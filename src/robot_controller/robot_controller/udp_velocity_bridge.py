from __future__ import annotations

import json
import math
import socket
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String


SOURCES = ("keyboard", "vln")


class UdpVelocityBridge(Node):
    """Select exactly one Twist source and transport it to Isaac Sim."""

    def __init__(self, **kwargs):
        super().__init__("b2w_udp_velocity_bridge", **kwargs)
        self.declare_parameter("command_source", "vln")
        self.declare_parameter("vln_topic", "/nav_vel")
        self.declare_parameter("keyboard_topic", "/keyboard/nav_vel")
        self.declare_parameter("source_topic", "/robot_controller/source")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("ground_truth_topic", "/ground_truth/odom")
        self.declare_parameter("command_host", "127.0.0.1")
        self.declare_parameter("command_port", 5820)
        self.declare_parameter("telemetry_host", "127.0.0.1")
        self.declare_parameter("telemetry_port", 5821)
        self.declare_parameter("debug_path_topic", "/vln/debug_path")
        self.declare_parameter("debug_path_host", "127.0.0.1")
        self.declare_parameter("debug_path_port", 5823)
        self.declare_parameter("command_timeout", 0.5)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("base_frame", "base_link")

        self.active_source = str(self.get_parameter("command_source").value)
        if self.active_source not in SOURCES:
            raise ValueError(f"command_source must be one of {SOURCES}")
        self.timeout = float(self.get_parameter("command_timeout").value)
        rate = float(self.get_parameter("publish_rate_hz").value)
        if self.timeout <= 0.0 or rate <= 0.0:
            raise ValueError("command_timeout and publish_rate_hz must be positive")

        self.command_address = (
            str(self.get_parameter("command_host").value),
            int(self.get_parameter("command_port").value),
        )
        self.debug_path_address = (
            str(self.get_parameter("debug_path_host").value),
            int(self.get_parameter("debug_path_port").value),
        )
        self.send_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receive_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receive_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.receive_socket.bind(
            (
                str(self.get_parameter("telemetry_host").value),
                int(self.get_parameter("telemetry_port").value),
            )
        )
        self.receive_socket.setblocking(False)

        self.latest = {source: (Twist(), None) for source in SOURCES}
        self.create_subscription(Twist, str(self.get_parameter("vln_topic").value), self.on_vln, 20)
        self.create_subscription(Twist, str(self.get_parameter("keyboard_topic").value), self.on_keyboard, 20)
        self.create_subscription(String, str(self.get_parameter("source_topic").value), self.on_source, 10)
        self.create_subscription(Path, str(self.get_parameter("debug_path_topic").value), self.on_debug_path, 10)
        self.create_subscription(String, "/episode/state", self.on_episode_state, 10)
        self.odom_pub = self.create_publisher(Odometry, str(self.get_parameter("odom_topic").value), 20)
        self.ground_truth_pub = self.create_publisher(
            Odometry, str(self.get_parameter("ground_truth_topic").value), 20
        )
        self.source_pub = self.create_publisher(String, "/robot_controller/active_source", 10)
        self.clock_pub = self.create_publisher(Clock, "/clock", 10)
        self.collision_pub = self.create_publisher(String, "/simulation/collision", 20)
        self.active_contacts = set()
        self.latest_world_pose = None
        self.create_timer(1.0 / rate, self.on_timer)
        self.send_zero()
        self.publish_source()
        self.get_logger().info(
            f"source={self.active_source}; UDP command={self.command_address}; telemetry port="
            f"{self.get_parameter('telemetry_port').value}; viewport path UDP={self.debug_path_address}"
        )

    @staticmethod
    def finite_twist(message: Twist):
        values = (message.linear.x, message.linear.y, message.angular.z)
        return all(math.isfinite(value) for value in values)

    def on_vln(self, message: Twist):
        self.on_velocity("vln", message)

    def on_keyboard(self, message: Twist):
        self.on_velocity("keyboard", message)

    def on_velocity(self, source: str, message: Twist):
        if not self.finite_twist(message):
            self.get_logger().warning(f"Rejected non-finite {source} velocity")
            return
        self.latest[source] = (message, time.monotonic())

    def on_source(self, message: String):
        source = message.data.strip().lower()
        if source not in SOURCES:
            self.get_logger().warning(f"Unknown source {source!r}; expected keyboard or vln")
            return
        if source != self.active_source:
            self.send_zero()
            self.send_debug_path([])
            self.active_source = source
            self.latest[source] = (Twist(), None)
            self.publish_source()
            self.get_logger().info(f"Switched command source to {source}; waiting for a fresh command")

    def publish_source(self):
        self.source_pub.publish(String(data=self.active_source))

    def on_episode_state(self, message: String):
        state = message.data.strip().upper()
        if state == "RESET":
            self.send_reset()
            self.send_debug_path([])
            self.latest = {source: (Twist(), None) for source in SOURCES}
        elif state in {"FINISH", "FAILED", "SUCCESS"}:
            self.send_zero()
            self.send_debug_path([])

    def on_debug_path(self, message: Path):
        if self.active_source != "vln":
            return
        world_frame = str(self.get_parameter("world_frame").value)
        base_frame = str(self.get_parameter("base_frame").value)
        if message.header.frame_id not in {world_frame, base_frame}:
            self.get_logger().warning(
                f"Rejected debug path frame {message.header.frame_id!r}; "
                f"expected {world_frame!r} or {base_frame!r}"
            )
            return
        if message.header.frame_id == base_frame and self.latest_world_pose is None:
            self.get_logger().warning("Rejected base_link debug path before simulator telemetry was received")
            return
        if message.header.frame_id == base_frame:
            position, yaw = self.latest_world_pose
            cosine, sine = math.cos(yaw), math.sin(yaw)
        points = []
        for pose in message.poses:
            values = (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z)
            if not all(math.isfinite(value) for value in values):
                self.get_logger().warning("Rejected non-finite debug path")
                return
            x, y, z = (float(value) for value in values)
            if message.header.frame_id == base_frame:
                x, y, z = (
                    position[0] + cosine * x - sine * y,
                    position[1] + sine * x + cosine * y,
                    position[2] + z,
                )
            points.append([x, y, z])
        if len(points) > 2048:
            self.get_logger().warning(f"Rejected debug path with {len(points)} points; maximum is 2048")
            return
        self.send_debug_path(points)

    def send_debug_path(self, points):
        payload = {
            "stamp": time.time(),
            "frame_id": str(self.get_parameter("world_frame").value),
            "points": points,
        }
        self.send_socket.sendto(
            json.dumps(payload, separators=(",", ":")).encode("utf-8"), self.debug_path_address
        )

    def send(self, message: Twist):
        payload = {
            "stamp": time.time(),
            "source": self.active_source,
            "vx": float(message.linear.x),
            "vy": float(message.linear.y),
            "wz": float(message.angular.z),
        }
        self.send_socket.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), self.command_address)

    def send_zero(self):
        self.send(Twist())

    def send_reset(self):
        payload = {"stamp": time.time(), "source": self.active_source, "reset": True}
        self.send_socket.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), self.command_address)

    def receive_telemetry(self):
        while True:
            try:
                raw, _ = self.receive_socket.recvfrom(65535)
            except BlockingIOError:
                return
            try:
                state = json.loads(raw.decode("utf-8"))
                position = [float(value) for value in state["position"]]
                orientation = [float(value) for value in state["orientation_wxyz"]]
                linear = [float(value) for value in state["linear_velocity_b"]]
                angular = [float(value) for value in state["angular_velocity_b"]]
                values = position + orientation + linear + angular
                if len(position) != 3 or len(orientation) != 4 or not all(math.isfinite(v) for v in values):
                    raise ValueError("invalid state vector")
            except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                self.get_logger().warning(f"Rejected simulator telemetry: {error}")
                continue
            sim_time = max(0.0, float(state.get("sim_time", 0.0)))
            message = Odometry()
            message.header.stamp.sec = int(sim_time)
            message.header.stamp.nanosec = int((sim_time - int(sim_time)) * 1.0e9)
            message.header.frame_id = str(self.get_parameter("world_frame").value)
            message.child_frame_id = str(self.get_parameter("base_frame").value)
            message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z = position
            message.pose.pose.orientation.w = orientation[0]
            message.pose.pose.orientation.x = orientation[1]
            message.pose.pose.orientation.y = orientation[2]
            message.pose.pose.orientation.z = orientation[3]
            message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.linear.z = linear
            message.twist.twist.angular.x, message.twist.twist.angular.y, message.twist.twist.angular.z = angular
            w, x, y, z = orientation
            yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            self.latest_world_pose = (position, yaw)
            self.odom_pub.publish(message)
            self.ground_truth_pub.publish(message)
            clock = Clock()
            clock.clock.sec = int(sim_time)
            clock.clock.nanosec = int((sim_time - int(sim_time)) * 1.0e9)
            self.clock_pub.publish(clock)
            contacts = state.get("contacts", [])
            current_contacts = {
                str(contact.get("robot_body", "unknown"))
                for contact in contacts
                if isinstance(contact, dict)
            }
            for contact in contacts:
                if not isinstance(contact, dict):
                    continue
                body = str(contact.get("robot_body", "unknown"))
                if body in self.active_contacts:
                    continue
                event = {
                    "time": sim_time,
                    "force": float(contact.get("force", 0.0)),
                    "object": str(contact.get("object", "unknown_static_geometry")),
                    "robot_body": body,
                    "is_pedestrian": bool(contact.get("is_pedestrian", False)),
                    "episode_failure": False,
                }
                if "distance" in contact:
                    event["distance"] = float(contact["distance"])
                self.collision_pub.publish(String(data=json.dumps(event, separators=(",", ":"))))
            self.active_contacts = current_contacts

    def on_timer(self):
        message, received = self.latest[self.active_source]
        if received is None or time.monotonic() - received > self.timeout:
            self.send_zero()
        else:
            self.send(message)
        self.receive_telemetry()

    def close(self):
        self.send_zero()
        self.send_debug_path([])
        self.send_socket.close()
        self.receive_socket.close()


def main(args=None):
    rclpy.init(args=args)
    node = UdpVelocityBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
