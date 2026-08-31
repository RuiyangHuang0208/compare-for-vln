from __future__ import annotations

import math

from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from vln_interfaces.msg import NavigationCommand

from ticvla_adapter.contracts import OfficialTicVlaCurvatureController

from .controller_probe_core import ALL_PROFILES, PATH_PROFILES, build_path, velocity_command


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class ControllerProbe(Node):
    """Deterministic source used only to validate high-level controller execution."""

    def __init__(self):
        super().__init__("controller_probe")
        self.declare_parameter("profile", "shared_pure_pursuit")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.profile = str(self.get_parameter("profile").value)
        if self.profile not in ALL_PROFILES:
            raise ValueError(f"profile must be one of {sorted(ALL_PROFILES)}")
        rate = float(self.get_parameter("publish_rate_hz").value)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive and finite")

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.command_pub = self.create_publisher(NavigationCommand, "/vln/command", 10)
        self.create_subscription(String, "/episode/state", self.on_state, qos)
        self.create_subscription(String, "/episode/id", self.on_episode_id, qos)
        self.create_subscription(PoseStamped, "/episode/goal", self.on_goal, qos)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.goal = None
        self.pose = None
        self.episode_id = ""
        self.active = False
        self.path_sent = False
        self.ticvla_controller = OfficialTicVlaCurvatureController(
            max_linear_velocity=1.0,
            max_angular_velocity=1.0,
        )
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"Controller acceptance probe profile={self.profile}")

    @property
    def motion(self):
        return "left" if self.episode_id.endswith("_left") else "straight"

    def on_episode_id(self, message):
        self.episode_id = message.data.strip()

    def on_goal(self, message):
        self.goal = np.asarray((message.pose.position.x, message.pose.position.y), dtype=np.float64)

    def on_odom(self, message):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = np.asarray((p.x, p.y, yaw_from_quaternion(q)), dtype=np.float64)
        if np.isfinite(pose).all():
            self.pose = pose

    def publish_stop(self):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.command_type = NavigationCommand.STOP
        message.valid = True
        message.confidence = 1.0
        self.command_pub.publish(message)

    def on_state(self, message):
        state = message.data.strip().upper()
        if state == "START":
            self.active = True
            self.path_sent = False
            self.ticvla_controller.reset()
        elif state in {"RESET", "FINISH", "FAILED", "SUCCESS"}:
            self.active = False
            self.path_sent = False
            self.ticvla_controller.reset()
            self.publish_stop()

    def local_goal(self):
        if self.goal is None or self.pose is None:
            return None
        delta = self.goal - self.pose[:2]
        c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
        return np.asarray((c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]))

    def make_message(self, command_type):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.command_type = command_type
        message.dt = 0.1
        message.horizon = 3.0
        message.confidence = 1.0
        message.valid = True
        return message

    def tick(self):
        if not self.active:
            return
        local_goal = self.local_goal()
        if local_goal is None:
            return
        if self.profile in PATH_PROFILES:
            if self.path_sent:
                return
            path = build_path(self.profile, self.motion, local_goal)
            message = self.make_message(NavigationCommand.TRAJECTORY)
            message.points = [
                Pose2D(x=float(x), y=float(y), theta=float(yaw)) for x, y, yaw in path
            ]
            self.command_pub.publish(message)
            self.path_sent = True
            self.get_logger().info(
                f"Published {self.profile} {self.motion} path with {len(path)} points"
            )
            return
        command = velocity_command(self.profile, local_goal, self.ticvla_controller)
        message = self.make_message(NavigationCommand.VELOCITY)
        message.velocity.linear.x = float(command[0])
        message.velocity.linear.y = float(command[1])
        message.velocity.angular.z = float(command[2])
        self.command_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = ControllerProbe()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
