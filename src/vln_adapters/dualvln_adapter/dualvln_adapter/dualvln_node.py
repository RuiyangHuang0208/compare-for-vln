from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import time

from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Odometry, Path
import numpy as np
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
import requests
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String
from vln_interfaces.msg import NavigationCommand

from .coordinates import trajectory_at_capture_to_current


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def image_stamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def decode_rgb(message: Image):
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(message.encoding.lower())
    if channels is None:
        raise ValueError(f"unsupported RGB encoding {message.encoding!r}")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    array = rows[:, : message.width * channels].reshape(message.height, message.width, channels)
    encoding = message.encoding.lower()
    if encoding.startswith("bgr"):
        array = array[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    return np.ascontiguousarray(array[..., :3])


def decode_depth(message: Image):
    encoding = message.encoding.upper()
    if encoding == "32FC1":
        dtype, scale = np.dtype("<f4"), 1.0
    elif encoding == "16UC1":
        dtype, scale = np.dtype("<u2"), 0.001
    else:
        raise ValueError(f"unsupported depth encoding {message.encoding!r}")
    if message.is_bigendian:
        dtype = dtype.newbyteorder(">")
    elements_per_row = message.step // dtype.itemsize
    rows = np.frombuffer(message.data, dtype=dtype).reshape(message.height, elements_per_row)
    return np.ascontiguousarray(rows[:, : message.width], dtype=np.float32) * scale


def discrete_action_trajectory(action):
    """Preserve confirmed InternNav action semantics while retaining the shared path follower."""
    if action == 1:
        return np.column_stack((np.linspace(0.1, 1.0, 10), np.zeros(10)))
    if action in (2, 3):
        sign = 1.0 if action == 2 else -1.0
        angles = sign * np.linspace(0.05, 0.65, 12)
        return np.column_stack((np.sin(np.abs(angles)), sign * (1.0 - np.cos(angles))))
    raise ValueError(f"unsupported DualVLN discrete action {action}")


class DualVlnAdapter(Node):
    def __init__(self, **kwargs):
        super().__init__("dualvln_adapter", **kwargs)
        defaults = {
            "model.server_url": "http://127.0.0.1:5801",
            "input.image_topic": "/camera/rgb/image_raw",
            "input.depth_topic": "/camera/depth/image_raw",
            "input.camera_info_topic": "/camera/rgb/camera_info",
            "input.odom_topic": "/odom",
            "input.instruction_topic": "/vln/instruction",
            "input.episode_state_topic": "/episode/state",
            "output.command_topic": "/vln/command",
            "output.debug_path_topic": "/vln/debug_path",
            "output.latency_topic": "/vln/inference_latency",
            "output.frame_id": "base_link",
            "runtime.inference_rate_hz": 2.0,
            "runtime.sensor_timeout": 0.5,
            "runtime.request_timeout": 180.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value
        rate = float(p("runtime.inference_rate_hz"))
        if rate <= 0.0:
            raise ValueError("inference_rate_hz must be positive")
        self.server_url = str(p("model.server_url")).rstrip("/")
        self.frame_id = str(p("output.frame_id"))
        self.sensor_timeout = float(p("runtime.sensor_timeout"))
        self.request_timeout = float(p("runtime.request_timeout"))
        self.rgb = None
        self.rgb_stamp = 0
        self.rgb_received_monotonic = None
        self.depth = None
        self.depth_stamp = 0
        self.depth_received_monotonic = None
        self.intrinsics = None
        self.pose = None
        self.instruction = ""
        self.generation = 0
        self.frame_number = 0
        self.reset_pending = True
        self.future = None
        self.http_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dualvln-http")

        self.command_pub = self.create_publisher(NavigationCommand, str(p("output.command_topic")), 10)
        self.path_pub = self.create_publisher(Path, str(p("output.debug_path_topic")), 10)
        self.latency_pub = self.create_publisher(Float32MultiArray, str(p("output.latency_topic")), 10)
        self.create_subscription(Image, str(p("input.image_topic")), self.on_rgb, 5)
        self.create_subscription(Image, str(p("input.depth_topic")), self.on_depth, 5)
        self.create_subscription(CameraInfo, str(p("input.camera_info_topic")), self.on_camera_info, 5)
        self.create_subscription(Odometry, str(p("input.odom_topic")), self.on_odom, 20)
        self.create_subscription(String, str(p("input.instruction_topic")), self.on_instruction, 10)
        self.create_subscription(String, str(p("input.episode_state_topic")), self.on_episode_state, 10)
        self.create_timer(1.0 / rate, self.on_timer)
        self.publish_stop(False)
        self.get_logger().info(f"Waiting for instruction and aligned RGB-D; service={self.server_url}")

    def on_rgb(self, message):
        try:
            self.rgb = decode_rgb(message)
            self.rgb_stamp = image_stamp_ns(message)
            self.rgb_received_monotonic = time.monotonic()
        except ValueError as error:
            self.get_logger().warning(str(error))

    def on_depth(self, message):
        try:
            self.depth = decode_depth(message)
            self.depth_stamp = image_stamp_ns(message)
            self.depth_received_monotonic = time.monotonic()
        except ValueError as error:
            self.get_logger().warning(str(error))

    def on_camera_info(self, message):
        matrix = np.asarray(message.k, dtype=np.float32).reshape(3, 3)
        if np.isfinite(matrix).all() and matrix[0, 0] > 0.0 and matrix[1, 1] > 0.0:
            self.intrinsics = matrix

    def on_odom(self, message):
        position = message.pose.pose.position
        values = np.asarray((position.x, position.y, yaw_from_quaternion(message.pose.pose.orientation)))
        if np.isfinite(values).all():
            self.pose = values

    def reset(self, reason):
        self.generation += 1
        self.rgb = self.depth = None
        self.rgb_stamp = self.depth_stamp = 0
        self.rgb_received_monotonic = self.depth_received_monotonic = None
        self.reset_pending = True
        self.publish_stop(False)
        self.get_logger().info(f"Cleared DualVLN adapter state: {reason}")

    def on_instruction(self, message):
        instruction = message.data.strip()
        if not instruction:
            self.instruction = ""
            self.reset("empty instruction")
            return
        self.instruction = instruction
        self.reset("new instruction")

    def on_episode_state(self, message):
        state = message.data.strip().upper()
        if state in {"RESET", "FINISH", "FAILED", "SUCCESS"}:
            self.instruction = ""
            self.reset(f"episode {state}")
        elif state == "START":
            self.reset("episode START")

    def publish_stop(self, goal_reached):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.STOP
        message.confidence = 1.0
        message.valid = True
        message.goal_reached = bool(goal_reached)
        self.command_pub.publish(message)

    def sensor_snapshot(self):
        if self.rgb is None or self.depth is None or self.intrinsics is None or self.pose is None:
            return None
        # SensorBridge stamps frames with Isaac simulation time. Compare age
        # using local receipt time instead of mixing simulation and wall clocks.
        now = time.monotonic()
        if self.rgb_received_monotonic is None or self.depth_received_monotonic is None:
            return None
        age = now - min(self.rgb_received_monotonic, self.depth_received_monotonic)
        alignment = abs(self.rgb_stamp - self.depth_stamp) / 1.0e9
        if age > self.sensor_timeout or alignment > 0.05 or self.rgb.shape[:2] != self.depth.shape:
            return None
        snapshot = {
            "rgb": self.rgb.copy(),
            "depth": self.depth.copy(),
            "intrinsics": self.intrinsics.copy(),
            "capture_pose": self.pose.copy(),
            "instruction": self.instruction,
            "generation": self.generation,
            "frame_number": self.frame_number,
            "reset": self.reset_pending,
        }
        self.frame_number += 1
        self.reset_pending = False
        return snapshot

    def request(self, snapshot):
        started = time.monotonic()
        health = requests.get(self.server_url + "/health", timeout=2.0)
        health.raise_for_status()
        rgb_buffer = io.BytesIO()
        PilImage.fromarray(snapshot["rgb"], mode="RGB").save(rgb_buffer, format="JPEG", quality=90)
        depth = np.nan_to_num(snapshot["depth"], nan=0.0, posinf=0.0, neginf=0.0)
        depth_u16 = np.clip(depth * 10000.0, 0, 65535).astype(np.uint16)
        depth_buffer = io.BytesIO()
        PilImage.fromarray(depth_u16).save(depth_buffer, format="PNG")
        metadata = {
            "instruction": snapshot["instruction"],
            "reset": snapshot["reset"],
            "frame_id": snapshot["frame_number"],
            "instruction_id": snapshot["generation"],
            "capture_pose": snapshot["capture_pose"].tolist(),
            "intrinsics": snapshot["intrinsics"].tolist(),
        }
        response = requests.post(
            self.server_url + "/step",
            files={
                "image": ("front.jpg", rgb_buffer.getvalue(), "image/jpeg"),
                "depth": ("front_depth.png", depth_buffer.getvalue(), "image/png"),
            },
            data={"json": json.dumps(metadata)},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        result = response.json()
        result["adapter_latency"] = time.monotonic() - started
        result["capture_pose"] = snapshot["capture_pose"]
        result["generation"] = snapshot["generation"]
        return result

    def publish_trajectory(self, points, confidence=1.0):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.TRAJECTORY
        message.points = [Pose2D(x=float(x), y=float(y)) for x, y in points]
        message.dt = 0.1
        message.horizon = 0.1 * len(points)
        message.confidence = float(np.clip(confidence, 0.0, 1.0))
        message.valid = True
        self.command_pub.publish(message)
        path = Path()
        path.header = message.header
        for x, y in points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    def consume_result(self):
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        try:
            result = future.result()
            if result["generation"] != self.generation or not self.instruction:
                return
            self.latency_pub.publish(
                Float32MultiArray(data=[float(result["adapter_latency"]), math.nan, math.nan])
            )
            if result.get("stop", False) or result.get("discrete_action") == [0]:
                self.publish_stop(True)
                return
            if "trajectory" in result:
                points = trajectory_at_capture_to_current(result["trajectory"], result["capture_pose"], self.pose)
            elif result.get("discrete_action"):
                points = discrete_action_trajectory(int(result["discrete_action"][0]))
            else:
                raise ValueError("service returned neither trajectory nor a supported action")
            self.publish_trajectory(points)
        except Exception as error:
            self.publish_stop(False)
            self.get_logger().error(f"DualVLN inference rejected; stopped: {type(error).__name__}: {error}")

    def on_timer(self):
        self.consume_result()
        if self.future is not None or not self.instruction:
            return
        snapshot = self.sensor_snapshot()
        if snapshot is not None:
            self.future = self.http_executor.submit(self.request, snapshot)

    def close(self):
        self.generation += 1
        if rclpy.ok():
            self.publish_stop(False)
        self.http_executor.shutdown(wait=False, cancel_futures=True)


def main(args=None):
    rclpy.init(args=args)
    node = DualVlnAdapter()
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
