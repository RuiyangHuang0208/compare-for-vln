from __future__ import annotations

from collections import deque
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
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
import requests
from vln_interfaces.msg import NavigationCommand

from .contracts import OfficialTicVlaCurvatureController, format_previous_waypoints, validate_waypoints


def yaw_from_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_seconds(message):
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) / 1.0e9


def decode_rgb(message):
    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported RGB encoding {message.encoding!r}")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, : message.width * channels].reshape(message.height, message.width, channels)
    if encoding.startswith("bgr"):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    return np.ascontiguousarray(image[..., :3])


def crop_to_aspect(image, aspect_ratio):
    """Center-crop only; preserve the calibrated horizontal field of view."""
    ratio = float(aspect_ratio)
    if ratio <= 0.0 or not math.isfinite(ratio):
        return image
    height, width = image.shape[:2]
    current = width / float(height)
    if math.isclose(current, ratio, rel_tol=1.0e-5, abs_tol=1.0e-5):
        return image
    if current < ratio:
        cropped_height = max(1, min(height, int(round(width / ratio))))
        top = (height - cropped_height) // 2
        return np.ascontiguousarray(image[top : top + cropped_height, :, :])
    cropped_width = max(1, min(width, int(round(height * ratio))))
    left = (width - cropped_width) // 2
    return np.ascontiguousarray(image[:, left : left + cropped_width, :])


class TicVlaAdapter(Node):
    def __init__(self, **kwargs):
        super().__init__("ticvla_adapter", **kwargs)
        defaults = {
            "model.base_model_path": "",
            "model.checkpoint_path": "",
            "model.device": "cuda:0",
            "model.dtype": "bfloat16",
            "model.server_url": "http://127.0.0.1:5802",
            "input.image_topic": "/camera/rgb/image_raw",
            "input.odom_topic": "/odom",
            "input.instruction_topic": "/vln/instruction",
            "input.episode_state_topic": "/episode/state",
            "input.history_length": 4,
            # Official DynaNav stores roughly 9 seconds of 10 Hz observations
            # and samples the context at 9, 6, 3, and 0 seconds.  Keep the
            # full window locally; history_length remains the model context
            # size, not the storage capacity.
            "input.history_capacity_frames": 190,
            "input.history_interval_frames": 30,
            "input.image_width": 640,
            "input.image_height": 480,
            # TIC-VLA's DynaNav camera is 16:9. The shared B2-W sensor remains
            # 640x480; this optional adapter crop preserves its horizontal FOV.
            "input.model_image_aspect_ratio": 0.0,
            "output.command_topic": "/vln/command",
            "output.debug_path_topic": "/vln/debug_path",
            "output.latency_topic": "/vln/inference_latency",
            "output.frame_id": "base_link",
            "output.action_horizon": 30,
            "output.action_dt": 0.1,
            "runtime.asynchronous_vlm": True,
            "runtime.initial_inference_timeout": 30.0,
            "runtime.command_timeout": 0.5,
            "runtime.publish_debug_path": True,
            "runtime.inference_rate_hz": 10.0,
            "evaluation.mode": "trajectory_normalized",
            "evaluation.direct_velocity": False,
            "native_output.lookahead_distance": 1.0,
            "native_output.angular_gain": 0.8,
            "native_output.yaw_filter_alpha": 0.35,
            "native_output.curvature_feedforward_gain": 0.5,
            "native_output.max_linear_velocity": 1.0,
            "native_output.max_angular_velocity": 1.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value
        if not bool(p("runtime.asynchronous_vlm")):
            raise ValueError("TIC-VLA must run asynchronously so inference cannot block robot control")
        self.history_length = int(p("input.history_length"))
        self.history_capacity_frames = int(p("input.history_capacity_frames"))
        self.history_interval_frames = int(p("input.history_interval_frames"))
        self.width = int(p("input.image_width"))
        self.height = int(p("input.image_height"))
        self.model_image_aspect_ratio = float(p("input.model_image_aspect_ratio"))
        if self.model_image_aspect_ratio < 0.0 or not math.isfinite(self.model_image_aspect_ratio):
            raise ValueError("input.model_image_aspect_ratio must be finite and non-negative")
        self.horizon = int(p("output.action_horizon"))
        self.action_dt = float(p("output.action_dt"))
        self.initial_timeout = float(p("runtime.initial_inference_timeout"))
        self.server_url = str(p("model.server_url")).rstrip("/")
        self.base_frame = str(p("output.frame_id"))
        self.debug_enabled = bool(p("runtime.publish_debug_path"))
        self.evaluation_mode = str(p("evaluation.mode"))
        self.direct_velocity = bool(p("evaluation.direct_velocity"))
        if self.evaluation_mode not in {"trajectory_normalized", "native_output"}:
            raise ValueError("TIC-VLA evaluation.mode must be trajectory_normalized or native_output")
        if self.direct_velocity != (self.evaluation_mode == "native_output"):
            raise ValueError("TIC-VLA native_output requires evaluation.direct_velocity=true")
        self.native_controller = OfficialTicVlaCurvatureController(
            lookahead=float(p("native_output.lookahead_distance")),
            angular_gain=float(p("native_output.angular_gain")),
            yaw_filter_alpha=float(p("native_output.yaw_filter_alpha")),
            curvature_feedforward_gain=float(p("native_output.curvature_feedforward_gain")),
            max_linear_velocity=float(p("native_output.max_linear_velocity")),
            max_angular_velocity=float(p("native_output.max_angular_velocity")),
        )
        rate = float(p("runtime.inference_rate_hz"))
        if (
            self.history_length < 2
            or self.history_capacity_frames < self.history_interval_frames + 1
            or self.history_interval_frames < 1
            or self.horizon != 30
            or self.action_dt <= 0.0
            or rate <= 0.0
        ):
            raise ValueError(
                "history_length>=2, history_capacity_frames>history_interval_frames, "
                "action_horizon=30, action_dt>0, and inference_rate_hz>0 are required"
            )

        self.history = deque(maxlen=self.history_capacity_frames)
        self.odom_history = deque(maxlen=256)
        self.pose = None
        self.velocity = np.zeros(3, dtype=np.float32)
        self.instruction = ""
        self.instruction_started = None
        self.generation = 0
        self.server_generation = -1
        self.future = None
        self.last_requested_stamp = None
        self.first_result = False
        self.timeout_reported = False
        self.last_native_velocity = np.zeros(3, dtype=np.float64)
        self.http_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ticvla-http")

        self.command_pub = self.create_publisher(NavigationCommand, str(p("output.command_topic")), 10)
        self.path_pub = self.create_publisher(Path, str(p("output.debug_path_topic")), 10)
        self.latency_pub = self.create_publisher(Float32MultiArray, str(p("output.latency_topic")), 10)
        self.create_subscription(Image, str(p("input.image_topic")), self.on_image, 10)
        self.create_subscription(Odometry, str(p("input.odom_topic")), self.on_odom, 20)
        self.create_subscription(String, str(p("input.instruction_topic")), self.on_instruction, 10)
        self.create_subscription(String, str(p("input.episode_state_topic")), self.on_episode_state, 10)
        self.create_timer(1.0 / rate, self.on_timer)
        self.publish_stop(False)
        self.get_logger().info(
            "TIC-VLA ROS adapter ready; model stays stopped until the first valid 30x2 trajectory; "
            f"service={self.server_url} mode={self.evaluation_mode}"
        )

    def on_odom(self, message):
        position = message.pose.pose.position
        pose = np.asarray((position.x, position.y, yaw_from_quaternion(message.pose.pose.orientation)))
        velocity = np.asarray(
            (message.twist.twist.linear.x, message.twist.twist.linear.y, message.twist.twist.angular.z),
            dtype=np.float32,
        )
        if np.isfinite(pose).all() and np.isfinite(velocity).all():
            self.pose = pose
            self.velocity = velocity
            self.odom_history.append((stamp_seconds(message), pose.copy(), velocity.copy()))

    def on_image(self, message):
        if self.pose is None:
            return
        if int(message.width) != self.width or int(message.height) != self.height:
            self.get_logger().warning(
                f"Rejected image {message.width}x{message.height}; expected {self.width}x{self.height}"
            )
            return
        try:
            image = crop_to_aspect(decode_rgb(message), self.model_image_aspect_ratio)
        except ValueError as error:
            self.get_logger().warning(str(error))
            return
        stamp = stamp_seconds(message)
        if self.history and stamp <= self.history[-1]["stamp"]:
            return
        pose, velocity = self.pose, self.velocity
        if self.odom_history:
            _, pose, velocity = min(self.odom_history, key=lambda item: abs(item[0] - stamp))
        self.history.append(
            {"image": image, "pose": pose.copy(), "velocity": velocity.copy(), "stamp": stamp}
        )

    def reset(self, reason):
        self.generation += 1
        self.history.clear()
        self.odom_history.clear()
        self.last_requested_stamp = None
        self.first_result = False
        self.timeout_reported = False
        self.native_controller.reset()
        self.last_native_velocity.fill(0.0)
        self.instruction_started = time.monotonic() if self.instruction else None
        self.publish_stop(False)
        self.get_logger().info(f"Cleared TIC-VLA history/cache state: {reason}")

    def on_instruction(self, message):
        self.instruction = message.data.strip()
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
        message.header.frame_id = self.base_frame
        message.command_type = NavigationCommand.STOP
        message.confidence = 1.0
        message.valid = True
        message.goal_reached = bool(goal_reached)
        self.command_pub.publish(message)

    def snapshot(self):
        if len(self.history) < 2 or self.pose is None:
            return None
        history = list(self.history)
        current = history[-1]
        # Match DynaNav's _get_sampled_image_paths(): oldest available frame
        # at -9/-6/-3 seconds followed by the current frame.  During startup,
        # use every available valid interval and never duplicate a frame.
        offsets = [
            self.history_interval_frames * (self.history_length - 1 - index)
            for index in range(self.history_length)
        ]
        frames = []
        for offset in offsets:
            index = len(history) - 1 - offset
            if index >= 0:
                frames.append(history[index])
        if not frames or frames[-1] is not current:
            frames.append(current)
        unique = []
        seen = set()
        for frame in frames:
            identity = frame["stamp"]
            if identity not in seen:
                unique.append(frame)
                seen.add(identity)
        frames = unique
        if not frames:
            return None
        return {
            "frames": [item["image"].copy() for item in frames],
            "velocity": [
                float(current["velocity"][0]),
                float(current["velocity"][1]),
                float(current["velocity"][2]),
            ],
            "current_pose": current["pose"].tolist(),
            "sim_step": int(round(current["stamp"] * 30.0)),
            "previous_waypoints_text": format_previous_waypoints(history, sample_interval=10),
            "instruction": self.instruction,
            "generation": self.generation,
            "frame_stamp": float(current["stamp"]),
        }

    def request(self, snapshot):
        health = requests.get(self.server_url + "/health", timeout=2.0)
        health.raise_for_status()
        if self.server_generation != snapshot["generation"]:
            reset = requests.post(self.server_url + "/reset", timeout=self.initial_timeout)
            reset.raise_for_status()
            self.server_generation = snapshot["generation"]
        files = []
        for index, image in enumerate(snapshot["frames"]):
            buffer = io.BytesIO()
            PilImage.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=90)
            files.append((f"image_{index}", (f"frame_{index}.jpg", buffer.getvalue(), "image/jpeg")))
        metadata = {
            "instruction": snapshot["instruction"],
            "velocity": snapshot["velocity"],
            "current_pose": snapshot["current_pose"],
            "sim_step": snapshot["sim_step"],
            "previous_waypoints_text": snapshot["previous_waypoints_text"],
        }
        response = requests.post(
            self.server_url + "/step",
            files=files,
            data={"json": json.dumps(metadata)},
            timeout=self.initial_timeout,
        )
        response.raise_for_status()
        result = response.json()
        result["generation"] = snapshot["generation"]
        return result

    def publish_trajectory(self, points):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame
        message.command_type = NavigationCommand.TRAJECTORY
        message.points = [Pose2D(x=float(x), y=float(y)) for x, y in points]
        message.dt = self.action_dt
        message.horizon = self.action_dt * len(points)
        message.confidence = 1.0
        message.valid = True
        self.command_pub.publish(message)
        self.publish_trajectory_debug(points, message.header)

    def publish_trajectory_debug(self, points, header=None):
        if self.debug_enabled:
            path = Path()
            path.header = header or NavigationCommand().header
            if header is None:
                path.header.stamp = self.get_clock().now().to_msg()
                path.header.frame_id = self.base_frame
            for x, y in points:
                pose = PoseStamped()
                pose.header = path.header
                pose.pose.position.x = float(x)
                pose.pose.position.y = float(y)
                pose.pose.orientation.w = 1.0
                path.poses.append(pose)
            self.path_pub.publish(path)

    def publish_velocity(self, velocity):
        values = np.asarray(velocity, dtype=np.float64)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("TIC-VLA native velocity must be finite [vx,vy,wz]")
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.base_frame
        message.command_type = NavigationCommand.VELOCITY
        message.velocity.linear.x = float(values[0])
        message.velocity.linear.y = float(values[1])
        message.velocity.angular.z = float(values[2])
        message.dt = self.action_dt
        message.horizon = self.action_dt
        message.confidence = 1.0
        message.valid = True
        self.command_pub.publish(message)

    def consume_result(self):
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        try:
            result = future.result()
            if result["generation"] != self.generation or not self.instruction:
                return
            points = validate_waypoints(result["waypoints"], self.horizon)
            raw_endpoint = float(np.linalg.norm(points[-1]))
            maximum_radius = float(np.linalg.norm(points, axis=1).max())
            if self.direct_velocity:
                velocity = self.native_controller.command(points)
                self.last_native_velocity[:] = velocity
                self.publish_velocity(velocity)
                self.publish_trajectory_debug(points)
            else:
                velocity = None
                self.publish_trajectory(points)
            self.first_result = True
            self.get_logger().info(
                "TIC-VLA trajectory: "
                f"pose=({self.pose[0]:.3f},{self.pose[1]:.3f},{self.pose[2]:.3f}) "
                f"points={len(points)} endpoint=({points[-1, 0]:.3f},{points[-1, 1]:.3f})m "
                f"radius={raw_endpoint:.3f}m max_radius={maximum_radius:.3f}m "
                f"kv_cache={bool(result.get('kv_cache_available', False))} "
                f"latency={float(result.get('total_latency', math.nan)):.3f}s "
                f"velocity={None if velocity is None else velocity.tolist()}"
            )
            self.latency_pub.publish(
                Float32MultiArray(
                    data=[
                        float(result.get("total_latency", math.nan)),
                        float(result.get("vlm_latency", math.nan)),
                        float(result.get("action_latency", math.nan)),
                    ]
                )
            )
        except Exception as error:
            self.first_result = False
            self.last_native_velocity.fill(0.0)
            self.publish_stop(False)
            self.get_logger().error(f"TIC-VLA inference rejected; stopped: {type(error).__name__}: {error}")

    def on_timer(self):
        self.consume_result()
        if not self.instruction:
            return
        # The public DynaNav behavior stores the most recent model action and
        # sends it on every simulation update until a newer inference arrives.
        # Republish here so the shared command watchdog sees the same behavior.
        if self.direct_velocity and self.first_result:
            self.publish_velocity(self.last_native_velocity)
        if (
            not self.first_result
            and not self.timeout_reported
            and self.instruction_started is not None
            and time.monotonic() - self.instruction_started > self.initial_timeout
        ):
            self.timeout_reported = True
            self.publish_stop(False)
            self.get_logger().error("Initial TIC-VLA inference timeout; robot remains stopped")
        if self.future is None:
            snapshot = self.snapshot()
            if snapshot is not None and snapshot["frame_stamp"] != self.last_requested_stamp:
                self.last_requested_stamp = snapshot["frame_stamp"]
                self.future = self.http_executor.submit(self.request, snapshot)

    def close(self):
        self.generation += 1
        if rclpy.ok():
            self.publish_stop(False)
        self.http_executor.shutdown(wait=False, cancel_futures=True)


def main(args=None):
    rclpy.init(args=args)
    node = TicVlaAdapter()
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
