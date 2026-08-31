from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import time
import uuid

from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Path
import numpy as np
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
import requests
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from vln_interfaces.msg import NavigationCommand

from .action_converter import action_to_trajectory
from .action_parser import parse_action
from .contracts import sample_episode_frames, validate_rgb


def decode_rgb(message):
    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported RGB encoding {message.encoding!r}")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, : message.width * channels].reshape(message.height, message.width, channels)
    if encoding.startswith("bgr"):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    return validate_rgb(np.ascontiguousarray(image[..., :3]))


class NavilaAdapter(Node):
    def __init__(self, **kwargs):
        super().__init__("navila_adapter", **kwargs)
        defaults = {
            "model.repository_path": "third_party/NaVILA",
            "model.checkpoint_path": "checkpoints/vln/navila",
            "model.device": "cuda:0",
            "model.dtype": "float16",
            "model.server_url": "http://127.0.0.1:5803",
            "input.image_topic": "/camera/rgb/image_raw",
            "input.instruction_topic": "/vln/instruction",
            "input.episode_topic": "/episode/state",
            "input.episode_id_topic": "/episode/id",
            "input.sensor_profile": "rgb_only",
            "output.command_topic": "/vln/command",
            "output.raw_action_topic": "/vln/navila/raw_action",
            "output.debug_path_topic": "/vln/debug_path",
            "output.latency_topic": "/vln/inference_latency",
            "output.model_stop_topic": "/episode/model_stop",
            "output.frame_id": "base_link",
            "conversion.trajectory_spacing": 0.1,
            "conversion.turn_radius": 0.0,
            "conversion.allowed_forward_cm": [25, 50, 75],
            "conversion.allowed_turn_degrees": [15, 30, 45],
            "runtime.asynchronous": True,
            "runtime.inference_timeout": 30.0,
            "runtime.command_timeout": 0.5,
            "runtime.inference_rate_hz": 2.0,
            "runtime.local_avoidance_enabled": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value
        if str(p("input.sensor_profile")) != "rgb_only":
            raise ValueError("NaVILA adapter only permits sensor_profile=rgb_only")
        if not bool(p("runtime.asynchronous")):
            raise ValueError("NaVILA inference must be asynchronous")
        if bool(p("runtime.local_avoidance_enabled")):
            raise ValueError("NaVILA local avoidance must remain disabled for fair comparison")
        self.turn_radius = float(p("conversion.turn_radius"))
        if not math.isfinite(self.turn_radius) or self.turn_radius <= 0.0:
            raise ValueError("conversion.turn_radius is uncalibrated; run the dummy B2-W arc test first")
        self.spacing = float(p("conversion.trajectory_spacing"))
        self.allowed_forward = tuple(int(value) for value in p("conversion.allowed_forward_cm"))
        self.allowed_turn = tuple(int(value) for value in p("conversion.allowed_turn_degrees"))
        self.server_url = str(p("model.server_url")).rstrip("/")
        self.frame_id = str(p("output.frame_id"))
        self.inference_timeout = float(p("runtime.inference_timeout"))
        rate = float(p("runtime.inference_rate_hz"))
        if self.inference_timeout <= 0.0 or rate <= 0.0:
            raise ValueError("inference_timeout and inference_rate_hz must be positive")

        self.latest_image = None
        self.latest_image_sequence = 0
        self.last_submitted_image_sequence = -1
        self.history = []
        self.instruction = ""
        self.episode_id = "unassigned"
        self.episode_active = False
        self.episode_started_once = False
        self.generation = 0
        self.future = None
        self.health_future = None
        self.future_started = None
        self.future_generation = None
        self.timeout_reported = False
        self.num_frames = None
        self.http_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="navila-http")

        self.command_pub = self.create_publisher(NavigationCommand, str(p("output.command_topic")), 10)
        self.raw_pub = self.create_publisher(String, str(p("output.raw_action_topic")), 10)
        self.path_pub = self.create_publisher(Path, str(p("output.debug_path_topic")), 10)
        self.latency_pub = self.create_publisher(Float32MultiArray, str(p("output.latency_topic")), 10)
        self.model_stop_pub = self.create_publisher(String, str(p("output.model_stop_topic")), 10)
        self.create_subscription(Image, str(p("input.image_topic")), self.on_image, 5)
        self.create_subscription(String, str(p("input.instruction_topic")), self.on_instruction, 10)
        self.create_subscription(String, str(p("input.episode_topic")), self.on_episode_state, 10)
        self.create_subscription(String, str(p("input.episode_id_topic")), self.on_episode_id, 10)
        self.create_timer(1.0 / rate, self.on_timer)
        self.publish_stop()
        self.get_logger().info(f"NaVILA RGB-only adapter waiting for service={self.server_url}")

    def on_image(self, message):
        try:
            self.latest_image = decode_rgb(message)
            self.latest_image_sequence += 1
        except ValueError as error:
            self.get_logger().warning(str(error))

    def on_episode_id(self, message):
        value = message.data.strip()
        if value:
            self.episode_id = value

    def reset(self, reason, clear_instruction=False, clear_image=False):
        self.generation += 1
        self.history.clear()
        if clear_image:
            self.latest_image = None
        self.last_submitted_image_sequence = self.latest_image_sequence - 1
        self.future = None
        self.future_started = None
        self.future_generation = None
        self.timeout_reported = False
        if clear_instruction:
            self.instruction = ""
        self.publish_stop()
        self.get_logger().info(f"Cleared NaVILA history and pending response: {reason}")

    def on_instruction(self, message):
        self.instruction = message.data.strip()
        resume_manual_demo = bool(self.instruction) and self.episode_started_once and not self.episode_active
        self.reset("new instruction", clear_instruction=False)
        if resume_manual_demo:
            self.episode_active = True
            self.get_logger().info(
                "Accepted a new instruction after episode completion; resumed manual demonstration mode"
            )

    def on_episode_state(self, message):
        state = message.data.strip().upper()
        if state in {"RESET", "FINISH", "FAILED", "SUCCESS"}:
            self.episode_active = False
            if state == "RESET":
                self.episode_started_once = False
            self.reset(f"episode {state}", clear_instruction=True, clear_image=True)
        elif state == "START":
            self.reset("episode START", clear_instruction=False, clear_image=False)
            self.episode_active = True
            self.episode_started_once = True

    def publish_stop(self):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.STOP
        message.confidence = 1.0
        message.valid = True
        message.goal_reached = False
        self.command_pub.publish(message)

    def publish_trajectory(self, values):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.TRAJECTORY
        message.points = [Pose2D(x=float(x), y=float(y), theta=float(yaw)) for x, y, yaw in values]
        message.dt = 0.1
        message.horizon = 0.1 * len(values)
        message.confidence = 1.0
        message.valid = True
        self.command_pub.publish(message)
        path = Path()
        path.header = message.header
        for x, y, _yaw in values:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self.path_pub.publish(path)

    def service_num_frames(self):
        response = requests.get(self.server_url + "/health", timeout=2.0)
        response.raise_for_status()
        value = int(response.json()["num_video_frames"])
        if value <= 0:
            raise ValueError("service returned invalid num_video_frames")
        return value

    def snapshot(self):
        if self.latest_image is None or not self.instruction:
            return None
        if self.num_frames is None:
            return None
        if self.latest_image_sequence == self.last_submitted_image_sequence:
            return None
        self.history.append(self.latest_image.copy())
        self.last_submitted_image_sequence = self.latest_image_sequence
        frames = sample_episode_frames(self.history, self.num_frames)
        request_id = f"{self.generation}-{uuid.uuid4().hex}"
        return {
            "frames": frames,
            "instruction": self.instruction,
            "episode_id": self.episode_id,
            "request_id": request_id,
            "generation": self.generation,
        }

    def request_inference(self, snapshot):
        files = []
        for index, image in enumerate(snapshot["frames"]):
            buffer = io.BytesIO()
            PilImage.fromarray(image, mode="RGB").save(buffer, format="PNG")
            files.append((f"image_{index}", (f"frame_{index}.png", buffer.getvalue(), "image/png")))
        metadata = {
            "instruction": snapshot["instruction"],
            "episode_id": snapshot["episode_id"],
            "request_id": snapshot["request_id"],
        }
        response = requests.post(
            self.server_url + "/step",
            files=files,
            data={"json": json.dumps(metadata)},
            timeout=self.inference_timeout,
        )
        response.raise_for_status()
        result = response.json()
        result["generation"] = snapshot["generation"]
        if result.get("request_id") != snapshot["request_id"]:
            raise ValueError("NaVILA service returned a mismatched request_id")
        if result.get("episode_id") != snapshot["episode_id"]:
            raise ValueError("NaVILA service returned a mismatched episode_id")
        return result

    def consume_result(self):
        if self.future is None or not self.future.done():
            return
        future, self.future = self.future, None
        self.future_started = None
        expected_generation, self.future_generation = self.future_generation, None
        try:
            result = future.result()
            if result["generation"] != self.generation or expected_generation != self.generation:
                return
            raw_action = str(result.get("raw_action", ""))
            self.raw_pub.publish(String(data=raw_action))
            self.latency_pub.publish(
                Float32MultiArray(data=[float(result["inference_s"]), math.nan, math.nan])
            )
            action = parse_action(raw_action, self.allowed_forward, self.allowed_turn)
            if action.kind == "stop":
                self.episode_active = False
                self.publish_stop()
                self.model_stop_pub.publish(
                    String(data=json.dumps({"episode_id": self.episode_id, "raw_action": raw_action}))
                )
                return
            trajectory = action_to_trajectory(action, self.spacing, self.turn_radius)
            self.publish_trajectory(trajectory)
            self.timeout_reported = False
            self.get_logger().info(
                f"NaVILA action={action.kind} value={action.value:g}; points={len(trajectory)}; "
                f"latency={float(result['inference_s']):.3f}s"
            )
        except Exception as error:
            self.publish_stop()
            self.get_logger().error(f"NaVILA inference/action rejected; stopped: {type(error).__name__}: {error}")

    def on_timer(self):
        self.consume_result()
        if self.num_frames is None:
            if self.health_future is None:
                self.health_future = self.http_executor.submit(self.service_num_frames)
            elif self.health_future.done():
                future, self.health_future = self.health_future, None
                try:
                    self.num_frames = future.result()
                    self.get_logger().info(f"NaVILA checkpoint requires {self.num_frames} RGB frames")
                except Exception as error:
                    self.publish_stop()
                    self.get_logger().error(
                        f"NaVILA service health check failed; stopped: {type(error).__name__}: {error}"
                    )
            return
        if self.future is not None:
            if (
                self.future_started is not None
                and not self.timeout_reported
                and time.monotonic() - self.future_started > self.inference_timeout
            ):
                self.timeout_reported = True
                self.publish_stop()
                self.get_logger().error("NaVILA inference timeout; robot stopped and late response will be discarded")
                self.generation += 1
            return
        if not self.episode_active or not self.instruction or self.latest_image is None:
            return
        try:
            snapshot = self.snapshot()
            if snapshot is None:
                return
            self.future_generation = self.generation
            self.future_started = time.monotonic()
            self.future = self.http_executor.submit(self.request_inference, snapshot)
        except Exception as error:
            self.publish_stop()
            self.get_logger().error(f"NaVILA service unavailable; stopped: {type(error).__name__}: {error}")

    def close(self):
        self.generation += 1
        if rclpy.ok():
            self.publish_stop()
        self.http_executor.shutdown(wait=False, cancel_futures=True)


def main(args=None):
    rclpy.init(args=args)
    node = NavilaAdapter()
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
