from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
import time
import uuid

from geometry_msgs.msg import Pose2D
import numpy as np
from PIL import Image as PilImage
import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, String
from vln_interfaces.msg import NavigationCommand

from .action_converter import actions_to_trajectory
from .action_parser import parse_action_sequence
from .contracts import response_is_current


def decode_rgb(message: Image):
    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"Unsupported RGB encoding {message.encoding!r}")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, : message.width * channels].reshape(message.height, message.width, channels)
    if encoding.startswith("bgr"):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    values = np.ascontiguousarray(image[..., :3])
    if values.shape != (message.height, message.width, 3):
        raise ValueError("Malformed RGB image")
    return values


class UniNaVidAdapter(Node):
    def __init__(self, **kwargs):
        super().__init__("uninavid_adapter", **kwargs)
        defaults = {
            "model.repository_path": "third_party/Uni-NaVid",
            "model.checkpoint_path": "checkpoints/vln/uninavid/uninavid-7b-full-224-video-fps-1-grid-2",
            "model.vision_encoder_path": "checkpoints/vln/uninavid/eva_vit_g.pth",
            "model.device": "cuda:0",
            "model.dtype": "auto",
            "model.run_type": "eval",
            "model.conversation_mode": "vicuna_v1",
            "model.server_url": "http://127.0.0.1:5804",
            "input.image_topic": "/camera/rgb/image_raw",
            "input.instruction_topic": "/vln/instruction",
            "input.episode_topic": "/episode/state",
            "input.episode_id_topic": "/episode/id",
            "input.sensor_profile": "rgb_only",
            "output.command_topic": "/vln/command",
            "output.raw_action_topic": "/vln/uninavid/raw_action",
            "output.latency_topic": "/vln/inference_latency",
            "output.model_stop_topic": "/episode/model_stop",
            "output.trajectory_finished_topic": "/navigation/trajectory_finished",
            "output.trajectory_failed_topic": "/navigation/trajectory_failed",
            "output.frame_id": "base_link",
            "input.frame_sample_hz": 0.0,
            "generation.do_sample": True,
            "generation.temperature": 0.2,
            "generation.offline_temperature": 0.5,
            "generation.max_new_tokens": 1024,
            "generation.predicted_action_limit": 4,
            "generation.executed_action_limit": 2,
            "action.forward_distance": 0.5,
            "action.turn_angle_degrees": 30.0,
            "conversion.turn_radius": 0.0,
            "conversion.trajectory_spacing": 0.1,
            "runtime.poll_rate_hz": 10.0,
            "runtime.inference_timeout": 30.0,
            "runtime.trajectory_timeout": 15.0,
            "runtime.command_timeout": 0.5,
            "runtime.asynchronous": True,
            "runtime.local_avoidance_enabled": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value
        if str(p("input.sensor_profile")) != "rgb_only":
            raise ValueError("Uni-NaVid only permits sensor_profile=rgb_only")
        if not bool(p("runtime.asynchronous")):
            raise ValueError("Uni-NaVid inference must be asynchronous")
        if bool(p("runtime.local_avoidance_enabled")):
            raise ValueError("Uni-NaVid local avoidance must remain disabled")
        self.frame_sample_hz = float(p("input.frame_sample_hz"))
        if not math.isfinite(self.frame_sample_hz) or self.frame_sample_hz <= 0.0:
            raise ValueError("input.frame_sample_hz is unconfirmed; benchmark launch is blocked")
        self.turn_radius = float(p("conversion.turn_radius"))
        if not math.isfinite(self.turn_radius) or self.turn_radius <= 0.0:
            raise ValueError("conversion.turn_radius is uncalibrated; run the dummy B2-W arc test first")
        self.forward_distance = float(p("action.forward_distance"))
        self.turn_degrees = float(p("action.turn_angle_degrees"))
        self.spacing = float(p("conversion.trajectory_spacing"))
        self.max_actions = int(p("generation.predicted_action_limit"))
        self.execute_actions = int(p("generation.executed_action_limit"))
        if self.max_actions != 4 or self.execute_actions != 2:
            raise ValueError("Official Uni-NaVid contract requires max_predicted_actions=4 and execute_actions=2")
        self.server_url = str(p("model.server_url")).rstrip("/")
        self.frame_id = str(p("output.frame_id"))
        self.inference_timeout = float(p("runtime.inference_timeout"))
        self.trajectory_timeout = float(p("runtime.trajectory_timeout"))

        self.instruction = ""
        self.episode_id = "unassigned"
        self.episode_active = False
        self.episode_started_once = False
        self.generation = 0
        self.pending_frames = []
        self.last_frame_time = None
        self.future = None
        self.future_kind = None
        self.future_generation = None
        self.future_started = None
        self.future_timeout_reported = False
        self.reset_pending = False
        self.cache_ready = False
        self.waiting_for_trajectory = False
        self.trajectory_started = None
        self.stop_after_trajectory = False
        self.http_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="uninavid-http")

        self.command_pub = self.create_publisher(NavigationCommand, str(p("output.command_topic")), 10)
        self.raw_pub = self.create_publisher(String, str(p("output.raw_action_topic")), 10)
        self.latency_pub = self.create_publisher(Float32MultiArray, str(p("output.latency_topic")), 10)
        self.model_stop_pub = self.create_publisher(String, str(p("output.model_stop_topic")), 10)
        self.create_subscription(Image, str(p("input.image_topic")), self.on_image, 5)
        self.create_subscription(String, str(p("input.instruction_topic")), self.on_instruction, 10)
        self.create_subscription(String, str(p("input.episode_topic")), self.on_episode_state, 10)
        self.create_subscription(String, str(p("input.episode_id_topic")), self.on_episode_id, 10)
        self.create_subscription(Bool, str(p("output.trajectory_finished_topic")), self.on_trajectory_finished, 10)
        self.create_subscription(Bool, str(p("output.trajectory_failed_topic")), self.on_trajectory_failed, 10)
        poll_rate = float(p("runtime.poll_rate_hz"))
        self.create_timer(1.0 / poll_rate, self.on_timer)
        self.publish_stop()
        self.get_logger().info(
            f"Uni-NaVid RGB-only adapter waiting for {self.server_url}; sample_rate={self.frame_sample_hz:.1f} Hz"
        )

    def on_image(self, message):
        if not self.episode_active:
            return
        now = time.monotonic()
        if self.last_frame_time is not None and now - self.last_frame_time < 1.0 / self.frame_sample_hz:
            return
        try:
            self.pending_frames.append(decode_rgb(message))
            self.last_frame_time = now
        except ValueError as error:
            self.get_logger().warning(str(error))

    def on_instruction(self, message):
        value = message.data.strip()
        resume_manual = bool(value) and self.episode_started_once and not self.episode_active
        # DDS does not guarantee ordering between the latched instruction and
        # START messages.  The first instruction received after START is the
        # episode instruction, not a runtime replacement, so it must not bump
        # the generation or invalidate the server cache.
        changed_during_run = (
            bool(value) and self.episode_active and bool(self.instruction) and value != self.instruction
        )
        self.instruction = value
        if resume_manual or changed_during_run:
            self.reset_local("new instruction")
            self.episode_active = True
            self.schedule_reset()

    def on_episode_id(self, message):
        if message.data.strip():
            self.episode_id = message.data.strip()

    def reset_local(self, reason, clear_instruction=False):
        self.generation += 1
        self.pending_frames.clear()
        self.last_frame_time = None
        self.cache_ready = False
        self.waiting_for_trajectory = False
        self.trajectory_started = None
        self.stop_after_trajectory = False
        if clear_instruction:
            self.instruction = ""
        self.publish_stop()
        self.get_logger().info(f"Cleared Uni-NaVid cache state and pending actions: {reason}")

    def on_episode_state(self, message):
        state = message.data.strip().upper()
        if state == "START":
            self.reset_local("episode START")
            self.episode_active = True
            self.episode_started_once = True
            self.schedule_reset()
        elif state in {"RESET", "FINISH", "FAILED", "SUCCESS"}:
            self.episode_active = False
            if state == "RESET":
                self.episode_started_once = False
            self.reset_local(f"episode {state}", clear_instruction=True)

    def request_reset(self, generation):
        response = requests.post(
            self.server_url + "/reset",
            json={"episode_id": self.episode_id, "generation": generation},
            timeout=self.inference_timeout,
        )
        response.raise_for_status()
        return response.json()

    def schedule_reset(self):
        if self.future is None:
            self.reset_pending = False
            self.submit_future("reset", self.generation, self.request_reset, self.generation)
        else:
            self.reset_pending = True

    def submit_future(self, kind, generation, function, *args):
        if self.future is not None:
            return False
        self.future = self.http_executor.submit(function, *args)
        self.future_kind = kind
        self.future_generation = generation
        self.future_started = time.monotonic()
        self.future_timeout_reported = False
        return True

    def request_step(self, snapshot):
        files = []
        for index, image in enumerate(snapshot["frames"]):
            buffer = io.BytesIO()
            PilImage.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=95)
            files.append((f"image_{index}", (f"frame_{index}.jpg", buffer.getvalue(), "image/jpeg")))
        metadata = {
            "instruction": snapshot["instruction"],
            "episode_id": snapshot["episode_id"],
            "generation": snapshot["generation"],
            "request_id": snapshot["request_id"],
        }
        response = requests.post(
            self.server_url + "/step",
            files=files,
            data={"json": json.dumps(metadata)},
            timeout=self.inference_timeout,
        )
        response.raise_for_status()
        return response.json()

    def publish_stop(self, model_requested=False):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.STOP
        message.confidence = 1.0
        message.valid = True
        self.command_pub.publish(message)
        if model_requested:
            self.model_stop_pub.publish(String(data=f"{self.episode_id}:uninavid"))

    def fail_safe_stop(self, reason):
        self.waiting_for_trajectory = False
        self.stop_after_trajectory = False
        self.publish_stop()
        self.get_logger().error(reason)

    def recover_cache_conflict(self, error):
        response = getattr(error, "response", None)
        try:
            payload = response.json() if response is not None else {}
        except (ValueError, requests.exceptions.RequestException):
            payload = {}
        server_episode = str(payload.get("episode_id", ""))
        if server_episode and server_episode != self.episode_id:
            self.fail_safe_stop(
                f"Uni-NaVid cache belongs to episode {server_episode!r}, expected {self.episode_id!r}"
            )
            return
        try:
            server_generation = int(payload["generation"])
        except (KeyError, TypeError, ValueError):
            server_generation = self.generation
        # Keep the newest generation observed by either side, then explicitly
        # reset the server cache before submitting another frame.
        self.generation = max(self.generation, server_generation)
        self.cache_ready = False
        self.waiting_for_trajectory = False
        self.stop_after_trajectory = False
        self.publish_stop()
        self.get_logger().warning(
            f"Uni-NaVid cache generation conflict; reinitializing generation={self.generation}"
        )
        self.schedule_reset()

    def publish_trajectory(self, points):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.TRAJECTORY
        message.points = [Pose2D(x=float(x), y=float(y), theta=float(yaw)) for x, y, yaw in points]
        message.dt = 0.1
        message.horizon = 0.1 * len(points)
        message.confidence = 1.0
        message.valid = True
        self.command_pub.publish(message)
        self.waiting_for_trajectory = True
        self.trajectory_started = time.monotonic()

    def on_trajectory_finished(self, message):
        if not message.data or not self.waiting_for_trajectory:
            return
        self.waiting_for_trajectory = False
        self.trajectory_started = None
        if self.stop_after_trajectory:
            self.stop_after_trajectory = False
            self.publish_stop(model_requested=True)

    def on_trajectory_failed(self, message):
        if not message.data or not self.waiting_for_trajectory:
            return
        self.waiting_for_trajectory = False
        self.trajectory_started = None
        self.stop_after_trajectory = False
        self.get_logger().warning("Shared trajectory execution failed; requesting a fresh observation")

    def consume_future(self):
        if self.future is None or not self.future.done():
            return
        future, kind = self.future, self.future_kind
        generation = self.future_generation
        timed_out = self.future_timeout_reported
        self.future = self.future_kind = self.future_generation = self.future_started = None
        self.future_timeout_reported = False
        if timed_out:
            self.generation += 1
            self.cache_ready = False
            self.schedule_reset()
            return
        if self.reset_pending and self.episode_active:
            self.schedule_reset()
            return
        if generation != self.generation:
            return
        try:
            result = future.result()
            if kind == "reset":
                if int(result["generation"]) != self.generation:
                    raise ValueError("reset generation mismatch")
                self.cache_ready = True
                return
            if result.get("request_id") is None or not response_is_current(
                result, self.episode_id, self.generation
            ):
                raise ValueError("inference response identity mismatch")
            raw = str(result.get("raw_action", ""))
            self.raw_pub.publish(String(data=raw))
            # Later entries are reserved for independently measured VLM and
            # ActionExpert latency. Uni-NaVid exposes only end-to-end latency.
            self.latency_pub.publish(Float32MultiArray(data=[float(result.get("latency", 0.0))]))
            plan = parse_action_sequence(raw, self.max_actions, self.execute_actions)
            if not plan.executed_actions:
                self.publish_stop(model_requested=plan.stop_after_trajectory)
                return
            points = actions_to_trajectory(
                plan.executed_actions,
                forward_distance=self.forward_distance,
                turn_degrees=self.turn_degrees,
                turn_radius=self.turn_radius,
                spacing=self.spacing,
            )
            self.stop_after_trajectory = plan.stop_after_trajectory
            self.publish_trajectory(points)
            self.get_logger().info(
                f"Uni-NaVid raw={raw!r}; executing={list(plan.executed_actions)}; points={len(points)}; "
                f"latency={float(result.get('latency', 0.0)):.3f}s"
            )
        except requests.HTTPError as error:
            if error.response is not None and error.response.status_code == 409:
                self.recover_cache_conflict(error)
            else:
                self.fail_safe_stop(f"Uni-NaVid {kind} failed: {type(error).__name__}: {error}")
        except Exception as error:
            self.fail_safe_stop(f"Uni-NaVid {kind} failed: {type(error).__name__}: {error}")

    def on_timer(self):
        self.consume_future()
        now = time.monotonic()
        if self.future is not None and self.future_started is not None:
            if now - self.future_started > self.inference_timeout and not self.future_timeout_reported:
                self.future_timeout_reported = True
                self.fail_safe_stop(f"Uni-NaVid {self.future_kind} timed out")
            return
        if self.waiting_for_trajectory:
            if self.trajectory_started is not None and now - self.trajectory_started > self.trajectory_timeout:
                self.fail_safe_stop("Uni-NaVid trajectory completion timed out")
            return
        if not (self.episode_active and self.cache_ready and self.instruction and self.pending_frames):
            return
        snapshot = {
            "frames": self.pending_frames[:],
            "instruction": self.instruction,
            "episode_id": self.episode_id,
            "generation": self.generation,
            "request_id": f"{self.generation}-{uuid.uuid4().hex}",
        }
        self.pending_frames.clear()
        self.submit_future("step", self.generation, self.request_step, snapshot)

    def destroy_node(self):
        self.http_executor.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = UniNaVidAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
