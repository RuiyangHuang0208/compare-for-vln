from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import math
import time
import uuid

from geometry_msgs.msg import Pose2D, PoseStamped
from nav_msgs.msg import Path
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, Header, MultiArrayDimension, String
from vln_interfaces.msg import NavigationCommand

from .contracts import (
    ACTION_SHAPE,
    CHECKPOINT_VARIANT,
    LANGUAGE_ONLY_MODALITY_ID,
    response_is_current,
    validate_language_only_contract,
)
from .inference_client import OmniVLAInferenceClient
from .trajectory_converter import (
    TrajectoryConversionError,
    convert_raw_trajectory,
    convert_raw_waypoint_to_velocity,
)


def decode_rgb(message: Image) -> np.ndarray:
    encoding = message.encoding.lower()
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4}.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported RGB encoding {message.encoding!r}")
    rows = np.frombuffer(message.data, dtype=np.uint8).reshape(message.height, message.step)
    image = rows[:, : message.width * channels].reshape(message.height, message.width, channels)
    if encoding.startswith("bgr"):
        image = image[..., [2, 1, 0] + ([3] if channels == 4 else [])]
    return np.ascontiguousarray(image[..., :3])


class OmniVLAAdapter(Node):
    def __init__(self, **kwargs):
        super().__init__("omnivla_adapter", **kwargs)
        defaults = {
            "model.repository_path": "third_party/OmniVLA",
            "model.checkpoint_path": "checkpoints/vln/omnivla/omnivla-original",
            "model.server_url": "http://127.0.0.1:5805",
            "model.checkpoint_variant": CHECKPOINT_VARIANT,
            "model.resume_step": 120000,
            "model.device": "cuda:0",
            "model.dtype": "bfloat16",
            "model.num_images_in_input": 2,
            "model.use_l1_regression": True,
            "model.use_diffusion": False,
            "model.use_film": False,
            "goal.profile": "language_only",
            "goal.modality_id": LANGUAGE_ONLY_MODALITY_ID,
            "goal.language_enabled": True,
            "goal.pose_enabled": False,
            "goal.image_enabled": False,
            "goal.satellite_enabled": False,
            "goal.placeholder_goal_image": "black",
            "goal.placeholder_goal_pose": [0.0, 0.0, 0.0, 0.0],
            "input.image_topic": "/camera/rgb/image_raw",
            "input.instruction_topic": "/vln/instruction",
            "input.episode_topic": "/episode/state",
            "input.episode_id_topic": "/episode/id",
            "input.sensor_profile": "rgb_only",
            "input.image_width": 640,
            "input.image_height": 480,
            "input.inference_rate_hz": 3.0,
            "action.chunk_size": 8,
            "action.action_dim": 4,
            "action.action_rate_hz": 3.0,
            "action.xy_scale_m": 0.1,
            "action.use_heading": False,
            "action.heading_epsilon": 1.0e-6,
            "action.max_abs_coordinate_m": 20.0,
            "output.command_topic": "/vln/command",
            "output.raw_trajectory_topic": "/vln/omnivla/raw_trajectory",
            "output.metadata_topic": "/vln/omnivla/metadata",
            "output.debug_path_topic": "/vln/debug_path",
            "output.latency_topic": "/vln/inference_latency",
            "output.frame_id": "base_link",
            "evaluation.mode": "trajectory_normalized",
            "evaluation.direct_velocity": False,
            "evaluation.external_local_avoidance": False,
            "runtime.asynchronous": True,
            "runtime.max_inflight_requests": 1,
            "runtime.inference_timeout": 30.0,
            "runtime.command_timeout": 0.5,
            "runtime.reject_stale_responses": True,
            "runtime.poll_rate_hz": 20.0,
            # START, episode id and instruction are published on separate DDS
            # topics.  Give the latched id/instruction a short settling window
            # before issuing the server reset, otherwise a new process can
            # reset OmniVLA with the previous episode's values.
            "runtime.episode_start_settle_time": 0.5,
            "runtime.allow_stub_server": False,
            "native_output.waypoint_index": 4,
            "native_output.controller_dt": 1.0 / 3.0,
            "native_output.max_linear_before_limit": 0.5,
            "native_output.max_angular_before_limit": 1.0,
            "native_output.max_v": 0.3,
            "native_output.max_w": 0.3,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value

        validate_language_only_contract(str(p("goal.profile")), int(p("goal.modality_id")))
        if str(p("input.sensor_profile")) != "rgb_only":
            raise ValueError("OmniVLA only permits sensor_profile=rgb_only")
        if str(p("model.checkpoint_variant")) != CHECKPOINT_VARIANT or int(p("model.resume_step")) != 120000:
            raise ValueError("OmniVLA adapter is fixed to omnivla-original step 120000")
        if (
            int(p("model.num_images_in_input")) != 2
            or not bool(p("model.use_l1_regression"))
            or bool(p("model.use_diffusion"))
            or bool(p("model.use_film"))
        ):
            raise ValueError("OmniVLA must use two images, L1 action head, no diffusion, and no FiLM")
        if not (
            bool(p("goal.language_enabled"))
            and not bool(p("goal.pose_enabled"))
            and not bool(p("goal.image_enabled"))
            and not bool(p("goal.satellite_enabled"))
            and str(p("goal.placeholder_goal_image")) == "black"
            and list(p("goal.placeholder_goal_pose")) == [0.0, 0.0, 0.0, 0.0]
        ):
            raise ValueError("OmniVLA language-only placeholders/modalities do not match the fixed contract")
        self.evaluation_mode = str(p("evaluation.mode"))
        self.direct_velocity = bool(p("evaluation.direct_velocity"))
        if self.evaluation_mode not in {"trajectory_normalized", "native_output"}:
            raise ValueError("OmniVLA evaluation.mode must be trajectory_normalized or native_output")
        if self.direct_velocity != (self.evaluation_mode == "native_output"):
            raise ValueError("OmniVLA direct_velocity must match evaluation.mode")
        if bool(p("evaluation.external_local_avoidance")):
            raise ValueError("external local avoidance must remain disabled")
        if not bool(p("runtime.asynchronous")) or int(p("runtime.max_inflight_requests")) != 1:
            raise ValueError("OmniVLA requires one asynchronous in-flight request")
        if not bool(p("runtime.reject_stale_responses")):
            raise ValueError("OmniVLA must reject stale responses")
        if (int(p("action.chunk_size")), int(p("action.action_dim"))) != ACTION_SHAPE:
            raise ValueError("OmniVLA action contract must remain 8x4")

        self.frame_id = str(p("output.frame_id"))
        self.width = int(p("input.image_width"))
        self.height = int(p("input.image_height"))
        self.rate = float(p("input.inference_rate_hz"))
        self.action_rate = float(p("action.action_rate_hz"))
        self.xy_scale = float(p("action.xy_scale_m"))
        self.use_heading = bool(p("action.use_heading"))
        self.heading_epsilon = float(p("action.heading_epsilon"))
        self.max_coordinate = float(p("action.max_abs_coordinate_m"))
        self.native_waypoint_index = int(p("native_output.waypoint_index"))
        self.native_dt = float(p("native_output.controller_dt"))
        self.native_max_linear = float(p("native_output.max_linear_before_limit"))
        self.native_max_angular = float(p("native_output.max_angular_before_limit"))
        self.native_max_v = float(p("native_output.max_v"))
        self.native_max_w = float(p("native_output.max_w"))
        if not 0 <= self.native_waypoint_index < ACTION_SHAPE[0]:
            raise ValueError("native_output.waypoint_index is outside the 8-point action chunk")
        if min(self.native_dt, self.native_max_linear, self.native_max_angular, self.native_max_v, self.native_max_w) <= 0.0:
            raise ValueError("native output timing and limits must be positive")
        self.timeout = float(p("runtime.inference_timeout"))
        self.episode_start_settle_time = float(p("runtime.episode_start_settle_time"))
        self.allow_stub = bool(p("runtime.allow_stub_server"))
        if min(self.rate, self.action_rate, self.timeout) <= 0.0 or self.episode_start_settle_time < 0.0:
            raise ValueError("inference/action rates and timeout must be positive; settle time must be non-negative")

        self.client = OmniVLAInferenceClient(str(p("model.server_url")), self.timeout)
        self.http_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnivla-http")
        self.future = None
        self.future_kind = None
        self.future_generation = None
        self.future_started = None
        self.future_timed_out = False
        self.reset_pending = False
        self.generation = 0
        self.episode_id = "unassigned"
        self.episode_active = False
        self.episode_started_once = False
        self.server_ready = False
        self.instruction = ""
        # DDS does not guarantee ordering across the separate /episode/id,
        # /vln/instruction and /episode/state topics.  Do not start a reset
        # with the placeholder ``unassigned`` episode id when START arrives
        # before the latched id/instruction messages.
        self.episode_start_pending = False
        self.start_not_before = 0.0
        self.latest_image = None
        self.latest_image_sequence = 0
        self.last_submitted_sequence = 0
        self.last_request_time = 0.0
        self.last_result_time = None

        self.command_pub = self.create_publisher(NavigationCommand, str(p("output.command_topic")), 10)
        self.raw_pub = self.create_publisher(Float32MultiArray, str(p("output.raw_trajectory_topic")), 10)
        self.metadata_pub = self.create_publisher(String, str(p("output.metadata_topic")), 10)
        self.path_pub = self.create_publisher(Path, str(p("output.debug_path_topic")), 10)
        self.latency_pub = self.create_publisher(Float32MultiArray, str(p("output.latency_topic")), 10)
        self.create_subscription(Image, str(p("input.image_topic")), self.on_image, 5)
        self.create_subscription(String, str(p("input.instruction_topic")), self.on_instruction, 10)
        self.create_subscription(String, str(p("input.episode_topic")), self.on_episode_state, 10)
        self.create_subscription(String, str(p("input.episode_id_topic")), self.on_episode_id, 10)
        self.create_timer(1.0 / float(p("runtime.poll_rate_hz")), self.on_timer)
        self.publish_stop()
        self.get_logger().info(
            f"OmniVLA RGB/language-only adapter waiting for {self.client.server_url}; "
            f"modality_id=7 mode={self.evaluation_mode} direct_velocity={self.direct_velocity}"
        )

    def on_image(self, message):
        if not self.episode_active:
            return
        if int(message.width) != self.width or int(message.height) != self.height:
            self.get_logger().warning(
                f"rejected RGB {message.width}x{message.height}; expected {self.width}x{self.height}"
            )
            return
        try:
            self.latest_image = decode_rgb(message)
            self.latest_image_sequence += 1
        except ValueError as error:
            self.get_logger().warning(str(error))

    def on_instruction(self, message):
        value = message.data.strip()
        resume_manual = bool(value) and self.episode_started_once and not self.episode_active
        changed = bool(value) and self.episode_active and value != self.instruction
        self.instruction = value
        if not value:
            self.episode_start_pending = False
            self.start_not_before = 0.0
            self.episode_active = False
            self.reset_local("empty instruction")
        elif self.episode_start_pending:
            # START may arrive before this latched instruction.  Keep the
            # adapter stopped and let maybe_start_episode() validate both
            # topics after the settle window.
            self.maybe_start_episode()
        elif resume_manual or changed:
            self.reset_local("new instruction")
            self.episode_active = True
            self.schedule_reset()

    def on_episode_id(self, message):
        value = message.data.strip()
        if not value or value == self.episode_id:
            return
        changed_active_episode = self.episode_active
        self.episode_id = value
        if changed_active_episode:
            # A new id invalidates any in-flight request/reset.  The next
            # reset is submitted only after the new id is installed locally.
            # The old instruction is stale only in this active-episode case;
            # keeping an instruction received before the first id preserves
            # the valid DDS ordering where instruction precedes episode id.
            self.instruction = ""
            self.episode_active = False
            self.episode_start_pending = True
            self.reset_local("new episode id")
        elif self.episode_start_pending:
            self.maybe_start_episode()

    def maybe_start_episode(self):
        """Start only after START, a real episode id and instruction exist."""
        if not self.episode_start_pending or self.episode_active:
            return
        if time.monotonic() < self.start_not_before:
            return
        if self.episode_id == "unassigned" or not self.instruction:
            return
        self.episode_start_pending = False
        self.episode_active = True
        self.schedule_reset()

    def on_episode_state(self, message):
        state = message.data.strip().upper()
        if state == "START":
            self.episode_start_pending = True
            self.episode_active = False
            self.start_not_before = time.monotonic() + self.episode_start_settle_time
            self.reset_local("episode START")
            self.episode_started_once = True
            self.maybe_start_episode()
        elif state in {"RESET", "FINISH", "FAILED", "SUCCESS"}:
            self.episode_start_pending = False
            self.start_not_before = 0.0
            self.episode_active = False
            if state == "RESET":
                self.episode_started_once = False
            self.reset_local(f"episode {state}", clear_instruction=True)

    def reset_local(self, reason, clear_instruction=False):
        self.generation += 1
        self.latest_image = None
        self.latest_image_sequence = 0
        self.last_submitted_sequence = 0
        self.last_result_time = None
        self.server_ready = False
        if clear_instruction:
            self.instruction = ""
        self.publish_stop()
        self.get_logger().info(f"cleared OmniVLA frame/future state: {reason}")

    def submit(self, kind, function, *args):
        if self.future is not None:
            return False
        self.future = self.http_executor.submit(function, *args)
        self.future_kind = kind
        self.future_generation = self.generation
        self.future_started = time.monotonic()
        self.future_timed_out = False
        return True

    def reset_service(self, episode_id, generation):
        health = self.client.health()
        variant = str(health.get("variant", ""))
        if variant != CHECKPOINT_VARIANT and not (self.allow_stub and variant == "stub"):
            raise ValueError(f"server variant {variant!r} is not {CHECKPOINT_VARIANT!r}")
        if int(health.get("modality_id", -1)) != LANGUAGE_ONLY_MODALITY_ID:
            raise ValueError("server does not advertise language-only modality_id=7")
        return self.client.reset(episode_id, generation)

    def schedule_reset(self):
        if self.episode_id == "unassigned" or not self.instruction:
            return
        if self.future is None:
            self.reset_pending = False
            self.submit("reset", self.reset_service, self.episode_id, self.generation)
        else:
            self.reset_pending = True

    def request_step(self, snapshot):
        metadata = {
            "instruction": snapshot["instruction"],
            "episode_id": snapshot["episode_id"],
            "generation": snapshot["generation"],
            "request_id": snapshot["request_id"],
            "goal_profile": "language_only",
            "modality_id": LANGUAGE_ONLY_MODALITY_ID,
        }
        return self.client.step(snapshot["image"], metadata)

    def publish_stop(self):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.STOP
        message.confidence = 1.0
        message.valid = True
        self.command_pub.publish(message)

    def publish_trajectory(self, points):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.TRAJECTORY
        message.points = [Pose2D(x=float(x), y=float(y), theta=float(yaw)) for x, y, yaw in points]
        message.dt = 1.0 / self.action_rate
        # Existing adapters define horizon as dt times the number of predicted points.
        message.horizon = message.dt * len(points)
        message.confidence = 1.0
        message.valid = True
        self.command_pub.publish(message)

        self.publish_debug_path(points, message.header)

    def publish_debug_path(self, points, header=None):
        """Publish a base_link path without sending a second navigation command."""
        if header is None:
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.frame_id
        path = Path()
        path.header = header
        for x, y, yaw in points:
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.z = math.sin(float(yaw) / 2.0)
            pose.pose.orientation.w = math.cos(float(yaw) / 2.0)
            path.poses.append(pose)
        self.path_pub.publish(path)

    def publish_velocity(self, velocity):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.VELOCITY
        message.velocity.linear.x = float(velocity[0])
        message.velocity.linear.y = float(velocity[1])
        message.velocity.angular.z = float(velocity[2])
        message.confidence = 1.0
        message.valid = True
        self.command_pub.publish(message)

    def publish_raw(self, raw):
        message = Float32MultiArray()
        message.layout.dim = [
            MultiArrayDimension(label="waypoint", size=8, stride=32),
            MultiArrayDimension(label="component[x,y,heading_x,heading_y]", size=4, stride=4),
        ]
        message.data = np.asarray(raw, dtype=np.float32).reshape(-1).tolist()
        self.raw_pub.publish(message)

    def consume_future(self):
        if self.future is None or not self.future.done():
            return
        future, kind = self.future, self.future_kind
        expected_request_id = getattr(future, "expected_request_id", "")
        generation, timed_out = self.future_generation, self.future_timed_out
        self.future = self.future_kind = self.future_generation = self.future_started = None
        self.future_timed_out = False
        if timed_out or generation != self.generation:
            if self.episode_active or self.episode_start_pending:
                self.schedule_reset()
            return
        if self.reset_pending and self.episode_active:
            self.schedule_reset()
            return
        try:
            result = future.result()
            if kind == "reset":
                if int(result["generation"]) != self.generation:
                    raise ValueError("reset generation mismatch")
                self.server_ready = True
                return
            if not response_is_current(result, self.episode_id, self.generation, expected_request_id):
                raise ValueError("inference response identity mismatch")
        except Exception as error:
            self.publish_stop()
            self.get_logger().error(f"OmniVLA service failed; stopped: {type(error).__name__}: {error}")
            return

        try:
            raw = np.asarray(result["raw_trajectory_8x4"], dtype=np.float64)
            self.publish_raw(raw)
            if self.direct_velocity:
                velocity = convert_raw_waypoint_to_velocity(
                    raw,
                    waypoint_index=self.native_waypoint_index,
                    dt=self.native_dt,
                    max_linear_before_limit=self.native_max_linear,
                    max_angular_before_limit=self.native_max_angular,
                    max_v=self.native_max_v,
                    max_w=self.native_max_w,
                    xy_scale_m=self.xy_scale,
                )
                self.publish_velocity(velocity)
                # The official controller can legitimately return an all-zero
                # chunk while stopping.  The normalized converter rejects a
                # stationary path by design, so only use it for visualization
                # here and never let it suppress the official velocity.
                try:
                    points, scaled = convert_raw_trajectory(
                        raw,
                        xy_scale_m=self.xy_scale,
                        use_heading=self.use_heading,
                        heading_epsilon=self.heading_epsilon,
                        max_abs_coordinate_m=self.max_coordinate,
                    )
                except TrajectoryConversionError:
                    points, scaled = np.empty((0, 3)), np.empty((0, 4))
                self.publish_debug_path(points)
            else:
                points, scaled = convert_raw_trajectory(
                    raw,
                    xy_scale_m=self.xy_scale,
                    use_heading=self.use_heading,
                    heading_epsilon=self.heading_epsilon,
                    max_abs_coordinate_m=self.max_coordinate,
                )
                velocity = None
                self.publish_trajectory(points)
            now = time.monotonic()
            actual_hz = 0.0 if self.last_result_time is None else 1.0 / max(now - self.last_result_time, 1.0e-9)
            self.last_result_time = now
            model_latency = float(result.get("inference_latency", math.nan))
            service_latency = float(result.get("service_latency", math.nan))
            # The shared evaluator interprets elements 1 and 2 as separate VLM
            # and action-expert latencies. Full OmniVLA exposes only total model
            # latency, so leave unavailable component timings as NaN.
            self.latency_pub.publish(Float32MultiArray(data=[model_latency, math.nan, math.nan]))
            metadata = {
                "request_id": result["request_id"],
                "episode_id": result["episode_id"],
                "generation": result["generation"],
                "variant": result.get("variant"),
                "resume_step": result.get("resume_step"),
                "modality_id": result["modality_id"],
                "raw_trajectory_shape": list(raw.shape),
                "raw_trajectory_8x4": raw.tolist(),
                "scaled_trajectory_8x4": scaled.tolist(),
                "inference_latency": model_latency,
                "service_latency": service_latency,
                "actual_inference_hz": actual_hz,
                "execution_mode": self.evaluation_mode,
                "direct_velocity": self.direct_velocity,
                "native_waypoint_index": self.native_waypoint_index if self.direct_velocity else None,
                "native_velocity": None if velocity is None else velocity.tolist(),
                "native_velocity_limits": {
                    "max_v": self.native_max_v,
                    "max_w": self.native_max_w,
                } if self.direct_velocity else None,
                "peak_gpu_memory_bytes": int(result.get("peak_gpu_memory_bytes", 0)),
                "language_only_leakage_verified": bool(
                    result.get("language_only_leakage_verified", False)
                ),
                "language_only_leakage_max_abs_difference": result.get(
                    "language_only_leakage_max_abs_difference"
                ),
            }
            self.metadata_pub.publish(String(data=json.dumps(metadata, separators=(",", ":"))))
            self.get_logger().info(
                f"OmniVLA 8x4 trajectory latency={model_latency:.3f}s actual_rate={actual_hz:.2f}Hz"
            )
        except (KeyError, TypeError, ValueError, TrajectoryConversionError) as error:
            self.publish_stop()
            self.get_logger().error(f"invalid OmniVLA trajectory; stopped: {error}")

    def on_timer(self):
        if (
            self.future is not None
            and not self.future.done()
            and not self.future_timed_out
            and time.monotonic() - self.future_started > self.timeout
        ):
            self.future_timed_out = True
            self.publish_stop()
            self.get_logger().error("OmniVLA inference timeout; robot stopped and late response will be discarded")
        self.consume_future()
        if self.episode_start_pending and self.future is None:
            self.maybe_start_episode()
        if not self.episode_active or not self.instruction or not self.server_ready or self.future is not None:
            return
        now = time.monotonic()
        if now - self.last_request_time < 1.0 / self.rate:
            return
        if self.latest_image is None or self.latest_image_sequence <= self.last_submitted_sequence:
            return
        request_id = str(uuid.uuid4())
        snapshot = {
            "image": self.latest_image.copy(),
            "instruction": self.instruction,
            "episode_id": self.episode_id,
            "generation": self.generation,
            "request_id": request_id,
        }
        self.last_submitted_sequence = self.latest_image_sequence
        self.last_request_time = now
        if self.submit("step", self.request_step, snapshot):
            # Store identity locally so a service cannot satisfy a new request with an old response.
            self.future.expected_request_id = request_id

    def close(self):
        self.generation += 1
        if rclpy.ok():
            self.publish_stop()
        self.http_executor.shutdown(wait=False, cancel_futures=True)


def main(args=None):
    rclpy.init(args=args)
    node = OmniVLAAdapter()
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
