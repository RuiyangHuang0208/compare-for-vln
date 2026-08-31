from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import math
import time
import uuid

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String
from vln_interfaces.msg import NavigationCommand

from .action_parser import ActionParseError, is_model_stop, parse_action
from .contracts import (
    DEPTH_FRAMES,
    EXPECTED_VECTOR_LENGTH,
    HISTORY_RGB_FRAMES,
    POINTCLOUD_POINTS,
    RequestIdentity,
    response_is_current,
    validate_runtime_contract,
)
from .inference_client import MobileVLAR1InferenceClient
from .pointcloud_from_depth import PointCloudError, pointcloud_from_depth
from .sensor_sync import ApproximateSensorSynchronizer, StampedValue, sample_rgb_history


def stamp_seconds(message) -> float:
    return float(message.header.stamp.sec) + float(message.header.stamp.nanosec) * 1.0e-9


def decode_rgb(message: Image) -> np.ndarray:
    if message.encoding.lower() != "rgb8":
        raise ValueError(f"RGB encoding must be rgb8, got {message.encoding!r}")
    rows = np.frombuffer(message.data, np.uint8).reshape(message.height, message.step)
    return np.ascontiguousarray(rows[:, : message.width * 3].reshape(message.height, message.width, 3))


def decode_depth_m(message: Image) -> np.ndarray:
    if message.encoding.upper() != "32FC1":
        raise ValueError(f"depth encoding must be 32FC1 meters, got {message.encoding!r}")
    row_values = message.step // np.dtype(np.float32).itemsize
    rows = np.frombuffer(message.data, np.float32).reshape(message.height, row_values)
    return np.ascontiguousarray(rows[:, : message.width], dtype=np.float32)


class MobileVLAR1Adapter(Node):
    def __init__(self, **kwargs):
        super().__init__("mobilevla_r1_adapter", **kwargs)
        defaults = {
            "model.repository_path": "third_party/MobileVLA-R1",
            "model.checkpoint_path": "checkpoints/vln/mobilevla_r1/MobileVLA-R1/weight/rl",
            "model.server_url": "http://127.0.0.1:5806",
            "input.rgb_topic": "/camera/rgb/image_raw",
            "input.depth_topic": "/camera/depth/image_raw",
            "input.camera_info_topic": "/camera/rgb/camera_info",
            "input.instruction_topic": "/vln/instruction",
            "input.episode_topic": "/episode/state",
            "input.episode_id_topic": "/episode/id",
            "input.sensor_profile": "rgb_d_pointcloud_from_depth",
            "input.sync_queue_size": 10,
            "input.sync_slop_s": 0.05,
            "input.maximum_sensor_age_s": 0.25,
            "input.history_frames": HISTORY_RGB_FRAMES,
            "input.depth_frames": DEPTH_FRAMES,
            "input.depth_unit": "m",
            "input.pointcloud_points": POINTCLOUD_POINTS,
            "input.pointcloud_normalize": True,
            "input.pointcloud_seed": 0,
            "parser.expected_vector_length": EXPECTED_VECTOR_LENGTH,
            "control.command_duration_s": 0.0,
            "control.velocity_units_confirmed": False,
            "control.coordinate_signs_confirmed": False,
            "output.command_topic": "/vln/command",
            "output.raw_response_topic": "/vln/mobilevla_r1/raw_response",
            "output.parsed_velocity_topic": "/vln/mobilevla_r1/parsed_velocity",
            "output.latency_topic": "/vln/inference_latency",
            "output.model_latency_topic": "/vln/mobilevla_r1/inference_latency",
            "output.status_topic": "/vln/mobilevla_r1/status",
            "output.frame_id": "base_link",
            "evaluation.mode": "native_output",
            "evaluation.direct_velocity": True,
            "evaluation.external_local_avoidance": False,
            "runtime.asynchronous": True,
            "runtime.max_inflight_requests": 1,
            "runtime.inference_timeout_s": 60.0,
            "runtime.command_timeout_s": 0.5,
            "runtime.poll_rate_hz": 20.0,
            "runtime.allow_stub_server": False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        p = lambda name: self.get_parameter(name).value

        self.allow_stub = bool(p("runtime.allow_stub_server"))
        self.command_duration = float(p("control.command_duration_s"))
        validate_runtime_contract(
            history_frames=int(p("input.history_frames")),
            depth_frames=int(p("input.depth_frames")),
            pointcloud_points=int(p("input.pointcloud_points")),
            expected_vector_length=int(p("parser.expected_vector_length")),
            command_duration_s=self.command_duration,
            allow_stub=self.allow_stub,
        )
        if str(p("input.sensor_profile")) != "rgb_d_pointcloud_from_depth":
            raise ValueError("MobileVLA-R1 requires sensor_profile=rgb_d_pointcloud_from_depth")
        if str(p("input.depth_unit")) != "m":
            raise ValueError("Isaac bridge publishes 32FC1 depth in meters")
        if str(p("evaluation.mode")) != "native_output" or not bool(p("evaluation.direct_velocity")):
            raise ValueError("MobileVLA-R1 requires native_output and direct_velocity=true")
        if bool(p("evaluation.external_local_avoidance")):
            raise ValueError("external local avoidance must remain disabled")
        if not bool(p("runtime.asynchronous")) or int(p("runtime.max_inflight_requests")) != 1:
            raise ValueError("exactly one asynchronous inference request is required")
        if not self.allow_stub and not (
            bool(p("control.velocity_units_confirmed")) and bool(p("control.coordinate_signs_confirmed"))
        ):
            raise ValueError(
                "official public code does not establish physical velocity units/signs; real benchmark remains disabled"
            )
        command_timeout = float(p("runtime.command_timeout_s"))
        if self.command_duration > command_timeout:
            raise ValueError("command_duration_s cannot exceed shared navigation_bridge command timeout")

        self.frame_id = str(p("output.frame_id"))
        self.history_count = int(p("input.history_frames"))
        self.point_count = int(p("input.pointcloud_points"))
        self.point_normalize = bool(p("input.pointcloud_normalize"))
        self.point_seed = int(p("input.pointcloud_seed"))
        self.timeout = float(p("runtime.inference_timeout_s"))
        self.synchronizer = ApproximateSensorSynchronizer(
            int(p("input.sync_queue_size")),
            float(p("input.sync_slop_s")),
            float(p("input.maximum_sensor_age_s")),
        )
        self.client = MobileVLAR1InferenceClient(str(p("model.server_url")), self.timeout)
        self.http_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mobilevla-r1-http")
        self.future = None
        self.future_kind = None
        self.future_started = None
        self.future_identity = None
        self.future_timeout_reported = False
        self.generation = 0
        self.episode_id = "unassigned"
        self.episode_active = False
        self.instruction = ""
        self.server_ready = False
        self.latest_bundle = None
        self.history = deque()
        self.bundle_sequence = 0
        self.submitted_sequence = 0
        self.command_expires = None
        self.stop_sent_for_command = True

        self.command_pub = self.create_publisher(NavigationCommand, str(p("output.command_topic")), 10)
        self.raw_pub = self.create_publisher(String, str(p("output.raw_response_topic")), 10)
        self.velocity_pub = self.create_publisher(Float32MultiArray, str(p("output.parsed_velocity_topic")), 10)
        self.latency_pub = self.create_publisher(Float32MultiArray, str(p("output.latency_topic")), 10)
        self.model_latency_pub = self.create_publisher(
            Float32MultiArray, str(p("output.model_latency_topic")), 10
        )
        self.status_pub = self.create_publisher(String, str(p("output.status_topic")), 10)
        self.create_subscription(Image, str(p("input.rgb_topic")), self.on_rgb, 5)
        self.create_subscription(Image, str(p("input.depth_topic")), self.on_depth, 5)
        self.create_subscription(CameraInfo, str(p("input.camera_info_topic")), self.on_camera_info, 5)
        self.create_subscription(String, str(p("input.instruction_topic")), self.on_instruction, 10)
        self.create_subscription(String, str(p("input.episode_topic")), self.on_episode_state, 10)
        self.create_subscription(String, str(p("input.episode_id_topic")), self.on_episode_id, 10)
        self.create_timer(1.0 / float(p("runtime.poll_rate_hz")), self.on_timer)
        self.publish_stop("startup", valid=True)

    def clock_seconds(self):
        return self.get_clock().now().nanoseconds * 1.0e-9

    def add_sensor(self, kind, stamp, value):
        if not self.episode_active:
            return
        bundle = self.synchronizer.add(
            kind,
            StampedValue(stamp, self.episode_id, value),
            self.clock_seconds(),
        )
        if bundle is not None:
            self.latest_bundle = bundle
            self.history.append(bundle.rgb.value)
            self.bundle_sequence += 1
        elif self.synchronizer.last_rejection_reason:
            self.safety_stop(self.synchronizer.last_rejection_reason)

    def on_rgb(self, message):
        try:
            self.add_sensor("rgb", stamp_seconds(message), decode_rgb(message))
        except ValueError as error:
            self.safety_stop("invalid_rgb", str(error))

    def on_depth(self, message):
        try:
            self.add_sensor("depth", stamp_seconds(message), decode_depth_m(message))
        except ValueError as error:
            self.safety_stop("invalid_depth", str(error))

    def on_camera_info(self, message):
        self.add_sensor("camera_info", stamp_seconds(message), np.asarray(message.k, np.float64).reshape(3, 3))

    def on_instruction(self, message):
        instruction = message.data.strip()
        changed = bool(instruction) and instruction != self.instruction
        self.instruction = instruction
        if changed and self.episode_active:
            self.reset_local("new instruction")
            self.episode_active = True
            self.schedule_reset()

    def on_episode_id(self, message):
        if message.data.strip():
            self.episode_id = message.data.strip()

    def on_episode_state(self, message):
        state = message.data.strip().upper()
        if state == "START":
            self.reset_local("episode START")
            self.episode_active = True
            self.schedule_reset()
        elif state in {"RESET", "FINISH", "FAILED", "SUCCESS"}:
            self.episode_active = False
            self.reset_local(f"episode {state}", clear_instruction=True)

    def reset_local(self, reason, clear_instruction=False):
        self.generation += 1
        self.synchronizer.clear()
        self.latest_bundle = None
        self.history.clear()
        self.bundle_sequence = 0
        self.submitted_sequence = 0
        self.command_expires = None
        self.stop_sent_for_command = True
        self.server_ready = False
        if clear_instruction:
            self.instruction = ""
        self.publish_stop("reset", valid=True, detail=reason)

    def submit(self, kind, function, *args, identity=None):
        if self.future is not None:
            return False
        self.future = self.http_executor.submit(function, *args)
        self.future_kind = kind
        self.future_started = time.monotonic()
        self.future_identity = identity
        self.future_timeout_reported = False
        return True

    def schedule_reset(self):
        if self.future is None:
            self.submit("reset", self.reset_service, self.episode_id, self.generation)

    def reset_service(self, episode_id, generation):
        health = self.client.health()
        variant = str(health.get("variant", ""))
        if variant != "official" and not (self.allow_stub and variant == "stub"):
            raise ValueError(f"unexpected inference server variant {variant!r}")
        for key, expected in (("history_frames", 8), ("depth_frames", 1), ("pointcloud_points", 2048), ("expected_vector_length", 12)):
            if int(health.get(key, -1)) != expected:
                raise ValueError(f"server {key}={health.get(key)!r}, expected {expected}")
        return self.client.reset(episode_id, generation)

    def request_step(self, rgb_history, depth, pointcloud, identity, sensor_stamp):
        metadata = {
            "instruction": self.instruction,
            "episode_id": identity.episode_id,
            "generation": identity.generation,
            "request_id": identity.request_id,
            "sensor_stamp": sensor_stamp,
            "history_frames": self.history_count,
            "depth_frames": 1,
            "pointcloud_points": self.point_count,
            "depth_unit": "m",
        }
        return self.client.step(rgb_history, depth, pointcloud, metadata)

    def publish_status(self, reason, detail="", **metadata):
        payload = {"reason": reason, "detail": detail, **metadata}
        self.status_pub.publish(String(data=json.dumps(payload, separators=(",", ":"), allow_nan=False)))

    def publish_stop(self, reason, valid=False, detail=""):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.STOP
        message.valid = bool(valid)
        message.confidence = 1.0 if valid else 0.0
        self.command_pub.publish(message)
        self.publish_status(reason, detail)

    def safety_stop(self, reason, detail=""):
        self.command_expires = None
        self.stop_sent_for_command = True
        self.publish_stop(reason, valid=False, detail=detail)

    def publish_velocity(self, velocity, full_vector):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.command_type = NavigationCommand.VELOCITY
        message.velocity.linear.x = float(velocity[0])
        message.velocity.linear.y = float(velocity[1])
        message.velocity.angular.z = float(velocity[2])
        message.valid = True
        message.confidence = 1.0
        self.command_pub.publish(message)
        self.velocity_pub.publish(Float32MultiArray(data=list(map(float, velocity))))
        self.command_expires = time.monotonic() + self.command_duration
        self.stop_sent_for_command = False
        self.publish_status(
            "velocity_command",
            full_action_vector=list(map(float, full_vector)),
            forwarded_indices=[0, 1, 2],
            forwarded_velocity=list(map(float, velocity)),
        )

    def handle_step(self, response, identity):
        if not response_is_current(response, identity) or identity.generation != self.generation:
            self.safety_stop("stale_response")
            return
        raw = str(response.get("raw_response", ""))
        self.raw_pub.publish(String(data=raw))
        try:
            action = parse_action(raw)
        except ActionParseError as error:
            self.safety_stop(error.reason, str(error))
            return
        latency = Float32MultiArray(data=[float(response["inference_latency"]), math.nan, math.nan])
        self.latency_pub.publish(latency)
        self.model_latency_pub.publish(latency)
        if is_model_stop(action.velocity):
            self.publish_stop("model_requested_stop", valid=True, detail="valid zero velocity")
            self.publish_status(
                "model_requested_stop",
                full_action_vector=list(action.vector),
                service_latency_s=float(response.get("service_latency", 0.0)),
                peak_gpu_memory_bytes=int(response.get("peak_gpu_memory_bytes", 0)),
            )
            return
        self.publish_velocity(action.velocity, action.vector)
        self.publish_status(
            "velocity_command_audit",
            full_action_vector=list(action.vector),
            forwarded_indices=[0, 1, 2],
            forwarded_velocity=list(action.velocity),
            inference_latency_s=float(response["inference_latency"]),
            service_latency_s=float(response.get("service_latency", 0.0)),
            peak_gpu_memory_bytes=int(response.get("peak_gpu_memory_bytes", 0)),
        )

    def poll_future(self):
        if self.future is None:
            return
        if not self.future.done():
            if (
                not self.future_timeout_reported
                and time.monotonic() - self.future_started > self.timeout
            ):
                self.future_timeout_reported = True
                self.safety_stop("timeout_stop")
            return
        future, kind, identity = self.future, self.future_kind, self.future_identity
        self.future = self.future_kind = self.future_identity = self.future_started = None
        self.future_timeout_reported = False
        try:
            response = future.result()
            if kind == "reset":
                self.server_ready = True
            else:
                self.handle_step(response, identity)
        except Exception as error:
            self.safety_stop("service_error", f"{type(error).__name__}: {error}")

    def maybe_submit_step(self):
        if (
            not self.episode_active
            or not self.server_ready
            or not self.instruction
            or self.latest_bundle is None
            or self.future is not None
            or self.bundle_sequence <= self.submitted_sequence
            or (self.command_expires is not None and time.monotonic() < self.command_expires)
        ):
            return
        try:
            rgb_history = np.stack(sample_rgb_history(self.history, self.history_count))
            depth = self.latest_bundle.depth.value
            pointcloud = pointcloud_from_depth(
                depth,
                self.latest_bundle.camera_info.value,
                max_points=self.point_count,
                normalize=self.point_normalize,
                seed=self.point_seed,
            )
        except (ValueError, PointCloudError) as error:
            self.safety_stop("sensor_error", str(error))
            return
        identity = RequestIdentity(self.episode_id, self.generation, uuid.uuid4().hex)
        self.submitted_sequence = self.bundle_sequence
        self.submit(
            "step",
            self.request_step,
            rgb_history,
            depth,
            pointcloud,
            identity,
            self.latest_bundle.stamp_s,
            identity=identity,
        )

    def on_timer(self):
        self.poll_future()
        sensor_failure = self.synchronizer.pending_failure(self.clock_seconds())
        if self.episode_active and sensor_failure:
            self.synchronizer.clear()
            self.safety_stop(sensor_failure)
        if self.command_expires is not None and time.monotonic() >= self.command_expires:
            self.command_expires = None
            if not self.stop_sent_for_command:
                self.stop_sent_for_command = True
                self.publish_stop("command_window_expired", valid=True)
        self.maybe_submit_step()

    def close(self):
        self.publish_stop("shutdown", valid=True)
        self.http_executor.shutdown(wait=False, cancel_futures=True)


def main(args=None):
    rclpy.init(args=args)
    node = MobileVLAR1Adapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
