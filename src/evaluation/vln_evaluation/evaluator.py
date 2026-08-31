from __future__ import annotations

import csv
import json
import math
import os
import time

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool, Float32MultiArray, String

from .metrics import MetricsAccumulator, classify_failure


def watchdog_expired(
    max_duration,
    sim_elapsed,
    wall_elapsed,
    clock_progress_age,
    wall_timeout_scale,
    clock_stall_timeout,
):
    """Use simulation time while /clock advances, with a bounded stalled-clock fallback."""
    if max_duration <= 0.0:
        return False
    if sim_elapsed is not None and sim_elapsed > max_duration:
        return True
    clock_stalled = clock_progress_age is None or clock_progress_age > clock_stall_timeout
    return bool(clock_stalled and wall_elapsed > max_duration * wall_timeout_scale)


def append_summary_row(summary_path, row):
    exists = os.path.isfile(summary_path) and os.path.getsize(summary_path) > 0
    existing_rows = []
    existing_fields = []
    if exists:
        with open(summary_path, newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            existing_fields = list(reader.fieldnames or [])
            existing_rows = [
                {key: value for key, value in existing.items() if key is not None}
                for existing in reader
            ]
    fields = existing_fields + [key for key in row if key not in existing_fields]
    if not fields:
        fields = list(row)
    rewrite = not exists or fields != existing_fields
    if rewrite:
        temporary = summary_path + ".tmp"
        with open(temporary, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(row)
        os.replace(temporary, summary_path)
    else:
        with open(summary_path, "a", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writerow(row)


class Evaluator(Node):
    def __init__(self, **kwargs):
        super().__init__("vln_evaluator", **kwargs)
        self.declare_parameter("model_name", "dummy")
        self.declare_parameter("experiment_name", "single_episode")
        self.declare_parameter("output_root", "outputs")
        self.declare_parameter("direct_velocity", False)
        self.declare_parameter("evaluation_mode", "trajectory_normalized")
        self.declare_parameter("shutdown_after_save", True)
        self.declare_parameter("wall_timeout_scale", 2.0)
        self.declare_parameter("clock_stall_timeout", 5.0)
        self.model_name = str(self.get_parameter("model_name").value)
        self.experiment_name = str(self.get_parameter("experiment_name").value)
        self.output_root = os.path.abspath(str(self.get_parameter("output_root").value))
        self.metadata = {}
        self.navigation_error = math.inf
        self.accumulator = None
        self.sim_time = None
        self.started_sim_time = None
        self.started_wall_time = None
        self.last_clock_wall_time = None
        self.last_clock_progress_wall_time = None
        self.watchdog_requested = False
        self.model_stop_requested = False
        self.wall_timeout_scale = float(self.get_parameter("wall_timeout_scale").value)
        self.clock_stall_timeout = float(self.get_parameter("clock_stall_timeout").value)
        if not math.isfinite(self.wall_timeout_scale) or self.wall_timeout_scale <= 1.0:
            raise ValueError("wall_timeout_scale must be finite and greater than 1")
        if not math.isfinite(self.clock_stall_timeout) or self.clock_stall_timeout <= 0.0:
            raise ValueError("clock_stall_timeout must be positive and finite")
        self.request_pub = self.create_publisher(String, "/episode/request", 10)
        self.create_subscription(String, "/episode/metadata", self.on_metadata, 10)
        self.create_subscription(String, "/episode/state", self.on_state, 10)
        self.create_subscription(String, "/episode/navigation_error", self.on_error, 10)
        self.create_subscription(String, "/episode/model_stop", self.on_model_stop, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self.on_odom, 20)
        self.create_subscription(Twist, "/nav_vel", self.on_control, 20)
        self.create_subscription(Float32MultiArray, "/vln/inference_latency", self.on_latency, 10)
        self.create_subscription(Bool, "/navigation/stuck", self.on_stuck, 10)
        self.create_subscription(String, "/simulation/collision", self.on_collision, 50)
        self.create_subscription(Clock, "/clock", self.on_clock, 20)
        self.create_timer(0.2, self.watchdog)
        self.shutdown_requested = False

    def on_metadata(self, message):
        try:
            self.metadata = json.loads(message.data)
        except json.JSONDecodeError:
            self.get_logger().warning("Rejected malformed /episode/metadata")

    def on_state(self, message):
        state = message.data.strip().upper()
        if state == "RESET":
            self.accumulator = None
            self.navigation_error = math.inf
            self.started_sim_time = None
            self.started_wall_time = None
            self.watchdog_requested = False
            self.model_stop_requested = False
        elif state == "START":
            self.model_stop_requested = False
            self.accumulator = MetricsAccumulator(
                self.model_name,
                self.experiment_name,
                bool(self.get_parameter("direct_velocity").value),
            )
            self.accumulator.start(self.metadata)
            self.started_sim_time = self.sim_time
            self.started_wall_time = time.monotonic()
            self.last_clock_progress_wall_time = self.started_wall_time
            self.watchdog_requested = False
        elif state in {"SUCCESS", "FAILED", "FINISH"} and self.accumulator is not None:
            reason = "timeout" if state == "FAILED" and self.watchdog_requested else state.lower()
            self.save(state == "SUCCESS", reason)

    def on_error(self, message):
        try:
            value = float(message.data)
            if math.isfinite(value):
                self.navigation_error = value
        except ValueError:
            pass

    def on_model_stop(self, _message):
        self.model_stop_requested = True

    def on_odom(self, message):
        if self.accumulator is not None:
            self.accumulator.add_pose(
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                self.sim_time,
            )

    def on_clock(self, message):
        value = float(message.clock.sec) + float(message.clock.nanosec) * 1.0e-9
        now = time.monotonic()
        if self.sim_time is None or value > self.sim_time + 1.0e-9:
            self.last_clock_progress_wall_time = now
        self.sim_time = value
        self.last_clock_wall_time = now

    def on_control(self, _message):
        if self.accumulator is not None:
            self.accumulator.add_control()

    def on_latency(self, message):
        if self.accumulator is not None and message.data:
            vlm = message.data[1] if len(message.data) > 1 else math.nan
            action_expert = message.data[2] if len(message.data) > 2 else math.nan
            self.accumulator.add_latency(message.data[0], vlm, action_expert)

    def on_stuck(self, message):
        if self.accumulator is not None:
            self.accumulator.set_stuck(message.data)

    def on_collision(self, message):
        if self.accumulator is None:
            return
        try:
            event = json.loads(message.data)
            required = ("time", "force", "object", "is_pedestrian", "episode_failure")
            if not all(key in event for key in required):
                raise ValueError("missing collision fields")
            self.accumulator.add_collision(event)
            if event["episode_failure"]:
                self.request_pub.publish(String(data="FAILED"))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            self.get_logger().warning(f"Rejected collision event: {error}")

    def watchdog(self):
        if self.accumulator is None or self.accumulator.started is None or self.watchdog_requested:
            return
        max_duration = float(self.accumulator.metadata.get("max_duration", 0.0))
        if max_duration > 0.0:
            now = time.monotonic()
            wall_elapsed = now - (self.started_wall_time or self.accumulator.started)
            if self.sim_time is not None and self.started_sim_time is not None:
                elapsed = self.sim_time - self.started_sim_time
            else:
                elapsed = 0.0
            # Isaac Sim can stop publishing /clock while a heavy USD payload is loading.
            # Keep simulation time as the primary metric, but never leave an episode
            # without a result when the wall clock has exceeded the bounded fallback.
            progress_age = (
                None
                if self.last_clock_progress_wall_time is None
                else now - self.last_clock_progress_wall_time
            )
            if watchdog_expired(
                max_duration,
                elapsed,
                wall_elapsed,
                progress_age,
                self.wall_timeout_scale,
                self.clock_stall_timeout,
            ):
                self.watchdog_requested = True
                self.request_pub.publish(String(data="FAILED"))

    def save(self, success, termination_reason):
        failure_attribution = classify_failure(
            success,
            watchdog_requested=self.watchdog_requested,
            model_stop_requested=self.model_stop_requested,
            physical_collision_count=self.accumulator.physical_collision_count,
            stuck_count=self.accumulator.stuck_count,
        )
        if failure_attribution == "model_behavior":
            termination_reason = "model_stop_before_goal"
        duration = None
        if self.sim_time is not None and self.started_sim_time is not None:
            elapsed = self.sim_time - self.started_sim_time
            if elapsed >= 0.0:
                duration = elapsed
        result = self.accumulator.finalize(
            success,
            self.navigation_error,
            duration=duration,
            termination_reason=termination_reason,
            failure_attribution=failure_attribution,
        )
        episode_id = str(result["episode_id"])
        episode_dir = os.path.join(self.output_root, self.model_name, self.experiment_name)
        os.makedirs(episode_dir, exist_ok=True)
        json_path = os.path.join(episode_dir, f"{episode_id}.json")
        with open(json_path, "w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        summary_dir = os.path.join(self.output_root, self.experiment_name)
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, "summary.csv")
        row = {key: value for key, value in result.items() if key != "collision_events"}
        row["model_inputs"] = json.dumps(row.get("model_inputs", []), separators=(",", ":"))
        row["derived_inputs"] = json.dumps(row.get("derived_inputs", []), separators=(",", ":"))
        row["model_runtime"] = json.dumps(row.get("model_runtime", {}), separators=(",", ":"))
        append_summary_row(summary_path, row)
        self.get_logger().info(f"Saved episode result: {json_path}")
        self.accumulator = None
        self.watchdog_requested = False
        self.model_stop_requested = False
        self.started_wall_time = None
        if bool(self.get_parameter("shutdown_after_save").value):
            self.shutdown_requested = True
            self.create_timer(0.25, self.shutdown_once)

    def shutdown_once(self):
        if self.shutdown_requested and rclpy.ok():
            self.shutdown_requested = False
            self.get_logger().info("Episode result saved; shutting down single-episode process")
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = Evaluator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
