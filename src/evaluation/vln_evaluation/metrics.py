from __future__ import annotations

import math
import statistics
import time


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def classify_failure(
    success,
    watchdog_requested=False,
    model_stop_requested=False,
    physical_collision_count=0,
    stuck_count=0,
):
    """Classify task outcomes without treating infrastructure errors as model errors."""
    if success:
        return "success"
    if watchdog_requested:
        return "inconclusive_timeout"
    if model_stop_requested and physical_collision_count == 0 and stuck_count == 0:
        return "model_behavior"
    if physical_collision_count > 0:
        return "collision_or_scene_contact"
    if stuck_count > 0:
        return "controller_stuck"
    return "inconclusive"


class MetricsAccumulator:
    def __init__(self, model_name, experiment_name, direct_velocity=False):
        self.model_name = model_name
        self.experiment_name = experiment_name
        self.direct_velocity = bool(direct_velocity)
        self.metadata = {}
        self.started = None
        self.start_pose = None
        self.last_pose = None
        self.last_pose_time = None
        self.path_length = 0.0
        self.latencies = []
        self.vlm_latencies = []
        self.action_expert_latencies = []
        self.control_times = []
        self.physical_collision_count = 0
        self.pedestrian_collision_count = 0
        self.collision_events = []
        self.stuck_count = 0
        self.recovery_count = 0
        self.stuck = False

    def start(self, metadata):
        self.metadata = dict(metadata)
        self.started = time.monotonic()

    def add_pose(self, x, y, timestamp=None):
        timestamp = time.monotonic() if timestamp is None else float(timestamp)
        if self.last_pose_time is not None and timestamp - self.last_pose_time < 0.1:
            return
        point = (float(x), float(y))
        if self.start_pose is None:
            self.start_pose = point
        if self.last_pose is not None:
            distance = math.hypot(point[0] - self.last_pose[0], point[1] - self.last_pose[1])
            if math.isfinite(distance) and distance < 2.0:
                self.path_length += distance
        self.last_pose = point
        self.last_pose_time = timestamp

    def add_control(self, timestamp=None):
        self.control_times.append(time.monotonic() if timestamp is None else float(timestamp))

    def add_latency(self, value, vlm=math.nan, action_expert=math.nan):
        value = float(value)
        if math.isfinite(value) and value >= 0.0:
            self.latencies.append(value)
        vlm = float(vlm)
        if math.isfinite(vlm) and vlm >= 0.0:
            self.vlm_latencies.append(vlm)
        action_expert = float(action_expert)
        if math.isfinite(action_expert) and action_expert >= 0.0:
            self.action_expert_latencies.append(action_expert)

    def add_collision(self, event):
        self.collision_events.append(dict(event))
        self.physical_collision_count += 1
        if event.get("is_pedestrian", False):
            self.pedestrian_collision_count += 1

    def set_stuck(self, value):
        value = bool(value)
        if value and not self.stuck:
            self.stuck_count += 1
        if not value and self.stuck:
            self.recovery_count += 1
        self.stuck = value

    def finalize(
        self,
        success,
        navigation_error,
        duration=None,
        termination_reason=None,
        failure_attribution=None,
    ):
        duration = max(0.0, time.monotonic() - self.started) if duration is None else float(duration)
        goal = self.metadata.get("goal")
        spawn = self.metadata.get("spawn")
        if self.start_pose and goal:
            shortest = math.hypot(float(goal[0]) - self.start_pose[0], float(goal[1]) - self.start_pose[1])
        elif goal and spawn:
            shortest = math.hypot(float(goal[0]) - float(spawn[0]), float(goal[1]) - float(spawn[1]))
        else:
            shortest = 0.0
        spl = float(bool(success)) * shortest / max(shortest, self.path_length, 1.0e-9)
        periods = [b - a for a, b in zip(self.control_times, self.control_times[1:]) if b > a]
        model_runtime = self.metadata.get("model_runtime", {})
        result = {
            "model_name": self.model_name,
            "experiment": self.experiment_name,
            "episode_id": self.metadata.get("episode_id", "unknown"),
            "scene": self.metadata.get("scene", "unknown"),
            "evaluation_mode": self.metadata.get("evaluation_mode", "trajectory_normalized"),
            "sensor_profile": self.metadata.get("sensor_profile", "unknown"),
            "comparison_track": self.metadata.get("comparison_track", "untracked"),
            "execution_profile": self.metadata.get("execution_profile", "fair"),
            "desired_speed": self.metadata.get("desired_speed"),
            "camera_horizontal_fov_degrees": self.metadata.get("camera_horizontal_fov_degrees"),
            "model_inputs": self.metadata.get("model_inputs", []),
            "model_runtime": model_runtime,
            "derived_inputs": model_runtime.get("derived_inputs", []),
            "model_internal_3d_perception": bool(model_runtime.get("model_internal_3d_perception", False)),
            "external_local_avoidance": bool(model_runtime.get("external_local_avoidance", False)),
            "uses_shared_path_follower": bool(model_runtime.get("uses_shared_path_follower", True)),
            "uses_shared_velocity_filter": bool(model_runtime.get("uses_shared_velocity_filter", True)),
            "high_level_controller": model_runtime.get(
                "high_level_controller", "shared_pure_pursuit"
            ),
            "model_native_high_level": bool(model_runtime.get("model_native_high_level", False)),
            "controller_fidelity": model_runtime.get("execution", {}).get("controller_fidelity"),
            "instruction": self.metadata.get("instruction", ""),
            "seed": self.metadata.get("seed"),
            "pedestrian_count": self.metadata.get("pedestrian_count", 0),
            "official_pedestrian_count": self.metadata.get(
                "official_pedestrian_count", self.metadata.get("pedestrian_count", 0)
            ),
            "effective_pedestrian_count": self.metadata.get(
                "effective_pedestrian_count", self.metadata.get("pedestrian_count", 0)
            ),
            "success": bool(success),
            "termination_reason": termination_reason,
            "failure_attribution": failure_attribution,
            "navigation_error": float(navigation_error) if math.isfinite(float(navigation_error)) else None,
            "start_pose": list(self.start_pose) if self.start_pose is not None else None,
            "final_pose": list(self.last_pose) if self.last_pose is not None else None,
            "spl": spl,
            "path_length": self.path_length,
            "duration": duration,
            "physical_collision_count": self.physical_collision_count,
            "pedestrian_collision_count": self.pedestrian_collision_count,
            "collision_rate": self.physical_collision_count / max(duration, 1.0e-9),
            "stuck_count": self.stuck_count,
            "recovery_count": self.recovery_count,
            "mean_inference_latency": statistics.fmean(self.latencies) if self.latencies else None,
            "p50_inference_latency": percentile(self.latencies, 0.50),
            "p95_inference_latency": percentile(self.latencies, 0.95),
            "max_inference_latency": max(self.latencies) if self.latencies else None,
            "mean_vlm_latency": statistics.fmean(self.vlm_latencies) if self.vlm_latencies else None,
            "mean_action_expert_latency": (
                statistics.fmean(self.action_expert_latencies) if self.action_expert_latencies else None
            ),
            "mean_control_frequency": 1.0 / statistics.fmean(periods) if periods else 0.0,
            "minimum_control_frequency": 1.0 / max(periods) if periods else 0.0,
            "direct_velocity": self.direct_velocity,
            "collision_events": self.collision_events,
        }
        return result
