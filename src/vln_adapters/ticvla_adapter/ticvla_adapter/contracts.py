from __future__ import annotations

import math

import numpy as np


def validate_waypoints(value, horizon=30):
    array = np.asarray(value, dtype=np.float32)
    if array.shape == (1, horizon, 2):
        array = array[0]
    if array.shape != (horizon, 2):
        raise ValueError(f"formal TIC-VLA output must be ({horizon},2) or (1,{horizon},2), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("TIC-VLA waypoints contain NaN/Inf")
    return array


class OfficialTicVlaCurvatureController:
    """Port of TIC-VLA's public Spot waypoint-to-velocity equations.

    The source is ``DynaNav/behavior/spot_test_ticvla.py`` at the repository
    commit recorded in ``configs/models.yaml``.  Only the final velocity limit
    is configurable so the same equations can respect the verified B2-W
    SRU-ONNX command range.
    """

    def __init__(
        self,
        *,
        lookahead=1.0,
        angular_gain=0.8,
        yaw_filter_alpha=0.35,
        curvature_feedforward_gain=0.5,
        max_linear_velocity=1.0,
        max_angular_velocity=1.0,
    ):
        self.lookahead = float(lookahead)
        self.angular_gain = float(angular_gain)
        self.yaw_filter_alpha = float(yaw_filter_alpha)
        self.curvature_feedforward_gain = float(curvature_feedforward_gain)
        self.max_linear_velocity = float(max_linear_velocity)
        self.max_angular_velocity = float(max_angular_velocity)
        values = (
            self.lookahead,
            self.angular_gain,
            self.yaw_filter_alpha,
            self.curvature_feedforward_gain,
            self.max_linear_velocity,
            self.max_angular_velocity,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("TIC-VLA controller parameters must be finite and positive")
        self.yaw_error_filtered = None

    def reset(self):
        self.yaw_error_filtered = None

    def command(self, waypoints):
        points = np.asarray(waypoints, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 5:
            raise ValueError("TIC-VLA official controller requires at least five Nx2 waypoints")
        if not np.isfinite(points).all():
            raise ValueError("TIC-VLA waypoints contain NaN/Inf")

        increments = np.diff(points, axis=0)
        arc_length = np.concatenate(([0.0], np.cumsum(np.hypot(increments[:, 0], increments[:, 1]))))
        index = int(np.searchsorted(arc_length, self.lookahead, side="left"))
        index = int(np.clip(index, 2, len(points) - 3))
        x_target, y_target = (float(value) for value in points[index])
        distance = float(math.hypot(x_target, y_target))
        epsilon = 1.0e-3
        if distance < epsilon:
            return np.zeros(3, dtype=np.float64)

        yaw_error = math.atan2(y_target, x_target)
        if self.yaw_error_filtered is None:
            self.yaw_error_filtered = yaw_error
        error = math.atan2(
            math.sin(yaw_error - self.yaw_error_filtered),
            math.cos(yaw_error - self.yaw_error_filtered),
        )
        self.yaw_error_filtered += self.yaw_filter_alpha * error
        curvature = 2.0 * y_target / (distance * distance)
        curvature_limited_speed = self.max_angular_velocity / (abs(curvature) + epsilon)
        velocity_x = float(
            np.clip(
                min(self.max_linear_velocity, curvature_limited_speed),
                0.0,
                self.max_linear_velocity,
            )
        )
        angular_z = (
            self.curvature_feedforward_gain * velocity_x * curvature
            + self.angular_gain * self.yaw_error_filtered
        )
        angular_z = float(
            np.clip(angular_z, -self.max_angular_velocity, self.max_angular_velocity)
        )
        return np.asarray((velocity_x, 0.0, angular_z), dtype=np.float64)


def displacement_in_delayed_frame(delayed_pose, current_pose):
    delayed = np.asarray(delayed_pose, dtype=np.float64)
    current = np.asarray(current_pose, dtype=np.float64)
    if delayed.shape != (3,) or current.shape != (3,) or not np.isfinite(delayed).all() or not np.isfinite(current).all():
        raise ValueError("poses must be finite [x,y,yaw]")
    delta = current[:2] - delayed[:2]
    c, s = np.cos(delayed[2]), np.sin(delayed[2])
    return np.asarray((c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]), dtype=np.float32)


def format_previous_waypoints(history, sample_interval=10):
    """Format executed one-second displacements like DynaNav's TIC-VLA behavior."""
    if sample_interval < 1:
        raise ValueError("sample_interval must be positive")
    records = list(history)
    if not records:
        return "From 0.0s to current timestamp time is 0.0s. No waypoints available."
    started = float(records[0]["stamp"])
    elapsed = max(0.0, float(records[-1]["stamp"]) - started)
    waypoints = []
    previous_index = 0
    for current_index in range(sample_interval, len(records), sample_interval):
        previous = records[previous_index]
        current = records[current_index]
        displacement = displacement_in_delayed_frame(previous["pose"], current["pose"])
        if np.isfinite(displacement).all():
            waypoints.append((float(displacement[0]), float(displacement[1]), 0.0))
        previous_index = current_index
    if not waypoints:
        return f"From 0.0s to current timestamp time is {elapsed:.1f}s. No waypoints available."
    values = ", ".join(f"({x:.2f}, {y:.2f}, {z:.2f})" for x, y, z in waypoints)
    return (
        f"From 0.0s to current timestamp time is {elapsed:.1f}s. "
        f"(a list of waypoints 1s in between): {values}\n"
        "Each waypoint (x, y, z) is the displacement over the previous 1.0s. "
        "x is forward, y is left, z is up."
    )


def trajectory_at_capture_to_current(trajectory, capture_pose, current_pose):
    points = np.asarray(trajectory, dtype=np.float64)
    capture = np.asarray(capture_pose, dtype=np.float64)
    current = np.asarray(current_pose, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("trajectory must be finite Nx2")
    if capture.shape != (3,) or current.shape != (3,):
        raise ValueError("poses must be [x,y,yaw]")
    capture_rotation = np.asarray(
        ((np.cos(capture[2]), -np.sin(capture[2])), (np.sin(capture[2]), np.cos(capture[2])))
    )
    current_rotation = np.asarray(
        ((np.cos(current[2]), -np.sin(current[2])), (np.sin(current[2]), np.cos(current[2])))
    )
    world = points @ capture_rotation.T + capture[:2]
    return (world - current[:2]) @ current_rotation
