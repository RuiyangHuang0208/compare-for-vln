from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def finite_vector(value, shape):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Expected finite shape {shape}, got {array.shape}")
    return array


def local_to_world(points, pose_xy_yaw):
    path = finite_vector(points, (len(points), 2))
    pose = finite_vector(pose_xy_yaw, (3,))
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.asarray(((c, -s), (s, c)), dtype=np.float64)
    return path @ rotation.T + pose[:2]


@dataclass(frozen=True)
class Limits:
    max_vx: float = 1.0
    max_vy: float = 0.5
    max_wz: float = 1.0
    max_linear_acceleration: float = 0.8
    max_angular_acceleration: float = 1.5
    max_linear_deceleration: float = 0.8
    max_angular_deceleration: float = 1.5


class VelocityFilter:
    def __init__(self, limits: Limits):
        self.limits = limits
        self.last = np.zeros(3, dtype=np.float64)

    def reset(self):
        self.last.fill(0.0)

    def apply(self, target, dt):
        target = finite_vector(target, (3,)).copy()
        target[0] = np.clip(target[0], -self.limits.max_vx, self.limits.max_vx)
        target[1] = np.clip(target[1], -self.limits.max_vy, self.limits.max_vy)
        target[2] = np.clip(target[2], -self.limits.max_wz, self.limits.max_wz)
        duration = max(0.0, float(dt))
        for index in (0, 1):
            accelerating = abs(target[index]) > abs(self.last[index]) and target[index] * self.last[index] >= 0.0
            rate = (
                self.limits.max_linear_acceleration
                if accelerating
                else self.limits.max_linear_deceleration
            )
            self.last[index] += np.clip(target[index] - self.last[index], -rate * duration, rate * duration)
        accelerating = abs(target[2]) > abs(self.last[2]) and target[2] * self.last[2] >= 0.0
        angular_rate = (
            self.limits.max_angular_acceleration
            if accelerating
            else self.limits.max_angular_deceleration
        )
        self.last[2] += np.clip(target[2] - self.last[2], -angular_rate * duration, angular_rate * duration)
        return self.last.copy()


class PurePursuitFollower:
    def __init__(
        self,
        lookahead=1.0,
        goal_tolerance=0.05,
        heading_tolerance=math.radians(5.0),
        heading_capture_distance=0.10,
        speed=1.0,
        yaw_gain=0.8,
        yaw_filter_alpha=0.35,
        curvature_feedforward_gain=0.5,
    ):
        self.lookahead = float(lookahead)
        self.goal_tolerance = float(goal_tolerance)
        self.heading_tolerance = float(heading_tolerance)
        self.heading_capture_distance = float(heading_capture_distance)
        self.speed = float(speed)
        self.yaw_gain = float(yaw_gain)
        self.yaw_filter_alpha = float(yaw_filter_alpha)
        self.curvature_feedforward_gain = float(curvature_feedforward_gain)
        self.path = None
        self.final_yaw = None
        self._yaw_error_filtered = None

    def set_path(self, path, final_yaw=None):
        array = np.asarray(path, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 2 or len(array) < 1 or not np.isfinite(array).all():
            raise ValueError(f"Path must be finite Nx2, got {array.shape}")
        if final_yaw is not None and not math.isfinite(final_yaw):
            raise ValueError("final_yaw must be finite or None")
        self.path = array.copy()
        self.final_yaw = None if final_yaw is None else wrap_angle(final_yaw)

    def clear(self):
        self.path = None
        self.final_yaw = None
        self._yaw_error_filtered = None

    @property
    def active(self):
        return self.path is not None

    def command(self, pose_xy_yaw):
        if self.path is None:
            return np.zeros(3), math.inf, False
        pose = finite_vector(pose_xy_yaw, (3,))
        delta = self.path - pose[:2]
        distances = np.linalg.norm(delta, axis=1)
        goal_distance = float(distances[-1])
        capture_distance = (
            max(self.goal_tolerance, self.heading_capture_distance)
            if self.final_yaw is not None
            else self.goal_tolerance
        )
        if goal_distance <= capture_distance:
            if self.final_yaw is not None:
                heading_error = wrap_angle(self.final_yaw - pose[2])
                if abs(heading_error) > self.heading_tolerance:
                    return np.asarray((0.0, 0.0, self.yaw_gain * heading_error)), goal_distance, False
            self.clear()
            return np.zeros(3), goal_distance, True
        nearest = int(np.argmin(distances))
        # Match TIC-VLA's DynaNav controller: choose the lookahead by arc
        # length from the current local prediction origin. Reconstructing the
        # local path avoids jumping to a nearest point on a later bend when a
        # rolling trajectory is refreshed.
        target_is_local = False
        if len(self.path) >= 5:
            cosine, sine = math.cos(float(pose[2])), math.sin(float(pose[2]))
            local_delta = self.path - pose[:2]
            local = np.column_stack(
                (
                    cosine * local_delta[:, 0] + sine * local_delta[:, 1],
                    -sine * local_delta[:, 0] + cosine * local_delta[:, 1],
                )
            )
            arc_length = np.concatenate(
                ([0.0], np.cumsum(np.linalg.norm(np.diff(local, axis=0), axis=1)))
            )
            target_index = int(np.searchsorted(arc_length, self.lookahead, side="left"))
            target_index = int(np.clip(target_index, 2, len(local) - 3))
            x_target, y_target = (float(value) for value in local[target_index])
            target_is_local = True
        else:
            target_index = nearest
            target = self.path[target_index]
            x_target = float(target[0] - pose[0])
            y_target = float(target[1] - pose[1])
        target_distance = max(float(math.hypot(x_target, y_target)), 1.0e-3)
        desired_heading = math.atan2(y_target, x_target)
        # The long-path branch above already expresses the target in base_link.
        # Subtracting the world yaw a second time makes a rotated straight path
        # look like a large turn (and can command zero forward speed).  The
        # short-path branch keeps the target in world coordinates.
        heading_error = desired_heading if target_is_local else wrap_angle(desired_heading - pose[2])
        if self._yaw_error_filtered is None:
            self._yaw_error_filtered = heading_error
        else:
            error_delta = wrap_angle(heading_error - self._yaw_error_filtered)
            self._yaw_error_filtered += self.yaw_filter_alpha * error_delta

        # Curvature feed-forward reduces the oscillation seen when the B2-W
        # follows successive short predictions around corners.
        curvature = 2.0 * y_target / (target_distance * target_distance)
        curvature_speed = self.yaw_gain / (abs(curvature) + 1.0e-3)
        vx = min(self.speed, curvature_speed, max(0.08, goal_distance))
        if abs(heading_error) > 1.0:
            vx = 0.0
        else:
            vx *= max(0.0, math.cos(heading_error))
        wz = self.curvature_feedforward_gain * vx * curvature + self.yaw_gain * self._yaw_error_filtered
        return np.asarray((vx, 0.0, wz)), goal_distance, False


