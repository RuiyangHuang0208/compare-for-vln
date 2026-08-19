"""Track a world-frame XY path with conservative B2W velocity commands."""

from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PathFollower:
    """Small pure-pursuit style follower for a single simulated robot."""

    def __init__(self, desired_speed=0.3, lookahead=0.6, goal_tolerance=0.25, yaw_gain=1.5):
        self.desired_speed = float(desired_speed)
        self.lookahead = float(lookahead)
        self.goal_tolerance = float(goal_tolerance)
        self.yaw_gain = float(yaw_gain)
        self.path = None
        self.generation = 0

    def set_path(self, path):
        value = np.asarray(path, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != 2 or value.shape[0] < 2:
            raise ValueError(f"Expected world path shape Nx2, got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError("World path contains non-finite values")
        self.path = value
        self.generation += 1

    def clear(self):
        self.path = None

    @property
    def active(self):
        return self.path is not None

    def command(self, position_xy, yaw):
        """Return [vx, vy, yaw_rate], remaining goal distance, and reached state."""
        if self.path is None:
            return np.zeros(3, dtype=np.float32), math.inf, False

        position = np.asarray(position_xy, dtype=np.float32)
        distances = np.linalg.norm(self.path - position, axis=1)
        nearest = int(np.argmin(distances))
        goal_distance = float(np.linalg.norm(self.path[-1] - position))
        if goal_distance <= self.goal_tolerance:
            self.clear()
            return np.zeros(3, dtype=np.float32), goal_distance, True

        target_index = nearest
        while target_index + 1 < len(self.path):
            if np.linalg.norm(self.path[target_index] - position) >= self.lookahead:
                break
            target_index += 1
        target = self.path[target_index]
        heading = math.atan2(float(target[1] - position[1]), float(target[0] - position[0]))
        heading_error = wrap_angle(heading - float(yaw))
        yaw_rate = self.yaw_gain * heading_error

        speed = min(self.desired_speed, max(0.08, goal_distance))
        if abs(heading_error) > 1.0:
            speed = 0.0
        else:
            speed *= max(0.0, math.cos(heading_error))
        return np.array((speed, 0.0, yaw_rate), dtype=np.float32), goal_distance, False
