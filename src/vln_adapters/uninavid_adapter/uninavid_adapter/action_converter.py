from __future__ import annotations

import math

import numpy as np


def _samples(length: float, spacing: float) -> int:
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("trajectory_spacing must be finite and positive")
    return max(2, int(math.ceil(length / spacing)))


def actions_to_trajectory(
    actions,
    *,
    forward_distance: float,
    turn_degrees: float,
    turn_radius: float,
    spacing: float,
):
    if not math.isfinite(forward_distance) or forward_distance <= 0.0:
        raise ValueError("forward_distance must be finite and positive")
    if not math.isfinite(turn_degrees) or turn_degrees <= 0.0:
        raise ValueError("turn_degrees must be finite and positive")
    if not math.isfinite(turn_radius) or turn_radius <= 0.0:
        raise ValueError("turn_radius is uncalibrated; run the dummy B2-W arc test first")

    pose = np.zeros(3, dtype=np.float64)
    points = []
    for action in actions:
        if action == "forward":
            count = _samples(forward_distance, spacing)
            for distance in np.linspace(forward_distance / count, forward_distance, count):
                points.append(
                    (
                        pose[0] + distance * math.cos(pose[2]),
                        pose[1] + distance * math.sin(pose[2]),
                        pose[2],
                    )
                )
            pose[:] = points[-1]
        elif action in {"left", "right"}:
            sign = 1.0 if action == "left" else -1.0
            angle = math.radians(turn_degrees)
            count = _samples(turn_radius * angle, spacing)
            start = pose.copy()
            for magnitude in np.linspace(angle / count, angle, count):
                yaw = start[2] + sign * magnitude
                x = start[0] + sign * turn_radius * (math.sin(yaw) - math.sin(start[2]))
                y = start[1] - sign * turn_radius * (math.cos(yaw) - math.cos(start[2]))
                points.append((x, y, yaw))
            pose[:] = points[-1]
        else:
            raise ValueError(f"Unsupported movement action {action!r}")
    values = np.asarray(points, dtype=np.float64)
    if not points:
        return np.empty((0, 3), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Converted Uni-NaVid trajectory is invalid")
    return values
