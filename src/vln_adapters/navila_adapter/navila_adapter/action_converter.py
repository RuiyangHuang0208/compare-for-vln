from __future__ import annotations

import math

import numpy as np

from .action_parser import ParsedAction


def _sample_count(length, spacing):
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("trajectory_spacing must be finite and positive")
    return max(2, int(math.ceil(length / spacing)))


def action_to_trajectory(action: ParsedAction, spacing: float, turn_radius: float):
    if action.kind == "stop":
        return np.empty((0, 3), dtype=np.float64)
    if action.kind == "forward":
        distance = action.value / 100.0
        count = _sample_count(distance, spacing)
        x = np.linspace(distance / count, distance, count)
        return np.column_stack((x, np.zeros(count), np.zeros(count)))
    if action.kind not in {"left", "right"}:
        raise ValueError(f"Unsupported action kind {action.kind!r}")
    if not math.isfinite(turn_radius) or turn_radius <= 0.0:
        raise ValueError("turn_radius must be calibrated and greater than zero")
    sign = 1.0 if action.kind == "left" else -1.0
    angle = math.radians(action.value)
    count = _sample_count(turn_radius * angle, spacing)
    samples = np.linspace(angle / count, angle, count)
    x = turn_radius * np.sin(samples)
    y = sign * turn_radius * (1.0 - np.cos(samples))
    yaw = sign * samples
    return np.column_stack((x, y, yaw))
