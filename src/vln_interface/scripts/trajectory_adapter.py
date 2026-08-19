"""Convert DualVLN ego-frame trajectories to world-frame XY paths."""

from __future__ import annotations

import math

import numpy as np


def dualvln_trajectory_to_world(trajectory, capture_pose, skip_points=3):
    """Transform an Nx2 DualVLN path using the pose captured with its RGB-D frame."""
    path = np.asarray(trajectory, dtype=np.float32)
    pose = np.asarray(capture_pose, dtype=np.float32)
    if path.ndim != 2 or path.shape[1] < 2 or path.shape[0] < 2:
        raise ValueError(f"Expected DualVLN trajectory shape Nx2, got {path.shape}")
    if pose.shape != (3,):
        raise ValueError(f"Expected capture pose [x, y, yaw], got {pose.shape}")
    if not np.isfinite(path[:, :2]).all() or not np.isfinite(pose).all():
        raise ValueError("DualVLN trajectory and capture pose must be finite")

    path = path[min(max(0, skip_points), path.shape[0] - 2) :, :2]
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    rotation = np.array(((c, -s), (s, c)), dtype=np.float32)
    return path @ rotation.T + pose[:2]
