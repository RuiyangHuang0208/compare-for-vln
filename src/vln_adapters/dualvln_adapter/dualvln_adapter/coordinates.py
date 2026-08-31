from __future__ import annotations

import math

import numpy as np


def trajectory_at_capture_to_current(trajectory, capture_pose, current_pose):
    """Express a metric XY trajectory from its capture base frame in the current base frame."""
    points = np.asarray(trajectory, dtype=np.float64)
    capture = np.asarray(capture_pose, dtype=np.float64)
    current = np.asarray(current_pose, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError(f"DualVLN trajectory must be finite Nx2 with N>=2, got {points.shape}")
    if capture.shape != (3,) or current.shape != (3,):
        raise ValueError("capture_pose and current_pose must be [x,y,yaw]")
    if not np.isfinite(points).all() or not np.isfinite(capture).all() or not np.isfinite(current).all():
        raise ValueError("trajectory and poses must be finite")
    capture_rotation = np.asarray(
        ((math.cos(capture[2]), -math.sin(capture[2])), (math.sin(capture[2]), math.cos(capture[2])))
    )
    current_rotation = np.asarray(
        ((math.cos(current[2]), -math.sin(current[2])), (math.sin(current[2]), math.cos(current[2])))
    )
    world = points @ capture_rotation.T + capture[:2]
    return (world - current[:2]) @ current_rotation
