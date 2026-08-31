from __future__ import annotations

import numpy as np


class PointCloudError(ValueError):
    pass


def pointcloud_from_depth(
    depth_m,
    intrinsic_matrix,
    *,
    max_points: int = 2048,
    normalize: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Project aligned metric depth into camera-optical XYZ and sample deterministically."""
    depth = np.asarray(depth_m, dtype=np.float32)
    matrix = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(3, 3)
    if depth.ndim != 2:
        raise PointCloudError("depth must be a 2-D array")
    if max_points <= 0:
        raise PointCloudError("max_points must be positive")
    fx, fy, cx, cy = matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]
    if not np.isfinite(matrix).all() or min(fx, fy) <= 0.0:
        raise PointCloudError("CameraInfo contains invalid intrinsics")
    rows, columns = np.indices(depth.shape, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    z = depth[valid]
    x = (columns[valid] - cx) * z / fx
    y = (rows[valid] - cy) * z / fy
    points = np.stack((x, y, z), axis=1).astype(np.float32, copy=False)
    if points.shape[0] == 0:
        raise PointCloudError("depth contains no positive finite samples")
    if points.shape[0] > max_points:
        indices = np.random.default_rng(seed).choice(points.shape[0], max_points, replace=False)
        points = points[indices]
    elif points.shape[0] < max_points:
        points = np.concatenate((points, np.zeros((max_points - points.shape[0], 3), np.float32)))
    if normalize:
        mean = points.mean(axis=0, keepdims=True)
        std = points.std(axis=0, keepdims=True)
        points = (points - mean) / np.maximum(std, 1.0e-6)
    if points.shape != (max_points, 3) or not np.isfinite(points).all():
        raise PointCloudError("point cloud is not finite with the requested shape")
    return np.ascontiguousarray(points, dtype=np.float32)

