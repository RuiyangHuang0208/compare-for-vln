from __future__ import annotations

import math

import numpy as np

from .contracts import ACTION_SHAPE


class TrajectoryConversionError(ValueError):
    pass


def convert_raw_waypoint_to_velocity(
    raw_trajectory,
    *,
    waypoint_index: int = 4,
    dt: float = 1.0 / 3.0,
    max_linear_before_limit: float = 0.5,
    max_angular_before_limit: float = 1.0,
    max_v: float = 0.3,
    max_w: float = 0.3,
    xy_scale_m: float = 0.1,
) -> np.ndarray:
    """Reproduce the official OmniVLA waypoint-4 velocity conversion.

    The public controller selects one waypoint, scales only its x/y values by
    the metric waypoint spacing (0.1 m), computes a one-step velocity, then
    clips the pair while preserving its linear/angular ratio.  The returned
    vector is ``[vx, vy, wz]`` for the workspace ``NavigationCommand``.
    """
    raw = np.asarray(raw_trajectory, dtype=np.float64)
    if raw.shape != ACTION_SHAPE or not np.isfinite(raw).all():
        raise TrajectoryConversionError(f"expected finite raw trajectory shape {ACTION_SHAPE}, got {raw.shape}")
    if not 0 <= int(waypoint_index) < ACTION_SHAPE[0]:
        raise TrajectoryConversionError("waypoint_index is outside the 8-point action chunk")
    values = (float(dt), float(max_linear_before_limit), float(max_angular_before_limit), float(max_v), float(max_w), float(xy_scale_m))
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise TrajectoryConversionError("native velocity timing, limits, and scale must be finite and positive")

    dx, dy, hx, hy = raw[int(waypoint_index)]
    dx, dy = float(dx) * xy_scale_m, float(dy) * xy_scale_m
    if abs(dx) < 1.0e-8 and abs(dy) < 1.0e-8:
        linear = 0.0
        angular = math.atan2(float(hy), float(hx)) / dt
    elif abs(dx) < 1.0e-8:
        linear = 0.0
        angular = math.copysign(math.pi / (2.0 * dt), dy)
    else:
        linear = dx / dt
        angular = math.atan(dy / dx) / dt

    # These are the two clipping stages in the upstream example.
    linear = float(np.clip(linear, 0.0, max_linear_before_limit))
    angular = float(np.clip(angular, -max_angular_before_limit, max_angular_before_limit))
    if abs(linear) <= max_v and abs(angular) <= max_w:
        return np.asarray((linear, 0.0, angular), dtype=np.float64)
    if abs(angular) <= 1.0e-3:
        return np.asarray((max_v * math.copysign(1.0, linear) if linear else 0.0, 0.0, 0.0), dtype=np.float64)
    ratio = linear / angular
    if abs(ratio) >= max_v / max_w:
        return np.asarray((max_v * math.copysign(1.0, linear), 0.0, max_v * math.copysign(1.0, angular) / abs(ratio)), dtype=np.float64)
    return np.asarray((max_w * math.copysign(abs(ratio), linear), 0.0, max_w * math.copysign(1.0, angular)), dtype=np.float64)


def _tangent(points: np.ndarray, index: int, epsilon: float) -> np.ndarray | None:
    candidates = []
    if index + 1 < len(points):
        candidates.append(points[index + 1] - points[index])
    if index > 0:
        candidates.append(points[index] - points[index - 1])
    if index == 0:
        candidates.append(points[index])
    for candidate in candidates:
        if np.isfinite(candidate).all() and float(np.linalg.norm(candidate)) > epsilon:
            return candidate
    return None


def convert_raw_trajectory(
    raw_trajectory,
    *,
    xy_scale_m: float = 0.1,
    use_heading: bool = False,
    heading_epsilon: float = 1.0e-6,
    max_abs_coordinate_m: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert absolute local [x,y,hx,hy] points.

    OmniVLA's official controller uses the heading vector for its instantaneous
    PD command, not as a final-yaw constraint.  The normalized shared path
    follower therefore defaults to XY-only points (theta=0).  Set
    ``use_heading=True`` only for an explicitly heading-aware consumer.
    """
    raw = np.asarray(raw_trajectory, dtype=np.float64)
    if raw.shape != ACTION_SHAPE:
        raise TrajectoryConversionError(f"expected raw trajectory shape {ACTION_SHAPE}, got {raw.shape}")
    if not np.isfinite(raw).all():
        raise TrajectoryConversionError("raw trajectory contains NaN or Inf")
    if not math.isfinite(xy_scale_m) or xy_scale_m <= 0.0:
        raise TrajectoryConversionError("xy_scale_m must be finite and positive")
    if not math.isfinite(heading_epsilon) or heading_epsilon <= 0.0:
        raise TrajectoryConversionError("heading_epsilon must be finite and positive")

    xy = raw[:, :2] * xy_scale_m
    if float(np.max(np.abs(xy))) > max_abs_coordinate_m:
        raise TrajectoryConversionError("scaled trajectory exceeds the configured safety bound")
    if float(np.max(np.linalg.norm(xy, axis=1))) <= heading_epsilon:
        raise TrajectoryConversionError("trajectory is stationary")

    result = np.empty((ACTION_SHAPE[0], 3), dtype=np.float64)
    result[:, :2] = xy
    if use_heading:
        for index, heading in enumerate(raw[:, 2:4]):
            if float(np.linalg.norm(heading)) <= heading_epsilon:
                heading = _tangent(xy, index, heading_epsilon)
                if heading is None:
                    raise TrajectoryConversionError(f"point {index} has zero heading and no valid tangent")
            result[index, 2] = math.atan2(float(heading[1]), float(heading[0]))
    else:
        result[:, 2] = 0.0
    return result, np.column_stack((xy, raw[:, 2:4]))
