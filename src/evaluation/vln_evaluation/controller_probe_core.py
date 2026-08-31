from __future__ import annotations

import math

import numpy as np

from navila_adapter.action_converter import action_to_trajectory
from navila_adapter.action_parser import ParsedAction
from omnivla_adapter.trajectory_converter import convert_raw_waypoint_to_velocity
from ticvla_adapter.contracts import OfficialTicVlaCurvatureController
from uninavid_adapter.action_converter import actions_to_trajectory


PATH_PROFILES = {"shared_pure_pursuit", "navila_discrete", "uninavid_discrete"}
VELOCITY_PROFILES = {"ticvla_official", "omnivla_official"}
ALL_PROFILES = PATH_PROFILES | VELOCITY_PROFILES


def _finite_goal(local_goal):
    goal = np.asarray(local_goal, dtype=np.float64)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError("local_goal must be finite [x,y]")
    return goal


def _compose(segments):
    pose = np.zeros(3, dtype=np.float64)
    result = []
    for segment in segments:
        values = np.asarray(segment, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
            raise ValueError("controller probe segment must be finite Nx3")
        c, s = math.cos(pose[2]), math.sin(pose[2])
        for x, y, yaw in values:
            result.append(
                (
                    pose[0] + c * x - s * y,
                    pose[1] + s * x + c * y,
                    pose[2] + yaw,
                )
            )
        if len(values):
            pose[:] = result[-1]
    return np.asarray(result, dtype=np.float64)


def build_path(profile, motion, local_goal):
    """Build deterministic paths through the actual adapter converters."""
    goal = _finite_goal(local_goal)
    if motion not in {"straight", "left"}:
        raise ValueError("motion must be straight or left")
    if profile == "shared_pure_pursuit":
        fractions = np.linspace(1.0 / 30.0, 1.0, 30)
        xy = fractions[:, None] * goal[None, :]
        return np.column_stack((xy, np.zeros(len(xy))))
    if profile == "navila_discrete":
        repeats = max(1, int(math.ceil(float(np.linalg.norm(goal)) / 0.75)))
        actions = (
            [ParsedAction("forward", 75.0)] * repeats
            if motion == "straight"
            else [ParsedAction("left", 45.0), ParsedAction("forward", 75.0)]
        )
        return _compose(
            [action_to_trajectory(action, spacing=0.1, turn_radius=1.0) for action in actions]
        )
    if profile == "uninavid_discrete":
        repeats = max(1, int(math.ceil(float(np.linalg.norm(goal)) / 0.5)))
        actions = (("forward",) * repeats) if motion == "straight" else ("left", "forward")
        return actions_to_trajectory(
            actions,
            forward_distance=0.5,
            turn_degrees=30.0,
            turn_radius=0.25,
            spacing=0.1,
        )
    raise ValueError(f"profile {profile!r} is not a path controller")


def make_ticvla_waypoints(local_goal, horizon=30):
    goal = _finite_goal(local_goal)
    fractions = np.linspace(1.0 / horizon, 1.0, horizon)
    return fractions[:, None] * goal[None, :]


def make_omnivla_raw(local_goal):
    goal = _finite_goal(local_goal)
    raw = np.zeros((8, 4), dtype=np.float64)
    fractions = np.linspace(1.0 / 8.0, 1.0, 8)
    raw[:, :2] = fractions[:, None] * goal[None, :] / 0.1
    norm = float(np.linalg.norm(goal))
    if norm > 1.0e-9:
        raw[:, 2:] = goal / norm
    else:
        raw[:, 2] = 1.0
    return raw


def velocity_command(profile, local_goal, ticvla_controller=None):
    goal = _finite_goal(local_goal)
    if float(np.linalg.norm(goal)) < 0.05:
        return np.zeros(3, dtype=np.float64)
    if profile == "ticvla_official":
        controller = ticvla_controller or OfficialTicVlaCurvatureController(
            max_linear_velocity=1.0,
            max_angular_velocity=1.0,
        )
        return controller.command(make_ticvla_waypoints(goal))
    if profile == "omnivla_official":
        return convert_raw_waypoint_to_velocity(make_omnivla_raw(goal))
    raise ValueError(f"profile {profile!r} is not a velocity controller")
