import numpy as np

from vln_evaluation.controller_probe_core import build_path, velocity_command


def test_all_path_profiles_produce_forward_motion():
    goals = {
        "shared_pure_pursuit": (2.0, 0.0),
        "navila_discrete": (2.25, 0.0),
        "uninavid_discrete": (1.0, 0.0),
    }
    for profile, goal in goals.items():
        path = build_path(profile, "straight", goal)
        assert path.ndim == 2 and path.shape[1] == 3
        assert np.isfinite(path).all()
        assert path[-1, 0] > 0.9
        assert abs(path[-1, 1]) < 1.0e-9


def test_discrete_left_profiles_have_positive_y_and_yaw():
    for profile in ("navila_discrete", "uninavid_discrete"):
        path = build_path(profile, "left", (1.0, 0.5))
        assert path[-1, 0] > 0.0
        assert path[-1, 1] > 0.0
        assert path[-1, 2] > 0.0


def test_official_velocity_profiles_go_forward_and_turn_left():
    for profile in ("ticvla_official", "omnivla_official"):
        straight = velocity_command(profile, (2.0, 0.0))
        left = velocity_command(profile, (1.5, 0.5))
        assert straight[0] > 0.0
        assert abs(straight[2]) < 1.0e-9
        assert left[0] > 0.0
        assert left[2] > 0.0
