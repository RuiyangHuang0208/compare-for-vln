import math

import numpy as np
import pytest

from navila_adapter.action_converter import action_to_trajectory
from navila_adapter.action_parser import ParsedAction


def test_forward_converts_centimeters_to_local_meters():
    points = action_to_trajectory(ParsedAction("forward", 50), spacing=0.1, turn_radius=1.0)
    assert points.shape == (5, 3)
    np.testing.assert_allclose(points[-1], [0.5, 0.0, 0.0])
    assert np.all(np.diff(points[:, 0]) > 0.0)


@pytest.mark.parametrize(("kind", "sign"), [("left", 1.0), ("right", -1.0)])
def test_turn_converts_to_executable_xy_arc(kind, sign):
    points = action_to_trajectory(ParsedAction(kind, 30), spacing=0.1, turn_radius=1.0)
    assert len(points) >= 2
    assert np.all(points[:, 0] > 0.0)
    assert np.all(sign * points[:, 1] > 0.0)
    assert math.isclose(points[-1, 2], sign * math.radians(30), rel_tol=1.0e-9)


def test_uncalibrated_turn_radius_is_rejected():
    with pytest.raises(ValueError, match="calibrated"):
        action_to_trajectory(ParsedAction("left", 15), spacing=0.1, turn_radius=0.0)
