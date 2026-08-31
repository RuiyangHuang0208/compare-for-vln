import math

import numpy as np
import pytest

from uninavid_adapter.action_converter import actions_to_trajectory


def convert(actions, radius=1.0):
    return actions_to_trajectory(
        actions,
        forward_distance=0.5,
        turn_degrees=30.0,
        turn_radius=radius,
        spacing=0.1,
    )


def test_two_forward_actions_form_one_meter_local_path():
    points = convert(("forward", "forward"))
    np.testing.assert_allclose(points[-1], [1.0, 0.0, 0.0], atol=1.0e-10)


@pytest.mark.parametrize(("action", "sign"), [("left", 1.0), ("right", -1.0)])
def test_turn_is_executable_xy_arc(action, sign):
    points = convert((action,))
    assert np.all(points[:, 0] > 0.0)
    assert np.all(sign * points[:, 1] > 0.0)
    assert math.isclose(points[-1, 2], sign * math.radians(30.0))


def test_sequential_action_uses_updated_heading():
    points = convert(("left", "forward"))
    assert points[-1, 1] > points[-2, 1]
    assert math.isclose(points[-1, 2], math.radians(30.0))


def test_uncalibrated_radius_is_rejected():
    with pytest.raises(ValueError, match="uncalibrated"):
        convert(("left",), radius=0.0)
