import math

import numpy as np
import pytest

from omnivla_adapter.trajectory_converter import (
    TrajectoryConversionError,
    convert_raw_trajectory,
    convert_raw_waypoint_to_velocity,
)


def straight_raw():
    raw = np.zeros((8, 4), dtype=np.float64)
    raw[:, 0] = np.arange(1, 9)
    raw[:, 2] = 1.0
    return raw


def test_scales_absolute_xy_without_cumsum():
    points, scaled = convert_raw_trajectory(straight_raw())
    np.testing.assert_allclose(points[:, 0], np.arange(1, 9) * 0.1)
    np.testing.assert_allclose(points[:, 1:], 0.0)
    np.testing.assert_allclose(scaled[:, 0], np.arange(1, 9) * 0.1)


@pytest.mark.parametrize(
    ("heading", "yaw"),
    [((1.0, 0.0), 0.0), ((0.0, 1.0), math.pi / 2.0), ((0.0, -1.0), -math.pi / 2.0)],
)
def test_heading_vector_to_yaw(heading, yaw):
    raw = straight_raw()
    raw[:, 2:] = heading
    points, _ = convert_raw_trajectory(raw, use_heading=True)
    np.testing.assert_allclose(points[:, 2], yaw)


def test_normalized_mode_ignores_heading_for_final_yaw():
    raw = straight_raw()
    raw[:, 2:] = (0.0, 1.0)
    points, scaled = convert_raw_trajectory(raw)
    np.testing.assert_allclose(points[:, 2], 0.0)
    np.testing.assert_allclose(scaled[:, 2:], np.tile((0.0, 1.0), (8, 1)))


def test_zero_heading_uses_trajectory_tangent():
    raw = straight_raw()
    raw[:, 2:] = 0.0
    points, _ = convert_raw_trajectory(raw, use_heading=True)
    np.testing.assert_allclose(points[:, 2], 0.0)


@pytest.mark.parametrize(
    "raw",
    [
        np.zeros((7, 4)),
        np.full((8, 4), np.nan),
        np.full((8, 4), np.inf),
        np.zeros((8, 4)),
    ],
)
def test_invalid_output_requires_stop(raw):
    with pytest.raises(TrajectoryConversionError):
        convert_raw_trajectory(raw)


def test_zero_heading_and_zero_local_tangent_is_rejected():
    raw = straight_raw()
    raw[3, :2] = raw[2, :2]
    raw[4, :2] = raw[2, :2]
    raw[3, 2:] = 0.0
    with pytest.raises(TrajectoryConversionError, match="zero heading"):
        convert_raw_trajectory(raw, use_heading=True)


def test_official_waypoint_velocity_uses_waypoint_four_and_limits_pair():
    raw = straight_raw()
    raw[:, 1] = 1.0
    raw[4, 0] = 4.0
    velocity = convert_raw_waypoint_to_velocity(raw)
    angular_pre = math.atan(0.1 / 0.4) / (1.0 / 3.0)
    assert velocity[0] == pytest.approx(0.3 * (0.5 / angular_pre), rel=1e-6)
    assert velocity[1] == pytest.approx(0.0)
    assert velocity[2] == pytest.approx(0.3)


def test_official_waypoint_velocity_all_zero_is_stop():
    raw = np.zeros((8, 4), dtype=np.float64)
    np.testing.assert_allclose(convert_raw_waypoint_to_velocity(raw), np.zeros(3))
