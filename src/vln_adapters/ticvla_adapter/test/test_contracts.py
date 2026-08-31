import math

import numpy as np
import pytest

from ticvla_adapter.contracts import (
    OfficialTicVlaCurvatureController,
    displacement_in_delayed_frame,
    format_previous_waypoints,
    validate_waypoints,
)
from ticvla_adapter.ticvla_node import crop_to_aspect


def test_accepts_only_formal_30_by_2_output_without_cumsum():
    expected = np.arange(60, dtype=np.float32).reshape(30, 2)
    np.testing.assert_array_equal(validate_waypoints(expected[None]), expected)


def test_rejects_delta_theta_shape():
    with pytest.raises(ValueError, match="formal TIC-VLA output"):
        validate_waypoints(np.zeros((30, 3)))


def test_displacement_uses_delayed_body_frame():
    result = displacement_in_delayed_frame((1.0, 2.0, math.pi / 2), (0.0, 3.0, 0.0))
    np.testing.assert_allclose(result, (1.0, 1.0), atol=1.0e-7)


def test_previous_waypoints_match_one_second_body_displacements():
    history = [
        {"stamp": index * 0.1, "pose": np.asarray((index * 0.1, 0.0, 0.0))}
        for index in range(21)
    ]
    result = format_previous_waypoints(history, sample_interval=10)
    assert "current timestamp time is 2.0s" in result
    assert result.count("(1.00, 0.00, 0.00)") == 2


def test_ticvla_crop_preserves_width_and_center_16_9():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[240, 320] = (1, 2, 3)
    cropped = crop_to_aspect(image, 16.0 / 9.0)
    assert cropped.shape == (360, 640, 3)
    np.testing.assert_array_equal(cropped[180, 320], (1, 2, 3))


def test_official_ticvla_controller_matches_straight_path_equations():
    controller = OfficialTicVlaCurvatureController(
        max_linear_velocity=1.0, max_angular_velocity=1.0
    )
    points = np.column_stack((np.linspace(0.1, 3.0, 30), np.zeros(30)))
    np.testing.assert_allclose(controller.command(points), (1.0, 0.0, 0.0), atol=1.0e-8)


def test_official_ticvla_controller_turns_left_and_resets_filter():
    controller = OfficialTicVlaCurvatureController(
        max_linear_velocity=1.0, max_angular_velocity=1.0
    )
    points = np.column_stack((np.linspace(0.1, 3.0, 30), np.linspace(0.0, 1.5, 30)))
    command = controller.command(points)
    assert 0.0 < command[0] <= 1.0
    assert command[1] == 0.0
    assert command[2] > 0.0
    controller.reset()
    assert controller.yaw_error_filtered is None
