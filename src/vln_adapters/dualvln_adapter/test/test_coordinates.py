import math

import numpy as np

from dualvln_adapter.coordinates import trajectory_at_capture_to_current


def test_same_pose_keeps_ego_trajectory():
    points = np.asarray(((0.0, 0.0), (1.0, 0.5), (2.0, 1.0)))
    np.testing.assert_allclose(trajectory_at_capture_to_current(points, (3, 4, 0.7), (3, 4, 0.7)), points)


def test_compensates_motion_during_inference():
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)))
    actual = trajectory_at_capture_to_current(points, (0, 0, 0), (0.5, 0, 0))
    np.testing.assert_allclose(actual, ((-0.5, 0), (0.5, 0), (1.5, 0)))


def test_compensates_current_heading():
    points = np.asarray(((0.0, 0.0), (1.0, 0.0)))
    actual = trajectory_at_capture_to_current(points, (0, 0, 0), (0, 0, math.pi / 2))
    np.testing.assert_allclose(actual, ((0, 0), (0, -1)), atol=1.0e-8)
