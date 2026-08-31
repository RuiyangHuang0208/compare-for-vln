import math

import numpy as np

from navigation_bridge.core import (
    Limits,
    PurePursuitFollower,
    VelocityFilter,
    local_to_world,
)


def test_local_to_world_rotates_and_translates():
    actual = local_to_world(np.asarray(((1.0, 0.0), (0.0, 1.0))), (2.0, 3.0, math.pi / 2.0))
    np.testing.assert_allclose(actual, ((2.0, 4.0), (1.0, 3.0)), atol=1.0e-8)


def test_velocity_filter_clips_speed_and_acceleration():
    filter_ = VelocityFilter(Limits(1.0, 0.5, 0.8, 0.5, 1.0))
    np.testing.assert_allclose(filter_.apply((5.0, -5.0, 5.0), 0.1), (0.05, -0.05, 0.1))
    for _ in range(100):
        result = filter_.apply((5.0, -5.0, 5.0), 0.1)
    np.testing.assert_allclose(result, (1.0, -0.5, 0.8))


def test_follower_stops_at_final_point():
    follower = PurePursuitFollower(goal_tolerance=0.25)
    follower.set_path(((0.5, 0.0), (1.0, 0.0)))
    command, distance, reached = follower.command((0.9, 0.0, 0.0))
    assert reached
    assert distance < 0.25
    np.testing.assert_array_equal(command, np.zeros(3))
    assert not follower.active


def test_short_rolling_trajectory_is_not_an_episode_goal():
    follower = PurePursuitFollower(goal_tolerance=0.05)
    follower.set_path(((0.10, 0.0), (0.25, 0.0)))
    command, distance, reached = follower.command((0.0, 0.0, 0.0))
    assert not reached
    assert math.isclose(distance, 0.25)
    assert command[0] > 0.0
    assert follower.active


def test_follower_turns_toward_path_without_lateral_velocity():
    follower = PurePursuitFollower()
    follower.set_path(((0.5, 0.5), (1.0, 1.0)))
    command, _, reached = follower.command((0.0, 0.0, 0.0))
    assert not reached
    assert command[0] > 0.0
    assert command[1] == 0.0
    assert command[2] > 0.0


def test_long_path_uses_local_heading_once_for_rotated_robot():
    follower = PurePursuitFollower(lookahead=1.0)
    local_path = np.asarray(tuple((0.5 * index, 0.0) for index in range(1, 8)))
    world_path = local_to_world(local_path, (0.0, 0.0, math.pi / 2.0))

    follower.set_path(world_path)
    command, _, reached = follower.command((0.0, 0.0, math.pi / 2.0))

    assert not reached
    assert command[0] > 0.0
    assert abs(float(command[2])) < 0.1


def test_follower_requires_explicit_final_heading_after_reaching_xy_goal():
    follower = PurePursuitFollower(goal_tolerance=0.05, heading_tolerance=math.radians(5.0))
    follower.set_path(((0.10, 0.0), (0.25, 0.0)), final_yaw=math.radians(30.0))

    command, distance, reached = follower.command((0.25, 0.0, 0.0))
    assert not reached
    assert math.isclose(distance, 0.0)
    np.testing.assert_allclose(command[:2], (0.0, 0.0))
    assert command[2] > 0.0
    assert follower.active

    command, _, reached = follower.command((0.25, 0.0, math.radians(27.0)))
    assert reached
    np.testing.assert_array_equal(command, np.zeros(3))
    assert not follower.active


def test_heading_path_enters_rotation_mode_inside_capture_distance():
    follower = PurePursuitFollower(
        goal_tolerance=0.05,
        heading_capture_distance=0.10,
        heading_tolerance=math.radians(5.0),
    )
    follower.set_path(((0.10, 0.0), (0.25, 0.0)), final_yaw=math.radians(60.0))
    command, distance, reached = follower.command((0.17, 0.0, math.radians(30.0)))
    assert math.isclose(distance, 0.08)
    assert not reached
    np.testing.assert_allclose(command[:2], (0.0, 0.0))
    assert command[2] > 0.0


def test_velocity_filter_uses_separate_deceleration_limit():
    filter_ = VelocityFilter(Limits(1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 3.0))
    filter_.last[:] = (1.0, 0.0, 1.0)
    np.testing.assert_allclose(filter_.apply((0.0, 0.0, 0.0), 0.1), (0.8, 0.0, 0.7))
