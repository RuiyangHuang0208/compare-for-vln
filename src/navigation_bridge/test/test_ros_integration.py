#!/usr/bin/env python3
from __future__ import annotations

import math
import json
import threading
import time
import unittest

from geometry_msgs.msg import Pose2D, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Float32, String
from vln_interfaces.msg import NavigationCommand

from navigation_bridge.navigation_bridge_node import NavigationBridge


class NavigationBridgeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        overrides = [
            Parameter("timeout.command_timeout", value=0.25),
            Parameter("timeout.trajectory_timeout", value=0.5),
            Parameter("limits.max_linear_acceleration", value=100.0),
            Parameter("limits.max_angular_acceleration", value=100.0),
            Parameter("limits.max_linear_deceleration", value=100.0),
            Parameter("limits.max_angular_deceleration", value=100.0),
        ]
        self.bridge = NavigationBridge(parameter_overrides=overrides)
        self.test_node = Node("navigation_bridge_test_client")
        self.command_pub = self.test_node.create_publisher(NavigationCommand, "/vln/command", 10)
        self.odom_pub = self.test_node.create_publisher(Odometry, "/odom", 10)
        self.speed_pub = self.test_node.create_publisher(Float32, "/navigation/desired_speed", 10)
        self.collision_pub = self.test_node.create_publisher(String, "/simulation/collision", 10)
        self.received = []
        self.trajectory_states = []
        self.trajectory_failures = []
        self.test_node.create_subscription(Twist, "/nav_vel", self.received.append, 10)
        self.test_node.create_subscription(
            Bool, "/navigation/trajectory_finished", self.trajectory_states.append, 10
        )
        self.test_node.create_subscription(
            Bool, "/navigation/trajectory_failed", self.trajectory_failures.append, 10
        )
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.bridge)
        self.executor.add_node(self.test_node)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        time.sleep(0.35)
        odom = Odometry()
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        odom.pose.pose.orientation.w = 1.0
        self.odom_pub.publish(odom)
        time.sleep(0.1)

    def tearDown(self):
        self.executor.shutdown(timeout_sec=1.0)
        self.thread.join(timeout=1.0)
        self.bridge.destroy_node()
        self.test_node.destroy_node()

    def command(self, kind, points=(), velocity=(0.0, 0.0, 0.0), valid=True, stale=False):
        message = NavigationCommand()
        message.header.stamp = self.test_node.get_clock().now().to_msg()
        if stale:
            message.header.stamp.sec -= 10
        message.header.frame_id = "base_link"
        message.command_type = kind
        message.points = [
            Pose2D(
                x=float(point[0]),
                y=float(point[1]),
                theta=float(point[2]) if len(point) > 2 else 0.0,
            )
            for point in points
        ]
        message.velocity.linear.x = float(velocity[0])
        message.velocity.linear.y = float(velocity[1])
        message.velocity.angular.z = float(velocity[2])
        message.confidence = 1.0
        message.valid = valid
        self.command_pub.publish(message)

    def wait_for(self, predicate, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.received and predicate(self.received[-1]):
                return self.received[-1]
            time.sleep(0.01)
        last = self.received[-1] if self.received else None
        self.fail(f"Timed out waiting for velocity; last={last}")

    @staticmethod
    def is_zero(message):
        return all(
            math.isclose(value, 0.0, abs_tol=1.0e-9)
            for value in (message.linear.x, message.linear.y, message.angular.z)
        )

    def test_all_command_routes_and_timeout(self):
        self.assertAlmostEqual(self.bridge.follower.speed, 1.0)
        self.assertAlmostEqual(self.bridge.maximum_path_speed, 2.0)

        self.command(NavigationCommand.TRAJECTORY, points=((0.5, 0.0), (2.0, 0.0)))
        trajectory = self.wait_for(lambda msg: msg.linear.x > 0.1)
        self.assertAlmostEqual(trajectory.linear.y, 0.0)

        self.speed_pub.publish(Float32(data=2.0))
        time.sleep(0.1)
        self.command(NavigationCommand.TRAJECTORY, points=((0.5, 0.0), (2.0, 0.0)))
        faster = self.wait_for(lambda msg: msg.linear.x > 1.9)
        self.assertLessEqual(faster.linear.x, 2.0)

        self.speed_pub.publish(Float32(data=2.5))
        time.sleep(0.1)
        self.command(NavigationCommand.TRAJECTORY, points=((0.5, 0.0), (2.0, 0.0)))
        rejected = self.wait_for(lambda msg: msg.linear.x > 1.9)
        self.assertLessEqual(rejected.linear.x, 2.0)

        self.command(NavigationCommand.WAYPOINT, points=((2.0, 1.0),))
        waypoint = self.wait_for(lambda msg: msg.linear.x > 0.1 and msg.angular.z > 0.1)
        self.assertAlmostEqual(waypoint.linear.y, 0.0)

        self.command(NavigationCommand.VELOCITY, velocity=(0.2, 0.1, -0.2))
        direct = self.wait_for(
            lambda msg: math.isclose(msg.linear.y, 0.1, abs_tol=1.0e-3)
            and math.isclose(msg.angular.z, -0.2, abs_tol=1.0e-3)
        )
        self.assertAlmostEqual(direct.linear.x, 0.2, places=3)

        self.command(NavigationCommand.STOP)
        self.wait_for(self.is_zero)

        self.command(NavigationCommand.TRAJECTORY, points=((math.nan, 0.0),), valid=False)
        self.wait_for(self.is_zero)

        self.command(NavigationCommand.VELOCITY, velocity=(0.2, 0.0, 0.0), stale=True)
        self.wait_for(self.is_zero)

        self.command(NavigationCommand.VELOCITY, velocity=(0.2, 0.0, 0.0))
        self.wait_for(lambda msg: msg.linear.x > 0.1)
        self.wait_for(self.is_zero, timeout=0.8)

    def test_fair_comparison_rejects_runtime_speed_change(self):
        self.bridge.lock_desired_speed = True
        self.assertAlmostEqual(self.bridge.follower.speed, 1.0)
        self.speed_pub.publish(Float32(data=2.0))
        time.sleep(0.15)
        self.assertAlmostEqual(self.bridge.follower.speed, 1.0)

    def test_trajectory_finished_is_only_true_after_reaching_path_end(self):
        self.command(NavigationCommand.TRAJECTORY, points=((0.01, 0.0), (0.02, 0.0)))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not any(item.data for item in self.trajectory_states):
            time.sleep(0.01)
        self.assertTrue(self.trajectory_states)
        self.assertFalse(self.trajectory_states[0].data)
        self.assertTrue(self.trajectory_states[-1].data)

    def test_trajectory_timeout_reports_failure_not_completion(self):
        self.command(NavigationCommand.TRAJECTORY, points=((2.0, 0.0), (3.0, 0.0)))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not any(item.data for item in self.trajectory_failures):
            time.sleep(0.01)
        self.assertTrue(any(item.data for item in self.trajectory_failures))
        self.assertFalse(any(item.data for item in self.trajectory_states))

    def test_static_collision_invalidates_active_trajectory_immediately(self):
        self.command(NavigationCommand.TRAJECTORY, points=((2.0, 0.0), (3.0, 0.0)))
        time.sleep(0.1)
        collision = String(data=json.dumps({"is_pedestrian": False, "object": "wall"}))
        self.collision_pub.publish(collision)
        deadline = time.monotonic() + 0.6
        while time.monotonic() < deadline and not any(item.data for item in self.trajectory_failures):
            time.sleep(0.01)
        self.assertTrue(any(item.data for item in self.trajectory_failures))

    def test_explicit_path_heading_is_completed_only_after_odom_reaches_yaw(self):
        self.command(
            NavigationCommand.TRAJECTORY,
            points=((0.01, 0.0, math.radians(15.0)), (0.02, 0.0, math.radians(30.0))),
        )
        turning = self.wait_for(lambda msg: msg.angular.z > 0.1)
        self.assertAlmostEqual(turning.linear.x, 0.0)
        self.assertFalse(any(item.data for item in self.trajectory_states))

        odom = Odometry()
        odom.header.frame_id = "world"
        odom.child_frame_id = "base_link"
        yaw = math.radians(30.0)
        odom.pose.pose.orientation.z = math.sin(yaw / 2.0)
        odom.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.odom_pub.publish(odom)
        deadline = time.monotonic() + 0.4
        while time.monotonic() < deadline and not any(item.data for item in self.trajectory_states):
            time.sleep(0.01)
        self.assertTrue(any(item.data for item in self.trajectory_states))


if __name__ == "__main__":
    unittest.main()
