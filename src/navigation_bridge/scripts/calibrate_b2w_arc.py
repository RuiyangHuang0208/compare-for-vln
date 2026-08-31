#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time

from geometry_msgs.msg import Pose2D
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, String
from vln_interfaces.msg import NavigationCommand


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


class ArcCalibration(Node):
    def __init__(self, radius: float, degrees: float, direction: str, spacing: float):
        super().__init__("b2w_arc_calibration")
        self.radius = radius
        self.angle = math.radians(degrees)
        self.sign = 1.0 if direction == "left" else -1.0
        self.spacing = spacing
        self.pose = None
        self.finished = False
        self.failed = False
        self.command_pub = self.create_publisher(NavigationCommand, "/vln/command", 10)
        self.state_pub = self.create_publisher(String, "/episode/state", 10)
        self.create_subscription(Odometry, "/odom", self.on_odom, 20)
        self.create_subscription(Bool, "/navigation/trajectory_finished", self.on_finished, 10)
        self.create_subscription(Bool, "/navigation/trajectory_failed", self.on_failed, 10)

    def on_odom(self, message: Odometry):
        p = message.pose.pose.position
        self.pose = (float(p.x), float(p.y), yaw_from_quaternion(message.pose.pose.orientation))

    def on_finished(self, message: Bool):
        self.finished = self.finished or bool(message.data)

    def on_failed(self, message: Bool):
        self.failed = self.failed or bool(message.data)

    def reset(self):
        self.state_pub.publish(String(data="RESET"))

    def publish_arc(self):
        count = max(2, math.ceil(self.radius * self.angle / self.spacing))
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "base_link"
        message.command_type = NavigationCommand.TRAJECTORY
        message.points = [
            Pose2D(
                x=self.radius * math.sin(self.angle * index / count),
                y=self.sign * self.radius * (1.0 - math.cos(self.angle * index / count)),
                theta=self.sign * self.angle * index / count,
            )
            for index in range(1, count + 1)
        ]
        message.dt = 0.1
        message.horizon = 15.0
        message.confidence = 1.0
        message.valid = True
        self.command_pub.publish(message)


def spin_until(node: ArcCalibration, predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Measure one B2-W arc through NavigationCommand.")
    parser.add_argument("direction", choices=("left", "right"))
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--degrees", type=float, default=30.0)
    parser.add_argument("--spacing", type=float, default=0.05)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args()
    if args.radius <= 0.0 or args.degrees <= 0.0 or args.spacing <= 0.0:
        parser.error("radius, degrees, and spacing must be positive")

    rclpy.init()
    node = ArcCalibration(args.radius, args.degrees, args.direction, args.spacing)
    try:
        if not spin_until(
            node,
            lambda: node.command_pub.get_subscription_count() > 0 and node.pose is not None,
            10.0,
        ):
            raise RuntimeError("navigation bridge or odometry is unavailable")
        node.reset()
        # Wait for reset telemetry to settle before defining the local path origin.
        settle_deadline = time.monotonic() + 1.5
        while time.monotonic() < settle_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
        start = node.pose
        node.finished = False
        node.failed = False
        node.publish_arc()
        completed = spin_until(node, lambda: node.finished or node.failed, args.timeout)
        end = node.pose
        result = {
            "direction": args.direction,
            "requested_radius_m": args.radius,
            "requested_yaw_degrees": args.degrees * node.sign,
            "completed": completed and node.finished and not node.failed,
            "failed": node.failed,
            "start": {"x": start[0], "y": start[1], "yaw_degrees": math.degrees(start[2])},
            "end": {"x": end[0], "y": end[1], "yaw_degrees": math.degrees(end[2])},
            "delta_distance_m": math.hypot(end[0] - start[0], end[1] - start[1]),
            "delta_yaw_degrees": math.degrees(wrap_angle(end[2] - start[2])),
        }
        print(json.dumps(result, indent=2))
        if not result["completed"]:
            raise SystemExit(1)
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
