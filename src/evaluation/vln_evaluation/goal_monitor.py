from __future__ import annotations

import json
import math

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vln_interfaces.msg import NavigationCommand


class GoalMonitor(Node):
    def __init__(self, **kwargs):
        super().__init__("goal_monitor", **kwargs)
        self.declare_parameter("success_threshold", 0.5)
        self.goal = None
        self.threshold = float(self.get_parameter("success_threshold").value)
        self.active = False
        self.completed = False
        self.last_distance = math.inf
        self.command_pub = self.create_publisher(NavigationCommand, "/vln/command", 10)
        self.velocity_pub = self.create_publisher(Twist, "/nav_vel", 10)
        self.request_pub = self.create_publisher(String, "/episode/request", 10)
        self.error_pub = self.create_publisher(String, "/episode/navigation_error", 10)
        self.create_subscription(PoseStamped, "/episode/goal", self.on_goal, 10)
        self.create_subscription(String, "/episode/metadata", self.on_metadata, 10)
        self.create_subscription(String, "/episode/state", self.on_state, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self.on_odom, 20)
        self.create_subscription(String, "/episode/model_stop", self.on_model_stop, 10)

    def on_goal(self, message):
        self.goal = (message.pose.position.x, message.pose.position.y)

    def on_metadata(self, message):
        try:
            metadata = json.loads(message.data)
            self.threshold = float(metadata.get("success_threshold", self.threshold))
        except (ValueError, TypeError, json.JSONDecodeError):
            self.get_logger().warning("Rejected invalid episode metadata")

    def on_state(self, message):
        state = message.data.strip().upper()
        if state == "RESET":
            self.active = False
            self.completed = False
            self.last_distance = math.inf
            self.publish_stop(False)
        elif state == "START":
            self.active = True
            self.completed = False
        elif state in {"FINISH", "FAILED", "SUCCESS"}:
            self.active = False

    def publish_stop(self, goal_reached):
        command = NavigationCommand()
        command.header.stamp = self.get_clock().now().to_msg()
        command.header.frame_id = "base_link"
        command.command_type = NavigationCommand.STOP
        command.confidence = 1.0
        command.valid = True
        command.goal_reached = bool(goal_reached)
        self.command_pub.publish(command)
        self.velocity_pub.publish(Twist())

    def on_odom(self, message):
        if self.goal is None:
            return
        position = message.pose.pose.position
        distance = math.hypot(self.goal[0] - position.x, self.goal[1] - position.y)
        self.last_distance = distance
        self.error_pub.publish(String(data=f"{distance:.9f}"))
        # TIC-VLA's DynaNav benchmark uses a strict distance comparison
        # (distance < success_threshold).  Keep the same rule here so the
        # ROS-side result cannot disagree at the exact boundary.
        if self.active and not self.completed and distance < self.threshold:
            self.completed = True
            self.active = False
            self.publish_stop(True)
            self.request_pub.publish(String(data="SUCCESS"))
            self.get_logger().info(f"Goal reached at navigation_error={distance:.3f} m")

    def on_model_stop(self, _message):
        if not self.active or self.completed:
            return
        self.completed = True
        self.active = False
        success = math.isfinite(self.last_distance) and self.last_distance < self.threshold
        self.publish_stop(success)
        self.request_pub.publish(String(data="SUCCESS" if success else "FAILED"))
        self.get_logger().info(
            f"Model requested STOP at navigation_error={self.last_distance:.3f} m; "
            f"episode={'SUCCESS' if success else 'FAILED'}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = GoalMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
