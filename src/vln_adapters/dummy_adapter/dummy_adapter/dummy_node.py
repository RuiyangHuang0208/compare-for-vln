from __future__ import annotations

import math

from geometry_msgs.msg import Pose2D
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from vln_interfaces.msg import NavigationCommand


VALID_MODES = (
    "straight",
    "left_turn",
    "right_turn",
    "waypoint",
    "velocity",
    "stop",
    "invalid",
    "stale",
)


class DummyAdapter(Node):
    def __init__(self):
        super().__init__("dummy_adapter")
        self.declare_parameter("mode", "straight")
        self.declare_parameter("publish_rate_hz", 5.0)
        self.declare_parameter("command_topic", "/vln/command")
        self.declare_parameter("episode_state_topic", "/episode/state")
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("wait_for_episode_start", False)
        self.declare_parameter("velocity.vx", 0.3)
        self.declare_parameter("velocity.vy", 0.0)
        self.declare_parameter("velocity.wz", 0.0)
        self.declare_parameter("trajectory.turn_radius", 1.0)
        self.declare_parameter("trajectory.turn_degrees", 45.0)
        self.declare_parameter("trajectory.spacing", 0.1)

        self.mode = str(self.get_parameter("mode").value)
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {self.mode!r}")
        rate = float(self.get_parameter("publish_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        topic = str(self.get_parameter("command_topic").value)
        episode_topic = str(self.get_parameter("episode_state_topic").value)
        self.publisher = self.create_publisher(NavigationCommand, topic, 10)
        self.create_subscription(String, episode_topic, self.on_episode_state, 10)
        self.enabled = not bool(self.get_parameter("wait_for_episode_start").value)
        self.create_timer(1.0 / rate, self.publish_command)
        self.get_logger().info(f"Publishing dummy mode={self.mode!r} on {topic}")

    def on_episode_state(self, message: String):
        state = message.data.strip().upper()
        if state in {"RESET", "FINISH", "FAILED", "SUCCESS"}:
            self.enabled = False
            self.publish_stop()
        elif state == "START":
            self.enabled = True

    def make_message(self) -> NavigationCommand:
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = str(self.get_parameter("frame_id").value)
        message.dt = 0.1
        message.horizon = 3.0
        message.confidence = 1.0
        message.valid = True
        message.goal_reached = False

        if self.mode == "straight":
            message.command_type = NavigationCommand.TRAJECTORY
            message.points = [Pose2D(x=0.1 * i, y=0.0, theta=0.0) for i in range(1, 31)]
        elif self.mode in {"left_turn", "right_turn"}:
            message.command_type = NavigationCommand.TRAJECTORY
            radius = float(self.get_parameter("trajectory.turn_radius").value)
            degrees = float(self.get_parameter("trajectory.turn_degrees").value)
            spacing = float(self.get_parameter("trajectory.spacing").value)
            if radius <= 0.0 or degrees <= 0.0 or spacing <= 0.0:
                raise ValueError("dummy arc radius, degrees, and spacing must be positive")
            angle = math.radians(degrees)
            count = max(2, math.ceil(radius * angle / spacing))
            sign = 1.0 if self.mode == "left_turn" else -1.0
            message.points = [
                Pose2D(
                    x=radius * math.sin(angle * i / count),
                    y=sign * radius * (1.0 - math.cos(angle * i / count)),
                    theta=sign * angle * i / count,
                )
                for i in range(1, count + 1)
            ]
        elif self.mode == "waypoint":
            message.command_type = NavigationCommand.WAYPOINT
            message.points = [Pose2D(x=2.0, y=1.0, theta=0.0)]
        elif self.mode == "velocity":
            message.command_type = NavigationCommand.VELOCITY
            message.velocity.linear.x = float(self.get_parameter("velocity.vx").value)
            message.velocity.linear.y = float(self.get_parameter("velocity.vy").value)
            message.velocity.angular.z = float(self.get_parameter("velocity.wz").value)
        elif self.mode == "stop":
            message.command_type = NavigationCommand.STOP
        elif self.mode == "invalid":
            message.command_type = NavigationCommand.TRAJECTORY
            message.points = [Pose2D(x=math.nan, y=0.0, theta=0.0)]
            message.valid = False
        elif self.mode == "stale":
            message.command_type = NavigationCommand.VELOCITY
            message.header.stamp.sec -= 10
            message.velocity.linear.x = 0.3
        return message

    def publish_stop(self):
        message = NavigationCommand()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = str(self.get_parameter("frame_id").value)
        message.command_type = NavigationCommand.STOP
        message.valid = True
        message.confidence = 1.0
        self.publisher.publish(message)

    def publish_command(self):
        if self.enabled:
            self.publisher.publish(self.make_message())


def main(args=None):
    rclpy.init(args=args)
    node = DummyAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_stop()
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
