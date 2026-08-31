from __future__ import annotations

import math
import json
import time

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String
from vln_interfaces.msg import NavigationCommand

from .core import (
    Limits,
    PurePursuitFollower,
    VelocityFilter,
    local_to_world,
    wrap_angle,
)


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NavigationBridge(Node):
    def __init__(self, **kwargs):
        super().__init__("navigation_bridge", **kwargs)
        defaults = {
            "command_topic": "/vln/command",
            "odom_topic": "/odom",
            "output_topic": "/nav_vel",
            "debug_path_topic": "/vln/debug_path",
            "desired_speed_topic": "/navigation/desired_speed",
            "current_speed_topic": "/navigation/desired_speed/current",
            "trajectory_finished_topic": "/navigation/trajectory_finished",
            "trajectory_failed_topic": "/navigation/trajectory_failed",
            "episode_state_topic": "/episode/state",
            "control_rate_hz": 20.0,
            "base_frame": "base_link",
            "world_frame": "world",
            "path_follower.lookahead_distance": 1.0,
            "path_follower.controller": "shared_pure_pursuit",
            "path_follower.goal_tolerance": 0.05,
            "path_follower.heading_tolerance": math.radians(5.0),
            "path_follower.heading_capture_distance": 0.10,
            "path_follower.desired_speed": 1.0,
            "path_follower.yaw_gain": 0.8,
            "path_follower.yaw_filter_alpha": 0.35,
            "path_follower.curvature_feedforward_gain": 0.5,
            "limits.max_vx": 2.0,
            "limits.max_vy": 0.5,
            "limits.max_wz": 1.0,
            "limits.max_linear_acceleration": 0.8,
            "limits.max_angular_acceleration": 1.5,
            "limits.max_linear_deceleration": 0.8,
            "limits.max_angular_deceleration": 1.5,
            "timeout.command_timeout": 0.5,
            "timeout.trajectory_timeout": 15.0,
            "runtime.lock_desired_speed": False,
            "stuck.command_speed_threshold": 0.1,
            "stuck.motion_speed_threshold": 0.03,
            "stuck.duration": 2.0,
            "stuck.recovery_enabled": True,
            "stuck.recovery_speed": -0.3,
            "stuck.recovery_duration": 1.5,
            "collision.recovery_enabled": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        param = lambda name: self.get_parameter(name).value
        self.base_frame = param("base_frame")
        self.world_frame = param("world_frame")
        self.command_timeout = float(param("timeout.command_timeout"))
        self.trajectory_timeout = float(param("timeout.trajectory_timeout"))
        self.control_rate = float(param("control_rate_hz"))
        limits = Limits(
            max_vx=float(param("limits.max_vx")),
            max_vy=float(param("limits.max_vy")),
            max_wz=float(param("limits.max_wz")),
            max_linear_acceleration=float(param("limits.max_linear_acceleration")),
            max_angular_acceleration=float(param("limits.max_angular_acceleration")),
            max_linear_deceleration=float(param("limits.max_linear_deceleration")),
            max_angular_deceleration=float(param("limits.max_angular_deceleration")),
        )
        desired_speed = float(param("path_follower.desired_speed"))
        if not math.isfinite(desired_speed) or desired_speed <= 0.0:
            raise ValueError("path_follower.desired_speed must be positive and finite")
        follower_kwargs = dict(
            lookahead=float(param("path_follower.lookahead_distance")),
            goal_tolerance=float(param("path_follower.goal_tolerance")),
            heading_tolerance=float(param("path_follower.heading_tolerance")),
            heading_capture_distance=float(param("path_follower.heading_capture_distance")),
            speed=min(desired_speed, limits.max_vx),
            yaw_gain=float(param("path_follower.yaw_gain")),
            yaw_filter_alpha=float(param("path_follower.yaw_filter_alpha")),
            curvature_feedforward_gain=float(param("path_follower.curvature_feedforward_gain")),
        )
        self.controller_mode = str(param("path_follower.controller"))
        if self.controller_mode == "shared_pure_pursuit":
            self.follower = PurePursuitFollower(**follower_kwargs)
        elif self.controller_mode == "discrete_action_path":
            # NaVILA and Uni-NaVid define their native discrete motion in the
            # adapter; the B2-W-specific execution of that path stays here.
            self.follower = PurePursuitFollower(**follower_kwargs)
        else:
            raise ValueError(
                "path_follower.controller must be shared_pure_pursuit or discrete_action_path"
            )
        self.velocity_filter = VelocityFilter(limits)
        self.maximum_path_speed = limits.max_vx
        self.lock_desired_speed = bool(param("runtime.lock_desired_speed"))
        self.velocity_target = np.zeros(3)
        self.mode = "stop"
        self.pose = None
        self.motion_speed = 0.0
        self.last_command_monotonic = None
        self.last_path_monotonic = None
        self.last_tick = time.monotonic()
        self.stuck_since = None
        self.stuck_active = False
        self.stuck_command_threshold = float(param("stuck.command_speed_threshold"))
        self.stuck_motion_threshold = float(param("stuck.motion_speed_threshold"))
        self.stuck_duration = float(param("stuck.duration"))
        self.recovery_enabled = bool(param("stuck.recovery_enabled"))
        self.recovery_speed = float(param("stuck.recovery_speed"))
        self.recovery_duration = float(param("stuck.recovery_duration"))
        if self.recovery_enabled and (self.recovery_speed >= 0.0 or self.recovery_duration <= 0.0):
            raise ValueError("stuck recovery requires recovery_speed < 0 and recovery_duration > 0")
        self.recovery_until = None
        self.collision_recovery_enabled = bool(param("collision.recovery_enabled"))

        self.velocity_pub = self.create_publisher(Twist, param("output_topic"), 10)
        self.path_pub = self.create_publisher(Path, param("debug_path_topic"), 10)
        self.stuck_pub = self.create_publisher(Bool, "/navigation/stuck", 10)
        self.current_speed_pub = self.create_publisher(Float32, param("current_speed_topic"), 10)
        self.trajectory_finished_pub = self.create_publisher(Bool, param("trajectory_finished_topic"), 10)
        self.trajectory_failed_pub = self.create_publisher(Bool, param("trajectory_failed_topic"), 10)
        self.create_subscription(NavigationCommand, param("command_topic"), self.on_command, 10)
        self.create_subscription(Odometry, param("odom_topic"), self.on_odom, 20)
        self.create_subscription(String, param("episode_state_topic"), self.on_episode_state, 10)
        self.create_subscription(String, "/simulation/collision", self.on_collision, 20)
        self.create_subscription(Float32, param("desired_speed_topic"), self.on_desired_speed, 10)
        self.create_timer(1.0 / self.control_rate, self.on_timer)
        self.publish_zero("startup")
        self.publish_current_speed()
        self.get_logger().info(f"High-level controller: {self.controller_mode}")

    def publish_current_speed(self):
        self.current_speed_pub.publish(Float32(data=float(self.follower.speed)))

    def on_desired_speed(self, message: Float32):
        speed = float(message.data)
        if self.lock_desired_speed:
            self.get_logger().warning(
                f"Rejected runtime speed change to {speed!r}; fair comparison speed is locked at "
                f"{self.follower.speed:.3f} m/s"
            )
            self.publish_current_speed()
            return
        if not math.isfinite(speed) or speed <= 0.0 or speed > self.maximum_path_speed:
            self.get_logger().warning(
                f"Rejected desired speed {speed!r}; expected 0 < speed <= {self.maximum_path_speed:.3f} m/s"
            )
            self.publish_current_speed()
            return
        self.follower.speed = speed
        self.publish_current_speed()
        self.get_logger().info(f"Path follower speed changed to {speed:.3f} m/s")

    def on_odom(self, message: Odometry):
        p, q = message.pose.pose.position, message.pose.pose.orientation
        values = np.asarray((p.x, p.y, yaw_from_quaternion(q)), dtype=np.float64)
        velocity = message.twist.twist
        if np.isfinite(values).all():
            self.pose = values
        speed_values = np.asarray((velocity.linear.x, velocity.linear.y, velocity.angular.z))
        self.motion_speed = float(np.linalg.norm(speed_values)) if np.isfinite(speed_values).all() else 0.0

    def on_episode_state(self, message: String):
        state = message.data.strip().upper()
        if state in {"RESET", "START", "FINISH", "FAILED", "SUCCESS"}:
            self.publish_zero(f"episode {state.lower()}")

    def on_collision(self, message: String):
        if not self.collision_recovery_enabled or self.recovery_until is not None or self.mode != "path":
            return
        try:
            event = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if bool(event.get("is_pedestrian", False)):
            return
        if self.mode == "path":
            # Invalidate discrete-action adapters immediately when recovery
            # clears the path; do not leave them waiting for a long timeout.
            self.trajectory_failed_pub.publish(Bool(data=True))
        self.follower.clear()
        self.mode = "stop"
        self.velocity_target.fill(0.0)
        self.velocity_filter.reset()
        self.recovery_until = time.monotonic() + self.recovery_duration
        self.get_logger().warning(
            f"Collision recovery: reversing at {self.recovery_speed:.2f} m/s "
            f"for {self.recovery_duration:.1f}s"
        )

    def _stamp_age(self, message: NavigationCommand) -> float:
        stamp_ns = int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)
        if stamp_ns <= 0:
            raise ValueError("zero timestamp")
        return (self.get_clock().now().nanoseconds - stamp_ns) / 1.0e9

    def _points(self, message: NavigationCommand):
        values = np.asarray([(point.x, point.y, point.theta) for point in message.points], dtype=np.float64)
        if values.ndim != 2 or values.shape[1:] != (3,) or not np.isfinite(values).all():
            raise ValueError("points must be a finite Nx3 Pose2D array")
        return values

    def on_command(self, message: NavigationCommand):
        if message.command_type == NavigationCommand.STOP:
            self.publish_zero("STOP command")
            return
        try:
            age = self._stamp_age(message)
            if age < -0.1 or age > self.command_timeout:
                raise ValueError(f"stale/future timestamp age={age:.3f}s")
            if not message.valid:
                raise ValueError("valid=false")
            if message.header.frame_id != self.base_frame:
                raise ValueError(f"frame_id must be {self.base_frame!r}")
            if not math.isfinite(message.confidence) or not 0.0 <= message.confidence <= 1.0:
                raise ValueError("confidence must be finite in [0,1]")
            now = time.monotonic()
            if message.command_type in (NavigationCommand.TRAJECTORY, NavigationCommand.WAYPOINT):
                if self.pose is None:
                    raise ValueError("odometry unavailable")
                points = self._points(message)
                minimum = 2 if message.command_type == NavigationCommand.TRAJECTORY else 1
                if len(points) < minimum:
                    raise ValueError(f"command requires at least {minimum} point(s)")
                world_path = local_to_world(points[:, :2], self.pose)
                # Existing continuous VLN trajectories usually leave theta at zero. Only
                # enforce heading when the adapter explicitly supplies a turn profile.
                heading_required = bool(np.any(np.abs(points[:, 2]) > 1.0e-6))
                final_yaw = wrap_angle(self.pose[2] + points[-1, 2]) if heading_required else None
                self.follower.set_path(world_path, final_yaw=final_yaw)
                self.mode = "path"
                self.last_path_monotonic = now
                self.trajectory_finished_pub.publish(Bool(data=False))
                self.trajectory_failed_pub.publish(Bool(data=False))
                self.publish_debug_path(world_path)
            elif message.command_type == NavigationCommand.VELOCITY:
                twist = message.velocity
                target = np.asarray((twist.linear.x, twist.linear.y, twist.angular.z), dtype=np.float64)
                if not np.isfinite(target).all():
                    raise ValueError("velocity contains NaN/Inf")
                self.follower.clear()
                self.velocity_target = target
                self.mode = "velocity"
            else:
                raise ValueError(f"unknown command_type={message.command_type}")
            self.last_command_monotonic = now
        except ValueError as error:
            self.get_logger().warning(f"Rejected NavigationCommand: {error}")
            self.publish_zero("invalid command")

    def publish_debug_path(self, world_path):
        message = Path()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.world_frame
        for x, y in world_path:
            pose = PoseStamped()
            pose.header = message.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            message.poses.append(pose)
        self.path_pub.publish(message)

    def publish_twist(self, command):
        message = Twist()
        message.linear.x, message.linear.y, message.angular.z = map(float, command)
        self.velocity_pub.publish(message)

    def publish_zero(self, reason: str):
        was_active = self.mode != "stop" or np.any(self.velocity_filter.last)
        self.mode = "stop"
        self.follower.clear()
        self.velocity_target.fill(0.0)
        self.velocity_filter.reset()
        self.last_command_monotonic = None
        self.last_path_monotonic = None
        self.stuck_since = None
        self.stuck_active = False
        self.recovery_until = None
        self.publish_twist(np.zeros(3))
        if was_active:
            self.publish_debug_path(np.empty((0, 2), dtype=np.float64))
            self.get_logger().info(f"Stopped: {reason}")

    def on_timer(self):
        now = time.monotonic()
        dt = max(0.0, now - self.last_tick)
        self.last_tick = now
        if self.recovery_until is not None:
            if now < self.recovery_until:
                command = self.velocity_filter.apply((self.recovery_speed, 0.0, 0.0), dt)
                self.publish_twist(command)
                return
            self.recovery_until = None
            self.velocity_filter.reset()
            self.mode = "stop"
            self.get_logger().info("Finished stuck recovery; waiting for a fresh trajectory")
        if self.mode == "stop":
            self.publish_twist(np.zeros(3))
            return
        if self.mode == "path":
            if self.last_path_monotonic is None or now - self.last_path_monotonic > self.trajectory_timeout:
                self.publish_zero("trajectory timeout")
                self.trajectory_failed_pub.publish(Bool(data=True))
                return
            if self.pose is None:
                self.publish_zero("odometry unavailable")
                self.trajectory_failed_pub.publish(Bool(data=True))
                return
            target, _, reached = self.follower.command(self.pose)
            if reached:
                self.publish_zero("trajectory complete")
                self.trajectory_finished_pub.publish(Bool(data=True))
                return
        else:
            if self.last_command_monotonic is None or now - self.last_command_monotonic > self.command_timeout:
                self.publish_zero("command timeout")
                return
            target = self.velocity_target
        command = self.velocity_filter.apply(target, dt)
        self.publish_twist(command)
        commanded_speed = float(np.linalg.norm(command))
        if commanded_speed > self.stuck_command_threshold and self.motion_speed < self.stuck_motion_threshold:
            self.stuck_since = self.stuck_since or now
            if not self.stuck_active and now - self.stuck_since >= self.stuck_duration:
                self.stuck_active = True
                self.stuck_pub.publish(Bool(data=True))
                if self.recovery_enabled and self.mode == "path":
                    self.trajectory_failed_pub.publish(Bool(data=True))
                    # Discard the trajectory that led into the obstacle. The model
                    # adapter will publish a fresh local trajectory after recovery.
                    self.follower.clear()
                    self.mode = "stop"
                    self.velocity_filter.reset()
                    self.recovery_until = now + self.recovery_duration
                    self.get_logger().warning(
                        f"Stuck recovery: reversing at {self.recovery_speed:.2f} m/s "
                        f"for {self.recovery_duration:.1f}s"
                    )
        else:
            if self.stuck_active:
                self.stuck_pub.publish(Bool(data=False))
            self.stuck_since = None
            self.stuck_active = False


def main():
    rclpy.init()
    node = NavigationBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.publish_zero("shutdown")
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
