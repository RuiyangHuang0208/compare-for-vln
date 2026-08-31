from __future__ import annotations

import json
import os

from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from sensor_msgs.msg import Image
import time
import yaml


class EpisodeManager(Node):
    def __init__(self, **kwargs):
        super().__init__("dynanav_episode_manager", **kwargs)
        default_config = os.path.join(get_package_share_directory("dynanav_bridge"), "config", "episodes.yaml")
        self.declare_parameter("episode_id", "hospital_001")
        self.declare_parameter("episodes_file", default_config)
        self.declare_parameter("auto_start_delay", 2.0)
        self.declare_parameter("model_name", "dummy")
        self.declare_parameter("evaluation_mode", "trajectory_normalized")
        self.declare_parameter("sensor_profile", "auto")
        self.declare_parameter("goal_profile", "none")
        self.declare_parameter("comparison_track", "untracked")
        self.declare_parameter("execution_profile", "fair")
        self.declare_parameter("desired_speed", 1.0)
        self.declare_parameter("camera_horizontal_fov_degrees", 79.0)
        self.declare_parameter("model_inputs", "")
        self.declare_parameter("model_runtime", "{}")
        self.declare_parameter("pedestrian_count", -1)
        episode_id = str(self.get_parameter("episode_id").value)
        with open(str(self.get_parameter("episodes_file").value), encoding="utf-8") as stream:
            episodes = yaml.safe_load(stream).get("episodes", {})
        if episode_id not in episodes:
            raise KeyError(f"Unknown episode {episode_id!r}; available={sorted(episodes)}")
        self.episode_id = episode_id
        self.episode = dict(episodes[episode_id])
        official_pedestrians = int(self.episode.get("pedestrian_count", 0))
        configured_pedestrians = int(self.get_parameter("pedestrian_count").value)
        if configured_pedestrians >= 0:
            self.episode["pedestrian_count"] = configured_pedestrians
        self.episode["official_pedestrian_count"] = official_pedestrians
        self.episode["effective_pedestrian_count"] = int(self.episode.get("pedestrian_count", 0))
        self.model_name = str(self.get_parameter("model_name").value)
        self.evaluation_mode = str(self.get_parameter("evaluation_mode").value)
        if self.evaluation_mode not in {"trajectory_normalized", "native_output"}:
            raise ValueError("evaluation_mode must be trajectory_normalized or native_output")
        self.sensor_profile = str(self.get_parameter("sensor_profile").value)
        if self.sensor_profile == "auto":
            self.sensor_profile = "rgb_d" if self.model_name == "dualvln" else "rgb_only"
        self.goal_profile = str(self.get_parameter("goal_profile").value)
        if self.episode.get("implemented", True) is False:
            raise RuntimeError(
                f"Episode {episode_id} requires DynaNav dynamic-person APIs not yet validated with Isaac Sim 5.1"
            )
        self.comparison_track = str(self.get_parameter("comparison_track").value)
        self.execution_profile = str(self.get_parameter("execution_profile").value)
        self.desired_speed = float(self.get_parameter("desired_speed").value)
        self.camera_horizontal_fov_degrees = float(
            self.get_parameter("camera_horizontal_fov_degrees").value
        )
        self.model_inputs = [
            item for item in str(self.get_parameter("model_inputs").value).split(",") if item
        ]
        try:
            self.model_runtime = json.loads(str(self.get_parameter("model_runtime").value))
        except json.JSONDecodeError as error:
            raise ValueError(f"model_runtime must be valid JSON: {error}") from error
        if not isinstance(self.model_runtime, dict):
            raise ValueError("model_runtime must decode to an object")
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.state_pub = self.create_publisher(String, "/episode/state", qos)
        self.episode_id_pub = self.create_publisher(String, "/episode/id", qos)
        self.instruction_pub = self.create_publisher(String, "/vln/instruction", qos)
        self.goal_pub = self.create_publisher(PoseStamped, "/episode/goal", qos)
        self.metadata_pub = self.create_publisher(String, "/episode/metadata", qos)
        self.create_subscription(String, "/episode/request", self.on_request, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self.on_odom, 10)
        self.create_subscription(Image, "/camera/rgb/image_raw", self.on_image, 5)
        self.started = False
        self.odom_ready = False
        self.image_ready = False
        self.ready_since = None
        self.start_delay = float(self.get_parameter("auto_start_delay").value)
        self.start_timer = self.create_timer(0.1, self.start_once)
        self.publish_state("RESET")
        self.get_logger().info(
            f"Selected {self.episode_id}: instruction={self.episode['instruction']!r} "
            f"scene={self.episode['scene']} goal={self.episode['goal']}"
        )

    def publish_state(self, state):
        self.state_pub.publish(String(data=state))

    def publish_episode(self):
        self.episode_id_pub.publish(String(data=self.episode_id))
        self.instruction_pub.publish(String(data=str(self.episode["instruction"])))
        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = "world"
        goal.pose.position.x = float(self.episode["goal"][0])
        goal.pose.position.y = float(self.episode["goal"][1])
        goal.pose.orientation.w = 1.0
        self.goal_pub.publish(goal)
        metadata = dict(self.episode)
        metadata["episode_id"] = self.episode_id
        metadata["model_name"] = self.model_name
        metadata["evaluation_mode"] = self.evaluation_mode
        metadata["sensor_profile"] = self.sensor_profile
        metadata["goal_profile"] = self.goal_profile
        metadata["comparison_track"] = self.comparison_track
        metadata["execution_profile"] = self.execution_profile
        metadata["desired_speed"] = self.desired_speed
        metadata["camera_horizontal_fov_degrees"] = self.camera_horizontal_fov_degrees
        metadata["model_inputs"] = self.model_inputs
        metadata["model_runtime"] = self.model_runtime
        self.metadata_pub.publish(String(data=json.dumps(metadata, separators=(",", ":"))))

    def start_once(self):
        if self.started:
            return
        if not self.odom_ready or not self.image_ready:
            return
        self.ready_since = self.ready_since or time.monotonic()
        if time.monotonic() - self.ready_since < self.start_delay:
            return
        self.started = True
        self.publish_episode()
        self.publish_state("START")
        self.get_logger().info(
            f"Started {self.episode_id}: instruction={self.episode['instruction']!r} "
            f"scene={self.episode['scene']} goal={self.episode['goal']} "
            f"pedestrians={self.episode.get('pedestrian_count', 0)}"
        )

    def on_odom(self, _message):
        self.odom_ready = True

    def on_image(self, _message):
        self.image_ready = True

    def on_request(self, message):
        request = message.data.strip().upper()
        if request == "RESET":
            self.started = False
            self.odom_ready = False
            self.image_ready = False
            self.ready_since = None
            self.publish_state("RESET")
            self.get_logger().info("Reset requested; waiting for fresh odometry and RGB before START")
        elif request in {"FINISH", "FAILED", "SUCCESS"}:
            self.publish_state(request)
        else:
            self.get_logger().warning(f"Unknown episode request {request!r}")


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
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
