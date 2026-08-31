from __future__ import annotations

import threading
import time
from unittest.mock import patch

import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, Header, String
from vln_interfaces.msg import NavigationCommand

from mobilevla_r1_adapter.mobilevla_r1_node import MobileVLAR1Adapter


def _image_messages(node):
    stamp = node.get_clock().now().to_msg()
    header = Header(stamp=stamp, frame_id="camera_front_optical")
    rgb = Image(header=header, height=4, width=6, encoding="rgb8", step=18)
    rgb.data = np.arange(72, dtype=np.uint8).tobytes()
    depth = Image(header=header, height=4, width=6, encoding="32FC1", step=24)
    depth.data = np.full((4, 6), 2.0, dtype=np.float32).tobytes()
    info = CameraInfo(header=header, height=4, width=6)
    info.k = [4.0, 0.0, 2.5, 0.0, 4.0, 1.5, 0.0, 0.0, 1.0]
    return rgb, depth, info


def test_stub_response_publishes_only_velocity_then_stop_without_privileged_topics():
    def health(_self):
        return {
            "status": "ready",
            "variant": "stub",
            "history_frames": 8,
            "depth_frames": 1,
            "pointcloud_points": 2048,
            "expected_vector_length": 12,
        }

    def reset(_self, episode_id, generation):
        return {"episode_id": episode_id, "generation": generation}

    def step(_self, rgb, depth, pointcloud, metadata):
        assert rgb.shape == (8, 4, 6, 3)
        assert depth.shape == (4, 6)
        assert pointcloud.shape == (2048, 3)
        return {
            "request_id": metadata["request_id"],
            "episode_id": metadata["episode_id"],
            "generation": metadata["generation"],
            "raw_response": (
                "<think>Use a moderate command.</think>"
                "<answer>[0.25, -0.1, 0.2, 9, 8, 7, 6, 5, 4, 3, 2, 1]</answer>"
            ),
            "inference_latency": 0.02,
        }

    rclpy.init()
    adapter = client = executor = thread = None
    with (
        patch("mobilevla_r1_adapter.inference_client.MobileVLAR1InferenceClient.health", health),
        patch("mobilevla_r1_adapter.inference_client.MobileVLAR1InferenceClient.reset", reset),
        patch("mobilevla_r1_adapter.inference_client.MobileVLAR1InferenceClient.step", step),
    ):
        try:
            adapter = MobileVLAR1Adapter(
                parameter_overrides=[
                    Parameter("runtime.allow_stub_server", value=True),
                    Parameter("control.command_duration_s", value=0.15),
                    Parameter("runtime.poll_rate_hz", value=100.0),
                    Parameter("input.maximum_sensor_age_s", value=2.0),
                ]
            )
            client = Node("mobilevla_r1_adapter_test_client")
            episode_id_pub = client.create_publisher(String, "/episode/id", 10)
            instruction_pub = client.create_publisher(String, "/vln/instruction", 10)
            state_pub = client.create_publisher(String, "/episode/state", 10)
            rgb_pub = client.create_publisher(Image, "/camera/rgb/image_raw", 10)
            depth_pub = client.create_publisher(Image, "/camera/depth/image_raw", 10)
            info_pub = client.create_publisher(CameraInfo, "/camera/rgb/camera_info", 10)
            commands, parsed = [], []
            client.create_subscription(NavigationCommand, "/vln/command", commands.append, 10)
            client.create_subscription(
                Float32MultiArray, "/vln/mobilevla_r1/parsed_velocity", parsed.append, 10
            )
            executor = MultiThreadedExecutor(num_threads=3)
            executor.add_node(adapter)
            executor.add_node(client)
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
            time.sleep(0.1)
            episode_id_pub.publish(String(data="stub_episode"))
            instruction_pub.publish(String(data="Walk to the chair"))
            state_pub.publish(String(data="START"))
            time.sleep(0.15)
            rgb, depth, info = _image_messages(client)
            rgb_pub.publish(rgb)
            depth_pub.publish(depth)
            info_pub.publish(info)

            deadline = time.monotonic() + 2.0
            velocity = None
            while time.monotonic() < deadline:
                velocity = next(
                    (item for item in commands if item.command_type == NavigationCommand.VELOCITY),
                    None,
                )
                if velocity is not None:
                    break
                time.sleep(0.01)
            assert velocity is not None
            assert velocity.header.frame_id == "base_link"
            assert velocity.velocity.linear.x == pytest.approx(0.25)
            assert velocity.velocity.linear.y == pytest.approx(-0.1)
            assert velocity.velocity.angular.z == pytest.approx(0.2)
            assert parsed and list(parsed[-1].data) == pytest.approx([0.25, -0.1, 0.2])

            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and commands[-1].command_type != NavigationCommand.STOP:
                time.sleep(0.01)
            assert commands[-1].command_type == NavigationCommand.STOP
            assert adapter.get_subscriptions_info_by_topic("/scan") == []
            assert adapter.get_subscriptions_info_by_topic("/points") == []
            assert adapter.get_publishers_info_by_topic("/nav_vel") == []
            assert adapter.get_publishers_info_by_topic("/joint_commands") == []
        finally:
            if executor is not None:
                executor.shutdown(timeout_sec=1.0)
            if thread is not None:
                thread.join(timeout=1.0)
            if adapter is not None:
                adapter.close()
                adapter.destroy_node()
            if client is not None:
                client.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
