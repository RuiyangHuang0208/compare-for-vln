import math
import threading
import time
from unittest.mock import patch

import numpy as np
import pytest
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String
from vln_interfaces.msg import NavigationCommand

from omnivla_adapter.omnivla_node import OmniVLAAdapter


def test_stub_service_publishes_eight_point_base_link_trajectory_and_reset_stop():
    rclpy.init()
    adapter = client = executor = thread = None

    def health(_self):
        return {"status": "ready", "variant": "stub", "modality_id": 7}

    def reset(_self, episode_id, generation):
        return {"episode_id": episode_id, "generation": generation}

    def step(_self, _image, metadata):
        raw = np.zeros((8, 4), dtype=float)
        raw[:, 0] = np.arange(1, 9)
        raw[:, 2] = 1.0
        return {
            "request_id": metadata["request_id"],
            "episode_id": metadata["episode_id"],
            "generation": metadata["generation"],
            "variant": "stub",
            "resume_step": None,
            "modality_id": 7,
            "raw_trajectory_shape": [8, 4],
            "raw_trajectory_8x4": raw.tolist(),
            "inference_latency": 0.01,
            "service_latency": 0.02,
            "peak_gpu_memory_bytes": 0,
        }

    with (
        patch("omnivla_adapter.inference_client.OmniVLAInferenceClient.health", health),
        patch("omnivla_adapter.inference_client.OmniVLAInferenceClient.reset", reset),
        patch("omnivla_adapter.inference_client.OmniVLAInferenceClient.step", step),
    ):
        try:
            adapter = OmniVLAAdapter(
                parameter_overrides=[
                    Parameter("runtime.allow_stub_server", value=True),
                    Parameter("input.image_width", value=2),
                    Parameter("input.image_height", value=2),
                    Parameter("input.inference_rate_hz", value=50.0),
                    Parameter("runtime.poll_rate_hz", value=100.0),
                ]
            )
            client = Node("omnivla_adapter_test_client")
            id_pub = client.create_publisher(String, "/episode/id", 10)
            instruction_pub = client.create_publisher(String, "/vln/instruction", 10)
            state_pub = client.create_publisher(String, "/episode/state", 10)
            image_pub = client.create_publisher(Image, "/camera/rgb/image_raw", 10)
            commands, raw_messages, latencies = [], [], []
            client.create_subscription(NavigationCommand, "/vln/command", commands.append, 10)
            client.create_subscription(
                Float32MultiArray, "/vln/omnivla/raw_trajectory", raw_messages.append, 10
            )
            client.create_subscription(Float32MultiArray, "/vln/inference_latency", latencies.append, 10)
            executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(adapter)
            executor.add_node(client)
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
            time.sleep(0.1)
            id_pub.publish(String(data="stub_episode"))
            instruction_pub.publish(String(data="Go to the chair"))
            state_pub.publish(String(data="START"))
            time.sleep(0.15)
            image = Image(height=2, width=2, encoding="rgb8", step=6)
            image.data = bytes(range(12))
            image_pub.publish(image)

            deadline = time.monotonic() + 2.0
            trajectory = None
            while time.monotonic() < deadline:
                trajectory = next(
                    (item for item in reversed(commands) if item.command_type == NavigationCommand.TRAJECTORY),
                    None,
                )
                if trajectory is not None:
                    break
                time.sleep(0.01)
            assert trajectory is not None
            assert trajectory.header.frame_id == "base_link"
            assert len(trajectory.points) == 8
            assert trajectory.points[-1].x == pytest.approx(0.8)
            assert trajectory.dt == pytest.approx(1.0 / 3.0)
            assert trajectory.horizon == pytest.approx(8.0 / 3.0)
            assert raw_messages and len(raw_messages[-1].data) == 32
            assert latencies and latencies[-1].data[0] == pytest.approx(0.01)
            assert math.isnan(latencies[-1].data[1])
            assert math.isnan(latencies[-1].data[2])

            state_pub.publish(String(data="RESET"))
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and (
                not commands or commands[-1].command_type != NavigationCommand.STOP
            ):
                time.sleep(0.01)
            assert commands[-1].command_type == NavigationCommand.STOP
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
