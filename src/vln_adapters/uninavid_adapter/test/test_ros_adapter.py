import json
import math
import threading
import time
from unittest.mock import patch

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float32MultiArray, String
from vln_interfaces.msg import NavigationCommand

from uninavid_adapter.uninavid_node import UniNaVidAdapter


class StubResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_stub_service_drives_trajectory_then_deferred_stop():
    rclpy.init()
    requests_seen = []
    adapter = None
    client = None
    executor = None
    thread = None

    def fake_post(url, **kwargs):
        requests_seen.append(url)
        if url.endswith("/reset"):
            metadata = kwargs["json"]
            return StubResponse({"episode_id": metadata["episode_id"], "generation": metadata["generation"]})
        metadata = json.loads(kwargs["data"]["json"])
        return StubResponse(
            {
                "request_id": metadata["request_id"],
                "episode_id": metadata["episode_id"],
                "generation": metadata["generation"],
                "raw_action": "forward stop",
                "latency": 0.01,
                "new_frames": len(kwargs["files"]),
            }
        )

    overrides = [
        Parameter("input.frame_sample_hz", value=100.0),
        Parameter("conversion.turn_radius", value=1.0),
        Parameter("runtime.poll_rate_hz", value=100.0),
    ]
    with patch("uninavid_adapter.uninavid_node.requests.post", side_effect=fake_post):
        try:
            adapter = UniNaVidAdapter(parameter_overrides=overrides)
            client = Node("uninavid_adapter_test_client")
            image_pub = client.create_publisher(Image, "/camera/rgb/image_raw", 10)
            instruction_pub = client.create_publisher(String, "/vln/instruction", 10)
            episode_id_pub = client.create_publisher(String, "/episode/id", 10)
            state_pub = client.create_publisher(String, "/episode/state", 10)
            finish_pub = client.create_publisher(Bool, "/navigation/trajectory_finished", 10)
            commands = []
            model_stops = []
            latencies = []
            client.create_subscription(NavigationCommand, "/vln/command", commands.append, 10)
            client.create_subscription(String, "/episode/model_stop", model_stops.append, 10)
            client.create_subscription(Float32MultiArray, "/vln/inference_latency", latencies.append, 10)
            executor = MultiThreadedExecutor(num_threads=2)
            executor.add_node(adapter)
            executor.add_node(client)
            thread = threading.Thread(target=executor.spin, daemon=True)
            thread.start()
            time.sleep(0.1)
            episode_id_pub.publish(String(data="stub_episode"))
            state_pub.publish(String(data="START"))
            # Exercise the DDS delivery order where START arrives before the
            # latched instruction.  The first instruction must not invalidate
            # the generation initialized by START.
            instruction_pub.publish(String(data="Walk forward."))
            time.sleep(0.15)
            image = Image()
            image.height = 2
            image.width = 2
            image.encoding = "rgb8"
            image.step = 6
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
            assert len(trajectory.points) == 5
            assert latencies and len(latencies[-1].data) == 1
            assert math.isclose(latencies[-1].data[0], 0.01, rel_tol=1.0e-5)
            assert all(
                math.isfinite(value)
                for point in trajectory.points
                for value in (point.x, point.y, point.theta)
            )
            finish_pub.publish(Bool(data=True))
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline and not model_stops:
                time.sleep(0.01)
            assert model_stops
            assert commands[-1].command_type == NavigationCommand.STOP
            assert sum(url.endswith("/reset") for url in requests_seen) == 1
            assert any(url.endswith("/step") for url in requests_seen)
        finally:
            if executor is not None:
                executor.shutdown(timeout_sec=1.0)
            if thread is not None:
                thread.join(timeout=1.0)
            if adapter is not None:
                adapter.destroy_node()
            if client is not None:
                client.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
