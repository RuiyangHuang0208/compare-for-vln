#!/usr/bin/env python3
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from vln_interfaces.msg import NavigationCommand

from navila_adapter.navila_node import NavilaAdapter


class Handler(BaseHTTPRequestHandler):
    raw_action = "The next action is move forward 50 cm"
    release = None

    def log_message(self, *_args):
        pass

    def reply(self, value):
        payload = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.reply({"status": "ready", "num_video_frames": 8})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        marker = b'name="json"\r\n\r\n'
        metadata = json.loads(body.split(marker, 1)[1].split(b"\r\n", 1)[0])
        if Handler.release is not None:
            Handler.release.wait(timeout=2.0)
        self.reply(
            {
                "request_id": metadata["request_id"],
                "episode_id": metadata["episode_id"],
                "raw_action": Handler.raw_action,
                "inference_s": 0.01,
            }
        )


class NavilaRosAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 15803), Handler)
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        Handler.raw_action = "The next action is move forward 50 cm"
        Handler.release = None
        self.adapter = NavilaAdapter(
            parameter_overrides=[
                Parameter("model.server_url", value="http://127.0.0.1:15803"),
                Parameter("conversion.turn_radius", value=1.0),
                Parameter("runtime.inference_rate_hz", value=20.0),
                Parameter("runtime.inference_timeout", value=2.0),
            ]
        )
        self.client = Node("navila_adapter_test_client")
        self.instruction_pub = self.client.create_publisher(String, "/vln/instruction", 10)
        self.episode_pub = self.client.create_publisher(String, "/episode/state", 10)
        self.episode_id_pub = self.client.create_publisher(String, "/episode/id", 10)
        self.image_pub = self.client.create_publisher(Image, "/camera/rgb/image_raw", 10)
        self.commands = []
        self.raw_actions = []
        self.client.create_subscription(NavigationCommand, "/vln/command", self.commands.append, 10)
        self.client.create_subscription(String, "/vln/navila/raw_action", self.raw_actions.append, 10)
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.adapter)
        self.executor.add_node(self.client)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        time.sleep(0.2)

    def tearDown(self):
        Handler.release = None
        self.executor.shutdown(timeout_sec=1.0)
        self.thread.join(timeout=1.0)
        self.adapter.close()
        self.adapter.destroy_node()
        self.client.destroy_node()

    def publish_image(self):
        image = Image(
            header=Header(stamp=self.client.get_clock().now().to_msg(), frame_id="camera_front_optical"),
            height=4,
            width=6,
            encoding="rgb8",
            step=18,
        )
        image.data = np.zeros((4, 6, 3), dtype=np.uint8).tobytes()
        self.image_pub.publish(image)

    def wait_for_trajectory(self, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            match = next(
                (item for item in reversed(self.commands) if item.command_type == NavigationCommand.TRAJECTORY),
                None,
            )
            if match is not None:
                return match
            time.sleep(0.01)
        return None

    def test_rgb_text_action_becomes_trajectory_without_depth_subscription(self):
        self.episode_id_pub.publish(String(data="test_episode"))
        self.instruction_pub.publish(String(data="Walk to the chair"))
        self.episode_pub.publish(String(data="START"))
        self.publish_image()
        trajectory = self.wait_for_trajectory()
        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory.header.frame_id, "base_link")
        self.assertAlmostEqual(trajectory.points[-1].x, 0.5)
        self.assertEqual(self.raw_actions[-1].data, Handler.raw_action)
        self.assertEqual(self.adapter.get_subscriptions_info_by_topic("/camera/depth/image_raw"), [])

    def test_reset_discards_late_response(self):
        Handler.release = threading.Event()
        self.episode_id_pub.publish(String(data="old_episode"))
        self.instruction_pub.publish(String(data="Walk to the chair"))
        self.episode_pub.publish(String(data="START"))
        self.publish_image()
        time.sleep(0.3)
        self.episode_pub.publish(String(data="RESET"))
        time.sleep(0.1)
        trajectory_count = sum(item.command_type == NavigationCommand.TRAJECTORY for item in self.commands)
        Handler.release.set()
        time.sleep(0.4)
        self.assertEqual(
            sum(item.command_type == NavigationCommand.TRAJECTORY for item in self.commands),
            trajectory_count,
        )
        self.assertEqual(self.commands[-1].command_type, NavigationCommand.STOP)

    def test_inference_timeout_stops_and_discards_response(self):
        self.adapter.inference_timeout = 0.2
        Handler.release = threading.Event()
        self.episode_id_pub.publish(String(data="timeout_episode"))
        self.instruction_pub.publish(String(data="Walk to the chair"))
        self.episode_pub.publish(String(data="START"))
        time.sleep(0.05)
        self.publish_image()
        time.sleep(0.6)
        self.assertEqual(self.commands[-1].command_type, NavigationCommand.STOP)
        trajectory_count = sum(item.command_type == NavigationCommand.TRAJECTORY for item in self.commands)
        Handler.release.set()
        time.sleep(0.3)
        self.assertEqual(
            sum(item.command_type == NavigationCommand.TRAJECTORY for item in self.commands),
            trajectory_count,
        )

    def test_new_instruction_after_terminal_state_resumes_manual_demo(self):
        self.episode_id_pub.publish(String(data="manual_demo"))
        self.instruction_pub.publish(String(data="Walk to the chair"))
        self.episode_pub.publish(String(data="START"))
        self.publish_image()
        self.assertIsNotNone(self.wait_for_trajectory())

        self.episode_pub.publish(String(data="FAILED"))
        time.sleep(0.2)
        trajectory_count = sum(item.command_type == NavigationCommand.TRAJECTORY for item in self.commands)
        self.instruction_pub.publish(String(data="Turn toward the door"))
        self.publish_image()
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            current_count = sum(
                item.command_type == NavigationCommand.TRAJECTORY for item in self.commands
            )
            if current_count > trajectory_count:
                break
            time.sleep(0.01)
        self.assertGreater(current_count, trajectory_count)
        self.assertTrue(self.adapter.episode_active)


if __name__ == "__main__":
    unittest.main()
