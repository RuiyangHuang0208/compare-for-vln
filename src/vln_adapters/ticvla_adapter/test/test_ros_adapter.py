#!/usr/bin/env python3
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import time
import unittest

from nav_msgs.msg import Odometry
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image
from std_msgs.msg import Header, String
from vln_interfaces.msg import NavigationCommand

from ticvla_adapter.ticvla_node import TicVlaAdapter


class Handler(BaseHTTPRequestHandler):
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
        self.reply({"status": "ready", "implementation": "ticvla.models.ticvla.TICVLA", "horizon": 30})

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.reply(
            {
                "waypoints": [[0.1 * (index + 1), 0.0] for index in range(30)],
                "total_latency": 0.2,
                "vlm_latency": 0.18,
                "action_latency": 0.02,
            }
        )


class TicVlaRosAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 15802), Handler)
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
        self.adapter = TicVlaAdapter(
            parameter_overrides=[
                Parameter("model.server_url", value="http://127.0.0.1:15802"),
                Parameter("input.history_length", value=2),
                Parameter("input.history_interval_frames", value=1),
                Parameter("input.image_width", value=6),
                Parameter("input.image_height", value=4),
                Parameter("runtime.inference_rate_hz", value=20.0),
            ]
        )
        self.client = Node("ticvla_adapter_test_client")
        self.instruction_pub = self.client.create_publisher(String, "/vln/instruction", 10)
        self.episode_pub = self.client.create_publisher(String, "/episode/state", 10)
        self.image_pub = self.client.create_publisher(Image, "/camera/rgb/image_raw", 10)
        self.odom_pub = self.client.create_publisher(Odometry, "/odom", 10)
        self.commands = []
        self.client.create_subscription(NavigationCommand, "/vln/command", self.commands.append, 10)
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.adapter)
        self.executor.add_node(self.client)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        time.sleep(0.3)

    def tearDown(self):
        self.executor.shutdown(timeout_sec=1.0)
        self.thread.join(timeout=1.0)
        self.adapter.close()
        self.adapter.destroy_node()
        self.client.destroy_node()

    def publish_odom(self, x=0.0):
        message = Odometry()
        message.pose.pose.position.x = x
        message.pose.pose.orientation.w = 1.0
        message.twist.twist.linear.x = 0.2
        self.odom_pub.publish(message)

    def publish_image(self):
        stamp = self.client.get_clock().now().to_msg()
        image = Image(
            header=Header(stamp=stamp, frame_id="camera"),
            height=4,
            width=6,
            encoding="rgb8",
            step=18,
        )
        image.data = np.zeros((4, 6, 3), dtype=np.uint8).tobytes()
        self.image_pub.publish(image)

    def test_30x2_output_and_reset_contract(self):
        self.instruction_pub.publish(String(data="Go to the chair"))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and self.adapter.instruction != "Go to the chair":
            time.sleep(0.01)
        self.assertEqual(self.adapter.instruction, "Go to the chair")
        self.publish_odom(0.0)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and self.adapter.pose is None:
            time.sleep(0.01)
        self.assertIsNotNone(self.adapter.pose)
        self.publish_image()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(self.adapter.history) < 1:
            time.sleep(0.01)
        self.assertEqual(len(self.adapter.history), 1)
        self.publish_odom(0.01)
        self.publish_image()
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and len(self.adapter.history) < 2:
            time.sleep(0.01)
        self.assertGreaterEqual(len(self.adapter.history), 2)
        deadline = time.monotonic() + 2.0
        trajectory = None
        while time.monotonic() < deadline:
            trajectory = next(
                (command for command in reversed(self.commands) if command.command_type == NavigationCommand.TRAJECTORY),
                None,
            )
            if trajectory is not None:
                break
            time.sleep(0.01)
        future_error = None
        if self.adapter.future is not None and self.adapter.future.done():
            future_error = repr(self.adapter.future.exception())
        self.assertIsNotNone(
            trajectory,
            f"future={self.adapter.future!r} future_error={future_error} snapshot={self.adapter.snapshot()!r}",
        )
        self.assertEqual(len(trajectory.points), 30)
        self.assertAlmostEqual(trajectory.points[-1].x, 3.0, places=4)
        self.assertEqual(trajectory.header.frame_id, "base_link")

        self.episode_pub.publish(String(data="RESET"))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and (
            not self.commands or self.commands[-1].command_type != NavigationCommand.STOP
        ):
            time.sleep(0.01)
        self.assertEqual(self.commands[-1].command_type, NavigationCommand.STOP)

    def test_native_mode_publishes_official_velocity_command(self):
        self.tearDown()
        self.adapter = TicVlaAdapter(
            parameter_overrides=[
                Parameter("model.server_url", value="http://127.0.0.1:15802"),
                Parameter("input.history_length", value=2),
                Parameter("input.history_interval_frames", value=1),
                Parameter("input.image_width", value=6),
                Parameter("input.image_height", value=4),
                Parameter("runtime.inference_rate_hz", value=20.0),
                Parameter("evaluation.mode", value="native_output"),
                Parameter("evaluation.direct_velocity", value=True),
            ]
        )
        self.client = Node("ticvla_native_adapter_test_client")
        self.instruction_pub = self.client.create_publisher(String, "/vln/instruction", 10)
        self.episode_pub = self.client.create_publisher(String, "/episode/state", 10)
        self.image_pub = self.client.create_publisher(Image, "/camera/rgb/image_raw", 10)
        self.odom_pub = self.client.create_publisher(Odometry, "/odom", 10)
        self.commands = []
        self.client.create_subscription(NavigationCommand, "/vln/command", self.commands.append, 10)
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.adapter)
        self.executor.add_node(self.client)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        time.sleep(0.3)

        self.instruction_pub.publish(String(data="Go straight"))
        self.publish_odom()
        time.sleep(0.1)
        self.publish_image()
        time.sleep(0.05)
        self.publish_image()
        deadline = time.monotonic() + 2.0
        velocity = None
        while time.monotonic() < deadline:
            velocity = next(
                (command for command in reversed(self.commands) if command.command_type == NavigationCommand.VELOCITY),
                None,
            )
            if velocity is not None:
                break
            time.sleep(0.01)
        self.assertIsNotNone(velocity)
        self.assertAlmostEqual(velocity.velocity.linear.x, 1.0, places=4)
        self.assertAlmostEqual(velocity.velocity.linear.y, 0.0, places=4)
        self.assertAlmostEqual(velocity.velocity.angular.z, 0.0, places=4)



if __name__ == "__main__":
    unittest.main()
