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
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header, String
from vln_interfaces.msg import NavigationCommand

from dualvln_adapter.dualvln_node import DualVlnAdapter


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def do_GET(self):
        payload = json.dumps({"status": "ready"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        payload = json.dumps({"trajectory": [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]], "inference_s": 0.01}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class DualVlnRosAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 15801), Handler)
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
        self.adapter = DualVlnAdapter(
            parameter_overrides=[
                Parameter("model.server_url", value="http://127.0.0.1:15801"),
                Parameter("runtime.inference_rate_hz", value=20.0),
                Parameter("sensor_timeout", value=2.0),
            ]
        )
        self.client = Node("dualvln_adapter_test_client")
        self.instruction_pub = self.client.create_publisher(String, "/vln/instruction", 10)
        self.episode_pub = self.client.create_publisher(String, "/episode/state", 10)
        self.rgb_pub = self.client.create_publisher(Image, "/camera/rgb/image_raw", 10)
        self.depth_pub = self.client.create_publisher(Image, "/camera/depth/image_raw", 10)
        self.info_pub = self.client.create_publisher(CameraInfo, "/camera/rgb/camera_info", 10)
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

    def publish_inputs(self):
        stamp = self.client.get_clock().now().to_msg()
        header = Header(stamp=stamp, frame_id="camera")
        rgb_array = np.zeros((4, 6, 3), dtype=np.uint8)
        rgb = Image(header=header, height=4, width=6, encoding="rgb8", step=18)
        rgb.data = rgb_array.tobytes()
        depth_array = np.ones((4, 6), dtype=np.float32)
        depth = Image(
            header=header,
            height=4,
            width=6,
            encoding="32FC1",
            step=24,
        )
        depth.data = depth_array.tobytes()
        info = CameraInfo(header=header, height=4, width=6)
        info.k = [4.0, 0.0, 3.0, 0.0, 4.0, 2.0, 0.0, 0.0, 1.0]
        odom = Odometry()
        odom.pose.pose.orientation.w = 1.0
        self.rgb_pub.publish(rgb)
        self.depth_pub.publish(depth)
        self.info_pub.publish(info)
        self.odom_pub.publish(odom)

    def test_http_trajectory_becomes_navigation_command_and_reset_stops(self):
        self.instruction_pub.publish(String(data="Walk to the door"))
        time.sleep(0.05)
        self.publish_inputs()
        deadline = time.monotonic() + 2.0
        trajectory = None
        while time.monotonic() < deadline:
            trajectory = next(
                (item for item in reversed(self.commands) if item.command_type == NavigationCommand.TRAJECTORY),
                None,
            )
            if trajectory is not None:
                break
            time.sleep(0.01)
        self.assertIsNotNone(trajectory)
        self.assertEqual(trajectory.header.frame_id, "base_link")
        self.assertEqual(len(trajectory.points), 3)
        self.assertAlmostEqual(trajectory.points[-1].x, 2.0)

        self.episode_pub.publish(String(data="RESET"))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and (
            not self.commands or self.commands[-1].command_type != NavigationCommand.STOP
        ):
            time.sleep(0.01)
        self.assertEqual(self.commands[-1].command_type, NavigationCommand.STOP)


if __name__ == "__main__":
    unittest.main()
