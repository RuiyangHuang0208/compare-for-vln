#!/usr/bin/env python3
from __future__ import annotations

import json
import socket
import threading
import time
import unittest

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String

from robot_controller.udp_velocity_bridge import UdpVelocityBridge


class UdpBridgeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        if rclpy.ok():
            rclpy.shutdown()

    def setUp(self):
        self.command_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_receiver.bind(("127.0.0.1", 15820))
        self.command_receiver.settimeout(1.0)
        self.telemetry_sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.path_receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.path_receiver.bind(("127.0.0.1", 15823))
        self.path_receiver.settimeout(1.0)
        overrides = [
            Parameter("command_port", value=15820),
            Parameter("telemetry_port", value=15821),
            Parameter("debug_path_port", value=15823),
            Parameter("command_timeout", value=0.2),
            Parameter("publish_rate_hz", value=50.0),
        ]
        self.bridge = UdpVelocityBridge(parameter_overrides=overrides)
        self.client = Node("udp_bridge_test_client")
        self.vln_pub = self.client.create_publisher(Twist, "/nav_vel", 10)
        self.keyboard_pub = self.client.create_publisher(Twist, "/keyboard/nav_vel", 10)
        self.source_pub = self.client.create_publisher(String, "/robot_controller/source", 10)
        self.path_pub = self.client.create_publisher(Path, "/vln/debug_path", 10)
        self.odometry = []
        self.client.create_subscription(Odometry, "/odom", self.odometry.append, 10)
        self.executor = MultiThreadedExecutor(num_threads=2)
        self.executor.add_node(self.bridge)
        self.executor.add_node(self.client)
        self.thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.thread.start()
        time.sleep(0.3)
        self.drain()

    def tearDown(self):
        self.executor.shutdown(timeout_sec=1.0)
        self.thread.join(timeout=1.0)
        self.bridge.close()
        self.bridge.destroy_node()
        self.client.destroy_node()
        self.command_receiver.close()
        self.telemetry_sender.close()
        self.path_receiver.close()

    def drain(self):
        self.command_receiver.setblocking(False)
        try:
            while True:
                self.command_receiver.recvfrom(4096)
        except BlockingIOError:
            pass
        self.command_receiver.settimeout(1.0)

    def receive_matching(self, predicate, timeout=1.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            payload, _ = self.command_receiver.recvfrom(4096)
            decoded = json.loads(payload)
            if predicate(decoded):
                return decoded
        self.fail("No matching UDP velocity received")

    def test_source_exclusion_timeout_and_telemetry(self):
        keyboard = Twist()
        keyboard.linear.x = 0.9
        self.keyboard_pub.publish(keyboard)
        time.sleep(0.08)
        self.assertEqual(self.receive_matching(lambda p: p["vx"] == 0.0)["source"], "vln")

        vln = Twist()
        vln.linear.x = 0.3
        vln.angular.z = -0.2
        self.vln_pub.publish(vln)
        command = self.receive_matching(lambda p: p["vx"] == 0.3)
        self.assertEqual(command["source"], "vln")
        self.assertEqual(command["wz"], -0.2)

        path = Path()
        path.header.frame_id = "world"
        for x, y in ((1.0, 2.0), (3.0, 4.0)):
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            path.poses.append(pose)
        self.path_pub.publish(path)
        payload, _ = self.path_receiver.recvfrom(65535)
        debug_path = json.loads(payload)
        self.assertEqual(debug_path["frame_id"], "world")
        self.assertEqual(debug_path["points"], [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]])

        self.receive_matching(lambda p: p["vx"] == 0.0, timeout=0.7)
        self.source_pub.publish(String(data="keyboard"))
        self.receive_matching(lambda p: p["source"] == "vln" and p["vx"] == 0.0)
        self.keyboard_pub.publish(keyboard)
        switched = self.receive_matching(lambda p: p["source"] == "keyboard" and p["vx"] == 0.9)
        self.assertEqual(switched["vy"], 0.0)

        telemetry = {
            "sim_time": 12.25,
            "position": [1.0, 2.0, 0.7],
            "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
            "linear_velocity_b": [0.3, 0.0, 0.0],
            "angular_velocity_b": [0.0, 0.0, 0.1],
        }
        self.telemetry_sender.sendto(json.dumps(telemetry).encode(), ("127.0.0.1", 15821))
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not self.odometry:
            time.sleep(0.01)
        self.assertTrue(self.odometry)
        self.assertEqual(self.odometry[-1].pose.pose.position.x, 1.0)
        self.assertEqual(self.odometry[-1].twist.twist.angular.z, 0.1)
        self.assertEqual(self.odometry[-1].header.stamp.sec, 12)
        self.assertEqual(self.odometry[-1].header.stamp.nanosec, 250000000)

        local_path = Path()
        local_path.header.frame_id = "base_link"
        for x, y in ((1.0, 0.0), (0.0, 2.0)):
            pose = PoseStamped()
            pose.pose.position.x = x
            pose.pose.position.y = y
            local_path.poses.append(pose)
        self.source_pub.publish(String(data="vln"))
        time.sleep(0.05)
        self.path_pub.publish(local_path)
        payload, _ = self.path_receiver.recvfrom(65535)
        debug_path = json.loads(payload)
        while not debug_path["points"]:
            payload, _ = self.path_receiver.recvfrom(65535)
            debug_path = json.loads(payload)
        self.assertEqual(debug_path["frame_id"], "world")
        self.assertEqual(debug_path["points"], [[2.0, 2.0, 0.7], [1.0, 4.0, 0.7]])


if __name__ == "__main__":
    unittest.main()
