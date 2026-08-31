from __future__ import annotations

import io
import json
from pathlib import Path
import queue
import socket
import struct
import threading

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
import numpy as np
from PIL import Image as PilImage
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import TransformBroadcaster


def receive_exact(connection, length):
    output = bytearray()
    while len(output) < length:
        block = connection.recv(length - len(output))
        if not block:
            raise ConnectionError("sensor stream disconnected")
        output.extend(block)
    return bytes(output)


def receive_blob(connection):
    length = struct.unpack("!I", receive_exact(connection, 4))[0]
    if length > 32 * 1024 * 1024:
        raise ValueError(f"sensor payload too large: {length}")
    return receive_exact(connection, length)


class SensorBridge(Node):
    def __init__(self, **kwargs):
        super().__init__("dynanav_sensor_bridge", **kwargs)
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 5822)
        self.declare_parameter("rgb_topic", "/camera/rgb/image_raw")
        self.declare_parameter("depth_topic", "/camera/depth/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/rgb/camera_info")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("camera_frame", "camera_front_optical")
        self.declare_parameter("debug_first_rgb_path", "")
        self.debug_frame_saved = False
        self.frames = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(
            (str(self.get_parameter("host").value), int(self.get_parameter("port").value))
        )
        self.server_socket.listen(1)
        self.server_socket.settimeout(0.5)
        self.thread = threading.Thread(target=self.receive_loop, name="isaac-rgbd", daemon=True)
        self.thread.start()
        self.rgb_pub = self.create_publisher(Image, str(self.get_parameter("rgb_topic").value), 5)
        self.depth_pub = self.create_publisher(Image, str(self.get_parameter("depth_topic").value), 5)
        self.info_pub = self.create_publisher(CameraInfo, str(self.get_parameter("camera_info_topic").value), 5)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value), self.on_odom, 20)
        self.create_timer(0.01, self.publish_latest)
        self.get_logger().info(f"Waiting for Isaac RGB-D TCP on port {self.get_parameter('port').value}")

    def receive_loop(self):
        while not self.closed.is_set():
            try:
                connection, _ = self.server_socket.accept()
            except socket.timeout:
                continue
            with connection:
                while not self.closed.is_set():
                    try:
                        header = json.loads(receive_blob(connection).decode("utf-8"))
                        rgb = np.asarray(PilImage.open(io.BytesIO(receive_blob(connection))).convert("RGB"))
                        depth_u16 = np.asarray(PilImage.open(io.BytesIO(receive_blob(connection))), dtype=np.uint16)
                        frame = (header, np.ascontiguousarray(rgb), depth_u16.astype(np.float32) / 10000.0)
                        try:
                            self.frames.put_nowait(frame)
                        except queue.Full:
                            self.frames.get_nowait()
                            self.frames.task_done()
                            self.frames.put_nowait(frame)
                    except (ConnectionError, OSError, ValueError, json.JSONDecodeError):
                        break

    def publish_latest(self):
        latest = None
        while True:
            try:
                latest = self.frames.get_nowait()
                self.frames.task_done()
            except queue.Empty:
                break
        if latest is None:
            return
        metadata, rgb, depth = latest
        height, width = rgb.shape[:2]
        if depth.shape != (height, width):
            self.get_logger().warning(f"Rejected unaligned RGB {rgb.shape} and depth {depth.shape}")
            return
        debug_path = str(self.get_parameter("debug_first_rgb_path").value).strip()
        if debug_path and not self.debug_frame_saved:
            destination = Path(debug_path).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            PilImage.fromarray(rgb, mode="RGB").save(destination)
            self.debug_frame_saved = True
            self.get_logger().info(f"Saved first RGB frame: {destination}")
        sim_time = max(0.0, float(metadata.get("sim_time", 0.0)))
        stamp = self.get_clock().now().to_msg()
        stamp.sec = int(sim_time)
        stamp.nanosec = int((sim_time - int(sim_time)) * 1.0e9)
        frame_id = str(self.get_parameter("camera_frame").value)
        rgb_message = Image()
        rgb_message.header.stamp = stamp
        rgb_message.header.frame_id = frame_id
        rgb_message.height, rgb_message.width = height, width
        rgb_message.encoding = "rgb8"
        rgb_message.step = width * 3
        rgb_message.data = rgb.tobytes()
        depth_message = Image()
        depth_message.header = rgb_message.header
        depth_message.height, depth_message.width = height, width
        depth_message.encoding = "32FC1"
        depth_message.step = width * 4
        depth_message.data = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
        info = CameraInfo()
        info.header = rgb_message.header
        info.height, info.width = height, width
        info.k = np.asarray(metadata["intrinsics"], dtype=np.float64).reshape(-1).tolist()
        info.p = [info.k[0], 0.0, info.k[2], 0.0, 0.0, info.k[4], info.k[5], 0.0, 0.0, 0.0, 1.0, 0.0]
        self.rgb_pub.publish(rgb_message)
        self.depth_pub.publish(depth_message)
        self.info_pub.publish(info)

    def on_odom(self, message):
        transform = TransformStamped()
        transform.header = message.header
        transform.child_frame_id = message.child_frame_id
        transform.transform.translation.x = message.pose.pose.position.x
        transform.transform.translation.y = message.pose.pose.position.y
        transform.transform.translation.z = message.pose.pose.position.z
        transform.transform.rotation = message.pose.pose.orientation
        self.tf_broadcaster.sendTransform(transform)

    def close(self):
        self.closed.set()
        self.server_socket.close()
        self.thread.join(timeout=1.0)


def main(args=None):
    rclpy.init(args=args)
    node = SensorBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.close()
        except KeyboardInterrupt:
            pass
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
