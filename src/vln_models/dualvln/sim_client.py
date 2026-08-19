"""Non-blocking HTTP client used by the Isaac Sim process."""

from __future__ import annotations

import io
import json
import queue
import threading
import time

import numpy as np
import requests
from PIL import Image


class AsyncDualVlnClient:
    """Keep at most one pending RGB-D frame so inference never blocks physics."""

    def __init__(self, server_url, request_timeout=180.0):
        server_url = server_url.rstrip("/")
        health = requests.get(server_url + "/health", timeout=5.0)
        health.raise_for_status()
        if health.json().get("status") != "ready":
            raise RuntimeError(f"DualVLN server is not ready: {health.text}")
        self.endpoint = server_url + "/step"
        self.request_timeout = float(request_timeout)
        self._requests = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._result = None
        self._error = None
        self._closed = threading.Event()
        self._session = requests.Session()
        self._worker = threading.Thread(target=self._run, name="dualvln-http", daemon=True)
        self._worker.start()

    @property
    def busy(self):
        return not self._requests.empty()

    def submit(self, rgb, depth_m, metadata):
        request = (np.asarray(rgb, dtype=np.uint8).copy(), np.asarray(depth_m, dtype=np.float32).copy(), dict(metadata))
        try:
            self._requests.put_nowait(request)
            return True
        except queue.Full:
            try:
                self._requests.get_nowait()
                self._requests.task_done()
                self._requests.put_nowait(request)
                return True
            except (queue.Empty, queue.Full):
                return False

    def take_result(self):
        with self._lock:
            result, self._result = self._result, None
            error, self._error = self._error, None
        return result, error

    def close(self):
        self._closed.set()
        self._session.close()

    def _run(self):
        while not self._closed.is_set():
            try:
                rgb, depth_m, metadata = self._requests.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                rgb_buffer = io.BytesIO()
                Image.fromarray(rgb, mode="RGB").save(rgb_buffer, format="JPEG", quality=90)
                depth_buffer = io.BytesIO()
                depth_u16 = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0)
                depth_u16 = np.clip(depth_u16 * 10000.0, 0, 65535).astype(np.uint16)
                Image.fromarray(depth_u16).save(depth_buffer, format="PNG")
                response = self._session.post(
                    self.endpoint,
                    files={
                        "image": ("front.jpg", rgb_buffer.getvalue(), "image/jpeg"),
                        "depth": ("front_depth.png", depth_buffer.getvalue(), "image/png"),
                    },
                    data={"json": json.dumps(metadata)},
                    timeout=self.request_timeout,
                )
                response.raise_for_status()
                payload = response.json()
                payload["capture_pose"] = metadata["capture_pose"]
                payload["frame_id"] = metadata["frame_id"]
                payload["instruction_id"] = metadata["instruction_id"]
                payload["received_monotonic"] = time.monotonic()
                with self._lock:
                    self._result = payload
                    self._error = None
            except Exception as exc:
                with self._lock:
                    self._error = {
                        "message": f"{type(exc).__name__}: {exc}",
                        "frame_id": metadata.get("frame_id", -1),
                        "instruction_id": metadata.get("instruction_id", -1),
                    }
            finally:
                self._requests.task_done()
