from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image
import requests


class MobileVLAR1InferenceClient:
    def __init__(self, server_url: str, timeout: float):
        self.server_url = server_url.rstrip("/")
        self.timeout = float(timeout)

    def health(self):
        response = requests.get(self.server_url + "/health", timeout=min(self.timeout, 2.0))
        response.raise_for_status()
        return response.json()

    def reset(self, episode_id: str, generation: int):
        response = requests.post(
            self.server_url + "/reset",
            json={"episode_id": episode_id, "generation": int(generation)},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def step(self, rgb_history, depth, pointcloud, metadata):
        archive = io.BytesIO()
        np.savez_compressed(
            archive,
            rgb=np.asarray(rgb_history, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            pointcloud=np.asarray(pointcloud, dtype=np.float32),
        )
        preview = io.BytesIO()
        Image.fromarray(np.asarray(rgb_history[-1], dtype=np.uint8), mode="RGB").save(preview, "JPEG")
        response = requests.post(
            self.server_url + "/step",
            files={
                "bundle": ("bundle.npz", archive.getvalue(), "application/octet-stream"),
                "preview": ("current.jpg", preview.getvalue(), "image/jpeg"),
            },
            data={"json": json.dumps(metadata, separators=(",", ":"))},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

