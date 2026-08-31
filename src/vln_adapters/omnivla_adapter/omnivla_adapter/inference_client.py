from __future__ import annotations

import io
import json

from PIL import Image
import requests


class OmniVLAInferenceClient:
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

    def step(self, image, metadata):
        buffer = io.BytesIO()
        Image.fromarray(image, mode="RGB").save(buffer, format="JPEG", quality=95)
        response = requests.post(
            self.server_url + "/step",
            files={"image": ("current.jpg", buffer.getvalue(), "image/jpeg")},
            data={"json": json.dumps(metadata, separators=(",", ":"))},
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

