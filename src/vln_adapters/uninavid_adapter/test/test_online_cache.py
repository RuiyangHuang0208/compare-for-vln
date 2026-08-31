import io
import json

from PIL import Image
import pytest

pytest.importorskip("flask")
from uninavid_adapter.inference_server import create_app


class FakeRuntime:
    def __init__(self):
        self.reset_count = 0
        self.frame_batches = []

    def reset_cache(self):
        self.reset_count += 1

    def infer(self, frames, instruction):
        self.frame_batches.append((len(frames), instruction))
        return "forward left stop"


def image_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (30, 60, 90)).save(buffer, format="PNG")
    return buffer.getvalue()


def step_payload(episode_id="episode_a", generation=2):
    metadata = {
        "episode_id": episode_id,
        "generation": generation,
        "request_id": "request-1",
        "instruction": "Walk to the chair.",
    }
    return {
        "json": json.dumps(metadata),
        "image_0": (io.BytesIO(image_bytes()), "frame.png"),
    }


def test_reset_initializes_cache_and_step_only_processes_sent_new_frames():
    runtime = FakeRuntime()
    client = create_app(runtime).test_client()
    response = client.post("/reset", json={"episode_id": "episode_a", "generation": 2})
    assert response.status_code == 200
    assert runtime.reset_count == 1
    response = client.post("/step", data=step_payload(), content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.json["raw_action"] == "forward left stop"
    assert response.json["new_frames"] == 1
    assert runtime.frame_batches == [(1, "Walk to the chair.")]


def test_old_episode_or_generation_is_rejected_before_inference():
    runtime = FakeRuntime()
    client = create_app(runtime).test_client()
    client.post("/reset", json={"episode_id": "episode_a", "generation": 2})
    response = client.post(
        "/step", data=step_payload(episode_id="episode_old", generation=1), content_type="multipart/form-data"
    )
    assert response.status_code == 409
    assert response.json["episode_id"] == "episode_a"
    assert response.json["generation"] == 2
    assert runtime.frame_batches == []
