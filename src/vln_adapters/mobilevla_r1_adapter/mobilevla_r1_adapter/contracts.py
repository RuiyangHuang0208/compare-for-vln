from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


EXPECTED_VECTOR_LENGTH = 12
HISTORY_RGB_FRAMES = 8
DEPTH_FRAMES = 1
POINTCLOUD_POINTS = 2048
VELOCITY_INDICES = (0, 1, 2)
OFFICIAL_FIELDS = (
    "x_vel_cmd",
    "y_vel_cmd",
    "yaw_vel_cmd",
    "body_height_cmd",
    "step_frequency_cmd",
    "gait1",
    "gait2",
    "gait3",
    "footswing_height_cmd",
    "pitch_cmd",
    "roll_cmd",
    "stance_width_cmd",
)
UNSUPPORTED_BEHAVIORS = ("jump", "dance", "hello", "stretch", "sit", "lie down")


@dataclass(frozen=True)
class RequestIdentity:
    episode_id: str
    generation: int
    request_id: str


def response_is_current(response: dict, identity: RequestIdentity) -> bool:
    try:
        return (
            str(response["episode_id"]) == identity.episode_id
            and int(response["generation"]) == identity.generation
            and str(response["request_id"]) == identity.request_id
        )
    except (KeyError, TypeError, ValueError):
        return False


def validate_runtime_contract(
    *,
    history_frames: int,
    depth_frames: int,
    pointcloud_points: int,
    expected_vector_length: int,
    command_duration_s: float,
    allow_stub: bool,
) -> None:
    expected = {
        "history_frames": (history_frames, HISTORY_RGB_FRAMES),
        "depth_frames": (depth_frames, DEPTH_FRAMES),
        "pointcloud_points": (pointcloud_points, POINTCLOUD_POINTS),
        "expected_vector_length": (expected_vector_length, EXPECTED_VECTOR_LENGTH),
    }
    invalid = [f"{name}={actual} (expected {required})" for name, (actual, required) in expected.items() if actual != required]
    if invalid:
        raise ValueError("MobileVLA-R1 official contract mismatch: " + ", ".join(invalid))
    if command_duration_s <= 0.0 and not allow_stub:
        raise ValueError(
            "MobileVLA-R1 command_duration_s is not established by the official repository; "
            "real benchmark is disabled until a verified execution window is configured"
        )


def checkpoint_status(checkpoint_path) -> dict[str, object]:
    root = Path(checkpoint_path).expanduser().resolve()
    required = (
        "config.json",
        "llm/config.json",
        "llm/tokenizer_config.json",
        "llm/model.safetensors.index.json",
        "vision_tower/model.safetensors",
        "mm_projector/model.safetensors",
    )
    missing = [name for name in required if not (root / name).is_file()]
    config = {}
    config_error = None
    if (root / "config.json").is_file():
        try:
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            config_error = str(error)
    index_keys = []
    index_path = root / "llm/model.safetensors.index.json"
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            index_keys = list(index.get("weight_map", {}))
        except (OSError, json.JSONDecodeError) as error:
            config_error = config_error or str(error)
    names_and_keys = [str(path.relative_to(root)).lower() for path in root.rglob("*") if path.is_file()]
    names_and_keys.extend(str(key).lower() for key in index_keys)
    depth_assets = any("depth_tower" in value or "depth_encoder" in value for value in names_and_keys)
    point_assets = any("point_tower" in value or "point_encoder" in value for value in names_and_keys)
    depth_enabled = config.get("use_depth_tower") is True
    point_enabled = config.get("use_point_tower") is True
    weights = sorted((root / "llm").glob("*.safetensors")) + sorted((root / "llm").glob("pytorch_model*.bin"))
    missing_modalities = []
    if not depth_enabled:
        missing_modalities.append("config.use_depth_tower=true")
    if not point_enabled:
        missing_modalities.append("config.use_point_tower=true")
    if not depth_assets:
        missing_modalities.append("depth tower/encoder weights")
    if not point_assets:
        missing_modalities.append("point tower/encoder weights")
    return {
        "path": str(root),
        "complete": root.is_dir() and not missing and bool(weights) and not missing_modalities and config_error is None,
        "missing": missing,
        "missing_modalities": missing_modalities,
        "config_error": config_error,
        "source_name_or_path": config.get("_name_or_path"),
        "history_frames": config.get("num_video_frames"),
        "depth_enabled": depth_enabled,
        "point_enabled": point_enabled,
        "depth_assets": depth_assets,
        "point_assets": point_assets,
        "weight_files": [item.name for item in weights],
        "non_lora_trainables": (root / "non_lora_trainables.bin").is_file(),
        "adapter_config": (root / "adapter_config.json").is_file(),
    }
