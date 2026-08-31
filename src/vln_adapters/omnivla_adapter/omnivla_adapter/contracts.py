from __future__ import annotations

from pathlib import Path


ACTION_SHAPE = (8, 4)
LANGUAGE_ONLY_MODALITY_ID = 7
CHECKPOINT_VARIANT = "omnivla-original"
RESUME_STEP = 120000
REQUIRED_CHECKPOINT_FILES = (
    "action_head--120000_checkpoint.pt",
    "proprio_projector--120000_checkpoint.pt",
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
    "model.safetensors.index.json",
    "config.json",
    "preprocessor_config.json",
)


def validate_language_only_contract(goal_profile: str, modality_id: int) -> None:
    if goal_profile != "language_only":
        raise ValueError("OmniVLA fair evaluation requires goal_profile=language_only")
    if int(modality_id) != LANGUAGE_ONLY_MODALITY_ID:
        raise ValueError("OmniVLA language-only inference requires modality_id=7")


def response_is_current(response, episode_id: str, generation: int, request_id: str) -> bool:
    try:
        return (
            str(response["episode_id"]) == str(episode_id)
            and int(response["generation"]) == int(generation)
            and str(response["request_id"]) == str(request_id)
            and int(response["modality_id"]) == LANGUAGE_ONLY_MODALITY_ID
        )
    except (KeyError, TypeError, ValueError):
        return False


def checkpoint_status(checkpoint_path) -> dict[str, object]:
    root = Path(checkpoint_path).expanduser().resolve()
    missing = [name for name in REQUIRED_CHECKPOINT_FILES if not (root / name).is_file()]
    lora = root / "lora_adapter"
    return {
        "path": str(root),
        "complete": not missing,
        "missing": missing,
        "lora_adapter_present": lora.is_dir(),
        "dist_head_present": (root / "dist_head--120000_checkpoint.pt").is_file(),
    }

