from pathlib import Path

import yaml


def test_omnivla_registry_is_full_language_only_model():
    workspace = Path(__file__).resolve().parents[4]
    registry = yaml.safe_load((workspace / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]
    entry = registry["omnivla"]
    assert entry["repository"] == "third_party/OmniVLA"
    assert entry["checkpoint_variant"] == "omnivla-original"
    assert entry["action_shape"] == [8, 4]
    assert entry["sensors"] == ["rgb"]
    assert entry["goal_profile"] == "language_only"
    assert entry["direct_velocity"] is False
    assert entry["local_avoidance"] is False


def test_omnivla_native_registry_is_separate_official_track():
    workspace = Path(__file__).resolve().parents[4]
    registry = yaml.safe_load((workspace / "configs" / "models.yaml").read_text(encoding="utf-8"))["models"]
    entry = registry["omnivla_native"]
    assert entry["output_type"] == "velocity"
    assert entry["evaluation_mode"] == "native_output"
    assert entry["direct_velocity"] is True
    assert entry["comparison_track"] == "omnivla_native"
    assert entry["model_inputs"] == ["rgb", "instruction"]
