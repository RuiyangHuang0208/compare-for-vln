from pathlib import Path

import numpy as np
import pytest

from omnivla_adapter.contracts import (
    LANGUAGE_ONLY_MODALITY_ID,
    checkpoint_status,
    validate_language_only_contract,
)
from omnivla_adapter.official_inference import StubOmniVLA


def test_language_only_is_modality_seven():
    validate_language_only_contract("language_only", LANGUAGE_ONLY_MODALITY_ID)
    with pytest.raises(ValueError):
        validate_language_only_contract("image_goal", LANGUAGE_ONLY_MODALITY_ID)
    with pytest.raises(ValueError):
        validate_language_only_contract("language_only", 6)


def test_stub_has_no_placeholder_leakage():
    runtime = StubOmniVLA()
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    first, _ = runtime.infer(image, "Go to the chair")
    changed = np.full_like(image, 255)
    second, _ = runtime.infer(changed, "Go to the chair")
    np.testing.assert_array_equal(first, second)
    assert runtime.metadata["modality_id"] == 7


def test_missing_checkpoint_is_explicit(tmp_path):
    status = checkpoint_status(tmp_path / "omnivla-original")
    assert status["complete"] is False
    assert "action_head--120000_checkpoint.pt" in status["missing"]


def test_ros_node_has_no_privileged_sensor_or_model_imports():
    source = (Path(__file__).parents[1] / "omnivla_adapter" / "omnivla_node.py").read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "Depth" not in source
    assert "LaserScan" not in source
    assert "PointCloud" not in source
    assert '"/odom"' not in source
    assert '"/episode/goal"' not in source
    assert '"/nav_vel"' not in source

