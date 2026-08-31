import json

from mobilevla_r1_adapter.contracts import checkpoint_status


def create_vila_layout(root, *, depth=True, point=True):
    for relative in ("llm", "vision_tower", "mm_projector"):
        (root / relative).mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(
        json.dumps(
            {
                "_name_or_path": "test-mobilevla-r1",
                "num_video_frames": 8,
                "use_depth_tower": depth,
                "use_point_tower": point,
            }
        ),
        encoding="utf-8",
    )
    (root / "llm/config.json").write_text("{}", encoding="utf-8")
    (root / "llm/tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "llm/model-00001-of-00001.safetensors").write_bytes(b"weight")
    (root / "llm/model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.depth_tower.layer.weight": "model-00001-of-00001.safetensors",
                    "model.point_tower.layer.weight": "model-00001-of-00001.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "vision_tower/model.safetensors").write_bytes(b"vision")
    (root / "mm_projector/model.safetensors").write_bytes(b"projector")


def test_complete_multimodal_checkpoint_is_accepted(tmp_path):
    create_vila_layout(tmp_path)
    status = checkpoint_status(tmp_path)
    assert status["complete"] is True
    assert status["history_frames"] == 8
    assert status["missing_modalities"] == []


def test_rgb_only_navila_checkpoint_is_rejected(tmp_path):
    create_vila_layout(tmp_path, depth=False, point=False)
    index = tmp_path / "llm/model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"model.layers.0.weight": "model-00001-of-00001.safetensors"}}))
    status = checkpoint_status(tmp_path)
    assert status["complete"] is False
    assert status["source_name_or_path"] == "test-mobilevla-r1"
    assert status["missing_modalities"] == [
        "config.use_depth_tower=true",
        "config.use_point_tower=true",
        "depth tower/encoder weights",
        "point tower/encoder weights",
    ]
