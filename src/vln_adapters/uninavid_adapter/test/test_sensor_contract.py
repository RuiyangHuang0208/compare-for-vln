from pathlib import Path


def test_adapter_has_no_privileged_sensor_subscription():
    source = (Path(__file__).parents[1] / "uninavid_adapter" / "uninavid_node.py").read_text()
    lowered = source.lower()
    for forbidden in ("depth_topic", "lidar", "pointcloud", "costmap", "goal_topic", "odom_topic"):
        assert forbidden not in lowered
    assert '"/nav_vel"' not in source


def test_config_requires_confirmed_sampling_and_turn_radius():
    config = (Path(__file__).parents[1] / "config" / "uninavid.yaml").read_text()
    assert "input.frame_sample_hz: 1.0" in config
    assert "conversion.turn_radius: 0.25" in config


def test_runtime_loader_relabels_released_llama_checkpoint_as_llava():
    source = (Path(__file__).parents[1] / "uninavid_adapter" / "inference_server.py").read_text()
    assert 'config["model_type"] = "llava"' in source
    assert 'config["architectures"] = ["LlavaLlamaAttForCausalLM"]' in source
