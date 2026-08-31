from types import SimpleNamespace

import pytest

from vln_evaluation.benchmark_runner import (
    episode_command,
    parse_args,
    resolve_execution,
    select_episodes,
    validate_episode,
)
from vln_evaluation.model_registry import load_model_registry


def test_all_requires_exactly_85_official_episodes():
    episodes = {
        f"episode_{index:03d}": {"suite": "dynanav_full"}
        for index in range(85)
    }
    episodes["local_smoke"] = {}
    selected = select_episodes(episodes, "all")
    assert len(selected) == 85
    assert "local_smoke" not in selected


def test_all_rejects_incomplete_official_suite():
    with pytest.raises(RuntimeError, match="Expected 85"):
        select_episodes({"only_one": {"suite": "dynanav_full"}}, "all")


def test_all_can_exclude_outdoor_without_changing_official_suite_validation():
    episodes = {
        f"episode_{index:03d}": {
            "suite": "dynanav_full",
            "scene": "outdoor" if index < 10 else "hospital",
        }
        for index in range(85)
    }
    selected = select_episodes(episodes, "all", ["outdoor"])
    assert len(selected) == 75
    assert all(episodes[key]["scene"] != "outdoor" for key in selected)


def test_exclude_scenes_rejects_unknown_scene():
    with pytest.raises(ValueError, match="Unsupported excluded scenes"):
        select_episodes({"episode": {"scene": "hospital"}}, "episode", ["mars"])


def test_episode_validation_and_command(tmp_path):
    asset_dir = tmp_path / "third_party" / "TIC-VLA" / "DynaNav" / "assets"
    asset_dir.mkdir(parents=True)
    (asset_dir / "office.usd").touch()
    config = {"scene": "office", "spawn": [1.0, 2.0, 0.0], "goal": [3.0, 4.0]}
    validate_episode("office_001", config, tmp_path)
    args = SimpleNamespace(model="dummy", experiment="dry", headless=True)
    command = episode_command(args, tmp_path, "office_001")
    assert "episode:=office_001" in command
    assert "headless:=true" in command
    assert "shutdown_after_finish:=true" in command


def test_resumable_benchmark_timeout_defaults_cover_slow_simulation():
    args = parse_args(["--model", "dummy"])
    assert args.timeout_scale == 5.0
    assert args.startup_grace == 120.0
    assert args.max_attempts == 1
    assert args.resume is False
    assert args.continue_on_error is False
    assert args.exclude_scenes == ""


def test_model_specific_execution_is_loaded_from_models_yaml():
    workspace = __import__("pathlib").Path(__file__).resolve().parents[3]
    registry = load_model_registry(workspace / "configs" / "models.yaml")
    args = parse_args(["--model", "omnivla", "--execution-profile", "model_specific"])
    resolve_execution(args, registry)
    assert args.launch_model == "omnivla_native"
    assert args.evaluation_mode == "native_output"
    assert args.desired_speed == 0.3
    assert args.camera_hfov == 90.0
    assert '"limits.max_vx":0.3' in args.navigation_overrides_json
    command = episode_command(args, workspace, "hospital_001")
    assert "model:=omnivla_native" in command
    assert "result_model_name:=omnivla" in command
    assert "comparison_track:=none" in command
    assert "execution_profile:=native" in command


def test_native_execution_selects_model_high_level_controller():
    workspace = __import__("pathlib").Path(__file__).resolve().parents[3]
    registry = load_model_registry(workspace / "configs" / "models.yaml")
    expected = {
        "dualvln": "shared_pure_pursuit",
        "ticvla": None,
        "navila": "discrete_action_path",
        "uninavid": "discrete_action_path",
    }
    for model, controller in expected.items():
        args = parse_args(["--model", model, "--execution-profile", "native"])
        resolve_execution(args, registry)
        if controller is None:
            assert args.launch_model == "ticvla_native"
            assert args.evaluation_mode == "native_output"
            assert "path_follower.controller" not in args.navigation_overrides_json
        else:
            assert f'"path_follower.controller":"{controller}"' in args.navigation_overrides_json
