from pathlib import Path

import pytest

from vln_evaluation.model_registry import (
    load_comparison_manifest,
    load_model_registry,
    select_model,
    validate_official_source,
    validate_comparison_run,
)


def write_registry(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def test_registry_loads_implemented_model(tmp_path):
    path = write_registry(
        tmp_path / "models.yaml",
        """models:\n  model_a:\n    package: model_a_adapter\n    launch_file: model_a.launch.py\n    output_type: trajectory\n""",
    )
    entry = select_model(load_model_registry(path), "model_a")
    assert entry["executable"] == "model_a_adapter"
    assert entry["evaluation_mode"] == "trajectory_normalized"
    assert entry["sensor_profile"] == "rgb_only"
    assert entry["comparison_track"] == "unclassified"


def test_registry_rejects_unknown_and_unimplemented(tmp_path):
    path = write_registry(
        tmp_path / "models.yaml",
        """models:\n  future:\n    package: future_adapter\n    launch_file: future.launch.py\n    output_type: waypoint\n    implemented: false\n""",
    )
    registry = load_model_registry(path)
    with pytest.raises(ValueError, match="not implemented"):
        select_model(registry, "future")
    with pytest.raises(ValueError, match="Unknown model"):
        select_model(registry, "missing")


def test_workspace_registry_contains_navila():
    workspace = Path(__file__).resolve().parents[3]
    navila = select_model(load_model_registry(workspace / "configs" / "models.yaml"), "navila")
    assert navila["output_type"] == "trajectory"
    assert navila["sensors"] == ["rgb"]
    assert navila["direct_velocity"] is False
    assert navila["comparison_track"] == "rgb_only"
    assert navila["model_inputs"] == ["rgb", "instruction"]
    assert navila["execution"]["desired_speed"] == 0.5
    assert navila["execution"]["official"]["forward_step_m"] == 0.25


def test_workspace_model_execution_parameters_are_embedded_and_validated():
    workspace = Path(__file__).resolve().parents[3]
    registry = load_model_registry(workspace / "configs" / "models.yaml")
    expected = {
        "ticvla": (1.0, 1.0, 1.0, 90.0),
        "omnivla": (0.3, 0.3, 0.3, 90.0),
        "dualvln": (0.3, 0.4, 0.4, 79.0),
        "navila": (0.5, 0.5, 0.6, 90.0),
        "uninavid": (0.5, 0.5, 1.0, 90.0),
    }
    for name, (speed, max_vx, max_wz, hfov) in expected.items():
        execution = registry[name]["execution"]
        assert execution["desired_speed"] == speed
        assert execution["navigation"]["max_vx"] == max_vx
        assert execution["navigation"]["max_wz"] == max_wz
        assert execution["camera_horizontal_fov_degrees"] == hfov
        assert execution["simulation_camera_resolution"] == [640, 480]
        assert execution["high_level_controller"]
        assert execution["controller_fidelity"]
        assert len(execution["upstream_commit"]) == 40
        assert len(execution["source_sha256"]) == 64
        validate_official_source(workspace, name, execution)


def test_registry_rejects_native_execution_outside_sru_onnx_command_range(tmp_path):
    path = write_registry(
        tmp_path / "models.yaml",
        """models:
  too_fast:
    package: example_adapter
    launch_file: example.launch.py
    output_type: trajectory
    execution:
      launch_model: too_fast
      high_level_controller: shared_pure_pursuit
      controller_fidelity: test
      source: upstream.py
      upstream_commit: "0000000000000000000000000000000000000000"
      source_sha256: "0000000000000000000000000000000000000000000000000000000000000000"
      evidence: test
      official_controller_available: true
      adapter_deviation: test
      camera_parameter_source: test
      camera_horizontal_fov_degrees: 90
      simulation_camera_resolution: [640, 480]
      desired_speed: 1.2
      official: {controller: test}
      navigation:
        lookahead_distance: 1.0
        goal_tolerance: 0.05
        yaw_gain: 0.8
        yaw_filter_alpha: 0.35
        curvature_feedforward_gain: 0.5
        max_vx: 1.2
        max_vy: 0.0
        max_wz: 1.0
        max_linear_acceleration: 0.8
        max_angular_acceleration: 1.5
        max_linear_deceleration: 0.8
        max_angular_deceleration: 1.5
        trajectory_timeout: 3.0
        recovery_speed: -0.3
        recovery_duration: 1.5
""",
    )
    with pytest.raises(ValueError, match="SRU-ONNX command range"):
        load_model_registry(path)


def test_workspace_registry_contains_full_language_only_omnivla():
    workspace = Path(__file__).resolve().parents[3]
    omnivla = select_model(load_model_registry(workspace / "configs" / "models.yaml"), "omnivla")
    assert omnivla["repository"] == "third_party/OmniVLA"
    assert omnivla["checkpoint_variant"] == "omnivla-original"
    assert omnivla["resume_step"] == 120000
    assert omnivla["raw_output_type"] == "local_trajectory_xy_heading"
    assert omnivla["action_shape"] == [8, 4]
    assert omnivla["sensors"] == ["rgb"]
    assert omnivla["model_inputs"] == ["rgb", "instruction"]
    assert omnivla["goal_profile"] == "language_only"
    assert omnivla["direct_velocity"] is False
    assert omnivla["local_avoidance"] is False

    manifest = load_comparison_manifest(workspace / "configs" / "fair_comparison.yaml")
    assert validate_comparison_run(
        manifest, "omnivla", omnivla, "auto", 1.0, "trajectory_normalized"
    ) == "rgb_only"


def test_workspace_fair_comparison_tracks_enforce_speed_and_sensor_group():
    workspace = Path(__file__).resolve().parents[3]
    registry = load_model_registry(workspace / "configs/models.yaml")
    manifest = load_comparison_manifest(workspace / "configs/fair_comparison.yaml")
    navila = select_model(registry, "navila")
    assert validate_comparison_run(
        manifest, "navila", navila, "auto", 1.0, "trajectory_normalized"
    ) == "rgb_only"
    assert manifest["tracks"]["rgb_only"]["camera_horizontal_fov_degrees"] == 90.0
    assert manifest["tracks"]["rgb_d"]["camera_horizontal_fov_degrees"] == 79.0
    with pytest.raises(ValueError, match="desired_speed=1"):
        validate_comparison_run(manifest, "navila", navila, "auto", 2.0, "trajectory_normalized")
    with pytest.raises(ValueError, match="not registered"):
        validate_comparison_run(manifest, "navila", navila, "rgb_d", 1.0, "trajectory_normalized")
    assert validate_comparison_run(
        manifest, "navila", navila, "none", 2.0, "trajectory_normalized"
    ) == "untracked"
