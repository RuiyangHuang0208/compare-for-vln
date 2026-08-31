from __future__ import annotations

from pathlib import Path
import hashlib
import math

import yaml


VALID_OUTPUT_TYPES = {"trajectory", "waypoint", "velocity", "stop"}
VALID_SENSOR_PROFILES = {"rgb_only", "rgb_d", "rgb_d_pointcloud_from_depth", "rgb_d_lidar"}
VALID_EVALUATION_MODES = {"trajectory_normalized", "native_output"}
EXECUTION_NAVIGATION_FIELDS = {
    "lookahead_distance",
    "goal_tolerance",
    "yaw_gain",
    "yaw_filter_alpha",
    "curvature_feedforward_gain",
    "max_vx",
    "max_vy",
    "max_wz",
    "max_linear_acceleration",
    "max_angular_acceleration",
    "max_linear_deceleration",
    "max_angular_deceleration",
    "trajectory_timeout",
    "recovery_speed",
    "recovery_duration",
}
VALID_HIGH_LEVEL_CONTROLLERS = {
    "shared_pure_pursuit",
    "discrete_action_path",
    "adapter_native_velocity",
}
SRU_ONNX_COMMAND_LIMITS = {"max_vx": 1.0, "max_vy": 1.0, "max_wz": 1.0}


def _validate_execution(name, raw):
    if raw in (None, {}):
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Model {name!r} execution must be a mapping")
    execution = dict(raw)
    for field in (
        "launch_model",
        "source",
        "evidence",
        "controller_fidelity",
        "upstream_commit",
        "source_sha256",
        "adapter_deviation",
        "camera_parameter_source",
    ):
        if not isinstance(execution.get(field), str) or not execution[field]:
            raise ValueError(f"Model {name!r} execution requires a non-empty {field}")
    if len(execution["upstream_commit"]) != 40:
        raise ValueError(f"Model {name!r} execution upstream_commit must be a full 40-character commit")
    digest = execution["source_sha256"].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"Model {name!r} execution source_sha256 must be a SHA256 hex digest")
    execution["source_sha256"] = digest
    if not isinstance(execution.get("official_controller_available"), bool):
        raise ValueError(f"Model {name!r} execution official_controller_available must be boolean")
    controller = execution.get("high_level_controller")
    if controller not in VALID_HIGH_LEVEL_CONTROLLERS:
        raise ValueError(
            f"Model {name!r} execution high_level_controller must be one of "
            f"{sorted(VALID_HIGH_LEVEL_CONTROLLERS)}"
        )
    for field in ("camera_horizontal_fov_degrees", "desired_speed"):
        value = float(execution.get(field, math.nan))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Model {name!r} execution requires positive finite {field}")
        execution[field] = value
    if not 1.0 < execution["camera_horizontal_fov_degrees"] < 179.0:
        raise ValueError(f"Model {name!r} execution camera HFOV must be between 1 and 179 degrees")
    resolution = execution.get("simulation_camera_resolution")
    if (
        not isinstance(resolution, list)
        or len(resolution) != 2
        or any(not isinstance(value, int) or value <= 0 for value in resolution)
    ):
        raise ValueError(
            f"Model {name!r} execution simulation_camera_resolution must be [width, height]"
        )
    official = execution.get("official")
    if not isinstance(official, dict) or not official:
        raise ValueError(f"Model {name!r} execution official evidence must be a non-empty mapping")
    navigation = execution.get("navigation")
    if not isinstance(navigation, dict) or set(navigation) != EXECUTION_NAVIGATION_FIELDS:
        raise ValueError(
            f"Model {name!r} execution navigation fields must equal "
            f"{sorted(EXECUTION_NAVIGATION_FIELDS)}"
        )
    execution["navigation"] = {key: float(value) for key, value in navigation.items()}
    if not all(math.isfinite(value) for value in execution["navigation"].values()):
        raise ValueError(f"Model {name!r} execution navigation values must be finite")
    non_negative = EXECUTION_NAVIGATION_FIELDS - {"recovery_speed"}
    if any(execution["navigation"][key] < 0.0 for key in non_negative):
        raise ValueError(f"Model {name!r} execution navigation limits must be non-negative")
    if execution["navigation"]["recovery_speed"] >= 0.0:
        raise ValueError(f"Model {name!r} execution recovery_speed must be negative")
    if execution["desired_speed"] > execution["navigation"]["max_vx"]:
        raise ValueError(f"Model {name!r} execution desired_speed exceeds max_vx")
    for field, policy_limit in SRU_ONNX_COMMAND_LIMITS.items():
        if execution["navigation"][field] > policy_limit:
            raise ValueError(
                f"Model {name!r} execution {field} exceeds the verified SRU-ONNX "
                f"command range ({policy_limit:g})"
            )
    return execution


def validate_official_source(workspace_root, model_name, execution):
    """Verify that the audited upstream source file has not drifted."""
    source = (Path(workspace_root).resolve() / execution["source"]).resolve()
    root = Path(workspace_root).resolve()
    try:
        source.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Model {model_name!r} official source escapes the workspace") from error
    if not source.is_file():
        raise FileNotFoundError(f"Model {model_name!r} official source is missing: {source}")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != execution["source_sha256"]:
        raise ValueError(
            f"Model {model_name!r} official source SHA256 drifted: expected "
            f"{execution['source_sha256']}, got {digest}"
        )
    repository = source
    while repository != root and not (repository / ".git").exists():
        repository = repository.parent
    if repository == root:
        raise ValueError(f"Model {model_name!r} official source is not inside a git repository")
    head_file = repository / ".git"
    # Submodules use a .git text pointer. Resolve HEAD without invoking a shell.
    if head_file.is_file():
        pointer = head_file.read_text(encoding="utf-8").strip()
        if not pointer.startswith("gitdir: "):
            raise ValueError(f"Model {model_name!r} has an invalid submodule .git pointer")
        git_dir = (repository / pointer[8:]).resolve()
    else:
        git_dir = head_file
    head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head[5:]
        ref_file = git_dir / ref
        if ref_file.is_file():
            commit = ref_file.read_text(encoding="utf-8").strip()
        else:
            commit = None
            packed = git_dir / "packed-refs"
            if packed.is_file():
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line and not line.startswith(("#", "^")):
                        value, name = line.split(" ", 1)
                        if name == ref:
                            commit = value
                            break
            if commit is None:
                raise ValueError(f"Model {model_name!r} cannot resolve official repository HEAD")
    else:
        commit = head
    if commit != execution["upstream_commit"]:
        raise ValueError(
            f"Model {model_name!r} official repository commit drifted: expected "
            f"{execution['upstream_commit']}, got {commit}"
        )
    return {"source": str(source), "sha256": digest, "commit": commit}


def load_model_registry(path):
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    models = document.get("models") if isinstance(document, dict) else None
    if not isinstance(models, dict) or not models:
        raise ValueError(f"Model registry {path} must contain a non-empty models mapping")
    validated = {}
    for name, raw in models.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise ValueError(f"Invalid model entry {name!r}")
        entry = dict(raw)
        for required in ("package", "launch_file", "output_type"):
            if not isinstance(entry.get(required), str) or not entry[required]:
                raise ValueError(f"Model {name!r} requires a non-empty {required}")
        entry.setdefault("executable", entry["package"])
        entry.setdefault("implemented", True)
        entry.setdefault("direct_velocity", entry["output_type"] == "velocity")
        entry.setdefault("evaluation_mode", "native_output" if entry["direct_velocity"] else "trajectory_normalized")
        entry.setdefault("sensor_profile", "rgb_only")
        entry.setdefault("sensors", [])
        entry.setdefault("derived_inputs", [])
        entry.setdefault("comparison_track", "unclassified")
        entry.setdefault("model_inputs", list(entry["sensors"]))
        entry.setdefault("launch_arguments", {})
        entry.setdefault("runtime_metadata", {})
        entry.setdefault("execution", {})
        if entry["output_type"] not in VALID_OUTPUT_TYPES:
            raise ValueError(f"Model {name!r} has unsupported output_type={entry['output_type']!r}")
        if entry["evaluation_mode"] not in VALID_EVALUATION_MODES:
            raise ValueError(f"Model {name!r} has unsupported evaluation_mode={entry['evaluation_mode']!r}")
        if entry["sensor_profile"] not in VALID_SENSOR_PROFILES:
            raise ValueError(f"Model {name!r} has unsupported sensor_profile={entry['sensor_profile']!r}")
        if not isinstance(entry["sensors"], list) or not all(isinstance(item, str) for item in entry["sensors"]):
            raise ValueError(f"Model {name!r} sensors must be a string list")
        if not isinstance(entry["derived_inputs"], list) or not all(
            isinstance(item, str) for item in entry["derived_inputs"]
        ):
            raise ValueError(f"Model {name!r} derived_inputs must be a string list")
        if not isinstance(entry["comparison_track"], str) or not entry["comparison_track"]:
            raise ValueError(f"Model {name!r} comparison_track must be a non-empty string")
        if not isinstance(entry["model_inputs"], list) or not all(
            isinstance(item, str) and item for item in entry["model_inputs"]
        ):
            raise ValueError(f"Model {name!r} model_inputs must be a non-empty string list")
        if not isinstance(entry["launch_arguments"], dict):
            raise ValueError(f"Model {name!r} launch_arguments must be a mapping")
        if not isinstance(entry["runtime_metadata"], dict):
            raise ValueError(f"Model {name!r} runtime_metadata must be a mapping")
        entry["execution"] = _validate_execution(name, entry["execution"])
        validated[name] = entry
    return validated


def select_model(registry, name):
    if name not in registry:
        raise ValueError(f"Unknown model {name!r}; available={sorted(registry)}")
    entry = registry[name]
    if not entry["implemented"]:
        raise ValueError(f"Model {name!r} is registered but not implemented")
    return entry


def load_comparison_manifest(path):
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    if not isinstance(document, dict) or not isinstance(document.get("shared"), dict):
        raise ValueError(f"Fair comparison manifest {path} requires a shared mapping")
    if not isinstance(document.get("tracks"), dict) or not document["tracks"]:
        raise ValueError(f"Fair comparison manifest {path} requires a non-empty tracks mapping")
    return document


def validate_comparison_run(manifest, model_name, model_entry, requested_track, desired_speed, evaluation_mode):
    if requested_track == "none":
        return "untracked"
    track_name = model_entry["comparison_track"] if requested_track == "auto" else requested_track
    tracks = manifest["tracks"]
    if track_name not in tracks:
        raise ValueError(f"Unknown comparison track {track_name!r}; available={sorted(tracks)}")
    track = tracks[track_name]
    models = track.get("models", [])
    if model_name not in models:
        raise ValueError(f"Model {model_name!r} is not registered in comparison track {track_name!r}")
    if track.get("sensor_profile") != model_entry["sensor_profile"]:
        raise ValueError(
            f"Comparison track {track_name!r} requires sensor_profile={track.get('sensor_profile')!r}, "
            f"but model {model_name!r} uses {model_entry['sensor_profile']!r}"
        )
    shared = manifest["shared"]
    expected_mode = str(track.get("evaluation_mode", shared.get("evaluation_mode", "")))
    if evaluation_mode != expected_mode or model_entry["evaluation_mode"] != expected_mode:
        raise ValueError(f"Fair comparison requires evaluation_mode={expected_mode!r}")
    expected_speed = float(shared.get("desired_speed", math.nan))
    speed = float(desired_speed)
    if not math.isfinite(expected_speed) or not math.isclose(speed, expected_speed, abs_tol=1.0e-9):
        raise ValueError(
            f"Fair comparison requires desired_speed={expected_speed:g} m/s; "
            "use comparison_track:=none for manual speed experiments"
        )
    if shared.get("locomotion") != "sru-onnx" or shared.get("local_avoidance") is not False:
        raise ValueError("Fair comparison manifest must require sru-onnx and local_avoidance=false")
    return track_name
