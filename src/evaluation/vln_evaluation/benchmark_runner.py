from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

try:
    import psutil
except ImportError:  # pragma: no cover - optional cleanup enhancement
    psutil = None

from ament_index_python.packages import get_package_share_directory
import yaml

from .model_registry import load_model_registry, select_model, validate_official_source


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run DynaNav-compatible episodes as isolated ROS/Isaac subprocesses.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--experiment", default="benchmark")
    parser.add_argument("--episodes", default="hospital_001", help="Comma-separated episode IDs; use 'all' explicitly.")
    parser.add_argument(
        "--exclude-scenes",
        default="",
        help="Comma-separated scene names to exclude after selecting episodes (for example: outdoor).",
    )
    parser.add_argument("--episodes-file", default="")
    parser.add_argument("--workspace-root", default=os.environ.get("ROBOT_VLN_WS", os.getcwd()))
    parser.add_argument("--models-file", default="")
    parser.add_argument(
        "--execution-profile",
        default="fair",
        choices=("fair", "native", "model_specific"),
        help="Use the shared fair controller or the model-native high-level controller. "
        "model_specific is a backward-compatible alias for native.",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate episode selection, scene assets, and launch commands without starting Isaac Sim.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip episodes whose result JSON already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Attempt the remaining episodes after an infrastructure failure.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=1,
        help="Maximum infrastructure attempts per episode.",
    )
    parser.add_argument(
        "--timeout-scale",
        type=float,
        default=5.0,
        help="Wall-clock multiplier applied to the episode simulation-time limit (slow GPU simulation fallback).",
    )
    parser.add_argument(
        "--startup-grace",
        type=float,
        default=120.0,
        help="Additional wall-clock seconds allowed for Isaac Sim startup and shutdown.",
    )
    return parser.parse_args(argv)


def select_episodes(episodes, requested, exclude_scenes=()):
    excluded = {str(scene).strip().lower() for scene in exclude_scenes if str(scene).strip()}
    supported_scenes = {"hospital", "office", "outdoor", "warehouse"}
    unknown_scenes = excluded - supported_scenes
    if unknown_scenes:
        raise ValueError(f"Unsupported excluded scenes: {sorted(unknown_scenes)}")
    selected = (
        [key for key, value in episodes.items() if value.get("suite") == "dynanav_full"]
        if requested == "all"
        else [item.strip() for item in requested.split(",") if item.strip()]
    )
    if requested == "all" and len(selected) != 85:
        raise RuntimeError(f"Expected 85 official DynaNav episodes, found {len(selected)}")
    unknown = sorted(set(selected) - set(episodes))
    if unknown:
        raise KeyError(f"Unknown episodes: {unknown}")
    if excluded:
        selected = [key for key in selected if str(episodes[key].get("scene", "")).lower() not in excluded]
    if not selected:
        suffix = f" after excluding scenes {sorted(excluded)}" if excluded else ""
        raise ValueError(f"No episodes selected{suffix}")
    return selected


def validate_episode(episode_id, config, workspace):
    scene = str(config.get("scene", ""))
    if scene not in {"hospital", "office", "outdoor", "warehouse"}:
        raise ValueError(f"Episode {episode_id} has unsupported scene {scene!r}")
    for field, size in (("spawn", 3), ("goal", 2)):
        value = config.get(field)
        if not isinstance(value, list) or len(value) < size:
            raise ValueError(f"Episode {episode_id} must define {field} with at least {size} values")
    if scene in {"office", "outdoor"}:
        filename = "office.usd" if scene == "office" else os.environ.get(
            "DYNANAV_OUTDOOR_ASSET", "outdoor_small.usd"
        )
        if Path(filename).name != filename:
            raise ValueError("DYNANAV_OUTDOOR_ASSET must be a file name")
        asset = workspace / "third_party" / "TIC-VLA" / "DynaNav" / "assets" / filename
        if not asset.is_file():
            raise FileNotFoundError(f"Episode {episode_id} is missing scene asset: {asset}")


def episode_command(args, workspace, episode_id):
    execution_profile = getattr(args, "execution_profile", "fair")
    if execution_profile == "model_specific":
        execution_profile = "native"
    launch_model = getattr(args, "launch_model", args.model)
    desired_speed = getattr(args, "desired_speed", 1.0)
    navigation_overrides_json = getattr(args, "navigation_overrides_json", "{}")
    execution_metadata_json = getattr(args, "execution_metadata_json", "{}")
    command = [
        "ros2",
        "launch",
        "robot_bringup",
        "dynanav_single_episode.launch.py",
        f"model:={launch_model}",
        f"result_model_name:={args.model}",
        f"episode:={episode_id}",
        f"experiment:={args.experiment}",
        f"workspace_root:={workspace}",
        f"headless:={'true' if args.headless else 'false'}",
        f"evaluation_mode:={getattr(args, 'evaluation_mode', 'trajectory_normalized')}",
        f"sensor_profile:={getattr(args, 'sensor_profile', 'auto')}",
        f"desired_speed:={desired_speed}",
        f"comparison_track:={'none' if execution_profile == 'native' else 'auto'}",
        f"execution_profile:={execution_profile}",
        f"navigation_overrides_json:={navigation_overrides_json}",
        f"execution_metadata_json:={execution_metadata_json}",
        "shutdown_after_finish:=true",
    ]
    camera_hfov = getattr(args, "camera_hfov", None)
    if camera_hfov is not None:
        command.append(f"camera_hfov_override:={camera_hfov}")
    return command


def resolve_execution(args, registry):
    """Resolve all runtime parameters from the selected entry in models.yaml."""
    requested_entry = select_model(registry, args.model)
    if args.execution_profile == "fair":
        args.launch_model = args.model
        args.evaluation_mode = requested_entry["evaluation_mode"]
        args.sensor_profile = requested_entry["sensor_profile"]
        args.desired_speed = 1.0
        args.camera_hfov = None
        args.navigation_overrides_json = "{}"
        args.execution_metadata_json = "{}"
        return requested_entry

    execution = requested_entry.get("execution", {})
    if not execution:
        raise ValueError(f"Model {args.model!r} has no execution mapping in configs/models.yaml")
    validate_official_source(Path(args.workspace_root), args.model, execution)
    args.launch_model = str(execution["launch_model"])
    launch_entry = select_model(registry, args.launch_model)
    args.evaluation_mode = launch_entry["evaluation_mode"]
    args.sensor_profile = launch_entry["sensor_profile"]
    args.desired_speed = float(execution["desired_speed"])
    args.camera_hfov = float(execution["camera_horizontal_fov_degrees"])
    parameter_names = {
        "lookahead_distance": "path_follower.lookahead_distance",
        "goal_tolerance": "path_follower.goal_tolerance",
        "yaw_gain": "path_follower.yaw_gain",
        "yaw_filter_alpha": "path_follower.yaw_filter_alpha",
        "curvature_feedforward_gain": "path_follower.curvature_feedforward_gain",
        "max_vx": "limits.max_vx",
        "max_vy": "limits.max_vy",
        "max_wz": "limits.max_wz",
        "max_linear_acceleration": "limits.max_linear_acceleration",
        "max_angular_acceleration": "limits.max_angular_acceleration",
        "max_linear_deceleration": "limits.max_linear_deceleration",
        "max_angular_deceleration": "limits.max_angular_deceleration",
        "trajectory_timeout": "timeout.trajectory_timeout",
        "recovery_speed": "stuck.recovery_speed",
        "recovery_duration": "stuck.recovery_duration",
    }
    navigation = {
        parameter_names[name]: value for name, value in execution["navigation"].items()
    }
    controller = str(execution["high_level_controller"])
    if controller != "adapter_native_velocity":
        navigation["path_follower.controller"] = controller
    args.navigation_overrides_json = json.dumps(navigation, separators=(",", ":"))
    metadata = dict(execution)
    metadata["requested_model"] = args.model
    metadata["resolved_launch_model"] = args.launch_model
    metadata["actual_camera_resolution"] = [640, 480]
    metadata["execution_profile"] = "native"
    args.execution_metadata_json = json.dumps(metadata, separators=(",", ":"))
    return launch_entry


def stop_process(process):
    descendants = []
    if psutil is not None:
        try:
            descendants = psutil.Process(process.pid).children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            descendants = []

    def signal_tree(signum):
        # Signal children first so Isaac Sim cannot outlive the ROS launch process.
        for child in reversed(descendants):
            try:
                child.send_signal(signum)
            except Exception:
                pass
        try:
            os.killpg(process.pid, signum)
        except ProcessLookupError:
            pass

    try:
        signal_tree(signal.SIGINT)
    except OSError:
        pass
    try:
        process.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        signal_tree(signal.SIGTERM)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            signal_tree(signal.SIGKILL)
            process.wait(timeout=5.0)
    if psutil is not None:
        for child in descendants:
            try:
                child.wait(timeout=2.0)
            except (psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied):
                pass


def main():
    args = parse_args()
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be at least 1")
    if args.timeout_scale <= 0.0 or args.startup_grace < 0.0:
        raise ValueError("Timeout scale must be positive and startup grace must be non-negative")
    workspace = Path(args.workspace_root).resolve()
    models_file = Path(args.models_file).resolve() if args.models_file else workspace / "configs" / "models.yaml"
    registry = load_model_registry(models_file)
    resolve_execution(args, registry)
    if not math.isfinite(args.desired_speed) or args.desired_speed <= 0.0:
        raise ValueError("Resolved desired speed must be positive and finite")
    default_episodes = Path(get_package_share_directory("dynanav_bridge")) / "config" / "episodes.yaml"
    episodes_file = Path(args.episodes_file).resolve() if args.episodes_file else default_episodes
    with open(episodes_file, encoding="utf-8") as stream:
        episodes = yaml.safe_load(stream).get("episodes", {})
    excluded_scenes = [item for item in args.exclude_scenes.split(",") if item.strip()]
    selected = select_episodes(episodes, args.episodes, excluded_scenes)
    scene_counts = {}
    for episode_id in selected:
        config = episodes[episode_id]
        validate_episode(episode_id, config, workspace)
        scene = str(config["scene"])
        scene_counts[scene] = scene_counts.get(scene, 0) + 1
    print(
        f"[BENCHMARK] PLAN model={args.model} launch_model={args.launch_model} "
        f"execution={args.execution_profile} speed={args.desired_speed:g}m/s "
        f"count={len(selected)} scenes={scene_counts}",
        flush=True,
    )
    if args.dry_run:
        for episode_id in selected:
            print(
                f"[BENCHMARK] DRY-RUN {episode_id}: " + " ".join(episode_command(args, workspace, episode_id)),
                flush=True,
            )
        print(f"[BENCHMARK] DRY-RUN COMPLETE count={len(selected)}", flush=True)
        return
    completed = 0
    skipped = 0
    failures = []
    for episode_id in selected:
        config = episodes[episode_id]
        if config.get("implemented", True) is False:
            print(f"[BENCHMARK] SKIP {episode_id}: not validated on Isaac Sim 5.1", flush=True)
            skipped += 1
            continue
        output = workspace / "outputs" / args.model / args.experiment / f"{episode_id}.json"
        if args.resume and output.is_file():
            print(f"[BENCHMARK] RESUME-SKIP {episode_id}: {output}", flush=True)
            skipped += 1
            continue
        episode_complete = False
        last_error = None
        for attempt in range(1, args.max_attempts + 1):
            previous_mtime = output.stat().st_mtime if output.exists() else 0.0
            command = episode_command(args, workspace, episode_id)
            wall_timeout = (
                float(config.get("max_duration", 60.0)) * args.timeout_scale
                + args.startup_grace
            )
            print(
                f"[BENCHMARK] START {episode_id} attempt={attempt}/{args.max_attempts} "
                f"wall_timeout={wall_timeout:.1f}s",
                flush=True,
            )
            log_dir = workspace / "logs" / args.model / args.experiment
            log_dir.mkdir(parents=True, exist_ok=True)
            episode_log = log_dir / f"{episode_id}_attempt{attempt}.log"
            log_stream = episode_log.open("w+", encoding="utf-8")
            process = subprocess.Popen(
                command,
                cwd=workspace,
                start_new_session=True,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
            )
            deadline = time.monotonic() + wall_timeout
            fatal_markers = (
                "ERROR_DEVICE_LOST",
                "GPU crash is detected",
                "ERROR_OUT_OF_DEVICE_MEMORY",
                "out of GPU memory",
                "cudaErrorMemoryAllocation",
            )
            try:
                while time.monotonic() < deadline:
                    if output.exists() and output.stat().st_mtime > previous_mtime:
                        print(f"[BENCHMARK] COMPLETE {episode_id}: {output}", flush=True)
                        episode_complete = True
                        completed += 1
                        break
                    if process.poll() is not None:
                        raise RuntimeError(
                            f"episode launch exited with code {process.returncode} before writing {output}"
                        )
                    log_stream.flush()
                    try:
                        log_stream.seek(0, os.SEEK_END)
                        end = log_stream.tell()
                        log_stream.seek(max(0, end - 65536), os.SEEK_SET)
                        recent_log = log_stream.read()
                    except (OSError, ValueError):
                        recent_log = ""
                    if any(marker in recent_log for marker in fatal_markers):
                        raise RuntimeError(
                            f"Isaac Sim GPU failure detected; see {episode_log}"
                        )
                    time.sleep(0.5)
                else:
                    raise TimeoutError(
                        f"episode {episode_id} did not produce a result within {wall_timeout:.1f}s"
                    )
            except (RuntimeError, TimeoutError) as error:
                last_error = error
                print(f"[BENCHMARK] ATTEMPT-ERROR {episode_id}: {error}", flush=True)
            finally:
                stop_process(process)
                log_stream.close()
            if episode_complete:
                break
        if not episode_complete:
            failures.append((episode_id, str(last_error)))
            print(f"[BENCHMARK] FAILED {episode_id}: {last_error}", flush=True)
            if not args.continue_on_error:
                raise RuntimeError(f"benchmark stopped after {episode_id}: {last_error}")
    print(
        f"[BENCHMARK] FINISHED selected={len(selected)} completed={completed} "
        f"skipped={skipped} infrastructure_failures={len(failures)}",
        flush=True,
    )
    if failures:
        failed_ids = ", ".join(episode_id for episode_id, _error in failures)
        raise RuntimeError(f"Benchmark attempted all episodes but infrastructure failures remain: {failed_ids}")
