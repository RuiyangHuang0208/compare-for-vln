import os
import json

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetLaunchConfiguration,
    Shutdown,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import yaml

from vln_evaluation.model_registry import (
    load_comparison_manifest,
    load_model_registry,
    select_model,
    validate_comparison_run,
)


def prepare_model(context):
    workspace_root = LaunchConfiguration("workspace_root").perform(context)
    model = LaunchConfiguration("model").perform(context)
    result_model_name = LaunchConfiguration("result_model_name").perform(context).strip() or model
    registry_path = os.path.join(workspace_root, "configs", "models.yaml")
    entry = select_model(load_model_registry(registry_path), model)
    episode_id = LaunchConfiguration("episode").perform(context)
    episodes_file = os.path.join(get_package_share_directory("dynanav_bridge"), "config", "episodes.yaml")
    with open(episodes_file, encoding="utf-8") as stream:
        episodes = yaml.safe_load(stream).get("episodes", {})
    if episode_id not in episodes:
        raise KeyError(f"Unknown episode {episode_id!r}; available={sorted(episodes)}")
    episode = episodes[episode_id]
    pedestrian_count = int(episode.get("pedestrian_count", 0))
    cap_text = os.environ.get("DYNANAV_PEDESTRIAN_CAP", "0")
    try:
        pedestrian_cap = int(cap_text)
    except ValueError as error:
        raise ValueError(f"DYNANAV_PEDESTRIAN_CAP must be a non-negative integer, got {cap_text!r}") from error
    if pedestrian_cap < 0:
        raise ValueError(f"DYNANAV_PEDESTRIAN_CAP must be non-negative, got {pedestrian_cap}")
    no_pedestrians = os.environ.get("DYNANAV_NO_PEDESTRIANS", "0").strip().lower() in {"1", "true", "yes"}
    effective_pedestrian_count = 0 if no_pedestrians else (
        min(pedestrian_count, pedestrian_cap) if pedestrian_cap > 0 else pedestrian_count
    )
    requested_evaluation = LaunchConfiguration("evaluation_mode").perform(context)
    required_evaluation = str(entry["evaluation_mode"])
    if requested_evaluation != required_evaluation:
        raise ValueError(
            f"model {model!r} requires evaluation_mode={required_evaluation!r}, got {requested_evaluation!r}"
        )
    requested_sensor = LaunchConfiguration("sensor_profile").perform(context)
    resolved_sensor = str(entry["sensor_profile"]) if requested_sensor == "auto" else requested_sensor
    if resolved_sensor != entry["sensor_profile"]:
        raise ValueError(
            f"model {model!r} requires sensor_profile={entry['sensor_profile']!r}, got {resolved_sensor!r}"
        )
    requested_goal = LaunchConfiguration("goal_profile").perform(context)
    registered_goal = str(entry.get("goal_profile", "none"))
    resolved_goal = registered_goal if requested_goal == "auto" else requested_goal
    if registered_goal != "none" and resolved_goal != registered_goal:
        raise ValueError(
            f"model {model!r} requires goal_profile={registered_goal!r}, got {resolved_goal!r}"
        )
    manifest = load_comparison_manifest(os.path.join(workspace_root, "configs", "fair_comparison.yaml"))
    execution_profile = LaunchConfiguration("execution_profile").perform(context)
    if execution_profile == "model_specific":
        execution_profile = "native"
    if execution_profile not in {"fair", "native"}:
        raise ValueError("execution_profile must be fair or native")
    comparison_track = validate_comparison_run(
        manifest,
        model,
        entry,
        LaunchConfiguration("comparison_track").perform(context),
        float(LaunchConfiguration("desired_speed").perform(context)),
        requested_evaluation,
    )
    camera_override = LaunchConfiguration("camera_hfov_override").perform(context).strip()
    camera_defaults = manifest["shared"].get("camera", {})
    if camera_override:
        camera_hfov = float(camera_override)
    elif comparison_track == "untracked":
        camera_hfov = float(camera_defaults.get("horizontal_fov_degrees", 79.0))
    else:
        camera_hfov = float(
            manifest["tracks"][comparison_track].get(
                "camera_horizontal_fov_degrees",
                camera_defaults.get("horizontal_fov_degrees", 79.0),
            )
        )
    if not 1.0 < camera_hfov < 179.0:
        raise ValueError(f"comparison track camera HFOV must be between 1 and 179 degrees; got {camera_hfov}")
    output_root = os.path.join(workspace_root, "outputs")
    evaluator = Node(
        package="vln_evaluation",
        executable="evaluator",
        parameters=[
            {
                "model_name": result_model_name,
                "experiment_name": LaunchConfiguration("experiment"),
                "output_root": output_root,
                "evaluation_mode": required_evaluation,
                "direct_velocity": bool(entry["direct_velocity"]),
                "shutdown_after_save": ParameterValue(
                    LaunchConfiguration("shutdown_after_finish"), value_type=bool
                ),
            }
        ],
        output="screen",
        on_exit=Shutdown(reason="single episode evaluator finished"),
    )
    runtime_metadata = dict(entry["runtime_metadata"])
    execution_metadata_text = LaunchConfiguration("execution_metadata_json").perform(context)
    try:
        execution_metadata = json.loads(execution_metadata_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"execution_metadata_json must be valid JSON: {error}") from error
    if not isinstance(execution_metadata, dict):
        raise ValueError("execution_metadata_json must decode to an object")
    registered_controller = entry["runtime_metadata"].get(
        "high_level_controller", "shared_pure_pursuit"
    )
    high_level_controller = str(
        execution_metadata.get("high_level_controller", registered_controller)
    )
    runtime_metadata.update(
        {
            "sensors": entry["sensors"],
            "derived_inputs": entry.get("derived_inputs", []),
            "model_internal_3d_perception": bool(entry.get("model_internal_3d_perception", False)),
            "external_local_avoidance": bool(entry.get("external_local_avoidance", False)),
            "uses_shared_path_follower": high_level_controller in {
                "shared_pure_pursuit", "discrete_action_path"
            },
            "uses_shared_velocity_filter": True,
            "execution_profile": execution_profile,
            "high_level_controller": high_level_controller,
            "model_native_high_level": execution_profile == "native",
            "execution": execution_metadata,
        }
    )
    adapter_arguments = {
        key: str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in entry["launch_arguments"].items()
    }
    if registered_goal != "none":
        adapter_arguments["goal_profile"] = resolved_goal
    adapter = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory(str(entry["package"])),
                "launch",
                str(entry["launch_file"]),
            )
        ),
        launch_arguments=adapter_arguments.items(),
    )
    return [
        SetLaunchConfiguration("execution_profile", execution_profile),
        SetLaunchConfiguration("resolved_sensor_profile", resolved_sensor),
        SetLaunchConfiguration("resolved_result_model_name", result_model_name),
        SetLaunchConfiguration("resolved_evaluation_mode", required_evaluation),
        SetLaunchConfiguration("resolved_comparison_track", comparison_track),
        SetLaunchConfiguration("resolved_camera_hfov", str(camera_hfov)),
        SetLaunchConfiguration("resolved_model_inputs", ",".join(entry["model_inputs"])),
        SetLaunchConfiguration("resolved_goal_profile", resolved_goal),
        SetLaunchConfiguration("resolved_high_level_controller", high_level_controller),
        SetLaunchConfiguration(
            "resolved_model_runtime", json.dumps(runtime_metadata, separators=(",", ":"))
        ),
        SetLaunchConfiguration(
            "resolved_lock_speed",
            "true" if execution_profile == "native" or comparison_track != "untracked" else "false",
        ),
        SetLaunchConfiguration("resolved_pedestrian_count", str(effective_pedestrian_count)),
        adapter,
        evaluator,
    ]


def launch_navigation_bridge(context):
    navigation_config = os.path.join(
        get_package_share_directory("navigation_bridge"), "config", "navigation.yaml"
    )
    overrides_text = LaunchConfiguration("navigation_overrides_json").perform(context)
    try:
        overrides = json.loads(overrides_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"navigation_overrides_json must be valid JSON: {error}") from error
    if not isinstance(overrides, dict):
        raise ValueError("navigation_overrides_json must decode to an object")
    allowed = {
        "path_follower.controller",
        "path_follower.lookahead_distance",
        "path_follower.goal_tolerance",
        "path_follower.yaw_gain",
        "path_follower.yaw_filter_alpha",
        "path_follower.curvature_feedforward_gain",
        "limits.max_vx",
        "limits.max_vy",
        "limits.max_wz",
        "limits.max_linear_acceleration",
        "limits.max_angular_acceleration",
        "limits.max_linear_deceleration",
        "limits.max_angular_deceleration",
        "timeout.trajectory_timeout",
        "stuck.recovery_speed",
        "stuck.recovery_duration",
    }
    unknown = sorted(set(overrides) - allowed)
    if unknown:
        raise ValueError(f"Unsupported navigation overrides: {unknown}")
    resolved = {
        name: (str(value) if name == "path_follower.controller" else float(value))
        for name, value in overrides.items()
    }
    resolved_controller = LaunchConfiguration("resolved_high_level_controller").perform(context)
    if resolved_controller in {"shared_pure_pursuit", "discrete_action_path"}:
        resolved.setdefault("path_follower.controller", resolved_controller)
    return [
        Node(
            package="navigation_bridge",
            executable="navigation_bridge",
            parameters=[
                navigation_config,
                resolved,
                {
                    "path_follower.desired_speed": ParameterValue(
                        LaunchConfiguration("desired_speed"), value_type=float
                    ),
                    "runtime.lock_desired_speed": ParameterValue(
                        LaunchConfiguration("resolved_lock_speed"), value_type=bool
                    ),
                },
            ],
            output="screen",
        )
    ]


def launch_simulator(context):
    workspace_root = LaunchConfiguration("workspace_root").perform(context)
    episode_id = LaunchConfiguration("episode").perform(context)
    episodes_file = os.path.join(get_package_share_directory("dynanav_bridge"), "config", "episodes.yaml")
    with open(episodes_file, encoding="utf-8") as stream:
        episodes = yaml.safe_load(stream).get("episodes", {})
    if episode_id not in episodes:
        raise KeyError(f"Unknown episode {episode_id!r}; available={sorted(episodes)}")
    episode = episodes[episode_id]
    effective_pedestrian_count = int(LaunchConfiguration("resolved_pedestrian_count").perform(context))
    print(f"[DYNANAV] pedestrians for {episode_id}: effective={effective_pedestrian_count}", flush=True)
    spawn = episode.get("spawn")
    if not isinstance(spawn, list) or len(spawn) != 3:
        raise ValueError(f"Episode {episode_id!r} must define spawn: [x, y, yaw]")
    script = os.path.join(workspace_root, "src", "robot_controller", "scripts", "run_b2w_hospital.sh")
    command = [
        script,
        "ros-vln",
        "--locomotion-policy",
        "sru-onnx",
        "--ros_sensor_tcp_port",
        "5822",
        "--spawn_x",
        str(spawn[0]),
        "--spawn_y",
        str(spawn[1]),
        "--spawn_z",
        str(float(episode.get("floor_z", 0.0)) + 0.75),
        "--spawn_yaw",
        str(spawn[2]),
        "--scene",
        str(episode["scene"]),
        "--pedestrian_count",
        str(effective_pedestrian_count),
        "--pedestrian_seed",
        str(int(episode.get("seed", 666))),
        "--camera_hfov",
        LaunchConfiguration("resolved_camera_hfov").perform(context),
    ]
    if episode["scene"] in {"office", "outdoor"}:
        scene_file = "office.usd" if episode["scene"] == "office" else os.environ.get(
            "DYNANAV_OUTDOOR_ASSET", "outdoor_small.usd"
        )
        if os.path.basename(scene_file) != scene_file:
            raise ValueError("DYNANAV_OUTDOOR_ASSET must be a file name within DynaNav/assets")
        scene_usd = os.path.join(workspace_root, "third_party", "TIC-VLA", "DynaNav", "assets", scene_file)
        if not os.path.isfile(scene_usd):
            raise FileNotFoundError(f"Missing local DynaNav scene: {scene_usd}")
        print(f"[DYNANAV] {episode['scene']} asset: {scene_usd}", flush=True)
        command.extend(("--scene_usd", scene_usd))
    if LaunchConfiguration("headless").perform(context).lower() in {"1", "true", "yes"}:
        command.append("--headless")
    return [ExecuteProcess(cmd=command, output="screen")]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "workspace_root",
                default_value=EnvironmentVariable("ROBOT_VLN_WS", default_value=os.getcwd()),
            ),
            DeclareLaunchArgument("model", default_value="dummy"),
            DeclareLaunchArgument("result_model_name", default_value=""),
            DeclareLaunchArgument("episode", default_value="hospital_001"),
            DeclareLaunchArgument("experiment", default_value="single_episode"),
            DeclareLaunchArgument("evaluation_mode", default_value="trajectory_normalized"),
            DeclareLaunchArgument("sensor_profile", default_value="auto"),
            DeclareLaunchArgument("goal_profile", default_value="auto"),
            DeclareLaunchArgument("comparison_track", default_value="auto"),
            DeclareLaunchArgument("execution_profile", default_value="fair"),
            DeclareLaunchArgument("navigation_overrides_json", default_value="{}"),
            DeclareLaunchArgument("execution_metadata_json", default_value="{}"),
            DeclareLaunchArgument("camera_hfov_override", default_value=""),
            DeclareLaunchArgument("resolved_camera_hfov", default_value="79.0"),
            DeclareLaunchArgument(
                "resolved_high_level_controller", default_value="shared_pure_pursuit"
            ),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("desired_speed", default_value="1.0"),
            DeclareLaunchArgument("shutdown_after_finish", default_value="false"),
            DeclareLaunchArgument("debug_first_rgb_path", default_value=""),
            OpaqueFunction(function=prepare_model),
            Node(
                package="dynanav_bridge",
                executable="sensor_bridge",
                parameters=[{"debug_first_rgb_path": LaunchConfiguration("debug_first_rgb_path")}],
                output="screen",
            ),
            Node(
                package="dynanav_bridge",
                executable="episode_manager",
                parameters=[{
                    "episode_id": LaunchConfiguration("episode"),
                    "model_name": LaunchConfiguration("resolved_result_model_name"),
                    "evaluation_mode": LaunchConfiguration("resolved_evaluation_mode"),
                    "sensor_profile": LaunchConfiguration("resolved_sensor_profile"),
                    "goal_profile": LaunchConfiguration("resolved_goal_profile"),
                    "comparison_track": LaunchConfiguration("resolved_comparison_track"),
                    "execution_profile": LaunchConfiguration("execution_profile"),
                    "desired_speed": ParameterValue(
                        LaunchConfiguration("desired_speed"), value_type=float
                    ),
                    "camera_horizontal_fov_degrees": ParameterValue(
                        LaunchConfiguration("resolved_camera_hfov"), value_type=float
                    ),
                    "model_inputs": LaunchConfiguration("resolved_model_inputs"),
                    "model_runtime": ParameterValue(
                        LaunchConfiguration("resolved_model_runtime"), value_type=str
                    ),
                    "pedestrian_count": ParameterValue(
                        LaunchConfiguration("resolved_pedestrian_count"), value_type=int
                    ),
                }],
                output="screen",
            ),
            Node(
                package="robot_controller",
                executable="udp_velocity_bridge",
                parameters=[{"command_source": "vln"}],
                output="screen",
            ),
            OpaqueFunction(function=launch_navigation_bridge),
            Node(package="vln_evaluation", executable="goal_monitor", output="screen"),
            OpaqueFunction(function=launch_simulator),
        ]
    )
