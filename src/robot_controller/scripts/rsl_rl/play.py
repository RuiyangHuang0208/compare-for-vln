# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import queue
import sys
import threading

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--scene",
    type=str,
    choices=("default", "hospital"),
    default="default",
    help="USD environment to load for policy inference.",
)
parser.add_argument("--spawn_x", type=float, default=None, help="Robot spawn x-coordinate in meters.")
parser.add_argument("--spawn_y", type=float, default=None, help="Robot spawn y-coordinate in meters.")
parser.add_argument("--spawn_yaw", type=float, default=None, help="Robot spawn yaw in radians.")
parser.add_argument("--goal_x", type=float, default=None, help="Goal x-coordinate in world frame, in meters.")
parser.add_argument("--goal_y", type=float, default=None, help="Goal y-coordinate in world frame, in meters.")
parser.add_argument("--goal_tolerance", type=float, default=0.35, help="Goal arrival tolerance in meters.")
parser.add_argument("--goal_speed", type=float, default=0.5, help="Maximum forward speed in goal mode, in m/s.")
parser.add_argument("--goal_yaw_rate", type=float, default=0.8, help="Maximum yaw rate in goal mode, in rad/s.")
parser.add_argument(
    "--interactive_goal",
    action="store_true",
    default=False,
    help="Accept runtime goal commands from stdin: 'goal X Y', 'stop', 'status', or 'quit'.",
)
parser.add_argument(
    "--b2w_test",
    action="store_true",
    default=False,
    help="Run the 40-second B2W velocity-command acceptance sequence.",
)
parser.add_argument(
    "--dualvln_sensors",
    action="store_true",
    default=False,
    help="Attach the simulated front RGB-D camera and expose B2W pose/velocity for DualVLN.",
)
parser.add_argument(
    "--sensor_test_steps",
    type=int,
    default=0,
    help="Save and validate the DualVLN RGB-D sensor after this many simulation steps, then exit.",
)
parser.add_argument(
    "--sensor_output",
    type=str,
    default="outputs/dualvln_sensor/front_rgb.png",
    help="PNG path written by --sensor_test_steps.",
)
parser.add_argument(
    "--depth_output",
    type=str,
    default="outputs/dualvln_sensor/front_depth.png",
    help="16-bit depth PNG path written by --sensor_test_steps.",
)
parser.add_argument("--dualvln", action="store_true", default=False, help="Run simulation-only DualVLN closed loop.")
parser.add_argument(
    "--dualvln_server", type=str, default="http://127.0.0.1:5801", help="DualVLN inference service URL."
)
parser.add_argument(
    "--instruction",
    type=str,
    default="",
    help="Optional initial English instruction; empty waits for runtime input.",
)
parser.add_argument(
    "--dualvln_plan_period", type=float, default=0.5, help="Minimum simulated seconds between RGB-D requests."
)
parser.add_argument(
    "--dualvln_result_timeout", type=float, default=15.0, help="Stop if the latest model result is older than this."
)
parser.add_argument(
    "--dualvln_duration", type=float, default=0.0, help="Exit after this many simulated seconds; zero runs forever."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# Camera sensors require the renderer even when the simulator is headless.
if args_cli.sensor_test_steps > 0 or args_cli.dualvln:
    args_cli.dualvln_sensors = True
if args_cli.video or args_cli.dualvln_sensors:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers import SPHERE_MARKER_CFG, VisualizationMarkers
from isaaclab.sensors import CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:

    def handle_deprecated_rsl_rl_cfg(agent_cfg, _installed_version):
        """Compatibility fallback for Isaac Lab releases without this helper."""
        return agent_cfg
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rl_utils import camera_follow

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src", "vln_models", "dualvln"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src", "vln_interface", "scripts"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src", "navigation_bridge", "scripts"))

from path_follower import PathFollower
from safety_filter import SafetyFilter
from sim_client import AsyncDualVlnClient
from trajectory_adapter import dualvln_trajectory_to_world

# PLACEHOLDER: Extension template (do not remove this comment)

B2W_TEST_PHASES = (
    ("forward", 7.0, (0.5, 0.0, 0.0)),
    ("stop_1", 3.0, (0.0, 0.0, 0.0)),
    ("backward", 7.0, (-0.5, 0.0, 0.0)),
    ("stop_2", 3.0, (0.0, 0.0, 0.0)),
    ("turn_left", 7.0, (0.0, 0.0, 0.5)),
    ("stop_3", 3.0, (0.0, 0.0, 0.0)),
    ("turn_right", 7.0, (0.0, 0.0, -0.5)),
    ("stop_4", 3.0, (0.0, 0.0, 0.0)),
)
HOSPITAL_USD_PATH = f"{ISAAC_NUCLEUS_DIR}/Environments/Hospital/hospital.usd"
DUALVLN_CAMERA_WIDTH = 640
DUALVLN_CAMERA_HEIGHT = 480
DUALVLN_DEFAULT_SPEED = 0.3
DUALVLN_MIN_SPEED = 0.05
DUALVLN_MAX_SPEED = 5.0
DUALVLN_MAX_YAW_RATE = 0.4
# InternNav's real-world reference uses 640x480 with about 79 degrees horizontal FOV.
DUALVLN_CAMERA_INTRINSICS = [
    386.5,
    0.0,
    320.0,
    0.0,
    386.5,
    240.0,
    0.0,
    0.0,
    1.0,
]


def make_dualvln_camera_cfg():
    """Create a robot-mounted aligned RGB-D camera outside the locomotion observation."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_rgb_camera",
        update_period=0.1,
        height=DUALVLN_CAMERA_HEIGHT,
        width=DUALVLN_CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=DUALVLN_CAMERA_INTRINSICS,
            width=DUALVLN_CAMERA_WIDTH,
            height=DUALVLN_CAMERA_HEIGHT,
            clipping_range=(0.05, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(
            # Unitree unitree_ros B2W joint f_oc_link, expressed from base_link.
            pos=(0.3993, 0.0, -0.01576),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


def validate_and_save_dualvln_sensor(env, rgb_output_path, depth_output_path):
    """Validate aligned RGB-D and odometry buffers and save inspection frames."""
    from PIL import Image

    camera = env.unwrapped.scene["front_rgb"]
    robot = env.unwrapped.scene["robot"]
    rgb = camera.data.output.get("rgb")
    depth = camera.data.output.get("distance_to_image_plane")
    if rgb is None or rgb.numel() == 0:
        raise RuntimeError("DualVLN front_rgb sensor returned no image")
    if depth is None or depth.numel() == 0:
        raise RuntimeError("DualVLN front_rgb sensor returned no aligned depth")

    rgb = rgb[0, ..., :3]
    depth = depth[0, ..., 0]
    expected_shape = (DUALVLN_CAMERA_HEIGHT, DUALVLN_CAMERA_WIDTH, 3)
    if tuple(rgb.shape) != expected_shape:
        raise RuntimeError(f"Unexpected front_rgb shape {tuple(rgb.shape)}; expected {expected_shape}")
    if not torch.isfinite(rgb.float()).all():
        raise RuntimeError("DualVLN front_rgb sensor contains non-finite values")

    rgb_std = rgb.float().std().item()
    if rgb_std < 1.0:
        raise RuntimeError(f"DualVLN front_rgb appears blank (pixel std={rgb_std:.3f})")

    expected_depth_shape = (DUALVLN_CAMERA_HEIGHT, DUALVLN_CAMERA_WIDTH)
    if tuple(depth.shape) != expected_depth_shape:
        raise RuntimeError(f"Unexpected aligned depth shape {tuple(depth.shape)}; expected {expected_depth_shape}")
    valid_depth = torch.isfinite(depth) & (depth > 0.05)
    valid_depth_fraction = valid_depth.float().mean().item()
    if valid_depth_fraction < 0.5:
        raise RuntimeError(f"DualVLN aligned depth has too few valid pixels ({valid_depth_fraction:.1%})")
    valid_depth_values = depth[valid_depth]
    depth_min = valid_depth_values.min().item()
    depth_max = valid_depth_values.max().item()

    position = robot.data.root_link_pos_w[0]
    orientation = robot.data.root_link_quat_w[0]
    linear_velocity = robot.data.root_lin_vel_b[0]
    angular_velocity = robot.data.root_ang_vel_b[0]
    state = torch.cat((position, orientation, linear_velocity, angular_velocity))
    if not torch.isfinite(state).all():
        raise RuntimeError("B2W simulated odometry contains non-finite values")

    rgb_output_path = os.path.abspath(rgb_output_path)
    depth_output_path = os.path.abspath(depth_output_path)
    os.makedirs(os.path.dirname(rgb_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(depth_output_path), exist_ok=True)
    Image.fromarray(rgb.detach().cpu().numpy()).save(rgb_output_path)
    depth_mm = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).mul(1000.0).clamp(0, 65535)
    Image.fromarray(depth_mm.to(torch.uint16).cpu().numpy()).save(depth_output_path)
    intrinsic_matrix = camera.data.intrinsic_matrices[0]
    print(
        f"[DUALVLN SENSOR] RGB PASS shape={tuple(rgb.shape)} std={rgb_std:.3f} saved={rgb_output_path}",
        flush=True,
    )
    print(
        "[DUALVLN SENSOR] DEPTH PASS "
        f"shape={tuple(depth.shape)} valid={valid_depth_fraction:.1%} "
        f"range=[{depth_min:.3f}, {depth_max:.3f}] m saved={depth_output_path}",
        flush=True,
    )
    print(
        f"[DUALVLN SENSOR] INTRINSICS PASS matrix={intrinsic_matrix.tolist()}",
        flush=True,
    )
    print(
        "[DUALVLN SENSOR] ODOM PASS "
        f"position={position.tolist()} quaternion_wxyz={orientation.tolist()} "
        f"linear_velocity_b={linear_velocity.tolist()} angular_velocity_b={angular_velocity.tolist()}",
        flush=True,
    )


def update_goal_command(command, robot, goal_xy, tolerance, max_speed, max_yaw_rate):
    """Update a velocity command that turns toward and drives to a world-frame XY goal."""
    position = robot.data.root_link_pos_w[0, :2]
    delta = goal_xy - position
    distance = torch.linalg.vector_norm(delta).item()
    if distance <= tolerance:
        command.zero_()
        return distance, True

    target_heading = math.atan2(delta[1].item(), delta[0].item())
    heading = robot.data.heading_w[0].item()
    heading_error = (target_heading - heading + math.pi) % (2.0 * math.pi) - math.pi

    yaw_rate = max(-max_yaw_rate, min(max_yaw_rate, 1.5 * heading_error))
    minimum_effective_speed = min(0.3, max_speed)
    forward_speed = min(max_speed, max(minimum_effective_speed, 0.8 * (distance - tolerance)))
    if abs(heading_error) > 1.0:
        forward_speed = 0.0
    else:
        forward_speed *= max(0.0, math.cos(heading_error))

    command.copy_(torch.tensor((forward_speed, 0.0, yaw_rate), dtype=command.dtype, device=command.device))
    return distance, False


def read_goal_commands(command_queue):
    """Read runtime goal commands without blocking the simulation loop."""
    while True:
        line = sys.stdin.readline()
        if not line:
            return
        command_queue.put(line.strip())


def get_b2w_test_phase(elapsed_s):
    """Return the active acceptance-test phase and time within it."""
    phase_start = 0.0
    for name, duration, command in B2W_TEST_PHASES:
        if elapsed_s < phase_start + duration:
            return name, elapsed_s - phase_start, command
        phase_start += duration
    return B2W_TEST_PHASES[-1][0], B2W_TEST_PHASES[-1][1], B2W_TEST_PHASES[-1][2]


def print_b2w_test_results(stats, stable_samples, min_height, max_leg_position):
    """Print data-backed acceptance results for the scripted command sequence."""

    def mean(phase_names, field):
        values = [sample[field] for name in phase_names for sample in stats[name]]
        return sum(values) / len(values)

    stop_phases = ("stop_1", "stop_2", "stop_3", "stop_4")
    results = {
        "Forward": mean(("forward",), "vx") > 0.2,
        "Backward": mean(("backward",), "vx") < -0.2,
        "Turn Left": mean(("turn_left",), "yaw_rate") > 0.2,
        "Turn Right": mean(("turn_right",), "yaw_rate") < -0.2,
        "Stop": (
            abs(mean(stop_phases, "vx")) < 0.2
            and abs(mean(stop_phases, "vy")) < 0.2
            and abs(mean(stop_phases, "yaw_rate")) < 0.2
        ),
    }
    upright_fraction = sum(stable_samples) / len(stable_samples)
    results["Stable 30s"] = upright_fraction >= 0.99 and min_height > 0.3 and max_leg_position < 3.2

    print("\n[B2W TEST] 40-second velocity-command acceptance results")
    print(f"Forward:   {'PASS' if results['Forward'] else 'FAIL'}  mean vx={mean(('forward',), 'vx'):.3f} m/s")
    print(f"Backward:  {'PASS' if results['Backward'] else 'FAIL'}  mean vx={mean(('backward',), 'vx'):.3f} m/s")
    print(
        f"Turn Left: {'PASS' if results['Turn Left'] else 'FAIL'}  "
        f"mean yaw_rate={mean(('turn_left',), 'yaw_rate'):.3f} rad/s"
    )
    print(
        f"Turn Right:{'PASS' if results['Turn Right'] else 'FAIL'}  "
        f"mean yaw_rate={mean(('turn_right',), 'yaw_rate'):.3f} rad/s"
    )
    print(
        f"Stop:      {'PASS' if results['Stop'] else 'FAIL'}  "
        f"mean [vx, vy, yaw_rate]=[{mean(stop_phases, 'vx'):.3f}, {mean(stop_phases, 'vy'):.3f}, "
        f"{mean(stop_phases, 'yaw_rate'):.3f}]"
    )
    print(
        f"Stable 30s:{'PASS' if results['Stable 30s'] else 'FAIL'}  upright={upright_fraction:.3%}, "
        f"min height={min_height:.3f} m, max leg position={max_leg_position:.3f} rad"
    )
    print(
        "Wheel speed: "
        f"forward={mean(('forward',), 'wheel_speed'):.3f}, "
        f"backward={mean(('backward',), 'wheel_speed'):.3f}, "
        f"left={mean(('turn_left',), 'wheel_speed'):.3f}, "
        f"right={mean(('turn_right',), 'wheel_speed'):.3f} rad/s"
    )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    fixed_goal_requested = args_cli.goal_x is not None or args_cli.goal_y is not None
    if fixed_goal_requested and (args_cli.goal_x is None or args_cli.goal_y is None):
        raise ValueError("--goal_x and --goal_y must be provided together")
    if args_cli.sensor_test_steps < 0:
        raise ValueError("--sensor_test_steps must be non-negative")
    if args_cli.dualvln_plan_period <= 0.0 or args_cli.dualvln_result_timeout <= 0.0:
        raise ValueError("DualVLN plan period and result timeout must be positive")
    if args_cli.dualvln_duration < 0.0:
        raise ValueError("--dualvln_duration must be non-negative")
    goal_enabled = fixed_goal_requested or args_cli.interactive_goal
    if goal_enabled and (args_cli.goal_tolerance <= 0.0 or args_cli.goal_speed <= 0.0 or args_cli.goal_yaw_rate <= 0.0):
        raise ValueError("Goal tolerance, speed, and yaw rate must be positive")
    if sum((args_cli.keyboard, args_cli.b2w_test, goal_enabled, args_cli.dualvln)) > 1:
        raise ValueError("--keyboard, --b2w_test, goal mode, and --dualvln are mutually exclusive")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.scene == "hospital":
        env_cfg.scene.num_envs = 1
        env_cfg.scene.terrain.terrain_type = "usd"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.usd_path = HOSPITAL_USD_PATH
        env_cfg.scene.sky_light = None
        env_cfg.terminations.terrain_out_of_bounds = None

    if args_cli.dualvln_sensors:
        env_cfg.scene.num_envs = 1
        env_cfg.scene.front_rgb = make_dualvln_camera_cfg()

    spawn_pos = list(env_cfg.scene.robot.init_state.pos)
    if args_cli.spawn_x is not None:
        spawn_pos[0] = args_cli.spawn_x
    if args_cli.spawn_y is not None:
        spawn_pos[1] = args_cli.spawn_y
    env_cfg.scene.robot.init_state.pos = tuple(spawn_pos)
    if args_cli.spawn_yaw is not None:
        half_yaw = 0.5 * args_cli.spawn_yaw
        env_cfg.scene.robot.init_state.rot = (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))

    # spawn the robot randomly in the grid (instead of their terrain levels)
    env_cfg.scene.terrain.max_init_terrain_level = None
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    env_cfg.events.randomize_apply_external_force_torque = None
    env_cfg.events.push_robot = None
    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None

    test_command = None
    goal_command = None
    if args_cli.keyboard or args_cli.b2w_test or goal_enabled or args_cli.dualvln:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.events.randomize_rigid_body_material = None
        env_cfg.events.randomize_rigid_body_mass_base = None
        env_cfg.events.randomize_rigid_body_mass_others = None
        env_cfg.events.randomize_com_positions = None
        env_cfg.events.randomize_actuator_gains = None
        env_cfg.events.randomize_reset_base.params["pose_range"] = {}
        env_cfg.events.randomize_reset_base.params["velocity_range"] = {}

    if args_cli.keyboard:
        config = Se2KeyboardCfg(
            v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
            v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
            omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
        )
        controller = Se2Keyboard(config)
        controller._INPUT_KEY_MAPPING.update(
            {
                "W": controller._INPUT_KEY_MAPPING["UP"],
                "S": controller._INPUT_KEY_MAPPING["DOWN"],
                "A": controller._INPUT_KEY_MAPPING["Z"],
                "D": controller._INPUT_KEY_MAPPING["X"],
            }
        )
        controller.add_callback("SPACE", controller.reset)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: torch.as_tensor(controller.advance(), dtype=torch.float32, device=env.device).unsqueeze(0),
        )
    elif args_cli.b2w_test:
        test_command = torch.zeros(3, dtype=torch.float32)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: test_command.unsqueeze(0).to(env.device),
        )
    elif goal_enabled:
        goal_command = torch.zeros(3, dtype=torch.float32)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: goal_command.unsqueeze(0).to(env.device),
        )
    elif args_cli.dualvln:
        dualvln_command = torch.zeros(3, dtype=torch.float32)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: dualvln_command.unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if args_cli.dualvln_sensors:
        print(
            "[DUALVLN SENSOR] Loaded aligned front RGB-D on Robot/base_link; "
            f"resolution={DUALVLN_CAMERA_WIDTH}x{DUALVLN_CAMERA_HEIGHT}, update_period=0.1 s",
            flush=True,
        )

    goal_robot = None
    goal_xy = None
    goal_marker = None
    goal_input_queue = None
    if goal_enabled:
        goal_robot = env.unwrapped.scene["robot"]
        marker_cfg = SPHERE_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/B2WGoal"
        marker_cfg.markers["sphere"].radius = 0.12
        goal_marker = VisualizationMarkers(marker_cfg)
        if fixed_goal_requested:
            goal_xy = torch.tensor((args_cli.goal_x, args_cli.goal_y), device=env.unwrapped.device)
            goal_marker.visualize(
                torch.tensor(((args_cli.goal_x, args_cli.goal_y, 0.2),), device=env.unwrapped.device)
            )
            print(
                f"[GOAL] Driving to world XY=({args_cli.goal_x:.2f}, {args_cli.goal_y:.2f}), "
                f"tolerance={args_cli.goal_tolerance:.2f} m",
                flush=True,
            )
        else:
            goal_marker.set_visibility(False)
            print("[GOAL] Waiting for a runtime goal; robot is stopped", flush=True)

        if args_cli.interactive_goal:
            goal_input_queue = queue.SimpleQueue()
            threading.Thread(target=read_goal_commands, args=(goal_input_queue,), daemon=True).start()
            print("[GOAL] Runtime commands: goal X Y | stop | status | quit", flush=True)

    dualvln_client = None
    dualvln_robot = None
    dualvln_camera = None
    dualvln_follower = None
    dualvln_safety = None
    dualvln_input_queue = None
    dualvln_ui = None
    dualvln_path_marker = None
    dualvln_instruction = args_cli.instruction.strip()
    if args_cli.dualvln:
        dualvln_robot = env.unwrapped.scene["robot"]
        dualvln_camera = env.unwrapped.scene["front_rgb"]
        dualvln_follower = PathFollower(
            desired_speed=DUALVLN_DEFAULT_SPEED,
            lookahead=0.6,
            goal_tolerance=0.25,
        )
        dualvln_safety = SafetyFilter(
            max_forward_speed=DUALVLN_MAX_SPEED,
            max_yaw_rate=DUALVLN_MAX_YAW_RATE,
        )
        if not dualvln_instruction:
            dualvln_safety.stop()
        dualvln_client = AsyncDualVlnClient(args_cli.dualvln_server)
        dualvln_input_queue = queue.SimpleQueue()
        threading.Thread(target=read_goal_commands, args=(dualvln_input_queue,), daemon=True).start()
        if not args_cli.headless:
            from dualvln_ui import DualVlnStatusWindow

            dualvln_ui = DualVlnStatusWindow(dualvln_instruction, dualvln_follower.desired_speed)
            path_marker_cfg = SPHERE_MARKER_CFG.copy()
            path_marker_cfg.prim_path = "/Visuals/DualVLNPredictedPath"
            path_marker_cfg.markers["sphere"].radius = 0.045
            dualvln_path_marker = VisualizationMarkers(path_marker_cfg)
            dualvln_path_marker.set_visibility(False)
        instruction_status = (
            f"Initial instruction: {dualvln_instruction}"
            if dualvln_instruction
            else "Waiting for an instruction; robot is stopped"
        )
        print(
            f"[DUALVLN] Server ready: {args_cli.dualvln_server}\n"
            f"[DUALVLN] {instruction_status}\n"
            f"[DUALVLN] Runtime commands: instruction TEXT | speed MPS | home | stop | status | quit",
            flush=True,
        )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # export the trained policy to JIT and ONNX formats
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    if version.parse(installed_version) >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
    else:
        # extract the neural network for rsl-rl < 4.0.0
        if version.parse(installed_version) >= version.parse("2.3.0"):
            policy_nn = runner.alg.policy
        else:
            policy_nn = runner.alg.actor_critic

        # extract the normalizer
        if hasattr(policy_nn, "actor_obs_normalizer"):
            normalizer = policy_nn.actor_obs_normalizer
        elif hasattr(policy_nn, "student_obs_normalizer"):
            normalizer = policy_nn.student_obs_normalizer
        else:
            normalizer = None

        # export to JIT and ONNX
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    test_stats = None
    test_step = 0
    if args_cli.b2w_test:
        robot = env.unwrapped.scene["robot"]
        wheel_joint_ids, _ = robot.find_joints(".*_foot_joint")
        leg_joint_ids, _ = robot.find_joints(".*_(hip|thigh|calf)_joint")
        test_stats = {name: [] for name, _, _ in B2W_TEST_PHASES}
        stable_samples = []
        min_height = math.inf
        max_leg_position = 0.0
        test_duration = sum(duration for _, duration, _ in B2W_TEST_PHASES)
        test_steps = math.ceil(test_duration / dt)

    goal_step = 0
    goal_arrived_once = False
    goal_report_interval = max(1, round(1.0 / dt))
    sensor_test_step = 0
    dualvln_step = 0
    dualvln_frame_id = 0
    dualvln_instruction_id = 0
    dualvln_request_interval = max(1, round(args_cli.dualvln_plan_period / dt))
    dualvln_reset_pending = True
    dualvln_manual_stop = args_cli.dualvln and not bool(dualvln_instruction)
    dualvln_received_result = False
    dualvln_last_result_time = time.monotonic()
    dualvln_last_report_step = 0
    dualvln_discrete_command = np.zeros(3, dtype=np.float32)
    dualvln_discrete_until = 0
    dualvln_goal_distance = math.inf
    dualvln_trajectory_count = 0
    dualvln_stable_samples = []
    dualvln_min_height = math.inf
    dualvln_max_command = 0.0

    # reset environment
    obs = env.get_observations()
    dualvln_start_position = (
        dualvln_robot.data.root_link_pos_w[0, :2].detach().cpu().numpy().copy() if args_cli.dualvln else None
    )
    if args_cli.dualvln:
        print(
            f"[HOME] Startup position registered: "
            f"({dualvln_start_position[0]:.3f}, {dualvln_start_position[1]:.3f})",
            flush=True,
        )
    timestep = 0
    # simulate environment
    exit_requested = False
    while simulation_app.is_running():
        start_time = time.time()
        if args_cli.interactive_goal:
            while not goal_input_queue.empty():
                runtime_command = goal_input_queue.get()
                parts = runtime_command.split()
                command_name = parts[0].lower() if parts else ""
                if command_name == "goal" and len(parts) == 3:
                    try:
                        target_x, target_y = float(parts[1]), float(parts[2])
                        if not (math.isfinite(target_x) and math.isfinite(target_y)):
                            raise ValueError
                    except ValueError:
                        print("[GOAL] Invalid target. Usage: goal X Y", flush=True)
                        continue
                    goal_xy = torch.tensor((target_x, target_y), device=env.unwrapped.device)
                    goal_marker.set_visibility(True)
                    goal_marker.visualize(
                        torch.tensor(((target_x, target_y, 0.2),), device=env.unwrapped.device)
                    )
                    goal_arrived_once = False
                    goal_step = 0
                    print(f"[GOAL] New target accepted: ({target_x:.2f}, {target_y:.2f})", flush=True)
                elif command_name == "stop" and len(parts) == 1:
                    goal_xy = None
                    goal_command.zero_()
                    goal_marker.set_visibility(False)
                    print("[GOAL] Target cancelled; stopping", flush=True)
                elif command_name == "status" and len(parts) == 1:
                    position = goal_robot.data.root_link_pos_w[0, :2]
                    if goal_xy is None:
                        print(f"[GOAL] Position=({position[0]:.3f}, {position[1]:.3f}); no active target", flush=True)
                    else:
                        distance = torch.linalg.vector_norm(goal_xy - position).item()
                        print(
                            f"[GOAL] Position=({position[0]:.3f}, {position[1]:.3f}); "
                            f"target=({goal_xy[0]:.3f}, {goal_xy[1]:.3f}); distance={distance:.3f} m",
                            flush=True,
                        )
                elif command_name == "quit" and len(parts) == 1:
                    goal_command.zero_()
                    exit_requested = True
                    print("[GOAL] Quit requested", flush=True)
                else:
                    print("[GOAL] Unknown command. Use: goal X Y | stop | status | quit", flush=True)
        if args_cli.dualvln:
            while not dualvln_input_queue.empty():
                runtime_command = dualvln_input_queue.get().strip()
                command_name, _, command_value = runtime_command.partition(" ")
                command_name = command_name.lower()
                if command_name == "instruction" and command_value.strip():
                    dualvln_instruction = command_value.strip()
                    dualvln_instruction_id += 1
                    dualvln_follower.clear()
                    dualvln_discrete_command.fill(0.0)
                    dualvln_discrete_until = dualvln_step
                    dualvln_command.zero_()
                    dualvln_safety.resume()
                    dualvln_manual_stop = False
                    dualvln_reset_pending = True
                    dualvln_received_result = False
                    if dualvln_path_marker is not None:
                        dualvln_path_marker.set_visibility(False)
                    if dualvln_ui is not None:
                        dualvln_ui.set_instruction(dualvln_instruction)
                        dualvln_ui.set_state("READY", "Waiting to submit RGB-D")
                        dualvln_ui.set_result("-", "No output for the new instruction", -1, math.nan)
                    print(f"[DUALVLN] New instruction: {dualvln_instruction}", flush=True)
                elif command_name == "speed" and command_value.strip():
                    try:
                        desired_speed = float(command_value.strip())
                        if not math.isfinite(desired_speed) or not (
                            DUALVLN_MIN_SPEED <= desired_speed <= DUALVLN_MAX_SPEED
                        ):
                            raise ValueError
                    except ValueError:
                        print(
                            f"[DUALVLN] Invalid speed. Usage: speed MPS "
                            f"({DUALVLN_MIN_SPEED:.2f}-{DUALVLN_MAX_SPEED:.2f})",
                            flush=True,
                        )
                        continue
                    dualvln_follower.desired_speed = desired_speed
                    if dualvln_ui is not None:
                        dualvln_ui.set_telemetry(desired_speed, dualvln_command.tolist(), dualvln_goal_distance)
                    print(f"[DUALVLN] Desired forward speed set to {desired_speed:.2f} m/s", flush=True)
                elif command_name == "home" and not command_value:
                    dualvln_instruction_id += 1
                    dualvln_instruction = ""
                    dualvln_received_result = False
                    dualvln_reset_pending = True
                    dualvln_manual_stop = True
                    dualvln_follower.clear()
                    dualvln_safety.stop()
                    dualvln_discrete_command.fill(0.0)
                    dualvln_discrete_until = dualvln_step
                    dualvln_command.zero_()
                    dualvln_goal_distance = 0.0
                    if dualvln_path_marker is not None:
                        dualvln_path_marker.set_visibility(False)

                    obs, _ = env.reset()
                    reset_mask = torch.ones(env.num_envs, dtype=torch.long, device=env.unwrapped.device)
                    if version.parse(installed_version) >= version.parse("4.0.0"):
                        policy.reset(reset_mask)
                    else:
                        policy_nn.reset(reset_mask)
                    if hasattr(camera_follow, "smooth_camera_positions"):
                        camera_follow.smooth_camera_positions.clear()

                    reset_position = dualvln_robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
                    if dualvln_ui is not None:
                        dualvln_ui.set_instruction("HOME: Reset to startup state")
                        dualvln_ui.set_state("STOPPED", "Reset at startup position")
                        dualvln_ui.set_result("Controller", "Environment reset completed", -1, math.nan)
                        dualvln_ui.set_telemetry(
                            dualvln_follower.desired_speed,
                            (0.0, 0.0, 0.0),
                            0.0,
                        )
                    print(
                        f"[HOME] Reset completed at "
                        f"({reset_position[0]:.3f}, {reset_position[1]:.3f}); robot stopped",
                        flush=True,
                    )
                elif command_name == "stop" and not command_value:
                    dualvln_instruction_id += 1
                    dualvln_manual_stop = True
                    dualvln_follower.clear()
                    dualvln_discrete_command.fill(0.0)
                    dualvln_discrete_until = dualvln_step
                    dualvln_safety.stop()
                    dualvln_command.zero_()
                    if dualvln_path_marker is not None:
                        dualvln_path_marker.set_visibility(False)
                    if dualvln_ui is not None:
                        dualvln_ui.set_state("STOPPED", "Manual STOP")
                    print("[DUALVLN] STOP: trajectory cleared and velocity set to zero", flush=True)
                elif command_name == "status" and not command_value:
                    position = dualvln_robot.data.root_link_pos_w[0, :2].tolist()
                    print(
                        f"[DUALVLN] position=({position[0]:.3f}, {position[1]:.3f}) "
                        f"home=({dualvln_start_position[0]:.3f}, {dualvln_start_position[1]:.3f}) "
                        f"instruction={dualvln_instruction!r} path_active={dualvln_follower.active} "
                        f"desired_speed={dualvln_follower.desired_speed:.2f}m/s "
                        f"command={dualvln_command.tolist()}",
                        flush=True,
                    )
                elif command_name == "quit" and not command_value:
                    dualvln_command.zero_()
                    exit_requested = True
                    if dualvln_ui is not None:
                        dualvln_ui.set_state("STOPPED", "Quit requested")
                    print("[DUALVLN] Quit requested", flush=True)
                else:
                    print(
                        "[DUALVLN] Unknown command. Use: "
                        "instruction TEXT | speed MPS | home | stop | status | quit",
                        flush=True,
                    )

            result, inference_error = dualvln_client.take_result()
            if inference_error is not None:
                inference_error_message = inference_error["message"]
                if inference_error["instruction_id"] != dualvln_instruction_id:
                    print(
                        f"[DUALVLN] Ignored stale inference error from frame "
                        f"{inference_error['frame_id']}: {inference_error_message}",
                        flush=True,
                    )
                else:
                    dualvln_follower.clear()
                    dualvln_safety.stop()
                    dualvln_manual_stop = True
                    dualvln_command.zero_()
                    if dualvln_path_marker is not None:
                        dualvln_path_marker.set_visibility(False)
                    if dualvln_ui is not None:
                        dualvln_ui.set_state("ERROR", inference_error_message)
                    print(f"[DUALVLN] Inference error; stopping: {inference_error_message}", flush=True)
            if result is not None and result["instruction_id"] != dualvln_instruction_id:
                print(f"[DUALVLN] Discarded stale frame {result['frame_id']} from an older instruction", flush=True)
            elif result is not None and dualvln_manual_stop:
                print(f"[DUALVLN] Discarded frame {result['frame_id']} while stopped", flush=True)
            elif result is not None:
                dualvln_last_result_time = time.monotonic()
                dualvln_received_result = True
                dualvln_reset_pending = False
                if result.get("stop", False) or result.get("discrete_action") == [0]:
                    dualvln_follower.clear()
                    dualvln_safety.stop()
                    dualvln_manual_stop = True
                    dualvln_command.zero_()
                    if dualvln_path_marker is not None:
                        dualvln_path_marker.set_visibility(False)
                    if dualvln_ui is not None:
                        dualvln_ui.set_state("STOPPED", "Model completed the task")
                        dualvln_ui.set_result(
                            "System 2",
                            "STOP",
                            result["frame_id"],
                            result.get("inference_s", math.nan),
                        )
                    print(f"[DUALVLN] Model returned STOP (frame {result['frame_id']})", flush=True)
                elif "trajectory" in result:
                    try:
                        world_path = dualvln_trajectory_to_world(result["trajectory"], result["capture_pose"])
                        dualvln_follower.set_path(world_path)
                        dualvln_trajectory_count += 1
                        dualvln_safety.resume()
                        dualvln_manual_stop = False
                        if dualvln_path_marker is not None:
                            marker_positions = torch.as_tensor(world_path, device=env.unwrapped.device)
                            marker_positions = torch.cat(
                                (
                                    marker_positions,
                                    torch.full(
                                        (marker_positions.shape[0], 1),
                                        0.08,
                                        device=env.unwrapped.device,
                                    ),
                                ),
                                dim=1,
                            )
                            dualvln_path_marker.set_visibility(True)
                            dualvln_path_marker.visualize(marker_positions)
                        if dualvln_ui is not None:
                            pixel_goal = result.get("pixel_goal")
                            pixel_text = f"; pixel goal={pixel_goal}" if pixel_goal is not None else ""
                            dualvln_ui.set_state("TRACKING", "Following predicted trajectory")
                            dualvln_ui.set_result(
                                "System 1",
                                f"Trajectory: {len(world_path)} points{pixel_text}",
                                result["frame_id"],
                                result.get("inference_s", math.nan),
                            )
                        print(
                            f"[DUALVLN] Trajectory frame={result['frame_id']} points={len(world_path)} "
                            f"inference={result.get('inference_s', math.nan):.2f}s "
                            f"goal=({world_path[-1, 0]:.2f}, {world_path[-1, 1]:.2f})",
                            flush=True,
                        )
                    except ValueError as exc:
                        dualvln_follower.clear()
                        dualvln_safety.stop()
                        dualvln_manual_stop = True
                        if dualvln_path_marker is not None:
                            dualvln_path_marker.set_visibility(False)
                        if dualvln_ui is not None:
                            dualvln_ui.set_state("ERROR", f"Invalid trajectory: {exc}")
                        print(f"[DUALVLN] Invalid trajectory; stopping: {exc}", flush=True)
                elif "discrete_action" in result:
                    action = result["discrete_action"][:1]
                    if action == [1]:
                        dualvln_discrete_command[:] = (dualvln_follower.desired_speed, 0.0, 0.0)
                    elif action == [2]:
                        dualvln_discrete_command[:] = (0.0, 0.0, DUALVLN_MAX_YAW_RATE)
                    elif action == [3]:
                        dualvln_discrete_command[:] = (0.0, 0.0, -DUALVLN_MAX_YAW_RATE)
                    else:
                        dualvln_discrete_command.fill(0.0)
                    dualvln_discrete_until = dualvln_step + max(1, round(0.5 / dt))
                    dualvln_safety.resume()
                    if dualvln_path_marker is not None:
                        dualvln_path_marker.set_visibility(False)
                    if dualvln_ui is not None:
                        action_name = {1: "FORWARD", 2: "TURN LEFT", 3: "TURN RIGHT"}.get(
                            action[0] if action else -1,
                            str(action),
                        )
                        dualvln_ui.set_state("ACTING", "Executing discrete action")
                        dualvln_ui.set_result(
                            "System 2",
                            action_name,
                            result["frame_id"],
                            result.get("inference_s", math.nan),
                        )
                    print(f"[DUALVLN] Discrete action={action}", flush=True)

            if (
                dualvln_received_result
                and time.monotonic() - dualvln_last_result_time > args_cli.dualvln_result_timeout
            ):
                dualvln_follower.clear()
                dualvln_safety.stop()
                dualvln_manual_stop = True
                if dualvln_path_marker is not None:
                    dualvln_path_marker.set_visibility(False)
                if dualvln_ui is not None:
                    dualvln_ui.set_state("TIMEOUT", "No recent model result; stopped")

            position = dualvln_robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
            heading = dualvln_robot.data.heading_w[0].item()
            if dualvln_step < dualvln_discrete_until:
                raw_command = dualvln_discrete_command
            else:
                raw_command, dualvln_goal_distance, reached = dualvln_follower.command(position, heading)
                if reached:
                    if dualvln_path_marker is not None:
                        dualvln_path_marker.set_visibility(False)
                    if dualvln_ui is not None:
                        dualvln_ui.set_state("WAITING", "Local trajectory reached")
                    print(
                        f"[DUALVLN] Local trajectory reached at {dualvln_goal_distance:.3f} m; stopping",
                        flush=True,
                    )
            safe_command = dualvln_safety.apply(raw_command)
            if dualvln_manual_stop:
                safe_command.fill(0.0)
            dualvln_command.copy_(torch.from_numpy(safe_command))

            if dualvln_step - dualvln_last_report_step >= max(1, round(1.0 / dt)):
                if dualvln_ui is not None:
                    dualvln_ui.set_telemetry(
                        dualvln_follower.desired_speed,
                        safe_command,
                        dualvln_goal_distance,
                    )
                print(
                    f"[DUALVLN] pose=({position[0]:.2f}, {position[1]:.2f}, {heading:.2f}) "
                    f"command=[{safe_command[0]:.2f}, {safe_command[1]:.2f}, {safe_command[2]:.2f}] "
                    f"remaining={dualvln_goal_distance:.2f}",
                    flush=True,
                )
                dualvln_last_report_step = dualvln_step
        if exit_requested:
            break

        if args_cli.b2w_test:
            phase_name, phase_elapsed, command = get_b2w_test_phase(test_step * dt)
            test_command.copy_(torch.tensor(command))
        elif goal_enabled and goal_xy is not None:
            goal_distance, goal_arrived = update_goal_command(
                goal_command,
                goal_robot,
                goal_xy,
                args_cli.goal_tolerance,
                args_cli.goal_speed,
                args_cli.goal_yaw_rate,
            )
            if goal_step % goal_report_interval == 0:
                print(
                    f"[GOAL] distance={goal_distance:.3f} m, "
                    f"command=[{goal_command[0]:.3f}, {goal_command[1]:.3f}, {goal_command[2]:.3f}]",
                    flush=True,
                )
            if goal_arrived and not goal_arrived_once:
                print(f"[GOAL] ARRIVED at distance={goal_distance:.3f} m; stopping", flush=True)
                goal_arrived_once = True
            goal_step += 1
        elif goal_enabled:
            goal_command.zero_()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            # reset recurrent states for episodes that have terminated
            if version.parse(installed_version) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        if args_cli.dualvln:
            height = dualvln_robot.data.root_link_pos_w[0, 2].item()
            gravity_z = dualvln_robot.data.projected_gravity_b[0, 2].item()
            dualvln_stable_samples.append(math.isfinite(height) and math.isfinite(gravity_z) and gravity_z < -0.7)
            dualvln_min_height = min(dualvln_min_height, height)
            dualvln_max_command = max(dualvln_max_command, float(torch.linalg.vector_norm(dualvln_command).item()))
            if not dualvln_manual_stop and dualvln_step % dualvln_request_interval == 0:
                rgb = dualvln_camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                depth = dualvln_camera.data.output["distance_to_image_plane"][0, ..., 0].detach().cpu().numpy()
                pose = [
                    dualvln_robot.data.root_link_pos_w[0, 0].item(),
                    dualvln_robot.data.root_link_pos_w[0, 1].item(),
                    dualvln_robot.data.heading_w[0].item(),
                ]
                accepted = dualvln_client.submit(
                    rgb,
                    depth,
                    {
                        "instruction": dualvln_instruction,
                        "reset": dualvln_reset_pending,
                        "frame_id": dualvln_frame_id,
                        "instruction_id": dualvln_instruction_id,
                        "sim_time": dualvln_step * dt,
                        "capture_pose": pose,
                        "intrinsics": dualvln_camera.data.intrinsic_matrices[0].detach().cpu().tolist(),
                    },
                )
                if accepted:
                    if dualvln_ui is not None:
                        activity = "Tracking current path while planning" if dualvln_follower.active else "Planning"
                        dualvln_ui.set_state("INFERENCE", f"{activity}; frame {dualvln_frame_id}")
                    dualvln_frame_id += 1
            dualvln_step += 1
            if args_cli.dualvln_duration > 0.0 and dualvln_step * dt >= args_cli.dualvln_duration:
                print(f"[DUALVLN] Reached requested duration {args_cli.dualvln_duration:.1f}s", flush=True)
                break

        if args_cli.sensor_test_steps > 0:
            sensor_test_step += 1
            if sensor_test_step >= args_cli.sensor_test_steps:
                validate_and_save_dualvln_sensor(env, args_cli.sensor_output, args_cli.depth_output)
                break

        if args_cli.b2w_test:
            data = robot.data
            gravity_z = data.projected_gravity_b[0, 2].item()
            height = data.root_pos_w[0, 2].item()
            leg_position = data.joint_pos[0, leg_joint_ids].abs().max().item()
            stable_samples.append(math.isfinite(height) and math.isfinite(gravity_z) and gravity_z < -0.7)
            min_height = min(min_height, height)
            max_leg_position = max(max_leg_position, leg_position)
            if phase_elapsed >= 2.0:
                test_stats[phase_name].append(
                    {
                        "vx": data.root_lin_vel_b[0, 0].item(),
                        "vy": data.root_lin_vel_b[0, 1].item(),
                        "yaw_rate": data.root_ang_vel_b[0, 2].item(),
                        "wheel_speed": data.joint_vel[0, wheel_joint_ids].abs().mean().item(),
                    }
                )
            test_step += 1
            if test_step >= test_steps:
                print_b2w_test_results(test_stats, stable_samples, min_height, max_leg_position)
                break
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        if (args_cli.keyboard or args_cli.b2w_test or goal_enabled or args_cli.dualvln) and (
            not args_cli.headless or args_cli.video
        ):
            camera_follow(env)

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if goal_enabled and goal_xy is not None:
        final_distance = torch.linalg.vector_norm(goal_xy - goal_robot.data.root_link_pos_w[0, :2]).item()
        status = "PASS" if final_distance <= args_cli.goal_tolerance else "FAIL"
        print(f"[GOAL] Final distance={final_distance:.3f} m: {status}")

    if dualvln_client is not None:
        final_position = dualvln_robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
        displacement = float(np.linalg.norm(final_position - dualvln_start_position))
        upright_fraction = sum(dualvln_stable_samples) / max(1, len(dualvln_stable_samples))
        print(
            "[DUALVLN TEST] "
            f"trajectories={dualvln_trajectory_count} displacement={displacement:.3f}m "
            f"max_command={dualvln_max_command:.3f} upright={upright_fraction:.3%} "
            f"min_height={dualvln_min_height:.3f}m",
            flush=True,
        )
        dualvln_command.zero_()
        dualvln_client.close()
        if dualvln_ui is not None:
            dualvln_ui.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
