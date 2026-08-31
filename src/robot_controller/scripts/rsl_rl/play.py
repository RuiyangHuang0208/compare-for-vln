# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import io
import json
import math
import queue
import socket
import struct
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
    "--locomotion-policy",
    "--locomotion_policy",
    dest="locomotion_policy",
    choices=("sru-onnx", "isaac-pt", "robotlab"),
    default="sru-onnx",
    help="Low-level B2W policy. SRU's Gazebo ONNX controller is the default; previous policies remain available.",
)
parser.add_argument(
    "--scene",
    type=str,
    choices=("default", "hospital", "office", "outdoor", "warehouse"),
    default="default",
    help="USD environment to load for policy inference.",
)
parser.add_argument("--scene_usd", type=str, default="", help="Optional USD path overriding the selected scene asset.")
parser.add_argument("--spawn_x", type=float, default=None, help="Robot spawn x-coordinate in meters.")
parser.add_argument("--spawn_y", type=float, default=None, help="Robot spawn y-coordinate in meters.")
parser.add_argument("--spawn_z", type=float, default=None, help="Robot base spawn z-coordinate in meters.")
parser.add_argument("--spawn_yaw", type=float, default=None, help="Robot spawn yaw in radians.")
parser.add_argument("--pedestrian_count", type=int, default=0, help="Number of DynaNav dynamic characters.")
parser.add_argument("--pedestrian_seed", type=int, default=666, help="DynaNav character command seed.")
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
parser.add_argument(
    "--ros_nav_udp",
    action="store_true",
    default=False,
    help="Receive [vx, vy, wz] from the ROS 2 loopback bridge and publish simulation odometry back to it.",
)
parser.add_argument("--command_udp_port", type=int, default=5820, help="Loopback UDP port for velocity commands.")
parser.add_argument("--telemetry_udp_port", type=int, default=5821, help="Loopback UDP port for simulation state.")
parser.add_argument(
    "--debug_path_udp_port",
    type=int,
    default=5823,
    help="Loopback UDP port for world-frame paths drawn in the Isaac Sim viewport.",
)
parser.add_argument("--command_timeout", type=float, default=0.5, help="Stop after this many seconds without a UDP command.")
parser.add_argument(
    "--ros_sensor_tcp_port",
    type=int,
    default=0,
    help="Loopback TCP port for aligned RGB-D; zero disables ROS sensor streaming.",
)
parser.add_argument(
    "--camera_hfov",
    type=float,
    default=79.0,
    help="Horizontal field of view for the simulated front RGB-D camera, in degrees.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if not math.isfinite(args_cli.camera_hfov) or not 1.0 < args_cli.camera_hfov < 179.0:
    parser.error("--camera_hfov must be finite and between 1 and 179 degrees")
# Camera sensors require the renderer even when the simulator is headless.
if args_cli.sensor_test_steps > 0 or args_cli.dualvln or args_cli.ros_sensor_tcp_port > 0:
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
import onnxruntime as ort
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
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
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src", "vln_adapters", "dualvln_adapter"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src", "vln_interface", "scripts"))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src", "navigation_bridge", "scripts"))

from path_follower import PathFollower
from safety_filter import SafetyFilter
from dualvln_adapter.service_client import AsyncDualVlnClient
from trajectory_adapter import dualvln_trajectory_to_world

sys.path.insert(0, os.path.join(WORKSPACE_ROOT, "src", "simulation_bridge", "dynanav_bridge"))
from dynanav_bridge.isaac_people import DynaNavPeopleRuntime

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
# DynaNav's published benchmark references Isaac 5.0 assets.  Keep the
# version configurable because Isaac Sim 5.1 can also resolve the matching
# Nucleus assets when the official 5.0 path is unavailable.
DYNANAV_ASSET_VERSION = os.environ.get("DYNANAV_ASSET_VERSION", "5.1").strip()
if DYNANAV_ASSET_VERSION not in {"5.0", "5.1"}:
    raise ValueError("DYNANAV_ASSET_VERSION must be 5.0 or 5.1")
DYNANAV_NUCLEUS_ROOT = (
    f"https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/{DYNANAV_ASSET_VERSION}/Isaac"
)
HOSPITAL_USD_PATH = os.environ.get(
    "DYNANAV_HOSPITAL_USD_PATH", f"{DYNANAV_NUCLEUS_ROOT}/Environments/Hospital/hospital.usd"
)
WAREHOUSE_USD_PATH = os.environ.get(
    "DYNANAV_WAREHOUSE_USD_PATH", f"{DYNANAV_NUCLEUS_ROOT}/Environments/Simple_Warehouse/full_warehouse.usd"
)
ISAAC_B2W_POLICY_PATH = os.path.join(
    WORKSPACE_ROOT, "checkpoints", "b2w_locomotion", "isaac_pt", "policy_b2w_new_2.pt"
)
SRU_ONNX_POLICY_PATH = os.path.join(
    WORKSPACE_ROOT, "checkpoints", "b2w_locomotion", "sru_onnx", "policy_force_new.onnx"
)
ISAAC_B2W_USD_PATH = os.path.join(
    WORKSPACE_ROOT,
    "third_party",
    "sru-navigation-sim",
    "isaaclab_nav_task",
    "navigation",
    "assets",
    "data",
    "Robots",
    "B2W",
    "b2w_rsl.usd",
)
ISAAC_B2W_JOINT_NAMES = (
    "FL_hip_joint",
    "FR_hip_joint",
    "RL_hip_joint",
    "RR_hip_joint",
    "FL_thigh_joint",
    "FR_thigh_joint",
    "RL_thigh_joint",
    "RR_thigh_joint",
    "FL_calf_joint",
    "FR_calf_joint",
    "RL_calf_joint",
    "RR_calf_joint",
    "FL_foot_joint",
    "FR_foot_joint",
    "RL_foot_joint",
    "RR_foot_joint",
)
ISAAC_B2W_DEFAULT_JOINT_POSITIONS = (0.0,) * 4 + (0.4,) * 4 + (-1.3,) * 4 + (0.0,) * 4
DUALVLN_CAMERA_WIDTH = 640
DUALVLN_CAMERA_HEIGHT = 480
DUALVLN_DEFAULT_SPEED = 0.3
DUALVLN_MIN_SPEED = 0.05
DUALVLN_MAX_SPEED = 5.0
DUALVLN_MAX_YAW_RATE = 0.4
def front_camera_intrinsics(horizontal_fov_degrees: float):
    focal = 0.5 * DUALVLN_CAMERA_WIDTH / math.tan(0.5 * math.radians(horizontal_fov_degrees))
    return [
        focal,
        0.0,
        0.5 * DUALVLN_CAMERA_WIDTH,
        0.0,
        focal,
        0.5 * DUALVLN_CAMERA_HEIGHT,
        0.0,
        0.0,
        1.0,
    ]


class RosNavUdpEndpoint:
    """Python-version-neutral loopback bridge between ROS 2 and Isaac Sim."""

    def __init__(self, command_port: int, telemetry_port: int, debug_path_port: int, command_timeout: float):
        self.command_timeout = float(command_timeout)
        # DynaNav's reference benchmark counts only sustained contact (100
        # simulation frames).  A single PhysX impulse is often a leg/base
        # settling spike, so never turn it into a navigation recovery event.
        try:
            self.contact_debounce_steps = max(
                1, int(os.environ.get("DYNANAV_CONTACT_DEBOUNCE_STEPS", "100"))
            )
        except ValueError as error:
            raise ValueError("DYNANAV_CONTACT_DEBOUNCE_STEPS must be a positive integer") from error
        self.contact_streaks = {}
        self.command = np.zeros(3, dtype=np.float32)
        self.last_command_time = None
        self.reset_requested = False
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.command_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.command_socket.bind(("127.0.0.1", int(command_port)))
        self.command_socket.setblocking(False)
        self.telemetry_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.telemetry_address = ("127.0.0.1", int(telemetry_port))
        self.debug_path_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.debug_path_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.debug_path_socket.bind(("127.0.0.1", int(debug_path_port)))
        self.debug_path_socket.setblocking(False)
        self.debug_path = np.empty((0, 3), dtype=np.float32)
        print(
            f"[ROS BRIDGE] Waiting for /nav_vel on UDP {command_port}; telemetry UDP {telemetry_port}; "
            f"viewport path UDP {debug_path_port}; contact debounce={self.contact_debounce_steps} steps",
            flush=True,
        )

    def advance(self):
        while True:
            try:
                payload, _ = self.command_socket.recvfrom(4096)
            except BlockingIOError:
                break
            try:
                decoded = json.loads(payload.decode("utf-8"))
                if decoded.get("reset", False):
                    self.reset_requested = True
                    self.command.fill(0.0)
                    self.last_command_time = None
                    continue
                values = np.asarray((decoded["vx"], decoded["vy"], decoded["wz"]), dtype=np.float32)
                if values.shape != (3,) or not np.isfinite(values).all():
                    raise ValueError("velocity must contain three finite values")
            except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                print(f"[ROS BRIDGE] Rejected UDP command: {error}", flush=True)
                continue
            self.command[:] = values
            self.last_command_time = time.monotonic()
        if self.last_command_time is None or time.monotonic() - self.last_command_time > self.command_timeout:
            self.command.fill(0.0)
        return self.command

    def take_reset(self):
        requested, self.reset_requested = self.reset_requested, False
        return requested

    def take_debug_path(self):
        updated = False
        while True:
            try:
                payload, _ = self.debug_path_socket.recvfrom(65535)
            except BlockingIOError:
                break
            try:
                decoded = json.loads(payload.decode("utf-8"))
                if decoded.get("frame_id") != "world":
                    raise ValueError("debug path frame must be 'world'")
                raw_points = decoded.get("points")
                if not isinstance(raw_points, list) or len(raw_points) > 2048:
                    raise ValueError("debug path must be a list with at most 2048 points")
                points = (
                    np.asarray(raw_points, dtype=np.float32)
                    if raw_points
                    else np.empty((0, 3), dtype=np.float32)
                )
                if points.shape != (len(raw_points), 3) or not np.isfinite(points).all():
                    raise ValueError("debug path must contain finite [x,y,z] points")
            except (TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as error:
                print(f"[ROS BRIDGE] Rejected viewport path: {error}", flush=True)
                continue
            self.debug_path = points
            updated = True
        return self.debug_path, updated

    def publish_telemetry(
        self, robot, sim_time: float, reset: bool = False, contact_sensor=None, extra_contacts=None
    ):
        if reset:
            # Contact debounce belongs to one episode.  Do not carry a
            # pre-reset contact streak into the next spawn.
            self.contact_streaks.clear()
        contacts = []
        if contact_sensor is not None:
            forces = torch.linalg.vector_norm(contact_sensor.data.net_forces_w[0], dim=-1)
            observed_bodies = set()
            for body_name, force in zip(contact_sensor.body_names, forces.detach().cpu().tolist()):
                # Wheel-ground contact is normal locomotion; report contacts made by other links.
                body_lower = body_name.lower()
                if "wheel" in body_lower or "foot" in body_lower:
                    continue
                if force > 100.0:
                    observed_bodies.add(body_name)
                    streak = self.contact_streaks.get(body_name, 0) + 1
                    self.contact_streaks[body_name] = streak
                    if streak >= self.contact_debounce_steps:
                        contacts.append(
                            {
                                "robot_body": body_name,
                                "force": float(force),
                                "contact_streak": streak,
                            }
                        )
                else:
                    self.contact_streaks.pop(body_name, None)
            for body_name in set(self.contact_streaks) - observed_bodies:
                self.contact_streaks.pop(body_name, None)
        if extra_contacts:
            contacts.extend(extra_contacts)
        payload = {
            "stamp": time.time(),
            "sim_time": float(sim_time),
            "position": robot.data.root_link_pos_w[0].detach().cpu().tolist(),
            "orientation_wxyz": robot.data.root_link_quat_w[0].detach().cpu().tolist(),
            "linear_velocity_b": robot.data.root_lin_vel_b[0].detach().cpu().tolist(),
            "angular_velocity_b": robot.data.root_ang_vel_b[0].detach().cpu().tolist(),
            "reset": bool(reset),
            "contacts": contacts,
        }
        self.telemetry_socket.sendto(json.dumps(payload, separators=(",", ":")).encode("utf-8"), self.telemetry_address)

    def close(self):
        self.command.fill(0.0)
        self.command_socket.close()
        self.telemetry_socket.close()
        self.debug_path_socket.close()


class RosSensorTcpEndpoint:
    """Asynchronously stream aligned RGB-D frames without blocking physics."""

    def __init__(self, port: int):
        self.address = ("127.0.0.1", int(port))
        self.frames = queue.Queue(maxsize=1)
        self.closed = threading.Event()
        self.socket = None
        self.worker = threading.Thread(target=self._run, name="ros-sensor-tcp", daemon=True)
        self.worker.start()
        print(f"[ROS SENSOR] RGB-D stream target TCP {self.address[0]}:{self.address[1]}", flush=True)

    def submit(self, camera, sim_time: float):
        rgb = camera.data.output.get("rgb")
        depth = camera.data.output.get("distance_to_image_plane")
        if rgb is None or depth is None:
            return
        frame = (
            rgb[0, ..., :3].detach().cpu().numpy().copy(),
            depth[0, ..., 0].detach().cpu().numpy().astype(np.float32, copy=True),
            camera.data.intrinsic_matrices[0].detach().cpu().numpy().astype(np.float32, copy=True),
            float(sim_time),
        )
        try:
            self.frames.put_nowait(frame)
        except queue.Full:
            try:
                self.frames.get_nowait()
                self.frames.task_done()
                self.frames.put_nowait(frame)
            except (queue.Empty, queue.Full):
                pass

    def _connect(self):
        if self.socket is not None:
            return True
        candidate = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        candidate.settimeout(1.0)
        try:
            candidate.connect(self.address)
        except OSError:
            candidate.close()
            return False
        candidate.settimeout(5.0)
        self.socket = candidate
        return True

    def _send_frame(self, frame):
        from PIL import Image

        rgb, depth, intrinsics, sim_time = frame
        rgb_buffer = io.BytesIO()
        Image.fromarray(rgb, mode="RGB").save(rgb_buffer, format="JPEG", quality=90)
        depth_buffer = io.BytesIO()
        depth_u16 = np.clip(
            np.nan_to_num(depth, nan=0.0, posinf=6.5535, neginf=0.0) * 10000.0,
            0,
            65535,
        ).astype(np.uint16)
        Image.fromarray(depth_u16).save(depth_buffer, format="PNG")
        header = json.dumps(
            {
                "sim_time": sim_time,
                "width": int(rgb.shape[1]),
                "height": int(rgb.shape[0]),
                "intrinsics": intrinsics.reshape(3, 3).tolist(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
        packet = (
            struct.pack("!I", len(header))
            + header
            + struct.pack("!I", len(rgb_buffer.getvalue()))
            + rgb_buffer.getvalue()
            + struct.pack("!I", len(depth_buffer.getvalue()))
            + depth_buffer.getvalue()
        )
        self.socket.sendall(packet)

    def _run(self):
        while not self.closed.is_set():
            try:
                frame = self.frames.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if not self._connect():
                    continue
                self._send_frame(frame)
            except OSError:
                if self.socket is not None:
                    self.socket.close()
                    self.socket = None
            finally:
                self.frames.task_done()

    def close(self):
        self.closed.set()
        if self.socket is not None:
            self.socket.close()
            self.socket = None


def make_isaac_b2w_robot_cfg():
    """Return the robot configuration paired with the official Isaac B2W policy."""
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ISAAC_B2W_USD_PATH,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=None,
                max_angular_velocity=None,
                max_depenetration_velocity=1.0,
                enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.75),
            joint_pos={
                name: value for name, value in zip(ISAAC_B2W_JOINT_NAMES, ISAAC_B2W_DEFAULT_JOINT_POSITIONS)
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.95,
        actuators={
            "hips_thighs": ImplicitActuatorCfg(
                joint_names_expr=list(ISAAC_B2W_JOINT_NAMES[:8]),
                effort_limit_sim=200.0,
                velocity_limit_sim=23.0,
                stiffness=100.0,
                damping=3.5,
                friction=0.0,
            ),
            "calves": ImplicitActuatorCfg(
                joint_names_expr=list(ISAAC_B2W_JOINT_NAMES[8:12]),
                effort_limit_sim=320.0,
                velocity_limit_sim=14.0,
                stiffness=100.0,
                damping=3.5,
                friction=0.0,
            ),
            "wheels": ImplicitActuatorCfg(
                joint_names_expr=list(ISAAC_B2W_JOINT_NAMES[12:]),
                effort_limit_sim=20.0,
                velocity_limit_sim=50.0,
                stiffness=0.0,
                damping=3.0,
                friction=0.0,
            ),
        },
    )


class IsaacB2WPolicy:
    """Build the official 60-D observation and run the recurrent Isaac policy."""

    def __init__(self, model_path, env, command_provider):
        self.env = env
        self.robot = env.unwrapped.scene["robot"]
        self.device = env.unwrapped.device
        self.command_provider = command_provider
        self.joint_ids = []
        for name in ISAAC_B2W_JOINT_NAMES:
            ids, names = self.robot.find_joints(name, preserve_order=True)
            if len(ids) != 1 or names != [name]:
                raise RuntimeError(f"Could not uniquely resolve official Isaac B2W joint {name!r}: {names}")
            self.joint_ids.append(ids[0])
        self.default_joint_positions = torch.tensor(
            ISAAC_B2W_DEFAULT_JOINT_POSITIONS, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        self.model = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.last_actions = torch.zeros((env.num_envs, 16), dtype=torch.float32, device=self.device)
        self.reset(torch.ones(env.num_envs, dtype=torch.bool, device=self.device))

    def _commands(self):
        command = torch.as_tensor(self.command_provider(), dtype=torch.float32, device=self.device)
        if command.ndim == 1:
            command = command.unsqueeze(0)
        if command.shape != (self.env.num_envs, 3):
            command = command.expand(self.env.num_envs, 3)
        return command

    def __call__(self, _robotlab_observation):
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_offsets = torch.remainder(joint_pos - self.default_joint_positions + 2.0 * math.pi, 4.0 * math.pi)
        joint_offsets = joint_offsets - 2.0 * math.pi
        observation = torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._commands(),
                joint_offsets,
                self.robot.data.joint_vel[:, self.joint_ids],
                self.last_actions,
            ),
            dim=-1,
        )
        if observation.shape[-1] != 60:
            raise RuntimeError(f"Official Isaac B2W observation must be 60-D, got {tuple(observation.shape)}")
        actions = self.model(observation)
        if actions.shape != (self.env.num_envs, 16):
            raise RuntimeError(f"Official Isaac B2W policy must return [N,16], got {tuple(actions.shape)}")
        self.last_actions.copy_(actions)
        return actions

    def reset(self, dones):
        reset_mask = torch.as_tensor(dones, dtype=torch.bool, device=self.device).reshape(-1)
        # The exported single-environment TorchScript clears its complete LSTM
        # state on every reset() call, even when the mask is all false.
        if not torch.any(reset_mask):
            return
        self.model.reset(reset_mask)
        self.last_actions[reset_mask] = 0.0


class SruOnnxB2WPolicy:
    """Exact 50 Hz Python adapter for SRU's Gazebo ONNX controller contract."""

    def __init__(self, model_path, env, command_provider):
        if env.num_envs != 1:
            raise ValueError("The official SRU ONNX model has a fixed batch size of one")
        self.env = env
        self.robot = env.unwrapped.scene["robot"]
        self.device = env.unwrapped.device
        self.command_provider = command_provider
        self.joint_ids = []
        for name in ISAAC_B2W_JOINT_NAMES:
            ids, names = self.robot.find_joints(name, preserve_order=True)
            if len(ids) != 1 or names != [name]:
                raise RuntimeError(f"Could not uniquely resolve official SRU B2W joint {name!r}: {names}")
            self.joint_ids.append(ids[0])

        self.default_joint_positions = torch.tensor(
            ISAAC_B2W_DEFAULT_JOINT_POSITIONS, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(model_path, sess_options=options, providers=["CPUExecutionProvider"])
        inputs = {item.name: item.shape for item in self.session.get_inputs()}
        outputs = {item.name: item.shape for item in self.session.get_outputs()}
        expected_inputs = {"obs": [1, 60], "h_in": [1, 1, 256], "c_in": [1, 1, 256]}
        expected_outputs = {"actions": [1, 16], "h_out": [1, 1, 256], "c_out": [1, 1, 256]}
        if inputs != expected_inputs or outputs != expected_outputs:
            raise RuntimeError(f"Unexpected official SRU ONNX contract: inputs={inputs}, outputs={outputs}")
        self.hidden = np.zeros((1, 1, 256), dtype=np.float32)
        self.cell = np.zeros((1, 1, 256), dtype=np.float32)
        self.last_actions = torch.zeros((1, 16), dtype=torch.float32, device=self.device)

    def _commands(self):
        command = torch.as_tensor(self.command_provider(), dtype=torch.float32, device=self.device)
        if command.ndim == 1:
            command = command.unsqueeze(0)
        if command.shape != (1, 3):
            command = command.expand(1, 3)
        return command

    def __call__(self, _robotlab_observation):
        joint_pos = self.robot.data.joint_pos[:, self.joint_ids]
        joint_offsets = torch.remainder(joint_pos - self.default_joint_positions + 2.0 * math.pi, 4.0 * math.pi)
        joint_offsets = joint_offsets - 2.0 * math.pi
        observation = torch.cat(
            (
                self.robot.data.root_lin_vel_b,
                self.robot.data.root_ang_vel_b,
                self.robot.data.projected_gravity_b,
                self._commands(),
                joint_offsets,
                self.robot.data.joint_vel[:, self.joint_ids],
                self.last_actions,
            ),
            dim=-1,
        )
        if observation.shape != (1, 60):
            raise RuntimeError(f"Official SRU ONNX observation must be [1,60], got {tuple(observation.shape)}")
        observation_np = observation.detach().cpu().numpy().astype(np.float32, copy=False)
        actions, hidden, cell = self.session.run(
            ["actions", "h_out", "c_out"],
            {"obs": observation_np, "h_in": self.hidden, "c_in": self.cell},
        )
        if actions.shape != (1, 16) or not np.isfinite(actions).all():
            raise RuntimeError(f"Official SRU ONNX returned invalid actions with shape {actions.shape}")
        self.hidden = hidden
        self.cell = cell
        self.last_actions.copy_(torch.from_numpy(actions).to(self.device))
        return self.last_actions

    def reset(self, dones):
        reset_mask = torch.as_tensor(dones, dtype=torch.bool, device=self.device).reshape(-1)
        if not torch.any(reset_mask):
            return
        self.hidden.fill(0.0)
        self.cell.fill(0.0)
        self.last_actions.zero_()


def make_dualvln_camera_cfg():
    """Create a robot-mounted aligned RGB-D camera outside the locomotion observation."""
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/front_rgb_camera",
        update_period=0.1,
        height=DUALVLN_CAMERA_HEIGHT,
        width=DUALVLN_CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=front_camera_intrinsics(args_cli.camera_hfov),
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
    if args_cli.command_timeout <= 0.0:
        raise ValueError("--command_timeout must be positive")
    if not all(
        1 <= port <= 65535
        for port in (args_cli.command_udp_port, args_cli.telemetry_udp_port, args_cli.debug_path_udp_port)
    ):
        raise ValueError("UDP ports must be in [1, 65535]")
    if not 0 <= args_cli.ros_sensor_tcp_port <= 65535:
        raise ValueError("--ros_sensor_tcp_port must be zero or a valid TCP port")
    goal_enabled = fixed_goal_requested or args_cli.interactive_goal
    if goal_enabled and (args_cli.goal_tolerance <= 0.0 or args_cli.goal_speed <= 0.0 or args_cli.goal_yaw_rate <= 0.0):
        raise ValueError("Goal tolerance, speed, and yaw rate must be positive")
    if sum((args_cli.keyboard, args_cli.b2w_test, goal_enabled, args_cli.dualvln, args_cli.ros_nav_udp)) > 1:
        raise ValueError("keyboard, test, goal, DualVLN, and ROS nav UDP modes are mutually exclusive")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64

    # handle deprecated configurations
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if args_cli.locomotion_policy in ("sru-onnx", "isaac-pt"):
        selected_policy_path = (
            SRU_ONNX_POLICY_PATH if args_cli.locomotion_policy == "sru-onnx" else ISAAC_B2W_POLICY_PATH
        )
        for required_path in (selected_policy_path, ISAAC_B2W_USD_PATH):
            if not os.path.isfile(required_path):
                raise FileNotFoundError(f"Missing official SRU B2W file: {required_path}")
        env_cfg.scene.robot = make_isaac_b2w_robot_cfg()
        env_cfg.actions.joint_pos.joint_names = list(ISAAC_B2W_JOINT_NAMES[:12])
        env_cfg.actions.joint_pos.scale = 0.5
        env_cfg.actions.joint_vel.joint_names = list(ISAAC_B2W_JOINT_NAMES[12:])
        env_cfg.actions.joint_vel.scale = 5.0
        official_contact_material = sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=0.8,
            restitution=0.1,
            compliant_contact_stiffness=5.0e5,
            compliant_contact_damping=300.0,
        )
        env_cfg.scene.terrain.physics_material = official_contact_material
        env_cfg.sim.physics_material = official_contact_material
        env_cfg.sim.disable_contact_processing = True
        if args_cli.locomotion_policy == "sru-onnx":
            # The deployment ONNX has fixed [1, ...] recurrent inputs, matching one Gazebo robot.
            env_cfg.scene.num_envs = 1

    if args_cli.scene != "default":
        scene_usd_paths = {
            "hospital": HOSPITAL_USD_PATH,
            "warehouse": WAREHOUSE_USD_PATH,
        }
        scene_usd = args_cli.scene_usd or scene_usd_paths.get(args_cli.scene)
        if not scene_usd:
            raise ValueError(f"--scene_usd is required for scene {args_cli.scene!r}")
        env_cfg.scene.num_envs = 1
        env_cfg.scene.terrain.terrain_type = "usd"
        env_cfg.scene.terrain.terrain_generator = None
        env_cfg.scene.terrain.usd_path = scene_usd
        env_cfg.scene.sky_light = None
        env_cfg.terminations.terrain_out_of_bounds = None
        print(f"[SCENE] {args_cli.scene}: {scene_usd}", flush=True)

    if args_cli.dualvln_sensors:
        env_cfg.scene.num_envs = 1
        env_cfg.scene.front_rgb = make_dualvln_camera_cfg()

    spawn_pos = list(env_cfg.scene.robot.init_state.pos)
    if args_cli.spawn_x is not None:
        spawn_pos[0] = args_cli.spawn_x
    if args_cli.spawn_y is not None:
        spawn_pos[1] = args_cli.spawn_y
    if args_cli.spawn_z is not None:
        spawn_pos[2] = args_cli.spawn_z
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

    controller = None
    test_command = None
    goal_command = None
    dualvln_command = None
    ros_nav_command = None
    if args_cli.keyboard or args_cli.b2w_test or goal_enabled or args_cli.dualvln or args_cli.ros_nav_udp:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        env_cfg.commands.base_velocity.debug_vis = False
        env_cfg.events.randomize_rigid_body_material = None
        env_cfg.events.randomize_rigid_body_mass_base = None
        env_cfg.events.randomize_rigid_body_mass_others = None
        env_cfg.events.randomize_com_positions = None
        env_cfg.events.randomize_actuator_gains = None
        if hasattr(env_cfg.events, "randomize_push_robot"):
            env_cfg.events.randomize_push_robot = None
        # DynaNav supplies an exact world-frame spawn pose.  Removing the
        # reset event is important: an empty range dictionary still invokes
        # reset_root_state_uniform and can replace the configured orientation
        # with the asset's default state on Isaac Lab versions that run the
        # event after the external pose override.
        env_cfg.events.randomize_reset_base = None

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
    elif args_cli.ros_nav_udp:
        ros_nav_command = torch.zeros(3, dtype=torch.float32)
        env_cfg.observations.policy.velocity_commands = ObsTerm(
            func=lambda env: ros_nav_command.unsqueeze(0).to(env.device),
        )

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.locomotion_policy == "sru-onnx":
        resume_path = SRU_ONNX_POLICY_PATH
    elif args_cli.locomotion_policy == "isaac-pt":
        resume_path = ISAAC_B2W_POLICY_PATH
    elif args_cli.use_pretrained_checkpoint:
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

    people_runtime = None
    if args_cli.pedestrian_count < 0:
        raise ValueError("--pedestrian_count must be non-negative")
    if args_cli.pedestrian_count > 0:
        selected_scene_usd = args_cli.scene_usd or {
            "hospital": HOSPITAL_USD_PATH,
            "warehouse": WAREHOUSE_USD_PATH,
        }.get(args_cli.scene, "")
        people_runtime = DynaNavPeopleRuntime(
            WORKSPACE_ROOT,
            args_cli.scene,
            selected_scene_usd,
            args_cli.pedestrian_count,
            args_cli.pedestrian_seed,
        )

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if people_runtime is not None:
        people_runtime.enable_extensions(simulation_app)
        people_runtime.setup(simulation_app)

    if args_cli.dualvln_sensors:
        print(
            "[DUALVLN SENSOR] Loaded aligned front RGB-D on Robot/base_link; "
            f"resolution={DUALVLN_CAMERA_WIDTH}x{DUALVLN_CAMERA_HEIGHT}, "
            f"hfov={args_cli.camera_hfov:.1f} deg, update_period=0.1 s",
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

    ros_nav_endpoint = None
    ros_contact_sensor = None
    ros_nav_robot = None
    ros_predicted_path_marker = None
    ros_actual_path_marker = None
    ros_actual_trace = []
    ros_path_update_count = 0
    ros_marker_height = (args_cli.spawn_z - 0.75 if args_cli.spawn_z is not None else 0.0) + 0.08
    if args_cli.ros_nav_udp:
        ros_nav_robot = env.unwrapped.scene["robot"]
        ros_contact_sensor = env.unwrapped.scene["contact_forces"]
        ros_nav_endpoint = RosNavUdpEndpoint(
            args_cli.command_udp_port,
            args_cli.telemetry_udp_port,
            args_cli.debug_path_udp_port,
            args_cli.command_timeout,
        )
        if not args_cli.headless:
            predicted_cfg = SPHERE_MARKER_CFG.copy()
            predicted_cfg.prim_path = "/Visuals/VLNPredictedPath"
            predicted_cfg.markers["sphere"].radius = 0.055
            predicted_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.8, 1.0)
            ros_predicted_path_marker = VisualizationMarkers(predicted_cfg)
            ros_predicted_path_marker.set_visibility(False)

            actual_cfg = SPHERE_MARKER_CFG.copy()
            actual_cfg.prim_path = "/Visuals/B2WActualPath"
            actual_cfg.markers["sphere"].radius = 0.035
            actual_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.75, 0.0)
            ros_actual_path_marker = VisualizationMarkers(actual_cfg)
            ros_actual_path_marker.set_visibility(False)
            print(
                "[ISAAC VIEWPORT] Path display enabled: cyan=VLN prediction, yellow=B2-W actual trace",
                flush=True,
            )
    ros_sensor_endpoint = None
    ros_sensor_camera = None
    if args_cli.ros_sensor_tcp_port > 0:
        ros_sensor_camera = env.unwrapped.scene["front_rgb"]
        ros_sensor_endpoint = RosSensorTcpEndpoint(args_cli.ros_sensor_tcp_port)

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

    print(f"[INFO]: Loading {args_cli.locomotion_policy} locomotion policy from: {resume_path}")
    policy_nn = None
    if args_cli.locomotion_policy in ("sru-onnx", "isaac-pt"):

        def sru_command_provider():
            if args_cli.keyboard:
                return controller.advance()
            if test_command is not None:
                return test_command
            if goal_command is not None:
                return goal_command
            if dualvln_command is not None:
                return dualvln_command
            if ros_nav_command is not None:
                ros_nav_command.copy_(torch.from_numpy(ros_nav_endpoint.advance()))
                return ros_nav_command
            return env.unwrapped.command_manager.get_command("base_velocity")

        if args_cli.locomotion_policy == "sru-onnx":
            policy = SruOnnxB2WPolicy(SRU_ONNX_POLICY_PATH, env, sru_command_provider)
        else:
            policy = IsaacB2WPolicy(ISAAC_B2W_POLICY_PATH, env, sru_command_provider)
        position_action_names = env.unwrapped.action_manager._terms["joint_pos"]._joint_names
        velocity_action_names = env.unwrapped.action_manager._terms["joint_vel"]._joint_names
        if position_action_names != list(ISAAC_B2W_JOINT_NAMES[:12]):
            raise RuntimeError(f"Isaac B2W leg action order mismatch: {position_action_names}")
        if velocity_action_names != list(ISAAC_B2W_JOINT_NAMES[12:]):
            raise RuntimeError(f"Isaac B2W wheel action order mismatch: {velocity_action_names}")
        backend_name = "Gazebo deployment ONNX" if args_cli.locomotion_policy == "sru-onnx" else "Isaac TorchScript"
        print(
            f"[INFO]: Official SRU B2W {backend_name} contract: observation=60 action=16 recurrent=true",
            flush=True,
        )
        print(f"[INFO]: Official SRU B2W joint order: {list(ISAAC_B2W_JOINT_NAMES)}", flush=True)
    else:
        # Preserve the original RobotLab RSL-RL checkpoint path and export behavior.
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        else:
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt
    if args_cli.locomotion_policy == "sru-onnx" and not math.isclose(dt, 0.02, abs_tol=1.0e-9):
        raise RuntimeError(f"SRU Gazebo ONNX requires 50 Hz inference (step_dt=0.02 s), got {dt:.9f} s")

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
    if args_cli.ros_nav_udp and ros_nav_robot is not None:
        startup_position = ros_nav_robot.data.root_link_pos_w[0, :2].detach().cpu().tolist()
        startup_quaternion = ros_nav_robot.data.root_link_quat_w[0].detach().cpu().tolist()
        startup_yaw = ros_nav_robot.data.heading_w[0].item()
        print(
            "[DYNANAV SPAWN] actual pose="
            f"({startup_position[0]:.3f}, {startup_position[1]:.3f}, {startup_yaw:.3f}) "
            f"quaternion_wxyz={[round(value, 5) for value in startup_quaternion]}",
            flush=True,
        )
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

                    # Isaac Lab state buffers are inference tensors after the first policy step.
                    # Reset them inside InferenceMode so their in-place initialization is legal.
                    with torch.inference_mode():
                        obs, _ = env.reset()
                        reset_mask = torch.ones(env.num_envs, dtype=torch.long, device=env.unwrapped.device)
                        if args_cli.locomotion_policy in ("sru-onnx", "isaac-pt") or version.parse(
                            installed_version
                        ) >= version.parse("4.0.0"):
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
        elif args_cli.ros_nav_udp:
            ros_nav_command.copy_(torch.from_numpy(ros_nav_endpoint.advance()))
            debug_path, path_updated = ros_nav_endpoint.take_debug_path()
            if path_updated and ros_predicted_path_marker is not None:
                if len(debug_path) == 0:
                    ros_predicted_path_marker.set_visibility(False)
                else:
                    marker_positions = debug_path.copy()
                    marker_positions[:, 2] = ros_marker_height
                    ros_predicted_path_marker.set_visibility(True)
                    ros_predicted_path_marker.visualize(
                        torch.as_tensor(marker_positions, dtype=torch.float32, device=env.unwrapped.device)
                    )
                    ros_path_update_count += 1
                    if ros_path_update_count == 1:
                        print(
                            f"[ISAAC VIEWPORT] Drew first predicted path with {len(marker_positions)} points",
                            flush=True,
                        )
            if ros_nav_endpoint.take_reset():
                ros_nav_command.zero_()
                # See the interactive home path above: state buffers are inference tensors.
                with torch.inference_mode():
                    obs, _ = env.reset()
                    reset_mask = torch.ones(env.num_envs, dtype=torch.bool, device=env.unwrapped.device)
                    if args_cli.locomotion_policy in ("sru-onnx", "isaac-pt") or version.parse(
                        installed_version
                    ) >= version.parse("4.0.0"):
                        policy.reset(reset_mask)
                    else:
                        policy_nn.reset(reset_mask)
                ros_actual_trace.clear()
                if ros_predicted_path_marker is not None:
                    ros_predicted_path_marker.set_visibility(False)
                if ros_actual_path_marker is not None:
                    ros_actual_path_marker.set_visibility(False)
                timestep = 0
                ros_nav_endpoint.publish_telemetry(ros_nav_robot, 0.0, True, ros_contact_sensor)
                print("[ROS BRIDGE] Episode reset completed; robot stopped", flush=True)
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if people_runtime is not None:
                people_runtime.advance(env.unwrapped.step_dt)
            # reset recurrent states for episodes that have terminated
            if args_cli.locomotion_policy in ("sru-onnx", "isaac-pt") or version.parse(
                installed_version
            ) >= version.parse("4.0.0"):
                policy.reset(dones)
            else:
                policy_nn.reset(dones)

        if ros_nav_endpoint is not None:
            if ros_actual_path_marker is not None:
                robot_xy = ros_nav_robot.data.root_link_pos_w[0, :2].detach().cpu().numpy()
                if not ros_actual_trace or np.linalg.norm(robot_xy - np.asarray(ros_actual_trace[-1][:2])) >= 0.08:
                    ros_actual_trace.append((float(robot_xy[0]), float(robot_xy[1]), ros_marker_height))
                    if len(ros_actual_trace) > 500:
                        del ros_actual_trace[: len(ros_actual_trace) - 500]
                    ros_actual_path_marker.set_visibility(True)
                    ros_actual_path_marker.visualize(
                        torch.tensor(ros_actual_trace, dtype=torch.float32, device=env.unwrapped.device)
                    )
            pedestrian_contacts = (
                people_runtime.contacts(ros_nav_robot.data.root_link_pos_w[0].detach().cpu().tolist())
                if people_runtime is not None
                else None
            )
            ros_nav_endpoint.publish_telemetry(
                ros_nav_robot,
                timestep * dt,
                bool(torch.any(dones).item()),
                ros_contact_sensor,
                pedestrian_contacts,
            )
            if ros_sensor_endpoint is not None and timestep % max(1, round(0.1 / dt)) == 0:
                ros_sensor_endpoint.submit(ros_sensor_camera, timestep * dt)
            timestep += 1

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

        if (args_cli.keyboard or args_cli.b2w_test or goal_enabled or args_cli.dualvln or args_cli.ros_nav_udp) and (
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

    if ros_nav_endpoint is not None:
        ros_nav_command.zero_()
        ros_nav_endpoint.close()
    if ros_sensor_endpoint is not None:
        ros_sensor_endpoint.close()
    if people_runtime is not None:
        people_runtime.close()

    # close the simulator
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        # Kit owns background CUDA/rendering threads, so it must also close when
        # Hydra setup or scene initialization raises before the play loop starts.
        simulation_app.close()
