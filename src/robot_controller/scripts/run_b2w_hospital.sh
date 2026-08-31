#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
ROBOT_LAB_SOURCE="${WORKSPACE_ROOT}/third_party/robot_lab/source/robot_lab"
CHECKPOINT="${WORKSPACE_ROOT}/checkpoints/b2w_locomotion/model_2600.pt"
ISAAC_POLICY="${WORKSPACE_ROOT}/checkpoints/b2w_locomotion/isaac_pt/policy_b2w_new_2.pt"
SRU_ONNX_POLICY="${WORKSPACE_ROOT}/checkpoints/b2w_locomotion/sru_onnx/policy_force_new.onnx"
ISAAC_ASSET="${WORKSPACE_ROOT}/third_party/sru-navigation-sim/isaaclab_nav_task/navigation/assets/data/Robots/B2W/b2w_rsl.usd"
PLAY_SCRIPT="${SCRIPT_DIR}/rsl_rl/play.py"
MODE="${1:-goal}"
CONDA_SETUP="${CONDA_SETUP:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
ISAACLAB_ENV="${ISAACLAB_ENV:-isaaclab232}"
ISAAC_SIM_ROOT="${ISAAC_SIM_ROOT:-${HOME}/isaacsim}"
ISAAC_SIM_SETUP="${ISAAC_SIM_SETUP:-${ISAAC_SIM_ROOT}/setup_conda_env.sh}"
LOCOMOTION_POLICY="${B2W_LOCOMOTION_POLICY:-sru-onnx}"

if [[ $# -gt 0 ]]; then
    shift
fi

if ! python -c "import isaaclab" >/dev/null 2>&1; then
    if [[ ! -f "${CONDA_SETUP}" ]]; then
        echo "Missing Conda setup script: ${CONDA_SETUP}" >&2
        exit 1
    fi
    echo "[B2W] Isaac Lab is unavailable in ${CONDA_DEFAULT_ENV:-the current environment}; activating ${ISAACLAB_ENV}."
    set +eu
    source "${CONDA_SETUP}"
    conda activate "${ISAACLAB_ENV}"
    activate_status=$?
    set -eu
    if [[ ${activate_status} -ne 0 ]]; then
        echo "Failed to activate Conda environment: ${ISAACLAB_ENV}" >&2
        exit "${activate_status}"
    fi
fi

if [[ ! -f "${ISAAC_SIM_SETUP}" ]]; then
    echo "Missing Isaac Sim environment setup: ${ISAAC_SIM_SETUP}" >&2
    exit 1
fi
set +eu
source "${ISAAC_SIM_SETUP}"
isaac_setup_status=$?
set -eu
if [[ ${isaac_setup_status} -ne 0 ]]; then
    echo "Failed to load Isaac Sim environment from ${ISAAC_SIM_SETUP}" >&2
    exit "${isaac_setup_status}"
fi

if ! python -c "import isaaclab; import onnxruntime; import torch" >/dev/null 2>&1; then
    echo "Python environment is still missing Isaac Lab, ONNX Runtime, or PyTorch after setup." >&2
    echo "Expected Conda environment: ${ISAACLAB_ENV}" >&2
    exit 1
fi

# Isaac Sim, SRU-ONNX and the locomotion policy are GPU-only in this project.
# Fail before launching the simulator when the shell/container cannot see the
# NVIDIA device; otherwise Isaac Sim may open partially and fail later with a
# misleading CUDA error.
if ! python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)' >/dev/null 2>&1; then
    echo "NVIDIA GPU is unavailable in this shell (torch.cuda.is_available()=False)." >&2
    echo "Check on the host: nvidia-smi && ls -l /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm" >&2
    echo "If nvidia-smi works in another terminal, run this command there rather than in a sandbox/container." >&2
    echo "CPU fallback is intentionally disabled for SRU-ONNX/Isaac Sim." >&2
    exit 78
fi

echo "[B2W] Environment ready: conda=${CONDA_DEFAULT_ENV:-unknown} python=$(command -v python)"

if [[ ! -d "${ROBOT_LAB_SOURCE}/robot_lab" ]]; then
    echo "Missing RobotLab package: ${ROBOT_LAB_SOURCE}/robot_lab" >&2
    exit 1
fi

if [[ "${LOCOMOTION_POLICY}" != "sru-onnx" && "${LOCOMOTION_POLICY}" != "isaac-pt" && "${LOCOMOTION_POLICY}" != "robotlab" ]]; then
    echo "B2W_LOCOMOTION_POLICY must be sru-onnx, isaac-pt, or robotlab; got: ${LOCOMOTION_POLICY}" >&2
    exit 2
fi
if [[ "${LOCOMOTION_POLICY}" == "sru-onnx" && (! -f "${SRU_ONNX_POLICY}" || ! -f "${ISAAC_ASSET}") ]]; then
    echo "Missing official SRU Gazebo ONNX policy or Isaac B2W asset." >&2
    echo "Policy: ${SRU_ONNX_POLICY}" >&2
    echo "Asset:  ${ISAAC_ASSET}" >&2
    exit 1
fi
if [[ "${LOCOMOTION_POLICY}" == "isaac-pt" && (! -f "${ISAAC_POLICY}" || ! -f "${ISAAC_ASSET}") ]]; then
    echo "Missing official Isaac B2W policy or asset." >&2
    echo "Policy: ${ISAAC_POLICY}" >&2
    echo "Asset:  ${ISAAC_ASSET}" >&2
    exit 1
fi
if [[ "${LOCOMOTION_POLICY}" == "robotlab" && ! -f "${CHECKPOINT}" ]]; then
    echo "Missing RobotLab B2W checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

echo "[B2W] Locomotion policy: ${LOCOMOTION_POLICY}"

export PYTHONPATH="${ROBOT_LAB_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS=(
    --task RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0
    --scene hospital
    --checkpoint "${CHECKPOINT}"
    --locomotion-policy "${LOCOMOTION_POLICY}"
    --device cuda:0
    --real-time
)

case "${MODE}" in
    env-check)
        python -c 'import sys, torch, isaaclab; print(f"[B2W] Python {sys.version.split()[0]}; Torch {torch.__version__}; Isaac Lab {isaaclab.__file__}")'
        ;;
    goal)
        exec python "${PLAY_SCRIPT}" "${COMMON_ARGS[@]}" --interactive_goal "$@"
        ;;
    keyboard)
        exec python "${PLAY_SCRIPT}" "${COMMON_ARGS[@]}" --keyboard "$@"
        ;;
    test)
        exec python "${PLAY_SCRIPT}" "${COMMON_ARGS[@]}" --b2w_test "$@"
        ;;
    dualvln-sensors)
        exec python "${PLAY_SCRIPT}" "${COMMON_ARGS[@]}" --interactive_goal --dualvln_sensors "$@"
        ;;
    sensor-test)
        exec python "${PLAY_SCRIPT}" "${COMMON_ARGS[@]}" --interactive_goal --dualvln_sensors \
            --sensor_test_steps 120 --headless "$@"
        ;;
    dualvln)
        exec python "${PLAY_SCRIPT}" "${COMMON_ARGS[@]}" --dualvln "$@"
        ;;
    ros-vln)
        exec python "${PLAY_SCRIPT}" "${COMMON_ARGS[@]}" --ros_nav_udp "$@"
        ;;
    *)
        echo "Usage: $0 [env-check|goal|keyboard|test|dualvln-sensors|sensor-test|dualvln|ros-vln] [additional play.py arguments]" >&2
        exit 2
        ;;
esac
