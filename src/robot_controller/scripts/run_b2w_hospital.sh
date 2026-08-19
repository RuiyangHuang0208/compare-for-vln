#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
ROBOT_LAB_SOURCE="${WORKSPACE_ROOT}/third_party/robot_lab/source/robot_lab"
CHECKPOINT="${WORKSPACE_ROOT}/checkpoints/b2w_locomotion/model_2600.pt"
PLAY_SCRIPT="${SCRIPT_DIR}/rsl_rl/play.py"
MODE="${1:-goal}"
CONDA_SETUP="/home/mifcom2/miniconda3/etc/profile.d/conda.sh"
ISAACLAB_ENV="isaaclab232"
ISAAC_SIM_SETUP="/home/mifcom2/isaacsim/setup_conda_env.sh"

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

if ! python -c "import isaaclab; import torch" >/dev/null 2>&1; then
    echo "Python environment is still missing Isaac Lab or PyTorch after setup." >&2
    echo "Expected Conda environment: ${ISAACLAB_ENV}" >&2
    exit 1
fi

echo "[B2W] Environment ready: conda=${CONDA_DEFAULT_ENV:-unknown} python=$(command -v python)"

if [[ ! -d "${ROBOT_LAB_SOURCE}/robot_lab" ]]; then
    echo "Missing RobotLab package: ${ROBOT_LAB_SOURCE}/robot_lab" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing B2W checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

export PYTHONPATH="${ROBOT_LAB_SOURCE}${PYTHONPATH:+:${PYTHONPATH}}"

COMMON_ARGS=(
    --task RobotLab-Isaac-Velocity-Flat-Unitree-B2W-v0
    --scene hospital
    --checkpoint "${CHECKPOINT}"
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
    *)
        echo "Usage: $0 [env-check|goal|keyboard|test|dualvln-sensors|sensor-test|dualvln] [additional play.py arguments]" >&2
        exit 2
        ;;
esac
