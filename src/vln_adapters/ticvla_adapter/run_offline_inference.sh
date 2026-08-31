#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${TICVLA_PYTHON:-${HOME}/miniconda3/envs/internnav/bin/python}"
BASE_MODEL="${TICVLA_BASE_MODEL_PATH:-${WORKSPACE_ROOT}/checkpoints/vln/ticvla/InternVL3-1B}"
CHECKPOINT="${TICVLA_CHECKPOINT_PATH:-${WORKSPACE_ROOT}/checkpoints/vln/ticvla/TIC-VLA-model.ckpt}"
IMAGE="${TICVLA_OFFLINE_IMAGE:-${WORKSPACE_ROOT}/outputs/dualvln_sensor/front_rgb.png}"
OUTPUT="${TICVLA_OFFLINE_OUTPUT:-${WORKSPACE_ROOT}/outputs/ticvla/offline_30x2.json}"

export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" -m ticvla_adapter.offline_inference \
    --repository "${WORKSPACE_ROOT}/third_party/TIC-VLA" \
    --base-model "${BASE_MODEL}" \
    --checkpoint "${CHECKPOINT}" \
    --image "${IMAGE}" \
    --output "${OUTPUT}" \
    "$@"
