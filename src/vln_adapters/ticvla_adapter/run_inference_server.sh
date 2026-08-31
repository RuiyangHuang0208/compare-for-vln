#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${TICVLA_PYTHON:-${HOME}/miniconda3/envs/internnav/bin/python}"
BASE_MODEL="${TICVLA_BASE_MODEL_PATH:-${WORKSPACE_ROOT}/checkpoints/vln/ticvla/InternVL3-1B}"
CHECKPOINT="${TICVLA_CHECKPOINT_PATH:-${WORKSPACE_ROOT}/checkpoints/vln/ticvla/TIC-VLA-model.ckpt}"

if ! "${PYTHON}" -c 'import flask, torch, transformers' >/dev/null 2>&1; then
    echo "TICVLA_PYTHON must point to a Python 3.11 environment with TIC-VLA dependencies." >&2
    exit 1
fi
if [[ ! -e "${BASE_MODEL}" || ! -f "${CHECKPOINT}" ]]; then
    echo "Missing TIC-VLA base model or checkpoint." >&2
    echo "TICVLA_BASE_MODEL_PATH=${BASE_MODEL}" >&2
    echo "TICVLA_CHECKPOINT_PATH=${CHECKPOINT}" >&2
    exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/ticvla_adapter/inference_server.py" \
    --repository "${WORKSPACE_ROOT}/third_party/TIC-VLA" \
    --base-model "${BASE_MODEL}" \
    --checkpoint "${CHECKPOINT}" \
    "$@"
