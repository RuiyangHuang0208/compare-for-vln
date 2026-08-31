#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${UNINAVID_PYTHON:-/home/mifcom2/miniconda3/envs/uninavid/bin/python}"
CHECKPOINT="${UNINAVID_CHECKPOINT_PATH:-${WORKSPACE_ROOT}/checkpoints/vln/uninavid/uninavid-7b-full-224-video-fps-1-grid-2}"
EVA_CHECKPOINT="${UNINAVID_EVA_PATH:-${WORKSPACE_ROOT}/checkpoints/vln/uninavid/eva_vit_g.pth}"

if ! "${PYTHON}" -c 'import flask, PIL, torch, transformers' >/dev/null 2>&1; then
    echo "UNINAVID_PYTHON must point to the independent Uni-NaVid Python 3.10 environment." >&2
    exit 1
fi
if [[ ! -f "${CHECKPOINT}/config.json" || ! -f "${EVA_CHECKPOINT}" ]]; then
    echo "Missing Uni-NaVid model files." >&2
    echo "UNINAVID_CHECKPOINT_PATH=${CHECKPOINT}" >&2
    echo "UNINAVID_EVA_PATH=${EVA_CHECKPOINT}" >&2
    exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/uninavid_adapter/inference_server.py" \
    --repository "${WORKSPACE_ROOT}/third_party/Uni-NaVid" \
    --checkpoint "${CHECKPOINT}" \
    --eva-checkpoint "${EVA_CHECKPOINT}" \
    "$@"
