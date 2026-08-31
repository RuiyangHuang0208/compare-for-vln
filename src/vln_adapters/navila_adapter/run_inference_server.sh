#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${NAVILA_PYTHON:-${HOME}/miniconda3/envs/navila/bin/python}"
REPOSITORY="${WORKSPACE_ROOT}/third_party/NaVILA"
CHECKPOINT="${WORKSPACE_ROOT}/checkpoints/vln/navila"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing NaVILA Python 3.10 environment: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${REPOSITORY}/LICENSE" || ! -f "${REPOSITORY}/llava/model/builder.py" ]]; then
    echo "Missing official NaVILA source in ${REPOSITORY}" >&2
    exit 1
fi
required=(
    config.json
    llm/model.safetensors.index.json
    llm/model-00001-of-00004.safetensors
    llm/model-00002-of-00004.safetensors
    llm/model-00003-of-00004.safetensors
    llm/model-00004-of-00004.safetensors
    mm_projector/model.safetensors
    vision_tower/model.safetensors
)
for relative_path in "${required[@]}"; do
    if [[ ! -f "${CHECKPOINT}/${relative_path}" ]]; then
        echo "Missing NaVILA checkpoint file: ${CHECKPOINT}/${relative_path}" >&2
        exit 1
    fi
done
if ! "${PYTHON}" -c 'import flask, torch, transformers; assert tuple(map(int, __import__("sys").version.split()[0].split(".")))[:2] == (3, 10)' >/dev/null 2>&1; then
    echo "NAVILA_PYTHON must be a Python 3.10 environment with Flask, PyTorch, and Transformers." >&2
    exit 1
fi

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${WORKSPACE_ROOT}"
exec "${PYTHON}" "${SCRIPT_DIR}/navila_adapter/inference_server.py" \
    --repository "${REPOSITORY}" \
    --checkpoint "${CHECKPOINT}" \
    "$@"
