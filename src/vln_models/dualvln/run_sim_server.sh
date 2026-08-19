#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="/home/mifcom2/miniconda3/envs/internnav/bin/python"
MODEL_DIR="${WORKSPACE_ROOT}/checkpoints/dualvln"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing InternNav Python environment: ${PYTHON}" >&2
    exit 1
fi
if [[ ! -f "${MODEL_DIR}/model.safetensors.index.json" ]]; then
    echo "Missing DualVLN checkpoint in ${MODEL_DIR}" >&2
    exit 1
fi
for shard in 00001 00002 00003 00004; do
    if [[ ! -f "${MODEL_DIR}/model-${shard}-of-00004.safetensors" ]]; then
        echo "Missing DualVLN checkpoint shard model-${shard}-of-00004.safetensors" >&2
        exit 1
    fi
done
if [[ ! -f "${MODEL_DIR}/depth_anything_v2_metric_hypersim_vits.pth" ]]; then
    echo "Missing DepthAnything checkpoint in ${MODEL_DIR}" >&2
    exit 1
fi

export INTERNNAV_MINIMAL_IMPORT=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "${WORKSPACE_ROOT}"
exec "${PYTHON}" "${SCRIPT_DIR}/sim_server.py" \
    --model-path "${MODEL_DIR}" \
    --aux-checkpoint-root "${MODEL_DIR}" \
    --attention-implementation sdpa \
    "$@"
