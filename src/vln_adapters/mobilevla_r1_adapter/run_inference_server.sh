#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${MOBILEVLA_R1_PYTHON:-${HOME}/miniconda3/envs/mobilevla_r1/bin/python}"
REPOSITORY="${WORKSPACE_ROOT}/third_party/MobileVLA-R1"
CHECKPOINT="${WORKSPACE_ROOT}/checkpoints/vln/mobilevla_r1/MobileVLA-R1/weight/rl"
STUB=false
for argument in "$@"; do
    [[ "${argument}" == "--stub" ]] && STUB=true
done

if [[ ! -f "${REPOSITORY}/inference.py" || ! -f "${REPOSITORY}/LICENSE" ]]; then
    echo "Missing official MobileVLA-R1 source: ${REPOSITORY}" >&2
    exit 1
fi
if [[ "${STUB}" == true && ! -x "${PYTHON}" ]]; then
    for candidate in "${HOME}/miniconda3/envs/omnivla/bin/python" "${HOME}/miniconda3/envs/internnav/bin/python" "$(command -v python3)"; do
        if [[ -x "${candidate}" ]] && "${candidate}" -c 'import flask, numpy, PIL' >/dev/null 2>&1; then
            PYTHON="${candidate}"
            break
        fi
    done
fi
if [[ "${STUB}" == false ]]; then
    if [[ ! -d "${CHECKPOINT}" || ! -f "${CHECKPOINT}/config.json" ]]; then
        echo "Missing extracted MobileVLA-R1 checkpoint: ${CHECKPOINT}" >&2
        echo "Official archive is 26.9 GB in weight.zip.part-aa through weight.zip.part-az; it was not auto-downloaded." >&2
        exit 1
    fi
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing independent MobileVLA-R1 Python 3.10 environment: ${PYTHON}" >&2
    exit 1
fi
if [[ "${STUB}" == false ]]; then
    if ! "${PYTHON}" -c 'import sys, torch, transformers; assert sys.version_info[:2] == (3, 10)' >/dev/null 2>&1; then
        echo "The independent environment must provide Python 3.10 plus the official Torch/Transformers stack." >&2
        exit 1
    fi
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
cd "${WORKSPACE_ROOT}"
exec "${PYTHON}" -m mobilevla_r1_adapter.inference_server \
    --repository "${REPOSITORY}" \
    --checkpoint "${CHECKPOINT}" \
    "$@"
