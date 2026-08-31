#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${OMNIVLA_PYTHON:-${HOME}/miniconda3/envs/omnivla/bin/python}"
REPOSITORY="${WORKSPACE_ROOT}/third_party/OmniVLA"
CHECKPOINT="${WORKSPACE_ROOT}/checkpoints/vln/omnivla/omnivla-original"
STUB=false
BLACKWELL_COMPAT="${OMNIVLA_ALLOW_BLACKWELL_COMPAT:-0}"
MIN_FREE_GB="${OMNIVLA_MIN_FREE_GB:-14}"
for argument in "$@"; do
    [[ "${argument}" == "--stub" ]] && STUB=true
done

if [[ ! -f "${REPOSITORY}/LICENSE" || ! -f "${REPOSITORY}/inference/run_omnivla.py" ]]; then
    echo "Missing official NHirose/OmniVLA source: ${REPOSITORY}" >&2
    exit 1
fi
if [[ "${STUB}" == true && ! -x "${PYTHON}" ]]; then
    for candidate in \
        "${HOME}/miniconda3/envs/navila/bin/python" \
        "${HOME}/miniconda3/envs/internnav/bin/python" \
        "$(command -v python3)"; do
        if [[ -x "${candidate}" ]] && "${candidate}" -c 'import flask, numpy, PIL' >/dev/null 2>&1; then
            PYTHON="${candidate}"
            break
        fi
    done
fi
if [[ "${STUB}" == false ]]; then
    required=(
        action_head--120000_checkpoint.pt
        proprio_projector--120000_checkpoint.pt
        model-00001-of-00004.safetensors
        model-00002-of-00004.safetensors
        model-00003-of-00004.safetensors
        model-00004-of-00004.safetensors
        model.safetensors.index.json
        config.json
        preprocessor_config.json
    )
    for relative_path in "${required[@]}"; do
        if [[ ! -f "${CHECKPOINT}/${relative_path}" ]]; then
            echo "Missing omnivla-original checkpoint file: ${CHECKPOINT}/${relative_path}" >&2
            exit 1
        fi
    done
fi
if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing independent OmniVLA Python 3.10 environment: ${PYTHON}" >&2
    exit 1
fi
if [[ "${STUB}" == true ]] && ! "${PYTHON}" -c 'import flask, numpy, PIL' >/dev/null 2>&1; then
    echo "Stub mode requires a Python environment with Flask, NumPy, and Pillow." >&2
    exit 1
fi
if [[ "${STUB}" == false && "${BLACKWELL_COMPAT}" != "1" ]] && ! "${PYTHON}" -c '
import sys, numpy, torch, torchvision, torchaudio, flash_attn
assert sys.version_info[:2] == (3, 10)
assert numpy.__version__ == "1.26.4"
assert torch.__version__.split("+")[0] == "2.2.0"
assert torchvision.__version__.split("+")[0] == "0.17.0"
assert torchaudio.__version__.split("+")[0] == "2.2.0"
assert flash_attn.__version__ == "2.5.5"
' >/dev/null 2>&1; then
    echo "OMNIVLA_PYTHON must match official Python 3.10, NumPy 1.26.4, Torch 2.2.0, TorchVision 0.17.0, TorchAudio 2.2.0, and FlashAttention 2.5.5." >&2
    exit 1
fi
if [[ "${STUB}" == false && "${BLACKWELL_COMPAT}" == "1" ]] && ! "${PYTHON}" -c '
import sys, numpy, torch, torchvision
torch_version = tuple(int(value) for value in torch.__version__.split("+")[0].split(".")[:2])
cuda_version = tuple(int(value) for value in torch.version.cuda.split(".")[:2])
assert sys.version_info[:2] == (3, 10)
assert numpy.__version__ == "1.26.4"
assert torch_version >= (2, 7)
assert cuda_version >= (12, 8)
assert torch.cuda.is_available()
' >/dev/null 2>&1; then
    echo "Blackwell compatibility mode requires Python 3.10, NumPy 1.26.4, Torch >=2.7 with CUDA >=12.8, a working CUDA device, and TorchVision." >&2
    exit 1
fi

# OmniVLA needs roughly 12 GB during model placement.  A stale Isaac Sim or
# model process can otherwise leave only a few hundred MiB and produce a long,
# misleading stack trace from transformers.  Refuse to load when the device is
# already too full and print the processes from nvidia-smi for cleanup.
if [[ "${STUB}" == false ]]; then
    if ! "${PYTHON}" - "${MIN_FREE_GB}" <<'PY'
import sys, torch
minimum = float(sys.argv[1]) * (1024 ** 3)
free, total = torch.cuda.mem_get_info()
if free < minimum:
    print(f"free={free / (1024 ** 3):.2f} GiB, required={minimum / (1024 ** 3):.2f} GiB", file=sys.stderr)
    raise SystemExit(1)
print(f"[OMNIVLA] CUDA memory available: {free / (1024 ** 3):.2f}/{total / (1024 ** 3):.2f} GiB", flush=True)
PY
    then
        echo "Insufficient free GPU memory for OmniVLA (threshold ${MIN_FREE_GB} GiB)." >&2
        echo "Run: nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv" >&2
        echo "Stop only stale Isaac/model processes, then retry this command." >&2
        exit 78
    fi
fi

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${WORKSPACE_ROOT}"
exec "${PYTHON}" -m omnivla_adapter.inference_server \
    --repository "${REPOSITORY}" \
    --checkpoint "${CHECKPOINT}" \
    "$@"
