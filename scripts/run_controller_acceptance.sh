#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT="${CONTROLLER_ACCEPTANCE_EXPERIMENT:-controller_acceptance}"
HEADLESS="${CONTROLLER_ACCEPTANCE_HEADLESS:-1}"

# ROS/colcon setup scripts read optional variables without `${name:-}`.  Keep
# nounset disabled only while sourcing them, then restore strict mode.
set +u
source /opt/ros/jazzy/setup.bash
source "${WORKSPACE_ROOT}/install/setup.bash"
set -u
export ROBOT_VLN_WS="${WORKSPACE_ROOT}"
export DYNANAV_NO_PEDESTRIANS=1
export ROS_LOG_DIR="${ROS_LOG_DIR:-${WORKSPACE_ROOT}/logs/ros}"
mkdir -p "${ROS_LOG_DIR}"

# Fail before opening ROS sockets or Isaac Sim when this terminal cannot see
# the host GPU. This avoids creating misleading partial acceptance results.
if ! /home/mifcom2/miniconda3/bin/conda run -n isaaclab232 \
    python -c 'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'; then
    echo "[CONTROLLER] GPU unavailable in this terminal; no acceptance case was run." >&2
    exit 78
fi

if [[ "${HEADLESS}" == "1" ]]; then
    headless_arg=(--headless)
else
    headless_arg=()
fi

tests=(
    "controller_probe_shared:controller_probe_straight"
    "controller_probe_shared:controller_probe_shared_left"
    "controller_probe_navila:controller_probe_straight"
    "controller_probe_navila:controller_probe_navila_left"
    "controller_probe_uninavid:controller_probe_straight"
    "controller_probe_uninavid:controller_probe_uninavid_left"
    "controller_probe_ticvla:controller_probe_straight"
    "controller_probe_ticvla:controller_probe_velocity_left"
    "controller_probe_omnivla:controller_probe_straight"
    "controller_probe_omnivla:controller_probe_velocity_left"
)

failures=()
for item in "${tests[@]}"; do
    model="${item%%:*}"
    episode="${item#*:}"
    echo "[CONTROLLER] START model=${model} episode=${episode}"
    if ros2 run vln_evaluation benchmark_runner \
        --workspace-root "${WORKSPACE_ROOT}" \
        --model "${model}" \
        --experiment "${EXPERIMENT}" \
        --episodes "${episode}" \
        --max-attempts 1 \
        --timeout-scale 3 \
        --startup-grace 120 \
        "${headless_arg[@]}"; then
        echo "[CONTROLLER] COMPLETE model=${model} episode=${episode}"
    else
        failures+=("${model}:${episode}:infrastructure")
    fi
    # Each benchmark episode owns one Isaac subprocess and waits for its full
    # process tree to exit before the next controller is started.
    sleep 5
done

python3 - "${WORKSPACE_ROOT}" "${EXPERIMENT}" "${tests[@]}" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
experiment = sys.argv[2]
failed = []
print("\nController acceptance summary")
print("controller                         motion     result  error(m) collisions stuck")
for item in sys.argv[3:]:
    model, episode = item.split(":", 1)
    path = root / "outputs" / model / experiment / f"{episode}.json"
    if not path.is_file():
        failed.append(f"{model}:{episode}:missing_result")
        print(f"{model:34} {episode.rsplit('_', 1)[-1]:10} FAIL    missing")
        continue
    result = json.loads(path.read_text(encoding="utf-8"))
    passed = bool(result.get("success")) and int(result.get("physical_collision_count", 0)) == 0
    status = "PASS" if passed else "FAIL"
    error = result.get("navigation_error")
    error_text = "none" if error is None else f"{float(error):.3f}"
    print(
        f"{model:34} {episode.rsplit('_', 1)[-1]:10} {status:7} "
        f"{error_text:8} {result.get('physical_collision_count', 0):10} {result.get('stuck_count', 0):5}"
    )
    if not passed:
        failed.append(f"{model}:{episode}:{result.get('termination_reason')}")

if failed:
    print("FAILED:", ", ".join(failed), file=sys.stderr)
    raise SystemExit(1)
print("All controller acceptance cases passed.")
PY

if (( ${#failures[@]} > 0 )); then
    printf '[CONTROLLER] infrastructure failures: %s\n' "${failures[*]}" >&2
    exit 1
fi
