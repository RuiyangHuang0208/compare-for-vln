#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPERIMENT="${DYNANAV_EXPERIMENT:-dynanav_full85_20260824}"
EPISODES="${DYNANAV_EPISODES:-all}"
RESUME="${DYNANAV_RESUME:-1}"
CONTINUE_ON_ERROR="${DYNANAV_CONTINUE_ON_ERROR:-1}"
PEDESTRIAN_CAP="${DYNANAV_PEDESTRIAN_CAP:-40}"
NO_PEDESTRIANS="${DYNANAV_NO_PEDESTRIANS:-1}"
EXCLUDE_SCENES="${DYNANAV_EXCLUDE_SCENES:-outdoor}"
ASSET_VERSION="${DYNANAV_ASSET_VERSION:-5.0}"
HEADLESS="${DYNANAV_HEADLESS:-1}"
TIMEOUT_SCALE="${DYNANAV_TIMEOUT_SCALE:-2.5}"
STARTUP_GRACE="${DYNANAV_STARTUP_GRACE:-120}"
OFFICIAL_BENCHMARK=0
MODEL_SPECIFIC_EXECUTION=0
MODELS=()
while (( $# > 0 )); do
    case "$1" in
        --help|-h)
            cat <<'USAGE'
用法：
  ./scripts/run_dynanav_full85_all.sh                 # 五个模型运行非 Outdoor 的75回合
  ./scripts/run_dynanav_full85_all.sh --official     # 官方85回合 + models.yaml内各模型执行参数
  ./scripts/run_dynanav_full85_all.sh --native --no-outdoor
  ./scripts/run_dynanav_full85_all.sh --official --no-resume ticvla
  ./scripts/run_dynanav_full85_all.sh dualvln navila  # 只运行指定模型
  DYNANAV_EXCLUDE_SCENES=outdoor ./scripts/run_dynanav_full85_all.sh
  DYNANAV_HEADLESS=0 ./scripts/run_dynanav_full85_all.sh --episodes simple_vending_machine
  DYNANAV_EPISODES=simple_forward_3m_standard ./scripts/run_dynanav_full85_all.sh omnivla dualvln navila uninavid
  ./scripts/run_dynanav_full85_all.sh --episodes hospital_001 omnivla_native

环境变量：
  DYNANAV_EXPERIMENT=名称       结果目录名称（默认 dynanav_full85_20260824）
  DYNANAV_EPISODES=列表         回合列表或 all（默认 all；逗号分隔）
  DYNANAV_RESUME=0              不跳过已有 JSON（默认跳过）
  DYNANAV_CONTINUE_ON_ERROR=0   某模型失败后停止（默认继续其他模型）
  DYNANAV_PEDESTRIAN_CAP=数量   动态行人显存保护上限（默认40，0表示不限制）
  DYNANAV_NO_PEDESTRIANS=1      强制所有回合不加载行人（默认；设为0才使用官方人数）
  DYNANAV_EXCLUDE_SCENES=场景列表  排除场景（默认 outdoor；逗号分隔，设为空字符串表示不排除）
  DYNANAV_ASSET_VERSION=5.0      DynaNav 官方 Isaac 资产版本（可选 5.0 或 5.1）
  DYNANAV_HEADLESS=1              无界面运行（默认）；设为0显示 Isaac Sim 窗口
  DYNANAV_TIMEOUT_SCALE=2.5       外层墙钟超时倍率（不改变仿真回合时限）
  DYNANAV_STARTUP_GRACE=120       每回合额外启动宽限秒数
  DYNANAV_OUTDOOR_ASSET=文件名  Outdoor 场景资产（默认 outdoor_small.usd，官方 benchmark 资产）

官方模式：
  --official                  使用官方 seed=666、每回合人数、场景、起点、目标、指令、超时和1.5m（或官方逐回合覆盖值）；不排除任何场景。
                              该选项会强制 DYNANAV_NO_PEDESTRIANS=0、DYNANAV_PEDESTRIAN_CAP=0、DYNANAV_EXCLUDE_SCENES=''。
                              同时直接读取 configs/models.yaml 中每个模型的执行参数。
                              始终使用本工程的 B2-W + SRU-ONNX 执行器。
  --native                    使用各模型原生高层控制，不改变回合/行人/场景筛选。
                              始终保留 B2-W SRU-ONNX 作为统一低层执行器。
  --model-specific            --native 的兼容别名。
USAGE
            exit 0
            ;;
        --experiment)
            [[ $# -ge 2 ]] || { echo "--experiment 需要名称" >&2; exit 2; }
            EXPERIMENT="$2"
            shift 2
            ;;
        --episodes)
            [[ $# -ge 2 ]] || { echo "--episodes 需要回合列表或 all" >&2; exit 2; }
            EPISODES="$2"
            shift 2
            ;;
        --no-resume)
            RESUME=0
            shift
            ;;
        --stop-on-error)
            CONTINUE_ON_ERROR=0
            shift
            ;;
        --exclude-scenes)
            [[ $# -ge 2 ]] || { echo "--exclude-scenes 需要场景列表" >&2; exit 2; }
            EXCLUDE_SCENES="$2"
            shift 2
            ;;
        --no-outdoor)
            EXCLUDE_SCENES="outdoor"
            shift
            ;;
        --gui|--no-headless)
            HEADLESS=0
            shift
            ;;
        --headless)
            HEADLESS=1
            shift
            ;;
        --official)
            OFFICIAL_BENCHMARK=1
            MODEL_SPECIFIC_EXECUTION=1
            shift
            ;;
        --native|--model-specific)
            MODEL_SPECIFIC_EXECUTION=1
            shift
            ;;
        --*)
            echo "未知参数：$1" >&2
            exit 2
            ;;
        *)
            MODELS+=("$1")
            shift
            ;;
    esac
done
if [[ "${OFFICIAL_BENCHMARK}" == "1" ]]; then
    # Match TIC-VLA DynaNav benchmark conditions.  Keep the B2-W executor and
    # workspace adapters unchanged; only the benchmark episode conditions are
    # switched to the upstream values.
    NO_PEDESTRIANS=0
    PEDESTRIAN_CAP=0
    EXCLUDE_SCENES=""
fi
if (( ${#MODELS[@]} == 0 )); then
    MODELS=(ticvla omnivla dualvln navila uninavid)
fi
SERVER_PID=""
FAILED_MODELS=()
SHUTTING_DOWN=0

declare -A PORTS=(
    [ticvla]=5802
    [ticvla_native]=5802
    [omnivla]=5805
    [omnivla_native]=5805
    [dualvln]=5801
    [navila]=5803
    [uninavid]=5804
)

cleanup_server() {
    local pid="${SERVER_PID}"
    [[ -n "${pid}" ]] || return 0
    SERVER_PID=""

    if kill -0 "${pid}" 2>/dev/null; then
        # Model servers may have CUDA/Flask worker threads that ignore SIGINT.
        # Never block the benchmark forever while waiting for one to exit.
        kill -INT "${pid}" 2>/dev/null || true
        for _ in $(seq 1 15); do
            kill -0 "${pid}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[ALL85] server pid=${pid} did not stop after SIGINT; sending SIGTERM" >&2
            kill -TERM "${pid}" 2>/dev/null || true
            for _ in $(seq 1 5); do
                kill -0 "${pid}" 2>/dev/null || break
                sleep 1
            done
        fi
        if kill -0 "${pid}" 2>/dev/null; then
            echo "[ALL85] server pid=${pid} still running; sending SIGKILL" >&2
            kill -KILL "${pid}" 2>/dev/null || true
        fi
    fi
    wait "${pid}" 2>/dev/null || true
}

handle_interrupt() {
    [[ "${SHUTTING_DOWN}" == "1" ]] && exit 130
    SHUTTING_DOWN=1
    cleanup_server
    exit 130
}

trap cleanup_server EXIT
trap handle_interrupt INT TERM

start_server() {
    local model="$1"
    local log="${WORKSPACE_ROOT}/logs/${model}_${EXPERIMENT}_server.log"
    case "${model}" in
        ticvla|ticvla_native)
            env TICVLA_PYTHON=/home/mifcom2/miniconda3/envs/internnav/bin/python \
                "${WORKSPACE_ROOT}/src/vln_adapters/ticvla_adapter/run_inference_server.sh" \
                >"${log}" 2>&1 &
            ;;
        omnivla)
            env OMNIVLA_ALLOW_BLACKWELL_COMPAT=1 \
                OMNIVLA_PYTHON=/home/mifcom2/miniconda3/envs/omnivla/bin/python \
                "${WORKSPACE_ROOT}/src/vln_adapters/omnivla_adapter/run_inference_server.sh" \
                >"${log}" 2>&1 &
            ;;
        omnivla_native)
            env OMNIVLA_ALLOW_BLACKWELL_COMPAT=1 \
                OMNIVLA_PYTHON=/home/mifcom2/miniconda3/envs/omnivla/bin/python \
                "${WORKSPACE_ROOT}/src/vln_adapters/omnivla_adapter/run_inference_server.sh" \
                >"${log}" 2>&1 &
            ;;
        dualvln)
            "${WORKSPACE_ROOT}/src/vln_adapters/dualvln_adapter/run_inference_server.sh" \
                >"${log}" 2>&1 &
            ;;
        navila)
            "${WORKSPACE_ROOT}/src/vln_adapters/navila_adapter/run_inference_server.sh" \
                >"${log}" 2>&1 &
            ;;
        uninavid)
            "${WORKSPACE_ROOT}/src/vln_adapters/uninavid_adapter/run_inference_server.sh" \
                >"${log}" 2>&1 &
            ;;
        *)
            echo "Unknown model: ${model}" >&2
            return 2
            ;;
    esac
    SERVER_PID=$!
}

wait_for_server() {
    local model="$1"
    local port="${PORTS[${model}]}"
    local deadline=$((SECONDS + 600))
    while (( SECONDS < deadline )); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[ALL85] ${model} server exited before readiness" >&2
            return 1
        fi
        if curl --fail --silent --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; then
            echo "[ALL85] ${model} server ready on port ${port}"
            return 0
        fi
        sleep 2
    done
    echo "[ALL85] ${model} server readiness timed out" >&2
    return 1
}

mkdir -p "${WORKSPACE_ROOT}/logs"
# Intentionally no global flock lock.  The operator requested manual control of
# concurrent runs; GPU/process conflicts are therefore the operator's responsibility.
cd "${WORKSPACE_ROOT}"
set +u
source /opt/ros/jazzy/setup.bash
source "${WORKSPACE_ROOT}/install/setup.bash"
set -u
if [[ ! "${PEDESTRIAN_CAP}" =~ ^[0-9]+$ ]]; then
    echo "DYNANAV_PEDESTRIAN_CAP must be a non-negative integer" >&2
    exit 2
fi
case "${NO_PEDESTRIANS}" in
    0|1) ;;
    *) echo "DYNANAV_NO_PEDESTRIANS must be 0 or 1" >&2; exit 2 ;;
esac
if ! [[ "${TIMEOUT_SCALE}" =~ ^[0-9]+([.][0-9]+)?$ ]] || [[ "${TIMEOUT_SCALE}" == "0" || "${TIMEOUT_SCALE}" == "0.0" ]]; then
    echo "DYNANAV_TIMEOUT_SCALE must be a positive number" >&2
    exit 2
fi
if ! [[ "${STARTUP_GRACE}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "DYNANAV_STARTUP_GRACE must be a non-negative number" >&2
    exit 2
fi
export DYNANAV_PEDESTRIAN_CAP="${PEDESTRIAN_CAP}"
export DYNANAV_NO_PEDESTRIANS="${NO_PEDESTRIANS}"
case "${ASSET_VERSION}" in
    5.0|5.1) ;;
    *) echo "DYNANAV_ASSET_VERSION must be 5.0 or 5.1" >&2; exit 2 ;;
esac
export DYNANAV_ASSET_VERSION="${ASSET_VERSION}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_MODULE_LOADING="${CUDA_MODULE_LOADING:-LAZY}"
echo "[ALL85] benchmark_standard=$([[ \"${OFFICIAL_BENCHMARK}\" == \"1\" ]] && echo ticvla_dynanav_official || echo workspace_smoke) execution=$([[ \"${MODEL_SPECIFIC_EXECUTION}\" == \"1\" ]] && echo native || echo fair)"
echo "[ALL85] GPU safeguards: pedestrian_cap=${DYNANAV_PEDESTRIAN_CAP}, no_pedestrians=${DYNANAV_NO_PEDESTRIANS}, asset_version=${DYNANAV_ASSET_VERSION}, PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
exclude_args=()
if [[ -n "${EXCLUDE_SCENES}" ]]; then
    exclude_args+=(--exclude-scenes "${EXCLUDE_SCENES}")
fi
echo "[ALL85] scene filter: exclude=${EXCLUDE_SCENES:-<none>}"

for model in "${MODELS[@]}"; do
    echo "[ALL85] START model=${model} experiment=${EXPERIMENT}"
    if ! start_server "${model}" || ! wait_for_server "${model}"; then
        FAILED_MODELS+=("${model}:server")
        cleanup_server
        [[ "${CONTINUE_ON_ERROR}" == "1" ]] || break
        continue
    fi

    benchmark_args=(
        --model "${model}"
        --experiment "${EXPERIMENT}"
        --episodes "${EPISODES}"
        --max-attempts 2
        --timeout-scale "${TIMEOUT_SCALE}"
        --startup-grace "${STARTUP_GRACE}"
    )
    [[ "${MODEL_SPECIFIC_EXECUTION}" == "1" ]] && benchmark_args+=(--execution-profile native)
    [[ "${HEADLESS}" == "1" ]] && benchmark_args+=(--headless)
    benchmark_args+=("${exclude_args[@]}")
    [[ "${RESUME}" == "1" ]] && benchmark_args+=(--resume)
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] && benchmark_args+=(--continue-on-error)

    if ros2 run vln_evaluation benchmark_runner "${benchmark_args[@]}" \
        >"${WORKSPACE_ROOT}/logs/${model}_${EXPERIMENT}.log" 2>&1; then
        echo "[ALL85] COMPLETE model=${model}"
    else
        echo "[ALL85] FAILED model=${model}; see logs/${model}_${EXPERIMENT}.log" >&2
        FAILED_MODELS+=("${model}:benchmark")
    fi
    cleanup_server
    if (( ${#FAILED_MODELS[@]} > 0 )) && [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
        break
    fi
    sleep 5
done

if (( ${#FAILED_MODELS[@]} > 0 )); then
    echo "[ALL85] FINISHED WITH FAILURES: ${FAILED_MODELS[*]}" >&2
    exit 1
fi
echo "[ALL85] FINISHED all models"
