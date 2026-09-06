#!/usr/bin/env bash
set -euo pipefail

# One independent 8-GPU RoboTwin evaluation group.
# Usage:
#   bash evaluation/RoboTwin/eval_8gpu_group.sh aha demo_clean
#   bash evaluation/RoboTwin/eval_8gpu_group.sh wan22 demo_randomized

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MODEL="${1:-${MODEL:-}}"
TASK_CONFIG="${2:-${TASK_CONFIG:-}}"
if [[ "${MODEL}" != "aha" && "${MODEL}" != "wan22" ]]; then
    echo "MODEL must be aha or wan22." >&2
    echo "Usage: bash evaluation/RoboTwin/eval_8gpu_group.sh aha|wan22 demo_clean|demo_randomized [output_root]" >&2
    exit 2
fi
if [[ "${TASK_CONFIG}" != "demo_clean" && "${TASK_CONFIG}" != "demo_randomized" ]]; then
    echo "TASK_CONFIG must be demo_clean or demo_randomized." >&2
    exit 2
fi

case "${MODEL}" in
    aha)
        CHECKPOINT="${AHA_CKPT:-/mnt/data/jiangjiahao/data/model/zaleni/AHA-A1}"
        MODEL_TAG="aha"
        ;;
    wan22)
        CHECKPOINT="${WAN22_CKPT:-/mnt/data/jiangjiahao/data/model/zaleni/Internvla-A1_5-Robotwin-60k}"
        MODEL_TAG="wan22"
        ;;
esac

OUTPUT_ROOT="${3:-${OUTPUT_ROOT:-${REPO_ROOT}/outputs/robotwin_eval/${MODEL_TAG}_60k/${TASK_CONFIG}}}"

export TASK_CONFIG
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-3}"
export NUM_EPISODES="${NUM_EPISODES:-100}"
export SEED="${SEED:-42}"
export STATS_KEY="${STATS_KEY:-aloha}"
export RESIZE_SIZE="${RESIZE_SIZE:-224}"
export ACTION_MODE="${ACTION_MODE:-abs}"
export DTYPE="${DTYPE:-bfloat16}"
export INFER_HORIZON="${INFER_HORIZON:-25}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
export START_TASK_IDX="${START_TASK_IDX:-0}"
export TASK_COUNT="${TASK_COUNT:-50}"
export MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
export RETRY_DELAY="${RETRY_DELAY:-10}"
export STAGGER_SECONDS="${STAGGER_SECONDS:-5}"
export FORCE_RERUN="${FORCE_RERUN:-false}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/third_party/RoboTwin}"

if [[ ! -f "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml" ]]; then
    for candidate in \
        /mnt/data/jiangjiahao/InternVLA-muon/third_party/RoboTwin \
        /data/jjhao/InternVLA-muon/third_party/RoboTwin; do
        if [[ -f "${candidate}/task_config/${TASK_CONFIG}.yml" ]]; then
            export ROBOTWIN_ROOT="${candidate}"
            break
        fi
    done
fi

echo "[RoboTwin group] model: ${MODEL_TAG}"
echo "[RoboTwin group] checkpoint: ${CHECKPOINT}"
echo "[RoboTwin group] task config: ${TASK_CONFIG}"
echo "[RoboTwin group] output: ${OUTPUT_ROOT}"
echo "[RoboTwin group] GPUs: ${GPU_IDS}; tasks: ${TASK_COUNT}; episodes/task: ${NUM_EPISODES}"
echo "[RoboTwin group] dtype/action/horizon: ${DTYPE}/${ACTION_MODE}/${INFER_HORIZON}"

if [[ "${MODEL}" == "aha" ]]; then
    exec bash "${SCRIPT_DIR}/eval_all_aha.sh" "${CHECKPOINT}" "${OUTPUT_ROOT}"
else
    exec bash "${SCRIPT_DIR}/eval_all_wan22_baseline.sh" "${CHECKPOINT}" "${OUTPUT_ROOT}"
fi
