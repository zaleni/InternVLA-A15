#!/usr/bin/env bash
set -euo pipefail

# Evaluate the two 60k RoboTwin runs one after another on the same 8 GPUs.
# Override the first two arguments when evaluating another checkpoint pair.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DEFAULT_AHA_CKPT="${REPO_ROOT}/outputs/internvla_a1_5/job-aha-wam-copy-master-0.job-aha-wam-copy-23456-internvla_a1_5-robotwin-all-27500-abs-finetune-aha-teacher/checkpoints/060000/pretrained_model"
DEFAULT_WAN22_CKPT="${REPO_ROOT}/outputs/internvla_a1_5/job-aha-wam-baseline-copy-master-0.job-aha-wam-baseline-copy-23456-internvla_a1_5-robotwin-all-27500-abs-finetune-wan22-baseline/checkpoints/060000/pretrained_model"

AHA_CKPT="${1:-${AHA_CKPT:-${DEFAULT_AHA_CKPT}}}"
WAN22_CKPT="${2:-${WAN22_CKPT:-${DEFAULT_WAN22_CKPT}}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs/robotwin_eval/a15_60k_compare}"

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
export NUM_EPISODES="${NUM_EPISODES:-100}"
export SEED="${SEED:-42}"
export ACTION_MODE="${ACTION_MODE:-abs}"
export DTYPE="${DTYPE:-bfloat16}"
export INFER_HORIZON="${INFER_HORIZON:-20}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
export START_TASK_IDX="${START_TASK_IDX:-0}"
export TASK_COUNT="${TASK_COUNT:-50}"
export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${REPO_ROOT}/third_party/RoboTwin}"

if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
    if [[ -d "/data/jjhao/InternVLA-muon/third_party/RoboTwin" ]]; then
        export ROBOTWIN_ROOT="/data/jjhao/InternVLA-muon/third_party/RoboTwin"
    else
        echo "RoboTwin checkout not found: ${ROBOTWIN_ROOT}" >&2
        exit 1
    fi
fi

mkdir -p "${OUTPUT_ROOT}"

echo "[compare] AHA checkpoint: ${AHA_CKPT}"
echo "[compare] Wan2.2 checkpoint: ${WAN22_CKPT}"
echo "[compare] output root: ${OUTPUT_ROOT}"
echo "[compare] GPUs/tasks/episodes: ${GPU_IDS}/${TASK_COUNT}/${NUM_EPISODES}"

echo "[compare] evaluating AHA"
bash "${SCRIPT_DIR}/eval_all_aha.sh" "${AHA_CKPT}" "${OUTPUT_ROOT}/aha" \
    --task-config "${TASK_CONFIG}" \
    --start-task-idx "${START_TASK_IDX}" \
    --task-count "${TASK_COUNT}" \
    --num-episodes "${NUM_EPISODES}" \
    --seed "${SEED}" \
    --action-mode "${ACTION_MODE}" \
    --dtype "${DTYPE}" \
    --infer-horizon "${INFER_HORIZON}" \
    --inference-backend "${INFERENCE_BACKEND}"

echo "[compare] evaluating Wan2.2 baseline"
bash "${SCRIPT_DIR}/eval_all_wan22_baseline.sh" "${WAN22_CKPT}" "${OUTPUT_ROOT}/wan22" \
    --task-config "${TASK_CONFIG}" \
    --start-task-idx "${START_TASK_IDX}" \
    --task-count "${TASK_COUNT}" \
    --num-episodes "${NUM_EPISODES}" \
    --seed "${SEED}" \
    --action-mode "${ACTION_MODE}" \
    --dtype "${DTYPE}" \
    --infer-horizon "${INFER_HORIZON}" \
    --inference-backend "${INFERENCE_BACKEND}"

echo "[compare] completed; summaries:"
echo "  ${OUTPUT_ROOT}/aha/summary.txt"
echo "  ${OUTPUT_ROOT}/wan22/summary.txt"
