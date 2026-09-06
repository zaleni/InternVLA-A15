#!/usr/bin/env bash
set -euo pipefail

# Eight-GPU RoboTwin evaluation for the standard Wan2.2 baseline checkpoint.
# The checkpoint path may be supplied as the first argument or PRETRAINED_CKPT.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${1:-${PRETRAINED_CKPT:-}}"
if [[ -z "${CHECKPOINT}" ]]; then
    echo "Usage: bash evaluation/RoboTwin/eval_all_wan22_baseline.sh /path/to/pretrained_model" >&2
    exit 2
fi
OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-${PWD}/outputs/robotwin/a15_wan22_baseline}}"

export RUN_TAG="${RUN_TAG:-wan22-baseline}"
export TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export DTYPE="${DTYPE:-bfloat16}"
export ACTION_MODE="${ACTION_MODE:-abs}"
export INFER_HORIZON="${INFER_HORIZON:-25}"
export GENERATE_SUBTASK=false

exec bash "${SCRIPT_DIR}/eval_all.sh" "${CHECKPOINT}" "${OUTPUT_ROOT}" "${@:3}"
