#!/usr/bin/env bash
set -euo pipefail

# Eight-GPU RoboTwin evaluation for an InternVLA-A1.5 checkpoint trained with
# AHA-WAM as the video teacher. AHA is not loaded at inference time.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="${1:-${PRETRAINED_CKPT:-}}"
if [[ -z "${CHECKPOINT}" ]]; then
    echo "Usage: bash evaluation/RoboTwin/eval_all_aha.sh /path/to/pretrained_model" >&2
    exit 2
fi
OUTPUT_ROOT="${2:-${OUTPUT_ROOT:-${PWD}/outputs/robotwin/a15_aha}}"

export RUN_TAG="${RUN_TAG:-aha-teacher}"
export TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export DTYPE="${DTYPE:-bfloat16}"
export ACTION_MODE="${ACTION_MODE:-abs}"
export INFER_HORIZON="${INFER_HORIZON:-25}"
export GENERATE_SUBTASK=false

exec bash "${SCRIPT_DIR}/eval_all.sh" "${CHECKPOINT}" "${OUTPUT_ROOT}" "${@:3}"
