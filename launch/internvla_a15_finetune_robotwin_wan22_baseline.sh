#!/usr/bin/env bash
set -euo pipefail

# Paired RoboTwin baseline: keep the original frozen Wan2.2 teacher behavior
# and the official A1.5 fine-tuning hyperparameters. The AHA comparison wrapper
# calls the same base launcher, so only teacher-specific settings differ.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if (( $# != 0 )); then
    echo "Usage: bash launch/internvla_a15_finetune_robotwin_wan22_baseline.sh" >&2
    exit 2
fi

export HF_HOME="${HF_HOME:-/data/jjhao/huggingface}"
export WAN_BASE_PATH="${WAN_BASE_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
export WAN_CHECKPOINT_PATH="${WAN_CHECKPOINT_PATH:-${WAN_BASE_PATH}}"
export WAN_CONFIG_PATH="${WAN_CONFIG_PATH:-${WAN_BASE_PATH}}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-${WAN_BASE_PATH}/Wan2.2_VAE.pth}"

# Original Wan2.2 auxiliary-training behavior.
export WAN_TEACHER_MODE=wan22
export ACTION_LOSS_ONLY=false
export FREEZE_WAN_DIT=true
export FREEZE_LEARNABLE_TOKENS=true
export VIDEO_LOSS_WEIGHT="${VIDEO_LOSS_WEIGHT:-1}"

# Paired experiment defaults requested for this run.
export PROC_PER_NODE="${PROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
export NODE_COUNT="${NODE_COUNT:-${SENSECORE_PYTORCH_NNODES:-2}}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export STEPS="${STEPS:-60000}"
export USE_SUBTASK_ANNOTATIONS=false
export RUN_TAG="${RUN_TAG:-wan22-baseline}"

echo "[Wan2.2 baseline] teacher: ${WAN_CHECKPOINT_PATH}"
echo "[Wan2.2 baseline] student: ${PRETRAINED_PATH:-/data/jjhao/data/model/a1.5_0600000_pretrained_model}"
echo "[Wan2.2 baseline] nodes/processes per node/batch per GPU/steps: ${NODE_COUNT}/${PROC_PER_NODE}/${BATCH_SIZE}/${STEPS}"
echo "[Wan2.2 baseline] foresight tokens frozen: ${FREEZE_LEARNABLE_TOKENS}"
echo "[Wan2.2 baseline] subtask annotations: ${USE_SUBTASK_ANNOTATIONS}"

exec bash "${SCRIPT_DIR}/internvla_a15_finetune_robotwin.sh"
