#!/usr/bin/env bash
set -euo pipefail

# Fine-tune InternVLA-A1.5 on RoboTwin with a frozen AHA-WAM Video Expert.
# Dataset/distributed/optimizer overrides are forwarded to the base launcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash launch/internvla_a15_finetune_robotwin_aha_teacher.sh

Optional checkpoint overrides:
  bash launch/internvla_a15_finetune_robotwin_aha_teacher.sh /path/to/aha_video_expert.pt

  AHA_TEACHER_CHECKPOINT=/path/to/aha_video_expert.pt \
    bash launch/internvla_a15_finetune_robotwin_aha_teacher.sh

Useful overrides:
  DATASET_ROOT=/data/jjhao/data
  DATASET_VARIANT='aloha-agilex*'         # default: all 27,500 episodes
  DATASET_REPO_ID="robotwin/task/aloha-agilex_clean_50 ..."
  ACTION_TYPE=abs                         # abs or delta
  STATS_PATH=/path/to/aggregated/stats.json
  PROC_PER_NODE=8 NODE_COUNT=2 BATCH_SIZE=8 STEPS=60000
  VIDEO_LOSS_WEIGHT=1 FREEZE_LEARNABLE_TOKENS=false DRY_RUN=true

The default checkpoint is:
  /data/jjhao/data/model/AHA-WAM-RoboTwin2.0/robotwin_ahawam_video_expert.pt

The checkpoint may be a raw AHA-WAM checkpoint or the compact Video Expert
checkpoint produced by util_scripts/extract_aha_wan_video_expert.py. The compact
checkpoint is recommended because every distributed worker loads it.
EOF
}

if (( $# > 1 )); then
    usage >&2
    exit 2
fi
if (( $# == 1 )); then
    if [[ -n "${AHA_TEACHER_CHECKPOINT:-}" && "${AHA_TEACHER_CHECKPOINT}" != "$1" ]]; then
        echo "AHA checkpoint was provided twice with different values." >&2
        exit 2
    fi
    AHA_TEACHER_CHECKPOINT="$1"
fi
AHA_TEACHER_CHECKPOINT="${AHA_TEACHER_CHECKPOINT:-/data/jjhao/data/model/AHA-WAM-RoboTwin2.0/robotwin_ahawam_video_expert.pt}"
if [[ ! -f "${AHA_TEACHER_CHECKPOINT}" ]]; then
    echo "AHA teacher checkpoint not found: ${AHA_TEACHER_CHECKPOINT}" >&2
    exit 1
fi
AHA_TEACHER_CHECKPOINT="$(realpath "${AHA_TEACHER_CHECKPOINT}")"

export HF_HOME="${HF_HOME:-/data/jjhao/huggingface}"
export WAN_BASE_PATH="${WAN_BASE_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"
export WAN_CHECKPOINT_PATH="${AHA_TEACHER_CHECKPOINT}"
export WAN_CONFIG_PATH="${WAN_CONFIG_PATH:-${WAN_BASE_PATH}}"
export WAN_VAE_PATH="${WAN_VAE_PATH:-${WAN_BASE_PATH}/Wan2.2_VAE.pth}"

if [[ ! -f "${WAN_CONFIG_PATH}/config.json" ]]; then
    echo "WAN config not found: ${WAN_CONFIG_PATH}/config.json" >&2
    exit 1
fi
if [[ ! -f "${WAN_VAE_PATH}" ]]; then
    echo "WAN VAE not found: ${WAN_VAE_PATH}" >&2
    exit 1
fi
export WAN_CONFIG_PATH="$(realpath "${WAN_CONFIG_PATH}")"
export WAN_VAE_PATH="$(realpath "${WAN_VAE_PATH}")"

# AHA supplies only a differentiable, frozen video teacher. Video loss still
# updates the trainable student path through the teacher, while neither action
# loss nor video loss updates AHA parameters.
export WAN_TEACHER_MODE=aha_wam
export ACTION_LOSS_ONLY=false
export FREEZE_WAN_DIT=true
# The AHA Video Expert has a different learned context distribution from the
# original WAN teacher. Train the foresight tokens and both context projections
# so the student can adapt, while the AHA DiT itself remains frozen.
export FREEZE_LEARNABLE_TOKENS=false
export VIDEO_LOSS_WEIGHT="${VIDEO_LOSS_WEIGHT:-1}"
export PROC_PER_NODE="${PROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
export NODE_COUNT="${NODE_COUNT:-${SENSECORE_PYTORCH_NNODES:-2}}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export STEPS="${STEPS:-60000}"
export USE_SUBTASK_ANNOTATIONS=false
export RUN_TAG="${RUN_TAG:-aha-teacher}"

echo "[AHA Robotwin] checkpoint: ${WAN_CHECKPOINT_PATH}"
echo "[AHA Robotwin] WAN config: ${WAN_CONFIG_PATH}"
echo "[AHA Robotwin] WAN VAE: ${WAN_VAE_PATH}"
echo "[AHA Robotwin] video loss weight: ${VIDEO_LOSS_WEIGHT}"
echo "[AHA Robotwin] nodes/processes per node/batch per GPU/steps: ${NODE_COUNT}/${PROC_PER_NODE}/${BATCH_SIZE}/${STEPS}"
echo "[AHA Robotwin] foresight tokens frozen: ${FREEZE_LEARNABLE_TOKENS}"
echo "[AHA Robotwin] subtask annotations: ${USE_SUBTASK_ANNOTATIONS}"

exec bash "${SCRIPT_DIR}/internvla_a15_finetune_robotwin.sh"
