#!/usr/bin/env bash
set -euo pipefail

# Fine-tune InternVLA-A1.5 on AC-One with a frozen AHA-WAM Video Expert as
# the video auxiliary teacher. All dataset, distributed, and optimization
# options are inherited from internvla_a15_finetune_acone.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<'EOF'
Usage:
  bash launch/internvla_a15_finetune_acone_aha_teacher.sh /path/to/aha_video_expert.pt

or:
  AHA_TEACHER_CHECKPOINT=/path/to/aha_video_expert.pt \
    bash launch/internvla_a15_finetune_acone_aha_teacher.sh

The checkpoint may be either a raw AHA-WAM training checkpoint or the compact
checkpoint produced by util_scripts/extract_aha_wan_video_expert.py. The compact
checkpoint is recommended for distributed training.
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

if [[ -z "${AHA_TEACHER_CHECKPOINT:-}" ]]; then
    usage >&2
    exit 2
fi
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

# Explicitly reproduce the forward semantics used to train AHA's Video Expert.
export WAN_TEACHER_MODE="aha_wam"

# This integration uses AHA only as a frozen video teacher. The InternVLA
# foresight tokens and their WAN context projection must adapt to that teacher.
export ACTION_LOSS_ONLY="false"
export FREEZE_WAN_DIT="true"
export FREEZE_LEARNABLE_TOKENS="false"
export VIDEO_LOSS_WEIGHT="${VIDEO_LOSS_WEIGHT:-1}"
export RUN_TAG="${RUN_TAG:-aha-teacher}"

echo "[AHA teacher] checkpoint: ${WAN_CHECKPOINT_PATH}"
echo "[AHA teacher] WAN config:  ${WAN_CONFIG_PATH}"
echo "[AHA teacher] WAN VAE:     ${WAN_VAE_PATH}"
echo "[AHA teacher] mode:        ${WAN_TEACHER_MODE}"
echo "[AHA teacher] video loss:  ${VIDEO_LOSS_WEIGHT}"

exec bash "${SCRIPT_DIR}/internvla_a15_finetune_acone.sh"
