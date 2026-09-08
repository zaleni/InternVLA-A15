#!/usr/bin/env bash
set -euo pipefail

# RoboTwin ablation: remove the VLM/FAST-token CE loss while keeping the FAST
# token representation in the input. This isolates the effect of the VLM loss;
# action flow-matching and WAN/AHA video losses remain enabled.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ENABLE_VQA_LOSS=false
export USE_FAST_ACTION_TOKENS=true
export USE_SUBTASK_ANNOTATIONS=false

# Official RoboTwin 2.0 scale from the paper: 24 GPUs, batch 16/GPU, 100k steps.
export PROC_PER_NODE="${PROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
export NODE_COUNT="${NODE_COUNT:-${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-3}}}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export STEPS="${STEPS:-100000}"
export OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-2000}"
export SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-140000}"
export SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-5e-6}"
export SAVE_FREQ="${SAVE_FREQ:-50000}"
export RUN_TAG="${RUN_TAG:-no-vlm-fast-loss}"
export GRADIENT_CHECKPOINTING=true

echo "[RoboTwin ablation] VLM/FAST loss: disabled (enable_vqa_loss=${ENABLE_VQA_LOSS})"
echo "[RoboTwin ablation] FAST tokens in input: ${USE_FAST_ACTION_TOKENS}"
echo "[RoboTwin ablation] nodes/processes per node/batch per GPU/steps: ${NODE_COUNT}/${PROC_PER_NODE}/${BATCH_SIZE}/${STEPS}"

exec bash "${SCRIPT_DIR}/internvla_a15_finetune_robotwin.sh"
