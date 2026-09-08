#!/usr/bin/env bash
set -euo pipefail

# RoboTwin ablation: keep the normal VLM/FAST-token loss and video setup, but
# initialize the InternVLA policy without loading an InternVLA pretrained ckpt.
# The Qwen VLM base is still loaded through INTERNVLA_VLM_PATH.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export PRETRAINED_PATH=""
export USE_SUBTASK_ANNOTATIONS=false

# Official RoboTwin 2.0 scale: 24 GPUs, batch 16/GPU, 100k steps.
export PROC_PER_NODE="${PROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
export NODE_COUNT="${NODE_COUNT:-${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-3}}}"
export BATCH_SIZE="${BATCH_SIZE:-16}"
export STEPS="${STEPS:-100000}"
export OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
export SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-2000}"
export SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-140000}"
export SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-5e-6}"
export SAVE_FREQ="${SAVE_FREQ:-50000}"
export RUN_TAG="${RUN_TAG:-no-internvla-pretrain}"
export GRADIENT_CHECKPOINTING=true

echo "[RoboTwin ablation] InternVLA pretrained checkpoint: disabled"
echo "[RoboTwin ablation] Qwen VLM base: ${INTERNVLA_VLM_PATH:-default}"
echo "[RoboTwin ablation] nodes/processes per node/batch per GPU/steps: ${NODE_COUNT}/${PROC_PER_NODE}/${BATCH_SIZE}/${STEPS}"

exec bash "${SCRIPT_DIR}/internvla_a15_finetune_robotwin.sh"
