#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

# Dataset config. DATASET_REPO_ID may contain whitespace-separated repo IDs.
DEFAULT_DATASET_ROOT="/data/datasets/internvla_data/arx_acone/Pour_liquid_from_beaker_into_Erlenmeyer_flask"
DATASET_ROOT="${DATASET_ROOT:-${DEFAULT_DATASET_ROOT}}"
DEFAULT_DATASET_NAME="$(basename "${DATASET_ROOT}")"
DATASET_REPO_ID="${DATASET_REPO_ID:-arx_acone/${DEFAULT_DATASET_NAME}}"
DATASET_TAG="${DATASET_TAG:-${DEFAULT_DATASET_NAME}}"
SAFE_DATASET_NAME="${DATASET_TAG//[^a-zA-Z0-9._-]/_}"
read -r -a DATASET_REPO_IDS <<< "${DATASET_REPO_ID}"
if (( ${#DATASET_REPO_IDS[@]} == 0 )); then
    echo "DATASET_REPO_ID must contain at least one repo ID." >&2
    exit 2
fi

# Local runtime and assets.
CONDA_ROOT="${CONDA_ROOT:-/data/jjhao/miniconda3}"
CONDA_ENV="${CONDA_ENV:-internvla_a1_5}"
echo "[InternVLA] Activating conda environment: ${CONDA_ENV}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export HF_HOME="${HF_HOME:-/data/jjhao/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/internvla_hf_datasets_cache}"
mkdir -p "${HF_DATASETS_CACHE}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export INTERNVLA_VLM_PATH="${INTERNVLA_VLM_PATH:-/data/jjhao/data/model/Qwen3.5-2B-Action}"
if [[ -z "${INTERNVLA_FAST_TOKENIZER_PATH:-}" ]]; then
    FAST_CACHE_DIR="${HF_HOME}/hub/models--physical-intelligence--fast"
    FAST_REVISION="$(<"${FAST_CACHE_DIR}/refs/main")"
    export INTERNVLA_FAST_TOKENIZER_PATH="${FAST_CACHE_DIR}/snapshots/${FAST_REVISION}"
fi
echo "[InternVLA] Dataset: ${DATASET_ROOT}"
echo "[InternVLA] Repo IDs: ${DATASET_REPO_ID}"
echo "[InternVLA] Qwen:   ${INTERNVLA_VLM_PATH}"
echo "[InternVLA] FAST:    ${INTERNVLA_FAST_TOKENIZER_PATH}"
echo "[InternVLA] Datasets cache: ${HF_DATASETS_CACHE}"

PRETRAINED_PATH="${PRETRAINED_PATH:-/data/jjhao/data/model/a1.5_0600000_pretrained_model}"
DEFAULT_STATS_PATH="${HF_HOME}/lerobot/stats/delta/${DATASET_REPO_ID}/stats.json"
STATS_PATH="${STATS_PATH:-${DEFAULT_STATS_PATH}}"
WAN_PATH="${WAN_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}"

# Recompute the 50-step delta-action statistics only when they are absent.
if [[ ! -f "${STATS_PATH}" ]]; then
    if (( ${#DATASET_REPO_IDS[@]} != 1 )); then
        echo "Aggregated delta stats are missing for this multi-repo run: ${STATS_PATH}" >&2
        echo "Compute them with compute_norm_stats_multi.py, then set STATS_PATH explicitly." >&2
        exit 2
    fi
    python util_scripts/compute_norm_stats_single.py \
        --action_mode delta \
        --chunk_size 50 \
        --repo_id "${DATASET_REPO_IDS[0]}" \
        --root "${DATASET_ROOT}" \
        --output_dir "${HF_HOME}/lerobot/stats"
fi

DATASET_INFO_PATHS=()
if (( ${#DATASET_REPO_IDS[@]} == 1 )) && [[ -f "${DATASET_ROOT}/meta/info.json" ]]; then
    DATASET_INFO_PATHS+=("${DATASET_ROOT}/meta/info.json")
else
    for repo_id in "${DATASET_REPO_IDS[@]}"; do
        DATASET_INFO_PATHS+=("${DATASET_ROOT}/${repo_id}/meta/info.json")
    done
fi

for required_path in \
    "${DATASET_INFO_PATHS[@]}" \
    "${PRETRAINED_PATH}/model.safetensors" \
    "${STATS_PATH}" \
    "${INTERNVLA_VLM_PATH}/config.json" \
    "${INTERNVLA_FAST_TOKENIZER_PATH}/tokenizer.json"; do
    if [[ ! -f "${required_path}" ]]; then
        echo "Required file not found: ${required_path}" >&2
        exit 1
    fi
done

ACTION_LOSS_ONLY="${ACTION_LOSS_ONLY:-true}"
if [[ "${ACTION_LOSS_ONLY}" != "true" ]]; then
    for required_path in \
        "${WAN_PATH}/config.json" \
        "${WAN_PATH}/Wan2.2_VAE.pth"; do
        if [[ ! -f "${required_path}" ]]; then
            echo "Required WAN file not found: ${required_path}" >&2
            exit 1
        fi
    done
fi

PROC_PER_NODE="${PROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
NODE_COUNT="${NODE_COUNT:-${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-2}}}"
NODE_RANK="${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${RANK:-}}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-}"

if [[ -z "${NODE_RANK}" ]]; then
    echo "NODE_RANK is missing; refusing to start multi-node training." >&2
    exit 2
fi

if [[ -z "${MASTER_ADDR}" || -z "${MASTER_PORT}" ]]; then
    echo "MASTER_ADDR or MASTER_PORT is missing." >&2
    exit 2
fi

NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))

echo "Distributed config: node_rank=${NODE_RANK}, nodes=${NODE_COUNT}, gpu/node=${PROC_PER_NODE}, master=${MASTER_ADDR}:${MASTER_PORT}"

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-60000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-12}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"

GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
if (( GPU_COUNT == 0 )); then
    echo "No CUDA GPU is visible; refusing to load the 15 GB checkpoint on CPU." >&2
    exit 2
fi
if (( PROC_PER_NODE > GPU_COUNT )); then
    echo "PROC_PER_NODE=${PROC_PER_NODE}, but only ${GPU_COUNT} CUDA GPUs are visible." >&2
    exit 2
fi

JOB_NAME="${JOB_NAME:-${SENSECORE_JOB_NAME:-$(date +'%Y_%m_%d_%H_%M_%S')}-internvla_a1_5-arx_acone-${SAFE_DATASET_NAME}-delta-finetune}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/internvla_a1_5/${JOB_NAME}}"
TASK_LOG_DIR="${TASK_LOG_DIR:-${PROJ_ROOT}/outputs/internvla_a1_5/logs/${SAFE_DATASET_NAME}}"
TRAIN_LOG="${TRAIN_LOG:-${TASK_LOG_DIR}/${JOB_NAME}.node-${NODE_RANK}.log}"
mkdir -p "${TASK_LOG_DIR}"
ln -sfn "$(basename "${TRAIN_LOG}")" "${TASK_LOG_DIR}/latest-node-${NODE_RANK}.log"
if (( NODE_RANK == 0 )); then
    ln -sfn "$(basename "${TRAIN_LOG}")" "${TASK_LOG_DIR}/latest.log"
fi

ACCELERATE_ARGS=(
    --num_processes="${NUM_PROCESSES}"
    --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}"
    --main_process_port="${MASTER_PORT}"
)
if (( NUM_PROCESSES > 1 )); then
    ACCELERATE_ARGS=(--multi_gpu "${ACCELERATE_ARGS[@]}")
fi

echo "Dataset root: ${DATASET_ROOT}"
echo "Delta stats:  ${STATS_PATH}"
echo "Checkpoint:   ${PRETRAINED_PATH}"
echo "Output:       ${OUTPUT_DIR}"
echo "Training log: ${TRAIN_LOG}"
echo "Processes:    ${NUM_PROCESSES}; batch/GPU: ${BATCH_SIZE}"

accelerate launch "${ACCELERATE_ARGS[@]}" src/lerobot/scripts/lerobot_train.py \
    --output_dir="${OUTPUT_DIR}" \
    --job_name="${JOB_NAME}" \
    --num_workers="${NUM_WORKERS}" \
    --policy.type=internvla_a1_5 \
    --policy.repo_id=lerobot_lab/internvla_a1_5 \
    --policy.pretrained_path="${PRETRAINED_PATH}" \
    --policy.vlm_model_name_or_path="${INTERNVLA_VLM_PATH}" \
    --policy.wan_checkpoint_path="${WAN_PATH}" \
    --policy.wan_config_path="${WAN_PATH}" \
    --policy.vae_path="${WAN_PATH}/Wan2.2_VAE.pth" \
    --policy.push_to_hub=false \
    --policy.dtype=bfloat16 \
    --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}" \
    --policy.optimizer_lr=5e-5 \
    --policy.scheduler_warmup_steps=2000 \
    --policy.scheduler_decay_steps="${STEPS}" \
    --policy.scheduler_decay_lr=5e-6 \
    --policy.freeze_vision_encoder=false \
    --policy.train_expert_only=false \
    --policy.enable_vqa_loss=true \
    --policy.tokenize_state=true \
    --policy.knowledge_insulation=false \
    --policy.video_loss_only=false \
    --policy.video_loss_weight=1 \
    --policy.action_loss_only="${ACTION_LOSS_ONLY}" \
    --policy.freeze_learnable_tokens=true \
    --policy.num_learnable_tokens=50 \
    --dataset.type=internvla_a1_5 \
    --dataset.repo_id="${DATASET_REPO_ID}" \
    --dataset.root="${DATASET_ROOT}" \
    --dataset.action_mode=delta \
    --dataset.use_external_stats=true \
    --dataset.external_stats_path="${STATS_PATH}" \
    --dataset.video_backend=pyav \
    --dataset.dist_loading=false \
    --dataset.tokenize_state=true \
    --dataset.use_fast_action_tokens=true \
    --seed=42 \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --save_freq="${SAVE_FREQ}" \
    --log_freq="${LOG_FREQ}" \
    --wandb.enable="${WANDB_ENABLE}" \
    --wandb.project=internvla_a1_5 \
    --wandb.mode=offline 2>&1 | tee "${TRAIN_LOG}"
