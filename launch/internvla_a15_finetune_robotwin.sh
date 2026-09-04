#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

# Runtime and local model assets.
CONDA_ROOT="${CONDA_ROOT:-${_CONDA_ROOT:-/data/jjhao/miniconda3}}"
CONDA_ENV="${CONDA_ENV:-internvla_a1_5}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export HF_HOME="${HF_HOME:-/data/jjhao/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/internvla_hf_datasets_cache}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
mkdir -p "${HF_DATASETS_CACHE}"

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

export INTERNVLA_VLM_PATH="${INTERNVLA_VLM_PATH:-/data/jjhao/data/model/Qwen3.5-2B-Action}"
if [[ -z "${INTERNVLA_FAST_TOKENIZER_PATH:-}" ]]; then
    FAST_CACHE_DIR="${HF_HOME}/hub/models--physical-intelligence--fast"
    if [[ ! -f "${FAST_CACHE_DIR}/refs/main" ]]; then
        echo "FAST tokenizer revision not found: ${FAST_CACHE_DIR}/refs/main" >&2
        exit 1
    fi
    FAST_REVISION="$(<"${FAST_CACHE_DIR}/refs/main")"
    export INTERNVLA_FAST_TOKENIZER_PATH="${FAST_CACHE_DIR}/snapshots/${FAST_REVISION}"
fi

# Student checkpoint and frozen video teacher.
POLICY="internvla_a1_5"
PRETRAINED_PATH="${PRETRAINED_PATH:-/data/jjhao/data/model/a1.5_0600000_pretrained_model}"
WAN_BASE_PATH="${WAN_BASE_PATH:-${WAN_PATH:-${HF_HOME}/hub/Wan2.2-TI2V-5B}}"
WAN_CHECKPOINT_PATH="${WAN_CHECKPOINT_PATH:-${WAN_BASE_PATH}}"
WAN_CONFIG_PATH="${WAN_CONFIG_PATH:-${WAN_BASE_PATH}}"
WAN_VAE_PATH="${WAN_VAE_PATH:-${WAN_BASE_PATH}/Wan2.2_VAE.pth}"
WAN_TEACHER_MODE="${WAN_TEACHER_MODE:-auto}"

ACTION_LOSS_ONLY="${ACTION_LOSS_ONLY:-false}"
FREEZE_WAN_DIT="${FREEZE_WAN_DIT:-true}"
FREEZE_LEARNABLE_TOKENS="${FREEZE_LEARNABLE_TOKENS:-true}"
VIDEO_LOSS_WEIGHT="${VIDEO_LOSS_WEIGHT:-1}"
USE_SUBTASK_ANNOTATIONS="${USE_SUBTASK_ANNOTATIONS:-false}"

# Dataset. Match the official RoboTwin launcher: use both clean_50 and
# randomized_500 for all 50 tasks (100 repos / 27,500 episodes). Repo IDs are
# paths relative to DATASET_ROOT.
DATASET_ROOT="${DATASET_ROOT:-/data/jjhao/data}"
DATASET_VARIANT="${DATASET_VARIANT:-aloha-agilex*}"
if [[ -z "${DATASET_REPO_ID:-}" ]]; then
    ROBOTWIN_ROOT="${DATASET_ROOT}/robotwin"
    if [[ ! -d "${ROBOTWIN_ROOT}" ]]; then
        echo "RoboTwin dataset directory not found: ${ROBOTWIN_ROOT}" >&2
        exit 1
    fi
    DATASET_REPO_ID="$({
        find -L "${ROBOTWIN_ROOT}" -mindepth 2 -maxdepth 2 \
            -type d -name "${DATASET_VARIANT}" 2>/dev/null \
        | while read -r dataset_dir; do
            if [[ -f "${dataset_dir}/meta/info.json" && -d "${dataset_dir}/videos" ]]; then
                echo "${dataset_dir#"${DATASET_ROOT}/"}"
            fi
        done \
        | sort -u
    } | tr '\n' ' ' | sed 's/[[:space:]]*$//')"
fi
read -r -a DATASET_REPO_IDS <<< "${DATASET_REPO_ID}"
if (( ${#DATASET_REPO_IDS[@]} == 0 )); then
    echo "No RoboTwin datasets matched DATASET_VARIANT=${DATASET_VARIANT}." >&2
    exit 2
fi

ACTION_TYPE="${ACTION_TYPE:-abs}" # abs | delta
USE_EXTERNAL_STATS="${USE_EXTERNAL_STATS:-true}"
if [[ -z "${STATS_PATH:-}" ]]; then
    case "${DATASET_VARIANT}" in
        aloha-agilex_clean_50)
            STATS_TAG="robotwin_clean_all"
            ;;
        "aloha-agilex*"|aloha-agilex_randomized_500)
            STATS_TAG="robotwin_clean_randomized_all"
            ;;
        *)
            STATS_TAG="robotwin_${DATASET_VARIANT}"
            ;;
    esac
    STATS_PATH="${HF_HOME}/lerobot/stats/aloha/${ACTION_TYPE}/${STATS_TAG}/stats.json"
fi

DROP_INCOMPLETE_ACTION_CHUNKS="${DROP_INCOMPLETE_ACTION_CHUNKS:-false}"
DROP_INCOMPLETE_ACTION_CHUNK_REPO_IDS="${DROP_INCOMPLETE_ACTION_CHUNK_REPO_IDS:-}"

# Distributed settings follow the AC-One launcher: the multi-node job platform
# runs this same command on every node and injects rank/rendezvous variables.
PROC_PER_NODE="${PROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
NODE_COUNT="${NODE_COUNT:-${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-2}}}"
NODE_RANK="${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${RANK:-}}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-}"
RUN_ID="${RUN_ID:-${SENSECORE_JOB_NAME:-}}"

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

if [[ -z "${DIST_LOADING:-}" ]]; then
    if (( NUM_PROCESSES > 1 )); then
        DIST_LOADING=true
    else
        DIST_LOADING=false
    fi
fi

BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS="${STEPS:-60000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-200}"
MODULE_GRAD_NORM_FREQ="${MODULE_GRAD_NORM_FREQ:-0}"
NUM_WORKERS="${NUM_WORKERS:-8}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_MODE="${WANDB_MODE:-offline}"
DRY_RUN="${DRY_RUN:-false}"

for boolean_name in ACTION_LOSS_ONLY FREEZE_WAN_DIT FREEZE_LEARNABLE_TOKENS \
    USE_EXTERNAL_STATS USE_SUBTASK_ANNOTATIONS DROP_INCOMPLETE_ACTION_CHUNKS DIST_LOADING \
    GRADIENT_CHECKPOINTING WANDB_ENABLE DRY_RUN; do
    boolean_value="${!boolean_name}"
    if [[ "${boolean_value}" != "true" && "${boolean_value}" != "false" ]]; then
        echo "${boolean_name} must be true or false, got: ${boolean_value}" >&2
        exit 2
    fi
done
if [[ "${ACTION_TYPE}" != "abs" && "${ACTION_TYPE}" != "delta" ]]; then
    echo "ACTION_TYPE must be abs or delta, got: ${ACTION_TYPE}" >&2
    exit 2
fi
if (( NODE_RANK < 0 || NODE_RANK >= NODE_COUNT )); then
    echo "NODE_RANK=${NODE_RANK} must be in [0, $((NODE_COUNT - 1))]." >&2
    exit 2
fi

# Fail before starting one process per GPU if a shared asset is missing.
REQUIRED_FILES=(
    "${INTERNVLA_VLM_PATH}/config.json"
    "${INTERNVLA_FAST_TOKENIZER_PATH}/tokenizer.json"
)
if [[ -d "${PRETRAINED_PATH}" ]]; then
    REQUIRED_FILES+=("${PRETRAINED_PATH}/config.json" "${PRETRAINED_PATH}/model.safetensors")
elif [[ "${HF_HUB_OFFLINE}" == "1" ]]; then
    echo "PRETRAINED_PATH is not a local directory while HF_HUB_OFFLINE=1: ${PRETRAINED_PATH}" >&2
    exit 1
fi
if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
    REQUIRED_FILES+=("${STATS_PATH}")
fi
if [[ "${ACTION_LOSS_ONLY}" != "true" ]]; then
    if [[ ! -e "${WAN_CHECKPOINT_PATH}" ]]; then
        echo "WAN/AHA checkpoint not found: ${WAN_CHECKPOINT_PATH}" >&2
        exit 1
    fi
    REQUIRED_FILES+=("${WAN_CONFIG_PATH}/config.json" "${WAN_VAE_PATH}")
fi
for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done
TOTAL_EPISODES=0
for repo_id in "${DATASET_REPO_IDS[@]}"; do
    INFO_PATH="${DATASET_ROOT}/${repo_id}/meta/info.json"
    if [[ ! -f "${INFO_PATH}" ]]; then
        echo "Dataset metadata not found: ${INFO_PATH}" >&2
        exit 1
    fi
    if [[ ! -d "${DATASET_ROOT}/${repo_id}/videos" ]]; then
        echo "Dataset videos not found: ${DATASET_ROOT}/${repo_id}/videos" >&2
        exit 1
    fi
    EPISODE_COUNT="$(sed -n 's/^[[:space:]]*"total_episodes":[[:space:]]*\([0-9][0-9]*\),*[[:space:]]*$/\1/p' "${INFO_PATH}" | head -n 1)"
    if [[ -z "${EPISODE_COUNT}" ]]; then
        echo "total_episodes is missing or invalid: ${INFO_PATH}" >&2
        exit 1
    fi
    TOTAL_EPISODES=$((TOTAL_EPISODES + EPISODE_COUNT))
done

if [[ "${DRY_RUN}" != "true" ]]; then
    GPU_COUNT="$(python -c 'import torch; print(torch.cuda.device_count())')"
    if (( GPU_COUNT == 0 )); then
        echo "No CUDA GPU is visible; refusing to start training." >&2
        exit 2
    fi
    if (( PROC_PER_NODE > GPU_COUNT )); then
        echo "PROC_PER_NODE=${PROC_PER_NODE}, but only ${GPU_COUNT} GPUs are visible." >&2
        exit 2
    fi
fi

if [[ "${DATASET_VARIANT}" == "aloha-agilex*" ]]; then
    DEFAULT_DATASET_TAG="robotwin-all-27500"
else
    DEFAULT_DATASET_TAG="robotwin-${DATASET_VARIANT}"
fi
DATASET_TAG="${DATASET_TAG:-${DEFAULT_DATASET_TAG}}"
SAFE_DATASET_TAG="${DATASET_TAG//[^a-zA-Z0-9._-]/_}"
RUN_TAG="${RUN_TAG:-}"
# SENSECORE_JOB_NAME is not present on every platform image. MASTER_ADDR and
# MASTER_PORT are shared by all nodes in one submitted job, so they provide a
# stable fallback instead of independent per-node timestamps.
RUN_PREFIX="${RUN_ID:-${MASTER_ADDR}-${MASTER_PORT}}"
SAFE_RUN_PREFIX="${RUN_PREFIX//[^a-zA-Z0-9._-]/_}"
JOB_NAME="${JOB_NAME:-${SAFE_RUN_PREFIX}-${POLICY}-${SAFE_DATASET_TAG}-${ACTION_TYPE}-finetune${RUN_TAG:+-${RUN_TAG}}}"
BASE_OUTPUT_DIR="${BASE_OUTPUT_DIR:-${PROJ_ROOT}/outputs/${POLICY}}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_OUTPUT_DIR}/${JOB_NAME}}"
TASK_LOG_DIR="${TASK_LOG_DIR:-${BASE_OUTPUT_DIR}/logs/${SAFE_DATASET_TAG}}"
TRAIN_LOG="${TRAIN_LOG:-${TASK_LOG_DIR}/${JOB_NAME}.node-${NODE_RANK}.log}"

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

DATASET_STATS_ARGS=(--dataset.use_external_stats="${USE_EXTERNAL_STATS}")
if [[ "${USE_EXTERNAL_STATS}" == "true" ]]; then
    DATASET_STATS_ARGS+=(--dataset.external_stats_path="${STATS_PATH}")
fi

CHUNK_FILTER_ARGS=(--dataset.drop_incomplete_action_chunks="${DROP_INCOMPLETE_ACTION_CHUNKS}")
if [[ -n "${DROP_INCOMPLETE_ACTION_CHUNK_REPO_IDS}" ]]; then
    CHUNK_FILTER_ARGS+=(
        --dataset.drop_incomplete_action_chunk_repo_ids="${DROP_INCOMPLETE_ACTION_CHUNK_REPO_IDS}"
    )
fi

TRAIN_ARGS=(
    src/lerobot/scripts/lerobot_train.py
    --output_dir="${OUTPUT_DIR}"
    --job_name="${JOB_NAME}"
    --num_workers="${NUM_WORKERS}"
    --policy.type="${POLICY}"
    --policy.repo_id="lerobot_lab/${POLICY}"
    --policy.pretrained_path="${PRETRAINED_PATH}"
    --policy.vlm_model_name_or_path="${INTERNVLA_VLM_PATH}"
    --policy.wan_checkpoint_path="${WAN_CHECKPOINT_PATH}"
    --policy.wan_config_path="${WAN_CONFIG_PATH}"
    --policy.vae_path="${WAN_VAE_PATH}"
    --policy.wan_teacher_mode="${WAN_TEACHER_MODE}"
    --policy.push_to_hub=false
    --policy.gradient_checkpointing="${GRADIENT_CHECKPOINTING}"
    --policy.dtype=bfloat16
    --policy.optimizer_lr=5e-5
    --policy.scheduler_warmup_steps=2000
    --policy.scheduler_decay_steps="${STEPS}"
    --policy.scheduler_decay_lr=5e-6
    --policy.freeze_vision_encoder=false
    --policy.train_expert_only=false
    --policy.enable_vqa_loss=true
    --policy.tokenize_state=true
    --policy.knowledge_insulation=false
    --policy.video_loss_only=false
    --policy.video_loss_weight="${VIDEO_LOSS_WEIGHT}"
    --policy.action_loss_only="${ACTION_LOSS_ONLY}"
    --policy.freeze_learnable_tokens="${FREEZE_LEARNABLE_TOKENS}"
    --policy.freeze_wan_dit="${FREEZE_WAN_DIT}"
    --policy.num_learnable_tokens=50
    --dataset.type="${POLICY}"
    --dataset.repo_id="${DATASET_REPO_ID}"
    --dataset.root="${DATASET_ROOT}"
    --dataset.action_mode="${ACTION_TYPE}"
    --dataset.use_subtask_annotations="${USE_SUBTASK_ANNOTATIONS}"
    "${DATASET_STATS_ARGS[@]}"
    --dataset.video_backend=pyav
    --dataset.dist_loading="${DIST_LOADING}"
    "${CHUNK_FILTER_ARGS[@]}"
    --dataset.tokenize_state=true
    --dataset.use_fast_action_tokens=true
    --seed=42
    --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}"
    --save_freq="${SAVE_FREQ}"
    --log_freq="${LOG_FREQ}"
    --module_grad_norm_freq="${MODULE_GRAD_NORM_FREQ}"
    --wandb.enable="${WANDB_ENABLE}"
    --wandb.project="${POLICY}"
    --wandb.mode="${WANDB_MODE}"
)

echo "[InternVLA Robotwin] datasets: ${#DATASET_REPO_IDS[@]} repos / ${TOTAL_EPISODES} episodes (${DATASET_VARIANT})"
echo "[InternVLA Robotwin] dataset root: ${DATASET_ROOT}"
echo "[InternVLA Robotwin] action/stats: ${ACTION_TYPE} / ${STATS_PATH}"
echo "[InternVLA Robotwin] student: ${PRETRAINED_PATH}"
echo "[InternVLA Robotwin] video teacher: ${WAN_CHECKPOINT_PATH}"
echo "[InternVLA Robotwin] teacher mode/frozen: ${WAN_TEACHER_MODE} / ${FREEZE_WAN_DIT}"
echo "[InternVLA Robotwin] subtask annotations: ${USE_SUBTASK_ANNOTATIONS}"
echo "[InternVLA Robotwin] processes: ${NUM_PROCESSES}; batch/GPU: ${BATCH_SIZE}"
echo "[InternVLA Robotwin] output: ${OUTPUT_DIR}"
echo "[InternVLA Robotwin] log: ${TRAIN_LOG}"

if [[ "${DRY_RUN}" == "true" ]]; then
    printf 'accelerate launch'
    printf ' %q' "${ACCELERATE_ARGS[@]}" "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

mkdir -p "${TASK_LOG_DIR}"
ln -sfn "$(basename "${TRAIN_LOG}")" "${TASK_LOG_DIR}/latest-node-${NODE_RANK}.log"
if (( NODE_RANK == 0 )); then
    ln -sfn "$(basename "${TRAIN_LOG}")" "${TASK_LOG_DIR}/latest.log"
fi
accelerate launch "${ACCELERATE_ARGS[@]}" "${TRAIN_ARGS[@]}" 2>&1 | tee "${TRAIN_LOG}"
