#!/usr/bin/env bash
set -euo pipefail

# 350 selected AC One demonstrations; 2 or 4 nodes x 8 GPUs, 30,000 optimizer steps.
# Prepare once: bash launch/pi05_finetune_acone_liquid350.sh --prepare-only
# SenseCore: submit this same shell command ONCE PER NODE (8 GPUs per node).
# The platform supplies SENSECORE_* or node-level WORLD_SIZE/RANK and MASTER_*.
# Manual launch: set NODE_RANK=<0..N-1>, MASTER_ADDR=<node0-ip>, MASTER_PORT=29545.
# --dry-run validates the roster/assets and prints the command without training.
MODE="${1:-train}"
case "${MODE}" in train|--prepare-only|--dry-run) ;; *) echo "Usage: $0 [--prepare-only|--dry-run]" >&2; exit 2 ;; esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJ_ROOT}"

# Resolve the node launcher environment before spawning any GPU workers.
# WORLD_SIZE/RANK are the node-level fallback used by the existing SenseCore
# acone launcher, NOT the 16-worker environment produced by accelerate.
PROC_PER_NODE="${PROC_PER_NODE:-${SENSECORE_ACCELERATE_DEVICE_COUNT:-8}}"
NODE_COUNT="${NODE_COUNT:-${SENSECORE_PYTORCH_NNODES:-${WORLD_SIZE:-4}}}"
NODE_RANK="${NODE_RANK:-${SENSECORE_PYTORCH_NODE_RANK:-${RANK:-}}}"
MASTER_ADDR="${MASTER_ADDR:-}"
MASTER_PORT="${MASTER_PORT:-}"
BATCH_SIZE="${BATCH_SIZE:-8}"
STEPS=30000
if [[ "${MODE}" != --prepare-only ]]; then
    if [[ -n "${LOCAL_RANK:-}" ]]; then
        echo "Run this shell once per node, not inside torchrun/accelerate (LOCAL_RANK is set)." >&2
        exit 2
    fi
    for value_name in PROC_PER_NODE NODE_COUNT NODE_RANK MASTER_PORT BATCH_SIZE; do
        value="${!value_name}"
        [[ "${value}" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "${value_name} is missing or not a nonnegative integer: '${value}'" >&2; exit 2; }
    done
    [[ ( "${NODE_COUNT}" == 2 || "${NODE_COUNT}" == 4 ) && "${PROC_PER_NODE}" == 8 ]] || {
        echo "This recipe requires 2 or 4 nodes x 8 GPUs. NODE_COUNT=${NODE_COUNT}, PROC_PER_NODE=${PROC_PER_NODE}; WORLD_SIZE fallback must be node count, not worker count." >&2
        exit 2
    }
    (( NODE_RANK < NODE_COUNT && MASTER_PORT > 0 && MASTER_PORT <= 65535 && BATCH_SIZE > 0 )) || {
        echo "Invalid node rank, master port, or batch size." >&2; exit 2;
    }
    case "${MASTER_ADDR}" in
        ''|localhost|127.*|::1|0.0.0.0) echo "MASTER_ADDR must be node 0's reachable address on both nodes." >&2; exit 2 ;;
    esac
fi
NUM_PROCESSES=$((NODE_COUNT * PROC_PER_NODE))
# A shared platform job name (or rendezvous address) keeps both nodes on the
# same output path. Never generate separate wall-clock timestamps per node.
RUN_ID="${RUN_ID:-${SENSECORE_JOB_NAME:-${MASTER_ADDR:-prepare}-${MASTER_PORT:-none}}}"
SAFE_RUN_ID="${RUN_ID//[^a-zA-Z0-9._-]/_}"
JOB_NAME="${JOB_NAME:-${SAFE_RUN_ID}-pi05-acone-liquid350-${NODE_COUNT}x${PROC_PER_NODE}-30k}"
[[ "${JOB_NAME}" =~ ^[a-zA-Z0-9._-]+$ && "${JOB_NAME}" != . && "${JOB_NAME}" != .. ]] || { echo "JOB_NAME must be a filename-safe run name." >&2; exit 2; }
OUTPUT_DIR="${OUTPUT_DIR:-${PROJ_ROOT}/outputs/pi05/acone_liquid350/${JOB_NAME}}"
TASK_LOG_DIR="${TASK_LOG_DIR:-${PROJ_ROOT}/outputs/pi05/acone_liquid350/logs}"
TRAIN_LOG="${TRAIN_LOG:-${TASK_LOG_DIR}/${JOB_NAME}.node-${NODE_RANK:-prepare}.log}"
if [[ "${MODE}" == train ]]; then
    [[ ! -e "${OUTPUT_DIR}" ]] || { echo "Output already exists; choose a new JOB_NAME: ${OUTPUT_DIR}" >&2; exit 2; }
    mkdir -p "${TASK_LOG_DIR}" "$(dirname "${TRAIN_LOG}")"
    ln -sfn "${TRAIN_LOG}" "${TASK_LOG_DIR}/latest-node-${NODE_RANK}.log"
    if (( NODE_RANK == 0 )); then ln -sfn "${TRAIN_LOG}" "${TASK_LOG_DIR}/latest.log"; fi
    # Include environment/preflight/stats errors as well as training output.
    exec > >(tee -a "${TRAIN_LOG}") 2>&1
fi
echo "Distributed config: node_rank=${NODE_RANK:-unset}, nodes=${NODE_COUNT}, gpu/node=${PROC_PER_NODE}, total_processes=${NUM_PROCESSES}, master=${MASTER_ADDR:-unset}:${MASTER_PORT:-unset}"
echo "Job: ${JOB_NAME}; output: ${OUTPUT_DIR}; log: ${TRAIN_LOG}"

CONDA_ROOT="${CONDA_ROOT:-${_CONDA_ROOT:-/data/jjhao/miniconda3}}"
CONDA_ENV="${CONDA_ENV:-internvla_a1_5}"
echo "[pi05] Activating conda environment: ${CONDA_ENV}"
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
# Always import this repository, independent of shell or editable-install state.
export PYTHONPATH="${PROJ_ROOT}/src"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export HF_HOME="${HF_HOME:-/data/jjhao/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/internvla_pi05_acone_liquid350_hf_cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/pi05_acone_liquid350_mpl}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

DATASET_ROOT="${DATASET_ROOT:-/data/heliqun/data-pipeline/annotated_datasets/Liquid-phase_synthesis_experiment}"
ASSET_DIR="${ASSET_DIR:-${PROJ_ROOT}/outputs/pi05/acone_liquid350/assets}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${HF_HOME}/hub/models--lerobot--pi05_base/snapshots/b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba}"
TOKENIZER_PATH="${PI05_TOKENIZER_PATH:-/data/jjhao/data/model/paligemma-3b-pt-224-tokenizer-modelscope}"
mkdir -p "${ASSET_DIR}" "${HF_DATASETS_CACHE}"

# Exact allowlist: never discover additional recovery/error-driven datasets.
# Content-addressed roster keeps statistics separate if DATASET_ROOT changes.
ROSTER_PATH="$(python - "${DATASET_ROOT}" "${ASSET_DIR}" <<'PY'
import hashlib, json, os, sys, tempfile
from pathlib import Path
root, assets = map(Path, sys.argv[1:])
selected = [
    ("Liquid-phase_synthesis_experiment_set0-1_designated", 100),
    ("Liquid-phase_synthesis_experiment_set0-2", 100),
    ("Liquid-phase_synthesis_experiment_correct_buttons_standard_start", 50),
    ("Liquid-phase_synthesis_experiment_correct_buttons_standard_start_part2", 50),
    ("Liquid-phase_synthesis_experiment_correct_buttons_varied_start", 50),
]
paths, frames = [], 0
for name, expected in selected:
    path = (root / name).resolve()
    info = json.loads((path / "meta/info.json").read_text())
    if info["total_episodes"] != expected:
        raise ValueError(f"{name}: expected {expected} episodes, got {info['total_episodes']}")
    if info["robot_type"] != "ARX AC One" or info["codebase_version"] != "v3.0" or info["fps"] != 30:
        raise ValueError(f"Unexpected robot/version/fps in {path}")
    for required in ["meta/stats.json", "meta/tasks.parquet"]:
        if not (path / required).is_file():
            raise FileNotFoundError(path / required)
    paths.append(str(path))
    frames += info["total_frames"]
    print(f"{expected:3d} episodes | {name}", file=sys.stderr)
text = "\n".join(paths) + "\n"
digest = hashlib.sha256(text.encode()).hexdigest()[:12]
roster = assets / f"repos_350_{digest}.txt"
with tempfile.NamedTemporaryFile(mode="w", dir=assets, delete=False) as f:
    f.write(text)
    temp = f.name
os.replace(temp, roster)
print(f"Selected 350 episodes / {frames:,} frames", file=sys.stderr)
print(roster)
PY
)"
mapfile -t DATASET_REPO_IDS < "${ROSTER_PATH}"
STATS_PATH="${ASSET_DIR}/$(basename "${ROSTER_PATH}" .txt)_delta_chunk50_stride2_stats.json"
for required in "${BASE_CHECKPOINT}/model.safetensors" "${BASE_CHECKPOINT}/config.json" "${TOKENIZER_PATH}/tokenizer.json"; do
    [[ -f "${required}" ]] || { echo "Missing asset: ${required}" >&2; exit 1; }
done

if [[ "${MODE}" == train ]]; then
    python - <<'PY'
import torch
if not torch.cuda.is_available() or torch.cuda.device_count() < 8:
    raise SystemExit("Eight visible CUDA GPUs are required on each node.")
PY
fi

if [[ "${MODE}" != --dry-run ]]; then
    # flock also handles simultaneous starts on shared storage. Publish only a
    # complete stats file; stats scan uses CPU and never decodes the videos.
    (
        flock -x 9
        if [[ ! -s "${STATS_PATH}" ]]; then
            TEMP_STATS="$(mktemp "${STATS_PATH}.tmp.XXXXXX")"
            trap 'rm -f "${TEMP_STATS}"' EXIT
            python util_scripts/compute_norm_stats_multi.py \
                --repo_ids "${DATASET_REPO_IDS[@]}" \
                --action_mode delta --chunk_size 50 --action_stride 2 --num_workers 5 \
                --output_path "${TEMP_STATS}"
            mv "${TEMP_STATS}" "${STATS_PATH}"
        fi
    ) 9>"${STATS_PATH}.lock"
fi
echo "Roster: ${ROSTER_PATH}"
echo "Stats:  ${STATS_PATH}"
if [[ "${MODE}" == --prepare-only ]]; then exit 0; fi

ARGS=(
    --multi_gpu --num_processes="${NUM_PROCESSES}" --num_machines="${NODE_COUNT}"
    --machine_rank="${NODE_RANK}"
    --main_process_ip="${MASTER_ADDR}" --main_process_port="${MASTER_PORT}"
    --mixed_precision=bf16
    src/lerobot/scripts/lerobot_train.py
    --output_dir="${OUTPUT_DIR}" --job_name="${JOB_NAME}"
    --num_workers="${NUM_WORKERS:-16}"
    --policy.type=pi05 --policy.repo_id=lerobot_lab/pi05-acone-liquid350
    --policy.pretrained_path="${BASE_CHECKPOINT}"
    --policy.device=cuda --policy.push_to_hub=false --policy.dtype=bfloat16
    --policy.gradient_checkpointing=false --policy.compile_model=false
    --policy.optimizer_lr=2.5e-5 --policy.scheduler_warmup_steps=1000
    --policy.scheduler_decay_steps="${STEPS}" --policy.scheduler_decay_lr=2.5e-6
    --policy.chunk_size=50 --policy.n_action_steps=50
    --policy.freeze_vision_encoder=false --policy.train_expert_only=false
    --dataset.type=pi05 --dataset.repo_id="${DATASET_REPO_IDS[*]}"
    --dataset.action_mode=delta --dataset.use_external_stats=true
    --dataset.external_stats_path="${STATS_PATH}"
    --dataset.pi05_tokenizer_path="${TOKENIZER_PATH}"
    --dataset.video_backend=pyav --dataset.dist_loading=false
    --dataset.image_transforms.enable=false
    --seed=42 --batch_size="${BATCH_SIZE}"
    --steps="${STEPS}" --save_checkpoint=true --save_freq=5000
    --log_freq=100 --eval_freq=0 --use_policy_training_preset=true
    --wandb.enable="${WANDB_ENABLE:-true}" --wandb.project=pi05_acone --wandb.mode=offline
)
echo "${NODE_COUNT} nodes x ${PROC_PER_NODE} GPUs; batch/GPU=${BATCH_SIZE}; global batch=$((NUM_PROCESSES * BATCH_SIZE)); steps=${STEPS}"
echo "Output: ${OUTPUT_DIR}"
if [[ "${MODE}" == --dry-run ]]; then
    printf '%q ' accelerate launch "${ARGS[@]}"
    printf '\n'
    exit 0
fi
exec accelerate launch "${ARGS[@]}"
