#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ROOT="${CONDA_ROOT:-${_CONDA_ROOT:-}}"
if [[ -z "${CONDA_ROOT}" ]]; then
    if [[ -f "/mnt/data/jiangjiahao/miniconda3/etc/profile.d/conda.sh" ]]; then
        CONDA_ROOT="/mnt/data/jiangjiahao/miniconda3"
    else
        CONDA_ROOT="/data/jjhao/miniconda3"
    fi
fi
CONDA_ENV="${CONDA_ENV:-internvla_a1_5}"
if [[ "${SKIP_CONDA_ACTIVATE:-false}" != "true" ]]; then
    if [[ ! -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
        echo "Conda initialization script not found: ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
        exit 2
    fi
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    export CONDA_EXE="${CONDA_ROOT}/bin/conda"
    export _CONDA_EXE="${CONDA_ROOT}/bin/conda"
    export CONDA_PYTHON_EXE="${CONDA_ROOT}/bin/python"
    export _CONDA_ROOT="${CONDA_ROOT}"
    export PATH="${CONDA_ROOT}/condabin:${CONDA_ROOT}/bin${PATH:+:${PATH}}"
    conda activate "${CONDA_ENV}"
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
else
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
fi

if (( $# < 2 )); then
    echo "Usage:" >&2
    echo "  bash evaluation/RoboTwin/eval_all.sh CHECKPOINT OUTPUT_ROOT" >&2
    echo "  bash evaluation/RoboTwin/eval_all.sh CHECKPOINT OUTPUT_ROOT --task-config demo_clean" >&2
    exit 2
fi

if [[ -z "${HF_HOME:-}" ]]; then
    if [[ -d "/mnt/data/jiangjiahao" ]]; then
        export HF_HOME="/mnt/data/jiangjiahao/huggingface"
    else
        export HF_HOME="/data/jjhao/huggingface"
    fi
fi
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
export GPU_IDS="${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"
export TASK_CONFIG="${TASK_CONFIG:-demo_randomized}"
export NUM_EPISODES="${NUM_EPISODES:-100}"
export SEED="${SEED:-42}"
export STATS_KEY="${STATS_KEY:-aloha}"
export RESIZE_SIZE="${RESIZE_SIZE:-224}"
export ACTION_MODE="${ACTION_MODE:-abs}"
export DTYPE="${DTYPE:-bfloat16}"
export INFER_HORIZON="${INFER_HORIZON:-25}"
export INFERENCE_BACKEND="${INFERENCE_BACKEND:-standard}"
export INSTRUCTION_TYPE="${INSTRUCTION_TYPE:-unseen}"
export FPS="${FPS:-30}"
export LOG_LEVEL="${LOG_LEVEL:-INFO}"
export MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
export RETRY_DELAY="${RETRY_DELAY:-10}"
export STAGGER_SECONDS="${STAGGER_SECONDS:-5}"
export FORCE_RERUN="${FORCE_RERUN:-false}"

if [[ -z "${ROBOTWIN_ROOT:-}" ]]; then
    if [[ -f "${REPO_ROOT}/third_party/RoboTwin/task_config/${TASK_CONFIG}.yml" ]]; then
        export ROBOTWIN_ROOT="${REPO_ROOT}/third_party/RoboTwin"
    elif [[ -f "/mnt/data/jiangjiahao/InternVLA-muon/third_party/RoboTwin/task_config/${TASK_CONFIG}.yml" ]]; then
        export ROBOTWIN_ROOT="/mnt/data/jiangjiahao/InternVLA-muon/third_party/RoboTwin"
    elif [[ -f "/data/jjhao/InternVLA-muon/third_party/RoboTwin/task_config/${TASK_CONFIG}.yml" ]]; then
        export ROBOTWIN_ROOT="/data/jjhao/InternVLA-muon/third_party/RoboTwin"
    elif [[ -f "/data/jjhao/AHA-WAM/third_party/RoboTwin/task_config/${TASK_CONFIG}.yml" ]]; then
        export ROBOTWIN_ROOT="/data/jjhao/AHA-WAM/third_party/RoboTwin"
    else
        export ROBOTWIN_ROOT="${REPO_ROOT}/third_party/RoboTwin"
    fi
fi

export PYTHONPATH="${REPO_ROOT}/src:${ROBOTWIN_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${CUDA_HOME:+:${CUDA_HOME}/lib64}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

echo "[RoboTwin eval] checkpoint: $1"
echo "[RoboTwin eval] output: $2"
echo "[RoboTwin eval] robotwin root: ${ROBOTWIN_ROOT}"
echo "[RoboTwin eval] GPUs: ${GPU_IDS}; jobs/GPU: ${MAX_JOBS_PER_GPU}"
echo "[RoboTwin eval] task config/episodes/dtype/horizon: ${TASK_CONFIG}/${NUM_EPISODES}/${DTYPE}/${INFER_HORIZON}"

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/eval_all.py" "$@"
