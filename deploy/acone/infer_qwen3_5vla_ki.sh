#!/usr/bin/env bash
# Compatibility launcher for the two-line hard-coded Python entry.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

ROS_ENV_FILE="${ROS_ENV_FILE:-${PROJECT_ROOT}/.env.humble.bash}"
if [[ -f "${ROS_ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    set +u
    source "${ROS_ENV_FILE}"
    set -u
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

python deploy/acone/lerobot_infer_qwen3_5vla_ki.py "$@"
