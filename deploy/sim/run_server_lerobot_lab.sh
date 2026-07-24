#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
SERVER_ENV="${SERVER_ENV:-lerobot_lab}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
# User-requested convention: lerobot_lab server uses lerobot_lab conda env.
conda activate "${SERVER_ENV}"

cd "${PROJ_ROOT}"
PYTHONPATH="${PROJ_ROOT}:${PROJ_ROOT}/src:${PYTHONPATH:-}" \
python deploy/sim/server_policy.py "$@"
