#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Optional deployment-machine environment setup:
#   ROS_ENV_FILE=/path/to/.env.humble.bash
#   ROS_SETUP=/opt/ros/humble/setup.bash
#   ROBOT_OVERLAY_SETUP=/path/to/robot_ws/install/setup.bash
#
# If the caller already sourced ROS, keep that environment and do not source it
# a second time.
if [[ -z "${ROS_DISTRO:-}" ]]; then
    ros_env_path="${ROS_ENV_FILE:-${PROJECT_ROOT}/.env.humble.bash}"
    if [[ -f "${ros_env_path}" ]]; then
        # shellcheck disable=SC1090
        set +u
        source "${ros_env_path}"
        set -u
    elif [[ -n "${ROS_SETUP:-}" ]]; then
        # shellcheck disable=SC1090
        set +u
        source "${ROS_SETUP}"
        set -u
    fi
fi
if [[ -n "${ROBOT_OVERLAY_SETUP:-}" ]]; then
    # shellcheck disable=SC1090
    set +u
    source "${ROBOT_OVERLAY_SETUP}"
    set -u
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

python deploy/acone/lerobot_infer_internvla_a1_5.py
