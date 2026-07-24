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

ARGS=(
    --robot-type "${ROBOT_TYPE:-arx_acone}"
    --inference-backend "${INFERENCE_BACKEND:-optimized}"
    --execute-steps "${EXECUTE_STEPS:-10}"
    --max-joint-step "${MAX_JOINT_STEP:-0.35}"
    --max-tracking-error "${MAX_TRACKING_ERROR:-0.5}"
    --max-inference-latency "${MAX_INFERENCE_LATENCY:-5.0}"
    --warmup-runs "${WARMUP_RUNS:-1}"
    --state-bound-margin "${STATE_BOUND_MARGIN:-0.1}"
    --execute
)

if [[ -n "${STATS_KEY:-}" ]]; then
    ARGS+=(--stats-key "${STATS_KEY}")
fi
if [[ -n "${LANGUAGE_MEMORY:-}" ]]; then
    ARGS+=(--language-memory "${LANGUAGE_MEMORY}")
fi
if [[ -n "${ROS_CONFIG:-}" ]]; then
    ARGS+=(--ros-config "${ROS_CONFIG}")
fi
if [[ -n "${CONTROL_HZ:-}" ]]; then
    ARGS+=(--control-hz "${CONTROL_HZ}")
fi
if [[ -n "${RIGHT_GRIPPER_ZERO_OFFSET:-}" ]]; then
    ARGS+=(--right-gripper-zero-offset "${RIGHT_GRIPPER_ZERO_OFFSET}")
fi
if [[ "${SKIP_RESET:-0}" == "1" ]]; then
    ARGS+=(--skip-reset)
fi
if [[ "${ENFORCE_STATE_BOUNDS:-1}" == "0" ]]; then
    ARGS+=(--no-enforce-state-bounds)
fi

python deploy/acone/lerobot_infer_internvla_a1_5.py "${ARGS[@]}" "$@"
