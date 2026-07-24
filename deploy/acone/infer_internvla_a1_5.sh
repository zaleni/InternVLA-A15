#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# Optional deployment-machine environment setup:
#   ROS_ENV_FILE=/path/to/.env.humble.bash
#   ROS_SETUP=/opt/ros/humble/setup.bash
#   ROBOT_OVERLAY_SETUP=/path/to/robot_ws/install/setup.bash
if [[ -n "${ROS_ENV_FILE:-}" ]]; then
    # shellcheck disable=SC1090
    source "${ROS_ENV_FILE}"
elif [[ -n "${ROS_SETUP:-}" ]]; then
    # shellcheck disable=SC1090
    source "${ROS_SETUP}"
fi
if [[ -z "${ROS_ENV_FILE:-}" && -n "${ROBOT_OVERLAY_SETUP:-}" ]]; then
    # shellcheck disable=SC1090
    source "${ROBOT_OVERLAY_SETUP}"
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

if [[ -z "${CKPT_PATH:-}" ]]; then
    echo "CKPT_PATH is required and must point to .../checkpoints/<step>/pretrained_model" >&2
    exit 2
fi
if [[ -z "${TASK:-}" ]]; then
    echo "TASK is required and should match the task text used during training." >&2
    exit 2
fi

ARGS=(
    --ckpt-path "${CKPT_PATH}"
    --task "${TASK}"
    --robot-type "${ROBOT_TYPE:-arx_acone}"
    --inference-backend "${INFERENCE_BACKEND:-standard}"
    --execute-steps "${EXECUTE_STEPS:-10}"
    --max-joint-step "${MAX_JOINT_STEP:-0.35}"
    --max-tracking-error "${MAX_TRACKING_ERROR:-0.5}"
    --max-inference-latency "${MAX_INFERENCE_LATENCY:-5.0}"
    --warmup-runs "${WARMUP_RUNS:-0}"
    --state-bound-margin "${STATE_BOUND_MARGIN:-0.1}"
)

if [[ -n "${VLM_PATH:-}" ]]; then
    ARGS+=(--vlm-path "${VLM_PATH}")
fi
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
if [[ "${EXECUTE:-0}" == "1" ]]; then
    ARGS+=(--execute)
fi
if [[ "${SKIP_RESET:-0}" == "1" ]]; then
    ARGS+=(--skip-reset)
fi
if [[ "${YES:-0}" == "1" ]]; then
    ARGS+=(--yes)
fi
if [[ "${ENFORCE_STATE_BOUNDS:-1}" == "0" ]]; then
    ARGS+=(--no-enforce-state-bounds)
fi

python deploy/acone/lerobot_infer_internvla_a1_5.py "${ARGS[@]}" "$@"
