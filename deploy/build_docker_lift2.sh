#!/usr/bin/env bash
set -euo pipefail

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_ROOT="${SCRIPT_DIR}/.."
DOCKER_DIR="${PROJ_ROOT}/.docker"

# -------- Configuration --------
IMAGE_NAME="lerobot_lab/lift2-qwen3_5"
CONTAINER_NAME="lerobot_lab-lift2-qwen3_5"
HOST_UID=$(id -u)
HOST_GID=$(id -g)
HOST_USER=${USER:-dev}
CONDA_DIR="/opt/conda"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_ENV=lerobot_lift2

FORCE_BUILD=${FORCE_BUILD:-0}

# -------- Build (content-hash) --------
BUILD_HASH=$(
  sha256sum \
    "${DOCKER_DIR}/Dockerfile.deploy.lift2" \
  | awk '{print $1}'
)
TAG="${IMAGE_NAME}:${BUILD_HASH}"

if [[ "${FORCE_BUILD}" == "1" ]] || ! docker image inspect "${TAG}" >/dev/null 2>&1; then
  echo "[info] Building ${TAG}..."
  docker build \
    -t "${TAG}" \
    -t "${IMAGE_NAME}:latest" \
    --build-arg GID="$(id -g)" \
    --build-arg UID="$(id -u)" \
    --build-arg CONDA_DIR="${CONDA_DIR}" \
    --build-arg PYTHON_VERSION="${PYTHON_VERSION}" \
    --build-arg ROS_HOSTNAME="172.16.0.51" \
    --build-arg ROS_IP="172.16.0.51" \
    --build-arg ROS_MASTER_URI="http://172.16.0.13:11311" \
    -f "${DOCKER_DIR}/Dockerfile.deploy.lift2" \
    "${PROJ_ROOT}"
else
  echo "[info] Up-to-date image found: ${TAG} (also tagged latest). Skipping build."
fi

# -------- Run --------
PROJECT_NAME=$(basename "$(dirname "$(dirname "$(realpath "$0")")")")

if [[ -n "${DISPLAY:-}" ]]; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

RUN_ARGS=(
  # --rm 
  -it
  --gpus all
  --network host
  --name "${CONTAINER_NAME}"
  -e UID="${HOST_UID}"
  -e GID="${HOST_GID}"
  -e USERNAME="${HOST_USER}"
  -v "/usr/local/cuda-12.8/:/usr/local/cuda/"
  -v `realpath .`:/workspace
  -w /workspace
  --shm-size=2g
  --ipc=host
)

if [[ -n "${DISPLAY:-}" ]]; then
  RUN_ARGS+=(
    -e DISPLAY="${DISPLAY}"
    -e QT_X11_NO_MITSHM=1
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw
  )
fi

docker run "${RUN_ARGS[@]}" "${IMAGE_NAME}:latest" bash

if [[ -n "${DISPLAY:-}" ]]; then
  xhost -local:docker >/dev/null 2>&1 || true
fi

# docker exec -it --user $USER lerobot_lab-lift2-qwen3_5 bash