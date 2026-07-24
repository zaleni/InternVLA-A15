#!/usr/bin/env bash
set -euo pipefail

CONDA_DIR="/opt/conda"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CONDA_ENV=lerobot_a2d

echo "[INFO] Installing Miniconda to ${CONDA_DIR}..."
# wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
bash /tmp/miniconda.sh -b -p "${CONDA_DIR}" && \
rm -f /tmp/miniconda.sh

echo "[INFO] Accepting Conda TOS automatically..."
export CONDA_OVERRIDE_TOS="yes"

echo "[INFO] Cleaning up Conda cache..."
"${CONDA_DIR}/bin/conda" clean -afy

source "${CONDA_DIR}/etc/profile.d/conda.sh"

echo "[INFO] Setting Conda Tsinghua mirrors..."
conda config --system --remove-key channels || true
conda config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --system --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
conda config --system --set show_channel_urls yes

echo "[INFO] Creating env ${CONDA_ENV}..."
conda create -y -n "${CONDA_ENV}" python="${PYTHON_VERSION}"

conda activate "${CONDA_ENV}"

echo "[INFO] Configuring pip mirror..."
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

python -m pip install --upgrade pip
conda install -c conda-forge ffmpeg=7.1.1 svt-av1 -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
  --index-url https://download.pytorch.org/whl/cu128
pip install torchcodec numpy scipy transformers==4.57.1 mediapy loguru pytest omegaconf
pip install protobuf==3.12.4 ruckig==0.14.0 opencv-python==4.10.0.84 zmq==0.0.0 pyzmq==26.2.0 matplotlib pynput

echo "[INFO] Cleaning pip cache..."
pip cache purge || true

rm -rf /tmp/deps || true

echo "[INFO] Final Conda clean..."
conda clean -afy

conda deactivate

echo "[INFO] A2D dependencies installation complete."