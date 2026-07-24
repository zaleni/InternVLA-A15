#!/usr/bin/env bash
set -euo pipefail

CONDA_DIR="/opt/conda"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
CONDA_ENV=lerobot_lift2

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

conda install -c conda-forge ffmpeg svt-av1 p11-kit -y
pip install torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install torchcodec numpy scipy transformers==5.2.0 mediapy loguru pytest omegaconf
pip install flash-linear-attention==0.4.2
pip install easydict ftfy imageio-ffmpeg librosa decord
pip install rospkg

pip cache purge
rm -rf /tmp/deps
conda clean -afy
conda deactivate

git clone https://github.com/ARXroboticsX/LIFT.git /opt/LIFT
git clone https://github.com/ARXroboticsX/R5.git /opt/LIFT/R5

source /opt/ros/noetic/setup.bash

cd /opt/LIFT/body/ROS
catkin_make
cd /opt/LIFT/R5/ROS/R5_ws
catkin_make
cd /opt/LIFT/R5/ARX_VR_SDK/ROS
catkin_make

echo "[INFO] LIFT2 dependencies installation complete."