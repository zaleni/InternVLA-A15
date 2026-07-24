set -ex

# export HF_TOKEN=${HF_TOKEN}
source /opt/conda/etc/profile.d/conda.sh
conda activate lerobot_lift2

export PYTHONPATH=/workspace:$PYTHONPATH
export ROS_MASTER_URI="http://192.3.8.133:11311"
export ROS_IP="192.3.8.52"
export ROS_HOSTNAME="192.3.8.52"
export HF_HOME="/workspace/hf_model"

export LD_LIBRARY_PATH=/opt/conda/envs/lerobot_lift2/lib:$LD_LIBRARY_PATH

cd /workspace

python deploy/lift2/lerobot_infer_qwen3_5vla.py
