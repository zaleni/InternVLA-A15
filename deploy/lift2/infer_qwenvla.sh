set -ex

# export HF_TOKEN=${HF_TOKEN}
# source /opt/conda/etc/profile.d/conda.sh
# conda activate lerobot_lift2

export PYTHONPATH=/workspace:$PYTHONPATH
export ROS_MASTER_URI="http://172.16.0.13:11311"

cd /workspace

python deploy/lift2/lerobot_infer_qwenvla.py
