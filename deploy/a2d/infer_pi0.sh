set -ex

source ext/a2d_sdk/env.sh
export PYTHONPATH=/workspace/src:${PYTHONPATH}
export PYTHONPATH=/workspace/lerobot/src:${PYTHONPATH}
python infer_pi0.py

exit