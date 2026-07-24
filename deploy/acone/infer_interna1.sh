set -ex

# 获取当前脚本所在目录的上两级文件夹路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOP_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

export PYTHONPATH=$TOP_DIR:$PYTHONPATH

python deploy/acone/infer_interna1.py
