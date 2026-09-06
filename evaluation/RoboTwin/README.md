# RoboTwin Evaluation

RoboTwin is the only bundled open evaluation entry.

## Setup

```bash
git submodule update --init third_party/RoboTwin
cp evaluation/RoboTwin/requirements.txt third_party/RoboTwin/script/requirements.txt
cd third_party/RoboTwin
bash script/_install.sh
bash script/_download_assets.sh
cd ../..
```

Follow the official RoboTwin documentation if your machine needs additional rendering dependencies.

## Run

```bash
bash evaluation/RoboTwin/eval.sh /path/to/checkpoint outputs/robotwin/internvla_a1_5 demo_clean 0 abs 50
```

Arguments:

- `checkpoint`: local checkpoint directory or Hugging Face repo id.
- `output_path`: directory where replay videos are saved.
- `task_config`: RoboTwin task config, such as `demo_clean` or `demo_randomized`.
- `task_idx`: index into `TASK_NAMES` in `evaluation/RoboTwin/inference.py`.
- `action_type`: `delta` or `abs`.
- `horizon`: number of predicted actions to enqueue per policy call.

Select a policy with the `POLICY_TYPE` environment variable:

```bash
POLICY_TYPE=pi05 bash evaluation/RoboTwin/eval.sh /path/to/pi05/checkpoint outputs/robotwin/pi05 demo_clean 0
```

Supported values are `pi0`, `pi0_fast`, `pi05`, and `internvla_a1_5`.

For `internvla_a1_5`, the inference entry defaults to `--action-loss-only`, which skips WAN weight loading. Use `--no-action-loss-only` only when the checkpoint and WAN assets are available.

## Results

Replay videos are written as `success_<id>.mp4` or `failure_<id>.mp4`.

For four independent 8-GPU evaluation groups (AHA/Wan2.2 × clean/randomized),
use `eval_8gpu_group.sh`. It runs all 50 tasks with one RoboTwin process per
GPU and uses an inference horizon of 25 by default. On the inference machine,
the two checkpoint defaults are:

```text
/mnt/data/jiangjiahao/data/model/zaleni/AHA-A1
/mnt/data/jiangjiahao/data/model/zaleni/Internvla-A1_5-Robotwin-60k
```

Run the four groups independently (one command on each 8-GPU machine):

```bash
bash evaluation/RoboTwin/eval_8gpu_group.sh aha demo_clean
bash evaluation/RoboTwin/eval_8gpu_group.sh aha demo_randomized
bash evaluation/RoboTwin/eval_8gpu_group.sh wan22 demo_clean
bash evaluation/RoboTwin/eval_8gpu_group.sh wan22 demo_randomized
```

Outputs are separated under `outputs/robotwin_eval/{aha,wan22}_60k/{task_config}`.
Override `AHA_CKPT`, `WAN22_CKPT`, `ROBOTWIN_ROOT`, `NUM_EPISODES`, or
`INFER_HORIZON` through environment variables when needed. The scheduler is
resumable and writes per-task logs and a run configuration into each output
directory.

The launcher follows the Muon evaluation environment: it sources Conda from
`/mnt/data/jiangjiahao/miniconda3` when available and activates
`internvla_robotwin`. Override `CONDA_ROOT` or `CONDA_ENV` if the inference
machine uses different names.

To summarize a completed evaluation directory:

```bash
python util_scripts/robotwin_result_stats.py outputs/robotwin/internvla_a1_5
```
