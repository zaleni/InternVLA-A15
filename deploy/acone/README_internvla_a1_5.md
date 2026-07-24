# InternVLA-A1.5 on AConE

Use `infer_internvla_a1_5.sh` for checkpoints trained with
`launch/internvla_a15_finetune_acone.sh`.

## Two-line hard-coded workflow

Edit only `task` and `ckpt_path` near the top of
`lerobot_infer_internvla_a1_5.py`, then run the InternVLA-A1.5 launcher:

```bash
bash deploy/acone/infer_internvla_a1_5.sh
```

The launcher selects the optimized backend and performs one real-observation
warmup. It does not require typing `EXECUTE`; after reset, press Enter once to
start inference and action publication. Pass `--no-execute` for a dry run.

The checkpoint path must resolve to a `pretrained_model` directory containing:

```text
config.json
model.safetensors
stats.json
train_config.json
```

The launcher accepts a step directory or training output directory too, and
will resolve `pretrained_model` or `checkpoints/last/pretrained_model`.

## Environment check

The same Python interpreter must be able to import PyTorch, ROS 2, the robot
messages, and this repository:

```bash
python -c "import torch, rclpy, cv_bridge; from arm_control.msg import PosCmd; from arx5_arm_msg.msg import RobotStatus"
```

Set ROS setup paths only when they are not already sourced:

```bash
export ROS_ENV_FILE=/path/to/.env.humble.bash
# Or set the two setup files separately:
export ROS_SETUP=/opt/ros/humble/setup.bash
export ROBOT_OVERLAY_SETUP=/path/to/robot_ws/install/setup.bash
```

## Dry run first

Dry-run mode subscribes to the real observations and performs inference but
does not reset or publish actions:

```bash
bash deploy/acone/infer_internvla_a1_5.sh --no-execute --max-control-steps 100
```

The hard-coded `task` must be the task sentence stored in the training dataset
metadata, not merely the dataset directory name; inspect `meta/tasks.jsonl`
when uncertain.

## Real execution

After checking camera order, qpos order, action values, reset pose, and gripper
zero point in dry-run mode:

```bash
bash deploy/acone/infer_internvla_a1_5.sh
```

There is no `EXECUTE` prompt. After the robot reaches its reset pose, press
Enter once to start inference. The launcher enables real action publication by
default.

The adapter rejects non-finite actions, excessive command steps, excessive
tracking error, stale inference results, and joint targets outside the
checkpoint's empirical state range. These checks supplement rather than
replace the robot controller's physical joint and emergency-stop limits.

Keyboard commands while running:

- `e`: exit
- `r`: clear the current chunk and reset (execute mode only)

The default `standard` backend is intended for the initial dry run.

## Important model semantics

- Policy type: `internvla_a1_5`
- Prompt control mode: `joint`
- Dataset action mode: loaded from `train_config.json` (the supplied training
  launcher uses `delta`)
- Physical order: left 6 joints, left gripper, right 6 joints, right gripper
- Joints are delta relative to the observation at the start of each chunk;
  grippers are absolute
- The internal 32-dimensional output is inverse-reordered through the
  `arx_acone` schema before it is sent to the 14-dimensional robot interface

Do not copy the old task-specific gripper thresholds or the commented `6.6`
right-gripper offset without calibrating them against the training dataset.
