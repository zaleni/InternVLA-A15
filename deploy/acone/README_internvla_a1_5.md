# InternVLA-A1.5 on AConE

Use `infer_internvla_a1_5.sh` for checkpoints trained with
`launch/internvla_a15_finetune_acone.sh`.

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
export CKPT_PATH=/path/to/output/checkpoints/060000/pretrained_model
export VLM_PATH=/path/on/inference-machine/Qwen3.5-2B-Action
export TASK='Fold the filter paper.'
export EXECUTE_STEPS=10
bash deploy/acone/infer_internvla_a1_5.sh --max-control-steps 100
```

`VLM_PATH` may be omitted when the absolute path stored in `config.json` also
exists on the inference machine. `TASK` must be the task sentence stored in the
training dataset metadata, not merely the dataset directory name; inspect
`meta/tasks.jsonl` when uncertain.

## Real execution

After checking camera order, qpos order, action values, reset pose, and gripper
zero point in dry-run mode:

```bash
export EXECUTE=1
bash deploy/acone/infer_internvla_a1_5.sh
```

The program requires typing `EXECUTE` before it publishes anything. Use
`YES=1` only for an already validated unattended setup.

For real execution, use the optimized backend and capture its CUDA graph before
the first executable plan:

```bash
export INFERENCE_BACKEND=optimized
export WARMUP_RUNS=1
export EXECUTE=1
bash deploy/acone/infer_internvla_a1_5.sh
```

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
