# InternVLA-A1.5 on AConE

This entry keeps the control behavior of the original, real-robot-tested
`lerobot_infer_qwen3_5vla_ki.py` and changes only what the current
InternVLA-A1.5 framework requires.

Edit these two lines near the top of
`lerobot_infer_internvla_a1_5.py`:

```python
task = "Fold the filter paper."
ckpt_path = Path("/path/to/training/output/or/pretrained_model")
```

The checkpoint path may be the training output directory or the final
`pretrained_model` directory. The launcher automatically uses the newest
numeric checkpoint when necessary.

If the ROS environment is not already loaded, the launcher sources
`${PROJECT_ROOT}/.env.humble.bash`. It does not source ROS a second time when
`ROS_DISTRO` is already set.

Run:

```bash
source /home/pjlab/caijh/InternVLA-A15/.env.humble.bash
bash /home/pjlab/caijh/InternVLA-A15/deploy/acone/infer_internvla_a1_5.sh
```

Runtime behavior matches the original entry:

- reset both arms to `[0, 0, 0, 0, 0, 0, -5]`;
- press Enter after reset;
- predict and execute the complete 50-step action chunk;
- convert delta joints to absolute targets while keeping grippers absolute;
- publish at 30 Hz;
- type `e` to exit or `r` to reset.

The required current-framework substitutions are:

- `InternVLAA15Config`;
- `InternVLAA15Policy` with its optimized action-inference backend;
- `InternVLAA15ChatProcessorTransformFn`;
- the `arx_acone` schema and checkpoint statistics.
