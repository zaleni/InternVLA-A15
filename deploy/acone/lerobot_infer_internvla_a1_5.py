#!/usr/bin/env python
"""Deploy an InternVLA-A1.5 checkpoint on the ARX AC One robot.

This intentionally follows the original, real-robot-tested
``lerobot_infer_qwen3_5vla_ki.py`` control loop. Only the model/config,
processor, schema name, and current-framework padding/reordering are adapted
for checkpoints trained by ``internvla_a15_finetune_acone.sh``.
"""

from __future__ import annotations

import logging
import select
import sys
import threading
import time
from collections import deque
from pathlib import Path

# Support direct execution without an editable install.
REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT / "src", REPO_ROOT):
    import_root_str = str(import_root)
    if import_root_str not in sys.path:
        sys.path.insert(0, import_root_str)

import numpy as np
import rclpy
import torch
from omegaconf import OmegaConf

from deploy.src.acone.ros2_operator import Rate, RosOperator
from lerobot.configs.policies import PreTrainedConfig
from lerobot.dataset_schemas import get_schema
from lerobot.datasets.utils import load_json
from lerobot.policies.internvla_a1_5 import InternVLAA15Config, InternVLAA15Policy
from lerobot.policies.internvla_a1_5.transform_internvla_a1_5 import (
    InternVLAA15ChatProcessorTransformFn,
)
from lerobot.transforms.core import (
    NormalizeTransformFn,
    PadStateAndActionTransformFn,
    RemapImageKeyTransformFn,
    ReorderStateActionTransform,
    ResizeImagesWithPadFn,
    UnNormalizeTransformFn,
    compose,
)
from lerobot.utils.constants import ACTION, OBS_STATE


# =============================================================================
# 和原部署脚本一样：通常只改下面两行。
# =============================================================================
task = "Fold the filter paper"
ckpt_path = Path("/home/pjlab/caijh/InternVLA-A15/outputs/internvla_a1_5/internvla_a1_5_fold_filter_paper_delta_60k")

# 推理机上的 Qwen3.5 主干路径；同一台机器通常不需要改。
vlm_path = Path(
    "/home/pjlab/caijh/qwen3_5_vqa/lerobot_lab/hf_model/hub/"
    "models--Qwen--Qwen3.5-2B-Action"
)


def resolve_checkpoint_path(path: Path) -> Path:
    """Accept the original pretrained_model path or a training output directory."""
    path = path.expanduser().resolve()
    candidates = [
        path,
        path / "pretrained_model",
        path / "checkpoints/last/pretrained_model",
    ]
    checkpoints_dir = path / "checkpoints"
    if checkpoints_dir.is_dir():
        numeric_steps = sorted(
            (
                item / "pretrained_model"
                for item in checkpoints_dir.iterdir()
                if item.is_dir() and item.name.isdigit()
            ),
            key=lambda item: int(item.parent.name),
            reverse=True,
        )
        candidates.extend(numeric_steps)
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (
            candidate / "model.safetensors"
        ).is_file():
            return candidate
    raise FileNotFoundError(
        f"Cannot find config.json and model.safetensors below checkpoint path: {path}"
    )


def resolve_vlm_path(path: Path) -> Path:
    """Accept either a model directory or a Hugging Face models--... cache root."""
    path = path.expanduser().resolve()
    if (path / "config.json").is_file():
        return path

    main_ref = path / "refs/main"
    if main_ref.is_file():
        revision = main_ref.read_text(encoding="utf-8").strip()
        snapshot = path / "snapshots" / revision
        if revision and (snapshot / "config.json").is_file():
            return snapshot

    snapshots_dir = path / "snapshots"
    if snapshots_dir.is_dir():
        snapshots = [
            item for item in snapshots_dir.iterdir() if (item / "config.json").is_file()
        ]
        if len(snapshots) == 1:
            return snapshots[0]
    raise FileNotFoundError(f"Cannot resolve Qwen3.5 model directory from: {path}")


def select_stats(stats_file: Path) -> dict:
    """Load current flat stats, with support for the old top-level robot key."""
    all_stats = load_json(stats_file)
    if OBS_STATE in all_stats or "states.left_joint.position" in all_stats:
        return all_stats
    if "arx_acone" in all_stats:
        return all_stats["arx_acone"]
    if "arx_lift2" in all_stats:
        return all_stats["arx_lift2"]
    if len(all_stats) == 1 and isinstance(next(iter(all_stats.values())), dict):
        return next(iter(all_stats.values()))
    raise KeyError(
        f"Cannot select AConE statistics from {stats_file}; keys={list(all_stats)}"
    )


def concatenate_stats(
    stats: dict,
    unified_key: str,
    feature_keys: list[str],
) -> dict[str, np.ndarray]:
    if unified_key in stats:
        return {
            name: np.asarray(value, dtype=np.float32)
            for name, value in stats[unified_key].items()
        }
    return {
        name: np.concatenate(
            [
                np.asarray(stats[feature_key][name], dtype=np.float32).reshape(-1)
                for feature_key in feature_keys
            ]
        )
        for name in ("min", "max", "mean", "std")
    }


class DeployACOne:
    """The same ROS wrapper used by the original deployment script."""

    def __init__(self, config, in_collect: bool = False):
        rclpy.init()
        self.ros_operator = RosOperator(config, in_collect=in_collect)
        self.rate = Rate(config.frame_rate)
        spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(self.ros_operator,),
            daemon=True,
        )
        spin_thread.start()

    def get_observation(self):
        while rclpy.ok():
            obs_dict = self.ros_operator.get_observation()
            if not obs_dict:
                print("sync fail")
                self.rate.sleep()
                continue
            obs_dict.update(
                {
                    "image_head": obs_dict["images"]["head"],
                    "image_left": obs_dict["images"]["left_wrist"],
                    "image_right": obs_dict["images"]["right_wrist"],
                }
            )
            return obs_dict
        raise RuntimeError("ROS stopped while waiting for an observation.")

    def reset(self):
        left = [0, 0, 0, 0, 0, 0, -5]
        right = [0, 0, 0, 0, 0, 0, -5]
        self.ros_operator.follow_arm_publish_continuous(left, right)
        input("Enter any key to continue: ")

    def step(self, action):
        self.ros_operator.follow_arm_publish(action[0:7], action[7:14])


def build_inputs(
    obs: dict,
    input_transforms,
    config: InternVLAA15Config,
    dtype: torch.dtype,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Apply the current A1.5 transforms while keeping the original sample layout."""
    init_action = (
        torch.as_tensor(obs["qpos"][None]).contiguous().cuda().to(torch.float32)
    )
    sample = {
        "images.rgb.head": (
            torch.as_tensor(obs["image_head"].copy()).contiguous().float() / 255.0
        ),
        "images.rgb.hand_left": (
            torch.as_tensor(obs["image_left"].copy()).contiguous().float() / 255.0
        ),
        "images.rgb.hand_right": (
            torch.as_tensor(obs["image_right"].copy()).contiguous().float() / 255.0
        ),
        # Hugging Face's processor runs on CPU; move its tensor outputs to CUDA
        # only after all preprocessing is complete.
        OBS_STATE: torch.as_tensor(obs["qpos"]).contiguous().float(),
        # The current padding transform handles state and action together.
        ACTION: torch.zeros(
            config.chunk_size,
            14,
            dtype=torch.float32,
        ),
        "task": task,
    }
    for key in list(sample):
        if "images" in key:
            sample[key] = sample[key].permute(2, 0, 1)

    sample = input_transforms(sample)
    inputs: dict[str, torch.Tensor] = {}
    for key, value in sample.items():
        if key == "task":
            inputs[key] = [value]
        elif isinstance(value, torch.Tensor):
            value = value[None].cuda()
            if value.dtype in (torch.int64, torch.bool):
                inputs[key] = value
            else:
                inputs[key] = value.to(dtype=dtype)
    return inputs, init_action


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    resolved_ckpt = resolve_checkpoint_path(ckpt_path)
    resolved_vlm = resolve_vlm_path(vlm_path)

    acone_config = OmegaConf.load("deploy/src/config/acone_ros2_config.yaml")
    acone = DeployACOne(acone_config, in_collect=False)

    config = PreTrainedConfig.from_pretrained(resolved_ckpt)
    assert isinstance(config, InternVLAA15Config), (
        f"Expected internvla_a1_5 checkpoint, got {type(config).__name__}"
    )
    config.vlm_model_name_or_path = str(resolved_vlm)
    config.device = "cuda"
    config.compile_model = False
    config.compile_mode = "reduce-overhead"
    config.gradient_checkpointing = False
    config.action_loss_only = True
    config.inference_backend = "optimized"
    config.inference_action_type = "fm"

    action_chunk = deque(maxlen=config.chunk_size)
    action_mode = "delta"
    dtype = torch.float32

    policy = InternVLAA15Policy.from_pretrained(
        config=config,
        pretrained_name_or_path=resolved_ckpt,
        strict=False,
    )
    policy.cuda()
    policy.to(dtype)
    policy.eval()

    total_params = sum(parameter.numel() for parameter in policy.parameters())
    qwen_params = sum(
        parameter.numel()
        for parameter in policy.model.qwen3_5_with_expert.qwen3_5.parameters()
    )
    expert_params = sum(
        parameter.numel()
        for parameter in policy.model.qwen3_5_with_expert.action_expert.parameters()
    )
    print(f"\nTotal parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"Qwen3_5 params: {qwen_params / 1e9:.2f}B")
    print(f"Qwen3_5_Expert params: {expert_params / 1e9:.2f}B")

    stats = select_stats(resolved_ckpt / "stats.json")
    schema = get_schema("arx_acone")
    state_stats = concatenate_stats(
        stats,
        OBS_STATE,
        schema.get_state_keys(),
    )
    action_stats = concatenate_stats(
        stats,
        ACTION,
        schema.get_action_keys(),
    )
    state_stat = {OBS_STATE: state_stats}
    action_stat = {ACTION: action_stats}
    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=[ACTION],
        norm_stats=action_stat,
    )

    input_transforms = compose(
        [
            ResizeImagesWithPadFn(
                height=224,
                width=224,
                mapping=schema.image_mapping,
            ),
            RemapImageKeyTransformFn(mapping=schema.image_mapping),
            NormalizeTransformFn(
                selected_keys=[OBS_STATE],
                norm_stats=state_stat,
            ),
            InternVLAA15ChatProcessorTransformFn(
                pretrained_model_name_or_path=str(resolved_vlm),
                max_length=650,
                tokenize_state=config.tokenize_state,
                max_state_dim=config.max_state_dim,
                use_fast_action_tokens=False,
                mode="eval",
                action_mode=schema.action_mode,
            ),
            PadStateAndActionTransformFn(
                max_state_dim=config.max_state_dim,
                max_action_dim=config.max_action_dim,
            ),
            ReorderStateActionTransform(
                state_reorder=schema.state_reorder,
                action_reorder=schema.action_reorder,
            ),
        ]
    )

    logger.info("policy warmup ...")
    dummy_obs = {
        "image_head": np.zeros((360, 640, 3), dtype=np.uint8),
        "image_left": np.zeros((480, 640, 3), dtype=np.uint8),
        "image_right": np.zeros((480, 640, 3), dtype=np.uint8),
        "qpos": np.zeros(14, dtype=np.float32),
    }
    dummy_inputs, _ = build_inputs(
        dummy_obs,
        input_transforms,
        config,
        dtype,
    )
    with torch.no_grad():
        policy.predict_action_chunk(dummy_inputs)
        policy.predict_action_chunk(dummy_inputs)

    print("acone reset !")
    acone.reset()

    while True:
        start_time = time.perf_counter()
        if len(action_chunk) == 0:
            print("predict new action chunk")
            obs = acone.get_observation()
            inputs, init_action = build_inputs(
                obs,
                input_transforms,
                config,
                dtype,
            )

            predict_started = time.perf_counter()
            with torch.no_grad():
                action_pred = policy.predict_action_chunk(inputs)[0, :, :16]
                action_pred = torch.cat(
                    [
                        action_pred[:, :6],
                        action_pred[:, 7:8],
                        action_pred[:, 8:14],
                        action_pred[:, 15:16],
                    ],
                    dim=1,
                )
                action_pred = unnormalize_fn({ACTION: action_pred})[ACTION]
                if action_mode == "delta":
                    init_action[:, 6] = 0.0
                    init_action[:, 13] = 0.0
                    action_pred += init_action
                action_chunk.extend(action_pred.unbind(dim=0))
            print(f"elapse time: {time.perf_counter() - predict_started:.4f}s")

        action = action_chunk.popleft().to(torch.float32).cpu().numpy()

        # Preserve the active gripper postprocessing from the original script.
        action[6] = 0 if action[6] > -0.5 else action[6]
        action[13] = 0 if action[13] > -0.5 else action[13]

        acone.step(action)
        elapsed = time.perf_counter() - start_time
        time.sleep(max(0, 1 / 30 - elapsed))

        if sys.stdin.isatty():
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if readable:
                user_key = sys.stdin.readline().strip().lower()
                if user_key == "e":
                    print("Early exit requested. Breaking current loop.")
                    acone.reset()
                    action_chunk.clear()
                    break
                if user_key == "r":
                    print("Resetting robot arm positions...")
                    acone.reset()
                    action_chunk.clear()
    print("End.")


if __name__ == "__main__":
    main()
