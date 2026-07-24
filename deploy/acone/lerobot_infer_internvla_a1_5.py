#!/usr/bin/env python
"""Deploy an InternVLA-A1.5 checkpoint on the ARX AC One robot.

This is the current-framework replacement for the old
``lerobot_infer_qwen3_5vla_ki.py`` development script.  It intentionally keeps
the ROS operator from that script, while using the public
``InternVLAA15Policy`` and the exact AConE training-time preprocessing layout.

The shell/CLI workflow is dry-run unless ``--execute`` is provided. When this
file is run without command-line arguments, it uses the two hard-coded values
below. Real execution waits only for Enter after the reset pose is reached; it
does not require typing a confirmation word.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import select
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Make direct ``python deploy/acone/<script>.py`` execution work without an
# editable install or a manually exported PYTHONPATH.
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
from lerobot.utils.constants import ACTION, OBS_STATE, OBS_STR


DEFAULT_ROS_CONFIG = REPO_ROOT / "deploy/src/config/acone_ros2_config.yaml"
STAT_NAMES = ("min", "max", "mean", "std", "q01", "q99", "mask")
INTEGER_DTYPES = (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8)

# =============================================================================
# 简单用法：通常只改下面 task 和 ckpt_path 两行，然后直接运行本文件。
# ckpt_path 可以是训练输出目录，也可以是 checkpoints/xxxxxx/pretrained_model。
# =============================================================================
task = "Fold the filter paper."
ckpt_path = Path("/home/pjlab/caijh/InternVLA-A15/outputs/internvla_a1_5/internvla_a1_5_fold_filter_paper_delta_50k")

# 推理机上固定不变的基础 Qwen 路径，不换机器通常不用改。
# 支持 Hugging Face cache 的 models--... 根目录，会自动解析 refs/main。
vlm_path = Path(
    "/home/pjlab/caijh/qwen3_5_vqa/lerobot_lab/hf_model/hub/"
    "models--Qwen--Qwen3.5-2B-Action"
)


def parse_args() -> argparse.Namespace:
    hardcoded_direct_run = len(sys.argv) == 1
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=ckpt_path,
        help="pretrained_model directory, checkpoint step directory, or training output directory.",
    )
    parser.add_argument("--task", default=task, help="Task text used during training.")
    parser.add_argument(
        "--language-memory",
        default="",
        help="Optional language_memory text if it was present in training annotations.",
    )
    parser.add_argument(
        "--vlm-path",
        type=Path,
        default=vlm_path,
        help="Deployment-machine Qwen3.5-Action directory; otherwise use checkpoint config.",
    )
    parser.add_argument("--robot-type", default="arx_acone")
    parser.add_argument(
        "--stats-key",
        default=None,
        help="Top-level stats.json key. Auto-selects robot_type or the only available key.",
    )
    parser.add_argument("--ros-config", type=Path, default=DEFAULT_ROS_CONFIG)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--inference-backend",
        choices=("standard", "optimized"),
        default="optimized" if hardcoded_direct_run else "standard",
    )
    parser.add_argument("--num-inference-steps", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--max-prompt-length", type=int, default=650)
    parser.add_argument(
        "--action-mode",
        choices=("auto", "abs", "delta"),
        default="auto",
        help="Dataset action representation, not the prompt control mode.",
    )
    parser.add_argument(
        "--normalization-mode",
        choices=("auto", "mean_std", "min_max", "q01_q99"),
        default="auto",
    )
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=10,
        help="Number of actions to execute from each predicted chunk before replanning.",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=0,
        help="0 uses frame_rate from the ROS config.",
    )
    parser.add_argument(
        "--max-control-steps",
        type=int,
        default=0,
        help="0 runs until Ctrl-C or the 'e' keyboard command.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1 if hardcoded_direct_run else 0,
        help="Extra predictions on the first real observation before control starts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--right-gripper-zero-offset",
        type=float,
        default=0.0,
        help="Dataset qpos minus robot qpos offset. Keep 0 unless it was calibrated in data collection.",
    )
    parser.add_argument(
        "--clip-to-stats",
        dest="clip_to_stats",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clip denormalized delta/absolute values to checkpoint stats before delta composition.",
    )
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=0.35,
        help="Abort if a commanded joint changes more than this from the previous command; 0 disables.",
    )
    parser.add_argument(
        "--max-tracking-error",
        type=float,
        default=0.5,
        help="Abort if measured joints lag the previous command by more than this; 0 disables.",
    )
    parser.add_argument(
        "--max-inference-latency",
        type=float,
        default=5.0,
        help="In execute mode, reject a plan older than this many seconds; 0 disables.",
    )
    parser.add_argument(
        "--enforce-state-bounds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require commanded joints to stay inside checkpoint state min/max plus a margin.",
    )
    parser.add_argument(
        "--state-bound-margin",
        type=float,
        default=0.1,
        help="Margin added to empirical training-state joint bounds.",
    )
    parser.add_argument(
        "--execute",
        action=argparse.BooleanOptionalAction,
        default=hardcoded_direct_run,
        help="Actually publish robot actions. Without this flag the full pipeline is dry-run only.",
    )
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument("--reset-left", default="0,0,0,0,0,0,-5")
    parser.add_argument("--reset-right", default="0,0,0,0,0,0,-5")
    return parser.parse_args()


def parse_pose(value: str, name: str) -> np.ndarray:
    try:
        pose = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"{name} must be seven comma-separated numbers, got {value!r}") from exc
    if pose.shape != (7,):
        raise ValueError(f"{name} must contain 7 values, got shape={pose.shape}")
    return pose


def validate_ros_message_runtime() -> None:
    """Fail before loading the large checkpoint when the robot overlay is absent."""
    try:
        for module_name in (
            "arm_control.msg._joint_control",
            "arm_control.msg._pos_cmd",
            "arx5_arm_msg.msg._robot_cmd",
            "arx5_arm_msg.msg._robot_status",
        ):
            importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            "AConE ROS message packages are unavailable in this Python environment. "
            "Source ROS Humble and the robot workspace (ROS_ENV_FILE/ROS_SETUP/"
            "ROBOT_OVERLAY_SETUP) before loading the checkpoint."
        ) from exc


def resolve_checkpoint_path(path: Path) -> Path:
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
                candidate / "pretrained_model"
                for candidate in checkpoints_dir.iterdir()
                if candidate.is_dir() and candidate.name.isdigit()
            ),
            key=lambda candidate: int(candidate.parent.name),
            reverse=True,
        )
        candidates.extend(numeric_steps)
    for candidate in candidates:
        if (candidate / "config.json").is_file() and (candidate / "model.safetensors").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find config.json + model.safetensors. Pass a pretrained_model directory, "
        f"a checkpoint step directory, or an output directory. Checked: {list(map(str, candidates))}"
    )


def resolve_vlm_path(path: Path) -> Path:
    """Resolve either a model directory or a Hugging Face cache model root."""
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
        snapshots = sorted(
            candidate
            for candidate in snapshots_dir.iterdir()
            if (candidate / "config.json").is_file()
        )
        if len(snapshots) == 1:
            return snapshots[0]

    raise FileNotFoundError(
        "Qwen VLM directory must contain config.json, or be a Hugging Face "
        f"models--... cache root with refs/main. Got: {path}"
    )


def load_train_config(ckpt_path: Path) -> dict[str, Any]:
    path = ckpt_path / "train_config.json"
    if not path.is_file():
        logging.warning("%s is absent; action/normalization modes will use compatibility defaults.", path)
        return {}
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return value


def resolve_action_mode(requested: str, train_config: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    value = str(train_config.get("dataset", {}).get("action_mode", "")).strip().lower()
    if value in {"abs", "delta"}:
        return value
    logging.warning("Could not infer dataset.action_mode; defaulting to delta for AConE compatibility.")
    return "delta"


def resolve_normalization_mode(requested: str, train_config: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    transforms = (
        train_config.get("dataset", {})
        .get("data_transforms", {})
        .get("inputs", [])
    )
    if isinstance(transforms, list):
        for transform in transforms:
            if not isinstance(transform, dict) or transform.get("type") != "normalize":
                continue
            value = str(transform.get("mode", "")).strip().lower()
            if value in {"mean_std", "min_max", "q01_q99"}:
                return value
    return "mean_std"


def looks_like_feature_stats(value: dict[str, Any], schema) -> bool:
    feature_keys = [OBS_STATE, ACTION, *schema.get_state_keys(), *schema.get_action_keys()]
    return any(key in value for key in feature_keys)


def select_stats(
    all_stats: dict[str, Any],
    *,
    requested_key: str | None,
    robot_type: str,
    schema,
) -> tuple[str, dict[str, Any]]:
    if requested_key:
        if requested_key not in all_stats:
            raise KeyError(
                f"stats_key={requested_key!r} is absent; available keys={list(all_stats.keys())}"
            )
        return requested_key, all_stats[requested_key]
    if robot_type in all_stats:
        return robot_type, all_stats[robot_type]
    if looks_like_feature_stats(all_stats, schema):
        return "<root>", all_stats
    if len(all_stats) == 1:
        key = next(iter(all_stats))
        return key, all_stats[key]
    raise ValueError(
        "stats.json has multiple robot keys. Pass --stats-key explicitly. "
        f"Available keys: {list(all_stats.keys())}"
    )


def numpy_stat_dict(value: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(stat, dtype=np.float32)
        for name, stat in value.items()
        if name in STAT_NAMES
    }


def concatenate_feature_stats(
    selected_stats: dict[str, Any],
    *,
    unified_key: str,
    feature_keys: list[str],
) -> dict[str, np.ndarray]:
    if unified_key in selected_stats:
        return numpy_stat_dict(selected_stats[unified_key])
    missing = [key for key in feature_keys if key not in selected_stats]
    if missing:
        raise KeyError(
            f"Cannot build {unified_key!r} stats; missing feature keys={missing}. "
            f"Available keys={list(selected_stats.keys())[:30]}"
        )
    common_names = [
        name
        for name in STAT_NAMES
        if all(name in selected_stats[key] for key in feature_keys)
    ]
    return {
        name: np.concatenate(
            [np.asarray(selected_stats[key][name], dtype=np.float32) for key in feature_keys],
            axis=-1,
        )
        for name in common_names
    }


def required_stats_for_mode(mode: str) -> tuple[str, str]:
    if mode == "mean_std":
        return "mean", "std"
    if mode == "min_max":
        return "min", "max"
    if mode == "q01_q99":
        return "q01", "q99"
    raise ValueError(f"Unsupported normalization mode: {mode}")


def validate_stats(stats: dict[str, np.ndarray], mode: str, label: str) -> int:
    required = required_stats_for_mode(mode)
    missing = [name for name in required if name not in stats]
    if missing:
        raise KeyError(f"{label} stats do not contain {missing} required by mode={mode!r}.")
    dims = {np.asarray(stats[name]).reshape(-1).shape[0] for name in required}
    if len(dims) != 1:
        raise ValueError(f"{label} stats have inconsistent dimensions: {dims}")
    return next(iter(dims))


def invert_reorder(tensor: torch.Tensor, reorder_spec: list[list[int]] | None) -> torch.Tensor:
    """Convert the model's padded/reordered layout back to compact robot order."""
    if not reorder_spec:
        return tensor
    source_dim = max(src_end for src_start, src_end, _, _ in reorder_spec)
    compact = torch.zeros(*tensor.shape[:-1], source_dim, dtype=tensor.dtype, device=tensor.device)
    for src_start, src_end, dst_start, dst_end in reorder_spec:
        if src_end - src_start != dst_end - dst_start:
            raise ValueError(f"Invalid reorder entry: {[src_start, src_end, dst_start, dst_end]}")
        if tensor.shape[-1] < dst_end:
            raise ValueError(
                f"Model action dim={tensor.shape[-1]} is too small for reorder destination {dst_end}."
            )
        compact[..., src_start:src_end] = tensor[..., dst_start:dst_end]
    return compact


def image_to_chw_float(image: np.ndarray) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB image, got shape={image.shape}.")
    tensor = torch.from_numpy(np.array(image, copy=True))
    if tensor.dtype == torch.uint8:
        tensor = tensor.float().div_(255.0)
    else:
        tensor = tensor.float()
        if float(tensor.max()) > 1.0:
            tensor = tensor.div_(255.0)
    return tensor.permute(2, 0, 1).contiguous()


@dataclass
class RuntimeContract:
    ckpt_path: Path
    stats_key: str
    action_mode: str
    normalization_mode: str
    state_dim: int
    action_dim: int


class InternVLAA15AConEInference:
    """Model, transforms, stats, and AConE action postprocessing."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ckpt_path = resolve_checkpoint_path(args.ckpt_path)
        train_config = load_train_config(self.ckpt_path)
        self.action_mode = resolve_action_mode(args.action_mode, train_config)
        self.normalization_mode = resolve_normalization_mode(
            args.normalization_mode, train_config
        )
        self.schema = get_schema(args.robot_type)

        config = PreTrainedConfig.from_pretrained(self.ckpt_path)
        if not isinstance(config, InternVLAA15Config):
            raise TypeError(
                "This deployment entry requires a checkpoint with policy.type=internvla_a1_5; "
                f"loaded {type(config).__name__}. Editing the type string alone cannot convert an old model."
            )
        if args.vlm_path is not None:
            config.vlm_model_name_or_path = str(resolve_vlm_path(args.vlm_path))
        vlm_path = str(config.vlm_model_name_or_path)
        if Path(vlm_path).is_absolute() and not Path(vlm_path).exists():
            raise FileNotFoundError(
                f"Qwen VLM path saved in the checkpoint does not exist: {vlm_path}. "
                "Pass --vlm-path with the deployment-machine path."
            )

        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
        config.device = str(device)
        config.compile_model = False
        config.gradient_checkpointing = False
        config.action_loss_only = True
        config.inference_backend = args.inference_backend
        config.inference_action_type = "fm"
        if args.num_inference_steps > 0:
            config.num_inference_steps = args.num_inference_steps

        stats_path = self.ckpt_path / "stats.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"Checkpoint stats are required for deployment: {stats_path}")
        all_stats = load_json(stats_path)
        stats_key, selected_stats = select_stats(
            all_stats,
            requested_key=args.stats_key,
            robot_type=args.robot_type,
            schema=self.schema,
        )
        state_stats = concatenate_feature_stats(
            selected_stats,
            unified_key=OBS_STATE,
            feature_keys=self.schema.get_state_keys(),
        )
        action_stats = concatenate_feature_stats(
            selected_stats,
            unified_key=ACTION,
            feature_keys=self.schema.get_action_keys(),
        )
        state_dim = validate_stats(state_stats, self.normalization_mode, "state")
        action_dim = validate_stats(action_stats, self.normalization_mode, "action")
        if state_dim != action_dim:
            raise ValueError(f"AConE state/action dimensions differ: {state_dim} vs {action_dim}")
        if state_dim != 14:
            raise ValueError(
                "This ROS adapter is for the 14-dimensional AConE joint interface; "
                f"selected schema/stats describe {state_dim} dimensions."
            )
        required_image_keys = {
            "images.rgb.head",
            "images.rgb.hand_left",
            "images.rgb.hand_right",
        }
        if set(self.schema.image_mapping) != required_image_keys:
            raise ValueError(
                "The selected robot schema does not have the AConE three-camera mapping. "
                f"Got raw image keys={list(self.schema.image_mapping)}"
            )
        if args.enforce_state_bounds:
            missing_bounds = [name for name in ("min", "max") if name not in state_stats]
            if missing_bounds:
                raise KeyError(
                    "--enforce-state-bounds is enabled, but checkpoint state stats "
                    f"do not contain {missing_bounds}."
                )
            for name in ("min", "max"):
                if np.asarray(state_stats[name]).reshape(-1).shape != (state_dim,):
                    raise ValueError(
                        f"State {name} bounds do not match the {state_dim}-dimensional AConE state."
                    )

        self.contract = RuntimeContract(
            ckpt_path=self.ckpt_path,
            stats_key=stats_key,
            action_mode=self.action_mode,
            normalization_mode=self.normalization_mode,
            state_dim=state_dim,
            action_dim=action_dim,
        )
        self.device = device
        self.config = config
        self.state_stats = state_stats
        self.action_stats = action_stats
        self.action_mask = self.schema.action_mask.cpu().numpy().astype(bool)
        if self.action_mask.size == 0:
            self.action_mask = np.ones(action_dim, dtype=bool)
        if self.action_mask.shape != (action_dim,):
            raise ValueError(
                f"Schema action mask shape={self.action_mask.shape}, expected {(action_dim,)}."
            )

        self.action_unnormalizer = UnNormalizeTransformFn(
            selected_keys=[ACTION],
            mode=self.normalization_mode,
            norm_stats={ACTION: action_stats},
        )
        self.input_transform = compose(
            [
                ResizeImagesWithPadFn(
                    height=args.resize_size,
                    width=args.resize_size,
                    mapping=self.schema.image_mapping,
                ),
                RemapImageKeyTransformFn(mapping=self.schema.image_mapping),
                NormalizeTransformFn(
                    selected_keys=[OBS_STATE],
                    mode=self.normalization_mode,
                    norm_stats={OBS_STATE: state_stats},
                ),
                InternVLAA15ChatProcessorTransformFn(
                    pretrained_model_name_or_path=vlm_path,
                    max_length=args.max_prompt_length,
                    tokenize_state=config.tokenize_state,
                    max_state_dim=config.max_state_dim,
                    use_fast_action_tokens=False,
                    mode="eval",
                    action_mode=self.schema.action_mode,
                ),
                # Training applies pad/reorder after chat processing. This order
                # keeps the prompt state compact while giving the numeric model
                # state the internal A1.5 layout.
                PadStateAndActionTransformFn(
                    max_state_dim=config.max_state_dim,
                    max_action_dim=config.max_action_dim,
                ),
                ReorderStateActionTransform(
                    state_reorder=self.schema.state_reorder,
                    action_reorder=self.schema.action_reorder,
                ),
            ]
        )

        logging.info("Loading InternVLA-A1.5 policy from %s", self.ckpt_path)
        load_started = time.perf_counter()
        self.policy = InternVLAA15Policy.from_pretrained(
            pretrained_name_or_path=self.ckpt_path,
            config=config,
            strict=False,
        )
        self.policy.to(device)
        self.policy.eval()
        logging.info("Policy loaded in %.1fs", time.perf_counter() - load_started)

        if config.dtype == "bfloat16":
            self.compute_dtype = torch.bfloat16
        elif config.dtype == "float32":
            self.compute_dtype = torch.float32
        else:
            raise ValueError(f"Unsupported checkpoint dtype={config.dtype!r}.")

    def _model_qpos(self, robot_qpos: np.ndarray) -> np.ndarray:
        qpos = np.asarray(robot_qpos, dtype=np.float32).copy()
        if qpos.shape != (self.contract.state_dim,):
            raise ValueError(
                f"ROS qpos shape={qpos.shape}, checkpoint expects {(self.contract.state_dim,)}."
            )
        qpos[13] += self.args.right_gripper_zero_offset
        return qpos

    def _build_sample(self, observation: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
        qpos = self._model_qpos(observation["qpos"])
        images = observation["images"]
        required = ("head", "left_wrist", "right_wrist")
        missing = [key for key in required if key not in images]
        if missing:
            raise KeyError(f"ROS observation is missing cameras: {missing}")

        sample: dict[str, Any] = {
            "images.rgb.head": image_to_chw_float(images["head"]),
            "images.rgb.hand_left": image_to_chw_float(images["left_wrist"]),
            "images.rgb.hand_right": image_to_chw_float(images["right_wrist"]),
            OBS_STATE: torch.from_numpy(qpos),
            # PadStateAndActionTransformFn mirrors training and requires ACTION.
            ACTION: torch.zeros(
                self.config.chunk_size,
                self.contract.action_dim,
                dtype=torch.float32,
            ),
            "task": self.args.task,
        }
        if self.args.language_memory:
            sample["language_memory"] = self.args.language_memory
        return self.input_transform(sample), qpos

    def _batch_for_policy(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        keys = (
            OBS_STATE,
            f"{OBS_STR}.pixel_values",
            f"{OBS_STR}.image_grid_thw",
            f"{OBS_STR}.input_ids",
            f"{OBS_STR}.attention_mask",
            f"{OBS_STR}.fast_token_mask",
        )
        inputs: dict[str, torch.Tensor] = {}
        for key in keys:
            value = sample.get(key)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Transformed input {key!r} must be a tensor, got {type(value)}")
            value = value.unsqueeze(0).to(self.device)
            if value.dtype in INTEGER_DTYPES or value.dtype == torch.bool:
                inputs[key] = value
            else:
                inputs[key] = value.float()
        return inputs

    def _clip_in_training_space(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.args.clip_to_stats:
            return actions
        if "min" not in self.action_stats or "max" not in self.action_stats:
            logging.warning("Action min/max stats are absent; skipping action clipping.")
            return actions
        low = torch.as_tensor(self.action_stats["min"], device=actions.device, dtype=actions.dtype)
        high = torch.as_tensor(self.action_stats["max"], device=actions.device, dtype=actions.dtype)
        return torch.maximum(torch.minimum(actions, high), low)

    @torch.inference_mode()
    def predict(self, observation: dict[str, Any]) -> tuple[np.ndarray, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        sample, model_qpos = self._build_sample(observation)
        inputs = self._batch_for_policy(sample)
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.compute_dtype,
            enabled=self.compute_dtype != torch.float32,
        ):
            model_actions = self.policy.predict_action_chunk(inputs)[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started

        compact = invert_reorder(model_actions, self.schema.action_reorder)
        compact = compact[..., : self.contract.action_dim].float()
        actions = self.action_unnormalizer({ACTION: compact})[ACTION]
        actions = self._clip_in_training_space(actions)

        if self.action_mode == "delta":
            initial = torch.as_tensor(model_qpos, device=actions.device, dtype=actions.dtype)
            mask = torch.as_tensor(self.action_mask, device=actions.device, dtype=torch.bool)
            actions = actions + torch.where(mask, initial, torch.zeros_like(initial))

        actions[..., 13] -= self.args.right_gripper_zero_offset
        result = actions.cpu().numpy().astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise FloatingPointError("Policy produced NaN or Inf; refusing to continue.")
        return result, elapsed

    def validate_absolute_joint_bounds(self, action: np.ndarray) -> None:
        if not self.args.enforce_state_bounds:
            return
        if "min" not in self.state_stats or "max" not in self.state_stats:
            raise KeyError(
                "--enforce-state-bounds requires min/max entries in checkpoint state stats."
            )
        margin = float(self.args.state_bound_margin)
        low = np.asarray(self.state_stats["min"], dtype=np.float32) - margin
        high = np.asarray(self.state_stats["max"], dtype=np.float32) + margin
        below = self.action_mask & (action < low)
        above = self.action_mask & (action > high)
        invalid = np.flatnonzero(below | above)
        if invalid.size:
            details = ", ".join(
                f"d{idx}={action[idx]:.4f} not in [{low[idx]:.4f}, {high[idx]:.4f}]"
                for idx in invalid
            )
            raise RuntimeError(
                "Absolute joint command is outside checkpoint state bounds; "
                f"action publication stopped ({details})."
            )


class DeployACOne:
    def __init__(self, config, *, execute: bool):
        rclpy.init()
        self.ros_operator = RosOperator(config, in_collect=False)
        self.rate = Rate(float(config.frame_rate))
        self.execute = execute
        self.spin_thread = threading.Thread(
            target=rclpy.spin,
            args=(self.ros_operator,),
            daemon=True,
        )
        self.spin_thread.start()

    def _keep_latest_observation_messages(self) -> None:
        # The copied development operator allows queues to grow to 2000 and
        # pops only one item per replan. Keep the newest message so long model
        # inference pauses do not accumulate a large stale image backlog.
        buffers = (
            self.ros_operator.img_head_deque,
            self.ros_operator.img_left_deque,
            self.ros_operator.img_right_deque,
            self.ros_operator.img_head_depth_deque,
            self.ros_operator.img_left_depth_deque,
            self.ros_operator.img_right_depth_deque,
            self.ros_operator.feedback_left_arm_deque,
            self.ros_operator.feedback_right_arm_deque,
        )
        for buffer in buffers:
            while len(buffer) > 1:
                buffer.popleft()

    def get_observation(self) -> dict[str, Any]:
        while rclpy.ok():
            self._keep_latest_observation_messages()
            observation = self.ros_operator.get_observation()
            if observation:
                return observation
            self.rate.sleep()
        raise RuntimeError("ROS shut down while waiting for an observation.")

    def reset(self, left: np.ndarray, right: np.ndarray) -> None:
        if not self.execute:
            logging.info("[dry-run] Would reset left=%s right=%s", left.tolist(), right.tolist())
            return
        self.ros_operator.follow_arm_publish_continuous(left.tolist(), right.tolist())

    def step(self, action: np.ndarray) -> None:
        if not self.execute:
            return
        self.ros_operator.follow_arm_publish(action[:7], action[7:14])

    def latest_qpos(self) -> np.ndarray | None:
        left = self.ros_operator.feedback_left_arm_deque
        right = self.ros_operator.feedback_right_arm_deque
        if not left or not right:
            return None
        qpos = np.concatenate(
            [
                np.asarray(left[-1].joint_pos, dtype=np.float32),
                np.asarray(right[-1].joint_pos, dtype=np.float32),
            ]
        )
        return qpos if qpos.shape == (14,) else None

    def close(self) -> None:
        try:
            self.ros_operator.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


def poll_keyboard() -> str | None:
    if not sys.stdin.isatty():
        return None
    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None
    return sys.stdin.readline().strip().lower()


def validate_joint_step(
    action: np.ndarray,
    reference: np.ndarray,
    action_mask: np.ndarray,
    max_joint_step: float,
) -> None:
    if max_joint_step <= 0:
        return
    joint_delta = np.abs(action[action_mask] - reference[action_mask])
    largest = float(joint_delta.max(initial=0.0))
    if largest > max_joint_step:
        raise RuntimeError(
            f"Joint command jump {largest:.4f} exceeds --max-joint-step={max_joint_step:.4f}; "
            "action publication stopped."
        )


def validate_tracking_error(
    measured: np.ndarray,
    commanded: np.ndarray,
    action_mask: np.ndarray,
    max_tracking_error: float,
) -> None:
    if max_tracking_error <= 0:
        return
    error = np.abs(measured[action_mask] - commanded[action_mask])
    largest = float(error.max(initial=0.0))
    if largest > max_tracking_error:
        raise RuntimeError(
            f"Measured joint tracking error {largest:.4f} exceeds "
            f"--max-tracking-error={max_tracking_error:.4f}; action publication stopped."
        )


def print_runtime_summary(
    args: argparse.Namespace,
    inference: InternVLAA15AConEInference,
    control_hz: float,
) -> None:
    contract = inference.contract
    mode = "EXECUTE" if args.execute else "DRY RUN"
    print("\n" + "=" * 72)
    print(f"Mode:                 {mode}")
    print(f"Policy:               internvla_a1_5 ({args.inference_backend})")
    print(f"Checkpoint:           {contract.ckpt_path}")
    print(f"Stats key:            {contract.stats_key}")
    print(f"Dataset action mode:  {contract.action_mode}")
    print(f"Prompt control mode:  {inference.schema.action_mode}")
    print(f"Normalization:        {contract.normalization_mode}")
    print(f"Physical action dim:  {contract.action_dim}")
    print(f"Chunk/execute steps:  {inference.config.chunk_size}/{args.execute_steps}")
    print(f"Control rate:         {control_hz:g} Hz")
    print(f"Task:                 {args.task}")
    print("=" * 72 + "\n")


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.execute_steps <= 0:
        raise ValueError("--execute-steps must be positive.")
    if args.warmup_runs < 0:
        raise ValueError("--warmup-runs cannot be negative.")
    if args.state_bound_margin < 0:
        raise ValueError("--state-bound-margin cannot be negative.")

    left_reset = parse_pose(args.reset_left, "--reset-left")
    right_reset = parse_pose(args.reset_right, "--reset-right")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    validate_ros_message_runtime()

    # Load the large model before starting ROS subscriptions so camera queues do
    # not grow for the duration of checkpoint loading.
    inference = InternVLAA15AConEInference(args)
    args.execute_steps = min(args.execute_steps, inference.config.chunk_size)
    if args.execute and not args.skip_reset:
        inference.validate_absolute_joint_bounds(np.concatenate([left_reset, right_reset]))

    ros_config_path = args.ros_config.expanduser().resolve()
    if not ros_config_path.is_file():
        raise FileNotFoundError(f"ROS config not found: {ros_config_path}")
    ros_config = OmegaConf.load(ros_config_path)
    if args.control_hz > 0:
        ros_config.frame_rate = float(args.control_hz)
    control_hz = float(ros_config.frame_rate)
    if control_hz <= 0:
        raise ValueError(f"frame_rate must be positive, got {control_hz}")

    print_runtime_summary(args, inference, control_hz)
    robot = DeployACOne(ros_config, execute=args.execute)
    action_queue: deque[np.ndarray] = deque()
    last_reference: np.ndarray | None = None
    first_prediction = True
    control_step = 0

    try:
        if not args.skip_reset:
            robot.reset(left_reset, right_reset)
            if args.execute:
                input("Reset complete. Press Enter to start inference: ")

        while rclpy.ok():
            loop_started = time.perf_counter()
            if args.max_control_steps > 0 and control_step >= args.max_control_steps:
                logging.info("Reached --max-control-steps=%d.", args.max_control_steps)
                break

            if not action_queue:
                observation = robot.get_observation()
                if first_prediction:
                    for run_idx in range(args.warmup_runs):
                        _, warmup_time = inference.predict(observation)
                        logging.info(
                            "Real-observation warmup %d/%d: %.3fs",
                            run_idx + 1,
                            args.warmup_runs,
                            warmup_time,
                        )
                    # Warmup/CUDA-graph capture can be slow. Do not execute a
                    # plan based on the observation used before warmup.
                    observation = robot.get_observation()
                    first_prediction = False

                plan, inference_time = inference.predict(observation)
                if (
                    args.execute
                    and args.max_inference_latency > 0
                    and inference_time > args.max_inference_latency
                ):
                    raise RuntimeError(
                        f"Inference latency {inference_time:.3f}s exceeds "
                        f"--max-inference-latency={args.max_inference_latency:.3f}s; "
                        "refusing to execute a stale plan."
                    )
                plan = plan[: args.execute_steps]
                action_queue.extend(plan)
                last_reference = np.asarray(observation["qpos"], dtype=np.float32).copy()
                logging.info(
                    "Predicted %d actions in %.3fs; executing %d before replanning.",
                    inference.config.chunk_size,
                    inference_time,
                    len(plan),
                )
                if not args.execute:
                    logging.info(
                        "[dry-run] chunk first action=%s",
                        np.array2string(plan[0], precision=4, suppress_small=True),
                    )

            action = action_queue.popleft()
            if last_reference is None:
                raise RuntimeError("Internal error: action reference was not initialized.")
            if args.execute and control_step > 0:
                measured_qpos = robot.latest_qpos()
                if measured_qpos is not None:
                    validate_tracking_error(
                        measured_qpos,
                        last_reference,
                        inference.action_mask,
                        args.max_tracking_error,
                    )
            validate_joint_step(
                action,
                last_reference,
                inference.action_mask,
                args.max_joint_step,
            )
            inference.validate_absolute_joint_bounds(action)
            robot.step(action)
            last_reference = action
            control_step += 1

            if not args.execute and (control_step == 1 or control_step % int(max(control_hz, 1)) == 0):
                logging.info(
                    "[dry-run] step=%d left_gripper=%.4f right_gripper=%.4f",
                    control_step,
                    float(action[6]),
                    float(action[13]),
                )

            command = poll_keyboard()
            if command == "e":
                logging.info("Keyboard exit requested.")
                break
            if command == "r":
                action_queue.clear()
                if args.execute:
                    robot.reset(left_reset, right_reset)
                else:
                    logging.info("[dry-run] Reset requested; cleared the action queue.")
                last_reference = None

            remaining = 1.0 / control_hz - (time.perf_counter() - loop_started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        logging.info("Interrupted by user.")
    finally:
        action_queue.clear()
        robot.close()
        logging.info("Deployment stopped after %d control steps.", control_step)


if __name__ == "__main__":
    main()
