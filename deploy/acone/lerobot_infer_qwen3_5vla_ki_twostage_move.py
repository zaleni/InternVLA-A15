import time
import copy
import math
import os
import sys
import select
import pdb
import logging
import threading
import termios
import tty

from pprint import pp
from pathlib import Path
from dataclasses import dataclass, replace
from collections import deque
from typing import Iterable, Optional
from omegaconf import OmegaConf
from pdb import set_trace

import torch
import numpy as np
from torch import Tensor
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
# from lerobot.policies.qwen3_5vla_ki import Qwen3_5VLAKIConfig, Qwen3_5VLAKIPolicy
# from lerobot.policies.qwen3_5vla_ki.modeling_qwen3_5vla_ki_fast import Qwen3_5VLAKIFastPolicy
from lerobot.policies.qwen3_5vla_ki_wan import Qwen3_5VLAKIWanConfig, Qwen3_5VLAKIWanPolicy
from lerobot.policies.qwen3_5vla_ki_wan.modeling_qwen3_5vla_ki_wan_opt import Qwen3_5VLAKIWanFastPolicy
from lerobot.datasets.utils import write_json, load_json
from lerobot.dataset_schemas import get_schema
from lerobot.datasets.factory import make_dataset
from lerobot.transforms.core import (
    ResizeImagesWithPadFn,
    NormalizeTransformFn,
    RemapImageKeyTransformFn,
    UnNormalizeTransformFn,
    ReorderStateActionTransform,
    PadStateTransformFn,
    compose,
)
from lerobot.utils.constants import OBS_IMAGES, OBS_STR, OBS_STATE, ACTION
from lerobot.policies.qwen3_5vla_ki.transform_qwen3_5vla_ki_v2 import Qwen3_5KIChatProcessorTransformFnV2

import rclpy

from deploy.src.acone.ros2_operator import RosOperator, Rate


def _arx_msg_bases() -> list[Path]:
    return [
        Path(f"/home/arx/ROS2_AC-one_Play/act/msg/{sys.version_info.major}.{sys.version_info.minor}"),
        Path("/home/arx/ROS2_AC-one_Play/act/msg/3.12"),
        Path("/home/arx/ROS2_AC-one_Play/act/msg/3.10"),
    ]


def _prepend_env_path(name: str, paths: Iterable[Path]) -> bool:
    existing = [p for p in os.environ.get(name, "").split(":") if p]
    additions = [str(path) for path in paths if path.exists() and str(path) not in existing]
    if not additions:
        return False
    os.environ[name] = ":".join(additions + existing)
    return True


def _ensure_arx_msg_runtime_paths() -> None:
    base = next((path for path in _arx_msg_bases() if path.exists()), None)
    if base is None:
        return

    changed = _prepend_env_path("PYTHONPATH", [base])
    changed |= _prepend_env_path("LD_LIBRARY_PATH", [base / "lib", base / "arm_control", base / "arx5_arm_msg"])
    if changed and os.environ.get("ARX_EE_JOG_REEXEC") != "1":
        os.environ["ARX_EE_JOG_REEXEC"] = "1"
        os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)

    if str(base) not in sys.path:
        sys.path.insert(0, str(base))


_ensure_arx_msg_runtime_paths()

from arm_control.msg import PosCmd
from arx5_arm_msg.msg import RobotStatus


LEFT_UP_KEY = "u"
LEFT_DOWN_KEY = "j"
RIGHT_UP_KEY = "i"
RIGHT_DOWN_KEY = "k"
JOG_KEYS = {LEFT_UP_KEY, LEFT_DOWN_KEY, RIGHT_UP_KEY, RIGHT_DOWN_KEY}
STOP_KEYS = {" ", "s"}
EXIT_KEYS = {"e", "q", "\x1b", "\x03"}
RESET_KEYS = {"r"}


class _StopOnFastToken(StoppingCriteria):
    def __init__(self, action_token_min: int, action_token_max: int):
        self.action_token_min = action_token_min
        self.action_token_max = action_token_max

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        del scores, kwargs
        last_tokens = input_ids[:, -1]
        is_fast = (last_tokens >= self.action_token_min) & (last_tokens <= self.action_token_max)
        return bool(is_fast.all().item())


def _ensure_no_fast_tokens(inputs: dict, action_token_min: int, action_token_max: int) -> None:
    key_ids = f"{OBS_STR}.input_ids"
    key_attn = f"{OBS_STR}.attention_mask"
    key_mask = f"{OBS_STR}.fast_token_mask"

    input_ids = inputs[key_ids]
    fast_pos = (input_ids >= action_token_min) & (input_ids <= action_token_max)
    if not fast_pos.any():
        inputs.pop(key_mask, None)
        return

    assert input_ids.shape[0] == 1, "FAST-strip path assumes batch size 1"
    keep = ~fast_pos
    inputs[key_ids] = input_ids[keep].unsqueeze(0)
    inputs[key_attn] = inputs[key_attn][keep].unsqueeze(0)
    inputs.pop(key_mask, None)


def _pack_qwen35_vision_inputs(
    pixel_values: Tensor, image_grid_thw: Tensor
) -> tuple[Tensor, Tensor]:
    image_grid_thw = image_grid_thw.reshape(-1, 3)
    if pixel_values.ndim > 2:
        pixel_values = pixel_values.reshape(-1, pixel_values.shape[-1])
    return pixel_values, image_grid_thw


@torch.no_grad()
def _append_subtask_until_first_fast_token(
    policy: Qwen3_5VLAKIWanFastPolicy,
    inputs: dict,
    action_token_min: int,
    action_token_max: int,
    max_new_tokens: int = 128,
    debug_tokenizer=None,
) -> int:
    key_ids = f"{OBS_STR}.input_ids"
    key_attn = f"{OBS_STR}.attention_mask"
    key_mask = f"{OBS_STR}.fast_token_mask"
    key_pixels = f"{OBS_STR}.pixel_values"
    key_grid = f"{OBS_STR}.image_grid_thw"

    _ensure_no_fast_tokens(inputs, action_token_min, action_token_max)

    lang_tokens = inputs[key_ids]
    lang_masks = inputs[key_attn]
    prompt_len = lang_tokens.shape[1]
    pixel_values, image_grid_thw = _pack_qwen35_vision_inputs(
        inputs[key_pixels], inputs[key_grid]
    )

    generated = policy.model.qwen3_5_with_expert.qwen3_5.generate(
        input_ids=lang_tokens,
        attention_mask=lang_masks,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
        do_sample=False,
        max_new_tokens=max_new_tokens,
        stopping_criteria=StoppingCriteriaList([
            _StopOnFastToken(action_token_min, action_token_max),
        ]),
    )

    new_tokens = generated[:, prompt_len:]
    if new_tokens.numel() == 0:
        inputs[key_mask] = torch.zeros_like(inputs[key_ids], dtype=torch.bool)
        return 0

    fast_pos = (new_tokens >= action_token_min) & (new_tokens <= action_token_max)
    if fast_pos.any():
        assert new_tokens.shape[0] == 1, "FAST-stop trimming assumes batch size 1"
        first_fast = fast_pos[0].nonzero(as_tuple=False)[0].item()
        subtask_tokens = new_tokens[:, :first_fast]
    else:
        subtask_tokens = new_tokens

    if debug_tokenizer is not None:
        subtask_text = debug_tokenizer.decode(
            subtask_tokens[0].detach().cpu().tolist(),
            skip_special_tokens=True,
        ).strip()
        print(f"  generated subtask: {subtask_text!r}")

    if subtask_tokens.numel() > 0:
        subtask_masks = torch.ones(
            subtask_tokens.shape,
            dtype=lang_masks.dtype,
            device=lang_masks.device,
        )
        inputs[key_ids] = torch.cat([lang_tokens, subtask_tokens], dim=1)
        inputs[key_attn] = torch.cat([lang_masks, subtask_masks], dim=1)

    _ensure_no_fast_tokens(inputs, action_token_min, action_token_max)
    inputs[key_mask] = torch.zeros_like(inputs[key_ids], dtype=torch.bool)
    return subtask_tokens.shape[1], subtask_text


def _append_cached_subtask_tokens(
    inputs: dict,
    cached_subtask_tokens: Tensor,
) -> None:
    key_ids = f"{OBS_STR}.input_ids"
    key_attn = f"{OBS_STR}.attention_mask"
    key_mask = f"{OBS_STR}.fast_token_mask"

    cached_subtask_tokens = cached_subtask_tokens.to(device=inputs[key_ids].device)
    subtask_masks = torch.ones(
        cached_subtask_tokens.shape,
        dtype=inputs[key_attn].dtype,
        device=inputs[key_attn].device,
    )
    inputs[key_ids] = torch.cat([inputs[key_ids], cached_subtask_tokens], dim=1)
    inputs[key_attn] = torch.cat([inputs[key_attn], subtask_masks], dim=1)
    inputs[key_mask] = torch.zeros_like(inputs[key_ids], dtype=torch.bool)


@dataclass
class ArmEeStatus:
    msg: Optional[RobotStatus] = None
    stamp: float = 0.0


class KeyboardJogMonitor:
    def __init__(self, key_hold_timeout: float = 0.20):
        self.key_hold_timeout = key_hold_timeout
        self._lock = threading.Lock()
        self._old_settings = None
        self._thread = None
        self._stop_event = threading.Event()
        self._last_key: Optional[str] = None
        self._last_key_time = 0.0
        self._exit_requested = False
        self._reset_requested = False
        self.enabled = False

    def start(self) -> None:
        if self.enabled:
            return
        if not sys.stdin.isatty():
            print("stdin is not a TTY; keyboard manual EE jog is disabled.")
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        self.enabled = True
        print("Keyboard control: hold u/j for left z +/-5mm, i/k for right z +/-5mm; release to resume policy; e/q exit, r reset.")

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
        self._thread = None
        self._old_settings = None
        self.enabled = False

    def pause_for_prompt(self) -> None:
        self.stop()

    def resume_after_prompt(self) -> None:
        self.start()

    def wait_for_enter(self, prompt: str = "Press Enter to continue: ") -> None:
        if not sys.stdin.isatty():
            return
        self.pause_for_prompt()
        try:
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
            input(prompt)
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        finally:
            self.resume_after_prompt()

    def _read_loop(self) -> None:
        while not self._stop_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.02)
            if not readable:
                continue
            key = sys.stdin.read(1).lower()
            now = time.monotonic()
            with self._lock:
                if key in JOG_KEYS:
                    self._last_key = key
                    self._last_key_time = now
                elif key in STOP_KEYS:
                    self._last_key = None
                    self._last_key_time = 0.0
                elif key in EXIT_KEYS:
                    self._exit_requested = True
                elif key in RESET_KEYS:
                    self._reset_requested = True

    def active_jog_key(self) -> Optional[str]:
        with self._lock:
            if self._last_key is None:
                return None
            if time.monotonic() - self._last_key_time > self.key_hold_timeout:
                self._last_key = None
                self._last_key_time = 0.0
                return None
            return self._last_key

    def consume_exit_requested(self) -> bool:
        with self._lock:
            requested = self._exit_requested
            self._exit_requested = False
            return requested

    def consume_reset_requested(self) -> bool:
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested


class ManualEeJogController:
    def __init__(
        self,
        node,
        left_feedback_topic: str = "/arm_slave_l_status",
        right_feedback_topic: str = "/arm_slave_r_status",
        left_cmd_topic: str = "/arm_l_ee_status",
        right_cmd_topic: str = "/arm_r_ee_status",
        step_m: float = 0.005,
        status_timeout: float = 0.5,
        qos: int = 10,
    ):
        self.node = node
        self.step_m = step_m
        self.status_timeout = status_timeout
        self.left = ArmEeStatus()
        self.right = ArmEeStatus()
        self._lock = threading.Lock()
        self._last_warn_time = 0.0
        self._last_log_time = 0.0
        self._publish_count = 0
        self._last_command_gripper = {"left": None, "right": None}

        self.left_pub = node.create_publisher(PosCmd, left_cmd_topic, qos)
        self.right_pub = node.create_publisher(PosCmd, right_cmd_topic, qos)
        node.create_subscription(RobotStatus, left_feedback_topic, self._left_cb, qos)
        node.create_subscription(RobotStatus, right_feedback_topic, self._right_cb, qos)
        node.get_logger().info(
            f"manual EE jog: sub left={left_feedback_topic} right={right_feedback_topic}; "
            f"pub left={left_cmd_topic} right={right_cmd_topic}; step={step_m:.4f}m"
        )

    def _left_cb(self, msg: RobotStatus) -> None:
        with self._lock:
            self.left = ArmEeStatus(msg=msg, stamp=time.monotonic())

    def _right_cb(self, msg: RobotStatus) -> None:
        with self._lock:
            self.right = ArmEeStatus(msg=msg, stamp=time.monotonic())

    def publish_for_key(self, key: str) -> bool:
        with self._lock:
            left = self.left
            right = self.right
        if key == LEFT_UP_KEY:
            return self._publish_jog("left", left, self.left_pub, 1.0)
        if key == LEFT_DOWN_KEY:
            return self._publish_jog("left", left, self.left_pub, -1.0)
        if key == RIGHT_UP_KEY:
            return self._publish_jog("right", right, self.right_pub, 1.0)
        if key == RIGHT_DOWN_KEY:
            return self._publish_jog("right", right, self.right_pub, -1.0)
        return False

    def _publish_jog(self, side: str, status: ArmEeStatus, publisher, direction: float) -> bool:
        now = time.monotonic()
        if status.msg is None:
            self._warn(f"manual EE jog skipped: no {side} feedback yet")
            return False
        age = now - status.stamp
        if age > self.status_timeout:
            self._warn(f"manual EE jog skipped: {side} feedback stale ({age:.3f}s)")
            return False

        end_pos = list(status.msg.end_pos)
        if len(end_pos) < 6:
            self._warn(f"manual EE jog skipped: {side} end_pos length {len(end_pos)} < 6")
            return False

        gripper, gripper_source = self._current_gripper(side, status.msg)
        if gripper is None:
            self._warn(f"manual EE jog skipped: {side} last-command gripper unavailable")
            return False

        cmd = PosCmd()
        cmd.x = float(end_pos[0])
        cmd.y = float(end_pos[1])
        cmd.z = float(end_pos[2]) + direction * self.step_m
        cmd.roll = float(end_pos[3])
        cmd.pitch = float(end_pos[4])
        cmd.yaw = float(end_pos[5])
        cmd.gripper = gripper
        cmd.time_count = int(time.time() * 1000) & 0x7FFFFFFF
        self._fill_quaternion(cmd)
        publisher.publish(cmd)

        self._publish_count += 1
        if now - self._last_log_time >= 1.0:
            self.node.get_logger().info(
                f"manual EE jog published {self._publish_count} commands; "
                f"{side} z {float(end_pos[2]):.4f} -> {cmd.z:.4f}; "
                f"gripper={cmd.gripper:.4f} ({gripper_source})"
            )
            self._last_log_time = now
        return True

    def update_command_grippers(self, left_gripper: float, right_gripper: float) -> None:
        self._last_command_gripper["left"] = float(left_gripper)
        self._last_command_gripper["right"] = float(right_gripper)

    def _current_gripper(self, side: str, msg: RobotStatus) -> tuple[Optional[float], str]:
        del msg
        # Feedback gripper is unreliable on this setup; preserve the gripper
        # by replaying the last gripper value we sent through reset or policy.
        cached_command = self._last_command_gripper.get(side)
        if cached_command is not None:
            return cached_command, "last_command"
        return None, "missing"

    def _fill_quaternion(self, cmd: PosCmd) -> None:
        cr = math.cos(cmd.roll * 0.5)
        sr = math.sin(cmd.roll * 0.5)
        cp = math.cos(cmd.pitch * 0.5)
        sp = math.sin(cmd.pitch * 0.5)
        cy = math.cos(cmd.yaw * 0.5)
        sy = math.sin(cmd.yaw * 0.5)
        cmd.quater_w = cr * cp * cy + sr * sp * sy
        cmd.quater_x = sr * cp * cy - cr * sp * sy
        cmd.quater_y = cr * sp * cy + sr * cp * sy
        cmd.quater_z = cr * cp * sy - sr * sp * cy

    def _warn(self, text: str) -> None:
        now = time.monotonic()
        if now - self._last_warn_time >= 1.0:
            self.node.get_logger().warn(text)
            self._last_warn_time = now


class DeployACOne:
    def __init__(self, config, in_collect=False):
        rclpy.init()
        self.ros_operator = RosOperator(config, in_collect=in_collect)
        self.rate = Rate(config.frame_rate)
        self.config = config
        self.ee_jog = ManualEeJogController(self.ros_operator)
        spin_thread = threading.Thread(target=rclpy.spin, args=(self.ros_operator,), daemon=True)
        spin_thread.start()

    def get_observation(self):
        while True and rclpy.ok():
            obs_dict = self.ros_operator.get_observation()
            if not obs_dict:
                print("sync fail")
                self.rate.sleep()
                continue
            obs_dict.update({
                "image_head": obs_dict['images']['head'],  # (360, 640, 3)
                "image_left": obs_dict['images']['left_wrist'],  # (480, 640, 3)
                "image_right": obs_dict['images']['right_wrist'],  # (480, 640, 3)
            })
            return obs_dict

    def _wait_for_enter(self, prompt: str = "Enter any key to continue: ") -> None:
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)
        input(prompt)
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin, termios.TCIFLUSH)

    def reset(self):
        left  = [0, 0, 0, 0, 0, 0, -5]
        right = [0, 0, 0, 0, 0, 0, -5]
        self.ee_jog.update_command_grippers(left[-1], right[-1])
        self.ros_operator.follow_arm_publish_continuous(left, right)
        self._wait_for_enter()

    def sleep(self):
        self.rate.sleep()

    def step(self, action):
        left_action = action[0:7]
        right_action = action[7:14]
        self.ee_jog.update_command_grippers(left_action[-1], right_action[-1])
        self.ros_operator.follow_arm_publish(left_action, right_action)

    def manual_ee_jog(self, key):
        self.ee_jog.publish_for_key(key)

def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    acone_config_path = "deploy/src/config/acone_ros2_config.yaml"
    acone_config = OmegaConf.load(acone_config_path)
    acone = DeployACOne(acone_config, in_collect=False)

    # task = "Zip the bag"
    # ckpt_path = Path("")
    # task = "Unscrew the cap"
    # ckpt_path = Path("")
    # task = "fold_the_airplane_box_blank"
    # task = "real three fold v2"
    # task = "Using the left arm, pick the orange tube to left box."
    # task = "Transfer the orange to the left box."
    # task = "Place the blue tube in the right box."
    # ckpt_path = Path("/home/pjlab/caijh/qwen3_5_vqa/lerobot_lab/outputs/qwen3_5vla_ki_wan/2026_06_10_13_18_55-qwen3_5vla_ki_wan-robotwin-delta-pretrain-600k-novideo-test-tube-sorting-stride2/checkpoints/060000/pretrained_model")
    # ckpt_path = Path("/home/pjlab/caijh/qwen3_5_vqa/lerobot_lab/outputs/qwen3_5vla_ki_wan/2026_06_11_14_09_30-qwen3_5vla_ki_wan-delta-pretrain-600k-novideo-test-tube-sorting-recover-clutter/checkpoints/060000/pretrained_model")
    # ckpt_path = Path("/home/pjlab/caijh/qwen3_5_vqa/lerobot_lab/outputs/qwen3_5svla_ki_wan/2026_06_10_13_27_34-qwen3_5vla_ki_wan-robotwin-delta-pretrain-600k-novideo-test-tube-sorting-stride2-bl-or/checkpoints/060000/pretrained_model")
    
    # task = "Hole 2: insert blue tube."
    # ckpt_path = Path("/home/pjlab/caijh/qwen3_5_vqa/lerobot_lab/outputs/qwen3_5vla_ki_wan/2026_06_17_16_53_49-qwen3_5vla_ki_wan-delta-pretrain-600k-insert-test-tube-iron-stand-new-mode-nob2o4/checkpoints/040000/pretrained_model")
    
    task = "Complete_chemical_reaction_experiment"
    ckpt_path = Path("/home/pjlab/caijh/qwen3_5_vqa/lerobot_lab/outputs/qwen3_5vla_ki_wan/2026_06_22_08_12_23-qwen3_5vla_ki_wan-delta-pretrain-600k-Complete_chemical_reaction_experiment-with-video/checkpoints/060000/pretrained_model")
    ckpt_name = ckpt_path.parts[2]
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.compile_model = False
    config.compile_mode = "reduce-overhead"
    action_chunk = deque(maxlen=config.chunk_size)
    action_mode = 'delta'  # abs | delta
    # dtype = torch.bfloat16
    dtype = torch.float32
    assert isinstance(config, Qwen3_5VLAKIWanConfig)
    policy = Qwen3_5VLAKIWanFastPolicy.from_pretrained(
        config=config,
        pretrained_name_or_path=ckpt_path,
    )
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"Qwen3_5 params: {sum(p.numel() for p in policy.model.qwen3_5_with_expert.qwen3_5.parameters()) / 1e9:.2f}B")
    print(f"Qwen3_5_Expert params: {sum(p.numel() for p in policy.model.qwen3_5_with_expert.action_expert.parameters()) / 1e9:.2f}B")
    policy.cuda()
    policy.to(dtype)
    policy.eval()
    debug_tokenizer = AutoTokenizer.from_pretrained(config.vlm_model_name_or_path)

    logger.info("policy warmup ... ")
    dummy_inputs = {
        f"{OBS_IMAGES}.image0": torch.rand((1, 3, 224, 224), dtype=dtype).cuda(),
        f"{OBS_IMAGES}.image1": torch.rand((1, 3, 224, 224), dtype=dtype).cuda(),
        f"{OBS_IMAGES}.image2": torch.rand((1, 3, 224, 224), dtype=dtype).cuda(),
        f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(),
        f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(),
        f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(),
        f"{OBS_STR}.pixel_values": torch.rand((1, 768, 1536), dtype=dtype).cuda(),
        f"{OBS_STR}.image_grid_thw": torch.tensor([[[1, 16, 16]] * 3]).cuda(),
        f"{OBS_STR}.input_ids": torch.tensor([[248056] * 192 + [777] * (48 + 6)]).cuda(),
        f"{OBS_STR}.attention_mask": torch.tensor([[1] * 246]).cuda(),
        OBS_STATE: torch.rand((1, 16), dtype=dtype).cuda(),
        ACTION: torch.rand((1, 50, 16), dtype=dtype).cuda(),
        "task": ["dummy sample"],
    }
    with torch.no_grad():
        policy.predict_action_chunk(dummy_inputs)
        policy.predict_action_chunk(dummy_inputs)

    stats = load_json(ckpt_path / "stats.json")["arx_lift2"]
    stat_keys = ['min', 'max', 'mean', 'std']
    state_stat = {
        stat_key: np.concatenate([
            stats["states.left_joint.position"][stat_key],
            stats["states.left_gripper.position"][stat_key],
            stats["states.right_joint.position"][stat_key],
            stats["states.right_gripper.position"][stat_key],
        ], axis=-1) for stat_key in stat_keys
    }
    action_stat = {
        stat_key: np.concatenate([
            stats["actions.left_joint.position"][stat_key],
            stats["actions.left_gripper.position"][stat_key],
            stats["actions.right_joint.position"][stat_key],
            stats["actions.right_gripper.position"][stat_key],
        ], axis=-1) for stat_key in stat_keys
    }
    # state_stat = {
    #     stat_key: np.asarray(stats["observation.state"][stat_key]) for stat_key in stat_keys
    # }
    
    # action_stat = {
    #     stat_key: np.asarray(stats["action"][stat_key]) for stat_key in stat_keys
    # }

    state_stat = {"observation.state": state_stat}
    action_stat = {"action": action_stat}
    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=["action"],
        norm_stats=action_stat,
    )

    schema = get_schema("arx_lift2")

    input_transforms = compose([
        # ResizeShortestCenterCropFn(height=448, width=448),
        ResizeImagesWithPadFn(height=224, width=224, mapping=schema.image_mapping),
        RemapImageKeyTransformFn(mapping=schema.image_mapping),
        NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
        Qwen3_5KIChatProcessorTransformFnV2(mode="eval"),
        PadStateTransformFn(
            max_state_dim=32,
        ),
    ])

    subtask_cache_tokens = None
    subtask_cache_uses_left = 0
    subtask_refresh_interval = 3

    print("acone reset !")
    acone.reset()
    keyboard = KeyboardJogMonitor()
    keyboard.start()
    subtask = ""
    manual_was_active = False
    try:
        while True:
            if keyboard.consume_exit_requested():
                print("Early exit requested. Breaking current loop.")
                break
            if keyboard.consume_reset_requested():
                print("Resetting robot arm positions...")
                keyboard.pause_for_prompt()
                acone.reset()
                keyboard.resume_after_prompt()
                action_chunk.clear()
                subtask_cache_tokens = None
                subtask_cache_uses_left = 0
                manual_was_active = False
                continue

            jog_key = keyboard.active_jog_key()
            if jog_key is not None:
                if not manual_was_active:
                    print("Manual EE jog active; clear policy actions and pause inference.")
                    action_chunk.clear()
                    subtask_cache_tokens = None
                    subtask_cache_uses_left = 0
                    manual_was_active = True
                acone.manual_ee_jog(jog_key)
                acone.sleep()
                continue
            if manual_was_active:
                print("Manual EE jog released; resume policy inference from fresh observation.")
                action_chunk.clear()
                subtask_cache_tokens = None
                subtask_cache_uses_left = 0
                manual_was_active = False

            start_time = time.perf_counter()
            if len(action_chunk) == 0:
                print('predict new action chunk')
                obs = acone.get_observation()
                init_action = torch.as_tensor(obs['qpos'][None]).contiguous().cuda()
                # Modify for right gripper zeropoint shifting
                print(f"Original right gripper position: {obs['qpos'][13]:.4f}")

                # obs['qpos'][13] += 6.6
                sample = {
                    f"images.rgb.head": torch.as_tensor(obs["image_head"].copy()).contiguous().cuda().to(dtype) / 255.0,
                    f"images.rgb.hand_left": torch.as_tensor(obs["image_left"].copy()).contiguous().cuda().to(dtype) / 255.0,
                    f"images.rgb.hand_right": torch.as_tensor(obs["image_right"].copy()).contiguous().cuda().to(dtype) / 255.0,
                    "observation.state": torch.as_tensor(obs['qpos']).contiguous().cuda(),
                    "task": task,
                }
                for key in sample.keys():
                    if "images" in key:
                        image = sample[key].permute(2, 0, 1)
                        sample[key] = image
                sample = input_transforms(sample)
                inputs = {}
                for key in sample.keys():
                    if key == 'task':
                        inputs[key] = [sample[key]]
                    elif sample[key].dtype == torch.int64 or sample[key].dtype == torch.bool:
                        inputs[key] = sample[key][None].cuda()
                    else:
                        inputs[key] = sample[key][None].cuda().to(dtype=dtype)
                sa = time.perf_counter()
                with torch.no_grad():
                    if subtask_cache_tokens is None or subtask_cache_uses_left <= 0:
                        prompt_len = inputs[f"{OBS_STR}.input_ids"].shape[1]
                        num_subtask_tokens, subtask_text = _append_subtask_until_first_fast_token(
                            policy,
                            inputs,
                            action_token_min=config.action_token_min,
                            action_token_max=config.action_token_max,
                            debug_tokenizer=debug_tokenizer,
                        )
                        subtask = subtask_text
                        subtask_cache_tokens = inputs[f"{OBS_STR}.input_ids"][:, prompt_len:].clone()
                        subtask_cache_uses_left = subtask_refresh_interval
                        print(f"  refreshed subtask cache: {num_subtask_tokens} tokens")
                    else:
                        _append_cached_subtask_tokens(inputs, subtask_cache_tokens)
                        print(f"  reuse cached subtask, uses left before this chunk: {subtask_cache_uses_left}")

                    action_pred = policy.predict_action_chunk(inputs)[0, :, :16]
                    subtask_cache_uses_left -= 1
                    action_pred = torch.cat([action_pred[:, :6], action_pred[:, 7:8], action_pred[:, 8:14], action_pred[:, 15:16]], dim=1)
                    
                    action_pred = unnormalize_fn({"action": action_pred})["action"]
                    if action_mode == 'delta':
                        init_action[:, 6] = 0.0
                        init_action[:, 13] = 0.0
                        action_pred += init_action
                    action_chunk.extend(list(action_pred.unbind(dim=0)))

                ea = time.perf_counter()
                print(f"elapse time: {ea - sa:.4f}s")

                if keyboard.active_jog_key() is not None:
                    print("Manual EE jog requested during inference; discard predicted chunk.")
                    action_chunk.clear()
                    subtask_cache_tokens = None
                    subtask_cache_uses_left = 0
                    continue

            if keyboard.active_jog_key() is not None:
                continue

            action = action_chunk.popleft()
            action = action.to(torch.float32).cpu().numpy()

            # fold the airplane box
            thresh = -0.8
            if "cylinder" in subtask:
                thresh = -2
            if "round-bottom" in subtask:
                thresh = -1.16
            if "stopper" in subtask:
                thresh = -1
            if "Insert the funnel" in subtask:
                thresh = -0.6
            action[6] = 0 if action[6] > thresh else action[6]
            action[13] = 0 if action[13] > thresh else action[13]

            # Modify for right gripper zeropoint shifting
            # action[13] -= 6.6
            acone.step(action)
            end_time = time.perf_counter()
            elapse_time = end_time - start_time
            time.sleep(max(0, 1 / 30 - elapse_time))
    finally:
        keyboard.stop()
    print("End.")


if __name__ == "__main__":
    main()
