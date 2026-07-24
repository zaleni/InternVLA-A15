import time
import copy
import sys
import select
import pdb
import logging
import threading

from pprint import pp
from pathlib import Path
from dataclasses import replace
from collections import deque
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


class DeployACOne:
    def __init__(self, config, in_collect=False):
        rclpy.init()
        self.ros_operator = RosOperator(config, in_collect=in_collect)
        self.rate = Rate(config.frame_rate)
        self.config = config
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

    def reset(self):
        left  = [0, 0, 0, 0, 0, 0, -5]
        right = [0, 0, 0, 0, 0, 0, -5]
        self.ros_operator.follow_arm_publish_continuous(left, right)
        input("Enter any key to continue: ")

    def sleep(self):
        self.rate.sleep()

    def step(self, action):
        left_action = action[0:7]
        right_action = action[7:14]
        self.ros_operator.follow_arm_publish(left_action, right_action)

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
    subtask_refresh_interval = 5

    print("acone reset !")
    acone.reset()
    subtask = ""
    while True:
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

        action = action_chunk.popleft()
        action = action.to(torch.float32).cpu().numpy()

        # fold the airplane box
        thresh = -0.8
        if "cylinder" in subtask:
            thresh = -2
        if "round-bottom" in subtask:
            thresh = -1.3
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

        if sys.stdin.isatty():
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if readable:
                user_key = sys.stdin.readline().strip().lower()
                if user_key == 'e':
                    print("Early exit requested. Breaking current loop.")
                    acone.reset()
                    action_chunk.clear()
                    subtask_cache_tokens = None
                    subtask_cache_uses_left = 0
                    break
                elif user_key == 'r':
                    print('Resetting robot arm positions...')
                    acone.reset()
                    action_chunk.clear()
                    subtask_cache_tokens = None
                    subtask_cache_uses_left = 0
    print("End.")


if __name__ == "__main__":
    main()
