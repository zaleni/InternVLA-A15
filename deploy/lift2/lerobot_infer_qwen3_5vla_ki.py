import time
import copy
import sys
import select
import pdb
import logging

from pprint import pp
from pathlib import Path
from dataclasses import replace
from collections import deque
from omegaconf import OmegaConf
from pdb import set_trace

import torch
import numpy as np

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.qwen3_5vla_ki import Qwen3_5VLAKIConfig, Qwen3_5VLAKIPolicy
from lerobot.policies.qwen3_5vla_ki.modeling_qwen3_5vla_ki_fast import Qwen3_5VLAKIFastPolicy
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
from lerobot.policies.qwen3_5vla_ki.transform_qwen3_5vla_ki import Qwen3_5KIChatProcessorTransformFn

import rospy

from deploy.src.lift2.ros_operator import RosOperator 


class DeployLIFT2:
    def __init__(self, config, in_collect=False):
        self.ros_operator = RosOperator(config, in_collect=in_collect)
        self.rate = rospy.Rate(config.frame_rate)
        self.count = 0
    
    def get_observation(self):
        while True and not rospy.is_shutdown():
            obs_dict = self.ros_operator.get_observation(self.count)
            if not obs_dict:
                print("sync fail")
                self.rate.sleep()
                continue
            obs_dict.update({
                "image_head": obs_dict['images']['head'],  # (360, 640, 3)
                "image_left": obs_dict['images']['left_wrist'],  # (480, 640, 3)
                "image_right": obs_dict['images']['right_wrist'],  # (480, 640, 3)
            })
            self.count += 1
            return obs_dict
    
    def reset(self):
        left  = [0, 0, 0, 0, 0, 0, 5]
        right = [0, 0, 0, 0, 0, 0, 5]
        self.ros_operator.follow_arm_publish_continuous(left, right)
        self.count = 0
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

    lift2_config_path = "deploy/src/config/lift2_ros_config.yaml"
    lift2_config = OmegaConf.load(lift2_config_path)
    lift2 = DeployLIFT2(lift2_config, in_collect=False)

    # task = "Sort the garbage on the desktop into recyclable and non-recyclable"
    # task = "Don't Move, go to the home!"
    # ckpt_path = Path(f"outputs/real-world/2026_05_16_11_49_26-qwen3_5vla_ki-stage2-30w-lift2-real-sort-garbage/checkpoints/060000/pretrained_model")
    # ckpt_path = Path(f"outputs/real-world/2026_05_15_21_33_24-qwen3_5vla_ki-stage2-30w-onlyfm-lift2-real-sort-garbage/checkpoints/060000/pretrained_model")
    task = "Make a beef sandwich"
    # ckpt_path = Path(f"outputs/real-world/2026_05_17_05_37_22-qwen3_5vla_ki-lift2-makesandwitch-delta-stage2-40w-onlyfm/checkpoints/060000/pretrained_model")
    ckpt_path = Path(f"outputs/real-world/2026_05_17_22_40_38-qwen3_5vla_ki-stage2-30w-lift2-real-make_sandwich/checkpoints/060000/pretrained_model")
    # task = "Bring the shipping label of the package into view, then grasp the package from the conveyor belt and orient the label to myself"
    # ckpt_path = Path(f"outputs/2025_12_08_15_40_29-qwenvla-lift2_real_Bring_the_shipping_label_of_the_package_into_view__then_grasp_the_package_from_the_conveyor_belt_and_orient_the_label_to_myself-delta-28l-pretrain_240k-finetune_60k-060000/pretrained_model")
    ckpt_name = ckpt_path.parts[2]
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.compile_model = False
    config.compile_mode = "reduce-overhead"
    action_chunk = deque(maxlen=config.chunk_size)
    action_mode = 'delta'  # abs | delta
    dtype = torch.bfloat16
    # dtype = torch.float32
    assert isinstance(config, Qwen3_5VLAKIConfig)
    policy = Qwen3_5VLAKIFastPolicy.from_pretrained(
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
        policy.predict_action_chunk(dummy_inputs, action_type="fm")
        policy.predict_action_chunk(dummy_inputs, action_type="fm")
    
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
    state_stat = {"observation.state": state_stat}
    action_stat = {
        stat_key: np.concatenate([
            stats["actions.left_joint.position"][stat_key], 
            stats["actions.left_gripper.position"][stat_key], 
            stats["actions.right_joint.position"][stat_key], 
            stats["actions.right_gripper.position"][stat_key],
        ], axis=-1) for stat_key in stat_keys
    }
    action_stat = {"action": action_stat}
    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=["action"], 
        mode="mean_std", 
        norm_stats=action_stat, 
    )

    schema = get_schema("arx_lift2")

    input_transforms = compose([
        # ResizeShortestCenterCropFn(height=448, width=448), 
        ResizeImagesWithPadFn(height=224, width=224, mapping=schema.image_mapping), 
        RemapImageKeyTransformFn(mapping=schema.image_mapping),
        NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
        Qwen3_5KIChatProcessorTransformFn(mode="eval"), 
        PadStateTransformFn(
            max_state_dim=32,
        ),
        # ReorderStateActionTransform(state_reorder=schema.state_reorder),
    ])

    lift2.reset()
    while True:
        start_time = time.perf_counter()
        if len(action_chunk) == 0:
            print('predict new action chunk')
            # sa = time.perf_counter()
            obs = lift2.get_observation()
            init_action = torch.as_tensor(obs['qpos'][None]).contiguous().cuda()
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
                action_pred = policy.predict_action_chunk(inputs, action_type="fm")[0, :, :16]
                action_pred = torch.cat([action_pred[:, :6], action_pred[:, 7:8], action_pred[:, 8:14], action_pred[:, 15:16]], dim=1)
                # action_pred = policy.predict_action_chunk(inputs, action_type="fast")[0, :, :14]
                action_pred = unnormalize_fn({"action": action_pred})["action"]
                if action_mode == 'delta':
                    init_action[:, 6] = 0.0
                    init_action[:, 13] = 0.0
                    action_pred += init_action.to(device=action_pred.device)
                action_chunk.extend(list(action_pred.unbind(dim=0)))

            ea = time.perf_counter()
            print(f"elapse time: {ea - sa:.4f}s")

        action = action_chunk.popleft()
        action = action.to(torch.float32).cpu().numpy()
        action[6] = 4.9 if action[6] > 2.5 else 0
        action[13] = 4.9 if action[13] > 2.5 else 0
        lift2.step(action)
        end_time = time.perf_counter()
        elapse_time = end_time - start_time
        time.sleep(max(0, 1 / 30 - elapse_time))

        if sys.stdin.isatty():
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if readable:
                user_key = sys.stdin.readline().strip().lower()
                if user_key == 'e':
                    print("Early exit requested. Breaking current loop.")
                    lift2.reset()
                    action_chunk.clear()
                    break
                elif user_key == 'r':
                    print('Resetting robot arm positions...')
                    lift2.reset()
                    action_chunk.clear()
    print("End.")


if __name__ == "__main__":
    main()
