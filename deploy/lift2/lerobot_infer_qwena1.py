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
from lerobot.policies.qwena1 import QwenA1Config, QwenA1Policy
from lerobot.datasets.utils import write_json, load_json
from lerobot.datasets.factory import make_dataset
from lerobot.transforms.core import (
    ResizeImagesWithPadFn, 
    ResizeShortestCenterCropFn, 
    NormalizeTransformFn, 
    UnNormalizeTransformFn, 
    ToTensorTransformFn, 
    compose, 
)
from lerobot.utils.constants import OBS_IMAGES, OBS_STR, OBS_STATE, ACTION
from lerobot.policies.qwena1.transform_qwena1 import Qwen3_VLProcessorTransformFn

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

    task = "Sort the garbage on the desktop into recyclable and non-recyclable"
    ckpt_path = Path(f"outputs/qwena1/2025_12_14_17_12_50-qwena1-lift2_real_Sort_the_garbage_on_the_desktop_into_recyclable_and_non-recyclable-delta-28l-scratch_60k-060000/pretrained_model")
    # ckpt_path = Path(f"outputs/qwena1/2025_12_15_01_59_20-qwena1-lift2_real_Sort_the_garbage_on_the_desktop_into_recyclable_and_non-recyclable-delta-28l-pretrain_100k-finetune_60k-060000/pretrained_model")
    
    # task = "Bring the shipping label of the package into view, then grasp the package from the conveyor belt and orient the label to myself"
    # ckpt_path = Path(f"outputs/2025_12_08_15_40_29-qwenvla-lift2_real_Bring_the_shipping_label_of_the_package_into_view__then_grasp_the_package_from_the_conveyor_belt_and_orient_the_label_to_myself-delta-28l-pretrain_240k-finetune_60k-060000/pretrained_model")
    ckpt_name = ckpt_path.parts[2]
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.compile_model = True
    config.compile_mode = "reduce-overhead"
    action_chunk = deque(maxlen=config.chunk_size)
    action_mode = 'delta'  # abs | delta
    # dtype = torch.bfloat16
    dtype = torch.float32
    decode_image_flag = False
    warmup_flag = True
    assert isinstance(config, QwenA1Config)
    policy = QwenA1Policy.from_pretrained(
        config=config, 
        pretrained_name_or_path=ckpt_path, 
    )
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"Und params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.und_expert.parameters()) / 1e9:.2f}B")
    print(f"Gen params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.gen_expert.parameters()) / 1e9:.2f}B")
    print(f"Act params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.act_expert.parameters()) / 1e9:.2f}B")
    policy.cuda()
    policy.to(dtype)
    policy.eval()

    logger.info("policy warmup ... ")
    
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

    input_transforms = compose([
        # ResizeShortestCenterCropFn(height=448, width=448), 
        ResizeImagesWithPadFn(height=224, width=224), 
        Qwen3_VLProcessorTransformFn(), 
        NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
    ])

    head_color_list = []
    hand_left_color_list = []
    hand_right_color_list = []
    image_history_interval = 15

    lift2.reset()
    while True:
        start_time = time.perf_counter()
        obs = lift2.get_observation()
        head_color_list.append(torch.as_tensor(obs["image_head"].copy()).contiguous().cuda().to(dtype) / 255.0)
        hand_left_color_list.append(torch.as_tensor(obs["image_left"].copy()).contiguous().cuda().to(dtype) / 255.0)    
        hand_right_color_list.append(torch.as_tensor(obs["image_right"].copy()).contiguous().cuda().to(dtype) / 255.0)
        while len(head_color_list) > image_history_interval+1:
            head_color_list.pop(0)
            hand_left_color_list.pop(0)
            hand_right_color_list.pop(0)
        if len(action_chunk) == 0:
            print('predict new action chunk')
            sa = time.perf_counter()
            init_action = torch.as_tensor(obs['qpos'][None]).contiguous().cuda()
            past_idx = max(len(head_color_list) - image_history_interval-1, 0)
            image_head_with_history = torch.stack([head_color_list[past_idx], head_color_list[-1]], dim=0)
            image_hand_left_with_history = torch.stack([hand_left_color_list[past_idx], hand_left_color_list[-1]], dim=0)
            image_hand_right_with_history = torch.stack([hand_right_color_list[past_idx], hand_right_color_list[-1]], dim=0)
            sample = {
                f"{OBS_IMAGES}.image0": image_head_with_history, 
                f"{OBS_IMAGES}.image1": image_hand_left_with_history, 
                f"{OBS_IMAGES}.image2": image_hand_right_with_history, 
                "observation.state": torch.as_tensor(obs['qpos']).contiguous().cuda(), 
                "task": task, 
            }
            for key in sample.keys():
                if "images" in key:
                    image = sample[key].permute(0, 3, 1, 2)
                    sample[key] = image
                    # sample[key] = torch.nn.functional.interpolate(image, (480, 640), mode='bilinear', align_corners=False)[0]
            sample = input_transforms(sample)
            inputs = {}
            for key in sample.keys():
                if key == 'task':
                    inputs[key] = [sample[key]]
                elif sample[key].dtype == torch.int64:
                    inputs[key] = sample[key][None].cuda()
                else:
                    inputs[key] = sample[key][None].cuda().to(dtype=dtype)
            inputs.update({
                    f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(), 
                    f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(), 
                    f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(), 
                })
            # import pdb; pdb.set_trace()
            with torch.no_grad():
                if warmup_flag:
                    for warmup_step in range(5):
                        print(f"warmup step: {warmup_step}")
                        action_pred, img_pred = policy.predict_action_chunk(inputs, decode_image=decode_image_flag)
                    warmup_flag = False
                action_pred, img_pred = policy.predict_action_chunk(inputs, decode_image=decode_image_flag)
                action_pred = action_pred[0, :, :14].clone()
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
