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

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.qwenvla import QwenVLAConfig, QwenVLAPolicy
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
from lerobot.policies.qwenvla.transform_qwenvla import Qwen3_VLProcessorTransformFn

from deploy.src.acone.ros2_operator import RosOperator, Rate
import rclpy


class DeployACOne:
    def __init__(self, config, in_collect=False):
        rclpy.init()
        self.ros_operator = RosOperator(config, in_collect=in_collect)
        self.rate = Rate(config.frame_rate)
        self.config = config
        spin_thread = threading.Thread(target=rclpy.spin, args=(self.ros_operator,), daemon=True)
        spin_thread.start()


    def reset(self):
        left1 = [0,  0, 0, 0, 0,  0, -5]
        right1 = [0,  0, 0, 0, 0,  0, -5]

        self.ros_operator.follow_arm_publish_continuous(left1, right1)

        input("Enter any key to continue: ")

    def get_observation(self):
        global obs_dict

        while True and rclpy.ok():
            obs_dict = self.ros_operator.get_observation()
            if not obs_dict:
                print("syn fail")
                self.rate.sleep()
                continue

            obs_dict.update({
                "image_head": obs_dict['images']['head'],  # (360, 640, 3)
                "image_left": obs_dict['images']['left_wrist'],  # (480, 640, 3)
                "image_right": obs_dict['images']['right_wrist'],  # (480, 640, 3)
            })

            return obs_dict

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

    # task = "Sort the garbage on the desktop into recyclable and non-recyclable"
    # ckpt_path = Path(f"outputs/2025_12_08_15_40_41-qwenvla-lift2_real_Sort_the_garbage_on_the_desktop_into_recyclable_and_non-recyclable-delta-28l-pretrain_240k-finetune_60k-060000/pretrained_model")
    # task = "Sort the garbage on the desktop into recyclable and non-recyclable"
    # ckpt_path = Path(f"outputs/qwenvla/2025_11_13_22_51_22-qwenvla-lift2_sort_garbage-delta-28l/pretrained_model")
    # task = "Bring the shipping label of the package into view, then grasp the package from the conveyor belt and orient the label to myself"
    # ckpt_path = Path(f"/home/pjlab/hongrui_workspace/qwenvla/finetune/2025_12_13_06_31_29-qwenvla-_oss-zhuhongrui_dataset_lerobot_v3_arx_lift2_x5_Green_bag-delta-28l-pretrain_240k-finetune_60k/checkpoints/060000/pretrained_model")
    
    # task = "Zip the bag"
    # ckpt_path = Path("/home/pjlab/hongrui_workspace/qwenvla/finetune/2025_12_26_09_30_44-qwenvla-acone_real_Zip_Bag-delta-28l-pretrain_500k-finetune_60k-060000/pretrained_model")

    task = "Pour the drink"
    ckpt_path = Path("/home/pjlab/hongrui_workspace/qwenvla/finetune/2025_12_26_09_30_46-qwenvla-acone_real_Unscrew_Cap-delta-28l-pretrain_500k-finetune_60k-060000/pretrained_model")

    ckpt_name = ckpt_path.parts[2]
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.compile_model = True
    config.compile_mode = "reduce-overhead"
    action_chunk = deque(maxlen=config.chunk_size)
    action_mode = 'delta'  # abs | delta
    # dtype = torch.bfloat16
    dtype = torch.float32
    assert isinstance(config, QwenVLAConfig)
    policy = QwenVLAPolicy.from_pretrained(
        config=config, 
        pretrained_name_or_path=ckpt_path, 
    )
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"Qwen3_VL params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.qwen3_vl.parameters()) / 1e9:.2f}B")
    print(f"Qwen3_Expert params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.qwen3_expert.parameters()) / 1e9:.2f}B")
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
        f"{OBS_STR}.input_ids": torch.tensor([[151655] * 192 + [777] * (48 + 6)]).cuda(), 
        f"{OBS_STR}.attention_mask": torch.tensor([[1] * 246]).cuda(), 
        OBS_STATE: torch.rand((1, 14), dtype=dtype).cuda(), 
        ACTION: torch.rand((1, 50, 14), dtype=dtype).cuda(), 
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

    print("acone reset !")
    acone.reset()
    while True:
        start_time = time.perf_counter()
        if len(action_chunk) == 0:
            print('predict new action chunk')
            sa = time.perf_counter()
            obs = acone.get_observation()
            init_action = torch.as_tensor(obs['qpos'][None]).contiguous().cuda()
            sample = {
                f"{OBS_IMAGES}.image0": torch.as_tensor(obs["image_head"].copy()).contiguous().cuda().to(dtype) / 255.0, 
                f"{OBS_IMAGES}.image1": torch.as_tensor(obs["image_left"].copy()).contiguous().cuda().to(dtype) / 255.0, 
                f"{OBS_IMAGES}.image2": torch.as_tensor(obs["image_right"].copy()).contiguous().cuda().to(dtype) / 255.0, 
                "observation.state": torch.as_tensor(obs['qpos']).contiguous().cuda(), 
                "task": task, 
            }
            for key in sample.keys():
                if "images" in key:
                    image = sample[key].permute(2, 0, 1)
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
                action_pred = policy.predict_action_chunk(inputs)[0, :, :14]
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

        # # Zip the bag
        # action[6] = 0 if action[6] > -1.5 else action[6]
        # action[13] = 0 if action[13] > -1.5 else action[13]

        # Pour the drink
        action[6] = -1.5 if action[6] > -2.3 else -3.3
        action[13] = -0.5 if action[13] > -1.7 else -3.3

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
                    break
                elif user_key == 'r':
                    print('Resetting robot arm positions...')
                    acone.reset()
                    action_chunk.clear()
    print("End.")


if __name__ == "__main__":
    main()
