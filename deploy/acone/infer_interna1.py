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
from lerobot.policies.internvla_a1 import InternVLAA1Config, InternVLAA1Policy
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
from lerobot.policies.internvla_a1.transform_a1 import InternVL3TokenizerTransformFn

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
        left1 = [0, 0, 0, 0, 0,  0, -5]
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

    # ckpt_path = Path("/home/pjlab/hongrui_workspace/interna1/finetune/a1_2b-unscrew_cap-ft_60k-2025-12-26-05-53/checkpoints/060000/pretrained_model")
    # task = "Pour the drink"

    ckpt_path = Path("/home/pjlab/hongrui_workspace/interna1/finetune/a1_2b-zip_bag-ft_60k-2025-12-26-05-35/checkpoints/060000/pretrained_model")
    task = "Zip the bag"
    
    ckpt_name = ckpt_path.parts[2]
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.compile_model = True
    config.compile_mode = "reduce-overhead"
    action_chunk = deque(maxlen=config.chunk_size)
    exec_steps = 50
    action_mode = 'delta'  # abs | delta
    dtype = torch.bfloat16
    dtype = torch.float32
    decode_image_flag = False
    warmup_flag = True
    assert isinstance(config, InternVLAA1Config)
    policy = InternVLAA1Policy.from_pretrained(
        config=config, 
        pretrained_name_or_path=ckpt_path, 
        strict=True,
    )
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"Und params: {sum(p.numel() for p in policy.model.internvl_with_expert.und_expert.parameters()) / 1e9:.2f}B")
    print(f"Gen params: {sum(p.numel() for p in policy.model.internvl_with_expert.gen_expert.parameters()) / 1e9:.2f}B")
    print(f"Act params: {sum(p.numel() for p in policy.model.internvl_with_expert.act_expert.parameters()) / 1e9:.2f}B")
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
        ResizeImagesWithPadFn(height=448, width=448), 
        InternVL3TokenizerTransformFn(), 
        NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
    ])

    head_color_list = []
    hand_left_color_list = []
    hand_right_color_list = []
    image_history_interval = 15

    acone.reset()
    while True:
        start_time = time.perf_counter()
        obs = acone.get_observation()
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
                "observation.state": torch.as_tensor(obs['qpos']).float().contiguous().cuda(), 
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
                    inputs[key] = sample[key][None].cuda() # .to(dtype=dtype)
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
                        # action_pred, img_pred = policy.predict_action_chunk(inputs) #  , decode_image=decode_image_flag)
                        action_pred = policy.predict_action_chunk(inputs) 
                    warmup_flag = False
                # action_pred, img_pred = policy.predict_action_chunk(inputs) # , decode_image=decode_image_flag)
                action_pred = policy.predict_action_chunk(inputs) # , decode_image=decode_image_flag)
                action_pred = action_pred[0, :, :14].clone()
                action_pred = unnormalize_fn({"action": action_pred})["action"]
                if action_mode == 'delta':
                    init_action[:, 6] = 0
                    init_action[:, 13] = 0
                    action_pred += init_action
                action_chunk.extend(list(action_pred.unbind(dim=0)))

            ea = time.perf_counter()
            print(f"elapse time: {ea - sa:.4f}s")

        action = action_chunk.popleft()
        action = action.to(torch.float32).cpu().numpy()
        
        # Zip the bag
        action[6] = 0 if action[6] > -1.5 else action[6]
        action[13] = 0 if action[13] > -1.5 else action[13]

        # # Pour the drink
        # action[6] = -1.5 if action[6] > -2.3 else -3.3
        # action[13] = -0.5 if action[13] > -1.7 else -3.3

        # if len(action_chunk) == (50 - exec_steps):
        #     action_chunk.clear()
        #     action_chunk = deque(maxlen=config.chunk_size)
        #     head_color_list = []
        #     hand_left_color_list = []
        #     hand_right_color_list = []

        # import pdb
        # pdb.set_trace()
        
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
                    action_chunk = deque(maxlen=config.chunk_size)
                    head_color_list = []
                    hand_left_color_list = []
                    hand_right_color_list = []
                    break
                elif user_key == 'r':
                    print('Resetting robot arm positions...')
                    acone.reset()
                    action_chunk.clear()
                    action_chunk = deque(maxlen=config.chunk_size)
                    head_color_list = []
                    hand_left_color_list = []
                    hand_right_color_list = []
    print("End.")

if __name__ == "__main__":
    main()
