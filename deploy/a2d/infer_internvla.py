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

import torch
import numpy as np
from PIL import Image


from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.policies.internvla import InternVLAConfig, InternVLAPolicy
from lerobot.datasets.utils import write_json, load_json
from lerobot.datasets.factory import make_dataset
from lerobot.transforms.core import (
    ResizeImagesWithPadFn, 
    NormalizeTransformFn, 
    UnNormalizeTransformFn, 
    ToTensorTransformFn, 
    compose, 
)
from lerobot.utils.constants import OBS_IMAGES, ACTION, OBS_STATE
from lerobot.policies.internvla.transform_internvla import InternVL3TokenizerTransformFn

from a2d.controller import A2DController


def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    ckpt_id = "060000"
    ckpt_path = Path(f"checkpoints/internvla/2025_11_28_13_46_28-internvla-jcaiaq_a2d_pick_pen-delta-24l-pretrain_a2d-060000/pretrained_model")
    ckpt_name = ckpt_path.parts[2]
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.compile_model = True
    config.compile_mode = "reduce-overhead"
    # dtype = torch.bfloat16
    dtype = torch.float32
    assert isinstance(config, InternVLAConfig)
    policy = InternVLAPolicy.from_pretrained(
        config=config, 
        pretrained_name_or_path=ckpt_path, 
    )
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    print(f"InternVL3 params: {sum(p.numel() for p in policy.model.internvl_with_expert.internvl.parameters()) / 1e9:.2f}B")
    print(f"Qwen2_Expert params: {sum(p.numel() for p in policy.model.internvl_with_expert.qwen2_expert.parameters()) / 1e9:.2f}B")
    policy.cuda()
    policy.to(dtype)
    policy.eval()
    dummy_inputs = {
        OBS_STATE: torch.rand((1, 32), dtype=dtype).cuda(), 
        ACTION: torch.rand((1, 50, 32), dtype=dtype).cuda(), 
        f"{OBS_IMAGES}.image0": torch.rand((1, 3, 448, 448), dtype=dtype).cuda(), 
        f"{OBS_IMAGES}.image1": torch.rand((1, 3, 448, 448), dtype=dtype).cuda(), 
        f"{OBS_IMAGES}.image2": torch.rand((1, 3, 448, 448), dtype=dtype).cuda(), 
        f"{OBS_IMAGES}.image0_mask": torch.tensor([True]).cuda(), 
        f"{OBS_IMAGES}.image1_mask": torch.tensor([True]).cuda(), 
        f"{OBS_IMAGES}.image2_mask": torch.tensor([True]).cuda(), 
        "task": ['Put the pen from the table into the pen holder.'], 
        "observation.language.tokens": torch.randint(0, 7777, (1, 48)).cuda(), 
        "observation.language.attention_mask": torch.ones((1, 48)).cuda(), 
    }
    with torch.no_grad():
        policy.predict_action_chunk(dummy_inputs)
        policy.predict_action_chunk(dummy_inputs)

    action_chunk = deque(maxlen=config.chunk_size)
    action_mode = 'delta'  # abs | delta
    
    stats = load_json(ckpt_path / "stats.json")["a2d"]
    stat_keys = ['min', 'max', 'mean', 'std']
    state_stat = {
        stat_key: np.concatenate([
            stats["observation.states.joint.position"][stat_key], 
            stats["observation.states.effector.position"][stat_key], 
        ], axis=-1) for stat_key in stat_keys
    }
    state_stat = {"observation.state": state_stat}
    action_stat = {
        stat_key: np.concatenate([
            stats["actions.joint.position"][stat_key], 
            stats["actions.effector.position"][stat_key], 
        ], axis=-1) for stat_key in stat_keys
    }
    action_stat = {"action": action_stat}
    unnormalize_fn = UnNormalizeTransformFn(
        selected_keys=["action"], 
        mode="mean_std", 
        norm_stats=action_stat, 
    )

    input_transforms = compose([
        ResizeImagesWithPadFn(height=448, width=448), 
        InternVL3TokenizerTransformFn(), 
        NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
    ])
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], dtype=dtype).view(3, 1, 1).cuda()
    imagenet_std = torch.tensor([0.229, 0.224, 0.225], dtype=dtype).view(3, 1, 1).cuda()

    camera_names = [
        "/camera/head_color",
        "/camera/hand_left_color",
        "/camera/hand_right_color",
    ]
    controller = A2DController(
        robot_name="A2D", 
        camera_names=camera_names, 
        config_file="deploy/src/config/a2d.yaml", 
        policy=None, 
        image_processor=None,
        langauage_processor=None,
        state_processor=None,
        action_processor=None, 
    )

    while True:
        controller.reset_to_initial()
        time.sleep(2)
        
        state = controller.get_observation_state()
        temp_gripper_action = state["gripper_joint_state"][:2]
        temp_gripper_action = np.array(temp_gripper_action).astype(float)
        temp_gripper_action = temp_gripper_action > 77.5
        infer_times = []

        for t in range(0, controller.max_step):
            start_time = time.perf_counter()
            if len(action_chunk) == 0:
                print('predict new action chunk')
                sa = time.perf_counter()
                images = controller.get_observation_images()
                state = controller.get_observation_state()
                init_joint_action = torch.from_numpy(np.array(state["arm_joint_state"][:14])[None]).cuda()
                sample = {
                    f"{OBS_IMAGES}.image0": torch.from_numpy(images["/camera/head_color"].copy()).cuda().to(dtype) / 255.0, 
                    f"{OBS_IMAGES}.image1": torch.from_numpy(images["/camera/hand_left_color"].copy()).cuda().to(dtype) / 255.0, 
                    f"{OBS_IMAGES}.image2": torch.from_numpy(images["/camera/hand_right_color"].copy()).cuda().to(dtype) / 255.0, 
                    "observation.state": torch.from_numpy(np.concatenate([state["arm_joint_state"][:14], state["gripper_joint_state"][:2]])).cuda(), 
                    "task": "Put the pen from the table into the pen holder.", 
                }
                for key in sample.keys():
                    if OBS_IMAGES in key:
                        image = sample[key].permute(2, 0, 1)[None]
                        sample[key] = torch.nn.functional.interpolate(image, (480, 640), mode='bilinear', align_corners=False)[0]

                sample = input_transforms(sample)
                sample.update({
                    f"{OBS_IMAGES}.image0": (sample[f"{OBS_IMAGES}.image0"] - imagenet_mean) / imagenet_std, 
                    f"{OBS_IMAGES}.image1": (sample[f"{OBS_IMAGES}.image1"] - imagenet_mean) / imagenet_std, 
                    f"{OBS_IMAGES}.image2": (sample[f"{OBS_IMAGES}.image2"] - imagenet_mean) / imagenet_std, 
                })
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
                with torch.no_grad():
                    action_pred = policy.predict_action_chunk(inputs)[0, :, :16]
                    action_pred = unnormalize_fn({"action": action_pred})["action"]
                    if action_mode == 'delta':
                        action_pred[:, :14] += init_joint_action
                    action_chunk.extend(list(action_pred.unbind(dim=0)))
                
                ea = time.perf_counter()
                print(f"elapse time: {ea - sa:.4f}s")
                infer_times.append(ea - sa)

            action = action_chunk.popleft()
            action = action.to(torch.float32).cpu().numpy()
            joint_action = action[:14]
            gripper_action = action[14:] > 0.95
            end_time = time.perf_counter()
            elapse_time = end_time - start_time

            controller.step(
                {
                    "arm_positions": joint_action,
                    "gripper_positions": gripper_action.astype(float),
                },
            )
            
            time.sleep(max(0, 1 / 30 - elapse_time))

            if sys.stdin.isatty():
                readable, _, _ = select.select([sys.stdin], [], [], 0)
                if readable:
                    user_key = sys.stdin.readline().strip().lower()
                    if user_key == 'e':
                        print("Early exit requested. Breaking current loop.")
                        action_chunk.clear()
                        break


if __name__ == "__main__":
    main()
