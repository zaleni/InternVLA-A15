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

from a2d.controller import A2DController



def main():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    ckpt_id = "060000"
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_11_09_14_33_27-qwenvla-jcaiaq_a2d_pick_pen-delta-28l/pretrained_model")
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_11_10_10_41_38-qwenvla-jcaiaq_a2d_pick_pen-delta-10l/pretrained_model")
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_11_15_21_55_15-qwenvla-jcaiaq_a2d_pick_pen-delta-28l-pretrain_50k-finetune_60k/pretrained_model")
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_11_15_21_58_04-qwenvla-jcaiaq_a2d_pick_pen-delta-28l-pretrain_100k-finetune_60k/pretrained_model")
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_11_28_00_33_40-qwenvla-jcaiaq_a2d_pick_pen-delta-28l-pretrain_30k-finetune_30k/pretrained_model")
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_11_28_08_52_23-qwenvla-jcaiaq_a2d_pick_pen-delta-28l-pretrain_50k-finetune_30k/pretrained_model")
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_12_01_08_16_00-qwenvla-jcaiaq_a2d_pick_pen-delta-28l-all_a2d_pretrain_30k-finetune_60k-060000/pretrained_model")
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_12_01_08_14_47-qwenvla-jcaiaq_a2d_pick_pen-delta-28l-all_a2d_pretrain_70k-finetune_60k-060000/pretrained_model")
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_12_01_08_06_09-qwenvla-jcaiaq_a2d_pick_pen-delta-28l-all_a2d_pretrain_130k-finetune_60k-060000/pretrained_model")
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_11_30_22_43_49-qwenvla-jcaiaq_a2d_pick_pen-delta-28l-all_a2d_sim_pretrain_30k-finetune_60k-060000/pretrained_model")
    # ckpt_path = Path(f"checkpoints/qwenvla/2025_11_30_17_34_56-qwenvla-jcaiaq_a2d_pick_pen-delta-28l-all_a2d_sim_pretrain_70k-finetune_60k-060000/pretrained_model")
    ckpt_path = Path(f"checkpoints/qwenvla/2025_12_01_08_03_23-qwenvla-jcaiaq_a2d_pick_pen-delta-28l-all_a2d_sim_pretrain_130k-finetune_60k-060000/pretrained_model")
    # ckpt_name = ckpt_path.parts[2]
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.compile_model = True
    config.compile_mode = "reduce-overhead"
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
        OBS_STATE: torch.rand((1, 16), dtype=dtype).cuda(), 
        ACTION: torch.rand((1, 50, 16), dtype=dtype).cuda(), 
        "task": ["dummy sample"], 
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
        # ResizeShortestCenterCropFn(height=448, width=448), 
        ResizeImagesWithPadFn(height=224, width=224), 
        Qwen3_VLProcessorTransformFn(), 
        NormalizeTransformFn(selected_keys=["observation.state"], norm_stats=state_stat),
    ])

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
        init_joint_action = copy.deepcopy(state["arm_joint_state"][:14])
        temp_joint_action = state["arm_joint_state"][:14]
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
