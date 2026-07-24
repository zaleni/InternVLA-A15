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
from torchvision.utils import save_image


from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.train import TrainPipelineConfig
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
from lerobot.policies.qwena1 import QwenA1Config, QwenA1Policy
from lerobot.policies.qwena1.transform_qwena1 import Qwen3_VLProcessorTransformFn

from a2d.controller import A2DController

task_instructions = {
    "task_1": "Put the pen from the table into the pen holder.",
    "task_2": "Pick up the flower and insert it into the vase.",
    "task_3": "Pickup a bag of potato chips and put it in the shopping cart.",
    "task_4": "Pickup a bottle of black tea and place it into the shopping cart.",
    "task_5": "pickup a bag of chip to the basket.",
    "task_6": "Pickup a bottle of black tea into the basket.",
    "task_7": "pickup a bag of bread into the basket.",
    "task_8": "Pick a bag of bread with the left arm, then handover, finally put it into the basket.",
    "task_9": "pick a bottle of black tea and hand over to the person.",
}


def main():
    task = "task_1"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # ckpt_path = Path(f"checkpoints/qwena1/2025_12_12_14_48_43-qwena1-jcaiaq_a2d_pick_pen-delta-28l-scratch_60k-060000/pretrained_model")
    ckpt_path = Path(f"checkpoints/qwena1/2025_12_14_18_54_37-qwena1-jcaiaq_a2d_pick_pen-delta-28l-pretrain_100k-finetune_60k-060000/pretrained_model")
    config = PreTrainedConfig.from_pretrained(ckpt_path)
    config.compile_model = True
    config.compile_mode = "reduce-overhead"
    warmup_flag = True
    decode_image_flag = False
    # dtype = torch.bfloat16
    dtype = torch.float32
    assert isinstance(config, QwenA1Config)
    policy = QwenA1Policy.from_pretrained(
        config=config, 
        pretrained_name_or_path=ckpt_path, 
    )
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"\nTotal parameters: {total_params:,}  ({total_params / 1e9:.2f}B)")
    # print(f"Qwen3_VL params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.qwen3_vl.parameters()) / 1e9:.2f}B")
    # print(f"Qwen3_Expert params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.qwen3_expert.parameters()) / 1e9:.2f}B")
    print(f"Und params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.und_expert.parameters()) / 1e9:.2f}B")
    print(f"Gen params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.gen_expert.parameters()) / 1e9:.2f}B")
    print(f"Act params: {sum(p.numel() for p in policy.model.qwen3_vl_with_expert.act_expert.parameters()) / 1e9:.2f}B")
    policy.cuda()
    policy.to(dtype)
    policy.eval()
    
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
        config_file="config/a2d.yaml", 
        policy=None, 
        image_processor=None,
        langauage_processor=None,
        state_processor=None,
        action_processor=None, 
    )

    head_color_list = []
    hand_left_color_list = []
    hand_right_color_list = []
    image_history_interval = 15

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
            sa = time.perf_counter()
            state = controller.get_observation_state()
            images = controller.get_observation_images()
            head_color_list.append(torch.from_numpy(images["/camera/head_color"].copy()).cuda() / 255.0)
            hand_left_color_list.append(torch.from_numpy(images["/camera/hand_left_color"].copy()).cuda() / 255.0)
            hand_right_color_list.append(torch.from_numpy(images["/camera/hand_right_color"].copy()).cuda() / 255.0)
            while len(head_color_list) > image_history_interval+1:
                head_color_list.pop(0)
                hand_left_color_list.pop(0)
                hand_right_color_list.pop(0)
            if len(action_chunk) == 0:
                print('predict new action chunk')
                init_joint_action = torch.from_numpy(np.array(state["arm_joint_state"][:14])[None]).cuda()
                past_idx = max(len(head_color_list) - image_history_interval-1, 0)
                image_head_with_history = torch.stack([head_color_list[past_idx], head_color_list[-1]], dim=0)
                image_hand_left_with_history = torch.stack([hand_left_color_list[past_idx], hand_left_color_list[-1]], dim=0)
                image_hand_right_with_history = torch.stack([hand_right_color_list[past_idx], hand_right_color_list[-1]], dim=0)

                sample = {
                    f"{OBS_IMAGES}.image0": image_head_with_history, 
                    f"{OBS_IMAGES}.image1": image_hand_left_with_history, 
                    f"{OBS_IMAGES}.image2": image_hand_right_with_history, 
                    "observation.state": torch.from_numpy(np.concatenate([state["arm_joint_state"][:14], state["gripper_joint_state"][:2]])).cuda(), 
                    "task": "Put the pen from the table into the pen holder.", 
                }
                for key in sample.keys():
                    if OBS_IMAGES in key:
                        image = sample[key].permute(0, 3, 1, 2)
                        sample[key] = torch.nn.functional.interpolate(image, (480, 640), mode='bilinear', align_corners=False)

                # import pdb; pdb.set_trace()
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
                    # import pdb; pdb.set_trace()
                    if warmup_flag:
                        for warmup_step in range(5):
                            print(f"warmup step: {warmup_step}")
                            action_pred, img_pred = policy.predict_action_chunk(inputs, decode_image=decode_image_flag)
                            action_pred = action_pred[0, :, :16].clone()
                        warmup_flag = False
                    action_pred, img_pred = policy.predict_action_chunk(inputs, decode_image=decode_image_flag)
                    action_pred = action_pred[0, :, :16].clone()
                    action_pred = unnormalize_fn({"action": action_pred})["action"]
                    if action_mode == 'delta':
                        action_pred[:, :14] += init_joint_action
                    action_chunk.extend(list(action_pred.unbind(dim=0)))
                    if img_pred is not None:
                        save_image((img_pred + 1) / 2, "img_pred.jpg", value_range=(0, 1))
                
                ea = time.perf_counter()
                print(f"elapse time: {ea - sa:.4f}s")
                infer_times.append(ea - sa)

            action = action_chunk.popleft()
            action = action.to(torch.float32).cpu().numpy()
            joint_action = action[:14]
            gripper_action = action[14:] > 0.95
            end_time = time.perf_counter()
            elapse_time = end_time - start_time

            # import pdb; pdb.set_trace()
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
        user_choice = input("Run again? (y/n, default: y): ").strip().lower()
        if user_choice == "":
            user_choice = "y"
        while user_choice not in ("y", "n"):
            user_choice = input("Please enter 'y' or 'n' (default: y): ").strip().lower()
            if user_choice == "":
                user_choice = "y"
                action_chunk = []
                head_color_list = []
                hand_left_color_list = []
                hand_right_color_list = []
        if user_choice == "n":
            controller.reset_to_initial()
            print("Exiting loop.")
            break
        print("Restarting loop.")
    print("Program ended.")


if __name__ == "__main__":
    main()
