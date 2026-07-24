#!/home/lin/software/miniconda3/envs/aloha/bin/python
# -- coding: UTF-8
"""
#!/usr/bin/python3
"""
import yaml
import torch
import numpy as np
import os
import pickle
import argparse
import cv2
# from controller.controller_real import PI0Controller
import collections
from collections import deque

import rospy
from std_msgs.msg import Header
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, Image
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import time
import threading
import lmdb
from pdb import set_trace
import threading
import select
# from controller.replay_controller import ReplayController
import matplotlib.pyplot as plt
import sys
from utils.ros_operator import RosOperator
from omegaconf import OmegaConf
sys.path.append("./")


def main():
    policy_name = "internvla_a1"
    action_type = "rel_qpos"  # abs_qpos or rel_qpos
    config_path = "config/controller.yaml"
    # language_instruction = "Sort the garbage on the desktop into recyclable and non-recyclable"
    language_instruction = "Bring the shipping label of the package into view, then grasp the package from the conveyor belt and orient the label to myself"

    config = OmegaConf.load(config_path)
    config["policy_name"] = policy_name
    config["action_type"] = action_type

    ros_operator = RosOperator(config, config, in_collect=False)

    control = DeployLift2(ros_operator=ros_operator, args=config)
    control.reset()
    control.model_control(
        policy=config["policy_name"],
        action_type=config["action_type"],
        chunk_size=config["chunk_size"],
        language_instruction=language_instruction
    )

class DeployLift2:
    def __init__(self, args, ros_operator):
        self.ros_operator = ros_operator
        self.args = args
        self.replay = False
        self.real_eval_max_steps = args['real_eval_max_steps']
    
    def get_observation(self, args, timestep):
        global obs_dict
        # print("get_observation")
        rate = rospy.Rate(args.frame_rate)
        while True and not rospy.is_shutdown():
            obs_dict = self.ros_operator.get_observation(ts=timestep)
            if not obs_dict:
                print("syn fail")
                rate.sleep()

                continue
      
            img_front = obs_dict['images']['head']
            # print("obs_dict['images']['head']:",obs_dict['images']['head'].shape)
            # img_front = cv2.resize(img_front, (640, 480))
            img_left = obs_dict['images']['left_wrist'] #(480, 640, 3)
            img_right = obs_dict['images']['right_wrist'] #(480, 640, 3)

            qpos = obs_dict['qpos'] #(14,)
            qvel = obs_dict['qvel'] #(14,)
            effort = obs_dict['effort'] #(14,)
            # obs_dict -> obs

            obs = {
                "color_image": [img_front, img_left, img_right,],
                "robot_state": {
                    "qpos": qpos,
                    "qvel": qvel,
                },
                "effort": effort
            }
            return obs

    def reset(self):
        left0  = [0, 0, 0, 0, 0, 0, 5]
        right0 = [0, 0, 0, 0, 0, 0, 5]

        left1  = [0.00889644,  1.6385847, -0.94616256, -0.04973284, 0.06998533,  0.07207861, 0.0715]
        right1 = [0.13147543,  1.7397599,  -1.05879847, -0.14471542,  0.18298756, 0.17756248, 0.073]
        
        # self.ros_operator.puppet_arm_publish_continuous(left1, right1)
        self.ros_operator.follow_arm_publish_continuous(left0, right0)
        input("Enter any key to continue: ")

    def model_control(self, policy=None, action_type='abs_qpos', chunk_size=1, language_instruction=""):
        if policy == "internvla_a1":
            from policy.internvla_a1.control.internvla_a1_controller import InternVLAA1PolicyController
            internvla_a1_config_path = "config/policy/internvla_a1.yaml"
            policy = InternVLAA1PolicyController(config_path=internvla_a1_config_path)

        step_id = 0
        avg_delay = 0

        infer_times = []

        while step_id < self.real_eval_max_steps and not rospy.is_shutdown():
            start_t = time.perf_counter()
            obs_dict_real = self.get_observation(self.args, step_id)
            if obs_dict_real is None:
                print("no obs")
                continue
            else:
                state = obs_dict_real['robot_state']['qpos']
                lang = language_instruction
                images = {
                    'head_color': obs_dict_real['color_image'][0],
                    'hand_left_color': obs_dict_real['color_image'][1],
                    'hand_right_color': obs_dict_real['color_image'][2],
                }
                

                if step_id % policy.action_pred_steps == 0:
                    state_wo_gripper = state.copy()
                    state_wo_gripper[6] = 0.0
                    state_wo_gripper[13] = 0.0
                
                action = policy.select_action(images, lang, state)

                if action_type == "abs_qpos":
                    target_qpos = action

                elif action_type == "rel_qpos":
                    action = action.cpu().numpy() + state_wo_gripper
                    target_qpos = action.copy()

                left_action = target_qpos[:7]
                right_action = target_qpos[7:14]

                # # apply action   
                self.ros_operator.follow_arm_publish(left_action, right_action)
                # self.ros_operator.puppet_arm_publish_continuous(left_action, right_action)

                # if self.args.use_robot_base:
                #     vel_action = action[14:16]
                #     self.ros_operator.robot_base_publish(vel_action)

                end_t = time.perf_counter()
                delay = end_t - start_t

                # if step_id % chunk_size != 0:
                # print(delay)
                time.sleep(max(0, 1/30 - delay))

                step_id += 1

                if sys.stdin.isatty():
                    readable, _, _ = select.select([sys.stdin], [], [], 0)
                    if readable:
                        user_key = sys.stdin.readline().strip().lower()
                        if user_key == 'r':
                            print('Resetting robot arm positions...')
                            self.reset()
                            policy.reset()
                            step_id = 0

                # if step_id > 20:
                #     avg_delay += delay

                # if step_id > 20 and step_id % 10 == 0:
                #     print(f"select action time: {delay}")
                #     infer_times.append(delay)
                #     print(f"max inference time: {max(infer_times)}")
                # print(step_id)



if __name__ == '__main__':
    main()

