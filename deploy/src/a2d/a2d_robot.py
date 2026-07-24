import cv2
import time
import ruckig
from a2d_sdk.robot import RobotDds
from a2d_sdk.robot import CosineCamera


class A2DRobot(RobotDds):
    """
    A2D Robot class that extends RobotDds to provide additional functionality.
    This class can be used to interact with the A2D robot and its components.
    """

    def __init__(self, *args, **kwargs):
        self.camera_names = kwargs.pop("camera_names", [
            "camera_front", "camera_back", "camera_left", "camera_right"
        ])
        self.camera_group = CosineCamera(self.camera_names)

        self.init_arm_positions = kwargs.pop("init_arm_positions", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.init_gripper_positions = kwargs.pop("init_gripper_positions", [34.94])
        # chips
        # self.init_head_positions = kwargs.pop("init_head_positions", [0.0, 0.17453292222222222])
        # self.init_waist_positions = kwargs.pop("init_waist_positions", [0.4363327457474638, 0.25])
        # # pen
        self.init_head_positions = kwargs.pop("init_head_positions", [0.0, 0.43633])
        self.init_waist_positions = kwargs.pop("init_waist_positions", [0.61086584, 0.32])

        self.joint_dim = 14

        super().__init__(*args, **kwargs)

    def get_observation_state(self):
        """
        Get the current observation state of the robot.
        Returns:
            dict: A dictionary containing the states of various joints.
        """
        arm_joint_timestamp = None

        while arm_joint_timestamp is None:
            # Wait until we have a valid arm joint state
            arm_joint_state, arm_joint_timestamp = self.arm_joint_states()
            gripper_joint_state, _ = self.gripper_states()
            left_effector_joint_state, right_effector_joint_state = gripper_joint_state
            head_joint_state, _ = self.head_joint_states()
            waist_joint_state, _ = self.waist_joint_states()

        return {
            "arm_joint_state": [float(x) for x in arm_joint_state],
            "gripper_joint_state": [float(x) for x in gripper_joint_state],
            "left_effector_joint_state": float(left_effector_joint_state),
            "right_effector_joint_state": float(right_effector_joint_state),
            "head_joint_state": [float(x) for x in head_joint_state],
            "waist_joint_state": [float(x) for x in waist_joint_state]
        }

    def get_observation_images(self, channel="rgb"):
        """
        Get the latest images from the robot's cameras.
        Returns:
            dict: A dictionary where keys are camera names and values are the latest images.
        """
        observation_images = {}
        for camera_name in self.camera_names:
            image, timestamp = self.camera_group.get_latest_image(camera_name)
            if channel == "rgb":
                image = image[::-1][::-1,:,:]
            elif channel == "gray":
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif channel == "bgr":
                pass
            observation_images[camera_name] = image

        return observation_images
    
    def reset(self, arm_positions=None, gripper_positions=None, hand_positions=None, waist_positions=None, head_positions=None):
        """
        Reset the robot's positions.
        """
        if gripper_positions is None:
            gripper_positions = self.init_gripper_positions
        if head_positions is None:
            head_positions = self.init_head_positions
        if waist_positions is None:
            waist_positions = self.init_waist_positions

        super().reset(
            arm_positions=arm_positions,
            gripper_positions=gripper_positions,
            hand_positions=hand_positions,
            waist_positions=waist_positions,
            head_positions=head_positions
        )
    
    def reset_to_initial(self):
        """
        Reset the robot to its initial positions.
        """
        print(self.init_head_positions)
        print(self.init_waist_positions)
        self.reset(
            arm_positions=self.init_arm_positions,
            gripper_positions=self.init_gripper_positions,
            hand_positions=None,  # Assuming no specific hand positions are set
            waist_positions=self.init_waist_positions,
            head_positions=self.init_head_positions
        )
    
    def set_target(self, action):
        self.target_action = action
        self.need_new_trajectory = True
        self.motion_finished = False

    def step(self, action):
        """
        Step the robot with the given action.
        Args:
            action (dict): A dictionary containing the actions for the robot.
        """
        # Assuming action is a dictionary with keys corresponding to joint names
        arm_positions = action.get("arm_positions", self.init_arm_positions)
        gripper_positions = action.get("gripper_positions", self.init_gripper_positions)

        self.move_arm(arm_positions)
        self.move_gripper(gripper_positions)

    def move_arm_smooth(self, positions):

        qpos = list(self._arm_joint_states.position)
        arm_target_positions = positions[:self.joint_dim]

        dof = self.joint_dim
        interval = 0.001
        rk = ruckig.Ruckig(dof, interval)
        rk_input = ruckig.InputParameter(dof)
        rk_output = ruckig.OutputParameter(dof)

        # set the current position
        rk_input.current_position = qpos
        rk_input.current_velocity = [0.0] * self.joint_dim
        rk_input.current_acceleration = [0.0] * self.joint_dim

        # set the target position
        rk_input.target_position = arm_target_positions
        rk_input.target_velocity = [0.0] * 14
        rk_input.target_acceleration = [0.0] * self.joint_dim

        # set the limits
        rk_input.max_velocity = [2.0] * self.joint_dim
        rk_input.max_acceleration = [1.0] * self.joint_dim
        rk_input.max_jerk = [5.0] * self.joint_dim

        # generate the trajectory
        trajs = []
        while rk.update(rk_input, rk_output) == ruckig.Result.Working:
            trajs.append(rk_output.new_position)
            rk_output.pass_to_input(rk_input)

        response = 1
        for traj in trajs:
            re = super().move_arm(traj)
            if re == 0: response = 0
            # time.sleep(interval)

        return response