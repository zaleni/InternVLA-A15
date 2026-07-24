
from omegaconf import OmegaConf
from pathlib import Path
import time
from PIL import Image
from deploy.src.a2d.a2d_robot import A2DRobot

class A2DController:
    def __init__(
        self, 
        robot_name, 
        camera_names, 
        config_file, 
        policy,
        image_processor=None,
        langauage_processor=None,
        state_processor=None,
        action_processor=None
    ):
        self.config = OmegaConf.load(Path(config_file))

        # load the initial state
        initial_state = self.config.initial_state
        arm_joint = initial_state.arm_joint
        gripper_joint = initial_state.gripper_joint
        head_joint = initial_state.head_joint
        waist_joint = initial_state.waist_joint

        self.robot_name = robot_name
            
        self.robot = A2DRobot(
            init_arm_positions=arm_joint,
            init_gripper_positions=gripper_joint,
            init_head_positions=head_joint,
            init_waist_positions=waist_joint,
            camera_names=camera_names,
        )
        time.sleep(2)   # to ensure the robot is ready
        self.robot.reset_to_initial()
        time.sleep(2)   # to ensure the robot is ready
        print("A2D Robot initialized.")

        # init policy and processors
        self.policy = policy
        self.image_processor = image_processor
        self.langauage_processor = langauage_processor
        self.state_processor = state_processor
        self.action_processor = action_processor

        self.max_step = self.config.max_step

    def get_observation_state(self):
        """
        Get the current observation state of the robot.
        Returns:
            dict: A dictionary containing the states of various joints.
        """
        return self.robot.get_observation_state()
    
    def get_observation_images(self):
        """
        Get the latest images from the robot's cameras.
        Returns:
            dict: A dictionary where keys are camera names and values are the latest images.
        """
        return self.robot.get_observation_images()
    
    def reset_to_initial(self):
        """
        Reset the robot to its initial positions.
        """
        self.robot.reset_to_initial()

    def step(self, action):

        assert isinstance(action, dict), "Action must be a dictionary"
        assert "arm_positions" in action, "Action must contain 'arm_positions'"
        assert "gripper_positions" in action, "Action must contain 'gripper_positions'"

        return self.robot.step(action)

    def select_action(self, images, lang, state):
        # For seer
        # for view, image in images.items():
        #     image = Image.fromarray(image).convert("RGB")
        #     image = self.image_processor([image])
        #     images[view] = image
        # start_time = time.time()
        images = {k.split("/")[-1]: v for k, v in images.items()}
        
        # images: {"head_color", "hand_left_color", "hand_right_color"}
        # 

        # import pdb
        # pdb.set_trace()
        if self.image_processor:
            images = self.image_processor(images)
        # end_time = time.time()
        # print(f"> ------------------- image processer time: {end_time - start_time:.4f}\n")
        # start_time = time.time()
        if self.langauage_processor:
            lang = self.langauage_processor(lang)
        # end_time = time.time()
        # print(f"> ------------------- language processer time: {end_time - start_time:.4f}\n")
        # start_time = time.time()
        if self.state_processor:
            state = self.state_processor(state)
        # end_time = time.time()
        # print(f"> ------------------- state processer time:  {end_time - start_time:.4f}\n")

        # import pdb; pdb.set_trace()
        if self.policy:
            action = self.policy.select_action(images, lang, state)
            # start_time = time.time()
            if self.action_processor is not None:
                action = self.action_processor(action)
            # end_time = time.time()
            # print(f"> ------------------- action processer time:  {end_time - start_time:.4f}\n")
        else: 
            action = None
        return action