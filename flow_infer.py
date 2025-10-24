import faulthandler
import pickle

import hydra
import numpy as np
import robosuite as suite
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as T
import wandb
from omegaconf import DictConfig, OmegaConf
from robosuite.controllers.composite.composite_controller_factory import (
    refactor_composite_controller_config,
)
from robosuite.environments import MujocoEnv
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from tqdm import tqdm

import robomimic.utils.obs_utils as ObsUtils
from flow_model import FlowMatching
from robomimic.utils.dataset import SequenceDataset
from train_flow import prepare_batch

faulthandler.enable()


@hydra.main(config_path="conf", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    num_epochs = cfg.training.num_epochs

    obs_window_size = cfg.flow.obs_window_size
    action_window_size = cfg.flow.action_window_size

    action_dim = 7  # 6 cartesian and gripper

    options = {}

    options["robots"] = "Panda"
    # arm_controller_config = suite.load_part_controller_config(
    #     default_controller="JOINT_POSITION"
    # )
    # robot = (
    #     options["robots"][0]
    #     if isinstance(options["robots"], list)
    #     else options["robots"]
    # )
    # options["controller_configs"] = refactor_composite_controller_config(
    #     arm_controller_config, robot, ["right", "left"]
    # )

    from robosuite.utils.placement_samplers import UniformRandomSampler

    cube_initializer = UniformRandomSampler(
        name="ObjectSampler",
        x_range=[-0.1, 0.1],
        y_range=[-0.1, 0.1],
        rotation=None,
        ensure_object_boundary_in_range=False,
        ensure_valid_placement=True,
        reference_pos=np.array((0, 0, 0.8)),
        z_offset=0.01,
    )

    env: ManipulationEnv = suite.make(
        "Lift",
        renderer="mujoco",
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=224,
        camera_widths=224,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="agentview",
        **options,
        # control_freq=10,
        placement_initializer=cube_initializer,
    )

    env.reset()

    obs = env.reset()

    model = FlowMatching(
        action_dim=action_dim, action_window_size=action_window_size, image_input=False
    )
    # DIR = "/home/coled/flow-learning/outputs/2025-10-22/08-52-39"
    # DIR = "/home/coled/flow-learning/outputs/2025-10-21/16-55-38"
    # DIR = "/home/coled/flow-learning/outputs/2025-10-22/16-37-42"
    # DIR = "/home/coled/flow-learning/outputs/2025-10-23/RAW_OBJECT_TEST"
    DIR = "/home/coled/flow-learning/outputs/2025-10-23/19-24-55"

    model.load_state_dict(torch.load(f"{DIR}/flow_epoch_200.pth"))
    with open(f"{DIR}/normalization_stats.pkl", "rb") as f:
        normalization_stats = pickle.load(f)
    with open(f"{DIR}/action_normalization_stats.pkl", "rb") as f:
        action_normalization_stats = pickle.load(f)
    # model.load_state_dict(torch.load("../../../flow_epoch_1000.pth"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    obs_history = []

    transform = Compose(
        [
            ToTensor(),  # Convert the image to a PyTorch tensor
            Resize((84, 84)),  # Resize to 224x224
        ]
    )
    for step in range(1000):
        current_obs = {
            "images": transform(obs["agentview_image"]).permute(1, 2, 0),
            "joints": np.concatenate(
                [
                    obs["robot0_joint_pos"],
                ]
            ),
            "gripper": np.array([obs["robot0_gripper_qpos"]]),
            "object": np.array([obs["object-state"]]),
        }
        obs_history.append(current_obs)

        if len(obs_history) > obs_window_size:
            obs_history.pop(0)

        if len(obs_history) < obs_window_size:
            action = np.zeros(action_dim)
            obs, reward, done, info = env.step(action)
            env.render()
        else:
            images = np.stack([h["images"] for h in obs_history])
            joints = np.stack([h["joints"] for h in obs_history])
            grippers = np.stack([h["gripper"] for h in obs_history])
            objects = np.stack([h["object"] for h in obs_history])

            obs_tensor = {
                "obs": {
                    "agentview_image": torch.tensor(
                        images, dtype=torch.float32
                    ).unsqueeze(0),
                    "robot0_joint_pos": torch.tensor(
                        joints, dtype=torch.float32
                    ).unsqueeze(0),
                    "robot0_gripper_qpos": torch.tensor(
                        grippers, dtype=torch.float32
                    ).unsqueeze(0),
                    "object": torch.tensor(objects, dtype=torch.float32).unsqueeze(0),
                }
            }

            obs_tensor["obs"]["robot0_gripper_qpos"] = obs_tensor["obs"][
                "robot0_gripper_qpos"
            ].reshape(1, obs_window_size, -1)

            obs_tensor = prepare_batch(
                normalization_stats,
                obs_tensor,
                obs_window_size,
                device,
                has_actions=False,
            )

            # obs_tensor = {
            #     "images": torch.tensor(images, dtype=torch.float32)
            #     .unsqueeze(0)
            #     .to(device),
            #     "joints": torch.tensor(joints, dtype=torch.float32)
            #     .unsqueeze(0)
            #     .to(device),
            #     "gripper": torch.tensor(grippers, dtype=torch.float32)
            #     .unsqueeze(0)
            #     .to(device),
            # }

            # mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(
            #     1, 1, 1, 1, 3
            # )
            # std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 1, 1, 3)
            # obs_tensor["images"] = ((obs_tensor["images"] / 255) - mean) / std

            # obs_tensor["images"] = obs_tensor["images"].permute(
            #     0, 1, 4, 2, 3
            # )  # to N T C H W
            # print(obs_tensor["images"].shape)

            with torch.no_grad():
                action = model.infer(obs_tensor["obs"], delta=0.1)

            scale = action_normalization_stats["actions"]["scale"]
            offset = action_normalization_stats["actions"]["offset"]
            scale = np.expand_dims(scale, axis=0)
            offset = np.expand_dims(offset, axis=0)

            action = action.cpu().numpy()

            # run the first 4/8 actions

            for i in range(4):
                action_i = action[0, i] * scale + offset
                action_i = action_i[0, 0]
                print("Inferred Action:", [round(x, 2) for x in action_i.tolist()])

                action_i = np.clip(action_i, env.action_spec[0], env.action_spec[1])
                action_i[3:6] = 0.0  # zero out rotation for simplicity
                obs, reward, done, info = env.step(action_i)
                env.render()

        if done:
            obs = env.reset()


if __name__ == "__main__":
    main()
