import faulthandler

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

from flow_model import FlowMatching
from robomimic.utils.dataset import SequenceDataset

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

    env: ManipulationEnv = suite.make(
        "Lift",
        renderer="mujoco",
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=84,
        camera_widths=84,
        has_renderer=True,
        has_offscreen_renderer=True,
        render_camera="frontview",
        **options,
    )

    env.reset()

    obs = env.reset()

    model = FlowMatching(action_dim=action_dim, action_window_size=action_window_size)
    model.load_state_dict(torch.load("../../../final_flow_40demo.pth"))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    obs_history = []

    transform = Compose(
        [
            ToTensor(),  # Convert the image to a PyTorch tensor
            Resize((84, 84)),  # Resize to 84x84
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
            "gripper": np.array([obs["robot0_gripper_qpos"][0]]),
        }
        obs_history.append(current_obs)

        if len(obs_history) > obs_window_size:
            obs_history.pop(0)

        if len(obs_history) < obs_window_size:
            action = np.zeros(action_dim)
        else:
            images = np.stack([h["images"] for h in obs_history])
            joints = np.stack([h["joints"] for h in obs_history])
            grippers = np.stack([h["gripper"] for h in obs_history])

            obs_tensor = {
                "images": torch.tensor(images, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                "joints": torch.tensor(joints, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
                "gripper": torch.tensor(grippers, dtype=torch.float32)
                .unsqueeze(0)
                .to(device),
            }

            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(
                1, 1, 1, 1, 3
            )
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 1, 1, 3)
            obs_tensor["images"] = ((obs_tensor["images"] / 255) - mean) / std

            with torch.no_grad():
                action = model.infer(obs_tensor, delta=0.1)

            action = action.cpu().numpy()
            action = action[0, 0]
            print("Inferred Action:", action)
            action = np.clip(action, env.action_spec[0], env.action_spec[1])

        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()


if __name__ == "__main__":
    main()
