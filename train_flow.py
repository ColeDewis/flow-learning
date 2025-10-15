import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as T
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from flow_model import FlowMatching
from robomimic.utils.dataset import SequenceDataset

# python my_app.py --multirun '+experiment=glob(*)' can run multiple configs


@hydra.main(config_path="conf", config_name="default", version_base=None)
def train(cfg: DictConfig) -> None:
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    num_epochs = cfg.training.num_epochs

    obs_window_size = cfg.flow.obs_window_size
    action_window_size = cfg.flow.action_window_size

    action_dim = 7

    model = FlowMatching(action_dim=action_dim, action_window_size=action_window_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    dataset = SequenceDataset(
        hdf5_path=cfg.dataset.path,
        obs_keys=(
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_joint_pos",
        ),
        action_keys=("actions",),
        action_config={"actions": {}},  # use default action config
        dataset_keys=("rewards", "dones"),
        seq_length=action_window_size,  # number of frames at state S and then AFTER S.
        frame_stack=obs_window_size,  # number of frames BEFORE state s to predict
        pad_seq_length=True,
        pad_frame_stack=True,
    )

    train_loader = DataLoader(
        dataset=dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=100,  # batches of size 100
        shuffle=True,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )

    # so with seq_length=2 and frame_stack=2, we have:
    # actions [s-1, s, s+1]
    # both obs and next_obs also see to have 3 entries,
    # don't think I need next_obs.

    # run = wandb.init(project="FlowMatchLearning")
    # wandb.config.update(cfg)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        ) as tepoch:
            for batch in tepoch:
                optimizer.zero_grad()
                batch["obs"]["images"] = batch["obs"].pop("agentview_image")
                batch["obs"]["joints"] = batch["obs"].pop("robot0_joint_pos")

                # obs is the values up to the obs length;
                # actions to predict starts at the "current" state which is obs_window_size-1
                batch["obs"]["images"] = batch["obs"]["images"][
                    :, :obs_window_size
                ].float()
                batch["obs"]["joints"] = batch["obs"]["joints"][
                    :, :obs_window_size
                ].float()
                batch["actions"] = batch["actions"][:, (obs_window_size - 1) :].float()
                preds, loss = model.forward(batch["actions"], batch["obs"])
                print("PREDICTIONS:", preds.shape)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())
                exit()

        # run validation after each epoch
        # model.eval()
        # val_loss = 0.0
        # with torch.no_grad():
        #     for batch in val_loader:
        #         preds, loss = model.forward(batch)
        #         val_loss += loss.item()
        # val_loss /= len(val_loader)
        # print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")

        scheduler.step()
        run.log(
            {
                "epoch": epoch,
                "loss": epoch_loss / len(train_loader),
                # "val_loss": val_loss,
                "lr": scheduler.get_lr()[0],
            }
        )

    wandb.finish()


if __name__ == "__main__":
    train()
