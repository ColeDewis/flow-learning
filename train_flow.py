import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as T
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from flow_model import FlowMatching
from robomimic.utils.dataset import SequenceDataset


def prepare_batch(
    batch: dict,
    obs_window_size: int,
    device: torch.device,
    img_mean: torch.Tensor,
    img_std: torch.Tensor,
) -> dict:
    """Processes the batch from robomimic into a nicer form by normalizing images
    and organizing to have nicer keys for the model.

    Args:
        batch (dict): batch of data from the data loader
        obs_window_size (int): size of the observation window
        device (torch.device): device to which the tensors should be moved
        img_mean (Tensor): mean values for image normalization
        img_std (Tensor): standard deviation values for image normalization

    Returns:
        dict: processed batch
    """

    # TODO: binary gripper is probably better

    batch["obs"]["images"] = batch["obs"].pop("agentview_image")
    batch["obs"]["joints"] = batch["obs"].pop("robot0_joint_pos")
    # gripper goes from 0 -> 0.02 in qpos, convert this to 0 -> 1
    batch["obs"]["gripper"] = batch["obs"].pop("robot0_gripper_qpos")[:, :, 0]
    batch["obs"]["gripper"] = batch["obs"]["gripper"] / 0.02
    batch["obs"]["gripper"] = 1 - torch.clamp(batch["obs"]["gripper"], 0.0, 1.0)
    # obs is the values up to the obs length;
    # actions to predict starts at the "current" state which is obs_window_size-1
    batch["obs"]["images"] = (
        batch["obs"]["images"][:, :obs_window_size].float().to(device)
    )

    # normalizing images
    batch["obs"]["images"] = ((batch["obs"]["images"] / 255) - img_mean) / img_std
    batch["obs"]["joints"] = (
        batch["obs"]["joints"][:, :obs_window_size].float().to(device)
    )
    batch["obs"]["gripper"] = (
        batch["obs"]["gripper"][:, :obs_window_size].float().to(device)
    )
    batch["actions"] = batch["actions"][:, (obs_window_size - 1) :].float().to(device)

    return batch


@hydra.main(config_path="conf", config_name="default", version_base=None)
def train(cfg: DictConfig) -> None:
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    num_epochs = cfg.training.num_epochs

    obs_window_size = cfg.flow.obs_window_size
    action_window_size = cfg.flow.action_window_size

    action_dim = 7
    joints_dim = 7 + 1  # 7 joints + 1 gripper

    model = FlowMatching(
        robot_state_dim=joints_dim,
        action_dim=action_dim,
        action_window_size=action_window_size,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    train_dataset = SequenceDataset(
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
        filter_by_attribute="train",
    )
    valid_dataset = SequenceDataset(
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
        filter_by_attribute="valid",
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        sampler=None,  # no custom sampling logic (uniform sampling)
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )

    # so with seq_length=2 and frame_stack=2, we have:
    # actions [s-1, s, s+1]
    # both obs and next_obs also see to have 3 entries,
    # don't think I need next_obs.

    run = wandb.init(project="FlowMatchLearning")
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True))

    best_loss = float("inf")
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 1, 1, 1, 3)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 1, 1, 1, 3)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        with tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"
        ) as tepoch:
            for batch in tepoch:
                optimizer.zero_grad()
                batch = prepare_batch(batch, obs_window_size, device, mean, std)

                preds, loss = model.forward(batch["actions"], batch["obs"])
                # print("PREDICTIONS:", preds.shape)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                tepoch.set_postfix(loss=loss.item())

        # run validation after each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in valid_loader:
                batch = prepare_batch(batch, obs_window_size, device, mean, std)
                preds, loss = model.forward(batch["actions"], batch["obs"])
                val_loss += loss.item()

            # sample inference
            batch["obs"]["images"] = batch["obs"]["images"][0].unsqueeze(0)
            batch["obs"]["joints"] = batch["obs"]["joints"][0].unsqueeze(0)
            batch["obs"]["gripper"] = batch["obs"]["gripper"][0].unsqueeze(0)
            example_pred = model.infer(batch["obs"], delta=0.1)
            print("Example Inference Action:", example_pred[0, 0])

        val_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_flow.pth")
            print(f"New best model saved with loss {best_loss:.4f}")

        scheduler.step()
        run.log(
            {
                "epoch": epoch,
                "loss": epoch_loss / len(train_loader),
                "val_loss": val_loss,
                "lr": scheduler.get_last_lr()[0],
            }
        )

    torch.save(model.state_dict(), "final_flow.pth")
    wandb.finish()


if __name__ == "__main__":
    train()
