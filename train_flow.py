import pickle

import cv2
import hydra
import numpy as np
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

import robomimic.utils.obs_utils as ObsUtils
from flow_model import FlowMatching
from robomimic.config import config_factory
from robomimic.utils.dataset import SequenceDataset


def prepare_batch(
    normalization_stats: dict,
    batch: dict,
    obs_window_size: int,
    device: torch.device,
    has_actions=True,
    inference=False,
) -> dict:
    """Processes the batch from robomimic into a nicer form by normalizing images
    and organizing to have nicer keys for the model.

    Args:
        batch (dict): batch of data from the data loader
        obs_window_size (int): size of the observation window
        device (torch.device): device to which the tensors should be moved

    Returns:
        dict: processed batch
    """
    for key in batch["obs"].keys():
        if "image" in key:
            batch["obs"][key] = ObsUtils.batch_image_hwc_to_chw(batch["obs"][key])
            # get rid of their image normalization to do it later.
            normalization_stats[key]["offset"] = np.zeros_like(
                normalization_stats[key]["offset"]
            )
            normalization_stats[key]["scale"] = np.ones_like(
                normalization_stats[key]["scale"]
            )
        batch["obs"][key] = batch["obs"][key].float()

    ObsUtils.normalize_dict(batch["obs"], normalization_stats=normalization_stats)

    batch["obs"]["images"] = batch["obs"].pop("agentview_image")

    # scale to [-1, 1]
    batch["obs"]["images"] = (batch["obs"]["images"] / 255.0) * 2.0 - 1.0

    batch["obs"]["joints"] = batch["obs"].pop("robot0_joint_pos")
    batch["obs"]["gripper"] = batch["obs"].pop("robot0_gripper_qpos")[:, :, 0]

    # TODO: try normalization to [-1, 1] by dividing by 255 then -1 * 2.
    # TODO: could consider adding a small random crop to the images

    # grab correct windows
    batch["obs"]["images"] = (
        batch["obs"]["images"][:, :obs_window_size].float().to(device)
    )
    batch["obs"]["joints"] = (
        batch["obs"]["joints"][:, :obs_window_size].float().to(device)
    )
    batch["obs"]["gripper"] = (
        batch["obs"]["gripper"][:, :obs_window_size].float().to(device)
    )
    # batch["obs"]["object"] = (
    #     batch["obs"]["object"][:, :obs_window_size].float().to(device)
    # )

    if has_actions:
        batch["actions"] = (
            batch["actions"][:, (obs_window_size - 1) :].float().to(device)
        )

    return batch


@hydra.main(config_path="conf", config_name="default", version_base=None)
def train(cfg: DictConfig) -> None:
    print("Dataset path:", cfg.dataset.path)
    lr = cfg.training.lr
    batch_size = cfg.training.batch_size
    num_epochs = cfg.training.num_epochs
    save_freq = cfg.training.save_freq

    obs_window_size = cfg.flow.obs_window_size
    action_window_size = cfg.flow.action_window_size

    action_dim = 7
    joints_dim = 7 + 1  # 7 joints + 1 gripper

    model = FlowMatching(
        robot_state_dim=joints_dim,
        obs_window_size=obs_window_size,
        action_dim=action_dim,
        action_window_size=action_window_size,
        image_input=True,
        # image_input=False,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)

    warmup_epochs = 10
    warmup_scheduler = lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    scheduler = lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    obs_modalities = {
        "obs": {
            "low_dim": [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_eef_quat_site",
                "robot0_gripper_qpos",
                "robot0_joint_pos",
                "robot0_eef_quat_site",
                "object",
                "robot0_gripper_qvel",
                "robot0_joint_pos_cos",
                "robot0_joint_pos_sin",
                "robot0_joint_vel",
            ],
            "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
        }
    }
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modalities)

    train_dataset = SequenceDataset(
        hdf5_path=cfg.dataset.path,
        obs_keys=(
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "agentview_image",
            "robot0_eye_in_hand_image",
            "robot0_joint_pos",
            # "object",
        ),
        action_keys=("actions",),
        action_config={"actions": {"normalization": "min_max"}},
        # action_config={"actions": {"normalization": None}},
        dataset_keys=("rewards", "dones"),
        seq_length=action_window_size,  # number of frames at state S and then AFTER S.
        frame_stack=obs_window_size,  # number of frames BEFORE state s to predict
        pad_seq_length=True,
        pad_frame_stack=True,
        filter_by_attribute="train",
        hdf5_normalize_obs=True,
        load_next_obs=False,
        hdf5_cache_mode="all",
        # demo_limit=50,
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
            # "object",
        ),
        action_keys=("actions",),
        action_config={"actions": {"normalization": "min_max"}},
        # action_config={"actions": {"normalization": None}},
        dataset_keys=("rewards", "dones"),
        seq_length=action_window_size,  # number of frames at state S and then AFTER S.
        frame_stack=obs_window_size,  # number of frames BEFORE state s to predict
        pad_seq_length=True,
        pad_frame_stack=True,
        filter_by_attribute="valid",
        load_next_obs=False,
        hdf5_normalize_obs=True,
        hdf5_cache_mode="all",
    )

    # Write normalization stats to a pickle file so we can load for inference later.
    normalization_stats = train_dataset.get_obs_normalization_stats()
    with open("normalization_stats.pkl", "wb") as f:
        pickle.dump(normalization_stats, f)

    action_normalization_stats = train_dataset.get_action_normalization_stats()
    with open("action_normalization_stats.pkl", "wb") as f:
        pickle.dump(action_normalization_stats, f)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        # shuffle=False,
        shuffle=True,
        num_workers=0,
        drop_last=True,  # don't provide last batch in dataset pass if it's less than 100 in size
    )

    # visualize the data loaders
    # print(train_dataset.n_demos, valid_dataset.n_demos)
    # data1 = train_dataset[0]
    # data2 = valid_dataset[0]
    # imgs1 = data1["obs"]["agentview_image"]
    # imgs2 = data2["obs"]["agentview_image"]
    # import cv2

    # for i in range(imgs1.shape[0]):
    #     img = imgs1[i]
    #     img = (img).astype("uint8")
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)

    # print("VALID")
    # for i in range(imgs2.shape[0]):
    #     img = imgs2[i]
    #     img = (img).astype("uint8")
    #     cv2.imshow("img", img)
    #     cv2.waitKey(0)

    # import cv2

    # # dataloaders look fine...
    # for i, data in enumerate(train_loader):
    #     imgs = data["obs"]["agentview_image"]
    #     for j in range(imgs.shape[0]):
    #         img = imgs[j]
    #         for k in range(img.shape[0]):
    #             im = img[k]
    #             im = im.numpy().astype("uint8")
    #             print(np.min(im), np.max(im), np.mean(im))
    #             cv2.imshow("img", im)
    #             cv2.waitKey(0)
    # exit()

    # so with seq_length=2 and frame_stack=2, we have:
    # actions [s-1, s, s+1]
    # both obs and next_obs also see to have 3 entries,
    # don't think I need next_obs.

    # TODO: add ema over model

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
                batch = prepare_batch(
                    normalization_stats, batch, obs_window_size, device
                )

                preds, loss, debug = model.forward(batch["actions"], batch["obs"])
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
                batch = prepare_batch(
                    normalization_stats, batch, obs_window_size, device
                )
                preds, loss, debug = model.forward(batch["actions"], batch["obs"])
                val_loss += loss.item()

            # sample inference
            r_sample = np.random.randint(0, batch["obs"]["images"].shape[0])
            batch["obs"]["images"] = batch["obs"]["images"][r_sample].unsqueeze(0)
            batch["obs"]["joints"] = batch["obs"]["joints"][r_sample].unsqueeze(0)
            batch["obs"]["gripper"] = batch["obs"]["gripper"][r_sample].unsqueeze(0)
            # batch["obs"]["object"] = batch["obs"]["object"][r_sample].unsqueeze(0)
            example_pred, debug = model.infer(batch["obs"], delta=0.1)
            print("Example Inference Action:", example_pred[0, 0])
            example_kp = debug["img_kp"][0].cpu().numpy()
            # print("Example Keypoints:", [round(x, 2) for x in example_kp.tolist()])

        val_loss /= len(valid_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Validation Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "best_flow.pth")
            print(f"New best model saved with loss {best_loss:.4f}")

        if (epoch + 1) % save_freq == 0:
            torch.save(model.state_dict(), f"flow_epoch_{epoch+1}.pth")

            # visualize keypoints on an example image from the validation set
            # example_im = batch["obs"]["images"][0, 0]  # first image in the window
            # example_im = example_im.cpu().numpy()
            # example_im = np.ascontiguousarray(example_im.transpose(1, 2, 0))
            # example_im = ((example_im + 1) / 2) * 255
            # example_im = example_im.astype(np.uint8)
            # img_h, img_w, _ = example_im.shape
            # example_kp = debug["img_kp"][0].cpu().numpy()
            # n_keypoints = len(example_kp) // 2
            # all_kp_x = example_kp[:n_keypoints]
            # all_kp_y = example_kp[n_keypoints:]
            # for k in range(n_keypoints):
            #     kp_x = int((all_kp_x[k] + 1) / 2 * img_w)
            #     kp_y = int((all_kp_y[k] + 1) / 2 * img_h)
            #     example_im = cv2.circle(example_im, (kp_x, kp_y), 2, (0, 255, 0), -1)
            # cv2.imwrite(f"keypoints_epoch_{epoch+1}.png", example_im)

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
