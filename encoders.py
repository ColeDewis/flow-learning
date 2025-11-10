import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.transforms import Resize


class ResnetEncoder(nn.Module):
    def __init__(self, flatten=False):
        """Resnet image encoder

        Args:
            flatten (bool, optional): if true, flatten the output and produce a vector, not an image. Defaults to False.
        """
        super(ResnetEncoder, self).__init__()
        self.resnet = resnet18(
            norm_layer=lambda c: nn.GroupNorm(8, c), pretrained=False
        )
        if flatten:
            self.resnet = nn.Sequential(
                *list(self.resnet.children())[:-1],
                nn.Flatten(),
            )
        else:
            self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # For ref if need a larger output image:
        # resnet.maxpool = nn.Identity()
        # for layer_name in ["layer3", "layer4"]:
        #     layer = getattr(resnet, layer_name)
        #     for block in layer:
        #         if isinstance(block.downsample, nn.Sequential):
        #             # Modify the stride in the downsample convolution
        #             block.downsample[0].stride = (1, 1)
        #         block.conv1.stride = (1, 1)

    def forward(self, x):
        x = self.resnet(x)
        return x


class PointCloudEncoder(nn.Module):
    """Point cloud encoder from 3D-Diffusion-Policy"""

    def __init__(self, in_ch=3, out_ch=256):
        super(PointCloudEncoder, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, out_ch),
            nn.LayerNorm(out_ch),
            nn.GELU(),
        )
        self.projection = nn.Sequential(nn.Linear(out_ch, out_ch), nn.LayerNorm(out_ch))

    def forward(self, pc):
        # pc: [B, N, 3]
        pc = self.mlp1(pc)
        pc = torch.max(pc, dim=1)[0]  # [B, out_ch]
        pc = self.projection(pc)
        return pc


class DinoEncoder(nn.Module):
    def __init__(
        self, output_dim, device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super(DinoEncoder, self).__init__()
        self.dinov2 = (
            torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")  # vitb14
            .eval()
            .to(device)
        )
        self.patch_size = 14  # DINOv2 patch size
        self.dino_embed_dim = 384  # DINOv2 embed dim
        self.input_size = 224

        for param in self.dinov2.parameters():
            param.requires_grad = False

        # project to smaller dim (3x3)
        self.projection = nn.Conv2d(
            in_channels=self.dino_embed_dim,
            out_channels=output_dim,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    def forward(self, image):
        dino_image = self.dinov2.get_intermediate_layers(
            image, n=1, reshape=True, return_class_token=False
        )[0]

        projected_im = self.projection(dino_image)

        return projected_im
