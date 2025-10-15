import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.half_dim = dim // 2

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device

        emb = math.log(10000) / (self.half_dim - 1)
        emb = torch.exp(torch.arange(self.half_dim, device=device) * -emb)

        emb = time[:, None] * emb[None, :]  # (B, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (B, dim)

        return emb


class FiLM(nn.Module):
    def __init__(self, in_dim: int, condition_dim: int):
        super(FiLM, self).__init__()
        self.in_dim = in_dim
        self.condition_dim = condition_dim

        # Linear layers to generate scale (gamma) and shift (beta) parameters
        self.gamma_layer = nn.Linear(condition_dim, in_dim)
        self.beta_layer = nn.Linear(condition_dim, in_dim)

    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma_layer(condition)  # (batch_size, in_dim)
        beta = self.beta_layer(condition)  # (batch_size, in_dim)

        gamma.unsqueeze_(-1)  # (batch_size, in_dim, 1)
        beta.unsqueeze_(-1)  # (batch_size, in_dim, 1)

        # Apply FiLM modulation: gamma * features + beta
        modulated_features = gamma * features + beta
        return modulated_features


class TemporalUNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        use_downsample: bool,
        is_decoder: bool = False,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)

        if use_downsample:
            # Stride 2 for temporal downsampling (reducing T_p by half)
            self.sampler = nn.Conv1d(
                out_channels, out_channels, kernel_size=4, stride=2, padding=1
            )
        else:
            # Transposed Conv for temporal upsampling (doubling T_p)
            self.sampler = nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            )

        # 3. FiLM and Normalization
        # BatchNorm1d is standard for channel-first conv features
        self.norm = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()

        self.film = FiLM(in_dim=out_channels, condition_dim=cond_dim)

        self.is_decoder = is_decoder
        self.use_downsample = use_downsample

    def forward(self, x: torch.Tensor, condition: torch.Tensor, skip_x=None):
        if self.is_decoder:
            # Upsampling
            x = self.sampler(x)

            # Skip Connection (Concatenation on the Channel dimension, dim=1)
            # The TemporalUNet implementation must ensure sizes match before cat
            if skip_x is not None:
                x = torch.cat([x, skip_x], dim=1)

        # --- 2. First Convolution + FiLM Modulation ---
        x = self.conv1(x)

        # Apply normalization: [B, D_model, T_p]
        norm_x = self.norm(x)

        # Apply FiLM: condition is [B, D_cond]. Features are [B, D_model, T_p]
        modulated_x = self.film(norm_x, condition)

        x = self.act(x + modulated_x)  # Residual connection + Activation

        x = self.conv2(x)
        x = self.act(x)

        # --- 4. Encoder Specific Operations (Downsample + Skip Output) ---
        if self.use_downsample:
            # Output for skip connection
            skip_out = x
            # Downsampling
            x = self.sampler(x)
            return x, skip_out

        # If it's the bottleneck or final decoder block
        return x, None


class TransformerBlock(nn.Module):
    # unused for now.
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(TransformerBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return out2


class CrossAttentionAggregator(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=num_heads, batch_first=True
        )
        # Query token size must match the new feature_dim (256)
        self.query_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, sequence_features: torch.Tensor) -> torch.Tensor:
        B = sequence_features.shape[0]
        Q = self.query_token.expand(B, -1, -1)
        attn_output, _ = self.attn(Q, sequence_features, sequence_features)
        summary_vector = self.norm(attn_output).squeeze(1)
        return summary_vector


class FlowMatching(nn.Module):
    def __init__(
        self,
        action_dim: int,
        action_window_size: int = 2,
        encoding_dim: int = 256,
        block_channels: list = [256, 512, 1024],
        block_depth: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # TODO: probably this is a UNet, transformer unet? Look at examples.
        super(FlowMatching, self).__init__()
        self.action_dim = action_dim
        self.device = device
        self.action_window_size = action_window_size
        self.encoding_dim = encoding_dim

        # --- OBSERVATION ENCODINGS
        # resnet-18 image encoder, as in diffusion policy.
        resnet = resnet18(pretrained=True)  # NOTE: DP uses untrained
        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-1],
            nn.Flatten(),
        )
        self.joint_encoder = nn.Sequential(
            nn.Linear(action_dim, 256),
            nn.GELU(),
            nn.Linear(256, 512),
        )
        self.observation_aggregator = CrossAttentionAggregator(
            feature_dim=512, num_heads=4
        )
        self.obs_resize = nn.Linear(512, self.encoding_dim // 2)
        # --- End

        self.time_embedding = SinusoidalPositionEmbeddings(dim=self.encoding_dim // 2)

        # --- MAIN TEMPORAL UNET ---
        self.initial_proj = nn.Linear(action_dim, self.encoding_dim)
        self.final_proj = nn.Linear(block_channels[0], action_dim)

        # down
        self.down_blocks = nn.ModuleList()
        curr_dim = self.encoding_dim
        num_levels = len(block_channels)
        for i in range(num_levels):
            # Define channel sizes for the current level
            in_d = block_channels[i - 1] if i > 0 else curr_dim
            out_d = block_channels[i]

            for d in range(block_depth):
                # The in_d for subsequent blocks (d > 0) is constant (out_d)
                current_in_d = in_d if d == 0 else out_d

                use_downsample = i < num_levels - 1 and d == block_depth - 1
                block = TemporalUNetBlock(
                    in_channels=current_in_d,
                    out_channels=out_d,
                    cond_dim=self.encoding_dim,
                    use_downsample=use_downsample,
                )
                self.down_blocks.append(block)
                in_d = out_d

        # bottleneck is in the last down block, since it won't use downsampling.
        # up
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(num_levels)):
            out_d = block_channels[i]
            in_d_prev = block_channels[i + 1] if i < num_levels - 1 else out_d

            # No first level again, it was bottleneck
            if i == num_levels - 1:
                continue

            for d in range(block_depth):
                if d == 0:
                    # Input to the first decoder block: (Prev Level Channel + Skip Channel)
                    current_in_d = in_d_prev
                    use_upsample = True
                else:
                    # Input to subsequent blocks: Constant channel size
                    current_in_d = out_d
                    use_upsample = False

                block = TemporalUNetBlock(
                    in_channels=current_in_d,
                    out_channels=out_d,
                    cond_dim=self.encoding_dim,
                    use_downsample=False,
                    is_decoder=use_upsample,  # Only the first block of the level needs upsampling logic
                )
                self.up_blocks.append(block)

    @torch.no_grad()
    def infer(self, obs, delta):
        # forward euler integration, with delta step size.
        action = torch.normal(0, 1, size=(self.action_dim,))
        for _ in range(0, 1, delta):
            action = action + delta * self.forward(action, obs)

        return action

    def sample_noise(self, shape):
        return torch.normal(0, 1, size=shape, device=self.device)

    def noise_action(self, action, noise, tau):
        # linearly interpolate between distribution of actions and noise
        tau = tau.unsqueeze(-1).unsqueeze(-1)  # Shape: (B, 1, 1)
        return tau * action + (1 - tau) * noise

    def predict(self, noisy_action, obs, tau):

        # TODO: test training. I'm not entirely sure the UNet is setup correctly,
        # since I had a lot of size issues.

        images = obs["images"]
        joints = obs["joints"]

        images = images.permute(0, 1, 4, 2, 3)  # (B, T_o, C, H, W)

        # reshape observations for encoders
        B, T_o = images.shape[:2]
        images = images.reshape(
            B * T_o, images.shape[2], images.shape[3], images.shape[4]
        )  # (B*T_o, C, H, W)
        joints = joints.reshape(B * T_o, -1)  # (B*T_o, joint_dim)
        img_enc = self.image_encoder(images)  # (B*T_o, 512)
        joint_enc = self.joint_encoder(joints)  # (B*T_o, 512)

        # combine the image and joint encodings, and reshape to observation sequence
        fused_features = img_enc + joint_enc  # (B*T_o, 512)
        fused_features = fused_features.view(B, T_o, -1)

        # fuse the entire observation sequence with cross-attention
        observation_vec = self.observation_aggregator(fused_features)  # (B, 512)
        observation_vec = self.obs_resize(observation_vec)  # (B, 256)

        # get sinusoidal position embeddings for time step
        time_enc = self.time_embedding(tau)  # (B, 256)

        # unified conditioning vector
        conditioning_vector = torch.cat(
            [observation_vec, time_enc], dim=-1
        )  # (B, encoding_dim)

        # pass through unet
        x = self.initial_proj(noisy_action)  # (B, T_p, encoding_dim)
        x = x.permute(0, 2, 1)  # (B, encoding_dim, T_p)

        # down + bottleneck
        skip_connections = []
        for block in self.down_blocks:
            x, skip_out = block(x, conditioning_vector)
            if skip_out is not None:
                skip_connections.append(skip_out)

        # up
        for block in self.up_blocks:
            skip_x = None
            if block.is_decoder:
                skip_x = skip_connections.pop() if skip_connections else None
            x, _ = block(x, conditioning_vector, skip_x=skip_x)

        x = x.permute(0, 2, 1)  # (B, T_p, encoding_dim)
        predicted_vector = self.final_proj(x)  # (B, T_p, action_dim)

        return predicted_vector

    def forward(self, action, obs, tau=None):
        if tau is None:
            # for now uniform distn; pi0 uses beta.
            tau = torch.rand(action.shape[0], device=self.device)

        noise = self.sample_noise(shape=action.shape)
        target = noise - action  # eps - A_t

        print(action.shape, noise.shape, target.shape, tau.shape)
        noisy_action = self.noise_action(action, noise=noise, tau=tau)

        pred = self.predict(noisy_action, obs, tau)

        # loss is MSE between predicted vector field and target
        loss = F.mse_loss(pred, target)

        return pred, loss
