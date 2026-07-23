"""
Lightweight Residual U-Net for Gaussian Scene Restoration.

Architecture:
  - 4-level encoder-decoder with skip connections
  - Each level: Conv3x3 + BatchNorm + ReLU + residual
  - Total params: ~840k at hidden_dim=64, ~3.3M at hidden_dim=128
  - Memory: ~1GB activations at 256x256 with 4 views batch

This module is designed as a drop-in replacement target for v2/v3 upgrades
(e.g., replacing with video foundation model features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Pre-activation residual block:
      in → BN → ReLU → Conv3×3 → BN → ReLU → Conv3×3 → +in
    """
    def __init__(self, channels: int):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.norm1(x)
        out = F.relu_(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu_(out)
        out = self.conv2(out)
        out = out + identity
        return out


class ResidualUNet(nn.Module):
    """
    Lightweight residual U-Net with skip connections.

    Args:
        in_channels:  Number of input channels (e.g., 128 DPT + 3 RGB = 131)
        hidden_dim:   Base channel count (default 64)
        num_blocks:   Number of encoder/decoder levels (default 4)

    Channel progression (num_blocks=4, hidden_dim=64):
        Encoder (down):  64 → 128 → 256 → 256
        Decoder (up):    256 → 128 → 64 → 64
        Each decoder level receives: upsampled_feat + skip → project → residual

    Forward:
        Input:  [B, in_channels, H, W]
        Output: [B, hidden_dim, H, W]
    """
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 64,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.hidden_dim = hidden_dim

        # --- Initial projection: in_channels → hidden_dim ---
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # --- Encoder (downsampling path) ---
        # Each level: residual block(s), then downsample
        enc_channels = []
        self.enc_blocks = nn.ModuleList()
        self.enc_down = nn.ModuleList()

        cur_ch = hidden_dim
        for i in range(num_blocks):
            out_ch = min(hidden_dim * (2 ** i), 256)
            enc_channels.append(cur_ch)

            block = nn.Sequential(
                ResidualBlock(cur_ch),
                ResidualBlock(cur_ch) if cur_ch <= 128 else nn.Identity(),
            )
            self.enc_blocks.append(block)

            if i < num_blocks - 1:
                next_ch = min(hidden_dim * (2 ** (i + 1)), 256)
                self.enc_down.append(
                    nn.Sequential(
                        nn.Conv2d(cur_ch, next_ch, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(next_ch),
                        nn.ReLU(inplace=True),
                    )
                )
                cur_ch = next_ch
            else:
                self.enc_down.append(nn.Identity())

        # Bottleneck residual after deepest level
        self.bottleneck = ResidualBlock(cur_ch)

        # --- Decoder (upsampling path) ---
        self.dec_up = nn.ModuleList()
        self.dec_proj = nn.ModuleList()  # project (upsampled + skip) to target ch
        self.dec_blocks = nn.ModuleList()

        for i in range(num_blocks - 1, -1, -1):
            skip_ch = enc_channels[i]
            target_ch = max(hidden_dim, cur_ch // 2) if i > 0 else hidden_dim

            # Upsample: cur_ch → target_ch
            if cur_ch != target_ch or i < num_blocks - 1:
                self.dec_up.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                        nn.Conv2d(cur_ch, target_ch, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(target_ch),
                        nn.ReLU(inplace=True),
                    )
                )
            else:
                self.dec_up.append(nn.Identity())

            # Project concatenated [target_ch + skip_ch] → target_ch
            cat_ch = target_ch + skip_ch
            self.dec_proj.append(
                nn.Sequential(
                    nn.Conv2d(cat_ch, target_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(target_ch),
                    nn.ReLU(inplace=True),
                )
            )

            # Residual blocks
            self.dec_blocks.append(
                nn.Sequential(
                    ResidualBlock(target_ch),
                    ResidualBlock(target_ch) if target_ch <= 128 else nn.Identity(),
                )
            )

            cur_ch = target_ch

        # --- Final refinement ---
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input projection
        x = self.input_proj(x)  # [B, hidden_dim, H, W]

        # Encoder
        skips = []
        for i in range(self.num_blocks):
            x = self.enc_blocks[i](x)
            skips.append(x)
            x = self.enc_down[i](x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for i in range(self.num_blocks):
            x = self.dec_up[i](x)
            skip = skips[-(i + 1)]
            # Align spatial dims if needed (due to odd resolutions)
            if x.shape[-2] != skip.shape[-2] or x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            # Project back to target channels
            x = self.dec_proj[i](x)
            # Residual blocks
            x = self.dec_blocks[i](x)

        x = self.final_conv(x)
        return x
