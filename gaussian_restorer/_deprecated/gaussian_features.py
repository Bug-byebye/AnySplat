"""
Gaussian Scene Restoration Framework — Gaussian Feature Extractor

Extracts meaningful feature representations from raw Gaussian parameters
or rendered GP-buffers for fusion with video priors.

Key design: produces pixel-aligned Gaussian feature maps that can be
directly fused with video latents at the same spatial resolution.
"""

from typing import Literal, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GaussianFeatureCfg


class GaussianFeatureExtractor(nn.Module):
    """
    Extracts pixel-aligned Gaussian features from raw parameters.

    Input:  [B*V, 83, H, W] raw Gaussian parameters (pixel-aligned)
            or rendered GP-buffer from 3D Gaussians.

    Output: [B*V, gs_feat_dim, H, W] encoded Gaussian features.

    Encoding strategies:
    - 'none': identity (passthrough)
    - 'conv_embed': Conv1x1 + LN + ReLU (lightweight)
    - 'gp_buffer': Split into semantic groups, per-group norm
    - 'gp_buffer_conv': gp_buffer + depthwise separable convolution per group
    """

    # Semantic grouping of raw GS params
    GROUP_NAMES = ["opacity", "scale", "rotation", "sh_coeffs"]
    GROUP_SLICES = [
        slice(0, 1),     # opacity logit
        slice(1, 4),     # scales (log-space)
        slice(4, 8),     # rotations (quaternion)
        slice(8, 83),    # SH coefficients (3*25=75)
    ]
    GROUP_DIMS = [1, 3, 4, 75]

    def __init__(self, cfg: GaussianFeatureCfg):
        super().__init__()
        self.cfg = cfg

        if cfg.encode_method == "none":
            self.encoder = nn.Identity()
            out_dim = cfg.gs_feat_dim

        elif cfg.encode_method == "conv_embed":
            self.encoder = nn.Sequential(
                nn.Conv2d(83, cfg.gs_feat_dim, kernel_size=1, bias=False),
                nn.LayerNorm(cfg.gs_feat_dim),
                nn.ReLU(inplace=True),
            )
            out_dim = cfg.gs_feat_dim

        elif cfg.encode_method == "gp_buffer":
            # Per-group independent processing
            self.per_group_norm = nn.ModuleDict()
            for name, dim, sl in zip(self.GROUP_NAMES, self.GROUP_DIMS, self.GROUP_SLICES):
                if dim == 1:
                    self.per_group_norm[name] = nn.Identity()
                else:
                    self.per_group_norm[name] = nn.LayerNorm(dim)
            # Project to output dim
            self.project = nn.Conv2d(83, cfg.gs_feat_dim, kernel_size=1, bias=True)
            out_dim = cfg.gs_feat_dim

        elif cfg.encode_method == "gp_buffer_conv":
            # Per-group conv + norm
            self.per_group_conv = nn.ModuleDict()
            total_out = 0
            for name, dim, sl in zip(self.GROUP_NAMES, self.GROUP_DIMS, self.GROUP_SLICES):
                out_ch = min(dim * 2, 32)
                groups = 1 if dim == 1 else min(dim, out_ch)  # groups must divide in_channels
                self.per_group_conv[name] = nn.Sequential(
                    nn.Conv2d(dim, out_ch, kernel_size=3, padding=1, groups=groups,
                              bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                total_out += out_ch
            self.project = nn.Conv2d(total_out, cfg.gs_feat_dim, kernel_size=1, bias=True)
            out_dim = cfg.gs_feat_dim

        self.out_dim = out_dim

    def _apply_gp_buffer(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-group normalization to raw params."""
        outputs = []
        for name, dim, sl in zip(self.GROUP_NAMES, self.GROUP_DIMS, self.GROUP_SLICES):
            group = x[:, sl]  # [B, dim, H, W]
            if dim > 1:
                # LayerNorm over channel dim
                group = group.permute(0, 2, 3, 1)  # [B, H, W, dim]
                group = self.per_group_norm[name](group)
                group = group.permute(0, 3, 1, 2)  # [B, dim, H, W]
            outputs.append(group)
        return torch.cat(outputs, dim=1)

    def _apply_gp_buffer_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-group depthwise conv to raw params."""
        outputs = []
        for name, dim, sl in zip(self.GROUP_NAMES, self.GROUP_DIMS, self.GROUP_SLICES):
            group = x[:, sl]
            outputs.append(self.per_group_conv[name](group))
        return torch.cat(outputs, dim=1)

    def forward(self, raw_gs_params: torch.Tensor) -> torch.Tensor:
        """
        Encode raw Gaussian parameters into feature representation.

        Args:
            raw_gs_params: [B*V, 83, H, W] pixel-aligned Gaussian params.

        Returns:
            gs_features: [B*V, gs_feat_dim, H, W] encoded features.
        """
        if self.cfg.encode_method == "gp_buffer":
            feats = self._apply_gp_buffer(raw_gs_params)
            return self.project(feats)
        elif self.cfg.encode_method == "gp_buffer_conv":
            feats = self._apply_gp_buffer_conv(raw_gs_params)
            return self.project(feats)
        else:
            # 'none' or 'conv_embed'
            return self.encoder(raw_gs_params)
