"""
Gaussian Scene Restoration Framework — Refiner Heads

Predicts Gaussian parameter refinements from fused features.
Multiple strategies available for the final refinement step.
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_post_dir = os.path.dirname(_script_dir)
if _post_dir not in sys.path:
    sys.path.insert(0, _post_dir)
from restorer.feature_extractor import ResidualUNet
from .config import RefinerCfg


class RefinerHead(nn.Module):
    """
    Abstract base for all refiner heads.

    Input:  fused_feats  [B*V, C_in, H, W] — fused features
            raw_params   [B*V, 83, H, W]   — raw Gaussian params (optional)

    Output: delta        [B*V, 83, H, W]   — refinement deltas
            (optional)   confidence         — [B*V, 1, H, W]
    """

    def __init__(self, cfg: RefinerCfg, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

    def forward(self, fused_feats: torch.Tensor,
                raw_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        raise NotImplementedError


class ResidualRefiner(RefinerHead):
    """
    Direct parameter residual prediction (v1/v3 approach).

    fused_feats → U-Net → Conv1x1 → Δ_params
    refined = raw + Δ

    Zero-initialized final layer for safe integration.
    """

    def __init__(self, cfg: RefinerCfg, in_channels: int):
        super().__init__(cfg, in_channels)

        self.num_blocks = cfg.num_blocks
        self.hidden_dim = cfg.hidden_dim

        # Use ResidualUNet as refinement backbone
        self.unet = ResidualUNet(
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            num_blocks=cfg.num_blocks,
        )

        # Residual head: zero-init
        self.residual_head = nn.Conv2d(cfg.hidden_dim, 83, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(self, fused_feats: torch.Tensor,
                raw_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.unet(fused_feats)
        delta = self.residual_head(feats)
        delta[:, 0:1] = delta[:, 0:1].clamp(-7, 7)

        if raw_params is not None:
            return raw_params + delta
        return delta


class FeatureResidualRefiner(RefinerHead):
    """
    Predict residuals in a learned lower-dimensional feature space,
    then decode to parameter space.

    fused_feats → encoder → latent_feats → decoder → Δ_params
    """

    def __init__(self, cfg: RefinerCfg, in_channels: int):
        super().__init__(cfg, in_channels)
        latent_dim = cfg.latent_dim

        self.encoder = ResidualUNet(
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            num_blocks=cfg.num_blocks,
        )

        self.latent_head = nn.Conv2d(cfg.hidden_dim, latent_dim, kernel_size=1)
        nn.init.zeros_(self.latent_head.weight)
        nn.init.zeros_(self.latent_head.bias)

        # Decoder: latent → 83-param delta
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(latent_dim * 2, 83, kernel_size=1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, fused_feats: torch.Tensor,
                raw_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.encoder(fused_feats)
        latent = self.latent_head(feats)
        delta = self.decoder(latent)
        delta[:, 0:1] = delta[:, 0:1].clamp(-7, 7)

        if raw_params is not None:
            return raw_params + delta
        return delta


class ConfidenceRefiner(RefinerHead):
    """
    Confidence-weighted refinement with interpretable per-pixel masking.

    Two-headed architecture:
    - Confidence head: predicts per-pixel importance (0-1)
    - Residual head: predicts refinement deltas

    refined = raw + confidence * delta

    The model learns which pixels need refinement (occluded regions,
    low-quality Gaussians) and which should remain unchanged.
    """

    def __init__(self, cfg: RefinerCfg, in_channels: int):
        super().__init__(cfg, in_channels)

        self.num_blocks = cfg.num_blocks
        self.hidden_dim = cfg.hidden_dim

        # Shared backbone
        self.unet = ResidualUNet(
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            num_blocks=cfg.num_blocks,
        )

        # Residual head: zero-init
        self.residual_head = nn.Conv2d(cfg.hidden_dim, 83, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        # Confidence head: initialize to conservative (low confidence)
        self.confidence_head = nn.Conv2d(cfg.hidden_dim, 1, kernel_size=1)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.constant_(self.confidence_head.bias, cfg.confidence_bias_init)

        self.use_confidence = cfg.use_confidence

    def forward(self, fused_feats: torch.Tensor,
                raw_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        feats = self.unet(fused_feats)

        delta = self.residual_head(feats)
        delta[:, 0:1] = delta[:, 0:1].clamp(-7, 7)

        if self.use_confidence:
            confidence = torch.sigmoid(self.confidence_head(feats))
            delta = delta * confidence

        if raw_params is not None:
            return raw_params + delta
        return delta

    def get_confidence(self, fused_feats: torch.Tensor) -> torch.Tensor:
        """Get confidence map for interpretability."""
        feats = self.unet(fused_feats)
        return torch.sigmoid(self.confidence_head(feats))


class MultiStageRefiner(RefinerHead):
    """
    Coarse-to-fine multi-stage refinement.

    Processes at progressively higher resolutions:
    Stage 1: 1/4 resolution — global structure
    Stage 2: 1/2 resolution — mid-level details
    Stage 3: Full resolution — fine details

    Each stage refines the output of the previous stage.
    """

    def __init__(self, cfg: RefinerCfg, in_channels: int):
        super().__init__(cfg, in_channels)
        self.num_stages = cfg.num_blocks  # Use num_blocks as proxy for stages
        self.hidden_dim = cfg.hidden_dim

        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage_hidden = max(cfg.hidden_dim // (2 ** i), 16)
            stage = nn.Sequential(
                nn.Conv2d(in_channels if i == 0 else 83, stage_hidden,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stage_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(stage_hidden, stage_hidden,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(stage_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(stage_hidden, 83, kernel_size=1),
            )
            # Zero-init final conv
            nn.init.zeros_(stage[-1].weight)
            nn.init.zeros_(stage[-1].bias)
            self.stages.append(stage)

    def forward(self, fused_feats: torch.Tensor,
                raw_params: Optional[torch.Tensor] = None) -> torch.Tensor:
        current = raw_params.clone() if raw_params is not None else None

        feats_pyramid = []
        f = fused_feats
        for i in range(self.num_stages):
            feats_pyramid.append(f)
            if i < self.num_stages - 1:
                f = F.avg_pool2d(f, kernel_size=2)

        for i in range(self.num_stages - 1, -1, -1):
            if i < self.num_stages - 1 and current is not None:
                current = F.interpolate(current, scale_factor=2,
                                        mode='bilinear', align_corners=False)

            stage_input = feats_pyramid[i] if current is None else \
                          torch.cat([feats_pyramid[i], current], dim=1) \
                          if feats_pyramid[i].shape[-2:] == current.shape[-2:] else \
                          feats_pyramid[i]

            # Align spatial dims
            if current is not None and stage_input.shape[-2:] != current.shape[-2:]:
                stage_input = F.interpolate(
                    stage_input, size=current.shape[-2:],
                    mode='bilinear', align_corners=False)

            delta = self.stages[i](stage_input)
            delta[:, 0:1] = delta[:, 0:1].clamp(-7, 7)

            if current is not None:
                if current.shape[-2:] != delta.shape[-2:]:
                    delta = F.interpolate(delta, size=current.shape[-2:],
                                          mode='bilinear', align_corners=False)
                current = current + delta
            else:
                current = delta

        return current


def build_refiner(cfg: RefinerCfg, in_channels: int) -> RefinerHead:
    """Factory: build refiner head from config."""
    if cfg.mode == "residual":
        return ResidualRefiner(cfg, in_channels)
    elif cfg.mode == "feature_residual":
        return FeatureResidualRefiner(cfg, in_channels)
    elif cfg.mode == "confidence":
        return ConfidenceRefiner(cfg, in_channels)
    elif cfg.mode == "multi_stage":
        return MultiStageRefiner(cfg, in_channels)
    else:
        raise ValueError(f"Unknown refiner mode: {cfg.mode}")
