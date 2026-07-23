"""
GaussianSceneRestorer — Per-pixel residual refinement of Gaussian parameters.

This is the entry point of the Gaussian Scene Restoration module.
It takes the intermediate DPT features (already computed by the frozen
VGGT_DPT_GS_Head) and the raw Gaussian parameters, and predicts per-pixel
residual deltas to refine the Gaussians.

Design rationale:
  - Operates pre-GaussianAdapter (on raw params), so the adapter's
    nonlinear transforms (softplus, quaternion norm) apply correctly.
  - Predicts residuals (not absolute values) — zero-initialized so
    the initial forward pass produces no degradation.
  - Only refines scales, rotations, SH coefficients, and opacity
    logit. Means are NOT modified (left to the depth/point head).
  - The FeatureExtractor (ResidualUNet) can be swapped for a video
    foundation model in future versions.

Reference:
  - AnchorSplat-style residual prediction on Gaussian parameters
  - GaussFusion-inspired use of external visual priors
    (currently the DPT multi-scale fused features serve as the prior)
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from .feature_extractor import ResidualUNet


@dataclass
class GaussianSceneRestorerCfg:
    """Configuration for the GaussianSceneRestorer module.

    Set enabled=True to activate. When disabled, the module is not created
    and the original forward pass is unchanged.
    """
    enabled: bool = False
    """Whether to enable the scene restorer. Default False preserves original behavior."""

    dpt_feat_dim: int = 128
    """Channel dimension of the DPT intermediate features (fixed by VGGT_DPT_GS_Head)."""

    hidden_dim: int = 64
    """Base channel count for the ResidualUNet feature extractor."""

    num_blocks: int = 4
    """Number of encoder/decoder levels in the U-Net."""

    use_input_images: bool = True
    """Whether to concatenate input RGB images as additional channels."""

    freeze_encoder: bool = True
    """When True, freezes all encoder parameters except the restorer during training."""


class GaussianSceneRestorer(nn.Module):
    """
    Per-pixel Gaussian parameter refiner.

    Refines raw Gaussian parameters (83 channels: 1 density + 78 GS params)
    by predicting per-pixel residuals from DPT intermediate features.

    The module is zero-initialized: at step 0, residuals are zero and
    the output equals the input. This ensures safe integration without
    breaking the existing pipeline.
    """

    def __init__(self, cfg: GaussianSceneRestorerCfg):
        super().__init__()
        self.cfg = cfg

        # Compute input channels for the U-Net
        in_channels = cfg.dpt_feat_dim
        if cfg.use_input_images:
            in_channels += 3  # RGB images

        self.feature_extractor = ResidualUNet(
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            num_blocks=cfg.num_blocks,
        )

        # Output head: per-pixel residual for each Gaussian parameter
        # raw_gs_dim = 1 (density/logit) + 7 (scales 3 + rotations 4) + 75 (SH 3*25) = 83
        self.residual_head = nn.Conv2d(cfg.hidden_dim, 83, kernel_size=1)

        # Zero-initialize the residual head so the first forward pass is safe
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        dpt_feats: torch.Tensor,
        raw_gs_params: torch.Tensor,
        input_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            dpt_feats:      DPT fused features           [B*V, 128, H, W]
            raw_gs_params:  Raw Gaussian parameters      [B*V, 83, H, W]
            input_images:   Original input images        [B*V, 3, H, W]

        Returns:
            refined_params: Refined Gaussian parameters  [B*V, 83, H, W]
        """
        # Concatenate inputs
        feats = dpt_feats
        if self.cfg.use_input_images and input_images is not None:
            feats = torch.cat([feats, input_images], dim=1)

        # Extract restoration features
        restorer_feats = self.feature_extractor(feats)  # [B*V, hidden_dim, H, W]

        # Predict residuals
        delta = self.residual_head(restorer_feats)  # [B*V, 83, H, W]

        # Clamp opacity logit delta (channel 0 = density in logit space)
        # This prevents the restorer from pushing opacity to extreme values
        delta[:, 0:1] = delta[:, 0:1].clamp(-7, 7)

        # Apply residual
        refined = raw_gs_params + delta
        return refined
