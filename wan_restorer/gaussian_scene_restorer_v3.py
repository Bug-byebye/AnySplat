"""
GaussianSceneRestorerV3 — Wan2.2 video model enhanced Gaussian refinement.

Replaces the v1 ResidualUNet+DPT feature extractor with Wan2.2's VAE encoder
and DiT transformer features, providing richer visual priors for per-pixel
Gaussian parameter refinement.

Architecture overview:
    Input images [B, V, 3, H, W]
        │
        ├── WanVAE Encoder → latents → DiT → WAN features [B*V, 128, H, W]
        ├── (optional) DPT Head → DPT features [B*V, 128, H, W]
        │
        └── Fusion: cat(WAN, DPT, RGB) → [B*V, 128+128+3, H, W]
                        │
                        ▼
                  Lightweight Refinement Head
                  (simpler than v1's ResidualUNet since
                   WAN features are already rich)
                        │
                        ▼
                  Δ residuals (zero-init) → refined = raw + Δ
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

import sys, os
_this_dir = os.path.dirname(os.path.abspath(__file__))
_post_dir = os.path.dirname(_this_dir)  # post/
if _post_dir not in sys.path:
    sys.path.insert(0, _post_dir)

from wan_restorer.config import WanRestorerCfg, compute_restorer_in_channels
from wan_restorer.wan_feature_extractor import WanFeatureExtractor
from restorer.feature_extractor import ResidualUNet


class GaussianSceneRestorerV3(nn.Module):
    """
    Wan2.2-enhanced Gaussian parameter refiner (v3).

    Uses Wan2.2 video generation model features as rich visual priors,
    replacing the DPT-only features used in v1.

    Key improvements over v1:
    - Wan's VAE encoder provides 48-channel latent with strong semantic encoding
    - Wan's DiT transformer adds contextual understanding via self-attention
    - Multi-view processing treats views as temporal frames, enabling
      cross-view feature communication
    - The refinement head can be simpler (fewer parameters) since
      WAN features are already high-quality

    Args:
        cfg: WanRestorerCfg configuration.
    """

    def __init__(self, cfg: WanRestorerCfg):
        super().__init__()
        self.cfg = cfg

        # 1. Wan2.2 Feature Extractor
        self.wan_extractor = WanFeatureExtractor(cfg)

        # 2. Compute input channels for the refinement head
        in_channels = compute_restorer_in_channels(cfg)

        # 3. Lightweight refinement head
        # WAN features are already rich, so we can use a smaller UNet
        # than v1 (which needed 4 blocks to compensate for weaker DPT features)
        self.refinement_head = ResidualUNet(
            in_channels=in_channels,
            hidden_dim=cfg.hidden_dim,
            num_blocks=cfg.num_blocks,
        )

        # 4. Residual head — zero-initialized for safe integration
        self.residual_head = nn.Conv2d(cfg.hidden_dim, 83, kernel_size=1)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

    def forward(
        self,
        input_images: torch.Tensor,           # [B, V, 3, H, W]
        raw_gs_params: torch.Tensor,          # [B*V, 83, H, W]
        dpt_feats: Optional[torch.Tensor] = None,  # [B*V, 128, H, W]
    ) -> torch.Tensor:
        """
        Refine Gaussian parameters using WAN-enhanced features.

        Args:
            input_images: Original input images [B, V, 3, H, W].
            raw_gs_params: Raw Gaussian parameters [B*V, 83, H, W].
            dpt_feats: Optional DPT intermediate features [B*V, 128, H, W].

        Returns:
            refined_params: Refined Gaussian parameters [B*V, 83, H, W].
        """
        b, v, _, h, w = input_images.shape

        # Step 1: Extract WAN features
        # Output: [B*V, wan_feat_dim, H, W]
        wan_feats = self.wan_extractor(input_images)

        # Step 2: Concatenate feature sources
        feats = [wan_feats]

        if self.cfg.fuse_with_dpt and dpt_feats is not None:
            assert dpt_feats.shape == wan_feats.shape[:2] + (h, w), \
                f"DPT feats shape {dpt_feats.shape} != wan shape {wan_feats.shape}"
            feats.append(dpt_feats)

        if self.cfg.use_input_images:
            # Flatten batch and views: [B*V, 3, H, W]
            flat_images = input_images.flatten(0, 1)
            feats.append(flat_images)

        concat_feats = torch.cat(feats, dim=1)  # [B*V, in_channels, H, W]

        # Step 3: Refinement
        refined_feats = self.refinement_head(concat_feats)

        # Step 4: Predict residuals
        delta = self.residual_head(refined_feats)  # [B*V, 83, H, W]

        # Clamp opacity logit delta
        delta[:, 0:1] = delta[:, 0:1].clamp(-7, 7)

        # Apply residual
        refined = raw_gs_params + delta
        return refined
