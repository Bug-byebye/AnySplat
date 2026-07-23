"""
Gaussian Scene Restoration Framework — Main Orchestrator

Coordinates all components to refine AnySplat's Gaussian scenes using
video foundation model priors.

Workflow:
  Input Images
    ├──→ VideoPriorProvider (frozen Wan2.2) → Video Latents [B*V, 128, H, W]
    │
    └──→ AnySplat (frozen) → Initial Gaussians
                              │
                              └──→ Render/Extract → Gaussian Features [B*V, C_g, H, W]
                                                      │
    Video Latents ──────────────────┐                 │
                                    ├──→ Fusion ──→ Fused Features
                                    │       ↑
                                    └───────┘
                                              │
                                              ▼
                                        Refiner → Δ_params
                                              │
                                              ▼
                                        Refined Gaussians
                                              │
                                              ▼
                                        Render → Loss
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
_script_dir = os.path.dirname(os.path.abspath(__file__))
_post_dir = os.path.dirname(_script_dir)
if _post_dir not in sys.path:
    sys.path.insert(0, _post_dir)
from gaussian_restorer.config import GaussianRestorerCfg
from .video_prior import VideoPriorProvider
from .gaussian_features import GaussianFeatureExtractor
from .fusion import build_fusion
from .refiner import build_refiner


class GaussianSceneRestorer(nn.Module):
    """
    Main orchestrator for Gaussian Scene Restoration.

    Modular architecture with swappable components controlled by config.

    Args:
        cfg: Master configuration.
    """

    def __init__(self, cfg: GaussianRestorerCfg, device: torch.device = None):
        super().__init__()
        self.cfg = cfg

        # 1. Video prior provider (frozen) — eager load if device given
        if cfg.video_prior.enabled:
            self.video_prior = VideoPriorProvider(cfg.video_prior, device=device)
        else:
            self.video_prior = None

        # 2. Gaussian feature extractor
        if cfg.gaussian_features.enabled:
            self.gs_feature_extractor = GaussianFeatureExtractor(cfg.gaussian_features)
            gs_dim = cfg.gaussian_features.gs_feat_dim
        else:
            self.gs_feature_extractor = nn.Identity()
            gs_dim = 83  # passthrough raw dim

        # 3. Compute fusion input channels
        video_dim = cfg.video_prior.wan_feat_dim if cfg.video_prior.enabled else 0
        if cfg.video_prior.enabled and cfg.video_prior.backbone == "none":
            video_dim = 0
        self.video_dim = video_dim
        self.gs_dim = gs_dim

        # 4. Fusion module
        if video_dim > 0 and gs_dim > 0:
            self.fusion = build_fusion(cfg.fusion, video_dim, gs_dim)
            fusion_out_dim = cfg.fusion.hidden_dim
        elif video_dim > 0:
            # Only video features
            self.fusion = nn.Identity()
            fusion_out_dim = video_dim
        elif gs_dim > 0:
            # Only Gaussian features
            self.fusion = nn.Identity()
            fusion_out_dim = gs_dim
        else:
            fusion_out_dim = 3  # RGB only

        # Optionally include RGB
        self.use_rgb = cfg.video_prior.enabled
        refiner_in = fusion_out_dim + (3 if self.use_rgb else 0)

        # 5. Refiner head
        self.refiner = build_refiner(cfg.refiner, refiner_in)

        # Track total params
        self._total_trainable = None

    def forward(
        self,
        input_images: torch.Tensor,         # [B, V, 3, H, W]
        raw_gs_params: torch.Tensor,         # [B*V, 83, H, W]
        return_confidence: bool = False,
    ) -> torch.Tensor:
        """
        Refine Gaussian parameters using video model priors.

        Args:
            input_images: [B, V, 3, H, W] normalized to [0,1].
            raw_gs_params: [B*V, 83, H, W] raw Gaussian params.
            return_confidence: Return confidence map for interpretability.

        Returns:
            refined_params: [B*V, 83, H, W] refined parameters.
        """
        # 1. Extract video prior features
        if self.video_prior is not None:
            video_feats = self.video_prior(input_images)
        else:
            video_feats = None

        # 2. Extract Gaussian features
        gs_feats = self.gs_feature_extractor(raw_gs_params)

        # 3. Fuse features
        if video_feats is not None:
            fused = self.fusion(video_feats, gs_feats)
        else:
            fused = gs_feats

        # 4. Optionally concatenate RGB
        if self.use_rgb:
            rgb = input_images.flatten(0, 1)
            if fused.shape[-2:] != rgb.shape[-2:]:
                fused = F.interpolate(fused, size=rgb.shape[-2:],
                                      mode='bilinear', align_corners=False)
            fused = torch.cat([fused, rgb], dim=1)

        # 5. Refine
        refined = self.refiner(fused, raw_gs_params)

        return refined

    def print_summary(self):
        """Print model summary with component-wise parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"GaussianSceneRestorer Summary:")
        print(f"  Total params:    {total/1e6:.2f}M")
        print(f"  Trainable params: {trainable/1e6:.2f}M")

        if hasattr(self, 'video_prior') and self.video_prior is not None:
            vp = sum(p.numel() for p in self.video_prior.parameters())
            print(f"  VideoPrior:      {vp/1e6:.2f}M (frozen)")
        if hasattr(self, 'gs_feature_extractor') and hasattr(self.gs_feature_extractor, 'parameters'):
            ge = sum(p.numel() for p in self.gs_feature_extractor.parameters())
            print(f"  GS Features:     {ge/1e3:.1f}K")
        if hasattr(self, 'fusion') and hasattr(self.fusion, 'parameters'):
            fu = sum(p.numel() for p in self.fusion.parameters())
            print(f"  Fusion:          {fu/1e3:.1f}K")
        if hasattr(self, 'refiner') and hasattr(self.refiner, 'parameters'):
            re = sum(p.numel() for p in self.refiner.parameters() if p.requires_grad)
            print(f"  Refiner:         {re/1e3:.1f}K")
