"""
Gaussian Scene Repair Framework — Unified Implementation (v4)
=============================================================

This is the SINGLE canonical implementation. The previous versions
(post/restorer/ v1, post/wan_restorer/ v3) are kept for reference
but deprecated for active development.

Core components:
  - VideoExtractor:     Temporal-aware Wan2.2 feature extraction
  - GaussianFeatureEncoder:  Gaussian encoding + video feature association
  - UnifiedRepairer:    PerGaussian analysis + Gaussian generator
  - RepairLoss:         Multi-component loss

Usage:
    from post.gaussian_restorer import (
        UnifiedRepairer, GaussianFeatureEncoder, VideoExtractor, RepairLoss
    )
"""

from .config import UnifiedCfg, VideoExtractorCfg, GaussianEncoderCfg, RepairerCfg
from .video_extractor import VideoExtractor, DiTAdapter
from .gaussian_encoder import GaussianFeatureEncoder
from .repairer import UnifiedRepairer, build_refined_gaussians, build_rotation_matrix
from .repair_loss import RepairLoss
from .utils import normalize_intrinsics, save_image, render_gaussians, extract_params_from_gaussians
