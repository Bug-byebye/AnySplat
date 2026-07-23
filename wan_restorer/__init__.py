"""
WAN-Enhanced Gaussian Scene Restorer (v3)

Uses Wan2.2 video generation model as a rich visual feature extractor
for per-pixel Gaussian parameter refinement.

Architecture:
  - Wan2.2 VAE Encoder for latent feature extraction
  - Optional Wan2.2 DiT blocks for contextual feature enhancement
  - Feature projector to pixel-aligned feature maps
  - Zero-init residual head for safe integration

Reference:
  - Wan: Open and Advanced Large-Scale Video Generative Models
    https://arxiv.org/abs/2503.20314
  - Wan2.2: https://github.com/Wan-Video/Wan2.2
"""

import sys, os
_this_dir = os.path.dirname(os.path.abspath(__file__))
_post_dir = os.path.dirname(_this_dir)
if _post_dir not in sys.path:
    sys.path.insert(0, _post_dir)

from wan_restorer.config import WanRestorerCfg
from wan_restorer.wan_feature_extractor import WanFeatureExtractor
from wan_restorer.gaussian_scene_restorer_v3 import GaussianSceneRestorerV3
