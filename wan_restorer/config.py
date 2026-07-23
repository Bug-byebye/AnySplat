"""
Configuration for Wan2.2-enhanced Gaussian Scene Restorer (v3).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class WanRestorerCfg:
    """Configuration for the Wan2.2-enhanced Gaussian Scene Restorer.

    Enables using Wan2.2 video generation model features as rich visual priors
    for Gaussian parameter refinement, replacing/augmenting the DPT features
    used in v1.
    """
    # === Wan model settings ===
    enabled: bool = False
    """Whether to enable the Wan-enhanced restorer."""

    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    """HuggingFace model ID for the Wan2.2 model."""

    use_fp16: bool = True
    """Whether to load Wan model in bfloat16 for memory efficiency."""

    cache_dir: Optional[str] = None
    """Optional local cache directory for model weights."""

    # === Feature extraction settings ===
    use_vae_encoder: bool = True
    """Whether to use the Wan VAE encoder for feature extraction."""

    num_dit_layers: int = 8
    """Number of DiT transformer layers to run (0 = VAE features only).
    More layers = richer features but slower and more memory."""

    dit_output_layer_indices: tuple = (4, 8)
    """Indices of DiT layers from which to extract intermediate features.
    Features are concatenated to form multi-scale representations."""

    wan_feat_dim: int = 128
    """Output channel dimension for Wan-extracted features."""

    # === Fusion settings ===
    fuse_with_dpt: bool = True
    """Whether to fuse Wan features with DPT intermediate features."""

    dpt_feat_dim: int = 128
    """Channel dimension of DPT features (when fuse_with_dpt=True)."""

    use_input_images: bool = True
    """Whether to concatenate input RGB images as additional channels."""

    # === Refinement head settings ===
    hidden_dim: int = 64
    """Base channel count for the refinement network."""

    num_blocks: int = 4
    """Number of encoder/decoder levels in the refinement U-Net."""

    # === Training settings ===
    freeze_encoder: bool = True
    """When True, freezes all encoder params except the restorer."""

    freeze_wan: bool = True
    """When True, freezes all Wan model parameters during training."""


# Input channel calculation helpers

def compute_restorer_in_channels(cfg: WanRestorerCfg) -> int:
    """Compute the total input channels for the refinement head."""
    channels = cfg.wan_feat_dim  # Wan features
    if cfg.fuse_with_dpt:
        channels += cfg.dpt_feat_dim  # DPT features
    if cfg.use_input_images:
        channels += 3  # RGB images
    return channels
