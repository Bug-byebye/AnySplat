"""
Unified Configuration for Gaussian Scene Repair.

Single source of truth: all hyperparameters in one place.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal


@dataclass
class VideoExtractorCfg:
    """Configuration for Wan2.2 video feature extraction."""
    enabled: bool = True
    model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    use_fp16: bool = True
    cache_dir: Optional[str] = None

    # Feature extraction depth
    num_dit_layers: int = 0
    """Number of DiT layers to run (0 = VAE only). More = richer, slower."""
    dit_output_indices: tuple = (4, 8)
    """DiT layers to extract features from (when num_dit_layers > 0)."""

    # Output format
    feat_dim: int = 48
    """Output feature channel dimension. Matches VAE latent dim (48) by default."""


@dataclass
class GaussianEncoderCfg:
    """Configuration for Gaussian feature encoding."""
    hidden_dim: int = 128
    """Hidden dimension for learned Gaussian features."""
    encode_method: Literal["mlp", "none"] = "mlp"
    """How to encode raw Gaussian params."""

    # Video-Gaussian association
    association: Literal["camera_projection", "cross_attention"] = "camera_projection"
    """How to associate video features with each Gaussian."""


@dataclass
class RepairerCfg:
    """Configuration for the Gaussian Repairer."""
    # Per-Gaussian analysis
    per_gaussian_hidden: int = 256
    delta_mean_scale: float = 0.01

    # Gaussian Generator (addition mechanism)
    n_new_queries: int = 256
    """Number of learnable query slots for new Gaussian generation."""
    generator_hidden: int = 256

    # Loss weights
    weight_params_reg: float = 1e-6
    weight_move_reg: float = 3e-4
    weight_delete_sparsity: float = 1e-3
    weight_generator_reg: float = 1e-5
    weight_lpips: float = 0.05

    # Training
    max_steps: int = 5000
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0


@dataclass
class UnifiedCfg:
    """Single unified configuration."""
    video_extractor: VideoExtractorCfg = field(default_factory=VideoExtractorCfg)
    gaussian_encoder: GaussianEncoderCfg = field(default_factory=GaussianEncoderCfg)
    repairer: RepairerCfg = field(default_factory=RepairerCfg)
