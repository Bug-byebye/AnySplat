"""
Config adapter — bridges between AnySplat's YAML-based config system and
the GaussianSceneRestorerCfg dataclass.

Use this to programmatically enable the restorer on an existing AnySplat config:

    from integrations.config_adapter import enable_restorer

    cfg = get_cfg()  # your AnySplat config
    enable_restorer(cfg, hidden_dim=128, use_input_images=True)
"""

from copy import deepcopy
from typing import Any, Dict, Optional


# Default restorer configuration (mirrors post/config/anysplat_restorer.yaml)
DEFAULT_RESTORER_CFG = {
    "enabled": True,
    "dpt_feat_dim": 128,
    "hidden_dim": 64,
    "num_blocks": 4,
    "use_input_images": True,
    "freeze_encoder": True,
}


def enable_restorer(
    cfg: Dict[str, Any],
    enabled: bool = True,
    hidden_dim: int = 64,
    num_blocks: int = 4,
    use_input_images: bool = True,
    freeze_encoder: bool = True,
    dpt_feat_dim: int = 128,
) -> Dict[str, Any]:
    """
    Enable the Gaussian Scene Restoration module on an AnySplat config dict.

    This is equivalent to adding the following to your YAML:

        model:
          encoder:
            gaussian_scene_restorer:
              enabled: true
              hidden_dim: 64
              num_blocks: 4
              use_input_images: true
              freeze_encoder: true

    Args:
        cfg: AnySplat configuration dictionary (nested).
        enabled: Whether to enable the restorer.
        hidden_dim: Base channel count for ResidualUNet.
        num_blocks: Number of U-Net encoder/decoder levels.
        use_input_images: Concatenate input RGB with DPT features.
        freeze_encoder: Freeze all encoder params except restorer.
        dpt_feat_dim: DPT feature channel dimension (fixed at 128).

    Returns:
        Modified config dict (in-place and returned).
    """
    # Navigate to model.encoder.gaussian_scene_restorer
    model = cfg.setdefault("model", {})
    encoder = model.setdefault("encoder", {})
    encoder["gaussian_scene_restorer"] = {
        "enabled": enabled,
        "dpt_feat_dim": dpt_feat_dim,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "use_input_images": use_input_images,
        "freeze_encoder": freeze_encoder,
    }
    return cfg


def enable_training_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply the full training configuration for restorer fine-tuning.

    Sets:
    - Freeze backbone/encoder
    - Disable distillation losses
    - Set appropriate learning rates
    - Disable unused loss terms

    See post/config/anysplat_restorer.yaml for the full reference.
    """
    cfg = enable_restorer(cfg)

    # Optimization: freeze backbone, zero its LR multiplier
    model = cfg.setdefault("model", {})
    encoder = model.setdefault("encoder", {})
    encoder["freeze_backbone"] = True
    encoder["freeze_module"] = "all"
    encoder["distill"] = False

    # Training params
    optimizer = cfg.setdefault("optimizer", {})
    optimizer.setdefault("lr", 1e-3)
    optimizer["backbone_lr_multiplier"] = 0.0
    optimizer.setdefault("warm_up_steps", 500)

    trainer = cfg.setdefault("trainer", {})
    trainer.setdefault("max_steps", 10000)

    # Disable distillation losses
    train_cfg = cfg.setdefault("train", {})
    train_cfg["weight_pose"] = 0.0
    train_cfg["weight_depth"] = 0.0
    train_cfg["weight_normal"] = 0.0
    train_cfg["pose_loss_alpha"] = 1.0

    return cfg
