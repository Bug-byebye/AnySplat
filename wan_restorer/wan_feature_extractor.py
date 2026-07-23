"""
Wan2.2 Feature Extractor — extracts rich visual features from Wan2.2 video model.

Uses the pretrained Wan2.2-TI2V-5B model's VAE encoder + DiT transformer to
produce high-quality feature maps for Gaussian scene restoration.

Architecture:
    Input images [B, V, 3, H, W]
        │
        ├── WanVAE Encoder → latents [B*V, 48, 1, H/16, W/16]
        ├── Patch Embedding (Conv3d) → tokens [B*V, N, D]
        ├── DiT Blocks (first N layers) → enriched tokens
        ├── Feature Projector → feature maps [B*V, C, H, W]
        │
        └── (optional) Multi-scale feature fusion

Reference:
    Wan: Open and Advanced Large-Scale Video Generative Models
    https://arxiv.org/abs/2503.20314
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import WanRestorerCfg


class WanFeatureExtractor(nn.Module):
    """
    Extracts visual features from Wan2.2 video model for Gaussian restoration.

    The extractor loads the pretrained Wan2.2-TI2V-5B model and uses its
    VAE encoder and DiT transformer to produce rich, multi-scale feature maps
    that serve as priors for Gaussian parameter refinement.

    Key design decisions:
    - Each input view is processed independently through the VAE encoder
      (temporal dim = 1, avoiding multi-frame padding issues)
    - View latents are stacked along the batch dimension for DiT processing
    - DiT features are extracted from intermediate layers and projected
      back to pixel-aligned feature maps
    - The module is fully frozen during training (BN stats also frozen)

    Args:
        cfg: WanRestorerCfg configuration object.
    """

    def __init__(self, cfg: WanRestorerCfg):
        super().__init__()
        self.cfg = cfg
        self.device = None  # set by forward()

        # Will be populated by _load_wan_model()
        self.vae = None
        self.transformer = None
        self.patch_size = None
        self.vae_scale_factor = None
        self.inner_dim = None
        self.dtype = torch.bfloat16 if cfg.use_fp16 else torch.float32

        # Feature projector: DiT tokens → pixel-aligned feature maps
        # Built after loading the model to infer dimensions
        self.feature_projector = None

    def _load_wan_model(self, device: torch.device):
        """Load Wan2.2 model components (called on first forward)."""
        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        print(f"[WanFeatureExtractor] Loading Wan model: {self.cfg.model_id}")

        # 1. Load VAE encoder
        self.vae = AutoencoderKLWan.from_pretrained(
            self.cfg.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,  # VAE in fp32 for numerical stability
            cache_dir=self.cfg.cache_dir,
            local_files_only=False,
        )
        # Only keep the encoder
        if hasattr(self.vae, "decoder"):
            del self.vae.decoder
        self.vae.encoder.requires_grad_(False)
        self.vae.eval()
        self.vae.to(device)

        # VAE scaling factors
        vae_cfg = self.vae.config
        self.vae_scale_factor = {
            "temporal": getattr(vae_cfg, "scale_factor_temporal", 4),
            "spatial": getattr(vae_cfg, "scale_factor_spatial", 8),
        }
        self.z_dim = getattr(vae_cfg, "z_dim", 48)

        # 2. Load DiT transformer
        wan_dtype = torch.bfloat16 if self.cfg.use_fp16 else torch.float32
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.cfg.model_id,
            subfolder="transformer",
            torch_dtype=wan_dtype,
            cache_dir=self.cfg.cache_dir,
            local_files_only=False,
        )
        self.transformer.requires_grad_(False)
        self.transformer.eval()
        self.transformer.to(device)

        # Transformer config
        t_cfg = self.transformer.config
        self.inner_dim = t_cfg.num_attention_heads * t_cfg.attention_head_dim
        self.patch_size = t_cfg.patch_size  # (t_patch, h_patch, w_patch)
        self.num_layers = t_cfg.num_layers

        print(f"[WanFeatureExtractor] Loaded:"
              f" VAE z_dim={self.z_dim},"
              f" Transformer inner_dim={self.inner_dim},"
              f" num_layers={self.num_layers},"
              f" patch_size={self.patch_size}")

        # 3. Build feature projector (DiT tokens → pixel feature maps)
        self._build_feature_projector(device)

    def _build_feature_projector(self, device: torch.device):
        """
        Build a lightweight projector that maps DiT token features to
        pixel-aligned feature maps at the original image resolution.

        The DiT processes latents at [B, D, F', H', W'] where
        F' = F / p_t, H' = H / (16 * p_h), W' = W / (16 * p_w).
        We need to upsample back to the original image resolution.

        Architecture:
            DiT tokens [B, N, C]
                → reshape to [B, C, H', W'] (single frame, no temporal)
                → Conv3x3 + ReLU
                → Pixel shuffle upsampling × factor
                → Conv1x1 → output channels
        """
        p_t, p_h, p_w = self.patch_size
        inner_dim = self.inner_dim

        # Single view is processed, so temporal dim post-patch = 1 (if p_t=1)
        # or ceil(1/p_t) = 1
        # Spatial dims post-patch: H//(16*p_h), W//(16*p_w)

        # Project inner_dim → wan_feat_dim
        in_channels = inner_dim
        if len(self.cfg.dit_output_layer_indices) > 1:
            # Multi-scale: sum features from multiple layers
            in_channels = inner_dim

        self.feature_projector = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.cfg.wan_feat_dim, kernel_size=1, bias=True),
        )

        # Initialize projector for stable output
        for m in self.feature_projector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.feature_projector.requires_grad_(False) if self.cfg.freeze_wan else None
        self.feature_projector.to(device)

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract feature maps from a batch of images.

        Args:
            images: Input images [B, V, 3, H, W] normalized to [0, 1] or [-1, 1].
                    where V = number of views.

        Returns:
            feature_maps: [B*V, wan_feat_dim, H, W] feature maps aligned to
                         the input image resolution.
        """
        if self.vae is None:
            self._load_wan_model(images.device)

        b, v, c, orig_h, orig_w = images.shape
        p_t, p_h, p_w = self.patch_size
        s_s = self.vae_scale_factor["spatial"]
        s_t = self.vae_scale_factor["temporal"]
        device = images.device

        # Pad to divisibility by VAE spatial factor
        pad_h = (s_s - orig_h % s_s) % s_s
        pad_w = (s_s - orig_w % s_s) % s_s

        # Dummy inputs for DiT condition (shared across all views)
        timestep = torch.zeros((1,), device=device, dtype=torch.long)
        dummy_text = torch.zeros((1, 512, self.transformer.config.text_dim),
                                  device=device, dtype=self.dtype)

        # Process each view independently
        all_feat_maps = []

        for batch_idx in range(b):
            for view_idx in range(v):
                # --- Step 1: VAE Encode ---
                img = images[batch_idx, view_idx]  # [3, H, W]
                # VAE runs in float32 (conv weights are float32)
                img_vae = img.to(torch.float32)
                # Create video with 4 identical frames: [1, 3, 4, H', W']
                img_4d = img_vae.unsqueeze(0).unsqueeze(2).repeat(1, 1, s_t, 1, 1)

                # Pad spatial dims
                if pad_h > 0 or pad_w > 0:
                    img_4d = F.pad(img_4d, (0, pad_w, 0, pad_h), mode="reflect")

                # VAE encode → [1, 96, 1, H/16, W/16], take mean → [1, 48, 1, H/16, W/16]
                h = self.vae._encode(img_4d)
                latent = h[:, :self.z_dim]  # [1, 48, 1, H/16, W/16]

                # --- Step 2: DiT Feature Extraction ---

                # --- Step 2: DiT Feature Extraction ---
                # Patch embed: [1, z_dim, 1, H', W'] → [1, N, inner_dim]
                latent_dit = latent.to(self.dtype)
                hidden = self.transformer.patch_embedding(latent_dit)
                hidden = hidden.flatten(2).transpose(1, 2)

                temb, ts_proj, enc_hidden, _ = self.transformer.condition_embedder(
                    timestep, dummy_text, None
                )
                ts_proj = ts_proj.unflatten(1, (6, -1))

                # Run DiT layers
                num_active = min(self.cfg.num_dit_layers, self.num_layers)
                rope_emb = self.transformer.rope(latent_dit)
                layer_outputs = []

                for i in range(num_active):
                    hidden = self.transformer.blocks[i](
                        hidden, enc_hidden, ts_proj, rope_emb
                    )
                    if i in self.cfg.dit_output_layer_indices or i == num_active - 1:
                        layer_outputs.append(hidden)

                # --- Step 3: Feature Projection ---
                if len(layer_outputs) > 1:
                    combined = sum(layer_outputs) / len(layer_outputs)
                else:
                    combined = layer_outputs[-1]

                # Reshape [1, N, C] → [1, C, H_p, W_p] → interpolate → project
                h_patch = latent.shape[3] // p_h
                w_patch = latent.shape[4] // p_w
                feat_tokens = combined[:, :h_patch * w_patch]
                feat_3d = feat_tokens.transpose(1, 2).reshape(
                    1, -1, h_patch, w_patch
                )

                # Resize to full res
                feat_r = F.interpolate(
                    feat_3d, size=(orig_h, orig_w),
                    mode="bilinear", align_corners=False
                )
                # Feature projector is in float32, cast input accordingly
                feat_out = self.feature_projector(feat_r.to(torch.float32))
                all_feat_maps.append(feat_out)

        # Stack: [B*V, wan_feat_dim, H, W]
        return torch.cat(all_feat_maps, dim=0)

    def get_feature_dim(self) -> int:
        """Return the output feature dimension."""
        return self.cfg.wan_feat_dim
