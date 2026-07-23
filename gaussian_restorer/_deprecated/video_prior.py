"""
Gaussian Scene Restoration Framework — Video Prior Provider

Extracts latent representations from pretrained video foundation models (Wan2.2)
as scene priors for Gaussian refinement. Designed for modular replacement —
swap backbone via config without changing pipeline code.

Key design principle: video model acts as a frozen Prior Provider, not a
generator. We extract internal latents (VAE + DiT intermediate features)
directly, avoiding encode-decode-reencode cycles.
"""

from typing import Optional, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import VideoPriorCfg


class VideoPriorProvider(nn.Module):
    """
    Extracts hierarchical latent features from pretrained video models.

    Supports multiple backbones via configurable `backbone` parameter.
    Currently implemented: 'wan' (Wan2.2-TI2V-5B).
    Future: 'dino', 'sd', 'siglip'.

    The provider is always frozen during training (acts as fixed prior).

    Args:
        cfg: VideoPriorCfg configuration.
    """

    def __init__(self, cfg: VideoPriorCfg, device: torch.device = None):
        super().__init__()
        self.cfg = cfg
        self._model = None
        self._device = device
        self._dtype = torch.bfloat16 if cfg.use_fp16 else torch.float32

        if device is not None and cfg.backbone != "none":
            self._load_model(device)

    def _load_model(self, device: torch.device):
        """Load the video backbone model (called eagerly if device given at init)."""
        self._device = device

        if self.cfg.backbone == "wan":
            self._load_wan(device)
        elif self.cfg.backbone == "none":
            pass  # No prior model, identity output
        else:
            raise ValueError(f"Unknown backbone: {self.cfg.backbone}")

    def _load_wan(self, device: torch.device):
        """Load Wan2.2 model components (VAE encoder + DiT)."""
        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        print(f"[VideoPriorProvider] Loading Wan model: {self.cfg.model_id}")

        # VAE in fp32 for numerical stability
        self.vae = AutoencoderKLWan.from_pretrained(
            self.cfg.model_id, subfolder="vae",
            torch_dtype=torch.float32, cache_dir=self.cfg.cache_dir,
        )
        self.vae.encoder.requires_grad_(False)
        self.vae.eval().to(device)

        # Only keep encoder, free memory
        if hasattr(self.vae, "decoder"):
            del self.vae.decoder

        self.z_dim = getattr(self.vae.config, "z_dim", 48)
        self.s_t = self.vae.config.scale_factor_temporal  # 4
        self.s_s = self.vae.config.scale_factor_spatial  # 16

        # DiT transformer in bf16
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.cfg.model_id, subfolder="transformer",
            torch_dtype=self._dtype, cache_dir=self.cfg.cache_dir,
        )
        self.transformer.requires_grad_(False)
        self.transformer.eval().to(device)

        t_cfg = self.transformer.config
        self.inner_dim = t_cfg.num_attention_heads * t_cfg.attention_head_dim
        self.num_layers = t_cfg.num_layers
        self.patch_size = t_cfg.patch_size

        print(f"  Loaded: VAE z_dim={self.z_dim}, DiT layers={self.num_layers}, "
              f"inner_dim={self.inner_dim}")

        # Feature projector: DiT tokens → pixel-aligned feature map
        self._build_projector(device)

    def _build_projector(self, device: torch.device):
        """Build lightweight projector from DiT features to pixel feature maps."""
        in_dim = self.inner_dim
        self.feature_projector = nn.Sequential(
            nn.Conv2d(in_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, self.cfg.wan_feat_dim, kernel_size=1, bias=True),
        )
        for m in self.feature_projector.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.feature_projector.to(device)

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract prior features from input images.

        Args:
            images: [B, V, 3, H, W] input images.

        Returns:
            prior_feats: [B*V, wan_feat_dim, H, W] pixel-aligned features.
        """
        if self.cfg.backbone != "none" and not hasattr(self, 'vae'):
            self._load_model(images.device)

        if self.cfg.backbone == "none":
            B, V, _, H, W = images.shape
            return torch.zeros(B * V, self.cfg.wan_feat_dim, H, W, device=images.device)

        return self._forward_wan(images)

    def _forward_wan(self, images: torch.Tensor) -> torch.Tensor:
        """Wan2.2 forward: VAE encode + DiT feature extraction."""
        B, V, C, H, W = images.shape
        device = images.device
        dtype = self._dtype
        num_layers = min(self.cfg.num_dit_layers, self.num_layers)

        # Dummy conditioning for feature extraction mode (timestep=0, zero text)
        timestep = torch.zeros((1,), device=device, dtype=torch.long)
        text_dim = self.transformer.config.text_dim
        dummy_text = torch.zeros((1, 512, text_dim), device=device, dtype=dtype)

        all_feats = []
        for b_idx in range(B):
            for v_idx in range(V):
                # --- VAE Encode ---
                img = images[b_idx, v_idx].to(torch.float32)
                video = img.unsqueeze(0).unsqueeze(2).repeat(1, 1, self.s_t, 1, 1)
                h = self.vae._encode(video)
                latent = h[:, :self.z_dim]  # [1, 48, 1, H/16, W/16]

                if num_layers == 0:
                    # VAE-only: project directly
                    feat = self._project_vae_latent(latent, H, W)
                else:
                    # DiT feature extraction
                    feat = self._forward_dit(latent, timestep, dummy_text,
                                              num_layers, H, W)

                all_feats.append(feat)

        return torch.cat(all_feats, dim=0)

    def _forward_dit(self, latent, timestep, dummy_text, num_layers, H, W):
        """Run DiT on latent and extract intermediate features."""
        latent_dit = latent.to(self._dtype)
        rope_emb = self.transformer.rope(latent_dit)

        # Patch embedding
        hidden = self.transformer.patch_embedding(latent_dit)
        hidden = hidden.flatten(2).transpose(1, 2)

        # Condition embedding
        temb, ts_proj, enc_hidden, _ = self.transformer.condition_embedder(
            timestep, dummy_text, None
        )
        ts_proj = ts_proj.unflatten(1, (6, -1))

        # Run DiT layers, collect features
        layer_outputs = []
        for i in range(num_layers):
            hidden = self.transformer.blocks[i](hidden, enc_hidden, ts_proj, rope_emb)
            if i in self.cfg.dit_output_indices or i == num_layers - 1:
                layer_outputs.append(hidden)

        # Aggregate and project
        if len(layer_outputs) > 1:
            combined = sum(layer_outputs) / len(layer_outputs)
        else:
            combined = layer_outputs[-1]

        # Reshape tokens → spatial map
        h_patch = latent.shape[3] // self.patch_size[1]
        w_patch = latent.shape[4] // self.patch_size[2]
        feat_tokens = combined[:, :h_patch * w_patch]
        feat_3d = feat_tokens.transpose(1, 2).reshape(1, -1, h_patch, w_patch)

        # Resize and project
        feat_r = F.interpolate(feat_3d, size=(H, W), mode="bilinear", align_corners=False)
        return self.feature_projector(feat_r.to(torch.float32))

    def _project_vae_latent(self, latent, H, W):
        """Project VAE latent directly to feature map (no DiT)."""
        feat = latent[:, :, 0].to(torch.float32)  # [1, 48, H/16, W/16]
        feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)
        # Use feature_projector (expects inner_dim input, but we have z_dim input)
        # For VAE-only, we need a separate projector
        if hasattr(self, 'vae_projector'):
            return self.vae_projector(feat)
        # Fallback: just interpolate to target dim
        projector = nn.Conv2d(self.z_dim, self.cfg.wan_feat_dim, 1).to(feat.device)
        return projector(feat)
