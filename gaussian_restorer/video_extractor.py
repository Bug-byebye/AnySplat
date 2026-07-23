"""
Video Feature Extractor — 利用 WAN 的生成先验
=============================================

核心设计：
  输入图像 → VAE Encode → latent_A → DiT 全40层 → latent_B → 两路输出
                                                         │
                                                         ├── 投影到 Repairer (latent级别，不经过像素)
                                                         └── VAE Decode → 视频帧 (仅用于观察)

你的洞察 — 完全正确：
  WAN 的 latent_B 已经包含了生成信息（补全、去遮挡、细节增强），
  不需要 decode → re-encode 的冗余过程。

  Adapter 的作用：让 DiT 的 latent space 更适配高斯修复任务，
  不是用来注入高斯信息的。WAN 生成时不受任何几何约束。
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiTAdapter(nn.Module):
    """
    Lightweight adapter 插入在每个 DiT block 之后。

    设计（LoRA风格）：
      hidden → LayerNorm → Linear(hidden, r) → ReLU → Linear(r, hidden) → *scale → +residual

    scale 初始化为 0 → 初始状态 adapter 无影响，原始 WAN 行为完全保留。
    随着训练，scale 逐渐增大，adapter 缓慢修正 WAN 的输出使其更适合修复任务。

    Args:
        hidden_dim: DiT 的隐藏维度 (Wan2.2-5B: 5120)
        bottleneck: 瓶颈维度 (默认 64，参数量 = 2 × 5120 × 64 ≈ 0.65M/层)
    """

    def __init__(self, hidden_dim: int, bottleneck: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.down = nn.Linear(hidden_dim, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, hidden_dim, bias=False)
        # scale=0 → 初始无影响，训练时逐渐增大
        self.scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        residual = hidden
        h = self.norm(hidden)
        h = self.down(h)
        h = F.relu(h)
        h = self.up(h)
        return residual + h * self.scale


class VideoExtractor(nn.Module):
    """
    WAN 视频特征提取器（含生成能力）。

    核心逻辑：
      [B,V,3,H,W] → VAE Encode → latent_A → DiT ×40层 → latent_B
                                                              │
                                          ┌───────────────────┤
                                          ▼                   ▼
                                      Projector           VAE Decoder
                                          │                   │
                                      [B*V,D,H,W]       [B,V,3,H,W] (观察用)
                                          │
                                      → Repairer

    Args:
        feat_dim: 输出特征维度（送给 Repairer 的通道数）。
        enable_decoder: 是否加载 VAE decoder（用于观察 WAN 生成的视频）。
        use_adapters: 是否在 DiT 中插入可训练的 Adapter。
        adapter_bottleneck: Adapter 的瓶颈维度。
        adapter_hidden_dim: DiT 隐藏层维度（自动获取，不需传入）。
    """

    def __init__(
        self,
        feat_dim: int = 48,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        use_fp16: bool = True,
        enable_decoder: bool = False,
        use_adapters: bool = False,
        adapter_bottleneck: int = 64,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.model_id = model_id
        self.use_fp16 = use_fp16
        self.enable_decoder = enable_decoder
        self.use_adapters = use_adapters
        self.adapter_bottleneck = adapter_bottleneck
        self.dtype = torch.bfloat16 if use_fp16 else torch.float32

        # Lazy-loaded
        self.vae = None
        self.transformer = None
        self.adapters = None  # nn.ModuleList of DiTAdapter
        self._projector = None
        self._z_dim = None
        self._vae_scale_factors = None
        self._patch_size = None
        self._text_dim = None
        self._inner_dim = None

    def _load_model(self, device: torch.device):
        """Load WAN (VAE + DiT) on first forward."""
        from diffusers import AutoencoderKLWan, WanTransformer3DModel

        # ---- VAE ----
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id, subfolder="vae", torch_dtype=torch.float32,
        )
        self.vae.encoder.requires_grad_(False)
        if self.vae.decoder is not None:
            self.vae.decoder.requires_grad_(False)
        self.vae.eval().to(device)

        vae_cfg = self.vae.config
        self._z_dim = getattr(vae_cfg, "z_dim", 48)
        self._vae_scale_factors = {
            "temporal": getattr(vae_cfg, "scale_factor_temporal", 4),
            "spatial": getattr(vae_cfg, "scale_factor_spatial", 8),
        }
        print(f"[VideoExtractor] VAE loaded: z_dim={self._z_dim}, "
              f"stride_t={self._vae_scale_factors['temporal']}, "
              f"stride_s={self._vae_scale_factors['spatial']}")

        if not self.enable_decoder and self.vae.decoder is not None:
            del self.vae.decoder
            self.vae.decoder = None

        # ---- DiT (全40层) ----
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_id, subfolder="transformer", torch_dtype=self.dtype,
        )
        self.transformer.requires_grad_(False)
        self.transformer.eval().to(device)

        t_cfg = self.transformer.config
        self._inner_dim = t_cfg.num_attention_heads * t_cfg.attention_head_dim
        self._patch_size = t_cfg.patch_size  # (t, h, w)
        self._text_dim = t_cfg.text_dim
        n_layers = t_cfg.num_layers
        print(f"[VideoExtractor] DiT loaded: inner_dim={self._inner_dim}, "
              f"layers={n_layers}, patch_size={self._patch_size}")

        # ---- Adapter (可选，插入在每个 DiT block 之后) ----
        if self.use_adapters:
            self.adapters = nn.ModuleList([
                DiTAdapter(self._inner_dim, bottleneck=self.adapter_bottleneck)
                for _ in range(n_layers)
            ]).to(device)
            print(f"[VideoExtractor] Adapters: {n_layers}× bottleneck={self.adapter_bottleneck} "
                  f"(trainable: {sum(p.numel() for p in self.adapters.parameters())/1e6:.2f}M)")

        # ---- Projector: latent_B [B, z_dim, T', H', W'] → [B*V, feat_dim, H, W] ----
        self._projector = nn.Sequential(
            nn.Conv2d(self._z_dim, self.feat_dim, kernel_size=1),
        )
        nn.init.kaiming_normal_(self._projector[0].weight, mode="fan_out", nonlinearity="linear")
        nn.init.zeros_(self._projector[0].bias)
        self._projector.to(device)

    def _prepare_video(self, images: torch.Tensor) -> Tuple[torch.Tensor, int, int, int]:
        """[B,V,C,H,W] → [B,C,T_pad,H_ps,W_ps] with temporal+spatial padding."""
        B, V, C, H, W = images.shape
        if self.vae is None:
            self._load_model(images.device)
        s_t = self._vae_scale_factors["temporal"]
        s_s = self._vae_scale_factors["spatial"]

        T_pad = ((V + s_t - 1) // s_t) * s_t
        video = images.permute(0, 2, 1, 3, 4)

        if T_pad > V:
            last = video[:, :, -1:, :, :]
            video = torch.cat([video, last.repeat(1, 1, T_pad - V, 1, 1)], dim=2)

        pad_h = (s_s - H % s_s) % s_s
        pad_w = (s_s - W % s_s) % s_s
        if pad_h > 0 or pad_w > 0:
            video = F.pad(video, (0, pad_w, 0, pad_h), mode="reflect")

        return video, T_pad, H, W

    # =================================================================
    # 主入口：VAE → DiT全40层 → latent_B → 投影 → 送给Repairer
    # =================================================================

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        主前向：输出 latent_B 投影后的特征，送给 Repairer。

        Args:
            images: [B, V, 3, H, W] normalized to [0,1].

        Returns:
            features: [B*V, feat_dim, H, W] 从 latent_B 投影得到的特征。
        """
        if self.vae is None:
            self._load_model(images.device)

        B, V = images.shape[:2]
        device = images.device

        # Step 1: VAE Encode → latent_A
        video_pad, T_pad, H, W_orig = self._prepare_video(images)
        h = self.vae._encode(video_pad.to(torch.float32))
        latent_a = h[:, :self._z_dim]  # [B, z_dim, T_pad//4, H_ps//16, W_ps//16]

        # Step 2: DiT 全40层 → latent_B (包含生成先验)
        latent_b = self._run_full_dit(latent_a)  # [B, z_dim, T', H', W']

        # Step 3: 从 latent_B 投影特征 → Repairer
        # 聚合 temporal 维度（取平均）
        lat = latent_b.mean(dim=2)  # [B, z_dim, H', W']
        features = F.interpolate(lat, size=(H, W_orig), mode="bilinear", align_corners=False)
        features = self._projector(features)  # [B, feat_dim, H, W]

        # Step 4: 扩展为逐视角特征
        features = features.unsqueeze(1).expand(-1, V, -1, -1, -1)
        features = features.flatten(0, 1)  # [B*V, feat_dim, H, W]

        return features

    # =================================================================
    # 视频生成出口（纯观察用 — 看 WAN 实际"想象"了什么）
    # =================================================================

    @torch.no_grad()
    def generate_video(self, images: torch.Tensor) -> torch.Tensor:
        """
        生成视频帧。输出 WAN 对场景的"想象"。

        流程：VAE Encode → DiT全40层 → VAE Decode → 视频帧

        这些帧展示了 WAN 认为场景应该长什么样（补全遮挡、增强细节）。
        注意：你不需要把这些帧编码回去。latent_B 已经包含了这些信息。

        Args:
            images: [B, V, 3, H, W] input.

        Returns:
            frames: [B, V, 3, H, W] WAN 生成的视频帧（仅在 enable_decoder=True 时有效）。
        """
        if self.vae is None:
            self._load_model(images.device)
        if not self.enable_decoder:
            print("[WARNING] VAE decoder not loaded. "
                  "Re-init with enable_decoder=True")
            return images

        B, V = images.shape[:2]
        video_pad, T_pad, H, W_orig = self._prepare_video(images)

        # Encode → DiT → Decode
        h = self.vae._encode(video_pad.to(torch.float32))
        latent_a = h[:, :self._z_dim]
        latent_b = self._run_full_dit(latent_a)

        # Decode
        decoder_in = torch.cat([latent_b, torch.zeros_like(latent_b)], dim=1)
        decoded = self.vae.decoder(decoder_in)
        decoded = decoded[:, :, :, :H, :W_orig]       # crop spatial
        decoded = decoded[:, :, :V, :, :]              # crop temporal
        frames = decoded.permute(0, 2, 1, 3, 4).clamp(0, 1)  # [B, V, 3, H, W]

        return frames

    # =================================================================
    # 视频插帧 / 外扩（利用 WAN 的生成先验）
    # =================================================================

    @torch.no_grad()
    def interpolate_video(
        self,
        images: torch.Tensor,
        target_num_frames: Optional[int] = None,
        extrapolate_front: int = 0,
        extrapolate_back: int = 0,
        interpolation_strength: float = 1.0,
        use_dit: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对低帧视频进行插帧 (interpolation) 和适当外扩 (extrapolation)。

        核心思路（2025-07 优化版）：
          1. 在像素域做时间维线性插值（比 latent 域插值更保真）
          2. 整体 VAE Encode → latent_A
          3. (可选) DiT 前向 → latent_B (利用生成先验增强)
          4. VAE Decode → 裁剪到目标帧数

        参数:
            images: [B, V, 3, H, W] 输入的低帧视频（归一化到 [0,1]）。
            target_num_frames: 插帧后的目标帧数。若为 None，则保持 V 不变。
            extrapolate_front: 在序列前额外生成的帧数（外扩）。
            extrapolate_back: 在序列后额外生成的帧数（外扩）。
            interpolation_strength: DiT 残差强度 [0,1]。
            use_dit: 是否使用 DiT 生成先验。

        返回:
            video_frames: [B, T_total, 3, H, W] 生成的视频。
            latent_b: [B, z_dim, T_lat_out, H_lat, W_lat] 解码前的 latent。
        """
        if self.vae is None:
            self._load_model(images.device)

        B, V, C, H_orig, W_orig = images.shape
        device = images.device
        s_t = self._vae_scale_factors["temporal"]   # 4
        s_s = self._vae_scale_factors["spatial"]     # 16

        target_frames = target_num_frames if target_num_frames is not None else V
        total_target = target_frames + extrapolate_front + extrapolate_back

        # ---- Step 1: 像素域时间线性插值 ----
        # [B,3,V,H,W] → [B,3,total_target,H,W]
        video_perm = images.permute(0, 2, 1, 3, 4)
        video_interp = F.interpolate(
            video_perm,
            size=(total_target, H_orig, W_orig),
            mode="trilinear",
            align_corners=False,
        )  # [B, 3, total_target, H, W]

        # ---- Step 2: 确保 latent token 数够解码 ----
        # VAE decoder: T_dec = 4*T_lat - 3, 需要 T_dec >= total_target
        # 所以需 T_lat >= ceil((total_target + 3) / 4)
        T_lat_needed = max((total_target + 3 + s_t - 1) // s_t, 1)
        T_pixel_needed = T_lat_needed * s_t
        T_pixel = video_interp.shape[2]

        if T_pixel < T_pixel_needed:
            last_frame = video_interp[:, :, -1:, :, :]
            video_interp = torch.cat(
                [video_interp, last_frame.repeat(1, 1, T_pixel_needed - T_pixel, 1, 1)],
                dim=2,
            )

        # ---- Step 3: VAE Encode ----
        pad_h = (s_s - H_orig % s_s) % s_s
        pad_w = (s_s - W_orig % s_s) % s_s
        if pad_h > 0 or pad_w > 0:
            video_interp = F.pad(video_interp, (0, pad_w, 0, pad_h), mode="reflect")

        h = self.vae._encode(video_interp.to(torch.float32))
        latent_a = h[:, :self._z_dim]
        _, _, T_lat, H_lat, W_lat = latent_a.shape

        if T_lat > T_lat_needed:
            latent_a = latent_a[:, :, :T_lat_needed]
        elif T_lat < T_lat_needed:
            latent_a = F.interpolate(
                latent_a, size=(T_lat_needed, H_lat, W_lat),
                mode="trilinear", align_corners=False,
            )

        # ---- Step 4: (可选) DiT 前向 ----
        if use_dit:
            latent_b = self._run_full_dit_interp(
                latent_a,
                latent_original=latent_a,
                T_lat_orig=T_lat_needed,
                strength=interpolation_strength,
            )
        else:
            latent_b = latent_a

        # ---- Step 5: VAE Decode ----
        decoded = self.vae.decode(latent_b.float())
        if hasattr(decoded, "sample"):
            decoded = decoded.sample
        elif isinstance(decoded, (tuple, list)):
            decoded = decoded[0]

        T_dec = decoded.shape[2]
        # 裁剪空间 (去掉 padding)
        decoded = decoded[:, :, :, :H_orig, :W_orig]
        # 裁剪时间维
        T_use = min(total_target, T_dec)
        decoded = decoded[:, :, :T_use, :, :]

        video_frames = decoded.permute(0, 2, 1, 3, 4).clamp(0, 1)
        return video_frames, latent_b

    def _run_full_dit_interp(
        self,
        latent: torch.Tensor,
        latent_original: torch.Tensor,
        T_lat_orig: int,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """
        用于插帧场景的 DiT 前向。

        用统一的 strength 控制 DiT 残差强度，适用于所有时间位置。
        对「原始帧对应区域」用较小权重 (0.1)，对「外扩/插值区域」
        用 strength 控制的大权重，使生成先验能更充分地填充细节。

        参数:
            latent: [B, 48, T_lat_target, H_lat, W_lat] 扩展后的 latent。
            latent_original: [B, 48, T_lat_orig, H_lat, W_lat] 原始 VAE latent。
            T_lat_orig: 原始 latent 时间维长度。
            strength: 插值/外扩位置的 DiT 残差强度 [0,1]。
                      0=纯线性插值，1=完全使用 DiT 生成。
        """
        B = latent.shape[0]
        device = latent.device
        dtype = self.dtype

        latent_in = latent.to(dtype)
        rope_emb = self.transformer.rope(latent_in)

        # Patch embedding
        hidden = self.transformer.patch_embedding(latent_in)
        hidden = hidden.flatten(2).transpose(1, 2)

        # Dummy conditioning
        timestep = torch.zeros((1,), device=device, dtype=torch.long)
        dummy_text = torch.zeros((1, 512, self._text_dim), device=device, dtype=dtype)
        temb, ts_proj, enc_hidden, _ = self.transformer.condition_embedder(
            timestep, dummy_text, None
        )
        ts_proj = ts_proj.unflatten(1, (6, -1))

        # 所有 DiT 层（30层）
        n_layers = self.transformer.config.num_layers
        for i in range(n_layers):
            hidden = self.transformer.blocks[i](hidden, enc_hidden, ts_proj, rope_emb)
            if self.use_adapters and self.adapters is not None:
                hidden = self.adapters[i](hidden)

        # 映射回 latent 空间
        hidden = self.transformer.norm_out(hidden)
        hidden = self.transformer.proj_out(hidden)

        # unpatchify
        B_tok, N_tok, D_tok = hidden.shape
        T_tok = latent.shape[2]
        Hp_tok = max(latent.shape[3] // 2, 1)
        Wp_tok = max(latent.shape[4] // 2, 1)
        n_expected = T_tok * Hp_tok * Wp_tok
        n_vis = min(n_expected, N_tok)
        h_vis = hidden[:, :n_vis]
        h_vis = h_vis.reshape(B_tok, T_tok, Hp_tok, Wp_tok, 48, 2, 2)
        h_vis = h_vis.permute(0, 4, 1, 2, 5, 3, 6)
        dit_out = h_vis.reshape(B_tok, self._z_dim, T_tok, Hp_tok * 2, Wp_tok * 2)

        # 调整 dit_out 的空间尺寸与 latent 匹配
        if dit_out.shape[-2] != latent.shape[-2] or dit_out.shape[-1] != latent.shape[-1]:
            # 5D 调整: 将 B*48 合并为 batch 维
            B_adj, C_adj, T_adj, H_adj, W_adj = dit_out.shape
            dit_out_flat = dit_out.transpose(1, 2).reshape(B_adj * T_adj, C_adj, H_adj, W_adj)
            dit_out_flat = F.interpolate(
                dit_out_flat, size=(latent.shape[-2], latent.shape[-1]),
                mode="bilinear", align_corners=False,
            )
            dit_out = dit_out_flat.reshape(B_adj, T_adj, C_adj, *dit_out_flat.shape[-2:])
            dit_out = dit_out.transpose(1, 2).contiguous()

        # === 残差应用 ===
        # 对插帧/外扩场景，所有时间位置都需要 DiT 的生成先验来修正插值伪影。
        # 使用强度为 strength * 0.3 的统一权重（介于保守 0.1 和激进 1.0 之间）。
        delta = dit_out - latent
        T_tok_now = latent.shape[2]
        weight = max(min(strength * 0.3, 0.8), 0.1)

        for t in range(T_tok_now):
            latent[:, :, t: t + 1] = latent[:, :, t: t + 1] + delta[:, :, t: t + 1] * weight

        return latent

    @torch.no_grad()
    def interpolate_video_flow_matching(
        self,
        images: torch.Tensor,
        target_num_frames: Optional[int] = None,
        extrapolate_front: int = 0,
        extrapolate_back: int = 0,
        num_denoising_steps: int = 15,
        noise_level: float = 0.6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于 Flow Matching 的 WAN 视频插帧 + 外扩（高质量）。

        流程：
          1. VAE Encode → latent_A
          2. 时间维上采样到目标帧数 → latent_expanded
          3. Flow Matching 加噪: z_t = (1-t)*z_0 + t*noise
          4. 多步去噪: DiT 各层前向 + scheduler.step
          5. VAE Decode → 输出视频帧

        相比 interpolate_video():
          - 使用完整去噪流程（而非单次 DiT 前向）
          - 更充分利用 WAN 的生成先验
          - 质量更好但更慢（~15× DiT 前向）

        参数:
            images: [B, V, 3, H, W] 输入的低帧视频（归一化到 [0,1]）。
            target_num_frames: 插帧后的目标帧数。
            extrapolate_front: 在序列前额外生成的帧数（外扩）。
            extrapolate_back: 在序列后额外生成的帧数（外扩）。
            num_denoising_steps: 去噪步数（越多质量越好，越慢）。
            noise_level: 加噪程度 [0,1]。0=不 noise，1=完全 noise。
                        推荐 0.5~0.7：保留输入结构 + 给生成留空间。

        返回:
            video_frames: [B, T_total, 3, H, W] 生成的视频。
            latent_b: [B, z_dim, T_lat_out, H_lat, W_lat] 解码前的 latent。
        """
        if self.vae is None:
            self._load_model(images.device)

        B, V, C, H_orig, W_orig = images.shape
        device = images.device
        s_t = self._vae_scale_factors["temporal"]
        s_s = self._vae_scale_factors["spatial"]

        # ---- Step 1: VAE Encode → latent_A ----
        video_pad, T_pad, H_ps, W_ps = self._prepare_video(images)
        h = self.vae._encode(video_pad.to(torch.float32))
        latent_a = h[:, :self._z_dim]  # [B, 48, T_lat_in, H_lat, W_lat]
        B_lat, _, T_lat_in, H_lat, W_lat = latent_a.shape

        # ---- Step 2: 计算目标 latent 时间维度 ----
        target_frames = target_num_frames if target_num_frames is not None else V
        total_target = target_frames + extrapolate_front + extrapolate_back
        T_lat_target = max((total_target + 3 + s_t - 1) // s_t, 1)

        # ---- Step 3: 构建扩展后的 latent 序列 ----
        if T_lat_target > T_lat_in:
            latent_expanded = F.interpolate(
                latent_a,
                size=(T_lat_target, H_lat, W_lat),
                mode="trilinear",
                align_corners=False,
            )
        elif T_lat_target < T_lat_in:
            latent_expanded = latent_a[:, :, :T_lat_target]
        else:
            latent_expanded = latent_a

        # ---- Step 4: Flow Matching 加噪 + 多步去噪 ----
        # 创建 flow-matching scheduler
        from diffusers import UniPCMultistepScheduler
        scheduler = UniPCMultistepScheduler(
            num_train_timesteps=1000,
            prediction_type="flow_prediction",
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="linear",
        )

        # 加噪: z_t = (1 - t_norm) * z_0 + t_norm * noise
        clean_latent = latent_expanded.clone()
        noise = torch.randn_like(clean_latent)
        t_norm = torch.tensor([noise_level], device=device)
        z_t = (1 - t_norm.view(1, 1, 1, 1, 1)) * clean_latent + t_norm.view(1, 1, 1, 1, 1) * noise

        # 多步去噪
        scheduler.set_timesteps(num_denoising_steps)
        timesteps = scheduler.timesteps.to(device)

        # 找到起始 timestep（对应 noise_level）
        noise_ts = int(noise_level * 1000)
        start_idx = (timesteps.float() - noise_ts).abs().argmin().item()
        active_ts = timesteps[start_idx:]

        latent_z = z_t.clone()
        dtype = self.dtype

        for step_i, ts in enumerate(active_ts):
            ts_b = ts.unsqueeze(0)

            # DiT forward
            latent_in = latent_z.to(dtype)
            rope_emb = self.transformer.rope(latent_in)

            hidden = self.transformer.patch_embedding(latent_in)
            hidden = hidden.flatten(2).transpose(1, 2)

            # Conditioning
            text_dim = self._text_dim or getattr(self.transformer.config, "text_dim", 4096)
            dummy_text = torch.zeros((1, 512, text_dim), device=device, dtype=dtype)
            temb, ts_proj, enc_hidden, _ = self.transformer.condition_embedder(
                ts_b, dummy_text, None
            )
            ts_proj = ts_proj.unflatten(1, (6, -1))

            # All DiT layers
            n_layers = self.transformer.config.num_layers
            for i in range(n_layers):
                hidden = self.transformer.blocks[i](hidden, enc_hidden, ts_proj, rope_emb)
                if self.use_adapters and self.adapters is not None:
                    hidden = self.adapters[i](hidden)

            # Output → latent space
            hidden = self.transformer.norm_out(hidden)
            hidden = self.transformer.proj_out(hidden)

            # unpatchify
            B_tok, N_tok, _ = hidden.shape
            Hp_tok = max(latent_in.shape[3] // 2, 1)
            Wp_tok = max(latent_in.shape[4] // 2, 1)
            n_expected = latent_in.shape[2] * Hp_tok * Wp_tok
            n_vis = min(n_expected, N_tok)
            h_vis = hidden[:, :n_vis]
            h_vis = h_vis.reshape(B_tok, latent_in.shape[2], Hp_tok, Wp_tok, 48, 2, 2)
            h_vis = h_vis.permute(0, 4, 1, 2, 5, 3, 6)
            pred_flow = h_vis.reshape(B_tok, self._z_dim, latent_in.shape[2], Hp_tok * 2, Wp_tok * 2)

            # Scheduler step (flow_prediction mode)
            latent_z = scheduler.step(pred_flow.float(), ts, latent_z.float()).prev_sample.to(dtype)

        latent_b = latent_z.float()

        # ---- Step 5: VAE Decode → 视频帧 ----
        decoded = self.vae.decode(latent_b)
        if hasattr(decoded, "sample"):
            decoded = decoded.sample
        elif isinstance(decoded, (tuple, list)):
            decoded = decoded[0]

        # 裁剪
        T_dec = decoded.shape[2]
        decoded = decoded[:, :, :, :H_ps, :W_ps]
        if H_orig < H_ps or W_orig < W_ps:
            decoded = decoded[:, :, :, :H_orig, :W_orig]

        T_use = min(total_target, T_dec)
        decoded = decoded[:, :, :T_use, :, :]

        video_frames = decoded.permute(0, 2, 1, 3, 4).clamp(0, 1)
        return video_frames, latent_b

    # =================================================================
    # DiT 全层前向（核心：latent_A → latent_B）
    # =================================================================

    def _run_full_dit(self, latent: torch.Tensor) -> torch.Tensor:
        """
        运行 DiT 全部 40 层，将 latent_A 转变为 latent_B。

        latent_A: VAE encoder 输出（压缩的，只含输入信息）。
        latent_B: 经过 40 层 transformer 处理（包含生成先验）。

        可选：如果 use_adapters=True，每层后运行 adapter（scale=0 初始无影响）。
        """
        B = latent.shape[0]
        device = latent.device
        dtype = self.dtype

        latent_in = latent.to(dtype)
        rope_emb = self.transformer.rope(latent_in)

        # Patch embedding
        hidden = self.transformer.patch_embedding(latent_in)
        hidden = hidden.flatten(2).transpose(1, 2)

        # Dummy conditioning (无文本 prompt)
        timestep = torch.zeros((1,), device=device, dtype=torch.long)
        dummy_text = torch.zeros((1, 512, self._text_dim), device=device, dtype=dtype)
        temb, ts_proj, enc_hidden, _ = self.transformer.condition_embedder(
            timestep, dummy_text, None
        )
        ts_proj = ts_proj.unflatten(1, (6, -1))

        # 运行全部 DiT 层（可附加 adapter 微调）
        n_layers = self.transformer.config.num_layers
        for i in range(n_layers):
            hidden = self.transformer.blocks[i](hidden, enc_hidden, ts_proj, rope_emb)
            if self.use_adapters and self.adapters is not None:
                hidden = self.adapters[i](hidden)

        # === 将 DiT token 输出映射回 latent 空间（使用预训练权重） ===
        # 之前犯的错：用了一个随机初始化的 _latent_proj，丢弃了预训练输出！
        # 修正：用 Wan2.2 自带的 norm_out + proj_out + unpatchify
        hidden = self.transformer.norm_out(hidden)     # [B, N, inner_dim=3072]
        hidden = self.transformer.proj_out(hidden)     # [B, N, 192]

        # unpatchify: [B, N, 192] → [B, 48, T, H, W]
        # 192 = 48 × 2 × 2（patch_embedding 的 kernel 大小）
        B_tok, N_tok, D_tok = hidden.shape
        T_tok = latent.shape[2]                     # T（时间维不变）
        Hp_tok = max(latent.shape[3] // 2, 1)       # H/2（spatial down 2×）
        Wp_tok = max(latent.shape[4] // 2, 1)       # W/2
        n_expected = T_tok * Hp_tok * Wp_tok

        # 取对应的视觉 token
        n_vis = min(n_expected, N_tok)
        h_vis = hidden[:, :n_vis]                   # [B, T*H/2*W/2, 192]

        # Reshape: 192 = 48 × 2(spatial) × 2(spatial)
        h_vis = h_vis.reshape(B_tok, T_tok, Hp_tok, Wp_tok, 48, 2, 2)
        h_vis = h_vis.permute(0, 4, 1, 2, 5, 3, 6)  # [B, 48, T, H/2, 2, W/2, 2]
        latent_b = h_vis.reshape(B_tok, self._z_dim, T_tok, Hp_tok * 2, Wp_tok * 2)

        # 小残差连接（可选）
        latent_b = latent + (latent_b - latent) * 0.1

        return latent_b

    def get_feat_dim(self) -> int:
        return self.feat_dim

    def is_decoder_loaded(self) -> bool:
        return self.enable_decoder and self.vae is not None and self.vae.decoder is not None
