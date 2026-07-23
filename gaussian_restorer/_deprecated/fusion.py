"""
Gaussian Scene Restoration Framework — Feature Fusion Modules

Fuses video prior features with Gaussian features before refinement.
Multiple strategies available, configurable via FusionCfg.strategy.
"""

from typing import Literal, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import FusionCfg


class FusionModule(nn.Module):
    """
    Abstract base for all fusion strategies.

    Inputs:
        video_feats: [B*V, C_v, H, W] — video prior features
        gs_feats:    [B*V, C_g, H, W] — Gaussian features

    Output:
        fused_feats: [B*V, C_out, H, W] — fused features for refiner
    """

    def __init__(self, cfg: FusionCfg, video_dim: int, gs_dim: int):
        super().__init__()
        self.cfg = cfg
        self.video_dim = video_dim
        self.gs_dim = gs_dim

    def forward(self, video_feats: torch.Tensor, gs_feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ConcatConvFusion(FusionModule):
    """
    Baseline fusion: concatenate + conv1x1 projection.

    Simplest approach — no learned interaction between modalities.
    Always available as the default fallback.
    """

    def __init__(self, cfg: FusionCfg, video_dim: int, gs_dim: int):
        super().__init__(cfg, video_dim, gs_dim)
        in_dim = video_dim + gs_dim
        self.proj = nn.Sequential(
            nn.Conv2d(in_dim, cfg.hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.hidden_dim, cfg.hidden_dim, kernel_size=1, bias=True),
        )

    def forward(self, video_feats: torch.Tensor, gs_feats: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([video_feats, gs_feats], dim=1)
        return self.proj(concat)


class CrossAttentionFusion(FusionModule):
    """
    Windowed cross-attention fusion.

    Video features attend to Gaussian features within local windows.
    Q = video features (prior-driven queries)
    K, V = Gaussian features (context to attend to)

    Windowed to avoid O(H²W²) memory cost of full spatial attention.
    """

    def __init__(self, cfg: FusionCfg, video_dim: int, gs_dim: int):
        super().__init__(cfg, video_dim, gs_dim)
        self.W = cfg.window_size
        self.num_heads = cfg.num_heads
        head_dim = max(cfg.hidden_dim // cfg.num_heads, 1)
        self.hidden_dim = head_dim * cfg.num_heads

        self.q_proj = nn.Linear(video_dim, self.hidden_dim, bias=False)
        self.k_proj = nn.Linear(gs_dim, self.hidden_dim, bias=False)
        self.v_proj = nn.Linear(gs_dim, self.hidden_dim, bias=False)
        self.out_proj = nn.Linear(self.hidden_dim, cfg.hidden_dim, bias=True)

        # Relative position bias for windowed attention
        self.rel_pos_bias = nn.Parameter(
            torch.randn(1, cfg.num_heads, self.W * self.W, self.W * self.W) * 0.02
        )

    def _window_partition(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Partition feature map into windows."""
        B, C, H, W = x.shape
        W_h, W_w = self.W, self.W
        pad_h = (W_h - H % W_h) % W_h
        pad_w = (W_w - W % W_w) % W_w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        B, C, Hp, Wp = x.shape
        n_h, n_w = Hp // W_h, Wp // W_w
        x = x.view(B, C, n_h, W_h, n_w, W_w)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
        windows = x.view(-1, W_h * W_w, C)
        return windows, n_h, n_w

    def _window_reverse(self, windows: torch.Tensor, n_h: int, n_w: int,
                         H: int, W: int) -> torch.Tensor:
        """Reverse window partition to full feature map."""
        B_ = windows.shape[0] // (n_h * n_w)
        C = windows.shape[-1]
        x = windows.view(B_, n_h, n_w, self.W, self.W, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B_, C, n_h * self.W, n_w * self.W)
        return x[:, :, :H, :W]

    def forward(self, video_feats: torch.Tensor, gs_feats: torch.Tensor) -> torch.Tensor:
        B, _, H, W = video_feats.shape
        device = video_feats.device

        # Project to hidden_dim
        v = video_feats.permute(0, 2, 3, 1)  # [B, H, W, C_v]
        g = gs_feats.permute(0, 2, 3, 1)     # [B, H, W, C_g]

        Q = self.q_proj(v)  # [B, H, W, D]
        K = self.k_proj(g)
        V = self.v_proj(g)

        # Window partition
        Q_win, n_h, n_w = self._window_partition(Q.permute(0, 3, 1, 2))
        K_win, _, _ = self._window_partition(K.permute(0, 3, 1, 2))
        V_win, _, _ = self._window_partition(V.permute(0, 3, 1, 2))

        # Multi-head attention
        n_windows = Q_win.shape[0]
        D_head = self.hidden_dim // self.num_heads
        Q_win = Q_win.view(n_windows, -1, self.num_heads, D_head).transpose(1, 2)
        K_win = K_win.view(n_windows, -1, self.num_heads, D_head).transpose(1, 2)
        V_win = V_win.view(n_windows, -1, self.num_heads, D_head).transpose(1, 2)

        attn = (Q_win @ K_win.transpose(-2, -1)) * (D_head ** -0.5)
        attn = attn + self.rel_pos_bias
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V_win).transpose(1, 2).contiguous().view(n_windows, -1, self.hidden_dim)
        out = self.out_proj(out)

        # Window reverse
        out_map = self._window_reverse(out, n_h, n_w, H, W)
        return out_map.permute(0, 3, 1, 2)  # [B, C_out, H, W]


class GatedFusion(FusionModule):
    """
    Gated fusion: learnable per-channel, per-pixel soft selection.

    gate = sigmoid(Linear(concat(video, gs)))
    output = gate * video_proj + (1 - gate) * gs_proj
    """

    def __init__(self, cfg: FusionCfg, video_dim: int, gs_dim: int):
        super().__init__(cfg, video_dim, gs_dim)
        hidden = cfg.hidden_dim

        self.video_proj = nn.Conv2d(video_dim, hidden, kernel_size=1, bias=False)
        self.gs_proj = nn.Conv2d(gs_dim, hidden, kernel_size=1, bias=False)
        self.gate_net = nn.Sequential(
            nn.Conv2d(video_dim + gs_dim, hidden, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=True),
        )

        # Initialize gate bias to favor balanced fusion
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, video_feats: torch.Tensor, gs_feats: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_net(torch.cat([video_feats, gs_feats], dim=1)))
        v_proj = self.video_proj(video_feats)
        g_proj = self.gs_proj(gs_feats)
        return gate * v_proj + (1 - gate) * g_proj


class TransformerDecoderFusion(FusionModule):
    """
    Transformer decoder fusion with learnable queries.

    Learnable queries attend to video and Gaussian features via cross-attention.
    Queries can specialize for different Gaussian parameter types.
    """

    def __init__(self, cfg: FusionCfg, video_dim: int, gs_dim: int):
        super().__init__(cfg, video_dim, gs_dim)

        from torch.nn import TransformerDecoder, TransformerDecoderLayer

        d_model = cfg.hidden_dim
        nhead = max(cfg.num_heads, 2)  # need at least 2 heads
        dim_feedforward = d_model * 4
        num_layers = cfg.num_decoder_layers

        # Learnable queries — one per Gaussian parameter type
        self.num_queries = cfg.num_queries
        self.queries = nn.Parameter(torch.randn(cfg.num_queries, d_model) * 0.02)

        # Cross-attention projections for video and GS features
        self.video_proj = nn.Conv2d(video_dim, d_model, kernel_size=1)
        self.gs_proj = nn.Conv2d(gs_dim, d_model, kernel_size=1)

        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                 batch_first=True, dropout=0.0)
        self.decoder = TransformerDecoder(decoder_layer, num_layers)

        # Output projection: queries back to spatial map
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # Spatial decoder: learnable position-based upsampling
        self.spatial_decoder = nn.ConvTranspose2d(
            (H // 8) * (W // 8), cfg.hidden_dim,
            kernel_size=3, stride=1, padding=1
        )
        # We don't know H, W at init — will build at first forward

        self._spatial_built = False

    def _build_spatial(self, H: int, W: int, device: torch.device):
        """Build spatial decoder for given resolution (lazy init)."""
        if self._spatial_built:
            return
        # We'll use a simpler approach: reshape queries into a coarse grid
        # and interpolate to full resolution
        self._spatial_built = True

    def forward(self, video_feats: torch.Tensor, gs_feats: torch.Tensor) -> torch.Tensor:
        B, _, H, W = video_feats.shape

        # Prepare memory (video + GS features flatten to tokens)
        # Video tokens: [B, H*W, d_model]
        v_tokens = self.video_proj(video_feats).flatten(2).permute(0, 2, 1)
        g_tokens = self.gs_proj(gs_feats).flatten(2).permute(0, 2, 1)

        # Concatenate as memory sequence
        memory = torch.cat([v_tokens, g_tokens], dim=1)  # [B, 2*H*W, d_model]

        # Expand queries to batch
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)  # [B, N_q, d_model]

        # Decode
        decoded = self.decoder(queries, memory)  # [B, N_q, d_model]

        # Project queries back to spatial map
        # Simple approach: average pool queries to a grid and interpolate
        grid_size = int(self.num_queries ** 0.5)
        if grid_size * grid_size == self.num_queries:
            # Reshape to grid
            feat_map = decoded.permute(0, 2, 1).view(B, -1, grid_size, grid_size)
        else:
            # Linear to fixed grid size
            feat_map = decoded.permute(0, 2, 1)  # [B, d_model, N_q]
            G = 7  # fixed coarse grid
            feat_map = F.adaptive_avg_pool1d(feat_map, G * G)
            feat_map = feat_map.view(B, -1, G, G)

        # Upsample to original resolution
        return F.interpolate(feat_map, size=(H, W), mode="bilinear", align_corners=False)


def build_fusion(cfg: FusionCfg, video_dim: int, gs_dim: int) -> FusionModule:
    """Factory: build fusion module from config."""
    if cfg.strategy == "concat_conv":
        return ConcatConvFusion(cfg, video_dim, gs_dim)
    elif cfg.strategy == "cross_attn":
        return CrossAttentionFusion(cfg, video_dim, gs_dim)
    elif cfg.strategy == "gated_fusion":
        return GatedFusion(cfg, video_dim, gs_dim)
    elif cfg.strategy == "transformer_decoder":
        return TransformerDecoderFusion(cfg, video_dim, gs_dim)
    else:
        raise ValueError(f"Unknown fusion strategy: {cfg.strategy}")
