"""
Gaussian Feature Encoder
========================

Encodes raw Gaussian parameters into a learned feature space and
associates them with video-derived features.

Two association modes:
  1. camera_projection (default): Project video features to each Gaussian
     using AnySplat's internally predicted cameras. Geometrically grounded.
  2. cross_attention: Learn 2D→3D correspondence via attention.
     Pose-free but needs more data.

Architecture:
  Raw Gaussian [B, N, 83] ──→ MLP ──→ Gaussian features [B, N, D]
                                          ↑
  Video features [B*V, D_v, H, W] ──→ project to each Gaussian's 3D pos
                                          ↑
                                 AnySplat predicted cameras (mode 1)
                                 or attention (mode 2)
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianFeatureEncoder(nn.Module):
    """
    Encodes Gaussian parameters and associates video features.

    Inputs:
        gaussian_params: [B, N, 83] — raw parameters (opacity+scale+rot+SH)
        gaussian_means:  [B, N, 3]  — 3D positions
        video_features:  [B*V, D_v, H, W] — from VideoExtractor
        extrinsic:       [B, V, 4, 4] (optional for camera_projection mode)
        intrinsic:       [B, V, 3, 3] (optional for camera_projection mode)

    Outputs:
        per_gaussian_feats: [B, N, D_out] — combined features for repairer
        global_scene_feat:  [B, D_out]     — global scene descriptor
    """

    def __init__(
        self,
        gaussian_dim: int = 83,
        video_feat_dim: int = 48,
        hidden_dim: int = 128,
        association: str = "camera_projection",
    ):
        super().__init__()
        self.association = association
        self.video_feat_dim = video_feat_dim

        # Encode raw Gaussian params into learned features
        self.gaussian_mlp = nn.Sequential(
            nn.Linear(gaussian_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Project video features to hidden dim
        self.video_proj = nn.Conv2d(video_feat_dim, hidden_dim, kernel_size=1)

        # Combine gaussian + video features
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        if association == "cross_attention":
            # Cross-attention: Gaussian queries attend to video features
            # Each Gaussian's position embedding is the query
            self.pos_embed = nn.Linear(3, hidden_dim)  # position → query
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, num_heads=4, batch_first=True
            )
            self.video_tokenizer = nn.Conv2d(video_feat_dim, hidden_dim, kernel_size=1)

    def forward(
        self,
        gaussian_params: torch.Tensor,   # [B, N, 83]
        gaussian_means: torch.Tensor,    # [B, N, 3]
        video_features: torch.Tensor,    # [B*V, D_v, H, W]
        extrinsic: Optional[torch.Tensor] = None,  # [B, V, 4, 4]
        intrinsic: Optional[torch.Tensor] = None,  # [B, V, 3, 3]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            per_gaussian_feats: [B, N, hidden_dim]
            global_scene_feat:  [B, hidden_dim]
        """
        B, N = gaussian_params.shape[:2]
        B2, V = video_features.shape[:2]
        H, W = video_features.shape[-2:]
        device = gaussian_params.device

        # ---- 1. Encode Gaussian params ----
        g_feats = self.gaussian_mlp(gaussian_params)  # [B, N, hidden]

        # ---- 2. Associate video features with each Gaussian ----
        if self.association == "camera_projection":
            assert extrinsic is not None and intrinsic is not None, \
                "camera_projection mode requires extrinsic and intrinsic"
            video_feats_per_gaussian = self._project_video_to_gaussians(
                video_features, gaussian_means, extrinsic, intrinsic, B, N, V, H, W
            )  # [B, N, hidden]
        elif self.association == "cross_attention":
            video_feats_per_gaussian = self._attend_video_to_gaussians(
                video_features, gaussian_means, B, N, V, H, W
            )  # [B, N, hidden]
        else:
            raise ValueError(f"Unknown association: {self.association}")

        # ---- 3. Fuse gaussian features + video features ----
        combined = torch.cat([g_feats, video_feats_per_gaussian], dim=-1)  # [B, N, 2*hidden]
        per_gaussian_feats = self.fusion(combined)  # [B, N, hidden]

        # ---- 4. Global scene descriptor ----
        global_scene_feat = per_gaussian_feats.mean(dim=1)  # [B, hidden]

        return per_gaussian_feats, global_scene_feat

    def _project_video_to_gaussians(
        self, video_features, gaussian_means, extrinsic, intrinsic,
        B, N, V, H, W
    ) -> torch.Tensor:
        """
        Project video features from each view to each Gaussian's 3D position,
        then aggregate across views.

        Uses AnySplat's predicted cameras (available from AnySplat forward pass).
        """
        pts = gaussian_means  # [B, N, 3]

        # Project video to hidden dim
        vid_proj = self.video_proj(video_features)  # [B*V, hidden, H, W]
        vid_proj = vid_proj.view(B, V, -1, H, W)   # [B, V, hidden, H, W]

        view_feats = []
        for vi in range(V):
            ext = extrinsic[:, vi]   # [B, 4, 4]
            intr = intrinsic[:, vi]  # [B, 3, 3]

            # World → camera
            R = ext[:, :3, :3]       # [B, 3, 3]
            t = ext[:, :3, 3:4]      # [B, 3, 1]
            cam_pts = (R.transpose(1, 2) @ (pts - t.transpose(1, 2)))  # [B, N, 3]

            # Camera → pixel
            uv = intr @ cam_pts.transpose(1, 2)  # [B, 3, N]
            u = uv[:, 0] / (uv[:, 2] + 1e-8)    # [B, N]
            v_ = uv[:, 1] / (uv[:, 2] + 1e-8)

            # Grid sample
            grid = torch.stack([
                (u / (W - 1)) * 2 - 1,
                (v_ / (H - 1)) * 2 - 1,
            ], dim=-1).unsqueeze(1)  # [B, 1, N, 2]

            sampled = F.grid_sample(
                vid_proj[:, vi], grid, mode='bilinear', align_corners=False
            )  # [B, hidden, 1, N]
            view_feats.append(sampled.squeeze(2))  # [B, hidden, N]

        # Aggregate across views (mean)
        video_feats = torch.stack(view_feats).mean(dim=0)  # [B, hidden, N]
        return video_feats.transpose(1, 2)  # [B, N, hidden]

    def _attend_video_to_gaussians(
        self, video_features, gaussian_means, B, N, V, H, W
    ) -> torch.Tensor:
        """
        Cross-attention: each Gaussian (position-encoded) attends to
        video feature tokens. No camera needed.

        Gaussian positions → position embedding → queries
        Video features (flattened) → keys/values
        """
        # Tokenize video features
        vid_tokens = self.video_tokenizer(video_features)  # [B*V, hidden, H, W]
        vid_tokens = vid_tokens.flatten(2).permute(0, 2, 1)  # [B*V, H*W, hidden]

        # Aggregate views (mean)
        vid_tokens = vid_tokens.view(B, V, -1, self.video_feat_dim).mean(dim=1)  # [B, H*W, hidden]

        # Position embedding for each Gaussian → queries
        gaussian_queries = self.pos_embed(gaussian_means)  # [B, N, hidden]

        # Cross-attention
        attn_out, _ = self.cross_attn(
            query=gaussian_queries,
            key=vid_tokens,
            value=vid_tokens,
        )  # [B, N, hidden]

        return attn_out
