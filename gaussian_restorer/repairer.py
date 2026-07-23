"""
Unified Gaussian Repairer
=========================

Three capabilities in one architecture:
  1. MODIFY: Per-Gaussian Δ_params + Δ_means
  2. DELETE: Per-Gaussian keep_prob
  3. ADD:    Query-based Gaussian generator (fully differentiable)

Key improvement over the old design:
  - GaussianGenerator replaces SceneDeficiencyAnalyzer + NewGaussianLifting
  - Instead of non-differentiable top-K 2D→3D lifting, we use learnable queries
    that cross-attend to video features and predict K new Gaussians directly
  - All K candidates are rendered with activation scores → fully differentiable!
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PerGaussianAnalysisHead(nn.Module):
    """
    Per-Gaussian analysis with three heads.

    Input:  [B, N, D] — per-Gaussian encoded features
    Output:
        delta_params  [B, N, 82] — scale(3) + rot(4) + SH(75) residuals
        delta_means   [B, N, 3]  — scaled position offsets
        keep_logit    [B, N, 1]  — retention logit
    """

    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(),
        )

        # Δ_params: zero-init for safe first forward
        self.param_head = nn.Linear(hidden, 82)
        nn.init.zeros_(self.param_head.weight)
        nn.init.zeros_(self.param_head.bias)

        # Δ_means: via small MLP, zero-init
        self.move_head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(),
            nn.Linear(32, 3),
        )
        nn.init.zeros_(self.move_head[0].weight)
        nn.init.zeros_(self.move_head[0].bias)
        nn.init.zeros_(self.move_head[2].weight)
        nn.init.zeros_(self.move_head[2].bias)

        # keep_prob: bias=0.5 → initial σ(0.5)≈0.62 (slight delete bias)
        self.confidence_head = nn.Linear(hidden, 1)
        nn.init.zeros_(self.confidence_head.weight)
        nn.init.constant_(self.confidence_head.bias, 0.5)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(features)
        delta_params = self.param_head(h)
        delta_means = self.move_head(h) * 0.01   # scaled → initially tiny moves
        keep_logit = self.confidence_head(h)
        return delta_params, delta_means, keep_logit


class GaussianGenerator(nn.Module):
    """
    Generate K new Gaussian candidates via learnable queries.

    Architecture (Perceiver/DETR-inspired):
      K learnable queries → cross-attend to video feature tokens
      Each query → [activation, position(3), params(82)]

    Totally differentiable: all K candidates are rendered with
    their activation as an opacity multiplier.

    Query initialization: learns "anchor" positions from data.
    During training, the rendering loss teaches queries where
    to place new Gaussians (holes, missing regions, etc.).
    """

    def __init__(
        self,
        n_queries: int = 256,
        video_feat_dim: int = 48,
        hidden: int = 256,
    ):
        super().__init__()
        self.n_queries = n_queries

        # Learnable query embeddings
        self.queries = nn.Parameter(torch.randn(n_queries, hidden) * 0.02)

        # Video feature tokenizer
        self.video_tokenizer = nn.Sequential(
            nn.Conv2d(video_feat_dim, hidden, kernel_size=1),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )

        # Cross-attention: queries ←→ video tokens
        self.cross_attn = nn.MultiheadAttention(
            hidden, num_heads=4, batch_first=True,
        )
        self.self_attn = nn.MultiheadAttention(
            hidden, num_heads=4, batch_first=True,
        )

        # Feed-forward after attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.ReLU(),
            nn.Linear(hidden * 2, hidden),
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

        # Prediction heads
        self.activation_head = nn.Linear(hidden, 1)   # → logit
        self.position_head = nn.Linear(hidden, 3)     # → 3D position delta
        self.param_head = nn.Linear(hidden, 82)       # → scale(3)+rot(4)+SH(75)

        # Global context conditioning
        self.global_cond = nn.Linear(hidden, hidden)

        # Zero-init output heads for safe start
        for head in [self.activation_head, self.position_head, self.param_head]:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        video_features: torch.Tensor,   # [B*V, D_v, H, W]
        global_scene_feat: torch.Tensor,  # [B, hidden]
        n_existing: int,                 # number of existing Gaussians N
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with:
            'new_means':     [B, K, 3]
            'new_params':    [B, K, 82]
            'new_activations': [B, K, 1]
        """
        B = global_scene_feat.shape[0]
        B2, _, H, W = video_features.shape

        # ---- 1. Tokenize video features ----
        vid_tokens = self.video_tokenizer(video_features)  # [B*V, hidden, H, W]
        vid_tokens = vid_tokens.flatten(2).permute(0, 2, 1)  # [B*V, H*W, hidden]

        # Aggregate views (mean across views)
        if B2 // B > 1:
            V = B2 // B
            vid_tokens = vid_tokens.view(B, V, -1, vid_tokens.shape[-1]).mean(dim=1)
        else:
            vid_tokens = vid_tokens  # [B, H*W, hidden]

        # ---- 2. Add global condition to queries ----
        global_vec = self.global_cond(global_scene_feat).unsqueeze(1)  # [B, 1, hidden]
        queries = self.queries.unsqueeze(0).expand(B, -1, -1) + global_vec  # [B, K, hidden]

        # ---- 3. Self-attention among queries (learn diversity) ----
        queries = queries + self.self_attn(queries, queries, queries)[0]
        queries = self.norm1(queries)

        # ---- 4. Cross-attention: queries → video tokens ----
        queries = queries + self.cross_attn(
            queries, vid_tokens, vid_tokens
        )[0]
        queries = self.norm2(queries)

        # ---- 5. FFN ----
        queries = queries + self.ffn(queries)

        # ---- 6. Predict outputs ----
        new_activations = torch.sigmoid(self.activation_head(queries))  # [B, K, 1]
        new_means = self.position_head(queries)                          # [B, K, 3]
        new_params = self.param_head(queries)                            # [B, K, 82]

        return {
            'new_means': new_means,
            'new_params': new_params,
            'new_activations': new_activations,
        }


class UnifiedRepairer(nn.Module):
    """
    Complete Gaussian Repairer with modify + delete + add.

    Architecture:
      1. PerGaussianAnalysisHead → Δ_params, Δ_means, keep_prob
      2. GaussianGenerator → K new Gaussian candidates
      3. Combine refined + new Gaussians into final set

    Fully differentiable: all operations (including generator)
    are on the computation graph.
    """

    def __init__(
        self,
        gaussian_feat_dim: int = 128,
        n_new_queries: int = 256,
        per_gaussian_hidden: int = 256,
        generator_hidden: int = 256,
        video_feat_dim: int = 48,
    ):
        super().__init__()
        self.n_new_queries = n_new_queries

        # Module 1: Per-Gaussian analysis
        self.per_gaussian_head = PerGaussianAnalysisHead(
            in_dim=gaussian_feat_dim, hidden=per_gaussian_hidden,
        )

        # Module 2: Gaussian generator (addition)
        self.generator = GaussianGenerator(
            n_queries=n_new_queries,
            video_feat_dim=video_feat_dim,
            hidden=generator_hidden,
        )

        # Module 3: Global scene encoder for generator conditioning
        self.scene_encoder = nn.Sequential(
            nn.Linear(gaussian_feat_dim, gaussian_feat_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        per_gaussian_feats: torch.Tensor,   # [B, N, D] from GaussianEncoder
        global_scene_feat: torch.Tensor,     # [B, D] from GaussianEncoder
        gaussian_means: torch.Tensor,        # [B, N, 3] original positions
        gaussian_params: torch.Tensor,       # [B, N, 83] original params
        video_features: torch.Tensor,        # [B*V, D_v, H, W] from VideoExtractor
    ) -> Dict[str, torch.Tensor]:
        """
        Full repair forward pass.

        Returns dict with:
          'delta_params':    [B, N, 82] — parameter residuals
          'delta_means':     [B, N, 3]  — position offsets (scaled)
          'keep_logit':      [B, N, 1]  — retention logits
          'new_means':       [B, K, 3]  — new Gaussian positions
          'new_params':      [B, K, 82] — new Gaussian parameters
          'new_activations': [B, K, 1]  — new Gaussian activation scores
          'keep_prob':       [B, N, 1]  — sigmoid(keep_logit)
        """
        B, N = gaussian_params.shape[:2]
        device = gaussian_params.device

        # ===== 1. Per-Gaussian Analysis =====
        delta_params, delta_means, keep_logit = self.per_gaussian_head(
            per_gaussian_feats
        )
        keep_prob = torch.sigmoid(keep_logit)

        # ===== 2. Gaussian Generator =====
        # Condition on aggregated scene understanding
        scene_context = self.scene_encoder(global_scene_feat)

        new_gaussians = self.generator(
            video_features=video_features,
            global_scene_feat=scene_context,
            n_existing=N,
            device=device,
        )

        return {
            'delta_params': delta_params,
            'delta_means': delta_means,
            'keep_logit': keep_logit,
            'keep_prob': keep_prob,
            'new_means': new_gaussians['new_means'],
            'new_params': new_gaussians['new_params'],
            'new_activations': new_gaussians['new_activations'],
        }

    def print_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"UnifiedRepairer Summary:")
        print(f"  Total params:     {total/1e3:.1f}K")
        print(f"  Trainable params: {trainable/1e3:.1f}K")

        pg = sum(p.numel() for p in self.per_gaussian_head.parameters())
        gen = sum(p.numel() for p in self.generator.parameters())
        print(f"  PerGaussianHead:  {pg/1e3:.1f}K")
        print(f"  GaussianGenerator: {gen/1e3:.1f}K")


def build_rotation_matrix(rotations: torch.Tensor) -> torch.Tensor:
    """Convert quaternions [w,x,y,z] to 3x3 rotation matrices."""
    w, x, y, z = rotations.unbind(-1)
    R = torch.stack([
        1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y,
        2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x,
        2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y,
    ], dim=-1).view(*rotations.shape[:-1], 3, 3)
    return R


def build_refined_gaussians(
    gaussian_params: torch.Tensor,   # [B, N, 83]
    gaussian_means: torch.Tensor,    # [B, N, 3]
    delta_params: torch.Tensor,      # [B, N, 82]
    delta_means: torch.Tensor,       # [B, N, 3]
    keep_prob: torch.Tensor,         # [B, N, 1]
    new_params: torch.Tensor,        # [B, K, 82]
    new_means: torch.Tensor,         # [B, K, 3]
    new_activations: torch.Tensor,   # [B, K, 1]
    device: torch.device,
) -> 'Gaussians':  # noqa: F821
    """
    Build the final Gaussians object from refined + new primitives.

    This function handles:
    1. Apply Δ_params to existing Gaussians
    2. Apply Δ_means to existing Gaussians' positions
    3. Apply keep_prob as opacity multiplier (deletion)
    4. Add new Gaussians with activation as opacity multiplier

    All operations are differentiable.
    """
    # Avoid circular import by importing here
    from src.model.types import Gaussians

    B, N = gaussian_params.shape[:2]
    K = new_params.shape[1]

    # ===== Refine existing Gaussians =====
    # (a) Move means
    refined_means = gaussian_means + delta_means  # [B, N, 3]

    # (b) Adjust scales
    refined_scales = 0.001 * F.softplus(
        (gaussian_params[:, :, 1:4] + delta_params[:, :, :3]).clamp(-10, 10)
    ).clamp(max=0.3)

    # (c) Adjust rotations
    rot_raw = gaussian_params[:, :, 4:8] + delta_params[:, :, 3:7]
    rot_norm = rot_raw.norm(dim=-1, keepdim=True)
    refined_rotations = rot_raw / (rot_norm + 1e-8)

    # (d) Adjust SH
    refined_harmonics = (gaussian_params[:, :, 8:] + delta_params[:, :, 7:]).view(B, N, 3, -1)

    # (e) Adjust opacity + apply deletion (keep_prob)
    raw_opacity = torch.sigmoid(gaussian_params[:, :, 0:1] + delta_params[:, :, 0:1])
    refined_opacities = (raw_opacity * keep_prob).squeeze(-1)  # [B, N]

    # (f) Build covariances
    R = build_rotation_matrix(refined_rotations)
    I = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
    refined_covariances = R @ (I * refined_scales.unsqueeze(-1)).pow(2) @ R.transpose(-1, -2)

    # ===== Add new Gaussians =====
    # Transform raw predictions
    new_scales = 0.001 * F.softplus(new_params[:, :, :3].clamp(-10, 10)).clamp(max=0.3)
    rot_raw_new = new_params[:, :, 3:7]
    rot_norm_new = rot_raw_new.norm(dim=-1, keepdim=True)
    new_rotations = rot_raw_new / (rot_norm_new + 1e-8)
    new_harmonics = new_params[:, :, 7:].view(B, K, 3, -1)
    new_opacities = torch.sigmoid(new_params[:, :, 0:1]) * new_activations  # [B, K]

    R_new = build_rotation_matrix(new_rotations)
    new_covariances = R_new @ (I * new_scales.unsqueeze(-1)).pow(2) @ R_new.transpose(-1, -2)

    # ===== Concatenate existing + new =====
    combined_means = torch.cat([refined_means, new_means], dim=1)
    combined_covariances = torch.cat([refined_covariances, new_covariances], dim=1)
    combined_harmonics = torch.cat([refined_harmonics, new_harmonics], dim=1)
    combined_opacities = torch.cat([refined_opacities, new_opacities.squeeze(-1)], dim=1)
    combined_scales = torch.cat([refined_scales, new_scales], dim=1)
    combined_rotations = torch.cat([refined_rotations, new_rotations], dim=1)

    return Gaussians(
        means=combined_means,
        covariances=combined_covariances,
        harmonics=combined_harmonics,
        opacities=combined_opacities,
        scales=combined_scales,
        rotations=combined_rotations,
    )
