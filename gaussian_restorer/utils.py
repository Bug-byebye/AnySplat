"""Utility functions for Gaussian repair training."""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image


def normalize_intrinsics(intr, W, H):
    """Normalize intrinsics as AnySplat encoder expects."""
    intr_n = intr.clone()
    intr_n[..., 0, :] = intr[..., 0, :] / W
    intr_n[..., 1, :] = intr[..., 1, :] / H
    return intr_n


def save_image(color_tensor, path):
    """Save a [3,H,W] tensor as an image file."""
    img = color_tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


@torch.no_grad()
def render_gaussians(decoder, gaussians, extrinsics, intrinsics, H, W):
    """Render Gaussians and return DecoderOutput."""
    out = decoder.forward(
        gaussians, extrinsics, intrinsics,
        torch.tensor([[0.1]], device=gaussians.means.device),
        torch.tensor([[100.0]], device=gaussians.means.device),
        (H, W),
    )
    return out


def extract_params_from_gaussians(g, B=1):
    """
    Extract raw parameter tensor from Gaussians object.

    Returns:
        params: [B, N, 83] — opacity(1) + scales(3) + rot(4) + SH(75)
        means:  [B, N, 3]
    """
    N = g.means.shape[1]
    params = torch.cat([
        g.opacities.unsqueeze(-1),              # [B, N, 1]
        g.scales,                                # [B, N, 3]
        g.rotations,                             # [B, N, 4]
        g.harmonics.view(B, N, -1),              # [B, N, 75]
    ], dim=-1)  # [B, N, 83]
    return params, g.means
