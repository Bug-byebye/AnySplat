"""
Visualization utilities for attention maps from the AnySplat (VGGT-based) model.

Produces several types of visualizations:
1. Query attention heatmap - For a selected query patch, show attention distribution
2. Attention entropy map - Per-patch entropy of attention distribution
3. Layer-wise comparison grid - Attention patterns across layers
4. Cross-view attention (global) - Attention between different views
5. Attention rollout - Aggregated attention (product of all layers)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from einops import rearrange, repeat
from torch import Tensor


# ---------------------------------------------------------------------------
#  Helper: convert attention weights to 2D heatmap for a given query position
# ---------------------------------------------------------------------------

def attention_to_heatmap(
    attn_weights: Tensor,
    query_idx: int,
    patch_grid: Tuple[int, int],
    head_idx: Optional[int] = None,
) -> np.ndarray:
    """
    Convert attention weights for a single query patch into a 2D heatmap.

    Args:
        attn_weights: [num_heads, N] (if head_idx given) or [N] (single head).
                      Already softmaxed weights for ONE query patch.
        query_idx: Index of the query patch (not used for selection, just tracking).
        patch_grid: (H, W) grid dimensions.
        head_idx: Which head to use. If None, average over all heads.

    Returns:
        heatmap: (H, W) numpy array of attention weights.
    """
    if attn_weights.ndim == 2:
        # attn_weights is [num_heads, N], average or select
        if head_idx is not None:
            weights = attn_weights[head_idx]
        else:
            weights = attn_weights.mean(dim=0)  # average over heads
    elif attn_weights.ndim == 1:
        weights = attn_weights
    else:
        raise ValueError(f"Unexpected attn_weights shape: {attn_weights.shape}")

    H, W = patch_grid
    # Exclude special tokens (camera + register). The first patch_start_idx tokens are special.
    # By default patch_start_idx = 1 + 4 = 5.
    n_patches = H * W
    if len(weights) > n_patches:
        weights = weights[-n_patches:]  # take only patch tokens
    elif len(weights) < n_patches:
        weights = weights[:n_patches]

    return weights.reshape(H, W).cpu().float().numpy()


def get_patch_grid(image_size: int, patch_size: int) -> Tuple[int, int]:
    """Get the patch grid dimensions for a given image size."""
    return image_size // patch_size, image_size // patch_size


def denormalize_image(img_tensor: Tensor) -> np.ndarray:
    """
    Convert normalized image tensor back to [0, 255] uint8 numpy for visualization.

    Handles [-1, 1] range (most common in this codebase).
    """
    if isinstance(img_tensor, Tensor):
        img = img_tensor.detach().cpu().float()
    else:
        img = torch.from_numpy(img_tensor).float()

    # If shape is [C, H, W] or [1, C, H, W], convert to [H, W, C]
    if img.ndim == 4:
        img = img.squeeze(0)

    if img.ndim == 3:
        # [C, H, W] -> [H, W, C]
        if img.shape[0] == 3 or img.shape[0] == 1:
            img = img.permute(1, 2, 0)
        # [H, W, C]

    # Handle range
    if img.min() < 0:
        img = (img + 1) / 2  # [-1, 1] -> [0, 1]
    img = img.clamp(0, 1).numpy()
    return (img * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
#  Core visualization functions
# ---------------------------------------------------------------------------


def visualize_query_attention(
    image: np.ndarray,
    attn_weights: Tensor,
    query_patches: List[Tuple[int, int]],
    patch_grid: Tuple[int, int],
    save_path: str,
    title: str = "Query Attention",
    head_idx: Optional[int] = None,
    cmap: str = "jet",
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (20, 5),
    dpi: int = 150,
):
    """
    Visualize attention maps for selected query patches.

    For each query patch (y, x) in the patch grid, shows which other patches
    it attends to, overlaid on the original image.

    Args:
        image: [H, W, 3] uint8 numpy array.
        attn_weights: [num_heads, N] tensor -- attention from one query to all keys.
                      Or [num_heads, N, N] tensor (takes the query_idx column).
        query_patches: List of (y, x) patch coordinates to visualize.
        patch_grid: (H_patches, W_patches).
        save_path: Where to save the figure.
        title: Title for the figure.
        head_idx: Which attention head to use. None = average all heads.
        cmap: Colormap for attention overlay.
        alpha: Overlay transparency (0 = transparent, 1 = opaque).
        figsize: Figure size (width, height) in inches.
        dpi: Figure DPI.
    """
    H, W = image.shape[:2]
    pH, pW = patch_grid
    patch_size_y = H / pH
    patch_size_x = W / pW

    # If attn_weights is [num_heads, N, N] matrix, extract the query_idx columns
    if attn_weights.ndim == 3:
        NH, Nq, Nk = attn_weights.shape
        # Per-query extraction happens per query in the loop below
        pass

    n_queries = len(query_patches)
    ncols = n_queries + 1  # +1 for original image marker

    fig, axes = plt.subplots(1, ncols, figsize=figsize, dpi=dpi)

    # First subplot: original image with markers
    ax = axes[0]
    ax.imshow(image)
    for qy, qx in query_patches:
        cy = qy * patch_size_y + patch_size_y / 2
        cx = qx * patch_size_x + patch_size_x / 2
        ax.scatter(cx, cy, c="white", s=80, marker="*", edgecolors="black", linewidths=1, zorder=5)
        ax.annotate(f"({qy},{qx})", (cx, cy), fontsize=8, color="white",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.7))
    ax.set_title("Query Points", fontsize=10)
    ax.axis("off")

    # For each query, show attention heatmap
    for i, (qy, qx) in enumerate(query_patches):
        query_idx = qy * pW + qx  # flat index
        ax = axes[i + 1]

        ax.imshow(image, alpha=1.0)

        # Get attention weights for this query
        if attn_weights.ndim == 2:
            weights = attn_weights
        elif attn_weights.ndim == 3:
            # [num_heads, N, N], select the query_idx column (keys)
            weights = attn_weights[:, query_idx, :]
        else:
            raise ValueError(f"Unexpected attn_weights shape: {attn_weights.shape}")

        # Convert to heatmap
        heatmap = attention_to_heatmap(weights, query_idx, (pH, pW), head_idx=head_idx)

        # Resize to image size
        heatmap_resized = _resize_heatmap(heatmap, (H, W))

        # Overlay heatmap
        masked_heatmap = np.ma.masked_where(heatmap_resized < 0.01, heatmap_resized)
        ax.imshow(masked_heatmap, cmap=cmap, alpha=alpha, vmin=0, vmax=heatmap.max())

        ax.set_title(f"Query ({qy},{qx})", fontsize=8)
        ax.axis("off")

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved query attention visualization to {save_path}")


def visualize_attention_entropy(
    image: np.ndarray,
    attn_weights: Tensor,
    patch_grid: Tuple[int, int],
    save_path: str,
    title: str = "Attention Entropy",
    cmap: str = "inferno",
    alpha: float = 0.6,
    figsize: Tuple[int, int] = (20, 5),
    dpi: int = 150,
):
    """
    Visualize attention entropy per patch.

    Low entropy = focused attention (model knows where to look).
    High entropy = diffuse attention (model is uncertain).

    Args:
        image: [H, W, 3] uint8 numpy array.
        attn_weights: [num_heads, N, N] tensor.
        patch_grid: (H_patches, W_patches).
        save_path: Where to save.
        title: Title.
        cmap: Colormap.
        alpha: Overlay transparency.
    """
    H, W = image.shape[:2]
    pH, pW = patch_grid

    if attn_weights.ndim == 4 and attn_weights.shape[0] == 1:
        attn_weights = attn_weights[0]

    if attn_weights.ndim == 3:
        # [num_heads, N, N]
        num_heads, N, _ = attn_weights.shape
        # Compute entropy per head: H(p) = -sum(p * log(p))
        entropy = -(attn_weights * torch.log(attn_weights.clamp(min=1e-10))).sum(dim=-1)  # [num_heads, N]
        # Average over heads
        entropy = entropy.mean(dim=0)  # [N]
    else:
        raise ValueError(f"Unexpected attn_weights shape: {attn_weights.shape}")

    # Take only patch tokens
    n_patches = pH * pW
    if len(entropy) > n_patches:
        entropy = entropy[-n_patches:]
    elif len(entropy) < n_patches:
        padded = torch.zeros(n_patches, device=entropy.device)
        padded[:len(entropy)] = entropy
        entropy = padded

    entropy_map = entropy.reshape(pH, pW).cpu().float().numpy()
    entropy_map_resized = _resize_heatmap(entropy_map, (H, W))

    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi)

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("Original Image", fontsize=10)
    axes[0].axis("off")

    # Entropy heatmap overlay
    axes[1].imshow(image, alpha=1.0)
    axes[1].imshow(entropy_map_resized, cmap=cmap, alpha=alpha)
    axes[1].set_title("Attention Entropy (overlay)", fontsize=10)
    axes[1].axis("off")

    # Pure entropy map
    im = axes[2].imshow(entropy_map_resized, cmap=cmap)
    axes[2].set_title("Attention Entropy (raw)", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved attention entropy visualization to {save_path}")


def visualize_layer_comparison(
    image: np.ndarray,
    attentions: Dict[Tuple[str, int], Tensor],
    patch_grid: Tuple[int, int],
    save_path: str,
    block_type: str = "frame",
    head_idx: Optional[int] = None,
    max_layers: int = 12,
    cmap: str = "jet",
    alpha: float = 0.6,
    dpi: int = 150,
):
    """
    Compare attention patterns across different layers for a fixed block type.

    Shows the mean attention heatmap (averaged over all query patches) for each layer,
    plus attention entropy per layer.

    Args:
        image: [H, W, 3] uint8 numpy array.
        attentions: Dict from (block_type, layer_idx) to [num_heads, N, N] tensor.
        patch_grid: (H_patches, W_patches).
        save_path: Output path.
        block_type: "frame" or "global".
        head_idx: Which head to visualize. None = average all heads.
        max_layers: Maximum number of layers to show.
        cmap: Colormap.
        alpha: Transparency.
        dpi: Figure DPI.
    """
    H, W = image.shape[:2]
    pH, pW = patch_grid

    # Get relevant layers
    layer_keys = sorted([k for k in attentions if k[0] == block_type])
    if len(layer_keys) > max_layers:
        # Uniformly sample
        indices = np.linspace(0, len(layer_keys) - 1, max_layers, dtype=int)
        layer_keys = [layer_keys[i] for i in indices]

    n_layers = len(layer_keys)
    ncols = n_layers

    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 4, 4), dpi=dpi)
    if ncols == 1:
        axes = [axes]

    for i, key in enumerate(layer_keys):
        attn = attentions[key]  # [num_heads, N, N]

        # Average over all query patches -> mean attention distribution
        if attn.ndim == 3:
            mean_attn = attn.mean(dim=1)  # [num_heads, N]
        else:
            raise ValueError(f"Unexpected shape: {attn.shape}")

        if head_idx is not None:
            weights = mean_attn[head_idx]
        else:
            weights = mean_attn.mean(dim=0)  # [N]

        # Get patch tokens only
        n_patches = pH * pW
        if len(weights) > n_patches:
            weights = weights[-n_patches:]

        heatmap = weights.reshape(pH, pW).cpu().float().numpy()
        heatmap_resized = _resize_heatmap(heatmap, (H, W))

        ax = axes[i]
        ax.imshow(image, alpha=1.0)
        masked = np.ma.masked_where(heatmap_resized < heatmap.max() * 0.1, heatmap_resized)
        ax.imshow(masked, cmap=cmap, alpha=alpha)
        layer_type, layer_idx = key
        ax.set_title(f"{layer_type} Layer {layer_idx}", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"Mean Attention per Layer ({block_type})", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved layer comparison to {save_path}")

    # Also save entropy per layer
    _save_entropy_per_layer(figsize=(ncols * 2, 4), attentions=attentions, block_type=block_type,
                            patch_grid=patch_grid, save_path=save_path.replace(".png", "_entropy.png"), dpi=dpi)


def _save_entropy_per_layer(
    figsize: Tuple[int, int],
    attentions: Dict[Tuple[str, int], Tensor],
    block_type: str,
    patch_grid: Tuple[int, int],
    save_path: str,
    dpi: int,
):
    """Helper: plot entropy vs layer index."""
    layer_keys = sorted([k for k in attentions if k[0] == block_type])
    pH, pW = patch_grid
    n_patches = pH * pW

    entropies = []
    for k in layer_keys:
        attn = attentions[k]
        if attn.ndim == 3:
            h, N, _ = attn.shape
            ent = -(attn * torch.log(attn.clamp(min=1e-10))).sum(dim=-1)
            ent = ent.mean(dim=0)
            if len(ent) > n_patches:
                ent = ent[-n_patches:]
            entropies.append(ent.mean().item())

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    if entropies:
        ax.plot(range(len(layer_keys)), entropies, "o-", linewidth=2, markersize=5)
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Mean Attention Entropy")
        ax.set_title(f"Attention Entropy across Layers ({block_type})")
        ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def visualize_mean_attention(
    image: np.ndarray,
    attn_weights: Tensor,
    patch_grid: Tuple[int, int],
    save_path: str,
    title: str = "Mean Attention",
    head_idx: Optional[int] = None,
    cmap: str = "jet",
    alpha: float = 0.6,
    dpi: int = 150,
):
    """
    Visualize the mean attention pattern averaged over all query patches.
    This shows where patches collectively "look" the most.

    Args:
        attn_weights: [num_heads, N, N] or [num_heads, 1, N] or [num_heads, N].
                      If 3D with shape [h, 1, N], treats it as mean attention [h, N].
    """
    H, W = image.shape[:2]
    pH, pW = patch_grid

    if attn_weights.ndim == 4 and attn_weights.shape[0] == 1:
        attn_weights = attn_weights[0]

    if attn_weights.ndim == 3:
        # [num_heads, N, N] -> mean over query dim -> [num_heads, N]
        mean_attn = attn_weights.mean(dim=1)
    elif attn_weights.ndim == 2:
        # [num_heads, N] - already processed
        mean_attn = attn_weights
    else:
        raise ValueError(f"Unexpected shape: {attn_weights.shape}")

    if head_idx is not None:
        weights = mean_attn[head_idx]
    else:
        weights = mean_attn.mean(dim=0)

    n_patches = pH * pW
    if len(weights) > n_patches:
        weights = weights[-n_patches:]
    elif len(weights) < n_patches:
        padded = torch.zeros(n_patches, device=weights.device)
        padded[:len(weights)] = weights
        weights = padded

    heatmap = weights.reshape(pH, pW).cpu().float().numpy()
    heatmap_resized = _resize_heatmap(heatmap, (H, W))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)
    axes[0].imshow(image)
    axes[0].imshow(heatmap_resized, cmap=cmap, alpha=alpha)
    axes[0].set_title(f"{title} (overlay)", fontsize=10)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap_resized, cmap=cmap)
    axes[1].set_title(f"{title} (raw)", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved mean attention to {save_path}")


def visualize_global_cross_view(
    images: List[np.ndarray],
    global_attn: Tensor,
    num_views: int,
    patch_grid: Tuple[int, int],
    save_path: str,
    view_pair: Optional[Tuple[int, int]] = None,
    head_idx: Optional[int] = None,
    cmap: str = "jet",
    alpha: float = 0.6,
    dpi: int = 150,
):
    """
    Visualize cross-view attention from global attention blocks.

    Shows which patches in a source view attend to which patches in a target view.

    Args:
        images: List of [H, W, 3] uint8 arrays, one per view.
        global_attn: [num_heads, S*P, S*P] tensor.
        num_views: Number of views S.
        patch_grid: (pH, pW).
        save_path: Output path.
        view_pair: (src_view, dst_view). The source view's patches attend to the dst view.
        head_idx: Which head.
        cmap, alpha, dpi: Visualization parameters.
    """
    H, W = images[0].shape[:2]
    pH, pW = patch_grid
    P = pH * pW

    if global_attn.ndim == 4 and global_attn.shape[0] == 1:
        global_attn = global_attn[0]

    if global_attn.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {global_attn.shape}")

    NH, SP, _ = global_attn.shape
    assert SP == num_views * (P + 5), f"Expected SP={num_views * (P + 5)}, got {SP}"

    if view_pair is None:
        view_pair = (0, 1)

    src_v, dst_v = view_pair

    # Get the sub-block: [num_heads, P_src, P_dst]
    # Tokens layout: [cam0, reg0_0..reg0_3, patches0, cam1, reg1_0..reg1_3, patches1, ...]
    tokens_per_view = SP // num_views
    src_start = src_v * tokens_per_view + 5  # skip camera(1) + register(4)
    src_end = src_start + P
    dst_start = dst_v * tokens_per_view + 5
    dst_end = dst_start + P

    sub_attn = global_attn[:, src_start:src_end, dst_start:dst_end]  # [num_heads, P, P]

    # Average over source queries -> mean attention from src to dst
    if head_idx is not None:
        mean_weights = sub_attn[head_idx].mean(dim=0)  # [P]
    else:
        mean_weights = sub_attn.mean(dim=(0, 1))  # [P]

    heatmap = mean_weights.reshape(pH, pW).cpu().float().numpy()
    heatmap_resized = _resize_heatmap(heatmap, (H, W))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)

    # Show the destination image with attention overlay
    axes[0].imshow(images[dst_v])
    axes[0].imshow(heatmap_resized, cmap=cmap, alpha=alpha)
    axes[0].set_title(f"View {src_v} → View {dst_v} Attention", fontsize=10)
    axes[0].axis("off")

    im = axes[1].imshow(heatmap_resized, cmap=cmap)
    axes[1].set_title("Raw Attention", fontsize=10)
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"Saved cross-view attention to {save_path}")


def visualize_attention_summary(
    images: List[np.ndarray],
    attentions: Dict[Tuple[str, int], Tensor],
    patch_grid: Tuple[int, int],
    output_dir: str,
    image_size: int = 518,
    patch_size: int = 14,
    num_views: int = 1,
    dpi: int = 150,
):
    """
    Generate a comprehensive set of attention visualizations.

    Args:
        images: List of [H, W, 3] uint8 arrays, one per view.
        attentions: Captured attention dict from AttentionCapture.
                    Values are either 4D [B, num_heads, N, N] or 3D [num_heads, N, N].
        patch_grid: (pH, pW).
        output_dir: Directory to save all visualizations.
        image_size: Input image size.
        patch_size: ViT patch size.
        num_views: Number of input views.
        dpi: Figure DPI.
    """
    os.makedirs(output_dir, exist_ok=True)

    ref_image = images[0] if isinstance(images, list) else images

    # Normalize attention tensors: keep 3D [num_heads, N, N] by selecting first batch item.
    normalized_attentions = {}
    for key, attn in attentions.items():
        if attn.ndim == 4:
            normalized_attentions[key] = attn[0]  # take first batch item
        else:
            normalized_attentions[key] = attn
    attentions = normalized_attentions

    # Get frame and global keys
    frame_keys = sorted([k for k in attentions if k[0] == "frame"])
    global_keys = sorted([k for k in attentions if k[0] == "global"])

    print(f"\n{'='*60}")
    print(f"Generating attention visualizations...")
    print(f"  Captured {len(frame_keys)} frame attention layers")
    print(f"  Captured {len(global_keys)} global attention layers")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # 1. Mean attention per layer (average focus pattern)
    if frame_keys:
        # Pick middle and last frame layer for quick overview
        mid_idx = frame_keys[len(frame_keys) // 2]
        last_idx = frame_keys[-1]

        # Mean attention for middle layer
        visualize_mean_attention(
            ref_image, attentions[mid_idx], patch_grid,
            save_path=os.path.join(output_dir, "frame_mean_attn_mid.png"),
            title=f"Frame Mean Attention (Layer {mid_idx[1]})",
            dpi=dpi,
        )

        # Mean attention for last layer
        visualize_mean_attention(
            ref_image, attentions[last_idx], patch_grid,
            save_path=os.path.join(output_dir, "frame_mean_attn_last.png"),
            title=f"Frame Mean Attention (Layer {last_idx[1]})",
            dpi=dpi,
        )

        # Mean attention across all frame layers
        all_means = []
        for k in frame_keys:
            attn = attentions[k]
            if attn.ndim == 3:
                all_means.append(attn.mean(dim=(0, 1)))  # [N]
        if all_means:
            all_attn = torch.stack(all_means).mean(dim=0)  # [N]
            n_patches = patch_grid[0] * patch_grid[1]
            if len(all_attn) > n_patches:
                all_attn = all_attn[-n_patches:]
            # Reshape as [1, N] since it's already averaged
            visualize_mean_attention(
                ref_image, all_attn.unsqueeze(0), patch_grid,
                save_path=os.path.join(output_dir, "frame_mean_attn_all_layers.png"),
                title="Frame Mean Attention (All Layers)",
                dpi=dpi,
            )

        # Attention entropy for last frame layer
        visualize_attention_entropy(
            ref_image, attentions[last_idx], patch_grid,
            save_path=os.path.join(output_dir, "frame_entropy_last.png"),
            title=f"Frame Attention Entropy (Layer {last_idx[1]})",
            dpi=dpi,
        )

        # Layer comparison
        visualize_layer_comparison(
            ref_image, attentions, patch_grid,
            save_path=os.path.join(output_dir, "frame_layer_comparison.png"),
            block_type="frame", max_layers=12, dpi=dpi,
        )

    # 2. Global attention cross-view (if we have multiple views)
    if global_keys and num_views > 1:
        # First and last global layers
        for k in [global_keys[0], global_keys[-1]]:
            visualize_global_cross_view(
                images, attentions[k], num_views, patch_grid,
                save_path=os.path.join(output_dir, f"global_crossview_layer{k[1]}.png"),
                dpi=dpi,
            )

        # Entropy for global attention
        visualize_attention_entropy(
            images[0] if isinstance(images, list) else ref_image,
            attentions[global_keys[-1]], patch_grid,
            save_path=os.path.join(output_dir, "global_entropy_last.png"),
            title="Global Attention Entropy",
            dpi=dpi,
        )

    # 3. Query-specific attention for some interesting patches
    if frame_keys:
        pH, pW = patch_grid
        # Example query patches: center and a few others
        query_patches = [
            (pH // 2, pW // 2),  # center
            (pH // 4, pW // 4),  # upper-left
            (3 * pH // 4, 3 * pW // 4),  # lower-right
        ]

        for k_idx, k in enumerate([frame_keys[0], frame_keys[len(frame_keys) // 2], frame_keys[-1]]):
            if k_idx > 0:
                break  # Just show the first/last pair to avoid too many files
            attn = attentions[k]
            if attn.ndim == 3:
                # Show query attention for center patch only
                qy, qx = pH // 2, pW // 2
                visualize_query_attention(
                    ref_image, attn, [(qy, qx)], patch_grid,
                    save_path=os.path.join(output_dir, f"frame_query_center_layer{k[1]}.png"),
                    title=f"Query ({qy},{qx}) Frame Layer {k[1]}",
                    dpi=dpi,
                )

    print(f"\nAll visualizations saved to {output_dir}")


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _resize_heatmap(heatmap_2d: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize a 2D heatmap to target (H, W) using bilinear interpolation."""
    H, W = target_size
    h, w = heatmap_2d.shape

    y_ratio = H / h
    x_ratio = W / w

    y_coords = np.clip(np.arange(H) / y_ratio, 0, h - 1).astype(np.float32)
    x_coords = np.clip(np.arange(W) / x_ratio, 0, w - 1).astype(np.float32)

    y0 = y_coords.astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x0 = x_coords.astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)

    wy = y_coords - y0
    wx = x_coords - x0

    result = (
        heatmap_2d[np.ix_(y0, x0)] * (1 - wy)[:, None] * (1 - wx)[None, :]
        + heatmap_2d[np.ix_(y0, x1)] * (1 - wy)[:, None] * wx[None, :]
        + heatmap_2d[np.ix_(y1, x0)] * wy[:, None] * (1 - wx)[None, :]
        + heatmap_2d[np.ix_(y1, x1)] * wy[:, None] * wx[None, :]
    )

    return result
