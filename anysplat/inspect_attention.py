#!/usr/bin/env python3
"""
Inspect Attention Maps in AnySplat Model.

This script loads the AnySplat model, runs inference on input images,
captures attention weights from the VGGT backbone, and generates
comprehensive visualizations to analyze what the model focuses on.

Usage:
    # Basic usage with default test images
    python inspect_attention.py

    # With specific images
    python inspect_attention.py --image_folder /path/to/images

    # Control which layers/heads to capture
    python inspect_attention.py --layer_indices 0 11 23 --head_indices 0 7

    # Save attentions to disk for later analysis
    python inspect_attention.py --save_attentions

    # Load previously saved attentions (skip model inference)
    python inspect_attention.py --load_attentions saved_attentions.pt
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from PIL import Image
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'anysplat'))

from src.model.model.anysplat import AnySplat
from src.analysis.attention_capture import AttentionCapture
from src.analysis.visualize_attention import (
    denormalize_image,
    get_patch_grid,
    visualize_attention_summary,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect attention maps in AnySplat model"
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="examples/vrnerf/riverview",
        help="Folder containing input images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="attention_analysis",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="lhjiang/anysplat",
        help="Model identifier (HF hub ID or local path)",
    )
    parser.add_argument(
        "--layer_indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific layers to capture (0-based). Default: all",
    )
    parser.add_argument(
        "--head_indices",
        type=int,
        nargs="*",
        default=None,
        help="Specific attention heads to capture. Default: all",
    )
    parser.add_argument(
        "--no_frame",
        action="store_true",
        help="Disable frame-level attention capture",
    )
    parser.add_argument(
        "--no_global",
        action="store_true",
        help="Disable global-level attention capture",
    )
    parser.add_argument(
        "--save_attentions",
        type=str,
        default=None,
        help="Save captured attentions to this .pt file",
    )
    parser.add_argument(
        "--load_attentions",
        type=str,
        default=None,
        help="Load attentions from .pt file (skip model inference)",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=448,
        help="Input image size (AnySplat expects 448)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=4,
        help="Maximum number of images to process",
    )
    return parser.parse_args()


def load_images_from_folder(folder: str, max_images: int, image_size: int):
    """Load and preprocess images from a folder.

    Returns:
        images_tensor: [1, V, 3, H, W] in range [0, 1] (as expected by model.inference)
        pil_images: List of PIL Images for visualization reference
    """
    supported = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
    image_paths = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(supported)
    ])

    if not image_paths:
        raise FileNotFoundError(f"No images found in {folder}")

    image_paths = image_paths[:max_images]
    print(f"Loading {len(image_paths)} images from {folder}")

    pil_images = []
    tensors = []
    for path in image_paths:
        img = Image.open(path).convert("RGB")
        pil_images.append(img)

        # Resize maintaining aspect ratio, then center crop
        w, h = img.size
        if w > h:
            new_h = image_size
            new_w = int(w * (new_h / h))
        else:
            new_w = image_size
            new_h = int(h * (new_w / w))
        img = img.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - image_size) // 2
        top = (new_h - image_size) // 2
        right = left + image_size
        bottom = top + image_size
        img = img.crop((left, top, right, bottom))

        # Convert to tensor in [-1, 1] range (then model will convert to [0, 1])
        img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 127.5 - 1.0
        tensors.append(img_tensor)

    images_tensor = torch.stack(tensors, dim=0).unsqueeze(0)  # [1, V, 3, H, W]
    print(f"  Tensor shape: {images_tensor.shape}, range: [{images_tensor.min():.2f}, {images_tensor.max():.2f}]")
    return images_tensor, pil_images


def main():
    args = parse_args()
    device = torch.device(args.device)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Attention Map Inspector for AnySplat")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")

    # ----- Load model -----
    if args.load_attentions is None:
        print(f"\n[1/3] Loading AnySplat model from '{args.model_path}'...")
        model = AnySplat.from_pretrained(args.model_path)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print("  Model loaded successfully.")

        # Get the aggregator from the encoder
        aggregator = model.encoder.aggregator
        num_frame_blocks = len(aggregator.frame_blocks)
        num_global_blocks = len(aggregator.global_blocks)
        print(f"  Aggregator: {num_frame_blocks} frame blocks, {num_global_blocks} global blocks")

        # ----- Load images -----
        print(f"\n[2/3] Loading images from '{args.image_folder}'...")
        images_tensor, pil_images = load_images_from_folder(
            args.image_folder, args.max_images, args.image_size
        )
        V = images_tensor.shape[1]
        images_tensor = images_tensor.to(device)

        # ----- Capture attention -----
        print(f"\n[3/3] Running inference with attention capture...")
        print(f"  Capturing: {'frame ' if not args.no_frame else ''}{'global' if not args.no_global else ''}")
        if args.layer_indices:
            print(f"  Layers: {args.layer_indices}")
        if args.head_indices:
            print(f"  Heads: {args.head_indices}")

        capturer = AttentionCapture(
            aggregator=aggregator,
            capture_frame=not args.no_frame,
            capture_global=not args.no_global,
            layer_indices=args.layer_indices,
            head_indices=args.head_indices,
            store_on_cpu=True,
        )

        with capturer:
            # model.inference expects [0, 1] range
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    gaussians, pred_context_pose = model.inference(
                        (images_tensor + 1) * 0.5  # [-1, 1] -> [0, 1]
                    )

        print(f"  Captured {len(capturer.attentions)} attention maps")

        if args.save_attentions:
            capturer.save(args.save_attentions)

    else:
        # Load pre-saved attentions
        print(f"\n[1/3] Loading pre-saved attentions from '{args.load_attentions}'...")
        capturer = AttentionCapture.__new__(AttentionCapture)
        capturer.aggregator = None
        capturer.attentions = {}
        capturer.load(args.load_attentions)
        V = 1  # unknown number of views

    # ----- Generate visualizations -----
    print(f"\nGenerating visualizations...")

    # Determine patch grid
    # VGGT uses patch_size=14, and the aggregator resizes to 518 internally
    # But AnySplat uses image_size=448 by default... Let me check.
    # The process_image in inference.py uses 448, but VGGT/Aggregator internally
    # might resize. Looking at aggregator code, it receives images at their
    # original size and patch_embed handles them.
    # For the default case, let's use the image_size from args.

    # Get a reference image for visualization
    if args.load_attentions is None:
        ref_images_np = [np.array(img) for img in pil_images]
    else:
        # Placeholder
        ref_images_np = [np.zeros((args.image_size, args.image_size, 3), dtype=np.uint8)]

    # Debug: print attention shapes
    print(f"  Captured {len(capturer.attentions)} attention maps:")
    for key, attn in capturer.attentions.items():
        print(f"    {key}: {attn.shape}")

    # Get patch grid from attention shape
    patch_grid = None
    if capturer.attentions:
        first_key = next(iter(capturer.attentions))
        first_attn = capturer.attentions[first_key]
        # Shape can be [B*S, num_heads, N, N] or [num_heads, N, N]
        N = first_attn.shape[-1]  # total tokens (last dim)
        block_type, layer_idx = first_key
        # Frame attention: N = tokens_per_view (1 camera + 4 register + patches)
        if block_type == "frame":
            patch_tokens = N - 5  # skip camera_token (1) + register_token (4)
        else:
            # Global attention: N = V * tokens_per_view
            patch_tokens = N // V - 5 if V > 0 else N - 5

        # Patch grid: square patch grid
        ps = int(round(patch_tokens ** 0.5))
        for guess_ps in range(max(ps - 2, 1), ps + 3):
            if guess_ps * guess_ps == patch_tokens:
                patch_grid = (guess_ps, guess_ps)
                break
        if patch_grid is None:
            for guess_ps in range(ps, 0, -1):
                if patch_tokens % guess_ps == 0:
                    patch_grid = (guess_ps, patch_tokens // guess_ps)
                    break
        if patch_grid:
            print(f"  Detected patch grid: {patch_grid} (from {patch_tokens} patches)")
        else:
            patch_grid = (32, 32)
            print(f"  Warning: Could not determine patch grid, using {patch_grid}")

    if patch_grid is None:
        patch_grid = (32, 32)
        print(f"  Using default patch grid: {patch_grid}")

    # Generate visualizations
    print(f"  Using {V} views, patch grid: {patch_grid}")
    visualize_attention_summary(
        images=ref_images_np,
        attentions=capturer.attentions,
        patch_grid=patch_grid,
        output_dir=output_dir,
        num_views=V,
        dpi=150,
    )

    print(f"\n{'='*60}")
    print(f"Done! All visualizations saved to: {output_dir}")
    print(f"{'='*60}")
    print(f"\nKey findings to look for:")
    print(f"  1. Mean attention: Does the model focus on specific regions?")
    print(f"  2. Attention entropy: Are some regions 'uncertain' (high entropy)?")
    print(f"  3. Per-layer patterns: How does attention evolve through layers?")
    print(f"  4. Cross-view (if multi-view): Does the model find correspondences?")
    print(f"\nTo disable attention capture for a particular block type:")
    print(f"    python inspect_attention.py --no_global")
    print(f"    python inspect_attention.py --no_frame")


if __name__ == "__main__":
    main()
