"""
Standalone demo: Test Wan2.2-enhanced Gaussian Scene Restorer inference.

This script:
  1. Loads the Wan2.2-TI2V-5B model
  2. Creates a test with synthetic images
  3. Runs WAN feature extraction
  4. Runs the full V3 restorer pipeline
  5. Reports memory usage and timing

Usage:
    # Basic test with synthetic data
    python -m post.wan_restorer.demo_inference

    # Test with actual images
    python -m post.wan_restorer.demo_inference --image_dir /path/to/images

    # Test with DPT features (simulated)
    python -m post.wan_restorer.demo_inference --use_dpt

    # Test with different Wan models
    python -m post.wan_restorer.demo_inference --model_id Wan-AI/Wan2.2-T2V-A14B-Diffusers
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

# Ensure the wan_restorer package directory is importable
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)  # post/
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from wan_restorer.config import WanRestorerCfg
from wan_restorer.wan_feature_extractor import WanFeatureExtractor
from wan_restorer.gaussian_scene_restorer_v3 import GaussianSceneRestorerV3


def load_test_images(image_dir: str = None, num_views: int = 2,
                     height: int = 224, width: int = 448) -> torch.Tensor:
    """
    Load test images from a directory or create synthetic ones.

    Returns:
        images: [1, num_views, 3, height, width] tensor in [0, 1] range.
    """
    if image_dir is not None:
        # Load actual images from directory
        img_paths = sorted(Path(image_dir).glob("*.jpg")) + \
                    sorted(Path(image_dir).glob("*.png"))
        img_paths = img_paths[:num_views]

        if len(img_paths) == 0:
            print(f"  No images found in {image_dir}, using synthetic data.")
            return create_synthetic_images(num_views, height, width)

        images = []
        for path in img_paths:
            img = Image.open(path).convert("RGB")
            img = img.resize((width, height), Image.LANCZOS)
            img_tensor = torch.tensor(
                (torch.from_numpy(np.array(img)).float() / 255.0)
            ).permute(2, 0, 1)  # [3, H, W]
            images.append(img_tensor)

        # Ensure num_views
        while len(images) < num_views:
            images.append(images[-1].clone())
        images = images[:num_views]

        return images.unsqueeze(0)  # [1, V, 3, H, W]

    else:
        return create_synthetic_images(num_views, height, width)


def create_synthetic_images(num_views: int = 2, height: int = 224,
                             width: int = 448) -> torch.Tensor:
    """Create synthetic test images with basic patterns."""
    images = []
    for v in range(num_views):
        # Create a simple pattern: gradient + random noise
        img = torch.zeros(3, height, width)
        for c in range(3):
            # Horizontal gradient with slight per-view shift
            shift = v * 0.1
            grad_h = torch.linspace(0, 1, height).view(-1, 1).expand(-1, width)
            grad_w = torch.linspace(0, 1, width).view(1, -1).expand(height, -1)
            img[c] = 0.5 + 0.3 * torch.sin(grad_h * 3 + grad_w * 2 + shift + c)
        # Add some noise
        img += 0.02 * torch.randn_like(img)
        img = img.clamp(0, 1)
        images.append(img)

    return torch.stack(images).unsqueeze(0)  # [1, V, 3, H, W]


@torch.no_grad()
def test_feature_extractor(cfg: WanRestorerCfg, images: torch.Tensor):
    """Test the WAN feature extractor standalone."""
    print(f"\n{'='*60}")
    print(f"Testing WAN Feature Extractor")
    print(f"{'='*60}")

    extractor = WanFeatureExtractor(cfg)

    # Warm up
    print("  Warming up...")
    _ = extractor(images.to(extractor.dtype))

    # Timed run
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    features = extractor(images.to(extractor.dtype))
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    current_mem = torch.cuda.memory_allocated() / 1e9

    print(f"  Input shape:  {tuple(images.shape)}")
    print(f"  Output shape: {tuple(features.shape)}")
    print(f"  Feature dtype: {features.dtype}")
    print(f"  Time:          {elapsed:.3f}s")
    print(f"  Peak VRAM:     {peak_mem:.2f} GB")
    print(f"  Current VRAM:  {current_mem:.2f} GB")
    print(f"  Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"  Feature mean:  {features.mean():.3f}")

    return extractor


@torch.no_grad()
def test_full_restorer(cfg: WanRestorerCfg, images: torch.Tensor):
    """Test the full V3 restorer pipeline."""
    print(f"\n{'='*60}")
    print(f"Testing Full V3 Restorer Pipeline")
    print(f"{'='*60}")

    device = images.device
    b, v, _, h, w = images.shape

    # Create synthetic raw GS params on GPU
    raw_gs = torch.randn(b * v, 83, h, w, device=device) * 0.1

    # Create optional DPT features on GPU
    dpt_feats = torch.randn(b * v, 128, h, w, device=device) * 0.1 if cfg.fuse_with_dpt else None

    # Initialize restorer and move to GPU
    restorer = GaussianSceneRestorerV3(cfg).to(device)

    # Warm up
    print("  Warming up...")
    _ = restorer(images.to(restorer.wan_extractor.dtype), raw_gs, dpt_feats)

    # Timed run
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    start_time = time.time()
    refined = restorer(images.to(restorer.wan_extractor.dtype), raw_gs, dpt_feats)
    torch.cuda.synchronize()
    elapsed = time.time() - start_time

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    current_mem = torch.cuda.memory_allocated() / 1e9

    # Verify zero-init: first forward → refined = raw
    delta = (refined - raw_gs).abs().max().item()

    print(f"  Input images:     {tuple(images.shape)}")
    print(f"  Raw GS params:    {tuple(raw_gs.shape)}")
    print(f"  Refined params:   {tuple(refined.shape)}")
    print(f"  DPT feats:        {dpt_feats.shape if dpt_feats is not None else 'None'}")
    print(f"  Max |refined-raw|: {delta:.6f}")
    print(f"  Total params:     {sum(p.numel() for p in restorer.parameters()):,}")
    print(f"  Wan params:       {sum(p.numel() for p in restorer.wan_extractor.parameters()):,}")
    print(f"  Time:             {elapsed:.3f}s")
    print(f"  Peak VRAM:        {peak_mem:.2f} GB")
    print(f"  Current VRAM:     {current_mem:.2f} GB")

    return restorer


def main():
    parser = argparse.ArgumentParser(description="Test Wan2.2-enhanced GSR")
    parser.add_argument("--model_id", type=str,
                        default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                        help="Wan model ID on HuggingFace")
    parser.add_argument("--image_dir", type=str, default=None,
                        help="Directory with input images")
    parser.add_argument("--num_views", type=int, default=2,
                        help="Number of views")
    parser.add_argument("--height", type=int, default=224,
                        help="Image height")
    parser.add_argument("--width", type=int, default=448,
                        help="Image width")
    parser.add_argument("--num_dit_layers", type=int, default=8,
                        help="Number of DiT layers to run")
    parser.add_argument("--use_dpt", action="store_true",
                        help="Use DPT features (simulated)")
    parser.add_argument("--use_fp16", action="store_true", default=True,
                        help="Use bfloat16 for Wan model")
    parser.add_argument("--skip_extractor", action="store_true",
                        help="Skip feature extractor test")
    parser.add_argument("--skip_restorer", action="store_true",
                        help="Skip full restorer test")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Wan2.2-Enhanced Gaussian Scene Restorer — Inference Test")
    print(f"{'='*60}")
    print(f"  Model:     {args.model_id}")
    print(f"  Views:     {args.num_views}")
    print(f"  Resolution: {args.height}x{args.width}")
    print(f"  DiT layers: {args.num_dit_layers}")
    print(f"  FP16:      {args.use_fp16}")

    # Check CUDA
    if not torch.cuda.is_available():
        print("  ERROR: CUDA not available! This test requires a GPU.")
        return

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    print(f"  GPU:       {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:      {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create test images
    try:
        import numpy as np
    except ImportError:
        np = None

    images = load_test_images(args.image_dir, args.num_views,
                              args.height, args.width)
    images = images.to(device)
    print(f"  Test images: {tuple(images.shape)}")

    # Config
    cfg = WanRestorerCfg(
        enabled=True,
        model_id=args.model_id,
        use_fp16=args.use_fp16,
        num_dit_layers=args.num_dit_layers,
        fuse_with_dpt=args.use_dpt,
        use_input_images=True,
    )

    # Run tests
    if not args.skip_extractor:
        extractor = test_feature_extractor(cfg, images)
        del extractor
        torch.cuda.empty_cache()

    if not args.skip_restorer:
        restorer = test_full_restorer(cfg, images)
        del restorer
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"All tests completed successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
