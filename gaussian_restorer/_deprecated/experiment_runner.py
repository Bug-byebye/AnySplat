"""
Gaussian Scene Restoration — Experiment Runner

Systematic experiments to evaluate different design choices:
1. Fusion strategies (concat_conv, cross_attn, gated_fusion, transformer_decoder)
2. Refiner designs (residual, feature_residual, confidence, multi_stage)
3. Video latent sources (VAE-only, DiT-4, DiT-8)
4. Gaussian feature encoding (none, conv_embed, gp_buffer)

Each experiment reports:
- Timing
- Memory
- Zero-init verification
- Output statistics
- Parameter counts

Usage:
    python post/gaussian_restorer/experiment_runner.py --quick
    python post/gaussian_restorer/experiment_runner.py --full
    python post/gaussian_restorer/experiment_runner.py --fusion cross_attn
"""

import argparse
import os
import sys
import time
from typing import Dict, Any, List, Optional
from dataclasses import replace

import torch
import torch.nn as nn

# Add post/ to path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_script_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from gaussian_restorer.config import (
    GaussianRestorerCfg, VideoPriorCfg, GaussianFeatureCfg,
    FusionCfg, RefinerCfg,
)
from gaussian_restorer.gaussian_scene_restorer import GaussianSceneRestorer


def create_test_data(B=1, V=2, H=224, W=448, device="cuda"):
    """Create synthetic test data."""
    images = torch.rand(B, V, 3, H, W, device=device) * 0.5 + 0.25
    # Structured pattern for more realistic features
    for v in range(V):
        for c in range(3):
            shift = v * 0.3 + c * 0.5
            grad_h = torch.linspace(0, 1, H, device=device).view(-1, 1).expand(-1, W)
            grad_w = torch.linspace(0, 1, W, device=device).view(1, -1).expand(H, -1)
            images[0, v, c] = 0.5 + 0.3 * torch.sin(grad_h * 3 + grad_w * 2 + shift)
    images = images.clamp(0, 1)

    # Raw GS params with realistic distribution
    raw_gs = torch.randn(B * V, 83, H, W, device=device) * 0.05
    # Opacity logit: centered at 0 (default = 0.5 sigmoid)
    raw_gs[:, 0] = torch.randn(B * V, H, W, device=device) * 0.1
    # Scales: log space, small positive centered
    raw_gs[:, 1:4] = torch.randn(B * V, 3, H, W, device=device) * 0.05
    # Rotations: quaternion [w,x,y,z], w biased to 1
    raw_gs[:, 4] = 1.0 + torch.randn(B * V, H, W, device=device) * 0.01
    raw_gs[:, 5:8] = torch.randn(B * V, 3, H, W, device=device) * 0.01
    # SH coefficients: small random
    raw_gs[:, 8:] = torch.randn(B * V, 75, H, W, device=device) * 0.01

    return images, raw_gs


class Experiment:
    """Single experiment tracking."""
    def __init__(self, name: str, config: GaussianRestorerCfg):
        self.name = name
        self.config = config
        self.results = {}

    def run(self, images: torch.Tensor, raw_gs: torch.Tensor):
        """Run the experiment and record results."""
        device = images.device
        print(f"\n{'='*70}")
        print(f"Experiment: {self.name}")
        print(f"{'='*70}")

        # Build model with device placement
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        try:
            model = GaussianSceneRestorer(self.config, device=device).to(device)
        except Exception as e:
            import traceback
            print(f"  BUILD ERROR: {e}")
            traceback.print_exc()
            self.results = {"error": str(e)}
            return self

        build_time = time.time() - t0
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"  Build time:     {build_time:.3f}s")
        print(f"  Total params:   {total_params/1e6:.2f}M")
        print(f"  Trainable:      {trainable_params/1e6:.2f}M")

        # Warm up
        try:
            with torch.no_grad():
                _ = model(images, raw_gs)
            torch.cuda.synchronize()
        except Exception as e:
            import traceback
            print(f"  WARMUP ERROR: {e}")
            traceback.print_exc()
            self.results = {"error": str(e)}
            return self

        # Timed forward (separate from warmup)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()

        with torch.no_grad():
            refined = model(images, raw_gs)

        torch.cuda.synchronize()
        forward_time = time.time() - t0

        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        current_mem = torch.cuda.memory_allocated() / 1e9
        max_delta = (refined - raw_gs).abs().max().item()
        mean_delta = (refined - raw_gs).abs().mean().item()
        refined_range = (refined.min().item(), refined.max().item())

        print(f"  Forward time:   {forward_time:.4f}s")
        print(f"  Peak VRAM:      {peak_mem:.2f} GB")
        print(f"  Current VRAM:   {current_mem:.2f} GB")
        print(f"  Input shape:    {tuple(raw_gs.shape)}")
        print(f"  Output shape:   {tuple(refined.shape)}")
        print(f"  Max |Δ|:        {max_delta:.6f}")
        print(f"  Mean |Δ|:       {mean_delta:.6f}")
        print(f"  Refined range:  [{refined_range[0]:.4f}, {refined_range[1]:.4f}]")
        print(f"  Zero-init?      {'✅' if max_delta < 1e-5 else '❌'}")

        self.results = {
            "build_time": build_time,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "forward_time": forward_time,
            "peak_vram_gb": peak_mem,
            "current_vram_gb": current_mem,
            "max_delta": max_delta,
            "mean_delta": mean_delta,
            "zero_init": max_delta < 1e-5,
            "output_shape": tuple(refined.shape),
        }

        return self


def experiment_quick(device="cuda"):
    """Quick sanity check with default config."""
    print("\n>>> QUICK SANITY CHECK <<<")
    images, raw_gs = create_test_data(device=device)

    cfg = GaussianRestorerCfg(enabled=True)
    cfg.video_prior.num_dit_layers = 4
    cfg.video_prior.wan_feat_dim = 128
    cfg.refiner.mode = "residual"
    cfg.refiner.hidden_dim = 64
    cfg.refiner.num_blocks = 4
    cfg.fusion.strategy = "concat_conv"

    exp = Experiment("quick_v3_baseline", cfg)
    exp.run(images, raw_gs)
    return exp


def experiment_fusion_strategies(device="cuda"):
    """Compare all fusion strategies."""
    print(f"\n{'#'*70}")
    print(f"# FUSION STRATEGY COMPARISON")
    print(f"{'#'*70}")

    images, raw_gs = create_test_data(device=device)
    strategies = ["concat_conv", "cross_attn", "gated_fusion"]

    results = []
    for strategy in strategies:
        cfg = GaussianRestorerCfg(enabled=True)
        cfg.video_prior.num_dit_layers = 4
        cfg.video_prior.wan_feat_dim = 128
        cfg.refiner.mode = "residual"
        cfg.refiner.hidden_dim = 64
        cfg.refiner.num_blocks = 4
        cfg.fusion.strategy = strategy  # type: ignore
        cfg.fusion.hidden_dim = 128

        exp = Experiment(f"fusion_{strategy}", cfg)
        exp.run(images, raw_gs)
        results.append(exp)

    # Summary table
    print(f"\n{'='*70}")
    print(f"FUSION STRATEGY COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Strategy':<20} {'Time(ms)':<12} {'VRAM(GB)':<12} {'Params(M)':<12} {'ZeroInit':<10}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    for exp in results:
        r = exp.results
        if "error" not in r:
            print(f"{exp.name:<20} {r['forward_time']*1000:<12.2f} "
                  f"{r['peak_vram_gb']:<12.2f} {r['total_params']/1e6:<12.2f} "
                  f"{'✅' if r['zero_init'] else '❌'}")
        else:
            print(f"{exp.name:<20} {'ERROR':<12} {'':<12} {'':<12} {'':<10}")

    return results


def experiment_refiner_designs(device="cuda"):
    """Compare all refiner designs."""
    print(f"\n{'#'*70}")
    print(f"# REFINER DESIGN COMPARISON")
    print(f"{'#'*70}")

    images, raw_gs = create_test_data(device=device)
    modes = ["residual", "feature_residual", "confidence", "multi_stage"]

    results = []
    for mode in modes:
        cfg = GaussianRestorerCfg(enabled=True)
        cfg.video_prior.num_dit_layers = 4
        cfg.video_prior.wan_feat_dim = 128
        cfg.refiner.mode = mode  # type: ignore
        cfg.refiner.hidden_dim = 64
        cfg.refiner.num_blocks = 4
        cfg.refiner.use_confidence = (mode == "confidence")
        cfg.fusion.strategy = "concat_conv"

        exp = Experiment(f"refiner_{mode}", cfg)
        exp.run(images, raw_gs)
        results.append(exp)

    print(f"\n{'='*70}")
    print(f"REFINER DESIGN COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Mode':<22} {'Time(ms)':<12} {'VRAM(GB)':<12} {'Params(M)':<12} {'ZeroInit':<10}")
    print(f"{'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    for exp in results:
        r = exp.results
        if "error" not in r:
            print(f"{exp.name:<22} {r['forward_time']*1000:<12.2f} "
                  f"{r['peak_vram_gb']:<12.2f} {r['total_params']/1e6:<12.2f} "
                  f"{'✅' if r['zero_init'] else '❌'}")
        else:
            print(f"{exp.name:<22} {'ERROR':<12} {'':<12} {'':<12} {'':<10}")

    return results


def experiment_video_latent_depth(device="cuda"):
    """Compare different video latent extraction depths."""
    print(f"\n{'#'*70}")
    print(f"# VIDEO LATENT DEPTH COMPARISON")
    print(f"{'#'*70}")

    images, raw_gs = create_test_data(device=device)
    depths = [0, 2, 4, 8]

    results = []
    for depth in depths:
        cfg = GaussianRestorerCfg(enabled=True)
        cfg.video_prior.num_dit_layers = depth
        cfg.video_prior.wan_feat_dim = 128
        cfg.refiner.mode = "residual"
        cfg.refiner.hidden_dim = 64
        cfg.refiner.num_blocks = 4
        cfg.fusion.strategy = "concat_conv"

        name = f"dit_{depth}layers" if depth > 0 else "vae_only"
        exp = Experiment(f"latent_{name}", cfg)
        exp.run(images, raw_gs)
        results.append(exp)

    print(f"\n{'='*70}")
    print(f"VIDEO LATENT DEPTH COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<22} {'Time(ms)':<12} {'VRAM(GB)':<12} {'Params(M)':<12} {'ZeroInit':<10}")
    print(f"{'-'*22} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    for exp in results:
        r = exp.results
        if "error" not in r:
            print(f"{exp.name:<22} {r['forward_time']*1000:<12.2f} "
                  f"{r['peak_vram_gb']:<12.2f} {r['total_params']/1e6:<12.2f} "
                  f"{'✅' if r['zero_init'] else '❌'}")
        else:
            print(f"{exp.name:<22} {'ERROR':<12} {'':<12} {'':<12} {'':<10}")

    return results


def experiment_gaussian_features(device="cuda"):
    """Compare Gaussian feature encoding strategies."""
    print(f"\n{'#'*70}")
    print(f"# GAUSSIAN FEATURE ENCODING COMPARISON")
    print(f"{'#'*70}")

    images, raw_gs = create_test_data(device=device)
    encodings = ["none", "conv_embed", "gp_buffer", "gp_buffer_conv"]

    results = []
    for enc in encodings:
        cfg = GaussianRestorerCfg(enabled=True)
        cfg.video_prior.num_dit_layers = 4
        cfg.video_prior.wan_feat_dim = 128
        cfg.gaussian_features.encode_method = enc  # type: ignore
        cfg.gaussian_features.gs_feat_dim = 83 if enc == "none" else 64
        cfg.refiner.mode = "residual"
        cfg.refiner.hidden_dim = 64
        cfg.refiner.num_blocks = 4
        cfg.fusion.strategy = "concat_conv"

        exp = Experiment(f"gs_feat_{enc}", cfg)
        exp.run(images, raw_gs)
        results.append(exp)

    print(f"\n{'='*70}")
    print(f"GAUSSIAN FEATURE ENCODING COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Encoding':<22} {'Time(ms)':<12} {'VRAM(GB)':<12} {'Params(M)':<12}")
    print(f"{'-'*22} {'-'*12} {'-'*12} {'-'*12}")
    for exp in results:
        r = exp.results
        if "error" not in r:
            print(f"{exp.name:<22} {r['forward_time']*1000:<12.2f} "
                  f"{r['peak_vram_gb']:<12.2f} {r['total_params']/1e6:<12.2f}")
        else:
            print(f"{exp.name:<22} {'ERROR':<12} {'':<12} {'':<12}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Gaussian Restoration Experiments")
    parser.add_argument("--quick", action="store_true", help="Quick sanity check only")
    parser.add_argument("--full", action="store_true", help="Run all experiments")
    parser.add_argument("--fusion", action="store_true", help="Fusion strategies")
    parser.add_argument("--refiner", action="store_true", help="Refiner designs")
    parser.add_argument("--latent", action="store_true", help="Latent depth")
    parser.add_argument("--gs_feat", action="store_true", help="GS feature encoding")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required!")
        return

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} | VRAM: {vram:.1f} GB")

    # Check GPU availability
    free_mem = []
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.get_device_properties(i).total_memory / 1e9
        allocated = torch.cuda.memory_reserved(i) / 1e9
        free_mem.append(mem - allocated)
        print(f"  GPU {i}: {mem:.1f} GB (free: {(mem-allocated):.1f} GB)")

    # Pick GPU with most free memory
    best_gpu = max(range(len(free_mem)), key=lambda i: free_mem[i])
    if best_gpu != 0:
        print(f"Using GPU {best_gpu} (most free memory)")
    torch.cuda.set_device(best_gpu)

    if args.quick or not any([args.full, args.fusion, args.refiner, args.latent, args.gs_feat]):
        experiment_quick(device)

    if args.fusion or args.full:
        experiment_fusion_strategies(device)

    if args.refiner or args.full:
        experiment_refiner_designs(device)

    if args.latent or args.full:
        experiment_video_latent_depth(device)

    if args.gs_feat or args.full:
        experiment_gaussian_features(device)

    print(f"\n{'='*70}")
    print(f"All experiments complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
