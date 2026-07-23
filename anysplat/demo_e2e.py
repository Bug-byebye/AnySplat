"""
End-to-End Demo: Gaussian Scene Restoration

Pipeline:
  1. Load AnySplat pretrained model
  2. Load sample images (context + target)
  3. Run AnySplat encoder → initial Gaussians
  4. Render initial Gaussians → "before" image
  5. Extract Wan2.2 video latents from context images
  6. Apply restorer (refine raw_gs_params with video priors)
  7. Re-run voxelization + GaussianAdapter → refined Gaussians
  8. Render refined Gaussians → "after" image
  9. Compare before/after vs ground truth

Usage:
    cd /home-ldap/sunchang/3dProjects/GuassDiff
    python demo_e2e.py --images examples/vrnerf/riverview/ --out outputs/demo
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import rearrange

# Add project to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'anysplat'))


def load_images(image_dir, num_context=2, max_size=448):
    """Load images from directory, sorted by name."""
    paths = sorted(Path(image_dir).glob("*.jpg")) + \
            sorted(Path(image_dir).glob("*.png"))
    if len(paths) < num_context + 1:
        print(f"Need {num_context+1} images, found {len(paths)}")
        return None, None

    imgs = []
    for p in paths[:num_context + 1]:
        img = Image.open(p).convert("RGB")
        w, h = img.size
        if max(w, h) > max_size:
            scale = max_size / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        # Ensure divisible by 16
        w, h = img.size
        w, h = w - w % 16, h - h % 16
        img = img.crop((0, 0, w, h))
        imgs.append(torch.tensor(np.array(img)).float().permute(2, 0, 1) / 255.0)

    context = torch.stack(imgs[:num_context])  # [V, 3, H, W]
    target = imgs[num_context]  # [3, H, W]
    return context, target


@torch.no_grad()
def run_anysplat(model, context_images, device):
    """Run AnySplat encoder to get Gaussians and intermediate outputs."""
    from src.model.encoder.anysplat import EncoderAnySplat

    model.eval()
    b, v, c, h, w = context_images.shape

    # Run VGGT aggregator
    with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
        aggregated_tokens, patch_start_idx = model.aggregator(
            context_images.to(torch.bfloat16),
            intermediate_layer_idx=model.cfg.intermediate_layer_idx,
        )

    # Camera head
    with torch.amp.autocast("cuda", enabled=False):
        pred_pose_enc = model.camera_head(aggregated_tokens)
        last_pose = pred_pose_enc[-1]
        from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        extrinsic, intrinsic = pose_encoding_to_extri_intri(last_pose, (h, w))

        # Depth head
        depth_map, depth_conf = model.depth_head(
            aggregated_tokens, images=context_images,
            patch_start_idx=patch_start_idx,
        )
        from src.model.encoder.vggt.utils.geometry import (
            batchify_unproject_depth_map_to_point_map,
        )
        pts_all = batchify_unproject_depth_map_to_point_map(
            depth_map, extrinsic, intrinsic
        )

    # DPT GS head → raw params
    out = model.gaussian_param_head(
        aggregated_tokens,
        pts_all.flatten(0, 1).permute(0, 3, 1, 2),
        context_images,
        patch_start_idx=patch_start_idx,
        image_size=(h, w),
    )

    raw_gs_params = out[:, :, :model.raw_gs_dim]  # [B, V, 83, H, W]

    # Get scene info
    neural_feats_list, neural_pts_list = [], []
    pts_flat = pts_all.flatten(2, 3)
    scene_scale = pts_flat.norm(dim=-1).mean().clip(min=1e-8)
    conf = out[:, :, model.raw_gs_dim]

    return {
        "raw_gs_params": raw_gs_params,
        "depth_map": depth_map,
        "pts_all": pts_all,
        "extrinsic": extrinsic,
        "intrinsic": intrinsic,
        "conf": conf,
        "scene_scale": scene_scale,
        "aggregated_tokens": aggregated_tokens,
        "patch_start_idx": patch_start_idx,
        "pred_pose_enc_list": pred_pose_enc,
    }


def render_gaussians(gaussians, decoder, target_extrinsic, target_intrinsic,
                      target_near, target_far, target_size):
    """Render a target view from Gaussians."""
    b = gaussians.means.shape[0]
    h, w = target_size
    out = decoder.forward(
        gaussians,
        target_extrinsic.unsqueeze(0),
        target_intrinsic.unsqueeze(0),
        torch.tensor([target_near]).unsqueeze(0),
        torch.tensor([target_far]).unsqueeze(0),
        (h, w),
    )
    return out.color[0], out.depth[0]


def build_restorer_and_refine(raw_gs_params, context_images, restorer_cfg, device):
    """Apply the restorer to refine raw_gs_params."""
    from gaussian_restorer import GaussianSceneRestorer

    restorer = GaussianSceneRestorer(restorer_cfg, device=device).to(device)
    restorer.eval()

    B, V, C, H, W = context_images.shape
    # Flatten raw_gs_params: [B, V, 83, H, W] → [B*V, 83, H, W]
    flat_params = raw_gs_params.flatten(0, 1)
    refined = restorer(context_images, flat_params)
    # Unflatten back: [B, V, 83, H, W]
    refined = refined.view(B, V, *refined.shape[1:])
    return refined, restorer


def main():
    parser = argparse.ArgumentParser(description="End-to-end GSR demo")
    parser.add_argument("--images", default="examples/vrnerf/riverview",
                        help="Image directory")
    parser.add_argument("--out", default="outputs/demo",
                        help="Output directory")
    parser.add_argument("--num_context", type=int, default=2,
                        help="Number of context views")
    parser.add_argument("--max_size", type=int, default=448,
                        help="Max image dimension")
    parser.add_argument("--steps", type=int, default=0,
                        help="Training steps (0 = zero-shot, no training)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate for training")
    args = parser.parse_args()

    device = torch.device("cuda")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Gaussian Scene Restoration — End-to-End Demo")
    print("=" * 60)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # 1. Load images
    print(f"\n[1] Loading images from {args.images}")
    context, target = load_images(args.images, args.num_context, args.max_size)
    if context is None:
        print("ERROR: Could not load sufficient images.")
        return
    print(f"  Context: {tuple(context.shape)}")
    print(f"  Target:  {tuple(target.shape)}")

    _, _, H, W = context.shape
    context_batch = context.unsqueeze(0).to(device)  # [1, V, 3, H, W]
    target_batch = target.unsqueeze(0).to(device)     # [1, 3, H, W]

    # Normalize for AnySplat ([-1, 1])
    context_norm = context_batch * 2 - 1
    target_norm = target_batch * 2 - 1

    # 2. Load AnySplat model
    print(f"\n[2] Loading AnySplat model...")
    from src.model.model.anysplat import ModelAnySplat
    from src.model.model_wrapper import OptimizerCfg, TestCfg, TrainCfg

    # Load pretrained model config
    import json
    with open(os.path.join(PROJECT_ROOT, "anysplat/pretrained_model/config.json")) as f:
        cfg_dict = json.load(f)

    model = ModelAnySplat(cfg_dict["encoder_cfg"], cfg_dict["decoder_cfg"])
    state = torch.load(os.path.join(PROJECT_ROOT, "anysplat/pretrained_model/model.safetensors"), map_location=device)
    model.load_state_dict(state, strict=False)
    model = model.to(device)
    model.eval()
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    # Get decoder
    decoder = model.decoder

    # 3. Run AnySplat encoder
    print(f"\n[3] Running AnySplat encoder...")
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    enc_out = run_anysplat(model.encoder, context_norm, device)
    raw_gs_params = enc_out["raw_gs_params"]

    print(f"  Raw GS params: {tuple(raw_gs_params.shape)}")
    print(f"  Time: {time.time()-t0:.2f}s")

    # Set up target camera (for simplicity, use a nearby view)
    target_extrinsic = enc_out["extrinsic"][:, 0]  # Use first view as proxy
    target_intrinsic = enc_out["intrinsic"][:, 0]
    # For demo, render the first context view (self-rendering)
    pg_gs_src = enc_out["pts_all"]  # [1, V, 3, H, W]

    # 4. Build initial Gaussians via adapter
    print(f"\n[4] Voxelize and build Gaussians...")
    # We need to go through voxelization + GaussianAdapter
    # This requires the post-processing steps from AnySplat's encoder
    # For the demo, let's use a simplified approach

    # Extract voxel features
    from src.model.encoder.anysplat import EncoderAnySplat
    encoder = model.encoder

    anchor_feats = raw_gs_params[0]  # [V, 83, H, W]
    pts_3d = enc_out["pts_all"][0]   # [V, 3, H, W]
    conf = enc_out["conf"][0]        # [V, H, W]

    # Voxelization
    neural_feats_list, neural_pts_list = [], []
    for b_i in range(1):
        neural_pts, neural_feats = encoder.voxelizaton_with_fusion(
            anchor_feats, pts_3d.permute(0, 2, 3, 1).contiguous(),
            encoder.voxel_size, conf=conf,
        )
        neural_feats_list.append(neural_feats)
        neural_pts_list.append(neural_pts)

    max_voxels = max(f.shape[0] for f in neural_feats_list)
    B = 1
    neural_feats = encoder.pad_tensor_list(neural_feats_list, (max_voxels,), value=-1e10)
    neural_pts = encoder.pad_tensor_list(neural_pts_list, (max_voxels,), -1e4)

    depths = neural_pts[..., -1].unsqueeze(-1)
    densities = neural_feats[..., 0].sigmoid()
    opacity = encoder.map_pdf_to_opacity(densities, global_step=100000)

    # Build initial Gaussians
    gaussians_init = encoder.gaussian_adapter.forward(
        neural_pts, depths, opacity, neural_feats[..., 1:].squeeze(2),
    )

    print(f"  Gaussians: {gaussians_init.means.shape[1]} splats")

    # 5. Render "before" image
    print(f"\n[5] Rendering 'before' (initial Gaussians)...")
    encoder_out = model.encoder(
        context_norm, 100000,
    )
    # Use the built-in encoder forward that includes rendering path
    # For the before case, just use the model's forward
    # (which gives us both Gaussians and rendered output)

    # Actually, let's use the ModelAnySplat forward to get the rendered output
    with torch.no_grad():
        encoder_output, model_output = model(context_norm, 100000)

    before_color = model_output.color[0]  # [V, 3, H, W]
    print(f"  Before color: {tuple(before_color.shape)}")
    print(f"  Before PSNR (vs context, first view): "
          f"{psnr(before_color[0], context_batch[0, 0]):.2f}")

    # 6. Extract Wan2.2 features (if Wan is available)
    print(f"\n[6] Extracting video priors from Wan2.2...")
    try:
        from diffusers import AutoencoderKLWan, WanTransformer3DModel
        WAN_AVAILABLE = True
    except ImportError:
        WAN_AVAILABLE = False
        print("  Wan2.2 not available - using zero features")

    if WAN_AVAILABLE:
        from wan_restorer.config import WanRestorerCfg
        from wan_restorer.wan_feature_extractor import WanFeatureExtractor

        wan_cfg = WanRestorerCfg(
            enabled=True, num_dit_layers=4, use_fp16=True,
            wan_feat_dim=128,
        )
        wan_extractor = WanFeatureExtractor(wan_cfg)
        wan_extractor.eval().to(device)

        t0 = time.time()
        with torch.no_grad():
            video_feats = wan_extractor(context_batch)
        print(f"  Video features: {tuple(video_feats.shape)} [{time.time()-t0:.2f}s]")

    # 7. Build restorer and refine
    print(f"\n[7] Refining with restorer...")
    from gaussian_restorer.config import GaussianRestorerCfg
    from gaussian_restorer.gaussian_scene_restorer import GaussianSceneRestorer

    restorer_cfg = GaussianRestorerCfg(
        enabled=True,
        use_rgb=True,
    )
    restorer_cfg.video_prior.num_dit_layers = 4
    restorer_cfg.fusion.strategy = "concat_conv"
    restorer_cfg.refiner.mode = "residual"
    restorer_cfg.refiner.hidden_dim = 64
    restorer_cfg.refiner.num_blocks = 4

    restorer = GaussianSceneRestorer(restorer_cfg, device=device).to(device)
    restorer.eval()

    flat_params = raw_gs_params.flatten(0, 1)
    with torch.no_grad():
        refined_params = restorer(context_batch, flat_params)

    refined_params = refined_params.view(1, -1, 83, *context_batch.shape[-2:])

    # Re-build Gaussians with refined params
    print(f"\n[8] Re-building Gaussians with refined params...")
    anchor_feats_refined = refined_params[0]
    neural_feats_list_r, neural_pts_list_r = [], []
    for b_i in range(1):
        neural_pts_r, neural_feats_r = encoder.voxelizaton_with_fusion(
            anchor_feats_refined, pts_3d.permute(0, 2, 3, 1).contiguous(),
            encoder.voxel_size, conf=conf,
        )
        neural_feats_list_r.append(neural_feats_r)
        neural_pts_list_r.append(neural_pts_r)

    max_voxels_r = max(f.shape[0] for f in neural_feats_list_r)
    neural_feats_r = encoder.pad_tensor_list(neural_feats_list_r, (max_voxels_r,), value=-1e10)
    neural_pts_r = encoder.pad_tensor_list(neural_pts_list_r, (max_voxels_r,), -1e4)
    depths_r = neural_pts_r[..., -1].unsqueeze(-1)
    densities_r = neural_feats_r[..., 0].sigmoid()
    opacity_r = encoder.map_pdf_to_opacity(densities_r, global_step=100000)

    gaussians_refined = encoder.gaussian_adapter.forward(
        neural_pts_r, depths_r, opacity_r, neural_feats_r[..., 1:].squeeze(2),
    )

    # 9. Render "after" image
    print(f"\n[9] Rendering 'after' (refined Gaussians)...")
    # Render from refined Gaussians at the same views as before
    from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri

    pred_pose = enc_out["pred_pose_enc_list"][-1]
    after_ext, after_int = pose_encoding_to_extri_intri(pred_pose, (H, W))

    after_color_list = []
    for v_idx in range(args.num_context):
        out_v = decoder.forward(
            gaussians_refined,
            after_ext[:, v_idx:v_idx+1],
            after_int[:, v_idx:v_idx+1],
            torch.tensor([[0.1]], device=device),
            torch.tensor([[100.0]], device=device),
            (H, W),
        )
        after_color_list.append(out_v.color[0])

    after_color = torch.stack(after_color_list, dim=1)[0]
    print(f"  After PSNR (vs context, first view): "
          f"{psnr(after_color[0], context_batch[0, 0]):.2f}")

    # 10. Save comparison
    print(f"\n[10] Saving comparison to {out_dir}...")
    save_comparison(context_batch[0], before_color, after_color,
                     target_batch, out_dir)

    print(f"\n{'='*60}")
    print(f"Demo complete! Results saved to {out_dir}")
    print(f"{'='*60}")


def psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def save_comparison(context, before, after, target, out_dir):
    """Save before/after comparison image."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, max(len(context), 2), figsize=(16, 8))

    # Row 1: Context views
    for i in range(len(context)):
        img = context[i].cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"Context View {i}")
        axes[0, i].axis("off")

    # Fill remaining columns
    for i in range(len(context), max(len(context), 2)):
        axes[0, i].axis("off")

    # Row 2: Before / After / Target
    # Before (first view)
    img_before = before[0].cpu().permute(1, 2, 0).numpy()
    img_before = np.clip(img_before, 0, 1)
    axes[1, 0].imshow(img_before)
    axes[1, 0].set_title(f"Before (PSNR: {psnr_before:.1f})" )

    # After (first view)
    img_after = after[0].cpu().permute(1, 2, 0).numpy()
    img_after = np.clip(img_after, 0, 1)
    axes[1, 1].imshow(img_after)
    axes[1, 1].set_title(f"After (PSNR: {psnr_after:.1f})")

    for j in range(2, max(3, len(context))):
        if j < max(3, len(context)):
            axes[1, j].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_dir / "comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Save individual images
    for i in range(len(context)):
        img = (context[i].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(img).save(str(out_dir / f"context_{i}.png"))

    img_before_np = (before[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img_before_np).save(str(out_dir / "before.png"))

    img_after_np = (after[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img_after_np).save(str(out_dir / "after.png"))

    print(f"  Comparison saved to {out_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
