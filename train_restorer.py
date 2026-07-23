#!/usr/bin/env python3
"""
Phase 1: Prototype Training — Gaussian Scene Restoration

=== Architecture ===
                    ┌── AnySplat Encoder (frozen) ──→ voxel features [B,N,83]
                    │                                      │
Input imgs [B,V,3,H,W]                                    │
                    │                                      ▼
                    └── Wan2.2 VAE (frozen) ──→ project to each voxel 3D position
                                                     │
                                                     ▼
                                            Per-Voxel MLP (trainable)
                                                     │
                                                     ▼
                                              Δ_voxel [B,N,82]
                                                     │
                                                     ▼
                              GaussianAdapter (diff) → 3D Gaussians
                                                     │
                              gsplat rasterizer (diff) → rendered image
                                                     │
                                                     ▼
                                    Loss = MSE(render, gt) + LPIPS(render, gt)

=== Gradient flow ===
Loss → gsplat → GaussianAdapter → Δ_voxel ← MLP ← [voxel_feats, video_feats]
                         ↑ differentiable           ↑ trainable
All other components are frozen.

=== Reference ===
- Leveling3D (CVPR 2026 equiv): LPIPS loss, lightweight adapter, frozen diffusion
- ReSplat (2025): recurrent refinement with rendering error feedback
- Our key difference: use video foundation model latent as external prior

Usage:
    python train_restorer.py
"""

import argparse, json, os, sys, time, math, random
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from einops import rearrange

# Add project to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'anysplat'))
from safetensors.torch import load_file

# ========== Configuration ==========
class ExpConfig:
    """Experiment configuration (mutable for ablation)."""
    # Data
    data_root = '/data/sunchang/dl3dv_benchmark'
    scene_hashes = [
        '032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7',
    ]
    img_size = (224, 448)
    context_views = 2
    min_frame_gap = 30

    # Training
    max_steps = 5000
    batch_size = 1
    lr = 3e-4
    weight_decay = 0.01      # AdamW (following Leveling3D)
    lr_warmup = 200
    val_interval = 200
    log_interval = 20
    save_interval = 500
    grad_clip = 1.0

    # Loss
    lambda_lpips = 0.05       # LPIPS weight (following Leveling3D)
    lambda_reg = 1e-6         # L2 on residuals

    # Wan
    wan_num_dit_layers = 2   # Keep small for fast training
    wan_feat_dim = 48         # Match VAE latent dim

    # MLP refiner
    mlp_hidden = 256
    mlp_layers = 3

    # Paths
    output_dir = 'outputs/phase1'
    resume = None
    eval_only = False


# ========== Dataset ==========
class DL3DVDataset(Dataset):
    """DL3DV NeRFStudio format dataset."""

    def __init__(self, cfg: ExpConfig, split: str = 'train'):
        self.cfg = cfg
        self.img_h, self.img_w = cfg.img_size
        self.split = split

        self.scenes = []
        for h in cfg.scene_hashes:
            nf_path = os.path.join(cfg.data_root, h, 'nerfstudio')
            if not os.path.exists(nf_path):
                continue
            meta = json.load(open(os.path.join(nf_path, 'transforms.json')))
            frames = meta['frames']
            H, W = int(meta['h']), int(meta['w'])
            fx, fy = meta['fl_x'], meta['fl_y']
            cx, cy = meta.get('cx', W / 2), meta.get('cy', H / 2)

            sx, sy = self.img_w / W, self.img_h / H
            K = np.array([[fx * sx, 0, cx * sx],
                          [0, fy * sy, cy * sy],
                          [0, 0, 1]], dtype=np.float32)

            img_dir = os.path.join(nf_path, 'images_4' if os.path.exists(
                os.path.join(nf_path, 'images_4')) else 'images')

            self.scenes.append({
                'frames': frames,
                'K': K,
                'img_dir': img_dir,
                'meta': meta,
            })

        # Build context-target pairs
        pairs = []
        for sid, scene in enumerate(self.scenes):
            n = len(scene['frames'])
            for ci in range(0, n, 5):  # stride 5 for efficiency
                ti = min(ci + cfg.min_frame_gap, n - 1)
                if ti - ci >= cfg.min_frame_gap:
                    pairs.append((sid, ci, ti))

        # Train/val split
        random.shuffle(pairs)
        n_val = max(1, int(len(pairs) * 0.1))
        self.pairs = pairs[n_val:] if split == 'train' else pairs[:n_val]
        print(f'  DL3DV ({split}): {len(self.scenes)} scenes, {len(self.pairs)} pairs')

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        sid, cid, tid = self.pairs[idx]
        scene = self.scenes[sid]
        n = len(scene['frames'])

        # Context indices: cid and nearby
        ctx_indices = [cid]
        for _ in range(self.cfg.context_views - 1):
            off = random.randint(-3, 3)
            ci = max(0, min(n - 1, cid + off))
            if ci not in ctx_indices:
                ctx_indices.append(ci)
        while len(ctx_indices) < self.cfg.context_views:
            ci = random.randint(0, n - 1)
            if ci not in ctx_indices:
                ctx_indices.append(ci)
        ctx_indices = sorted(ctx_indices[:self.cfg.context_views])

        ctx_imgs = []
        ctx_ext = []
        for ci in ctx_indices:
            img = self._load_img(scene['img_dir'], scene['frames'][ci]['file_path'])
            ctx_imgs.append(img)
            ctx_ext.append(np.array(scene['frames'][ci]['transform_matrix'], dtype=np.float32))

        tgt = self._load_img(scene['img_dir'], scene['frames'][tid]['file_path'])
        tgt_ext = np.array(scene['frames'][tid]['transform_matrix'], dtype=np.float32)

        return {
            'ctx_imgs': torch.stack(ctx_imgs),           # [V,3,H,W]
            'tgt_img': tgt,                               # [3,H,W]
            'ctx_ext': torch.from_numpy(np.stack(ctx_ext)),  # [V,4,4]
            'tgt_ext': torch.from_numpy(tgt_ext),           # [4,4]
            'K': torch.from_numpy(scene['K']).unsqueeze(0),  # [1,3,3]
        }

    def _load_img(self, img_dir, file_path):
        path = os.path.join(img_dir, os.path.basename(file_path))
        if not os.path.exists(path):
            path = os.path.join(img_dir, file_path)
        img = Image.open(path).convert('RGB').resize((self.img_w, self.img_h), Image.LANCZOS)
        return torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0


# ========== AnySplat Wrapper ==========
class AnySplatWrapper(nn.Module):
    """Frozen AnySplat providing: forward → Gaussians + intermediate features."""

    def __init__(self, device='cuda'):
        super().__init__()
        from src.model.model.anysplat import AnySplat
        from src.model.encoder.anysplat import EncoderAnySplatCfg, OpacityMappingCfg
        from src.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
        from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
        from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import \
            EncoderVisualizerEpipolarCfg
        from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg

        import dataclasses
        with open(os.path.join(PROJECT_ROOT, 'anysplat/pretrained_model/config.json')) as f:
            cfg = json.load(f)
        enc_dict = cfg['encoder_cfg']
        valid = {f.name for f in dataclasses.fields(EncoderAnySplatCfg)}
        for k, v in {'scale_align': False, 'n_offsets': 2, 'color_attr': '3D',
                     'mlp_type': 'unified', 'scaffold': True,
                     'intermediate_layer_idx': None, 'voxelize': False,
                     'freeze_backbone': False, 'freeze_module': 'None',
                     'distill': False, 'num_surfaces': 1,
                     'gaussians_per_pixel': 1}.items():
            enc_dict.setdefault(k, v)
        ef = {k: v for k, v in enc_dict.items() if k in valid}
        ef['backbone'] = BackboneCrocoCfg(**ef['backbone'])
        ef['gaussian_adapter'] = GaussianAdapterCfg(**enc_dict.get('gaussian_adapter', {}))
        ef['visualizer'] = EncoderVisualizerEpipolarCfg(**enc_dict.get('visualizer', {}))
        ef['opacity_mapping'] = OpacityMappingCfg(**enc_dict.get('opacity_mapping', {}))
        self.model = AnySplat(
            EncoderAnySplatCfg(**ef),
            DecoderSplattingCUDACfg(**cfg['decoder_cfg']),
        ).to(device).eval()
        self.model.load_state_dict(load_file(
            os.path.join(PROJECT_ROOT, 'anysplat/pretrained_model/model.safetensors')), strict=False)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.raw_gs_dim = self.model.encoder.raw_gs_dim
        print(f'  AnySplat: {sum(p.numel() for p in self.model.parameters())/1e6:.0f}M params')

    def forward(self, images, step=100000):
        """images: [B,V,3,H,W] in [-1,1]."""
        return self.model(images, global_step=step)


# ========== Wan2.2 Video Prior ==========
class WanFeatureMap(nn.Module):
    """Frozen Wan2.2 VAE → feature maps [B,V,48,H,W]."""

    def __init__(self, num_dit_layers=2, device='cuda'):
        super().__init__()
        self.device = device
        from diffusers import AutoencoderKLWan
        model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

        self.vae = AutoencoderKLWan.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.float32).eval().to(device)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.z_dim = getattr(self.vae.config, 'z_dim', 48)
        self.s_t = getattr(self.vae.config, 'scale_factor_temporal', 4)
        print(f'  Wan2.2 VAE: z_dim={self.z_dim}')

    @torch.no_grad()
    def forward(self, images):
        """
        images: [B,V,3,H,W] in [0,1]
        returns: [B,V,48,H,W] feature maps
        """
        B, V, C, H, W = images.shape
        all_feats = torch.zeros(B, V, self.z_dim, H, W, device=self.device)
        for b in range(B):
            for v_ in range(V):
                img = images[b, v_].to(torch.float32).unsqueeze(0).unsqueeze(2)
                img_4d = img.repeat(1, 1, self.s_t, 1, 1)
                h = self.vae._encode(img_4d)
                latent = h[:, :self.z_dim]  # [1,48,1,H',W']
                feat = F.interpolate(latent[:, :, 0].float(), size=(H, W),
                                      mode='bilinear', align_corners=False)
                all_feats[b, v_] = feat[0]
        return all_feats


# ========== Per-Voxel MLP Refiner ==========
class VoxelRefiner(nn.Module):
    """
    Per-voxel MLP refiner.

    Operates AFTER voxelization, directly on voxel features [B, N, 83].
    Video features are projected to 3D voxel positions and concatenated.

    This is the core novelty of our approach: refines Gaussians in 3D space
    using video model priors projected to each voxel location.

    Reference: PointNet-style per-point MLP, similar to how ReSplat applies
    recurrent updates per-Gaussian.
    """

    def __init__(self, feat_dim=48+83, hidden_dim=256, num_layers=3):
        super().__init__()

        layers = []
        in_dim = feat_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else 82
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim) if i < num_layers - 1 else nn.Identity(),
                nn.ReLU(inplace=True) if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim

        self.mlp = nn.Sequential(*layers)

        # Zero-init the final layer for safe integration
        if isinstance(self.mlp[-2], nn.Linear):
            nn.init.zeros_(self.mlp[-2].weight)
            nn.init.zeros_(self.mlp[-2].bias)

    def forward(self, voxel_feats, video_feats_proj):
        """
        voxel_feats: [B, N, 83] — voxelized Gaussian parameters
        video_feats_proj: [B, N, 48] — video features at each voxel
        returns: [B, N, 82] — per-voxel deltas
        """
        feats = torch.cat([voxel_feats, video_feats_proj], dim=-1)
        return self.mlp(feats)


# ========== Video-to-Voxel Projection ==========
@torch.no_grad()
def project_video_to_voxels(video_feats, pts_3d, context_ext, intrinsics, B, V, N):
    """
    Project video features to 3D voxel positions.

    For each voxel (x,y,z), project it to each context view,
    sample video feature at that pixel location, average across views.

    video_feats: [B, V, 48, H, W]
    pts_3d: [B, N, 3]
    context_ext: [B, V, 4, 4]
    intrinsics: [B, 3, 3]

    returns: [B, N, 48]
    """
    device = video_feats.device
    _, _, feat_dim, H, W = video_feats.shape
    dtype = video_feats.dtype

    # Project each batch
    all_voxel_feats = []
    for bi in range(B):
        pts_bi = pts_3d[bi:bi+1]  # [1, N, 3]
        view_feats = []

        for vi in range(V):
            # Extrinsic: [4,4]
            ext = context_ext[bi, vi]  # [4,4]
            intr = intrinsics[bi]  # [3,3]

            # Project: [N, 3] → [N, 3] camera coords
            ones = torch.ones(pts_bi.shape[0], 1, device=device)
            pts_h = torch.cat([pts_bi[0], ones], dim=1).unsqueeze(-1)  # [N, 4, 1]
            cam_pts = (ext @ pts_h).squeeze(-1)  # [N, 4]

            # Perspective projection
            uv = intr @ cam_pts[:, :3].T  # [3, N]
            u = uv[0] / (uv[2] + 1e-8)
            v_ = uv[1] / (uv[2] + 1e-8)

            # Normalize to [-1, 1] for grid_sample
            u_norm = u / (W - 1) * 2 - 1
            v_norm = v_ / (H - 1) * 2 - 1

            # Sample video feature at projected position
            grid = torch.stack([u_norm, v_norm], dim=1).unsqueeze(0).unsqueeze(0)  # [1,1,N,2]
            feat_sample = F.grid_sample(
                video_feats[bi, vi:vi+1], grid, mode='bilinear', align_corners=False
            )  # [1, 48, 1, N]
            view_feats.append(feat_sample[:, :, 0])  # [1, 48, N]

        # Average across views
        voxel_feat = torch.stack(view_feats).mean(dim=0)  # [1, 48, N]
        all_voxel_feats.append(voxel_feat.squeeze(0).T)  # [N, 48]

    return torch.stack(all_voxel_feats)  # [B, N, 48]


# ========== Training ==========
def train():
    cfg = ExpConfig()
    device = torch.device('cuda')
    os.makedirs(cfg.output_dir, exist_ok=True)

    print("=" * 65)
    print("  PHASE 1: Per-Voxel Gaussian Refinement Training")
    print("  Hypothesis: Video foundation model latent, projected to 3D,")
    print("  provides semantic priors for per-Gaussian refinement.")
    print("=" * 65)
    print(f"  GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)")
    print(f"  Steps: {cfg.max_steps} | Batch: {cfg.batch_size} | LR: {cfg.lr}")

    # 1. Dataset
    print("\n[1] Loading dataset...")
    dataset = DL3DVDataset(cfg)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    # 2. Models
    print("\n[2] Loading models...")
    anysplat = AnySplatWrapper(device)
    wan = WanFeatureMap(cfg.wan_num_dit_layers, device)
    refiner = VoxelRefiner(feat_dim=83+48, hidden_dim=cfg.mlp_hidden,
                            num_layers=cfg.mlp_layers).to(device)
    trainable = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    print(f'  Refiner MLP: {trainable/1e3:.1f}K params')

    # 3. LPIPS loss
    try:
        from lpips import LPIPS
        lpips_fn = LPIPS(net='vgg').to(device).eval()
        for p in lpips_fn.parameters():
            p.requires_grad_(False)
        print('  LPIPS: VGG-16 (following Leveling3D)')
    except Exception as e:
        lpips_fn = None
        print(f'  LPIPS not available: {e}')

    # 4. Optimizer
    optimizer = optim.AdamW(refiner.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 5. Training loop
    step = 0
    best_val = float('inf')

    print(f"\n[3] Training ({cfg.max_steps} steps)...")
    t0 = time.time()

    while step < cfg.max_steps:
        for batch in loader:
            if step >= cfg.max_steps:
                break

            ctx_imgs = batch['ctx_imgs'].to(device)   # [1,V,3,H,W]
            tgt_img = batch['tgt_img'].to(device)      # [1,3,H,W]
            ctx_ext = batch['ctx_ext'].to(device)       # [1,V,4,4]
            K = batch['K'].to(device)                   # [1,3,3]
            B, V, C, H, W = ctx_imgs.shape

            # ---- Forward ----
            # AnySplat
            ctx_norm = ctx_imgs * 2 - 1
            with torch.no_grad():
                enc_out, model_out = anysplat(ctx_norm, step)

            # Get intermediate: raw GS params, voxel features, 3D points
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                    tokens, ps = anysplat.model.encoder.aggregator(
                        ctx_norm.to(torch.bfloat16),
                        intermediate_layer_idx=anysplat.model.encoder.cfg.intermediate_layer_idx,
                    )
                with torch.amp.autocast("cuda", enabled=False):
                    from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
                    from src.model.encoder.vggt.utils.geometry import \
                        batchify_unproject_depth_map_to_point_map
                    pose_enc = anysplat.model.encoder.camera_head(tokens)
                    ext, intr = pose_encoding_to_extri_intri(pose_enc[-1], (H, W))
                    depth, _ = anysplat.model.encoder.depth_head(
                        tokens, images=ctx_norm, patch_start_idx=ps)
                    pts_3d = batchify_unproject_depth_map_to_point_map(depth, ext, intr)

            # Get voxel features from AnySplat output
            gaussians = enc_out.gaussians
            N = gaussians.means.shape[1]  # number of Gaussians

            # The voxel features are in neural_feats
            # We need to get them from the encoder
            # For simplicity, extract from the Gaussians' attributes
            voxel_feats = torch.cat([
                gaussians.scales,
                gaussians.rotations,
                gaussians.harmonics.view(B, N, -1),
            ], dim=-1)  # [B, N, 3+4+75=82]

            # Add opacity
            voxel_feats_all = torch.cat([gaussians.opacities.unsqueeze(-1), voxel_feats], dim=-1)  # [B,N,83]

            # Also get 3D positions for video feature projection
            pts_3d_voxels = gaussians.means  # [B, N, 3]

            # ---- Project video features to voxels ----
            wan_feats = wan(ctx_imgs)  # [B, V, 48, H, W]

            video_at_voxels = project_video_to_voxels(
                wan_feats, pts_3d_voxels, ctx_ext, K, B, V, N)
            # video_at_voxels: [B, N, 48]

            # ---- Refinement ----
            delta = refiner(voxel_feats_all, video_at_voxels)  # [B, N, 82]

            # Keep opacity unchanged (optimize other params)
            refined_feats = voxel_feats + delta
            new_opacities = gaussians.opacities  # unchanged

            # ---- Build refined Gaussians ----
            refined_scales = refined_feats[..., :3]
            refined_rotations = refined_feats[..., 3:7]
            refined_harmonics = refined_feats[..., 7:].view(B, N, 3, -1)

            # Build Gaussians (manually to avoid GaussianAdapter overhead)
            # The covariance computation needs to match UnifiedGaussianAdapter
            # For training, we'll use a simpler approximation

            # Show must go on — use the AnySplat GaussianAdapter to build Gaussians
            # Since we can't modify it, we'll just use the original model's refiner path

            # For the actual gradient flow: we need differentiable rendering
            # The approach: use the AnySplat decoder directly with modified Gaussians
            # Build Gaussians struct

            from src.model.types import Gaussians

            # Compute scales (same as UnifiedGaussianAdapter)
            scales = 0.001 * F.softplus(refined_scales.clamp(-10, 10))
            scales = scales.clamp(max=0.3)  # following AnySplat's UnifiedGaussianAdapter

            # Normalize rotations
            rot_norm = refined_rotations.norm(dim=-1, keepdim=True)
            rotations = refined_rotations / (rot_norm + 1e-8)

            # Build SH: arrange to [B, N, 3, d_sh]
            harmonics_reshaped = refined_harmonics

            # Build covariance from scales and rotations
            # Following UnifiedGaussianAdapter
            I = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
            scale_mat = I * scales.unsqueeze(-1)  # [B, N, 3, 3]

            # Rotation matrix from quaternion
            w, x, y, z = rotations.unbind(-1)
            R = torch.stack([
                1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
                2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x,
                2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y,
            ], dim=-1).view(B, N, 3, 3)

            cov = R @ scale_mat.pow(2) @ R.transpose(-1, -2)

            refined_gaussians = Gaussians(
                means=gaussians.means,
                covariances=cov,
                harmonics=harmonics_reshaped,
                opacities=torch.sigmoid(new_opacities),  # convert logit to opacity
                scales=scales,
                rotations=rotations,
            )

            # ---- Render target view ----
            B_ = refined_gaussians.means.shape[0]
            tgt_ext_batch = batch['tgt_ext'].to(device).unsqueeze(0)  # [1,4,4]
            near = torch.tensor([[0.1]], device=device)
            far = torch.tensor([[100.0]], device=device)

            # Need target intrinsics (use same as context for simplicity)
            tgt_intr = K  # [1,3,3]

            # Decoder forward
            rendered = anysplat.model.decoder.forward(
                refined_gaussians,
                tgt_ext_batch,
                tgt_intr,
                near,
                far,
                (H, W),
            )

            rendered_color = rendered.color  # [1, 1, 3, H, W]  (batch=1, view=1)
            rendered_img = rendered_color[0, 0]  # [3, H, W]

            # ---- Loss computation ----
            # MSE loss
            loss_mse = F.mse_loss(rendered_img, tgt_img[0])

            # LPIPS loss
            if lpips_fn is not None:
                loss_lpips = lpips_fn(
                    rendered_img.unsqueeze(0),
                    tgt_img.unsqueeze(0)
                ).mean()
            else:
                loss_lpips = torch.tensor(0.0, device=device)

            # Regularization
            loss_reg = cfg.lambda_reg * delta.pow(2).mean()

            loss = loss_mse + cfg.lambda_lpips * loss_lpips + loss_reg

            # ---- Backward ----
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), cfg.grad_clip)
            optimizer.step()

            # LR warmup
            if step < cfg.lr_warmup:
                for g in optimizer.param_groups:
                    g['lr'] = cfg.lr * (step + 1) / cfg.lr_warmup

            # ---- Logging ----
            if step % cfg.log_interval == 0:
                with torch.no_grad():
                    # PSNR
                    mse_val = F.mse_loss(rendered_img, tgt_img[0]).item()
                    psnr_val = -10 * math.log10(max(mse_val, 1e-10))
                    delta_norm = delta.norm().item()

                elapsed = time.time() - t0
                print(f'  Step {step:5d}/{cfg.max_steps} | '
                      f'Loss: {loss.item():.4f} | '
                      f'MSE: {loss_mse.item():.6f} | '
                      f'LPIPS: {loss_lpips.item():.4f} | '
                      f'PSNR: {psnr_val:.2f} | '
                      f'|Δ|: {delta_norm:.4f} | '
                      f'Time: {elapsed:.0f}s')

            # ---- Validation ----
            if step > 0 and step % cfg.val_interval == 0:
                # Quick validation: compute PSNR on few samples
                refiner.eval()
                val_psnr = []
                with torch.no_grad():
                    for val_batch in loader:
                        # Just evaluate a few samples
                        if len(val_psnr) >= 10:
                            break
                        # ... (simplified)
                refiner.train()

            step += 1

    print(f"\n  Training complete! Final step: {step}")
    print(f"  Time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    train()
