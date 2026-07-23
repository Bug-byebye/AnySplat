#!/usr/bin/env python3
"""
Phase 1 v3: Gaussian Repair Network — Delete, Move, and Add Primitives
======================================================================

Extends the Phase 1 v2 training with a full GaussianRepairNetwork that can:
1. DELETE: Prune artifact-causing Gaussians (via keep_prob head)
2. MOVE:  Correct Gaussian positions (via Δ_means head)
3. ADD:   Generate new Gaussians to fill holes (via SceneDeficiencyAnalyzer)

Architecture:
  Input Images → AnySplat (frozen) → Initial Gaussians G₀
                                       │
                    ┌────────────────────┴────────────────────┐
                    │  Render G₀ at context views (no_grad)    │
                    │  → ctx_color, ctx_depth, ctx_alpha       │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │   GaussianRepairNetwork                  │
                    │  ├── PerGaussianAnalysisHead             │
                    │  │   → Δ_params [N,82]                   │
                    │  │   → Δ_means  [N,3]                    │
                    │  │   → keep_logit [N,1]                  │
                    │  └── SceneDeficiencyAnalyzer             │
                    │      → deficiency_map + new Gaussians    │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │  Apply Δ to G₀ → G₁ (refined)           │
                    │  Add new Gaussians → G_new              │
                    │  Merge: G_all = G₁ ∪ G_new              │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │  Render G_all at target view → Loss      │
                    └─────────────────────────────────────────┘

Usage:
  CUDA_VISIBLE_DEVICES=2 python3 phase1_train_repairer.py
"""

import os, sys, json, time, math, random, dataclasses, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file

PROJECT_ROOT = '/home-ldap/sunchang/3dProjects/GuassDiff'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'anysplat'))
device = 'cuda:0'

# ====== Config ======
B, V, H, W = 1, 2, 224, 448
MAX_STEPS = 5000
LR = 3e-4
WD = 0.01
LOG_INT = 10
SAVE_INT = 500
VAL_INT = 200
TGT_GAP_MIN = 8
TGT_GAP_MAX = 15

# Repair loss weights
W_PARAMS = 1e-6       # parameter residual regularization
W_MOVE   = 3e-4       # position movement regularization
W_DELETE = 1e-3       # deletion sparsity weight
W_ADD    = 0.01       # addition deficiency BCE weight
W_LPIPS  = 0.05       # LPIPS weight

# Addition budget
N_NEW_GAUSSIANS = 512   # maximum new Gaussians to add

OUT_DIR = Path('/data/sunchang/exp_train/phase1_v3_repairer')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR = OUT_DIR / 'results'
RESULT_DIR.mkdir(exist_ok=True)

# DL3DV scene
SCENE = '/data/sunchang/dl3dv_benchmark/032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7/nerfstudio'

# ====== Data ======
print('[Data] Loading DL3DV...')
with open(os.path.join(SCENE, 'transforms.json')) as f:
    meta = json.load(f)
frames = meta['frames']
img_dir = os.path.join(SCENE, 'images_4')
H_full, W_full = int(meta['h']), int(meta['w'])
fx, fy = meta['fl_x'], meta['fl_y']
cx, cy = meta.get('cx', W_full/2), meta.get('cy', H_full/2)
sx, sy = W / W_full, H / W_full
K_orig = np.array([[fx*sx, 0, cx*sx], [0, fy*sy, cy*sy], [0, 0, 1]], dtype=np.float32)

n_total = len(frames)
print(f'  {n_total} frames')

def load_img(idx):
    fp = frames[idx]['file_path']
    path = os.path.join(img_dir, os.path.basename(fp))
    if not os.path.exists(path):
        path = os.path.join(img_dir, fp)
    img = Image.open(path).convert('RGB').resize((W, H), Image.LANCZOS)
    return torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0

def get_ext(idx):
    return torch.tensor(frames[idx]['transform_matrix'], dtype=torch.float32)  # c2w

# Build training pairs
pairs = []
for ci in range(0, n_total - 20, 5):
    for gap in range(TGT_GAP_MIN, TGT_GAP_MAX + 1):
        ti = ci + gap
        if ti < n_total:
            pairs.append((ci, ti))
random.shuffle(pairs)
n_val = max(1, int(len(pairs) * 0.02))
train_pairs = pairs[n_val:]
val_pairs = pairs[:n_val]
print(f'  {len(train_pairs)} train, {len(val_pairs)} val pairs')

# ====== Models ======
print('\n[Model] Loading AnySplat...')
t0 = time.time()
from src.model.model.anysplat import AnySplat
from src.model.encoder.anysplat import EncoderAnySplatCfg, OpacityMappingCfg
from src.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg, DecoderSplattingCUDA
from src.model.types import Gaussians

with open(os.path.join(PROJECT_ROOT, 'anysplat/pretrained_model/config.json')) as f:
    cfg = json.load(f)
enc_dict = cfg['encoder_cfg']
valid = {f.name for f in dataclasses.fields(EncoderAnySplatCfg)}
defaults = {'scale_align':False,'n_offsets':2,'color_attr':'3D','mlp_type':'unified','scaffold':True,
    'intermediate_layer_idx':None,'voxelize':False,'freeze_backbone':False,'freeze_module':'None',
    'distill':False,'num_surfaces':1,'gaussians_per_pixel':1}
for k,v in defaults.items():
    enc_dict.setdefault(k,v)
ef = {k:v for k,v in enc_dict.items() if k in valid}
ef['backbone'] = BackboneCrocoCfg(**ef['backbone'])
ef['gaussian_adapter'] = GaussianAdapterCfg(**enc_dict.get('gaussian_adapter', {}))
ef['visualizer'] = EncoderVisualizerEpipolarCfg(**enc_dict.get('visualizer', {}))
ef['opacity_mapping'] = OpacityMappingCfg(**enc_dict.get('opacity_mapping', {}))

model = AnySplat(EncoderAnySplatCfg(**ef), DecoderSplattingCUDACfg(**cfg['decoder_cfg'])).to(device)
model.load_state_dict(load_file(os.path.join(PROJECT_ROOT, 'anysplat/pretrained_model/model.safetensors')),
                      strict=False)
model.eval()
for p in model.parameters(): p.requires_grad_(False)
print(f'  Loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M ({time.time()-t0:.1f}s)')

# Decoder for rendering
decoder = DecoderSplattingCUDA(
    DecoderSplattingCUDACfg(name='splatting_cuda', background_color=[1, 1, 1],
                            make_scale_invariant=False)
).to(device)

print('[Model] Loading Wan2.2 VAE...')
t0 = time.time()
from diffusers import AutoencoderKLWan
vae = AutoencoderKLWan.from_pretrained(
    'Wan-AI/Wan2.2-TI2V-5B-Diffusers',
    subfolder='vae', torch_dtype=torch.float32).eval().to(device)
for p in vae.parameters(): p.requires_grad_(False)
z_dim = getattr(vae.config, 'z_dim', 48)
s_t = getattr(vae.config, 'scale_factor_temporal', 4)
print(f'  VAE z_dim={z_dim} ({time.time()-t0:.1f}s)')

try:
    from lpips import LPIPS
    lpips_fn = LPIPS(net='vgg').to(device).eval()
    for p in lpips_fn.parameters(): p.requires_grad_(False)
    print('[Model] LPIPS: loaded')
except:
    lpips_fn = None
    print('[Model] LPIPS: N/A')

# ====== Gaussian Repair Network ======
print('[Model] Building GaussianRepairNetwork...')
from gaussian_restorer.repairer import GaussianRepairNetwork
from gaussian_restorer.repair_loss import RepairLoss

repairer = GaussianRepairNetwork(
    per_gaussian_dim=134,   # 83(gaussian) + 3(mean) + 48(video_latent)
    hidden=256,
    n_new_queries=N_NEW_GAUSSIANS,
    enable_addition=True,
).to(device)
repairer.print_summary()

opt = torch.optim.AdamW(repairer.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEPS)

loss_fn = RepairLoss(
    weight_params=W_PARAMS,
    weight_move=W_MOVE,
    weight_delete=W_DELETE,
    weight_add=W_ADD,
    weight_lpips=W_LPIPS,
    lpips_fn=lpips_fn,
).to(device)

# ====== Helper Functions ======

@torch.no_grad()
def extract_wan_features(ctx_imgs):
    """Extract Wan2.2 VAE features from context images."""
    B, V, _, H, W = ctx_imgs.shape
    wan = torch.zeros(B, V, z_dim, H, W, device=device)
    for vi in range(V):
        img = ctx_imgs[0, vi].float().unsqueeze(0).unsqueeze(2)
        img_4d = img.repeat(1, 1, s_t, 1, 1)
        h = vae._encode(img_4d)
        lat = h[:, :z_dim]
        feat = F.interpolate(lat[:, :, 0].float(), size=(H, W),
                             mode='bilinear', align_corners=False)
        wan[0, vi] = feat[0]
    return wan


def build_rotation_matrix(rotations):
    """Convert quaternions [w,x,y,z] to 3x3 rotation matrices."""
    w, x, y, z = rotations.unbind(-1)
    R = torch.stack([
        1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y,
        2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x,
        2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y,
    ], dim=-1).view(*rotations.shape[:-1], 3, 3)
    return R


@torch.no_grad()
def render_gaussians(g, extrinsics, intrinsics, H, W, return_alpha=True):
    """Render Gaussians and return color, depth, alpha."""
    out = decoder.forward(
        g, extrinsics, intrinsics,
        torch.tensor([[0.1]], device=device),
        torch.tensor([[100.0]], device=device),
        (H, W),
    )
    if return_alpha:
        return out.color, out.depth, out.alpha
    return out.color, out.depth


def normalize_intrinsics(intr):
    """Normalize intrinsics as AnySplat encoder expects."""
    intr_n = intr.clone()
    intr_n[..., 0, :] = intr[..., 0, :] / W
    intr_n[..., 1, :] = intr[..., 1, :] / H
    return intr_n


def save_rendered_image(color_tensor, path):
    """Save a [3,H,W] tensor as an image."""
    img = color_tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


# ====== Train ======
print(f'\n{"="*60}')
print(f'  PHASE 1 V3 — Gaussian Repair Network')
print(f'  Capabilities: Delete (prune) + Move (Δ_means) + Add (deficiency fill)')
print(f'  Data: {len(train_pairs)} train, {len(val_pairs)} val pairs')
print(f'  Output: {OUT_DIR}')
print(f'{"="*60}\n')

step = 0
t_start = time.time()
K_tensor = torch.from_numpy(K_orig).to(device)

# Run baseline (step 0) and save initial state
print('[Init] Running baseline AnySplat and saving initial state...')
with torch.no_grad():
    base_ci, base_ti = train_pairs[0]
    ctx_base = torch.stack([load_img(base_ci), load_img(base_ci + 2)]).unsqueeze(0).to(device)
    ctx_norm = ctx_base * 2 - 1
    enc_out, _ = model(ctx_norm, 100000)
    g_init = enc_out.gaussians
    N_init = g_init.means.shape[1]
    print(f'  Initial Gaussians: {N_init}')

    # Get camera from AnySplat encoder
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        tokens, ps = model.encoder.aggregator(
            ctx_norm.to(torch.bfloat16),
            intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
    with torch.amp.autocast('cuda', enabled=False):
        from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        pose_enc = model.encoder.camera_head(tokens)
        ext_pred, intr_pred = pose_encoding_to_extri_intri(pose_enc[-1], (H, W))

    # Save initial renders
    bottom = torch.tensor([0,0,0,1], device=device).reshape(1,1,1,4).expand(1, V, -1, -1)
    ext_c2w = torch.cat([ext_pred, bottom], dim=2).inverse()
    intr_n = normalize_intrinsics(intr_pred)

    out0 = decoder.forward(g_init, ext_c2w, intr_n,
        torch.tensor([[0.1]], device=device), torch.tensor([[100.0]], device=device), (H, W))
    for vi in range(V):
        save_rendered_image(out0.color[0, vi], RESULT_DIR / 'step_0' / f'init_ctx_v{vi}.png')

    # Target view initial render
    tgt_ext = get_ext(base_ti).to(device).unsqueeze(0).unsqueeze(0)
    out_tgt = decoder.forward(g_init, tgt_ext, intr_n[:, 0:1],
        torch.tensor([[0.1]], device=device), torch.tensor([[100.0]], device=device), (H, W))
    save_rendered_image(out_tgt.color[0, 0], RESULT_DIR / 'step_0' / 'init_target.png')

print('  Baseline saved. Starting training...')

# Training loop
while step < MAX_STEPS:
    random.shuffle(train_pairs)

    for ci, ti in train_pairs:
        if step >= MAX_STEPS:
            break

        # ===== Load data =====
        ctx_indices = sorted(set([
            ci,
            min(n_total - 1, max(0, ci + random.randint(-3, 3))),
        ]))[:V]
        while len(ctx_indices) < V:
            ci2 = random.randint(0, n_total - 1)
            if ci2 not in ctx_indices:
                ctx_indices.append(ci2)
        ctx_indices = sorted(ctx_indices[:V])

        ctx_imgs = torch.stack([load_img(i) for i in ctx_indices]).unsqueeze(0).to(device)
        tgt_img = load_img(ti).unsqueeze(0).to(device)
        ctx_norm = ctx_imgs * 2 - 1

        # ===== AnySplat forward (frozen) =====
        with torch.no_grad():
            enc_out, _ = model(ctx_norm, global_step=100000)
            g = enc_out.gaussians
            N = g.means.shape[1]
            if N == 0:
                print(f'  WARNING: No Gaussians generated at step {step}, skipping')
                step += 1
                continue

            # Get camera
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                tokens, ps = model.encoder.aggregator(
                    ctx_norm.to(torch.bfloat16),
                    intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx)
            with torch.amp.autocast('cuda', enabled=False):
                pose_enc = model.encoder.camera_head(tokens)
                ext, intr = pose_encoding_to_extri_intri(pose_enc[-1], (H, W))

            # Wan features
            wan_feats = extract_wan_features(ctx_imgs)

            # GT cameras for context views
            ctx_gt_ext = torch.stack([get_ext(i) for i in ctx_indices]).unsqueeze(0).to(device)

        # ===== Render initial Gaussians at context views (for deficiency analysis) =====
        with torch.no_grad():
            ctx_render = decoder.forward(g, ctx_gt_ext, intr[:, 0:1].expand(-1, V, -1, -1),
                torch.tensor([[0.1]], device=device).expand(B, V),
                torch.tensor([[100.0]], device=device).expand(B, V),
                (H, W))
            ctx_color = ctx_render.color        # [1, V, 3, H, W]
            ctx_depth = ctx_render.depth         # [1, V, H, W]
            ctx_alpha = ctx_render.alpha         # [1, V, H, W]

        # ===== Build per-Gaussian features =====
        voxel_feats = torch.cat([
            g.opacities.unsqueeze(-1),
            g.scales,
            g.rotations,
            g.harmonics.view(B, N, -1),
        ], dim=-1)  # [B, N, 83]

        # ===== Gaussian Repair Network forward =====
        outputs = repairer(
            gaussian_params=voxel_feats,
            gaussian_means=g.means,
            video_latents=wan_feats,
            extrinsic=ext,
            intrinsic=intr,
            ctx_rendered_images=ctx_color,
            ctx_rendered_depth=ctx_depth.unsqueeze(2),  # [1, V, 1, H, W]
            ctx_rendered_opacity=ctx_alpha.unsqueeze(2),  # [1, V, 1, H, W]
            ctx_gt_images=ctx_imgs,
        )

        delta_params = outputs['delta_params']    # [B, N, 82]
        delta_means = outputs['delta_means']      # [B, N, 3]
        keep_logit = outputs['keep_logit']        # [B, N, 1]
        keep_prob = torch.sigmoid(keep_logit)     # [B, N, 1]

        # ===== Apply refinement to existing Gaussians =====
        # 1. Move means
        refined_means = g.means + delta_means

        # 2. Adjust scales
        refined_scales = 0.001 * F.softplus(
            (voxel_feats[:, :, 1:4] + delta_params[:, :, :3]).clamp(-10, 10)
        ).clamp(max=0.3)

        # 3. Adjust rotations (normalize quaternion)
        rot_raw = voxel_feats[:, :, 4:8] + delta_params[:, :, 3:7]
        rot_norm = rot_raw.norm(dim=-1, keepdim=True)
        refined_rotations = rot_raw / (rot_norm + 1e-8)

        # 4. Adjust SH coefficients
        refined_harmonics = (voxel_feats[:, :, 8:] + delta_params[:, :, 7:]).view(B, N, 3, -1)

        # 5. Adjust opacity and apply deletion (keep_prob)
        raw_opacity = torch.sigmoid(voxel_feats[:, :, 0:1] + delta_params[:, :, 0:1])
        refined_opacities = (raw_opacity * keep_prob).squeeze(-1)  # [B, N]

        # 6. Build covariance matrices
        R = build_rotation_matrix(refined_rotations)  # [B, N, 3, 3]
        I = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
        refined_covariances = R @ (I * refined_scales.unsqueeze(-1)).pow(2) @ R.transpose(-1, -2)

        # ===== Add new Gaussians from deficiency analyzer =====
        new_means_list = outputs.get('new_means', [])
        new_scales_list = outputs.get('new_scales', [])
        new_rotations_list = outputs.get('new_rotations', [])
        new_harmonics_list = outputs.get('new_harmonics', [])
        new_opacities_list = outputs.get('new_opacities', [])

        has_new_gaussians = (len(new_means_list) > 0 and
                            len(new_means_list[0]) > 0)

        if has_new_gaussians:
            n_new = new_means_list[0].shape[0]
            # We have [B=1], expand to batch dim
            new_means = new_means_list[0].unsqueeze(0)
            new_scales = new_scales_list[0].unsqueeze(0)
            new_rotations = new_rotations_list[0].unsqueeze(0)
            new_harmonics = new_harmonics_list[0].unsqueeze(0)
            new_opacities = new_opacities_list[0].unsqueeze(0)

            # Build covariances for new Gaussians
            R_new = build_rotation_matrix(new_rotations)
            I_new = torch.eye(3, device=device).unsqueeze(0).unsqueeze(0)
            new_covariances = R_new @ (I_new * new_scales.unsqueeze(-1)).pow(2) @ R_new.transpose(-1, -2)

            # Concatenate refined + new
            combined_means = torch.cat([refined_means, new_means], dim=1)
            combined_covariances = torch.cat([refined_covariances, new_covariances], dim=1)
            combined_harmonics = torch.cat([refined_harmonics, new_harmonics], dim=1)
            combined_opacities = torch.cat([refined_opacities, new_opacities.squeeze(-1)], dim=1)
            combined_scales = torch.cat([refined_scales, new_scales], dim=1)
            combined_rotations = torch.cat([refined_rotations, new_rotations], dim=1)

            n_new = new_means.shape[1]
            n_total_gaussians = N + n_new
        else:
            combined_means = refined_means
            combined_covariances = refined_covariances
            combined_harmonics = refined_harmonics
            combined_opacities = refined_opacities
            combined_scales = refined_scales
            combined_rotations = refined_rotations
            n_new = 0
            n_total_gaussians = N

        # ===== Build Gaussians object =====
        refined_g = Gaussians(
            means=combined_means,
            covariances=combined_covariances,
            harmonics=combined_harmonics,
            opacities=combined_opacities,
            scales=combined_scales,
            rotations=combined_rotations,
        )

        # ===== Render at target view =====
        tgt_ext = get_ext(ti).to(device).unsqueeze(0).unsqueeze(0)
        tgt_intr = intr[:, 0:1]  # Use predicted intrinsics (same calibration)
        tgt_out = decoder.forward(
            refined_g, tgt_ext, tgt_intr,
            torch.tensor([[0.1]], device=device),
            torch.tensor([[100.0]], device=device),
            (H, W),
        )
        rendered_color = tgt_out.color[0, 0]  # [3, H, W]

        # ===== Compute loss =====
        loss_dict = loss_fn(
            rendered_color=rendered_color,
            target_image=tgt_img[0],
            delta_params=delta_params,
            delta_means=delta_means,
            keep_logit=keep_logit,
            deficiency_map=outputs.get('deficiency_map'),
            new_gauss_params=outputs.get('new_gauss_params'),
            ctx_rendered=ctx_color,
            ctx_gt=ctx_imgs,
        )

        loss = loss_dict['total']

        # ===== Backward =====
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(repairer.parameters(), 1.0)
        opt.step()
        sched.step()

        # ===== Logging =====
        if step % LOG_INT == 0:
            psnr = -10 * math.log10(max(loss_dict['render_mse'].item(), 1e-10))
            lr_now = opt.param_groups[0]['lr']
            keep_mean = keep_prob.mean().item()
            delta_norm = delta_params.norm().item()
            move_norm = delta_means.norm().item()
            deferred_str = ''
            if deficiency_map is not None:
                def_mean = outputs['deficiency_map'].mean().item()
                deferred_str += f' | def={def_mean:.3f}'
            if has_new_gaussians:
                deferred_str += f' | +{n_new}G'
            print(f'  Step {step:5d}/{MAX_STEPS} | '
                  f'PSNR: {psnr:.2f} | '
                  f'|Δ|={delta_norm:.2f} |Δμ|={move_norm:.4f} '
                  f'keep={keep_mean:.3f}'
                  f'{deferred_str} | LR: {lr_now:.2e}')

        # ===== Save intermediate results =====
        if step > 0 and step % SAVE_INT == 0:
            with torch.no_grad():
                step_dir = RESULT_DIR / f'step_{step}'
                step_dir.mkdir(exist_ok=True)

                # Save refined render at target
                save_rendered_image(rendered_color, step_dir / 'refined_target.png')

                # Save renders at context views
                ctx_refined = decoder.forward(refined_g, ctx_gt_ext,
                    intr[:, 0:1].expand(-1, V, -1, -1),
                    torch.tensor([[0.1]], device=device).expand(B, V),
                    torch.tensor([[100.0]], device=device).expand(B, V),
                    (H, W))
                for vi in range(V):
                    save_rendered_image(ctx_refined.color[0, vi],
                                        step_dir / f'refined_ctx_v{vi}.png')

                # Save delta norm and keep prob maps
                delta_map = delta_params.norm(dim=-1).squeeze().cpu().numpy()
                np.save(str(step_dir / 'delta_norm.npy'), delta_map)
                keep_map = keep_prob.squeeze().cpu().numpy()
                np.save(str(step_dir / 'keep_prob.npy'), keep_map)

                # Log info
                n_deleted = int((N * (1 - keep_mean)).item())
                print(f'  ** Saved step {step}: '
                      f'N={N}→{n_total_gaussians} '
                      f'(kept ≈{int(N*keep_mean)}, '
                      f'deleted ≈{n_deleted}, '
                      f'added {n_new})')

            torch.save(repairer.state_dict(), CKPT_DIR / f'repairer_step{step}.pt')
            print(f'  ** Checkpoint saved')

        step += 1

# ====== Final Save ======
torch.save(repairer.state_dict(), CKPT_DIR / 'repairer_final.pt')
print(f'\n{"="*60}')
print(f'  TRAINING COMPLETE: {MAX_STEPS} steps in {time.time()-t_start:.0f}s')
print(f'  Results saved to {OUT_DIR}')
print(f'{"="*60}')
