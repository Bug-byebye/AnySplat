#!/usr/bin/env python3
"""
Unified Training Script — Gaussian Repair Network (v4)
======================================================

Single unified architecture: VideoExtractor → GaussianFeatureEncoder → UnifiedRepairer

Key improvements over previous versions:
  1. VideoExtractor treats frames as VIDEO CLIP (temporal VAE, not per-view)
  2. GaussianFeatureEncoder uses AnySplat's predicted cameras (no GT poses)
  3. Query-based GaussianGenerator (fully differentiable, no top-K lifting)
  4. Clean unified config (single source of truth)

Pipeline per step:
  Input RGB [B,V,3,H,W] (pure images, no poses/depth)
    │
    ├── AnySplat (frozen) → Gaussians [N, ...] + predicted cameras
    │
    ├── VideoExtractor (frozen) → video features [B*V, 48, H, W]
    │   (temporal-aware: all V frames as a video clip)
    │
    ├── GaussianFeatureEncoder → per-Gaussian features [B, N, 128]
    │                              + global scene descriptor [B, 128]
    │
    ├── UnifiedRepairer
    │   ├── Δ_params [B, N, 82] + Δ_means [B, N, 3] + keep_prob [B, N, 1]
    │   └── K new Gaussians [B, K, 85] (via learnable queries)
    │
    ├── build_refined_gaussians() → combined Gaussians [B, N+K, ...]
    │
    └── Render at target view → Loss → Backward

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 phase1_train_unified.py
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
B, V, H, W = 1, 2, 224, 448          # batch=1 initially
MAX_STEPS = 5000
LR = 3e-4
WD = 0.01
LOG_INT = 10
SAVE_INT = 500
VAL_INT = 200
TGT_GAP_MIN = 8
TGT_GAP_MAX = 15
N_NEW_QUERIES = 256

# Data
SCENE = '/data/sunchang/dl3dv_benchmark/032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7/nerfstudio'
OUT_DIR = Path('/data/sunchang/exp_train/phase1_v4_unified')
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR = OUT_DIR / 'results'
RESULT_DIR.mkdir(exist_ok=True)

# ====== Data Loading ======
print('[Data] Loading DL3DV...')
with open(os.path.join(SCENE, 'transforms.json')) as f:
    meta = json.load(f)
frames = meta['frames']
img_dir = os.path.join(SCENE, 'images_4')
H_full, W_full = int(meta['h']), int(meta['w'])
fx, fy = meta['fl_x'], meta['fl_y']
cx, cy = meta.get('cx', W_full/2), meta.get('cy', H_full/2)
sx, sy = W / W_full, H / W_full

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
    """Get GT camera (c2w) from DL3DV transforms.json."""
    return torch.tensor(frames[idx]['transform_matrix'], dtype=torch.float32)

# Build (context_idx, target_idx) pairs
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
# --- AnySplat (frozen) ---
print('\n[Model] Loading AnySplat...')
t0 = time.time()
from src.model.model.anysplat import AnySplat
from src.model.encoder.anysplat import EncoderAnySplatCfg, OpacityMappingCfg
from src.model.encoder.backbone.backbone_croco import BackboneCrocoCfg
from src.model.encoder.common.gaussian_adapter import GaussianAdapterCfg
from src.model.encoder.visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDACfg, DecoderSplattingCUDA

with open(os.path.join(PROJECT_ROOT, 'anysplat/pretrained_model/config.json')) as f:
    cfg_any = json.load(f)
enc_dict = cfg_any['encoder_cfg']
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

anysplat = AnySplat(EncoderAnySplatCfg(**ef),
                    DecoderSplattingCUDACfg(**cfg_any['decoder_cfg'])).to(device)
anysplat.load_state_dict(load_file(os.path.join(PROJECT_ROOT, 'anysplat/pretrained_model/model.safetensors')),
                          strict=False)
anysplat.eval()
for p in anysplat.parameters(): p.requires_grad_(False)
print(f'  Loaded: {sum(p.numel() for p in anysplat.parameters())/1e6:.0f}M ({time.time()-t0:.1f}s)')

# Decoder for rendering
decoder = DecoderSplattingCUDA(
    DecoderSplattingCUDACfg(name='splatting_cuda', background_color=[1, 1, 1],
                            make_scale_invariant=False)
).to(device)

# --- Video Extractor (frozen) ---
print('[Model] Loading VideoExtractor...')
t0 = time.time()
from gaussian_restorer import VideoExtractor
video_extractor = VideoExtractor(
    feat_dim=48,
    num_dit_layers=0,  # VAE-only for Phase 1
    use_fp16=True,
).to(device)
for p in video_extractor.parameters(): p.requires_grad_(False)
print(f'  Loaded ({time.time()-t0:.1f}s)')

# LPIPS for perceptual loss
try:
    from lpips import LPIPS
    lpips_fn = LPIPS(net='vgg').to(device).eval()
    for p in lpips_fn.parameters(): p.requires_grad_(False)
    print('[Model] LPIPS: loaded')
except:
    lpips_fn = None
    print('[Model] LPIPS: N/A')

# ====== Gaussian Encoder + Repairer ======
print('[Model] Building UnifiedRepairer...')
from gaussian_restorer import GaussianFeatureEncoder, UnifiedRepairer, RepairLoss

gaussian_encoder = GaussianFeatureEncoder(
    gaussian_dim=83,
    video_feat_dim=48,
    hidden_dim=128,
    association="camera_projection",
).to(device)

repairer = UnifiedRepairer(
    gaussian_feat_dim=128,
    n_new_queries=N_NEW_QUERIES,
    per_gaussian_hidden=256,
    generator_hidden=256,
    video_feat_dim=48,
).to(device)
repairer.print_summary()

# Optimizer (only repairer + gaussian_encoder are trainable)
train_params = list(repairer.parameters()) + list(gaussian_encoder.parameters())
opt = torch.optim.AdamW(train_params, lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEPS)

loss_fn = RepairLoss(
    weight_params=1e-6,
    weight_move=3e-4,
    weight_delete=1e-3,
    weight_gen=1e-5,
    weight_lpips=0.05,
    lpips_fn=lpips_fn,
)

print(f'  Total trainable: {sum(p.numel() for p in train_params)/1e3:.1f}K')

# ====== Helpers ======
from gaussian_restorer import build_refined_gaussians, extract_params_from_gaussians
from gaussian_restorer import normalize_intrinsics, save_image

def get_anysplat_cameras(ctx_norm):
    """Extract camera predictions from AnySplat's encoder."""
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        tokens, ps = anysplat.encoder.aggregator(
            ctx_norm.to(torch.bfloat16),
            intermediate_layer_idx=anysplat.encoder.cfg.intermediate_layer_idx)
    with torch.amp.autocast('cuda', enabled=False):
        from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        pose_enc = anysplat.encoder.camera_head(tokens)
        ext, intr = pose_encoding_to_extri_intri(pose_enc[-1], (H, W))
    return ext, intr

# ====== Train ======
print(f'\n{"="*60}')
print(f'  PHASE 1 V4 — Unified Repairer')
print(f'  Video: temporal-aware (clip of {V} frames)')
print(f'  Generator: {N_NEW_QUERIES} query-based new Gaussians')
print(f'  Cameras: AnySplat predicted (no GT poses)')
print(f'  Output: {OUT_DIR}')
print(f'{"="*60}\n')

step = 0
t_start = time.time()

# Step 0: baseline
print('[Init] Running baseline...')
with torch.no_grad():
    base_ci, base_ti = train_pairs[0]
    ctx_base = torch.stack([load_img(base_ci), load_img(base_ci + 2)]).unsqueeze(0).to(device)
    ctx_norm = ctx_base * 2 - 1
    enc_out0, _ = anysplat(ctx_norm, 100000)
    g0 = enc_out0.gaussians
    N0 = g0.means.shape[1]
    print(f'  Initial Gaussians: {N0}')

    # Render at target view
    tgt_ext0 = get_ext(base_ti).to(device).unsqueeze(0).unsqueeze(0)
    out0 = decoder.forward(g0, tgt_ext0,
        normalize_intrinsics(torch.eye(3, device=device).unsqueeze(0).unsqueeze(0), W, H),
        torch.tensor([[0.1]], device=device), torch.tensor([[100.0]], device=device), (H, W))
    save_image(out0.color[0, 0], RESULT_DIR / 'step_0' / 'baseline_target.png')

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
            enc_out, _ = anysplat(ctx_norm, global_step=100000)
            g = enc_out.gaussians
            N = g.means.shape[1]
            if N == 0:
                step += 1
                continue

            # Get cameras from AnySplat (predicted, not GT)
            ext, intr = get_anysplat_cameras(ctx_norm)

        # ===== Video features (temporal-aware, frozen) =====
        with torch.no_grad():
            video_feats = video_extractor(ctx_imgs)  # [B*V, 48, H, W]

        # ===== Gaussian feature encoding =====
        gaussian_params, gaussian_means = extract_params_from_gaussians(g)
        # gaussian_params: [B, N, 83], gaussian_means: [B, N, 3]

        per_gaussian_feats, global_scene_feat = gaussian_encoder(
            gaussian_params=gaussian_params,
            gaussian_means=gaussian_means,
            video_features=video_feats,
            extrinsic=ext,   # AnySplat predicted cameras
            intrinsic=intr,  # AnySplat predicted cameras
        )

        # ===== Repairer forward =====
        repair_out = repairer(
            per_gaussian_feats=per_gaussian_feats,
            global_scene_feat=global_scene_feat,
            gaussian_means=gaussian_means,
            gaussian_params=gaussian_params,
            video_features=video_feats,
        )

        # ===== Build refined Gaussians =====
        refined_g = build_refined_gaussians(
            gaussian_params=gaussian_params,
            gaussian_means=gaussian_means,
            delta_params=repair_out['delta_params'],
            delta_means=repair_out['delta_means'],
            keep_prob=repair_out['keep_prob'],
            new_params=repair_out['new_params'],
            new_means=repair_out['new_means'],
            new_activations=repair_out['new_activations'],
            device=device,
        )
        n_total_gaussians = refined_g.means.shape[1]
        n_new = n_total_gaussians - N

        # ===== Render at target view =====
        tgt_ext = get_ext(ti).to(device).unsqueeze(0).unsqueeze(0)
        # Use normalized intrinsics from AnySplat for rendering
        tgt_intr_n = normalize_intrinsics(intr[:, 0:1], W, H)
        rendered = decoder.forward(
            refined_g, tgt_ext, tgt_intr_n,
            torch.tensor([[0.1]], device=device),
            torch.tensor([[100.0]], device=device),
            (H, W),
        )
        rendered_color = rendered.color[0, 0]  # [3, H, W]

        # ===== Loss =====
        loss_dict = loss_fn(
            rendered_color=rendered_color,
            target_image=tgt_img[0],
            delta_params=repair_out['delta_params'],
            delta_means=repair_out['delta_means'],
            keep_prob=repair_out['keep_prob'],
            new_params=repair_out['new_params'],
            new_activations=repair_out['new_activations'],
        )
        loss = loss_dict['total']

        # ===== Backward =====
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, 1.0)
        opt.step()
        sched.step()

        # ===== Logging =====
        if step % LOG_INT == 0:
            psnr = -10 * math.log10(max(loss_dict['render_mse'].item(), 1e-10))
            keep_mean = repair_out['keep_prob'].mean().item()
            act_mean = repair_out['new_activations'].mean().item()
            delta_norm = repair_out['delta_params'].norm().item()
            move_norm = repair_out['delta_means'].norm().item()
            lr_now = opt.param_groups[0]['lr']

            log_str = (f'  Step {step:5d}/{MAX_STEPS} | '
                       f'{loss_fn.log(loss_dict, psnr)} | '
                       f'keep={keep_mean:.3f} act={act_mean:.3f} '
                       f'|Δ|={delta_norm:.2f} |Δμ|={move_norm:.4f} '
                       f'+{n_new}G | LR={lr_now:.2e}')
            print(log_str)

        # ===== Save =====
        if step > 0 and step % SAVE_INT == 0:
            with torch.no_grad():
                step_dir = RESULT_DIR / f'step_{step}'
                step_dir.mkdir(exist_ok=True)
                save_image(rendered_color, step_dir / 'refined_target.png')

                # Stats
                np.save(str(step_dir / 'keep_prob.npy'),
                        repair_out['keep_prob'].squeeze().cpu().numpy())
                np.save(str(step_dir / 'new_activations.npy'),
                        repair_out['new_activations'].squeeze().cpu().numpy())
                print(f'  ** Saved step {step}: N={N} → {n_total_gaussians} (+{n_new})')

            torch.save({
                'repairer': repairer.state_dict(),
                'gaussian_encoder': gaussian_encoder.state_dict(),
                'opt': opt.state_dict(),
                'step': step,
            }, CKPT_DIR / f'unified_step{step}.pt')

        step += 1

# ====== Final Save ======
torch.save({
    'repairer': repairer.state_dict(),
    'gaussian_encoder': gaussian_encoder.state_dict(),
    'config': {'N_NEW_QUERIES': N_NEW_QUERIES, 'MAX_STEPS': MAX_STEPS},
}, CKPT_DIR / 'unified_final.pt')
print(f'\n{"="*60}')
print(f'  TRAINING COMPLETE: {MAX_STEPS} steps in {time.time()-t_start:.0f}s')
print(f'  Results saved to {OUT_DIR}')
print(f'{"="*60}')
