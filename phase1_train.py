#!/usr/bin/env python3
"""
Phase 1 统一训练入口 — YAML 配置驱动
======================================

Usage:
    # 使用默认配置
    CUDA_VISIBLE_DEVICES=2 python3 phase1_train.py

    # 使用自定义配置
    CUDA_VISIBLE_DEVICES=2 python3 phase1_train.py --config config/my_exp.yaml

    # 仅测试数据加载
    CUDA_VISIBLE_DEVICES=2 python3 phase1_train.py --dry-run
"""

import os, sys, json, time, math, random, argparse, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from safetensors.torch import load_file

PROJECT_ROOT = '/home-ldap/sunchang/3dProjects/GuassDiff'
sys.path.insert(0, PROJECT_ROOT)                              # for data, gaussian_restorer, etc.
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'anysplat'))  # for from src.* imports (AnySplat)

device = 'cuda:0'

# ============================================================================
# 参数
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='config/default.yaml')
parser.add_argument('--dry-run', action='store_true', help='仅加载数据和模型，不训练')
args = parser.parse_args()

# 加载 YAML 配置
from data import load_config
CFG = load_config(args.config)

EXPERIMENT = CFG['experiment']
DATA_CFG = CFG['data']
MODEL_CFG = CFG['model']
TRAIN_CFG = CFG['training']

OUT_DIR = Path(EXPERIMENT['output_dir'])
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR = OUT_DIR / 'checkpoints'
CKPT_DIR.mkdir(exist_ok=True)
RESULT_DIR = OUT_DIR / 'results'
RESULT_DIR.mkdir(exist_ok=True)

B, V = DATA_CFG.get('batch_size', 1), DATA_CFG.get('context_views', 2)
H, W = DATA_CFG.get('resolution', [224, 448])
MAX_STEPS = TRAIN_CFG.get('max_steps', 5000)
LR = TRAIN_CFG.get('learning_rate', 3e-4)
WD = TRAIN_CFG.get('weight_decay', 0.01)
LOG_INT = TRAIN_CFG.get('log_interval', 10)
SAVE_INT = TRAIN_CFG.get('save_interval', 500)
VAL_INT = TRAIN_CFG.get('val_interval', 200)
TGT_MIN, TGT_MAX = DATA_CFG.get('target_gap', [8, 15])

# ============================================================================
# 数据
# ============================================================================
print(f'\n[Data] Loading dataset: {DATA_CFG["dataset"]}/{DATA_CFG["scene"]}...')
from data import build_dataset

train_data = build_dataset(DATA_CFG, split='train')
print(f'  Train pairs: {len(train_data)}')

if args.dry_run:
    sample = train_data[0]
    print(f'  Sample keys: {list(sample.keys())}')
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f'    {k}: {tuple(v.shape)}')
    print('[Dry run] Data OK. Exiting.')
    sys.exit(0)

# ============================================================================
# 模型
# ============================================================================
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
anysplat.load_state_dict(load_file(os.path.join(PROJECT_ROOT, 'anysplat/pretrained_model/model.safetensors'),
                                    strict=False))
anysplat.eval()
for p in anysplat.parameters(): p.requires_grad_(False)
print(f'  Loaded: {sum(p.numel() for p in anysplat.parameters())/1e6:.0f}M ({time.time()-t0:.1f}s)')

# Decoder
decoder = DecoderSplattingCUDA(
    DecoderSplattingCUDACfg(name='splatting_cuda', background_color=[1,1,1],
                            make_scale_invariant=False)
).to(device)

# Video Extractor
print('[Model] Loading VideoExtractor...')
t0 = time.time()
from gaussian_restorer import VideoExtractor

ve_cfg = MODEL_CFG.get('video_extractor', {})
video_extractor = VideoExtractor(
    feat_dim=ve_cfg.get('feat_dim', 48),
    enable_decoder=ve_cfg.get('enable_decoder', False),
    use_adapters=ve_cfg.get('use_adapters', False),
    adapter_bottleneck=ve_cfg.get('adapter_bottleneck', 64),
).to(device)
for p in video_extractor.parameters(): p.requires_grad_(False)
print(f'  Loaded ({time.time()-t0:.1f}s)')

# LPIPS
try:
    from lpips import LPIPS
    lpips_fn = LPIPS(net='vgg').to(device).eval()
    for p in lpips_fn.parameters(): p.requires_grad_(False)
    print('[Model] LPIPS: loaded')
except:
    lpips_fn = None

# Gaussian Encoder + Repairer
print('[Model] Building GaussianFeatureEncoder + UnifiedRepairer...')
from gaussian_restorer import GaussianFeatureEncoder, UnifiedRepairer, RepairLoss

ge_cfg = MODEL_CFG.get('gaussian_encoder', {})
gaussian_encoder = GaussianFeatureEncoder(
    gaussian_dim=83,
    video_feat_dim=ve_cfg.get('feat_dim', 48),
    hidden_dim=ge_cfg.get('hidden_dim', 128),
    association=ge_cfg.get('association', 'camera_projection'),
).to(device)

r_cfg = MODEL_CFG.get('repairer', {})
repairer = UnifiedRepairer(
    gaussian_feat_dim=ge_cfg.get('hidden_dim', 128),
    n_new_queries=r_cfg.get('n_new_queries', 256),
    per_gaussian_hidden=r_cfg.get('per_gaussian_hidden', 256),
    generator_hidden=r_cfg.get('generator_hidden', 256),
    video_feat_dim=ve_cfg.get('feat_dim', 48),
).to(device)
repairer.print_summary()

l_cfg = TRAIN_CFG.get('loss', {})
loss_fn = RepairLoss(
    weight_params=l_cfg.get('weight_params', 1e-6),
    weight_move=l_cfg.get('weight_move', 3e-4),
    weight_delete=l_cfg.get('weight_delete', 1e-3),
    weight_gen=l_cfg.get('weight_gen', 1e-5),
    weight_lpips=l_cfg.get('weight_lpips', 0.05),
    lpips_fn=lpips_fn,
)

train_params = list(repairer.parameters()) + list(gaussian_encoder.parameters())
opt = torch.optim.AdamW(train_params, lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEPS)
print(f'  Trainable: {sum(p.numel() for p in train_params)/1e3:.1f}K')

# ============================================================================
# 工具函数
# ============================================================================
from gaussian_restorer import (build_refined_gaussians, extract_params_from_gaussians,
                                      normalize_intrinsics, save_image, render_gaussians)

def get_anysplat_cameras(ctx_norm):
    """提取 AnySplat 内部预测的相机。"""
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
        tokens, ps = anysplat.encoder.aggregator(
            ctx_norm.to(torch.bfloat16),
            intermediate_layer_idx=anysplat.encoder.cfg.intermediate_layer_idx)
    with torch.amp.autocast('cuda', enabled=False):
        from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
        pe = anysplat.encoder.camera_head(tokens)
        ext, intr = pose_encoding_to_extri_intri(pe[-1], (H, W))
    return ext, intr

# ============================================================================
# 训练
# ============================================================================
print(f'\n{"="*60}')
print(f'  {EXPERIMENT["name"]}')
print(f'  Dataset: {DATA_CFG["dataset"]}/{DATA_CFG["scene"]}')
if DATA_CFG.get('cameras'):
    print(f'  Cameras: {DATA_CFG["cameras"]}')
print(f'  V={V}, H={H}, W={W}, steps={MAX_STEPS}')
print(f'  Output: {OUT_DIR}')
print(f'{"="*60}\n')

step = 0
t_start = time.time()

# Step 0: baseline
print('[Init] Baseline...')
with torch.no_grad():
    sample0 = train_data[0]
    ctx0 = sample0['ctx_images'].unsqueeze(0).to(device)
    ctx_norm0 = ctx0 * 2 - 1
    enc_out0, _ = anysplat(ctx_norm0, 100000)
    g0 = enc_out0.gaussians
    N0 = g0.means.shape[1]
    print(f'  Initial: {N0} Gaussians')

    # Render at target
    tgt_c2w = sample0['tgt_c2w'].unsqueeze(0).unsqueeze(0).to(device)
    tgt_intr = sample0['tgt_intr'].unsqueeze(0).unsqueeze(0).to(device)
    out0 = decoder.forward(g0, tgt_c2w, tgt_intr,
        torch.tensor([[0.1]], device=device), torch.tensor([[100.0]], device=device), (H, W))
    save_image(out0.color[0, 0], RESULT_DIR / 'step_0' / 'baseline.png')

print('  Baseline saved. Training...\n')

while step < MAX_STEPS:
    # Shuffle
    indices = list(range(len(train_data)))
    random.shuffle(indices)

    for idx in indices:
        if step >= MAX_STEPS:
            break

        sample = train_data[idx]

        # ---- 数据 ----
        ctx_imgs = sample['ctx_images'].unsqueeze(0).to(device)   # [1, V, 3, H, W]
        tgt_img = sample['tgt_image'].unsqueeze(0).to(device)     # [1, 3, H, W]
        ctx_norm = ctx_imgs * 2 - 1

        # ---- AnySplat ----
        with torch.no_grad():
            enc_out, _ = anysplat(ctx_norm, 100000)
            g = enc_out.gaussians
            N = g.means.shape[1]
            if N == 0:
                step += 1; continue
            ext, intr = get_anysplat_cameras(ctx_norm)

        # ---- WAN latent_B 特征 ----
        with torch.no_grad():
            # video_extractor.forward() 内部: VAE → DiT全40层 → latent_B → 投影
            video_feats = video_extractor(ctx_imgs)  # [B*V, feat_dim, H, W]

        # ---- 高斯编码 + Repair ----
        g_params, g_means = extract_params_from_gaussians(g)

        per_g_feats, global_feat = gaussian_encoder(
            gaussian_params=g_params, gaussian_means=g_means,
            video_features=video_feats, extrinsic=ext, intrinsic=intr,
        )

        repair_out = repairer(
            per_gaussian_feats=per_g_feats, global_scene_feat=global_feat,
            gaussian_means=g_means, gaussian_params=g_params,
            video_features=video_feats,
        )

        # ---- 构建修复后的高斯 ----
        refined_g = build_refined_gaussians(
            gaussian_params=g_params, gaussian_means=g_means,
            delta_params=repair_out['delta_params'],
            delta_means=repair_out['delta_means'],
            keep_prob=repair_out['keep_prob'],
            new_params=repair_out['new_params'],
            new_means=repair_out['new_means'],
            new_activations=repair_out['new_activations'],
            device=device,
        )
        n_new = refined_g.means.shape[1] - N

        # ---- 渲染目标视图 ----
        tgt_c2w = sample['tgt_c2w'].unsqueeze(0).unsqueeze(0).to(device)
        tgt_intr = sample['tgt_intr'].unsqueeze(0).unsqueeze(0).to(device)
        rendered = decoder.forward(refined_g, tgt_c2w, tgt_intr,
            torch.tensor([[0.1]], device=device), torch.tensor([[100.0]], device=device), (H, W))
        rendered_color = rendered.color[0, 0]

        # ---- Loss ----
        loss_dict = loss_fn(
            rendered_color=rendered_color, target_image=tgt_img[0],
            delta_params=repair_out['delta_params'],
            delta_means=repair_out['delta_means'],
            keep_prob=repair_out['keep_prob'],
            new_params=repair_out['new_params'],
            new_activations=repair_out['new_activations'],
        )
        loss = loss_dict['total']

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(train_params, TRAIN_CFG.get('grad_clip', 1.0))
        opt.step()
        sched.step()

        # ---- Log ----
        if step % LOG_INT == 0:
            psnr = -10 * math.log10(max(loss_dict['render_mse'].item(), 1e-10))
            keep_m = repair_out['keep_prob'].mean().item()
            act_m = repair_out['new_activations'].mean().item()
            dn = repair_out['delta_params'].norm().item()
            mn = repair_out['delta_means'].norm().item()
            print(f'  {step:5d}/{MAX_STEPS} | '
                  f'PSNR={psnr:.1f} |Δ|={dn:.2f} |Δμ|={mn:.4f} '
                  f'keep={keep_m:.3f} act={act_m:.3f} +{n_new}G | '
                  f'{loss_fn.log(loss_dict, psnr)}')

        # ---- Save ----
        if step > 0 and step % SAVE_INT == 0:
            with torch.no_grad():
                sd = RESULT_DIR / f'step_{step}'
                sd.mkdir(exist_ok=True)
                save_image(rendered_color, sd / 'refined_target.png')
                torch.save({
                    'repairer': repairer.state_dict(),
                    'gaussian_encoder': gaussian_encoder.state_dict(),
                    'opt': opt.state_dict(),
                    'step': step,
                }, CKPT_DIR / f'unified_step{step}.pt')
                print(f'  ** Saved step {step}')

        step += 1

# Final save
torch.save({
    'repairer': repairer.state_dict(),
    'gaussian_encoder': gaussian_encoder.state_dict(),
}, CKPT_DIR / 'unified_final.pt')
print(f'\n{"="*60}')
print(f'  COMPLETE: {MAX_STEPS} steps in {time.time()-t_start:.0f}s')
print(f'{"="*60}')
