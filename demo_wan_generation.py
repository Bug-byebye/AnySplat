#!/usr/bin/env python3
"""
Demo: WAN 视频生成效果 — VRNeRF 均匀采样
===========================================

用法:
    CUDA_VISIBLE_DEVICES=2 python3 demo_wan_generation.py
"""

import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
import numpy as np
from PIL import Image
from pathlib import Path

PROJECT_ROOT = '/home-ldap/sunchang/3dProjects/GuassDiff'
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'anysplat'))
device = 'cuda:0'


def save(tensor, path):
    img = tensor.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def make_gif(dir_path, out_path, pattern="*.png", duration=500):
    import glob
    files = sorted(glob.glob(str(dir_path / pattern)))
    if not files:
        return
    imgs = [Image.open(f) for f in files]
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:],
                 duration=duration, loop=0)
    print(f"  GIF: {out_path}")


# ====== 配置 ======
SCENE = "apartment"
CAMERA = "10"
SAMPLE_STEP = 2
MAX_FRAMES = 8
RES = [224, 448]
OUT = Path(f'/data/sunchang/exp_train/wan_generation_demo/{SCENE}_cam{CAMERA}')
OUT.mkdir(parents=True, exist_ok=True)

# ====== 加载数据 ======
print(f'[Data] VRNeRF {SCENE}, camera={CAMERA}, step={SAMPLE_STEP}...')
from data.vrnerf import VRNeRFVideoSequence
seq = VRNeRFVideoSequence(
    scene_dir=Path(f'/data/sunchang/datasets-raw/vrnerf/{SCENE}'),
    camera_id=CAMERA, sample_step=SAMPLE_STEP, max_frames=MAX_FRAMES,
    resolution=RES,
)
video = seq.frames.to(device)  # [T, 3, H, W]
T = video.shape[0]
print(f'  {T} frames')

(OUT / 'input').mkdir(exist_ok=True)
for t in range(T):
    save(video[t], OUT / 'input' / f'{t:02d}.png')
make_gif(OUT / 'input', OUT / 'input.gif')

# ====== 加载 VideoExtractor（加载完整的 Wan2.2 VAE） ======
print('\n[Model] Loading VideoExtractor...')
from gaussian_restorer import VideoExtractor
ext = VideoExtractor(feat_dim=48, enable_decoder=True).to(device)

# 触发 lazy-load
_ = ext.forward(video.unsqueeze(0))
print(f'  Decoder: {ext.is_decoder_loaded()}')
vae = ext.vae

# ====== 处理 ======
print(f'\n[Processing] {T} frames as video...')
(OUT / 'recon_vae').mkdir(exist_ok=True)
(OUT / 'wan_gen').mkdir(exist_ok=True)

with torch.no_grad():
    # 整段视频: [1, T, 3, H, W]
    inp = video.unsqueeze(0)

    # === Step 1: VAE encode / decode (compression test) ===
    # 用 vae.encode() 走完整流程（含 patchify、temporal chunking 等）
    v_pad, T_pad, Ho, Wo = ext._prepare_video(inp)

    # encode
    enc_out = vae.encode(v_pad.float())
    if hasattr(enc_out, 'latent_dist'):
        z = enc_out.latent_dist.mean  # [1, 48, T_pad//4, H/16, W/16]
    else:
        z = enc_out

    print(f'  latent: {z.shape}')

    # 取 mean（z_dim 以内的通道）
    z_a = z[:, :ext._z_dim]  # [1, 48, T', H', W']

    # decode（VAE-only）
    dec_out = vae.decode(z_a)
    if hasattr(dec_out, 'sample'):
        recon = dec_out.sample
    else:
        recon = dec_out
    # Output temporal dim may differ from input T_pad. Take what we can.
    T_out = recon.shape[2]
    T_use = min(T, T_out)
    print(f'  VAE recon: {recon.shape} (taking {T_use} of {T} input frames)')
    for t in range(T_use):
        save(recon[0, :, t].clamp(0, 1), OUT / 'recon_vae' / f'{t:02d}.png')

    # === Step 2: DiT → latent_B → decode ===
    z_b = ext._run_full_dit(z_a)
    print(f'  latent_B (after DiT): {z_b.shape}')

    # decode WAN-generated
    dec_out = vae.decode(z_b)
    if hasattr(dec_out, 'sample'):
        wan_f = dec_out.sample
    else:
        wan_f = dec_out
    T_wan = wan_f.shape[2]
    T_use_w = min(T, T_wan)
    print(f'  WAN gen: {wan_f.shape} (taking {T_use_w} of {T} input frames)')
    for t in range(T_use_w):
        save(wan_f[0, :, t].clamp(0, 1), OUT / 'wan_gen' / f'{t:02d}.png')

# GIFs
for sub in ['input', 'recon_vae', 'wan_gen']:
    d = OUT / sub
    if d.exists():
        make_gif(d, OUT / f'{sub}.gif')

# ====== PSNR ======
print(f'\n{"="*60}')
print(f'  PSNR vs input:')
print(f'{"="*60}')
psnrs_v, psnrs_w = [], []
T_show = min(T_use, T_use_w)
for t in range(T_show):
    mv = torch.nn.functional.mse_loss(recon[0, :, t].clamp(0,1), video[t]).item()
    mw = torch.nn.functional.mse_loss(wan_f[0, :, t].clamp(0,1), video[t]).item()
    pv = -10 * np.log10(max(mv, 1e-10))
    pw = -10 * np.log10(max(mw, 1e-10))
    psnrs_v.append(pv); psnrs_w.append(pw)
    print(f'  Frame {t:2d}: VAE={pv:.1f}dB  WAN={pw:.1f}dB  Δ={pw-pv:+.1f}dB')
print(f'  Mean:     VAE={np.mean(psnrs_v):.1f}dB  WAN={np.mean(psnrs_w):.1f}dB  Δ={np.mean(psnrs_w)-np.mean(psnrs_v):+.1f}dB')
print(f'\nOutput: {OUT}')
