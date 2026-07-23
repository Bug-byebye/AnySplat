#!/usr/bin/env python3
"""
WAN 完整推理 — Flow Matching 加噪 + 逐步去噪
===============================================

正确的 WAN 生成流程：
  输入帧 → VAE → latent → 加噪(flow matching) → 逐步去噪(UniPCMultistepScheduler) → VAE解码 → 输出

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 demo_wan_proper.py
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
    imgs[0].save(out_path, save_all=True, append_images=imgs[1:], duration=duration, loop=0)
    print(f"  GIF: {out_path}")


# ====== 配置 ======
SCENE = "apartment"
CAMERA = "10"
SAMPLE_STEP = 2
MAX_FRAMES = 8
RES = [224, 448]
OUT = Path(f'/data/sunchang/exp_train/wan_generation_demo/{SCENE}_cam{CAMERA}_proper')
OUT.mkdir(parents=True, exist_ok=True)

# ====== 加载数据 ======
print(f'[Data] VRNeRF {SCENE}, camera={CAMERA}...')
from data.vrnerf import VRNeRFVideoSequence
seq = VRNeRFVideoSequence(
    scene_dir=Path(f'/data/sunchang/datasets-raw/vrnerf/{SCENE}'),
    camera_id=CAMERA, sample_step=SAMPLE_STEP, max_frames=MAX_FRAMES,
    resolution=RES,
)
video = seq.frames.to(device)
T = video.shape[0]
H, W = RES

for t in range(T):
    save(video[t], OUT / f'input_{t:02d}.png')

# ====== 加载 VideoExtractor（包含 VAE + DiT） ======
print('\n[Model] Loading VideoExtractor...')
from gaussian_restorer import VideoExtractor
ext = VideoExtractor(feat_dim=48, enable_decoder=True, use_fp16=True).to(device)
_ = ext.forward(video.unsqueeze(0))  # trigger lazy load
vae = ext.vae
transformer = ext.transformer
print(f'  Decoder: {ext.is_decoder_loaded()}')

# ====== 创建 Flow Matching Scheduler ======
print('\n[Scheduler] UniPCMultistepScheduler (flow_prediction)...')
from diffusers import UniPCMultistepScheduler
scheduler = UniPCMultistepScheduler(
    num_train_timesteps=1000,
    prediction_type='flow_prediction',
    beta_start=0.0001, beta_end=0.02, beta_schedule='linear',
)
print(f'  Timesteps range: [{scheduler.timesteps[0]}, {scheduler.timesteps[-1]}]')

# ====== 逐帧生成 ======
print(f'\n[Generating] {T} frames with proper flow-matching denoising...')
(OUT / 'wan_proper').mkdir(exist_ok=True)

for t in range(T):
    inp = video[t:t+1].unsqueeze(0)  # [1, 1, 3, H, W]

    with torch.no_grad():
        # ---- Step 1: VAE Encode ----
        v_pad, T_pad, Ho, Wo = ext._prepare_video(inp)
        h = vae.encode(v_pad.float())
        if hasattr(h, 'latent_dist'):
            z = h.latent_dist.mean
        else:
            z = h
        z_0 = z[:, :48]  # [1, 48, T_pad//4, H/16, W/16]
        print(f'  latent: {z_0.shape}')

        # ---- Step 2: Flow Matching — 加噪 ----
        # For flow matching: z_t = (1 - t/T) * z_0 + (t/T) * noise
        noise = torch.randn_like(z_0)
        # Use t=400 (moderate noise — enough for generation, keeps structure)
        timestep = torch.tensor([400], device=device)
        t_norm = timestep.float() / 1000.0  # normalized to [0, 1]
        z_t = (1 - t_norm) * z_0 + t_norm * noise
        print(f'  noise level t={timestep[0].item()}')

        # ---- Step 3: Multi-step denoising ----
        scheduler.set_timesteps(20)  # 20 denoising steps
        scheduler_t = scheduler.timesteps.to(device)

        # Find where to start (closest to our timestep)
        start_idx = (scheduler_t.float() - timestep[0].float()).abs().argmin().item()
        active_ts = scheduler_t[start_idx:]

        latent_z = z_t.clone()
        dtype = torch.bfloat16

        for step_i, ts in enumerate(active_ts):
            ts_b = ts.unsqueeze(0)

            # DiT forward
            latent_dit = latent_z.to(dtype)
            rope_emb = transformer.rope(latent_dit)
            hidden = transformer.patch_embedding(latent_dit)
            hidden = hidden.flatten(2).transpose(1, 2)

            # Conditioning (dummy text, actual timestep)
            text_dim = transformer.config.text_dim
            dummy_text = torch.zeros((1, 512, text_dim), device=device, dtype=dtype)
            temb, ts_proj, enc_hidden, _ = transformer.condition_embedder(
                ts_b, dummy_text, None
            )
            ts_proj = ts_proj.unflatten(1, (6, -1))

            # All DiT layers
            for i in range(transformer.config.num_layers):
                hidden = transformer.blocks[i](hidden, enc_hidden, ts_proj, rope_emb)

            # Output → latent space (norm_out + proj_out + unpatchify)
            hidden = transformer.norm_out(hidden)
            hidden = transformer.proj_out(hidden)  # [B, N, 192]
            T_lat = z_0.shape[2]
            Hp = max(z_0.shape[3] // 2, 1)
            Wp = max(z_0.shape[4] // 2, 1)
            n_vis = min(T_lat * Hp * Wp, hidden.shape[1])
            h_vis = hidden[:, :n_vis]
            h_vis = h_vis.reshape(h_vis.shape[0], T_lat, Hp, Wp, 48, 2, 2)
            h_vis = h_vis.permute(0, 4, 1, 2, 5, 3, 6)
            pred_flow = h_vis.reshape(h_vis.shape[0], 48, T_lat, Hp * 2, Wp * 2)

            # Scheduler step (flow_prediction mode)
            latent_z = scheduler.step(pred_flow, ts, latent_z).prev_sample

        # ---- Step 4: VAE Decode ----
        dec_out = vae.decode(latent_z.float())
        if hasattr(dec_out, 'sample'):
            frame = dec_out.sample
        else:
            frame = dec_out

        frame = frame[:, :, 0, :Ho, :Wo]  # first temporal slice
        frame = frame.clamp(0, 1)

        psnr = -10 * np.log10(max(torch.nn.functional.mse_loss(frame[0], video[t]).item(), 1e-10))
        save(frame[0], OUT / 'wan_proper' / f'{t:02d}.png')
        save(video[t], OUT / 'wan_proper' / f'{t:02d}_input.png')
        print(f'  Frame {t+1}/{T}: PSNR={psnr:.1f} dB')

# GIF
make_gif(OUT / 'wan_proper', OUT / 'wan_proper.gif', pattern='*_input.png')
make_gif(OUT / 'wan_proper', OUT / 'wan_proper_gen.gif', pattern='[0-9]*.png')
print(f'\n{"="*60}')
print(f'  Done. Results: {OUT}')
print(f'{"="*60}')
