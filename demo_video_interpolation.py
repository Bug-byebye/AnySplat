#!/usr/bin/env python3
"""
WAN 视频插帧 + 外扩 Demo
=========================
用 WAN (Wan2.2-TI2V-5B) 的生成先验对低帧视频进行插帧和合理外扩。

流程:
  1. 读取输入视频 → 抽帧（模拟低帧率输入）
  2. VAE Encode → latent_A
  3. 时间维上采样（插帧 + 外扩）
  4. DiT 全层前向 → latent_B（利用生成先验填充细节）
  5. VAE Decode → 输出视频
  6. 同时返回 latent_B（可下游使用）

Usage:
    CUDA_VISIBLE_DEVICES=2 python3 demo_video_interpolation.py
    # 或指定 GPU
    CUDA_VISIBLE_DEVICES=0 python3 demo_video_interpolation.py
"""

import os
import sys
import warnings
import argparse
from pathlib import Path

warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "anysplat"))


# ============================================================
# 视频读取 / 写入工具
# ============================================================

def read_video_frames(path: str, max_frames: int = 200) -> torch.Tensor:
    """
    读入视频 → [T, H, W, 3] numpy uint8 → [T, 3, H, W] torch float32 [0,1].
    使用多种后端尝试。
    """
    path = str(path)

    # Method 1: torchvision
    try:
        import torchvision.io
        vframes, _, meta = torchvision.io.read_video(path, pts_unit="sec")
        # vframes: [T, H, W, 3] uint8
        vframes = vframes[:max_frames]
        frames = vframes.permute(0, 3, 1, 2).float() / 255.0  # [T, 3, H, W]
        print(f"  [torchvision] loaded {frames.shape[0]} frames, {meta}")
        return frames
    except Exception as e:
        print(f"  [torchvision] failed: {e}")

    # Method 2: imageio
    try:
        import imageio
        reader = imageio.get_reader(path)
        frames_list = []
        for i, frame in enumerate(reader):
            if i >= max_frames:
                break
            # frame: [H, W, 3] uint8
            frames_list.append(torch.from_numpy(frame.copy()).permute(2, 0, 1).float() / 255.0)
        frames = torch.stack(frames_list)
        print(f"  [imageio] loaded {frames.shape[0]} frames")
        return frames
    except Exception as e:
        print(f"  [imageio] failed: {e}")

    # Method 3: OpenCV
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        frames_list = []
        while len(frames_list) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # OpenCV: [H, W, 3] BGR uint8
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames_list.append(torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0)
        cap.release()
        frames = torch.stack(frames_list)
        print(f"  [OpenCV] loaded {frames.shape[0]} frames")
        return frames
    except Exception as e:
        print(f"  [OpenCV] failed: {e}")

    raise RuntimeError(f"Cannot read video: {path}")


def save_video_frames(
    frames: torch.Tensor,
    out_dir: Path,
    prefix: str = "frame",
    make_gif: bool = True,
):
    """
    保存视频帧为 PNG + GIF。
    frames: [T, 3, H, W] in [0,1].
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    T = frames.shape[0]
    for t in range(T):
        img = frames[t].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
        Image.fromarray((img * 255).astype(np.uint8)).save(
            out_dir / f"{prefix}_{t:04d}.png"
        )
    print(f"  Saved {T} frames to {out_dir}/{prefix}_*.png")

    if make_gif:
        try:
            img_list = []
            for t in range(T):
                img = frames[t].clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                img_list.append(Image.fromarray((img * 255).astype(np.uint8)))
            gif_path = out_dir / f"{prefix}.gif"
            img_list[0].save(
                gif_path,
                save_all=True,
                append_images=img_list[1:],
                duration=200,
                loop=0,
            )
            print(f"  Saved GIF: {gif_path}")
        except Exception as e:
            print(f"  GIF save skipped: {e}")


def write_mp4(frames: torch.Tensor, out_path: str, fps: float = 10.0):
    """
    将帧序列写出为 MP4 视频。
    frames: [T, 3, H, W] in [0,1].
    """
    try:
        import cv2
        T, C, H, W = frames.shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        for t in range(T):
            img = (frames[t].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            out.write(img_bgr)
        out.release()
        print(f"  MP4 saved: {out_path} ({T} frames @ {fps}fps)")
    except Exception as e:
        print(f"  MP4 save failed (CV2): {e}")
        # Fallback: use imageio
        try:
            import imageio
            writer = imageio.get_writer(out_path, fps=fps)
            T = frames.shape[0]
            for t in range(T):
                img = (frames[t].clamp(0, 1).cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                writer.append_data(img)
            writer.close()
            print(f"  MP4 saved (imageio): {out_path} ({T} frames @ {fps}fps)")
        except Exception as e2:
            print(f"  MP4 save failed (imageio): {e2}")


# ============================================================
# 处理函数
# ============================================================

@torch.no_grad()
def process_video(
    video: torch.Tensor,          # [T_in, 3, H, W] float32 [0,1]
    subsample_step: int = 3,       # 抽帧步长
    interpolation_factor: float = 3.0,   # 插帧倍数（相对抽帧后的帧数）
    extrapolate_front: int = 2,    # 前向外扩帧数
    extrapolate_back: int = 2,     # 后向外扩帧数
    target_height: int = 256,      # 目标高度（需能被 32 整除）
    target_width: int = 448,       # 目标宽度（需能被 32 整除）
    interpolation_strength: float = 1.0,
    use_dit: bool = False,
    method: str = "dit_single",    # "dit_single", "flow_matching"
    denoising_steps: int = 15,
    noise_level: float = 0.6,
    enable_decoder: bool = True,
    device: torch.device = None,
) -> dict:
    """
    完整流程：抽帧 → 插帧 → 外扩 → WAN 生成 → 输出。

    method:
      "dit_single" - 单次 DiT 前向（快速，使用 _run_full_dit_interp）
      "flow_matching" - Flow Matching 多步去噪（高质量，使用 interpolate_video_flow_matching）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    T_in = video.shape[0]

    # ---- Step 1: 预处理 ----
    H_orig, W_orig = video.shape[2], video.shape[3]
    if H_orig != target_height or W_orig != target_width:
        video_resized = F.interpolate(
            video, size=(target_height, target_width),
            mode="bilinear", align_corners=False,
        )
    else:
        video_resized = video

    # ---- Step 2: 抽帧（模拟低帧率输入） ----
    sub_indices = list(range(0, T_in, subsample_step))
    video_subsampled = video_resized[sub_indices]  # [T_sub, 3, H, W]
    T_sub = video_subsampled.shape[0]
    print(f"\n[Input] 原始 {T_in} 帧 → 抽帧 (step={subsample_step}) → {T_sub} 帧")

    # ---- Step 3: WAN 插帧 + 外扩 ----
    print(f"[WAN] 加载 VideoExtractor...")
    from gaussian_restorer import VideoExtractor

    extractor = VideoExtractor(
        feat_dim=48,
        use_fp16=True,
        enable_decoder=enable_decoder,
    ).to(device)

    # 触发加载
    _ = extractor.forward(video_subsampled.unsqueeze(0).to(device))
    print(f"  VAE decoder ready: {extractor.is_decoder_loaded()}")

    # 计算目标帧数
    target_num_frames = int(np.ceil(T_sub * interpolation_factor))
    total_target = target_num_frames + extrapolate_front + extrapolate_back
    print(f"[Interp] 插帧 {T_sub}→{target_num_frames} 帧 "
          f"(factor={interpolation_factor})")
    print(f"[Extra]  外扩前{extrapolate_front}+后{extrapolate_back}={total_target} 帧")

    # 执行插帧 + 外扩
    batch = video_subsampled.unsqueeze(0).to(device)  # [1, T_sub, 3, H, W]

    if method == "flow_matching":
        print(f"[Method] Flow Matching ({denoising_steps} steps, noise={noise_level})")
        generated_frames, latent_b = extractor.interpolate_video_flow_matching(
            batch,
            target_num_frames=target_num_frames,
            extrapolate_front=extrapolate_front,
            extrapolate_back=extrapolate_back,
            num_denoising_steps=denoising_steps,
            noise_level=noise_level,
        )
    else:
        print(f"[Method] Pixel-interp + VAE (use_dit={use_dit}, strength={interpolation_strength})")
        generated_frames, latent_b = extractor.interpolate_video(
            batch,
            target_num_frames=target_num_frames,
            extrapolate_front=extrapolate_front,
            extrapolate_back=extrapolate_back,
            interpolation_strength=interpolation_strength,
            use_dit=use_dit,
        )

    print(f"  生成视频: {generated_frames.shape}")
    print(f"  latent_B: {latent_b.shape}")

    # ---- Step 4: 返回结果 ----
    result = {
        "input_original": video,
        "input_sub": video_subsampled,
        "output": generated_frames[0],
        "latent_b": latent_b[0],
        "sub_indices": sub_indices,
        "target_height": target_height,
        "target_width": target_width,
        "extrapolate_front": extrapolate_front,
    }
    return result


# ============================================================
# 主函数
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="WAN Video Interpolation Demo")
    parser.add_argument("--video", type=str,
                        default="examples/video/llff_horns.mp4",
                        help="Input video path")
    parser.add_argument("--out", type=str,
                        default="outputs/video_interpolation",
                        help="Output directory")
    parser.add_argument("--subsample_step", type=int, default=3,
                        help="Frame decimation step (default: 3 → take 1 of 3 frames)")
    parser.add_argument("--interp_factor", type=float, default=3.0,
                        help="Interpolation factor relative to subsampled frames")
    parser.add_argument("--extrapolate_front", type=int, default=2,
                        help="Extra frames before sequence")
    parser.add_argument("--extrapolate_back", type=int, default=2,
                        help="Extra frames after sequence")
    parser.add_argument("--height", type=int, default=256,
                        help="Target height (must be multiple of 32)")
    parser.add_argument("--width", type=int, default=448,
                        help="Target width (must be multiple of 32)")
    parser.add_argument("--strength", type=float, default=1.0,
                        help="DiT interpolation strength [0,1]")
    parser.add_argument("--use_dit", action="store_true", default=False,
                        help="Enable DiT refinement (may cause color shift)")
    parser.add_argument("--method", type=str, default="dit_single",
                        choices=["dit_single", "flow_matching"],
                        help="Interpolation method (dit_single uses interpolate_video)")
    parser.add_argument("--denoising_steps", type=int, default=15,
                        help="Number of denoising steps for flow matching")
    parser.add_argument("--noise_level", type=float, default=0.6,
                        help="Noise level [0,1] for flow matching")
    parser.add_argument("--fps", type=float, default=8.0,
                        help="Output video FPS")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 检测 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")
    else:
        print("WARNING: CUDA not available! This requires a GPU.")
        return

    # ---- 读取视频 ----
    print(f"\n{'='*60}")
    print(f"Reading video: {video_path}")
    print(f"{'='*60}")
    video = read_video_frames(str(video_path))
    T_in, C, H, W = video.shape
    print(f"  Original: {T_in} frames, {H}x{W}")

    # ---- 执行插帧 + 外扩 ----
    result = process_video(
        video=video,
        subsample_step=args.subsample_step,
        interpolation_factor=args.interp_factor,
        extrapolate_front=args.extrapolate_front,
        extrapolate_back=args.extrapolate_back,
        target_height=args.height,
        target_width=args.width,
        interpolation_strength=args.strength,
        use_dit=args.use_dit,
        method=args.method,
        denoising_steps=args.denoising_steps,
        noise_level=args.noise_level,
        enable_decoder=True,
        device=device,
    )

    # ---- 保存结果 ----
    print(f"\n{'='*60}")
    print(f"Saving results to: {out_dir}")
    print(f"{'='*60}")

    # 1. 输入帧（原始）
    save_video_frames(result["input_original"], out_dir / "input_original", "input")

    # 2. 抽帧后的低帧输入
    save_video_frames(result["input_sub"], out_dir / "input_subsampled", "sub")

    # 3. 插帧+外扩后的输出视频
    out_frames = result["output"]
    save_video_frames(out_frames, out_dir / "output", "gen")

    # 4. 写出 MP4
    output_mp4 = str(out_dir / "result.mp4")
    write_mp4(out_frames, output_mp4, fps=args.fps)

    # 5. 保存 latent_B 信息
    latent_b = result["latent_b"]
    print(f"\n[Latent_B] 形状: {latent_b.shape}")
    print(f"  dtype: {latent_b.dtype}, "
          f"range: [{latent_b.min():.4f}, {latent_b.max():.4f}]")
    # 保存 latent_B 为 .pt 文件供后续使用
    torch.save(latent_b.cpu(), out_dir / "latent_b.pt")
    print(f"  Saved: {out_dir / 'latent_b.pt'}")

    # 6. 对比：输入 vs 输出的帧数信息
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"  原始帧数:          {T_in}")
    print(f"  抽帧后输入:        {result['input_sub'].shape[0]} 帧 "
          f"(step={args.subsample_step})")
    print(f"  插帧+外扩输出:     {out_frames.shape[0]} 帧 "
          f"(插帧 {result['input_sub'].shape[0]}→{int(np.ceil(result['input_sub'].shape[0]*args.interp_factor))}"
          f" + 前扩{args.extrapolate_front} + 后扩{args.extrapolate_back})")
    print(f"  latent_B shape:    {list(latent_b.shape)}")
    print(f"  输出视频:          {output_mp4}")
    print(f"  Done!")


if __name__ == "__main__":
    main()
