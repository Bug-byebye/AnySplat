from pathlib import Path
import argparse
import torch
import os
import json
from datetime import datetime

import wandb
from omegaconf import OmegaConf
from src.model.model.anysplat import AnySplat
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.utils.image import process_image
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.misc.image_io import save_image
from scripts.vrnerf_sampler import (
    load_scenes_used,
    get_dense_sparse_splits_with_paths,
)


def evaluate_scene_with_method(
    model: AnySplat,
    ctx_images: torch.Tensor,
    tgt_images: torch.Tensor,
    method_name: str,
    output_folder: Path,
    device: torch.device,
):
    """
    Evaluate a scene using either feed_forward or self_supervise method.
    
    Args:
        model: AnySplat model
        ctx_images: Context images [1, K, 3, H, W] in [0, 1]
        tgt_images: Target GT images [1, T, 3, H, W] in [0, 1]
        method_name: "feed_forward" or "self_supervise"
        output_folder: Output directory for this method
        device: torch device
    """
    b, v_ctx, _, h, w = ctx_images.shape
    _, v_tgt, _, _, _ = tgt_images.shape
    
    # Step 1: Scene reconstruction using context images only
    print(f"[{method_name}] Step 1: Reconstructing scene from {v_ctx} context images...")

    encoder_output = model.encoder(
        ctx_images,
        global_step=0,
        visualization_dump={},
    )
    gaussians = encoder_output.gaussians
    pred_context_pose = encoder_output.pred_context_pose
    if method_name == "self_supervise":
        # Self-supervise: iterative refinement
        from self_supervise import pose_interpolation, render_images, save_images, load_images
        import imageio
        import random
        
        ss_cfg = OmegaConf.to_container(OmegaConf.load("config/self_supervise.yaml"), resolve=True)
        num_interp_frames = ss_cfg["images"].get("interp_frames", 3)
        iter_num = ss_cfg["inference"].get("iter_num", 5)
        image_sorted = ss_cfg["images"].get("image_sorted", True)
        
        # 将 context 图像从 [0,1] 映射回 [-1,1]，并拆成单张列表，便于与新渲染的视图拼接
        combined_images = []
        for i in range(v_ctx):
            img = ctx_images[0, i]          # [3, H, W] in [0,1]
            img = img * 2.0 - 1.0           # -> [-1,1]
            combined_images.append(img.cpu())

        # Iterative refinement
        for i in range(iter_num):
            print(f"[{method_name}] Iteration {i+1}/{iter_num}...")
            
            # Interpolate poses
            interpolated_pose = pose_interpolation(
                pred_context_pose,
                num_interp_frames=num_interp_frames,
                image_sorted=image_sorted
            )
            
            # Render interpolated views (returns [N, C, H, W] in [0, 1])
            selected_images = render_images(model.decoder, interpolated_pose, gaussians)
            
            # Save intermediate results (optional, for debugging)
            ss_temp_folder = output_folder / f"ss_iter_{i:04d}"
            ss_temp_folder.mkdir(parents=True, exist_ok=True)
            save_images(selected_images, ss_temp_folder)
            
            # 将渲染出的图像（[0,1]）转换到 [-1,1] 后加入 combined_images 列表
            new_images = load_images(ss_temp_folder)  # list of tensors in [-1,1]
            # load_images 已经返回 [-1,1]，这里直接扩展列表即可
            combined_images.extend([img.cpu() for img in new_images])

            # 使用当前所有图像重新推理（拼成 [1, V, 3, H, W]）
            combined_tensor = torch.stack(combined_images, dim=0).unsqueeze(0).to(device)
            print(f"[{method_name}] Re-inferencing with {combined_tensor.shape[1]} total images...")
            encoder_output = model.encoder(
                (combined_tensor + 1) * 0.5,  # encoder 期望 [0,1]
                global_step=0,
                visualization_dump={},
            )
            gaussians = encoder_output.gaussians
            pred_context_pose = encoder_output.pred_context_pose
    else:
        if method_name != "feed_forward":
            raise ValueError(f"Unknown method: {method_name}")
    
    # Step 2: Predict poses for context + target images
    print(f"[{method_name}] Step 2: Predicting poses for {v_ctx} context + {v_tgt} target images...")
    vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        aggregated_tokens_list, patch_start_idx = model.encoder.aggregator(
            vggt_input_image, 
            intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx
        )
    
    with torch.cuda.amp.autocast(enabled=False):
        fp32_tokens = [token.float() for token in aggregated_tokens_list]
        pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
        pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(
            pred_all_pose_enc, 
            vggt_input_image.shape[-2:]
        )
    
    # Convert to world-to-camera format (inverse)
    extrinsic_padding = torch.tensor(
        [0, 0, 0, 1], 
        device=pred_all_extrinsic.device, 
        dtype=pred_all_extrinsic.dtype
    ).view(1, 1, 1, 4).repeat(b, vggt_input_image.shape[1], 1, 1)
    pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()
    
    # Normalize intrinsics
    pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
    pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h
    
    # Split context and target poses
    pred_all_context_extrinsic, pred_all_target_extrinsic = (
        pred_all_extrinsic[:, :v_ctx], 
        pred_all_extrinsic[:, v_ctx:]
    )
    pred_all_context_intrinsic, pred_all_target_intrinsic = (
        pred_all_intrinsic[:, :v_ctx], 
        pred_all_intrinsic[:, v_ctx:]
    )
    
    # Scale alignment: align predicted context poses with reconstructed context poses
    scale_factor = (
        pred_context_pose['extrinsic'][:, :, :3, 3].mean() / 
        pred_all_context_extrinsic[:, :, :3, 3].mean()
    )
    pred_all_target_extrinsic[..., :3, 3] = pred_all_target_extrinsic[..., :3, 3] * scale_factor
    pred_all_context_extrinsic[..., :3, 3] = pred_all_context_extrinsic[..., :3, 3] * scale_factor
    print(f"[{method_name}] Scale factor: {scale_factor.item():.4f}")
    
    # Step 3: Render target views using reconstructed gaussians and predicted target poses
    print(f"[{method_name}] Step 3: Rendering {v_tgt} target views...")
    output = model.decoder.forward(
        gaussians,
        pred_all_target_extrinsic,
        pred_all_target_intrinsic.float(),
        torch.ones(1, v_tgt, device=device) * 0.01,
        torch.ones(1, v_tgt, device=device) * 100,
        (h, w)
    )
    
    # Step 4: Compute metrics
    print(f"[{method_name}] Step 4: Computing metrics...")
    psnr = compute_psnr(tgt_images[0], output.color[0])
    ssim = compute_ssim(tgt_images[0], output.color[0])
    lpips = compute_lpips(tgt_images[0], output.color[0])
    
    print(f"[{method_name}] Results:")
    print(f"  PSNR: {psnr.mean().item():.2f} (mean), {psnr.min().item():.2f} (min), {psnr.max().item():.2f} (max)")
    print(f"  SSIM: {ssim.mean().item():.3f} (mean), {ssim.min().item():.3f} (min), {ssim.max().item():.3f} (max)")
    print(f"  LPIPS: {lpips.mean().item():.3f} (mean), {lpips.min().item():.3f} (min), {lpips.max().item():.3f} (max)")
    
    # Step 5: Save results
    pred_dir = output_folder / "pred"
    gt_dir = output_folder / "gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, (gt_image, pred_image) in enumerate(zip(tgt_images[0], output.color[0])):
        # 文件名中不再加入时间戳，时间戳仅体现在上层目录路径中
        save_image(gt_image, gt_dir / f"{idx:06d}.jpg")
        save_image(pred_image, pred_dir / f"{idx:06d}.jpg")
    
    return {
        "psnr": psnr.mean().item(),
        "ssim": ssim.mean().item(),
        "lpips": lpips.mean().item(),
        "psnr_per_view": psnr.cpu().tolist(),
        "ssim_per_view": ssim.cpu().tolist(),
        "lpips_per_view": lpips.cpu().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="NVS compare with wandb logging")
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name for this run (overrides config). "
             "Can also be set via env WANDB_RUN_NAME.",
    )
    args = parser.parse_args()

    scenes = load_scenes_used()

    # ------------------------------------------------------------------
    # 1. 读取 NVS 评估配置（优先 config/nvs_compare.yaml）
    # ------------------------------------------------------------------
    try:
        nvs_cfg = OmegaConf.load("config/nvs_compare.yaml")
    except Exception:
        nvs_cfg = OmegaConf.create({})

    exp_cfg = nvs_cfg.get("experiment", {})
    # 视角策略：dense | sparse（由 YAML 配置）
    view_strategy = str(exp_cfg.get("strategy", "dense")).strip().lower()
    if view_strategy not in ("dense", "sparse"):
        raise ValueError(f"experiment.strategy 必须为 'dense' 或 'sparse'，当前为: {view_strategy}")
    # 基于 VR-NeRF camera block 的 dense/sparse 设置
    num_context = int(exp_cfg.get("num_context", 32))
    camera_id = str(exp_cfg.get("camera_id", "20"))
    metrics_root_dir = Path(str(exp_cfg.get("metrics_dir", "outputs/nvs_compare")))
    image_root_dir = Path(str(exp_cfg.get("image_root_dir", "exp-results")))

    # 输出目录加时间戳，便于多次实验区分
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_output_dir = metrics_root_dir / f"run_{timestamp}"

    # ------------------------------------------------------------------
    # 2. 加载模型
    # ------------------------------------------------------------------
    print("Loading AnySplat model...")
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Model loaded on {device}")

    # ------------------------------------------------------------------
    # 3. dense/sparse 相关参数（先读 vrnerf_sampler.yaml，再用 nvs_compare.yaml 覆盖）
    # ------------------------------------------------------------------
    try:
        vr_cfg = OmegaConf.load("config/vrnerf_sampler.yaml")
        dense_sparse_cfg = vr_cfg.get("dense_sparse", {})
    except Exception:
        dense_sparse_cfg = {}
    # 允许在 nvs_compare.yaml 中覆盖 dense_sparse 字段
    nvs_dense_sparse_cfg = nvs_cfg.get("dense_sparse", {})
    if nvs_dense_sparse_cfg:
        # OmegaConf get 返回的是 DictConfig，这里用 dict(...) 兼容 .get
        tmp = OmegaConf.to_container(nvs_dense_sparse_cfg, resolve=True)
        dense_sparse_cfg = {**dense_sparse_cfg, **tmp}

    pool_size = int(dense_sparse_cfg.get("pool_size", 72))
    pool_stride = int(dense_sparse_cfg.get("pool_stride", 2))
    dense_seed = int(dense_sparse_cfg.get("seed", 0))
    images_subdir = str(dense_sparse_cfg.get("images_subdir", "images-jpeg-1k"))

    # Set up Weights & Biases logging (reuse main.yaml where possible).
    wandb_cfg = None
    try:
        main_cfg = OmegaConf.load("config/main.yaml")
        wandb_cfg = main_cfg.get("wandb", None)
    except Exception:
        wandb_cfg = None

    wandb_mode = "disabled"
    wandb_project = "anysplat-nvs"
    wandb_name = "nvs_compare"
    wandb_tags = ["nvs", "vrnerf"]

    if wandb_cfg is not None:
        wandb_mode = str(wandb_cfg.get("mode", wandb_mode))
        wandb_project = str(wandb_cfg.get("project", wandb_project))
        wandb_name = f"{wandb_cfg.get('name', wandb_name)}-nvs-compare"
        extra_tags = wandb_cfg.get("tags", None)
        if extra_tags is not None:
            wandb_tags = list(wandb_tags) + list(extra_tags)

    # 命令行或环境变量指定的 run 名称优先
    run_name_override = os.environ.get("WANDB_RUN_NAME") or args.wandb_name
    if run_name_override:
        wandb_name = run_name_override

    if wandb_mode != "disabled":
        run = wandb.init(
            project=wandb_project,
            mode=wandb_mode,
            name=wandb_name,
            tags=wandb_tags,
            config={
                "strategy": view_strategy,
                "num_context": num_context,
                "dense_sparse": {
                    "pool_size": pool_size,
                    "pool_stride": pool_stride,
                    "seed": dense_seed,
                    "images_subdir": images_subdir,
                    "camera_id": camera_id,
                },
                "device": str(device),
                "scenes": [s["scene"] for s in scenes],
            },
        )
    else:
        run = None

    all_results = {}

    # 为本次实验构造图像保存根目录：<image_root_dir>/vrnerf_<timestamp>/
    image_run_root = image_root_dir / f"vrnerf_{timestamp}"

    for item in scenes:
        scene = item["scene"]
        scene_dir: Path = item["scene_dir"]
        if not scene_dir.exists():
            print(f"[skip] scene dir not found: {scene_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing scene: {scene}")
        print(f"{'='*60}")

        # 使用 dense/sparse camera-block 策略获取 input/test 视角及对应路径
        print(f"[info] 使用 camera block {camera_id}，strategy={view_strategy}")
        try:
            dense_splits = get_dense_sparse_splits_with_paths(
                scene=scene,
                scene_dir=scene_dir,
                camera_id=camera_id,
                setting=view_strategy,
                num_input=num_context if view_strategy == "dense" else None,
                pool_size=pool_size,
                pool_stride=pool_stride,
                seed=dense_seed,
                images_subdir=images_subdir,
            )
        except ValueError as e:
            # 场景在对应 camera block 下可用图片数不足等
            print(f"[warn] scene {scene}: 构建 {view_strategy}-view 候选池失败（{e}），跳过该场景")
            continue

        if not dense_splits:
            print(f"[warn] scene {scene}: 未获取到有效 {view_strategy} split，跳过")
            continue

        split = dense_splits[0]
        input_paths = split["input_paths"]
        test_paths = split["test_paths"]

        print(f"[info] 输入视角数: {len(input_paths)}, 测试视角数: {len(test_paths)}")

        # Load and preprocess context (input) images: [-1, 1] -> [0, 1] after process_image
        ctx_images_raw = [process_image(str(p)) for p in input_paths]
        ctx_images = torch.stack(ctx_images_raw, dim=0).unsqueeze(0).to(device)  # [1, K, 3, 448, 448]
        ctx_images = (ctx_images + 1) * 0.5  # Convert to [0, 1] for model.encoder

        # Load and preprocess target images
        tgt_images_raw = [process_image(str(p)) for p in test_paths]
        tgt_images = torch.stack(tgt_images_raw, dim=0).unsqueeze(0).to(device)  # [1, T, 3, 448, 448]
        tgt_images = (tgt_images + 1) * 0.5  # Convert to [0, 1] for model.encoder

        # Save input images (与 output_ff/output_ss 同级)
        input_images_dir = image_run_root / scene / "input_images"
        input_images_dir.mkdir(parents=True, exist_ok=True)
        print(f"[info] 保存 {len(input_paths)} 张输入视图到 {input_images_dir}")
        for idx, img_tensor in enumerate(ctx_images[0]):  # ctx_images[0] is [K, 3, H, W] in [0,1]
            save_image(img_tensor, input_images_dir / f"{idx:06d}.jpg")

        scene_results = {}

        # Evaluate with feed_forward method
        print(f"\n--- Evaluating with feed_forward method ---")
        ff_output_folder = image_run_root / scene / "output_ff"
        ff_output_folder.mkdir(parents=True, exist_ok=True)
        ff_results = evaluate_scene_with_method(
            model=model,
            ctx_images=ctx_images,
            tgt_images=tgt_images,
            method_name="feed_forward",
            output_folder=ff_output_folder,
            device=device,
        )
        scene_results["feed_forward"] = ff_results

        # Evaluate with self_supervise method
        print(f"\n--- Evaluating with self_supervise method ---")
        ss_output_folder = image_run_root / scene / "output_ss"
        ss_output_folder.mkdir(parents=True, exist_ok=True)
        ss_results = evaluate_scene_with_method(
            model=model,
            ctx_images=ctx_images,
            tgt_images=tgt_images,
            method_name="self_supervise",
            output_folder=ss_output_folder,
            device=device,
        )
        scene_results["self_supervise"] = ss_results

        all_results[scene] = scene_results

        # 控制台对比当前场景两种方法
        print(f"\n--- Comparison for {scene} ---")
        print(f"Feed Forward:  PSNR={ff_results['psnr']:.2f}, SSIM={ff_results['ssim']:.3f}, LPIPS={ff_results['lpips']:.3f}")
        print(f"Self Supervise: PSNR={ss_results['psnr']:.2f}, SSIM={ss_results['ssim']:.3f}, LPIPS={ss_results['lpips']:.3f}")

    # Print summary
    print(f"\n{'='*60}")
    print("Summary across all scenes:")
    print(f"{'='*60}")
    print(f"{'Scene':<20} {'Method':<15} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8}")
    print("-" * 60)
    for scene, results in all_results.items():
        for method, metrics in results.items():
            print(f"{scene:<20} {method:<15} {metrics['psnr']:<8.2f} {metrics['ssim']:<8.3f} {metrics['lpips']:<8.3f}")
    
    # Compute averages
    ff_psnr = [r["feed_forward"]["psnr"] for r in all_results.values()]
    ff_ssim = [r["feed_forward"]["ssim"] for r in all_results.values()]
    ff_lpips = [r["feed_forward"]["lpips"] for r in all_results.values()]
    ss_psnr = [r["self_supervise"]["psnr"] for r in all_results.values()]
    ss_ssim = [r["self_supervise"]["ssim"] for r in all_results.values()]
    ss_lpips = [r["self_supervise"]["lpips"] for r in all_results.values()]
    
    avg_ff_psnr = sum(ff_psnr) / len(ff_psnr)
    avg_ff_ssim = sum(ff_ssim) / len(ff_ssim)
    avg_ff_lpips = sum(ff_lpips) / len(ff_lpips)
    avg_ss_psnr = sum(ss_psnr) / len(ss_psnr)
    avg_ss_ssim = sum(ss_ssim) / len(ss_ssim)
    avg_ss_lpips = sum(ss_lpips) / len(ss_lpips)

    print(f"\nAverage Feed Forward:  PSNR={avg_ff_psnr:.2f}, SSIM={avg_ff_ssim:.3f}, LPIPS={avg_ff_lpips:.3f}")
    print(f"Average Self Supervise: PSNR={avg_ss_psnr:.2f}, SSIM={avg_ss_ssim:.3f}, LPIPS={avg_ss_lpips:.3f}")

    # 将指标写入 JSON / 文本文件，便于后续分析
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    # 记录本次实验的关键配置 + 结果，方便之后完全复现实验
    config_section = {
        "experiment": {
            "num_context": num_context,
            "camera_id": camera_id,
            "metrics_root_dir": str(metrics_root_dir),
            "metrics_run_dir": str(metrics_output_dir),
            "image_root_dir": str(image_root_dir),
        },
        "dense_sparse": {
            "pool_size": pool_size,
            "pool_stride": pool_stride,
            "seed": dense_seed,
            "images_subdir": images_subdir,
        },
    }
    # 仅记录 wandb 的关键信息，避免过多无关字段
    config_section["wandb"] = {
        "mode": wandb_mode,
        "project": wandb_project,
        "name": wandb_name,
        "tags": wandb_tags,
    }

    metrics_payload = {
        "config": config_section,
        "per_scene": all_results,
        "average": {
            "feed_forward": {
                "psnr": avg_ff_psnr,
                "ssim": avg_ff_ssim,
                "lpips": avg_ff_lpips,
            },
            "self_supervise": {
                "psnr": avg_ss_psnr,
                "ssim": avg_ss_ssim,
                "lpips": avg_ss_lpips,
            },
        },
    }

    json_path = metrics_output_dir / f"metrics_{timestamp}.json"
    txt_path = metrics_output_dir / f"metrics_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        # 先写配置，再写实验结果
        f.write("Config:\n")
        f.write(f"  strategy         : {view_strategy}\n")
        f.write(f"  num_context      : {num_context}\n")
        f.write(f"  camera_id        : {camera_id}\n")
        f.write(f"  metrics_root_dir : {metrics_root_dir}\n")
        f.write(f"  metrics_run_dir  : {metrics_output_dir}\n")
        f.write(f"  image_root_dir   : {image_root_dir}\n")
        f.write("  dense_sparse:\n")
        f.write(f"    pool_size   : {pool_size}\n")
        f.write(f"    pool_stride : {pool_stride}\n")
        f.write(f"    seed        : {dense_seed}\n")
        f.write(f"    images_subdir: {images_subdir}\n")
        f.write("  wandb:\n")
        f.write(f"    mode    : {wandb_mode}\n")
        f.write(f"    project : {wandb_project}\n")
        f.write(f"    name    : {wandb_name}\n")
        f.write(f"    tags    : {wandb_tags}\n")

        f.write("\nPer-scene metrics:\n")
        for scene, results in all_results.items():
            for method, m in results.items():
                f.write(
                    f"{scene:20s} {method:15s} "
                    f"PSNR={m['psnr']:.2f} SSIM={m['ssim']:.3f} LPIPS={m['lpips']:.3f}\n"
                )
        f.write("\nAverages:\n")
        f.write(
            f"Feed Forward:  PSNR={avg_ff_psnr:.2f}, SSIM={avg_ff_ssim:.3f}, LPIPS={avg_ff_lpips:.3f}\n"
        )
        f.write(
            f"Self Supervise: PSNR={avg_ss_psnr:.2f}, SSIM={avg_ss_ssim:.3f}, LPIPS={avg_ss_lpips:.3f}\n"
        )

    # 使用 wandb.Table + 曲线图，将所有场景数据画在同一张图上
    if run is not None:
        rows = []
        for scene, results in all_results.items():
            for method, m in results.items():
                rows.append(
                    [scene, method, m["psnr"], m["ssim"], m["lpips"]]
                )

        table = wandb.Table(
            data=rows,
            columns=["scene", "method", "psnr", "ssim", "lpips"],
        )

        wandb.log(
            {
                # 场景级曲线图：横轴为 scene，纵轴为对应指标值，method 区分曲线
                "psnr_vs_scene": wandb.plot.line(
                    table,
                    "scene",
                    "psnr",
                    title="PSNR vs Scene",
                    stroke="method",
                ),
                "ssim_vs_scene": wandb.plot.line(
                    table,
                    "scene",
                    "ssim",
                    title="SSIM vs Scene",
                    stroke="method",
                ),
                "lpips_vs_scene": wandb.plot.line(
                    table,
                    "scene",
                    "lpips",
                    title="LPIPS vs Scene",
                    stroke="method",
                ),
                # 仍然记录一下总体平均，方便比较
                "summary/feed_forward/psnr": avg_ff_psnr,
                "summary/feed_forward/ssim": avg_ff_ssim,
                "summary/feed_forward/lpips": avg_ff_lpips,
                "summary/self_supervise/psnr": avg_ss_psnr,
                "summary/self_supervise/ssim": avg_ss_ssim,
                "summary/self_supervise/lpips": avg_ss_lpips,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
