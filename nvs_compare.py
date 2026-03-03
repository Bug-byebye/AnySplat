from pathlib import Path
import argparse
import torch
import os
import json
from datetime import datetime
from dataclasses import replace, fields

import wandb
from omegaconf import OmegaConf
from src.model.model.anysplat import AnySplat
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.utils.image import process_image
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.misc.image_io import save_image
from scripts.vrnerf.vrnerf_sampler import (
    load_scenes_used,
    get_dense_sparse_splits_with_paths,
)
from scripts.vrnerf.process_fisheye import (
    process_scene_cameras_fisheye,
    get_processed_image_path,
)
from ttt import TTTConfig, load_ttt_config, run_ttt
from itr import ITRConfig, itr, load_itr_config
from src.utils.model_loading import load_model_with_fallback


def load_local_model(device: torch.device, local_path: str | None = None) -> AnySplat:
    """Load model with local-first, HuggingFace fallback strategy.
    
    Args:
        device: Torch device to move model to
        local_path: Path to local pretrained model. If None, uses default.
                   If relative path, will be resolved relative to current working directory.
    """
    if local_path:
        if not Path(local_path).is_absolute():
            local_path = str(Path.cwd() / local_path)
    
    return load_model_with_fallback(
        local_path=local_path,
        device=device,
    )


def evaluate_scene_with_method(
    model: AnySplat,
    ctx_images: torch.Tensor,
    tgt_images: torch.Tensor,
    method_name: str,
    output_folder: Path,
    device: torch.device,
    ttt_input_folder: Path | None = None,
    ttt_overrides: dict | None = None,
    itr_input_folder: Path | None = None,
    itr_overrides: dict | None = None,
):
    """
    Evaluate a scene using either feed_forward, ttt, or itr method.
    
    Args:
        model: AnySplat model
        ctx_images: Context images [1, K, 3, H, W] in [0, 1]
        tgt_images: Target GT images [1, T, 3, H, W] in [0, 1]
        method_name: "feed_forward" | "ttt" | "itr"
        output_folder: Output directory for this method
        device: torch device
    """
    b, v_ctx, _, h, w = ctx_images.shape
    _, v_tgt, _, _, _ = tgt_images.shape

    if method_name == "ttt":
        if ttt_input_folder is None:
            raise ValueError("ttt_input_folder is required when method_name='ttt'")
        ttt_output_folder = output_folder / "ttt_outputs"
        ttt_output_folder.mkdir(parents=True, exist_ok=True)
        ttt_cfg = load_ttt_config(Path("config/ttt.yaml"))
        if ttt_overrides:
            ttt_cfg = replace(ttt_cfg, **ttt_overrides)
        ttt_cfg = replace(
            ttt_cfg,
            input_folder=ttt_input_folder,
            output_folder=ttt_output_folder,
            device=str(device),
        )
        print(f"[{method_name}] Running TTT with input folder: {ttt_input_folder}")
        run_ttt(model, ttt_cfg)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    if method_name == "itr":
        if itr_input_folder is None:
            raise ValueError("itr_input_folder is required when method_name='itr'")
        itr_output_folder = output_folder / "itr_outputs"
        itr_output_folder.mkdir(parents=True, exist_ok=True)
        itr_cfg = load_itr_config(Path("config/itr.yaml"))
        if itr_overrides:
            itr_cfg = replace(itr_cfg, **itr_overrides)
        itr_cfg = replace(
            itr_cfg,
            input_folder=itr_input_folder,
            output_folder=itr_output_folder,
            device=str(device),
        )
        print(f"[{method_name}] Running ITR with input folder: {itr_input_folder}")
        itr(model, itr_cfg)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    
    # Step 1: Scene reconstruction using context images only
    print(f"[{method_name}] Step 1: Reconstructing scene from {v_ctx} context images...")

    encoder_output = model.encoder(
        ctx_images,
        global_step=0,
        visualization_dump={},
    )
    gaussians = encoder_output.gaussians
    pred_context_pose = encoder_output.pred_context_pose
    # if method_name == "self_supervise":
    #     # Self-supervise: iterative refinement
    #     from self_supervise import pose_interpolation, render_images, save_images, load_images
    #     import imageio
    #     import random
    #
    #     ss_cfg = OmegaConf.to_container(OmegaConf.load("config/self_supervise.yaml"), resolve=True)
    #     num_interp_frames = ss_cfg["images"].get("interp_frames", 3)
    #     iter_num = ss_cfg["inference"].get("iter_num", 5)
    #     image_sorted = ss_cfg["images"].get("image_sorted", True)
    #
    #     # 将 context 图像从 [0,1] 映射回 [-1,1]，并拆成单张列表，便于与新渲染的视图拼接
    #     combined_images = []
    #     for i in range(v_ctx):
    #         img = ctx_images[0, i]          # [3, H, W] in [0,1]
    #         img = img * 2.0 - 1.0           # -> [-1,1]
    #         combined_images.append(img.cpu())
    #
    #     # Iterative refinement
    #     for i in range(iter_num):
    #         print(f"[{method_name}] Iteration {i+1}/{iter_num}...")
    #
    #         # Interpolate poses
    #         interpolated_pose = pose_interpolation(
    #             pred_context_pose,
    #             num_interp_frames=num_interp_frames,
    #             image_sorted=image_sorted
    #         )
    #
    #         # Render interpolated views (returns [N, C, H, W] in [0, 1])
    #         selected_images = render_images(model.decoder, interpolated_pose, gaussians)
    #
    #         # Save intermediate results (optional, for debugging)
    #         ss_temp_folder = output_folder / f"ss_iter_{i:04d}"
    #         ss_temp_folder.mkdir(parents=True, exist_ok=True)
    #         save_images(selected_images, ss_temp_folder)
    #
    #         # 将渲染出的图像（[0,1]）转换到 [-1,1] 后加入 combined_images 列表
    #         new_images = load_images(ss_temp_folder)  # list of tensors in [-1,1]
    #         # load_images 已经返回 [-1,1]，这里直接扩展列表即可
    #         combined_images.extend([img.cpu() for img in new_images])
    #
    #         # 使用当前所有图像重新推理（拼成 [1, V, 3, H, W]）
    #         combined_tensor = torch.stack(combined_images, dim=0).unsqueeze(0).to(device)
    #         print(f"[{method_name}] Re-inferencing with {combined_tensor.shape[1]} total images...")
    #         encoder_output = model.encoder(
    #             (combined_tensor + 1) * 0.5,  # encoder 期望 [0,1]
    #             global_step=0,
    #             visualization_dump={},
    #         )
    #         gaussians = encoder_output.gaussians
    #         pred_context_pose = encoder_output.pred_context_pose
    if method_name not in {"feed_forward", "ttt", "itr"}:
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

    # ------------------------------------------------------------------
    # 1. 读取 NVS 评估配置（优先 config/nvs_compare.yaml）
    # ------------------------------------------------------------------
    try:
        nvs_cfg = OmegaConf.load("config/nvs_compare.yaml")
    except Exception:
        nvs_cfg = OmegaConf.create({})

    try:
        vr_cfg = OmegaConf.load("config/vrnerf_sampler.yaml")
    except Exception:
        vr_cfg = OmegaConf.create({})

    nvs_dict = OmegaConf.to_container(nvs_cfg, resolve=True) if nvs_cfg is not None else {}
    vr_dict = OmegaConf.to_container(vr_cfg, resolve=True) if vr_cfg is not None else {}
    if not isinstance(nvs_dict, dict):
        nvs_dict = {}
    if not isinstance(vr_dict, dict):
        vr_dict = {}

    dataset_root = nvs_dict.get("dataset_root", vr_dict.get("dataset_root", "datasets-raw/vrnerf"))
    scenes_path = nvs_dict.get("scenes", vr_dict.get("scenes", "datasets-raw/vrnerf/scenes_used.json"))
    scenes = load_scenes_used(
        dataset_root=Path(str(dataset_root)),
        scenes_used_path=Path(str(scenes_path)),
    )
    def extract_ttt_overrides(cfg) -> dict:
        if cfg is None:
            return {}
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            return {}
        candidate = cfg_dict.get("ttt") if isinstance(cfg_dict.get("ttt"), dict) else cfg_dict
        valid_keys = {field.name for field in fields(TTTConfig)}
        return {k: v for k, v in candidate.items() if k in valid_keys}

    def extract_itr_overrides(cfg) -> dict:
        if cfg is None:
            return {}
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            return {}
        candidate = cfg_dict.get("itr") if isinstance(cfg_dict.get("itr"), dict) else cfg_dict
        valid_keys = {field.name for field in fields(ITRConfig)}
        return {k: v for k, v in candidate.items() if k in valid_keys}

    ttt_overrides = extract_ttt_overrides(nvs_cfg)
    itr_overrides = extract_itr_overrides(nvs_cfg)

    exp_cfg = nvs_cfg.get("experiment", {})
    # 固定使用 dense 视角采样
    num_context = int(exp_cfg.get("num_context", 32))
    camera_id = str(exp_cfg.get("camera_id", "20"))
    fisheye_camera_id = str(exp_cfg.get("fisheye_camera_id", "4"))
    metrics_root_dir = Path(str(exp_cfg.get("metrics_dir", "outputs/nvs_compare")))
    image_root_dir = Path(str(exp_cfg.get("image_root_dir", "exp-results")))

    pretrained_model_path = exp_cfg.get("pretrained_model_path", "pretrained_model")
    if isinstance(pretrained_model_path, str):
        pretrained_model_path = str(pretrained_model_path)
    else:
        pretrained_model_path = "pretrained_model"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_output_dir = metrics_root_dir / f"run_{timestamp}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # dense/sparse 相关参数（先读 vrnerf_sampler.yaml，再用 nvs_compare.yaml 覆盖）
    # ------------------------------------------------------------------
    try:
        dense_sparse_cfg = vr_cfg.get("dense_sparse", {})
    except Exception:
        dense_sparse_cfg = {}
    # 允许在 nvs_compare.yaml 中覆盖 dense_sparse 字段
    nvs_dense_sparse_cfg = nvs_cfg.get("dense_sparse", {})
    if nvs_dense_sparse_cfg:
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
                "num_context": num_context,
                "camera_id": camera_id,
                "fisheye_camera_id": fisheye_camera_id,
                "dense_sparse": {
                    "pool_size": pool_size,
                    "pool_stride": pool_stride,
                    "seed": dense_seed,
                    "images_subdir": images_subdir,
                },
                "device": str(device),
                "scenes": [s["scene"] for s in scenes],
            },
        )
    else:
        run = None

    # Load fisheye configuration
    fisheye_scenes = []
    fisheye_json_path = Path(dataset_root) / "fisheye.json"
    if fisheye_json_path.exists():
        try:
            with open(fisheye_json_path, "r") as f:
                fisheye_scenes = json.load(f)
            print(f"[info] Loaded fisheye scenes: {fisheye_scenes}")
        except Exception as e:
            print(f"[warn] Failed to load fisheye.json: {e}")
    else:
        print(f"[info] fisheye.json not found at {fisheye_json_path}")

    all_results = {}

    # 为实验构造图像保存根目录：<image_root_dir>/vrnerf_<timestamp>/
    image_run_root = image_root_dir / f"vrnerf_{timestamp}"

    for scene_idx, item in enumerate(scenes):
        scene = item["scene"]
        scene_dir: Path = item["scene_dir"]
        if not scene_dir.exists():
            print(f"[skip] scene dir not found: {scene_dir}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing scene: {scene}")
        print(f"{'='*60}")

        # Determine camera_id to use for this scene
        is_fisheye_scene = scene in fisheye_scenes
        active_camera_id = fisheye_camera_id if is_fisheye_scene else camera_id

        # Process fisheye images if this scene is in fisheye list
        if is_fisheye_scene:
            # Check if fisheye images are already processed
            undistorted_dir = scene_dir / images_subdir / fisheye_camera_id / "undistorted"
            if undistorted_dir.exists() and any(undistorted_dir.glob("*.jpg")):
                print(f"[info] Scene {scene} is fisheye scene, using camera_id={fisheye_camera_id}")
                print(f"[info] Found existing processed images in {undistorted_dir}, skipping processing")
            else:
                print(f"[info] Scene {scene} detected as fisheye scene, using camera_id={fisheye_camera_id}, processing...")
                try:
                    processed_cameras = process_scene_cameras_fisheye(
                        scene_dir=scene_dir,
                        camera_ids=[fisheye_camera_id],
                        images_subdir=images_subdir,
                        output_subdir="undistorted",
                        balance=0.0,
                        crop=True,
                        verbose=True,
                    )
                    if processed_cameras:
                        print(f"[info] Successfully processed fisheye cameras: {processed_cameras}")
                    else:
                        print(f"[warn] No fisheye cameras were successfully processed")
                except Exception as e:
                    print(f"[warn] Failed to process fisheye images for {scene}: {e}")
        else:
            print(f"[info] Scene {scene} is normal scene, using camera_id={camera_id}")

        # 使用 dense camera-block 策略获取 input/test 视角及对应路径
        print(f"[info] 使用 camera block {active_camera_id}，strategy=dense{'（鱼眼场景）' if is_fisheye_scene else ''}")
        try:
            dense_splits = get_dense_sparse_splits_with_paths(
                scene=scene,
                scene_dir=scene_dir,
                camera_id=active_camera_id,
                setting="dense",
                num_input=num_context,
                pool_size=pool_size,
                pool_stride=pool_stride,
                seed=dense_seed,
                images_subdir=images_subdir,
            )
        except ValueError as e:
            # 场景在对应 camera block 下可用图片数不足等
            print(f"[warn] scene {scene}: 构建 dense-view 候选池失败（{e}），跳过该场景")
            continue

        if not dense_splits:
            print(f"[warn] scene {scene}: 未获取到有效 dense split，跳过")
            continue

        split = dense_splits[0]
        input_paths = split["input_paths"]
        test_paths = split["test_paths"]

        print(f"[info] 输入视角数: {len(input_paths)}, 测试视角数: {len(test_paths)}")

        # Load and preprocess context (input) images
        # For fisheye scenes, try to use processed images; for normal scenes, use original images
        # [-1, 1] -> [0, 1] after process_image
        if is_fisheye_scene:
            ctx_images_raw = [
                process_image(str(get_processed_image_path(p, images_subdir=images_subdir)))
                for p in input_paths
            ]
        else:
            ctx_images_raw = [
                process_image(str(p))
                for p in input_paths
            ]
        ctx_images = torch.stack(ctx_images_raw, dim=0).unsqueeze(0).to(device)  # [1, K, 3, 448, 448]
        ctx_images = (ctx_images + 1) * 0.5  # Convert to [0, 1] for model.encoder

        # Load and preprocess target images
        if is_fisheye_scene:
            tgt_images_raw = [
                process_image(str(get_processed_image_path(p, images_subdir=images_subdir)))
                for p in test_paths
            ]
        else:
            tgt_images_raw = [
                process_image(str(p))
                for p in test_paths
            ]
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
            model=load_local_model(device, local_path=pretrained_model_path),
            ctx_images=ctx_images,
            tgt_images=tgt_images,
            method_name="feed_forward",
            output_folder=ff_output_folder,
            device=device,
            ttt_overrides=ttt_overrides,
        )
        scene_results["feed_forward"] = ff_results

        # Evaluate with self_supervise method
        # print(f"\n--- Evaluating with self_supervise method ---")
        # ss_output_folder = image_run_root / scene / "output_ss"
        # ss_output_folder.mkdir(parents=True, exist_ok=True)
        # ss_results = evaluate_scene_with_method(
        #     model=model,
        #     ctx_images=ctx_images,
        #     tgt_images=tgt_images,
        #     method_name="self_supervise",
        #     output_folder=ss_output_folder,
        #     device=device,
        # )
        # scene_results["self_supervise"] = ss_results

        # Evaluate with ttt method
        print(f"\n--- Evaluating with ttt method ---")
        ttt_output_folder = image_run_root / scene / "output_ttt"
        ttt_output_folder.mkdir(parents=True, exist_ok=True)
        ttt_results = evaluate_scene_with_method(
            model=load_local_model(device, local_path=pretrained_model_path),
            ctx_images=ctx_images,
            tgt_images=tgt_images,
            method_name="ttt",
            output_folder=ttt_output_folder,
            device=device,
            ttt_input_folder=input_images_dir,
            ttt_overrides=ttt_overrides,
        )
        scene_results["ttt"] = ttt_results
        # Evaluate with itr method
        print(f"\n--- Evaluating with itr method ---")
        itr_output_folder = image_run_root / scene / "output_itr"
        itr_output_folder.mkdir(parents=True, exist_ok=True)
        itr_results = evaluate_scene_with_method(
            model=load_local_model(device, local_path=pretrained_model_path),
            ctx_images=ctx_images,
            tgt_images=tgt_images,
            method_name="itr",
            output_folder=itr_output_folder,
            device=device,
            itr_input_folder=input_images_dir,
            itr_overrides=itr_overrides,
        )
        scene_results["itr"] = itr_results

        all_results[scene] = scene_results
        if run is not None:
            wandb.log(
                {
                    "scene_idx": scene_idx,
                    "metrics/feed_forward/psnr": ff_results["psnr"],
                    "metrics/feed_forward/ssim": ff_results["ssim"],
                    "metrics/feed_forward/lpips": ff_results["lpips"],
                    "metrics/ttt/psnr": ttt_results["psnr"],
                    "metrics/ttt/ssim": ttt_results["ssim"],
                    "metrics/ttt/lpips": ttt_results["lpips"],
                    "metrics/itr/psnr": itr_results["psnr"],
                    "metrics/itr/ssim": itr_results["ssim"],
                    "metrics/itr/lpips": itr_results["lpips"],
                },
                step=scene_idx,
            )

        print(f"\n--- Comparison for {scene} ---")
        print(f"Feed Forward:  PSNR={ff_results['psnr']:.2f}, SSIM={ff_results['ssim']:.3f}, LPIPS={ff_results['lpips']:.3f}")
        print(f"TTT:          PSNR={ttt_results['psnr']:.2f}, SSIM={ttt_results['ssim']:.3f}, LPIPS={ttt_results['lpips']:.3f}")
        print(f"ITR:          PSNR={itr_results['psnr']:.2f}, SSIM={itr_results['ssim']:.3f}, LPIPS={itr_results['lpips']:.3f}")

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
    itr_psnr = [r["itr"]["psnr"] for r in all_results.values()]
    itr_ssim = [r["itr"]["ssim"] for r in all_results.values()]
    itr_lpips = [r["itr"]["lpips"] for r in all_results.values()]
    # ss_psnr = [r["self_supervise"]["psnr"] for r in all_results.values()]
    # ss_ssim = [r["self_supervise"]["ssim"] for r in all_results.values()]
    # ss_lpips = [r["self_supervise"]["lpips"] for r in all_results.values()]
    ttt_psnr = [r["ttt"]["psnr"] for r in all_results.values()]
    ttt_ssim = [r["ttt"]["ssim"] for r in all_results.values()]
    ttt_lpips = [r["ttt"]["lpips"] for r in all_results.values()]
    
    avg_ff_psnr = sum(ff_psnr) / len(ff_psnr)
    avg_ff_ssim = sum(ff_ssim) / len(ff_ssim)
    avg_ff_lpips = sum(ff_lpips) / len(ff_lpips)
    # avg_ss_psnr = sum(ss_psnr) / len(ss_psnr)
    # avg_ss_ssim = sum(ss_ssim) / len(ss_ssim)
    # avg_ss_lpips = sum(ss_lpips) / len(ss_lpips)
    avg_ttt_psnr = sum(ttt_psnr) / len(ttt_psnr)
    avg_ttt_ssim = sum(ttt_ssim) / len(ttt_ssim)
    avg_ttt_lpips = sum(ttt_lpips) / len(ttt_lpips)
    avg_itr_psnr = sum(itr_psnr) / len(itr_psnr)
    avg_itr_ssim = sum(itr_ssim) / len(itr_ssim)
    avg_itr_lpips = sum(itr_lpips) / len(itr_lpips)

    print(f"\nAverage Feed Forward:  PSNR={avg_ff_psnr:.2f}, SSIM={avg_ff_ssim:.3f}, LPIPS={avg_ff_lpips:.3f}")
    print(f"Average TTT:          PSNR={avg_ttt_psnr:.2f}, SSIM={avg_ttt_ssim:.3f}, LPIPS={avg_ttt_lpips:.3f}")
    print(f"Average ITR:          PSNR={avg_itr_psnr:.2f}, SSIM={avg_itr_ssim:.3f}, LPIPS={avg_itr_lpips:.3f}")

    # 将指标写入 JSON / 文本文件
    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    # 加载并记录 TTT 和 ITR 的完整配置
    try:
        ttt_base_cfg = OmegaConf.load("config/ttt.yaml")
        ttt_cfg_dict = OmegaConf.to_container(ttt_base_cfg, resolve=True)
        # 应用 overrides
        if ttt_overrides:
            ttt_cfg_dict.update(ttt_overrides)
    except Exception as e:
        ttt_cfg_dict = {"error": f"Failed to load ttt.yaml: {e}"}
    
    try:
        itr_base_cfg = OmegaConf.load("config/itr.yaml")
        itr_cfg_dict = OmegaConf.to_container(itr_base_cfg, resolve=True)
        # 应用 overrides
        if itr_overrides:
            itr_cfg_dict.update(itr_overrides)
    except Exception as e:
        itr_cfg_dict = {"error": f"Failed to load itr.yaml: {e}"}

    # 记录本次实验的关键配置 + 结果
    config_section = {
        "experiment": {
            "num_context": num_context,
            "camera_id": camera_id,
            "fisheye_camera_id": fisheye_camera_id,
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
        "ttt": ttt_cfg_dict,
        "itr": itr_cfg_dict,
    }

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
            # "self_supervise": {
            #     "psnr": avg_ss_psnr,
            #     "ssim": avg_ss_ssim,
            #     "lpips": avg_ss_lpips,
            # },
            "ttt": {
                "psnr": avg_ttt_psnr,
                "ssim": avg_ttt_ssim,
                "lpips": avg_ttt_lpips,
            },
            "itr": {
                "psnr": avg_itr_psnr,
                "ssim": avg_itr_ssim,
                "lpips": avg_itr_lpips,
            },
        },
    }

    json_path = metrics_output_dir / f"metrics_{timestamp}.json"
    txt_path = metrics_output_dir / f"metrics_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Config:\n")
        f.write(f"  num_context      : {num_context}\n")
        f.write(f"  camera_id        : {camera_id}\n")
        f.write(f"  fisheye_camera_id: {fisheye_camera_id}\n")
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
        
        # TTT 配置
        f.write("  ttt:\n")
        if isinstance(ttt_cfg_dict, dict) and "error" not in ttt_cfg_dict:
            for key, val in ttt_cfg_dict.items():
                if key not in ["input_folder", "output_folder", "device"]:
                    f.write(f"    {key:<20s}: {val}\n")
        else:
            f.write(f"    error: {ttt_cfg_dict.get('error', 'Unknown')}\n")
        
        # ITR 配置
        f.write("  itr:\n")
        if isinstance(itr_cfg_dict, dict) and "error" not in itr_cfg_dict:
            for key, val in itr_cfg_dict.items():
                if key not in ["input_folder", "output_folder", "device"]:
                    f.write(f"    {key:<20s}: {val}\n")
        else:
            f.write(f"    error: {itr_cfg_dict.get('error', 'Unknown')}\n")

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
        # f.write(
        #     f"Self Supervise: PSNR={avg_ss_psnr:.2f}, SSIM={avg_ss_ssim:.3f}, LPIPS={avg_ss_lpips:.3f}\n"
        # )
        f.write(
            f"TTT:          PSNR={avg_ttt_psnr:.2f}, SSIM={avg_ttt_ssim:.3f}, LPIPS={avg_ttt_lpips:.3f}\n"
        )
        f.write(
            f"ITR:          PSNR={avg_itr_psnr:.2f}, SSIM={avg_itr_ssim:.3f}, LPIPS={avg_itr_lpips:.3f}\n"
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
                # 记录平均值到同一个 namespace，覆盖循环中最后记录的单个场景值
                "metrics/feed_forward/psnr": avg_ff_psnr,
                "metrics/feed_forward/ssim": avg_ff_ssim,
                "metrics/feed_forward/lpips": avg_ff_lpips,
                "metrics/ttt/psnr": avg_ttt_psnr,
                "metrics/ttt/ssim": avg_ttt_ssim,
                "metrics/ttt/lpips": avg_ttt_lpips,
                "metrics/itr/psnr": avg_itr_psnr,
                "metrics/itr/ssim": avg_itr_ssim,
                "metrics/itr/lpips": avg_itr_lpips,
                # 仍然记录一下在 summary namespace，方便查看详细信息
                "summary/feed_forward/psnr": avg_ff_psnr,
                "summary/feed_forward/ssim": avg_ff_ssim,
                "summary/feed_forward/lpips": avg_ff_lpips,
                # "summary/self_supervise/psnr": avg_ss_psnr,
                # "summary/self_supervise/ssim": avg_ss_ssim,
                # "summary/self_supervise/lpips": avg_ss_lpips,
                "summary/ttt/psnr": avg_ttt_psnr,
                "summary/ttt/ssim": avg_ttt_ssim,
                "summary/ttt/lpips": avg_ttt_lpips,
                "summary/itr/psnr": avg_itr_psnr,
                "summary/itr/ssim": avg_itr_ssim,
                "summary/itr/lpips": avg_itr_lpips,
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
