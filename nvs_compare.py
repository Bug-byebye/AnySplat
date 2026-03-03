from pathlib import Path
import argparse
import json
import os
from dataclasses import fields, replace
from datetime import datetime

import torch
import wandb
from omegaconf import OmegaConf

from itr import ITRConfig, itr, load_itr_config
from scripts.nvs_compare import build_dataset_adapter
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.misc.image_io import save_image
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image
from src.utils.model_loading import load_model_with_fallback
from ttt import TTTConfig, load_ttt_config, run_ttt


def load_local_model(device: torch.device, local_path: str | None = None) -> AnySplat:
    if local_path:
        if not Path(local_path).is_absolute():
            local_path = str(Path.cwd() / local_path)
    return load_model_with_fallback(local_path=local_path, device=device)


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


def load_eval_images(image_paths: list[Path], device: torch.device) -> torch.Tensor:
    images_raw = [process_image(str(p)) for p in image_paths]
    images = torch.stack(images_raw, dim=0).unsqueeze(0).to(device)
    return (images + 1) * 0.5


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

    if method_name not in {"feed_forward", "ttt", "itr"}:
        raise ValueError(f"Unknown method: {method_name}")

    print(f"[{method_name}] Step 1: Reconstructing scene from {v_ctx} context images...")
    encoder_output = model.encoder(ctx_images, global_step=0, visualization_dump={})
    gaussians = encoder_output.gaussians
    pred_context_pose = encoder_output.pred_context_pose

    print(f"[{method_name}] Step 2: Predicting poses for {v_ctx} context + {v_tgt} target images...")
    vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        aggregated_tokens_list, _ = model.encoder.aggregator(
            vggt_input_image,
            intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx,
        )

    with torch.cuda.amp.autocast(enabled=False):
        fp32_tokens = [token.float() for token in aggregated_tokens_list]
        pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
        pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(
            pred_all_pose_enc,
            vggt_input_image.shape[-2:],
        )

    extrinsic_padding = torch.tensor(
        [0, 0, 0, 1],
        device=pred_all_extrinsic.device,
        dtype=pred_all_extrinsic.dtype,
    ).view(1, 1, 1, 4).repeat(b, vggt_input_image.shape[1], 1, 1)
    pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()

    pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
    pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h

    pred_all_context_extrinsic, pred_all_target_extrinsic = (
        pred_all_extrinsic[:, :v_ctx],
        pred_all_extrinsic[:, v_ctx:],
    )
    _, pred_all_target_intrinsic = (
        pred_all_intrinsic[:, :v_ctx],
        pred_all_intrinsic[:, v_ctx:],
    )

    scale_factor = (
        pred_context_pose["extrinsic"][:, :, :3, 3].mean()
        / pred_all_context_extrinsic[:, :, :3, 3].mean()
    )
    pred_all_target_extrinsic[..., :3, 3] = pred_all_target_extrinsic[..., :3, 3] * scale_factor
    pred_all_context_extrinsic[..., :3, 3] = pred_all_context_extrinsic[..., :3, 3] * scale_factor
    print(f"[{method_name}] Scale factor: {scale_factor.item():.4f}")

    print(f"[{method_name}] Step 3: Rendering {v_tgt} target views...")
    output = model.decoder.forward(
        gaussians,
        pred_all_target_extrinsic,
        pred_all_target_intrinsic.float(),
        torch.ones(1, v_tgt, device=device) * 0.01,
        torch.ones(1, v_tgt, device=device) * 100,
        (h, w),
    )

    print(f"[{method_name}] Step 4: Computing metrics...")
    psnr = compute_psnr(tgt_images[0], output.color[0])
    ssim = compute_ssim(tgt_images[0], output.color[0])
    lpips = compute_lpips(tgt_images[0], output.color[0])

    pred_dir = output_folder / "pred"
    gt_dir = output_folder / "gt"
    pred_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    for idx, (gt_image, pred_image) in enumerate(zip(tgt_images[0], output.color[0])):
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


def init_wandb(
    *,
    args,
    dataset_name: str,
    adapter_wandb_cfg: dict,
    num_context: int,
    dense_sparse_cfg: dict,
    scene_names: list[str],
    device: torch.device,
):
    wandb_cfg = None
    try:
        main_cfg = OmegaConf.load("config/main.yaml")
        wandb_cfg = main_cfg.get("wandb", None)
    except Exception:
        wandb_cfg = None

    wandb_mode = "disabled"
    wandb_project = "anysplat-nvs"
    wandb_name = "nvs_compare"
    wandb_tags = ["nvs", dataset_name.replace("-", "_")]

    if wandb_cfg is not None:
        wandb_mode = str(wandb_cfg.get("mode", wandb_mode))
        wandb_project = str(wandb_cfg.get("project", wandb_project))
        wandb_name = f"{wandb_cfg.get('name', wandb_name)}-nvs-compare"
        extra_tags = wandb_cfg.get("tags", None)
        if extra_tags is not None:
            wandb_tags = list(wandb_tags) + list(extra_tags)

    run_name_override = os.environ.get("WANDB_RUN_NAME") or args.wandb_name
    if run_name_override:
        wandb_name = run_name_override

    run = None
    if wandb_mode != "disabled":
        dense_sparse_payload = {
            "pool_size": int(dense_sparse_cfg.get("pool_size", 72)),
            "pool_stride": int(dense_sparse_cfg.get("pool_stride", 2)),
            "seed": int(dense_sparse_cfg.get("seed", 0)),
        }
        if "images_subdir" in dense_sparse_cfg:
            dense_sparse_payload["images_subdir"] = dense_sparse_cfg["images_subdir"]

        wandb_config = {
            "dataset": dataset_name,
            "num_context": num_context,
            "dense_sparse": dense_sparse_payload,
            "device": str(device),
            "scenes": scene_names,
            **adapter_wandb_cfg,
        }
        run = wandb.init(
            project=wandb_project,
            mode=wandb_mode,
            name=wandb_name,
            tags=wandb_tags,
            config=wandb_config,
        )

    return run, wandb_mode, wandb_project, wandb_name, wandb_tags


def main():
    parser = argparse.ArgumentParser(description="NVS compare with wandb logging")
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Weights & Biases run name for this run (overrides config). Can also be set via env WANDB_RUN_NAME.",
    )
    args = parser.parse_args()

    try:
        nvs_cfg = OmegaConf.load("config/nvs_compare.yaml")
    except Exception:
        nvs_cfg = OmegaConf.create({})

    exp_cfg = nvs_cfg.get("experiment", {})
    dataset_name = str(exp_cfg.get("dataset", "vr-nerf"))
    num_context = int(exp_cfg.get("num_context", 32))

    pretrained_model_path = exp_cfg.get("pretrained_model_path", "pretrained_model")
    if not isinstance(pretrained_model_path, str):
        pretrained_model_path = "pretrained_model"

    output_metrics_root_dir = Path(str(exp_cfg.get("output_metrics_dir", "outputs/nvs_compare")))
    output_image_root_dir = Path(str(exp_cfg.get("output_image_root_dir", "exp-results")))

    dense_sparse_cfg = nvs_cfg.get("dense_sparse", {})
    dense_sparse_cfg = OmegaConf.to_container(dense_sparse_cfg, resolve=True) if dense_sparse_cfg else {}
    if not isinstance(dense_sparse_cfg, dict):
        dense_sparse_cfg = {}

    adapter = build_dataset_adapter(nvs_cfg)

    try:
        scene_batches = list(adapter.iter_scene_batches(num_context=num_context, dense_sparse_cfg=dense_sparse_cfg))
    except Exception as e:
        raise RuntimeError(f"Failed to prepare scene batches for dataset '{dataset_name}': {e}") from e

    if not scene_batches:
        raise RuntimeError(f"No valid scenes found for dataset '{dataset_name}'.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    metrics_output_dir = output_metrics_root_dir / f"run_{timestamp}"
    image_run_root = output_image_root_dir / f"{dataset_name.replace('-', '')}_{timestamp}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    run, wandb_mode, wandb_project, wandb_name, wandb_tags = init_wandb(
        args=args,
        dataset_name=dataset_name,
        adapter_wandb_cfg=adapter.get_wandb_config(),
        num_context=num_context,
        dense_sparse_cfg=dense_sparse_cfg,
        scene_names=[x.name for x in scene_batches],
        device=device,
    )

    ttt_overrides = extract_ttt_overrides(nvs_cfg)
    itr_overrides = extract_itr_overrides(nvs_cfg)

    all_results: dict[str, dict] = {}

    for scene_idx, batch in enumerate(scene_batches):
        scene = batch.name
        input_paths = batch.sample.input_paths
        test_paths = batch.sample.test_paths

        print(f"\n{'=' * 60}")
        print(f"Processing scene: {scene}")
        print(f"Dataset: {dataset_name}")
        print(f"Input views: {len(input_paths)} | Test views: {len(test_paths)}")
        print(f"{'=' * 60}")

        ctx_images = load_eval_images(input_paths, device=device)
        tgt_images = load_eval_images(test_paths, device=device)

        input_images_dir = image_run_root / scene / "input_images"
        input_images_dir.mkdir(parents=True, exist_ok=True)
        for idx, img_tensor in enumerate(ctx_images[0]):
            save_image(img_tensor, input_images_dir / f"{idx:06d}.jpg")

        scene_results = {}

        ff_output_folder = image_run_root / scene / "output_ff"
        ff_output_folder.mkdir(parents=True, exist_ok=True)
        ff_results = evaluate_scene_with_method(
            model=load_local_model(device, local_path=pretrained_model_path),
            ctx_images=ctx_images,
            tgt_images=tgt_images,
            method_name="feed_forward",
            output_folder=ff_output_folder,
            device=device,
        )
        scene_results["feed_forward"] = ff_results

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

    print(f"\n{'=' * 60}")
    print("Summary across all scenes:")
    print(f"{'=' * 60}")
    print(f"{'Scene':<20} {'Method':<15} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8}")
    print("-" * 60)
    for scene, results in all_results.items():
        for method, metrics in results.items():
            print(f"{scene:<20} {method:<15} {metrics['psnr']:<8.2f} {metrics['ssim']:<8.3f} {metrics['lpips']:<8.3f}")

    ff_psnr = [r["feed_forward"]["psnr"] for r in all_results.values()]
    ff_ssim = [r["feed_forward"]["ssim"] for r in all_results.values()]
    ff_lpips = [r["feed_forward"]["lpips"] for r in all_results.values()]
    ttt_psnr = [r["ttt"]["psnr"] for r in all_results.values()]
    ttt_ssim = [r["ttt"]["ssim"] for r in all_results.values()]
    ttt_lpips = [r["ttt"]["lpips"] for r in all_results.values()]
    itr_psnr = [r["itr"]["psnr"] for r in all_results.values()]
    itr_ssim = [r["itr"]["ssim"] for r in all_results.values()]
    itr_lpips = [r["itr"]["lpips"] for r in all_results.values()]

    avg_ff_psnr = sum(ff_psnr) / len(ff_psnr)
    avg_ff_ssim = sum(ff_ssim) / len(ff_ssim)
    avg_ff_lpips = sum(ff_lpips) / len(ff_lpips)
    avg_ttt_psnr = sum(ttt_psnr) / len(ttt_psnr)
    avg_ttt_ssim = sum(ttt_ssim) / len(ttt_ssim)
    avg_ttt_lpips = sum(ttt_lpips) / len(ttt_lpips)
    avg_itr_psnr = sum(itr_psnr) / len(itr_psnr)
    avg_itr_ssim = sum(itr_ssim) / len(itr_ssim)
    avg_itr_lpips = sum(itr_lpips) / len(itr_lpips)

    print(f"\nAverage Feed Forward:  PSNR={avg_ff_psnr:.2f}, SSIM={avg_ff_ssim:.3f}, LPIPS={avg_ff_lpips:.3f}")
    print(f"Average TTT:          PSNR={avg_ttt_psnr:.2f}, SSIM={avg_ttt_ssim:.3f}, LPIPS={avg_ttt_lpips:.3f}")
    print(f"Average ITR:          PSNR={avg_itr_psnr:.2f}, SSIM={avg_itr_ssim:.3f}, LPIPS={avg_itr_lpips:.3f}")

    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        ttt_base_cfg = OmegaConf.load("config/ttt.yaml")
        ttt_cfg_dict = OmegaConf.to_container(ttt_base_cfg, resolve=True)
        if ttt_overrides:
            ttt_cfg_dict.update(ttt_overrides)
    except Exception as e:
        ttt_cfg_dict = {"error": f"Failed to load ttt.yaml: {e}"}

    try:
        itr_base_cfg = OmegaConf.load("config/itr.yaml")
        itr_cfg_dict = OmegaConf.to_container(itr_base_cfg, resolve=True)
        if itr_overrides:
            itr_cfg_dict.update(itr_overrides)
    except Exception as e:
        itr_cfg_dict = {"error": f"Failed to load itr.yaml: {e}"}

    config_section = {
        "experiment": {
            "num_context": num_context,
            "metrics_root_dir": str(output_metrics_root_dir),
            "metrics_run_dir": str(metrics_output_dir),
            "image_root_dir": str(output_image_root_dir),
            **adapter.get_metrics_experiment_fields(),
        },
        "dense_sparse": {
            "pool_size": int(dense_sparse_cfg.get("pool_size", 72)),
            "pool_stride": int(dense_sparse_cfg.get("pool_stride", 2)),
            "seed": int(dense_sparse_cfg.get("seed", 0)),
            **({"images_subdir": dense_sparse_cfg.get("images_subdir")} if "images_subdir" in dense_sparse_cfg else {}),
        },
        "ttt": ttt_cfg_dict,
        "itr": itr_cfg_dict,
        "wandb": {
            "mode": wandb_mode,
            "project": wandb_project,
            "name": wandb_name,
            "tags": wandb_tags,
        },
    }

    metrics_payload = {
        "config": config_section,
        "per_scene": all_results,
        "average": {
            "feed_forward": {"psnr": avg_ff_psnr, "ssim": avg_ff_ssim, "lpips": avg_ff_lpips},
            "ttt": {"psnr": avg_ttt_psnr, "ssim": avg_ttt_ssim, "lpips": avg_ttt_lpips},
            "itr": {"psnr": avg_itr_psnr, "ssim": avg_itr_ssim, "lpips": avg_itr_lpips},
        },
    }

    json_path = metrics_output_dir / f"metrics_{timestamp}.json"
    txt_path = metrics_output_dir / f"metrics_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("Config:\n")
        for key, value in config_section["experiment"].items():
            f.write(f"  {key:<16}: {value}\n")
        f.write("  dense_sparse:\n")
        for key, value in config_section["dense_sparse"].items():
            f.write(f"    {key:<14}: {value}\n")
        f.write("\nPer-scene metrics:\n")
        for scene, results in all_results.items():
            for method, m in results.items():
                f.write(
                    f"{scene:20s} {method:15s} "
                    f"PSNR={m['psnr']:.2f} SSIM={m['ssim']:.3f} LPIPS={m['lpips']:.3f}\n"
                )
        f.write("\nAverages:\n")
        f.write(f"Feed Forward:  PSNR={avg_ff_psnr:.2f}, SSIM={avg_ff_ssim:.3f}, LPIPS={avg_ff_lpips:.3f}\n")
        f.write(f"TTT:          PSNR={avg_ttt_psnr:.2f}, SSIM={avg_ttt_ssim:.3f}, LPIPS={avg_ttt_lpips:.3f}\n")
        f.write(f"ITR:          PSNR={avg_itr_psnr:.2f}, SSIM={avg_itr_ssim:.3f}, LPIPS={avg_itr_lpips:.3f}\n")

    if run is not None:
        rows = []
        for scene, results in all_results.items():
            for method, m in results.items():
                rows.append([scene, method, m["psnr"], m["ssim"], m["lpips"]])

        table = wandb.Table(data=rows, columns=["scene", "method", "psnr", "ssim", "lpips"])
        wandb.log(
            {
                "psnr_vs_scene": wandb.plot.line(table, "scene", "psnr", title="PSNR vs Scene", stroke="method"),
                "ssim_vs_scene": wandb.plot.line(table, "scene", "ssim", title="SSIM vs Scene", stroke="method"),
                "lpips_vs_scene": wandb.plot.line(table, "scene", "lpips", title="LPIPS vs Scene", stroke="method"),
                "summary/feed_forward/psnr": avg_ff_psnr,
                "summary/feed_forward/ssim": avg_ff_ssim,
                "summary/feed_forward/lpips": avg_ff_lpips,
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
