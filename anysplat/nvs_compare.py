import sys
from pathlib import Path
import argparse
import json
import os
from dataclasses import fields, replace
from datetime import datetime

import torch
import wandb
from omegaconf import OmegaConf

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, 'anysplat'))

from methods.itr import ITRConfig, itr, load_itr_config
from scripts.nvs_compare import build_dataset_adapter
from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.misc.image_io import save_image
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image
from src.utils.model_loading import load_model_with_fallback
from methods.ttt import TTTConfig, load_ttt_config, run_ttt


SUPPORTED_METHODS = {"feed_forward", "ttt", "itr", "diffi_enhance"}
SUPPORTED_TEST_SAMPLING_MODES = {"uniform", "pose_extreme"}


def _lazy_import_diffi_enhance():
    """
    Diffi enhance depends on external difix3d components and may not be available.
    Import it only when the method is actually requested.
    """
    from methods.diffi_enhance import (  # type: ignore
        DiffiEnhanceConfig,
        diffi_enhance,
        load_diffi_enhance_config,
    )

    return DiffiEnhanceConfig, diffi_enhance, load_diffi_enhance_config


def load_local_model(device: torch.device, local_path: str | None = None) -> AnySplat:
    if local_path:
        if not Path(local_path).is_absolute():
            local_path = str(Path.cwd() / local_path)
    return load_model_with_fallback(local_path=local_path, device=device)


def snapshot_model_state_dict(model: AnySplat) -> dict[str, torch.Tensor]:
    # Keep a CPU copy of baseline weights to restore quickly without reloading from disk.
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_model_from_snapshot(
    model: AnySplat,
    snapshot: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    model.load_state_dict(snapshot, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False


def _deduplicate_pool_paths(paths: list[Path]) -> list[Path]:
    deduped: list[Path] = []
    seen: set[str] = set()
    for p in paths:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    return deduped


def _sample_indices_uniform(n_total: int, n_pick: int, seed: int) -> list[int]:
    if n_pick <= 0:
        return []
    if n_pick > n_total:
        raise ValueError(f"Cannot sample {n_pick} items from only {n_total} candidates")
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    return sorted(perm[:n_pick])


def _compute_pose_order_scores(extrinsic: torch.Tensor) -> torch.Tensor:
    # Use camera-center trajectory projected on principal axis to define two pose extremes.
    centers = extrinsic[:, :3, 3].float()
    centered = centers - centers.mean(dim=0, keepdim=True)
    if centers.shape[0] < 2 or torch.allclose(centered, torch.zeros_like(centered)):
        return centers[:, 0]
    _, _, vh = torch.linalg.svd(centered, full_matrices=False)
    principal_axis = vh[0]
    return centered @ principal_axis


def resample_scene_views_pose_extreme(
    model: AnySplat,
    pool_paths: list[Path],
    num_context: int,
    num_test: int,
    seed: int,
    device: torch.device,
) -> tuple[list[Path], list[Path]]:
    if num_test <= 0:
        raise ValueError("num_test must be > 0 for pose_extreme sampling")
    if len(pool_paths) < (num_context + num_test):
        raise ValueError(
            f"pool size {len(pool_paths)} is smaller than num_context + num_test = {num_context + num_test}"
        )

    pool_images = load_eval_images(pool_paths, device=device)
    with torch.no_grad():
        _, pred_context_pose = model.inference(pool_images)

    extrinsic = pred_context_pose["extrinsic"][0]
    if extrinsic.shape[0] != len(pool_paths):
        raise RuntimeError(
            f"pose count {extrinsic.shape[0]} does not match pool size {len(pool_paths)}"
        )

    scores = _compute_pose_order_scores(extrinsic)
    sorted_idx = torch.argsort(scores).tolist()

    # Half of test views come from both ends of sorted poses; the rest are random from remaining views.
    extreme_quota = num_test // 2
    left_quota = extreme_quota // 2
    right_quota = extreme_quota - left_quota

    left_idx = sorted_idx[:left_quota] if left_quota > 0 else []
    right_idx = sorted_idx[-right_quota:] if right_quota > 0 else []

    test_idx_set: set[int] = set(left_idx + right_idx)
    remaining_test_quota = num_test - len(test_idx_set)

    remaining_idx = [i for i in range(len(pool_paths)) if i not in test_idx_set]
    if remaining_test_quota > len(remaining_idx):
        raise ValueError(
            f"Not enough remaining views for random test sampling: need {remaining_test_quota}, have {len(remaining_idx)}"
        )

    if remaining_test_quota > 0:
        sampled_local_idx = _sample_indices_uniform(
            n_total=len(remaining_idx),
            n_pick=remaining_test_quota,
            seed=seed + 17,
        )
        for li in sampled_local_idx:
            test_idx_set.add(remaining_idx[li])

    test_indices = sorted(test_idx_set)
    input_candidates = [i for i in range(len(pool_paths)) if i not in test_idx_set]

    if num_context > len(input_candidates):
        raise ValueError(
            f"Not enough views left for context: need {num_context}, have {len(input_candidates)}"
        )

    chosen_ctx_local = _sample_indices_uniform(
        n_total=len(input_candidates),
        n_pick=num_context,
        seed=seed + 29,
    )
    input_indices = sorted(input_candidates[i] for i in chosen_ctx_local)

    input_paths = [pool_paths[i] for i in input_indices]
    test_paths = [pool_paths[i] for i in test_indices]
    return input_paths, test_paths


def extract_ttt_overrides(cfg) -> dict:
    if cfg is None:
        return {}
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        return {}
    
    # Try to extract from 'ttt' section first, then from 'experiment' section
    ttt_section = cfg_dict.get("ttt", {})
    exp_section = cfg_dict.get("experiment", {})
    
    # Merge both sections, with ttt section taking precedence
    candidate = {**exp_section, **ttt_section}
    
    valid_keys = {field.name for field in fields(TTTConfig)}
    return {k: v for k, v in candidate.items() if k in valid_keys}


def extract_itr_overrides(cfg) -> dict:
    if cfg is None:
        return {}
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        return {}
    
    # Try to extract from 'itr' section first, then from 'experiment' section
    itr_section = cfg_dict.get("itr", {})
    exp_section = cfg_dict.get("experiment", {})
    
    # Merge both sections, with itr section taking precedence
    candidate = {**exp_section, **itr_section}
    
    valid_keys = {field.name for field in fields(ITRConfig)}
    return {k: v for k, v in candidate.items() if k in valid_keys}


def extract_diffi_overrides(cfg) -> dict:
    # DiffiEnhanceConfig may be unavailable if difix3d deps are not installed.
    try:
        DiffiEnhanceConfig, _, _ = _lazy_import_diffi_enhance()
    except Exception:
        return {}
    if cfg is None:
        return {}
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg_dict, dict):
        return {}

    # Try to extract from 'diffi_enhance' section first, then from 'experiment' section
    diffi_section = cfg_dict.get("diffi_enhance", {})
    exp_section = cfg_dict.get("experiment", {})

    # Merge both sections, with diffi_enhance section taking precedence
    candidate = {**exp_section, **diffi_section}

    valid_keys = {field.name for field in fields(DiffiEnhanceConfig)}
    return {k: v for k, v in candidate.items() if k in valid_keys}


def load_eval_images(image_paths: list[Path], device: torch.device) -> torch.Tensor:
    images_raw = [process_image(str(p)) for p in image_paths]
    images = torch.stack(images_raw, dim=0).unsqueeze(0).to(device)
    return (images + 1) * 0.5


def prepare_output_root(path_value: str, fallback_relative: str, label: str) -> Path:
    configured_path = Path(path_value)
    target_path = configured_path if configured_path.is_absolute() else (Path.cwd() / configured_path)
    try:
        target_path.mkdir(parents=True, exist_ok=True)
        return target_path
    except PermissionError:
        fallback_path = (Path.cwd() / fallback_relative)
        fallback_path.mkdir(parents=True, exist_ok=True)
        print(
            f"[warn] Cannot write {label} to '{target_path}' due to permission. "
            f"Falling back to '{fallback_path}'."
        )
        return fallback_path


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
    diffi_input_folder: Path | None = None,
    diffi_overrides: dict | None = None,
):
    if method_name == "ttt":
        if ttt_input_folder is None:
            raise ValueError("ttt_input_folder is required when method_name='ttt'")
        ttt_output_folder = output_folder / "ttt_outputs"
        ttt_output_folder.mkdir(parents=True, exist_ok=True)
        ttt_cfg = load_ttt_config(Path(_PROJECT_ROOT) / "anysplat/config/ttt.yaml")
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
        itr_cfg = load_itr_config(Path(_PROJECT_ROOT) / "anysplat/config/itr.yaml")
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

    if method_name == "diffi_enhance":
        if diffi_input_folder is None:
            raise ValueError("diffi_input_folder is required when method_name='diffi_enhance'")
        try:
            _, diffi_enhance, load_diffi_enhance_config = _lazy_import_diffi_enhance()
        except Exception as e:
            raise RuntimeError(
                "diffi_enhance method requested but DifixPipeline/difix3d dependency is not available. "
                "Please install/enable difix3d or remove 'diffi_enhance' from experiment.methods."
            ) from e
        diffi_output_folder = output_folder / "diffi_enhance_outputs"
        diffi_output_folder.mkdir(parents=True, exist_ok=True)
        diffi_cfg = load_diffi_enhance_config(Path(_PROJECT_ROOT) / "anysplat/config/diffi_enhance.yaml")
        if diffi_overrides:
            diffi_cfg = replace(diffi_cfg, **diffi_overrides)
        diffi_cfg = replace(
            diffi_cfg,
            input_folder=diffi_input_folder,
            output_folder=diffi_output_folder,
            device=str(device),
        )
        print(f"[{method_name}] Running Diffi Enhance with input folder: {diffi_input_folder}")
        enhanced_virtual = diffi_enhance(model, diffi_cfg)
        if enhanced_virtual is not None and enhanced_virtual.shape[1] > 0:
            # Evaluate with "original context + enhanced virtual views".
            enhanced_virtual = enhanced_virtual.to(device)
            ctx_images = torch.cat([ctx_images, enhanced_virtual], dim=1)
            print(
                f"[{method_name}] Using augmented context for evaluation: "
                f"original={ctx_images.shape[1] - enhanced_virtual.shape[1]}, "
                f"enhanced_virtual={enhanced_virtual.shape[1]}, total={ctx_images.shape[1]}"
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

    if method_name not in {"feed_forward", "ttt", "itr", "diffi_enhance"}:
        raise ValueError(f"Unknown method: {method_name}")

    b, v_ctx, _, h, w = ctx_images.shape
    _, v_tgt, _, _, _ = tgt_images.shape

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
        main_cfg = OmegaConf.load(os.path.join(_PROJECT_ROOT, "anysplat/config/main.yaml"))
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
        nvs_cfg = OmegaConf.load(os.path.join(_PROJECT_ROOT, "anysplat/config/nvs_compare.yaml"))
    except Exception:
        print("Error loading nvs_compare.yaml")
        nvs_cfg = OmegaConf.create({})

    exp_cfg = nvs_cfg.get("experiment", {})
    dataset_name = str(exp_cfg.get("dataset"))
    num_context = int(exp_cfg.get("num_context", 32))
    configured_methods = exp_cfg.get("methods", ["feed_forward", "ttt", "diffi_enhance"])
    if configured_methods is None:
        configured_methods = ["feed_forward", "ttt", "diffi_enhance"]
    methods = [str(m) for m in configured_methods]
    if not methods:
        raise ValueError("experiment.methods cannot be empty")
    invalid_methods = [m for m in methods if m not in SUPPORTED_METHODS]
    if invalid_methods:
        raise ValueError(
            f"Unsupported methods in experiment.methods: {invalid_methods}. "
            f"Supported methods: {sorted(SUPPORTED_METHODS)}"
        )

    pretrained_model_path = exp_cfg.get("pretrained_model_path", "anysplat/pretrained_model")
    if not isinstance(pretrained_model_path, str):
        pretrained_model_path = "anysplat/pretrained_model"

    output_metrics_root_dir = prepare_output_root(
        str(exp_cfg.get("output_metrics_dir", "outputs/nvs_compare")),
        "outputs/nvs_compare",
        "metrics output",
    )
    output_image_root_dir = prepare_output_root(
        str(exp_cfg.get("output_image_root_dir", "exp-results")),
        "exp-results",
        "image output",
    )

    # Build dense_sparse_cfg from experiment section (parameters moved from dense_sparse)
    dense_sparse_cfg = {
        "pool_size": int(exp_cfg.get("pool_size", 72)),
        "pool_stride": int(exp_cfg.get("pool_stride", 2)),
        "num_test": int(exp_cfg.get("num_test", 8)),
        "seed": int(exp_cfg.get("seed", 0)),
    }
    test_sampling_mode = str(exp_cfg.get("test_sampling_mode", "uniform")).strip().lower()
    if test_sampling_mode not in SUPPORTED_TEST_SAMPLING_MODES:
        raise ValueError(
            f"Unsupported experiment.test_sampling_mode='{test_sampling_mode}'. "
            f"Supported modes: {sorted(SUPPORTED_TEST_SAMPLING_MODES)}"
        )

    adapter = build_dataset_adapter(nvs_cfg)

    try:
        scene_batches = list(adapter.iter_scene_batches(num_context=num_context, dense_sparse_cfg=dense_sparse_cfg))
    except Exception as e:
        raise RuntimeError(f"Failed to prepare scene batches for dataset '{dataset_name}': {e}") from e

    if not scene_batches:
        raise RuntimeError(f"No valid scenes found for dataset '{dataset_name}'.")

    max_scenes_raw = exp_cfg.get("max_scenes", None)
    if max_scenes_raw is not None and str(max_scenes_raw).lower() != "null":
        try:
            max_scenes = int(max_scenes_raw)
        except Exception:
            max_scenes = None
        if max_scenes is not None:
            if max_scenes <= 0:
                raise ValueError("experiment.max_scenes must be > 0 or null")
            scene_batches = scene_batches[:max_scenes]

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

    print(f"[nvs_compare] Enabled methods: {methods}")
    print(f"[nvs_compare] Test sampling mode: {test_sampling_mode}")

    exp_cfg_dict = OmegaConf.to_container(exp_cfg, resolve=True) if exp_cfg else {}
    if not isinstance(exp_cfg_dict, dict):
        exp_cfg_dict = {}

    ttt_overrides = extract_ttt_overrides(nvs_cfg)
    itr_overrides = extract_itr_overrides(nvs_cfg)
    diffi_overrides = extract_diffi_overrides(nvs_cfg)

    print("[nvs_compare] Loading AnySplat model once...")
    shared_model = load_local_model(device, local_path=pretrained_model_path)
    base_model_snapshot = snapshot_model_state_dict(shared_model)
    print("[nvs_compare] AnySplat model loaded and baseline snapshot cached in memory.")

    all_results: dict[str, dict] = {}

    for scene_idx, batch in enumerate(scene_batches):
        scene = batch.name
        input_paths = batch.sample.input_paths
        test_paths = batch.sample.test_paths

        if test_sampling_mode == "pose_extreme":
            restore_model_from_snapshot(shared_model, base_model_snapshot, device)
            pool_paths = _deduplicate_pool_paths(list(input_paths) + list(test_paths))
            if len(pool_paths) < (num_context + dense_sparse_cfg["num_test"]):
                raise RuntimeError(
                    f"Scene '{scene}' has insufficient pooled views ({len(pool_paths)}) for "
                    f"num_context={num_context}, num_test={dense_sparse_cfg['num_test']}"
                )
            input_paths, test_paths = resample_scene_views_pose_extreme(
                model=shared_model,
                pool_paths=pool_paths,
                num_context=num_context,
                num_test=dense_sparse_cfg["num_test"],
                seed=dense_sparse_cfg["seed"] + scene_idx,
                device=device,
            )
            print(
                f"[nvs_compare] pose_extreme resample -> pool={len(pool_paths)}, "
                f"input={len(input_paths)}, test={len(test_paths)}"
            )

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

        for method_name in methods:
            method_output_folder = image_run_root / scene / f"output_{method_name}"
            method_output_folder.mkdir(parents=True, exist_ok=True)

            # Ensure each method/scene starts from the same baseline weights.
            restore_model_from_snapshot(shared_model, base_model_snapshot, device)

            method_kwargs = {
                "model": shared_model,
                "ctx_images": ctx_images,
                "tgt_images": tgt_images,
                "method_name": method_name,
                "output_folder": method_output_folder,
                "device": device,
            }

            if method_name == "ttt":
                method_kwargs["ttt_input_folder"] = input_images_dir
                method_kwargs["ttt_overrides"] = ttt_overrides
            elif method_name == "itr":
                method_kwargs["itr_input_folder"] = input_images_dir
                method_kwargs["itr_overrides"] = itr_overrides
            elif method_name == "diffi_enhance":
                method_kwargs["diffi_input_folder"] = input_images_dir
                method_kwargs["diffi_overrides"] = diffi_overrides

            method_results = evaluate_scene_with_method(**method_kwargs)

            if method_name == "diffi_enhance":
                # Load virtual view -> reference view mapping if available
                diffi_enhance_outputs = method_output_folder / "diffi_enhance_outputs"
                if diffi_enhance_outputs.exists():
                    mapping_file = diffi_enhance_outputs / "virtual_ref_mapping.json"
                    if mapping_file.exists():
                        with mapping_file.open("r") as f:
                            virtual_ref_mapping = json.load(f)
                        print(
                            f"[nvs_compare] Loaded {len(virtual_ref_mapping)} "
                            f"virtual-reference view mappings for {scene}"
                        )
                        method_results["virtual_ref_mapping"] = virtual_ref_mapping

            scene_results[method_name] = method_results

        all_results[scene] = scene_results

        if run is not None:
            wandb_payload = {"scene_idx": scene_idx}
            for method_name in methods:
                method_metrics = scene_results[method_name]
                wandb_payload[f"metrics/{method_name}/psnr"] = method_metrics["psnr"]
                wandb_payload[f"metrics/{method_name}/ssim"] = method_metrics["ssim"]
                wandb_payload[f"metrics/{method_name}/lpips"] = method_metrics["lpips"]
            wandb.log(wandb_payload, step=scene_idx)

    print(f"\n{'=' * 60}")
    print("Summary across all scenes:")
    print(f"{'=' * 60}")
    print(f"{'Scene':<20} {'Method':<15} {'PSNR':<8} {'SSIM':<8} {'LPIPS':<8}")
    print("-" * 60)
    for scene, results in all_results.items():
        for method, metrics in results.items():
            print(f"{scene:<20} {method:<15} {metrics['psnr']:<8.2f} {metrics['ssim']:<8.3f} {metrics['lpips']:<8.3f}")

    avg_metrics = {}
    for method_name in methods:
        psnr_values = [r[method_name]["psnr"] for r in all_results.values()]
        ssim_values = [r[method_name]["ssim"] for r in all_results.values()]
        lpips_values = [r[method_name]["lpips"] for r in all_results.values()]
        avg_metrics[method_name] = {
            "psnr": sum(psnr_values) / len(psnr_values),
            "ssim": sum(ssim_values) / len(ssim_values),
            "lpips": sum(lpips_values) / len(lpips_values),
        }

    print("\nAverage metrics:")
    for method_name in methods:
        m = avg_metrics[method_name]
        print(f"  {method_name:<15} PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.3f}, LPIPS={m['lpips']:.3f}")

    metrics_output_dir.mkdir(parents=True, exist_ok=True)

    # Load nvs_compare experiment configuration
    nvs_experiment_cfg = {}
    if nvs_cfg is not None and nvs_cfg.get("experiment") is not None:
        nvs_experiment_cfg = OmegaConf.to_container(nvs_cfg.get("experiment"), resolve=True)
        if not isinstance(nvs_experiment_cfg, dict):
            nvs_experiment_cfg = {}
    
    # Load dataset-specific configuration
    dataset_cfg = {}
    if nvs_cfg is not None and dataset_name:
        dataset_section = nvs_cfg.get(dataset_name)
        if dataset_section is not None:
            dataset_cfg = OmegaConf.to_container(dataset_section, resolve=True)
            if not isinstance(dataset_cfg, dict):
                dataset_cfg = {}
    
    # Load ttt.yaml
    try:
        ttt_base_cfg = OmegaConf.load(os.path.join(_PROJECT_ROOT, "anysplat/config/ttt.yaml"))
        ttt_cfg_dict = OmegaConf.to_container(ttt_base_cfg, resolve=True)
        if ttt_overrides:
            ttt_cfg_dict.update(ttt_overrides)
    except Exception as e:
        ttt_cfg_dict = {"error": f"Failed to load ttt.yaml: {e}"}

    # Load itr.yaml
    try:
        itr_base_cfg = OmegaConf.load(os.path.join(_PROJECT_ROOT, "anysplat/config/itr.yaml"))
        itr_cfg_dict = OmegaConf.to_container(itr_base_cfg, resolve=True)
        if itr_overrides:
            itr_cfg_dict.update(itr_overrides)
    except Exception as e:
        itr_cfg_dict = {"error": f"Failed to load itr.yaml: {e}"}

    # Load diffi_enhance.yaml
    try:
        # Only include diffi_enhance config when dependency is importable.
        _lazy_import_diffi_enhance()
        diffi_base_cfg = OmegaConf.load(os.path.join(_PROJECT_ROOT, "anysplat/config/diffi_enhance.yaml"))
        diffi_cfg_dict = OmegaConf.to_container(diffi_base_cfg, resolve=True)
        if diffi_overrides:
            diffi_cfg_dict.update(diffi_overrides)
    except Exception as e:
        diffi_cfg_dict = {"error": f"Failed to load diffi_enhance.yaml: {e}"}

    config_section = {
        "experiment": nvs_experiment_cfg,
        f"dataset_config_{dataset_name}": dataset_cfg,
        "dense_sparse": {
            "pool_size": int(dense_sparse_cfg.get("pool_size", 72)),
            "pool_stride": int(dense_sparse_cfg.get("pool_stride", 2)),
            "seed": int(dense_sparse_cfg.get("seed", 0)),
            **({"images_subdir": dense_sparse_cfg.get("images_subdir")} if "images_subdir" in dense_sparse_cfg else {}),
        },
        "ttt_config": ttt_cfg_dict,
        "itr_config": itr_cfg_dict,
        "diffi_enhance_config": diffi_cfg_dict,
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
        "average": avg_metrics,
    }

    json_path = metrics_output_dir / f"metrics_{timestamp}.json"
    txt_path = metrics_output_dir / f"metrics_{timestamp}.txt"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=False)

    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Configuration Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Experiment Config (nvs_compare.yaml::experiment):\n")
        f.write("-" * 80 + "\n")
        for key, value in config_section["experiment"].items():
            f.write(f"  {key:<24}: {value}\n")
        
        f.write(f"\nDataset-Specific Config ({dataset_name}):\n")
        f.write("-" * 80 + "\n")
        for key, value in config_section.get(f"dataset_config_{dataset_name}", {}).items():
            f.write(f"  {key:<24}: {value}\n")
        
        f.write("\nDense-Sparse Sampling:\n")
        f.write("-" * 80 + "\n")
        for key, value in config_section["dense_sparse"].items():
            f.write(f"  {key:<24}: {value}\n")
        
        f.write("\nTTT Config (config/ttt.yaml):\n")
        f.write("-" * 80 + "\n")
        for key, value in config_section.get("ttt_config", {}).items():
            f.write(f"  {key:<24}: {value}\n")
        
        f.write("\nITR Config (config/itr.yaml):\n")
        f.write("-" * 80 + "\n")
        for key, value in config_section.get("itr_config", {}).items():
            f.write(f"  {key:<24}: {value}\n")

        f.write("\nDiffi Enhance Config (config/diffi_enhance.yaml):\n")
        f.write("-" * 80 + "\n")
        for key, value in config_section.get("diffi_enhance_config", {}).items():
            f.write(f"  {key:<24}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("\nPer-scene metrics:\n")
        f.write("=" * 80 + "\n")
        for scene, results in all_results.items():
            for method, m in results.items():
                f.write(
                    f"{scene:20s} {method:15s} "
                    f"PSNR={m['psnr']:.2f} SSIM={m['ssim']:.3f} LPIPS={m['lpips']:.3f}\n"
                )
        f.write("\nAverages:\n")
        for method_name in methods:
            m = avg_metrics[method_name]
            f.write(
                f"{method_name:15s} "
                f"PSNR={m['psnr']:.2f}, SSIM={m['ssim']:.3f}, LPIPS={m['lpips']:.3f}\n"
            )

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
                **{
                    f"summary/{method_name}/psnr": avg_metrics[method_name]["psnr"]
                    for method_name in methods
                },
                **{
                    f"summary/{method_name}/ssim": avg_metrics[method_name]["ssim"]
                    for method_name in methods
                },
                **{
                    f"summary/{method_name}/lpips": avg_metrics[method_name]["lpips"]
                    for method_name in methods
                },
            }
        )
        wandb.finish()


if __name__ == "__main__":
    main()
