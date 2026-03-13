from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.model.model.anysplat import AnySplat
from src.model.ply_export import export_ply
from src.evaluation.metrics import get_lpips
from src.misc.image_io import save_image
from src.utils.model_loading import load_model_with_fallback
from methods.self_supervise import pose_interpolation
from methods.ttt import (
	get_ttt_parameters,
	group_images,
	list_image_paths,
	load_image_tensors,
	render_views,
	select_train_test_images,
	split_image_paths_evenly,
)


@dataclass
class ITRConfig:
	input_folder: Path
	output_folder: Optional[Path]
	iters: int = 5
	image_sorted: bool = True
	group_size: int = 8
	interp_frames: int = 4
	num_train: Optional[int] = None
	train_distribution: str = "uniform"
	seed: int = 0
	lr: float = 1e-4
	train_components: Optional[List[str]] = None
	device: str = "auto"
	export_ply: bool = True
	loss_type: str = "l1_mse"
	l1_weight: float = 1.0
	mse_weight: float = 1.0
	lpips_weight: float = 0.1
	context_gt_weight: float = 1.0
	context_consistency_weight: float = 1.0
	target_consistency_weight: float = 0.1
	pretrained_model_path: Optional[str] = None
	target_consistency_weight: float = 1.0


def load_itr_config(config_path: Path) -> ITRConfig:
	if not config_path.exists():
		raise FileNotFoundError(f"ITR config not found: {config_path}")

	raw_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
	if not isinstance(raw_cfg, dict):
		raise ValueError(f"Invalid ITR config format: {config_path}")

	cfg = ITRConfig(**raw_cfg)

	input_folder = Path(cfg.input_folder).expanduser().resolve()
	if cfg.output_folder is None:
		output_root = (input_folder / "itr_outputs").expanduser().resolve()
	else:
		output_root = Path(cfg.output_folder).expanduser().resolve()

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	output_folder = output_root / f"itr_{timestamp}"

	if cfg.device in (None, "auto"):
		device = "cuda" if torch.cuda.is_available() else "cpu"
	else:
		device = str(cfg.device)

	return replace(
		cfg,
		input_folder=input_folder,
		output_folder=output_folder,
		device=device,
	)


def _compute_basic_losses(
	pred: torch.Tensor,
	target: torch.Tensor,
	lpips_fn,
	use_lpips: bool,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
	loss_l1 = F.l1_loss(pred, target)
	loss_mse = F.mse_loss(pred, target)
	loss_lpips = None
	if use_lpips and lpips_fn is not None:
		loss_lpips = lpips_fn(pred, target, normalize=True).mean()
	return loss_l1, loss_mse, loss_lpips


def itr(model: AnySplat, cfg: ITRConfig) -> None:
	device = cfg.device
	model = model.to(device)

	for p in model.parameters():
		p.requires_grad = False
	itr_params = get_ttt_parameters(model, train_components=cfg.train_components)
	for p in itr_params:
		p.requires_grad = True

	optimizer = torch.optim.Adam(itr_params, lr=cfg.lr)
	itr_params_snapshot = [p.detach().clone() for p in itr_params]

	use_lpips = cfg.loss_type.lower() in {"lpips", "l1_mse_lpips"}
	lpips_fn = get_lpips(device) if use_lpips else None

	image_paths = list_image_paths(cfg.input_folder, cfg.image_sorted)
	if len(image_paths) < 2:
		raise ValueError("ITR requires at least 2 input images.")

	train_paths, test_paths = select_train_test_images(
		image_paths=image_paths,
		num_train=cfg.num_train,
		train_distribution=cfg.train_distribution,
		seed=cfg.seed,
	)
	train_images = load_image_tensors(train_paths)
	test_images = load_image_tensors(test_paths) if test_paths else []

	print(
		f"[itr] total images: {len(image_paths)} | "
		f"train set: {len(train_images)} ({cfg.train_distribution}) | "
		f"test set: {len(test_images)}"
	)

	groups = group_images(train_images, cfg.group_size)
	print(f"[itr] train set grouped by {cfg.group_size}, groups: {len(groups)}")

	total_groups = len(groups)
	for it in range(cfg.iters):
		print(f"\n[itr] ===== Epoch {it + 1}/{cfg.iters} =====")
		rng = random.Random(cfg.seed + it)

		if it < total_groups:
			selected_groups = [(it, groups[it])]
		else:
			random_idx = rng.randrange(total_groups)
			selected_groups = [(random_idx, groups[random_idx])]

		for group_idx, group in selected_groups:
			context_images = group
			ctx_tensor = torch.stack(context_images, dim=0).unsqueeze(0).to(device)
			ctx_01 = (ctx_tensor + 1.0) * 0.5
			_, v_ctx, _, h, w = ctx_01.shape

			model.train()
			optimizer.zero_grad(set_to_none=True)

			encoder_output_1 = model.encoder(ctx_01, global_step=0, visualization_dump={})
			gaussians_1 = encoder_output_1.gaussians
			pred_context_pose_1 = encoder_output_1.pred_context_pose

			y_ctx_1 = render_views(
				model.decoder,
				gaussians_1,
				pred_context_pose_1["extrinsic"],
				pred_context_pose_1["intrinsic"],
				(h, w),
			)

			random.seed(cfg.seed + it * 100 + group_idx)
			interp_pose = pose_interpolation(
				pred_context_pose_1,
				num_interp_frames=cfg.interp_frames,
				image_sorted=cfg.image_sorted,
			)
			y_tgt_1 = render_views(
				model.decoder,
				gaussians_1,
				interp_pose["extrinsic"],
				interp_pose["intrinsic"],
				(h, w),
			)

			combined_01 = torch.cat([ctx_01, y_tgt_1.unsqueeze(0)], dim=1)
			encoder_output_2 = model.encoder(combined_01, global_step=0, visualization_dump={})
			gaussians_2 = encoder_output_2.gaussians
			pred_all_pose_2 = encoder_output_2.pred_context_pose

			pred_ctx_pose_2 = {
				"extrinsic": pred_all_pose_2["extrinsic"][:, :v_ctx],
				"intrinsic": pred_all_pose_2["intrinsic"][:, :v_ctx],
			}
			pred_tgt_pose_2 = {
				"extrinsic": pred_all_pose_2["extrinsic"][:, v_ctx:],
				"intrinsic": pred_all_pose_2["intrinsic"][:, v_ctx:],
			}

			y_ctx_2 = render_views(
				model.decoder,
				gaussians_2,
				pred_ctx_pose_2["extrinsic"],
				pred_ctx_pose_2["intrinsic"],
				(h, w),
			)
			y_tgt_2 = render_views(
				model.decoder,
				gaussians_2,
				pred_tgt_pose_2["extrinsic"],
				pred_tgt_pose_2["intrinsic"],
				(h, w),
			)

			render_dir = cfg.output_folder / "train_renders" / f"epoch_{it + 1:04d}" / f"group_{group_idx + 1:04d}"
			render_dir.mkdir(parents=True, exist_ok=True)
			for view_idx, img in enumerate(y_ctx_1):
				save_image(img, render_dir / f"ctx1_{view_idx:03d}.jpg")
			for view_idx, img in enumerate(y_tgt_1):
				save_image(img, render_dir / f"tgt1_{view_idx:03d}.jpg")
			for view_idx, img in enumerate(y_ctx_2):
				save_image(img, render_dir / f"ctx2_{view_idx:03d}.jpg")
			for view_idx, img in enumerate(y_tgt_2):
				save_image(img, render_dir / f"tgt2_{view_idx:03d}.jpg")

			ctx_gt = ctx_01[0]
			y_ctx_1_detached = y_ctx_1.detach()
			y_tgt_1_detached = y_tgt_1.detach()

			ctx_cons_l1, ctx_cons_mse, ctx_cons_lpips = _compute_basic_losses(
				y_ctx_2, y_ctx_1_detached, lpips_fn, False
			)
			tgt_cons_l1, tgt_cons_mse, tgt_cons_lpips = _compute_basic_losses(
				y_tgt_2, y_tgt_1_detached, lpips_fn, False
			)
			ctx_gt1_l1, ctx_gt1_mse, ctx_gt1_lpips = _compute_basic_losses(
				y_ctx_1, ctx_gt, lpips_fn, use_lpips
			)
			ctx_gt2_l1, ctx_gt2_mse, ctx_gt2_lpips = _compute_basic_losses(
				y_ctx_2, ctx_gt, lpips_fn, use_lpips
			)

			loss = cfg.l1_weight * (
				cfg.context_consistency_weight * ctx_cons_l1
				+ cfg.target_consistency_weight * tgt_cons_l1
				+ cfg.context_gt_weight * (ctx_gt1_l1 + ctx_gt2_l1)
			) + cfg.mse_weight * (
				cfg.context_consistency_weight * ctx_cons_mse
				+ cfg.target_consistency_weight * tgt_cons_mse
				+ cfg.context_gt_weight * (ctx_gt1_mse + ctx_gt2_mse)
			)

			if use_lpips:
				loss = loss + cfg.lpips_weight * (
					cfg.context_gt_weight * (ctx_gt1_lpips + ctx_gt2_lpips)
				)

			loss.backward()
			if group_idx == 0:
				has_grad = any(
					(p.grad is not None) and torch.isfinite(p.grad).all()
					for p in itr_params
				)
				print(f"[itr] epoch {it + 1} has_grad={has_grad}")

			optimizer.step()
			max_delta = max(
				(p.detach() - b).abs().max().item()
				for p, b in zip(itr_params, itr_params_snapshot)
			)
			print(f"[itr] epoch {it + 1} max_param_delta={max_delta:.6e}")

			prefix = f"[itr] epoch {it + 1} group {group_idx + 1}/{len(groups)}"
			if not use_lpips:
				print(
					f"{prefix}: loss={loss.item():.6f} "
					f"(CTX_CONS_L1={ctx_cons_l1.item():.6f}, TGT_CONS_L1={tgt_cons_l1.item():.6f}, "
					f"CTX_GT1_L1={ctx_gt1_l1.item():.6f}, CTX_GT2_L1={ctx_gt2_l1.item():.6f})"
				)
			else:
				print(
					f"{prefix}: loss={loss.item():.6f} "
					f"(CTX_CONS_L1={ctx_cons_l1.item():.6f}, TGT_CONS_L1={tgt_cons_l1.item():.6f}, "
					f"CTX_GT1_L1={ctx_gt1_l1.item():.6f}, CTX_GT2_L1={ctx_gt2_l1.item():.6f}, "
					f"CTX_GT1_LPIPS={ctx_gt1_lpips.item():.6f}, CTX_GT2_LPIPS={ctx_gt2_lpips.item():.6f})"
				)

	print("\n[itr] ITR done, reconstructing with all images...")
	model.eval()
	with torch.no_grad():
		all_images = load_image_tensors(image_paths)
		all_tensor = torch.stack(all_images, dim=0).unsqueeze(0).to(device)
		all_01 = (all_tensor + 1.0) * 0.5
		encoder_output = model.encoder(all_01, global_step=0, visualization_dump={})
		gaussians_final = encoder_output.gaussians

	if cfg.export_ply:
		ply_path = cfg.output_folder / "gaussians_itr.ply"
		export_ply(
			gaussians_final.means[0],
			gaussians_final.scales[0],
			gaussians_final.rotations[0],
			gaussians_final.harmonics[0],
			gaussians_final.opacities[0],
			ply_path,
		)
		print(f"[itr] exported final gaussians to: {ply_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="AnySplat Iterative Test-time Refinement (ITR)")
	parser.add_argument(
		"--config",
		type=str,
		default=str(Path("config/itr.yaml")),
		help="ITR YAML config path.",
	)
	return parser.parse_args()


def main() -> None:

	cfg = load_itr_config(Path("config/itr.yaml").expanduser().resolve())
	cfg.output_folder.mkdir(parents=True, exist_ok=True)
	print(cfg.train_components)

	device = torch.device(cfg.device)

	print(f"[itr] device: {device}")
	print(f"[itr] input folder: {cfg.input_folder}")
	print(f"[itr] output folder: {cfg.output_folder}")

	print("[itr] loading AnySplat pretrained model...")

	local_model_path = cfg.pretrained_model_path
	if local_model_path:
		if not Path(local_model_path).is_absolute():
			local_model_path = Path.cwd() / local_model_path
	
	model = load_model_with_fallback(
		local_path=local_model_path,
		device=device,
	)

	itr(model, replace(cfg, device=device))


if __name__ == "__main__":
	main()
