from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToTensor

from difix3d_service import difix_infer
from methods.self_supervise import pose_interpolation
from methods.ttt import (
	group_images,
	list_image_paths,
	load_image_tensors,
	render_views,
	select_train_test_images,
)
from src.dataset.view_sampler.view_sampler_rank import extrinsic_distance_batch
from src.misc.image_io import save_image
from src.model.model.anysplat import AnySplat
from src.model.ply_export import export_ply
from src.utils.model_loading import load_model_with_fallback


@dataclass
class DiffiEnhanceConfig:
	input_folder: Path
	output_folder: Optional[Path]
	iters: int = 5
	image_sorted: bool = True
	lr: float = 1e-4
	train_components: Optional[List[str]] = None
	loss_type: str = "l1_mse"
	l1_weight: float = 1.0
	mse_weight: float = 1.0
	lpips_weight: float = 0.1
	context_loss_weight: float = 1.0
	group_size: int = 8
	group_mode: str = "sequential"
	group_stride: Optional[int] = None
	interp_frames: int = 2
	num_train: Optional[int] = None
	train_distribution: str = "uniform"
	seed: int = 0
	device: str = "auto"
	export_ply: bool = True
	pretrained_model_path: Optional[str] = None

	# Difix3D enhancement settings
	difix_prompt: str = "remove degradation"
	difix_num_inference_steps: int = 1
	difix_timesteps: Optional[List[int]] = None
	difix_guidance_scale: float = 0.0


def load_diffi_enhance_config(config_path: Path) -> DiffiEnhanceConfig:
	if not config_path.exists():
		raise FileNotFoundError(f"Diffi enhance config not found: {config_path}")

	raw_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
	if not isinstance(raw_cfg, dict):
		raise ValueError(f"Invalid diffi enhance config format: {config_path}")

	cfg = DiffiEnhanceConfig(**raw_cfg)

	input_folder = Path(cfg.input_folder).expanduser().resolve()
	if cfg.output_folder is None:
		output_root = (input_folder / "diffi_enhance_outputs").expanduser().resolve()
	else:
		output_root = Path(cfg.output_folder).expanduser().resolve()

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	output_folder = output_root / f"diffi_enhance_{timestamp}"

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


def _tensor01_to_pil(img_01_chw: torch.Tensor) -> Image.Image:
	img = img_01_chw.detach().clamp(0.0, 1.0).cpu()
	img_hwc = (img.permute(1, 2, 0) * 255.0).to(torch.uint8).numpy()
	return Image.fromarray(img_hwc)


def _pil_to_tensor01(img: Image.Image) -> torch.Tensor:
	return ToTensor()(img.convert("RGB"))


def diffi_enhance(model: AnySplat, cfg: DiffiEnhanceConfig) -> Optional[torch.Tensor]:
	device = cfg.device
	model = model.to(device)
	model.eval()
	for p in model.parameters():
		p.requires_grad = False

	image_paths = list_image_paths(cfg.input_folder, cfg.image_sorted)
	if len(image_paths) < 2:
		raise ValueError("Diffi enhance requires at least 2 input images.")

	train_paths, test_paths = select_train_test_images(
		image_paths=image_paths,
		num_train=cfg.num_train,
		train_distribution=cfg.train_distribution,
		seed=cfg.seed,
	)
	train_images = load_image_tensors(train_paths)
	test_images = load_image_tensors(test_paths) if test_paths else []

	print(
		f"[diffi] total images: {len(image_paths)} | "
		f"train set: {len(train_images)} ({cfg.train_distribution}) | "
		f"test set: {len(test_images)}"
	)

	groups = group_images(
		train_images,
		group_size=cfg.group_size,
		mode=cfg.group_mode,
		stride=cfg.group_stride,
	)
	if cfg.group_mode in {"sliding", "slide", "overlap"}:
		stride_info = cfg.group_stride if cfg.group_stride not in (None, 0) else max(1, cfg.group_size // 2)
		print(
			f"[diffi] train grouping(sliding): size={cfg.group_size}, stride={stride_info}, groups={len(groups)}"
		)
	else:
		print(f"[diffi] train grouping(sequential): size={cfg.group_size}, groups={len(groups)}")

	enhanced_virtual_views: List[torch.Tensor] = []
	virtual_ref_mapping: List[dict] = []  # Record virtual view -> reference view mapping
	total_groups = len(groups)

	with torch.no_grad():
		for it in range(cfg.iters):
			print(f"\n[diffi] ===== Epoch {it + 1}/{cfg.iters} =====")
			rng = random.Random(cfg.seed + it)

			if it < total_groups:
				selected_groups = [(it, groups[it])]
			else:
				random_idx = rng.randrange(total_groups)
				selected_groups = [(random_idx, groups[random_idx])]

			for group_idx, group in selected_groups:
				ctx_tensor = torch.stack(group, dim=0).unsqueeze(0).to(device)
				ctx_01 = (ctx_tensor + 1.0) * 0.5
				_, v_ctx, _, h, w = ctx_01.shape

				encoder_output = model.encoder(ctx_01, global_step=0, visualization_dump={})
				gaussians = encoder_output.gaussians
				pred_context_pose = encoder_output.pred_context_pose

				random.seed(cfg.seed + it * 100 + group_idx)
				interp_pose = pose_interpolation(
					pred_context_pose,
					num_interp_frames=cfg.interp_frames,
					image_sorted=cfg.image_sorted,
				)

				y_virtual = render_views(
					model.decoder,
					gaussians,
					interp_pose["extrinsic"],
					interp_pose["intrinsic"],
					(h, w),
				)

				render_dir = (
					cfg.output_folder
					/ "virtual_renders"
					/ f"epoch_{it + 1:04d}"
					/ f"group_{group_idx + 1:04d}"
				)
				render_dir.mkdir(parents=True, exist_ok=True)

			# Compute pose distances from each virtual view to all context views
			# Context poses: [4, 4] per view, virtual poses: [4, 4] per view
			virtual_extrinsics = interp_pose["extrinsic"][0]  # [num_virtual, 4, 4]
			context_extrinsics = pred_context_pose["extrinsic"][0]  # [num_context, 4, 4]
			
			# Compute all pairwise distances: [num_virtual, num_context]
			pose_distances = []
			for v_idx in range(virtual_extrinsics.shape[0]):
				v_ext = virtual_extrinsics[v_idx:v_idx+1]  # [1, 4, 4]
				# Compute distance to each context view
				dists = extrinsic_distance_batch(
					torch.cat([v_ext, context_extrinsics], dim=0),
					lambda_t=1.0
				)[0, 1:]  # Get distances from virtual view (index 0) to all context views
				pose_distances.append(dists)
			pose_distances = torch.stack(pose_distances, dim=0)  # [num_virtual, num_context]

			for view_idx, virtual_img_01 in enumerate(y_virtual):
				save_image(virtual_img_01, render_dir / f"virtual_raw_{view_idx:03d}.jpg")

				# Select context view with smallest pose distance as reference
				nearest_ctx_idx = pose_distances[view_idx].argmin().item()
				
				virtual_pil = _tensor01_to_pil(virtual_img_01)
				ref_pil = _tensor01_to_pil(ctx_01[0, nearest_ctx_idx])
				enhanced_pil = difix_infer(
					input_image=virtual_pil,
					ref_image=ref_pil,
					prompt=cfg.difix_prompt,
					num_inference_steps=cfg.difix_num_inference_steps,
					timesteps=cfg.difix_timesteps,
					guidance_scale=cfg.difix_guidance_scale,
				)

				enhanced_tensor_01 = _pil_to_tensor01(enhanced_pil)
				enhanced_virtual_views.append(enhanced_tensor_01.cpu())
				save_image(enhanced_tensor_01, render_dir / f"virtual_difix_{view_idx:03d}.jpg")
				
				# Save reference view and record mapping
				ref_filename = f"virtual_ref_{view_idx:03d}.jpg"
				# Convert PIL to tensor for saving
				ref_tensor_01 = _pil_to_tensor01(ref_pil)
				save_image(ref_tensor_01, render_dir / ref_filename)
				virtual_ref_mapping.append({
					"virtual_view_idx": len(enhanced_virtual_views) - 1,
					"virtual_raw_file": f"virtual_raw_{view_idx:03d}.jpg",
					"virtual_difix_file": f"virtual_difix_{view_idx:03d}.jpg",
					"reference_file": ref_filename,
					"reference_ctx_idx": nearest_ctx_idx,
					"pose_distance": pose_distances[view_idx, nearest_ctx_idx].item(),
					"epoch": it + 1,
					"group": group_idx + 1,
				})

			print(
				f"[diffi] epoch {it + 1} group {group_idx + 1}/{len(groups)}: "
				f"generated {y_virtual.shape[0]} virtual views, "
				f"enhanced total={len(enhanced_virtual_views)}"
			)

	print("\n[diffi] Group processing done, reconstructing with original + enhanced virtual views...")
	with torch.no_grad():
		all_images = load_image_tensors(image_paths)
		all_tensor = torch.stack(all_images, dim=0).unsqueeze(0).to(device)
		all_01 = (all_tensor + 1.0) * 0.5

		if enhanced_virtual_views:
			enhanced_tensor = torch.stack(enhanced_virtual_views, dim=0).unsqueeze(0).to(device)
			recon_input_01 = torch.cat([all_01, enhanced_tensor], dim=1)
		else:
			recon_input_01 = all_01

		print(
			f"[diffi] final reconstruction input views: "
			f"original={all_01.shape[1]}, enhanced_virtual={recon_input_01.shape[1] - all_01.shape[1]}, "
			f"total={recon_input_01.shape[1]}"
		)

		encoder_output = model.encoder(recon_input_01, global_step=0, visualization_dump={})
		gaussians_final = encoder_output.gaussians

	if cfg.export_ply:
		ply_path = cfg.output_folder / "gaussians_diffi_enhance.ply"
		export_ply(
			gaussians_final.means[0],
			gaussians_final.scales[0],
			gaussians_final.rotations[0],
			gaussians_final.harmonics[0],
			gaussians_final.opacities[0],
			ply_path,
		)
		print(f"[diffi] exported final gaussians to: {ply_path}")

	# Save virtual view -> reference view mapping
	if virtual_ref_mapping:
		import json
		mapping_path = cfg.output_folder / "virtual_ref_mapping.json"
		with open(mapping_path, "w") as f:
			json.dump(virtual_ref_mapping, f, indent=2)
		print(f"[diffi] saved virtual-reference view mapping to: {mapping_path}")

	if enhanced_virtual_views:
		return torch.stack(enhanced_virtual_views, dim=0).unsqueeze(0)

	return None


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="AnySplat Diffi Enhance")
	parser.add_argument(
		"--config",
		type=str,
		default=str(Path("config/diffi_enhance.yaml")),
		help="Diffi enhance YAML config path.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	cfg = load_diffi_enhance_config(Path(args.config).expanduser().resolve())
	cfg.output_folder.mkdir(parents=True, exist_ok=True)

	device = torch.device(cfg.device)

	print(f"[diffi] device: {device}")
	print(f"[diffi] input folder: {cfg.input_folder}")
	print(f"[diffi] output folder: {cfg.output_folder}")

	print("[diffi] loading AnySplat pretrained model...")
	local_model_path = cfg.pretrained_model_path
	if local_model_path:
		if not Path(local_model_path).is_absolute():
			local_model_path = Path.cwd() / local_model_path

	model = load_model_with_fallback(
		local_path=local_model_path,
		device=device,
	)

	_ = diffi_enhance(model, replace(cfg, device=device))


if __name__ == "__main__":
	main()
