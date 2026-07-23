import argparse
import datetime
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from src.misc.image_io import save_image
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full NVS evaluation without video/image dumping."
    )
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory containing per-scene image folders.",
    )
    parser.add_argument(
        "--scene_index",
        type=str,
        default="",
        help="Optional JSON file listing scene folder names.",
    )
    parser.add_argument(
        "--llffhold",
        type=int,
        default=8,
        help="LLFF holdout step for context/target split.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help='Device, e.g. "cuda" or "cpu".',
    )
    parser.add_argument(
        "--summary_prefix",
        type=str,
        default="nvs_results",
        help="Output summary filename prefix.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/nvs_full_eval",
        help="Root directory for per-scene artifacts (gt/pred).",
    )
    parser.add_argument(
        "--category_split_token",
        type=str,
        default="__",
        help="Token used to infer category from scene name suffix.",
    )
    parser.add_argument(
        "--save_json",
        action="store_true",
        help="Also save per-scene raw metrics in JSON.",
    )
    return parser.parse_args()


def load_scene_names(data_root: Path, scene_index: str) -> list[str]:
    if scene_index:
        with open(scene_index, "r", encoding="utf-8") as f:
            names = json.load(f)
        return [str(x) for x in names]
    return sorted([p.name for p in data_root.iterdir() if p.is_dir()])


def infer_category(scene_name: str, split_token: str) -> str:
    if split_token and split_token in scene_name:
        return scene_name.rsplit(split_token, 1)[-1]
    return "uncategorized"


@torch.no_grad()
def evaluate_one_scene(
    model: AnySplat,
    scene_dir: Path,
    llffhold: int,
    device: torch.device,
    output_root: Path,
) -> dict:
    image_names = sorted(
        [
            str(p)
            for p in scene_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
    )
    if len(image_names) < 2:
        return {"ok": False, "error": "not enough images"}

    images = [process_image(p) for p in image_names]
    ctx_indices = [idx for idx in range(len(image_names)) if idx % llffhold != 0]
    tgt_indices = [idx for idx in range(len(image_names)) if idx % llffhold == 0]
    if not ctx_indices or not tgt_indices:
        return {"ok": False, "error": "invalid context/target split"}

    ctx_images = torch.stack([images[i] for i in ctx_indices], dim=0).unsqueeze(0).to(device)
    tgt_images = torch.stack([images[i] for i in tgt_indices], dim=0).unsqueeze(0).to(device)
    ctx_images = (ctx_images + 1) * 0.5
    tgt_images = (tgt_images + 1) * 0.5
    b, v, _, h, w = tgt_images.shape

    encoder_output = model.encoder(
        ctx_images,
        global_step=0,
        visualization_dump={},
    )
    gaussians, pred_context_pose = encoder_output.gaussians, encoder_output.pred_context_pose

    num_context_view = ctx_images.shape[1]
    vggt_input_image = torch.cat((ctx_images, tgt_images), dim=1).to(torch.bfloat16)
    with torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
        aggregated_tokens_list, _ = model.encoder.aggregator(
            vggt_input_image,
            intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx,
        )
    with torch.cuda.amp.autocast(enabled=False):
        fp32_tokens = [token.float() for token in aggregated_tokens_list]
        pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
        pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(
            pred_all_pose_enc, vggt_input_image.shape[-2:]
        )

    extrinsic_padding = (
        torch.tensor([0, 0, 0, 1], device=pred_all_extrinsic.device, dtype=pred_all_extrinsic.dtype)
        .view(1, 1, 1, 4)
        .repeat(b, vggt_input_image.shape[1], 1, 1)
    )
    pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()

    pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
    pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h
    pred_all_context_extrinsic = pred_all_extrinsic[:, :num_context_view]
    pred_all_target_extrinsic = pred_all_extrinsic[:, num_context_view:]
    pred_all_target_intrinsic = pred_all_intrinsic[:, num_context_view:]

    scale_factor = (
        pred_context_pose["extrinsic"][:, :, :3, 3].mean()
        / pred_all_context_extrinsic[:, :, :3, 3].mean()
    )
    pred_all_target_extrinsic[..., :3, 3] = pred_all_target_extrinsic[..., :3, 3] * scale_factor

    output = model.decoder.forward(
        gaussians,
        pred_all_target_extrinsic,
        pred_all_target_intrinsic.float(),
        torch.ones(1, v, device=device) * 0.01,
        torch.ones(1, v, device=device) * 100,
        (h, w),
    )

    psnr = compute_psnr(output.color[0], tgt_images[0]).mean().item()
    ssim = compute_ssim(output.color[0], tgt_images[0]).mean().item()
    lpips = compute_lpips(output.color[0], tgt_images[0]).mean().item()

    scene_out = output_root / scene_dir.name
    for idx, (gt_image, pred_image) in enumerate(zip(tgt_images[0], output.color[0])):
        save_image(gt_image, scene_out / "gt" / f"{idx:0>6}.jpg")
        save_image(pred_image, scene_out / "pred" / f"{idx:0>6}.jpg")

    return {"ok": True, "psnr": psnr, "ssim": ssim, "lpips": lpips}


def write_summary(
    output_txt: Path,
    results: list[dict],
    category_split_token: str,
) -> None:
    per_category = defaultdict(list)
    for r in results:
        if r.get("ok"):
            c = infer_category(r["scene"], category_split_token)
            per_category[c].append(r)

    with output_txt.open("w", encoding="utf-8") as f:
        f.write("NVS Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write("Per-category results:\n")
        f.write("-" * 50 + "\n")
        for c in sorted(per_category.keys()):
            vals = per_category[c]
            f.write(f"{c:<22} PSNR: {sum(v['psnr'] for v in vals) / len(vals):.4f}\n")
            f.write(f"{c:<22} SSIM: {sum(v['ssim'] for v in vals) / len(vals):.4f}\n")
            f.write(f"{c:<22} LPIPS: {sum(v['lpips'] for v in vals) / len(vals):.4f}\n")
            f.write("\n")

        ok_vals = [r for r in results if r.get("ok")]
        f.write("-" * 50 + "\n")
        if ok_vals:
            f.write(f"Mean PSNR: {sum(v['psnr'] for v in ok_vals) / len(ok_vals):.4f}\n")
            f.write(f"Mean SSIM: {sum(v['ssim'] for v in ok_vals) / len(ok_vals):.4f}\n")
            f.write(f"Mean LPIPS: {sum(v['lpips'] for v in ok_vals) / len(ok_vals):.4f}\n")
            f.write(f"Num scenes (success): {len(ok_vals)}\n")

        fail_vals = [r for r in results if not r.get("ok")]
        if fail_vals:
            f.write(f"Num scenes (failed): {len(fail_vals)}\n")
        f.write("\n" + "=" * 50 + "\n")


def main() -> None:
    args = setup_args()
    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, fallback to CPU.", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("Loading AnySplat model...", flush=True)
    model = AnySplat.from_pretrained("lhjiang/anysplat")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"Using device: {device}", flush=True)

    scene_names = load_scene_names(data_root, args.scene_index)
    print(f"Found {len(scene_names)} scenes to evaluate.", flush=True)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, scene_name in enumerate(scene_names, start=1):
        scene_dir = data_root / scene_name
        if not scene_dir.is_dir():
            results.append({"scene": scene_name, "ok": False, "error": "scene folder missing"})
            print(f"[{i}/{len(scene_names)}] FAILED {scene_name}: folder missing", flush=True)
            continue
        try:
            one = evaluate_one_scene(
                model=model,
                scene_dir=scene_dir,
                llffhold=args.llffhold,
                device=device,
                output_root=output_root,
            )
            one["scene"] = scene_name
            results.append(one)
            if one.get("ok"):
                print(
                    f"[{i}/{len(scene_names)}] {scene_name} -> "
                    f"PSNR {one['psnr']:.2f}, SSIM {one['ssim']:.3f}, LPIPS {one['lpips']:.3f}",
                    flush=True,
                )
            else:
                print(f"[{i}/{len(scene_names)}] FAILED {scene_name}: {one.get('error')}", flush=True)
        except Exception as e:
            results.append({"scene": scene_name, "ok": False, "error": str(e)})
            print(f"[{i}/{len(scene_names)}] FAILED {scene_name}: {e}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = Path.cwd() / f"{args.summary_prefix}_{timestamp}.txt"
    write_summary(summary_path, results, args.category_split_token)
    print(f"Summary saved to: {summary_path}", flush=True)

    if args.save_json:
        raw_path = Path.cwd() / f"{args.summary_prefix}_{timestamp}.json"
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Raw scene metrics saved to: {raw_path}", flush=True)


if __name__ == "__main__":
    main()
