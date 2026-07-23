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

REAL_DATA_ROOT_DEFAULT = "/data/sunchang/real_data"
OUTPUT_BASE_DEFAULT = "outputs/nvs_full_eval_real_data"

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NVS evaluation on real_data (default /data/sunchang/real_data, scene_*/color). "
            "Each run writes to <output_parent>/<run_timestamp>/ so prior outputs are kept."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=REAL_DATA_ROOT_DEFAULT,
        help=f"Root directory containing per-scene image folders (default: {REAL_DATA_ROOT_DEFAULT}).",
    )
    parser.add_argument(
        "--scene_index",
        type=str,
        default="",
        help=(
            "Optional JSON file: list of paths relative to data_root. Each entry is either "
            "the scene root (parent of --image_subdir) e.g. multi_metal/scene1, or the image "
            "folder itself if its name matches --image_subdir."
        ),
    )
    parser.add_argument(
        "--image_subdir",
        type=str,
        default="color",
        help=(
            "Subfolder under each scene that holds input views (default: color). "
            "When scene_index is omitted, matching .../<scene_dir>/<image_subdir>/ are used."
        ),
    )
    parser.add_argument(
        "--scene_dir_prefix",
        type=str,
        default="scene_",
        help=(
            "Auto-discovery: only include folders whose immediate parent name starts with this "
            '(default: "scene_" for scene_bear / scene_tree style). Use empty string to disable.'
        ),
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
        default="nvs_results_real_data",
        help="Output summary filename prefix (written inside the run directory).",
    )
    parser.add_argument(
        "--output_parent",
        type=str,
        default=OUTPUT_BASE_DEFAULT,
        help=(
            "Parent directory for this run. Actual output is "
            "<output_parent>/<YYYYMMDD_HHMMSS>/ so prior runs are never overwritten."
        ),
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


def _count_images(image_dir: Path) -> int:
    return sum(1 for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)


def _path_has_hidden_component(p: Path) -> bool:
    return any(part.startswith(".") for part in p.parts)


def _scene_key_to_output_dirname(scene_key: str) -> str:
    return scene_key.replace("\\", "/").replace("/", "__")


def resolve_image_dir(data_root: Path, rel: str, image_subdir: str) -> Path | None:
    """Return directory containing views, or None if invalid."""
    r = Path(rel)
    direct = data_root / r
    if direct.is_dir() and direct.name == image_subdir:
        return direct
    under = data_root / r / image_subdir
    if under.is_dir():
        return under
    return None


def discover_scenes(
    data_root: Path,
    scene_index: str,
    image_subdir: str,
    scene_dir_prefix: str,
) -> list[tuple[str, Path | None]]:
    """
    Returns (scene_key, image_dir) sorted by scene_key.
    scene_key is path relative to data_root of the scene root (parent of image_subdir).
    """
    if scene_index:
        with open(scene_index, "r", encoding="utf-8") as f:
            entries = json.load(f)
        out: list[tuple[str, Path | None]] = []
        for x in entries:
            rel = str(x).strip().replace("\\", "/")
            img_dir = resolve_image_dir(data_root, rel, image_subdir)
            if img_dir is None:
                out.append((rel, None))
                continue
            if img_dir.name == image_subdir:
                key = str(img_dir.parent.relative_to(data_root)).replace("\\", "/")
            else:
                key = rel
            out.append((key, img_dir))
        return sorted(out, key=lambda t: t[0])

    found: list[tuple[str, Path]] = []
    for img_dir in sorted(data_root.rglob(image_subdir)):
        if not img_dir.is_dir() or img_dir.name != image_subdir:
            continue
        if _count_images(img_dir) < 2:
            continue
        try:
            rel_parent = img_dir.parent.relative_to(data_root)
        except ValueError:
            continue
        if _path_has_hidden_component(rel_parent):
            continue
        if scene_dir_prefix and not img_dir.parent.name.startswith(scene_dir_prefix):
            continue
        key = str(rel_parent).replace("\\", "/")
        found.append((key, img_dir))
    return sorted(found, key=lambda t: t[0])


def infer_category(scene_name: str, split_token: str) -> str:
    norm = scene_name.replace("\\", "/")
    if "/" in norm:
        return norm.split("/")[0]
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
    scene_output_name: str,
) -> dict:
    image_names = sorted(
        [
            str(p)
            for p in scene_dir.iterdir()
            if p.suffix.lower() in _IMAGE_SUFFIXES
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

    scene_out = output_root / scene_output_name
    for idx, (gt_image, pred_image) in enumerate(zip(tgt_images[0], output.color[0])):
        save_image(gt_image, scene_out / "gt" / f"{idx:0>6}.jpg")
        save_image(pred_image, scene_out / "pred" / f"{idx:0>6}.jpg")

    return {"ok": True, "psnr": psnr, "ssim": ssim, "lpips": lpips}


def write_summary(
    output_txt: Path,
    results: list[dict],
    category_split_token: str,
    *,
    data_root: str,
    output_root: str,
    run_timestamp: str,
) -> None:
    per_category = defaultdict(list)
    for r in results:
        if r.get("ok"):
            c = infer_category(r["scene"], category_split_token)
            per_category[c].append(r)

    with output_txt.open("w", encoding="utf-8") as f:
        f.write("NVS Evaluation Results (real_data)\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"data_root: {data_root}\n")
        f.write(f"output_root: {output_root}\n")
        f.write(f"run_timestamp: {run_timestamp}\n\n")
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
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    output_root = (Path(args.output_parent) / run_timestamp).resolve()

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
    print(f"data_root: {data_root}", flush=True)
    print(f"output_root: {output_root}", flush=True)

    scenes = discover_scenes(
        data_root,
        args.scene_index,
        args.image_subdir,
        args.scene_dir_prefix,
    )
    print(
        f"Found {len(scenes)} scenes (image_subdir={args.image_subdir}, "
        f"scene_dir_prefix={args.scene_dir_prefix!r}).",
        flush=True,
    )
    output_root.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for i, (scene_key, scene_img_dir) in enumerate(scenes, start=1):
        out_name = _scene_key_to_output_dirname(scene_key)
        if scene_img_dir is None:
            results.append({"scene": scene_key, "ok": False, "error": "could not resolve image folder"})
            print(f"[{i}/{len(scenes)}] FAILED {scene_key}: could not resolve image folder", flush=True)
            continue
        if not scene_img_dir.is_dir():
            results.append({"scene": scene_key, "ok": False, "error": "image folder missing"})
            print(f"[{i}/{len(scenes)}] FAILED {scene_key}: image folder missing", flush=True)
            continue
        try:
            one = evaluate_one_scene(
                model=model,
                scene_dir=scene_img_dir,
                llffhold=args.llffhold,
                device=device,
                output_root=output_root,
                scene_output_name=out_name,
            )
            one["scene"] = scene_key
            results.append(one)
            if one.get("ok"):
                print(
                    f"[{i}/{len(scenes)}] {scene_key} -> "
                    f"PSNR {one['psnr']:.2f}, SSIM {one['ssim']:.3f}, LPIPS {one['lpips']:.3f}",
                    flush=True,
                )
            else:
                print(f"[{i}/{len(scenes)}] FAILED {scene_key}: {one.get('error')}", flush=True)
        except Exception as e:
            results.append({"scene": scene_key, "ok": False, "error": str(e)})
            print(f"[{i}/{len(scenes)}] FAILED {scene_key}: {e}", flush=True)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    summary_path = output_root / f"{args.summary_prefix}_{run_timestamp}.txt"
    write_summary(
        summary_path,
        results,
        args.category_split_token,
        data_root=str(data_root),
        output_root=str(output_root),
        run_timestamp=run_timestamp,
    )
    print(f"Summary saved to: {summary_path}", flush=True)

    if args.save_json:
        raw_path = output_root / f"{args.summary_prefix}_{run_timestamp}.json"
        payload = {
            "data_root": str(data_root),
            "output_root": str(output_root),
            "run_timestamp": run_timestamp,
            "results": results,
        }
        with raw_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Raw scene metrics saved to: {raw_path}", flush=True)


if __name__ == "__main__":
    main()
