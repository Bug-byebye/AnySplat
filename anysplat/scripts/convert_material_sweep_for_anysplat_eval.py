#!/usr/bin/env python3
import argparse
import gzip
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert /data/jiachen/output_material_sweep_gpu1 style scenes into "
            "AnySplat evaluation-ready dataset (NVS + pose)."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/jiachen/output_material_sweep_gpu1"),
        help="Source root where each scene has rgb/ and transforms.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/sunchang/evaldataset_anysplat"),
        help="Target root for converted dataset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-root before conversion.",
    )
    parser.add_argument(
        "--max-frames-per-scene",
        type=int,
        default=0,
        help="Optional cap for frames per scene. 0 means keep all.",
    )
    parser.add_argument(
        "--min-frames-per-scene",
        type=int,
        default=10,
        help="Skip scene if valid frames are less than this.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Optional resize width for all copied images. 0 means keep original.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=0,
        help="Optional resize height for all copied images. 0 means keep original.",
    )
    parser.add_argument(
        "--transform-kind",
        choices=("c2w", "w2c"),
        default="c2w",
        help="Interpretation of frame transform_matrix in transforms.json.",
    )
    parser.add_argument(
        "--apply-opengl-to-opencv-fix",
        action="store_true",
        help=(
            "If set, apply axis fix diag(1,-1,-1,1) before generating w2c. "
            "Useful for Blender/OpenGL style camera convention."
        ),
    )
    parser.add_argument(
        "--log-every-scenes",
        type=int,
        default=20,
        help="Print progress every N scenes.",
    )
    return parser.parse_args()


def discover_scenes(input_root: Path) -> list[Path]:
    scenes: list[Path] = []
    for tf_path in sorted(input_root.rglob("transforms.json")):
        scene_dir = tf_path.parent
        if (scene_dir / "rgb").is_dir():
            scenes.append(scene_dir)
    return scenes


def infer_category_from_scene_name(scene_name: str) -> str:
    if "__" in scene_name:
        return scene_name.rsplit("__", 1)[-1]
    return "uncategorized"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalize_frame_path(scene_dir: Path, frame_entry: dict[str, Any]) -> Path:
    raw = str(frame_entry.get("file_path", "")).strip()
    if not raw:
        raise ValueError(f"Empty file_path in scene: {scene_dir}")
    rel = Path(raw)
    if rel.suffix == "":
        for ext in (".png", ".jpg", ".jpeg"):
            candidate = scene_dir / f"{raw}{ext}"
            if candidate.exists():
                return candidate.resolve()
    candidate = (scene_dir / rel).resolve()
    if candidate.exists():
        return candidate
    # Common fallback: only filename given in transforms, image in rgb/
    fallback = (scene_dir / "rgb" / rel.name).resolve()
    return fallback


def to_w2c(transform_matrix: Any, transform_kind: str, apply_opengl_to_opencv_fix: bool) -> np.ndarray:
    pose = np.asarray(transform_matrix, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"transform_matrix must be 4x4, got shape {pose.shape}")

    if apply_opengl_to_opencv_fix:
        # Convert OpenGL camera basis to OpenCV basis.
        gl_to_cv = np.diag([1.0, -1.0, -1.0, 1.0])
        pose = pose @ gl_to_cv

    if transform_kind == "c2w":
        return np.linalg.inv(pose)
    return pose


def opencv_w2c_to_pt3d_rt(w2c: np.ndarray) -> tuple[list[list[float]], list[float]]:
    """
    Build R,T that are expected by eval_pose.py before convert_pt3d_RT_to_opencv().
    """
    r_cv = w2c[:3, :3]
    t_cv = w2c[:3, 3]

    # Inverse mapping of convert_pt3d_RT_to_opencv() used in eval_pose.py
    rot_pt3d = r_cv.T.copy()
    rot_pt3d[:, :2] *= -1.0
    trans_pt3d = t_cv.copy()
    trans_pt3d[:2] *= -1.0

    return rot_pt3d.tolist(), trans_pt3d.tolist()


def maybe_resize_copy(src: Path, dst: Path, resize_wh: tuple[int, int] | None) -> tuple[int, int]:
    if resize_wh is None:
        shutil.copy2(src, dst)
        with Image.open(dst) as im:
            w, h = im.size
        return w, h

    with Image.open(src) as im:
        im = im.convert("RGB")
        im = im.resize(resize_wh, Image.Resampling.LANCZOS)
        dst_suffix = dst.suffix.lower()
        if dst_suffix in {".jpg", ".jpeg"}:
            im.save(dst, quality=95)
        else:
            im.save(dst)
        w, h = im.size
    return w, h


def convert_one_scene(
    scene_dir: Path,
    nvs_scene_dir: Path,
    pose_scene_dir: Path,
    pose_category: str,
    max_frames: int,
    min_frames: int,
    resize_wh: tuple[int, int] | None,
    transform_kind: str,
    apply_opengl_to_opencv_fix: bool,
) -> tuple[list[dict[str, Any]], int]:
    tf = load_json(scene_dir / "transforms.json")
    frames = tf.get("frames", [])
    valid_frames: list[dict[str, Any]] = []
    for fr in frames:
        img_path = normalize_frame_path(scene_dir, fr)
        if img_path.exists():
            valid_frames.append(fr)

    if max_frames > 0 and len(valid_frames) > max_frames:
        valid_frames = valid_frames[:max_frames]

    if len(valid_frames) < min_frames:
        return [], 0

    nvs_scene_dir.mkdir(parents=True, exist_ok=True)
    pose_scene_dir.mkdir(parents=True, exist_ok=True)

    pose_records: list[dict[str, Any]] = []
    for idx, fr in enumerate(valid_frames):
        src_img = normalize_frame_path(scene_dir, fr)
        suffix = src_img.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            suffix = ".png"

        out_name = f"{idx:06d}{suffix}"
        nvs_img_path = nvs_scene_dir / out_name
        pose_img_path = pose_scene_dir / out_name

        maybe_resize_copy(src_img, nvs_img_path, resize_wh)
        maybe_resize_copy(src_img, pose_img_path, resize_wh)

        w2c = to_w2c(
            fr["transform_matrix"],
            transform_kind=transform_kind,
            apply_opengl_to_opencv_fix=apply_opengl_to_opencv_fix,
        )
        r_pt3d, t_pt3d = opencv_w2c_to_pt3d_rt(w2c)
        pose_records.append(
            {
                "filepath": str(Path(pose_category) / scene_dir.name / out_name),
                "R": r_pt3d,
                "T": t_pt3d,
            }
        )

    return pose_records, len(valid_frames)


def main() -> None:
    args = parse_args()
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")

    if args.overwrite and args.output_root.exists():
        log(f"Removing existing output root: {args.output_root}")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)

    nvs_root = args.output_root / "nvs_scenes"
    pose_data_root = args.output_root / "pose_co3d"
    pose_anno_root = args.output_root / "pose_annotations"
    nvs_root.mkdir(parents=True, exist_ok=True)
    pose_data_root.mkdir(parents=True, exist_ok=True)
    pose_anno_root.mkdir(parents=True, exist_ok=True)

    resize_wh = None
    if args.resize_width > 0 and args.resize_height > 0:
        resize_wh = (args.resize_width, args.resize_height)

    scenes = discover_scenes(args.input_root)
    if not scenes:
        raise RuntimeError("No valid scene found: expected .../<scene>/rgb and transforms.json")
    log(f"Found {len(scenes)} candidate scenes.")

    annotation_by_category: dict[str, dict[str, list[dict[str, Any]]]] = {}
    nvs_scene_names: list[str] = []
    skipped = 0

    for idx, scene_dir in enumerate(scenes, start=1):
        scene_name = scene_dir.name
        pose_category = infer_category_from_scene_name(scene_name)
        pose_records, kept = convert_one_scene(
            scene_dir=scene_dir,
            nvs_scene_dir=nvs_root / scene_name,
            pose_scene_dir=pose_data_root / pose_category / scene_name,
            pose_category=pose_category,
            max_frames=args.max_frames_per_scene,
            min_frames=args.min_frames_per_scene,
            resize_wh=resize_wh,
            transform_kind=args.transform_kind,
            apply_opengl_to_opencv_fix=args.apply_opengl_to_opencv_fix,
        )
        if kept == 0:
            skipped += 1
            continue
        if pose_category not in annotation_by_category:
            annotation_by_category[pose_category] = {}
        annotation_by_category[pose_category][scene_name] = pose_records
        nvs_scene_names.append(scene_name)

        if idx % max(1, args.log_every_scenes) == 0 or idx == len(scenes):
            log(
                f"Processed {idx}/{len(scenes)} scenes, "
                f"kept={len(nvs_scene_names)}, skipped={skipped}"
            )

    if not annotation_by_category:
        raise RuntimeError("No scene kept after filtering. Please lower min-frames-per-scene.")

    written_annos: list[str] = []
    for pose_category, annotation in sorted(annotation_by_category.items()):
        anno_path = pose_anno_root / f"{pose_category}_test.jgz"
        with gzip.open(anno_path, "wt", encoding="utf-8") as f:
            json.dump(annotation, f)
        written_annos.append(str(anno_path))

    with (args.output_root / "nvs_scene_index.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(nvs_scene_names), f, indent=2)

    run_hint = {
        "nvs_example": (
            "python src/eval_nvs.py "
            "--data_dir /data/sunchang/evaldataset_anysplat/nvs_scenes/<scene_name>"
        ),
        "pose_example": (
            "python src/eval_pose.py "
            "--co3d_dir /data/sunchang/evaldataset_anysplat/pose_co3d "
            "--co3d_anno_dir /data/sunchang/evaldataset_anysplat/pose_annotations "
            f"--min_num_images {args.min_frames_per_scene}"
        ),
        "notes": [
            "Pose categories are inferred from scene suffix after '__' (e.g., '__conductor').",
            "Use eval_pose.py with --categories auto to evaluate all generated categories.",
        ],
    }
    with (args.output_root / "README_evaldataset_anysplat.json").open("w", encoding="utf-8") as f:
        json.dump(run_hint, f, indent=2)

    meta = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "pose_categories": sorted(annotation_by_category.keys()),
        "num_scenes_kept": len(nvs_scene_names),
        "num_scenes_skipped": skipped,
        "min_frames_per_scene": args.min_frames_per_scene,
        "max_frames_per_scene": args.max_frames_per_scene,
        "resize_width": args.resize_width,
        "resize_height": args.resize_height,
        "transform_kind": args.transform_kind,
        "apply_opengl_to_opencv_fix": args.apply_opengl_to_opencv_fix,
    }
    with (args.output_root / "conversion_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    log("Conversion complete.")
    log(f"NVS scenes root: {nvs_root}")
    log(f"Pose data root: {pose_data_root}")
    log(f"Pose annotations: {', '.join(written_annos)}")
    log(f"Kept scenes: {len(nvs_scene_names)}")


if __name__ == "__main__":
    main()
