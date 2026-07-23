#!/usr/bin/env python3
import argparse
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

from PIL import Image


def log(message: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert /data/jiachen/output_2 style dataset into DL3DV-like format."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/jiachen/output_2"),
        help="Root directory of the source dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/data/sunchang/output2_bench"),
        help="Output directory in DL3DV-like layout.",
    )
    parser.add_argument(
        "--split-strategy",
        choices=("scene", "within_scene"),
        default="scene",
        help=(
            "scene: split by scene (recommended for no leakage); "
            "within_scene: split each scene by frames into *_train/*_test."
        ),
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="Train ratio for split strategy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for deterministic split.",
    )
    parser.add_argument(
        "--max-frames-per-scene",
        type=int,
        default=50,
        help="Max frames kept per scene after selection.",
    )
    parser.add_argument(
        "--frame-sampling",
        choices=("uniform", "head", "random"),
        default="uniform",
        help="How to choose frames when source has more than max-frames-per-scene.",
    )
    parser.add_argument(
        "--min-frames-per-scene",
        type=int,
        default=20,
        help="Skip scene if valid frames less than this value.",
    )
    parser.add_argument(
        "--resize-width",
        type=int,
        default=0,
        help="Optional output image width for images_4. 0 means keep original size.",
    )
    parser.add_argument(
        "--resize-height",
        type=int,
        default=0,
        help="Optional output image height for images_4. 0 means keep original size.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output-root before conversion.",
    )
    parser.add_argument(
        "--log-every-scenes",
        type=int,
        default=20,
        help="Print progress every N scenes during scan/convert.",
    )
    return parser.parse_args()


def discover_scenes(input_root: Path) -> list[Path]:
    scene_roots: list[Path] = []
    for tf_path in sorted(input_root.rglob("transforms.json")):
        scene_dir = tf_path.parent
        if (scene_dir / "rgb").is_dir():
            scene_roots.append(scene_dir)
    return scene_roots


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def infer_intrinsics_from_angle_x(angle_x: float, width: int, height: int) -> tuple[float, float, float, float]:
    fx = 0.5 * width / math.tan(0.5 * angle_x)
    fy = fx
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def select_indices(total: int, keep: int, mode: str, rng: random.Random) -> list[int]:
    if keep >= total:
        return list(range(total))
    if mode == "head":
        return list(range(keep))
    if mode == "random":
        idxs = list(range(total))
        rng.shuffle(idxs)
        return sorted(idxs[:keep])

    # uniform
    if keep <= 1:
        return [0]
    step = (total - 1) / (keep - 1)
    picks = sorted({int(round(i * step)) for i in range(keep)})
    while len(picks) < keep:
        for i in range(total):
            if i not in picks:
                picks.append(i)
                if len(picks) == keep:
                    break
    return sorted(picks[:keep])


def normalize_frame_path(scene_dir: Path, frame_entry: dict[str, Any]) -> Path:
    file_path = frame_entry["file_path"]
    return (scene_dir / file_path).resolve()


def convert_scene(
    scene_dir: Path,
    dst_scene_dir: Path,
    selected_frames: list[dict[str, Any]],
    resize_wh: tuple[int, int] | None,
) -> tuple[int, int]:
    dst_images_4 = dst_scene_dir / "images_4"
    dst_images_4.mkdir(parents=True, exist_ok=True)

    h = -1
    w = -1
    out_frames = []
    for i, frame in enumerate(selected_frames):
        src_path = normalize_frame_path(scene_dir, frame)
        if not src_path.exists():
            raise FileNotFoundError(f"Missing frame image: {src_path}")
        suffix = src_path.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg"}:
            suffix = ".png"
        out_name = f"{i:06d}{suffix}"
        dst_path = dst_images_4 / out_name

        # Keep original image format by default.
        if resize_wh is None:
            shutil.copy2(src_path, dst_path)
            with Image.open(dst_path) as copied_img:
                w, h = copied_img.size
        else:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize(resize_wh, Image.Resampling.LANCZOS)
                w, h = img.size
                if suffix == ".png":
                    img.save(dst_path)
                else:
                    img.save(dst_path, quality=95)

        out_frames.append(
            {
                "file_path": f"images/{out_name}",
                "transform_matrix": frame["transform_matrix"],
            }
        )

    tf_src = load_json(scene_dir / "transforms.json")
    angle_x = tf_src.get("camera_angle_x")
    if angle_x is None:
        raise ValueError(f"camera_angle_x is missing in {scene_dir / 'transforms.json'}")
    fx, fy, cx, cy = infer_intrinsics_from_angle_x(float(angle_x), w, h)
    tf_dst = {
        "h": h,
        "w": w,
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "frames": out_frames,
    }
    with (dst_scene_dir / "transforms.json").open("w", encoding="utf-8") as f:
        json.dump(tf_dst, f, indent=2)
    return h, w


def main() -> None:
    args = parse_args()
    log("Starting dataset conversion.")
    log(
        f"Config: input={args.input_root}, output={args.output_root}, "
        f"split={args.split_strategy}, train_ratio={args.train_ratio}, seed={args.seed}"
    )
    if not args.input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    if args.overwrite and args.output_root.exists():
        log(f"Overwrite enabled, removing existing output root: {args.output_root}")
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    log(f"Output root is ready: {args.output_root}")

    rng = random.Random(args.seed)
    log(f"Scanning scenes under {args.input_root} (this may take a while)...")
    scene_dirs = discover_scenes(args.input_root)
    log(f"Scene scan complete. Found {len(scene_dirs)} candidate scenes.")
    if not scene_dirs:
        raise RuntimeError("No valid scenes found (expected .../scene/transforms.json with rgb/).")

    resize_wh = None
    if args.resize_width > 0 and args.resize_height > 0:
        resize_wh = (args.resize_width, args.resize_height)

    prepared_scenes: list[dict[str, Any]] = []
    skipped_too_few = 0
    for idx_scene, scene_dir in enumerate(scene_dirs, start=1):
        tf = load_json(scene_dir / "transforms.json")
        frames = tf.get("frames", [])
        valid_frames = [f for f in frames if normalize_frame_path(scene_dir, f).exists()]
        if len(valid_frames) < args.min_frames_per_scene:
            skipped_too_few += 1
            continue
        idx = select_indices(
            total=len(valid_frames),
            keep=min(args.max_frames_per_scene, len(valid_frames)),
            mode=args.frame_sampling,
            rng=rng,
        )
        sampled_frames = [valid_frames[i] for i in idx]
        scene_key = f"{scene_dir.parent.name}_{scene_dir.name}"
        prepared_scenes.append(
            {
                "scene_dir": scene_dir,
                "scene_key": scene_key,
                "frames": sampled_frames,
            }
        )
        if idx_scene % max(1, args.log_every_scenes) == 0:
            log(
                f"Prepared {idx_scene}/{len(scene_dirs)} scenes, "
                f"valid={len(prepared_scenes)}, skipped_too_few={skipped_too_few}"
            )

    if not prepared_scenes:
        raise RuntimeError("No scene passed filtering. Adjust min-frames/max-frames options.")
    log(
        f"Preparation complete: {len(prepared_scenes)} usable scenes "
        f"(skipped_too_few={skipped_too_few})."
    )

    train_scene_names: list[str] = []
    test_scene_names: list[str] = []
    metadata: dict[str, Any] = {
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "split_strategy": args.split_strategy,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "max_frames_per_scene": args.max_frames_per_scene,
        "frame_sampling": args.frame_sampling,
        "num_total_scenes": len(prepared_scenes),
        "scenes": [],
    }

    if args.split_strategy == "scene":
        order = list(range(len(prepared_scenes)))
        rng.shuffle(order)
        split = max(1, int(len(order) * args.train_ratio))
        train_ids = set(order[:split])

        log("Converting scenes with scene-level split...")
        for i, item in enumerate(prepared_scenes, start=1):
            scene_name = item["scene_key"]
            dst_scene_dir = args.output_root / scene_name
            h, w = convert_scene(
                scene_dir=item["scene_dir"],
                dst_scene_dir=dst_scene_dir,
                selected_frames=item["frames"],
                resize_wh=resize_wh,
            )
            if i in train_ids:
                train_scene_names.append(scene_name)
                split_name = "train"
            else:
                test_scene_names.append(scene_name)
                split_name = "test"
            metadata["scenes"].append(
                {
                    "scene": scene_name,
                    "split": split_name,
                    "num_frames": len(item["frames"]),
                    "image_h": h,
                    "image_w": w,
                }
            )
            if i % max(1, args.log_every_scenes) == 0 or i == len(prepared_scenes):
                log(
                    f"Converted {i}/{len(prepared_scenes)} scenes "
                    f"(train={len(train_scene_names)}, test={len(test_scene_names)})"
                )
    else:
        # within_scene split: each original scene is duplicated as *_train and *_test
        log("Converting scenes with within-scene split...")
        for i, item in enumerate(prepared_scenes, start=1):
            scene_name = item["scene_key"]
            frames = item["frames"]
            if len(frames) < 2:
                continue
            split_idx = max(1, int(len(frames) * args.train_ratio))
            split_idx = min(split_idx, len(frames) - 1)
            train_frames = frames[:split_idx]
            test_frames = frames[split_idx:]

            train_name = f"{scene_name}_train"
            test_name = f"{scene_name}_test"
            h_train, w_train = convert_scene(
                scene_dir=item["scene_dir"],
                dst_scene_dir=args.output_root / train_name,
                selected_frames=train_frames,
                resize_wh=resize_wh,
            )
            h_test, w_test = convert_scene(
                scene_dir=item["scene_dir"],
                dst_scene_dir=args.output_root / test_name,
                selected_frames=test_frames,
                resize_wh=resize_wh,
            )
            train_scene_names.append(train_name)
            test_scene_names.append(test_name)
            metadata["scenes"].append(
                {
                    "scene": scene_name,
                    "split": "within_scene",
                    "num_frames_train": len(train_frames),
                    "num_frames_test": len(test_frames),
                    "image_h_train": h_train,
                    "image_w_train": w_train,
                    "image_h_test": h_test,
                    "image_w_test": w_test,
                }
            )
            if i % max(1, args.log_every_scenes) == 0 or i == len(prepared_scenes):
                log(
                    f"Converted {i}/{len(prepared_scenes)} source scenes "
                    f"into paired train/test scene folders"
                )

    with (args.output_root / "train_index.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(train_scene_names), f, indent=2)
    with (args.output_root / "test_index.json").open("w", encoding="utf-8") as f:
        json.dump(sorted(test_scene_names), f, indent=2)
    with (args.output_root / "conversion_meta.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    log("Conversion complete.")
    log(f"Output root: {args.output_root}")
    log(f"Train scenes: {len(train_scene_names)}")
    log(f"Test scenes: {len(test_scene_names)}")
    log("Index files: train_index.json, test_index.json")


if __name__ == "__main__":
    main()
