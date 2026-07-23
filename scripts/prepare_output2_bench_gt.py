#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from generate_output2_depth import depth_is_plausible, render_bench_scene_depth


def iter_depth_candidates(raw_chunk_dir: Path, stem: str):
    for folder in ("depth",):
        for suffix in (".npy", ".npz", ".exr", ".png"):
            yield raw_chunk_dir / folder / f"{stem}{suffix}"


def load_depth_array(depth_path: Path) -> np.ndarray:
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        depth = np.load(depth_path)
    elif suffix == ".npz":
        arr = np.load(depth_path)
        key = "depth" if "depth" in arr else list(arr.keys())[0]
        depth = arr[key]
    else:
        raise ValueError(
            f"Refusing to use image file as GT depth: {depth_path}. "
            "Use scripts/generate_output2_depth.py to raycast mesh depth."
        )
    if depth.ndim == 3:
        depth = depth[..., 0]
    return depth.astype(np.float32)


def prepare_gt(
    source_root: Path,
    target_root: Path,
    overwrite_depth: bool = True,
    generate_depth: bool = True,
    ray_batch_size: int = 1_000_000,
    log_every: int = 100,
) -> None:
    if not source_root.exists():
        raise FileNotFoundError(f"source_root does not exist: {source_root}")
    if not target_root.exists():
        raise FileNotFoundError(f"target_root does not exist: {target_root}")

    scene_dirs = sorted([p for p in target_root.iterdir() if p.is_dir() and "_" in p.name])
    total_scenes = len(scene_dirs)
    print(
        f"[prepare_output2_bench_gt] start | source={source_root} | target={target_root} | "
        f"overwrite_depth={overwrite_depth} | generate_depth={generate_depth} | "
        f"total_scenes={total_scenes}",
        flush=True,
    )

    synced = 0
    missing_raw = 0
    missing_depth = 0
    copied_depth_files = 0
    skipped_existing_depth = 0
    rendered_depth_files = 0
    repaired_invalid_depth_files = 0
    depth_render_failures = 0
    missing_transforms = 0

    for idx, scene_dir in enumerate(scene_dirs, start=1):
        raw_scene, raw_chunk = scene_dir.name.rsplit("_", 1)
        raw_chunk_dir = source_root / raw_scene / raw_chunk
        if not raw_chunk_dir.exists():
            missing_raw += 1
            if missing_raw <= 5:
                print(f"[warn] missing raw chunk: {raw_chunk_dir}", flush=True)
            continue

        target_tf = scene_dir / "transforms.json"
        raw_tf = raw_chunk_dir / "transforms.json"
        if raw_tf.exists() and not target_tf.exists():
            target_tf.write_text(raw_tf.read_text())
        if not target_tf.exists():
            missing_transforms += 1
            continue

        with target_tf.open("r") as f:
            tf_data = json.load(f)
        frames = tf_data.get("frames", [])
        if not frames:
            continue

        if generate_depth:
            try:
                stats = render_bench_scene_depth(
                    raw_chunk_dir=raw_chunk_dir,
                    bench_scene_dir=scene_dir,
                    overwrite=overwrite_depth,
                    repair_invalid=False,
                    ray_batch_size=ray_batch_size,
                )
                rendered_depth_files += stats["rendered"]
                skipped_existing_depth += stats["skipped_existing"]
                repaired_invalid_depth_files += stats["repaired_invalid"]
                missing_depth += stats["missing_pose"]
            except Exception as exc:
                depth_render_failures += 1
                if depth_render_failures <= 5:
                    print(
                        f"[warn] depth render failed for {scene_dir.name}: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
        else:
            depth_dir = scene_dir / "depth"
            depth_dir.mkdir(parents=True, exist_ok=True)
            for frame in frames:
                frame_stem = Path(frame.get("file_path", "")).stem
                bench_depth = depth_dir / f"{frame_stem}.npy"
                if bench_depth.exists() and not overwrite_depth:
                    skipped_existing_depth += 1
                    continue

                src_depth = None
                for stem in (frame_stem, frame_stem[-3:]):
                    for candidate in iter_depth_candidates(raw_chunk_dir, stem):
                        if candidate.exists():
                            src_depth = candidate
                            break
                    if src_depth is not None:
                        break

                if src_depth is None:
                    missing_depth += 1
                    continue

                depth = load_depth_array(src_depth)
                if not depth_is_plausible(depth):
                    missing_depth += 1
                    continue
                np.save(bench_depth, depth)
                copied_depth_files += 1

        synced += 1
        if idx % max(log_every, 1) == 0 or idx == total_scenes:
            print(
                f"[progress] {idx}/{total_scenes} scenes | synced={synced} | "
                f"copied_depth_files={copied_depth_files} | "
                f"rendered_depth_files={rendered_depth_files} | "
                f"repaired_invalid_depth_files={repaired_invalid_depth_files} | "
                f"skipped_existing_depth={skipped_existing_depth} | "
                f"missing_raw={missing_raw} | missing_transforms={missing_transforms} | "
                f"missing_depth={missing_depth} | depth_render_failures={depth_render_failures}",
                flush=True,
            )

    print(
        "[prepare_output2_bench_gt] done | "
        f"synced={synced} | copied_depth_files={copied_depth_files} | "
        f"rendered_depth_files={rendered_depth_files} | "
        f"repaired_invalid_depth_files={repaired_invalid_depth_files} | "
        f"skipped_existing_depth={skipped_existing_depth} | missing_raw={missing_raw} | "
        f"missing_transforms={missing_transforms} | missing_depth={missing_depth} | "
        f"depth_render_failures={depth_render_failures}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject GT pose/depth metadata into /data/sunchang/output2_bench."
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/jiachen/output_2"),
        help="Raw source dataset root.",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("/data/sunchang/output2_bench"),
        help="Bench dataset root used by training.",
    )
    parser.add_argument(
        "--overwrite-depth",
        action="store_true",
        dest="overwrite_depth",
        default=True,
        help="Overwrite existing depth/*.npy in target dataset.",
    )
    parser.add_argument(
        "--skip-existing-depth",
        action="store_false",
        dest="overwrite_depth",
        help="Skip existing depth/*.npy files instead of regenerating them.",
    )
    parser.add_argument(
        "--no-generate-depth",
        action="store_true",
        help="Do not raycast mesh depth; only copy existing raw depth files if present.",
    )
    parser.add_argument(
        "--ray-batch-size",
        type=int,
        default=1_000_000,
        help="Number of rays per Open3D cast_rays call.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Print progress every N scenes.",
    )
    args = parser.parse_args()

    prepare_gt(
        args.source_root,
        args.target_root,
        overwrite_depth=args.overwrite_depth,
        generate_depth=not args.no_generate_depth,
        ray_batch_size=args.ray_batch_size,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
