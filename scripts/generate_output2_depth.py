#!/usr/bin/env python3
"""Generate metric z-depth for output_2/output2_bench scenes.

The raw output_2 chunks contain NeRF/Blender-style camera-to-world transforms
and a mesh_raw.ply.  This script raycasts the mesh and writes OpenCV-style
camera z-depth maps, with invalid pixels set to zero.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import open3d as o3d


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def raw_chunk_from_bench_scene(source_root: Path, bench_scene_name: str) -> Path:
    raw_scene, raw_chunk = bench_scene_name.rsplit("_", 1)
    return source_root / raw_scene / raw_chunk


def find_mesh_path(raw_chunk_dir: Path) -> Path:
    for name in ("mesh_raw.ply", "mesh.ply", "mesh.obj"):
        path = raw_chunk_dir / name
        if path.exists():
            return path
    candidates = sorted(raw_chunk_dir.glob("*.ply")) + sorted(raw_chunk_dir.glob("*.obj"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No mesh file found in {raw_chunk_dir}")


def frame_stem(frame: dict[str, Any]) -> str:
    raw = str(frame.get("file_path", "")).strip()
    if not raw:
        raise ValueError("Frame is missing file_path")
    return Path(raw).stem


def source_frame_lookup(raw_frames: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for raw_frame in raw_frames:
        stem = frame_stem(raw_frame)
        lookup[stem] = raw_frame
        lookup[stem.zfill(6)] = raw_frame
        if len(stem) >= 3:
            lookup[stem[-3:]] = raw_frame
            lookup[stem[-3:].zfill(6)] = raw_frame
    return lookup


def ensure_bench_intrinsics(
    target_transforms: dict[str, Any],
    raw_transforms: dict[str, Any],
    image_width: int,
    image_height: int,
) -> None:
    if all(k in target_transforms for k in ("h", "w", "fl_x", "fl_y", "cx", "cy")):
        return

    camera_angle_x = raw_transforms.get("camera_angle_x")
    if camera_angle_x is None:
        raise KeyError(
            "Target transforms lack intrinsics, and raw transforms lack camera_angle_x."
        )

    fl_x = 0.5 * image_width / math.tan(0.5 * float(camera_angle_x))
    target_transforms["h"] = int(image_height)
    target_transforms["w"] = int(image_width)
    target_transforms["fl_x"] = float(fl_x)
    target_transforms["fl_y"] = float(fl_x)
    target_transforms["cx"] = float(image_width) / 2.0
    target_transforms["cy"] = float(image_height) / 2.0


def make_raycast_scene(mesh_path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise ValueError(f"Open3D failed to load mesh: {mesh_path}")
    if hasattr(mesh, "triangulate"):
        mesh = mesh.triangulate()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)
    return scene


def build_nerf_camera_rays(
    width: int,
    height: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Return unnormalized OpenGL/NeRF camera rays with z == -1.

    Because z is exactly -1, the ray parameter returned by Open3D is equal to
    positive OpenCV camera z-depth after the Blender/OpenGL -> OpenCV axis flip.
    """
    u, v = np.meshgrid(
        np.arange(width, dtype=np.float64),
        np.arange(height, dtype=np.float64),
    )
    x = (u - float(cx)) / float(fx)
    y = -(v - float(cy)) / float(fy)
    z = -np.ones_like(x, dtype=np.float64)
    return np.stack([x, y, z], axis=-1).astype(np.float32).reshape(-1, 3)


def render_depth(
    scene: o3d.t.geometry.RaycastingScene,
    rays_camera: np.ndarray,
    c2w: np.ndarray,
    height: int,
    width: int,
    ray_batch_size: int,
) -> np.ndarray:
    origin = c2w[:3, 3].astype(np.float32)
    rotation = c2w[:3, :3].astype(np.float32)
    depth = np.zeros((rays_camera.shape[0],), dtype=np.float32)

    for start in range(0, rays_camera.shape[0], ray_batch_size):
        end = min(start + ray_batch_size, rays_camera.shape[0])
        dirs_world = rays_camera[start:end] @ rotation.T
        origins = np.broadcast_to(origin, dirs_world.shape)
        rays = np.concatenate([origins, dirs_world], axis=1).astype(np.float32)
        ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))
        geometry_ids = ans["geometry_ids"].numpy()
        t_hit = ans["t_hit"].numpy().astype(np.float32)
        hit = np.isfinite(t_hit) & (geometry_ids >= 0) & (t_hit > 0)
        depth[start:end][hit] = t_hit[hit]

    return depth.reshape(height, width)


def depth_is_plausible(depth: np.ndarray, min_valid_ratio: float = 0.001) -> bool:
    if depth.ndim != 2 or depth.dtype.kind not in {"f", "u", "i"}:
        return False
    finite = np.isfinite(depth)
    valid = finite & (depth > 0)
    if float(valid.mean()) < min_valid_ratio:
        return False
    positive = depth[valid]
    if positive.size < 16:
        return False
    sample = positive[: min(positive.size, 100_000)]
    unique_count = np.unique(sample).size
    if unique_count <= 8 and float(positive.max()) <= 255.0:
        return False
    return True


def render_bench_scene_depth(
    raw_chunk_dir: Path,
    bench_scene_dir: Path,
    overwrite: bool = False,
    repair_invalid: bool = True,
    ray_batch_size: int = 1_000_000,
    min_valid_ratio: float = 0.001,
) -> dict[str, int]:
    raw_tf_path = raw_chunk_dir / "transforms.json"
    target_tf_path = bench_scene_dir / "transforms.json"
    if not raw_tf_path.exists():
        raise FileNotFoundError(f"Missing raw transforms: {raw_tf_path}")
    if not target_tf_path.exists():
        raise FileNotFoundError(f"Missing target transforms: {target_tf_path}")

    raw_tf = read_json(raw_tf_path)
    target_tf = read_json(target_tf_path)
    frames = target_tf.get("frames", [])
    raw_frames = raw_tf.get("frames", [])
    if not frames:
        raise ValueError(f"No frames in {target_tf_path}")
    if not raw_frames:
        raise ValueError(f"No frames in {raw_tf_path}")

    width = int(target_tf.get("w", 1600))
    height = int(target_tf.get("h", 1200))
    ensure_bench_intrinsics(target_tf, raw_tf, width, height)
    if target_tf != read_json(target_tf_path):
        write_json(target_tf_path, target_tf)

    fx = float(target_tf["fl_x"])
    fy = float(target_tf["fl_y"])
    cx = float(target_tf["cx"])
    cy = float(target_tf["cy"])
    rays_camera = build_nerf_camera_rays(width, height, fx, fy, cx, cy)

    mesh_path = find_mesh_path(raw_chunk_dir)
    scene = make_raycast_scene(mesh_path)
    raw_by_stem = source_frame_lookup(raw_frames)

    depth_dir = bench_scene_dir / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)

    rendered = 0
    skipped_existing = 0
    repaired_invalid = 0
    missing_pose = 0

    for target_frame in frames:
        target_stem = frame_stem(target_frame)
        out_path = depth_dir / f"{target_stem}.npy"

        should_render = overwrite or not out_path.exists()
        if out_path.exists() and not should_render and repair_invalid:
            try:
                should_render = not depth_is_plausible(
                    np.load(out_path), min_valid_ratio=min_valid_ratio
                )
                if should_render:
                    repaired_invalid += 1
            except Exception:
                should_render = True
                repaired_invalid += 1

        if not should_render:
            skipped_existing += 1
            continue

        raw_frame = raw_by_stem.get(target_stem) or raw_by_stem.get(target_stem[-3:])
        pose_frame = raw_frame or target_frame
        if "transform_matrix" not in pose_frame:
            missing_pose += 1
            continue

        c2w = np.asarray(pose_frame["transform_matrix"], dtype=np.float64)
        if c2w.shape != (4, 4):
            missing_pose += 1
            continue

        depth = render_depth(
            scene,
            rays_camera,
            c2w,
            height,
            width,
            max(1, int(ray_batch_size)),
        )
        np.save(out_path, depth.astype(np.float32))
        rendered += 1

    return {
        "rendered": rendered,
        "skipped_existing": skipped_existing,
        "repaired_invalid": repaired_invalid,
        "missing_pose": missing_pose,
    }


def iter_bench_scenes(target_root: Path, scene_names: list[str] | None) -> list[Path]:
    if scene_names:
        return [target_root / name for name in scene_names]
    return sorted([p for p in target_root.iterdir() if p.is_dir() and "_" in p.name])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate true mesh-raycast depth for output2_bench scenes."
    )
    parser.add_argument("--source-root", type=Path, default=Path("/data/jiachen/output_2"))
    parser.add_argument("--target-root", type=Path, default=Path("/data/sunchang/output2_bench"))
    parser.add_argument("--scene", action="append", default=None, help="Bench scene name to process.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate all depth files.")
    parser.add_argument(
        "--no-repair-invalid",
        action="store_true",
        help="Keep existing depth files even if they look quantized/invalid.",
    )
    parser.add_argument("--ray-batch-size", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    totals = {
        "scenes": 0,
        "missing_raw": 0,
        "failed": 0,
        "rendered": 0,
        "skipped_existing": 0,
        "repaired_invalid": 0,
        "missing_pose": 0,
    }

    scenes = iter_bench_scenes(args.target_root, args.scene)
    for idx, bench_scene_dir in enumerate(scenes, start=1):
        if not bench_scene_dir.exists():
            totals["failed"] += 1
            print(f"[warn] missing bench scene: {bench_scene_dir}", flush=True)
            continue

        raw_chunk_dir = raw_chunk_from_bench_scene(args.source_root, bench_scene_dir.name)
        if not raw_chunk_dir.exists():
            totals["missing_raw"] += 1
            print(f"[warn] missing raw chunk: {raw_chunk_dir}", flush=True)
            continue

        try:
            stats = render_bench_scene_depth(
                raw_chunk_dir=raw_chunk_dir,
                bench_scene_dir=bench_scene_dir,
                overwrite=args.overwrite,
                repair_invalid=not args.no_repair_invalid,
                ray_batch_size=args.ray_batch_size,
            )
        except Exception as exc:
            totals["failed"] += 1
            print(f"[warn] failed {bench_scene_dir.name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        totals["scenes"] += 1
        for key, value in stats.items():
            totals[key] += value

        if idx % max(1, args.log_every) == 0 or idx == len(scenes):
            print(
                "[progress] "
                f"{idx}/{len(scenes)} | rendered={totals['rendered']} | "
                f"repaired_invalid={totals['repaired_invalid']} | "
                f"skipped_existing={totals['skipped_existing']} | "
                f"missing_raw={totals['missing_raw']} | failed={totals['failed']}",
                flush=True,
            )

    print(f"[done] {totals}", flush=True)


if __name__ == "__main__":
    main()
