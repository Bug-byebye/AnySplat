#!/usr/bin/env python3
"""
DL3DV-Evaluation dataset sampler for nvs_compare.py.

Expected scene structure (actual dataset layout):
  dataset_root/
    images/
      {scene_hash}/
        {scene_hash}/
          gaussian_splat/{index}/*.png|*.jpg
          nerfstudio/transforms.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def farthest_point_sampling(points: np.ndarray, num_samples: int, seed: int = 0) -> np.ndarray:
    if len(points) <= num_samples:
        return np.arange(len(points))

    rng = np.random.default_rng(seed)
    selected_indices: list[int] = [int(rng.integers(0, len(points)))]
    min_distances = np.full(len(points), np.inf)

    for _ in range(num_samples - 1):
        last_point = points[selected_indices[-1]]
        distances = np.linalg.norm(points - last_point, axis=1)
        min_distances = np.minimum(min_distances, distances)
        next_idx = int(np.argmax(min_distances))
        selected_indices.append(next_idx)
        min_distances[next_idx] = 0

    return np.array(selected_indices, dtype=np.int64)


def _resolve_images_root(dataset_root: Path) -> Path:
    dataset_root = Path(dataset_root)
    images_root = dataset_root / "images"
    if images_root.exists():
        return images_root
    return dataset_root


def _resolve_scene_dir(scene_hash_dir: Path) -> Path:
    nested_dir = scene_hash_dir / scene_hash_dir.name
    if nested_dir.exists() and nested_dir.is_dir():
        return nested_dir
    return scene_hash_dir


def load_scenes_used(
    dataset_root: Path,
    split: str = "test",
    max_scenes: Optional[int] = None,
) -> List[Dict[str, Any]]:
    del split

    images_root = _resolve_images_root(Path(dataset_root))
    if not images_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {images_root}")

    scenes: list[dict[str, Any]] = []

    for scene_hash_dir in sorted(images_root.iterdir()):
        if not scene_hash_dir.is_dir():
            continue

        scene_dir = _resolve_scene_dir(scene_hash_dir)
        transforms_file = scene_dir / "nerfstudio" / "transforms.json"
        gaussian_splat_dir = scene_dir / "gaussian_splat"

        if not transforms_file.exists() or not gaussian_splat_dir.exists():
            continue

        scenes.append({
            "scene": scene_hash_dir.name,
            "scene_dir": scene_dir,
        })

        if max_scenes is not None and len(scenes) >= max_scenes:
            break

    return scenes


def load_scene_transforms(scene_dir: Path) -> Dict[str, Any]:
    transforms_file = scene_dir / "nerfstudio" / "transforms.json"
    if not transforms_file.exists():
        raise FileNotFoundError(f"transforms.json not found: {transforms_file}")

    with transforms_file.open("r") as f:
        return json.load(f)


def extract_camera_centers(transforms: Dict[str, Any]) -> np.ndarray:
    centers = []
    for frame in transforms["frames"]:
        transform_matrix = np.array(frame["transform_matrix"])
        centers.append(transform_matrix[:3, 3])
    return np.array(centers)


def _resolve_image_path(image_dir: Path, frame_file_path: str | None, frame_idx: int) -> Path | None:
    candidates: list[Path] = []

    if frame_file_path:
        frame_name = Path(frame_file_path).name
        candidates.append(image_dir / frame_name)

    candidates.append(image_dir / f"{frame_idx:04d}.png")

    for base in candidates:
        if base.exists():
            return base
        for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
            alt = base.with_suffix(ext)
            if alt.exists():
                return alt

    return None


def get_dense_sparse_splits_with_paths(
    scene_dir: Path,
    index: str,
    num_input: int,
    num_test: int = 8,
    seed: int = 0,
    pool_size: int | None = None,
    pool_stride: int = 1,
) -> List[Dict[str, Any]]:
    """
    Get input/test view splits for DL3DV-Evaluation dataset.
    
    Args:
        scene_dir: Path to scene directory
        index: Image subdirectory (e.g., "images_4")
        num_input: Number of input views
        num_test: Number of test views
        seed: Random seed for sampling
        pool_size: Size of candidate pool. If None, use all available frames
        pool_stride: Stride when building pool (1=consecutive, 2=every other, etc.)
    
    Returns:
        List of dicts with "input_paths", "test_paths", etc.
    """
    image_dir = scene_dir / "gaussian_splat" / index
    if not image_dir.exists():
        print(f"[warn] Image directory not found: {image_dir}")
        return []

    try:
        transforms = load_scene_transforms(scene_dir)
    except Exception as e:
        print(f"[warn] Failed to load transforms.json in {scene_dir}: {e}")
        return []

    frames = transforms.get("frames", [])
    if len(frames) < num_input + num_test:
        print(
            f"[warn] Scene {scene_dir.name} has only {len(frames)} frames, "
            f"need {num_input + num_test}"
        )
        return []

    # Build candidate pool using stride-based sampling
    # Select every pool_stride-th frame: 0, stride, 2*stride, ...
    candidate_indices = []
    for i in range(0, len(frames), pool_stride):
        candidate_indices.append(i)
        if pool_size is not None and len(candidate_indices) >= pool_size:
            break
    
    if pool_size is not None:
        candidate_indices = candidate_indices[:pool_size]
    
    if len(candidate_indices) < num_input + num_test:
        print(
            f"[warn] Scene {scene_dir.name}: candidate pool has {len(candidate_indices)} frames, "
            f"need {num_input + num_test} (pool_stride={pool_stride})"
        )
        return []

    # Split test views from candidate pool uniformly
    n_pool = len(candidate_indices)
    step = n_pool / num_test
    test_positions = [int(i * step + step / 2) for i in range(num_test)]
    test_positions = [min(pos, n_pool - 1) for pos in test_positions]
    test_positions = sorted(set(test_positions))[:num_test]
    
    test_indices = [candidate_indices[pos] for pos in test_positions]
    test_set = set(test_indices)
    
    # Remaining candidates are for input
    input_candidate_indices = [idx for idx in candidate_indices if idx not in test_set]
    
    # Sample num_input views from input candidates
    if num_input >= len(input_candidate_indices):
        input_indices = input_candidate_indices
    else:
        # Random sample from input candidates
        rng = np.random.default_rng(seed)
        sample_positions = rng.choice(len(input_candidate_indices), size=num_input, replace=False)
        sample_positions = np.sort(sample_positions)
        input_indices = [input_candidate_indices[pos] for pos in sample_positions]
    
    print(f"[sampler] test_length: {len(test_indices)}, input_candidate_length: {len(input_candidate_indices)}")

    input_paths: list[Path] = []
    for idx in input_indices:
        frame = frames[idx]
        img_path = _resolve_image_path(image_dir, frame.get("file_path"), idx)
        if img_path is not None:
            input_paths.append(img_path)
        else:
            print(f"[warn] Input image not found in {image_dir}, frame idx={idx}")

    test_paths: list[Path] = []
    for idx in test_indices:
        frame = frames[idx]
        img_path = _resolve_image_path(image_dir, frame.get("file_path"), idx)
        if img_path is not None:
            test_paths.append(img_path)
        else:
            print(f"[warn] Test image not found in {image_dir}, frame idx={idx}")

    if len(input_paths) < num_input or len(test_paths) == 0:
        print(
            f"[warn] Scene {scene_dir.name}: got {len(input_paths)} inputs, "
            f"{len(test_paths)} tests (need {num_input} inputs, {num_test} tests)"
        )
        return []

    return [{
        "input_paths": input_paths,
        "test_paths": test_paths,
        "num_input": len(input_paths),
        "num_test": len(test_paths),
    }]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="DL3DV-Evaluation sampler test")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--index", type=str, default="images_4")
    parser.add_argument("--num_input", type=int, default=32)
    parser.add_argument("--num_test", type=int, default=8)
    parser.add_argument("--max_scenes", type=int, default=5)
    args = parser.parse_args()

    scenes = load_scenes_used(Path(args.dataset_root), max_scenes=args.max_scenes)
    print(f"Loaded {len(scenes)} scenes")

    for scene in scenes:
        print(f"\nScene: {scene['scene']}")
        splits = get_dense_sparse_splits_with_paths(
            scene_dir=scene["scene_dir"],
            index=args.index,
            num_input=args.num_input,
            num_test=args.num_test,
        )
        if splits:
            split = splits[0]
            print(f"  Input views: {split['num_input']}")
            print(f"  Test views: {split['num_test']}")
        else:
            print("  Failed to generate splits")


if __name__ == "__main__":
    main()
