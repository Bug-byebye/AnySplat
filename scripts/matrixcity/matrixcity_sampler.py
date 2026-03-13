#!/usr/bin/env python3
"""
MatrixCity sampler for NVS comparison.

This module provides data loading utilities for MatrixCity dataset,
mirroring the interface of vrnerf_sampler.py for compatibility.

MatrixCity dataset structure (example):
    datasets/matrixcity/
        <scope>/          # big | small | fusion
            <species>/    # street | aerial | all
                <scene>/
                    images/       # or other image directories
                        *.jpg
                    cameras.json  # or pose files
                    splits.json   # train/test splits (optional)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def _read_json(path: Path) -> dict:
    """Read JSON file."""
    with path.open("r") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    """Write JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def load_scenes_used(
    *,
    dataset_root: Path,
    scope: str = "big",
    species: str = "street",
    scenes_used_path: Optional[Path] = None,
) -> List[dict]:
    """
    Load MatrixCity scenes.

    Args:
        dataset_root: Root directory of MatrixCity dataset
        scope: "big" | "small" | "fusion"
        species: "street" | "aerial" | "all"
        scenes_used_path: Optional path to JSON file listing scene names.
                          If None, will automatically discover scenes from directory structure.

    Returns:
        List of dicts with keys:
            - "scene": scene name
            - "scene_dir": Path to scene directory
    """
    dataset_root = Path(dataset_root)
    
    # Determine scope/species path
    scope_dir = dataset_root / scope / species
    if not scope_dir.exists():
        # Try alternative: direct scene folders in dataset_root
        scope_dir = dataset_root
    
    # Load from scenes_used.json if provided
    if scenes_used_path and scenes_used_path.exists():
        data = _read_json(scenes_used_path)
        if isinstance(data, list):
            scenes = [str(x) for x in data]
        elif isinstance(data, dict) and "scenes" in data:
            scenes = [str(x) for x in data["scenes"]]
        else:
            raise ValueError(f"Unsupported scenes file format: {scenes_used_path}")
        
        return [{"scene": s, "scene_dir": scope_dir / s} for s in scenes]
    
    # Auto-discover scenes from directory structure
    scenes = []
    if scope_dir.exists():
        for item in sorted(scope_dir.iterdir()):
            if item.is_dir():
                # Check if this looks like a scene directory
                # (has images/ subfolder or other typical scene content)
                if (item / "images").exists() or any(item.glob("*.json")):
                    scenes.append({
                        "scene": item.name,
                        "scene_dir": item,
                    })
    
    if not scenes:
        print(f"[Warning] No scenes found in {scope_dir}")
        return []
    
    return scenes


def _resolve_image_path(
    scene_dir: Path,
    frame_id: str,
    images_subdir: str = "images",
) -> Path:
    """
    Resolve a frame_id to its image file path.
    
    Args:
        scene_dir: Scene directory
        frame_id: Frame identifier (e.g., "000001" or "cam01_000001")
        images_subdir: Subdirectory containing images
    
    Returns:
        Path to the image file
    """
    img_dir = scene_dir / images_subdir
    
    # Try common image extensions
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        img_path = img_dir / f"{frame_id}{ext}"
        if img_path.exists():
            return img_path
    
    # If not found, return the .jpg path anyway (will fail later if actually missing)
    return img_dir / f"{frame_id}.jpg"


def get_dense_sparse_splits_with_paths(
    *,
    scene: str,
    scene_dir: Path,
    camera_id: Optional[str] = None,
    num_input: int = 64,
    num_test: int = 8,
    pool_size: int = 200,
    pool_stride: int = 2,
    seed: int = 0,
    images_subdir: str = "images",
) -> List[dict]:
    """
    Sample input and test views for MatrixCity scene.
    
    Args:
        scene: Scene name
        scene_dir: Path to scene directory
        camera_id: Optional camera block identifier (may not apply to MatrixCity)
        num_input: Number of input/context views
        num_test: Number of test views
        pool_size: Size of candidate pool for sampling
        pool_stride: Stride for building candidate pool
        seed: Random seed
        images_subdir: Subdirectory containing images
    
    Returns:
        List of dicts, each containing:
            - "input": List of input frame IDs
            - "test": List of test frame IDs
            - "input_paths": List of Path objects for input images
            - "test_paths": List of Path objects for test images
    """
    scene_dir = Path(scene_dir)
    img_dir = scene_dir / images_subdir
    
    if not img_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Collect all image files
    exts = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    image_files = sorted([
        f for f in img_dir.iterdir()
        if f.is_file() and f.suffix in exts
    ])
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {img_dir}")
    
    # Build candidate pool with stride
    candidate_files = []
    for i in range(0, len(image_files), pool_stride):
        candidate_files.append(image_files[i])
        if len(candidate_files) >= pool_size:
            break
    
    if len(candidate_files) < num_input + num_test:
        raise ValueError(
            f"Not enough images in pool: {len(candidate_files)} < {num_input + num_test}"
        )
    
    # 1. Uniformly select test views from pool
    n_pool = len(candidate_files)
    step = n_pool / num_test
    test_positions = [int(i * step + step / 2) for i in range(num_test)]
    test_positions = [min(pos, n_pool - 1) for pos in test_positions]
    test_positions = sorted(set(test_positions))[:num_test]
    
    # 2. Remaining candidates for input
    test_set = set(test_positions)
    input_candidate_indices = [i for i in range(n_pool) if i not in test_set]
    
    # 3. Sample input views from candidates
    if num_input >= len(input_candidate_indices):
        input_indices = input_candidate_indices
    else:
        rng = np.random.default_rng(seed)
        sample_positions = rng.choice(len(input_candidate_indices), size=num_input, replace=False)
        sample_positions = np.sort(sample_positions)
        input_indices = [input_candidate_indices[pos] for pos in sample_positions]
    
    # Get file paths
    input_paths = [candidate_files[i] for i in input_indices]
    test_paths = [candidate_files[i] for i in test_positions]
    
    # Get frame IDs (stem without extension)
    input_ids = [p.stem for p in input_paths]
    test_ids = [p.stem for p in test_paths]
    
    print(f"[sampler] test_length: {len(test_ids)}, input_candidate_length: {len(input_candidate_indices)}")
    
    return [{
        "input": input_ids,
        "test": test_ids,
        "input_paths": input_paths,
        "test_paths": test_paths,
        "num_input": len(input_ids),
        "num_test": len(test_ids),
        "seed": seed,
    }]


def _farthest_point_sample(
    points: np.ndarray,
    k: int,
    seed: int = 0,
) -> List[int]:
    """
    Greedy farthest-point sampling.
    
    Args:
        points: Array of shape [N, D] representing points
        k: Number of points to sample
        seed: Random seed for initial point selection
    
    Returns:
        List of selected indices
    """
    n = len(points)
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    
    pts = points.astype(np.float64)
    selected = []
    
    # Select first point randomly
    rng = np.random.default_rng(seed)
    selected.append(int(rng.integers(0, n)))
    
    # Maintain min squared distance to selected set
    d2 = np.full((n,), np.inf, dtype=np.float64)
    
    while len(selected) < k:
        # Update distances with last selected point
        last_idx = selected[-1]
        diff = pts - pts[last_idx]
        d2 = np.minimum(d2, np.sum(diff * diff, axis=1))
        
        # Mask already selected points
        d2[selected] = -1.0
        
        # Select point with maximum distance
        nxt = int(np.argmax(d2))
        selected.append(nxt)
    
    return selected


# Placeholder for camera pose utilities
# These will be implemented once actual MatrixCity data format is known

def load_camera_poses(
    scene_dir: Path,
    camera_file: str = "cameras.json",
) -> Dict[str, np.ndarray]:
    """
    Load camera poses from MatrixCity scene.
    
    This is a placeholder implementation.
    
    Returns:
        Dict mapping frame_id to 4x4 camera-to-world matrix
    """
    camera_path = scene_dir / camera_file
    if not camera_path.exists():
        print(f"[Warning] Camera file not found: {camera_path}")
        return {}
    
    # TODO: Implement based on actual MatrixCity format
    # For now, return empty dict
    return {}
