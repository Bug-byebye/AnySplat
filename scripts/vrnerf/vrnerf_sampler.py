#!/usr/bin/env python3
"""
VR-NeRF few-shot sampler

Goal
  Create small "few-shot" splits suitable for AnySplat-style reconstruction:
    - train: context/input views (few-shot)
    - test : target/evaluation views

Input (per scene under dataset_root/<scene>/)
  - splits.json   : {"train": [...], "test": [...]}
  - cameras.json  : {"KRT": [{"cameraId": "...", "T": [[...],[...],[...],[...]], ...}, ...]}

Output (per scene)
  - splits_fewshot_{K}v.json : {"train": [...], "test": [...]}

Sampling
  - Candidate pool comes from splits.json (default: "train" list; optionally union train+test).
  - Optional block filter (e.g. "10/") to restrict to a capture block.
  - Context views: farthest-point sampling (max coverage) in "camera center" space.
  - Test views: farthest-point sampling on remaining frames to diversify evaluation.

Note on camera centers
  VR-NeRF cameras.json stores a 4x4 matrix "T". Different datasets use different conventions.
  For robust sampling, we only need a *consistent* 3D embedding per frame, so we use:
    center = np.array(T)[3, :3]
  which matches the observed file structure (last row contains translation-like values).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from omegaconf import OmegaConf
import torch

from src.misc.image_io import load_image


def _read_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def _load_scene_list(scenes_path: Path) -> List[str]:
    data = _read_json(scenes_path)
    if isinstance(data, list):
        return [str(x) for x in data]
    if isinstance(data, dict):
        # allow {"scenes": [...]} as a fallback
        if "scenes" in data and isinstance(data["scenes"], list):
            return [str(x) for x in data["scenes"]]
    raise ValueError(f"Unsupported scenes file format: {scenes_path}")


# -----------------------------------------------------------------------------
# Dense-view / Sparse-view evaluation pipeline (camera-block based)
# -----------------------------------------------------------------------------


def _build_candidate_pool_from_camera_block(
    scene_dir: Path,
    camera_id: str | int,
    pool_size: int = 72,
    stride: int = 2,
    images_subdir: str = "images-jpeg-1k",
) -> List[str]:
    """
    Build candidate view pool from a single camera block directory.

    From e.g. datasets-raw/vrnerf/kitchen/images-jpeg-1k/20/, list images,
    sort by filename, then take every `stride`-th image (1st, 3rd, 5th... when stride=2)
    until `pool_size` images are collected.

    Returns:
        List of frame_ids in format "{camera_id}/{filename_without_ext}", e.g. "20/20_DSC0010"
    """
    camera_id = str(camera_id)
    block_dir = scene_dir / images_subdir / camera_id
    if not block_dir.exists():
        raise FileNotFoundError(f"Camera block dir not found: {block_dir}")

    exts = (".jpg", ".jpeg", ".png")
    files = [
        f for f in block_dir.iterdir()
        if f.is_file() and f.suffix.lower() in exts
    ]
    files = sorted(files, key=lambda p: p.name)

    # Take every stride-th image: indices 0, stride, 2*stride, ...
    selected = []
    for i in range(0, len(files), stride):
        if len(selected) >= pool_size:
            break
        selected.append(files[i])

    if len(selected) < pool_size:
        raise ValueError(
            f"Not enough images in {block_dir}: got {len(selected)} with stride={stride}, need {pool_size}"
        )

    selected = selected[:pool_size]
    # frame_id format: "20/20_DSC0010"
    return [f"{camera_id}/{f.stem}" for f in selected]


def _split_test_views_from_pool(
    candidate_pool: List[str],
    setting: str,
) -> dict:
    """
    Split test views from candidate pool according to evaluation setting.

    Args:
        candidate_pool: List of frame_ids (sorted by acquisition order)
        setting: "dense" or "sparse"

    Returns:
        {
            "test": list of test frame_ids,
            "input_candidate": list of remaining frame_ids (for input sampling)
        }
    """
    n = len(candidate_pool)
    if setting == "dense":
        # Every 9th image → 8 test views, 64 input candidate
        step = 9
        test_indices = list(range(4, n, step))
    elif setting == "sparse":
        # Every 3nd image → 36 test views, 36 input candidate
        step = 3
        test_indices = list(range(1, n, step))
    else:
        raise ValueError(f"setting must be 'dense' or 'sparse', got {setting}")

    test_set = set(test_indices)
    input_candidate = [candidate_pool[i] for i in range(n) if i not in test_set]
    test = [candidate_pool[i] for i in test_indices]

    print(f"test_length: {len(test)}, input_candidate_length: {len(input_candidate)}")

    return {"test": test, "input_candidate": input_candidate}


def _sample_input_views_for_dense_setting(
    input_candidate: List[str],
    num_input: int,
    seed: int = 0,
) -> List[str]:
    """
    Sample input views from input candidate pool (dense-view setting only).

    For 64-view: use all 64.
    For 48-view: random sample 48.
    For 32-view: random sample 32.

    Reproducible via seed.
    """
    n = len(input_candidate)
    if num_input >= n:
        return list(input_candidate)

    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=num_input, replace=False)
    indices = np.sort(indices)
    return [input_candidate[i] for i in indices]


def get_dense_sparse_splits(
    *,
    scene: str,
    scene_dir: Path,
    camera_id: str | int,
    setting: str,
    num_input: int | None = None,
    pool_size: int = 72,
    pool_stride: int = 2,
    seed: int = 0,
    images_subdir: str = "images-jpeg-1k",
) -> List[dict]:
    """
    Get input/test view splits for Dense-view or Sparse-view evaluation.

    Args:
        scene: Scene name (e.g. "kitchen")
        scene_dir: Path to scene directory (e.g. datasets-raw/vrnerf/kitchen)
        camera_id: Camera block id (e.g. "20")
        setting: "dense" or "sparse"
        num_input: For dense-view only. None = use all input candidates.
                   For dense-view, can also pass 64, 48, 32 to get specific sizes.
                   When setting is "sparse", this is ignored.
        pool_size: Size of candidate pool (default 72)
        pool_stride: Stride when building pool (default 2: every other image)
        seed: Random seed for input sampling (dense 48/32-view)
        images_subdir: Subdir under scene_dir for images

    Returns:
        List of dicts, each:
        {
            "input": list of input frame_ids,
            "test": list of test frame_ids,
            "setting": "dense" | "sparse",
            "num_input": int,
            "num_test": int,
        }

        For dense-view: returns 3 items (64-view, 48-view, 32-view) if num_input is None.
        For dense-view with num_input=32: returns 1 item (32-view only).
        For sparse-view: returns 1 item (all input candidates as input).
    """
    scene_dir = Path(scene_dir)
    # 1) Build candidate pool
    candidate_pool = _build_candidate_pool_from_camera_block(
        scene_dir=scene_dir,
        camera_id=camera_id,
        pool_size=pool_size,
        stride=pool_stride,
        images_subdir=images_subdir,
    )

    # 2) Split test views
    split = _split_test_views_from_pool(candidate_pool, setting=setting)
    test = split["test"]
    input_candidate = split["input_candidate"]

    if setting == "sparse":
        return [
            {
                "input": input_candidate,
                "test": test,
                "setting": "sparse",
                "num_input": len(input_candidate),
                "num_test": len(test),
            }
        ]

    # 3) Dense-view: construct 64/48/32 input sizes
    assert setting == "dense"
    results = []

    if num_input is not None:
        sizes = [num_input]
    else:
        sizes = [64, 48, 32]

    for k in sizes:
        if k > len(input_candidate):
            continue
        input_ids = _sample_input_views_for_dense_setting(
            input_candidate=input_candidate,
            num_input=k,
            seed=seed,
        )
        results.append({
            "input": input_ids,
            "test": test,
            "setting": "dense",
            "num_input": len(input_ids),
            "num_test": len(test),
        })

    return results


def get_dense_sparse_splits_with_paths(
    *,
    scene: str,
    scene_dir: Path,
    camera_id: str | int,
    setting: str,
    num_input: int | None = None,
    pool_size: int = 72,
    pool_stride: int = 2,
    seed: int = 0,
    images_subdir: str = "images-jpeg-1k",
) -> List[dict]:
    """
    Same as get_dense_sparse_splits, but each returned dict also includes
    "input_paths" and "test_paths" (list of Path).
    """
    splits = get_dense_sparse_splits(
        scene=scene,
        scene_dir=scene_dir,
        camera_id=camera_id,
        setting=setting,
        num_input=num_input,
        pool_size=pool_size,
        pool_stride=pool_stride,
        seed=seed,
        images_subdir=images_subdir,
    )
    for s in splits:
        s["input_paths"] = [
            _resolve_vrnerf_image_path(Path(scene_dir), fid, images_subdir)
            for fid in s["input"]
        ]
        s["test_paths"] = [
            _resolve_vrnerf_image_path(Path(scene_dir), fid, images_subdir)
            for fid in s["test"]
        ]
    return splits


def load_scenes_used(
    *,
    dataset_root: Path,
    scenes_used_path: Path
) -> list[dict]:
    """
    Read `scenes_used.json` and return a structured list for downstream scripts.

    Returns a list of dicts:
      [{"scene": <scene_name>, "scene_dir": <Path to scene directory>}, ...]
    """
    
    dataset_root = Path(dataset_root)
    scenes_used_path = Path(scenes_used_path)
    scenes = _load_scene_list(scenes_used_path)
    return [{"scene": s, "scene_dir": dataset_root / s} for s in scenes]


def _camera_centers_from_cameras_json(cameras_json: dict) -> Dict[str, np.ndarray]:
    if "KRT" not in cameras_json or not isinstance(cameras_json["KRT"], list):
        raise ValueError("cameras.json missing 'KRT' list")
    out: Dict[str, np.ndarray] = {}
    for item in cameras_json["KRT"]:
        cam_id = item.get("cameraId", None)
        T = item.get("T", None)
        if cam_id is None or T is None:
            continue
        M = np.array(T, dtype=np.float64)
        if M.shape != (4, 4):
            continue
        out[str(cam_id)] = M[3, :3].copy()
    return out


def _farthest_point_sample(
    ids: Sequence[str],
    points: Sequence[np.ndarray],
    k: int,
    seed: int = 0,
    start: str = "first",
    initial_selected: Optional[Sequence[int]] = None,
) -> List[int]:
    """
    Greedy farthest-point sampling:
      pick a start, then iteratively pick the point with max distance to the selected set.
    """
    n = len(ids)
    if k <= 0:
        return []
    if n == 0:
        raise ValueError("Empty candidate set")
    if k >= n:
        return list(range(n))

    pts = np.stack(points, axis=0).astype(np.float64)  # [n,3]

    selected: List[int] = []
    if initial_selected:
        selected.extend([int(i) for i in initial_selected])
        selected = [i for i in selected if 0 <= i < n]
        # de-dup while preserving order
        seen = set()
        selected = [i for i in selected if not (i in seen or seen.add(i))]

    rng = np.random.default_rng(seed)

    if not selected:
        if start == "random":
            selected.append(int(rng.integers(0, n)))
        else:
            selected.append(0)

    # Maintain min squared distance to selected for each point
    d2 = np.full((n,), np.inf, dtype=np.float64)
    for s in selected:
        diff = pts - pts[s]
        d2 = np.minimum(d2, np.einsum("ij,ij->i", diff, diff))

    while len(selected) < k:
        # avoid picking already selected
        d2[selected] = -1.0
        nxt = int(np.argmax(d2))
        selected.append(nxt)
        diff = pts - pts[nxt]
        d2 = np.minimum(d2, np.einsum("ij,ij->i", diff, diff))

        if not np.isfinite(d2).any():
            break

    return selected[:k]


def _uniform_stride(ids: Sequence[str], k: int) -> List[int]:
    n = len(ids)
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    # deterministic coverage across the sorted list
    idxs = [int(round(i * (n - 1) / (k - 1))) for i in range(k)]
    # de-dup, then fill if needed
    uniq = []
    seen = set()
    for i in idxs:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    j = 0
    while len(uniq) < k and j < n:
        if j not in seen:
            uniq.append(j)
            seen.add(j)
        j += 1
    return uniq[:k]


def build_fewshot_split_for_scene(
    scene_dir: Path,
    num_context: int,
    num_test: int,
    seed: int,
    pool_from: str,
    block_prefix: Optional[str],
    test_strategy: str,
) -> dict:
    splits_path = scene_dir / "splits.json"
    cameras_path = scene_dir / "cameras.json"
    if not splits_path.exists():
        raise FileNotFoundError(f"Missing {splits_path}")
    if not cameras_path.exists():
        raise FileNotFoundError(f"Missing {cameras_path}")

    splits = _read_json(splits_path)
    cams = _read_json(cameras_path)
    centers = _camera_centers_from_cameras_json(cams)

    train_ids = list(map(str, splits.get("train", [])))
    test_ids = list(map(str, splits.get("test", [])))

    if pool_from == "train":
        pool = train_ids
    elif pool_from == "all":
        # preserve order but de-dup
        pool = []
        seen = set()
        for x in train_ids + test_ids:
            if x not in seen:
                pool.append(x)
                seen.add(x)
    else:
        raise ValueError(f"pool_from must be 'train' or 'all', got {pool_from}")

    if block_prefix:
        pool_f = [x for x in pool if x.startswith(block_prefix)]
        # fallback if too small to satisfy split sizes
        if len(pool_f) >= (num_context + num_test):
            pool = pool_f

    # Keep only ids that have camera centers
    pool = [x for x in pool if x in centers]
    if len(pool) < (num_context + num_test):
        raise ValueError(
            f"{scene_dir.name}: not enough usable frames after filtering: "
            f"{len(pool)} < {num_context + num_test}"
        )

    # Stable order for determinism (VR-NeRF ids include a directory-like prefix)
    pool = sorted(pool)
    pool_pts = [centers[x] for x in pool]

    # context
    ctx_idx = _farthest_point_sample(
        ids=pool,
        points=pool_pts,
        k=num_context,
        seed=seed,
        start="first",
    )
    ctx_set = set(ctx_idx)
    context = [pool[i] for i in ctx_idx]

    # candidates for test
    remaining = [(i, pool[i]) for i in range(len(pool)) if i not in ctx_set]
    rem_ids = [x for _, x in remaining]
    rem_pts = [centers[x] for x in rem_ids]

    if test_strategy == "farthest":
        tst_idx_local = _farthest_point_sample(
            ids=rem_ids,
            points=rem_pts,
            k=num_test,
            seed=seed + 999,
            start="first",
        )
    elif test_strategy == "uniform":
        tst_idx_local = _uniform_stride(rem_ids, num_test)
    else:
        raise ValueError(f"test_strategy must be 'farthest' or 'uniform', got {test_strategy}")

    test = [rem_ids[i] for i in tst_idx_local]

    # Safety: ensure disjoint
    if set(context) & set(test):
        raise RuntimeError("Context/test overlap detected (should not happen)")

    return {"train": context, "test": test}


def _resolve_vrnerf_image_path(
    scene_dir: Path,
    frame_id: str,
    images_subdir: str = "images-jpeg-1k",
) -> Path:
    """
    Map a VR-NeRF frame id (e.g. "10/000123") to an on-disk image file path.

    The downloader saves images under:
      <scene_dir>/images-jpeg-1k/<frame_id>.jpg

    This resolver is lenient:
    - if frame_id already has an extension, we use it directly
    - otherwise we try common extensions in order
    """
    base = scene_dir / images_subdir / frame_id
    if base.suffix.lower() in (".jpg", ".jpeg", ".png"):
        return base

    for ext in (".jpg", ".jpeg", ".png"):
        cand = base.with_suffix(ext)
        if cand.exists():
            return cand

    # Fall back to the most likely path (jpg) for a clear error upstream.
    return base.with_suffix(".jpg")


def load_vrnerf_fewshot_images(
    *,
    scene_dir: Path | str,
    k: int,
    split_path: Path | str | None = None,
    images_subdir: str = "images-jpeg-1k",
    device: torch.device | str | None = None,
    return_paths_only: bool = False,
) -> tuple[
    torch.Tensor | list[Path],
    torch.Tensor | list[Path],
    list[str],
    list[str],
    list[Path],
    list[Path],
]:
    """
    One-call helper to load few-shot train/test images for a VR-NeRF scene.

    What you pass in:
    - scene_dir: the scene folder containing splits_fewshot_{k}v.json and images
    - k: the few-shot context size (matches the file name)

    What it returns (in this order):
    - train_images: (K, 3, H, W) float tensor in [0,1]  OR list[Path] if return_paths_only
    - test_images : (T, 3, H, W) float tensor in [0,1]  OR list[Path] if return_paths_only
    - train_ids   : list[str] (frame ids)
    - test_ids    : list[str]
    - train_paths : list[Path] (resolved image file paths)
    - test_paths  : list[Path]
    """
    scene_dir = Path(scene_dir)
    if split_path is None:
        split_path = scene_dir / f"splits_fewshot_{int(k)}v.json"
    else:
        split_path = Path(split_path)

    split = _read_json(split_path)
    train_ids = list(map(str, split.get("train", [])))
    test_ids = list(map(str, split.get("test", [])))
    if not train_ids or not test_ids:
        raise ValueError(f"Invalid split file (empty train/test): {split_path}")

    train_paths = [_resolve_vrnerf_image_path(scene_dir, fid, images_subdir) for fid in train_ids]
    test_paths = [_resolve_vrnerf_image_path(scene_dir, fid, images_subdir) for fid in test_ids]

    if return_paths_only:
        return train_paths, test_paths, train_ids, test_ids, train_paths, test_paths

    dev = torch.device(device) if device is not None else torch.device("cpu")

    train_imgs = [load_image(p) for p in train_paths]
    test_imgs = [load_image(p) for p in test_paths]
    train_images = torch.stack(train_imgs, dim=0).to(dev)
    test_images = torch.stack(test_imgs, dim=0).to(dev)

    return train_images, test_images, train_ids, test_ids, train_paths, test_paths


def sample_and_load_vrnerf_fewshot_images(
    *,
    scene_dir: Path | str,
    num_context: int,
    num_test: int = 50,
    seed: int = 0,
    pool_from: str = "train",
    block_prefix: str | None = None,
    test_strategy: str = "farthest",
    images_subdir: str = "images-jpeg-1k",
    device: torch.device | str | None = None,
    write_split_json: bool = False,
    out_dir: Path | str | None = None,
    return_paths_only: bool = False,
) -> tuple[
    torch.Tensor | list[Path],
    torch.Tensor | list[Path],
    list[str],
    list[str],
    list[Path],
    list[Path],
    dict,
]:
    """
    One-call helper: sample few-shot split (like main/build_fewshot_split_for_scene),
    then immediately load and return the corresponding train/test images.

    This is the "do everything" function you can call from e.g. nvs_compare.py:
      - run sampling (context + test) based on cameras.json + splits.json
      - optionally write splits_fewshot_{K}v.json
      - resolve image paths and load images into tensors

    Returns (in this order):
      train_images, test_images, train_ids, test_ids, train_paths, test_paths, split_dict
    where split_dict == {"train": [...], "test": [...]}.
    """
    scene_dir = Path(scene_dir)

    # 1) Sample the split (this is the same core logic used by main()).
    split = build_fewshot_split_for_scene(
        scene_dir=scene_dir,
        num_context=int(num_context),
        num_test=int(num_test),
        seed=int(seed),
        pool_from=str(pool_from),
        block_prefix=None if block_prefix in (None, "null", "") else str(block_prefix),
        test_strategy=str(test_strategy),
    )

    # 2) Optionally write the split json (same naming convention as main()).
    if write_split_json:
        dst_dir = Path(out_dir) / scene_dir.name if out_dir is not None else scene_dir
        out_path = dst_dir / f"splits_fewshot_{int(num_context)}v.json"
        _write_json(out_path, split)

    train_ids = list(map(str, split["train"]))
    test_ids = list(map(str, split["test"]))
    train_paths = [_resolve_vrnerf_image_path(scene_dir, fid, images_subdir) for fid in train_ids]
    test_paths = [_resolve_vrnerf_image_path(scene_dir, fid, images_subdir) for fid in test_ids]

    # 3) Optionally return just paths (fast path).
    if return_paths_only:
        return train_paths, test_paths, train_ids, test_ids, train_paths, test_paths, split

    # 4) Load images and move to device.
    dev = torch.device(device) if device is not None else torch.device("cpu")
    train_images = torch.stack([load_image(p) for p in train_paths], dim=0).to(dev)
    test_images = torch.stack([load_image(p) for p in test_paths], dim=0).to(dev)

    return train_images, test_images, train_ids, test_ids, train_paths, test_paths, split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        type=Path,
        default=Path("config/vrnerf_sampler.yaml"),
        help="YAML config path",
    )
    args = ap.parse_args()

    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping, got {type(cfg)}")

    dataset_root = Path(str(cfg.get("dataset_root", "datasets-raw/vrnerf")))
    scenes_path = Path(str(cfg.get("scenes", "datasets-raw/vrnerf/scenes_used.json")))
    out_dir_raw = cfg.get("out_dir", None)
    out_dir = Path(str(out_dir_raw)) if out_dir_raw not in (None, "null") else None

    num_context = cfg.get("num_context", [2, 5, 20])
    if isinstance(num_context, str):
        ks = [int(x.strip()) for x in num_context.split(",") if x.strip()]
    else:
        ks = [int(x) for x in list(num_context)]
    if not ks:
        raise ValueError("num_context produced empty list")

    num_test = int(cfg.get("num_test", 50))
    seed = int(cfg.get("seed", 0))
    pool_from = str(cfg.get("pool_from", "train"))
    block_prefix = cfg.get("block_prefix", None)
    block_prefix = None if block_prefix in (None, "null", "") else str(block_prefix)
    test_strategy = str(cfg.get("test_strategy", "farthest"))

    scenes = _load_scene_list(scenes_path)

    for scene in scenes:
        scene_dir = dataset_root / scene
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene dir not found: {scene_dir}")

        for k in ks:
            split = build_fewshot_split_for_scene(
                scene_dir=scene_dir,
                num_context=k,
                num_test=num_test,
                seed=seed,
                pool_from=pool_from,
                block_prefix=block_prefix,
                test_strategy=test_strategy,
            )
            dst_dir = out_dir / scene if out_dir is not None else scene_dir
            out_path = dst_dir / f"splits_fewshot_{k}v.json"
            _write_json(out_path, split)

    print(f"Done. Wrote few-shot splits for {len(scenes)} scenes.")


if __name__ == "__main__":
    main()

