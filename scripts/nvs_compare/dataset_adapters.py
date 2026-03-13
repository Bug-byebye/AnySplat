from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from scripts.matrixcity.matrixcity_sampler import (
    load_scenes_used as matrixcity_load_scenes_used,
    get_dense_sparse_splits_with_paths as matrixcity_get_dense_sparse_splits_with_paths,
)
from scripts.vrnerf.process_fisheye import (
    process_scene_cameras_fisheye,
    get_processed_image_path,
)
from scripts.vrnerf.vrnerf_sampler import (
    load_scenes_used as vrnerf_load_scenes_used,
    get_dense_sparse_splits_with_paths as vrnerf_get_dense_sparse_splits_with_paths,
)
from scripts.dl3dv_evaluation.dl3dv_evaluation_sampler import (
    load_scenes_used as dl3dv_evaluation_load_scenes_used,
    get_dense_sparse_splits_with_paths as dl3dv_evaluation_get_dense_sparse_splits_with_paths,
)


@dataclass
class SceneSample:
    input_paths: list[Path]
    test_paths: list[Path]


@dataclass
class SceneBatch:
    name: str
    scene_dir: Path
    sample: SceneSample


class DatasetAdapter:
    def __init__(self, dataset_name: str, cfg: Any, cfg_dict: dict[str, Any]):
        self.dataset_name = dataset_name
        self.cfg = cfg
        self.cfg_dict = cfg_dict

    def get_wandb_config(self) -> dict[str, Any]:
        return {"dataset": self.dataset_name}

    def get_metrics_experiment_fields(self) -> dict[str, Any]:
        return {"dataset": self.dataset_name}

    def iter_scene_batches(self, num_context: int, dense_sparse_cfg: dict[str, Any]):
        raise NotImplementedError


class VRNerfAdapter(DatasetAdapter):
    def __init__(self, cfg: Any, cfg_dict: dict[str, Any], vr_sampler_cfg: Any, vr_sampler_dict: dict[str, Any]):
        super().__init__("vr-nerf", cfg, cfg_dict)
        self.vr_sampler_cfg = vr_sampler_cfg
        self.vr_sampler_dict = vr_sampler_dict

        vr_cfg = cfg.get("vr-nerf", {})
        # Try vr-nerf config, then vr_sampler config, then default
        self.dataset_root = Path(str(vr_cfg.get("dataset_root", vr_sampler_dict.get("dataset_root", "datasets-raw/vrnerf"))))
        self.scenes_path = Path(str(vr_cfg.get("scenes", vr_sampler_dict.get("scenes", "datasets-raw/vrnerf/scenes_used.json"))))

        self.camera_id = str(vr_cfg.get("camera_id", "20"))
        self.fisheye_camera_id = str(vr_cfg.get("fisheye_camera_id", "4"))
        
        # Handle images_subdir which can be null or a string
        images_subdir_raw = vr_cfg.get("images_subdir", None)
        if images_subdir_raw is None or (isinstance(images_subdir_raw, str) and images_subdir_raw == "null"):
            # Use default if null
            self.images_subdir = "images-jpeg-1k"
        else:
            self.images_subdir = str(images_subdir_raw)

        self.fisheye_scenes = self._load_fisheye_scenes()

    def _load_fisheye_scenes(self) -> list[str]:
        fisheye_json_path = self.dataset_root / "fisheye.json"
        if not fisheye_json_path.exists():
            return []
        try:
            with fisheye_json_path.open("r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _ensure_fisheye_processed(self, scene_name: str, scene_dir: Path) -> None:
        if scene_name not in self.fisheye_scenes:
            return
        undistorted_dir = scene_dir / self.images_subdir / self.fisheye_camera_id / "undistorted"
        if undistorted_dir.exists() and any(undistorted_dir.glob("*.jpg")):
            return
        process_scene_cameras_fisheye(
            scene_dir=scene_dir,
            camera_ids=[self.fisheye_camera_id],
            images_subdir=self.images_subdir,
            output_subdir="undistorted",
            balance=0.0,
            crop=True,
            verbose=True,
        )

    def get_wandb_config(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "camera_id": self.camera_id,
            "fisheye_camera_id": self.fisheye_camera_id,
            "images_subdir": self.images_subdir,
        }

    def get_metrics_experiment_fields(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "camera_id": self.camera_id,
            "fisheye_camera_id": self.fisheye_camera_id,
        }

    def iter_scene_batches(self, num_context: int, dense_sparse_cfg: dict[str, Any]):
        scenes = vrnerf_load_scenes_used(
            dataset_root=self.dataset_root,
            scenes_used_path=self.scenes_path,
        )
        pool_size = int(dense_sparse_cfg.get("pool_size", 72))
        pool_stride = int(dense_sparse_cfg.get("pool_stride", 2))
        num_test = int(dense_sparse_cfg.get("num_test", 8))
        seed = int(dense_sparse_cfg.get("seed", 0))

        for item in scenes:
            scene_name = item["scene"]
            scene_dir = item["scene_dir"]
            if not scene_dir.exists():
                continue
            try:
                self._ensure_fisheye_processed(scene_name, scene_dir)
                active_camera_id = self.fisheye_camera_id if scene_name in self.fisheye_scenes else self.camera_id
                splits = vrnerf_get_dense_sparse_splits_with_paths(
                    scene=scene_name,
                    scene_dir=scene_dir,
                    camera_id=active_camera_id,
                    num_input=num_context,
                    num_test=num_test,
                    pool_size=pool_size,
                    pool_stride=pool_stride,
                    seed=seed,
                    images_subdir=self.images_subdir,
                )
                if not splits:
                    continue
                split = splits[0]

                if scene_name in self.fisheye_scenes:
                    input_paths = [get_processed_image_path(p, images_subdir=self.images_subdir) for p in split["input_paths"]]
                    test_paths = [get_processed_image_path(p, images_subdir=self.images_subdir) for p in split["test_paths"]]
                else:
                    input_paths = split["input_paths"]
                    test_paths = split["test_paths"]

                yield SceneBatch(
                    name=scene_name,
                    scene_dir=scene_dir,
                    sample=SceneSample(input_paths=list(input_paths), test_paths=list(test_paths)),
                )
            except Exception as e:
                print(f"[warn] Skip vr-nerf scene '{scene_name}': {e}")


class MatrixCityAdapter(DatasetAdapter):
    def __init__(self, cfg: Any, cfg_dict: dict[str, Any]):
        super().__init__("matrixcity", cfg, cfg_dict)
        matrix_cfg = cfg.get("matrixcity", {})
        self.dataset_root = Path(str(matrix_cfg.get("dataset_root", "datasets/matrixcity")))
        self.scope = str(matrix_cfg.get("scope", "big"))
        self.species = str(matrix_cfg.get("species", "street"))
        
        # Handle images_subdir which can be null or a string
        images_subdir_raw = matrix_cfg.get("images_subdir", None)
        if images_subdir_raw is None or (isinstance(images_subdir_raw, str) and images_subdir_raw == "null"):
            # Use default if null
            self.images_subdir = "images"
        else:
            self.images_subdir = str(images_subdir_raw)
        
        scenes_raw = matrix_cfg.get("scenes", None)
        self.scenes_path = Path(str(scenes_raw)) if scenes_raw else None

    def get_wandb_config(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "scope": self.scope,
            "species": self.species,
            "images_subdir": self.images_subdir,
        }

    def get_metrics_experiment_fields(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "scope": self.scope,
            "species": self.species,
        }

    def iter_scene_batches(self, num_context: int, dense_sparse_cfg: dict[str, Any]):
        scenes = matrixcity_load_scenes_used(
            dataset_root=self.dataset_root,
            scope=self.scope,
            species=self.species,
            scenes_used_path=self.scenes_path,
        )
        pool_size = int(dense_sparse_cfg.get("pool_size", 72))
        pool_stride = int(dense_sparse_cfg.get("pool_stride", 2))
        num_test = int(dense_sparse_cfg.get("num_test", 8))
        seed = int(dense_sparse_cfg.get("seed", 0))

        for item in scenes:
            scene_name = item["scene"]
            scene_dir = item["scene_dir"]
            if not scene_dir.exists():
                continue
            try:
                splits = matrixcity_get_dense_sparse_splits_with_paths(
                    scene=scene_name,
                    scene_dir=scene_dir,
                    camera_id=None,
                    num_input=num_context,
                    num_test=num_test,
                    pool_size=pool_size,
                    pool_stride=pool_stride,
                    seed=seed,
                    images_subdir=self.images_subdir,
                )
                if not splits:
                    continue
                split = splits[0]
                yield SceneBatch(
                    name=scene_name,
                    scene_dir=scene_dir,
                    sample=SceneSample(input_paths=list(split["input_paths"]), test_paths=list(split["test_paths"])),
                )
            except Exception as e:
                print(f"[warn] Skip matrixcity scene '{scene_name}': {e}")
class DL3DVEvaluationAdapter(DatasetAdapter):
    def __init__(self, cfg: Any, cfg_dict: dict[str, Any]):
        super().__init__("dl3dv-evaluation", cfg, cfg_dict)
        dl3dv_eval_cfg = cfg.get("dl3dv-evaluation", cfg.get("dl3dv", {}))
        
        # dataset_root is required
        self.dataset_root = Path(str(dl3dv_eval_cfg.get("dataset_root", "datasets/dl3dv/DL3DV-Evaluation")))
        
        # images_subdir can be null; combine with dataset_root to get base_dir
        images_subdir_raw = dl3dv_eval_cfg.get("images_subdir", None)
        if images_subdir_raw is None or (isinstance(images_subdir_raw, str) and images_subdir_raw == "null"):
            self.base_dir = self.dataset_root
        else:
            self.base_dir = self.dataset_root / str(images_subdir_raw)
        
        # index specifies which subdirectory under gaussian_splat to use for images
        self.index = str(dl3dv_eval_cfg.get("index", "images_4"))
        
        # split determines which index file to load (train/test)
        self.split = str(dl3dv_eval_cfg.get("split", "test"))
        
        # max_scenes limits the number of scenes to process
        self.max_scenes = dl3dv_eval_cfg.get("max_scenes", None)

    def get_wandb_config(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "index": self.index,
            "split": self.split,
            "base_dir": str(self.base_dir),
        }

    def get_metrics_experiment_fields(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "split": self.split,
        }

    def iter_scene_batches(self, num_context: int, dense_sparse_cfg: dict[str, Any]):
        # Load scenes from dataset_root (will auto-detect /images subdirectory if needed)
        # Pass the dataset_root, not base_dir, so the function can handle /images lookup
        scenes = dl3dv_evaluation_load_scenes_used(
            dataset_root=self.dataset_root,
            split=self.split,
            max_scenes=self.max_scenes,
        )
        pool_size = dense_sparse_cfg.get("pool_size", None)
        pool_stride = int(dense_sparse_cfg.get("pool_stride", 1))
        num_test = int(dense_sparse_cfg.get("num_test", 8))
        seed = int(dense_sparse_cfg.get("seed", 0))

        for item in scenes:
            scene_name = item["scene"]
            scene_dir = item["scene_dir"]
            if not scene_dir.exists():
                continue
            try:
                splits = dl3dv_evaluation_get_dense_sparse_splits_with_paths(
                    scene_dir=scene_dir,
                    index=self.index,
                    num_input=num_context,
                    num_test=num_test,
                    seed=seed,
                    pool_size=pool_size,
                    pool_stride=pool_stride,
                )
                if not splits:
                    continue
                split = splits[0]
                yield SceneBatch(
                    name=scene_name,
                    scene_dir=scene_dir,
                    sample=SceneSample(input_paths=list(split["input_paths"]), test_paths=list(split["test_paths"])),
                )
            except Exception as e:
                print(f"[warn] Skip dl3dv-evaluation scene '{scene_name}': {e}")




def build_dataset_adapter(cfg: Any) -> DatasetAdapter:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}
    if not isinstance(cfg_dict, dict):
        cfg_dict = {}

    exp_cfg = cfg.get("experiment", {}) if cfg is not None else {}
    dataset_name = str(exp_cfg.get("dataset", "vr-nerf"))

    if dataset_name == "vr-nerf":
        try:
            vr_sampler_cfg = OmegaConf.load("config/vrnerf_sampler.yaml")
        except Exception:
            vr_sampler_cfg = OmegaConf.create({})
        vr_sampler_dict = OmegaConf.to_container(vr_sampler_cfg, resolve=True)
        if not isinstance(vr_sampler_dict, dict):
            vr_sampler_dict = {}
        return VRNerfAdapter(cfg, cfg_dict, vr_sampler_cfg, vr_sampler_dict)

    if dataset_name == "matrixcity":
        return MatrixCityAdapter(cfg, cfg_dict)

    if dataset_name in {"dl3dv-evaluation", "dl3dv"}:
        return DL3DVEvaluationAdapter(cfg, cfg_dict)

    raise ValueError(f"Unknown dataset: {dataset_name}. Supported: 'vr-nerf', 'matrixcity', 'dl3dv-evaluation'")
