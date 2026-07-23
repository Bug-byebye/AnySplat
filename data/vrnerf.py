"""
VRNeRF Dataset — 多焦距相机场景数据集。
=======================================

数据路径: /data/sunchang/datasets-raw/vrnerf/{scene}/
结构:
  cameras.json       — 全部相机参数（KRT 列表，3804条）
  images-jpeg-1k/    — 图像，按相机ID分目录
    {cam_id}/{cam_id}_DSC{number}.jpg
  splits.json        — train/test 划分

每个相机ID对应不同的焦距（10=24mm, 24=85mm, 等），
每个有 ~180 张不同视角的图片。

Usage:
    from post.data import build_dataset
    dataset = build_dataset({
        'dataset': 'vrnerf',
        'root': '/data/sunchang/datasets-raw/vrnerf',
        'scene': 'apartment',
        'cameras': ['10', '24'],     # 空列表 = 全部相机
        'context_views': 2,
        'target_gap': [8, 15],
        'resolution': [224, 448],
    }, split='train')
    sample = dataset[0]  # -> dict with ctx_images, ctx_c2w, ctx_intr, tgt_image, ...
"""

import json
import os
import random
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image
import torch

from . import register


@register("vrnerf")
def make_vrnerf_dataset(cfg: dict, split: str = "train") -> "VRNeRFDataset":
    """
    VRNeRF 数据集工厂函数。

    Args:
        cfg: 数据配置 dict，含 root, scene, cameras, context_views, target_gap, resolution。
        split: "train" 或 "test"。

    Returns:
        VRNeRFDataset 实例。
    """
    root = Path(cfg["root"])
    scene = cfg.get("scene", "apartment")
    cameras = cfg.get("cameras", [])  # [] = all
    ctx_v = cfg.get("context_views", 2)
    tgt_gap = cfg.get("target_gap", [8, 15])
    resolution = cfg.get("resolution", [224, 448])

    return VRNeRFDataset(
        scene_dir=root / scene,
        split=split,
        camera_ids=cameras,
        context_views=ctx_v,
        target_gap=tgt_gap,
        resolution=resolution,
    )


def load_video_sequence(cfg: dict) -> "VRNeRFVideoSequence":
    """
    从 VRNeRF 场景加载均匀采样的视频序列（用于 WAN 生成演示）。

    Args:
        cfg: 需含 root, scene, camera, sample_step, max_frames, resolution。

    Returns:
        VRNeRFVideoSequence 实例。
    """
    root = Path(cfg["root"])
    scene = cfg.get("scene", "apartment")
    camera = cfg.get("camera", "10")
    step = cfg.get("sample_step", 2)       # 每 step 帧取一帧
    max_frames = cfg.get("max_frames", 16)  # 最多取多少帧
    resolution = cfg.get("resolution", [224, 448])

    return VRNeRFVideoSequence(
        scene_dir=root / scene,
        camera_id=camera,
        sample_step=step,
        max_frames=max_frames,
        resolution=resolution,
    )


class VRNeRFDataset:
    """
    VRNeRF 数据集。

    每次 __getitem__ 返回一个 dict：
        ctx_images:  [V, 3, H, W]  上下文帧
        ctx_c2w:     [V, 4, 4]     上下文 c2w
        ctx_intr:    [V, 3, 3]     上下文内参
        tgt_image:   [3, H, W]     目标帧
        tgt_c2w:     [4, 4]        目标 c2w
        tgt_intr:    [3, 3]        目标内参
        scene_name:  str           场景名
    """

    def __init__(
        self,
        scene_dir: Path,
        split: str = "train",
        camera_ids: List[str] = None,
        context_views: int = 2,
        target_gap: Tuple[int, int] = (8, 15),
        resolution: Tuple[int, int] = (224, 448),
    ):
        self.scene_dir = Path(scene_dir)
        self.split = split
        self.camera_filter = camera_ids or None  # None = use all
        self.context_views = context_views
        self.target_gap = target_gap
        self.H, self.W = resolution
        self.scene_name = scene_dir.name

        # ---- 加载相机参数 ----
        with open(self.scene_dir / "cameras.json") as f:
            cam_data = json.load(f)
        krt_list = cam_data["KRT"]
        # 构建 cameraId → {K, T, w, h} 的索引
        self.camera_index = {}
        for entry in krt_list:
            cid = entry["cameraId"]  # e.g. "10/10_DSC0001"
            self.camera_index[cid] = {
                "K": np.array(entry["K"], dtype=np.float32).reshape(3, 3).T,
                "T": np.array(entry["T"], dtype=np.float32).reshape(4, 4),  # w2c
                "width": entry["width"],
                "height": entry["height"],
            }

        # ---- 加载划分 ----
        with open(self.scene_dir / "splits.json") as f:
            self.splits = json.load(f)

        all_entries_raw = self.splits.get(split, [])

        # 过滤掉没有 KRT 相机参数的条目
        all_entries = [e for e in all_entries_raw if e in self.camera_index]
        skipped = len(all_entries_raw) - len(all_entries)
        print(f"[VRNeRF] {scene_dir.name}/{split}: {len(all_entries_raw)} total, "
              f"{skipped} missing KRT, {len(all_entries)} usable")

        # ---- 按相机 ID 过滤（焦距选择） ----
        if self.camera_filter:
            filtered = []
            for entry in all_entries:
                cam_id = entry.split("/")[0]
                if cam_id in self.camera_filter:
                    filtered.append(entry)
            all_entries = filtered
            print(f"         -> after camera filter {camera_ids}: {len(all_entries)} entries")

        # ---- 构建 (context_idx, target_idx) 配对 ----
        # 同一个相机内的连续帧
        self.pairs = []

        # 按相机分组
        from collections import defaultdict
        cam_groups = defaultdict(list)
        for entry in all_entries:
            cam_id, frame_name = entry.split("/")
            # 提取帧号
            num = int(frame_name.split("_DSC")[1])  # e.g. 10_DSC0001 → 1
            cam_groups[cam_id].append((num, entry))

        # 每个相机内: 按帧号排序, 用滑动窗口构建 (ctx_start, target) 配对
        for cam_id, frames in cam_groups.items():
            frames.sort(key=lambda x: x[0])
            # 帧号列表
            nums = [f[0] for f in frames]
            entries = [f[1] for f in frames]
            n = len(nums)

            min_gap, max_gap = target_gap
            for t_idx in range(min_gap, n):
                for gap in range(min_gap, min(max_gap + 1, n - t_idx)):
                    ctx_start = t_idx - gap
                    if ctx_start < 0:
                        continue
                    # 从 ctx_start 取 context_views 帧（连续）
                    if ctx_start + context_views <= t_idx:
                        self.pairs.append((
                            entries[ctx_start:ctx_start + context_views],
                            entries[t_idx],
                        ))

        random.shuffle(self.pairs)
        print(f"         -> {len(self.pairs)} (ctx, tgt) pairs")

    def _index_of(self, camera_id: str):
        """查找 camera_id 是否在 camera_index 中。"""
        if camera_id in self.camera_index:
            return camera_id
        return None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        ctx_ids, tgt_id = self.pairs[idx]

        # ---- 加载上下文帧 ----
        ctx_images = []
        ctx_c2w = []
        ctx_intr = []

        for cid in ctx_ids:
            cam_id, frame_name = cid.split("/")
            # 加载图像
            img_path = self.scene_dir / "images-jpeg-1k" / cam_id / f"{frame_name}.jpg"
            img = Image.open(img_path).convert("RGB").resize((self.W, self.H), Image.LANCZOS)
            img_t = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
            ctx_images.append(img_t)

            # 加载相机参数
            cam_entry = self.camera_index[cid]
            ctx_c2w.append(self._w2c_to_c2w(cam_entry["T"]))
            ctx_intr.append(self._scale_intrinsics(
                cam_entry["K"], cam_entry["width"], cam_entry["height"]
            ))

        # ---- 加载目标帧 ----
        t_cam_id, t_frame_name = tgt_id.split("/")
        tgt_path = self.scene_dir / "images-jpeg-1k" / t_cam_id / f"{t_frame_name}.jpg"
        tgt_img = Image.open(tgt_path).convert("RGB").resize((self.W, self.H), Image.LANCZOS)
        tgt_t = torch.tensor(np.array(tgt_img), dtype=torch.float32).permute(2, 0, 1) / 255.0

        tgt_entry = self.camera_index[tgt_id]
        tgt_c2w = self._w2c_to_c2w(tgt_entry["T"])
        tgt_intr = self._scale_intrinsics(
            tgt_entry["K"], tgt_entry["width"], tgt_entry["height"]
        )

        return {
            "ctx_images": torch.stack(ctx_images),           # [V, 3, H, W]
            "ctx_c2w": torch.stack(ctx_c2w),                 # [V, 4, 4]
            "ctx_intr": torch.stack(ctx_intr),               # [V, 3, 3]
            "tgt_image": tgt_t,                              # [3, H, W]
            "tgt_c2w": tgt_c2w,                              # [4, 4]
            "tgt_intr": tgt_intr,                            # [3, 3]
            "scene_name": self.scene_name,
            "ctx_ids": ctx_ids,
            "tgt_id": tgt_id,
        }

    def _w2c_to_c2w(self, T_w2c: np.ndarray) -> torch.Tensor:
        """world-to-camera 4x4 → camera-to-world 4x4。"""
        T_c2w = np.linalg.inv(T_w2c)
        return torch.from_numpy(T_c2w.astype(np.float32))

    def _scale_intrinsics(self, K: np.ndarray, orig_w: int, orig_h: int) -> torch.Tensor:
        """将内参缩放到当前分辨率。"""
        K = K.copy()
        sx = self.W / orig_w
        sy = self.H / orig_h
        K[0] *= sx   # fx, 0, cx * sx
        K[1] *= sy   # 0, fy, cy * sy
        return torch.from_numpy(K.astype(np.float32))


class VRNeRFVideoSequence:
    """
    均匀采样的视频序列（用于 WAN 生成演示）。

    从单个相机的所有帧中均匀采样，形成视频片段。

    Args:
        scene_dir: 场景路径。
        camera_id: 相机 ID（如 "10" 表示 24mm 焦距）。
        sample_step: 每隔 step 帧取一帧。
        max_frames: 最多取多少帧。
        resolution: (H, W)。

    Usage:
        seq = VRNeRFVideoSequence(
            scene_dir=Path("/data/.../vrnerf/apartment"),
            camera_id="10", sample_step=2, max_frames=8,
        )
        frames = seq.frames  # [T, 3, H, W] 视频帧
        c2w = seq.c2w        # [T, 4, 4] 每帧的相机位姿
        intr = seq.intr      # [T, 3, 3] 每帧的内参
    """

    def __init__(
        self,
        scene_dir: Path,
        camera_id: str = "10",
        sample_step: int = 2,
        max_frames: int = 16,
        resolution: Tuple[int, int] = (224, 448),
    ):
        self.scene_dir = Path(scene_dir)
        self.camera_id = camera_id
        self.sample_step = sample_step
        self.max_frames = max_frames
        self.H, self.W = resolution

        # ---- 加载相机参数 ----
        with open(self.scene_dir / "cameras.json") as f:
            cam_data = json.load(f)

        # 构建索引
        self.camera_index = {}
        for entry in cam_data["KRT"]:
            self.camera_index[entry["cameraId"]] = {
                "K": np.array(entry["K"], dtype=np.float32).reshape(3, 3).T,
                "T": np.array(entry["T"], dtype=np.float32).reshape(4, 4),
                "width": entry["width"],
                "height": entry["height"],
            }

        # ---- 获取该相机的所有帧 ----
        img_dir = self.scene_dir / "images-jpeg-1k" / camera_id
        all_images = sorted(os.listdir(img_dir))  # ["XX_DSC0001.jpg", ...]

        # 构建完整 ID 列表
        frame_ids = []
        for fname in all_images:
            name = fname.replace(".jpg", "")
            cid = f"{camera_id}/{name}"
            if cid in self.camera_index:
                frame_ids.append((name, cid))

        # 按帧号排序
        def frame_num(name):
            return int(name.split("_DSC")[1])
        frame_ids.sort(key=lambda x: frame_num(x[0]))

        # ---- 均匀采样 ----
        sampled = frame_ids[::sample_step][:max_frames]
        print(f"[VRNeRFVideo] camera={camera_id}, step={sample_step}, "
              f"total={len(frame_ids)} frames → sampled {len(sampled)} frames")

        # ---- 加载 ----
        self.frames_list = []
        self.c2w_list = []
        self.intr_list = []
        self.frame_names = []

        for name, cid in sampled:
            # 图像
            img_path = img_dir / f"{name}.jpg"
            img = Image.open(img_path).convert("RGB").resize(
                (self.W, self.H), Image.LANCZOS)
            img_t = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1) / 255.0
            self.frames_list.append(img_t)

            # 相机
            entry = self.camera_index[cid]
            T_w2c = entry["T"]
            T_c2w = np.linalg.inv(T_w2c)
            self.c2w_list.append(torch.from_numpy(T_c2w.astype(np.float32)))

            K = entry["K"].copy()
            sx = self.W / entry["width"]
            sy = self.H / entry["height"]
            K[0] *= sx
            K[1] *= sy
            self.intr_list.append(torch.from_numpy(K.astype(np.float32)))

            self.frame_names.append(name)

        self.frames = torch.stack(self.frames_list)  # [T, 3, H, W]
        self.c2w = torch.stack(self.c2w_list)         # [T, 4, 4]
        self.intr = torch.stack(self.intr_list)       # [T, 3, 3]

        print(f"  -> video shape: {self.frames.shape}")

    def __len__(self):
        return len(self.frames)


def quick_test():
    """快速测试：加载一个 sample 并打印。"""
    cfg = {
        "root": "/data/sunchang/datasets-raw/vrnerf",
        "scene": "apartment",
        "cameras": ["10"],
        "context_views": 2,
        "target_gap": [8, 15],
        "resolution": [224, 448],
    }
    ds = make_vrnerf_dataset(cfg, split="train")
    print(f"Dataset: {len(ds)} pairs")
    sample = ds[0]
    print(f"ctx_images: {sample['ctx_images'].shape}")
    print(f"ctx_c2w: {sample['ctx_c2w'].shape}")
    print(f"ctx_intr: {sample['ctx_intr'].shape}")
    print(f"tgt_image: {sample['tgt_image'].shape}")
    print(f"scene_name: {sample['scene_name']}")
    print(f"ctx_ids: {sample['ctx_ids']}")
    print(f"tgt_id: {sample['tgt_id']}")
    return sample


if __name__ == "__main__":
    quick_test()
