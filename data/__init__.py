"""
数据模块 — 统一的数据集接口。

所有数据集实现相同的接口（返回 dict）：
  {
    'ctx_images':   [V, 3, H, W] 上下文图像, 归一化到 [0,1]
    'ctx_c2w':      [V, 4, 4]     上下文相机 (camera-to-world)
    'ctx_intr':     [V, 3, 3]     上下文内参
    'tgt_image':    [3, H, W]     目标图像
    'tgt_c2w':      [4, 4]        目标相机
    'tgt_intr':     [3, 3]        目标内参
    'scene_name':   str           场景名
  }

添加新数据集：在 post/data/ 下新建文件，实现 make_dataset() 函数，
并在此 register 中注册。
"""

from typing import Dict, Optional
from pathlib import Path
import yaml


# 数据集注册表：名称 → 构建函数
_REGISTRY: Dict[str, callable] = {}


def register(name: str):
    """装饰器：注册数据集构建函数。"""
    def wrapper(fn):
        _REGISTRY[name] = fn
        return fn
    return wrapper


def build_dataset(cfg: dict, split: str = "train"):
    """
    根据配置构建数据集。

    Args:
        cfg: data 部分的配置 dict，必须有 'dataset' 和 'root' 字段。
        split: "train" / "val" / "test"

    Returns:
        dataset: 可迭代的数据集对象（list of dicts 或 torch Dataset）。
    """
    name = cfg.get("dataset", "vrnerf")
    if name not in _REGISTRY:
        raise ValueError(f"未知数据集 '{name}'。可用: {list(_REGISTRY.keys())}")

    builder = _REGISTRY[name]
    return builder(cfg, split=split)


def load_config(path: str) -> dict:
    """加载 YAML 配置文件。"""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# 导入数据集实现，触发注册
from . import vrnerf
