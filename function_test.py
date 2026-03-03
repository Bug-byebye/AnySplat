#!/usr/bin/env python3
"""
测试 get_dense_sparse_splits 与 get_dense_sparse_splits_with_paths 函数
"""

from pathlib import Path

from scripts.vrnerf.vrnerf_sampler import (
    get_dense_sparse_splits,
    get_dense_sparse_splits_with_paths,
)


def main():
    scene = "kitchen"
    scene_dir = Path("datasets-raw/vrnerf/kitchen")
    camera_id = "20"

    if not scene_dir.exists():
        print(f"[错误] 场景目录不存在: {scene_dir}")
        return

    print("=" * 70)
    print("测试 get_dense_sparse_splits / get_dense_sparse_splits_with_paths")
    print("=" * 70)
    print(f"场景: {scene}, 相机块: {camera_id}, 场景目录: {scene_dir}")
    print()

    # -------------------------------------------------------------------------
    # 1. Dense-view 测试
    # -------------------------------------------------------------------------
    print("-" * 70)
    print("【1】Dense-view 设置 (num_input=None → 返回 64/48/32 三种规模)")
    print("-" * 70)

    dense_splits = get_dense_sparse_splits(
        scene=scene,
        scene_dir=scene_dir,
        camera_id=camera_id,
        setting="dense",
        num_input=None,
        pool_size=72,
        pool_stride=2,
        seed=0,
    )

    print(f"返回 split 数量: {len(dense_splits)}")
    for i, s in enumerate(dense_splits):
        print(f"\n  Split[{i}]:")
        print(f"    setting: {s['setting']}")
        print(f"    num_input: {s['num_input']}")
        print(f"    num_test: {s['num_test']}")
        print(f"    input 前 3 个 frame_id: {s['input'][:3]}")
        print(f"    test 全部 frame_id: {s['test']}")
        # 校验：三种设置的 test 必须完全一致
        if i > 0:
            assert s["test"] == dense_splits[0]["test"], "test views 应完全一致"
            print(f"    [校验] test 与 Split[0] 一致 ✓")

    # 校验 test 无重叠
    for s in dense_splits:
        overlap = set(s["input"]) & set(s["test"])
        assert len(overlap) == 0, f"input 与 test 不应有重叠: {overlap}"
    print("\n  [校验] 所有 split 的 input 与 test 无重叠 ✓")

    # -------------------------------------------------------------------------
    # 2. Dense-view 指定 num_input
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("【2】Dense-view 设置 (num_input=32 → 仅返回 32-view)")
    print("-" * 70)

    dense_32 = get_dense_sparse_splits(
        scene=scene,
        scene_dir=scene_dir,
        camera_id=camera_id,
        setting="dense",
        num_input=32,
        seed=42,
    )
    print(f"返回 split 数量: {len(dense_32)}")
    s = dense_32[0]
    print(f"  num_input: {s['num_input']}, num_test: {s['num_test']}")
    print(f"  input 前 5 个: {s['input'][:5]}")

    # 可复现性：相同 seed 应得到相同结果
    dense_32_repeat = get_dense_sparse_splits(
        scene=scene,
        scene_dir=scene_dir,
        camera_id=camera_id,
        setting="dense",
        num_input=32,
        seed=42,
    )
    assert dense_32[0]["input"] == dense_32_repeat[0]["input"], "相同 seed 应可复现"
    print("  [校验] seed=42 重复调用结果一致 ✓")

    # -------------------------------------------------------------------------
    # 3. Sparse-view 测试
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("【3】Sparse-view 设置")
    print("-" * 70)

    sparse_splits = get_dense_sparse_splits(
        scene=scene,
        scene_dir=scene_dir,
        camera_id=camera_id,
        setting="sparse",
    )
    print(f"返回 split 数量: {len(sparse_splits)}")
    s = sparse_splits[0]
    print(f"  num_input: {s['num_input']}, num_test: {s['num_test']}")
    print(f"  input 前 3 个: {s['input'][:3]}")
    print(f"  test 前 5 个: {s['test'][:5]}")

    # -------------------------------------------------------------------------
    # 4. get_dense_sparse_splits_with_paths 测试
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("【4】get_dense_sparse_splits_with_paths (带路径)")
    print("-" * 70)

    splits_with_paths = get_dense_sparse_splits_with_paths(
        scene=scene,
        scene_dir=scene_dir,
        camera_id=camera_id,
        setting="dense",
        num_input=32,
        seed=0,
    )
    s = splits_with_paths[0]
    print(f"  num_input: {s['num_input']}, num_test: {s['num_test']}")
    print(f"  input_paths 前 2 个: {s['input_paths'][:2]}")
    print(f"  test_paths 前 2 个: {s['test_paths'][:2]}")

    # 校验路径存在
    for p in s["input_paths"][:3] + s["test_paths"][:2]:
        exists = p.exists()
        print(f"    路径存在? {exists}: {p}")
        if not exists:
            print(f"    [警告] 路径不存在!")

    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)


if __name__ == "__main__":
    main()
