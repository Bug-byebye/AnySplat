#!/usr/bin/env python3
"""
鱼眼图像去畸变处理脚本
功能: 将鱼眼图像转换为普通图像
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, Tuple
import argparse


def estimate_fisheye_params(img_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    估算鱼眼相机的内参矩阵和畸变系数
    
    Args:
        img_shape: 图像形状 (height, width)
    
    Returns:
        K: 相机内参矩阵
        D: 畸变系数
    """
    h, w = img_shape
    
    # 估算相机内参矩阵
    # 焦距通常约为图像宽度
    focal_length = w * 0.8
    cx, cy = w / 2, h / 2
    
    K = np.array([[focal_length, 0, cx],
                  [0, focal_length, cy],
                  [0, 0, 1]], dtype=np.float64)
    
    # 鱼眼畸变系数 [k1, k2, k3, k4]
    # 这些是经验值,可能需要根据实际情况调整
    D = np.array([[-0.3], [0.1], [0.0], [0.0]], dtype=np.float64)
    
    return K, D


def undistort_fisheye_image(
    img: np.ndarray,
    K: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
    balance: float = 0.0,
    crop: bool = True
) -> np.ndarray:
    """
    对单张鱼眼图像进行去畸变处理
    
    Args:
        img: 输入的鱼眼图像
        K: 相机内参矩阵 (3x3), 如果为None则自动估算
        D: 畸变系数 (4x1), 如果为None则自动估算
        balance: 平衡参数 (0.0-1.0), 0保留所有像素,1最大化视野
        crop: 是否裁剪黑边
    
    Returns:
        去畸变后的图像
    """
    h, w = img.shape[:2]
    
    # 如果没有提供相机参数,则估算
    if K is None or D is None:
        K, D = estimate_fisheye_params((h, w))
    
    # 计算新的相机矩阵
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )
    
    # 计算映射表
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2
    )
    
    # 应用去畸变
    undistorted = cv2.remap(
        img, map1, map2, 
        interpolation=cv2.INTER_LINEAR, 
        borderMode=cv2.BORDER_CONSTANT
    )
    
    # 可选: 裁剪黑边
    if crop:
        # 找到有效区域
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY) if len(undistorted.shape) == 3 else undistorted
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            x, y, w_crop, h_crop = cv2.boundingRect(contours[0])
            # 添加一些边距以避免过度裁剪
            margin = 5
            x = max(0, x - margin)
            y = max(0, y - margin)
            w_crop = min(w - x, w_crop + 2 * margin)
            h_crop = min(h - y, h_crop + 2 * margin)
            undistorted = undistorted[y:y+h_crop, x:x+w_crop]
    
    return undistorted


def process_fisheye_images(
    input_dir: str,
    output_subdir: str = "undistorted",
    K: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
    balance: float = 0.0,
    crop: bool = True,
    image_extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
) -> None:
    """
    批量处理目录下的鱼眼图像
    
    Args:
        input_dir: 输入图像目录的绝对路径
        output_subdir: 输出子目录名称,将在input_dir下创建
        K: 相机内参矩阵,为None时自动估算
        D: 畸变系数,为None时自动估算
        balance: 平衡参数 (0.0-1.0)
        crop: 是否裁剪黑边
        image_extensions: 要处理的图像文件扩展名
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        raise ValueError(f"输入目录不存在: {input_dir}")
    
    # 创建输出目录
    output_path = input_path / output_subdir
    output_path.mkdir(exist_ok=True)
    
    print(f"输入目录: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"处理参数: balance={balance}, crop={crop}")
    print("-" * 60)
    
    # 获取所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_path.glob(f"*{ext}")))
        image_files.extend(list(input_path.glob(f"*{ext.upper()}")))
    
    if not image_files:
        print(f"在 {input_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像")
    
    # 处理每张图像
    success_count = 0
    for i, img_file in enumerate(image_files, 1):
        try:
            print(f"[{i}/{len(image_files)}] 处理: {img_file.name} ... ", end="", flush=True)
            
            # 读取图像
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"失败 (无法读取)")
                continue
            
            # 去畸变
            undistorted = undistort_fisheye_image(img, K, D, balance, crop)
            
            # 保存
            output_file = output_path / img_file.name
            cv2.imwrite(str(output_file), undistorted)
            
            print(f"完成 (原始: {img.shape[1]}x{img.shape[0]} -> 处理后: {undistorted.shape[1]}x{undistorted.shape[0]})")
            success_count += 1
            
        except Exception as e:
            print(f"失败 ({str(e)})")
    
    print("-" * 60)
    print(f"处理完成! 成功: {success_count}/{len(image_files)}")
    print(f"输出目录: {output_path}")


def process_fisheye_images_with_custom_params(
    input_dir: str,
    output_subdir: str = "undistorted",
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    k1: float = -0.3,
    k2: float = 0.1,
    k3: float = 0.0,
    k4: float = 0.0,
    balance: float = 0.0,
    crop: bool = True
) -> None:
    """
    使用自定义相机参数批量处理鱼眼图像
    
    Args:
        input_dir: 输入图像目录
        output_subdir: 输出子目录名称
        fx, fy: 焦距参数
        cx, cy: 主点坐标
        k1, k2, k3, k4: 畸变系数
        balance: 平衡参数
        crop: 是否裁剪黑边
    """
    # 读取一张图像以获取尺寸
    input_path = Path(input_dir)
    sample_images = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))
    
    if not sample_images:
        raise ValueError(f"目录中没有图像文件: {input_dir}")
    
    sample_img = cv2.imread(str(sample_images[0]))
    h, w = sample_img.shape[:2]
    
    # 构建相机参数
    if fx is None:
        fx = w * 0.8
    if fy is None:
        fy = fx
    if cx is None:
        cx = w / 2
    if cy is None:
        cy = h / 2
    
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]], dtype=np.float64)
    
    D = np.array([[k1], [k2], [k3], [k4]], dtype=np.float64)
    
    print(f"相机内参矩阵 K:")
    print(K)
    print(f"\n畸变系数 D: {D.flatten()}")
    print()
    
    # 调用主处理函数
    process_fisheye_images(input_dir, output_subdir, K, D, balance, crop)


def process_scene_cameras_fisheye(
    scene_dir: Path | str,
    camera_ids: list[str | int],
    images_subdir: str = "images-jpeg-1k",
    output_subdir: str = "undistorted",
    balance: float = 0.0,
    crop: bool = True,
    verbose: bool = True,
) -> list[str]:
    """
    For a given scene, process fisheye images in specified camera blocks.
    
    Args:
        scene_dir: Path to scene directory
        camera_ids: List of camera IDs to process (e.g., ["5", "20"])
        images_subdir: Name of images subdirectory
        output_subdir: Name of output subdirectory for undistorted images
        balance: Balance parameter for undistortion
        crop: Whether to crop black borders
        verbose: Print processing info
    
    Returns:
        List of camera IDs that were successfully processed
    """
    scene_dir = Path(scene_dir)
    processed_cameras = []
    
    for camera_id in camera_ids:
        camera_id = str(camera_id)
        block_dir = scene_dir / images_subdir / camera_id
        
        if not block_dir.exists():
            if verbose:
                print(f"[warn] Camera block dir not found: {block_dir}")
            continue
        
        if verbose:
            print(f"[info] Processing fisheye images in camera {camera_id}: {block_dir}")
        
        try:
            # Process the images
            process_fisheye_images(
                input_dir=str(block_dir),
                output_subdir=output_subdir,
                K=None,  # Auto-estimate
                D=None,  # Auto-estimate
                balance=balance,
                crop=crop,
            )
            processed_cameras.append(camera_id)
            if verbose:
                print(f"[info] Successfully processed camera {camera_id}")
        except Exception as e:
            if verbose:
                print(f"[warn] Failed to process camera {camera_id}: {e}")
    
    return processed_cameras


def get_processed_image_path(
    original_path: Path | str,
    images_subdir: str = "images-jpeg-1k",
    processed_subdir: str = "undistorted",
) -> Path:
    """
    Get the path to processed (undistorted) image if it exists, otherwise return original.
    
    Assumes path structure: <scene_dir>/{images_subdir}/{camera_id}/{filename}
    Processed path would be: <scene_dir>/{images_subdir}/{camera_id}/{processed_subdir}/{filename}
    
    Args:
        original_path: Path to original image
        images_subdir: Name of images directory
        processed_subdir: Name of processed images subdirectory
    
    Returns:
        Path to processed image if exists, otherwise original path
    """
    original_path = Path(original_path)
    
    try:
        parts = original_path.parts
        if images_subdir not in parts:
            return original_path
        
        idx = parts.index(images_subdir)
        if idx + 2 >= len(parts):  # Need idx, camera_id, filename
            return original_path
        
        camera_id = parts[idx + 1]
        filename = parts[-1]
        
        # Reconstruct: scene_dir / images_subdir / camera_id / processed_subdir / filename
        scene_dir = Path(*parts[:idx])
        processed_path = scene_dir / images_subdir / camera_id / processed_subdir / filename
        
        if processed_path.exists():
            return processed_path
        return original_path
    except Exception:
        return original_path


def main():
    parser = argparse.ArgumentParser(
        description="鱼眼图像去畸变处理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用 (自动估算参数)
  python fisheye_undistort.py /path/to/images
  
  # 指定输出目录名称
  python fisheye_undistort.py /path/to/images --output rectified
  
  # 调整balance参数以保留更多视野
  python fisheye_undistort.py /path/to/images --balance 0.5
  
  # 不裁剪黑边
  python fisheye_undistort.py /path/to/images --no-crop
  
  # 使用自定义畸变参数
  python fisheye_undistort.py /path/to/images --k1 -0.4 --k2 0.15
        """
    )
    
    parser.add_argument(
        'input_dir',
        type=str,
        help='包含鱼眼图像的输入目录绝对路径'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='undistorted',
        help='输出子目录名称 (默认: undistorted)'
    )
    
    parser.add_argument(
        '--balance', '-b',
        type=float,
        default=0.0,
        help='平衡参数 0.0-1.0, 0保留所有像素, 1最大化视野 (默认: 0.0)'
    )
    
    parser.add_argument(
        '--no-crop',
        action='store_true',
        help='不裁剪黑边'
    )
    
    parser.add_argument(
        '--k1',
        type=float,
        default=-0.3,
        help='畸变系数k1 (默认: -0.3)'
    )
    
    parser.add_argument(
        '--k2',
        type=float,
        default=0.1,
        help='畸变系数k2 (默认: 0.1)'
    )
    
    parser.add_argument(
        '--k3',
        type=float,
        default=0.0,
        help='畸变系数k3 (默认: 0.0)'
    )
    
    parser.add_argument(
        '--k4',
        type=float,
        default=0.0,
        help='畸变系数k4 (默认: 0.0)'
    )
    
    args = parser.parse_args()
    
    try:
        process_fisheye_images_with_custom_params(
            input_dir=args.input_dir,
            output_subdir=args.output,
            k1=args.k1,
            k2=args.k2,
            k3=args.k3,
            k4=args.k4,
            balance=args.balance,
            crop=not args.no_crop
        )
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


