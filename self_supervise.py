
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
import os
import sys
import imageio
from omegaconf import OmegaConf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.misc.image_io import save_interpolated_video,save_rendered_video_no_interpolation
from src.model.ply_export import export_ply
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image
from src.visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics
)
from src.dataset.view_sampler.view_sampler_rank import extrinsic_distance_batch


def load_config(config_path="config/self_supervise.yaml"):
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        OmegaConf.DictConfig: 配置对象
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    cfg = OmegaConf.load(config_path)
    return cfg


def load_images(image_folder):
    """
    加载图像文件夹中的所有图像
    
    Args:
        image_folder: 图像文件夹路径
        
    Returns:
        list: 处理后的图像列表
    """
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"图像文件夹不存在: {image_folder}")
    
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    image_paths = [os.path.join(image_folder, f) for f in image_files]
    images = [process_image(img_path) for img_path in image_paths]
    return images


def sort_cameras_by_trajectory(extrinsics, lambda_t=1.0, normalize=True):
    """
    根据相机位姿对相机进行排序，形成合理的相机轨迹。
    借鉴 view_sampler_rank 的外参距离度量（位置 + 朝向），使用贪心最近邻构建路径。
    
    改进点：
    1. 使用 extrinsic_distance（位置 + 旋转）替代纯欧氏距离，相邻相机在视点上更连贯
    2. 从质心最近的相机作为起点，避免固定从索引 0 带来的轨迹方向随意性
    3. 可选归一化使不同尺度场景下的旋转/平移权重更平衡
    
    Args:
        extrinsics: [batch, num_views, 4, 4] 外参矩阵 (c2w)
        lambda_t: 平移相对于旋转的权重，越大越强调位置邻近
        normalize: 是否对平移部分归一化（按平均距离缩放）
        
    Returns:
        sorted_indices: [num_views] 排序后的索引
    """
    # extrinsics: [batch, num_views, 4, 4] -> [num_views, 4, 4]
    extrinsics_single = extrinsics[0].clone()
    num_views = extrinsics_single.shape[0]
    device = extrinsics.device
    
    if num_views <= 1:
        return torch.arange(num_views, device=device)
    
    # 归一化：使平移尺度与旋转尺度更匹配（与 view_sampler_rank 一致）
    if normalize:
        camera_center = extrinsics_single[:, :3, 3].clone()
        avg_scale = torch.norm(camera_center, dim=1).mean()
        if avg_scale > 1e-8:
            extrinsics_single = extrinsics_single.clone()
            extrinsics_single[:, :3, 3] = extrinsics_single[:, :3, 3] / avg_scale
    
    # 计算成对外参距离（位置 + 朝向），[N, N]
    dists = extrinsic_distance_batch(extrinsics_single, lambda_t=lambda_t)
    
    # 选择起点：质心最近的相机，使轨迹更自然
    positions = extrinsics_single[:, :3, 3]
    centroid = positions.mean(dim=0)
    dists_to_centroid = torch.norm(positions - centroid.unsqueeze(0), dim=1)
    start_idx = dists_to_centroid.argmin().item()
    
    # 贪心最近邻构建轨迹
    sorted_indices = [start_idx]
    visited = {start_idx}
    current_idx = start_idx
    
    for _ in range(num_views - 1):
        # 当前相机到所有相机的距离，未访问的中选最小
        row_dists = dists[current_idx].clone()
        row_dists[list(visited)] = float('inf')
        next_idx = row_dists.argmin().item()
        
        sorted_indices.append(next_idx)
        visited.add(next_idx)
        current_idx = next_idx
    
    return torch.tensor(sorted_indices, device=device)


def pose_interpolation(pred_context_pose, num_interp_frames, image_sorted=True):
    """
    对输入的相机位姿进行插值，生成新视角的位姿。
    每次随机选择一对相邻视角，在这对视角之间插值1帧，重复num_interp_frames次。
    
    Args:
        pred_context_pose: 包含'extrinsic'和'intrinsic'的字典
            - extrinsic: [batch, num_views, 4, 4] 外参矩阵
            - intrinsic: [batch, num_views, 3, 3] 内参矩阵
        num_interp_frames: 总共要插值的帧数（每次随机选择一对，插值1帧，重复num_interp_frames次）
        image_sorted: 是否对相机进行排序
        
    Returns:
        interpolated_poses: 包含插值后的'extrinsic'和'intrinsic'的字典
            - extrinsic: [batch, num_interp_frames, 4, 4]
            - intrinsic: [batch, num_interp_frames, 3, 3]
    """
    import random
    
    pred_all_extrinsic = pred_context_pose['extrinsic']  # [batch, num_views, 4, 4]
    pred_all_intrinsic = pred_context_pose['intrinsic']  # [batch, num_views, 3, 3]
    
    batch_size, num_views = pred_all_extrinsic.shape[:2]
    device = pred_all_extrinsic.device
    
    # 对相机进行排序（如果需要）
    if image_sorted:
        sorted_indices = sort_cameras_by_trajectory(pred_all_extrinsic)
        sorted_extrinsics = pred_all_extrinsic[:, sorted_indices]  # [batch, num_views, 4, 4]
        sorted_intrinsics = pred_all_intrinsic[:, sorted_indices]  # [batch, num_views, 3, 3]
    else:
        sorted_extrinsics = pred_all_extrinsic
        sorted_intrinsics = pred_all_intrinsic
        sorted_indices = torch.arange(num_views, device=device)

    num_views = sorted_extrinsics.shape[1]
    
    # 检查是否有足够的视角对进行插值
    if num_views < 2:
        raise ValueError(f"至少需要2个视角才能进行插值，当前只有 {num_views} 个视角")
    
    # 存储所有插值结果
    interpolated_extrinsics_list = []
    interpolated_intrinsics_list = []
    selected_pairs = []
    
    # 循环 num_interp_frames 次，每次随机选择一对视角插值1帧
    for frame_idx in range(num_interp_frames):
        # 随机选择一对相邻视角
        # 可选的对数：num_views - 1
        pair_idx = random.randint(0, num_views - 2)
        selected_pairs.append((pair_idx, pair_idx + 1))
        
        # 获取选中的一对视角
        start_extrinsic = sorted_extrinsics[:, pair_idx]  # [batch, 4, 4]
        end_extrinsic = sorted_extrinsics[:, pair_idx + 1]  # [batch, 4, 4]
        start_intrinsic = sorted_intrinsics[:, pair_idx]  # [batch, 3, 3]
        end_intrinsic = sorted_intrinsics[:, pair_idx + 1]  # [batch, 3, 3]
        
        # 随机选择插值位置（在0到1之间随机选择一个t值）
        # 这样可以确保每次插值的位置都不同
        t = torch.rand(1, device=device)  # [1] 随机值在 [0, 1) 之间
        
        # 插值外参（使用高级插值方法，基于焦点点旋转）
        # interpolate_extrinsics 返回 [batch, 1, 4, 4]
        interp_extrinsic = interpolate_extrinsics(
            start_extrinsic, 
            end_extrinsic, 
            t
        )  # [batch, 1, 4, 4]
        
        # 插值内参（线性插值）
        # interpolate_intrinsics 返回 [batch, 1, 3, 3]
        interp_intrinsic = interpolate_intrinsics(
            start_intrinsic,
            end_intrinsic,
            t
        )  # [batch, 1, 3, 3]
        
        # 添加到列表
        interpolated_extrinsics_list.append(interp_extrinsic)
        interpolated_intrinsics_list.append(interp_intrinsic)
    
    # 拼接所有插值结果
    # 每个元素是 [batch, 1, 4, 4]，拼接后是 [batch, num_interp_frames, 4, 4]
    interp_extrinsics = torch.cat(interpolated_extrinsics_list, dim=1)  # [batch, num_interp_frames, 4, 4]
    interp_intrinsics = torch.cat(interpolated_intrinsics_list, dim=1)  # [batch, num_interp_frames, 3, 3]
    
    print(f"[info] 随机插值完成：共 {num_interp_frames} 帧，选中的视角对: {selected_pairs[:5]}..." if len(selected_pairs) > 5 else f"[info] 随机插值完成：共 {num_interp_frames} 帧，选中的视角对: {selected_pairs}")
    
    return {
        'extrinsic': interp_extrinsics,  # [batch, num_interp_frames, 4, 4]
        'intrinsic': interp_intrinsics,  # [batch, num_interp_frames, 3, 3]
        'selected_pairs': selected_pairs,  # 返回所有选中的视角对索引列表
        'sorted_indices': sorted_indices  # 返回排序索引，方便调试
    }

def render_images(decoder, pred_context_pose, gaussians):
    pred_all_extrinsic = pred_context_pose['extrinsic']
    pred_all_intrinsic = pred_context_pose['intrinsic']
    num_frames = pred_all_extrinsic.shape[1]
    h, w = 448, 448
    interpolated_output = decoder.forward(
        gaussians,
        pred_all_extrinsic,
        pred_all_intrinsic.float(),
        torch.ones(1, num_frames, device=pred_all_extrinsic.device) * 0.1,
        torch.ones(1, num_frames, device=pred_all_extrinsic.device) * 100,
        (h, w),
    )

    images = interpolated_output.color[0].clip(min=0, max=1)
    return images

def save_images(images, save_path):
    for i in range(images.shape[0]):
        image = images[i].cpu().numpy()
        image = image.transpose(1, 2, 0)
        image = image * 255.0
        image = image.astype(np.uint8)
        imageio.imwrite(save_path / f"{i:04d}.png", image)

def self_supervise(model, images, output_folder, cfg):

    gaussians, pred_context_pose = model.inference((images+1)*0.5)
    num_interp_frames = cfg.images.get("interp_frames", 5)
    iter_num = cfg.inference.get("iter_num", 5)
    image_sorted = cfg.images.get("image_sorted", True)
    combined_images = images
    selected_folder = cfg.images.get("selected_folder", None)

    for i in range(iter_num):
        print(f"[ss-info] 迭代 {i+1}/{iter_num}")
        interpolated_pose = pose_interpolation(
            pred_context_pose, 
            num_interp_frames=num_interp_frames,
            image_sorted=image_sorted
        )
        selected_images = render_images(model.decoder, interpolated_pose, gaussians)
        
        if not selected_folder:
            selected_folder = os.path.join(output_folder, f"selected_images_{i:04d}")
        else:
            selected_folder = os.path.join(output_folder, selected_folder, f"iter_{i:04d}")
        
        os.makedirs(selected_folder, exist_ok=True)
        save_images(selected_images, Path(selected_folder))
        print(f"[ss-info] 图像已保存到: {selected_folder}，数量: {selected_images.shape[0]}")
        
        selected_images = load_images(selected_folder)
        combined_images = combined_images + selected_images
        images = torch.stack(combined_images, dim=0).unsqueeze(0).to(device)  # [1, K+N, 3, 448, 448]
        
        print(f"[ss-info] 使用合并后的图像重新推理，图像数量: {images.shape[1]}")
        gaussians, pred_context_pose = model.inference((images+1)*0.5)

def main():
    cfg = load_config("config/self_supervise.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    if cfg.model.ckpts and cfg.model.ckpt_path:
        print(f"[info] 从检查点加载模型: {cfg.model.ckpt_path}")
        model = build_model_from_checkpoint(cfg.model.ckpt_path, device)
    else:
        print("[info] 从 Hugging Face 加载预训练模型")
        model = AnySplat.from_pretrained("lhjiang/anysplat")
    
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    input_folder = cfg.images.input_folder
    if not input_folder:
        raise ValueError("配置文件中 images.input_folder 不能为空")
    
    print(f"[info] 从文件夹加载图像: {input_folder}")
    raw_images = load_images(input_folder)
    images = torch.stack(raw_images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, 448, 448]
    b, v, _, h, w = images.shape
    print(f"[info] 图像形状: {images.shape}")
    
    print("[info] 运行推理...")
    gaussians, pred_context_pose = model.inference((images+1)*0.5)

    # pred_all_extrinsic = pred_context_pose['extrinsic']
    # pred_all_intrinsic = pred_context_pose['intrinsic']

    output_folder = cfg.images.get("output_folder", None)
    if not output_folder:
        output_folder = os.path.join(input_folder, "output")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"{output_folder}_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)
    
    num_interp_frames = cfg.images.get("interp_frames", 5)
    iter_num = cfg.inference.get("iter_num", 5)
    image_sorted = cfg.images.get("image_sorted", True)
    combined_images = raw_images
    print(f"[info] 开始自监督训练，迭代次数: {iter_num}, 每次插值帧数: {num_interp_frames}")
    
    for i in range(iter_num):
        print(f"[info] 迭代 {i+1}/{iter_num}")
        interpolated_pose = pose_interpolation(
            pred_context_pose, 
            num_interp_frames=num_interp_frames,
            image_sorted=image_sorted
        )
        selected_images = render_images(model.decoder, interpolated_pose, gaussians)
        
        selected_folder = cfg.images.get("selected_folder", None)
        if not selected_folder:
            selected_folder = os.path.join(output_folder, f"selected_images_{i:04d}")
        else:
            selected_folder = os.path.join(output_folder, selected_folder, f"iter_{i:04d}")
        
        os.makedirs(selected_folder, exist_ok=True)
        save_images(selected_images, Path(selected_folder))
        print(f"[info] 图像已保存到: {selected_folder}，数量: {selected_images.shape[0]}")
        

        selected_images = load_images(selected_folder)
        combined_images = combined_images + selected_images
        images = torch.stack(combined_images, dim=0).unsqueeze(0).to(device)  # [1, K+N, 3, 448, 448]
        
        print(f"[info] 使用合并后的图像重新推理，图像数量: {images.shape[1]}")
        gaussians, pred_context_pose = model.inference((images+1)*0.5)

        # pred_all_extrinsic = pred_context_pose['extrinsic']
        # pred_all_intrinsic = pred_context_pose['intrinsic']

        # print(f"[info] 外参形状: {pred_all_extrinsic.shape}, 内参形状: {pred_all_intrinsic.shape}")

    ply_path = Path(output_folder) / "gaussians.ply"
    export_ply(
        gaussians.means[0], 
        gaussians.scales[0], 
        gaussians.rotations[0], 
        gaussians.harmonics[0], 
        gaussians.opacities[0], 
        ply_path
    )
    print(f"[info] 高斯点云已导出到: {ply_path}")
    
    # 保存插值视频（使用最后一次迭代的位姿）
    # 注意：如果希望保存原始位姿的插值视频，应该使用第一次推理的结果
    pred_all_extrinsic = pred_context_pose['extrinsic']
    pred_all_intrinsic = pred_context_pose['intrinsic']
    current_batch_size = pred_all_extrinsic.shape[0]
    # save_rendered_video_no_interpolation(
    save_interpolated_video(
        pred_all_extrinsic, 
        pred_all_intrinsic, 
        current_batch_size,  # 使用当前的batch size
        h, w, 
        gaussians, 
        output_folder, 
        model.decoder
    )
    print(f"[info] 渲染视频已保存到: {output_folder}")

if __name__ == "__main__":
    main()