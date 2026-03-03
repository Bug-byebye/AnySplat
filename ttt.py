from __future__ import annotations

import argparse
import random
from datetime import datetime
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from src.model.model.anysplat import AnySplat
from src.model.ply_export import export_ply
from src.utils.image import process_image
from src.evaluation.metrics import get_lpips
from src.misc.image_io import save_image
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from src.utils.model_loading import load_model_with_fallback

# 复用 self_supervise 中已经实现的一些工具函数


@dataclass
class TTTConfig:
    input_folder: Path
    output_folder: Optional[Path]
    iters: int = 5
    # interp_frames: int = 5
    image_sorted: bool = True
    lr: float = 1e-4
    train_components: Optional[List[str]] = None
    device: str = "auto"
    export_ply: bool = True
    loss_type: str = "l1_mse"
    l1_weight: float = 1.0
    mse_weight: float = 1.0
    lpips_weight: float = 0.1
    context_loss_weight: float = 1.0
    group_size: int = 8
    group_mode: str = "sequential"
    group_stride: Optional[int] = None
    seed: int = 0
    pretrained_model_path: Optional[str] = None


def load_ttt_config(config_path: Path) -> TTTConfig:
    if not config_path.exists():
        raise FileNotFoundError(f"TTT 配置文件不存在: {config_path}")

    raw_cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    if not isinstance(raw_cfg, dict):
        raise ValueError(f"TTT 配置文件格式错误: {config_path}")

    cfg = TTTConfig(**raw_cfg)

    input_folder = Path(cfg.input_folder).expanduser().resolve()
    if cfg.output_folder is None:
        output_root = (input_folder / "ttt_outputs").expanduser().resolve()
    else:
        output_root = Path(cfg.output_folder).expanduser().resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = output_root / f"ttt_{timestamp}"

    if cfg.device in (None, "auto"):
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = str(cfg.device)

    return replace(
        cfg,
        input_folder=input_folder,
        output_folder=output_folder,
        device=device,
    )


def get_ttt_parameters(model: AnySplat, train_components: Optional[List[str]] = None) -> List[torch.nn.Parameter]:

    params: List[torch.nn.Parameter] = []

    if not train_components:
        train_components = ["encoder.gaussian_adapter", "encoder.gaussian_param_head"]

    # 打印所有模型参数名，帮助调试
    print(f"\n[DEBUG] TTT 寻找参数: train_components={train_components}")
    print(f"[DEBUG] 模型中所有参数（包括冻结的）:")
    all_param_names = []
    trainable_param_names = []
    for name, p in model.named_parameters():
        all_param_names.append((name, p.requires_grad))
        if p.requires_grad:
            trainable_param_names.append(name)
    
    if all_param_names:
        print(f"  总数: {len(all_param_names)}")
        print(f"  可训练的参数: {len(trainable_param_names)}")
        print(f"\n  参数列表 (前 30 个):")
        for name, requires_grad in all_param_names[:30]:
            status = "✓" if requires_grad else "✗"
            print(f"    [{status}] {name}")
        if len(all_param_names) > 30:
            print(f"    ... 还有 {len(all_param_names) - 30} 个参数")
    else:
        print("  (模型中没有任何参数！)")

    for name, p in model.named_parameters():
        if any(component in name for component in train_components):
            params.append(p)

    if not params:
        print(f"\n[ERROR] 未找到匹配的参数！")
        print(f"[ERROR] train_components={train_components}")
        print(f"[ERROR] 请检查:")
        print(f"  1. train_components 中的字符串是否与模型参数名称匹配")
        print(f"  2. 模型是否正确加载")
        print(f"  3. 当前模型中包含的参数模块名称（供参考）:")
        
        # 列出所有可能的组件名称
        matching_modules = set()
        for name, p in model.named_parameters():
            parts = name.split(".")
            if len(parts) >= 2:
                # 提取不同长度的前缀
                matching_modules.add(parts[0])
                matching_modules.add(".".join(parts[:2]))
                if len(parts) >= 3:
                    matching_modules.add(".".join(parts[:3]))
        
        if matching_modules:
            for module in sorted(matching_modules)[:20]:  # 最多显示20个
                print(f"     - \"{module}\"")
        
        raise RuntimeError("未找到可用于 TTT 的参数，请检查模型结构或筛选逻辑。")

    print(f"[DEBUG] 找到 {len(params)} 个参数用于 TTT 训练\n")
    return params


def list_image_paths(input_folder: Path, image_sorted: bool) -> List[Path]:
    if not input_folder.exists():
        raise FileNotFoundError(f"输入图像文件夹不存在: {input_folder}")

    exts = {".png", ".jpg", ".jpeg"}
    paths = [p for p in input_folder.iterdir() if p.suffix.lower() in exts]
    if image_sorted:
        paths = sorted(paths)
    return paths


def split_image_paths_evenly(image_paths: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    将图像路径均匀划分为两组（交替分配），保证两组数量尽可能接近。
    """
    set_a = image_paths[::2]
    set_b = image_paths[1::2]
    return set_a, set_b


def load_image_tensors(image_paths: List[Path]) -> List[torch.Tensor]:
    images = [process_image(str(p)) for p in image_paths]
    if len(images) == 0:
        raise ValueError("未加载到任何图像，请检查输入目录。")
    return images


def group_images(
    images: List[torch.Tensor],
    group_size: int,
    mode: str = "sequential",
    stride: Optional[int] = None,
) -> List[List[torch.Tensor]]:
    if group_size < 2:
        raise ValueError("group_size 必须 >= 2")

    if not images:
        raise ValueError("输入图像为空，无法构建分组。")

    mode = str(mode or "sequential").strip().lower()

    if mode == "sequential":
        groups = [images[i : i + group_size] for i in range(0, len(images), group_size)]
        if len(groups[-1]) < 2:
            if len(groups) == 1:
                raise ValueError("至少需要 2 张图像才能进行 TTT 训练。")
            groups[-2].extend(groups[-1])
            groups.pop()
        return groups

    if mode in {"sliding", "slide", "overlap"}:
        if len(images) < group_size:
            raise ValueError("图像数量不足，无法构建滑动窗口分组。")
        step = int(stride) if stride not in (None, 0) else max(1, group_size // 2)
        if step < 1:
            raise ValueError("group_stride 必须 >= 1")
        groups = [
            images[i : i + group_size]
            for i in range(0, len(images) - group_size + 1, step)
        ]
        if not groups:
            raise ValueError("未生成任何滑动窗口分组，请检查 group_size/group_stride。")
        return groups

    raise ValueError("group_mode 必须为 'sequential' 或 'sliding'")


def render_views(
    decoder,
    gaussians,
    extrinsic: torch.Tensor,
    intrinsic: torch.Tensor,
    image_size: Tuple[int, int],
    near: float = 0.01,
    far: float = 100.0,
) -> torch.Tensor:
    num_views = extrinsic.shape[1]
    output = decoder.forward(
        gaussians,
        extrinsic,
        intrinsic.float(),
        torch.ones(1, num_views, device=extrinsic.device) * near,
        torch.ones(1, num_views, device=extrinsic.device) * far,
        image_size,
    )
    return output.color[0].clip(min=0, max=1)


def predict_context_and_target_poses(
    model: AnySplat,
    ctx_images_01: torch.Tensor,
    tgt_images_01: torch.Tensor,
    pred_context_pose: dict,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    b, v_ctx, _, h, w = ctx_images_01.shape
    _, v_tgt, _, _, _ = tgt_images_01.shape

    use_amp = ctx_images_01.is_cuda
    vggt_input_image = torch.cat((ctx_images_01, tgt_images_01), dim=1)
    vggt_input_image = vggt_input_image.to(torch.bfloat16 if use_amp else torch.float32)

    if use_amp:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=False, dtype=torch.bfloat16):
            aggregated_tokens_list, _ = model.encoder.aggregator(
                vggt_input_image,
                intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx,
            )
    else:
        with torch.no_grad():
            aggregated_tokens_list, _ = model.encoder.aggregator(
                vggt_input_image,
                intermediate_layer_idx=model.encoder.cfg.intermediate_layer_idx,
            )

    with torch.no_grad():
        fp32_tokens = [token.float() for token in aggregated_tokens_list]
        pred_all_pose_enc = model.encoder.camera_head(fp32_tokens)[-1]
        pred_all_extrinsic, pred_all_intrinsic = pose_encoding_to_extri_intri(
            pred_all_pose_enc,
            vggt_input_image.shape[-2:],
        )

    extrinsic_padding = torch.tensor(
        [0, 0, 0, 1],
        device=pred_all_extrinsic.device,
        dtype=pred_all_extrinsic.dtype,
    ).view(1, 1, 1, 4).repeat(b, vggt_input_image.shape[1], 1, 1)
    pred_all_extrinsic = torch.cat([pred_all_extrinsic, extrinsic_padding], dim=2).inverse()

    pred_all_intrinsic[:, :, 0] = pred_all_intrinsic[:, :, 0] / w
    pred_all_intrinsic[:, :, 1] = pred_all_intrinsic[:, :, 1] / h

    pred_all_context_extrinsic = pred_all_extrinsic[:, :v_ctx]
    pred_all_target_extrinsic = pred_all_extrinsic[:, v_ctx:]
    pred_all_context_intrinsic = pred_all_intrinsic[:, :v_ctx]
    pred_all_target_intrinsic = pred_all_intrinsic[:, v_ctx:]

    scale_factor = (
        pred_context_pose["extrinsic"][:, :, :3, 3].mean()
        / pred_all_context_extrinsic[:, :, :3, 3].mean()
    )
    pred_all_target_extrinsic[..., :3, 3] = pred_all_target_extrinsic[..., :3, 3] * scale_factor
    pred_all_context_extrinsic[..., :3, 3] = pred_all_context_extrinsic[..., :3, 3] * scale_factor

    return (
        pred_all_context_extrinsic,
        pred_all_context_intrinsic,
        pred_all_target_extrinsic,
        pred_all_target_intrinsic,
    )


def run_ttt(model: AnySplat, cfg: TTTConfig) -> None:
    """
    Test-Time Training 主循环：

    1. 输入图像均匀划分为集合一/二；
    2. 集合一按顺序每 8 张分组，每组随机选 1 张为 target，其余为 context；
    3. 用 context 重建高斯，并渲染 target/context 计算损失微调模型；
    4. 集合一全部使用后，用微调模型在集合一 + 集合二上重建最终场景。
    """
    device = cfg.device
    model = model.to(device)

    # 先将所有参数冻结，再打开 TTT 相关参数
    for p in model.parameters():
        p.requires_grad = False
    ttt_params = get_ttt_parameters(model, train_components=cfg.train_components)
    for p in ttt_params:
        p.requires_grad = True

    optimizer = torch.optim.Adam(ttt_params, lr=cfg.lr)
    ttt_params_snapshot = [p.detach().clone() for p in ttt_params]
    lpips_fn = None
    if cfg.loss_type.lower() in {"lpips", "l1_mse_lpips"}:
        lpips_fn = get_lpips(device)

    image_paths = list_image_paths(cfg.input_folder, cfg.image_sorted)
    if len(image_paths) < 2:
        raise ValueError("至少需要 2 张图像才能进行 TTT。")

    set1_paths, set2_paths = split_image_paths_evenly(image_paths)
    set1_images = load_image_tensors(set1_paths)
    set2_images = load_image_tensors(set2_paths) if set2_paths else []

    print(
        f"[ttt] 输入图像总数: {len(image_paths)} | "
        f"集合一: {len(set1_images)} | 集合二: {len(set2_images)}"
    )

    groups = group_images(
        set1_images,
        group_size=cfg.group_size,
        mode=cfg.group_mode,
        stride=cfg.group_stride,
    )
    if cfg.group_mode in {"sliding", "slide", "overlap"}:
        stride_info = cfg.group_stride if cfg.group_stride not in (None, 0) else max(1, cfg.group_size // 2)
        print(
            f"[ttt] 集合一使用滑动窗口分组: size={cfg.group_size}, stride={stride_info}, groups={len(groups)}"
        )
    else:
        print(f"[ttt] 集合一按 {cfg.group_size} 张分组，共 {len(groups)} 组")

    total_groups = len(groups)
    for it in range(cfg.iters):
        print(f"\n[ttt] ===== Epoch {it + 1}/{cfg.iters} =====")
        rng = random.Random(cfg.seed + it)

        if it < total_groups:
            selected_groups = [(it, groups[it])]
        else:
            random_idx = rng.randrange(total_groups)
            selected_groups = [(random_idx, groups[random_idx])]

        for group_idx, group in selected_groups:
            target_idx = rng.randrange(len(group))
            target_image = group[target_idx]
            context_images = [img for i, img in enumerate(group) if i != target_idx]

            ctx_tensor = torch.stack(context_images, dim=0).unsqueeze(0).to(device)
            tgt_tensor = target_image.unsqueeze(0).unsqueeze(0).to(device)

            _, _, _, h, w = ctx_tensor.shape

            ctx_01 = (ctx_tensor + 1.0) * 0.5
            tgt_01 = (tgt_tensor + 1.0) * 0.5

            model.train()
            optimizer.zero_grad(set_to_none=True)

            encoder_output = model.encoder(ctx_01, global_step=0, visualization_dump={})
            gaussians = encoder_output.gaussians
            pred_context_pose = encoder_output.pred_context_pose

            (
                pred_ctx_extrinsic,
                pred_ctx_intrinsic,
                pred_tgt_extrinsic,
                pred_tgt_intrinsic,
            ) = predict_context_and_target_poses(
                model=model,
                ctx_images_01=ctx_01,
                tgt_images_01=tgt_01,
                pred_context_pose=pred_context_pose,
            )

            y_tgt = render_views(
                model.decoder,
                gaussians,
                pred_tgt_extrinsic,
                pred_tgt_intrinsic,
                (h, w),
            )
            y_ctx = render_views(
                model.decoder,
                gaussians,
                pred_ctx_extrinsic,
                pred_ctx_intrinsic,
                (h, w),
            )

            render_dir = cfg.output_folder / "train_renders" / f"epoch_{it + 1:04d}" / f"group_{group_idx + 1:04d}"
            render_dir.mkdir(parents=True, exist_ok=True)
            for view_idx, img in enumerate(y_ctx):
                save_image(img, render_dir / f"ctx_{view_idx:03d}.jpg")
            for view_idx, img in enumerate(y_tgt):
                save_image(img, render_dir / f"tgt_{view_idx:03d}.jpg")

            target_gt = tgt_01[0]
            ctx_gt = ctx_01[0]

            loss_l1 = F.l1_loss(y_tgt, target_gt)
            loss_mse = F.mse_loss(y_tgt, target_gt)
            loss_ctx_l1 = F.l1_loss(y_ctx, ctx_gt)
            loss_ctx_mse = F.mse_loss(y_ctx, ctx_gt)

            loss = (
                cfg.l1_weight * (loss_l1 + cfg.context_loss_weight * loss_ctx_l1)
                + cfg.mse_weight * (loss_mse + cfg.context_loss_weight * loss_ctx_mse)
            )

            loss_lpips = None
            loss_ctx_lpips = None
            if lpips_fn is not None:
                loss_lpips = lpips_fn(y_tgt, target_gt, normalize=True).mean()
                loss_ctx_lpips = lpips_fn(y_ctx, ctx_gt, normalize=True).mean()
                loss = loss + cfg.lpips_weight * (
                    loss_lpips + cfg.context_loss_weight * loss_ctx_lpips
                )

            loss.backward()
            if group_idx == 0:
                has_grad = any(
                    (p.grad is not None) and torch.isfinite(p.grad).all()
                    for p in ttt_params
                )
                print(f"[ttt] epoch {it + 1} has_grad={has_grad}")
            optimizer.step()
            max_delta = max(
                (p.detach() - b).abs().max().item()
                for p, b in zip(ttt_params, ttt_params_snapshot)
            )
            print(f"[ttt] epoch {it + 1} max_param_delta={max_delta:.6e}")
            prefix = f"[ttt] epoch {it + 1} group {group_idx + 1}/{len(groups)}"
            if loss_lpips is None:
                print(
                    f"{prefix}: loss={loss.item():.6f} "
                    f"(L1={loss_l1.item():.6f}, MSE={loss_mse.item():.6f}, "
                    f"CTX_L1={loss_ctx_l1.item():.6f}, CTX_MSE={loss_ctx_mse.item():.6f})"
                )
            else:
                print(
                    f"{prefix}: loss={loss.item():.6f} "
                    f"(L1={loss_l1.item():.6f}, MSE={loss_mse.item():.6f}, LPIPS={loss_lpips.item():.6f}, "
                    f"CTX_L1={loss_ctx_l1.item():.6f}, CTX_MSE={loss_ctx_mse.item():.6f}, "
                    f"CTX_LPIPS={loss_ctx_lpips.item():.6f})"
                )

    # --------------------------------------------------------------
    # 最后一次使用更新后的模型，导出 PLY 和插值视频（可选）
    # --------------------------------------------------------------
    print("\n[ttt] TTT 完成，使用微调后的模型做最终场景重建...")
    model.eval()
    with torch.no_grad():
        all_images = load_image_tensors(image_paths)
        all_tensor = torch.stack(all_images, dim=0).unsqueeze(0).to(device)
        all_01 = (all_tensor + 1.0) * 0.5
        encoder_output = model.encoder(all_01, global_step=0, visualization_dump={})
        gaussians_final = encoder_output.gaussians

    if cfg.export_ply:
        ply_path = cfg.output_folder / "gaussians_ttt.ply"
        export_ply(
            gaussians_final.means[0],
            gaussians_final.scales[0],
            gaussians_final.rotations[0],
            gaussians_final.harmonics[0],
            gaussians_final.opacities[0],
            ply_path,
        )
        print(f"[ttt] 最终场景高斯已导出到: {ply_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AnySplat Test-Time Training (TTT)")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path("config/ttt.yaml")),
        help="TTT YAML 配置文件路径。",
    )
    return parser.parse_args()


def main() -> None:

    cfg = load_ttt_config(Path("config/ttt.yaml").expanduser().resolve())
    cfg.output_folder.mkdir(parents=True, exist_ok=True)
    print(cfg.train_components)

    device = torch.device(cfg.device)

    print(f"[ttt] 使用设备: {device}")
    print(f"[ttt] 输入图像目录: {cfg.input_folder}")
    print(f"[ttt] 输出目录: {cfg.output_folder}")

    print("[ttt] 加载 AnySplat 预训练模型...")
    local_model_path = cfg.pretrained_model_path
    if local_model_path:
        if not Path(local_model_path).is_absolute():
            local_model_path = Path.cwd() / local_model_path
    
    model = load_model_with_fallback(
        local_path=local_model_path,
        device=device,
    )

    run_ttt(model, replace(cfg, device=device))


if __name__ == "__main__":
    main()

