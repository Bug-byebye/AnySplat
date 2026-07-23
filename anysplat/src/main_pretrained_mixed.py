"""
在已完成预训练的 AnySplat 上做混合数据训练：
- 本仓库 GT 数据集（DatasetDL3DVGT，配置名 dl3dv_gt）：与 main_pretrained_gt 相同，使用深度与位姿真值监督。
- 其他常规数据集：encoder.distill=True，DistillLoss 走 VGGT teacher 伪标签（与主训练 src/main.py 一致）。

启动示例：
  python src/main_pretrained_mixed.py dataset.dl3dv.roots=[/path/to/bench] dataset.co3d.roots=[/path/to/co3d]
"""
import os
from pathlib import Path

import hydra
import torch
import wandb
from colorama import Fore
from jaxtyping import install_import_hook
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers.wandb import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict

import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model.model import get_model
from src.model.model.anysplat import AnySplat as HubAnySplat

import warnings

warnings.filterwarnings("ignore")

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg, get_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.model_wrapper import ModelWrapper
    from src.dataset.dataset_dl3dv_gt import DatasetDL3DVGT
    from src.loss.loss_distill import extri_intri_to_pose_encoding, huber_loss
    from src.utils.point import get_normal_map


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def initialize_from_local_pretrained(model: torch.nn.Module, model_path: str) -> torch.nn.Module:
    local_model_dir = Path(model_path).expanduser()
    if not local_model_dir.exists():
        raise FileNotFoundError(
            f"Local pretrained model directory does not exist: {local_model_dir}"
        )

    print(cyan(f"Initializing model from local pretrained AnySplat: {local_model_dir}"))
    pretrained_model = HubAnySplat.from_pretrained(str(local_model_dir))
    missing, unexpected = model.load_state_dict(pretrained_model.state_dict(), strict=False)
    print(
        cyan(
            "Pretrained weight load completed "
            f"(missing keys: {len(missing)}, unexpected keys: {len(unexpected)})."
        )
    )
    return model


def register_gt_dataset() -> None:
    from src.dataset import DATASETS

    DATASETS["dl3dv_gt"] = DatasetDL3DVGT


def register_mixed_prob_mapping(cfg_dict: DictConfig) -> None:
    """为 ConcatDataset 中每个数据源类注册 MixedBatchSampler 的采样权重。"""
    from src.dataset.data_module import prob_mapping
    from src.dataset.dataset_dl3dv import DatasetDL3DV
    from src.dataset.dataset_co3d import DatasetCo3d
    from src.dataset.dataset_scannetpp import DatasetScannetpp

    def resolve_class(key: str) -> type:
        entry = cfg_dict.dataset[key]
        name = entry.get("name", "")
        if key == "dl3dv":
            return DatasetDL3DVGT if name == "dl3dv_gt" else DatasetDL3DV
        if key == "co3d":
            return DatasetCo3d
        if key == "scannetpp":
            return DatasetScannetpp
        raise ValueError(
            f"Unknown dataset key '{key}'. Extend resolve_class() for new dataset types."
        )

    keys = list(cfg_dict.dataset.keys())
    mixed_training = cfg_dict.get("mixed_training")
    weights = (mixed_training or {}).get("sampling_weights")

    for k in keys:
        cls = resolve_class(k)
        if weights is None:
            prob_mapping[cls] = prob_mapping.get(cls, 1.0)
        else:
            if k not in weights:
                raise ValueError(
                    f"mixed_training.sampling_weights must define an entry for each dataset key: missing '{k}'"
                )
    if weights is None:
        print(
            cyan(
                f"Mixed sampling: default weight 1.0 per dataset class (PyTorch multinomial normalizes); "
                f"order={keys}"
            )
        )
        return

    raw = [float(weights[k]) for k in keys]
    s = sum(raw)
    if s <= 0:
        raise ValueError("mixed_training.sampling_weights must sum to a positive value")
    for k, w in zip(keys, raw):
        prob_mapping[resolve_class(k)] = w / s
    print(
        cyan(
            "Mixed sampling weights (normalized to sum=1): "
            + ", ".join(f"{k}={w/s:.4f}" for k, w in zip(keys, raw))
        )
    )


def prepare_mixed_finetune_cfg(cfg_dict: DictConfig) -> None:
    """保持 VGGT teacher（与 main.py 一致），同时允许 GT 分支；移除与 GT 微调冲突的 depth_consis。"""
    with open_dict(cfg_dict):
        if "model" in cfg_dict and "encoder" in cfg_dict.model:
            cfg_dict.model.encoder.distill = True

        if "loss" in cfg_dict and "depth_consis" in cfg_dict.loss:
            del cfg_dict.loss["depth_consis"]
            print(cyan("Removed depth_consis for mixed GT + distill training."))

        if "train" in cfg_dict:
            current_depth_weight = float(cfg_dict.train.get("weight_depth", 1.0))
            if current_depth_weight < 1.0:
                cfg_dict.train.weight_depth = 1.0
                print(
                    cyan(
                        "Raised train.weight_depth from "
                        f"{current_depth_weight:g} to 1.0 for the GT supervision branch."
                    )
                )


def _batch_uses_gt_supervision(batch: dict) -> bool:
    """DatasetDL3DVGT 在 collate 后提供 use_gt_supervision；其它数据集无此字段。"""
    flag = batch.get("use_gt_supervision")
    if flag is None:
        return False
    if isinstance(flag, torch.Tensor):
        return bool(flag.all().item())
    return bool(flag)


def _context_depth_and_mask(
    pred_depth: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    gt_depth = batch["context"]["depth"].to(device)
    if gt_depth.ndim == 5 and gt_depth.shape[2] == 1:
        gt_depth = gt_depth.squeeze(2)

    mask = torch.isfinite(gt_depth) & (gt_depth > 0) & torch.isfinite(pred_depth)
    if "valid_mask" in batch["context"]:
        valid_mask = batch["context"]["valid_mask"].to(device)
        if valid_mask.ndim == 5 and valid_mask.shape[2] == 1:
            valid_mask = valid_mask.squeeze(2)
        mask = mask & (valid_mask > 0)
    return pred_depth, gt_depth, mask


def _flatten_context_depth(
    pred_depth: torch.Tensor,
    batch: dict,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_depth, gt_depth, mask = _context_depth_and_mask(pred_depth, batch, device)
    return pred_depth.flatten(0, 1), gt_depth.flatten(0, 1), mask.flatten(0, 1)


def _align_and_clip_depth_per_view(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
    low_quantile: float = 0.01,
    high_quantile: float = 0.99,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred_safe = pred_depth.clamp(min=1e-6)
    pred_aligned = pred_safe.clone()
    gt_clamped = gt_depth.clone()
    robust_mask = valid_mask.clone()

    for i in range(pred_depth.shape[0]):
        cur_mask = robust_mask[i]
        if not cur_mask.any():
            continue

        cur_gt = gt_depth[i][cur_mask]
        if cur_gt.numel() >= 16:
            gt_lo = torch.quantile(cur_gt, low_quantile).detach()
            gt_hi = torch.quantile(cur_gt, high_quantile).detach()
            cur_mask = cur_mask & (gt_depth[i] >= gt_lo) & (gt_depth[i] <= gt_hi)
            robust_mask[i] = cur_mask
            gt_clamped[i] = gt_depth[i].clamp(min=gt_lo, max=gt_hi)

        if not cur_mask.any():
            continue

        cur_pred = pred_safe[i][cur_mask]
        cur_gt = gt_clamped[i][cur_mask]
        pred_med = torch.median(cur_pred).clamp(min=1e-6)
        gt_med = torch.median(cur_gt).clamp(min=1e-6)
        pred_aligned[i] = pred_safe[i] * (gt_med / pred_med).detach()

    return pred_aligned, gt_clamped.clamp(min=1e-6), robust_mask


def _depth_absrel(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(min=1e-6)
    gt = gt.clamp(min=1e-6)
    return ((pred[mask] - gt[mask]).abs() / gt[mask]).mean()


def _depth_delta1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = pred.clamp(min=1e-6)
    gt = gt.clamp(min=1e-6)
    ratio = torch.maximum(pred[mask] / gt[mask], gt[mask] / pred[mask])
    return (ratio < 1.25).float().mean()


def patch_distill_loss_mixed() -> None:
    from src.loss import loss_distill as loss_distill_module

    if getattr(loss_distill_module.DistillLoss, "_mixed_dispatch_installed", False):
        return

    _teacher_forward = loss_distill_module.DistillLoss.forward

    def forward_with_gt(self, distill_infos, pred_pose_enc_list, prediction, batch):
        loss_pose = torch.zeros((), device=prediction.depth.device)
        if pred_pose_enc_list is not None:
            gt_pose_enc = extri_intri_to_pose_encoding(
                batch["context"]["extrinsics"][:, :, :3, :],
                batch["context"]["intrinsics"],
            ).to(prediction.depth.device)
            num_predictions = len(pred_pose_enc_list)
            for i, cur_pred_pose_enc in enumerate(pred_pose_enc_list):
                i_weight = self.gamma ** (num_predictions - i - 1)
                loss_pose = loss_pose + i_weight * huber_loss(
                    cur_pred_pose_enc, gt_pose_enc
                ).mean()
            loss_pose = loss_pose / max(num_predictions, 1)

        pred_depth, gt_depth, valid_mask = _flatten_context_depth(
            prediction.depth, batch, prediction.depth.device
        )
        if valid_mask.any():
            pred_depth = pred_depth.clamp(min=1e-6)
            pred_depth_aligned, gt_depth_clamped, robust_mask = (
                _align_and_clip_depth_per_view(pred_depth, gt_depth, valid_mask)
            )

            if robust_mask.any():
                raw_rel_residual = (
                    pred_depth[robust_mask] - gt_depth_clamped[robust_mask]
                ) / gt_depth_clamped[robust_mask]
                raw_log_residual = (
                    pred_depth[robust_mask].log()
                    - gt_depth_clamped[robust_mask].log()
                )
                aligned_rel_residual = (
                    pred_depth_aligned[robust_mask] - gt_depth_clamped[robust_mask]
                ) / gt_depth_clamped[robust_mask]

                zero_raw = torch.zeros_like(raw_rel_residual)
                zero_log = torch.zeros_like(raw_log_residual)
                zero_aligned = torch.zeros_like(aligned_rel_residual)

                loss_depth_raw = 0.5 * torch.nn.functional.huber_loss(
                    raw_rel_residual, zero_raw, delta=0.25, reduction="mean"
                ) + 0.5 * torch.nn.functional.huber_loss(
                    raw_log_residual, zero_log, delta=0.20, reduction="mean"
                )
                loss_depth_aligned = torch.nn.functional.huber_loss(
                    aligned_rel_residual, zero_aligned, delta=0.15, reduction="mean"
                )
                loss_depth = 0.75 * loss_depth_raw + 0.25 * loss_depth_aligned
            else:
                loss_depth = torch.zeros((), device=prediction.depth.device)

            if self.weight_normal > 0 and robust_mask.any():
                render_normal = get_normal_map(
                    pred_depth_aligned, batch["context"]["intrinsics"].flatten(0, 1)
                )
                gt_normal = get_normal_map(
                    gt_depth_clamped, batch["context"]["intrinsics"].flatten(0, 1)
                )
                alpha1_loss = (
                    1 - (render_normal[robust_mask] * gt_normal[robust_mask]).sum(-1)
                ).mean()
                alpha2_loss = torch.nn.functional.l1_loss(
                    render_normal[robust_mask],
                    gt_normal[robust_mask],
                    reduction="mean",
                )
                loss_normal = (alpha1_loss + alpha2_loss) / 2
            else:
                loss_normal = torch.zeros((), device=prediction.depth.device)
        else:
            loss_depth = torch.zeros((), device=prediction.depth.device)
            loss_normal = torch.zeros((), device=prediction.depth.device)

        loss_distill = (
            loss_pose * self.weight_pose
            + loss_depth * self.weight_depth
            + loss_normal * self.weight_normal
        )
        loss_distill = torch.nan_to_num(loss_distill, nan=0.0, posinf=0.0, neginf=0.0)
        return {
            "loss_distill": loss_distill,
            "loss_pose": loss_pose * self.weight_pose,
            "loss_depth": loss_depth * self.weight_depth,
            "loss_normal": loss_normal * self.weight_normal,
        }

    def forward_mixed(self, distill_infos, pred_pose_enc_list, prediction, batch):
        if _batch_uses_gt_supervision(batch):
            return forward_with_gt(self, distill_infos, pred_pose_enc_list, prediction, batch)
        return _teacher_forward(self, distill_infos, pred_pose_enc_list, prediction, batch)

    loss_distill_module.DistillLoss.forward = forward_mixed
    loss_distill_module.DistillLoss._mixed_dispatch_installed = True


def patch_gt_batch_sanity_log() -> None:
    from src.model import model_wrapper as model_wrapper_module

    original_training_step = model_wrapper_module.ModelWrapper.training_step
    relaxed_skip_threshold = 10.0
    huge_loss_threshold = 1000.0

    def training_step_with_sanity(self, batch, batch_idx):
        depth = batch["context"]["depth"]
        valid = depth > 0
        valid_ratio = valid.float().mean().item()
        valid_count = int(valid.sum().item())
        total_count = int(valid.numel())
        if valid.any():
            valid_depth = depth[valid]
            d_min = valid_depth.min().item()
            d_max = valid_depth.max().item()
        else:
            d_min, d_max = 0.0, 0.0
        t = batch["context"]["extrinsics"][..., :3, 3]
        t_min = t.min().item()
        t_max = t.max().item()
        gt_flag = _batch_uses_gt_supervision(batch)

        if not getattr(self, "_mixed_sanity_logged", False):
            print(
                cyan(
                    "[Mixed sanity] "
                    f"use_gt_supervision={gt_flag}, "
                    f"valid_depth_ratio={valid_ratio:.4f}, "
                    f"depth_min={d_min:.4f}, depth_max={d_max:.4f}, "
                    f"pose_t_min={t_min:.4f}, pose_t_max={t_max:.4f}"
                )
            )
            self._mixed_sanity_logged = True

        out = original_training_step(self, batch, batch_idx)

        diag_file = Path(self.train_cfg.output_path) / "skipped_batches_diagnostics.txt"
        scene_repr = batch.get("scene", "unknown")
        if isinstance(scene_repr, list):
            scene_repr = ",".join([str(x) for x in scene_repr])

        def _write_diag(reason: str, estimated_total_loss: float):
            with diag_file.open("a") as f:
                f.write(
                    f"reason={reason}, step={self.global_step}, batch_idx={batch_idx}, "
                    f"use_gt_supervision={gt_flag}, "
                    f"estimated_total_loss={estimated_total_loss:.6f}, "
                    f"relaxed_threshold={relaxed_skip_threshold:.3f}, huge_threshold={huge_loss_threshold:.3f}, "
                    f"scene={scene_repr}, "
                    f"valid_ratio={valid_ratio:.6f}, valid_count={valid_count}, total_count={total_count}, "
                    f"depth_min={d_min:.6f}, depth_max={d_max:.6f}, "
                    f"pose_t_min={t_min:.6f}, pose_t_max={t_max:.6f}\n"
                )

        if isinstance(out, torch.Tensor):
            out_val = float(out.detach().item())
            if out_val > huge_loss_threshold:
                print(
                    cyan(
                        f"[Mixed huge-loss skip] step={self.global_step}, "
                        f"loss={out_val:.6f} > {huge_loss_threshold:.3f}"
                    )
                )
                _write_diag("huge_loss_any_step", out_val)
                return out * 1e-10

            if self.global_step > 1000 and out_val < 1e-6:
                estimated_total_loss = out_val * 1e10
                if estimated_total_loss <= relaxed_skip_threshold:
                    print(
                        cyan(
                            f"[Mixed skip relax] restore batch at step {self.global_step} "
                            f"(estimated_loss={estimated_total_loss:.6f}, "
                            f"new_threshold={relaxed_skip_threshold:.3f})"
                        )
                    )
                    return out * 1e10

                _write_diag("post1000_skip", estimated_total_loss)
        return out

    model_wrapper_module.ModelWrapper.training_step = training_step_with_sanity


def patch_optimizer_to_avoid_backbone_lr_raise() -> None:
    from src.model import model_wrapper as mw

    def configure_optimizers_fixed(self):
        new_params, pretrained_params = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "gaussian_param_head" in name or "interm" in name:
                new_params.append(param)
            else:
                pretrained_params.append(param)

        param_dicts = [
            {"params": new_params, "lr": self.optimizer_cfg.lr},
            {
                "params": pretrained_params,
                "lr": self.optimizer_cfg.lr * self.optimizer_cfg.backbone_lr_multiplier,
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.optimizer_cfg.lr, weight_decay=0.05, betas=(0.9, 0.95)
        )

        warm_up_steps = max(int(self.optimizer_cfg.warm_up_steps), 1)
        warm_up = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / warm_up_steps,
            end_factor=1.0,
            total_iters=warm_up_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=get_cfg()["trainer"]["max_steps"],
            eta_min=0.0,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warm_up, cosine],
            milestones=[warm_up_steps],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    mw.ModelWrapper.configure_optimizers = configure_optimizers_fixed


def patch_metrics_to_use_gt_only() -> None:
    """验证集仅使用 cfg.dataset 中第一个数据源；若需 GT 指标，请将带真值的 dl3dv_gt 放在第一位。"""
    from src.model import model_wrapper as mw
    import torch.nn.functional as F

    def validation_step_gt(self, batch, batch_idx, dataloader_idx=0):
        batch = self.data_shim(batch)
        b, v, _, _, _ = batch["context"]["image"].shape
        assert b == 1

        context_image = (batch["context"]["image"] + 1) / 2
        encoder_output, output = self.model(context_image, self.global_step, visualization_dump=None)

        pred_depth, gt_depth, mask = _context_depth_and_mask(
            output.depth, batch, output.depth.device
        )

        if mask.any():
            self.log("val/gt_depth_absrel", _depth_absrel(pred_depth, gt_depth, mask))
            self.log("val/gt_depth_delta1", _depth_delta1(pred_depth, gt_depth, mask))
            self.log(
                "val/gt_depth_mse",
                F.mse_loss(pred_depth.clamp(min=1e-6)[mask], gt_depth[mask]).mean(),
            )

            flat_pred, flat_gt, flat_mask = (
                pred_depth.flatten(0, 1),
                gt_depth.flatten(0, 1),
                mask.flatten(0, 1),
            )
            aligned_pred, clipped_gt, aligned_mask = _align_and_clip_depth_per_view(
                flat_pred, flat_gt, flat_mask
            )
            if aligned_mask.any():
                self.log(
                    "val/gt_depth_absrel_aligned",
                    _depth_absrel(aligned_pred, clipped_gt, aligned_mask),
                )
                self.log(
                    "val/gt_depth_delta1_aligned",
                    _depth_delta1(aligned_pred, clipped_gt, aligned_mask),
                )
                self.log(
                    "val/gt_depth_mse_aligned",
                    F.mse_loss(aligned_pred[aligned_mask], clipped_gt[aligned_mask]).mean(),
                )
            else:
                z = torch.zeros((), device=output.depth.device)
                self.log("val/gt_depth_absrel_aligned", z)
                self.log("val/gt_depth_delta1_aligned", z)
                self.log("val/gt_depth_mse_aligned", z)
        else:
            z = torch.zeros((), device=output.depth.device)
            self.log("val/gt_depth_absrel", z)
            self.log("val/gt_depth_delta1", z)
            self.log("val/gt_depth_mse", z)
            self.log("val/gt_depth_absrel_aligned", z)
            self.log("val/gt_depth_delta1_aligned", z)
            self.log("val/gt_depth_mse_aligned", z)

        if encoder_output.pred_pose_enc_list is not None and len(encoder_output.pred_pose_enc_list) > 0:
            gt_pose_enc = extri_intri_to_pose_encoding(
                batch["context"]["extrinsics"][:, :, :3, :],
                batch["context"]["intrinsics"],
            ).to(output.depth.device)
            pred_pose_enc = encoder_output.pred_pose_enc_list[-1].to(output.depth.device)
            self.log("val/gt_pose_t_l2", (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1).mean())
            self.log("val/gt_pose_q_l1", (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs().mean())
            self.log("val/gt_pose_fov_l1", (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs().mean())

        from src.evaluation.metrics import compute_psnr, compute_ssim, compute_lpips

        rgb_gt = (batch["context"]["image"][0].float() + 1) / 2
        rgb_pred = output.color[0].float()
        self.log("val/psnr", compute_psnr(rgb_gt, rgb_pred).mean())
        self.log("val/ssim", compute_ssim(rgb_gt, rgb_pred).mean())
        self.log("val/lpips", compute_lpips(rgb_gt, rgb_pred).mean())

        return None

    mw.ModelWrapper.validation_step = validation_step_gt


def patch_encoder_distill_info_fallback() -> None:
    from src.model.encoder.anysplat import EncoderAnySplat

    if getattr(EncoderAnySplat, "_gt_distill_info_fallback_patched", False):
        return

    original_forward = EncoderAnySplat.forward

    def forward_with_fallback_distill_infos(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)
        distill_infos = output.distill_infos
        if not self.distill and (
            distill_infos is None or "conf_mask" not in distill_infos
        ):
            distill_infos = {} if distill_infos is None else dict(distill_infos)
            depth = output.depth_dict.get("depth") if output.depth_dict is not None else None
            if depth is not None:
                if depth.ndim == 5 and depth.shape[-1] == 1:
                    depth = depth.squeeze(-1)
                distill_infos["conf_mask"] = torch.ones_like(depth, dtype=torch.bool)
                output.distill_infos = distill_infos
        return output

    EncoderAnySplat.forward = forward_with_fallback_distill_infos
    EncoderAnySplat._gt_distill_info_fallback_patched = True


def record_runtime_config(cfg_dict: DictConfig, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_cfg_path = output_dir / "runtime_config_raw.yaml"
    resolved_cfg_path = output_dir / "runtime_config_resolved.yaml"

    raw_cfg_text = OmegaConf.to_yaml(cfg_dict, resolve=False)
    raw_cfg_path.write_text(raw_cfg_text)

    cfg_for_resolve = OmegaConf.create(OmegaConf.to_container(cfg_dict, resolve=False))
    try:
        if "hydra" not in cfg_for_resolve:
            cfg_for_resolve["hydra"] = {}
        if "run" not in cfg_for_resolve["hydra"]:
            cfg_for_resolve["hydra"]["run"] = {}
        cfg_for_resolve["hydra"]["run"]["dir"] = str(output_dir)
        if "train" in cfg_for_resolve:
            cfg_for_resolve["train"]["output_path"] = str(output_dir)
        resolved_cfg_text = OmegaConf.to_yaml(cfg_for_resolve, resolve=True)
        resolved_cfg_path.write_text(resolved_cfg_text)
    except Exception as e:
        resolved_cfg_path.write_text(
            f"# Failed to resolve config due to: {type(e).__name__}: {e}\n\n"
            + OmegaConf.to_yaml(cfg_for_resolve, resolve=False)
        )

    dataset_name = "unknown"
    dataset_root = "unknown"
    if "dataset" in cfg_dict and "dl3dv" in cfg_dict.dataset:
        dataset_name = cfg_dict.dataset.dl3dv.get("name", dataset_name)
        roots = cfg_dict.dataset.dl3dv.get("roots", [])
        if roots:
            dataset_root = str(roots[0])

    print(
        cyan(
            "Runtime config snapshot saved: "
            f"raw={raw_cfg_path}, resolved={resolved_cfg_path}"
        )
    )
    print(
        cyan(
            "Runtime config key fields: "
            f"mode={cfg_dict.get('mode', 'unknown')}, "
            f"dataset_name={dataset_name}, "
            f"dataset_root={dataset_root}, "
            f"wandb_name={cfg_dict.wandb.get('name', 'unknown')}, "
            f"max_steps={cfg_dict.trainer.get('max_steps', 'unknown')}"
        )
    )


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="main_pretrained_mixed",
)
def train(cfg_dict: DictConfig):
    register_gt_dataset()
    if "dl3dv" in cfg_dict.dataset:
        cfg_dict.dataset.dl3dv.name = "dl3dv_gt"

    register_mixed_prob_mapping(cfg_dict)
    prepare_mixed_finetune_cfg(cfg_dict)
    patch_distill_loss_mixed()
    patch_gt_batch_sanity_log()
    patch_optimizer_to_avoid_backbone_lr_raise()
    patch_metrics_to_use_gt_only()
    patch_encoder_distill_info_fallback()

    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(cyan(f"Saving outputs to {output_dir}."))
    record_runtime_config(cfg_dict, output_dir)

    cfg.train.output_path = output_dir

    callbacks = []
    if cfg_dict.wandb.mode != "disabled":
        logger = WandbLogger(
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name} ({output_dir.parent.name}/{output_dir.name})",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=False,
            save_dir=output_dir,
            config=OmegaConf.to_container(cfg_dict),
        )
        callbacks.append(LearningRateMonitor("step", True))
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    callbacks.append(
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            save_weights_only=cfg.checkpointing.save_weights_only,
            save_last=True,
            monitor="info/global_step",
            mode="max",
        )
    )
    callbacks[-1].CHECKPOINT_EQUALS_CHAR = "_"

    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    step_tracker = StepTracker()

    trainer = Trainer(
        max_epochs=-1,
        num_nodes=cfg.trainer.num_nodes,
        accelerator="gpu",
        logger=logger,
        devices="auto",
        strategy=(
            "ddp_find_unused_parameters_true"
            if torch.cuda.device_count() > 1
            else "auto"
        ),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,
        enable_progress_bar=False,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        inference_mode=False if (cfg.mode == "test" and cfg.test.align_pose) else True,
    )
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)

    model = get_model(cfg.model.encoder, cfg.model.decoder)
    local_pretrained_path = cfg_dict.get(
        "pretrained_model_path",
        os.environ.get("ANYSPLAT_PRETRAINED_PATH", "pretrained_model"),
    )
    model = initialize_from_local_pretrained(model, local_pretrained_path)

    model_wrapper = ModelWrapper(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        model,
        get_losses(cfg.loss),
        step_tracker,
    )
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )

    if cfg.mode == "train":
        trainer.fit(model_wrapper, datamodule=data_module, ckpt_path=checkpoint_path)
    else:
        trainer.test(
            model_wrapper,
            datamodule=data_module,
            ckpt_path=checkpoint_path,
        )


if __name__ == "__main__":
    train()
