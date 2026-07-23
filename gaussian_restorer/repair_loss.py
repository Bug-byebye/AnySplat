"""
Unified Repair Loss Functions.

L_total = L_render + λ_params·L_params + λ_move·L_move
          + λ_del·L_delete + λ_gen·L_gen
"""

from typing import Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RepairLoss(nn.Module):
    """
    Multi-component loss for the UnifiedRepairer.

    Components:
      render:  MSE + LPIPS
      params:  |Δ_params|²     — keep modifications conservative
      move:    |Δ_means|²      — strong regularizer prevents jitter
      delete:  (1-keep_prob)   — gentle sparsity pressure
      gen:     new_params reg  — keep new Gaussians conservative at start
    """

    def __init__(self,
                 weight_params: float = 1e-6,
                 weight_move: float = 3e-4,
                 weight_delete: float = 1e-3,
                 weight_gen: float = 1e-5,
                 weight_lpips: float = 0.05,
                 lpips_fn: Optional[nn.Module] = None,
                 ):
        super().__init__()
        self.weight_params = weight_params
        self.weight_move = weight_move
        self.weight_delete = weight_delete
        self.weight_gen = weight_gen
        self.weight_lpips = weight_lpips
        self.lpips_fn = lpips_fn

    def forward(
        self,
        rendered_color: torch.Tensor,     # [B, 3, H, W]
        target_image: torch.Tensor,        # [B, 3, H, W]
        # Repairer outputs
        delta_params: torch.Tensor,        # [B, N, 82]
        delta_means: torch.Tensor,         # [B, N, 3]
        keep_prob: torch.Tensor,           # [B, N, 1]
        new_params: torch.Tensor,          # [B, K, 82]
        new_activations: torch.Tensor,     # [B, K, 1]
    ) -> Dict[str, torch.Tensor]:
        """Compute loss and return dict of components."""
        loss_dict = {}

        # ===== 1. Rendering Loss =====
        loss_mse = F.mse_loss(rendered_color, target_image)
        loss_lpips = torch.tensor(0.0, device=rendered_color.device)
        if self.lpips_fn is not None:
            try:
                loss_lpips = self.lpips_fn(
                    rendered_color.unsqueeze(0) if rendered_color.dim() == 3 else rendered_color,
                    target_image.unsqueeze(0) if target_image.dim() == 3 else target_image,
                ).mean()
            except Exception:
                pass
        loss_dict['render_mse'] = loss_mse
        loss_dict['render_lpips'] = loss_lpips
        loss_dict['render'] = loss_mse + self.weight_lpips * loss_lpips

        # ===== 2. Parameter Regularization =====
        loss_dict['params'] = delta_params.pow(2).mean()

        # ===== 3. Movement Regularization =====
        loss_dict['move'] = delta_means.pow(2).mean()

        # ===== 4. Deletion Sparsity =====
        loss_dict['delete'] = (1 - keep_prob).mean()

        # ===== 5. Generator Regularization =====
        loss_dict['gen_params'] = new_params.pow(2).mean()
        # Encourage some activations (penalize all-zero)
        act_mean = new_activations.mean()
        loss_dict['gen_act'] = (1 - act_mean).clamp(min=0) * 0.1  # gentle push toward some activation
        loss_dict['gen'] = loss_dict['gen_params'] + loss_dict['gen_act']

        # ===== Total =====
        loss_dict['total'] = (
            loss_dict['render']
            + self.weight_params * loss_dict['params']
            + self.weight_move * loss_dict['move']
            + self.weight_delete * loss_dict['delete']
            + self.weight_gen * loss_dict['gen']
        )

        return loss_dict

    def log(self, loss_dict: Dict[str, torch.Tensor], psnr: float) -> str:
        """Format for console logging."""
        parts = [f"PSNR={psnr:.1f}"]
        for key in ['total', 'render_mse', 'params', 'move', 'delete', 'gen_params', 'gen_act']:
            if key in loss_dict:
                v = loss_dict[key].item()
                if abs(v) < 0.001:
                    parts.append(f"{key}={v:.6f}")
                elif v < 10:
                    parts.append(f"{key}={v:.4f}")
                else:
                    parts.append(f"{key}={v:.1f}")
        return " | ".join(parts)
