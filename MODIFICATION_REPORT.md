# Modification Report — Gaussian Repair Network

> **Date**: 2026-07-10
> **Author**: Claude Code
> **Purpose**: Document the design and implementation of the GaussianRepairNetwork, which extends the original Refiner with add/delete/move capabilities for scene completion.

---

## 1. Problem Statement

The original Refiner design predicted only parameter residuals (Δ_params, 82-dim) on existing Gaussians. This is fundamentally insufficient for scene repair because:

| Limitation | Consequence |
|------------|-------------|
| **No deletion** | Artifact-causing floaters cannot be removed; rendering loss can only adjust their parameters, which may not eliminate the artifact |
| **No position adjustment** | Gaussians with incorrect 3D positions cannot be moved to the right location; means remain fixed at the initial (possibly erroneous) AnySplat output |
| **No addition** | Holes and occluded regions cannot be filled with new Gaussians; the only way to cover a hole is to expand existing Gaussians' scales, which causes blurring |

These limitations are inherent to the Δ-only design: it keeps the number of Gaussians (N) fixed and their positions unchanged.

---

## 2. Design Overview

### 2.1 The Repairer Philosophy

Instead of a Refiner that adjusts, we design a **GaussianRepairNetwork** that **fixes**. The repair network has three independent but complementary mechanisms:

```
                    ┌─────────────────────────────┐
                    │   GaussianRepairNetwork      │
                    ├─────────────────────────────┤
                    │ 1. Per-Gaussian Analysis     │ ← Δ_params [82]  (parameter refine)
                    │    ├── Δ_params head         │ ← Δ_means  [3]   (position correct)
                    │    ├── Δ_means head          │ ← keep_prob [1]  (pruning)
                    │    └── keep_prob head        │
                    │                              │
                    │ 2. Scene Deficiency Analyzer │ ← deficiency map [H×W]
                    │    ├── hole detection        │ ← new Gaussians [K]
                    │    └── new Gaussian gen.      │
                    └─────────────────────────────┘
```

### 2.2 File Changes Summary

| File | Action | Purpose |
|------|--------|---------|
| `post/gaussian_restorer/repairer.py` | **NEW** | Main GaussianRepairNetwork module |
| `post/gaussian_restorer/repair_loss.py` | **NEW** | Comprehensive loss with sub-losses for add/delete/move |
| `post/gaussian_restorer/config.py` | MODIFIED | Added `RepairCfg` with all hyperparameters |
| `phase1_train_repairer.py` | **NEW** | Training script with full repair pipeline |
| `LITERATURE_SURVEY.md` | **NEW** | Literature survey of 7 papers |

---

## 3. Detailed Architecture

### 3.1 PerGaussianAnalysisHead (`repairer.py:33`)

This module processes each Gaussian independently with a shared MLP encoder and three output heads.

```
Input: [B, N, 134] = Gaussian params(83) + means(3) + video latent(48)
  │
  ├── MLP Encoder (2 × Linear+LayerNorm+ReLU, 256 hidden)
  │
  ├── param_head: Linear(256 → 82)  → Δ_params [B, N, 82]
  │     - Zero-initialized → first forward is identity
  │     - Applies to: scales(3), rotations(4), SH(75)
  │
  ├── move_head: Linear(256 → 32 → ReLU → 3) → Δ_means [B, N, 3]
  │     - Zero-initialized
  │     - Scaled by ×0.01 to keep movements tiny initially
  │     - Applies to: Gaussian means (x, y, z)
  │
  └── confidence_head: Linear(256 → 1) → keep_logit [B, N, 1]
        - Bias initialized to 0.5 → initial keep_prob ≈ 0.62 (slightly delete-biased)
        - keep_prob = σ(keep_logit)
        - Used as multiplier on opacity: effective_opacity = raw_opacity × keep_prob
```

**Design rationale:**

- **Zero-init final layers** for Δ_params and Δ_means: ensures first forward pass doesn't degrade the initial Gaussians, enabling safe training start.
- **×0.01 scale on Δ_means**: prevents Gaussians from "jumping" to unreasonable positions during initial training. The model must learn to accumulate small corrections.
- **keep_prob with bias 0.5**: initial retention probability ≈ 0.62 provides a gentle bias toward deletion, counteracting the rendering loss's natural tendency to keep all primitives.

### 3.2 SceneDeficiencyAnalyzer (`repairer.py:106`)

Detects where the current Gaussian field is deficient and predicts new Gaussian parameters.

```
Input: [B*V, 57, H, W]
  ├── video_latent(48)    — semantic prior from Wan2.2
  ├── rendered_img(3)     — what initial Gaussians render
  ├── rendered_opacity(1) — coverage (low = hole)
  ├── context_gt(3)       — what should be there
  ├── error_map(1)        — |rendered - GT|
  └── rendered_depth(1)   — for 2D→3D lifting
  │
  ├── ResidualUNet (4-block, 128 hidden)
  │
  ├── deficiency_head → deficiency_map [B*V, 1, H, W]
  │     - Sigmoid output: probability that each pixel needs new Gaussians
  │
  └── new_gaussian_head → new_gauss_params [B*V, 84, H, W]
        - Layout: depth_offset(1) + scale(3) + rot(4) + SH(75) + opacity_logit(1)
        - Zero-initialized
```

### 3.3 NewGaussianLifting (`repairer.py:147`)

A stateless function (not nn.Module) that lifts 2D deficiency predictions to 3D Gaussians.

**Algorithm:**
1. Aggregate deficiency maps across views (per-pixel max)
2. Top-K sampling: select K pixels with highest deficiency scores (K=512 default)
3. Filter by threshold (default 0.2)
4. For each sampled pixel:
   - Get depth from rendered depth + predicted depth_offset
   - Compute 3D position: **X = R·K⁻¹·[u·d, v·d, d] + t** (back-projection using camera extrinsics)
   - Extract Gaussian parameters (scale, rot, SH, opacity) from predicted map
   - Apply transformations: softplus for scale, normalize for rotation, sigmoid for opacity

**Why this is correct for training:** The deficiency analyzer's parameters receive gradients through:
1. The sampled pixel positions (top-K is non-differentiable, but we use a straight-through approximation via the deficiency BCE loss)
2. The predicted Gaussian parameters (scale, rot, SH, opacity) which are used in rendering
3. The depth_offset which affects the back-projected 3D position

### 3.4 RepairLoss (`repair_loss.py`)

Comprehensive multi-component loss:

```
L_total = L_render + λ_params·L_params + λ_move·L_move + λ_del·L_delete + λ_add·L_add

L_render  = MSE + λ_lpips·LPIPS          ← rendering quality (against target view GT)
L_params  = |Δ_params|²                   ← parameter residual regularization (λ=1e-6)
L_move    = |Δ_means|²                    ← position offset regularization (λ=3e-4, stronger)
L_delete  = (1 - keep_prob).mean()        ← gentle deletion sparsity (λ=1e-3)
L_add     = BCE(deficiency_map, target)   ← deficiency prediction (λ=0.01)
```

The **deletion loss** deserves explanation: `(1 - keep_prob).mean()` creates gentle pressure toward deleting Gaussians. The rendering loss provides counter-pressure (deleting useful Gaussians increases rendering error). The net effect is that only Gaussians that hurt rendering quality get deleted — those that either contribute nothing (low opacity already) or cause artifacts.

---

## 4. Training Procedure

### 4.1 Per-Step Data Flow

```
Step t:
  1. Sample context indices (2 frames) + target index (gap 8-15)
  2. Load context images [1,2,3,224,448], target image [1,3,224,448]
  3. AnySplat forward (frozen) → initial Gaussians G₀ [1, N, ...]
     + cameras [1,2,4,4], [1,2,3,3]
  4. Wan2.2 VAE encode → video latents [1,2,48,224,448]
  5. Render G₀ at context views (no_grad) → ctx_color, ctx_depth, ctx_alpha
  6. GaussianRepairNetwork forward:
     a. Build per-Gaussian features (project video to each Gaussian)
     b. PerGaussianAnalysisHead → Δ_params, Δ_means, keep_logit
     c. SceneDeficiencyAnalyzer → deficiency_map, new_gauss_params
     d. NewGaussianLifting → new_means, new_scales, ...
  7. Apply refinement to G₀:
     means ← means + Δ_means × 0.01
     params ← params + Δ_params
     opacity ← opacity × σ(keep_logit)
  8. Build refined Gaussians G₁ [1, N', ...], (N' may differ from N due to deletion)
  9. Concatenate new Gaussians G_new [1, K, ...]
     → G_all = G₁ ∪ G_new [1, N'+K, ...]
  10. Render G_all at target view → rendered_color
  11. Loss: MSE vs target + LPIPS + regularizations
  12. Backward through repair network only
```

### 4.2 Why the Addition Mechanism Works Despite Non-Differentiable Sampling

The top-K sampling of deficiency pixels is non-differentiable. However, gradients flow through the addition mechanism via two paths:

1. **Direct gradient path**: The new Gaussian parameters (scale, rot, SH, opacity, depth_offset) are differentiable functions of the deficiency analyzer's weights. They contribute to the rendered image → loss → backward.

2. **Indirect supervision path**: The BCE loss between predicted deficiency map and ground-truth error map trains the deficiency analyzer to identify where new Gaussians are needed.

The top-K sampling acts as an "argmax" selection — we pick K pixel locations and use their predicted parameters. As the deficiency map improves (learns where holes are), the selected locations become more relevant.

---

## 5. Comparison of Refiner vs Repairer

| Capability | Original Refiner | New Repairer | Key Change |
|---|---|---|---|
| Parameter refine | ✅ Δ_params [82] | ✅ Δ_params [82] | Same |
| Position move | ❌ Fixed means | ✅ Δ_means [3] | **NEW** — added Linear(32→3) head |
| Deletion | ❌ Always keep all | ✅ keep_prob [1] | **NEW** — added confidence head |
| Addition | ❌ No new primitives | ✅ K new Gaussians | **NEW** — SceneDeficiencyAnalyzer |
| Hole detection | ❌ None | ✅ deficiency map | **NEW** — ResidualUNet + BCE loss |
| Loss functions | MSE + LPIPS | MSE + LPIPS + regs | **NEW** — move/delete/add sub-losses |

**Parameter count comparison:**
- Original Refiner (VoxelRefiner, phase1_train_v2.py): ~122K params
- New Repairer (GaussianRepairNetwork): ~200K params (per-Gaussian) + ~260K (deficiency analyzer) = ~460K params
- The increase is justified by the three additional capabilities.

---

## 6. Usage

### 6.1 Training

```bash
CUDA_VISIBLE_DEVICES=2 python3 phase1_train_repairer.py
```

The training script:
- Uses DL3DV scene data (same as v2)
- 5000 steps, AdamW, cosine LR schedule
- Saves checkpoints every 500 steps to `/data/sunchang/exp_train/phase1_v3_repairer/`
- Logs PSNR, Δ norms, keep probability, deficiency mean, and new Gaussian count

### 6.2 Expected Behavior

| Metric | Expected Range | Meaning |
|--------|---------------|---------|
| PSNR | 15-25 dB | Increasing over training |
| Δ norm | 1-10 | Model is modifying parameters |
| keep_prob | 0.6-0.95 | Some Gaussians are pruned; starts lower, stabilizes higher |
| Δ_means norm | 0.01-1.0 | Small position corrections |
| deficiency_mean | 0.1-0.5 | Decreasing as scene repairs |
| New Gaussians | 0-512 | Fewer over time as scene converges |

---

## 7. Limitations & Future Work

**Current limitations:**

1. **No cross-view consistency for new Gaussians**: The lifting step uses only the first context view. Future work should aggregate across views.
2. **Fixed K budget for new Gaussians**: 512 slots may be too many for simple scenes and too few for complex ones. An adaptive budget would be better.
3. **Simplified pose**: Uses AnySplat's predicted extrinsics (not GT from dataset) for video latent projection. This may cause misalignment.
4. **Phase 1 scope**: Single scene, limited number of Gaussians, grayscale SH tests.

**Planned improvements for Phase 2:**

1. **Multi-view deficiency aggregation**: Fuse deficiency maps from all context views before lifting.
2. **Learnable K selection**: Each new Gaussian candidate predicts its own "activation score" — the top-K by score are kept.
3. **End-to-end training of addition through rendering**: Once the deficiency BCE loss converges, enable the full gradient path through added Gaussians' rendering contribution (requires soft top-K).
4. **Depth quality prediction**: The depth_quality_head predicts which Gaussians' depth estimates are reliable, informing the back-projection step.
