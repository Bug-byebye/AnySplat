# Gaussian Scene Restoration — 研究计划

> 本文档是科研计划，而非工程方案。每一步都包含假设（Hypothesis）、验证方法、以及失败时的替代路径。

---

## 1. 核心研究问题

### 1.1 我们要解决什么问题

**问题陈述**：前馈式 3D Gaussian 重建方法（如 AnySplat）在遮挡区域、稀疏视图、以及纹理模糊处会产生不完美的高斯场——包含漂浮物、空洞、和细节丢失。如何在**不修改重建流程**的前提下，利用大规模预训练视频模型的先验知识，对这些高斯场进行**一次前向的修复**？

### 1.2 为什么现有方法不够好

| 方法 | 缺陷 |
|------|------|
| **直接优化（per-scene 3DGS）** | 需要测试时优化数分钟，不是前馈式 |
| **ReSplat / Fuse-and-Refine** | 纯几何 refine，无语义先验，被遮挡区域无从推断 |
| **GaussFusion** | latent → 生成图像 → 重新编码 → loss，**信息流损失大**（LLM 领域叫"chain-of-thought 崩溃"） |
| **GSFixer** | 需要 diffusion 迭代采样（多步），不是一次前向 |

### 1.3 我们的假设（Hypothesis）

**核心假设**：Wan2.2 这类视频 foundation model 在其 VAE 的 latent space 中已经编码了丰富的场景层级先验（物体形状、材质、光照、遮挡关系）。如果可以**直接**将这个 latent 关联到 3D 高斯场的每个体元上，就可以用一个轻量级 MLP 学习从"latent + 原始高斯参数"到"残差"的映射，从而一次前向修复整个高斯场。

> **Hypothesis 1**：Wan2.2 VAE latent 在 3D 空间中的采样，比在 2D 像素平面上做 conv refine，能提供更语义一致的精化信号。

> **Hypothesis 2**：每个高斯体元独立 refine（per-Gaussian MLP）比 2D U-Net refine 更高效，且天然多视图一致。

> **Hypothesis 3**：video model 的 latent space 包含普通单目模型不具备的时间/运动先验，这对修复遮挡区域尤为关键。

---

## 2. 方法设计

### 2.1 整体架构

```
输入图像 [B, V, 3, H, W]
    │
    ├──────────────────────────────────────────────────────────────┐
    │                                                              │
    ▼                                                              ▼
AnySplat Encoder (frozen)                               Wan2.2 VAE (frozen)
    │                                                              │
    ├──→ 深度图 + 相机参数                                          ├──→ VAE Latent [48ch, 16× down]
    ├──→ 3D点云 (从深度反投影)                                       │
    ├──→ 体素化 → GaussianAdapter                                   ▼
    └──→ 3D Gaussians [N, 82+3D+O]                     Latent Grid [48, H/16, W/16]
              │                                                 │
              │                         对于每个Gaussians点(x,y,z) │
              │                              ▼                   │
              │                   投影到每张context view:(u_i,v_i) │
              │                              │                   │
              │                   在Latent Grid采样 → 48-dim vec  │
              │                              │                   │
              │                   cross-view聚合 (mean/attention) │
              │                              │                   │
              └──────────────┬───────────────┘                   │
                             │                                    │
                             ▼                                    │
              Per-Gaussian Feature Vector [82+3+1+48]
                  (scales, rot, SH, opacity, means, video_latent)
                             │
                             ▼
                  Lightweight MLP (3层, LayerNorm + ReLU)
                             │
                             ▼
                      Δ_params [82]
                             │
                             ▼
              refined_gaussians = gaussians + Δ
                             │
                             ▼
          Render target views (gsplat, differentiable)
                             │
                             ▼
              Loss = MSE + LPIPS (vs ground truth)
```

### 2.2 关键模块设计

#### 2.2.1 Video Latent 采样器

给定一个 3D 高斯点 `p = (x, y, z)`，以及一个 context view 的相机参数 `(extrinsic, intrinsic)`：

```
p_view = extrinsic @ p  →  (u, v) = intrinsic @ p_view
latent_feat = bilinear_sample(latent_grid, u/16, v/16)  # [48]
```

**跨视图聚合**：对于 V 个 context view，每个 view 提供一个 48-dim 向量。聚合方式：
- **Mean**：简单平均（基线）
- **Attention**：learnable 查询聚合（更强的选择）

**假设**（待验证）：
> Mean 聚合已经足够好，因为 VAE latent 本身已经被视频模型训练到了对视角变化鲁棒的特征空间。如果 Mean 不够，再尝试 Attention。

#### 2.2.2 Per-Gaussian MLP Refiner

```
Input: [82 + 3 + 1 + 48] = [134]
  └── 82 = scales(3) + rotations(4) + SH(75)           ← 高斯属性
  └── 3 = normalized_means                               ← 位置
  └── 1 = opacity                                         ← 透明度
  └── 48 = video_latent                                   ← 视频先验

MLP:
  Linear(134 → 256) → LayerNorm → ReLU
  Linear(256 → 128) → LayerNorm → ReLU
  Linear(128 → 82)  → zero-init                          ← 残差输出

Output: Δ[82] (与高斯参数空间一致)
refined = raw_params + Δ
```

**设计理由**：
1. **Zero-init**：最后一层 Linear 初始化为 0，确保首次前向 Δ=0，不破坏初始高斯场
2. **LayerNorm**：视频 latent 与高斯参数的量级差异大，需要归一化
3. **残差形式**：每个高斯只需预测微小的调整，不需要重新预测全部参数
4. **为什么轻量 MLP 够用**：Wan2.2 VAE latent 已经完成了绝大部分"理解"，MLP 只需要学习从"理解"到"调整"的简单映射

#### 2.2.3 Loss 设计

```
L_total = L_mse + λ_lpips * L_lpips + λ_reg * L_reg

L_mse = MSE(rendered, ground_truth)             ← 像素级精度
L_lpips = LPIPS(rendered, ground_truth)          ← 感知质量
L_reg = |Δ|² (L2 regularization on residuals)    ← 防止过度修改
```

**Loss 选择理由**：
- **不需要 Feature Consistency Loss**：在 prototype 阶段先不加，因为需要 auxiliary decoder，增加复杂度。如果 MSE+LPIPS 效果不够好再加
- **L2 正则化**：防止 refiner 对高斯参数做大幅修改，保持初始高斯场的结构
- 初始权重：`λ_lpips=0.05`, `λ_reg=1e-5`

---

## 3. 数据集选择

### 3.1 第一阶段：CO3Dv2（验证可行性）

| 属性 | 值 |
|------|-----|
| 类别 | `car`（~100 个 scene） |
| 每 scene 帧数 | 50-200 帧 |
| 使用方式 | 选 2 帧 context + 1 帧 target |
| 训练/验证划分 | 80/20 |
| 输入分辨率 | 224×448 |
| 总训练样本数 | ~8,000 个三元组 |

**选择理由**：CO3D v2 已有本地数据，`car` 类物体形状规整、光照变化丰富、遮挡情况多样。足够验证方案可行性。

### 3.2 第二阶段：DL3DV（论文级实验）

| 属性 | 值 |
|------|-----|
| 场景数 | 140 个 benchmark scenes |
| 使用方式 | 2/4 context views, 1 target view |
| 训练/验证 | 100/40 |
| 分辨率 | 512×960（需多卡）/ 256×512（单卡） |

### 3.3 数据加载流程

```
每个训练 step:
  1. 随机选一个 scene
  2. 随机选 2 帧 context + 1 帧 target（间隔 > threshold）
  3. 加载 context 图像 (224×448)
  4. 加载 target 图像 → 作为 ground truth
  5. 返回：context [2,3,H,W], target [3,H,W]
```

---

## 4. 实验计划

### 4.1 实验 1：Prototype 验证（3-5 天）

**目标**：验证整个 pipeline 能否跑通，refiner 能否降低 loss

| 配置 | 值 |
|------|-----|
| 数据集 | CO3Dv2 car (80/20 split) |
| Context views | 2 |
| Resolution | 224×448 |
| Training steps | 10,000 |
| Batch size | 1 |
| Learning rate | 1e-4, cosine decay |
| Optimizer | AdamW |
| 显存预计 | ~40 GB |
| 时间预计 | ~4 小时 |

**成功标准**：
- [ ] 训练 loss 稳定下降
- [ ] 验证集 PSNR 有提升趋势
- [ ] Δ 的 norm 不为 0（refiner 在真正修改参数）
- [ ] 未出现训练崩溃（loss NaN / 渲染发散）

**预期困难**：
- gsplat 可微渲染连接 MLP 输出的梯度流可能不稳定
- 多 view video latent 采样的坐标对齐精度
- 132K 个高斯 × 3 层 MLP × 每个 batch = 计算量

**失败预案**：
- 如果 Per-Gaussian MLP 训练不收敛，回退到 2D U-Net + Concat fusion（当前实现）
- 如果 Video Latent 采样对齐不准，改用双线性插值的 softened 版本

### 4.2 实验 2：Ablation Study（1 周）

| 实验 | 变量 | 问题 |
|------|------|------|
| E2.1 | Video Latent 有无 | Video latent 是否真的有帮助？ |
| E2.2 | VAE-only vs VAE+DiT | DiT 的语义特征是否比 VAE 底层的几何特征更有效？ |
| E2.3 | Context views 数量 (1/2/4) | 更多 view 是否提升效果？ |
| E2.4 | MLP 深度 (1/3/5 layers) | 更深的网络是否必要？ |
| E2.5 | 残差预测 vs 直接预测 | 残差形式是否确实更好？ |

### 4.3 实验 3：Cross-Attention Fusion（2 周）

如果 Mean 聚合不够好——也就是 video latent 在不同 view 间差异大，导致 Mean 聚合信息损失——则实现 Attention-based 聚合：

```
对于每个高斯:
  Q = learned_query (per-Gaussian learnable)
  K_i = video_latent_from_view_i (i=1..V)
  V_i = K_i
  attn_i = softmax(Q @ K_i / sqrt(d))
  aggregated_latent = sum(attn_i * V_i)
```

### 4.4 实验 4：与 Baseline 对比（论文必备）

| Baseline | 对比维度 |
|----------|----------|
| AnySplat 原始输出 | 直接对比 PSNR/LPIPS/SSIM |
| AnySplat + v3 restorer（2D U-Net）| 我们的 3D per-Gaussian vs 2D pixel-level |
| 不使用 video latent（只有高斯参数） | Video latent 的贡献 |

---

## 5. 论文故事线（暂定）

**标题**：VideoPriorGS: Feed-Forward 3D Gaussian Refinement with Video Foundation Model Priors

**核心贡献**：
1. 首次将视频 foundation model 的 latent space **直接**（而非生成→编码）用于 3DGS 精化
2. 提出 Per-Gaussian Video Latent Sampling 方法，在 3D 空间而非 2D 像素空间做先验融合
3. 轻量级 MLP（~1M 参数）即可实现有效的 feed-forward 精化

**差异化**：
- vs GaussFusion：latent 级融合，无需图像生成步骤
- vs ReSplat：引入视频语义先验，而非纯几何 recurrent
- vs GSFixer：一次前向，无需多步 diffusion

---

## 6. 风险管理

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 训练不收敛 | 中 | 高 | 先用 synthetic data 验证梯度流 |
| Δ 始终接近 0 | 中 | 高 | 降低 L2 reg 权重，监控 Δ norm |
| Video Latent 信息不足 | 低 | 中 | 尝试 DiT 深层特征替代 VAE-only |
| 渲染 Loss 与 3D 参数梯度耦合差 | 中 | 高 | 尝试 Feature Consistency Loss |
| gsplat 可微渲染在 132K 高斯上慢 | 低 | 中 | 降低分辨率或高斯数量 |

---

## 7. 迭代策略

```
Phase 1 ───→ 跑通流程，验证收敛（当前目标）
  │
  ├── Pass? → Phase 2: Ablation + 系统实验
  │
  └── Fail? → 诊断原因：
              ├── 梯度问题 → 改用 2D U-Net + Per-Pixel refine（v3路径）
              ├── Video Latent问题 → 替换为 DINOv2 / SD VAE
              └── 数据问题 → 换用 synthetic data
```

**关键决策点**：如果在 Phase 1 的 10K steps 内验证集 PSNR 完全没有提升趋势（且 Δ norm 极小），则说明 per-Gaussian MLP 无法有效利用 video latent。此时应优先尝试 **Cross-Attention Fusion** 或回退到 **2D U-Net** 方案。
