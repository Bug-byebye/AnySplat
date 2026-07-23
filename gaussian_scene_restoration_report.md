# Gaussian Scene Restoration — v1 原型实现报告

## 1. 背景与动机

Feed-forward 3D Gaussian 重建方法（如 AnySplat）虽然速度快，但输出质量受限于：
- **漂浮高斯**（floating Gaussians）：由于遮挡或匹配歧义产生的虚假几何
- **几何缺失**：输入视图覆盖不足的区域表现空洞
- **纹理模糊**：前馈网络容量有限导致的细节丢失

本项目的目标是在 **不修改原有重建流程** 的前提下，新增一个独立的 **Gaussian Scene Restoration** 模块，对已生成的高斯场进行后处理优化，并为后续引入视频生成模型等更强的先验预留接口。

## 2. 整体思路

```
参考工作：AnchorSplat (2026)     → 残差式高斯修正（在已有高斯上预测 Δ）
         GaussFusion (2025)     → 利用生成模型的几何先验

本方案：结合两者理念，但用"已有 VGGT 的 DPT 中间特征"替代完整视频模型，
        实现一个可直接运行的轻量级原型。
```

### 核心设计原则

| 原则 | 具体实践 |
|------|----------|
| **最小侵入** | Restorer 作为 `EncoderAnySplat` 内部可选子模块，默认 `enabled: false` |
| **残差式** | 预测 Δraw_GS，zero-init，不改变初始高斯场的结构 |
| **轻量化** | ~7M 参数，单 GPU (RTX PRO 6000) 可训练 |
| **可替换** | FeatureExtractor 可被后续更强的视频模型特征提取器替换 |

## 3. 技术选型

### 3.1 特征来源：DPT 中间特征（"免费"的高质量先验）

选择原因：
- VGGT_DPT_GS_Head 内部已经计算了 128ch 的多尺度融合特征图（来自 VGGT 的 [4,11,17,23] 层）
- 该特征图已编码了丰富的多视图几何信息，且无需额外计算
- 与完整视频模型相比：**零推理开销，零额外显存**

### 3.2 网络结构：ResidualUNet（而非逐高斯 MLP）

| 对比项 | ResidualUNet（本方案） | 逐高斯 MLP |
|--------|----------------------|------------|
| **参数** | ~7M | ~5-10M |
| **空间上下文** | ✓ 卷积天然考虑邻域 | ✗ 逐点独立 |
| **对齐难度** | 低（像素对齐的特征） | 高（需建立像素→高斯映射） |
| **替换灵活性** | 高（换特征提取器即可） | 低 |

### 3.3 残差位置：Pre-GaussianAdapter（而非后适配器）

在 GaussianAdapter 之前修正 raw GS params（密度+尺度+旋转+SH），因为：
- Adapter 内含 softplus/quaternion norm 等非线性变换
- Raw space 中的残差更简单可预测
- **不预测 Δmeans**：位置由深度头产生，几何含义明确；low-opacity 即可消除坏点

### 3.4 Zero-initialization

残差头（Conv2d）权重和偏置均初始化为 0：
- 首次前向时 refined = raw，**零风险集成**
- 训练过程中梯度从优化后的渲染质量反传

## 4. 数据流总览

```
Input images [B,V,3,H,W]
    │
    ├── VGGT aggregator (frozen)     ← facebook/VGGT-1B 预训练权重
    ├── Camera/Depth head (frozen)
    │
    └── VGGT_DPT_GS_Head (frozen)
            │
            ├── DPT 多尺度融合 → 128ch 特征图 [B*V,128,H,W]  ← 新增 return_intermediate 接口
            │
            ├── output_conv2 → 原始GS参数 [B*V,84,H,W] (83 params + 1 confidence)
            │
            ▼  cat(DPT_feats, RGB) → [B*V,131,H,W]
            │
    ┌───────┴──────────────────────────────────────────┐
    │  GaussianSceneRestorer (训练)                     │
    │                                                   │
    │  ┌──────────────────┐   ┌──────────────────┐     │
    │  │ ResidualUNet     │ → │ Conv2d(64→83, 1) │ → Δ  │
    │  │ 131→64→128→256→  │   │ (zero-init)      │     │
    │  │ 256→256→128→64→64│   └──────────────────┘     │
    │  └──────────────────┘         │                   │
    │                               ▼                   │
    │                    refined = raw + clamp(Δ)       │
    └───────────────────────┬──────────────────────────┘
                            │
    ▼  refined → 替换 anchor_feats 前 raw_gs_dim 通道      │
    │                                                      │
    ├── Voxelization (unchanged)
    ├── GaussianAdapter (unchanged) → Gaussians
    ├── gsplat rasterization → rendered images
    └── Loss: MSE + LPIPS
```

## 5. 文件结构

```
src/model/restorer/
├── __init__.py                      # 包入口
├── feature_extractor.py             # ResidualUNet 实现（~100行）
└── gaussian_scene_restorer.py       # GaussianSceneRestorer + Config（~120行）

config/experiment/
└── anysplat_restorer.yaml           # 训练实验配置（~90行）

Modified:
├── src/model/encoder/heads/vggt_dpt_gs_head.py   # +6行 return_intermediate
├── src/model/encoder/anysplat.py                  # +30行 集成restorer
├── src/model/model_wrapper.py                     # +1行 优化器分组
└── config/model/encoder/anysplat.yaml             # +10行 默认配置
```

**涉及修改的总代码量**：新增 ~230 行，修改 ~47 行。

## 6. 训练策略

| 项目 | 值 |
|------|-----|
| **可训练参数** | GaussianSceneRestorer 仅 ~7M |
| **冻结部分** | VGGT aggregator, camera_head, depth_head, DPT_GS_Head |
| **优化器** | AdamW, lr=1e-3, weight_decay=0.05 |
| **LR Scheduler** | Linear warmup 500 + CosineAnnealing |
| **Loss** | MSE (w=1.0) + LPIPS (w=0.05) |
| **Steps** | 10,000 |
| **显存** | 基础 ~13GB + restorer ~3GB ≈ 16GB（远低于 96GB） |

## 7. 未来升级路径

```
v1 (当前)     U-Net + DPT 特征 → 像素级残差
   ↓
v1.1         增加 rendered image skip（渲染反馈）
   ↓
v2           逐高斯 MLP（后适配器残差 + 高斯级特征）
   ↓
v3           视频 Foundation Model 替换特征提取器
   ↓
v4           Diffusion 先验 → latent 去噪 → GS 梯度
```

## 8. 验证状态

- [x] 模块导入与实例化：✓
- [x] Zero-init 首次前向 Δ=0：✓
- [x] DPT 中间特征返回接口：✓
- [x] Config 字段注册与默认值：✓
- [x] 优化器参数分组：✓
- [x] YAML 配置解析：✓
- [x] 所有文件语法编译：✓
- [ ] 实际训练收敛（需启动实验验证）— **下一步**
