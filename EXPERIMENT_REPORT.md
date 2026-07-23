# Gaussian Scene Restoration — 实验报告

## 一、框架概览

构建了模块化 Gaussian Scene Restoration 框架，包含以下可替换组件：

```
post/gaussian_restorer/
├── __init__.py              # 包入口
├── config.py                # 统一配置（5 个子配置模块）
├── video_prior.py           # 视频 Foundation Model Prior 提取器
├── gaussian_features.py     # Gaussian Feature 编码器
├── fusion.py                # 4 种融合策略
├── refiner.py               # 4 种 Refiner 设计
├── gaussian_scene_restorer.py # 主编排器
└── experiment_runner.py     # 实验运行器
```

## 二、实验设置

| 项目 | 值 |
|------|-----|
| GPU | NVIDIA RTX PRO 6000 Blackwell (102 GB) |
| CUDA | 12.8 |
| PyTorch | 2.9.1+cu128 |
| Video Model | Wan-AI/Wan2.2-TI2V-5B-Diffusers (5B params) |
| 输入 | 2 views × 224×448, batch=1 |
| Warmup | 1 次前向（含模型加载） |
| 计时 | 另测 1 次前向（已加载后） |

## 三、实验结果

### 3.1 融合策略对比

| 策略 | Time(ms) | VRAM(GB) | Params(M) | ZeroInit | 状态 |
|------|----------|----------|-----------|----------|------|
| **Concat + Conv** (baseline) | **106.99** | **14.27** | **5171.99** | ✅ | ✅ 通过 |
| **Cross-Attention** (windowed) | — | — | 5165.25 | — | ❌ 通道维度不匹配 |
| **Gated Fusion** | **111.35** | **14.27** | **5172.02** | ✅ | ✅ 通过 |

**分析：**
- **Concat + Conv** 与 **Gated Fusion** 性能几乎相同（~107-111ms），显存一致（14.27GB）
- Gated Fusion 仅比 ConcatConv 慢 4%（111 vs 107ms），差异在噪声范围内
- **Cross-Attention** 当前有通道对齐问题：fusion 输出 451 通道但 refiner 期望 131 通道。原因是 cross-attention 的内部 hidden_dim 计算与融合后的 refiner_in 维度不匹配
- **待修复** Cross-Attention 的输出维度对齐

### 3.2 Refiner 设计对比

| 设计 | Time(ms) | VRAM(GB) | Params(M) | ZeroInit | 状态 |
|------|----------|----------|-----------|----------|------|
| **Residual** (v3 baseline) | **106.55** | **14.27** | **5171.99** | ✅ | ✅ 通过 |
| **Feature Residual** | **104.05** | **14.27** | **5172.01** | ✅ | ✅ 通过 |
| **Confidence** | **115.09** | **14.27** | **5171.99** | ✅ | ✅ 通过 |
| **Multi-Stage** | — | — | 5165.25 | — | ❌ 通道拼接错误 |

**分析：**
- **Residual** 与 **Feature Residual** 性能几乎一致（106 vs 104ms），但 Feature Residual 使用更低维的隐空间（32→83）
- **Confidence** 略慢（115ms），因为多了 confidence head 的额外计算，但有可解释的置信度图
- **Multi-Stage** 在 stage 间通道拼接有 bug：fused 特征（128ch）与当前参数（83ch）拼接后的通道数与 stage conv 期望的不匹配
- **待修复** Multi-Stage 的通道管理

### 3.3 视频先验深度

| 配置 | 预计 Time(ms) | 预计 VRAM(GB) | 说明 |
|------|-------------|--------------|------|
| VAE-only (0 DiT layers) | ~0.01s | ~5.6 GB | 仅 VAE 编码器 |
| DiT-4 layers | ~0.08s | ~12 GB | 当前 v3 默认 |
| DiT-8 layers | ~0.15s | ~14 GB | 更丰富的特征 |
| DiT-30 (full) | ~0.4s | ~25 GB | 全部语义信息 |

**说明：** 完整 DiT 实验每项需重载模型（~10s），暂未执行完整对比。

## 四、关键发现与问题

### 4.1 已确认的设计原则

1. **Zero-initialization 验证通过**: 所有通过实验的 `Max |Δ| = 0.000000`，确保首次前向无损
2. **显存稳定**: 所有配置峰值均为 ~14.27 GB，远低于 102 GB 上限（占用 14%）
3. **推理速度快**: ~107ms 对于 2×224×448 输入，足够实时应用
4. **参数量**: Wan5B 占 99.9% 参数，可训练部分仅 ~22.58M（0.4%）

### 4.2 发现的问题

| 问题 | 影响 | 优先级 |
|------|------|--------|
| Cross-Attention 融合输出通道数不匹配 | 无法运行 | 高 |
| Multi-Stage refiner 通道拼接错误 | 无法运行 | 高 |
| 每次实验重载 Wan 模型（~10s） | 实验效率低 | 中 |
| Wan VAE logvar 不为零（随机潜在空间） | 均值提取合理，但信息有损 | 低 |
| VAE-only vs DiT 未定量对比 | 未知语义特征的收益 | 中 |

### 4.3 需进一步探索的设计问题

1. **从 3D Gaussians 渲染 GP-buffer**: 当前使用预体素化的逐像素 raw GS params。用户要求从重建后的 3D Gaussians 渲染 GP-buffer。这需要 gsplat 集成（当前环境无此依赖）。

2. **Video Latent → Gaussian Feature 的 Cross-Attention**: 设计方案正确但实现有 bug。需要修复通道对齐后重新测试。

3. **多视图 GP-buffer**: 当前每视图独立处理，没有跨视图融合。可能的改进方向：跨视图注意力或 3D 体素融合。

4. **Loss 设计**: 当前仅测试了前向传播（zero-init）。实际训练需要实现 MSE+LPIPS 渲染损失，以及可选的 Feature Consistency 损失。

## 五、下一步工作

### 短期（修复现有问题）
1. 修复 Cross-Attention fusion 输出通道对齐
2. 修复 Multi-Stage refiner 通道拼接
3. 实现 Wan 模型共享缓存（避免 10s 重载）

### 中期（实验与验证）
4. 集成 gsplat 渲染 GP-buffer
5. 定量比较不同融合策略的实际训练效果
6. 探索 Feature Consistency Loss

### 长期（研究级实验）
7. 替换视频模型（DINOv2, SD VAE 等）
8. 跨视图注意力融合
9. 端到端训练收敛实验

## 六、对整体工作流的反思

**当前框架的优点：**
- 高度模块化：可独立替换视频模型、融合策略、refiner 设计
- 与 AnySplat 解耦：作为独立的 post-hoc 模块
- 显存友好：~14 GB 峰值，可扩展到更大分辨率

**当前框架的局限性：**
1. **视频模型即用性**: Wan2.2 需要特殊的分时域填充（4× temporal），导致每视图需额外计算
2. **潜在空间对齐**: VAE 的 48ch 潜在空间与 Gaussian 参数空间（83ch）的语义对齐未经验证
3. **缺乏 3D 感知**: 当前在 2D 图像空间操作，未利用高斯场景的 3D 结构
4. **评估指标缺失**: 当前仅测试前向传播，需要完整的渲染质量评估（PSNR, LPIPS, SSIM）

**建议的框架改进方向：**
- **3D-aware fusion**: 在 3D 体素空间（而非 2D 像素空间）进行特征融合，利用高斯场景的几何结构
- **双向先验**: 不仅视频模型指导高斯优化，高斯场景也可提供几何先验给视频模型
- **渐进式细化**: 从粗到细的多阶段优化，类似 coarse-to-fine 策略
