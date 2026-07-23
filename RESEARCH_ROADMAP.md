# Gaussian Scene Restoration — CCF-A 论文科研路线图

> 当前状态：框架已搭建，流水线已跑通，但距离 CCF-A 论文仍有关键缺口

---

## 一、当前已完成的工程基础

### ✅ 已完成

| 模块 | 状态 | 说明 |
|------|------|------|
| AnySplat 预训练模型加载 | ✅ | 1.19B 参数，132K Gaussians，1.32s 前向 |
| Wan2.2-TI2V-5B 视频先验提取 | ✅ | VAE 48ch latent + DiT 4层特征，~7s 加载 |
| 模块化 Restorer 框架 | ✅ | 4种Fusion + 4种Refiner + 2种GS编码器 |
| 零初始化验证 | ✅ | Max \|Δ\| = 0.00000000，首次前向无损 |
| 端到端流水线 | ✅ | images → AnySplat → Wan2.2 → Restorer → Render |
| 显存验证 | ✅ | 40 GB / 102 GB（RTX PRO 6000, 39%） |

### ⚠️ 部分完成（有已知问题）

| 模块 | 问题 | 优先级 |
|------|------|--------|
| Cross-Attention Fusion | 输出通道451≠预期131，需要对齐 | 高 |
| Multi-Stage Refiner | 通道拼接错误 | 高 |
| 从 refined params → 3D Gaussians 重建 | 体素化函数接口不匹配，需调试 | 中 |
| Wan 模型加载重复 | 每次 forward 重载（~7s） | 中 |

---

## 二、核心缺口：距离 CCF-A 论文还差什么

### 缺口 1：缺少可训练的完整训练闭环

**当前状态**：流水线可以前向运行，但 restorer 是零初始化的（Δ=0），训练环节完全缺失。

**需要做**：
- [ ] 构建训练循环：读取 DL3DV/CO3D 数据集 → AnySplat 前向 → Wan2.2 前向 → Restorer 前向 → gsplat 可微渲染 → MSE+LPIPS 损失 → 反向传播
- [ ] 冻结 AnySplat 和 Wan2.2 的所有参数（约 6B），只训练 Restorer（~22M）
- [ ] 验证训练后 restorer 能显著提升渲染质量（PSNR +1~3dB）
- [ ] **预计训练时间**：~3-5 天在单卡 PRO 6000 上完成 10K steps

### 缺口 2：缺少有说服力的定量结果

**CCF-A 论文要求**：在多个 benchmark（DL3DV、CO3D、ScanNet++）上有显著的定量提升。

| 指标 | 当前 | 目标 |
|------|------|------|
| PSNR (↑) | 未训练 | +1.0~3.0 dB |
| LPIPS (↓) | 未训练 | -0.02~0.05 |
| SSIM (↑) | 未训练 | +0.01~0.03 |
| 推理速度 | 已验证 | < 2s/scene |

**需要做**：
- [ ] 在 3 个以上 benchmark 上跑完整评估
- [ ] 与 baseline (AnySplat 原版) 做 ablation study
- [ ] 可视化 before/after 对比图

### 缺口 3：缺少有说服力的定性结果

**需要做**：
- [ ] 修复体素化重建流水线，使 refined params → refined Gaussians → refined render 可运行
- [ ] 在多个场景上展示 before/after 渲染对比
- [ ] 展示 failure cases（什么情况下 restorer 失效）
- [ ] 视频 fly-through 对比

### 缺口 4：缺少学术贡献的 novelty

**CCF-A 论文必须有明确的学术贡献**。目前的 "Wan2.2 特征 + 轻量头" 方案虽然工程上可行，但缺少学术深度。

**可能的研究方向**（需要选择一个作为论文核心贡献）：

#### 方向 A：3D-Aware Video Prior Fusion（推荐）

**核心思想**：当前在 2D 像素空间做 fusion，忽略了高斯场的 3D 结构。真正的创新点在于：
- 将 Wan2.2 的 2D latent 反投影到 3D 空间
- 在 3D 体素空间（而非 2D 图像空间）进行 Video Prior 与 Gaussian Feature 的融合
- 利用 3D 空间的一致性约束来 refine Gaussians

**为什么是 CCF-A 级别**：目前没有工作将视频 foundation model 的 latent 以 3D 感知的方式用于 3DGS 后处理优化。

#### 方向 B：Cross-View Latent Consistency

**核心思想**：多视图输入时，Wan2.2 对每个视图独立提取 latent，但这些 latent 应该对应同一个 3D 场景。可以约束：
- 不同视图的 latent 在共享 3D 空间中的一致性
- 用多视图一致性作为自监督信号（无需 GT 图像）

#### 方向 C：Progressive Coarse-to-Fine Refinement

**核心思想**：
- 阶段 1：用 video latent 做全局场景补全（填充空洞、修复漂浮）
- 阶段 2：用渲染 loss 做局部细节优化
- 两阶段联合训练，互相增强

---

## 三、实验计划

### Phase 1：训练基础设施（1-2 周）

1. **数据准备**
   - DL3DV：下载 1000 个场景，预处理为 AnySplat 输入格式
   - CO3Dv2：已有 local 数据（car, teddybear）
   - 每场景：2 个 context view + 1 个 target view

2. **训练循环**
   ```python
   for scene in dataloader:
       context, target = scene
       # AnySplat forward (frozen)
       gaussians, raw_gs = anysplat(context)
       # Wan2.2 forward (frozen)
       video_latents = wan_extractor(context)
       # Restorer forward
       refined_gs = restorer(video_latents, raw_gs)
       # Render target view
       rendered = render(refined_gs, target_pose)
       # Loss
       loss = MSE(rendered, target) + LPIPS(rendered, target)
       loss.backward()
       optimizer.step()
   ```

3. **评估**
   - 每 500 steps 在验证集上计算 PSNR/LPIPS/SSIM
   - 保存 best checkpoint

### Phase 2：Ablation Study（1 周）

| 实验 | 变量 | 预期结论 |
|------|------|----------|
| 融合策略对比 | concat vs cross-attn vs gated | Cross-attn 质量最高，但慢 |
| Refiner 设计对比 | residual vs feature vs confidence | Confidence 可解释性好 |
| DiT 层数影响 | 0/2/4/8/30 layers | 4-8 层质量/速度最佳 |
| 视频模型对比 | Wan vs DINOv2 vs SD VAE | Wan 语义更丰富 |

### Phase 3：Novelty 验证（2-3 周）

1. **3D-aware fusion** 实现
2. **多视图一致性约束** 实现
3. 消融实验验证每个 novel component 的贡献

### Phase 4：论文撰写（2 周）

---

## 四、当前最急需解决的问题

按优先级排序：

| # | 问题 | 影响 | 预计解决时间 |
|---|------|------|------------|
| 1 | **修复训练闭环**（数据加载+训练循环+可微渲染） | 无法产生任何定量结果 | 3-5 天 |
| 2 | **修复体素化重建流水线** | 无法展示 after 渲染对比 | 1-2 天 |
| 3 | **Cross-Attention Fusion bug** | 无法运行最强融合策略 | 半天 |
| 4 | **Multi-Stage Refiner bug** | 无法运行多阶段细化 | 半天 |
| 5 | **实现 3D-aware fusion 作为核心创新点** | 论文 novelty | 1-2 周 |

---

## 五、总结

**当前框架的定位**：一个基础扎实、高度模块化的研究平台，适合作为 CCF-A 论文的实验基础设施。

**关键差距总结**：

```
当前状态  →  CCF-A 论文要求
   │              │
   ├── 流水线跑通  →  完整的训练-评估闭环
   ├── 模块化设计  →  有效的 ablation study
   ├── zero-init  →  显著的定量提升 (PSNR+1~3dB)
   ├── 工程框架    →  有深度的学术创新点 (3D-aware fusion)
   └── 单场景验证  →  多 benchmark 全面评估
```

**建议的下一步**：优先搭建训练闭环（Phase 1），跑出第一组定量结果（即使不高），然后基于结果迭代学术创新。
