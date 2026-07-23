# Gaussian Scene Restoration — 独立模块

本目录包含从 AnySplat 代码库中提取出的 **Gaussian Scene Restoration** 模块及相关集成文件，实现了与 AnySplat 代码文件层面的解耦。

## 目录结构

```
post/
├── README.md                           # 本文件：概述与集成指南
├── gaussian_scene_restoration_report.md # 原设计文档
│
├── restorer/                           # ★ v1 核心模块（完全独立，仅依赖 torch）
│   ├── __init__.py
│   ├── feature_extractor.py            # ResidualUNet 实现
│   └── gaussian_scene_restorer.py      # GaussianSceneRestorer + Config
│
├── wan_restorer/                       # ★ v3 WAN增强模块（新增）
│   ├── __init__.py                     # 包入口
│   ├── config.py                       # WanRestorerCfg 配置
│   ├── wan_feature_extractor.py        # Wan2.2 VAE+DiT 特征提取器
│   ├── gaussian_scene_restorer_v3.py   # GaussianSceneRestorerV3
│   ├── demo_inference.py               # 独立推理测试脚本
│   ├── test_download.py                # 模型下载与验证脚本
│   └── environment.yaml                # Conda 环境定义
│
├── config/                             # 实验配置文件
│   └── anysplat_restorer.yaml          # v1 训练实验配置
│
├── patches/                            # 对 AnySplat 源码的最小修改（可 git apply）
│   ├── 0001-vggt-dpt-gs-head-return-intermediate.patch
│   ├── 0002-anysplat-encoder-restorer-integration.patch
│   ├── 0003-model-wrapper-optimizer-groups.patch
│   └── 0004-config-restorer-defaults.patch
│
└── integrations/                       # 集成辅助代码（依赖 AnySplat）
    ├── __init__.py
    ├── anysplat_composer.py            # 通过子类化添加 Restorer
    └── config_adapter.py               # 配置文件辅助工具
```

## 模块总览

### `post/restorer/` — 核心模块（完全独立）

- **零依赖**：仅依赖 `torch` 和 `torch.nn`，不导入任何 AnySplat 代码
- **即插即用**：可直接 `import` 到任何 PyTorch 项目
- **模型结构**：`ResidualUNet(131→64→128→256→256→128→64→64) → Conv2d(64→83, zero-init)`
- **参数量**：约 840K（hidden_dim=64）~ 3.3M（hidden_dim=128）

```python
from post.restorer import GaussianSceneRestorer, GaussianSceneRestorerCfg

cfg = GaussianSceneRestorerCfg(enabled=True, hidden_dim=64)
restorer = GaussianSceneRestorer(cfg)

refined = restorer(
    dpt_feats=dpt_fused_feats,      # [B*V, 128, H, W]
    raw_gs_params=raw_params,       # [B*V, 83, H, W]
    input_images=images,            # [B*V, 3, H, W]  (optional)
)
```

### 数据流

```
Input images [B,V,3,H,W]
    │
    ├── VGGT aggregator (frozen)
    ├── Camera/Depth head (frozen)
    │
    └── VGGT_DPT_GS_Head (frozen)
            │
            ├── DPT 多尺度融合 → 128ch 特征图 [B*V,128,H,W]
            │                                ↑ return_intermediate=True
            ├── output_conv2 → 原始GS参数 [B*V,83,H,W]
            │
            ▼  cat(DPT_feats, RGB) → [B*V,131,H,W]
            │
    ┌───────┴──────────────────────┐
    │  GaussianSceneRestorer      │
    │  ┌──────────────────┐       │
    │  │ ResidualUNet      │ → Δ  │  (zero-init: 首次 Δ=0)
    │  └──────────────────┘       │
    │           ▼                  │
    │  refined = raw + clamp(Δ)   │
    └───────────┬──────────────────┘
                │
    refined → 替换 anchor_feats 前 raw_gs_dim 通道
                │
    ├── Voxelization + GaussianAdapter (unchanged) → Gaussians
    ├── gsplat rasterization → rendered images
    └── Loss: MSE + LPIPS
```

## 集成方式

### 方式 A：应用补丁（推荐，最小侵入）

```bash
cd /path/to/anysplat
git am /path/to/GuassDiff/post/patches/0001-*.patch
git am /path/to/GuassDiff/post/patches/0002-*.patch
git am /path/to/GuassDiff/post/patches/0003-*.patch
git am /path/to/GuassDiff/post/patches/0004-*.patch
```

补丁修改的文件：
| 文件 | 修改内容 |
|------|----------|
| `vggt_dpt_gs_head.py` | +6行：`return_intermediate` 参数 + fused_features 捕获 |
| `anysplat.py` | +30行：restorer 导入、config字段、初始化、forward逻辑 |
| `model_wrapper.py` | +1行：优化器分组中加入 restorer |
| `encoder/anysplat.yaml` | +10行：默认配置 |

### 方式 B：子类化 AnySplat（无需修改原文件）

```python
from post.integrations import AnySplatWithRestorer

# 直接替代 EncoderAnySplat
encoder = AnySplatWithRestorer(cfg)
# 当 cfg.gaussian_scene_restorer.enabled = True 时自动启用
```

### 方式 C：单独使用 Restorer 模块

将 `post/restorer/` 目录复制到你的项目，通过提供相应的特征和参数即可使用，无需 AnySplat：

```python
from your_project.restorer import GaussianSceneRestorer

restorer = GaussianSceneRestorer(cfg)
refined_params = restorer(dpt_feats, raw_gs_params, input_images)
```

## 训练 (v1)

配置参考：`post/config/anysplat_restorer.yaml`

```bash
python src/main_pretrained.py +experiment=anysplat_restorer
```

关键训练参数：
- 可训练参数：仅 GaussianSceneRestorer ~7M
- 冻结部分：VGGT aggregator, camera_head, depth_head, DPT_GS_Head
- 优化器：AdamW, lr=1e-3, weight_decay=0.05
- LR Scheduler：Linear warmup 500 + CosineAnnealing
- Loss：MSE (w=1.0) + LPIPS (w=0.05)
- Steps：10,000
- 显存：基础 ~13GB + restorer ~3GB ≈ 16GB


## V3：Wan2.2 视频模型增强（`post/wan_restorer/`）

### 概述

v3 用 **Wan2.2-TI2V-5B** 视频生成模型的 VAE 编码器 + DiT Transformer 替换 v1 的 ResidualUNet，
提供更丰富的视觉先验来优化高斯参数。

| 对比项 | v1 (ResidualUNet + DPT) | v3 (Wan2.2 VAE + DiT) |
|--------|------------------------|----------------------|
| **特征来源** | DPT 中间特征（128ch） | Wan VAE latent（48ch）+ DiT 自注意力 |
| **参数量** | ~7M | ~5B（frozen）+ ~7M（trainable） |
| **空间上下文** | 卷积局部感受野 | Transformer 全局自注意力 |
| **语义先验** | 仅几何特征 | 语义+几何+运动先验 |
| **GPU 需求** | ~16GB | ~30-40GB（97GB available） |

### 架构

```
Input images [B,V,3,H,W]
    │
    ├── VGGT aggregator (frozen, optional)
    │
    └── Wan2.2 Feature Extractor (frozen)      ★ NEW
        ├── VAE Encoder → latents [48ch, H/16, W/16]
        ├── 3D Patch Embedding → tokens
        ├── DiT Transformer (first N layers)
        └── Feature Projector → [B*V, 128, H, W]
    
    cat(WAN_feats, DPT_feats, RGB) → [B*V, in_channels, H, W]
                    │
                    ▼
              Lightweight Refinement Head
                    │
                    ▼
              Δ residuals (zero-init)
```

### 使用方式

```bash
# 1. 激活环境
conda activate wan_restorer

# 2. 下载并验证模型
python -m post.wan_restorer.test_download

# 3. 运行推理测试
python -m post.wan_restorer.demo_inference --num_views 2 --num_dit_layers 8

# 4. 使用实际图像测试
python -m post.wan_restorer.demo_inference --image_dir /path/to/images --use_dpt
```

### 代码结构

| 文件 | 说明 |
|------|------|
| `config.py` | WanRestorerCfg 配置类（模型ID、层数、特征维度等） |
| `wan_feature_extractor.py` | 核心：加载 Wan2.2 VAE + DiT，提取中间特征并投影到像素空间 |
| `gaussian_scene_restorer_v3.py` | 完整 V3 流程：WAN特征→融合→轻量头→残差 |
| `demo_inference.py` | 独立测试脚本（支持合成/真实图像） |
| `test_download.py` | 模型下载与基础推理验证 |
| `environment.yaml` | Conda 环境配置（Python 3.10 + CUDA 12.8） |

### 环境配置

```bash
conda env create -f post/wan_restorer/environment.yaml
conda activate wan_restorer
```

模型（Wan2.2-TI2V-5B, ~10GB）首次运行时自动从 HuggingFace 下载，
缓存在 `~/.cache/huggingface/`。

### 推理性能（预估）

| 配置 | VRAM | 时间（224x448, 2 views） |
|------|------|------------------------|
| VAE only (num_dit_layers=0) | ~15 GB | ~0.5s |
| 4 DiT layers | ~25 GB | ~2s |
| 8 DiT layers | ~35 GB | ~4s |

## 设计原则

1. **最小侵入** — 通过 enabled: false 默认禁用，不影响原有流程
2. **残差式** — 预测 Δ，zero-init，首次前向 Δ=0
3. **轻量化** — 可训练部分仅 ~7M 参数
4. **可替换** — FeatureExtractor 可不断升级（v1→v3→v4）

## 升级路径

| 版本 | 方案 | 状态 |
|------|------|------|
| v1 | U-Net + DPT 特征 → 像素级残差 | ✅ 发布 |
| v1.1 | 增加 rendered image skip（渲染反馈） | 📋 计划 |
| v2 | 逐高斯 MLP（后适配器残差 + 高斯级特征） | 📋 计划 |
| **v3** | **视频 Foundation Model（Wan2.2）替换特征提取器** | **✅ 当前** |
| v4 | Diffusion 先验 → latent 去噪 → GS 梯度 | 📋 计划 |

## License

本模块的代码按与 AnySplat 相同的许可协议（CC BY-NC-SA 4.0）发布。
