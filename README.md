# Gaussian Scene Restoration

视频模型指导的高斯场后处理修复完善（Video Model-Guided Gaussian Field Post-Processing Restoration）

## 项目结构

```
GuassDiff/
├── gaussian_restorer/          # ★ 核心：高斯基元修复框架（v4 统一架构）
│   ├── VideoExtractor          # Wan2.2 时序特征提取
│   ├── GaussianFeatureEncoder  # 高斯基元特征编码
│   ├── UnifiedRepairer         # 统一修复网络（per-Gaussian + 新基元生成）
│   └── RepairLoss              # 多组件损失函数
│
├── wan_restorer/               # ★ Wan2.2 视频模型特征提取模块
│   ├── GaussianSceneRestorerV3 # WAN增强修复器
│   └── WanFeatureExtractor     # Wan2.2 VAE+DiT 特征提取器
│
├── restorer/                   # ★ V1 修复模块（独立，仅依赖 torch）
│   └── GaussianSceneRestorer   # 像素级残差修复网络
│
├── data/                       # ★ 数据集接口
│   ├── vrnerf.py               # VRNeRF 多焦距数据集
│   └── __init__.py             # 数据集注册表（可扩展）
│
├── config/                     # ★ 实验配置文件
│   ├── default.yaml            # 默认训练配置
│   └── anysplat_restorer.yaml  # 修复器实验配置
│
├── patches/                    # AnySplat 源码补丁
│
├── phase1_train.py             # Phase1 统一训练入口（YAML配置驱动）
├── phase1_train_unified.py     # 统一修复网络训练（v4）
├── phase1_train_v2.py          # 基于点的修复训练
├── phase1_train_repairer.py    # 修复器训练
├── train_restorer.py           # 原型训练脚本
├── validate_pipeline.py        # 梯度流验证脚本
├── experiment_proof.py         # 300步概念验证
│
├── demo_wan_proper.py          # WAN2.2 完整推理流程演示
├── demo_wan_generation.py      # WAN2.2 视频生成演示
├── demo_video_interpolation.py # 视频插帧演示
│
├── scripts/                    # 工具脚本
│   ├── convert_output2_to_dl3dv.py
│   ├── generate_output2_depth.py
│   ├── prepare_output2_bench_gt.py
│   └── wandb_data_export.py
│
├── anysplat/                   # AnySplat 依赖（封装）— 仅作初始输入
│   ├── src/                    # AnySplat 核心代码
│   ├── config/                 # AnySplat 配置
│   ├── pretrained_model/       # 预训练权重
│   ├── integrations/           # 桥接代码
│   └── ...（演示/工具脚本）
│
└── *.md                        # 设计文档、研究计划、实验报告
```

## 使用方式

```bash
# 激活环境
conda activate wan_restorer

# 训练（YAML配置驱动）
CUDA_VISIBLE_DEVICES=0 python phase1_train.py --config config/default.yaml

# 用自定义配置
CUDA_VISIBLE_DEVICES=0 python phase1_train.py --config config/my_exp.yaml

# 仅测试数据加载
CUDA_VISIBLE_DEVICES=0 python phase1_train.py --dry-run
```

## 依赖关系

- **AnySplat** (`anysplat/`)：提供初始高斯基元输入，仅用作 frozen backbone
- **Wan2.2**：视频先验模型，用于特征提取
- 核心修复模块 (`gaussian_restorer/`, `restorer/`, `wan_restorer/`) 独立于 AnySplat
