# AnySplat: Feed-forward 3D Gaussian Splatting from Unconstrained Views

[![Project Website](https://img.shields.io/badge/AnySplat-Website-4CAF50?logo=googlechrome&logoColor=white)](https://city-super.github.io/anysplat/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=b31b1b)](https://arxiv.org/pdf/2505.23716)
[![Gradio Demo](https://img.shields.io/badge/Gradio-Demo-orange?style=flat&logo=Gradio&logoColor=red)](https://huggingface.co/spaces/alexnasa/AnySplat)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/lhjiang/anysplat)

[Lihan Jiang*](https://jianglh-whu.github.io/), [Yucheng Mao*](https://myc634.github.io/yuchengmao/), [Linning Xu](https://eveneveno.github.io/lnxu),
[Tao Lu](https://inspirelt.github.io/), [Kerui Ren](https://github.com/tongji-rkr), [Yichen Jin](), [Xudong Xu](https://scholar.google.com.hk/citations?user=D8VMkA8AAAAJ&hl=en), [Mulin Yu](https://scholar.google.com/citations?user=w0Od3hQAAAAJ), [Jiangmiao Pang](https://oceanpang.github.io/), [Feng Zhao](https://scholar.google.co.uk/citations?user=r6CvuOUAAAAJ&hl=en), [Dahua Lin](http://dahua.site/), [Bo Dai<sup>†</sup>](https://daibo.info/) <br />


## Overview
<p align="center">
<img src="assets/pipeline.jpg" width="100%" height="auto" class="center">
</p>

Starting from a set of uncalibrated images, a transformer-based geometry encoder is followed by three decoder heads: <i>F<sub>G</sub></i>, <i>F<sub>D</sub></i>, and <i>F<sub>C</sub></i>, which respectively predict the Gaussian parameters (μ, σ, r, s, c), the depth map <i>D</i>, and the camera poses <i>p</i>. These outputs are used to construct a set of pixel-wise 3D Gaussians, which is then voxelized into pre-voxel 3D Gaussians with the proposed Differentiable Voxelization module. From the voxelized 3D Gaussians, multi-view images and depth maps are subsequently rendered. The rendered images are supervised using an RGB loss against the ground truth image, while the rendered depth maps, along with the decoded depth <i>D</i> and camera poses <i>p</i>, are used to compute geometry losses. The geometries are supervised by pseudo-geometry priors obtained by the pretrained VGGT.

## Installation

Our code relies on Python 3.10+, and is developed based on PyTorch 2.2.0 and CUDA 12.1, but it should work with other PyTorch/CUDA versions as well.

1. Clone AnySplat.
```bash
git clone https://github.com/OpenRobotLab/AnySplat.git
cd AnySplat
```

2. Create the environment, here we show an example using conda.
```bash
conda create -y -n anysplat python=3.10
conda activate anysplat
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

3. (Optional) Download pretrained model locally:
```bash
# The model will be automatically downloaded from Hugging Face if not found locally
# To download manually for offline use, visit: https://huggingface.co/lhjiang/anysplat
mkdir -p pretrained_model
# Place model files in pretrained_model/
```

## Model Loading Strategy

This repository implements an intelligent **local-first, HuggingFace-fallback** model loading mechanism:

- **Priority 1**: Load from local path (configurable in YAML files)
- **Priority 2**: Automatically download from HuggingFace Hub (`lhjiang/anysplat`) if local model not found

Configure the model path in respective YAML config files:
```yaml
# config/ttt.yaml, config/itr.yaml
pretrained_model_path: "pretrained_model"  # Relative path from project root

# config/nvs_compare.yaml
experiment:
  pretrained_model_path: "pretrained_model"
```

You can also specify absolute paths:
```yaml
pretrained_model_path: "/path/to/your/model"
```

## Usage

### 1. Basic Feed-Forward Inference

Quick inference with the pretrained model (`inference.py`):

```bash
python inference.py
```

Or use in Python:

```python
from pathlib import Path
import torch
from src.model.model.anysplat import AnySplat
from src.utils.image import process_image
from src.utils.model_loading import load_model_with_fallback

# Load model with automatic fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_with_fallback(
    local_path="pretrained_model",  # Try local first, fallback to HF Hub
    device=device,
)

# Load and preprocess images
image_paths = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"] 
images = [process_image(img) for img in image_paths]
images = torch.stack(images, dim=0).unsqueeze(0).to(device)  # [1, K, 3, 448, 448]

# Run inference
gaussians, pred_context_pose = model.inference((images+1)*0.5)
```

### 2. Test-Time Training (TTT)

**File**: `ttt.py`

Refine the model at test time on input images to improve reconstruction quality. TTT fine-tunes the Gaussian adapter and parameter prediction heads using the test images themselves.

```bash
python ttt.py
```

**Key Features:**
- Fine-tunes only specific model components (Gaussian adapter, parameter heads) while freezing the backbone
- Groups input images sequentially and selects **1 target + N-1 context** per training iteration
- Optimizes both **target view rendering loss** and **context consistency loss**
- Supports flexible loss combinations: L1, MSE, and LPIPS
- Saves training intermediate renders for visualization

**Configuration** (`config/ttt.yaml`):
```yaml
# Model loading
pretrained_model_path: "pretrained_model"

# Input/Output
input_folder: "path/to/your/images"
output_folder: "exp-results/ttt-exp"

# Training parameters
iters: 5                    # Number of training epochs
group_size: 8               # Images per training group
lr: 0.0002                  # Learning rate
seed: 123                   # Random seed

# Loss configuration
loss_type: "l1_mse_lpips"   # Options: "l1_mse" | "lpips" | "l1_mse_lpips"
l1_weight: 1.0
mse_weight: 1.0
lpips_weight: 0.05
context_loss_weight: 1.0    # Weight for context reconstruction loss

# Trainable components
train_components:
  - "encoder.gaussian_adapter"
  - "encoder.gaussian_param_head"
```

**Output Structure:**
```
output_folder/
└── ttt_TIMESTAMP/
    ├── gaussians_ttt.ply              # Refined 3D Gaussians
    └── train_renders/                 # Training visualizations
        └── epoch_XXXX/
            └── group_XXXX/
                ├── ctx_000.jpg        # Context view renders
                ├── ctx_001.jpg
                └── tgt_000.jpg        # Target view renders
```

### 3. Iterative Test-time Refinement (ITR)

**File**: `itr.py`

Two-pass refinement strategy with render-based consistency constraints:

```bash
python itr.py
```

**Key Features:**
- **Round 1**: Reconstruct from context images only, interpolate poses to generate virtual target views
- **Round 2**: Reconstruct from context + Round-1 rendered targets, with consistency loss between rounds
- **Gradient Control**: Round-1 outputs are detached before computing consistency loss (prevents gradient backprop through Round-1)
- **LPIPS Restriction**: LPIPS loss only applied to context-GT anchoring, NOT to consistency losses

**Configuration** (`config/itr.yaml`):
```yaml
# Model loading
pretrained_model_path: "pretrained_model"

# Input/Output
input_folder: "path/to/your/images"
output_folder: "exp-results/itr-exp"

# Training parameters
iters: 5
group_size: 8
interp_frames: 2              # Virtual views interpolated between context pairs
lr: 0.0002
seed: 123

# Loss configuration
loss_type: "l1_mse_lpips"
l1_weight: 1.0
mse_weight: 1.0
lpips_weight: 0.05

# ITR-specific loss weights
context_gt_weight: 1.0                # Context vs GT loss weight
context_consistency_weight: 0.1       # Round1-Round2 context consistency weight
target_consistency_weight: 0.05       # Round1-Round2 target consistency weight

# Trainable components
train_components:
  - "encoder.gaussian_adapter"
  - "encoder.gaussian_param_head"
```

**Output Structure:**
```
output_folder/
└── itr_TIMESTAMP/
    ├── gaussians_itr.ply              # Final refined 3D Gaussians
    └── train_renders/                 # Both round renders saved
        └── epoch_XXXX/
            └── group_XXXX/
                ├── ctx1_000.jpg       # Round 1 context renders
                ├── tgt1_000.jpg       # Round 1 target renders
                ├── ctx2_000.jpg       # Round 2 context renders
                └── tgt2_000.jpg       # Round 2 target renders
```

**ITR Algorithm Overview:**
```
For each training group:
  1. Round 1: Encode context → Render context & interpolated targets
  2. Detach Round-1 renders (stop gradient)
  3. Round 2: Encode (context + Round-1 targets) → Render both
  4. Compute losses:
     - Context-GT loss (with LPIPS)
     - Context consistency: L1/MSE between Round1-ctx and Round2-ctx (NO LPIPS)
     - Target consistency: L1/MSE between Round1-tgt and Round2-tgt (NO LPIPS)
  5. Backprop only through Round-2 outputs
```

### 4. Novel View Synthesis Comparison

**File**: `nvs_compare.py`

Comprehensive evaluation tool that compares three methods: **Feed-Forward**, **TTT**, and **ITR** on the same scenes.

```bash
python nvs_compare.py --wandb-name "nvs-comparison-experiment"
```

**Key Features:**
- Evaluates multiple scenes from VR-NeRF dataset automatically
- Tests three methods **independently** with **fresh model instances** per scene (prevents parameter contamination)
- Computes **PSNR**, **SSIM**, **LPIPS** metrics for each method
- Logs results to **Weights & Biases** with per-scene and aggregate summaries
- Saves rendered images and training intermediate outputs to disk

**Configuration** (`config/nvs_compare.yaml`):
```yaml
experiment:
  pretrained_model_path: "pretrained_model"
  num_context: 64           # Number of context views for reconstruction
  camera_id: "20"           # VR-NeRF camera block ID
  metrics_dir: "exp-results/ttt-exp/nvs_compare"
  image_root_dir: "exp-results/ttt-exp"

dense_sparse:
  pool_size: 72             # Total available views in dense sampling pool
  pool_stride: 3            # Stride for dense sampling
  seed: 123                 # Random seed for sampling
  images_subdir: "images-jpeg-1k"  # Image subdirectory in VR-NeRF dataset

# Dataset paths (inherits from vrnerf_sampler.yaml)
dataset_root: "datasets-raw/vrnerf"
scenes: "datasets-raw/vrnerf/scenes_used.json"
```

**Output Structure:**
```
exp-results/
└── ttt-exp/
    └── vrnerf_TIMESTAMP/
        └── scene_name/
            ├── input_images/           # Input context images used for all methods
            │   ├── 000000.jpg
            │   ├── 000001.jpg
            │   └── ...
            │
            ├── output_ff/              # Feed-forward results
            │   ├── rendered_000.jpg    # Rendered novel views
            │   ├── rendered_001.jpg
            │   └── ...
            │
            ├── output_ttt/             # TTT results
            │   ├── rendered_000.jpg
            │   ├── ...
            │   └── ttt_outputs/
            │       └── ttt_TIMESTAMP/
            │           ├── gaussians_ttt.ply
            │           └── train_renders/  # TTT training visualizations
            │
            └── output_itr/             # ITR results
                ├── rendered_000.jpg
                ├── ...
                └── itr_outputs/
                    └── itr_TIMESTAMP/
                        ├── gaussians_itr.ply
                        └── train_renders/  # ITR training visualizations (both rounds)
```

**Weights & Biases Logging:**
- Per-scene metrics: `metrics/{method}/{psnr,ssim,lpips}`
- Aggregate summaries: `summary/{method}/{psnr,ssim,lpips}`
- Training configuration logged automatically

**Command-line Options:**
```bash
# Custom W&B run name
python nvs_compare.py --wandb-name "exp-64ctx-ttt-itr"

# Or set via environment variable
export WANDB_RUN_NAME="my-experiment"
python nvs_compare.py
```

### 5. Self-Supervised Refinement

**File**: `self_supervise.py`

Self-supervised training with pose interpolation and virtual view synthesis:

```bash
python self_supervise.py
```

Configuration in `config/self_supervise.yaml`. This method uses pose interpolation to generate additional training views for model refinement.

### 6. Interactive Gradio Demo

**File**: `demo_gradio.py`

Launch an interactive web interface for 3D reconstruction:

```bash
python demo_gradio.py
```

The demo automatically downloads the pretrained model (if not available locally) and starts a Gradio web server. You can:
- Upload multiple images or a video
- Visualize the reconstructed 3D Gaussian Splat
- View rendered RGB and depth videos
- Export PLY files for external viewers

![demo_gradio](assets/demo_gradio.gif)

## Training (Advanced)

Train AnySplat from scratch on your own datasets:

```bash
# Single node:
python src/main.py +experiment=dl3dv trainer.num_nodes=1

# Multi-node distributed training:
export GPU_NUM=8
export NUM_NODES=2
torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$GPU_NUM \
  --rdzv_id=test \
  --rdzv_backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  -m src.main +experiment=multi-dataset +hydra.job.config.store_config=false
```

We provide example configurations for three datasets: [CO3Dv2](https://github.com/facebookresearch/co3d), [DL3DV](https://dl3dv-10k.github.io/DL3DV-10K/), and [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/), each representing different training view sampling strategies.

## Post Optimization

Apply per-scene optimization for higher quality:

```bash
python src/post_opt/simple_trainer.py default --data_dir /path/to/scene
```

## Evaluation

Standard evaluation pipelines:

```bash
# Novel View Synthesis evaluation
python src/eval_nvs.py --data_dir /path/to/test/data

# Pose Estimation evaluation
python src/eval_pose.py --co3d_dir /path/to/co3d --co3d_anno_dir /path/to/annotations
```

## Dataset Preprocessing

We use the original data from DL3DV datasets. For other datasets (CO3D, ScanNet++, etc.), follow [CUT3R's data preprocessing instructions](https://github.com/naver/dust3r/tree/main?tab=readme-ov-file#datasets).

## Project Structure

```
AnySplat/
├── src/                          # Core implementation
│   ├── model/                    # Model architecture
│   ├── dataset/                  # Dataset loaders
│   ├── utils/                    # Utilities
│   │   └── model_loading.py     # Smart model loading with fallback
│   ├── evaluation/               # Metrics
│   └── ...
│
├── config/                       # YAML configurations
│   ├── ttt.yaml                  # TTT configuration
│   ├── itr.yaml                  # ITR configuration
│   ├── nvs_compare.yaml          # NVS comparison configuration
│   └── ...
│
├── ttt.py                        # Test-Time Training script
├── itr.py                        # Iterative Test-time Refinement script
├── nvs_compare.py                # NVS comparison evaluation script
├── inference.py                  # Basic feed-forward inference
├── self_supervise.py             # Self-supervised training
├── demo_gradio.py                # Interactive web demo
│
└── scripts/                      # Helper scripts
    ├── vrnerf_sampler.py         # VR-NeRF dataset sampling
    └── vrnerf_download.py        # VR-NeRF dataset download
```

## Key Improvements in This Fork

### 1. Intelligent Model Loading
- **File**: `src/utils/model_loading.py`
- **Feature**: Automatically tries local path first, falls back to HuggingFace Hub
- **Benefit**: Works seamlessly in both online and offline environments

### 2. Test-Time Training (TTT)
- **File**: `ttt.py`
- **Feature**: Fine-tune model parameters at test time using input images
- **Benefit**: Improved reconstruction quality for challenging scenes

### 3. Iterative Test-time Refinement (ITR)
- **File**: `itr.py`
- **Feature**: Two-pass refinement with consistency constraints
- **Benefit**: Better handling of view interpolation and temporal consistency

### 4. Comprehensive NVS Comparison
- **File**: `nvs_compare.py`
- **Feature**: Side-by-side evaluation of Feed-Forward, TTT, and ITR
- **Benefit**: Quantitative analysis with PSNR/SSIM/LPIPS metrics

### 5. Configurable via YAML
- All methods configurable through YAML files
- Easy to adjust hyperparameters without code changes
- Reproducible experiments with version-controlled configs

## Citation

If you find our work helpful, please consider citing:

```bibtex
@article{jiang2025anysplat,
  title={Anysplat: Feed-forward 3d gaussian splatting from unconstrained views},
  author={Jiang, Lihan and Mao, Yucheng and Xu, Linning and Lu, Tao and Ren, Kerui and Jin, Yichen and Xu, Xudong and Yu, Mulin and Pang, Jiangmiao and Zhao, Feng and others},
  journal={ACM Transactions on Graphics (TOG)},
  volume={44},
  number={6},
  pages={1--16},
  year={2025},
  publisher={ACM New York, NY, USA}
}
```

## Acknowledgement

We thank all authors behind these repositories for their excellent work: [VGGT](https://github.com/facebookresearch/vggt), [NoPoSplat](https://github.com/cvg/NoPoSplat), [CUT3R](https://github.com/CUT3R/CUT3R/tree/main) and [gsplat](https://github.com/nerfstudio-project/gsplat).
