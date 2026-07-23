# Literature Survey Report — Gaussian Editing for Scene Repair

> **Date**: 2026-07-10
> **Context**: Research on 3D Gaussian Splatting (3DGS) completion/inpainting methods to guide the design of a GaussianRepairNetwork with add/delete/move capabilities

---

## 1. Introduction

This survey covers seven key papers published in 2025 that address 3D Gaussian editing, inpainting, and density control. The goal is to identify transferable techniques for designing a **Gaussian Repair Network** that can fix imperfect Gaussian fields from feed-forward reconstruction methods (e.g., AnySplat), with capabilities for **adding, deleting, and moving** Gaussian primitives.

---

## 2. Directly Requested Papers

### 2.1 InterGSEdit: Interactive 3D Gaussian Splatting Editing with 3D Geometry-Consistent Attention Prior

| | |
|---|---|
| **Venue** | ICCV 2025 |
| **Authors** | Wen et al. (Nanjing Univ. of Aeronautics & Astronautics) |
| **Links** | [arXiv:2507.04961](https://arxiv.org/abs/2507.04961) |

**Core method:** InterGSEdit addresses multi-view inconsistency in 3DGS editing by constructing a **3D Geometry-Consistent Attention Prior (GAP³ᴰ)**. The pipeline is:

1. **CLIP-based Semantic Consistency Selection (CSCS)**: For a user-selected key view, selects semantically consistent reference views using CLIP embeddings.
2. **GAP³ᴰ Construction**: Extracts cross-attention maps from the diffusion model's denoising process for each reference view. These 2D maps are **lifted into 3D** via weighted Gaussian Splatting unprojection — splatting the attention values back onto 3D Gaussians based on their positions and known camera poses.
3. **Attention Fusion Network (AFN)**: Fuses 3D-constrained attention (from projecting GAP³ᴰ back to 2D) with 2D cross-attention maps using a learned gating mechanism. Early inference prioritizes 3D-constrained attention for geometric consistency; later stages shift to 2D attention for fine details.

**Relevance to our design:**
- **2D→3D Lifting**: The weighted Gaussian unprojection of 2D attention maps provides a template for lifting 2D deficiency signals to 3D Gaussian positions — directly applicable to our addition mechanism.
- **Multi-view Feature Aggregation**: The CSCS approach shows that CLIP-based semantic selection helps identify which views provide the most relevant information for a given 3D region.
- **Stage-wise Fusion**: The AFN's adaptive gating suggests that early geometric consistency followed by fine detail is a viable training strategy.

**Limitation:** Per-scene optimization, not feed-forward. No explicit Gaussian add/delete mechanisms.

---

### 2.2 EditSplat: Multi-View Fusion and Attention-Guided Optimization for View-Consistent 3D Scene Editing with 3D Gaussian Splatting

| | |
|---|---|
| **Venue** | CVPR 2025 |
| **Authors** | Lee et al. (Korea Univ., Yonsei Univ.) |
| **Links** | [arXiv:2412.11520](https://arxiv.org/abs/2412.11520) |

**Core method:** EditSplat is a text-driven 3D scene editing framework with two key contributions:

1. **Multi-View Fusion Guidance (MFG)**: Projects edited multi-view images onto a target view using 3DGS depth maps, then blends based on depth values for multi-view consistency.
2. **Attention-Guided Trimming (AGT)**: Extracts cross-attention maps from the diffusion model for the target text concept (e.g., "clown"). Projects these 2D attention maps onto 3D Gaussians, and **prunes Gaussians with high attention weight before editing**. This removes Gaussians that retain excessive source information in semantically meaningful regions, enabling more efficient optimization.

**Relevance to our design:**
- **Gaussian Deletion via Attention**: AGT is the most directly relevant mechanism for **deleting Gaussians**. It demonstrates that attention-guided selection of which Gaussians to remove is effective.
- **Selective Optimization**: After pruning, only the remaining high-attention Gaussians are optimized, enabling precise local edits.
- **Key insight**: Pre-editing pruning removes "source information contamination" — in our repair context, this corresponds to removing artifact-causing Gaussians before generating replacements.

**Limitation:** Requires a diffusion model for attention maps. Our feed-forward approach uses video latent features instead.

---

## 3. Inpainting & Completion Papers

### 3.1 RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors

| | |
|---|---|
| **Venue** | ICCV 2025 |
| **Authors** | Paliwal et al. (Texas A&M, Meta Reality Labs, MPI) |
| **Links** | [arXiv:2503.10860](https://arxiv.org/abs/2503.10860) |

**Core method:** RI3D separates view synthesis into **two tasks** with dedicated diffusion models:

1. **Repair Model**: Takes a rendered image as input, predicts a high-quality "pseudo GT" image → focuses on reconstructing **visible regions**.
2. **Inpainting Model**: Hallucinates details in **unobserved/missing areas**.
3. **Two-stage optimization**: Stage 1 uses only the repair model (unseen regions remain empty). Stage 2 activates the inpainting model to fill missing regions.
4. **Novel Gaussian Initialization**: Combines DUSt3R 3D-consistent depth with detailed monocular depth via bilateral filtering, producing high-quality depth maps for initializing Gaussians.

**Relevance to our design:**
- **Two-stage philosophy**: Stage 1 (repair existing) + Stage 2 (generate new) maps perfectly to our per-Gaussian analysis head + deficiency analyzer design.
- **Hybrid Depth Initialization**: Provides a strategy for estimating where new Gaussians should be placed (in 3D space) from 2D observations.
- **Task Separation**: Shows that repairing and inpainting are fundamentally different tasks that benefit from specialized sub-modules.

**Key technique transfer:** We adopt the two-stage approach: our PerGaussianAnalysisHead handles "repair" (Δ_params + Δ_means + keep_prob), while the SceneDeficiencyAnalyzer handles "inpaint" (new Gaussians).

---

### 3.2 ShareGS: Hole Completion with Sparse Inputs Based on Reusing Selected Scene Information

| | |
|---|---|
| **Venue** | Pattern Recognition (2025) |
| **Authors** | Hong, Tao, Gong |
| **Links** | [DOI:10.1016/j.patcog.2025.111729](https://doi.org/10.1016/j.patcog.2025.111729) |

**Core method:** ShareGS addresses scene holes caused by limited perspective coverage:

1. **Feature and Scale-Guided Gaussian Homogenization**: Spreads Gaussians from dense coverage areas into gaps, increasing Gaussian density in hole regions.
2. **Feature-Consistent Hole Selection & Projection Transformation**: Generates pseudo-views near input viewpoints. A projection transformation strategy allows reference views to supervise regions containing holes.
3. **Unsupervised Depth-Color Consistency Regularization**: Ensures continuity at hole boundaries.

**Relevance to our design:**
- **Gaussian Homogenization** is a direct "addition" mechanism: it adds Gaussians to fill holes by redistributing from dense regions.
- **Pseudo-view generation** for supervising hole completion suggests that our deficiency analyzer can learn from rendered views even without GT for those views.
- The feature-scale guided selection of transition regions provides a method for locating WHERE new Gaussians are needed.

---

### 3.3 AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360° Unbounded Scene Inpainting

| | |
|---|---|
| **Venue** | CVPR 2025 |
| **Authors** | Wu et al. (NVIDIA, NCTU) |
| **Links** | [arXiv:2502.05176](https://arxiv.org/abs/2502.05176) |

**Core method:** Object removal and hole filling in unbounded 3D scenes:

1. **Depth-Aware Unseen Mask Generation**: Uses depth warping across multiple views to identify truly occluded regions.
2. **Adaptive Guided Depth Diffusion (AGDD)**: Zero-shot depth alignment that aligns monocular depth with existing scene geometry.
3. **SDEdit-Based Detail Enhancement**: Preserves structure from reference view while enhancing fine details.

**Relevance to our design:**
- **Depth-Aware Mask Generation** provides a blueprint for our deficiency map: low accumulated alpha + high rendering error = need new Gaussians.
- The AGDD approach suggests that depth can be corrected/refined from video latent features (we use a depth_offset head in our new Gaussian params).

---

## 4. Adaptive Density Control (Primitive Add/Delete)

### 4.1 Original 3DGS Adaptive Density Control (Kerbl et al., 2023)

**Core mechanism:**
- **Clone**: Under-reconstructed regions (small Gaussians with large view-space positional gradients) → clone a copy in the direction of the gradient.
- **Split**: Over-reconstructed regions (large Gaussians with large gradients) → split into two smaller ones.
- **Prune**: Remove Gaussians with very low opacity (α < ε) after periodic opacity reset.

**Relevance:** The canonical framework for Gaussian density management. Our keep_prob mechanism is a learnable, differentiable version of opacity-based pruning.

### 4.2 EGU-GS: Efficient Gaussian Utilization (Zheng et al., 2025)

| | |
|---|---|
| **Venue** | Image and Vision Computing |
| **Links** | [DOI:10.1016/j.imavis.2025.105687](https://doi.org/10.1016/j.imavis.2025.105687) |

**Key improvements:**
- **Cross-Section-Oriented Splitting**: Preserves shape information, reduces overlap.
- **Heterogeneous Cloning**: Uses probability sampling instead of simple attribute replication.
- **Opacity Adaptive Pruning**: Removes low-opacity Gaussians with adaptive thresholds.
- **Gaussian Importance Weights**: Refines selection for densification.
- **Result**: 42% reduction in Gaussian count with improved PSNR.

**Relevance:** The importance-weighted selection and adaptive pruning strategies informed our keep_prob head design — specifically, making deletion a learned, continuous decision rather than a hard threshold.

### 4.3 Metropolis-Hastings Sampling for 3D Gaussian Reconstruction (Kim et al., NeurIPS 2025)

**Key idea:** Replaces heuristic clone/split/prune with a **probabilistic sampling framework** using multi-view photometric error + opacity scores, with Bayesian acceptance tests.

**Relevance:** Demonstrates that learned/learnable density control outperforms hard-coded heuristics — supporting our approach of making keep_prob a learned output of the repair network.

---

## 5. Synthesis: Towards a Gaussian Repair Network

### 5.1 Design Principles Extracted

| Principle | Source Papers | Application in Our Design |
|---|---|---|
| **Two-stage (repair + inpaint)** | RI3D | PerGaussianHead (repair) + SceneDeficiencyAnalyzer (inpaint) |
| **Attention-guided pruning** | EditSplat (AGT) | keep_prob head (learned deletion) |
| **2D→3D lifting of signals** | InterGSEdit (GAP³ᴰ) | Deficiency map → 3D Gaussian generation via back-projection |
| **Depth-aware hole detection** | AuraFusion360 | Deficiency maps from rendered alpha + error maps |
| **Gaussian homogenization** | ShareGS | New Gaussian generation in deficient regions |
| **Learned density control** | EGU-GS, MH-GS | Differentiable keep_prob instead of hard pruning |
| **Gradient-based position updates** | Original 3DGS | Δ_means head with 0.01 scaling |

### 5.2 Key Differences from Existing Methods

Our approach differs from all surveyed papers in a critical way: **it is feed-forward, not per-scene optimized**. All of the editing papers (InterGSEdit, EditSplat, RI3D, ShareGS) optimize per scene, requiring minutes of compute per instance. Our repair network learns a **single forward pass** from video latent features directly to repair decisions, enabling real-time application at inference time.

### 5.3 References

1. Wen et al., *InterGSEdit: Interactive 3D Gaussian Splatting Editing with 3D Geometry-Consistent Attention Prior*, ICCV 2025. [arXiv:2507.04961](https://arxiv.org/abs/2507.04961)
2. Lee et al., *EditSplat: Multi-View Fusion and Attention-Guided Optimization for View-Consistent 3D Scene Editing with 3D Gaussian Splatting*, CVPR 2025. [arXiv:2412.11520](https://arxiv.org/abs/2412.11520)
3. Paliwal et al., *RI3D: Few-Shot Gaussian Splatting With Repair and Inpainting Diffusion Priors*, ICCV 2025. [arXiv:2503.10860](https://arxiv.org/abs/2503.10860)
4. Hong et al., *ShareGS: Hole Completion with Sparse Inputs Based on Reusing Selected Scene Information*, Pattern Recognition 2025. [DOI:10.1016/j.patcog.2025.111729](https://doi.org/10.1016/j.patcog.2025.111729)
5. Wu et al., *AuraFusion360: Augmented Unseen Region Alignment for Reference-based 360° Unbounded Scene Inpainting*, CVPR 2025. [arXiv:2502.05176](https://arxiv.org/abs/2502.05176)
6. Zheng et al., *EGU-GS: Efficient Gaussian Utilization for Real-time 3D Gaussian Splatting*, Image and Vision Computing 2025. [DOI:10.1016/j.imavis.2025.105687](https://doi.org/10.1016/j.imavis.2025.105687)
7. Kerbl et al., *3D Gaussian Splatting for Real-Time Radiance Field Rendering*, ACM TOG 2023. [arXiv:2308.04079](https://arxiv.org/abs/2308.04079)
