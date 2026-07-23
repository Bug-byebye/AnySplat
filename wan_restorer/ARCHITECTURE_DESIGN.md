# Modular Gaussian Scene Restoration Framework — Architecture Design

## Table of Contents

1. [Video Latent Acquisition](#1-video-latent-acquisition)
2. [Gaussian Feature Construction](#2-gaussian-feature-construction)
3. [Fusion Strategies](#3-fusion-strategies)
4. [Gaussian Refiner Designs](#4-gaussian-refiner-designs)
5. [Loss Functions](#5-loss-functions)
6. [Integration Roadmap](#6-integration-roadmap)

---

## 1. Video Latent Acquisition

### 1.1 Wan2.2 DiT Layer Analysis

The Wan2.2 VAE encoder produces a 48-channel latent at 1/16 spatial resolution. The DiT transformer (text-conditioned 3D) then processes this latent through ~40 blocks. Different layers encode different levels of visual structure:

| Layer Group | Typical Indices | Content | Resolution Level | Use Case |
|---|---|---|---|---|
| Early | 0--10 | Local structure, edges, textures, low-level patterns | Per-pixel alignment | Best for fine-grained Gaussian position/scale refinement |
| Mid | 10--25 | Semantics, object parts, appearance composition | Object-level | Best for appearance (SH) and rotation refinement |
| Late | 25+ | Global context, scene layout, long-range dependencies | Scene-level | Useful for occlusion reasoning and global consistency |

**Current implementation** (in `wan_feature_extractor.py`): extracts layers `{4, 8}` from an 8-layer subset, averages them. This is a reasonable starting point but can be extended.

### 1.2 Multi-Scale Feature Pyramid from DiT

**Option A: Simple averaging (current)**
```
features = sum(layer_outputs[i] for i in indices) / len(indices)
--> project --> feature map
```
- Pros: Simple, memory-efficient, works well when layers are complementary.
- Cons: May lose scale-specific information; early and late features are mixed indiscriminately.

**Option B: Concatenation + channel reduction**
```
features = concat(layer_outputs[early], layer_outputs[mid], layer_outputs[late])
--> Conv1x1 (channel reduction) --> feature map
```
- Pros: Preserves per-scale information; each scale can specialize.
- Cons: Higher channel dimension in the middle.

**Option C: Feature Pyramid Network (FPN) style**
```
Late features --> upsample --> fuse with mid features --> upsample --> fuse with early features
```
- Pros: Standard approach in detection/segmentation; proven to preserve both semantic and spatial accuracy.
- Cons: Higher complexity; the FPN neck needs to be trained (not frozen).

**Option D: Independent per-scale heads + learned fusion**
```
Early --> Projector_E --> [scale-specific features]
Mid   --> Projector_M --> [scale-specific features]
Late  --> Projector_L --> [scale-specific features]
Fusion: learned weights per pixel or per Gaussian
```
- Pros: Maximum flexibility; each scale head can specialize.
- Cons: Parameter-heavy; risk of overfitting with limited training data.

**Recommendation:** Start with Option A (current) for simplicity, then migrate to Option B with 3 groups (early: 0--8, mid: 9--16, late: 17+).

### 1.3 VAE-Only (No DiT) Sufficiency

| Criterion | VAE Only | VAE + DiT (current) |
|---|---|---|
| Feature quality | 48-channel latent, spatially localized | Same latent + enriched by self-attention across spatial positions |
| Contextual awareness | None (purely local encoding) | Global via self-attention |
| Memory | ~2 GB | ~8--12 GB (DiT is the dominant cost) |
| Speed | ~10 ms/view | ~150--300 ms/view |
| Quality ceiling | Limited (no high-level semantics) | High (rich semantic + detailed) |

**Verdict:** VAE-only is NOT sufficient for scenes requiring semantic understanding (e.g., filling in occluded regions, recognizing object categories). However, for simple refinement tasks where local texture matching is enough, VAE-only may be a viable fast option. Consider a **configurable depth** parameter (`num_dit_layers: 0` = VAE-only, `8` = lightweight, `40` = full).

### 1.4 Alternatives to Wan2.2

If Wan2.2 is unsuitable (license, model size, inference speed):

| Alternative | Pro | Con |
|---|---|---|
| **Stable Diffusion VAE + UNet** | Widely available; 4-channel latent; lightweight | 2D-only; no temporal modeling |
| **DINOv2** (ViT features) | Excellent semantic features; efficient (400M params) | Single-scale; no video prior |
| **CLIP** (visual encoder) | Strong semantic alignment with text | Low spatial resolution; coarse |
| **SigLIP / SigLIP2** | Better spatial detail than CLIP | Still not pixel-level accurate |
| **Depth-Anything / DPT** (already present) | Good geometry; lightweight | Limited appearance semantics |
| **Custom VAE + Small ViT** | Fully controllable; efficient | Needs training from scratch |

**Recommendation:** Keep Wan2.2 as the primary extractor. Add a `feature_backbone` config option to switch between `"wan"`, `"dino"`, `"sd"`, and `"none"` (DPT-only fallback).

---

## 2. Gaussian Feature Construction

### 2.1 Rasterized GP-Buffer Features

Currently the refiner receives a dense [B*V, 83, H, W] tensor of raw Gaussian parameters. To enable richer per-pixel features for the refiner, we can rasterize Gaussian properties into GP-buffer-style feature maps before refinement.

**Feature channels that can be rasterized:**

| Channel | Source | Semantic Value |
|---|---|---|
| Position (depth) | `pts_all` after unprojection | 3D geometry cue |
| Scale (log) | Raw scale parameters (channels 1--3) | Surface roughness / detail size |
| Opacity (logit) | Channel 0 | Visibility / coverage |
| SH coefficients | Channels 4--78 | Appearance / color |
| Features | AnySplat neural features (if available) | Learned per-Gaussian descriptor |

**Current GP-buffer construction** (can be added as a preprocessing step):

```
raw_gs_params = [B*V, 83, H, W]
       |
       v
GP_buffer = {
    'opacity':    raw_gs_params[:, 0:1],        # [B*V, 1, H, W]
    'scale':      raw_gs_params[:, 1:4],         # [B*V, 3, H, W]
    'rotation':   raw_gs_params[:, 4:8],         # [B*V, 4, H, W]
    'sh_coeffs':  raw_gs_params[:, 8:83],        # [B*V, 75, H, W]
}
```

These can be normalized (e.g., layer norm per channel group) and concatenated to the feature input of the refiner.

### 2.2 Neighbor Aggregation (KNN or Attention)

Current Gaussian parameters are per-pixel dense maps. However, Gaussians have relationships with neighbors that the per-pixel refinement cannot capture.

**Option A: Dilated convolution (lightweight)**
```
Conv3x3, dilation=2 --> captures 5x5 receptive field
```
- Pros: Simple, fast, built into any Conv2d.
- Cons: Fixed grid; does not adapt to scene content.

**Option B: KNN in pixel space (medium)**
```
For each pixel (i,j):
    features = gather neighbors in a 3x3 or 5x5 region
    aggregate via learned weights (small MLP)
```
- Pros: Captures local structure.
- Cons: Grid-based, not Gaussian-position-aware.

**Option C: KNN in 3D (expensive but principled)**
```
For each pixel (i,j):
    unproject to 3D position
    find K nearest Gaussians in 3D space
    aggregate their features via attention or MLP
```
- Pros: True 3D neighbor relations.
- Cons: Very expensive; requires per-pixel 3D KNN search (not CUDA-friendly in dense form).

**Option D: Per-pixel local self-attention through the refinement U-Net**
```
Already implicit in the U-Net's downsampling + upsampling:
    - Encoder downsampling expands receptive field
    - Skip connections preserve local detail
    - The U-Net already captures multi-scale context
```
- Pros: Already happening in the existing `ResidualUNet` (receptive field grows with each downsampling level).
- Cons: Not explicit; limited by U-Net depth.

**Recommendation:** The existing U-Net in `GaussianSceneRestorerV3` already provides multi-scale context through its encoder-decoder structure. Add explicit GP-buffer features as input channels, but do not add explicit KNN aggregation unless ablation studies show the U-Net's implicit context is insufficient.

### 2.3 Feature Encoding Module (New)

Propose a configurable `GaussianFeatureEncoder` module:

```
GaussianFeatureEncoder:
  - Input:  raw_gs_params [B*V, 83, H, W]
  - Options:
      'none':          identity (pass through)
      'conv_embed':     Conv1x1 --> LN --> ReLU  (lightweight embedding)
      'gp_buffer':      Split into semantic groups, norm each group
      'gp_buffer_conv': gp_buffer + depthwise separable conv per group
  - Output: [B*V, gs_feat_dim, H, W]
```

---

## 3. Fusion Strategies

Fusion refers to combining video latent features (from Wan/DINO etc.) with Gaussian features (from raw params or GP-buffer) before the refinement head.

### 3.1 Cross-Attention (Highest Priority)

**Design:**
```
Video latent features: [B*V, wan_feat_dim, H, W]  --> Q projection
Gaussian features:     [B*V, gs_feat_dim, H, W]    --> K, V projection

For each pixel position (i,j):
    Q_i = W_q * video_feat_i
    K_i = W_k * gs_feat_i
    V_i = W_v * gs_feat_i
    
    # Self-contained per-position cross-attention (not spatial)
    attn = softmax(Q_i @ K_i^T / sqrt(d))
    out  = attn @ V_i
    
    # Or spatial cross-attention (attend across positions)
    Q flat: [B*V, H*W, d]
    K flat: [B*V, H*W, d]
    V flat: [B*V, H*W, d]
    attn = softmax(Q @ K^T / sqrt(d))  # [B*V, H*W, H*W] -- very large!
    out  = attn @ V
```

**Spatial cross-attention is memory-prohibitive** for high-resolution images (e.g., H*W = 65536 at 256x256). Use **windowed cross-attention**:

```
Window size W_w x W_h (e.g., 16x16):
    Q: [B*V, num_windows, W_w*W_h, d]
    K: [B*V, num_windows, W_w*W_h, d]
    V: [B*V, num_windows, W_w*W_h, d]
    attn = softmax(Q @ K^T / sqrt(d))  # [B*V, num_windows, 256, 256]
```

| Variant | Memory | Quality |
|---|---|---|
| Per-position (channel-only) | Low | Poor (no spatial mixing) |
| Full spatial | Prohibitive | Best |
| Windowed (16x16) | Tractable | Good |
| Windowed (8x8) | Cheap | Moderate |

**Implementation Plan:**

```
class CrossAttentionFusion(nn.Module):
    def __init__(self, wan_dim, gs_dim, hidden_dim, window_size=16):
        self.q_proj = nn.Linear(wan_dim, hidden_dim)
        self.k_proj = nn.Linear(gs_dim, hidden_dim)
        self.v_proj = nn.Linear(gs_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.window_size = window_size
    
    def forward(self, video_feats, gs_feats):
        # Both [B*V, C, H, W]
        # Apply windowed cross-attention
        ...
        return fused_feats  # [B*V, hidden_dim, H, W]
```

**Pros:**
- Video features act as queries, allowing flexible selection of relevant Gaussian information.
- The attention mechanism can learn which video features matter for which Gaussian parameter.
- More expressive than concatenation.

**Cons:**
- Higher memory and compute cost than concat.
- Needs careful windowing to be tractable.
- Requires more training data to learn attention patterns.

### 3.2 Concat + Conv (Baseline, Already Implemented)

**Current implementation** (in `GaussianSceneRestorerV3`):
```
concat_feats = cat([wan_feats, dpt_feats, input_images], dim=1)
--> Conv3x3 + BN + ReLU --> hidden_dim
--> U-Net refinement
```

**Pros:**
- Simple, proven baseline.
- No additional parameters beyond the projection conv.
- Fast inference.

**Cons:**
- No learned interaction between modalities (just concatenation).
- Each modality competes for channels in the subsequent conv layers.
- Less expressive than attention-based methods.

### 3.3 Transformer Decoder with Learnable Queries

**Design:**
```
Learnable queries (Q):  [num_queries, hidden_dim]
    |
    v
TransformerDecoder(video_feats, gs_feats):
    Layer 0: Self-Attn(Q) --> Cross-Attn(Q, video_feats) --> FFN
    Layer 1: Self-Attn(Q) --> Cross-Attn(Q, gs_feats)   --> FFN
    Layer N: ...
    |
    v
Output: [num_queries, hidden_dim]
    |
    v
Upsample to [H, W] via learned position decoder
```

**Key differences from cross-attention:**
- Queries are learnable embeddings, not derived from video features.
- Queries can represent Gaussian parameter groups (e.g., one query per SH band).
- The decoder naturally performs iterative refinement across layers.

| Number of Queries | Use Case |
|---|---|
| 1 | Global scene descriptor |
| 16 | Per-region refinement (coarse) |
| 83 = num_gs_channels | Per-parameter refinement (fine-grained) |
| 256 | Dense refinement (near-pixel-level) |

**Pros:**
- Queries can specialize for different parameter types (opacity, scale, rotation, SH).
- Iterative refinement through decoder layers.
- No quadratic spatial attention (queries are fixed).

**Cons:**
- Additional decoder parameters (~10--50M for a 6-layer decoder).
- Query interpretation is not trivial (queries are learned, not supervised).
- Needs a decoder to project queries back to pixel grid.

### 3.4 Gated Fusion

**Design:**
```
video_feat_proj = Linear(video_feats)   # [B*V, hidden_dim, H, W]
gs_feat_proj    = Linear(gs_feats)       # [B*V, hidden_dim, H, W]

gate = sigmoid(Linear(cat([video_feats, gs_feats])))  # [B*V, hidden_dim, H, W]
output = gate * video_feat_proj + (1 - gate) * gs_feat_proj
```

**Variants:**

| Variant | Equation | Property |
|---|---|---|
| Simple gate | `g * a + (1-g) * b` | Per-channel, per-pixel soft selection |
| Bilinear gate | `g * a * b` | Multiplicative interaction |
| Gated with bias | `g * a + (1-g) * b + bias` | Includes additive bias term |

**Pros:**
- Learnable per-pixel, per-channel weighting between modalities.
- Parameter-efficient (only gate projection weights).
- Intuitive: the model learns when to trust video features vs. Gaussian features.

**Cons:**
- Still a weighted sum; no complex interaction (unlike attention).
- May collapse to always-pick-one modality if training data is limited.

### 3.5 Fusion Strategy Comparison

| Strategy | Params | Memory | Expressivity | Implementation Difficulty | Priority |
|---|---|---|---|---|---|
| Concat + Conv | Very low | Low | Low | Trivial | 2 (baseline) |
| Gated Fusion | Low | Low | Medium | Easy | 4 |
| Cross-Attention (windowed) | Medium | Medium-High | High | Medium | 1 |
| Transformer Decoder | High | Medium | Highest | Hard | 3 |

**Recommendation:**
1. Implement Cross-Attention (windowed 16x16) as the primary learned fusion.
2. Keep Concat + Conv as the always-available baseline.
3. Add Gated Fusion as a lightweight alternative.
4. Add Transformer Decoder as an experimental high-capacity option.

---

## 4. Gaussian Refiner Designs

### 4.1 Residual Prediction (Current v1/v3 Approach)

**Current architecture:**
```
Inputs --> [WAN feats, DPT feats, RGB] --> U-Net --> Conv1x1 --> Delta [B*V, 83, H, W]
Refined = Raw + Delta
Delta[:, 0] = clamp(Delta[:, 0], -7, 7)  # opacity logit clamp
```

**Properties:**
- Zero-initialized final layer (safe at step 0).
- Predicts additive residuals directly in parameter space.
- Opacity delta is clamped to prevent extreme values.

**Pros:**
- Simple, directly modifies Gaussian parameters.
- Zero-init guarantees safe integration.
- Easy to debug (inspect deltas to see what changes).

**Cons:**
- Parameter space is not Euclidean (scale is log, rotation is quaternion).
- Large deltas in one parameter can destabilize rendering.
- No notion of parameter importance (all 83 channels treated equally).

### 4.2 Feature Residual (Predict Offsets in Feature Space)

**Design:**
```
Inputs --> Encoder --> features [B*V, latent_dim, H, W]
                        |
                        v
Feature Delta Head --> delta_feats [B*V, latent_dim, H, W]
                        |
                        v
Refined features = raw_features + delta_feats
                        |
                        v
Parameter decoder --> refined_params [B*V, 83, H, W]
```

Instead of predicting `Delta` for the 83-dimensional parameter space, predict a delta in a lower-dimensional **feature space**, then decode to parameters.

**Pros:**
- Feature space is smoother and better conditioned than parameter space.
- Latent dimension can be smaller than 83 (e.g., 32 vs. 83).
- Allows multi-modal mapping (feature space can represent more than one parameter configuration).

**Cons:**
- Requires a decoder (``latent --> 83 params`), which needs training.
- Adds a bottleneck that may lose detail.
- The decoder may need careful initialization to preserve zero-init property.

**Implementation Sketch:**
```
class FeatureResidualRefiner(nn.Module):
    def __init__(self, latent_dim=32, gs_dim=83):
        self.feature_encoder = ...  # Process fusion features to latent_dim
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim*2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim*2, gs_dim, 1),
        )
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)
    
    def forward(self, fusion_feats, raw_params):
        # Predict feature offset
        features = self.feature_encoder(fusion_feats)
        # Decode directly (no residual in feature space for simplicity)
        delta = self.decoder(features)
        return raw_params + delta
```

### 4.3 Confidence-Based Masking + Refinement

**Design:**
```
Inputs --> Confidence Head --> confidence_map [B*V, 1, H, W] (sigmoid)
                               |
                               v
Inputs --> Refinement Head --> delta [B*V, 83, H, W]
                               |
                               v
Refined = Raw + confidence_map * delta
```

The model predicts both a per-pixel confidence (0--1) and a refinement delta. The confidence acts as a soft mask: low confidence means "don't change this pixel."

**Extension: Two-headed architecture:**
```
Fusion features
    ├──> Confidence Head (1-channel, sigmoid)
    └──> Refinement Head (83-channel, zero-init)
    
Refined = Raw + confidence * delta
```

**Pros:**
- The model learns which pixels need refinement (e.g., occluded regions, low-quality Gaussians).
- Naturally sparse: most well-rendered pixels get low delta.
- The confidence map acts as an interpretable attention mask.

**Cons:**
- The confidence head may collapse to always-0 or always-1.
- Need to balance confidence and refinement training.
- Adds one extra output head.

**Mitigation:** Initialize the confidence head bias to `log(0.1 / 0.9)` so initial confidence is ~0.1 (cautious). Train with a small L1 penalty on confidence to encourage sparsity.

### 4.4 Multi-Stage Coarse-to-Fine

**Design:**
```
Stage 1 (Coarse): 
    Low-res features --> predict low-res delta --> upsample
Stage 2 (Medium):
    Upsampled + mid-res features --> predict mid-res delta --> upsample
Stage 3 (Fine):
    Upsampled + high-res features --> predict final delta
    
Params flow through stages (iterative refinement):
    raw --> Coarse --> Medium --> Fine --> refined
```

**Resolution progression example:**
- Stage 1: run at 1/4 resolution (64x64 at 256x256 input).
- Stage 2: run at 1/2 resolution (128x128).
- Stage 3: run at full resolution (256x256).

**Pros:**
- Computationally efficient (coarse stages are cheap).
- Naturally handles multi-scale refinement (large structures first, details later).
- Each stage can specialize: global appearance in coarse, local detail in fine.

**Cons:**
- Complex; harder to train end-to-end.
- Needs careful design of inter-stage connections.
- May overfit with limited data.

**Implementation Sketch:**
```
class CoarseToFineRefiner(nn.Module):
    def __init__(self, num_stages=3):
        self.stages = nn.ModuleList([
            RefinementStage(...) for _ in range(num_stages)
        ])
    
    def forward(self, fusion_feats_pyramid, raw_params):
        # fusion_feats_pyramid: list of [B*V, C, H/s_i, W/s_i]
        current = raw_params
        for stage, feats in zip(self.stages, fusion_feats_pyramid):
            delta = stage(feats, current)
            current = F.interpolate(current + delta, scale_factor=2, mode='bilinear')
        return current
```

### 4.5 Refiner Design Comparison

| Design | Complexity | Parameter Efficiency | Quality Ceiling | Training Difficulty | Suitability |
|---|---|---|---|---|---|
| Residual (v1/v3) | Low | High | Medium | Low | Good baseline, always available |
| Feature Residual | Medium | Medium | High | Medium | Best when feature space is well-structured |
| Confidence + Refine | Low-Medium | High | Medium-High | Medium | Good for selective refinement |
| Coarse-to-Fine | High | Medium | High | High | Best for large scenes / high-res |

**Recommendation:**
1. Keep Residual Prediction (v3) as the default, always-enabled refiner.
2. Implement Confidence-Based Masking next -- it adds interpretability and sparsity with low overhead.
3. Feature Residual as an experimental configuration option.
4. Coarse-to-Fine as a future upgrade path for high-resolution scenes.

---

## 5. Loss Functions

### 5.1 MSE + LPIPS (Rendering Loss, Required)

**Current setup (implied from training integration):**
```
novel_view_render = render(refined_params, target_view)
render_loss = MSE(novel_view_render, gt_image) + LPIPS(novel_view_render, gt_image)
```

**Key design decisions:**
- The loss is on rendered images (pixel space), not on the Gaussian parameters directly.
- This means the refiner learns to improve rendering quality, not parameter accuracy per se.
- MSE ensures pixel-level accuracy; LPIPS ensures perceptual similarity.

**Implementation:**
```
mse_loss = F.mse_loss(rendered, gt)
lpips_loss = lpips_fn(rendered, gt)  # pre-trained AlexNet-based
render_loss = mse_loss + lambda_lpips * lpips_loss
```

**Why this is required:**
- The downstream task is rendering quality; there is no "ground truth" Gaussian parameter.
- Rendering loss is the only direct signal for whether the refinement helped.
- MSE alone can lead to blurry outputs; LPIPS sharpens perceptual quality.

### 5.2 Feature Consistency Loss

**Key idea:** The rendered feature map (from refined Gaussians) should be consistent with the predicted video features (from Wan/DiT).

```
refined_params --> renderer from target view --> rendered_feature_map
                                                    |
                                                    v
Feature consistency = MSE(rendered_feature_map, wan_feature_map)

total_loss = render_loss + lambda_feat * feature_consistency
```

**Challenge:** Gaussians render RGB images, not arbitrary feature maps. To get rendered feature maps, we need to:

**Option A: Render additional feature channels with Gaussians:**
- Attach a small feature vector (e.g., 16-dim) to each Gaussian, in addition to SH colors.
- Render both RGB and feature maps in the same rasterization pass.
- Loss on rendered features vs. Wan features.

**Option B: Use an auxiliary decoder:**
```
refined_params --> renderer --> RGB image
                                |
                                v
Auxiliary decoder (CNN) --> predicted_feature_map
                                |
                                v
Loss = MSE(predicted_feature_map, wan_feature_map)
```
- Pros: No modification to Gaussian rendering; the decoder learns to predict features from RGB.
- Cons: Decoder adds parameters; indirect signal.

**Option C: Feature distillation via correlation:**
```
Compute correlation matrix between rendered RGB and Wan features:
    C_ij = cosine_sim(rendered_RGB_i, wan_feature_j)
    
Loss = -trace(C)  (encourage positive correlation) 
  or MSE(C, identity) (encourage diagonal correlation)
```

**Recommendation:** Start with Option B (auxiliary decoder) since it doesn't modify the Gaussian renderer. It's a lightweight add-on during training (discarded at inference).

### 5.3 Latent Consistency with Video Model

**Idea:** The refined Gaussian parameters, when rendered, should produce images whose VAE latent space representation is consistent with the Wan VAE latent of the ground truth.

```
refined_params --> renderer --> rendered_image
                                 |
                                 v
WanVAE encoder --> rendered_latent [48, H/16, W/16]
                         |
                         v
latent_loss = MSE(rendered_latent, gt_latent)
                            
rendered_image --> WanVAE + DiT --> rendered_dit_features
                                        |
                                        v
dit_loss = MSE(rendered_dit_features, gt_dit_features)
```

**Pros:**
- Latent space is smoother than pixel space; gradients are more stable.
- DiT features encode high-level semantics; consistency helps with semantic quality.

**Cons:**
- Requires running Wan VAE (or DiT) on rendered images during training → expensive.
- The VAE/DiT encoder is frozen, so the loss only affects the refiner.
- Latent MSE does not always translate to perceptual pixel quality.

**Recommendation:** Add as an optional auxiliary loss and explore its effect in ablation. Likely most beneficial in early training to guide the refiner toward semantically meaningful refinements.

### 5.4 Loss Schedule

Progressive training can use different loss combinations at different stages:

| Training Stage | Loss Configuration | Purpose |
|---|---|---|
| Warm-up (0--1k steps) | MSE only | Stable initial convergence |
| Main (1k--20k steps) | MSE + LPIPS | Perceptual + pixel accuracy |
| Fine-tuning (20k+) | MSE + LPIPS + Feature Consistency | Semantic alignment |
| Optional refinement | Add Latent Consistency if needed | Video model alignment |

### 5.5 Loss Function Summary

| Loss | Domain | Required? | Compute Cost | Priority |
|---|---|---|---|---|
| MSE | Pixel (RGB) | Yes | Low | 1 |
| LPIPS | Perceptual feature | Yes | Medium | 1 |
| Feature Consistency | Feature map | No | Medium (aux decoder) | 2 |
| Latent Consistency | VAE latent | No | High (VAE encode) | 3 |

---

## 6. Integration Roadmap

### Phase 1: Framework Refactoring (Minimal Changes)

1. **Add `GaussianFeatureEncoder`** module with configurable encoding options.
2. **Add `FusionLayer`** abstract base class with `ConcatConv` implementation.
3. **Add `RefinerHead`** abstract base class with `ResidualPrediction` implementation.
4. **Keep `WanFeatureExtractor`** unchanged.

Files to create/modify:
- `post/wan_restorer/gaussian_feature_encoder.py` (new)
- `post/wan_restorer/fusion.py` (new)
- `post/wan_restorer/refiner_head.py` (new)
- `post/wan_restorer/gaussian_scene_restorer_v3.py` (refactor to use modular components)

### Phase 2: Fusion Strategies

5. **Implement windowed Cross-Attention fusion.**
6. **Implement Gated Fusion.**
7. **Reg-test all fusion strategies on a validation set.**

### Phase 3: Advanced Refiner Designs

8. **Implement Confidence-Based Masking refiner head.**
9. **Implement Feature Residual refiner head.**
10. **Implement Coarse-to-Fine multi-stage refiner.**

### Phase 4: Loss Functions

11. **Add Feature Consistency loss** (auxiliary decoder).
12. **Add optional Latent Consistency loss.**

### Phase 5: Full Integration

13. **Wire all components into `WanRestorerCfg`** with configurable options.
14. **Write unit tests for each component.**
15. **Write end-to-end test** with AnySplat integration.

---

## Appendix: GPU Memory Budget

Estimated GPU memory for each component at 256x256, 4 views, batch size 1:

| Component | Memory (GB) | Notes |
|---|---|---|
| Wan VAE Encoder | ~1.5 | Single view at a time |
| Wan DiT (8 layers) | ~4.0 | Processed per view |
| Feature Projector | ~0.3 | Conv only |
| U-Net Refiner | ~0.8 | 4-level, 64 hidden |
| Cross-Attention Fusion | ~2.0 | Windowed 16x16 |
| Rendering | ~3.0 | Gaussian rasterization |
| **Total (current v3)** | **~9.6** | VAE + DiT + U-Net |
| **Total (with cross-attn)** | **~11.6** | + Cross-Attention |

At 512x512, memory approximately doubles. Consider gradient checkpointing for DiT.
