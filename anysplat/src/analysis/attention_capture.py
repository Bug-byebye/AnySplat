"""
Attention Capture Utility for AnySplat (VGGT-based) Model.

Provides a context manager that monkey-patches the VGGT Attention modules
to force them to use the unfused attention path (which exposes the attention
weight matrix) and captures those weights for visualization.

Usage:
    model = AnySplat.from_pretrained(...)
    capturer = AttentionCapture(model)
    with capturer:
        output = model.inference(images)

    # Access captured attentions
    for key, attn_weights in capturer.attentions.items():
        block_type, layer_idx = key  # "frame" or "global", layer index
        print(attn_weights.shape)  # [B, num_heads, N, N]

    # Save / restore later
    capturer.save("captured_attentions.pt")
    capturer.load("captured_attentions.pt")
"""

from __future__ import annotations

import copy
import functools
import os
import torch
from typing import Dict, List, Optional, Tuple, Union

from src.model.encoder.vggt.models.aggregator import Aggregator


class AttentionCapture:
    """
    Captures attention weights from the VGGT Aggregator's attention modules.

    This works by monkey-patching the forward methods of all Attention modules
    inside the aggregator's frame_blocks and global_blocks. The patching forces
    fused_attn=False so that the explicit attention matrix is computed, then
    captures it after softmax.

    Use as a context manager:
        with AttentionCapture(model.encoder.aggregator) as ac:
            model.inference(images)
        ac.attentions  # dict of captured weights
    """

    def __init__(
        self,
        aggregator: Aggregator,
        capture_frame: bool = True,
        capture_global: bool = True,
        layer_indices: Optional[List[int]] = None,
        head_indices: Optional[List[int]] = None,
        store_on_cpu: bool = True,
    ):
        """
        Args:
            aggregator: The VGGT Aggregator module that contains frame_blocks and global_blocks.
            capture_frame: Whether to capture frame-level self-attention.
            capture_global: Whether to capture global-level cross-attention.
            layer_indices: Which layers to capture (0-based). None = all layers.
            head_indices: Which attention heads to capture. None = all heads.
            store_on_cpu: Move captured weights to CPU immediately to save GPU memory.
        """
        self.aggregator = aggregator
        self.capture_frame = capture_frame
        self.capture_global = capture_global
        self.layer_indices = layer_indices
        self.head_indices = head_indices
        self.store_on_cpu = store_on_cpu

        self.attentions: Dict[Tuple[str, int], torch.Tensor] = {}
        self._patches: List[tuple] = []  # (block_type, block_idx, original_forward, original_fused)
        self._block_counters: Dict[str, int] = {"frame": 0, "global": 0}

    def _make_patched_forward(self, block_type: str, block_idx: int, attn_module: torch.nn.Module):
        """Create a patched forward that captures attention weights."""
        capture_ref = self  # avoid closure issues
        block_counter_key = block_type

        @functools.wraps(attn_module.forward)
        def patched_forward(x: torch.Tensor, pos: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Only capture if layer_indices is not set or this layer is requested
            if capture_ref.layer_indices is not None and block_idx not in capture_ref.layer_indices:
                # Fall back to original behavior
                return _run_original_forward(attn_module, x, pos)

            # Force unfused to get explicit attention weights
            attn_module.fused_attn = False

            B, N, C = x.shape
            qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            q, k = attn_module.q_norm(q), attn_module.k_norm(k)

            if attn_module.rope is not None:
                q = attn_module.rope(q, pos)
                k = attn_module.rope(k, pos)

            # Compute attention explicitly
            q = q * attn_module.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)

            # Sub-select heads if requested
            attn_to_store = attn
            if capture_ref.head_indices is not None:
                attn_to_store = attn[:, capture_ref.head_indices]

            # Store with block type as key
            with torch.no_grad():
                if capture_ref.store_on_cpu:
                    attn_to_store = attn_to_store.cpu()
                capture_ref.attentions[(block_type, block_idx)] = attn_to_store.detach()

            attn = attn_module.attn_drop(attn)
            x = attn @ v
            x = x.transpose(1, 2).reshape(B, N, C)
            x = attn_module.proj(x)
            x = attn_module.proj_drop(x)

            return x

        return patched_forward

    def _patch_module(self, block_type: str, block_idx: int, attn_module: torch.nn.Module):
        """Replace forward method and store original."""
        original_forward = attn_module.forward
        original_fused = attn_module.fused_attn
        attn_module.forward = self._make_patched_forward(block_type, block_idx, attn_module)
        self._patches.append((block_type, block_idx, original_forward, original_fused))

    def _restore_module(self, block_type: str, block_idx: int, original_forward, original_fused, attn_module=None):
        """Restore original forward method."""
        if attn_module is None:
            # Look up the module
            if block_type == "frame":
                attn_module = self.aggregator.frame_blocks[block_idx].attn
            else:
                attn_module = self.aggregator.global_blocks[block_idx].attn
        attn_module.forward = original_forward
        attn_module.fused_attn = original_fused

    def __enter__(self):
        """Apply all patches to capture attention."""
        self.attentions.clear()
        self._patches.clear()

        # Patch frame attention blocks
        if self.capture_frame:
            for idx, block in enumerate(self.aggregator.frame_blocks):
                self._patch_module("frame", idx, block.attn)

        # Patch global attention blocks
        if self.capture_global:
            for idx, block in enumerate(self.aggregator.global_blocks):
                self._patch_module("global", idx, block.attn)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore all original forward methods."""
        for block_type, block_idx, original_forward, original_fused in self._patches:
            if block_type == "frame":
                attn_module = self.aggregator.frame_blocks[block_idx].attn
            else:
                attn_module = self.aggregator.global_blocks[block_idx].attn
            attn_module.forward = original_forward
            attn_module.fused_attn = original_fused
        self._patches.clear()

    def get_layer_keys(self) -> List[Tuple[str, int]]:
        """Get sorted list of (block_type, layer_idx) for captured layers."""
        return sorted(self.attentions.keys())

    def get_num_layers(self) -> int:
        """Get number of captured layers."""
        return len(self.attentions)

    def get_attention(
        self,
        block_type: str,
        layer_idx: int,
    ) -> torch.Tensor:
        """
        Get attention weights for a specific layer.

        Returns tensor of shape [B, num_heads, N, N] or subset thereof.
        For frame attention: N = num_patches in one view
        For global attention: N = total_patches across all views
        """
        return self.attentions[(block_type, layer_idx)]

    def save(self, path: str):
        """Save captured attentions to disk."""
        torch.save(self.attentions, path)
        print(f"Saved {len(self.attentions)} attention maps to {path}")

    def load(self, path: str):
        """Load captured attentions from disk."""
        self.attentions = torch.load(path, map_location="cpu", weights_only=True)
        print(f"Loaded {len(self.attentions)} attention maps from {path}")


def _run_original_forward(attn_module, x, pos=None):
    """Run the original forward logic (used as fallback when a layer is not captured)."""
    B, N, C = x.shape
    qkv = attn_module.qkv(x).reshape(B, N, 3, attn_module.num_heads, attn_module.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    q, k = attn_module.q_norm(q), attn_module.k_norm(k)

    if attn_module.rope is not None:
        q = attn_module.rope(q, pos)
        k = attn_module.rope(k, pos)

    if attn_module.fused_attn:
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            dropout_p=attn_module.attn_drop.p if attn_module.training else 0.0,
        )
    else:
        q = q * attn_module.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = attn_module.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B, N, C)
    x = attn_module.proj(x)
    x = attn_module.proj_drop(x)
    return x
