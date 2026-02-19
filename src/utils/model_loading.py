"""
Model loading utilities with fallback support.
Prioritizes loading from local path, falls back to HuggingFace Hub if not available.
"""

from pathlib import Path
from typing import Optional

import torch

from src.model.model.anysplat import AnySplat


# HuggingFace Hub model identifier
HF_MODEL_ID = "lhjiang/anysplat"


def load_model_with_fallback(
    local_path: Optional[str] = None,
    hf_model_id: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> AnySplat:
    """
    Load AnySplat model with fallback: prioritize local path, then HuggingFace Hub.

    Args:
        local_path: Path to local pretrained model directory. If None, uses default.
        hf_model_id: HuggingFace Hub model ID. If None, uses "lhjiang/anysplat".
        device: Torch device to move model to. If None, uses CUDA if available.

    Returns:
        AnySplat model loaded and moved to the specified device.

    Raises:
        FileNotFoundError: If local path is specified but doesn't exist and HF loading fails.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hf_model_id is None:
        hf_model_id = HF_MODEL_ID

    # Try loading from local path first
    if local_path is not None:
        local_path_obj = Path(local_path).expanduser().resolve()
        if local_path_obj.exists():
            print(f"[model_loading] Loading model from local path: {local_path_obj}")
            try:
                model = AnySplat.from_pretrained(str(local_path_obj))
                model = model.to(device)
                model.eval()
                for param in model.parameters():
                    param.requires_grad = False
                return model
            except Exception as e:
                print(f"[model_loading] Failed to load from local path: {e}")
                print(f"[model_loading] Attempting to load from HuggingFace Hub: {hf_model_id}")
        else:
            print(f"[model_loading] Local path not found: {local_path_obj}")
            print(f"[model_loading] Attempting to load from HuggingFace Hub: {hf_model_id}")

    # Fall back to HuggingFace Hub
    print(f"[model_loading] Loading model from HuggingFace Hub: {hf_model_id}")
    try:
        model = AnySplat.from_pretrained(hf_model_id)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        print(f"[model_loading] Successfully loaded from HuggingFace Hub: {hf_model_id}")
        return model
    except Exception as e:
        raise RuntimeError(
            f"[model_loading] Failed to load model from both local path ({local_path}) "
            f"and HuggingFace Hub ({hf_model_id}). Error: {e}"
        ) from e
