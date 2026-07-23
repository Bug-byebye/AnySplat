"""
Force re-download of the full Wan2.2 model to complete the cache.
"""
import os, shutil, torch

model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_cache = os.path.join(cache_dir, f"models--{model_id.replace('/', '--')}")

# Delete the incomplete cache
if os.path.exists(model_cache):
    print(f"Removing incomplete cache ({model_cache})...")
    shutil.rmtree(model_cache)
    print("Cache cleared.")

from diffusers import AutoencoderKLWan, WanTransformer3DModel

print(f"\nDownloading {model_id} VAE...")
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
print("VAE downloaded OK")

print(f"\nDownloading {model_id} Transformer (this may take a while, ~10GB)...")
transformer = WanTransformer3DModel.from_pretrained(
    model_id, subfolder="transformer", torch_dtype=torch.bfloat16
)
print("Transformer downloaded OK")

print("\n✓ Model download complete!")

# Verify sizes
blobs_dir = os.path.join(model_cache, "blobs")
if os.path.exists(blobs_dir):
    total = sum(os.path.getsize(os.path.join(blobs_dir, f)) for f in os.listdir(blobs_dir))
    print(f"Cache size: {total/1e9:.1f} GB")
