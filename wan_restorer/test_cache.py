"""
Check HuggingFace cache and test Wan model loading + inference.
"""
import os, torch, time

cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
model_dir = os.path.join(cache_dir, "models--Wan-AI--Wan2.2-TI2V-5B-Diffusers")

print(f"Model cache: {model_dir}")
print(f"Exists: {os.path.exists(model_dir)}")

if os.path.exists(model_dir):
    blobs = os.path.join(model_dir, "blobs")
    if os.path.exists(blobs):
        files = os.listdir(blobs)
        print(f"Blobs: {len(files)} files, {sum(os.path.getsize(os.path.join(blobs,f)) for f in files)/1e9:.1f} GB")

from diffusers import AutoencoderKLWan, WanTransformer3DModel

device = torch.device("cuda")
model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

# Load VAE
t0 = time.time()
vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
print(f"VAE loaded in {time.time()-t0:.1f}s")
print(f"  z_dim={vae.config.z_dim}, in_channels={vae.config.in_channels}, "
      f"scale_factor_temporal={vae.config.scale_factor_temporal}")

# Load Transformer
t0 = time.time()
transformer = WanTransformer3DModel.from_pretrained(model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
print(f"Transformer loaded in {time.time()-t0:.1f}s")
t_cfg = transformer.config
print(f"  {t_cfg.num_layers} layers, inner_dim={t_cfg.num_attention_heads * t_cfg.attention_head_dim}, "
      f"patch_size={t_cfg.patch_size}, text_dim={t_cfg.text_dim}")

# ---- VAE Encode Test ----
print("\n--- VAE Encode Test ---")
vae = vae.to(device).eval()

# Wan VAE expects [B, C*T, 1, H, W] where T = scale_factor_temporal
# For single images, we repeat 4 times and stack in channel dim
s_t = vae.config.scale_factor_temporal  # 4
s_s = vae.config.scale_factor_spatial   # 16 (actually 8 per the code)
H, W = 224, 448

# Pad to divisibility by spatial factor (8)
pad_h = (s_s - H % s_s) % s_s
pad_w = (s_s - W % s_s) % s_s
in_h, in_w = H + pad_h, W + pad_w

# Create test video [B, C, T, H, W] - T=s_t is 4 temporal frames
test_video = torch.randn(1, 3, s_t, in_h, in_w).to(device)

print(f"Input shape: {tuple(test_video.shape)}")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()
with torch.no_grad():
    # Use vae._encode() for deterministic encoding (returns 96-ch parameters)
    # Then take only the mean (first 48 channels)
    h = vae._encode(test_video)  # [1, 96, 1, H', W']
    latent = h[:, :vae.config.z_dim]  # [1, 48, 1, H', W'] - take mean only
torch.cuda.synchronize()
print(f"Encode time: {time.time()-t0:.3f}s")
print(f"Latent shape: {tuple(latent.shape)}")
print(f"  dtype: {latent.dtype}, mean={latent.mean():.3f}, std={latent.std():.3f}")

# ---- DiT Forward Test ----
print("\n--- DiT Forward Test (first 4 layers) ---")
transformer = transformer.to(device).eval()
dtype = torch.bfloat16

# Dummy text embeddings
dummy_text = torch.zeros((1, 512, t_cfg.text_dim), device=device, dtype=dtype)
timestep = torch.zeros((1,), device=device, dtype=torch.long)

latent_dit = latent.to(dtype)
hidden_states = transformer.patch_embedding(latent_dit)
hidden_states = hidden_states.flatten(2).transpose(1, 2)
print(f"Post-patch: {tuple(hidden_states.shape)}")

temb, timestep_proj, enc_hidden, _ = transformer.condition_embedder(timestep, dummy_text, None)
timestep_proj = timestep_proj.unflatten(1, (6, -1))

t0 = time.time()
hs = hidden_states
with torch.no_grad():
    for i in range(min(4, t_cfg.num_layers)):
        hs = transformer.blocks[i](hs, enc_hidden, timestep_proj, transformer.rope(latent_dit))
torch.cuda.synchronize()
print(f"4 DiT layers: {time.time()-t0:.3f}s")
print(f"Output shape: {tuple(hs.shape)}")
print(f"  mean={hs.mean():.3f}, std={hs.std():.3f}")

peak = torch.cuda.max_memory_allocated() / 1e9
current = torch.cuda.memory_allocated() / 1e9
print(f"\nPeak VRAM: {peak:.2f} GB")
print(f"Current VRAM: {current:.2f} GB")
print("SUCCESS!")
