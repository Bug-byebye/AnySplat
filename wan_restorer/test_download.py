"""
Quick test: download and verify Wan2.2 model loads and runs.
"""
import torch, time, os, sys

# Ensure proper path
_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_script_dir)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

def main():
    print("=" * 60)
    print("Wan2.2 Model Download & Verification Test")
    print("=" * 60)

    # Check CUDA
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Check cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.exists(cache_dir):
        items = [d for d in os.listdir(cache_dir) if "wan" in d.lower()]
        print(f"Existing Wan cache items: {items}")

    model_id = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    print(f"\nModel: {model_id}")

    # Step 1: Load VAE
    print("\n--- Step 1: Loading VAE ---")
    from diffusers import AutoencoderKLWan

    t0 = time.time()
    vae = AutoencoderKLWan.from_pretrained(
        model_id, subfolder="vae",
        torch_dtype=torch.float32,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"  VAE: z_dim={vae.config.z_dim}, "
          f"spatial_scale={vae.config.scale_factor_spatial}, "
          f"temporal_scale={vae.config.scale_factor_temporal}")

    # Step 2: Load Transformer (big download)
    print("\n--- Step 2: Loading DiT Transformer ---")
    from diffusers import WanTransformer3DModel

    t0 = time.time()
    transformer = WanTransformer3DModel.from_pretrained(
        model_id, subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    t_cfg = transformer.config
    print(f"  Transformer: layers={t_cfg.num_layers}, "
          f"heads={t_cfg.num_attention_heads}, "
          f"head_dim={t_cfg.attention_head_dim}, "
          f"ffn_dim={t_cfg.ffn_dim}")
    inner_dim = t_cfg.num_attention_heads * t_cfg.attention_head_dim
    print(f"  inner_dim={inner_dim}, patch_size={t_cfg.patch_size}")

    # Model size
    vae_params = sum(p.numel() for p in vae.parameters())
    transformer_params = sum(p.numel() for p in transformer.parameters())
    print(f"  VAE params: {vae_params/1e6:.1f}M")
    print(f"  Transformer params: {transformer_params/1e6:.1f}M")
    print(f"  Total: {(vae_params + transformer_params)/1e6:.1f}M")

    # Step 3: VAE encode test
    print("\n--- Step 3: VAE Encode Test ---")
    vae = vae.to(device).eval()

    # Input: [B=1, C=3, F=1, H=224, W=448]
    test_input = torch.randn(1, 3, 1, 224, 448).to(device)

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    with torch.no_grad():
        latent = vae.encoder(test_input)
        latent = vae.quant_conv(latent)
        latent = latent * vae.config.scaling_factor
    torch.cuda.synchronize()
    print(f"  Encode time: {time.time() - t0:.3f}s")
    print(f"  Latent shape: {tuple(latent.shape)}")
    print(f"  Latent dtype: {latent.dtype}")
    print(f"  Latent stats: mean={latent.mean():.3f}, std={latent.std():.3f}")

    # Step 4: DiT forward test
    print("\n--- Step 4: DiT Forward Test (first 4 layers) ---")
    transformer = transformer.to(device).eval()

    dtype = torch.bfloat16
    text_dim = t_cfg.text_dim
    dummy_text = torch.zeros((1, 512, text_dim), device=device, dtype=dtype)
    timestep = torch.zeros((1,), device=device, dtype=torch.long)

    latent_dit = latent.to(dtype)
    hidden_states = transformer.patch_embedding(latent_dit)
    hidden_states = hidden_states.flatten(2).transpose(1, 2)
    print(f"  Post-patch: {tuple(hidden_states.shape)}")

    temb, timestep_proj, enc_hidden, _ = transformer.condition_embedder(
        timestep, dummy_text, None
    )
    timestep_proj = timestep_proj.unflatten(1, (6, -1))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    hs = hidden_states
    for i in range(min(4, t_cfg.num_layers)):
        hs = transformer.blocks[i](
            hs, enc_hidden, timestep_proj, transformer.rope(latent_dit)
        )
    torch.cuda.synchronize()
    print(f"  Forward time: {time.time() - t0:.3f}s")
    print(f"  Output shape: {tuple(hs.shape)}")
    print(f"  Output stats: mean={hs.mean():.3f}, std={hs.std():.3f}")

    # Memory summary
    peak = torch.cuda.max_memory_allocated() / 1e9
    current = torch.cuda.memory_allocated() / 1e9
    print(f"\n--- Memory Summary ---")
    print(f"  Peak VRAM: {peak:.2f} GB")
    print(f"  Current VRAM: {current:.2f} GB")

    print("\n" + "=" * 60)
    print("✓ Model download and inference test PASSED!")
    print("=" * 60)

    # Clean up
    del vae, transformer, latent, hs
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
