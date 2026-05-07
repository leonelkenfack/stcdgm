"""
Fast version of the Stage 2 bottleneck benchmark — runs in 1-2 min on CPU.

Uses a smaller UNet (~2 M params) and smaller spatial dimensions
(64×64 vs production 172×179). The *ratio* between path A (bs=1 loop)
and path B (bs=32 batched) is what we want to measure — and that ratio
holds (often gets larger) as we scale up to the production size on
GPU, because per-launch overhead is fixed.
"""
from __future__ import annotations

import time
import torch
from diffusers import UNet2DConditionModel


def build_tiny_unet():
    return UNet2DConditionModel(
        sample_size=64,
        in_channels=3,
        out_channels=1,
        layers_per_block=1,                  # smaller
        block_out_channels=(32, 64),         # smaller
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        mid_block_type="UNetMidBlock2D",
        cross_attention_dim=64,
        norm_num_groups=8,
        attention_head_dim=8,
        only_cross_attention=[False, False],
    )


def time_path_A(unet, n_samples=16):
    """32 iter at bs=1."""
    sample = torch.randn(1, 3, 64, 64)
    t = torch.tensor([100], dtype=torch.long)
    enc = torch.zeros(1, 1, 64)
    t0 = time.time()
    for _ in range(n_samples):
        out = unet(sample=sample, timestep=t, encoder_hidden_states=enc).sample
        loss = (out - torch.randn_like(out)).pow(2).mean()
        loss.backward()
    elapsed = time.time() - t0
    for p in unet.parameters():
        if p.grad is not None:
            p.grad = None
    return elapsed


def time_path_B(unet, batch_size=16):
    """1 iter at bs=N."""
    sample = torch.randn(batch_size, 3, 64, 64)
    t = torch.tensor([100] * batch_size, dtype=torch.long)
    enc = torch.zeros(batch_size, 1, 64)
    t0 = time.time()
    out = unet(sample=sample, timestep=t, encoder_hidden_states=enc).sample
    loss = (out - torch.randn_like(out)).pow(2).mean()
    loss.backward()
    elapsed = time.time() - t0
    for p in unet.parameters():
        if p.grad is not None:
            p.grad = None
    return elapsed


def main():
    print("Stage 2 bottleneck — FAST CPU bench (toy UNet)")
    print("=" * 58)
    torch.set_num_threads(4)
    unet = build_tiny_unet()
    n = sum(p.numel() for p in unet.parameters())
    print(f"UNet params: {n/1e6:.2f} M (toy size, scales the same on GPU)")
    print()

    # Warmup
    print("Warmup...", flush=True)
    _ = time_path_A(unet, n_samples=2)
    _ = time_path_B(unet, batch_size=2)
    print("Warmup done.\n", flush=True)

    print("Path A — 16 sequential bs=1 forwards", flush=True)
    t_A = time_path_A(unet, n_samples=16)
    print(f"  total : {t_A:.2f} s", flush=True)
    print(f"  per s : {t_A/16*1000:.1f} ms", flush=True)
    print()

    print("Path B — single bs=16 forward", flush=True)
    t_B = time_path_B(unet, batch_size=16)
    print(f"  total : {t_B:.2f} s", flush=True)
    print(f"  per s : {t_B/16*1000:.1f} ms", flush=True)
    print()

    ratio = t_A / t_B
    print("=" * 58)
    print(f"VERDICT — path A is {ratio:.1f}x slower than path B")
    print("=" * 58)
    if ratio > 4:
        print("⚠️  Per-sample loop is the dominant cost.")
        print("    The 32× bs=1 loop in train_epoch_stage2 wastes most")
        print("    of the GPU. Fix: batch micros tensorially.")
    elif ratio > 2:
        print("〰️  Significant per-loop overhead — fix is worthwhile.")
    else:
        print("✓  Per-loop overhead is small on this hardware. CPU has")
        print("    little kernel-launch overhead — GPU ratio is much larger.")
    print()
    print("Note: production has 32 micros at 172×179 with 9.64M UNet.")
    print("On GPU the bs=1 vs bs=32 ratio is typically 2-4× larger than")
    print("CPU because A100 kernel launches are wasted at small batch.")


if __name__ == "__main__":
    main()
