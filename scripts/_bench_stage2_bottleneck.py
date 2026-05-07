"""
Stage 2 bottleneck benchmark — synthetic, CPU-only.

Purpose: confirm or refute the hypothesis that ``train_epoch_stage2``'s
per-micro forward loop (``for micro in batches:``) is the architectural
bottleneck causing 20 sec/batch on A100 80GB.

The training pipeline does, per logical batch from the dataloader:
  for micro in batches:                 # 32 iterations (bs=32 → 32 micros)
      Stage 1 forward (bs=1) + Stage 2 forward+backward (bs=1)

Whereas the proper way would be:
  forward all 32 samples in one tensor pass (bs=32).

This script mocks the UNet2DConditionModel at CorrDiff-Mini sizing
([64, 128, 128]) and measures both paths on CPU. The CPU/GPU absolute
numbers differ, but the *ratio* between path A (bs=1 loop) and path B
(bs=32 batched) is what tells us if the design is the issue.

Output: wall-clock, per-sample throughput, and the ratio.
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


def build_mock_unet():
    """CorrDiff-Mini sizing as in training_config_corrdiff_mini.yaml."""
    return UNet2DConditionModel(
        sample_size=176,                 # ≈ HR shape
        in_channels=3,                   # [δ_noisy, μ_HR, baseline_log]
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 128),
        down_block_types=(
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ),
        up_block_types=(
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "UpBlock2D",
        ),
        mid_block_type="UNetMidBlock2D",
        cross_attention_dim=128,
        norm_num_groups=32,
        attention_head_dim=32,
        only_cross_attention=[False, False, False],
        class_embed_type="projection",
        projection_class_embeddings_input_dim=768,
    )


def time_path_A_per_sample_loop(unet, n_samples=32, n_runs=3):
    """Mimic train_epoch_stage2: 32 sequential forward+backward at bs=1."""
    sample = torch.randn(1, 3, 172, 179)
    timestep = torch.tensor([100], dtype=torch.long)
    enc_hidden = torch.zeros(1, 1, 128)
    class_labels = torch.zeros(1, 768)

    times = []
    for run in range(n_runs):
        t0 = time.time()
        loss_total = 0.0
        for _ in range(n_samples):
            out = unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=enc_hidden,
                class_labels=class_labels,
            ).sample
            loss = (out - torch.randn_like(out)).pow(2).mean()
            loss.backward()
        elapsed = time.time() - t0
        times.append(elapsed)
        # Reset grads
        for p in unet.parameters():
            if p.grad is not None:
                p.grad = None

    return min(times)


def time_path_B_batched(unet, batch_size=32, n_runs=3):
    """Proper batched path: single forward+backward at bs=32."""
    sample = torch.randn(batch_size, 3, 172, 179)
    timestep = torch.tensor([100] * batch_size, dtype=torch.long)
    enc_hidden = torch.zeros(batch_size, 1, 128)
    class_labels = torch.zeros(batch_size, 768)

    times = []
    for run in range(n_runs):
        t0 = time.time()
        out = unet(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=enc_hidden,
            class_labels=class_labels,
        ).sample
        loss = (out - torch.randn_like(out)).pow(2).mean()
        loss.backward()
        elapsed = time.time() - t0
        times.append(elapsed)
        for p in unet.parameters():
            if p.grad is not None:
                p.grad = None

    return min(times)


def main():
    print("=" * 70)
    print("Stage 2 bottleneck benchmark — UNet CorrDiff-Mini, CPU")
    print("=" * 70)
    torch.set_num_threads(4)  # match a typical workstation
    print(f"torch threads: {torch.get_num_threads()}")

    unet = build_mock_unet()
    n_params = sum(p.numel() for p in unet.parameters())
    print(f"UNet params: {n_params/1e6:.2f} M")
    print()

    # Warmup
    print("Warmup (bs=1, 1 iter)...")
    _ = time_path_A_per_sample_loop(unet, n_samples=1, n_runs=1)
    _ = time_path_B_batched(unet, batch_size=2, n_runs=1)
    print("Warmup done.\n")

    print("Path A — sequential per-sample loop (32 × bs=1)")
    print("  (mimics current train_epoch_stage2 inner loop)")
    t_A = time_path_A_per_sample_loop(unet, n_samples=32, n_runs=3)
    print(f"  best of 3 : {t_A:.3f} s for 32 samples")
    print(f"  per sample: {t_A/32*1000:.1f} ms")
    print()

    print("Path B — proper batched (1 × bs=32)")
    print("  (what a normal training loop would do)")
    t_B = time_path_B_batched(unet, batch_size=32, n_runs=3)
    print(f"  best of 3 : {t_B:.3f} s")
    print(f"  per sample: {t_B/32*1000:.1f} ms")
    print()

    ratio = t_A / t_B
    print("=" * 70)
    print(f"VERDICT")
    print("=" * 70)
    print(f"Path A is {ratio:.1f}× slower than Path B for the same 32 samples")
    print()
    if ratio > 4:
        print(f"⚠️  ARCHITECTURAL BOTTLENECK CONFIRMED")
        print(f"    The per-micro forward loop is wasting GPU/CPU time.")
        print(f"    Fix: batch micros into a single tensor before forwarding.")
    elif ratio > 2:
        print(f"〰️  Significant overhead but not the dominant cost.")
        print(f"    Other factors (I/O, Stage 1 forward) likely contribute.")
    else:
        print(f"✓  Per-sample loop is not the main bottleneck on this hardware.")
        print(f"    Suspect I/O or Stage 1 forward.")
    print()
    print("Note: this is a CPU benchmark. On A100 GPU the absolute times")
    print("are 10-50× faster, but the bs=1 vs bs=32 *ratio* is typically")
    print("AT LEAST as bad (often worse, because per-op kernel launch")
    print("overhead is fixed and dominates at small batch sizes).")


if __name__ == "__main__":
    main()
