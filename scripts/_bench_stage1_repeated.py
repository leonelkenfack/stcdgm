"""
Bench: cost of repeating Stage 1 forward 32× per logical batch.

The hypothesis: in train_epoch_stage2, encoder.init_state +
rcn_runner.run (16 timesteps) + regression_head.forward is run
*for every micro* in the batch. Even though Stage 1 weights are frozen,
this forward pass is *not* skippable in the current code.

We mock the equivalent compute load with stacked GraphConv-like ops
(SAGEConv would require pyg-cuda which isn't here). The key cost is
the *sequence-length* sequential dependence — RCN unrolls over 16
timesteps, which can't be parallelized within a sample.

Output: sequential cost per sample × 32 samples.
"""
from __future__ import annotations

import time
import torch
import torch.nn as nn


class FakeRCNCell(nn.Module):
    """Mimics CausalRCNCell compute cost: 2 linear layers + DAGMA-style
    M-matrix multiply per timestep."""

    def __init__(self, hidden_dim=128, num_vars=6, num_nodes=2048):
        super().__init__()
        self.hidden = hidden_dim
        self.num_vars = num_vars
        self.driver_proj = nn.Linear(2, hidden_dim)         # LR driver dim ~2
        self.update_lin = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dag_W = nn.Parameter(torch.randn(num_vars, num_vars) * 0.1)
        self.recon_head = nn.Linear(hidden_dim, 2)

    def forward(self, H, driver):
        # H: [q, N, hidden_dim] — q=num_vars, N=num_nodes
        # driver: [N, driver_dim]
        d_emb = self.driver_proj(driver)  # [N, hidden]
        # broadcast over q
        d_emb_expanded = d_emb.unsqueeze(0).expand_as(H)
        # DAG-conditioned mixing along q axis
        H_mixed = torch.einsum('qr,rnh->qnh', self.dag_W, H)
        # Update via concat + lin
        cat = torch.cat([H_mixed, d_emb_expanded], dim=-1)
        H_next = torch.tanh(self.update_lin(cat))
        return H_next


class FakeRegressionHead(nn.Module):
    """Mimics GraphToGridDecoder cost (~407k params)."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 64 * 22 * 23)  # H_T → grid prefig
        self.refine = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, H_T):
        # H_T: [q, N, hidden] — pool to [B, q*hidden]
        B, q, N, h = 1, 6, 2048, 128
        pooled = H_T.permute(2, 0, 1).reshape(N, q * h)[:1].mean(dim=0, keepdim=True)
        # ↑ trivial pool — production may differ
        out = self.proj(pooled)               # [B, 64*22*23]
        grid = out.view(B, 64, 22, 23)
        grid = self.refine(grid)              # [B, 1, 22, 23]
        # Upsample to HR
        return torch.nn.functional.interpolate(grid, size=(172, 179), mode='bilinear')


def time_stage1_per_micro(cell, head, n_samples=32, seq_len=16):
    """Mimic train_epoch_stage2: 32 sequential Stage 1 forwards, each
    unrolling RCN over 16 timesteps."""
    H_init = torch.randn(6, 2048, 128)  # [q, N, hidden]
    drivers = [torch.randn(2048, 2) for _ in range(seq_len)]

    cell.eval(); head.eval()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_samples):
            H = H_init.clone()
            for t in range(seq_len):
                H = cell(H, drivers[t])
            mu = head(H)
    return time.time() - t0


def time_stage1_oneshot(cell, head, n_samples=32, seq_len=16):
    """If Stage 1 outputs were precomputed and cached (BS32b), the
    Stage 2 loop just loads them — essentially free."""
    cache_mu = torch.randn(n_samples, 1, 172, 179)
    t0 = time.time()
    for i in range(n_samples):
        _ = cache_mu[i]  # like loading from .npy mmap
    return time.time() - t0


def main():
    print("Stage 1 forward repetition cost — CPU bench")
    print("=" * 58)
    torch.set_num_threads(4)

    cell = FakeRCNCell()
    head = FakeRegressionHead()
    n_params = sum(p.numel() for p in cell.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"Stage 1 mock (cell + head) params: {n_params/1e6:.2f} M")
    print()

    # Warmup
    print("Warmup...", flush=True)
    _ = time_stage1_per_micro(cell, head, n_samples=2, seq_len=4)
    print("Warmup done.\n", flush=True)

    print("Path A — recompute Stage 1 every batch (current code):")
    t_A = time_stage1_per_micro(cell, head, n_samples=32, seq_len=16)
    print(f"  32 samples × 16 timesteps : {t_A:.2f} s", flush=True)
    print(f"  per sample                : {t_A/32*1000:.1f} ms", flush=True)
    print()

    print("Path B — read precomputed μ_HR from cache:")
    t_B = time_stage1_oneshot(cell, head, n_samples=32, seq_len=16)
    print(f"  32 samples loaded from RAM : {t_B*1000:.1f} ms", flush=True)
    print()

    print("=" * 58)
    print(f"Stage 1 cost per logical batch on CPU: ~{t_A:.1f} s")
    print(f"After pre-cache:                       ~{t_B*1000:.1f} ms")
    print(f"Speedup: {t_A/t_B:.0f}× by skipping Stage 1 entirely")
    print()
    print("On A100 GPU, Stage 1 forward is faster (cuDNN), but the")
    print("16-step sequential RCN unroll cannot be parallelized within")
    print("a sample. The 32× per-batch repetition is the waste.")


if __name__ == "__main__":
    main()
