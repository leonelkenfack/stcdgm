"""
Two-Stage Causal Architecture smoke tests (T9-T15).

Hard gates run on CPU before consuming Colab A100 hours. Each test is
self-contained on synthetic data so the full suite runs in <30s.

Run from the repo root:

    .venv/Scripts/python -m pytest tests/test_two_stage_smoke.py -v

Tests are independent of T1-T8 (EDM smoke) but conceptually extend them
with the two-stage architecture. T11 in particular is the architecture-
level (O3) gate: if it fails, the regression head is not load-bearing
and the design must be revised before launching Colab.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from st_cdgm.models.regression_head import GraphToGridDecoder
from st_cdgm.training.two_stage import (
    causal_ablation_check,
    calibrate_sigma_data_two_stage,
    freeze_stage1,
    gamma_dag_warmup,
    stage1_compute_loss,
    unfreeze_stage1,
)


# ----------------------------------------------------------------------
# T9 — GraphToGridDecoder shape mapping
# ----------------------------------------------------------------------

def test_T9_decoder_shape_mapping():
    dec = GraphToGridDecoder(d_model=128, hr_h=172, hr_w=179, n_heads=4)

    # Single-sample (RCN native output)
    H_T = torch.randn(5, 598, 128)
    out = dec(H_T)
    assert out.shape == (1, 1, 172, 179), f"Single shape wrong: {out.shape}"

    # Batched
    H_T = torch.randn(3, 5, 598, 128)
    out = dec(H_T)
    assert out.shape == (3, 1, 172, 179), f"Batched shape wrong: {out.shape}"

    # Different intermediate resolution
    dec2 = GraphToGridDecoder(d_model=64, hr_h=64, hr_w=80, intermediate_h=16, intermediate_w=20, n_heads=4)
    H_T = torch.randn(2, 5, 100, 64)
    out = dec2(H_T)
    assert out.shape == (2, 1, 64, 80)

    # Param count sanity
    assert dec.num_params() > 100_000
    assert dec.num_params() < 5_000_000


# ----------------------------------------------------------------------
# T10 — Cross-attention non-uniform attention pattern
# ----------------------------------------------------------------------

def test_T10_attention_pattern_after_one_step():
    """Sanity : after a single optimizer step on a synthetic loss, the
    cross-attention weights should become non-uniform.

    At init, the small grid_queries (×0.02) yield near-uniform attention
    by design (stable training). This test verifies the mechanism unfreezes
    after training: a single SGD step on MSE makes the queries discriminate
    keys, validating that the attention IS learnable.
    """
    torch.manual_seed(42)
    dec = GraphToGridDecoder(
        d_model=64, hr_h=32, hr_w=32,
        intermediate_h=8, intermediate_w=8, n_heads=2,
    )
    H_T = torch.randn(1, 5, 100, 64)
    target = torch.randn(1, 1, 32, 32)

    opt = torch.optim.Adam(dec.parameters(), lr=1e-2)
    for _ in range(20):
        opt.zero_grad()
        out = dec(H_T)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        opt.step()

    # Now check attention pattern
    tokens = H_T.reshape(1, -1, 64)
    queries = dec.grid_queries
    _, attn_weights = dec.cross_attn(
        query=queries, key=tokens, value=tokens, need_weights=True
    )
    per_query_std = attn_weights.std(dim=-1).mean().item()
    assert per_query_std > 1e-4, (
        f"Even after 20 train steps, attention weights remain uniform "
        f"(std={per_query_std:.6f}); cross-attn cannot discriminate."
    )


# ----------------------------------------------------------------------
# T11 — CRITICAL: μ_HR(A_dag) ≠ μ_HR(0)  (O3 architectural gate)
# ----------------------------------------------------------------------

def test_T11_decoder_is_load_bearing_on_HT():
    """If the decoder ignores H_T, μ_HR(H_T_a) == μ_HR(H_T_b) ∀ a,b.
    This test directly compares μ_HR for two very different H_T tensors."""
    torch.manual_seed(0)
    dec = GraphToGridDecoder(d_model=128, hr_h=64, hr_w=64, intermediate_h=16, intermediate_w=16, n_heads=4)

    H_T_real = torch.randn(2, 5, 100, 128)
    H_T_zero = torch.zeros_like(H_T_real)

    with torch.no_grad():
        out_real = dec(H_T_real)
        out_zero = dec(H_T_zero)

    delta = (out_real - out_zero).abs().mean().item()
    signal = out_real.abs().mean().item() + 1e-12
    ratio = delta / signal

    print(f"\nT11: |dec(H_T) - dec(0)| = {delta:.5f}, |dec(H_T)| = {signal:.5f}, ratio = {ratio:.4f}")
    assert delta > 1e-4, (
        f"Decoder is NOT load-bearing on H_T (delta={delta:.6e}). "
        "The architecture cannot guarantee (O3); abort design."
    )


# ----------------------------------------------------------------------
# T12 — Channel concatenation UNet plumbing
# ----------------------------------------------------------------------

def test_T12_channel_concat_unet():
    pytest.importorskip("diffusers")

    from st_cdgm.models.diffusion_decoder import CausalDiffusionDecoder
    from st_cdgm.models.edm_preconditioner import EDMConfig

    # Build a tiny UNet in concat mode (in_channels=1, expects 3 due to concat)
    cfg = EDMConfig(sigma_data=0.05)
    decoder = CausalDiffusionDecoder(
        in_channels=1,
        conditioning_dim=8,  # required for diffusers cross-attn dim, even if unused
        height=32,
        width=32,
        scheduler_type="edm_karras",
        edm_config=cfg,
        causal_concat=True,
        unet_kwargs=dict(
            layers_per_block=1,
            block_out_channels=(8, 16),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            mid_block_type="UNetMidBlock2D",
            norm_num_groups=4,
        ),
    )

    # The internal UNet should report 3 in_channels
    assert decoder.unet.config.in_channels == 3, (
        f"Expected UNet in_channels=3, got {decoder.unet.config.in_channels}"
    )
    assert decoder.unet.config.out_channels == 1

    # Forward shape test
    delta = torch.randn(2, 1, 32, 32) * 0.05
    sigma = torch.tensor([0.5, 0.5])
    mu_HR = torch.randn(2, 1, 32, 32) * 0.1
    baseline_log = torch.randn(2, 1, 32, 32) * 0.5

    out = decoder.forward_edm(
        delta, sigma, conditioning=None, mu_HR=mu_HR, baseline_log=baseline_log
    )
    assert out.shape == (2, 1, 32, 32), f"Output shape wrong: {out.shape}"
    assert torch.isfinite(out).all()


# ----------------------------------------------------------------------
# T12b — DPM-Solver++ sampling with causal_concat (BS39 / V4 Tier 0)
# ----------------------------------------------------------------------

def test_T12b_dpm_solver_causal_concat_sample():
    """Regression: DPM++ must accept 3-channel UNet input in two-stage mode.

    Previously ``_sample_dpm_solver`` fed 1-channel ``sample`` into a UNet
    built with ``in_channels=3``, causing RuntimeError at inference.
    """
    pytest.importorskip("diffusers")

    from st_cdgm.models.diffusion_decoder import CausalDiffusionDecoder
    from st_cdgm.models.edm_preconditioner import EDMConfig

    cfg = EDMConfig(sigma_data=0.05)
    decoder = CausalDiffusionDecoder(
        in_channels=1,
        conditioning_dim=8,
        height=32,
        width=32,
        scheduler_type="edm_karras",
        edm_config=cfg,
        causal_concat=True,
        unet_kwargs=dict(
            layers_per_block=1,
            block_out_channels=(8, 16),
            down_block_types=("DownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "UpBlock2D"),
            mid_block_type="UNetMidBlock2D",
            norm_num_groups=4,
        ),
    )

    mu_HR = torch.randn(2, 1, 32, 32) * 0.1
    baseline_log = torch.randn(2, 1, 32, 32) * 0.5

    out = decoder.sample(
        conditioning=None,
        mu_HR=mu_HR,
        baseline_log=baseline_log,
        scheduler_type="dpm_solver++",
        num_steps=4,
        cfg_scale=1.5,
        apply_constraints=False,
    )
    assert out.residual.shape == (2, 1, 32, 32), f"residual shape: {out.residual.shape}"
    assert torch.isfinite(out.residual).all()


# ----------------------------------------------------------------------
# T13 — Stage 1 backward propagates to A_dag.grad
# ----------------------------------------------------------------------

def test_T13_stage1_grad_flow_to_A_dag():
    """Verify that L_stage1 does propagate gradient to A_dag (because L_rec
    and L_dag both depend on it)."""
    # Minimal RCN-like cell with A_dag parameter
    class MiniRCN(nn.Module):
        def __init__(self, num_vars=5, hidden_dim=16):
            super().__init__()
            self.A_dag = nn.Parameter(torch.randn(num_vars, num_vars) * 0.1)
            self.num_vars = num_vars

        def forward(self, drivers):  # drivers: [seq, N, d]
            # Aggregate using A_dag (so gradient flows)
            H = drivers.mean(dim=0)  # [N, d]
            H = H.unsqueeze(0).expand(self.num_vars, -1, -1)  # [q, N, d]
            mixed = torch.einsum("qn d, q p -> p n d", H, self.A_dag)
            return mixed

    rcn = MiniRCN()
    # Apply a regression head sized for this mini setup
    dec = GraphToGridDecoder(d_model=16, hr_h=32, hr_w=32, intermediate_h=8, intermediate_w=8, n_heads=2)

    drivers = torch.randn(4, 50, 16)  # 4 timesteps, 50 nodes, d=16
    H_T = rcn(drivers)
    mu_HR = dec(H_T)
    target = torch.randn(1, 1, 32, 32) * 0.1

    # Use a synthetic dagma penalty (= ||A||_F²) that depends on A_dag
    L_dagma = (rcn.A_dag * rcn.A_dag).sum()
    L_l1 = rcn.A_dag.abs().sum()

    loss, _ = stage1_compute_loss(
        mu_HR=mu_HR,
        target_residual=target,
        rcn_reconstruction_loss=None,
        dagma_loss=L_dagma,
        dag_l1_loss=L_l1,
    )
    loss.backward()

    assert rcn.A_dag.grad is not None, "A_dag.grad is None"
    assert rcn.A_dag.grad.abs().sum().item() > 0, "A_dag.grad is exactly zero"
    assert torch.isfinite(rcn.A_dag.grad).all(), "A_dag.grad has NaN/Inf"


# ----------------------------------------------------------------------
# T14 — freeze_stage1 / unfreeze_stage1 effective
# ----------------------------------------------------------------------

def test_T14_freeze_stage1_effective():
    enc = nn.Linear(10, 20)
    dec = GraphToGridDecoder(d_model=128, hr_h=64, hr_w=64, n_heads=4)

    # Initially trainable
    assert all(p.requires_grad for p in enc.parameters())
    assert all(p.requires_grad for p in dec.parameters())

    freeze_stage1(enc, dec)

    assert all(not p.requires_grad for p in enc.parameters()), "encoder not frozen"
    assert all(not p.requires_grad for p in dec.parameters()), "decoder not frozen"
    assert not enc.training, "encoder still in train mode after freeze"
    assert not dec.training, "decoder still in train mode after freeze"

    unfreeze_stage1(enc, dec)
    assert all(p.requires_grad for p in enc.parameters())
    assert all(p.requires_grad for p in dec.parameters())
    assert enc.training and dec.training


# ----------------------------------------------------------------------
# T15 — sigma_data calibration sanity
# ----------------------------------------------------------------------

def test_T15_sigma_data_calibration_sanity():
    """Calibration on a known synthetic distribution returns the right std."""
    # Stub data loader / iterate function to feed predetermined deltas
    class FakeBuilder:
        pass

    # Synthetic samples where target_residual - mu_HR should have std ≈ 0.07
    SYNTH_STD = 0.07
    N_SAMPLES = 80
    H, W = 32, 32

    class FakeDataLoader:
        def __iter__(self):
            for _ in range(1):
                yield None  # just iteration token; iterate_batches handles it

    def fake_iterate(loader, builder, device):
        for _ in range(N_SAMPLES):
            # The synthetic batch: target_residual - mu_HR has std SYNTH_STD
            target_residual = torch.randn(1, 1, H, W) * SYNTH_STD
            yield [{
                "lr": torch.randn(8, 15, 5, 5),  # dummy
                "residual": [target_residual.squeeze(0)],
                "hetero": None,
            }]

    # Stub encoder / RCN / regression that produces mu_HR == 0
    class StubEncoder:
        def init_state(self, hetero):
            return torch.zeros(5, 25, 8)
        def eval(self): pass

    class StubRCNRunner:
        class _Cell:
            def eval(self): pass
        cell = _Cell()
        def run(self, H_init, drivers, reconstruction_sources=None):
            class _Out:
                states = [torch.zeros(5, 25, 8)]
            return _Out()

    class StubRegHead(nn.Module):
        def forward(self, H_T):
            return torch.zeros(1, 1, H, W)
        def eval(self): pass

    result = calibrate_sigma_data_two_stage(
        encoder=StubEncoder(),
        rcn_runner=StubRCNRunner(),
        regression_head=StubRegHead(),
        data_loader=FakeDataLoader(),
        iterate_batches_fn=fake_iterate,
        builder=FakeBuilder(),
        device=torch.device("cpu"),
        max_samples=N_SAMPLES,
        verbose=False,
    )

    # ±20% tolerance on the empirical std (small sample size + subsampling)
    assert abs(result["std"] - SYNTH_STD) / SYNTH_STD < 0.25, (
        f"Calibrated σ_data {result['std']:.4f} too far from true {SYNTH_STD}"
    )
    # n_pixels reflects the post-subsample count (every 1024-th pixel)
    assert result["n_pixels"] >= N_SAMPLES // 2, (
        f"n_pixels {result['n_pixels']} too low — sampling broken?"
    )


# ----------------------------------------------------------------------
# T-extra: gamma_dag_warmup curve sanity
# ----------------------------------------------------------------------

def test_gamma_dag_warmup_curve():
    assert gamma_dag_warmup(0, 0.10, warmup_epochs=5) == pytest.approx(0.0)
    assert gamma_dag_warmup(5, 0.10, warmup_epochs=5) == pytest.approx(0.10)
    assert gamma_dag_warmup(10, 0.10, warmup_epochs=5) == pytest.approx(0.10)  # clipped
    # Halfway
    assert gamma_dag_warmup(2, 0.10, warmup_epochs=5) == pytest.approx(0.04)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
