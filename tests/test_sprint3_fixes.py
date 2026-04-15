"""
Regression tests for Sprint 3 — « polish probabiliste ».

Each test pins one Sprint 3 component from ORACLE_HYPERPLAN_ANALYSIS.md §3 so
future refactors cannot silently break the probabilistic pipeline.

Covered:
- ``loss_precip_physical`` scalar behaviour (positivity, mass, quantile).
- ``loss_precip_physical`` gradient on predicted x0.
- ``loss_precip_physical`` NaN robustness (geographic masks).
- ``CausalDiffusionDecoder._sample_dpm_solver`` actually honours
  ``cfg_scale`` (output differs between cfg=0 and cfg=2).
- Min-SNR weighting exists and does not blow up at extreme SNR.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from st_cdgm.models.diffusion_decoder import CausalDiffusionDecoder
from st_cdgm.training.training_loop import loss_precip_physical


# ---------------------------------------------------------------------------
# loss_precip_physical
# ---------------------------------------------------------------------------


def test_precip_physical_zero_when_perfect():
    """Perfect prediction on non-negative target ⇒ all three terms = 0."""
    target = torch.rand(1, 1, 16, 16)  # non-negative by construction
    pred = target.clone()
    loss = loss_precip_physical(pred, target)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_precip_physical_penalises_negative_pred():
    """Positivity hinge: negative pred raises the loss above zero."""
    target = torch.ones(1, 1, 8, 8)
    pred = torch.full_like(target, -0.5)
    loss = loss_precip_physical(pred, target, w_mass=0.0, w_quantile=0.0)
    # positivity = mean(relu(-(-0.5))) = 0.5
    assert loss.item() == pytest.approx(0.5, rel=1e-4)


def test_precip_physical_mass_component():
    """Mass conservation term penalises global-mean drift."""
    target = torch.full((1, 1, 8, 8), 1.0)
    pred = torch.full((1, 1, 8, 8), 1.3)   # +0.3 global bias
    loss = loss_precip_physical(
        pred, target, w_positivity=0.0, w_quantile=0.0, w_mass=1.0
    )
    assert loss.item() == pytest.approx(0.09, rel=1e-4)  # 0.3^2


def test_precip_physical_quantile_component():
    """Quantile term penalises mismatched tails without touching the bulk."""
    torch.manual_seed(7)
    # 64 pixels of zero + 4 extreme pixels of 10.0 → target p99 ≈ 10, p95 > 0.
    target = torch.zeros(1, 1, 8, 10)
    target[0, 0, 0, -4:] = 10.0
    # Prediction has the same mean but flattens the tails completely.
    pred = torch.full_like(target, target.mean())
    loss = loss_precip_physical(
        pred, target, w_positivity=0.0, w_mass=0.0, w_quantile=1.0
    )
    # The quantile mismatch should drive the loss well above zero.
    assert loss.item() > 0.1


def test_precip_physical_grad_flows_to_pred():
    """Gradient of the composite loss must reach the predicted x0 tensor."""
    target = torch.rand(1, 1, 16, 16)
    pred = torch.rand(1, 1, 16, 16, requires_grad=True)
    loss = loss_precip_physical(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum().item() > 0.0


def test_precip_physical_handles_nan_mask():
    """Geographic NaN mask must not propagate into the loss value."""
    target = torch.ones(1, 1, 8, 8)
    target[0, 0, 0, 0] = float("nan")
    pred = torch.ones(1, 1, 8, 8)
    loss = loss_precip_physical(pred, target)
    assert math.isfinite(loss.item())
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_precip_physical_zero_when_no_valid_pixels():
    """All-NaN target short-circuits to 0 instead of dividing by zero."""
    target = torch.full((1, 1, 4, 4), float("nan"))
    pred = torch.zeros(1, 1, 4, 4)
    loss = loss_precip_physical(pred, target)
    assert loss.item() == 0.0


# ---------------------------------------------------------------------------
# CFG through DPM-Solver++
# ---------------------------------------------------------------------------


def _make_tiny_decoder() -> CausalDiffusionDecoder:
    """A minimal UNet-backed decoder that fits in a unit test."""
    return CausalDiffusionDecoder(
        in_channels=1,
        conditioning_dim=8,
        height=16,
        width=16,
        num_diffusion_steps=50,
        unet_kwargs=dict(
            layers_per_block=1,
            block_out_channels=(8, 16),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            mid_block_type="UNetMidBlock2D",
            norm_num_groups=4,
            class_embed_type="projection",
            projection_class_embeddings_input_dim=16,  # q*cond = 2*8
            resnet_time_scale_shift="scale_shift",
            attention_head_dim=4,
            only_cross_attention=[False, True],
        ),
        scheduler_type="dpm_solver++",
    )


def _run_dpm(decoder, conditioning, cfg_scale, seed):
    gen = torch.Generator().manual_seed(seed)
    decoder.eval()
    with torch.no_grad():
        return decoder.sample(
            conditioning,
            num_steps=4,
            scheduler_type="dpm_solver++",
            cfg_scale=cfg_scale,
            generator=gen,
            apply_constraints=False,
        )


def test_dpm_solver_cfg_actually_changes_output():
    """
    Sprint 3: setting ``cfg_scale > 0`` must change the DPM-Solver++ output.
    Before the fix, ``_sample_dpm_solver`` silently ignored ``cfg_scale``,
    so different cfg values produced identical residuals — this is the bug
    that made the YAML knob a no-op.
    """
    try:
        decoder = _make_tiny_decoder()
    except ImportError:
        pytest.skip("diffusers not available")
    conditioning = torch.randn(1, 2, 8)

    out_cfg0 = _run_dpm(decoder, conditioning, cfg_scale=0.0, seed=123)
    out_cfg2 = _run_dpm(decoder, conditioning, cfg_scale=2.0, seed=123)
    # cfg=0 and cfg=2 must differ once CFG is wired — allow a small tolerance
    # in case of numerical ties at t=0 with an untrained UNet.
    diff = (out_cfg2.residual - out_cfg0.residual).abs().max().item()
    assert diff > 1e-4, (
        f"cfg_scale had no effect on DPM-Solver++ output (max diff={diff})."
    )


def test_dpm_solver_cfg_handles_spatial_conditioning():
    """Smoke test: CFG branch accepts a spatial conditioning tensor."""
    try:
        decoder = _make_tiny_decoder()
    except ImportError:
        pytest.skip("diffusers not available")
    conditioning = torch.randn(1, 2, 8)
    cond_spatial = torch.randn(1, 12, 8)  # 12 spatial tokens
    out = _run_dpm_with_spatial(decoder, conditioning, cond_spatial, 2.0, 0)
    assert torch.isfinite(out.residual).all()


def _run_dpm_with_spatial(decoder, conditioning, cond_spatial, cfg_scale, seed):
    gen = torch.Generator().manual_seed(seed)
    decoder.eval()
    with torch.no_grad():
        return decoder.sample(
            conditioning,
            num_steps=4,
            scheduler_type="dpm_solver++",
            cfg_scale=cfg_scale,
            generator=gen,
            conditioning_spatial=cond_spatial,
            apply_constraints=False,
        )


# ---------------------------------------------------------------------------
# Min-SNR weighting — numerical stability
# ---------------------------------------------------------------------------


def test_min_snr_weighting_finite_extremes():
    """
    Min-SNR should produce a finite weight for the extreme timesteps of the
    DDPM schedule (alphas_cumprod near 1 at t=0, near 0 at t=T-1). This is
    a safety check against accidental division-by-zero if someone rewrites
    the weighting formula.
    """
    try:
        decoder = _make_tiny_decoder()
    except ImportError:
        pytest.skip("diffusers not available")
    target = torch.rand(1, 1, 16, 16)
    conditioning = torch.randn(1, 2, 8)
    # Several random time samples — the loss must be finite for all of them.
    for seed in (0, 1, 2, 3):
        torch.manual_seed(seed)
        loss = decoder.compute_loss(target, conditioning)
        assert math.isfinite(loss.item()), f"Min-SNR loss exploded at seed={seed}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
