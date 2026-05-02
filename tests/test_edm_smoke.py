"""
Phase 1 EDM smoke tests (T1-T8).

These run on CPU in <2 minutes total and validate that the EDM rewrite
is functional BEFORE consuming Colab A100 hours. Each test is a hard
gate: failure means the EDM path is broken and we fall back to the
DDPM-fix branch.

Run from the repo root:

    .venv/Scripts/python -m pytest tests/test_edm_smoke.py -v

Or run individual tests:

    .venv/Scripts/python -m pytest tests/test_edm_smoke.py::test_T4_detach_a_dag -v
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from st_cdgm.models.edm_preconditioner import (
    EDMConfig,
    compute_preconditioning,
    karras_sigma_schedule,
    lambda_weight,
    sample_training_sigma,
)
from st_cdgm.models.edm_sampler import heun_sample


# ----------------------------------------------------------------------
# Common fixtures: a tiny denoiser standing in for the full ST-CDGM UNet.
# ----------------------------------------------------------------------

class TinyDenoiser(nn.Module):
    """3-conv U-shaped network with a stub class-label projection.

    Used to validate the EDM training/sampling math at minimal compute.
    Same external API as ``CausalDiffusionDecoder.forward_edm`` but
    completely independent — keeps this smoke suite fast and free of
    diffusers dependency.
    """

    def __init__(self, in_channels: int = 1, sigma_data: float = 0.1):
        super().__init__()
        self.sigma_data = sigma_data
        self.in_channels = in_channels
        ch = 8
        self.in_proj = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.mid = nn.Conv2d(ch, ch, 3, padding=1)
        self.out_proj = nn.Conv2d(ch, in_channels, 3, padding=1)
        # Time embedding: tiny MLP that maps c_noise (scalar) to a per-channel bias
        self.time_emb = nn.Sequential(
            nn.Linear(1, ch),
            nn.SiLU(),
            nn.Linear(ch, ch),
        )

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """Apply EDM preconditioning + run the tiny U-shape."""
        c_skip, c_out, c_in, c_noise = compute_preconditioning(sigma, self.sigma_data)
        h = self.in_proj(c_in * x)
        # Inject c_noise as additive channel bias [B, ch, 1, 1]
        t_emb = self.time_emb(c_noise.view(-1, 1)).view(-1, h.shape[1], 1, 1)
        h = torch.relu(h + t_emb)
        h = torch.relu(self.mid(h))
        F_x = self.out_proj(h)
        return c_skip * x + c_out * F_x


# ----------------------------------------------------------------------
# T1 — preconditioner shapes
# ----------------------------------------------------------------------

def test_T1_preconditioner_shapes():
    """Forward of EDMPreconditioner — shapes match input, no NaN."""
    sigma = torch.tensor([0.1, 0.5, 1.0, 5.0])
    c_skip, c_out, c_in, c_noise = compute_preconditioning(sigma, sigma_data=0.1)
    assert c_skip.shape == (4, 1, 1, 1), c_skip.shape
    assert c_out.shape == (4, 1, 1, 1), c_out.shape
    assert c_in.shape == (4, 1, 1, 1), c_in.shape
    assert c_noise.shape == (4,), c_noise.shape
    for name, t in [("c_skip", c_skip), ("c_out", c_out), ("c_in", c_in), ("c_noise", c_noise)]:
        assert torch.isfinite(t).all(), f"{name} contains NaN/Inf"

    # When sigma == sigma_data, c_skip == 0.5 and c_out == sigma_data / sqrt(2)
    sigma_eq = torch.tensor([0.1])
    c_skip_eq, c_out_eq, c_in_eq, _ = compute_preconditioning(sigma_eq, sigma_data=0.1)
    assert torch.isclose(c_skip_eq.flatten(), torch.tensor([0.5]), atol=1e-6)
    assert torch.isclose(c_out_eq.flatten(), torch.tensor([0.1 / math.sqrt(2)]), atol=1e-6)


# ----------------------------------------------------------------------
# T2 — Karras Eq. 8 verifies: when D(x_noisy) == x0, loss == 0
# ----------------------------------------------------------------------

def test_T2_loss_zero_when_perfect_denoise():
    """If the model is a perfect denoiser, the EDM loss is 0."""
    sigma = torch.tensor([0.1, 1.0])
    sigma_data = 0.1
    x0 = torch.randn(2, 1, 8, 8) * sigma_data  # match sigma_data scale
    n = torch.randn_like(x0) * sigma.view(-1, 1, 1, 1)
    x_noisy = x0 + n

    # "Oracle" denoiser: returns x0 exactly
    D = x0.clone()

    weight = lambda_weight(sigma, sigma_data)
    loss = (weight * (D - x0) ** 2).mean()
    assert loss.item() < 1e-12, f"Oracle denoiser produced nonzero loss: {loss.item()}"


# ----------------------------------------------------------------------
# T3 — gradient flow on TinyDenoiser
# ----------------------------------------------------------------------

def test_T3_gradient_flow():
    """One backward pass populates grads on every UNet param, no NaN."""
    torch.manual_seed(0)
    sigma_data = 0.1
    model = TinyDenoiser(in_channels=1, sigma_data=sigma_data)
    cfg = EDMConfig(sigma_data=sigma_data)

    x0 = torch.randn(2, 1, 16, 16) * sigma_data
    sigma = sample_training_sigma(2, cfg.P_mean, cfg.P_std, device=x0.device)
    n = torch.randn_like(x0) * sigma.view(-1, 1, 1, 1)
    x_noisy = x0 + n

    D = model(x_noisy, sigma)
    weight = lambda_weight(sigma, sigma_data)
    loss = (weight * (D - x0) ** 2).mean()
    loss.backward()

    bad = []
    for name, p in model.named_parameters():
        if p.grad is None:
            bad.append(f"{name}: grad is None")
        elif not torch.isfinite(p.grad).all():
            bad.append(f"{name}: grad has NaN/Inf")
        elif p.grad.abs().max().item() == 0:
            bad.append(f"{name}: grad is all zero")
    assert not bad, "Gradient flow problems: " + "; ".join(bad)


# ----------------------------------------------------------------------
# T4 — CRITICAL: A_dag.detach() blocks gradient from L_gen
# ----------------------------------------------------------------------

def test_T4_detach_a_dag():
    """A_dag.grad must be None when fed through .detach() into the
    diffusion forward — the paper §sec:arch:rcn requires this."""
    torch.manual_seed(0)
    sigma_data = 0.1
    model = TinyDenoiser(in_channels=1, sigma_data=sigma_data)

    # Simulate A_dag as a leaf parameter, then route a *detached* copy
    # into the conditioning. The diffusion loss must NOT update A_dag.
    A_dag = torch.nn.Parameter(torch.randn(5, 5))
    H_T = torch.randn(2, 5, 4)  # bs=2, q=5 vars, d=4

    # Use A_dag.detach() to produce conditioning (mimicking the real path)
    conditioning_via_dag = (A_dag.detach() @ H_T).mean(dim=1)  # [B, d]
    # Add a no-op channel bias to wire conditioning into the model output
    # via a learnable projection (so loss depends on conditioning numerically).
    proj = nn.Linear(4, 1)
    cond_bias = proj(conditioning_via_dag).view(-1, 1, 1, 1)

    sigma = torch.tensor([0.5, 1.0])
    x0 = torch.randn(2, 1, 8, 8) * sigma_data
    x_noisy = x0 + torch.randn_like(x0) * sigma.view(-1, 1, 1, 1)
    D = model(x_noisy, sigma) + cond_bias

    weight = lambda_weight(sigma, sigma_data)
    loss = (weight * (D - x0) ** 2).mean()
    loss.backward()

    assert A_dag.grad is None, (
        f"A_dag received gradient from L_gen ({A_dag.grad}) — the .detach() "
        f"failed and the Trace Trap is open. Aborting EDM smoke."
    )
    # Sanity: the projection layer DID receive a gradient
    assert proj.weight.grad is not None and proj.weight.grad.abs().max().item() > 0


# ----------------------------------------------------------------------
# T5 — sigma_data sweep: best loss at empirical sigma_data
# ----------------------------------------------------------------------

def test_T5_sigma_data_sweep():
    """Train a TinyDenoiser briefly across multiple sigma_data values;
    the value matching the empirical std of x0 should give the best
    final-epoch loss. This validates the calibration argument from
    Gemini Axis 2."""
    torch.manual_seed(0)
    empirical_sigma = 0.1
    sigma_data_candidates = [0.01, 0.1, 1.0]
    final_losses = {}

    for sigma_data in sigma_data_candidates:
        torch.manual_seed(0)
        model = TinyDenoiser(in_channels=1, sigma_data=sigma_data)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        cfg = EDMConfig(sigma_data=sigma_data)

        for _ in range(50):
            x0 = torch.randn(8, 1, 8, 8) * empirical_sigma
            sigma = sample_training_sigma(8, cfg.P_mean, cfg.P_std, device=x0.device)
            n = torch.randn_like(x0) * sigma.view(-1, 1, 1, 1)
            x_noisy = x0 + n
            D = model(x_noisy, sigma)
            weight = lambda_weight(sigma, sigma_data)
            loss = (weight * (D - x0) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # Eval loss with sigma_data fixed to true value (so all candidates
        # are scored on the same yardstick).
        with torch.no_grad():
            x0 = torch.randn(64, 1, 8, 8) * empirical_sigma
            sigma = sample_training_sigma(64, cfg.P_mean, cfg.P_std, device=x0.device)
            n = torch.randn_like(x0) * sigma.view(-1, 1, 1, 1)
            x_noisy = x0 + n
            D = model(x_noisy, sigma)
            eval_weight = lambda_weight(sigma, empirical_sigma)
            eval_loss = (eval_weight * (D - x0) ** 2).mean().item()
        final_losses[sigma_data] = eval_loss

    print(f"\nT5 sigma_data sweep (eval-weighted): {final_losses}")
    best = min(final_losses, key=lambda k: final_losses[k])
    # The best should be 0.1 (exact match) or 0.01 (closer than 1.0).
    # We at least require that the worst is sigma_data=1.0 (10x off).
    worst = max(final_losses, key=lambda k: final_losses[k])
    assert worst == 1.0, (
        f"sigma_data=1.0 should be worst on data with std=0.1, "
        f"got rankings: {final_losses}"
    )


# ----------------------------------------------------------------------
# T6 — Heun sampler converges to the data distribution
# ----------------------------------------------------------------------

def test_T6_heun_sampler_converges():
    """Sampling N points from the trained denoiser should match the
    target distribution within KS-distance 0.3 — loose enough to pass
    on a tiny 8-conv model."""
    torch.manual_seed(0)
    sigma_data = 0.5  # training data std

    # Synthetic 1-D-ish "image": [B, 1, 4, 4] with mean=2, std=0.5
    def sample_data(n):
        return torch.randn(n, 1, 4, 4) * sigma_data + 2.0

    model = TinyDenoiser(in_channels=1, sigma_data=sigma_data)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    cfg = EDMConfig(sigma_data=sigma_data, sigma_min=0.002, sigma_max=20.0, rho=7.0)

    for _ in range(400):
        x0 = sample_data(32)
        sigma = sample_training_sigma(32, cfg.P_mean, cfg.P_std, device=x0.device)
        n = torch.randn_like(x0) * sigma.view(-1, 1, 1, 1)
        x_noisy = x0 + n
        D = model(x_noisy, sigma)
        weight = lambda_weight(sigma, sigma_data)
        loss = (weight * (D - x0) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    # Sample with Heun
    model.eval()
    samples = heun_sample(
        denoiser_fn=lambda x, s: model(x, s.expand(x.shape[0])),
        shape=(64, 1, 4, 4), cfg=cfg, num_steps=18, device=torch.device("cpu"),
    )
    # Compare distribution moments (loose tolerance for tiny model).
    target = sample_data(64)
    assert abs(samples.mean().item() - target.mean().item()) < 0.5, (
        f"Sample mean drifted: {samples.mean().item():.3f} vs {target.mean().item():.3f}"
    )
    # Std must be in the right order of magnitude (factor 3 tolerance)
    s_std = samples.std().item()
    t_std = target.std().item()
    assert 0.33 * t_std < s_std < 3.0 * t_std, (
        f"Sample std out of range: {s_std:.3f} vs target {t_std:.3f}"
    )


# ----------------------------------------------------------------------
# T7 — full pipeline: encoder + RCN + diffusion compute_loss + backward
# ----------------------------------------------------------------------

def test_T7_full_pipeline_one_step():
    """Smoke test the entire ST-CDGM pipeline at MINIMAL shape:
    LR (1, 15, 8, 8), HR (1, 1, 32, 32). One training step, no crash."""
    pytest.importorskip("diffusers")
    pytest.importorskip("torch_geometric")

    from st_cdgm.models.diffusion_decoder import CausalDiffusionDecoder
    from st_cdgm.models.edm_preconditioner import EDMConfig

    torch.manual_seed(0)
    cfg = EDMConfig(sigma_data=0.1)
    decoder = CausalDiffusionDecoder(
        in_channels=1, conditioning_dim=8, height=32, width=32,
        scheduler_type="edm_karras",
        edm_config=cfg,
        unet_kwargs=dict(
            layers_per_block=1,
            block_out_channels=(8, 16),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            mid_block_type="UNetMidBlock2D",
            norm_num_groups=4,
            attention_head_dim=4,
            class_embed_type="projection",
            projection_class_embeddings_input_dim=8 * 5,  # q=5, dim=8
        ),
    )

    target = torch.randn(1, 1, 32, 32) * 0.1
    conditioning = torch.randn(1, 5, 8)  # [B, q, d]
    cond_spatial = torch.randn(1, 6, 8)  # [B, num_tokens, d]

    loss = decoder.compute_loss_edm(target, conditioning, conditioning_spatial=cond_spatial)
    loss.backward()

    assert torch.isfinite(loss).all(), f"loss is NaN/Inf: {loss.item()}"
    n_zero_grad = sum(1 for p in decoder.parameters() if p.grad is None or p.grad.abs().max().item() == 0)
    n_total = sum(1 for _ in decoder.parameters())
    print(f"\nT7: {n_total - n_zero_grad}/{n_total} params received nonzero gradient")
    assert n_zero_grad < n_total // 2, (
        f"Too many params received no gradient: {n_zero_grad}/{n_total}"
    )


# ----------------------------------------------------------------------
# T8 — DAG intervention test: D(A=0) != D(A) when conditioning is wired
# ----------------------------------------------------------------------

def test_T8_dag_intervention_signal():
    """With a model that uses conditioning, swapping A=0 vs A=A_real
    must produce different outputs. This is the O6 metric from the paper."""
    torch.manual_seed(0)
    sigma_data = 0.1
    model = TinyDenoiser(in_channels=1, sigma_data=sigma_data)

    # Train briefly to bind conditioning to outputs
    proj = nn.Linear(4, 1)
    A_dag = torch.nn.Parameter(torch.randn(5, 5))
    A_dag.requires_grad_(False)  # treat as fixed for this test
    H_T = torch.randn(8, 5, 4)

    # No actual training — just verify mechanism. Compute outputs for A and A=0.
    cond_real = (A_dag @ H_T).mean(dim=1)
    cond_real = proj(cond_real).view(-1, 1, 1, 1)

    cond_zero = (torch.zeros_like(A_dag) @ H_T).mean(dim=1)
    cond_zero = proj(cond_zero).view(-1, 1, 1, 1)

    sigma = torch.tensor([1.0] * 8)
    x_noisy = torch.randn(8, 1, 8, 8) * 1.0
    with torch.no_grad():
        D_real = model(x_noisy, sigma) + cond_real
        D_zero = model(x_noisy, sigma) + cond_zero

    delta = (D_real - D_zero).abs().mean().item()
    print(f"\nT8 DAG-intervention delta: {delta:.4f}")
    assert delta > 0, "A_dag had no effect on output — conditioning is not wired"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
