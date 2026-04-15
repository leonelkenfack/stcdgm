"""
Regression tests locking in the Sprint 1 / ORACLE hyperplan fixes.

These tests are deliberately small and synthetic (no NetCDF) so they can run
on any CI/dev box in under a second. Each test pins a specific bug from
``ORACLE_HYPERPLAN_ANALYSIS.md`` so future refactors cannot silently re-break
it.

Covered fixes:
- B1 : ``RCNCell.reconstruction_decoder`` now accepts H_prev flattened
       ([N, num_vars * hidden_dim]) and produces a non-trivial L_rec.
- B1+: the reconstruction gradient reaches the structural MLPs (i.e. L_rec
       is not a dead branch).
- DAG prior init: ``dag_prior`` is honoured and ``A_dag`` starts close to
       the supplied matrix (plus the requested noise).
- sigma_r NaN: ``compute_ensemble_sigma_r`` returns a finite value for a
       non-degenerate ensemble and a non-constant target.
"""

from __future__ import annotations

import math

import pytest
import torch

from st_cdgm.evaluation.evaluation_xai import compute_ensemble_sigma_r
from st_cdgm.models.causal_rcn import RCNCell


def test_b1_recon_decoder_accepts_hidden_state():
    """Regression for B1: Linear(num_vars*hidden_dim, recon_dim)."""
    q, hidden, driver, N = 5, 16, 8, 12
    cell = RCNCell(
        num_vars=q,
        hidden_dim=hidden,
        driver_dim=driver,
        reconstruction_dim=driver,
    )
    # Expected shape of the recon decoder weight matrix.
    assert cell.reconstruction_decoder is not None
    assert cell.reconstruction_decoder.in_features == q * hidden
    assert cell.reconstruction_decoder.out_features == driver

    H_prev = torch.randn(q, N, hidden)
    u_t = torch.randn(N, driver)

    H_next, recon, A_masked = cell(H_prev, u_t)
    assert recon is not None, "Recon decoder must run when recon_dim is set."
    assert recon.shape == (N, driver)
    assert torch.isfinite(recon).all()


def test_b1_recon_loss_reaches_structural_mlps():
    """
    With the B1 fix in place, L_rec must produce non-zero gradients on the
    structural MLP weights (``struct_W1``). Before the fix L_rec crashed or
    was silently zero, which left A_dag without its main supervision signal.
    """
    q, hidden, driver, N = 3, 8, 4, 6
    cell = RCNCell(
        num_vars=q,
        hidden_dim=hidden,
        driver_dim=driver,
        reconstruction_dim=driver,
    )
    H_prev = torch.randn(q, N, hidden, requires_grad=False)
    u_t = torch.randn(N, driver)

    _, recon, _ = cell(H_prev, u_t)
    loss = torch.nn.functional.mse_loss(recon, u_t)
    loss.backward()

    # struct_W1 feeds the recon decoder through H_next, so it must get grad.
    assert cell.struct_W1.grad is not None
    assert cell.struct_W1.grad.abs().sum().item() > 0.0
    # Recon decoder itself obviously gets grad.
    assert cell.reconstruction_decoder.weight.grad is not None
    assert cell.reconstruction_decoder.weight.grad.abs().sum().item() > 0.0


def test_dag_prior_init_honoured():
    """
    Passing ``dag_prior`` must initialise ``A_dag`` near the prior (not
    Xavier random) so that the RCN starts from a physically meaningful
    topology. We allow ``+- 4 * noise`` tolerance on the off-diagonal.
    """
    q = 5
    prior = torch.tensor([
        [0.0, 0.2, 0.0, 0.0, 0.0],
        [0.2, 0.0, 0.2, 0.0, 0.0],
        [0.0, 0.2, 0.0, 0.2, 0.0],
        [0.0, 0.0, 0.2, 0.0, 0.2],
        [0.0, 0.0, 0.0, 0.2, 0.0],
    ])
    noise = 0.01
    torch.manual_seed(0)
    cell = RCNCell(
        num_vars=q,
        hidden_dim=8,
        driver_dim=4,
        reconstruction_dim=None,
        dag_prior=prior,
        dag_prior_noise=noise,
    )
    A = cell.A_dag.detach()
    # Diagonal must be zero.
    assert torch.allclose(torch.diagonal(A), torch.zeros(q), atol=1e-6)
    # Off-diagonal must be close to the prior within a few noise sigmas.
    off_mask = ~torch.eye(q, dtype=torch.bool)
    diff = (A - prior)[off_mask].abs()
    assert diff.max().item() < 4.0 * noise + 1e-6, (
        f"DAG init deviates too far from prior: max diff {diff.max().item()} "
        f"> 4*noise={4*noise}"
    )


def test_dag_prior_init_disabled_falls_back_to_xavier():
    """Without ``dag_prior``, A_dag should be the Xavier-random tensor."""
    torch.manual_seed(0)
    cell = RCNCell(num_vars=4, hidden_dim=6, driver_dim=3)
    A = cell.A_dag.detach()
    # Xavier will not produce an all-zero off-diagonal pattern.
    off_mask = ~torch.eye(4, dtype=torch.bool)
    assert A[off_mask].abs().mean().item() > 0.05


def test_sigma_r_finite_for_real_ensemble():
    """Regression for sigma_r NaN: a proper ensemble + non-trivial target."""
    torch.manual_seed(1)
    ensemble = [torch.randn(1, 1, 16, 16) for _ in range(5)]
    target = torch.randn(1, 1, 16, 16)
    out = compute_ensemble_sigma_r(ensemble, target)
    assert math.isfinite(out["sigma_r"]), out
    assert math.isfinite(out["sigma_r_pct"]), out
    assert out["sigma_r"] > 0.0


def test_sigma_r_nan_on_degenerate_ensemble():
    """A singleton ensemble legitimately yields NaN (no spread defined)."""
    ensemble = [torch.randn(1, 1, 8, 8)]
    target = torch.randn(1, 1, 8, 8)
    out = compute_ensemble_sigma_r(ensemble, target)
    assert math.isnan(out["sigma_r"])


def test_sigma_r_nan_on_constant_target():
    """A constant target has zero spatial std → ratio is undefined (NaN)."""
    ensemble = [torch.randn(1, 1, 8, 8) for _ in range(3)]
    target = torch.full((1, 1, 8, 8), 2.5)
    out = compute_ensemble_sigma_r(ensemble, target)
    assert math.isnan(out["sigma_r"])


def test_dag_spectral_projection_keeps_acyclicity():
    """
    Sanity check on the hard spectral projection that complements DAGMA.
    The projection must leave the spectral radius of A ∘ A strictly below
    the configured max_radius, and leave the diagonal at zero.
    """
    q = 6
    cell = RCNCell(num_vars=q, hidden_dim=4, driver_dim=3)
    # Blow up A_dag so projection has to kick in.
    with torch.no_grad():
        cell.A_dag.data = torch.randn(q, q) * 3.0
    max_radius = 0.9
    cell.project_dag_spectral(max_radius=max_radius)

    A = cell.A_dag.data
    assert torch.allclose(torch.diagonal(A), torch.zeros(q), atol=1e-6)
    # Gershgorin bound used inside project_dag_spectral.
    rho_bound = (A * A).sum(dim=1).max().item()
    assert rho_bound <= max_radius + 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
