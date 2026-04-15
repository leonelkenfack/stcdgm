"""
Regression tests for Sprint 2 — « recâbler le gradient causal ».

Each test pins one Sprint 2 component from ORACLE_HYPERPLAN_ANALYSIS.md §2 so
future changes cannot silently break the DAG↔diffusion coupling.

Covered:
- ``dag_grad_gate`` straight-through gating (gate=0 reproduces Sprint 1
  detached behaviour, gate=1 restores full gradient flow to A_dag).
- ``set_dag_grad_gate`` clamps to [0, 1] and rejects non-finite values.
- ``CausalConditioningProjector`` appends DAG tokens to spatial tokens,
  and gradient flows from the output tokens back to ``A_dag``.
- ``HRTargetIdentifiabilityHead`` produces stable stats and gradient on
  the causal state (identifiability pressure on the SCM parameters).
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from st_cdgm.models.causal_rcn import RCNCell
from st_cdgm.models.intelligible_encoder import (
    CausalConditioningProjector,
    HRTargetIdentifiabilityHead,
    SpatialConditioningProjector,
)


# ---------------------------------------------------------------------------
# dag_grad_gate
# ---------------------------------------------------------------------------


def _make_cell(q: int = 3, hidden: int = 8, driver: int = 4) -> RCNCell:
    return RCNCell(
        num_vars=q,
        hidden_dim=hidden,
        driver_dim=driver,
        reconstruction_dim=driver,
    )


def test_dag_grad_gate_default_is_zero():
    """Sprint 2: the gate must boot at 0 so cold-start reproduces Sprint 1."""
    cell = _make_cell()
    assert float(cell.dag_grad_gate.item()) == pytest.approx(0.0)


def test_set_dag_grad_gate_clamps():
    cell = _make_cell()
    cell.set_dag_grad_gate(2.5)
    assert float(cell.dag_grad_gate.item()) == pytest.approx(1.0)
    cell.set_dag_grad_gate(-0.3)
    assert float(cell.dag_grad_gate.item()) == pytest.approx(0.0)
    cell.set_dag_grad_gate(0.4)
    assert float(cell.dag_grad_gate.item()) == pytest.approx(0.4)


def test_set_dag_grad_gate_rejects_nan():
    cell = _make_cell()
    with pytest.raises(ValueError):
        cell.set_dag_grad_gate(float("nan"))
    with pytest.raises(ValueError):
        cell.set_dag_grad_gate(float("inf"))


def test_gate_zero_blocks_gradient_to_A_dag_via_state():
    """
    With gate=0 the structural path is fully detached from A_dag, so any
    loss computed purely from H_next must produce a zero (or ``None``)
    gradient on A_dag. This is the Sprint 1 safety property we want to
    preserve as a fallback.
    """
    q, hidden, driver, N = 3, 6, 3, 5
    cell = _make_cell(q, hidden, driver)
    cell.set_dag_grad_gate(0.0)
    H_prev = torch.randn(q, N, hidden)
    u_t = torch.randn(N, driver)
    H_next, _, _ = cell(H_prev, u_t)
    # Loss that only depends on H_next — no L_rec, no L_dag, no L_dagma.
    loss = (H_next ** 2).mean()
    loss.backward()
    grad = cell.A_dag.grad
    if grad is not None:
        assert grad.abs().sum().item() == pytest.approx(0.0, abs=1e-6)


def test_gate_one_unlocks_gradient_to_A_dag_via_state():
    """
    With gate=1 the same loss must push a non-zero gradient into A_dag,
    because the attached branch is now fully weighted. This is the whole
    point of Sprint 2: L_gen can now train the DAG.
    """
    q, hidden, driver, N = 3, 6, 3, 5
    cell = _make_cell(q, hidden, driver)
    cell.set_dag_grad_gate(1.0)
    H_prev = torch.randn(q, N, hidden)
    u_t = torch.randn(N, driver)
    H_next, _, _ = cell(H_prev, u_t)
    loss = (H_next ** 2).mean()
    loss.backward()
    grad = cell.A_dag.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0


def test_gate_zero_preserves_forward_output():
    """
    Straight-through gating must not change the *values* — only the
    gradient pathway. Forward output with gate=0 and gate=1 on the same
    input must match exactly.
    """
    q, hidden, driver, N = 3, 6, 3, 5
    cell = _make_cell(q, hidden, driver)
    H_prev = torch.randn(q, N, hidden)
    u_t = torch.randn(N, driver)

    cell.set_dag_grad_gate(0.0)
    with torch.no_grad():
        H_a, _, _ = cell(H_prev, u_t)
    cell.set_dag_grad_gate(1.0)
    with torch.no_grad():
        H_b, _, _ = cell(H_prev, u_t)
    assert torch.allclose(H_a, H_b, atol=1e-6)


# ---------------------------------------------------------------------------
# CausalConditioningProjector
# ---------------------------------------------------------------------------


def test_causal_projector_shape():
    q, hidden, cond = 5, 16, 12
    lat, lon = 4, 5
    th, tw = 2, 3
    proj = CausalConditioningProjector(
        num_vars=q,
        hidden_dim=hidden,
        conditioning_dim=cond,
        lr_shape=(lat, lon),
        target_shape=(th, tw),
        num_dag_tokens=2,
    )
    state = torch.randn(q, lat * lon, hidden)
    A = torch.randn(q, q, requires_grad=True)
    out = proj(state, A)
    # Spatial tokens = q * th * tw ; DAG tokens = 2.
    assert out.shape == (1, q * th * tw + 2, cond)


def test_causal_projector_grad_reaches_A_dag():
    """The whole point: a loss on the projector output must produce a
    non-zero gradient on ``A_dag``."""
    q, hidden, cond = 3, 8, 8
    lat, lon = 3, 3
    proj = CausalConditioningProjector(
        num_vars=q,
        hidden_dim=hidden,
        conditioning_dim=cond,
        lr_shape=(lat, lon),
        target_shape=(2, 2),
        num_dag_tokens=1,
    )
    state = torch.randn(q, lat * lon, hidden, requires_grad=False)
    A = torch.randn(q, q, requires_grad=True)
    out = proj(state, A)
    loss = (out ** 2).mean()
    loss.backward()
    assert A.grad is not None
    assert A.grad.abs().sum().item() > 0.0


def test_causal_projector_rejects_wrong_A_shape():
    q = 4
    proj = CausalConditioningProjector(
        num_vars=q,
        hidden_dim=4,
        conditioning_dim=4,
        lr_shape=(2, 2),
        target_shape=(1, 1),
    )
    state = torch.randn(q, 4, 4)
    A = torch.randn(3, 3)  # wrong q
    with pytest.raises(ValueError):
        proj(state, A)


def test_causal_projector_intervention_changes_output():
    """
    Editing ``A_dag`` (simulating a do-intervention) must change the
    projector output — otherwise the UNet would see the same tokens
    regardless of the causal structure, which defeats the purpose.
    """
    q, hidden, cond = 4, 8, 6
    proj = CausalConditioningProjector(
        num_vars=q,
        hidden_dim=hidden,
        conditioning_dim=cond,
        lr_shape=(2, 3),
        target_shape=(1, 1),
    )
    state = torch.randn(q, 6, hidden)
    A1 = torch.eye(q) * 0.0           # null DAG
    A2 = torch.eye(q)                 # identity DAG (not acyclic but OK for test)
    with torch.no_grad():
        out1 = proj(state, A1)
        out2 = proj(state, A2)
    # Spatial tokens are identical (they don't see A); DAG tokens differ.
    spatial_len = q * 1 * 1
    assert torch.allclose(out1[:, :spatial_len], out2[:, :spatial_len], atol=1e-6)
    assert not torch.allclose(out1[:, spatial_len:], out2[:, spatial_len:], atol=1e-6)


# ---------------------------------------------------------------------------
# HRTargetIdentifiabilityHead
# ---------------------------------------------------------------------------


def test_hr_ident_head_output_shape():
    head = HRTargetIdentifiabilityHead(num_vars=5, hidden_dim=16)
    state = torch.randn(1, 5, 16)  # [batch, q, hidden]
    out = head(state)
    assert out.shape == (1, 4)  # default stats {mean, std, p95, p99}


def test_hr_ident_head_extract_stats_known_values():
    target = torch.zeros(1, 1, 8, 8)
    target[0, 0, 0, 0] = 10.0
    target[0, 0, 0, 1] = 5.0
    stats = HRTargetIdentifiabilityHead.extract_target_stats(
        target, stats=("mean", "max")
    )
    assert stats.shape == (1, 2)
    # Only 2 non-zero pixels among 64 → mean = 15/64 ≈ 0.234, max = 10.
    assert stats[0, 0].item() == pytest.approx(15.0 / 64.0, abs=1e-4)
    assert stats[0, 1].item() == pytest.approx(10.0, abs=1e-4)


def test_hr_ident_head_grad_to_state():
    """
    Gradient from the head's output must flow back to the causal state.
    When this state is itself the RCN output, that gradient pressurises
    struct_W, GRU and (with dag_grad_gate > 0) A_dag.
    """
    head = HRTargetIdentifiabilityHead(num_vars=3, hidden_dim=4)
    state = torch.randn(1, 3, 4, requires_grad=True)
    pred = head(state)
    target = torch.tensor([[0.5, 0.2, 1.0, 1.5]])
    loss = nn.functional.smooth_l1_loss(pred, target)
    loss.backward()
    assert state.grad is not None
    assert state.grad.abs().sum().item() > 0.0


def test_hr_ident_head_handles_nan_pixels():
    """Geographic NaN masks must not poison the statistic extraction."""
    target = torch.full((1, 1, 8, 8), float("nan"))
    target[0, 0, :, 0] = 1.0  # one valid column
    stats = HRTargetIdentifiabilityHead.extract_target_stats(
        target, stats=("mean",)
    )
    assert math.isfinite(stats[0, 0].item())
    assert stats[0, 0].item() == pytest.approx(1.0, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
