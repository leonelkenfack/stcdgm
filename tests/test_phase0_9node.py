"""Phase 0 (option C) — sanity tests for the opt-in 9-node extension.

These tests guard the deterministic, GPU-free parts of the 9-node DAG
extension (physics prior + graph builder) WITHOUT touching the 6-node
defaults that the current baseline relies on.

Spec : path_c_plus/audit/PHASE0_SPEC_9NODE.md
Validated-learnable by : path_c_plus/scripts/smoke_9node_extended_dag.ipynb
"""

from __future__ import annotations

import pytest

from st_cdgm.training.physics_prior import (
    EXPECTED_EDGES,
    EXPECTED_EDGES_9NODE,
    VAR_LABELS,
    VAR_LABELS_9NODE,
    build_physical_mask,
)


# ---------------------------------------------------------------------------
# Physics prior — 6-node defaults must remain untouched (non-regression)
# ---------------------------------------------------------------------------
def test_6node_defaults_unchanged():
    assert len(VAR_LABELS) == 6
    assert len(EXPECTED_EDGES) == 5
    G = build_physical_mask(num_vars=6)
    assert tuple(G.shape) == (6, 6)
    assert int((G != 0).sum().item()) == 5
    assert float(G.sum().item()) == 5.0  # all +1


# ---------------------------------------------------------------------------
# Physics prior — 9-node extension
# ---------------------------------------------------------------------------
def test_9node_labels_canonical_order():
    # Convention §1 : 5 dry metapaths, then humid chain Q850/W500/IVT,
    # then SP_HR static LAST (index 8).
    assert len(VAR_LABELS_9NODE) == 9
    assert VAR_LABELS_9NODE[:5] == VAR_LABELS[:5]
    assert VAR_LABELS_9NODE[5:8] == ["Q850", "W500", "IVT"]
    assert VAR_LABELS_9NODE[8] == "SP_HR"


def test_9node_mask_has_10_physical_edges():
    assert len(EXPECTED_EDGES_9NODE) == 10
    G = build_physical_mask(
        num_vars=9,
        var_labels=VAR_LABELS_9NODE,
        expected_edges=EXPECTED_EDGES_9NODE,
    )
    assert tuple(G.shape) == (9, 9)
    assert int((G != 0).sum().item()) == 10
    assert float(G.sum().item()) == 10.0  # all +1


def test_9node_humid_chain_targets_surface():
    G = build_physical_mask(
        num_vars=9,
        var_labels=VAR_LABELS_9NODE,
        expected_edges=EXPECTED_EDGES_9NODE,
    )
    idx = {lab: i for i, lab in enumerate(VAR_LABELS_9NODE)}
    # Held-Soden closure : W500 -> SP_HR and IVT -> SP_HR
    assert G[idx["W500"], idx["SP_HR"]].item() == 1.0
    assert G[idx["IVT"], idx["SP_HR"]].item() == 1.0
    # Q850 -> IVT (moisture feeds vapor transport)
    assert G[idx["Q850"], idx["IVT"]].item() == 1.0
    # no self-loops
    for i in range(9):
        assert G[i, i].item() == 0.0


def test_9node_edges_reference_only_known_labels():
    labels = set(VAR_LABELS_9NODE)
    for src, tgt, sign in EXPECTED_EDGES_9NODE:
        assert src in labels, f"unknown src {src}"
        assert tgt in labels, f"unknown tgt {tgt}"
        assert sign in (-1, +1)


# ---------------------------------------------------------------------------
# Graph builder — opt-in extended_9node flag
# ---------------------------------------------------------------------------
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")
from st_cdgm.models.graph_builder import HeteroGraphBuilder  # noqa: E402


def test_graph_builder_6node_default_has_no_humid_nodes():
    b = HeteroGraphBuilder(lr_shape=(4, 5), hr_shape=(8, 10))
    assert b.dynamic_node_types == ["GP850", "GP500", "GP250"]
    assert not b.extended_9node


def test_graph_builder_extended_adds_3_nodes_and_spatial_edges():
    b = HeteroGraphBuilder(lr_shape=(4, 5), hr_shape=(8, 10), extended_9node=True)
    assert b.dynamic_node_types == [
        "GP850", "GP500", "GP250", "Q850", "W500", "IVT",
    ]
    data, report = b.build()
    for nt in ("Q850", "W500", "IVT"):
        assert data[nt].num_nodes == b.num_nodes_lr
        assert (nt, "spat_adj", nt) in data.edge_types
        assert nt in report.edges_spatial
    # Humid-chain directed edges must NOT be baked as topology (carried by A_dag)
    assert ("GP850", "causes", "Q850") not in data.edge_types
    assert ("Q850", "causes", "IVT") not in data.edge_types
