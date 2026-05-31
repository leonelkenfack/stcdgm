"""Static checks for the non-causal CorrDiff baseline protocol.

This is intentionally lightweight and GPU-free. It catches the regression that
motivated the fix: a notebook claiming ``run_variant=noncausal`` while still
executing the causal DAG training loop.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _nb_text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _parse_notebook_cells(name: str) -> None:
    nb = json.loads((ROOT / name).read_text(encoding="utf-8"))
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        src = "\n".join(
            line
            for line in src.splitlines()
            if not line.lstrip().startswith(("!", "%"))
        )
        if src.strip():
            ast.parse(src, filename=f"{name}:cell-{i}")
    print(f"[OK] {name}: code cells parse")


def main() -> None:
    for nbname in [
        "st_cdgm_noncausal_training.ipynb",
        "st_cdgm_training_evaluation.ipynb",
        "st_cdgm_validation_inference.ipynb",
        "st_cdgm_results_presentation.ipynb",
        "st_cdgm_dag_intervention_test.ipynb",
        "st_cdgm_causal_vs_noncausal_comparison.ipynb",
    ]:
        _parse_notebook_cells(nbname)

    nc = _nb_text("st_cdgm_noncausal_training.ipynb")
    required_markers = [
        "train_epoch_stage1_noncausal",
        "precompute_stage1_outputs_variant",
        "validate_stage1_gate",
        "STAGE 1 — Non-causal CorrDiff Mean Prediction",
        "BS35 causal DAG ablation skipped",
        # Variant-aware init : cell 36 doit skip encoder/RCN en noncausal.
        "Encoder/RCN skip (noncausal)",
        # Resume noncausal : cell 51 doit lire epoch_last.pth.
        "NONCAUSAL_RESUME",
        # Save/eval guards (cells 56/61) : encoder/rcn None-tolerants.
        "NONCAUSAL_VARIANT_SAFE",
    ]
    missing = [m for m in required_markers if m not in nc]
    if missing:
        raise AssertionError(f"noncausal notebook missing protocol markers: {missing}")

    if "force_zero_dag_train=True" in nc.lower():
        raise AssertionError("noncausal notebook still advertises force_zero_dag_train=True")

    # >>> NONCAUSAL_CELL36_GUARD
    # Pattern qui interdit la resurgence du bug 'encoder + RCN construits sans
    # variant guard'. Le notebook noncausal NE doit PAS contenir les chaines de
    # construction non gardees ; on cherche le pattern bare (= debut de ligne).
    forbidden_bare_patterns = [
        "\nencoder = IntelligibleVariableEncoder(",
        "\nrcn_cell = RCNCell(",
    ]
    for bad in forbidden_bare_patterns:
        if bad in nc:
            raise AssertionError(
                f"noncausal notebook re-introduces causal build path : "
                f"{bad.strip()!r} (must be guarded by RUN_VARIANT check)"
            )

    # >>> NONCAUSAL_MARKDOWN_GUARD
    # Cell 60 markdown ne doit plus annoncer 'Intervention DAG' comme titre.
    nb_nc = json.loads((ROOT / "st_cdgm_noncausal_training.ipynb").read_text(encoding="utf-8"))
    for i, cell in enumerate(nb_nc["cells"]):
        if cell.get("cell_type") != "markdown":
            continue
        src = "".join(cell.get("source", []))
        if "Validation finale + Intervention DAG" in src:
            raise AssertionError(
                f"noncausal notebook markdown cell {i} still titles itself "
                f"'Validation finale + Intervention DAG' (must be noncausal-aligned)"
            )

    main_nb = _nb_text("st_cdgm_training_evaluation.ipynb")
    if "is causal-only" not in main_nb:
        raise AssertionError("main training notebook must reject run_variant='noncausal'")

    validation_nb = _nb_text("st_cdgm_validation_inference.ipynb")
    for marker in ["RegressionMeanPredictor", "DAG intervention skipped", "SHD/DAG prior test skipped"]:
        if marker not in validation_nb:
            raise AssertionError(f"validation notebook missing marker: {marker}")

    comparison_nb = _nb_text("st_cdgm_causal_vs_noncausal_comparison.ipynb")
    if "artefact probablement produit avant le fix" not in comparison_nb:
        raise AssertionError("comparison notebook missing contamination warning")

    print("\nALL NONCAUSAL PROTOCOL CHECKS PASSED")


if __name__ == "__main__":
    main()
