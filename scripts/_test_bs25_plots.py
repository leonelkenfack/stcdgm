"""Smoke test for BS25 two-stage-aware loss curves cell."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"


def get_cell_53_source() -> str:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        s = "".join(c.get("source", []))
        if "BS25_PLOTS_TWO_STAGE" in s:
            return s
    raise RuntimeError("BS25 cell not found")


def run_with(globals_dict):
    src = get_cell_53_source()
    # Strip plt.show() to avoid GUI
    src = src.replace("plt.show()", "pass  # show suppressed")
    exec(src, globals_dict)


def case_two_stage_only_s1():
    print("--- case: Stage 1 only (val_mse_history populated) ---")
    g = {"val_mse_history": [0.0498, 0.0486, 0.0485, 0.0490, 0.0492, 0.0489, 0.0488],
         "history": None}
    run_with(g)


def case_two_stage_both():
    print("\n--- case: Stage 1 + Stage 2 ---")
    g = {"val_mse_history": [0.05, 0.048, 0.047],
         "history": {"loss_diff_train": [0.6, 0.45, 0.30, 0.22, 0.20], "epoch_time": [120.5, 119.2, 118.0, 117.5, 117.2]}}
    run_with(g)


def case_legacy():
    print("\n--- case: legacy single-stage history ---")
    g = {"history": {
        "loss": [1.5, 1.2, 0.9, 0.6],
        "loss_gen": [0.8, 0.6, 0.45, 0.30],
        "loss_rec": [0.5, 0.4, 0.30, 0.20],
        "loss_dag": [0.2, 0.1, 0.05, 0.01],
    }}
    run_with(g)


def case_empty():
    print("\n--- case: nothing populated ---")
    g = {"history": None}
    run_with(g)


if __name__ == "__main__":
    case_two_stage_only_s1()
    case_two_stage_both()
    case_legacy()
    case_empty()
    print("\nAll BS25 smoke cases ran without raising.")
