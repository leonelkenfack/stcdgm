"""
BS33b — bump eval K_SAMPLES 4 → 16 (paper Tab.1 ensemble size 8-16).

The Bomgni 2026 paper (oracle.tex Tab.1, "Inference / evaluation"
block) states ensemble size 8-16 for evaluation. Our cell 60 was at
K_SAMPLES=4 (a compromise between speed and statistics chosen during
debugging). Now that BS32b has eval at ~5s/batch instead of 28s/batch,
the cost is negligible (16 batches × 16 samples × 5s ≈ 6 min vs
current 1.5 min).

Idempotent — sentinel ``BS33b_K_SAMPLES``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


OLD = '''N_TEST_BATCHES = 16
K_SAMPLES = 4              # ensemble pour CRPS/spread
N_INTERVENTION = 4         # nb de batches pour l'ablation μ_HR'''


NEW = '''# >>> BS33b_K_SAMPLES : bump 4 → 16 (paper Tab.1 says 8-16).
# Eval cost reasonable post-BS32b (~6 min for 16×16 samples vs 1.5 min before).
N_TEST_BATCHES = 16
K_SAMPLES = 16             # ensemble pour CRPS/spread (was 4 → 16)
N_INTERVENTION = 4         # nb de batches pour l'ablation μ_HR'''


def patch_cell_60() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "FINAL_VALIDATION (BS30" in s)
    if idx is None:
        print("  ! FINAL_VALIDATION cell not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS33b_K_SAMPLES" in src:
        print(f"  = cell {idx} already patched (BS33b)")
        return 0
    if OLD not in src:
        print(f"  ! cell {idx}: K_SAMPLES anchor not found")
        return 0
    src = src.replace(OLD, NEW, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {idx}: K_SAMPLES 4 → 16")
    return 1


def main() -> int:
    print("=== BS33b : K_SAMPLES 4 → 16 (paper-aligned ensemble eval) ===")
    n = patch_cell_60()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
