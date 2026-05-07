"""
BS29 — fix scope bug in BS28 generate_prediction.

Inside a function body, ``dir()`` returns local names only (no
``globals()`` lookup). BS28 used::

    if _causal_concat and "regression_head" in dir() and regression_head is not None:

That check is *always False* because ``regression_head`` is defined at
module/cell scope, not inside ``generate_prediction``. So the
``mu_HR`` / ``baseline_log`` branch never executed and
``diffusion.sample()`` raised the same ValueError BS28 was meant to
prevent.

Fix: resolve via ``globals().get("regression_head")``.
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


OLD_GUARD = '''    if _causal_concat and "regression_head" in dir() and regression_head is not None:
        # Build mu_HR + baseline_log just like Stage-2 training.
        target_shape = batch["residual"][-1].to(device).shape  # (C, H, W) or (B, C, H, W)
        mu_HR = regression_head(H_T)'''

NEW_GUARD = '''    # >>> BS29_FIX_SCOPE
    # ``regression_head`` is a module-level / cell-level global; ``dir()``
    # inside a function only sees locals, so the BS28 guard never fired.
    # Resolve through ``globals()`` instead.
    _rh = globals().get("regression_head", None)
    if _causal_concat and _rh is not None:
        # Build mu_HR + baseline_log just like Stage-2 training.
        target_shape = batch["residual"][-1].to(device).shape  # (C, H, W) or (B, C, H, W)
        mu_HR = _rh(H_T)'''


def patch_cell_57() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "BS28_GENERATE_PREDICTION" in s)
    if idx is None:
        print("  ! cell with BS28_GENERATE_PREDICTION not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS29_FIX_SCOPE" in src:
        print(f"  = cell {idx} already patched (BS29)")
        return 0
    if OLD_GUARD not in src:
        print(f"  ! cell {idx}: BS28 guard pattern not found — was BS28 customized?")
        return 0
    src = src.replace(OLD_GUARD, NEW_GUARD, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {idx}: regression_head resolved via globals()")
    return 1


def main() -> int:
    print("=== BS29 : fix scope bug in BS28 generate_prediction ===")
    n = patch_cell_57()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
