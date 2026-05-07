"""
BS26 — restore plot history lists on resume.

Symptom: after a session restart, the loss-curves cell only plots
the epochs trained *in the current session*. ``val_mse_history`` and
``history`` are reinitialized to ``[]`` / ``{loss_diff_train: []}`` at
the start of each Stage in cell 50, and BS18 RESUME never repopulated
them — only the model weights and epoch counters were restored.

What we need:
- ``val_mse_stage1`` list lives inside ``payload["history"]`` (per BS18
  patcher: ``history={"val_mse_stage1": val_mse_history}``).
- Stage 2's ``history`` dict (with ``loss_diff_train`` + ``epoch_time``)
  is also stored in ``payload["history"]``.

Fix:
1. In the BS18 RESUME read block (before ``del _ck_bs18``), extract
   ``ck_history`` and stash::

       _resume_val_mse_history  = ck_history.get("val_mse_stage1", []) or []
       _resume_history_s2       = {k: v for k, v in ck_history.items()
                                   if k != "val_mse_stage1"}

2. Right after ``val_mse_history = []``, hydrate from
   ``_resume_val_mse_history`` if non-empty.

3. Right after ``history = {k: [] for k in ...}``, merge any matching
   keys from ``_resume_history_s2``.

Idempotent — sentinel ``BS26_RESTORE_HISTORY``.
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


# ---- Patch 1: extract history before `del _ck_bs18` ------------------

OLD_DEL = '''        del _ck_bs18'''

NEW_DEL = '''        # >>> BS26_RESTORE_HISTORY (extract)
        # Pull out the list-style training metrics so we can re-hydrate
        # the in-session ``val_mse_history`` and ``history`` dict after
        # they're (re)initialized below. Without this, plot cell 53 only
        # shows whatever epochs ran *in the current session*.
        _ck_history = _ck_bs18.get("history", {}) or {}
        _resume_val_mse_history = list(_ck_history.get("val_mse_stage1", []) or [])
        _resume_history_s2 = {
            k: list(v) for k, v in _ck_history.items()
            if k != "val_mse_stage1" and isinstance(v, list)
        }
        if _resume_val_mse_history or _resume_history_s2:
            _msg = []
            if _resume_val_mse_history:
                _msg.append(f"Stage1 ValMSE×{len(_resume_val_mse_history)}")
            for _k, _v in _resume_history_s2.items():
                _msg.append(f"{_k}×{len(_v)}")
            print(f"   ↳ history extracted: {', '.join(_msg)}")

        del _ck_bs18'''


# ---- Patch 2: hydrate val_mse_history right after `[]` init ----------

OLD_S1_INIT = '''val_mse_history = []
best_s1_val = math.inf
best_s1_epoch = 0
patience = int(ts_cfg.stage1.early_stop_patience)
no_improve_s1 = 0'''

NEW_S1_INIT = '''val_mse_history = []
# >>> BS26_RESTORE_HISTORY (Stage 1)
if "_resume_val_mse_history" in dir() and _resume_val_mse_history:
    val_mse_history = list(_resume_val_mse_history)
    print(f"   ↳ BS26: restored {len(val_mse_history)} ValMSE entries from checkpoint")
best_s1_val = math.inf
best_s1_epoch = 0
patience = int(ts_cfg.stage1.early_stop_patience)
no_improve_s1 = 0
# Update best_s1_val from restored history so early-stopping still has context.
if val_mse_history:
    best_s1_val = float(min(val_mse_history))
    best_s1_epoch = int(val_mse_history.index(best_s1_val) + 1)'''


# ---- Patch 3: hydrate Stage 2 ``history`` dict -----------------------

OLD_S2_INIT = '''history = {k: [] for k in ["loss_diff_train", "epoch_time"]}

for s2_epoch in range(_s2_resume_from, S2_EPOCHS):'''

NEW_S2_INIT = '''history = {k: [] for k in ["loss_diff_train", "epoch_time"]}
# >>> BS26_RESTORE_HISTORY (Stage 2)
if "_resume_history_s2" in dir() and _resume_history_s2:
    for _k, _v in _resume_history_s2.items():
        if _k in history:
            history[_k] = list(_v)
    _restored = {k: len(v) for k, v in history.items() if v}
    if _restored:
        print(f"   ↳ BS26: restored Stage 2 history: {_restored}")

for s2_epoch in range(_s2_resume_from, S2_EPOCHS):'''


def patch_cell_50() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "TWO_STAGE_TRAINING_LOOP" in s and "for s2_epoch" in s)
    if idx is None:
        print("  ! cell 50 not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS26_RESTORE_HISTORY" in src:
        print("  = cell 50 already patched (BS26)")
        return 0

    n = 0
    for old, new, label in (
        (OLD_DEL, NEW_DEL, "extract block before del"),
        (OLD_S1_INIT, NEW_S1_INIT, "Stage 1 hydration"),
        (OLD_S2_INIT, NEW_S2_INIT, "Stage 2 hydration"),
    ):
        if old in src:
            src = src.replace(old, new, 1)
            n += 1
            print(f"  ~ cell 50: {label}")
        else:
            print(f"  ! cell 50: pattern for '{label}' not found")
    if n > 0:
        cells[idx]["source"] = src.splitlines(keepends=True)
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def main() -> int:
    print("=== BS26 : restore history on resume (cell 50) ===")
    n = patch_cell_50()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
