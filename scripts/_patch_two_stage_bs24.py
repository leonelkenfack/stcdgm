"""
BS24 — fix lazy-param crash + resume print NameError.

Two issues exposed by BS22+BS23 working correctly:

1. ``_persist_load_state_dict`` crashed on uninitialized lazy parameters
   (SAGEConv ``in_channels=-1``) because BS23's per-tensor shape check
   accesses ``target_sd[tk].shape`` which raises on
   ``UninitializedParameter``. Encoder weights were lost.

   Fix: wrap shape access in ``try/except``. If the live param is
   uninitialized, accept the saved tensor — PyTorch's lazy module
   ``_load_from_state_dict`` hook will materialize from the saved shape.

2. The BS18 resume banner referenced ``ts_cfg.stage1.epochs_max`` but
   ``ts_cfg = CONFIG.two_stage`` is assigned *after* the resume block.
   Pre-existing latent bug — never manifested because earlier runs
   never had ``_s1_resume_from > 0``. Now that resume works, this branch
   fires → NameError, halting the cell.

   Fix: substitute ``CONFIG.two_stage`` directly (equivalent value,
   available at this point in cell execution).

Idempotent — sentinels ``BS24 — lazy-param tolerance`` and
``BS24_RESUME_PRINT_CONFIG``.
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


# =====================================================================
# Cell 47 patch — lazy param tolerance in _persist_load_state_dict
# =====================================================================

OLD_LAZY = '''        v = stripped_sd[norm_tk]
        # BS23: per-tensor shape check.
        live_shape = tuple(target_sd[tk].shape) if hasattr(target_sd[tk], "shape") else None
        ckpt_shape = tuple(v.shape) if hasattr(v, "shape") else None
        if live_shape is not None and ckpt_shape is not None and live_shape != ckpt_shape:
            skipped_shape.append((tk, ckpt_shape, live_shape))
            continue
        matched[tk] = v'''

NEW_LAZY = '''        v = stripped_sd[norm_tk]
        # BS23: per-tensor shape check.
        # BS24 — lazy-param tolerance: ``UninitializedParameter`` (e.g.
        # SAGEConv with in_channels=-1) raises on ``.shape`` access.
        # If live shape can't be read, accept the saved tensor and let
        # PyTorch's lazy module hook materialize during load_state_dict.
        try:
            live_shape = tuple(target_sd[tk].shape) if hasattr(target_sd[tk], "shape") else None
        except (RuntimeError, ValueError):
            live_shape = None  # uninitialized lazy param
        ckpt_shape = tuple(v.shape) if hasattr(v, "shape") else None
        if live_shape is not None and ckpt_shape is not None and live_shape != ckpt_shape:
            skipped_shape.append((tk, ckpt_shape, live_shape))
            continue
        matched[tk] = v'''


def patch_cell_47() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "def _persist_load_state_dict" in s
                     and "PERSIST_HELPERS" in s)
    if idx is None:
        print("  ! cell 47 not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS24 — lazy-param tolerance" in src:
        print("  = cell 47 already patched (BS24)")
        return 0
    if OLD_LAZY not in src:
        print("  ! cell 47: BS23 baseline pattern not found — apply BS23 first")
        return 0
    src = src.replace(OLD_LAZY, NEW_LAZY, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  ~ cell 47: BS24 lazy-param try/except wrapped around shape access")
    return 1


# =====================================================================
# Cell 50 patch — fix ts_cfg NameError in resume banner
# =====================================================================

OLD_PRINT = '''if _s1_resume_from > 0 or _s2_resume_from > 0:
    print(
        f"🔁 BS18 resume: Stage1 done={_s1_resume_from}/{int(ts_cfg.stage1.epochs_max)}, "
        f"Stage2 done={_s2_resume_from}/{int(ts_cfg.stage2.epochs_max)}, "
        f"calib={'skip' if _skip_calibration else 'redo'}, "
        f"ablation={'skip' if _skip_ablation else 'redo'}"
    )'''

NEW_PRINT = '''# >>> TWO_STAGE_BS24_RESUME_PRINT_CONFIG
# Use CONFIG.two_stage directly — ``ts_cfg`` alias is assigned *after*
# this block. Pre-existing latent NameError exposed once resume works.
if _s1_resume_from > 0 or _s2_resume_from > 0:
    print(
        f"🔁 BS18 resume: Stage1 done={_s1_resume_from}/{int(CONFIG.two_stage.stage1.epochs_max)}, "
        f"Stage2 done={_s2_resume_from}/{int(CONFIG.two_stage.stage2.epochs_max)}, "
        f"calib={'skip' if _skip_calibration else 'redo'}, "
        f"ablation={'skip' if _skip_ablation else 'redo'}"
    )'''


def patch_cell_50() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "TWO_STAGE_TRAINING_LOOP" in s and "for s2_epoch" in s)
    if idx is None:
        print("  ! cell 50 not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "TWO_STAGE_BS24_RESUME_PRINT_CONFIG" in src:
        print("  = cell 50 already patched (BS24)")
        return 0
    if OLD_PRINT not in src:
        print("  ! cell 50: resume print pattern not found")
        return 0
    src = src.replace(OLD_PRINT, NEW_PRINT, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  ~ cell 50: ts_cfg → CONFIG.two_stage in resume print")
    return 1


def main() -> int:
    print("=== BS24a : lazy-param tolerance (cell 47) ===")
    n1 = patch_cell_47()
    print("\n=== BS24b : ts_cfg NameError fix (cell 50) ===")
    n2 = patch_cell_50()
    print(f"\n{n1 + n2} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
