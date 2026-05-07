"""
BS27 — fix smoke-test cell 58 to prioritize current-format checkpoints
and use the robust BS22+BS23+BS24 loader.

Symptoms:
- Cell picks ``st_cdgm_checkpoint_best.pth`` (old single-stage save with
  a different UNet architecture) instead of ``epoch_last.pth`` /
  ``epoch_best.pth`` (current two-stage saves).
- Even when the right file is found, it calls
  ``diff_target.load_state_dict(strip_torch_compile_prefix(...))``
  with the *strict* default → crashes on nested ``_orig_mod`` mismatch
  (UNet inside the diffusion gets ``_orig_mod`` only at load time).

Patch:
1. Reorder ``candidates`` so ``epoch_last.pth`` / ``epoch_best.pth``
   come first.
2. Replace inline ``strip_torch_compile_prefix + load_state_dict``
   with calls to ``_persist_load_state_dict`` (defined in cell 47, with
   BS22 nested-_orig_mod, BS23 shape-mismatch tolerance, BS24 lazy-param
   tolerance).

Idempotent — sentinel ``BS27_SMOKE_FIX``.
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


# Old candidates list (legacy first, current never first).
OLD_CANDIDATES = '''candidates = [
    save_dir / "st_cdgm_checkpoint_best.pth",
    save_dir / "st_cdgm_checkpoint.pth",
    save_dir / "st_cdgm_checkpoint_last.pth",
]
if save_dir.is_dir():
    candidates.extend(sorted(save_dir.glob("*.pth")))'''

NEW_CANDIDATES = '''# >>> BS27_SMOKE_FIX (priority: current two-stage saves first)
candidates = [
    save_dir / "epoch_last.pth",
    save_dir / "epoch_best.pth",
    save_dir / "st_cdgm_checkpoint_best.pth",
    save_dir / "st_cdgm_checkpoint.pth",
    save_dir / "st_cdgm_checkpoint_last.pth",
]
if save_dir.is_dir():
    candidates.extend(sorted(save_dir.glob("*.pth")))'''


# Old inline load_state_dict (crashes on nested _orig_mod).
OLD_LOADS = '''encoder_target.load_state_dict(strip_torch_compile_prefix(loaded_ckpt["encoder_state_dict"]))
rcn_target.load_state_dict(strip_torch_compile_prefix(loaded_ckpt["rcn_cell_state_dict"]))
diff_target.load_state_dict(strip_torch_compile_prefix(loaded_ckpt["diffusion_state_dict"]))'''

NEW_LOADS = '''# >>> BS27_SMOKE_FIX (robust loader with shape/lazy/orig_mod tolerance)
_persist_load_state_dict(encoder, loaded_ckpt["encoder_state_dict"])
_persist_load_state_dict(rcn_cell, loaded_ckpt["rcn_cell_state_dict"])
_persist_load_state_dict(diffusion, loaded_ckpt["diffusion_state_dict"])
if "regression_head" in dir() and regression_head is not None and "regression_head_state_dict" in loaded_ckpt:
    _persist_load_state_dict(regression_head, loaded_ckpt["regression_head_state_dict"])'''


def patch_cell_58() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "[SMOKE]" in s and "Checkpoint smoke-test" in s)
    if idx is None:
        print("  ! smoke-test cell not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS27_SMOKE_FIX" in src:
        print(f"  = cell {idx} already patched (BS27)")
        return 0

    n = 0
    if OLD_CANDIDATES in src:
        src = src.replace(OLD_CANDIDATES, NEW_CANDIDATES, 1)
        n += 1
        print(f"  ~ cell {idx}: candidates priority reordered")
    else:
        print(f"  ! cell {idx}: candidates pattern not found")

    if OLD_LOADS in src:
        src = src.replace(OLD_LOADS, NEW_LOADS, 1)
        n += 1
        print(f"  ~ cell {idx}: load_state_dict swapped for _persist_load_state_dict")
    else:
        print(f"  ! cell {idx}: load pattern not found")

    if n > 0:
        cells[idx]["source"] = src.splitlines(keepends=True)
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


# =====================================================================
# Cell 60 patch — FINAL_VALIDATION should prefer epoch_last.pth
# =====================================================================

OLD_FINAL = '''_ckpt_path = _best if _best.exists() else _last'''
NEW_FINAL = '''# BS27_SMOKE_FIX: prefer ``epoch_last.pth`` — ``epoch_best.pth`` may be
# stale (BS18 Stage 2 always passes improved=False, so best is whatever
# the last Stage-1 best was, possibly from a previous run with a
# different architecture).
_ckpt_path = _last if _last.exists() else _best'''


def patch_cell_final_validation() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "FINAL_VALIDATION" in s
                     and "_ckpt_path = _best if _best.exists()" in s)
    if idx is None:
        # Already patched, or anchor changed.
        idx2 = _find_cell(cells, lambda s: "FINAL_VALIDATION" in s)
        if idx2 is not None and "BS27_SMOKE_FIX: prefer ``epoch_last.pth``" in "".join(cells[idx2]["source"]):
            print("  = FINAL_VALIDATION cell already patched (BS27)")
        else:
            print("  ! FINAL_VALIDATION cell pattern not found")
        return 0
    src = "".join(cells[idx]["source"])
    if OLD_FINAL not in src:
        print("  ! FINAL_VALIDATION pattern unmatched")
        return 0
    src = src.replace(OLD_FINAL, NEW_FINAL, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {idx}: FINAL_VALIDATION now prefers epoch_last.pth")
    return 1


def main() -> int:
    print("=== BS27a : smoke-test cell prefers epoch_last.pth + uses robust loader ===")
    n1 = patch_cell_58()
    print("\n=== BS27b : FINAL_VALIDATION prefers epoch_last.pth ===")
    n2 = patch_cell_final_validation()
    print(f"\n{n1 + n2} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
