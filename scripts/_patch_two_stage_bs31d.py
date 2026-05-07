"""
BS31d — fix BS31c position + auto-detect Stage 2 architecture drift.

Two issues observed on the first BS31b/c run:

1. **BS31c was positioned wrong**. The flag block was inserted INSIDE
   the BS18 try-block, right after ``_s2_resume_from`` assignment, but
   BEFORE the lines that read ``_saved_sigma_data`` and set
   ``_skip_calibration = True``. Even when FORCE_S2_RESTART=True is
   honored, the original code overwrites our resets a few lines later.
   Result: ``calib=skip, ablation=skip`` regardless of the flag.

2. **No auto-detection**. The user has to remember to set
   ``FORCE_S2_RESTART = True`` manually. Otherwise the loop happily
   resumes Stage 2 from epoch 16/20 with a freshly-xavier UNet → 5
   epochs of training is not enough to recover.

This patch:
- Captures the saved ``config`` dict from the checkpoint into
  ``_saved_config`` (right before ``del _ck_bs18``).
- Removes the in-try BS31c block (replaced with a no-op comment).
- Adds a NEW post-try block that:
    a) compares saved Stage-2 ``block_out_channels`` to current CONFIG
       — any difference triggers ``_arch_drift = True``
    b) computes ``_force_s2 = FORCE_S2_RESTART or _arch_drift`` and,
       when truthy, resets ``_s2_resume_from``, ``_skip_calibration``,
       ``_saved_sigma_data``, ``_skip_ablation``, ``_saved_ablation_ratio``.
- Logs explicitly which trigger fired (manual flag vs. arch drift).

Idempotent — sentinels ``BS31d_ARCH_DRIFT`` and ``BS31d_POST_TRY_OVERRIDE``.
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
# Patch 1 — capture _saved_config before del _ck_bs18
# =====================================================================

OLD_DEL = '''        del _ck_bs18'''

NEW_DEL_WITH_CAPTURE = '''        # >>> BS31d_ARCH_DRIFT
        # Capture the saved CONFIG so we can detect Stage 2 architecture
        # drift after the try block (block_out_channels mismatch ⇒ saved
        # diffusion weights are useless ⇒ Stage 2 epoch counter is
        # meaningless ⇒ force restart).
        _saved_config = _ck_bs18.get("config", {}) or {}

        del _ck_bs18'''


# =====================================================================
# Patch 2 — remove the in-try BS31c block (it ran before the
# overwriting code, which made it ineffective on _skip_calibration etc.)
# Replace with a comment + a no-op so the position is preserved for
# future reference.
# =====================================================================

OLD_INTRY_BS31C = '''        _s2_resume_from = int(_ts_state.get("stage2_epoch_done", 0))
        # >>> BS31c_FORCE_S2_RESTART
        # When the user has changed the diffusion architecture (e.g. via
        # CorrDiff-Mini override BS31b), Stage 2 weights are no longer
        # compatible — BS23 will skip them as shape mismatches → de-facto
        # xavier init. In that case the Stage 2 epoch counter from the
        # old run is meaningless; force it back to 0 so we retrain from
        # scratch. Set ``FORCE_S2_RESTART = True`` at cell scope before
        # running this cell to activate.
        if globals().get("FORCE_S2_RESTART", False):
            if _s2_resume_from > 0:
                print(f"   ⚡ FORCE_S2_RESTART=True — skipping Stage 2 resume "
                      f"(was at epoch {_s2_resume_from}/{int(_ts_state.get('stage2_epoch_done', 0))}).")
            _s2_resume_from = 0
            _skip_calibration = False  # recalibrate σ_data on the new residual scale
            _saved_sigma_data = None
            _saved_sigma_min = None
            _skip_ablation = False     # re-run O3 ablation with restored Stage 1 weights'''

NEW_INTRY_NOOP = '''        _s2_resume_from = int(_ts_state.get("stage2_epoch_done", 0))
        # BS31c moved out of the try-block by BS31d (the override has to
        # run *after* _saved_sigma_data is read, otherwise the original
        # ``if _saved_sigma_data is not None: _skip_calibration = True``
        # below clobbers our reset).'''


# =====================================================================
# Patch 3 — insert post-try override + arch-drift auto-detect.
# Anchor : the line right after the except-clause and before the
# ``# >>> TWO_STAGE_BS22_RESUME_STATUS`` block.
# =====================================================================

OLD_BEFORE_BS22 = '''except Exception as _e_bs18:
    print(f"   (BS18 resume read failed: {_e_bs18})")

# >>> TWO_STAGE_BS22_RESUME_STATUS'''

NEW_BEFORE_BS22 = '''except Exception as _e_bs18:
    print(f"   (BS18 resume read failed: {_e_bs18})")

# >>> BS31d_POST_TRY_OVERRIDE
# Combined override: manual ``FORCE_S2_RESTART`` flag OR auto-detected
# Stage 2 architecture drift. Runs *after* BS18 finishes assigning
# every ``_saved_*`` and ``_skip_*`` so its resets are the final word.
_arch_drift = False
try:
    _saved_blocks = (globals().get("_saved_config", {}) or {}) \\
        .get("diffusion", {}).get("unet_kwargs", {}).get("block_out_channels")
    _cur_blocks = list(CONFIG.diffusion.unet_kwargs.block_out_channels)
    if _saved_blocks is not None and list(_saved_blocks) != _cur_blocks:
        _arch_drift = True
        print(f"   🔀 Stage 2 arch drift: ckpt block_out_channels={list(_saved_blocks)} "
              f"vs current={_cur_blocks}")
except Exception:
    pass  # silent fallback when no checkpoint or no saved config

_force_s2 = bool(globals().get("FORCE_S2_RESTART", False)) or _arch_drift

if _force_s2 and (_s2_resume_from > 0 or _skip_calibration or _skip_ablation):
    _trigger = "FORCE_S2_RESTART=True" if globals().get("FORCE_S2_RESTART", False) else "auto-detected arch drift"
    print(f"   ⚡ Stage 2 reset (trigger: {_trigger})")
    print(f"      was: s2_done={_s2_resume_from}, calib_skip={_skip_calibration}, "
          f"ablation_skip={_skip_ablation}")
    _s2_resume_from = 0
    _skip_calibration = False
    _saved_sigma_data = None
    _saved_sigma_min = None
    _skip_ablation = False
    _saved_ablation_ratio = None
    print(f"      now: s2 retrains from epoch 1, σ_data recalibrated, O3 gate re-tested")

# >>> TWO_STAGE_BS22_RESUME_STATUS'''


def patch_cell_50() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "TWO_STAGE_TRAINING_LOOP" in s and "for s2_epoch" in s)
    if idx is None:
        print("  ! cell 50 not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS31d_POST_TRY_OVERRIDE" in src:
        print(f"  = cell {idx} already patched (BS31d)")
        return 0

    n = 0
    if OLD_DEL in src:
        src = src.replace(OLD_DEL, NEW_DEL_WITH_CAPTURE, 1)
        n += 1
        print(f"  ~ cell {idx}: capture _saved_config before del")
    else:
        print(f"  ! cell {idx}: del _ck_bs18 anchor not found")

    if OLD_INTRY_BS31C in src:
        src = src.replace(OLD_INTRY_BS31C, NEW_INTRY_NOOP, 1)
        n += 1
        print(f"  ~ cell {idx}: in-try BS31c neutralised")
    else:
        print(f"  ! cell {idx}: in-try BS31c block not found (ok if already removed)")

    if OLD_BEFORE_BS22 in src:
        src = src.replace(OLD_BEFORE_BS22, NEW_BEFORE_BS22, 1)
        n += 1
        print(f"  ~ cell {idx}: post-try override block inserted")
    else:
        print(f"  ! cell {idx}: BS22 status anchor not found")

    if n > 0:
        cells[idx]["source"] = src.splitlines(keepends=True)
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def main() -> int:
    print("=== BS31d : arch-drift auto-detect + post-try override (cell 50) ===")
    n = patch_cell_50()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
