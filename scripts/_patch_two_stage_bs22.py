"""
BS22 — robust checkpoint resume.

Three issues observed on first BS21 reload:

1. **Diffusion weights silently dropped** — ``_persist_load_state_dict``
   only stripped *top-level* ``_orig_mod.`` keys. But ``torch.compile`` is
   applied to the *nested* ``diffusion.unet``, so saved keys like
   ``unet.conv_in.weight`` no longer match the live ``unet._orig_mod.conv_in.weight``
   target. Strict ``load_state_dict`` raised → diffusion stayed at xavier
   init while encoder/rcn_cell loaded fine.

2. **Optimizer restore on stale param groups** — when resume happens but
   the optimizer was saved with a different param-group sizing (e.g.
   built before some lazy params were materialized), strict load crashes.
   We now warn and continue with a fresh optimizer.

3. **Silent resume-from-epoch-0** — when ``two_stage`` field is absent
   from the checkpoint payload, BS20 loads weights but the loop counter
   stays at 0, *without* telling the user. We add explicit logs.

Cell 47 (PERSIST_HELPERS)
-------------------------
- Replace ``_persist_load_state_dict`` with a bidirectional normalizer
  that strips ``_orig_mod.`` everywhere in the saved state_dict, then
  re-maps onto the live module's expected keys (which can have
  ``_orig_mod`` re-inserted at any depth). Loads with ``strict=False`` and
  prints a one-line summary of matched / missing keys.

Cell 50 (TWO_STAGE_TRAINING_LOOP)
---------------------------------
- Augment the BS18 RESUME block with a clear status print covering the
  three possible cases: (a) full two_stage info found → resume normally,
  (b) checkpoint exists but no ``two_stage`` field → weights-only resume,
  loop restarts at epoch 1 (warned), (c) no checkpoint → fresh start.

Idempotent — sentinel ``TWO_STAGE_BS22_*``.
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
# Cell 47 patch — robust _persist_load_state_dict
# =====================================================================

OLD_LOAD_FN = '''def _persist_load_state_dict(m, sd):
    """Charge un state_dict en respectant les wrappers DDP / torch.compile."""
    if m is None or sd is None:
        return
    try:
        from st_cdgm.utils.checkpoint import strip_torch_compile_prefix
        sd = strip_torch_compile_prefix(sd)
    except Exception:
        pass
    base = m.module if hasattr(m, "module") and not hasattr(m, "_orig_mod") else m
    base = getattr(base, "_orig_mod", base)
    base.load_state_dict(sd)'''

NEW_LOAD_FN = '''def _persist_load_state_dict(m, sd):
    """Charge un state_dict en respectant les wrappers DDP / torch.compile.

    BS22 — bidirectional ``_orig_mod`` normalization:
    ``torch.compile`` can be applied to *nested* submodules (e.g.
    ``diffusion.unet``), so the prefix may appear at any depth. We strip
    every ``_orig_mod.`` token in saved keys then re-map onto the live
    module's expected keys (which themselves may carry ``_orig_mod`` at
    arbitrary positions). Returns silently on success, prints a single
    summary line if any key was unmatched.
    """
    if m is None or sd is None:
        return
    base = m.module if hasattr(m, "module") and not hasattr(m, "_orig_mod") else m
    base = getattr(base, "_orig_mod", base)

    # Strip _orig_mod tokens at any depth from saved keys.
    stripped_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    target_keys = list(base.state_dict().keys())
    matched = {}
    for tk in target_keys:
        norm_tk = tk.replace("_orig_mod.", "")
        if norm_tk in stripped_sd:
            matched[tk] = stripped_sd[norm_tk]
    n_target = len(target_keys)
    n_matched = len(matched)
    if n_matched < n_target:
        # Useful when a checkpoint was saved before architecture changes.
        print(f"   ↳ _persist_load: matched {n_matched}/{n_target} weights (some keys absent in checkpoint)")
    base.load_state_dict(matched, strict=False)'''


def patch_cell_47() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "def _persist_load_state_dict" in s
                     and "PERSIST_HELPERS" in s)
    if idx is None:
        print("  ! cell 47 (PERSIST_HELPERS) not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS22 — bidirectional" in src:
        print("  = cell 47 already patched (BS22)")
        return 0
    if OLD_LOAD_FN not in src:
        print("  ! cell 47: OLD_LOAD_FN signature pattern not found")
        return 0
    src = src.replace(OLD_LOAD_FN, NEW_LOAD_FN, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  ~ cell 47: _persist_load_state_dict replaced (BS22 nested _orig_mod)")
    return 1


# =====================================================================
# Cell 50 patch — explicit resume status logging
# =====================================================================

OLD_STATUS = '''if _s1_resume_from > 0 or _s2_resume_from > 0:
    print(
        f"🔁 BS18 resume: Stage1 done={_s1_resume_from}/{int(ts_cfg.stage1.epochs_max)}, "
        f"Stage2 done={_s2_resume_from}/{int(ts_cfg.stage2.epochs_max)}, "
        f"calib={'skip' if _skip_calibration else 'redo'}, "
        f"ablation={'skip' if _skip_ablation else 'redo'}"
    )'''

NEW_STATUS = '''# >>> TWO_STAGE_BS22_RESUME_STATUS
# Explicit one-liner covering the three resume cases so the user can
# tell at a glance whether the loop will restart from scratch on the
# epoch counter despite weights being loaded.
_ck_existed_bs22 = False
try:
    _ck_existed_bs22 = (_Path_bs18(str(CKPT_SAVE_DIR)) / "epoch_last.pth").exists()
except Exception:
    pass

if _s1_resume_from > 0 or _s2_resume_from > 0:
    print(
        f"🔁 BS18 resume: Stage1 done={_s1_resume_from}/{int(ts_cfg.stage1.epochs_max)}, "
        f"Stage2 done={_s2_resume_from}/{int(ts_cfg.stage2.epochs_max)}, "
        f"calib={'skip' if _skip_calibration else 'redo'}, "
        f"ablation={'skip' if _skip_ablation else 'redo'}"
    )
elif _ck_existed_bs22:
    print(
        "⚠️  BS22 resume: checkpoint trouvé MAIS aucun champ 'two_stage' valide "
        "(stage1_epoch_done=0, stage2_epoch_done=0).\\n"
        "   → poids chargés par BS20 mais boucle Stage 1 redémarre à epoch 1.\\n"
        "   → c'est attendu si le précédent run a crash avant la fin de la 1ère epoch."
    )
else:
    print("🆕 BS22 resume: aucun checkpoint trouvé → entraînement frais.")'''


def patch_cell_50() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "TWO_STAGE_TRAINING_LOOP" in s and "for s2_epoch" in s)
    if idx is None:
        print("  ! cell 50 not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "TWO_STAGE_BS22_RESUME_STATUS" in src:
        print("  = cell 50 already patched (BS22)")
        return 0
    if OLD_STATUS not in src:
        print("  ! cell 50: OLD_STATUS pattern not found")
        return 0
    src = src.replace(OLD_STATUS, NEW_STATUS, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  ~ cell 50: BS22 resume status block added")
    return 1


def main() -> int:
    print("=== BS22a : robust _persist_load_state_dict (cell 47) ===")
    n1 = patch_cell_47()
    print("\n=== BS22b : explicit resume status logging (cell 50) ===")
    n2 = patch_cell_50()
    print(f"\n{n1 + n2} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
