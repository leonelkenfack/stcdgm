"""
BS33 — fix BS22 status print mismatch with actual S{1,2}_EPOCHS.

Symptom (user log): the line
    🔁 BS18 resume: Stage1 done=7/7, Stage2 done=0/20, ...
shows ``/20`` while the actual training loop a few lines down says
``--- Stage 2 — epoch 1/10 ---``. This happens because the BS22
print was inserted *before* ``ts_cfg = CONFIG.two_stage`` and
``S{1,2}_EPOCHS = int(...)`` are computed. If CONFIG is mutated
between the BS22 print and the loop (e.g. cell 13 re-run with a
newer override file mid-run), or if the cell is partially
re-executed, the print can show a stale value while the loop uses
the live one.

Fix: move the status print *after* ``S2_EPOCHS = int(...)`` and use
``S1_EPOCHS`` / ``S2_EPOCHS`` directly. Single source of truth for
both the print and the loop bounds.

Idempotent — sentinel ``BS33_STATUS_AFTER_EPOCHS``.
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


# Old BS22 status block — to be removed from current position.
OLD_BS22_BLOCK = '''# >>> TWO_STAGE_BS22_RESUME_STATUS
# Explicit one-liner covering the three resume cases so the user can
# tell at a glance whether the loop will restart from scratch on the
# epoch counter despite weights being loaded.
_ck_existed_bs22 = False
try:
    _ck_existed_bs22 = (_Path_bs18(str(CKPT_SAVE_DIR)) / "epoch_last.pth").exists()
except Exception:
    pass

# >>> TWO_STAGE_BS24_RESUME_PRINT_CONFIG
# Use CONFIG.two_stage directly — ``ts_cfg`` alias is assigned *after*
# this block. Pre-existing latent NameError exposed once resume works.
if _s1_resume_from > 0 or _s2_resume_from > 0:
    print(
        f"🔁 BS18 resume: Stage1 done={_s1_resume_from}/{int(CONFIG.two_stage.stage1.epochs_max)}, "
        f"Stage2 done={_s2_resume_from}/{int(CONFIG.two_stage.stage2.epochs_max)}, "
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
    print("🆕 BS22 resume: aucun checkpoint trouvé → entraînement frais.")

ts_cfg = CONFIG.two_stage
S1_EPOCHS = int(ts_cfg.stage1.epochs_max)
S2_EPOCHS = int(ts_cfg.stage2.epochs_max)'''


# New: ts_cfg/S_EPOCHS first, then status print using those variables.
NEW_BLOCK = '''ts_cfg = CONFIG.two_stage
S1_EPOCHS = int(ts_cfg.stage1.epochs_max)
S2_EPOCHS = int(ts_cfg.stage2.epochs_max)

# >>> BS33_STATUS_AFTER_EPOCHS
# Status print moved AFTER S{1,2}_EPOCHS so the displayed totals are
# guaranteed to match what the training loops will use. Previously the
# print read CONFIG.two_stage.* before ts_cfg was assigned; if CONFIG
# was mutated (override merge re-run, partial cell re-execution),
# the print could show a stale value while the loop used the live one.
_ck_existed_bs33 = False
try:
    _ck_existed_bs33 = (_Path_bs18(str(CKPT_SAVE_DIR)) / "epoch_last.pth").exists()
except Exception:
    pass

if _s1_resume_from > 0 or _s2_resume_from > 0:
    print(
        f"🔁 BS18 resume: Stage1 done={_s1_resume_from}/{S1_EPOCHS}, "
        f"Stage2 done={_s2_resume_from}/{S2_EPOCHS}, "
        f"calib={'skip' if _skip_calibration else 'redo'}, "
        f"ablation={'skip' if _skip_ablation else 'redo'}"
    )
elif _ck_existed_bs33:
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
    if "BS33_STATUS_AFTER_EPOCHS" in src:
        print(f"  = cell {idx} already patched (BS33)")
        return 0
    if OLD_BS22_BLOCK not in src:
        print(f"  ! cell {idx}: BS22 status block pattern not found "
              f"(maybe already manually edited or earlier patch differs)")
        return 0
    src = src.replace(OLD_BS22_BLOCK, NEW_BLOCK, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {idx}: BS22 status moved AFTER S{{1,2}}_EPOCHS, uses those variables")
    return 1


def main() -> int:
    print("=== BS33 : status print uses S{1,2}_EPOCHS, after their assignment ===")
    n = patch_cell_50()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
