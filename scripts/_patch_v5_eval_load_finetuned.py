"""Patch Phase G : expose CHECKPOINT_NAME dans le notebook d'evaluation.

Permet de switcher entre le checkpoint V5-mini baseline (`epoch_last.pth`) et
le checkpoint post-Phase F fine-tuné (`epoch_finetuned.pth`) sans toucher
manuellement aux paths.

Modifications :
- Cell 2 (config) : ajout variable `CHECKPOINT_NAME = "epoch_last"`. Le user
  peut la changer en `"epoch_finetuned"` pour evaluer le checkpoint post-train.
- Cell 4 (build_stack) : utilise `V5_DIR / f"{CHECKPOINT_NAME}.pth"` et idem
  pour `NONCAUSAL_DIR`. Le CorrDiff baseline reste sur `epoch_last.pth`
  (jamais fine-tune).

Usage :
    python scripts/_patch_v5_eval_load_finetuned.py
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NB = Path("st_cdgm_v5_evaluation.ipynb")
with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak_phaseG")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup cree : {BACKUP}")

# === Patch Cell 2 — ajout CHECKPOINT_NAME apres NONCAUSAL_DIR ===
cell2_src = "".join(nb["cells"][2]["source"])

# On l'insere juste apres RESULTS_DIR.mkdir
OLD_C2 = 'RESULTS_DIR.mkdir(parents=True, exist_ok=True)'
NEW_C2 = ('RESULTS_DIR.mkdir(parents=True, exist_ok=True)\n\n'
          '# === Phase G (post-V5-mini, 2026-06-09) — selecteur de checkpoint ===\n'
          '# "epoch_last" = V5-mini baseline (gel §10.8)\n'
          '# "epoch_finetuned" = checkpoint post-Phase F (Bundle B + CASTLE + G_phys)\n'
          'CHECKPOINT_NAME = "epoch_last"   # change a "epoch_finetuned" pour evaluer post-train\n'
          'print(f"[Phase G] CHECKPOINT_NAME = {CHECKPOINT_NAME}")')

if OLD_C2 in cell2_src and "CHECKPOINT_NAME" not in cell2_src:
    cell2_src_new = cell2_src.replace(OLD_C2, NEW_C2)
    nb["cells"][2]["source"] = [l + "\n" for l in cell2_src_new.split("\n")[:-1]] + [cell2_src_new.split("\n")[-1]]
    print("[OK] Cell 2 : CHECKPOINT_NAME ajoute")
else:
    if "CHECKPOINT_NAME" in cell2_src:
        print("[SKIP] Cell 2 : CHECKPOINT_NAME deja present")
    else:
        print("[KO]  Cell 2 : motif RESULTS_DIR.mkdir introuvable")

# === Patch Cell 2 — verification d'existence avec CHECKPOINT_NAME ===
cell2_src = "".join(nb["cells"][2]["source"])
OLD_CHK = 'ckpt = p / "epoch_last.pth"'
NEW_CHK = 'ckpt = p / f"{CHECKPOINT_NAME}.pth"'
if OLD_CHK in cell2_src:
    cell2_src_new = cell2_src.replace(OLD_CHK, NEW_CHK)
    nb["cells"][2]["source"] = [l + "\n" for l in cell2_src_new.split("\n")[:-1]] + [cell2_src_new.split("\n")[-1]]
    print("[OK] Cell 2 : verification existence utilise CHECKPOINT_NAME")
else:
    if "CHECKPOINT_NAME.pth" in cell2_src:
        print("[SKIP] Cell 2 : verification deja patchee")
    else:
        print("[KO]  Cell 2 : motif epoch_last.pth check introuvable")

# === Patch Cell 4 — build_stack charge dynamiquement ===
cell4_src = "".join(nb["cells"][4]["source"])
OLD_C4 = 'stack_v5 = build_stack(V5_DIR / "epoch_last.pth", "V5")\nstack_nc = build_stack(NONCAUSAL_DIR / "epoch_last.pth", "Noncausal")'
NEW_C4 = ('# Phase G : CHECKPOINT_NAME permet de switcher entre baseline et post-finetune.\n'
          '# Le baseline noncausal reste sur "epoch_last.pth" (jamais fine-tune).\n'
          'stack_v5 = build_stack(V5_DIR / f"{CHECKPOINT_NAME}.pth", "V5")\n'
          'stack_nc = build_stack(NONCAUSAL_DIR / "epoch_last.pth", "Noncausal")')

if OLD_C4 in cell4_src:
    cell4_src_new = cell4_src.replace(OLD_C4, NEW_C4)
    nb["cells"][4]["source"] = [l + "\n" for l in cell4_src_new.split("\n")[:-1]] + [cell4_src_new.split("\n")[-1]]
    print("[OK] Cell 4 : build_stack utilise CHECKPOINT_NAME pour V5/ORACLE")
else:
    if 'CHECKPOINT_NAME}.pth' in cell4_src:
        print("[SKIP] Cell 4 : deja patchee")
    else:
        print("[KO]  Cell 4 : motif build_stack introuvable")

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print("=" * 60)
print("Pour evaluer le baseline V5-mini : CHECKPOINT_NAME = \"epoch_last\"")
print("Pour evaluer post-Phase F        : CHECKPOINT_NAME = \"epoch_finetuned\"")
print("=" * 60)
