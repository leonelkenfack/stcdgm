"""Patch : renomme RESULTS_DIR de v5_evaluation -> oracle_evaluation.

Le user veut isoler les resultats du nouveau modele (post-Phase F) des resultats
baseline V5-mini. Plutot que d'ecraser les anciens JSONs/figures, on les laisse
dans v5_evaluation/ et le nouveau run va dans oracle_evaluation/.

Effet :
- Cell 0 (intro) : update du path de sortie affiche
- Cell 2 (config) : RESULTS_DIR -> oracle_evaluation
- Suppression des cells de snapshot baseline (devenues inutiles)

Le path v5_evaluation/ n'est PAS supprime sur Drive.
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak_oracle_results")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup cree : {BACKUP}")


def _patch_cell_src(idx, replacements):
    src = "".join(nb["cells"][idx]["source"])
    n = 0
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new)
            n += 1
    nb["cells"][idx]["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    return n


# === Cell 0 : intro MD ===
n0 = _patch_cell_src(0, [
    ("**Sortie** : /content/drive/MyDrive/climate_data/results/v5_evaluation/",
     "**Sortie** (post-Phase F) : `/content/drive/MyDrive/climate_data/results/oracle_evaluation/`  \n**Baseline V5-mini** (intacte) : `/content/drive/MyDrive/climate_data/results/v5_evaluation/`"),
])
print(f"[OK] Cell 0 intro : {n0} remplacements")


# === Cell 2 : RESULTS_DIR ===
n2 = _patch_cell_src(2, [
    ('RESULTS_DIR = Path("/content/drive/MyDrive/climate_data/results/v5_evaluation")',
     '# Phase F (post-V5-mini, 2026-06-10) : nouveau dossier pour ne pas ecraser le baseline.\n'
     'RESULTS_DIR = Path("/content/drive/MyDrive/climate_data/results/oracle_evaluation")\n'
     'V5_BASELINE_RESULTS_DIR = Path("/content/drive/MyDrive/climate_data/results/v5_evaluation")  # baseline intact'),
])
print(f"[OK] Cell 2 RESULTS_DIR : {n2} remplacements")


# === Supprime les 2 cells snapshot (cells 5+6 = MD + code save baseline) ===
# Avant : cell 5 = MD snapshot, cell 6 = code snapshot, cell 7 = MD Phase F
# Apres suppression : cell 5 = MD Phase F (le precedent cell 7)
snapshot_md_idx = None
snapshot_code_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] == "markdown" and "Snapshot baseline V5-mini" in src:
        snapshot_md_idx = i
    elif c["cell_type"] == "code" and "Snapshot baseline V5-mini" in src:
        snapshot_code_idx = i

removed = []
# Supprimer dans l'ordre decroissant pour ne pas decaler les indices
for idx in sorted([snapshot_md_idx, snapshot_code_idx], reverse=True):
    if idx is not None:
        removed.append(idx)
        del nb["cells"][idx]
if removed:
    print(f"[OK] Cells snapshot baseline supprimees aux indices : {sorted(removed)}")
    print(f"     (les anciens resultats restent intacts dans /v5_evaluation/)")
else:
    print(f"[SKIP] Cells snapshot baseline introuvables")


# === Save ===
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print(f"[OK] Notebook : {len(nb['cells'])} cellules au total")
print(f"     Nouveaux runs ecrivent dans : results/oracle_evaluation/")
print(f"     Baseline V5-mini intact dans : results/v5_evaluation/")
