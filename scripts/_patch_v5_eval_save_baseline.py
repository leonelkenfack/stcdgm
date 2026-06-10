"""Patch : insere une cellule 'sauvegarde du baseline V5-mini' juste avant Phase F.

Inseres entre Cell 4 (bootstrap+stacks) et Cell 5 (Phase F MD) :
- Cell MD : explication "snapshot avant fine-tune"
- Cell code : shutil.copy des JSONs et figures actuelles vers v5_baseline/

A executer UNE SEULE FOIS avant la premiere execution de Phase F. La
cellule est idempotente : si v5_baseline/ existe deja, elle skip.
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak_save_baseline")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup cree : {BACKUP}")

# Sanity : on n'insere qu'une fois
if any("Sauvegarde du baseline V5-mini" in "".join(c["source"]) for c in nb["cells"]):
    print("[SKIP] Cellule baseline-save deja presente, abort")
    sys.exit(0)

CELL_MD = """---

## Snapshot baseline V5-mini (a executer UNE FOIS avant la 1ere execution de Phase F)

Avant de lancer le fine-tune (Phase F), on copie les JSONs et figures actuels
(baseline V5-mini) vers `results/v5_baseline/` pour pouvoir comparer ensuite
le checkpoint post-fine-tune contre cette reference.

**Cellule idempotente** : si `v5_baseline/` existe deja, elle skip silencieusement.

A executer apres la Phase 12 du baseline (ou tout au moins apres avoir les
JSONs en place) et **avant** d'executer Cell 8 (Phase F).
"""

CELL_CODE = '''# === Snapshot baseline V5-mini ===
import shutil
from pathlib import Path

BASELINE_DIR = Path("/content/drive/MyDrive/climate_data/results/v5_baseline")

if BASELINE_DIR.exists() and any(BASELINE_DIR.iterdir()):
    print(f"[SKIP] Baseline deja sauvegarde : {BASELINE_DIR}")
    print(f"       (pour re-snapshot, supprime ce dossier d'abord)")
else:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    n_copied = 0

    # 1. JSONs Phase 6/7 dans les ckpt dirs (V5 cote)
    for f in V5_DIR.glob("*.json"):
        shutil.copy(f, BASELINE_DIR / f.name)
        n_copied += 1

    # 2. JSONs Phase 6/7 cote Noncausal (pour referentiel comparaison)
    nc_baseline = BASELINE_DIR / "noncausal"
    nc_baseline.mkdir(exist_ok=True)
    for f in NONCAUSAL_DIR.glob("*.json"):
        shutil.copy(f, nc_baseline / f.name)
        n_copied += 1

    # 3. Sous-dossiers de results/v5_evaluation (figures + JSONs Phases 8-12)
    src_results = Path("/content/drive/MyDrive/climate_data/results/v5_evaluation")
    for sub in ["phase8_figures", "phase9_climate_standards",
                 "phase10_extreme_bias_maps", "phase11_causal_advanced",
                 "phase12_integrated_gradients", "phase7_runs"]:
        src_sub = src_results / sub
        if src_sub.exists():
            shutil.copytree(src_sub, BASELINE_DIR / sub, dirs_exist_ok=True)
            n_copied += sum(1 for _ in src_sub.rglob("*") if _.is_file())

    # 4. Recap JSONs a la racine de results/v5_evaluation
    for name in ["phase6_in_distribution.json", "phase7_ood_aligned.json",
                  "phase8_interpretability.json"]:
        src_file = src_results / name
        if src_file.exists():
            shutil.copy(src_file, BASELINE_DIR / name)
            n_copied += 1

    print(f"[OK] Baseline V5-mini sauvegarde dans {BASELINE_DIR}")
    print(f"     {n_copied} fichiers copies (JSONs + figures + npz Phase 7)")
    print(f"     Apres Phase F + re-eval, comparer avec :")
    print(f"     !python -m scripts.compare_eval_results \\\\")
    print(f"         --results-baseline {BASELINE_DIR} \\\\")
    print(f"         --baseline-dir {BASELINE_DIR} \\\\")
    print(f"         --results-finetuned {src_results} \\\\")
    print(f"         --finetuned-dir {V5_DIR} \\\\")
    print(f"         --out {src_results}/comparison_finetuned_vs_v5mini.md")
'''


def _make_cell(cell_type, source_str):
    lines = source_str.split("\n")
    if len(lines) > 1:
        source_list = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    else:
        source_list = [source_str]
    cell = {"cell_type": cell_type, "metadata": {}, "source": source_list}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


# Trouve la cell Phase F MD pour inserer juste avant
insert_at = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] == "markdown" and "Phase F (post-V5-mini" in src:
        insert_at = i
        break

if insert_at is None:
    print("[KO] Impossible de trouver la cell Phase F MD")
    sys.exit(1)

print(f"[OK] Cell {insert_at} = Phase F MD (insertion juste avant)")

new_cells = [
    _make_cell("markdown", CELL_MD),
    _make_cell("code", CELL_CODE),
]
nb["cells"] = nb["cells"][:insert_at] + new_cells + nb["cells"][insert_at:]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] 2 cellules inserees (snapshot baseline)")
print(f"     Notebook : {len(nb['cells'])} cellules au total")
