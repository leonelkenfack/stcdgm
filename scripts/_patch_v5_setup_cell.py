"""Aligne la cellule de setup de v5_evaluation sur celle de training_evaluation."""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

NB_V5 = Path("st_cdgm_v5_evaluation.ipynb")
NB_TRAIN = Path("st_cdgm_training_evaluation.ipynb")

with NB_V5.open(encoding="utf-8") as f:
    nb_v5 = json.load(f)

with NB_TRAIN.open(encoding="utf-8") as f:
    nb_train = json.load(f)

# Cellule 5 du training notebook = bootstrap Colab
bootstrap_cell = nb_train["cells"][5]
src_bootstrap = "".join(bootstrap_cell["source"])

# On reprend exactement cette cellule MAIS on ajoute un smoke test V5 a la fin
SMOKE_TEST = """

# === V5 specific smoke test (apres bootstrap) ===
try:
    from st_cdgm.models import ConditionalSkipBlock
    print("[V5 smoke] ConditionalSkipBlock importable - V5 features pretes")
except ImportError as e:
    print(f"[V5 smoke] ConditionalSkipBlock NON disponible : {e}")
    print("           Verifier que la branche two-stage-causal contient src/st_cdgm/models/skip_direct.py")

try:
    from scripts.intervention_test import INTERVENTIONS
    print(f"[V5 smoke] Phase 8 helpers OK ({len(INTERVENTIONS)} interventions)")
except ImportError as e:
    print(f"[V5 smoke] scripts.intervention_test NON disponible : {e}")
"""

new_source = src_bootstrap.rstrip() + SMOKE_TEST

# Construire la nouvelle cellule 1 du notebook V5
new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [ln + "\n" for ln in new_source.split("\n")[:-1]] + [new_source.split("\n")[-1]],
}

# Backup
BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak3")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb_v5, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #3 cree : {BACKUP}")

# Remplace cell 1 (la cellule de setup)
nb_v5["cells"][1] = new_cell

NB_V5.write_text(json.dumps(nb_v5, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Cell 1 remplacee par le bootstrap de training_evaluation")
print(f"[OK] {len(nb_v5['cells'])} cellules au total")
print()
print("Constantes utilisees (heritees de training_evaluation.ipynb cell 5) :")
print("  GIT_URL    = https://github.com/leonelkenfack/stcdgm.git")
print("  GIT_BRANCH = two-stage-causal")
print("  LOCAL_PROJECT = /content/climate_data (dossier local, pas le nom du repo)")
