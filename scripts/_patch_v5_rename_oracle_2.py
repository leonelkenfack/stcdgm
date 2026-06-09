"""Patch additionnel : fix le print de la boucle resume Q_int dans Cell 10
pour afficher 'ORACLE' au lieu de 'Noncausal'."""
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

src = "".join(nb["cells"][10]["source"])

# Le print de la boucle "for variant in [V5, Noncausal]"
OLD = r'print(f"\n{variant:10s} : Q_int = {q_int:.3f}  ({n_match}/{len(results)} signes corrects)")'
NEW = (r'_disp = "ORACLE" if variant == "Noncausal" else variant' "\n"
       r'        print(f"\n{_disp:10s} : Q_int = {q_int:.3f}  ({n_match}/{len(results)} signes corrects)")')

if OLD in src:
    src = src.replace(OLD, NEW)
    nb["cells"][10]["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK] summary print : variant -> _disp avec mapping Noncausal->ORACLE")
else:
    print("[KO] motif introuvable")
    print(f"   OLD repr (premier chars): {OLD[:80]!r}")
