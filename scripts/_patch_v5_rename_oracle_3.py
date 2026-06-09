"""Phase 7 recap : remplace `--- {variant} ---` par `--- {_disp} ---`
ou _disp mappe Noncausal -> ORACLE."""
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

src = "".join(nb["cells"][8]["source"])

OLD = 'for variant in ["V5", "Noncausal"]:\n    print(f"\\n--- {variant} ---")'
NEW = ('for variant in ["V5", "Noncausal"]:\n'
       '    _disp = "ORACLE" if variant == "Noncausal" else variant\n'
       '    print(f"\\n--- {_disp} ---")')

if OLD in src:
    src = src.replace(OLD, NEW)
    nb["cells"][8]["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("[OK] Phase 7 recap header : Noncausal -> ORACLE")
else:
    print("[KO] motif introuvable")
    print(f"  OLD repr: {OLD!r}")
