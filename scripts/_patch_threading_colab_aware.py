"""
Make cell 4's CPU thread default Colab-aware.

Default 48 threads (the original) is fine on a 48-core CPU host but starves
the Colab T4 runtime (2-8 vCPUs, GPU does the math anyway). On Colab, we
want intra-op = ~4 so BLAS doesn't fight the dataloader workers.

Detection: same heuristic as the bootstrap cell — ``google.colab`` module
present or ``/content`` path exists.

Idempotent: detects sentinel ``# >>> THREADING_COLAB_AWARE``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL = "# >>> THREADING_COLAB_AWARE"

NEW_CELL = '''\
# >>> THREADING_COLAB_AWARE
import os
import sys
from pathlib import Path

# Défaut device-aware :
#  - Colab / runtime GPU léger : 4 threads (T4 a 2-8 vCPU, le GPU calcule)
#  - Host CPU dédié (workstation, serveur) : 48 (ou ce que l'env. impose)
# Override systématiquement via l'env. var ST_CDGM_CPU_THREADS=N.
_IS_COLAB = "google.colab" in sys.modules or Path("/content").exists()
_DEFAULT_THREADS = "4" if _IS_COLAB else "48"
CPU_THREADS = int(os.environ.get("ST_CDGM_CPU_THREADS", _DEFAULT_THREADS))

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = str(CPU_THREADS)

print(f"[PERF] CPU_THREADS={CPU_THREADS} (Colab={_IS_COLAB})")
'''


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    target_idx = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if SENTINEL in src:
            target_idx = i
            print(f"✓ Cellule threading déjà patchée (cell {i}) — réécriture.")
            break
        if 'ST_CDGM_CPU_THREADS' in src and 'CPU_THREADS = int' in src:
            target_idx = i
            print(f"  ↪ Remplace cell threading {i}.")
            break

    if target_idx is None:
        print("[ERROR] Cellule threading introuvable.")
        return 2

    cells[target_idx]["source"] = NEW_CELL.splitlines(keepends=True)
    cells[target_idx]["outputs"] = []
    cells[target_idx]["execution_count"] = None

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Saved {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
