"""
Three OOM fixes for T4 (14.5 GB VRAM):

1. ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` posé AVANT l'import
   de torch (cell 4 = threading). Élimine la fragmentation : les 6.47 GB
   "reserved but unallocated" devenaient innaccessibles pour des allocations
   contigües (ex. 7.14 GB pour la matrice d'attention).

2. ``use_gradient_checkpointing: true`` dans le YAML diffusion. Divise par
   ~2 la VRAM des activations UNet (recompute en backward, +30% compute).
   Indispensable sur T4 pour `block_out_channels` actuel.

3. ``compile.diffusion_mode: "default"`` au lieu de ``reduce-overhead``.
   Le mode reduce-overhead utilise CUDA graphs qui capturent et FIGENT la
   mémoire — ingérable quand on est à 8 GB / 14.5 GB de marge. ``default``
   compile sans CUDA graphs, on perd ~5% de speedup mais on garde 1-2 GB.
   Encoder/RCN restent en reduce-overhead (peu coûteux).

Idempotent : sentinelle ``# >>> OOM_FIX`` dans cell 4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
YAML = Path(__file__).resolve().parent.parent / "config" / "training_config.yaml"
SENTINEL = "# >>> OOM_FIX"

# Cell 4 (threading) avec PYTORCH_CUDA_ALLOC_CONF posé EN PREMIER
CELL_4_NEW = '''\
# >>> OOM_FIX
# CRITIQUE : ``PYTORCH_CUDA_ALLOC_CONF`` doit être posé AVANT le moindre
# import torch — l'allocateur lit cette variable une seule fois à l'init.
# ``expandable_segments:True`` permet à PyTorch d'agrandir les segments au
# lieu de réserver des blocs contigus → élimine la fragmentation qui faisait
# OOM sur les 7+ GB d'attention sur T4 (14.5 GB VRAM).
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# >>> THREADING_COLAB_AWARE
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
print(f"[PERF] PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")
'''


def patch_notebook() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Find cell 4 (threading)
    target = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if SENTINEL in src and "PYTORCH_CUDA_ALLOC_CONF" in src:
            print(f"✓ Cell {i} déjà patchée (OOM fix présent).")
            return 0
        if "ST_CDGM_CPU_THREADS" in src and "CPU_THREADS = int" in src:
            target = i
            break

    if target is None:
        print("[ERROR] Cellule threading introuvable.")
        return 2

    cells[target]["source"] = CELL_4_NEW.splitlines(keepends=True)
    cells[target]["outputs"] = []
    cells[target]["execution_count"] = None
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ↪ Cell {target} (threading) → ajouté PYTORCH_CUDA_ALLOC_CONF=expandable_segments")
    print(f"💾 Saved {NB}")
    return 0


def patch_yaml() -> int:
    src = YAML.read_text(encoding="utf-8")
    changed = []

    # 1. use_gradient_checkpointing: false → true
    if "use_gradient_checkpointing: false" in src:
        src = src.replace(
            "use_gradient_checkpointing: false",
            "use_gradient_checkpointing: true  # OOM fix T4 : divise les activations UNet par ~2",
            1,
        )
        changed.append("use_gradient_checkpointing → true")
    elif "use_gradient_checkpointing: true" in src:
        print("✓ use_gradient_checkpointing déjà à true.")
    else:
        print("[WARN] Clé use_gradient_checkpointing introuvable.")

    # 2. compile.diffusion_mode: "reduce-overhead" → "default"
    if 'diffusion_mode: "reduce-overhead"' in src:
        src = src.replace(
            'diffusion_mode: "reduce-overhead"',
            'diffusion_mode: "default"  # OOM fix T4 : reduce-overhead ⇒ CUDA graphs ⇒ mémoire figée',
            1,
        )
        changed.append('compile.diffusion_mode → "default"')
    elif 'diffusion_mode: "default"' in src:
        print("✓ compile.diffusion_mode déjà à default.")

    if changed:
        YAML.write_text(src, encoding="utf-8")
        print(f"💾 YAML patché : {', '.join(changed)}")
    return 0


def main() -> int:
    rc = patch_notebook()
    rc |= patch_yaml()
    return rc


if __name__ == "__main__":
    sys.exit(main())
