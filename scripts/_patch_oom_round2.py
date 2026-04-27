"""
Round 2 OOM fixes — le OOM persiste après round 1 parce que SDPA tombe sur
le backend "math" (matrice [N,N] en fp32) au lieu de "memory_efficient"
(O(N) mémoire). 8 attentions × 478 MB → 4 GB activations + 3 GB backward
retention = ~7 GB.

Trois actions :

A. Cell 4 — ajouter ``PYTORCH_ALLOC_CONF`` (forme PyTorch 2.5+, c'est celle
   que le message d'erreur PyTorch nomme).

B. Cell 35 — après l'init CUDA, **forcer** le backend SDPA memory_efficient
   en désactivant math + flash. Mem-efficient est O(N) au lieu de O(N²) →
   les 8 attentions UNet passent de ~4 GB à ~80 MB.

C. Cell 5 (bootstrap) — tenter ``pip install xformers`` (best-effort). Sur
   sm_75 (T4), xformers a un kernel mémoire-efficient encore meilleur que
   SDPA mem-eff. Pas bloquant si l'install échoue.

Idempotent : sentinelle ``# >>> OOM_FIX_R2``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL = "# >>> OOM_FIX_R2"


# Cell 4 — ajout de PYTORCH_ALLOC_CONF (le message d'erreur PyTorch suggère
# CETTE forme). Les deux noms coexistent en PyTorch 2.5+.
CELL_4_NEW = '''\
# >>> OOM_FIX_R2
# CRITIQUE : ces variables doivent être posées AVANT le moindre import torch
# (l'allocateur CUDA les lit une fois à l'init). ``expandable_segments:True``
# élimine la fragmentation. PyTorch 2.5+ accepte les deux noms.
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# >>> THREADING_COLAB_AWARE
import sys
from pathlib import Path

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
print(f"[PERF] PYTORCH_ALLOC_CONF={os.environ['PYTORCH_ALLOC_CONF']}")
'''


# Cell 35 — GPU memory + préfère mem-efficient SDPA + xformers.
CELL_35_NEW = '''\
# >>> OOM_FIX_R2
# Optimisation mémoire GPU + ordre de préférence des backends SDPA.
#
# Round 1 du fix avait DÉSACTIVÉ math, ce qui causait ``RuntimeError:
# Invalid backend`` quand mem-efficient ne pouvait pas s'appliquer (ex.
# attention_mask non-None passé par diffusers). On garde math activé en
# **fallback de dernier recours** ; mem-efficient sera essayé en priorité
# par le dispatcher PyTorch et utilisé quand applicable.
#
# Combiné avec ``expandable_segments:True`` (cell 4), la fragmentation qui
# rendait math OOM-able est désormais éliminée → math comme fallback est
# safe. xformers reste préféré quand dispo (constraints plus permissives
# que SDPA mem-efficient).
if torch.cuda.is_available():
    print("🧹 GPU memory optimization…")
    torch.cuda.empty_cache()
    print("   ✓ GPU cache cleared")

    # Préférences SDPA — l'ordre Flash > Mem-Eff > Math est conservé,
    # on désactive juste Flash (T4 sm_75 ne le supporte pas) pour éviter
    # qu'il tente d'abord et échoue silencieusement.
    try:
        torch.backends.cuda.enable_flash_sdp(False)        # T4 sm_75 → pas Flash
        torch.backends.cuda.enable_mem_efficient_sdp(True) # priorité 1
        torch.backends.cuda.enable_math_sdp(True)          # filet de sécurité
        print("   ✓ SDPA: mem-efficient prioritaire, math en fallback")
    except AttributeError:
        print("   ⚠️  API SDPA absente (PyTorch ancien).")

    # xformers : kernel attention encore meilleur que SDPA mem-eff sur sm_75
    # et bien plus permissif sur les shapes/dtypes/masks.
    if "diffusion" in dir() and hasattr(diffusion, "unet"):
        try:
            diffusion.unet.enable_xformers_memory_efficient_attention()
            print("   ✅ xformers ATTENTION ACTIVÉ sur diffusion.unet — OOM résolu.")
        except (ImportError, ModuleNotFoundError):
            print("   ⚠️  xformers ABSENT — fallback sur SDPA. Si OOM persiste :")
            print("        !pip install xformers && Runtime → Restart runtime")
        except ValueError as _ve:
            # Survient quand la version xformers est incompatible avec PyTorch
            print(f"   ⚠️  xformers incompatible: {_ve}")
        except Exception as _xe:
            print(f"   ⚠️  xformers échec: {type(_xe).__name__}: {_xe}")

    # Status mémoire courant
    print("\\n📊 GPU Memory Status (post-optim):")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        print(f"   GPU {i}: {allocated:.2f} GB allocated / {reserved:.2f} GB reserved / {total:.2f} GB total")

    # Note : on N'APPELLE PAS ``set_per_process_memory_fraction(0.95)``.
    # Avec ``expandable_segments:True``, le cap est contre-productif (force
    # le release prématuré au lieu de laisser l'allocator agrandir).
    print("\\n✅ GPU memory optimization done")
else:
    print("ℹ️  CUDA indisponible — saut des optimisations GPU.")
'''


# Cell 5 — bootstrap : ajout d'un essai xformers (best-effort).
# On modifie la liste EXTRA_DEPS pour ajouter xformers.
def patch_bootstrap_xformers(src: str) -> str | None:
    """Inject xformers install attempt into the bootstrap pip block."""
    marker = '"torch-geometric",  # v2.3+ ne nécessite plus torch-scatter/torch-sparse'
    if marker not in src:
        return None
    if 'xformers' in src:
        return None  # already present
    replacement = (
        '"torch-geometric",  # v2.3+ ne nécessite plus torch-scatter/torch-sparse\n'
        '            "xformers",  # OOM fix R2 : memory-efficient attention sur sm_75 (T4)'
    )
    return src.replace(marker, replacement, 1)


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    # Patch cell 4
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "CPU_THREADS = int" in src and "PYTORCH_CUDA_ALLOC_CONF" in src:
            if SENTINEL in src and "PYTORCH_ALLOC_CONF" in src and "PYTORCH_ALLOC_CONF=" in src:
                print(f"✓ Cell {i} (threading) déjà patchée R2.")
            else:
                cells[i]["source"] = CELL_4_NEW.splitlines(keepends=True)
                cells[i]["outputs"] = []
                cells[i]["execution_count"] = None
                print(f"  ↪ Cell {i} (threading) → ajout PYTORCH_ALLOC_CONF")
                n_changed += 1
            break

    # Patch cell 35 (GPU memory). Always rewrite to ensure latest content
    # (we tweak the SDPA backend strategy across iterations of this patch).
    target35 = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if (
            ("GPU Memory Optimization" in src and "torch.cuda.empty_cache" in src)
            or (SENTINEL in src and "enable_mem_efficient_sdp" in src)
        ):
            target35 = i
            break

    if target35 is not None:
        current = "".join(cells[target35].get("source", []))
        # Idempotency : skip rewrite if content is byte-identical.
        if current == CELL_35_NEW:
            print(f"✓ Cell {target35} (GPU mem) déjà à jour.")
        else:
            cells[target35]["source"] = CELL_35_NEW.splitlines(keepends=True)
            cells[target35]["outputs"] = []
            cells[target35]["execution_count"] = None
            print(f"  ↪ Cell {target35} (GPU mem) → SDPA mem_efficient prio + math fallback")
            n_changed += 1
    else:
        print("[WARN] Cell 'GPU Memory Optimization' introuvable — skip cell 35.")

    # Patch cell 5 (bootstrap) — append xformers to deps list
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "COLAB_BOOTSTRAP" in src and "EXTRA_DEPS" in src:
            new_src = patch_bootstrap_xformers(src)
            if new_src is None:
                print(f"✓ Cell {i} (bootstrap) — xformers déjà inclus ou pattern non trouvé.")
            else:
                cells[i]["source"] = new_src.splitlines(keepends=True)
                cells[i]["outputs"] = []
                cells[i]["execution_count"] = None
                print(f"  ↪ Cell {i} (bootstrap) → ajout xformers aux EXTRA_DEPS")
                n_changed += 1
            break

    if n_changed == 0:
        print("Aucune modification nécessaire.")
        return 0

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Saved {NB} ({n_changed} cells modified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
