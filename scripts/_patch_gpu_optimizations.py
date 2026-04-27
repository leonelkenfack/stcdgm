"""
Three GPU-correctness fixes in ``st_cdgm_training_evaluation.ipynb``.

A. Cell 40 — ``cudnn.benchmark = True`` was gated on ``USE_MULTI_GPU``. On a
   single T4 it never fired. This is a 10-30% wallclock loss on conv-heavy
   networks at fixed input shapes. Activate it whenever CUDA is available.
   Also add ``torch.set_float32_matmul_precision("high")`` (free on sm_70+).

B. Cell 41 — ``torch.compile`` block recreates ``optimizer`` from
   ``encoder.parameters() + rcn_cell.parameters() + diffusion.parameters()``
   ONLY. The original optimizer (created earlier) included the spatial
   projector and the HR ident head; after cell 41 those params are silently
   dropped from optimization. Worst case: the DAG-token MLP inside
   ``CausalConditioningProjector`` stops training → the contrastive DAG
   loss has nothing to push. Patch the optimizer reconstruction to include
   every param group that existed before.

C. Cell 41 — also wrap ``spatial_projector`` and ``hr_ident_head`` in
   ``torch.compile`` (when present) for consistency with the other modules.

Idempotent: sentinel ``# >>> GPU_OPT_FIX``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL = "# >>> GPU_OPT_FIX"

CELL_40_NEW = '''\
# >>> GPU_OPT_FIX
# Optimisations GPU générales (s'appliquent en single-GPU comme en multi-GPU).
# cudnn.benchmark : sélectionne le meilleur kernel conv par shape (les nôtres
# sont fixes à 172×179) et le cache. Gain typique 10-30% sur UNet conv-lourd.
# set_float32_matmul_precision("high") : autorise TF32 sur Ampere+, no-op sur
# T4 (sm_75) — donc gratuit dans tous les cas.
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    cudnn.benchmark = True
    cudnn.enabled = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    print("✅ cudnn.benchmark + set_float32_matmul_precision activés.")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
        alloc = torch.cuda.memory_allocated(i) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
        print(f"   GPU {i} {name} — {total:.1f} GB total, "
              f"{alloc:.2f} alloc / {reserved:.2f} reserved")
else:
    print("ℹ️  Pas de CUDA — saut des optimisations GPU.")

# Multi-GPU specific (load balancing) — kept for compatibility.
if USE_MULTI_GPU:
    torch.set_num_threads(4)
    print(f"   threads/worker (multi-GPU): {torch.get_num_threads()}")
'''


CELL_41_NEW = '''\
# >>> GPU_OPT_FIX
# torch.compile sur tous les modules entraînables, en préservant l'optimizer
# d'origine (qui inclut spatial_projector + hr_ident_head si présents).
# Le bug précédent : cette cellule reconstruisait optimizer SANS le spatial
# projector → le MLP des DAG-tokens cessait de s'entraîner et la perte
# contrastive n'avait plus rien à propager.
if CONFIG.training.get("compile", {}).get("enabled", False):
    print("🔧 Compiling models with torch.compile…")
    compile_cfg = CONFIG.training.compile

    def _unwrap(m):
        """Strip DDP / DataParallel before compile (eager core)."""
        if m is None:
            return None
        return m.module if hasattr(m, "module") else m

    encoder_base = _unwrap(encoder)
    rcn_base = _unwrap(rcn_cell)
    diffusion_base = _unwrap(diffusion)
    sp_base = _unwrap(spatial_projector) if "spatial_projector" in dir() and spatial_projector is not None else None
    hr_base = _unwrap(hr_ident_head) if "hr_ident_head" in dir() and hr_ident_head is not None else None

    def _try_compile(m, mode_key, default="reduce-overhead"):
        if m is None:
            return None
        try:
            return torch.compile(m, mode=compile_cfg.get(mode_key, default), fullgraph=False)
        except Exception as _e:
            print(f"⚠️  torch.compile({mode_key}) a échoué : {_e} — fallback eager.")
            return m

    encoder_compiled = _try_compile(encoder_base, "encoder_mode")
    rcn_compiled = _try_compile(rcn_base, "rcn_mode")
    diffusion_compiled = _try_compile(diffusion_base, "diffusion_mode")
    sp_compiled = _try_compile(sp_base, "spatial_projector_mode")
    hr_compiled = _try_compile(hr_base, "hr_ident_mode")

    # Re-wrap with DataParallel if the originals were wrapped.
    if hasattr(encoder, "module"):
        gpus = CONFIG.training.multi_gpu.get("gpus", [0])
        encoder = torch.nn.DataParallel(encoder_compiled, device_ids=gpus)
        rcn_cell = torch.nn.DataParallel(rcn_compiled, device_ids=gpus)
        diffusion = torch.nn.DataParallel(diffusion_compiled, device_ids=gpus)
        if sp_compiled is not None:
            spatial_projector = torch.nn.DataParallel(sp_compiled, device_ids=gpus)
        if hr_compiled is not None:
            hr_ident_head = torch.nn.DataParallel(hr_compiled, device_ids=gpus)
    else:
        encoder = encoder_compiled
        rcn_cell = rcn_compiled
        diffusion = diffusion_compiled
        if sp_compiled is not None:
            spatial_projector = sp_compiled
        if hr_compiled is not None:
            hr_ident_head = hr_compiled

    # Reconstruct optimizer including ALL trainable modules. We preserve the
    # param groups (DAG group has its own LR/betas) of the existing optimizer
    # if present, otherwise rebuild a flat single-group optimizer.
    _all_params = (
        list(encoder.parameters())
        + list(rcn_cell.parameters())
        + list(diffusion.parameters())
    )
    if "spatial_projector" in dir() and spatial_projector is not None:
        _all_params += list(spatial_projector.parameters())
    if "hr_ident_head" in dir() and hr_ident_head is not None:
        _all_params += list(hr_ident_head.parameters())

    # Try to preserve existing param-group structure (LR multipliers etc.).
    if "optimizer" in dir() and optimizer is not None and len(optimizer.param_groups) > 1:
        # Existing groups are referenced by *params*, but compile changes the
        # parameter tensor identities → we must rebuild from scratch using the
        # same hyperparameters as before.
        _g0 = optimizer.param_groups[0]
        _gN = optimizer.param_groups[-1]
        # Group 0: tout sauf A_dag. Group last: A_dag (DAG-decouple LR mult).
        _dag_names = {id(p) for n, p in _unwrap(rcn_cell).named_parameters() if n == "A_dag"}
        _other = [p for p in _all_params if id(p) not in _dag_names]
        _dag = [p for p in _all_params if id(p) in _dag_names]
        optimizer = torch.optim.Adam([
            {"params": _other, "lr": _g0["lr"]},
            {"params": _dag, "lr": _gN["lr"], "betas": _gN.get("betas", (0.9, 0.999))},
        ])
        print("   ✓ optimizer reconstruit avec 2 groups (tout + A_dag)")
    else:
        optimizer = torch.optim.Adam(_all_params, lr=CONFIG.training.lr)
        print("   ✓ optimizer reconstruit (single group)")

    print("✅ Models compiled.")
    print(f"   - encoder: {compile_cfg.get('encoder_mode', 'reduce-overhead')}")
    print(f"   - rcn:     {compile_cfg.get('rcn_mode', 'reduce-overhead')}")
    print(f"   - diff:    {compile_cfg.get('diffusion_mode', 'reduce-overhead')}")
    if sp_compiled is not None:
        print(f"   - spatial_projector: {compile_cfg.get('spatial_projector_mode', 'reduce-overhead')}")
    if hr_compiled is not None:
        print(f"   - hr_ident_head:     {compile_cfg.get('hr_ident_mode', 'reduce-overhead')}")
else:
    print("ℹ️  torch.compile désactivé dans la config.")
'''


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Locate cells by content (cell numbers shifted after previous patches).
    # Detection order: original phrase → patched sentinel → fail.
    target40 = target41 = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if (
            ("GPU Load Balancing Optimization" in src and "USE_MULTI_GPU" in src)
            or ("cudnn.benchmark + set_float32_matmul_precision" in src)
        ) and target40 is None:
            target40 = i
        if (
            ("Apply torch.compile if enabled" in src)
            or ("torch.compile sur tous les modules entraînables" in src)
        ) and target41 is None:
            target41 = i

    if target40 is None or target41 is None:
        print(f"[ERROR] Cellules introuvables (40={target40}, 41={target41}).")
        return 2

    src40 = "".join(cells[target40].get("source", []))
    src41 = "".join(cells[target41].get("source", []))
    n_changed = 0

    if SENTINEL not in src40:
        cells[target40]["source"] = CELL_40_NEW.splitlines(keepends=True)
        cells[target40]["outputs"] = []
        cells[target40]["execution_count"] = None
        print(f"  ↪ Patched cell {target40} (cudnn.benchmark global).")
        n_changed += 1
    else:
        print(f"✓ Cellule {target40} déjà patchée.")

    if SENTINEL not in src41:
        cells[target41]["source"] = CELL_41_NEW.splitlines(keepends=True)
        cells[target41]["outputs"] = []
        cells[target41]["execution_count"] = None
        print(f"  ↪ Patched cell {target41} (compile + optimizer fix).")
        n_changed += 1
    else:
        print(f"✓ Cellule {target41} déjà patchée.")

    if n_changed:
        NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"💾 Saved {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
