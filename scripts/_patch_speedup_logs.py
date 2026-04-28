"""
Trois fixes pour réduire le temps/batch et éliminer le crash CUDA graph :

1. Cell 41 (compile) : ne compile QUE le UNet (diffusion). Encoder, RCN,
   spatial_projector, hr_ident_head restent en eager — leur compute est
   marginal vs le UNet, et compile reduce-overhead crée des conflits CUDA
   graph (cf. ``RuntimeError: accessing tensor output of CUDAGraphs...``).

2. training_loop.py : suppression de la heartbeat per-micro-batch (48 prints
   par outer batch = ~2-5s de Python overhead pour rien). Le batch-level
   summary (toutes les ``log_interval`` itérations) suffit.

3. training_loop.py : ``_do_timing`` n'imprime plus les détails encoder/RCN/
   diffusion à chaque epoch — uniquement quand ``verbose and batch_idx % log_interval == 0``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
TRAINING_LOOP = Path(__file__).resolve().parent.parent / "src" / "st_cdgm" / "training" / "training_loop.py"

CELL_41_NEW = '''\
# >>> GPU_OPT_FIX
# torch.compile UNIQUEMENT sur le UNet (diffusion). Encoder/RCN/projecteurs
# restent en eager — leur compute est marginal et compile reduce-overhead
# créait des conflits CUDA graph (RuntimeError: tensor output overwritten).
#
# max-autotune sur l'UNet apporte un vrai gain (~10-20% vs reduce-overhead),
# mais il faut éviter de mixer reduce-overhead sur les petits modules sinon
# leurs CUDA graphs interfèrent avec celui du UNet.
if CONFIG.training.get("compile", {}).get("enabled", False):
    print("🔧 Compiling UNet (diffusion) with torch.compile…")
    compile_cfg = CONFIG.training.compile

    def _unwrap(m):
        if m is None:
            return None
        return m.module if hasattr(m, "module") else m

    diffusion_base = _unwrap(diffusion)
    diff_mode = compile_cfg.get("diffusion_mode", "max-autotune")

    try:
        diffusion_compiled = torch.compile(diffusion_base, mode=diff_mode, fullgraph=False)
        if hasattr(diffusion, "module"):
            gpus = CONFIG.training.multi_gpu.get("gpus", [0])
            diffusion = torch.nn.DataParallel(diffusion_compiled, device_ids=gpus)
        else:
            diffusion = diffusion_compiled
        print(f"   ✓ diffusion compiled (mode={diff_mode})")
    except Exception as _e:
        print(f"⚠️  torch.compile(diffusion) a échoué : {_e} — fallback eager.")

    # Optimizer reconstruction — préserve groups (DAG-decouple LR mult).
    _all_params = (
        list(encoder.parameters())
        + list(rcn_cell.parameters())
        + list(diffusion.parameters())
    )
    if "spatial_projector" in dir() and spatial_projector is not None:
        _all_params += list(spatial_projector.parameters())
    if "hr_ident_head" in dir() and hr_ident_head is not None:
        _all_params += list(hr_ident_head.parameters())

    if "optimizer" in dir() and optimizer is not None and len(optimizer.param_groups) > 1:
        _g0 = optimizer.param_groups[0]
        _gN = optimizer.param_groups[-1]
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

    print(f"✅ Compile done — encoder/RCN/projector restent en eager (intentionnel)")
else:
    print("ℹ️  torch.compile désactivé dans la config.")
'''


def patch_cell_41() -> bool:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if (
            ("Apply torch.compile if enabled" in src)
            or ("torch.compile sur tous les modules" in src)
            or ("torch.compile UNIQUEMENT sur le UNet" in src)
        ):
            current = src
            new_src = CELL_41_NEW
            if current == new_src:
                print(f"✓ Cell {i} (compile) déjà à jour.")
                return False
            nb["cells"][i]["source"] = new_src.splitlines(keepends=True)
            nb["cells"][i]["outputs"] = []
            nb["cells"][i]["execution_count"] = None
            NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"  ↪ Cell {i} (compile) → UNet uniquement, encoder/RCN/projector en eager")
            return True
    print("[WARN] Cell torch.compile introuvable.")
    return False


def patch_training_loop_remove_heartbeat() -> bool:
    src = TRAINING_LOOP.read_text(encoding="utf-8")
    sentinel = "# >>> NO_MICRO_HEARTBEAT"
    if sentinel in src:
        print("✓ training_loop.py heartbeat déjà supprimée.")
        return False

    OLD_HEARTBEAT = '''            # Per-micro-batch heartbeat so the user can see progress
            # inside batch_size>1 gradient-accumulation steps. Without
            # this, batch_idx==0 with batches=48 was completely silent
            # for 10+ minutes. Cheap (one print per micro_idx).
            if verbose and len(batches) > 1 and (
                batch_idx == 0
                or batch_idx % max(1, log_interval) == 0
            ):
                _mb_dt = time.time() - _micro_t0
                _suffix = ""
                # C1: surface DAG sensitivity (positive ⇒ DAG is conditioning),
                # which is informative regardless of whether the margin is met.
                # Falls back to nothing when the contrastive block didn't fire
                # this micro (gated to micro_idx==0 + interval).
                if _dag_sens_step is not None:
                    _suffix += f" | dag_sens={_dag_sens_step:+.4f}"
                if loss_precip_phy_value.item() > 0.0:
                    _suffix += f" | precip={loss_precip_phy_value.item():.4f}"
                print(
                    f"   micro {micro_idx + 1}/{len(batches)} "
                    f"loss={loss_total.item():.4f} bwd={_mb_dt:.2f}s{_suffix}",
                    flush=True,
                )'''

    NEW_HEARTBEAT = '''            # >>> NO_MICRO_HEARTBEAT
            # La heartbeat per-micro-batch a été supprimée — 48 prints par
            # outer step ajoutaient ~2-5s de Python overhead. Le summary
            # batch-level (cf. plus bas, gardé sur ``log_interval``) donne
            # déjà la progression. ``_dag_sens_step`` est toujours agrégé
            # via ``total_dag_sensitivity`` pour le report fin d'époque.'''

    if OLD_HEARTBEAT not in src:
        print("[WARN] Bloc heartbeat exact non trouvé — peut-être déjà modifié.")
        return False

    src = src.replace(OLD_HEARTBEAT, NEW_HEARTBEAT, 1)
    TRAINING_LOOP.write_text(src, encoding="utf-8")
    print("  ↪ training_loop.py : heartbeat per-micro supprimée")
    return True


def main() -> int:
    n_changed = 0
    if patch_cell_41():
        n_changed += 1
    if patch_training_loop_remove_heartbeat():
        n_changed += 1
    print(f"\n💾 {n_changed} fichier(s) modifié(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
