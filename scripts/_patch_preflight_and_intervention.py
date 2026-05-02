"""
Ajoute deux cells diagnostic aux notebooks ST-CDGM (Phase 2/4 du hyperplan).

1. Training notebook : EDM_PREFLIGHT_TRAINING — 100 training steps de
   diagnostic AVANT le grand training. Surveille loss + ||A_dag.grad|| /
   ||UNet.grad|| pour valider que la pipeline EDM est saine et que le
   detach sur A_dag fonctionne (Trace Trap fermé). Aucun optimizer.step()
   n'est appelé : le modèle reste à son état d'init / resume.

2. Validation notebook : EDM_DAG_INTERVENTION — test causal d'intervention
   (Objectif O6 du papier oracle.tex). Calcule
   delta_zero/signal = MSE(D(A=0), D(A_real)) / MSE(D(A_real), target)
   sur N samples. > 0.05 = DAG conditionne effectivement la diffusion.

Idempotent — sentinel-guarded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"
VAL_NB = ROOT / "st_cdgm_validation_inference.ipynb"


PREFLIGHT_TRAINING_CELL = r'''# >>> EDM_PREFLIGHT_TRAINING
# 100 training steps de diagnostic AVANT le grand training (~10h Colab).
# Surveille :
#   - loss : doit décroitre log-linéairement
#   - ||A_dag.grad|| / ||UNet.grad|| : doit rester < 0.1 (Trace Trap fermé)
#   - aucun NaN/Inf
# Aucun optimizer.step() n'est appelé — le modèle reste à son état initial.
import math
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

# Skip si on resume depuis un checkpoint (pas besoin de re-vérifier la pipeline)
_PREFLIGHT_SKIP = False
try:
    from pathlib import Path as _Path
    _ckpt_dir = _Path(str(CKPT_SAVE_DIR))
    if (_ckpt_dir / "epoch_last.pth").exists():
        print("ℹ️  Checkpoint existant — pre-flight sauté (mode resume).")
        _PREFLIGHT_SKIP = True
except Exception:
    pass

if not _PREFLIGHT_SKIP:
    print("🛫 Pre-flight EDM : 100 training steps de diagnostic")
    print("=" * 70)

    _PF_N_STEPS = 100
    _pf_loader = DataLoader(
        dataset, batch_size=1, num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
        collate_fn=lambda x: x,
    )

    encoder.train()
    rcn_runner.cell.train()
    diffusion.train()
    if "spatial_projector" in dir() and spatial_projector is not None:
        spatial_projector.train()

    optimizer.zero_grad(set_to_none=True)

    _pf_losses, _pf_dag_g, _pf_unet_g = [], [], []
    _pf_t0 = time.time()
    _pf_step = 0

    from st_cdgm.training.training_loop import loss_diffusion as _pf_loss_fn

    for _converted in iterate_batches(_pf_loader, builder, DEVICE):
        if _pf_step >= _PF_N_STEPS:
            break
        for _batch in _converted:
            if _pf_step >= _PF_N_STEPS:
                break

            _lr = _batch["lr"].to(DEVICE)
            _target = _batch["residual"][-1].to(DEVICE)
            if _target.dim() == 3:
                _target = _target.unsqueeze(0)

            _H_init = encoder.init_state(_batch["hetero"]).to(DEVICE)
            _drivers = [_lr[t] for t in range(_lr.shape[0])]
            _seq = rcn_runner.run(_H_init, _drivers, reconstruction_sources=None)
            _H_last = _seq.states[-1]
            _cond = encoder.project_state_tensor(_H_last).to(DEVICE)

            _cond_sp = None
            if "spatial_projector" in dir() and spatial_projector is not None:
                _sp = spatial_projector.module if hasattr(spatial_projector, "module") else spatial_projector
                _sp = getattr(_sp, "_orig_mod", _sp)
                if hasattr(_sp, "dag_mlp"):
                    _rcn_base = rcn_runner.cell.module if hasattr(rcn_runner.cell, "module") else rcn_runner.cell
                    _rcn_base = getattr(_rcn_base, "_orig_mod", _rcn_base)
                    _A = _rcn_base.A_dag
                    _A_masked = _A - torch.diag(torch.diagonal(_A))
                    _cond_sp = spatial_projector(_H_last, _A_masked).to(DEVICE)
                else:
                    _cond_sp = spatial_projector(_H_last).to(DEVICE)

            _loss = _pf_loss_fn(diffusion, _target, _cond, conditioning_spatial=_cond_sp)
            optimizer.zero_grad(set_to_none=True)
            _loss.backward()

            # Norms — only inspect, no optimizer.step()
            _A_grad_n = float("nan")
            try:
                _rcn_b = rcn_runner.cell.module if hasattr(rcn_runner.cell, "module") else rcn_runner.cell
                _rcn_b = getattr(_rcn_b, "_orig_mod", _rcn_b)
                if _rcn_b.A_dag.grad is not None:
                    _A_grad_n = _rcn_b.A_dag.grad.norm().item()
            except Exception:
                pass
            _unet_sq = 0.0
            for _p in diffusion.parameters():
                if _p.grad is not None:
                    _unet_sq += _p.grad.float().norm().item() ** 2
            _unet_n = math.sqrt(_unet_sq)

            _pf_losses.append(_loss.item())
            _pf_dag_g.append(_A_grad_n)
            _pf_unet_g.append(_unet_n)
            _pf_step += 1

            if _pf_step == 1 or _pf_step % 20 == 0:
                _r = _A_grad_n / max(_unet_n, 1e-12)
                print(f"  step {_pf_step:3d} | loss={_loss.item():.4e} | "
                      f"||UNet.g||={_unet_n:.3e} | ||A_dag.g||={_A_grad_n:.3e} | "
                      f"ratio={_r:.2e}")

    _pf_dt = time.time() - _pf_t0
    print("=" * 70)
    print(f"✓ {_pf_step} steps en {_pf_dt:.1f}s ({_pf_step/_pf_dt:.2f} it/s)")

    _losses = np.asarray(_pf_losses)
    _dag_grads = np.asarray([g for g in _pf_dag_g if not math.isnan(g)])
    _unet_grads = np.asarray(_pf_unet_g)

    print("\n📊 Diagnostic")
    print(f"  loss     : start={_losses[0]:.3e} | mean(last 10)={_losses[-10:].mean():.3e}"
          f" | trend = {_losses[0]/max(_losses[-10:].mean(),1e-12):.2f}x")
    if len(_dag_grads):
        print(f"  A_dag.g  : mean={_dag_grads.mean():.3e}, max={_dag_grads.max():.3e}")
    print(f"  UNet.g   : mean={_unet_grads.mean():.3e}, max={_unet_grads.max():.3e}")

    _PF_PASS = True
    if not np.isfinite(_losses).all():
        print("  🚨 NaN/Inf détecté dans la loss — ABORT")
        _PF_PASS = False

    if len(_dag_grads):
        _ratio = _dag_grads.mean() / max(_unet_grads.mean(), 1e-12)
        print(f"  ratio dag/unet : {_ratio:.2e}")
        if _ratio > 0.1:
            print("  🚨 ratio > 0.1 — A_dag absorbe trop de gradient.")
            print("       Vérifier que rcn.dag_grad_gate.enabled = false dans le YAML.")
            _PF_PASS = False
        elif _ratio < 1e-4:
            print("  ✓ ratio < 1e-4 — detach effectif (Trace Trap fermé).")
        else:
            print(f"  ⚠️  ratio modéré ({_ratio:.2e}). Tolérable mais à surveiller.")

    if _losses[-10:].mean() > _losses[0]:
        print("  ⚠️  loss n'a pas décru sur 100 steps — modèle peut-être bloqué.")
    else:
        print(f"  ✓ loss décroit (×{_losses[0]/max(_losses[-10:].mean(),1e-12):.2f}).")

    print("=" * 70)
    if _PF_PASS:
        print("🚦 GATE : PASS — lance la cellule training principale.")
    else:
        print("🛑 GATE : FAIL — kill runtime et investigate AVANT le grand training.")
    print("=" * 70)

    # Reset état pour démarrer le training propre
    optimizer.zero_grad(set_to_none=True)
    torch.manual_seed(int(CONFIG.training.get("seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(CONFIG.training.get("seed", 42)))
'''


DAG_INTERVENTION_CELL = r'''# >>> EDM_DAG_INTERVENTION
# Objectif O6 du papier oracle.tex : prouver que A_dag conditionne
# effectivement le modèle de diffusion. Si setting A_dag = 0 ne change
# rien à la sortie, la DAG est décorative.
#
# Métrique : delta_zero/signal = MSE(D(A=0), D(A_real)) / MSE(D(A_real), target)
# Seuil de validation : > 0.05 (5% de variance attribuable à la DAG)
import numpy as np
import torch

_INTERVENTION_N_SAMPLES = 20
_INTERVENTION_SEED = 12345


def _intervention_run_one(sample, A_override=None):
    """Run diffusion sample once, optionally with A_dag overridden in-place.
    Returns (pred_t_mean_cpu, target_cpu)."""
    batch = convert_sample_to_batch(sample, builder, DEVICE)
    lr_data = batch["lr"].to(DEVICE)
    target_batch, baseline_batch = extract_target_and_baseline_t_mean(batch, DEVICE)

    _rcn_base = rcn_runner.cell.module if hasattr(rcn_runner.cell, "module") else rcn_runner.cell
    _rcn_base = getattr(_rcn_base, "_orig_mod", _rcn_base)
    _A_saved = _rcn_base.A_dag.data.clone()

    if A_override is not None:
        _rcn_base.A_dag.data.copy_(A_override.to(_rcn_base.A_dag.device))

    try:
        H_init = encoder.init_state(batch["hetero"]).to(DEVICE)
        drivers = [lr_data[t] for t in range(lr_data.shape[0])]
        seq_out = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
        H_last = seq_out.states[-1]
        cond = encoder.project_state_tensor(H_last).to(DEVICE)

        cond_sp = None
        _sp = globals().get("spatial_projector", None)
        if _sp is not None:
            _sp_base = _sp.module if hasattr(_sp, "module") else _sp
            _sp_base = getattr(_sp_base, "_orig_mod", _sp_base)
            if hasattr(_sp_base, "dag_mlp"):
                A_now = _rcn_base.A_dag
                A_masked = A_now - torch.diag(torch.diagonal(A_now))
                cond_sp = _sp(H_last, A_masked).to(DEVICE)
            else:
                cond_sp = _sp(H_last).to(DEVICE)

        gen = torch.Generator(device=DEVICE).manual_seed(_INTERVENTION_SEED)
        out = diffusion.sample(
            cond,
            num_steps=int(CONFIG.diffusion.get("eval_num_steps", 18)),
            scheduler_type=CONFIG.diffusion.get("scheduler_type", "edm_karras"),
            apply_constraints=False,
            baseline=baseline_batch,
            cfg_scale=float(CONFIG.diffusion.get("cfg_scale", 1.0)),
            conditioning_spatial=cond_sp,
            generator=gen,
        )
        return out.t_mean.detach().cpu(), target_batch.cpu()
    finally:
        _rcn_base.A_dag.data.copy_(_A_saved)


print(f"🧪 DAG intervention test sur {_INTERVENTION_N_SAMPLES} samples du GCM {TEST_GCMS[0]}")

_test_pipe = build_test_pipeline(TEST_GCMS[0])
_test_ds = _test_pipe.build_sequence_dataset(seq_len=SEQ_LEN, as_torch=True)
_iv_iter = iter(_test_ds)

_rcn_base_for_zero = rcn_runner.cell.module if hasattr(rcn_runner.cell, "module") else rcn_runner.cell
_rcn_base_for_zero = getattr(_rcn_base_for_zero, "_orig_mod", _rcn_base_for_zero)
A_zero = torch.zeros_like(_rcn_base_for_zero.A_dag.data)

_iv_ratios = []
_iv_mse_real_target = []
_iv_mse_zero_real = []

with torch.no_grad():
    for _i in range(_INTERVENTION_N_SAMPLES):
        try:
            _s = next(_iv_iter)
        except StopIteration:
            break

        pred_real, target = _intervention_run_one(_s, A_override=None)
        pred_zero, _ = _intervention_run_one(_s, A_override=A_zero)

        _mse_rt = float((pred_real - target).pow(2).mean())
        _mse_zr = float((pred_zero - pred_real).pow(2).mean())
        _ratio = _mse_zr / max(_mse_rt, 1e-12)

        _iv_mse_real_target.append(_mse_rt)
        _iv_mse_zero_real.append(_mse_zr)
        _iv_ratios.append(_ratio)

        if (_i + 1) % 5 == 0:
            print(f"  sample {_i+1:2d}/{_INTERVENTION_N_SAMPLES} | "
                  f"MSE(D(A),target)={_mse_rt:.3e} | "
                  f"MSE(D(0),D(A))={_mse_zr:.3e} | ratio={_ratio:.4f}")

_arr = np.asarray(_iv_ratios)
print(f"\n📊 DAG Intervention Summary (n={len(_arr)})")
print(f"   delta_zero/signal mean   = {_arr.mean():.4f}")
print(f"   delta_zero/signal median = {np.median(_arr):.4f}")
print(f"   delta_zero/signal p95    = {np.percentile(_arr, 95):.4f}")
print(f"   MSE(real,target) mean    = {np.mean(_iv_mse_real_target):.3e}")
print(f"   MSE(zero,real) mean      = {np.mean(_iv_mse_zero_real):.3e}")

if _arr.mean() > 0.05:
    print("   ✓ ratio > 0.05 — la DAG conditionne EFFECTIVEMENT (O6 paper).")
elif _arr.mean() > 0.01:
    print("   ⚠️  ratio ∈ [0.01, 0.05] — conditioning faible. Plus d'epochs ou cross-attn.")
else:
    print("   🚨 ratio < 0.01 — DAG décorative (UNet ignore A_dag).")

# Plots
import matplotlib.pyplot as plt
_fig, _ax = plt.subplots(1, 2, figsize=(11, 3.5))
_ax[0].hist(_arr, bins=15, edgecolor="k", alpha=0.7)
_ax[0].axvline(0.05, color="r", linestyle="--", label="seuil 0.05")
_ax[0].set_xlabel("delta_zero/signal")
_ax[0].set_ylabel("count")
_ax[0].set_title("DAG Intervention — distribution")
_ax[0].legend()

_x = np.array(_iv_mse_real_target)
_y = np.array(_iv_mse_zero_real)
_ax[1].scatter(_x, _y, alpha=0.6, s=30)
if _x.max() > 0:
    _xl = np.linspace(0, _x.max(), 50)
    _ax[1].plot(_xl, 0.05 * _xl, "r--", alpha=0.6, label="ratio = 0.05")
_ax[1].set_xlabel("MSE(D(A_real), target)")
_ax[1].set_ylabel("MSE(D(A=0), D(A_real))")
_ax[1].set_title("Sample-level scatter")
_ax[1].legend()
plt.tight_layout()
plt.show()

# Persiste pour le rapport final
intervention_report = {
    "n_samples": int(len(_arr)),
    "delta_signal_ratio_mean": float(_arr.mean()),
    "delta_signal_ratio_median": float(np.median(_arr)),
    "delta_signal_ratio_p95": float(np.percentile(_arr, 95)),
    "mse_real_target_mean": float(np.mean(_iv_mse_real_target)),
    "mse_zero_real_mean": float(np.mean(_iv_mse_zero_real)),
    "passes_o6_threshold": bool(_arr.mean() > 0.05),
}
'''


def _make_code_cell(src, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


def patch_training_notebook() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    if _find_cell(cells, lambda s: "EDM_PREFLIGHT_TRAINING" in s) is not None:
        print("  = training EDM_PREFLIGHT_TRAINING déjà présent")
        return 0

    # Insérer après EDM_PREFLIGHT_SIGMA_DATA, avant la cellule training principale.
    sigma_idx = _find_cell(cells, lambda s: "EDM_PREFLIGHT_SIGMA_DATA" in s)
    if sigma_idx is None:
        print("  ! EDM_PREFLIGHT_SIGMA_DATA introuvable — fallback : avant la cellule training")
        train_idx = _find_cell(cells, lambda s: "Entraînement strict: split train/val" in s)
        if train_idx is None:
            print("  ! Impossible de localiser un point d'insertion - skip")
            return 0
        insert_at = train_idx
    else:
        insert_at = sigma_idx + 1

    cells.insert(insert_at, _make_code_cell(PREFLIGHT_TRAINING_CELL, "edm_preflight_training"))
    n_changed += 1
    print(f"  + training EDM_PREFLIGHT_TRAINING inséré en position {insert_at}")

    if n_changed > 0:
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n_changed


def patch_validation_notebook() -> int:
    nb = json.loads(VAL_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    if _find_cell(cells, lambda s: "EDM_DAG_INTERVENTION" in s) is not None:
        print("  = validation EDM_DAG_INTERVENTION déjà présent")
        return 0

    # Insérer après la cellule SHD test (sentinel EDM_SHD_TEST), ou avant le main eval loop.
    shd_idx = _find_cell(cells, lambda s: "EDM_SHD_TEST" in s)
    if shd_idx is None:
        print("  ! SHD anchor introuvable — fallback : avant l'eval principal")
        eval_idx = _find_cell(cells, lambda s: "for gcm in TEST_GCMS:" in s)
        if eval_idx is None:
            print("  ! Impossible de localiser un point d'insertion - skip")
            return 0
        insert_at = eval_idx
    else:
        insert_at = shd_idx + 1

    cells.insert(insert_at, _make_code_cell(DAG_INTERVENTION_CELL, "edm_dag_intervention"))
    n_changed += 1
    print(f"  + validation EDM_DAG_INTERVENTION inséré en position {insert_at}")

    if n_changed > 0:
        VAL_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n_changed


def main() -> int:
    print("=== Training notebook ===")
    n1 = patch_training_notebook()
    print("\n=== Validation notebook ===")
    n2 = patch_validation_notebook()
    print(f"\n{n1 + n2} modification(s) appliquée(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
