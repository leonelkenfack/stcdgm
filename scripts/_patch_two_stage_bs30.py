"""
BS30 — rewrite FINAL_VALIDATION cell for causal_concat + correct iteration.

Two bugs surfaced after running cell 60 with the BS18-trained checkpoint:

1. Iteration: ``for _bi, _raw_batch in enumerate(val_dataloader)`` yields
   *lists* of samples (the dataloader uses ``collate_fn=lambda x: x``).
   ``convert_sample_to_batch`` expects a dict → TypeError on
   ``sample["lr"]``. Training paths use ``iterate_batches(...)`` to
   normalize this, but the eval cell rolled its own loop.

2. Sampling: ``_sample_once`` calls ``diffusion.sample(...,
   conditioning_spatial=_cond_sp, apply_constraints=True)`` without
   ``mu_HR`` / ``baseline_log``. With ``causal_concat=True`` (current
   architecture), the same ValueError BS28 patches in cell 57 crops up
   here too. ``conditioning_spatial`` is also unused in causal_concat
   mode.

Patch (cell 60): full rewrite that
- detects ``causal_concat`` from the eager core
- loads ``regression_head`` weights from the checkpoint
- builds ``mu_HR = regression_head(H_T)`` + ``baseline_log = baseline[-1]``
  (mirrors Stage-2 training)
- calls ``diffusion.sample(..., mu_HR=, baseline_log=, scheduler_type=
  "edm_karras", apply_constraints=False)``
- iterates via ``iterate_batches(val_dataloader, builder, DEVICE)``
- replaces the now-irrelevant DAG-tokens ablation with a *μ_HR
  ablation* (zero out the Stage-1 head output and measure how much the
  diffusion output shifts → measures whether Stage-1 conditions Stage-2)

Idempotent — sentinel ``BS30_FINAL_VALIDATION``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


NEW_SOURCE = '''# >>> FINAL_VALIDATION (BS30 — causal_concat aware)
# Validation finale post-training : sampling + ablation μ_HR.
import json
import time
from pathlib import Path
import numpy as np
import torch

from st_cdgm.evaluation import (
    compute_f1_extremes,
    compute_spectrum_distance,
)


# ── 1. Recharger le checkpoint last (fallback best) ─────────────────
_ckpt_dir = Path(CONFIG.checkpoint.get("save_dir", "models"))
# BS27_SMOKE_FIX: prefer ``epoch_last.pth`` — ``epoch_best.pth`` may be
# stale (BS18 Stage 2 always passes improved=False).
_last = _ckpt_dir / "epoch_last.pth"
_best = _ckpt_dir / "epoch_best.pth"
_ckpt_path = _last if _last.exists() else _best
if not _ckpt_path.exists():
    raise FileNotFoundError(
        f"Aucun checkpoint dans {_ckpt_dir}. La boucle d'entraînement a-t-elle "
        f"persisté au moins une époque ?"
    )
print(f"📦 Loading checkpoint: {_ckpt_path}")
_ckpt = torch.load(_ckpt_path, map_location=DEVICE, weights_only=False)
print(f"   epoch={_ckpt.get('epoch')} / {_ckpt.get('epochs_total')}, "
      f"val_loss={_ckpt.get('val_loss')}, best_val_loss={_ckpt.get('best_val_loss')}")

# Recharge state_dicts via le helper BS22+BS23+BS24 (cell 47).
_persist_load_state_dict(encoder, _ckpt.get("encoder_state_dict"))
_persist_load_state_dict(rcn_cell, _ckpt.get("rcn_cell_state_dict"))
_persist_load_state_dict(diffusion, _ckpt.get("diffusion_state_dict"))
_rh = globals().get("regression_head", None)
if _rh is not None and _ckpt.get("regression_head_state_dict") is not None:
    _persist_load_state_dict(_rh, _ckpt["regression_head_state_dict"])
if "spatial_projector" in globals() and spatial_projector is not None:
    _persist_load_state_dict(spatial_projector, _ckpt.get("spatial_projector_state_dict"))
if "hr_ident_head" in globals() and hr_ident_head is not None:
    _persist_load_state_dict(hr_ident_head, _ckpt.get("hr_ident_head_state_dict"))

encoder.eval()
rcn_runner.cell.eval()
diffusion.eval()
if _rh is not None:
    _rh.eval()
if "spatial_projector" in globals() and spatial_projector is not None:
    spatial_projector.eval()


# ── 2. Detect mode ──────────────────────────────────────────────────
_diff_core = getattr(diffusion, "_orig_mod", diffusion)
_diff_core = getattr(_diff_core, "module", _diff_core)
_causal_concat = bool(getattr(_diff_core, "causal_concat", False))
print(f"   causal_concat = {_causal_concat}, regression_head = {_rh is not None}")


# ── 3. Helpers : conditioning + sampling ────────────────────────────
def _build_inputs(_batch):
    """Returns (conditioning, mu_HR, baseline_log, target) on DEVICE."""
    _hetero = _batch["hetero"]
    _lr = _batch["lr"]
    if isinstance(_lr, torch.Tensor):
        _lr = _lr.to(DEVICE)
    _H_init = encoder.init_state(_hetero).to(DEVICE)
    _drivers = [_lr[t] for t in range(_lr.shape[0])]
    _seq = rcn_runner.run(_H_init, _drivers, reconstruction_sources=None)
    _H_T = _seq.states[-1]
    _cond = encoder.project_state_tensor(_H_T).to(DEVICE)

    _target = _batch[CONFIG.training.get("residual_key", "residual")][-1].to(DEVICE)
    if _target.dim() == 3:
        _target = _target.unsqueeze(0)

    _mu_HR = None
    _baseline_log = None
    if _causal_concat and _rh is not None:
        _mu_HR = _rh(_H_T)
        if _mu_HR.shape[-2:] != _target.shape[-2:]:
            _mu_HR = torch.nn.functional.interpolate(
                _mu_HR, size=_target.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        _baseline_t = _batch["baseline"][-1].to(DEVICE)
        if _baseline_t.dim() == _mu_HR.dim() - 1:
            _baseline_t = _baseline_t.unsqueeze(0)
        _baseline_log = _baseline_t  # already log1p-encoded
        _mu_HR = torch.nan_to_num(_mu_HR, nan=0.0, posinf=0.0, neginf=0.0)
        _baseline_log = torch.nan_to_num(_baseline_log, nan=0.0, posinf=0.0, neginf=0.0)
    return _cond, _mu_HR, _baseline_log, _target


def _sample_once(_cond, _mu_HR, _baseline_log):
    _kwargs = dict(
        num_steps=int(CONFIG.diffusion.get(
            "eval_num_steps", 18 if _causal_concat else 30)),
        scheduler_type=("edm_karras" if _causal_concat
                        else CONFIG.diffusion.get("scheduler_type", "dpm_solver++")),
        cfg_scale=float(CONFIG.diffusion.get("cfg_scale", 0.0)),
        apply_constraints=False,  # 1-channel residual: constraints branch unsafe
    )
    if _causal_concat:
        _kwargs["mu_HR"] = _mu_HR
        _kwargs["baseline_log"] = _baseline_log
    return _diff_core.sample(conditioning=_cond, **_kwargs).residual


# ── 4. Boucle d'évaluation ──────────────────────────────────────────
N_TEST_BATCHES = 16
K_SAMPLES = 4              # ensemble pour CRPS/spread
N_INTERVENTION = 4         # nb de batches pour l'ablation μ_HR

print(f"\\n🧪 Sampling {N_TEST_BATCHES} batches × {K_SAMPLES} samples")
print(f"   eval_num_steps={int(CONFIG.diffusion.get('eval_num_steps', 18 if _causal_concat else 30))}, "
      f"scheduler={'edm_karras' if _causal_concat else CONFIG.diffusion.get('scheduler_type', 'dpm_solver++')}, "
      f"cfg_scale={CONFIG.diffusion.get('cfg_scale', 0.0)}")

_all_means = []
_all_stds = []
_all_targets = []
_intervention = []
_t_eval_start = time.time()
_count = 0

with torch.no_grad():
    for converted_batches in iterate_batches(val_dataloader, builder, DEVICE):
        for _batch in converted_batches:
            if _count >= N_TEST_BATCHES:
                break
            _cond, _mu_HR, _baseline_log, _target = _build_inputs(_batch)

            _samples_k = torch.stack(
                [_sample_once(_cond, _mu_HR, _baseline_log) for _ in range(K_SAMPLES)],
                dim=0,
            )
            _all_means.append(_samples_k.mean(dim=0))
            _all_stds.append(_samples_k.std(dim=0))
            _all_targets.append(_target)

            # Ablation μ_HR : measure Stage-1 → Stage-2 contribution.
            if _count < N_INTERVENTION and _causal_concat and _mu_HR is not None:
                _mu_zero = torch.zeros_like(_mu_HR)
                _s_real = _sample_once(_cond, _mu_HR, _baseline_log)
                _s_zero = _sample_once(_cond, _mu_zero, _baseline_log)
                _delta = (_s_real - _s_zero).abs().mean().item()
                _signal = _s_real.abs().mean().item()
                _ratio = _delta / max(_signal, 1e-8)
                _intervention.append(_ratio)
                print(f"   batch {_count+1:2d} | μ_HR ablation Δ/signal = {_ratio*100:6.2f}%")
            else:
                print(f"   batch {_count+1:2d} | sampled K={K_SAMPLES}")

            _count += 1
        if _count >= N_TEST_BATCHES:
            break

_eval_time = time.time() - _t_eval_start


# ── 5. Métriques agrégées ───────────────────────────────────────────
_pred_mean = torch.cat(_all_means, dim=0).cpu()
_pred_std = torch.cat(_all_stds, dim=0).cpu()
_targets = torch.cat(_all_targets, dim=0).cpu()
_valid = torch.isfinite(_targets)
_pred_clean = torch.where(_valid, _pred_mean, torch.zeros_like(_pred_mean))
_targ_clean = torch.where(_valid, _targets, torch.zeros_like(_targets))
_diff_sq = ((_pred_clean - _targ_clean) ** 2)[_valid]
_rmse = float(_diff_sq.mean().sqrt().item()) if _valid.any() else float("nan")
_mae = float((_pred_clean - _targ_clean).abs()[_valid].mean().item()) if _valid.any() else float("nan")
_spread = float(_pred_std[_valid].mean().item()) if _valid.any() else float("nan")

_f1 = {}
try:
    _f1 = compute_f1_extremes(_pred_clean, _targ_clean, threshold_percentiles=[95.0, 99.0])
except Exception as _e:
    print(f"⚠️  F1 extremes a échoué : {_e}")

_rapsd_d = None
try:
    _rapsd_d = float(compute_spectrum_distance(_pred_clean[0], _targ_clean[0]))
except Exception as _e:
    print(f"⚠️  RAPSD distance a échoué : {_e}")

_dag_avg = float(np.mean(_intervention)) if _intervention else None


# ── 6. Print + JSON ─────────────────────────────────────────────────
print("\\n" + "=" * 72)
print("📊 FINAL VALIDATION METRICS")
print("=" * 72)
print(f"  Checkpoint            : {_ckpt_path.name}  (epoch {_ckpt.get('epoch')})")
print(f"  causal_concat         : {_causal_concat}")
print(f"  Test batches          : {len(_all_targets)}")
print(f"  Samples/batch         : {K_SAMPLES}")
print(f"  Eval time             : {_eval_time:.1f}s ({_eval_time/max(1,len(_all_targets)):.2f}s/batch)")
print()
print(f"  RMSE (ensemble mean)  : {_rmse:.6f}")
print(f"  MAE                   : {_mae:.6f}")
print(f"  Spread (ensemble std) : {_spread:.6f}")
for _k, _v in _f1.items():
    print(f"  {_k:<22}: {_v:.4f}")
if _rapsd_d is not None:
    print(f"  RAPSD distance        : {_rapsd_d:.6f}")
print()
if _dag_avg is not None:
    _pct = _dag_avg * 100.0
    print(f"  🧠 μ_HR ABLATION (Stage-1 → Stage-2 conditioning):")
    print(f"     Δ/signal = {_pct:.3f}%   (sur {len(_intervention)} batches)")
    if _dag_avg < 0.001:
        verdict = "MU_HR_IGNORED"
        print(f"     ⚠️  μ_HR IGNORÉ — diffusion ne dépend pas de Stage 1 (Δ < 0.1%)")
    elif _dag_avg < 0.01:
        verdict = "WEAK"
        print(f"     ⚠️  μ_HR FAIBLEMENT utilisé (0.1% ≤ Δ < 1%)")
    else:
        verdict = "MU_HR_CONDITIONS"
        print(f"     ✅ μ_HR CONDITIONNE la diffusion (Δ ≥ 1%)")
else:
    verdict = "N/A"
    print("  ⚠️  Ablation μ_HR non exécutée (causal_concat=False ou regression_head absent)")

# Sauvegarde JSON pour comparaison entre runs.
_metrics = {
    "checkpoint": str(_ckpt_path),
    "epoch": _ckpt.get("epoch"),
    "epochs_total": _ckpt.get("epochs_total"),
    "best_val_loss": _ckpt.get("best_val_loss"),
    "causal_concat": _causal_concat,
    "n_test_batches": len(_all_targets),
    "k_samples": K_SAMPLES,
    "eval_time_s": _eval_time,
    "rmse": _rmse,
    "mae": _mae,
    "spread_mean": _spread,
    "f1_extremes": _f1,
    "rapsd_distance": _rapsd_d,
    "mu_HR_ablation": {
        "delta_signal_ratio_avg": _dag_avg,
        "per_batch": _intervention,
        "verdict": verdict,
    },
    "config_eval_num_steps": int(CONFIG.diffusion.get(
        "eval_num_steps", 18 if _causal_concat else 30)),
    "config_cfg_scale": float(CONFIG.diffusion.get("cfg_scale", 0.0)),
    "config_block_out_channels": list(CONFIG.diffusion.unet_kwargs.block_out_channels),
}
_metrics_path = _ckpt_dir / "final_validation_metrics.json"
_metrics_path.write_text(json.dumps(_metrics, indent=2, default=str))
print(f"\\n💾 Métriques sauvegardées : {_metrics_path}")
'''


def patch_cell_60() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "FINAL_VALIDATION" in s
                     and "compute_f1_extremes" in s)
    if idx is None:
        print("  ! FINAL_VALIDATION cell not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS30_FINAL_VALIDATION" in src or "FINAL_VALIDATION (BS30" in src:
        print(f"  = cell {idx} already patched (BS30)")
        return 0
    cells[idx]["source"] = NEW_SOURCE.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {idx}: FINAL_VALIDATION rewritten for causal_concat + iterate_batches")
    return 1


def main() -> int:
    print("=== BS30 : FINAL_VALIDATION rewrite (cell 60) ===")
    n = patch_cell_60()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
