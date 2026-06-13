# >>> FINAL_VALIDATION (BS30 — causal_concat aware)
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
# >>> NONCAUSAL_VARIANT_SAFE — guard encoder/rcn_cell None en noncausal.
if globals().get("encoder") is not None:
    _persist_load_state_dict(encoder, _ckpt.get("encoder_state_dict"))
if globals().get("rcn_cell") is not None:
    _persist_load_state_dict(rcn_cell, _ckpt.get("rcn_cell_state_dict"))
# BS37_EMA_EVAL — preferer les poids EMA pour l'inference si presents.
_ema_sd_eval = _ckpt.get("diffusion_ema_state_dict")
# >>> BS41 FIX F4 — bypass-EMA escape hatch. Set this global to True
# *before* running the FINAL_VALIDATION cell to load the LIVE diffusion
# weights instead of the EMA shadow (useful when the EMA is suspected to
# be stale due to the V5 ema_steps reset bug).
_bs41_force_live = bool(globals().get("BS41_FORCE_LIVE_INFERENCE", False))
if _ema_sd_eval is not None and not _bs41_force_live:
    print("  🌗 BS37 EMA detected in checkpoint — loading EMA weights for FINAL_VALIDATION")
    _persist_load_state_dict(diffusion, _ema_sd_eval)
elif _bs41_force_live:
    print("  ⚠️  BS41 FORCE_LIVE_INFERENCE=True — bypass EMA, loading live diffusion weights")
    _persist_load_state_dict(diffusion, _ckpt.get("diffusion_state_dict"))
else:
    _persist_load_state_dict(diffusion, _ckpt.get("diffusion_state_dict"))
_rh = globals().get("regression_head", None)
if _rh is not None and _ckpt.get("regression_head_state_dict") is not None:
    _persist_load_state_dict(_rh, _ckpt["regression_head_state_dict"])
if "spatial_projector" in globals() and spatial_projector is not None:
    _persist_load_state_dict(spatial_projector, _ckpt.get("spatial_projector_state_dict"))
if "hr_ident_head" in globals() and hr_ident_head is not None:
    _persist_load_state_dict(hr_ident_head, _ckpt.get("hr_ident_head_state_dict"))

if globals().get("encoder") is not None:
    encoder.eval()
if globals().get("rcn_runner") is not None and hasattr(rcn_runner, "cell"):
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
from st_cdgm.evaluation.two_stage_inference import build_two_stage_inputs
from st_cdgm.training.stage1_paths import resolve_run_variant as _resolve_run_variant_final
_RUN_VARIANT_FINAL = _resolve_run_variant_final(CONFIG)

def _build_inputs(_batch):
    """Returns (conditioning, mu_HR, baseline_log, target) on DEVICE."""
    return build_two_stage_inputs(
        _batch,
        variant=_RUN_VARIANT_FINAL,
        regression_head=_rh,
        encoder=encoder if _RUN_VARIANT_FINAL == "causal" else None,
        rcn_runner=rcn_runner if _RUN_VARIANT_FINAL == "causal" else None,
        builder=builder,
        device=DEVICE,
    )


def _sample_once(_cond, _mu_HR, _baseline_log):
    # BS39: DPM-Solver++ supports causal_concat in diffusion_decoder._sample_dpm_solver.
    _kwargs = dict(
        num_steps=int(CONFIG.diffusion.get(
            "eval_num_steps", 18 if _causal_concat else 30)),
        scheduler_type=str(CONFIG.diffusion.get(
            "scheduler_type", "edm_karras" if _causal_concat else "dpm_solver++")),
        cfg_scale=float(CONFIG.diffusion.get("cfg_scale", 0.0)),
        apply_constraints=False,  # 1-channel residual: constraints branch unsafe
    )
    if _causal_concat:
        _kwargs["mu_HR"] = _mu_HR
        _kwargs["baseline_log"] = _baseline_log
    return _diff_core.sample(conditioning=_cond, **_kwargs).residual


# ── 4. Boucle d'évaluation ──────────────────────────────────────────
# >>> BS33b_K_SAMPLES : bump 4 → 16 (paper Tab.1 says 8-16).
# Eval cost reasonable post-BS32b (~6 min for 16×16 samples vs 1.5 min before).
N_TEST_BATCHES = 16
K_SAMPLES = int(globals().get("K_SAMPLES_OVERRIDE", 64))  # V5 ensemble averaging boost (16 -> 64); override via globals()
N_INTERVENTION = 4         # nb de batches pour l'ablation μ_HR

_eval_scheduler = str(CONFIG.diffusion.get(
    "scheduler_type", "edm_karras" if _causal_concat else "dpm_solver++"
))

print(f"\n🧪 Sampling {N_TEST_BATCHES} batches × {K_SAMPLES} samples")
print(f"   eval_num_steps={int(CONFIG.diffusion.get('eval_num_steps', 18 if _causal_concat else 30))}, "
      f"scheduler={_eval_scheduler}, "
      f"cfg_scale={CONFIG.diffusion.get('cfg_scale', 0.0)}")

_all_means = []
_all_stds = []
_all_targets = []
_intervention = []
# >>> BS31_DIAG_SHORTCUT
_all_mu_HR = []  # accumulate Stage-1 μ_HR per batch for shortcut probe
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
            # BS31_DIAG_SHORTCUT
            if _causal_concat and _mu_HR is not None:
                _all_mu_HR.append(_mu_HR.detach())

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

# >>> BS31f_FULL_PREDICTION
# In causal_concat mode the diffusion outputs δ̂ (the residual on top of
# μ_HR), but ``_targets`` is HR_residual = HR - baseline. To measure the
# end-to-end reconstruction skill, build ``_pred_full = μ_HR + δ̂`` and
# use it for all pixel-level metrics. Keep ``_pred_mean`` raw for the
# shortcut diagnostic (which inspects δ̂ specifically).
if _all_mu_HR:
    _mu_concat_eval = torch.cat(_all_mu_HR, dim=0).cpu()
    if _mu_concat_eval.shape == _pred_mean.shape:
        _pred_full = _pred_mean + _mu_concat_eval
        _metrics_scope = "full_prediction (μ_HR + δ̂)"
    else:
        print(f"   ⚠️  BS31f : μ_HR shape mismatch — fallback raw δ̂ for metrics")
        _pred_full = _pred_mean
        _metrics_scope = "delta_only (BS31f fallback)"
else:
    _pred_full = _pred_mean
    _metrics_scope = "delta_only (no μ_HR cached)"
print(f"   📐 metrics scope : {_metrics_scope}")

_pred_clean = torch.where(_valid, _pred_full, torch.zeros_like(_pred_full))
_targ_clean = torch.where(_valid, _targets, torch.zeros_like(_targets))
_diff_sq = ((_pred_clean - _targ_clean) ** 2)[_valid]
_rmse = float(_diff_sq.mean().sqrt().item()) if _valid.any() else float("nan")
_mae = float((_pred_clean - _targ_clean).abs()[_valid].mean().item()) if _valid.any() else float("nan")
_spread = float(_pred_std[_valid].mean().item()) if _valid.any() else float("nan")

# >>> BS31_DIAG_SHORTCUT — empirical residual-collapse probe
_shortcut = {
    "norm_output": None,
    "norm_mu_HR": None,
    "norm_target": None,
    "norm_output_minus_mu_HR": None,
    "norm_target_minus_mu_HR": None,
    "shortcut_ratio": None,
    "verdict": "N/A",
}
if _all_mu_HR:
    _mu_concat = torch.cat(_all_mu_HR, dim=0).cpu()
    # Align shapes: _pred_mean and _targets are already CPU
    if _mu_concat.shape == _pred_mean.shape:
        _valid_mu = _valid & torch.isfinite(_mu_concat)
        _out = _pred_mean[_valid_mu]
        _mu = _mu_concat[_valid_mu]
        _tg = _targets[_valid_mu]
        _shortcut["norm_output"] = float(_out.abs().mean().item())
        _shortcut["norm_mu_HR"] = float(_mu.abs().mean().item())
        _shortcut["norm_target"] = float(_tg.abs().mean().item())
        _shortcut["norm_output_minus_mu_HR"] = float((_out - _mu).abs().mean().item())
        _shortcut["norm_target_minus_mu_HR"] = float((_tg - _mu).abs().mean().item())
        _ratio = (_shortcut["norm_output_minus_mu_HR"]
                  / max(_shortcut["norm_output"], 1e-12))
        _shortcut["shortcut_ratio"] = float(_ratio)
        if _ratio < 0.10:
            _shortcut["verdict"] = "SHORTCUT_CONFIRMED"
        elif _ratio < 0.30:
            _shortcut["verdict"] = "AMBIGUOUS"
        else:
            _shortcut["verdict"] = "REFINEMENT_OK"
    else:
        print(f"   ⚠️  shortcut diag : shape mismatch μ_HR{tuple(_mu_concat.shape)} vs pred{tuple(_pred_mean.shape)}")

# >>> BS31e_PEARSON_CORR — paper metric (Bomgni et al. 2026 target ~0.896)
def _pearson(a, b, eps=1e-8):
    """Pearson correlation between two flat tensors. Assumes inputs already finite."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    den = torch.sqrt((a_c * a_c).sum() * (b_c * b_c).sum() + eps)
    return float((num / den).item())

# Global Pearson over all valid pixels concatenated.
_corr_global = float("nan")
_corr_per_sample = float("nan")
_corr_per_sample_list = []
try:
    # BS31f : Pearson computed on the FULL prediction (μ_HR + δ̂),
    # not on the raw residual δ̂. See sentinel BS31f_FULL_PREDICTION.
    _p_flat = _pred_full[_valid]
    _t_flat = _targets[_valid]
    if _p_flat.numel() > 1 and _t_flat.numel() > 1:
        _corr_global = _pearson(_p_flat, _t_flat)
    # Per-sample correlation (matches paper convention).
    for _i in range(_pred_full.shape[0]):
        _vi = _valid[_i]
        if _vi.sum() < 2:
            continue
        _pi = _pred_full[_i][_vi]
        _ti = _targets[_i][_vi]
        _c = _pearson(_pi, _ti)
        if _c == _c:  # filter NaN
            _corr_per_sample_list.append(_c)
    if _corr_per_sample_list:
        _corr_per_sample = float(np.mean(_corr_per_sample_list))
except Exception as _e:
    print(f"⚠️  Pearson correlation a échoué : {_e}")

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
print("\n" + "=" * 72)
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
# BS31e_PEARSON_CORR print
print(f"  Pearson Corr (global) : {_corr_global:.4f}")
print(f"  Pearson Corr (per-sample avg) : {_corr_per_sample:.4f} "
      f"(over {len(_corr_per_sample_list)} samples)  ← paper target ~0.896")
for _k, _v in _f1.items():
    print(f"  {_k:<22}: {_v:.4f}")
if _rapsd_d is not None:
    print(f"  RAPSD distance        : {_rapsd_d:.6f}")
print()
# >>> BS31_DIAG_SHORTCUT print block
if _shortcut["shortcut_ratio"] is not None:
    print(f"  🔬 SHORTCUT DIAGNOSTIC (residual collapse probe):")
    print(f"     ‖output‖              = {_shortcut['norm_output']:.5f}")
    print(f"     ‖μ_HR‖                = {_shortcut['norm_mu_HR']:.5f}")
    print(f"     ‖target‖              = {_shortcut['norm_target']:.5f}")
    print(f"     ‖output − μ_HR‖       = {_shortcut['norm_output_minus_mu_HR']:.5f}")
    print(f"     ‖target − μ_HR‖ (true)= {_shortcut['norm_target_minus_mu_HR']:.5f}")
    _r = _shortcut['shortcut_ratio']
    print(f"     shortcut_ratio        = {_r:.4f}  →  verdict: {_shortcut['verdict']}")
    if _shortcut['verdict'] == "SHORTCUT_CONFIRMED":
        print(f"     ⚠️  Diffusion ≈ identité sur μ_HR. UNet sous-capacité probable.")
    elif _shortcut['verdict'] == "AMBIGUOUS":
        print(f"     〰️  Zone grise — examiner aussi F1, RAPSD, ablation μ_HR.")
    else:
        print(f"     ✅ Diffusion produit un raffinement réel sur μ_HR.")
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
    "metrics_scope": _metrics_scope,
    "eval_time_s": _eval_time,
    "rmse": _rmse,
    "mae": _mae,
    "spread_mean": _spread,
    "f1_extremes": _f1,
    "pearson_corr": {
        "global": _corr_global,
        "per_sample_avg": _corr_per_sample,
        "per_sample_n": len(_corr_per_sample_list),
        "per_sample_list": _corr_per_sample_list[:64],  # cap for JSON size
    },
    "rapsd_distance": _rapsd_d,
    "mu_HR_ablation": {
        "delta_signal_ratio_avg": _dag_avg,
        "per_batch": _intervention,
        "verdict": verdict,
    },
    "shortcut_diagnostic": _shortcut,
    "config_eval_num_steps": int(CONFIG.diffusion.get(
        "eval_num_steps", 18 if _causal_concat else 30)),
    "config_cfg_scale": float(CONFIG.diffusion.get("cfg_scale", 0.0)),
    "config_block_out_channels": list(CONFIG.diffusion.unet_kwargs.block_out_channels),
}
_metrics_path = _ckpt_dir / "final_validation_metrics.json"
_metrics_path.write_text(json.dumps(_metrics, indent=2, default=str))
print(f"\n💾 Métriques sauvegardées : {_metrics_path}")
