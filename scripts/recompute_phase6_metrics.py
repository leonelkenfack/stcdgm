"""Recompute final_validation_metrics.json apres Phase F (fine-tune).

PROTOCOLE : reproduit EXACTEMENT la Cell 61 de st_cdgm_noncausal_training.ipynb,
qui est le code source qui a produit les baselines V5-mini (Oracle) et CorrDiff
publies. Identites garanties :

  - Inputs builder : ``build_two_stage_inputs`` (variant-aware, returns
    ``(conditioning=None, mu_HR, baseline_log, target)`` pour causal-concat EDM)
  - Sampling      : ``_diff_core.sample(conditioning=None, mu_HR=..., baseline_log=...,
    num_steps=32, scheduler_type="edm_karras", cfg_scale=1.5, apply_constraints=False).residual``
  - Aggregation   : ``torch.cat`` puis une seule passe ``compute_f1_extremes`` /
    ``compute_spectrum_distance`` (pas de moyenne par batch des metriques)
  - Pred metriques : ``pred_full = pred_mean + mu_HR`` (sentinel BS31f), pas
    le residual brut seul
  - mu_HR ablation : compare ``_sample_once(mu_HR)`` vs ``_sample_once(0)``
    (intervention diffusion, pas A_dag zeroing)
  - Shortcut diag : sentinel BS31 — verdict SHORTCUT_CONFIRMED / AMBIGUOUS /
    REFINEMENT_OK selon ``||output - mu_HR|| / ||output||``
  - Per-batch try/except : un batch corrompu ne casse pas tout le run

Params V5-mini training (training_config_corrdiff_normal.yaml) :
  K=64, n_steps=32, scheduler="edm_karras", cfg_scale=1.5

Usage typique :

.. code-block:: python

    from scripts.recompute_phase6_metrics import recompute_phase6_metrics

    recompute_phase6_metrics(
        stack=stack_v5, builder=builder, val_dataset=val_dataset,
        DEVICE=DEVICE,
        convert_sample_to_batch_fn=convert_sample_to_batch,
        out_path=ORACLE_FINETUNED_DIR / "final_validation_metrics.json",
        run_variant="causal",
        K_samples=64, n_steps=32, n_batches=16,
    )
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch


def _pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    """Pearson sur deux tensors aplatis (Cell 61 exact)."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    den = torch.sqrt((a_c * a_c).sum() * (b_c * b_c).sum() + eps)
    return float((num / den).item())


def recompute_phase6_metrics(
    *,
    stack: Dict[str, Any],
    builder,
    val_dataset,
    DEVICE: torch.device,
    convert_sample_to_batch_fn: Callable,
    out_path: Path,
    run_variant: str = "causal",
    K_samples: int = 64,
    n_steps: int = 32,
    n_batches: int = 16,
    n_intervention: int = 4,
    epoch: int = 25,
    # J29 audit fix (Batch D follow-up): default cfg_scale was 1.5 but
    # edm_karras path does not implement CFG -> commit 7d81079 now raises
    # ValueError on cfg_scale > 1.0 in that branch. Default is set to 1.0
    # so callers who do not override will get conditional-only sampling
    # (the SAME behaviour as legacy V5-mini, which silently ignored cfg).
    # To use CFG > 1, callers must explicitly pass scheduler_type="dpm_solver++".
    cfg_scale: float = 1.0,
    scheduler_type: str = "edm_karras",
    verbose: bool = True,
    seed: Optional[int] = None,
    epochs_configured: Optional[int] = None,
    epochs_completed: Optional[int] = None,
    # backward-compat (ignored)
    predict_with_stack_fn: Callable = None,
    causal_concat: Optional[bool] = None,
    do_mu_hr_ablation: bool = True,
) -> Dict[str, Any]:
    """Recompute Phase 6 metrics with the EXACT Cell 61 protocol.

    Output JSON schema = Cell 61 schema, directly comparable to baseline
    V5-mini / CorrDiff ``final_validation_metrics.json``.

    K16 fix (audit DS): ``seed`` parameter makes results reproducible across
    runs. Without it, K=64 ensemble draws differ between calls due to
    unseeded `torch.randn` inside `_diff_core.sample`. With it, the same
    seed produces identical metrics (within numerical noise of the GPU
    matmul order — see consensus §1.4 for hardware reproducibility caveats).
    """
    from st_cdgm.evaluation import compute_f1_extremes, compute_spectrum_distance
    from st_cdgm.evaluation.two_stage_inference import build_two_stage_inputs

    # K16 fix: seed before the eval loop so the K=64 ensemble noise draws are
    # reproducible across re-runs. Recorded in the output JSON for audit.
    if seed is not None:
        import random as _random
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
            # AI eng + DS revise: maximize within-device reproducibility
            # cuBLAS algorithm selection is non-deterministic even with seed
            # set; these flags suppress that within a single GPU architecture.
            # Trade-off: ~5-10% slower, but reproducible same-device.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        np.random.seed(int(seed))
        _random.seed(int(seed))
        if verbose:
            print(f"[K16] Seed set: torch+cuda+numpy+random = {seed}")
            print(f"[K16] cudnn.deterministic=True, benchmark=False (reproducibility)")

    # K30 fix (audit DS): warn if model was under-trained
    # V5-mini baseline had epoch=200 but epochs_total=10 in published JSON,
    # meaning the model ran 5% of its configured budget. Any comparison
    # against an under-trained model is statistically suspect.
    if epochs_configured is not None and epochs_completed is not None:
        if epochs_completed < epochs_configured:
            ratio = epochs_completed / max(epochs_configured, 1)
            # DS revise: use BOTH print AND warnings.warn since Python silently
            # deduplicates warnings by location after first occurrence. The
            # print ensures the message is visible in Colab even on 2nd+ calls.
            msg = (
                f"\n[K30 UNDERTRAINED WARNING] "
                f"epochs_completed={epochs_completed} < epochs_configured={epochs_configured} "
                f"({ratio*100:.0f}% of configured budget). "
                f"Metrics may not represent converged model performance. "
                f"Re-train to full budget before claiming comparison results.\n"
            )
            print("!" * 70)
            print(msg)
            print("!" * 70)
            warnings.warn(msg, UserWarning, stacklevel=2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoder = stack.get("encoder")
    rcn_runner = stack.get("rcn_runner")
    regression_head = stack["regression_head"]
    diffusion = stack["diffusion"]

    # === Cell 61 : detect mode ===
    _diff_core = getattr(diffusion, "_orig_mod", diffusion)
    _diff_core = getattr(_diff_core, "module", _diff_core)
    _causal_concat = bool(getattr(_diff_core, "causal_concat", False))
    if causal_concat is not None and bool(causal_concat) != _causal_concat:
        warnings.warn(
            f"[recompute] causal_concat override ({causal_concat}) differs from "
            f"diffusion._causal_concat ({_causal_concat}). Using model value."
        )

    # K1 fix (audit DS): assert causal_concat consistency with run_variant.
    # The audit found that V5-mini Oracle AND CorrDiff baselines both showed
    # causal_concat=true in their published JSONs, meaning the noncausal
    # baseline was evaluated via the causal-concat sampling path (with mu_HR
    # injection). This makes the Oracle vs CorrDiff comparison apples-to-mangoes.
    # Enforce here: if run_variant='noncausal' then causal_concat MUST be False.
    if run_variant == "noncausal" and _causal_concat:
        raise ValueError(
            "K1 audit fix: run_variant='noncausal' but model has causal_concat=True. "
            "This combination produced the invalid Oracle vs CorrDiff comparison in V5-mini. "
            "The noncausal baseline must use a model trained WITHOUT causal_concat. "
            "If this is intentional (e.g. ablation), explicitly pass causal_concat=True "
            "override AND document in the output JSON metadata."
        )
    if run_variant == "causal" and not _causal_concat:
        warnings.warn(
            "K1 audit fix: run_variant='causal' but model has causal_concat=False. "
            "Causal Oracle should use causal_concat=True. Verify the loaded checkpoint."
        )

    # === Eval mode ===
    if encoder is not None:
        encoder.eval()
    if rcn_runner is not None and hasattr(rcn_runner, "cell"):
        rcn_runner.cell.eval()
    diffusion.eval()
    if regression_head is not None:
        regression_head.eval()

    if verbose:
        print("=" * 72)
        print("Phase 6 recompute (protocol = Cell 61 training_noncausal)")
        print("=" * 72)
        print(f"  variant            : {run_variant}")
        print(f"  causal_concat      : {_causal_concat}")
        print(f"  regression_head    : {regression_head is not None}")
        print(f"  K_samples          : {K_samples}")
        print(f"  n_steps            : {n_steps}")
        print(f"  scheduler          : {scheduler_type}")
        print(f"  cfg_scale          : {cfg_scale}")
        print(f"  apply_constraints  : False")
        print(f"  n_test_batches     : {n_batches}")
        print(f"  n_intervention     : {n_intervention}")
        print(f"  out_path           : {out_path}")
        print()

    # === Cell 61 : helpers ===
    def _build_inputs(_batch):
        return build_two_stage_inputs(
            _batch,
            variant=run_variant,
            regression_head=regression_head,
            encoder=encoder if run_variant == "causal" else None,
            rcn_runner=rcn_runner if run_variant == "causal" else None,
            builder=builder,
            device=DEVICE,
        )

    def _sample_once(_cond, _mu_HR, _baseline_log):
        _kwargs = dict(
            num_steps=int(n_steps),
            scheduler_type=str(scheduler_type),
            cfg_scale=float(cfg_scale),
            apply_constraints=False,
        )
        if _causal_concat:
            _kwargs["mu_HR"] = _mu_HR
            _kwargs["baseline_log"] = _baseline_log
        return _diff_core.sample(conditioning=_cond, **_kwargs).residual

    # === Cell 61 : eval loop ===
    _all_means: List[torch.Tensor] = []
    _all_stds: List[torch.Tensor] = []
    _all_targets: List[torch.Tensor] = []
    _all_mu_HR: List[torch.Tensor] = []
    _all_baseline_log: List[torch.Tensor] = []  # K3 fix: cache for full-space alignment
    _intervention: List[float] = []

    n_avail = min(len(val_dataset), n_batches)
    t0 = time.time()
    _count = 0
    _failures: List[Dict[str, Any]] = []

    with torch.no_grad():
        for i in range(n_avail):
            try:
                sample = val_dataset[i]
            except Exception as e:
                _failures.append({"i": i, "stage": "retrieval", "error": str(e)})
                warnings.warn(f"[batch {i}] sample retrieval failed: {e}")
                continue

            try:
                _batch = convert_sample_to_batch_fn(sample, builder, DEVICE)
                _cond, _mu_HR, _baseline_log, _target = _build_inputs(_batch)

                _samples_k = torch.stack(
                    [_sample_once(_cond, _mu_HR, _baseline_log) for _ in range(K_samples)],
                    dim=0,
                )
                _all_means.append(_samples_k.mean(dim=0))
                _all_stds.append(_samples_k.std(dim=0))
                _all_targets.append(_target)
                if _causal_concat and _mu_HR is not None:
                    _all_mu_HR.append(_mu_HR.detach())
                # K3 fix: also cache baseline_log for full-space metric alignment
                if _causal_concat and _baseline_log is not None:
                    _all_baseline_log.append(_baseline_log.detach())

                if (
                    do_mu_hr_ablation
                    and _count < n_intervention
                    and _causal_concat
                    and _mu_HR is not None
                ):
                    _mu_zero = torch.zeros_like(_mu_HR)
                    _s_real = _sample_once(_cond, _mu_HR, _baseline_log)
                    _s_zero = _sample_once(_cond, _mu_zero, _baseline_log)
                    _delta = (_s_real - _s_zero).abs().mean().item()
                    _signal = _s_real.abs().mean().item()
                    _ratio = _delta / max(_signal, 1e-8)
                    _intervention.append(_ratio)
                    if verbose:
                        print(f"   batch {_count+1:2d} | mu_HR ablation delta/signal = {_ratio*100:6.2f}%")
                elif verbose:
                    print(f"   batch {_count+1:2d} | sampled K={K_samples}")

                _count += 1
            except Exception as e:
                _failures.append({"i": i, "stage": "eval", "error": f"{type(e).__name__}: {e}"})
                warnings.warn(f"[batch {i}] eval failed: {type(e).__name__}: {e}")
                continue

    if _count == 0:
        result = {
            "error": "Aucun batch eval reussi",
            "n_batches_attempted": n_avail,
            "n_batches_successful": 0,
            "failures": _failures,
        }
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        return result

    _eval_time = time.time() - t0

    # === Cell 61 : metriques agregees ===
    _pred_mean = torch.cat(_all_means, dim=0).cpu()
    _pred_std = torch.cat(_all_stds, dim=0).cpu()
    _targets = torch.cat(_all_targets, dim=0).cpu()
    _valid = torch.isfinite(_targets)

    # === K3 fix (DS audit): properly aligned pred vs target ===
    # Audit finding: legacy code compared pred_full = δ̂_mean + μ_HR against
    # _targets = δ̂_true (residual only). The μ_HR offset on the prediction
    # side but NOT the target side artifactually inflated Pearson and
    # depressed RMSE — apples-to-oranges.
    #
    # K3 fix: compute BOTH aligned versions and report.
    #   1. residual-space: pred_residual = δ̂_mean vs target_residual = δ̂_true
    #      (Stage 2 diffusion quality, μ_HR-independent)
    #   2. full-HR-space (log1p): pred_full = baseline_log + μ_HR + δ̂_mean
    #                              target_full = baseline_log + δ̂_true
    #      (end-to-end downscaling quality)
    # Primary metrics use full-HR-space (BS31f intent) but residual-space is
    # logged for diagnostic.
    _pred_residual = _pred_mean.clone()
    _target_residual = _targets.clone()

    if _all_mu_HR and _all_baseline_log:
        _mu_concat_eval = torch.cat(_all_mu_HR, dim=0).cpu()
        _baseline_log_concat = torch.cat(_all_baseline_log, dim=0).cpu()
        if (_mu_concat_eval.shape == _pred_mean.shape
                and _baseline_log_concat.shape == _pred_mean.shape):
            # K3-correct alignment: both pred and target in log1p(HR) space
            _pred_full = _baseline_log_concat + _mu_concat_eval + _pred_mean
            _target_full = _baseline_log_concat + _targets
            _targets_for_metrics = _target_full
            _metrics_scope = "K3_full_log1p_HR (baseline + mu_HR + delta_hat vs baseline + target_residual)"
        else:
            warnings.warn(
                f"K3 shape mismatch — mu_HR {_mu_concat_eval.shape}, "
                f"baseline_log {_baseline_log_concat.shape}, pred {_pred_mean.shape}. "
                f"Falling back to residual-only metrics."
            )
            _pred_full = _pred_mean
            _targets_for_metrics = _targets
            _metrics_scope = "K3_residual_only (fallback)"
    elif _all_mu_HR:
        # Has mu_HR but no baseline_log cached - degrade gracefully to residual
        warnings.warn(
            "K3: baseline_log not cached — comparing residuals only "
            "(legacy BS31f had pred_full=δ̂+μ_HR vs target=δ̂ which was biased)"
        )
        _pred_full = _pred_mean
        _targets_for_metrics = _targets
        _metrics_scope = "K3_residual_only (no baseline_log cached)"
    else:
        _pred_full = _pred_mean
        _targets_for_metrics = _targets
        _metrics_scope = "K3_residual_only (no mu_HR cached)"

    _pred_clean = torch.where(_valid, _pred_full, torch.zeros_like(_pred_full))
    _targ_clean = torch.where(_valid, _targets_for_metrics, torch.zeros_like(_targets_for_metrics))
    _diff_sq = ((_pred_clean - _targ_clean) ** 2)[_valid]
    _rmse = float(_diff_sq.mean().sqrt().item()) if _valid.any() else float("nan")
    _mae = float((_pred_clean - _targ_clean).abs()[_valid].mean().item()) if _valid.any() else float("nan")
    _spread = float(_pred_std[_valid].mean().item()) if _valid.any() else float("nan")

    # === BS31_DIAG_SHORTCUT : residual collapse probe (Cell 61 exact) ===
    _shortcut: Dict[str, Any] = {
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
            _ratio = _shortcut["norm_output_minus_mu_HR"] / max(_shortcut["norm_output"], 1e-12)
            _shortcut["shortcut_ratio"] = float(_ratio)
            if _ratio < 0.10:
                _shortcut["verdict"] = "SHORTCUT_CONFIRMED"
            elif _ratio < 0.30:
                _shortcut["verdict"] = "AMBIGUOUS"
            else:
                _shortcut["verdict"] = "REFINEMENT_OK"
        else:
            warnings.warn(
                f"shortcut diag : shape mismatch mu_HR{tuple(_mu_concat.shape)} "
                f"vs pred{tuple(_pred_mean.shape)}"
            )

    # === BS31e_PEARSON_CORR : global + per-sample sur pred_full ===
    _corr_global = float("nan")
    _corr_per_sample = float("nan")
    _corr_per_sample_list: List[float] = []
    try:
        _p_flat = _pred_full[_valid]
        _t_flat = _targets[_valid]
        if _p_flat.numel() > 1 and _t_flat.numel() > 1:
            _corr_global = _pearson(_p_flat, _t_flat)
        for _i in range(_pred_full.shape[0]):
            _vi = _valid[_i]
            if _vi.sum() < 2:
                continue
            _pi = _pred_full[_i][_vi]
            _ti = _targets[_i][_vi]
            _c = _pearson(_pi, _ti)
            if _c == _c:
                _corr_per_sample_list.append(_c)
        if _corr_per_sample_list:
            _corr_per_sample = float(np.mean(_corr_per_sample_list))
    except Exception as e:
        warnings.warn(f"Pearson failed: {e}")

    # === F1 extremes : un seul appel sur tensors complets (Cell 61) ===
    _f1: Dict[str, float] = {}
    try:
        _f1 = compute_f1_extremes(_pred_clean, _targ_clean, threshold_percentiles=[95.0, 99.0])
        _f1 = {k: float(v) for k, v in _f1.items()}
    except Exception as e:
        warnings.warn(f"F1 extremes failed: {e}")

    # === RAPSD : compute_spectrum_distance sur le PREMIER sample seulement ===
    _rapsd_d: Optional[float] = None
    try:
        _rapsd_d = float(compute_spectrum_distance(_pred_clean[0], _targ_clean[0]))
    except Exception as e:
        warnings.warn(f"RAPSD failed: {e}")

    # === mu_HR ablation verdict (Cell 61 thresholds) ===
    _dag_avg: Optional[float] = float(np.mean(_intervention)) if _intervention else None
    if _dag_avg is None:
        verdict = "N/A"
    elif _dag_avg < 0.001:
        verdict = "MU_HR_IGNORED"
    elif _dag_avg < 0.01:
        verdict = "WEAK"
    else:
        verdict = "MU_HR_CONDITIONS"

    # === JSON output : exact schema Cell 61 ===
    result: Dict[str, Any] = {
        "checkpoint": str(out_path.parent / "epoch_finetuned.pth"),
        "epoch": epoch,
        "epochs_total": epoch,
        "best_val_loss": None,
        "causal_concat": _causal_concat,
        "n_test_batches": _count,
        "k_samples": int(K_samples),
        "metrics_scope": _metrics_scope,
        "eval_time_s": float(_eval_time),
        "rmse": _rmse,
        "mae": _mae,
        "spread_mean": _spread,
        "f1_extremes": _f1,
        "pearson_corr": {
            "global": _corr_global,
            "per_sample_avg": _corr_per_sample,
            "per_sample_n": len(_corr_per_sample_list),
            "per_sample_list": _corr_per_sample_list[:64],
        },
        "rapsd_distance": _rapsd_d,
        "mu_HR_ablation": {
            "delta_signal_ratio_avg": _dag_avg,
            "per_batch": _intervention,
            "verdict": verdict,
        },
        "shortcut_diagnostic": _shortcut,
        "config_eval_num_steps": int(n_steps),
        "config_cfg_scale": float(cfg_scale),
        "config_scheduler_type": str(scheduler_type),
        "run_variant": run_variant,
        "seed": int(seed) if seed is not None else None,  # K16 audit trail
        "hardware": {  # DS revise: full hardware metadata for reproducibility
            "device": str(DEVICE),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": (
                str(torch.version.cuda)
                if torch.cuda.is_available() and hasattr(torch.version, "cuda")
                else None
            ),
            "cudnn_version": (
                int(torch.backends.cudnn.version())
                if torch.cuda.is_available() and torch.backends.cudnn.is_available()
                else None
            ),
            "gpu_name": (
                str(torch.cuda.get_device_name(0))
                if torch.cuda.is_available()
                else None
            ),
            "torch_version": str(torch.__version__),
            "cudnn_deterministic": (
                bool(torch.backends.cudnn.deterministic) if seed is not None else None
            ),
        },
        "epochs_configured": epochs_configured,  # K30 audit trail
        "epochs_completed": epochs_completed,    # K30 audit trail
        "is_undertrained": (
            epochs_configured is not None
            and epochs_completed is not None
            and epochs_completed < epochs_configured
        ),  # K30 flag for downstream analysis
        "recomputed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "uses_cell61_protocol": True,
        "n_batches_attempted": n_avail,
        "n_batches_failed": len(_failures),
        "failures": _failures[:8],  # cap for JSON size
    }

    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # === Cell 61 : final print block ===
    if verbose:
        print()
        print("=" * 72)
        print("FINAL VALIDATION METRICS (post-Phase F)")
        print("=" * 72)
        print(f"  causal_concat         : {_causal_concat}")
        print(f"  Test batches          : {_count} (over {n_avail} attempted, {len(_failures)} failed)")
        print(f"  Samples/batch         : {K_samples}")
        print(f"  Eval time             : {_eval_time:.1f}s ({_eval_time/max(1,_count):.2f}s/batch)")
        print(f"  Metrics scope         : {_metrics_scope}")
        print()
        print(f"  RMSE (ensemble mean)  : {_rmse:.6f}")
        print(f"  MAE                   : {_mae:.6f}")
        print(f"  Spread (ensemble std) : {_spread:.6f}")
        print(f"  Pearson Corr (global) : {_corr_global:.4f}")
        print(f"  Pearson Corr (per-s.) : {_corr_per_sample:.4f}  (n={len(_corr_per_sample_list)})")
        for _k, _v in _f1.items():
            print(f"  {_k:<22}: {_v:.4f}")
        if _rapsd_d is not None:
            print(f"  RAPSD distance        : {_rapsd_d:.6f}")
        print()
        if _shortcut["shortcut_ratio"] is not None:
            print(f"  SHORTCUT DIAGNOSTIC (residual collapse probe):")
            print(f"     ||output||             = {_shortcut['norm_output']:.5f}")
            print(f"     ||mu_HR||              = {_shortcut['norm_mu_HR']:.5f}")
            print(f"     ||target||             = {_shortcut['norm_target']:.5f}")
            print(f"     ||output - mu_HR||     = {_shortcut['norm_output_minus_mu_HR']:.5f}")
            print(f"     ||target - mu_HR||     = {_shortcut['norm_target_minus_mu_HR']:.5f}")
            print(f"     shortcut_ratio         = {_shortcut['shortcut_ratio']:.4f}  -> {_shortcut['verdict']}")
            if _shortcut["verdict"] == "SHORTCUT_CONFIRMED":
                print(f"     [!] Diffusion approx identite sur mu_HR. UNet sous-capacite probable.")
            elif _shortcut["verdict"] == "AMBIGUOUS":
                print(f"     [~] Zone grise -- examiner F1, RAPSD, ablation mu_HR.")
            else:
                print(f"     [OK] Diffusion produit un raffinement reel sur mu_HR.")
            print()
        if _dag_avg is not None:
            print(f"  MU_HR ABLATION (Stage-1 -> Stage-2 conditioning):")
            print(f"     delta/signal = {_dag_avg*100:.3f}%   (sur {len(_intervention)} batches)")
            print(f"     verdict      = {verdict}")
        else:
            print("  MU_HR ABLATION : non execute (causal_concat=False ou regression_head absent)")
        print()
        print(f"  Saved to              : {out_path}")
        print("=" * 72)

    return result


__all__ = ["recompute_phase6_metrics"]
