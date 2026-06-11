"""Recompute final_validation_metrics.json apres Phase F (fine-tune).

Phase 6 du notebook lit ce JSON statique. Apres un fine-tune, il faut le
regenerer pour que la comparaison reflete le nouveau modele.

Calcule sur le val_dataset :
- RMSE, MAE (sur mu_HR + delta vs truth, log1p space)
- Pearson global + per_sample_avg
- spread_mean (ensemble std moyen)
- F1-p95, F1-p99
- RAPSD distance
- mu_HR ablation (delta_signal_ratio_avg : impact de A_dag sur mu_HR)

Format du JSON aligne sur ce que Phase 6 attend (cles "rmse", "mae",
"spread_mean", "pearson_corr.global", "f1_extremes.p95/p99",
"rapsd_distance", "mu_HR_ablation.delta_signal_ratio_avg").

Usage typique (depuis le notebook apres Phase F)
-------------------------------------------------
.. code-block:: python

    from scripts.recompute_phase6_metrics import recompute_phase6_metrics

    recompute_phase6_metrics(
        stack=stack_v5, builder=builder, val_dataset=val_dataset,
        DEVICE=DEVICE, predict_with_stack_fn=predict_with_stack,
        convert_sample_to_batch_fn=convert_sample_to_batch,
        out_path=V5_DIR / "final_validation_metrics.json",
        K_samples=12, n_steps=18, n_batches=16,
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
from torch import Tensor


# ---------------------------------------------------------------------
# Metriques unitaires
# ---------------------------------------------------------------------


def _rmse_mae(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple:
    """RMSE et MAE sur pixels valides."""
    p = pred[mask]; t = target[mask]
    if p.size == 0:
        return float("nan"), float("nan")
    diff = p - t
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))
    return rmse, mae


def _pearson(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Pearson sur pixels valides."""
    p = pred[mask]; t = target[mask]
    if p.size < 2:
        return float("nan")
    p_c = p - p.mean(); t_c = t - t.mean()
    denom = np.sqrt((p_c ** 2).sum() * (t_c ** 2).sum()) + 1e-12
    return float((p_c * t_c).sum() / denom)


def _f1_at_quantile(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, q: float) -> float:
    """F1 binarise au quantile q de la truth (sur valid mask)."""
    p = pred[mask]; t = target[mask]
    if p.size == 0:
        return float("nan")
    thr = np.quantile(t, q)
    pred_pos = (p >= thr).astype(np.int8)
    tgt_pos = (t >= thr).astype(np.int8)
    tp = int(((pred_pos == 1) & (tgt_pos == 1)).sum())
    fp = int(((pred_pos == 1) & (tgt_pos == 0)).sum())
    fn = int(((pred_pos == 0) & (tgt_pos == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


def _radial_power_spectrum(field: np.ndarray) -> np.ndarray:
    """RAPSD : moyenne radiale de la PSD 2D."""
    fft = np.fft.fft2(field)
    psd2d = np.abs(np.fft.fftshift(fft)) ** 2
    H, W = field.shape
    cy, cx = H // 2, W // 2
    Y, X = np.indices(field.shape)
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(int)
    R_max = min(cx, cy)
    radial = np.zeros(R_max)
    for r in range(R_max):
        m = R == r
        if m.sum() > 0:
            radial[r] = psd2d[m].mean()
    return radial


def _rapsd_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """L1 distance entre RAPSD du pred et du target (cartes 2D)."""
    sp = _radial_power_spectrum(pred)
    st = _radial_power_spectrum(target)
    n = min(len(sp), len(st))
    return float(np.abs(sp[:n] - st[:n]).sum())


# ---------------------------------------------------------------------
# Recompute principal
# ---------------------------------------------------------------------


def recompute_phase6_metrics(
    *,
    stack: Dict[str, Any],
    builder,
    val_dataset,
    DEVICE: torch.device,
    predict_with_stack_fn: Callable,
    convert_sample_to_batch_fn: Callable,
    out_path: Path,
    K_samples: int = 12,
    n_steps: int = 18,
    n_batches: int = 16,
    epoch: int = 25,
    causal_concat: bool = True,
    cfg_scale: float = 1.5,
    do_mu_hr_ablation: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Calcule les metriques Phase 6 et ecrit out_path.

    Parameters
    ----------
    stack : dict
        Stack avec encoder, rcn_runner, regression_head, skip_block, diffusion.
        Si stack contient une cle "A_dag" Tensor, l'ablation mu_HR la
        zeroise temporairement pour mesurer son impact.
    val_dataset : iterable indexable
        Dataset de validation.
    predict_with_stack_fn : callable
        Signature (stack, batch, K, n_steps) -> Tensor [K, B, 1, H, W].
    convert_sample_to_batch_fn : callable
        Signature (sample, builder, device) -> batch dict.
    out_path : Path
        Chemin du JSON de sortie.

    Returns
    -------
    dict des metriques calculees.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[Recompute Phase 6] K={K_samples}, n_steps={n_steps}, n_batches={n_batches}")
        print(f"  out_path : {out_path}")

    t0 = time.time()
    pearson_per_sample: List[float] = []
    rmse_per_sample: List[float] = []
    mae_per_sample: List[float] = []
    rapsd_per_sample: List[float] = []
    spread_per_sample: List[float] = []
    pred_all: List[np.ndarray] = []
    target_all: List[np.ndarray] = []
    mu_HR_only_all: List[np.ndarray] = []   # pour ablation

    n_avail = min(len(val_dataset), n_batches)

    for i in range(n_avail):
        try:
            sample = val_dataset[i]
        except Exception:
            continue
        batch = convert_sample_to_batch_fn(sample, builder, DEVICE)

        # Inference ensemble
        with torch.no_grad():
            ens = predict_with_stack_fn(stack, batch, K=K_samples, n_steps=n_steps)
            # ens : [K, B=1, 1, H, W] (ou similar)
            while ens.dim() > 4:
                ens = ens.squeeze(1)
            # ens : [K, ..., H, W]
            ens_np = ens.cpu().numpy()
            if ens_np.ndim == 3:
                # [K, H, W]
                ens_arr = ens_np
            elif ens_np.ndim == 4:
                # [K, 1, H, W] ou [K, H, W, 1] — squeeze le canal
                ens_arr = ens_np.squeeze(1) if ens_np.shape[1] == 1 else ens_np.squeeze(-1)
            else:
                ens_arr = ens_np.reshape(K_samples, -1)
            pred_mean = ens_arr.mean(axis=0)   # [H, W]
            spread = ens_arr.std(axis=0).mean()

            target = (batch["baseline"][-1] + batch["residual"][-1]).cpu().numpy()
            target = target.squeeze()
            mask = np.isfinite(target)
            target_clean = np.where(mask, target, 0.0)
            pred_clean = np.where(mask, pred_mean, 0.0)

            # Metriques per-sample
            rmse_i, mae_i = _rmse_mae(pred_clean, target_clean, mask)
            pear_i = _pearson(pred_clean, target_clean, mask)
            rapsd_i = _rapsd_distance(pred_clean, target_clean)

            pearson_per_sample.append(pear_i)
            rmse_per_sample.append(rmse_i)
            mae_per_sample.append(mae_i)
            rapsd_per_sample.append(rapsd_i)
            spread_per_sample.append(float(spread))
            pred_all.append(pred_clean)
            target_all.append(target_clean)

            # mu_HR-only (sans Stage 2) pour ablation
            if do_mu_hr_ablation:
                try:
                    mu_only = _compute_mu_HR_only(stack, batch, DEVICE)
                    mu_HR_only_all.append(mu_only)
                except Exception as e:
                    warnings.warn(f"mu_HR only failed sample {i}: {e}")
                    mu_HR_only_all.append(np.zeros_like(pred_clean))

        if verbose and (i + 1) % 5 == 0:
            print(f"  batch {i+1}/{n_avail} : RMSE={rmse_i:.4f}, MAE={mae_i:.4f}, "
                  f"Pearson={pear_i:.3f}, RAPSD={rapsd_i:.1f}, spread={spread:.4f}")

    # Guard : si tous les batches ont fail, retourne early avec un message
    if not pred_all or not target_all:
        print("[WARN] Tous les batches ont fail. Recompute Phase 6 skip.")
        return {
            "error": "Tous les batches ont fail dans recompute_phase6_metrics",
            "n_batches_attempted": n_avail,
            "n_batches_successful": 0,
        }

    # Metriques globales (agreg)
    pred_global = np.stack(pred_all, axis=0)
    target_global = np.stack(target_all, axis=0)
    mask_global = np.isfinite(target_global)

    rmse_global, mae_global = _rmse_mae(pred_global, target_global, mask_global)
    pearson_global = _pearson(pred_global, target_global, mask_global)
    f1_p95 = _f1_at_quantile(pred_global, target_global, mask_global, 0.95)
    f1_p99 = _f1_at_quantile(pred_global, target_global, mask_global, 0.99)
    rapsd_distance = float(np.mean(rapsd_per_sample))

    # mu_HR ablation (impact de A_dag)
    mu_HR_ablation = None
    if do_mu_hr_ablation and mu_HR_only_all and "A_dag" in stack and stack["A_dag"] is not None:
        try:
            mu_HR_ablation = _compute_mu_HR_ablation(
                stack, val_dataset, builder, DEVICE,
                convert_sample_to_batch_fn, n_batches=min(4, n_avail),
            )
        except Exception as e:
            warnings.warn(f"mu_HR ablation failed: {e}")

    result = {
        "checkpoint": str(out_path.parent / "epoch_last.pth"),
        "epoch": epoch,
        "epochs_total": epoch,
        "best_val_loss": None,
        "causal_concat": causal_concat,
        "n_test_batches": n_avail,
        "k_samples": K_samples,
        "metrics_scope": "full_prediction (mu_HR + delta_hat)",
        "eval_time_s": float(time.time() - t0),
        "rmse": rmse_global,
        "mae": mae_global,
        "spread_mean": float(np.mean(spread_per_sample)),
        "f1_extremes": {"p95": f1_p95, "p99": f1_p99},
        "pearson_corr": {
            "global": pearson_global,
            "per_sample_avg": float(np.mean(pearson_per_sample)),
            "per_sample_n": len(pearson_per_sample),
            "per_sample_list": [float(x) for x in pearson_per_sample],
        },
        "rapsd_distance": rapsd_distance,
        "mu_HR_ablation": mu_HR_ablation or {
            "delta_signal_ratio_avg": float("nan"),
            "per_batch": [],
            "verdict": "SKIPPED",
        },
        "config_eval_num_steps": n_steps,
        "config_cfg_scale": cfg_scale,
        "recomputed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print()
        print("=" * 60)
        print(f"[OK] Phase 6 recomputed in {result['eval_time_s']:.1f}s")
        print(f"  RMSE             : {result['rmse']:.4f}")
        print(f"  MAE              : {result['mae']:.4f}")
        print(f"  Pearson global   : {result['pearson_corr']['global']:.4f}")
        print(f"  Pearson per-samp : {result['pearson_corr']['per_sample_avg']:.4f}")
        print(f"  F1-p95           : {result['f1_extremes']['p95']:.4f}")
        print(f"  F1-p99           : {result['f1_extremes']['p99']:.4f}")
        print(f"  RAPSD distance   : {result['rapsd_distance']:.4f}")
        print(f"  spread_mean      : {result['spread_mean']:.4f}")
        if mu_HR_ablation:
            print(f"  mu_HR ablation   : {result['mu_HR_ablation']['delta_signal_ratio_avg']:.4f}")
        print(f"  Saved to         : {out_path}")
        print("=" * 60)

    return result


def _compute_mu_HR_only(stack, batch, DEVICE) -> np.ndarray:
    """Forward Stage 1 seulement (sans diffusion) pour ablation."""
    import torch.nn.functional as F
    enc = stack["encoder"]; rcn = stack["rcn_runner"]; rh = stack["regression_head"]
    skip = stack.get("skip_block")
    lr = batch["lr"].to(DEVICE)
    H_init = enc.init_state(batch["hetero"]).to(DEVICE)
    drivers = [lr[t] for t in range(lr.shape[0])]
    seq = rcn.run(H_init, drivers, reconstruction_sources=None)
    mu_c = rh(seq.states[-1])
    tshape = batch["residual"][-1].to(DEVICE).shape
    if tshape[-2:] != mu_c.shape[-2:]:
        mu_c = F.interpolate(mu_c, size=tshape[-2:], mode="bilinear", align_corners=False)
    if skip is not None:
        lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
        try:
            mu, _ = skip(lr_last, mu_c)
        except Exception:
            mu = mu_c
    else:
        mu = mu_c
    return mu.cpu().squeeze().numpy()


def _compute_mu_HR_ablation(
    stack, val_dataset, builder, DEVICE, convert_sample_to_batch_fn,
    n_batches: int = 4,
) -> Dict[str, Any]:
    """Ablation A_dag : compare mu_HR(A_dag) vs mu_HR(A_dag=0)."""
    rcn_cell = stack["rcn_runner"].cell
    A_orig = rcn_cell.A_dag.detach().clone()
    per_batch: List[float] = []
    for i in range(n_batches):
        try:
            sample = val_dataset[i]
        except Exception:
            continue
        batch = convert_sample_to_batch_fn(sample, builder, DEVICE)
        with torch.no_grad():
            mu_full = _compute_mu_HR_only(stack, batch, DEVICE)
            rcn_cell.A_dag.data.zero_()
            mu_zero = _compute_mu_HR_only(stack, batch, DEVICE)
            rcn_cell.A_dag.data.copy_(A_orig)
        delta = np.abs(mu_full - mu_zero).mean()
        signal = np.abs(mu_full).mean() + 1e-12
        per_batch.append(float(delta / signal))
    return {
        "delta_signal_ratio_avg": float(np.mean(per_batch)) if per_batch else float("nan"),
        "per_batch": per_batch,
        "verdict": "MU_HR_CONDITIONS" if per_batch and np.mean(per_batch) > 0.5 else "MU_HR_WEAK",
    }


__all__ = ["recompute_phase6_metrics"]
