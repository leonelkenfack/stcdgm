"""
Dual-convention evaluation metrics — shared source of truth.

Extracted from _eval_3way_dual_convention.ipynb Cell 5 so that BOTH the 3-way
eval notebook AND the V6' notebook import the SAME metric code (avoids the C5
"F1 keys mismatch" class of bug — one definition, one behaviour).

Convention A : per-gridpoint ETCCDI (climatological per-pixel threshold) — the
               NIWA/Rampal/WMO standard, co-primary metric for V6'.
Convention B : pooled global quantile threshold — the cGAN-literature standard.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from st_cdgm.evaluation.evaluation_xai import (
    compute_f1_extremes,
    compute_spectrum_distance,
)

PRECIP_DELTA = 0.01  # pipeline.py precipitation_delta


def to_mm_day(x_log1p: torch.Tensor) -> torch.Tensor:
    """log1p(pr + delta) -> mm/day, clamped [0, 500]."""
    return (torch.expm1(x_log1p) - PRECIP_DELTA).clamp(min=0.0, max=500.0)


def pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    den = torch.sqrt((a_c * a_c).sum() * (b_c * b_c).sum() + eps)
    return float((num / den).item())


def compute_pearson_global_and_per_sample(pred_full: torch.Tensor, targets: torch.Tensor):
    _valid = torch.isfinite(targets) & torch.isfinite(pred_full)
    corr_global = float("nan")
    corr_ps_list = []
    try:
        p_flat = pred_full[_valid]
        t_flat = targets[_valid]
        if p_flat.numel() > 1:
            corr_global = pearson(p_flat, t_flat)
        for i in range(pred_full.shape[0]):
            vi = _valid[i]
            if vi.sum() < 2:
                continue
            c = pearson(pred_full[i][vi], targets[i][vi])
            if c == c:
                corr_ps_list.append(c)
    except Exception as _e:  # noqa: BLE001
        print(f"  WARNING pearson failed: {_e}")
    corr_ps = float(np.mean(corr_ps_list)) if corr_ps_list else float("nan")
    return corr_global, corr_ps, corr_ps_list


def compute_f1_both_conventions(
    pred: torch.Tensor,
    target: torch.Tensor,
    clim_p99: torch.Tensor,
    clim_p95: torch.Tensor,
) -> dict:
    """Both Convention B (pooled) and Convention A (ETCCDI per-pixel).

    pred / target in mm/day (finite, clipped). clim_* are per-pixel thresholds.
    """
    result = {}
    try:
        _b = compute_f1_extremes(pred, target, threshold_percentiles=[95.0, 99.0])
        result["conv_B_F1p99"] = _b.get("p99", float("nan"))
        result["conv_B_F1p95"] = _b.get("p95", float("nan"))
    except Exception as _e:  # noqa: BLE001
        print(f"  WARNING conv_B F1 failed: {_e}")
        result["conv_B_F1p99"] = float("nan")
        result["conv_B_F1p95"] = float("nan")
    try:
        _a99 = compute_f1_extremes(pred, target, threshold_percentiles=[99.0], climatology=clim_p99)
        _a95 = compute_f1_extremes(pred, target, threshold_percentiles=[95.0], climatology=clim_p95)
        result["conv_A_F1p99"] = _a99.get("p99", float("nan"))
        result["conv_A_F1p95"] = _a95.get("p95", float("nan"))
    except Exception as _e:  # noqa: BLE001
        print(f"  WARNING conv_A F1 failed: {_e}")
        result["conv_A_F1p99"] = float("nan")
        result["conv_A_F1p95"] = float("nan")
    return result


def compute_rmse_mae_spread(pred_mean: torch.Tensor, pred_std: Optional[torch.Tensor], targets: torch.Tensor):
    _valid = torch.isfinite(targets) & torch.isfinite(pred_mean)
    if not _valid.any():
        return float("nan"), float("nan"), float("nan")
    diff = pred_mean[_valid] - targets[_valid]
    rmse = float(diff.pow(2).mean().sqrt().item())
    mae = float(diff.abs().mean().item())
    spread = float(pred_std[_valid].mean().item()) if pred_std is not None else float("nan")
    return rmse, mae, spread


def compute_rapsd_batch(pred_mean: torch.Tensor, targets: torch.Tensor, n_samples: int = 4) -> float:
    dists = []
    for i in range(min(n_samples, pred_mean.shape[0])):
        try:
            dists.append(float(compute_spectrum_distance(pred_mean[i], targets[i])))
        except Exception:  # noqa: BLE001
            pass
    return float(np.mean(dists)) if dists else float("nan")


def rx1day_bias_mm(pred_mm: torch.Tensor, target_mm: torch.Tensor) -> float:
    """Annual-max daily precip bias (mean over batch of per-sample max)."""
    try:
        pr_max = pred_mm.reshape(pred_mm.shape[0], -1).max(dim=1).values
        tg_max = target_mm.reshape(target_mm.shape[0], -1).max(dim=1).values
        return float((pr_max - tg_max).mean().item())
    except Exception:  # noqa: BLE001
        return float("nan")


def evaluate_ensemble(
    ensemble_residual_log1p: torch.Tensor,   # [K, B, 1, H, W] diffusion residual
    mu_HR: torch.Tensor,                      # [B, 1, H, W] log1p
    baseline_log: torch.Tensor,              # [B, 1, H, W] log1p
    target_residual_log1p: torch.Tensor,      # [B, 1, H, W] delta target (HR - baseline - mu_HR)
    clim_p99: torch.Tensor,
    clim_p95: torch.Tensor,
) -> dict:
    """Compose HR = baseline + mu_HR + mean(ensemble residual), convert to mm/day,
    and compute the full metric suite (both conventions + RMSE/MAE/Pearson/RAPSD/Rx1day).

    All inputs in log1p space. Returns a flat dict of scalar metrics.
    """
    K = ensemble_residual_log1p.shape[0]
    res_mean = ensemble_residual_log1p.mean(dim=0)   # [B,1,H,W]
    res_std = ensemble_residual_log1p.std(dim=0) if K > 1 else torch.zeros_like(res_mean)

    hr_pred_log = baseline_log + mu_HR + res_mean
    hr_target_log = baseline_log + mu_HR + target_residual_log1p  # = true HR log1p

    pred_mm = to_mm_day(hr_pred_log).squeeze(1)      # [B,H,W]
    target_mm = to_mm_day(hr_target_log).squeeze(1)

    out = {}
    out.update(compute_f1_both_conventions(pred_mm, target_mm, clim_p99, clim_p95))
    cg, cps, _ = compute_pearson_global_and_per_sample(pred_mm, target_mm)
    out["pearson_global"] = cg
    out["pearson_per_sample"] = cps
    rmse, mae, spr = compute_rmse_mae_spread(
        to_mm_day(hr_pred_log).squeeze(1),
        to_mm_day(baseline_log + mu_HR + res_std).squeeze(1) - to_mm_day(baseline_log + mu_HR).squeeze(1),
        target_mm,
    )
    out["rmse"] = rmse
    out["mae"] = mae
    out["rapsd_distance"] = compute_rapsd_batch(pred_mm, target_mm)
    out["rx1day_bias"] = rx1day_bias_mm(pred_mm, target_mm)
    return out


__all__ = [
    "PRECIP_DELTA",
    "to_mm_day",
    "pearson",
    "compute_pearson_global_and_per_sample",
    "compute_f1_both_conventions",
    "compute_rmse_mae_spread",
    "compute_rapsd_batch",
    "rx1day_bias_mm",
    "evaluate_ensemble",
]
