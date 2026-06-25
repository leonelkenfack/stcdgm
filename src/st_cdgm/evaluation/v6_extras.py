"""
V6 MVP — extra evaluation utilities :
  - CRPS multi-échelle (S4.3, Gneiting-Raftery 2007)
  - OOD distribution shift monitoring (S4.2)
  - do(SST+2K) test causal interventionnel (S4.4)

Independent of the training loop. Designed to be called post-training on
checkpointed Stage 2 outputs.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# S4.3 — CRPS multi-échelle (Gneiting-Raftery 2007 JASA)
# --------------------------------------------------------------------------- #
def crps_ensemble(
    ensemble: Tensor, observation: Tensor, *, valid_mask: Optional[Tensor] = None
) -> Tensor:
    """Empirical CRPS for an ensemble vs observation, mean over pixels.

    Hersbach 2000 / Gneiting-Raftery 2007 formulation :
        CRPS(F, y) = E|X − y| − 0.5 · E|X − X'|
    where X, X' ~ F (independent ensemble members).

    Parameters
    ----------
    ensemble : Tensor
        Shape ``[K, B, 1, H, W]`` (K members per sample) or ``[K, B, H, W]``.
    observation : Tensor
        Shape ``[B, 1, H, W]`` or ``[B, H, W]``.
    valid_mask : Optional[Tensor]
        Bool mask of same shape as observation.

    Returns
    -------
    Scalar tensor : mean CRPS over valid pixels.
    """
    if ensemble.dim() == 5:
        ensemble = ensemble.squeeze(2)
    if observation.dim() == 4:
        observation = observation.squeeze(1)
    K, B, H, W = ensemble.shape
    # E|X - y|
    abs_dev = (ensemble - observation.unsqueeze(0)).abs().mean(dim=0)  # [B, H, W]
    # E|X - X'| approximation : sort-based O(K log K) per pixel
    ens_sorted, _ = torch.sort(ensemble, dim=0)
    # E|X - X'| via sorted-ensemble formula (Hersbach 2000) :
    #   E|X - X'| = 2 / K^2 · sum_i (i - (K-1)/2) · ens_sorted_i
    idx_i = torch.arange(K, device=ensemble.device, dtype=ensemble.dtype)
    coef = (idx_i - (K - 1) / 2.0).view(-1, 1, 1, 1)
    abs_pair = (2.0 / (K * K)) * (coef * ens_sorted).sum(dim=0)  # [B, H, W]
    crps_per_pixel = abs_dev - 0.5 * abs_pair

    if valid_mask is not None:
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(1)
        crps_per_pixel = crps_per_pixel[valid_mask]
    return crps_per_pixel.mean()


def crps_multiscale(
    ensemble: Tensor,
    observation: Tensor,
    *,
    scales: tuple[int, ...] = (1, 3, 9),
    valid_mask: Optional[Tensor] = None,
) -> dict:
    """CRPS at multiple spatial scales (avg-pool the ensemble + obs by scale).

    Useful to characterize skill across spatial scales (sub-grid to synoptic).
    """
    out = {}
    for s in scales:
        if s == 1:
            ens_s = ensemble
            obs_s = observation
            mask_s = valid_mask
        else:
            # Avg-pool
            def _pool(x: Tensor) -> Tensor:
                if x.dim() == 5:
                    K, B, C, H, W = x.shape
                    flat = x.reshape(K * B, C, H, W)
                    pooled = torch.nn.functional.avg_pool2d(flat, kernel_size=s, stride=s)
                    return pooled.reshape(K, B, C, *pooled.shape[-2:])
                elif x.dim() == 4:
                    B, C, H, W = x.shape
                    return torch.nn.functional.avg_pool2d(x, kernel_size=s, stride=s)
                elif x.dim() == 3:
                    return torch.nn.functional.avg_pool2d(x.unsqueeze(1), kernel_size=s, stride=s).squeeze(1)
                return x

            ens_s = _pool(ensemble.float())
            obs_s = _pool(observation.float())
            if valid_mask is not None:
                mask_s = _pool(valid_mask.float()) >= 0.5
            else:
                mask_s = None
        crps_s = crps_ensemble(ens_s, obs_s, valid_mask=mask_s)
        out[f"crps_scale_{s}"] = float(crps_s.item())
    return out


# --------------------------------------------------------------------------- #
# S4.2 — OOD distribution shift monitor
# --------------------------------------------------------------------------- #
def wasserstein_1d_distance(p_samples: np.ndarray, q_samples: np.ndarray) -> float:
    """1-D Wasserstein distance between two empirical distributions.

    Sorted-quantile formula, O((n+m) log(n+m)).
    """
    p_sorted = np.sort(p_samples.flatten())
    q_sorted = np.sort(q_samples.flatten())
    # Resample to same length by quantile interpolation
    n = max(len(p_sorted), len(q_sorted))
    qs = np.linspace(0.0, 1.0, n)
    p_at = np.quantile(p_sorted, qs)
    q_at = np.quantile(q_sorted, qs)
    return float(np.abs(p_at - q_at).mean())


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation."""
    rx = np.argsort(np.argsort(x.flatten())).astype(np.float64)
    ry = np.argsort(np.argsort(y.flatten())).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / (np.sqrt((rx ** 2).sum() * (ry ** 2).sum()) + 1e-12))


def ood_distribution_shift_report(
    *,
    train_precip_in_distrib: np.ndarray,   # ACCESS-CM2 train precip flat
    eval_precip_ood: np.ndarray,            # EC-Earth3 (or NorESM2) eval precip flat
    mu_refined_ood: Optional[np.ndarray] = None,
    mu_base_ood: Optional[np.ndarray] = None,
    window_days: int = 30,
) -> dict:
    """Compute OOD distribution shift metrics per V6 plan §3.7 + A7.

    Parameters
    ----------
    train_precip_in_distrib : np.ndarray
        Flat array of training precip values (ACCESS-CM2 train split).
    eval_precip_ood : np.ndarray
        Flat array of eval precip values (EC-Earth3 or NorESM2 test split).
    mu_refined_ood : np.ndarray, optional
        Stage 2 output WITH r_phi (the V6 refined prediction).
    mu_base_ood : np.ndarray, optional
        Stage 1 output WITHOUT r_phi (the V5_causal-equivalent).
    window_days : int
        Window size for time-window Wasserstein (currently a global scalar).

    Returns
    -------
    dict
        Keys :
          - ``wasserstein_train_vs_ood`` : W1 distance between train and OOD
                histograms (alerte si > 2× la valeur train, Pan 2022 GMD).
          - ``spearman_corr_refined_vs_base`` (si mu_refined + mu_base) :
                Spearman corr (< 0.7 = alerte overfitting GCM-specific).
    """
    out = {}
    out["wasserstein_train_vs_ood"] = wasserstein_1d_distance(
        train_precip_in_distrib, eval_precip_ood
    )
    if mu_refined_ood is not None and mu_base_ood is not None:
        out["spearman_corr_refined_vs_base"] = spearman_corr(mu_refined_ood, mu_base_ood)
    # Red flag : Spearman < 0.7 (V6 plan §4.3 Climat A7)
    if "spearman_corr_refined_vs_base" in out:
        out["spearman_red_flag"] = bool(out["spearman_corr_refined_vs_base"] < 0.7)
    return out


# --------------------------------------------------------------------------- #
# S4.4 — Interventional test causal do(SST+2K)
# --------------------------------------------------------------------------- #
def precip_response_to_sst_intervention(
    *,
    sample_fn,
    lr_normal: dict,
    lr_intervened: dict,
    k_samples: int = 16,
    device: torch.device | None = None,
) -> dict:
    """Test interventionnel : do(SST+2K) -> response on precip (V6 plan §4.2).

    Parameters
    ----------
    sample_fn : callable(lr_batch_dict, K) -> ensemble_tensor
        Function that produces an ensemble forecast given an LR batch.
    lr_normal : dict
        LR batch BEFORE intervention.
    lr_intervened : dict
        LR batch AFTER intervention (e.g., SST_TASMAN feature + 2K).
    k_samples : int
        Ensemble size.

    Returns
    -------
    dict with :
      - ``response_pct_per_K`` : observed %/K response
      - ``CC_expected_pct_per_K`` : 7%/K (Clausius-Clapeyron, Pall 2007)
      - ``CC_consistency`` : ``|response - CC|`` within tolerance (3%/K)
    """
    ens_normal = sample_fn(lr_normal, k_samples)
    ens_intervened = sample_fn(lr_intervened, k_samples)
    mean_normal = float(ens_normal.mean().item())
    mean_intervened = float(ens_intervened.mean().item())
    response_pct_per_K = 100.0 * (mean_intervened - mean_normal) / max(1e-6, mean_normal) / 2.0
    CC_pct = 7.0
    tol = 3.0
    return {
        "mean_normal_mm_per_d": mean_normal,
        "mean_intervened_mm_per_d": mean_intervened,
        "response_pct_per_K": response_pct_per_K,
        "CC_expected_pct_per_K": CC_pct,
        "tolerance_pct_per_K": tol,
        "CC_consistent": bool(abs(response_pct_per_K - CC_pct) <= tol),
    }


__all__ = [
    "crps_ensemble",
    "crps_multiscale",
    "wasserstein_1d_distance",
    "spearman_corr",
    "ood_distribution_shift_report",
    "precip_response_to_sst_intervention",
]
