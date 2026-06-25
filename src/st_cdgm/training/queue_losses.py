"""
Queue-aware + rank-promoting losses — V6 MVP Stage 2.

Two orthogonal additional losses for ``train_epoch_stage2_cached`` :

* **Pinball loss multi-τ** (Koenker & Bassett 1978 ; Gneiting & Raftery 2007) —
  brise le pillar P1 de la duality MSE-bypass (CST-DAG §2.2). Bayes optimum
  devient quantile conditionnel, pas mean → l'optimiseur cible explicitement
  les queues p99 au lieu d'écraser vers la moyenne.

* **Log-det rank-promoting penalty** sur la covariance résiduelle (CST-DAG §2.4).
  Pénalise rank-deficient covariance → pousse l'optimiseur hors de l'attracteur
  rank-5 (plateau B₁, F2/F6 ECHECS_ET_LECONS.md). Subsample 256-512 pixels
  stratifiés par région NZ pour rester praticable (au lieu de 30k×30k = 8 GB).

References
----------
- Koenker & Bassett 1978 *Econometrica* 46:33 — pinball loss
- Gneiting & Raftery 2007 *JASA* 102:359 — proper scoring rules
- CST-DAG §2.2/§2.4 (ARCHITECTURE_PROPOSEE_RIGOUREUSE.md)
- Math ronde 4 (V6 plan §2.2/§2.3) — λ_pinball ≤ 0.1·λ_MSE,
  log-det subsample 256-512 stratifié, batch ≥ 128 obligatoire
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


# --------------------------------------------------------------------------- #
# Pinball (quantile) loss
# --------------------------------------------------------------------------- #
def pinball_loss(pred: Tensor, target: Tensor, tau: float) -> Tensor:
    """Pinball loss at quantile ``tau`` ∈ (0, 1).

    L(pred, target ; τ) = mean( max(τ · (target − pred), (τ − 1) · (target − pred)) )

    Equivalent to ``mean( (τ − I{target < pred}) · (target − pred) )`` — the
    asymmetric weighting drives Bayes optimum to the τ-quantile of target | pred.

    Parameters
    ----------
    pred, target : Tensor
        Same shape. NaN in target are masked out.
    tau : float
        Quantile, must be in (0, 1).
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    valid = torch.isfinite(target)
    if not valid.any():
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=True)
    diff = target - pred
    # max(τ·diff, (τ-1)·diff) = (τ - I{diff<0}) · diff with positive value
    loss = torch.where(diff >= 0, tau * diff, (tau - 1.0) * diff)
    return loss[valid].mean()


def pinball_multi_tau(
    pred: Tensor,
    target: Tensor,
    taus: Sequence[float] = (0.5, 0.95, 0.99),
) -> Tensor:
    """Sum of pinball losses at multiple τ values (V6 default {0.5, 0.95, 0.99}).

    Returns the SUM (not mean) so the magnitude scales with len(taus). The
    caller applies ``λ_pinball ≤ 0.1·λ_MSE`` per Math ronde 4.
    """
    total = pred.new_tensor(0.0)
    for tau in taus:
        total = total + pinball_loss(pred, target, tau)
    return total


# --------------------------------------------------------------------------- #
# Log-det rank-promoting penalty
# --------------------------------------------------------------------------- #
def make_stratified_subsample_indices(
    region_masks: Tensor,
    k_per_region: int,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample ``k_per_region`` pixel indices per region (flat HR grid indices).

    Parameters
    ----------
    region_masks : Tensor
        Shape ``[n_regions, hr_h, hr_w]``, values in [0, 1]. We treat each
        region's mask > 0.5 as the pool of valid pixels for that region.
    k_per_region : int
        Number of pixels to sample per region.
    generator : torch.Generator, optional
        For reproducibility.

    Returns
    -------
    Tensor
        Flat indices in [0, hr_h·hr_w), shape ``[n_regions · k_per_region]``.
    """
    if region_masks.dim() != 3:
        raise ValueError(f"region_masks must be [n_regions, H, W]; got {tuple(region_masks.shape)}")
    n_regions, H, W = region_masks.shape
    all_indices = []
    for r in range(n_regions):
        flat_mask = region_masks[r].reshape(-1) > 0.5
        valid = torch.nonzero(flat_mask, as_tuple=False).flatten()
        if valid.numel() == 0:
            # Region empty (unlikely if masks well-defined) — pad with zeros
            all_indices.append(torch.zeros(k_per_region, dtype=torch.long, device=region_masks.device))
            continue
        # Sample with replacement if region too small (rare for k=96 per region)
        replace = valid.numel() < k_per_region
        if replace:
            sel = valid[torch.randint(
                0, valid.numel(), (k_per_region,),
                generator=generator, device=region_masks.device
            )]
        else:
            perm = torch.randperm(valid.numel(), generator=generator, device=region_masks.device)
            sel = valid[perm[:k_per_region]]
        all_indices.append(sel)
    return torch.cat(all_indices, dim=0)


def log_det_rank_penalty(
    residual: Tensor,
    subsample_indices: Tensor,
    delta: float = 1e-3,
) -> Tensor:
    """Log-det rank-promoting penalty on subsampled residual covariance.

    L_rank = − log det( δ·I + Cov_batch_subsampled(residual) )

    The negative log-det grows as the smallest eigenvalues of Cov shrink → the
    optimiser is pushed away from rank-deficient covariances (CST-DAG §2.4).

    IMPORTANT : batch_size ≥ 128 required (Math ronde 4). Smaller batches make
    the empirical covariance rank-deficient by sample-count alone (rank ≤ B),
    which would dominate the penalty and saturate it.

    Parameters
    ----------
    residual : Tensor
        Shape ``[B, 1, H, W]`` or ``[B, H, W]`` — typically ``x_HR_pred - μ_HR``.
    subsample_indices : Tensor
        Flat indices in [0, H·W) produced by ``make_stratified_subsample_indices``.
        Shape ``[K]`` where K = n_regions · k_per_region.
    delta : float
        Numerical floor for log-det. Default 1e-3.

    Returns
    -------
    Scalar tensor. Lower = better (more rank-full).
    """
    if residual.dim() == 4:
        if residual.shape[1] != 1:
            raise ValueError(f"residual channels must be 1, got {residual.shape[1]}")
        residual = residual.squeeze(1)
    if residual.dim() != 3:
        raise ValueError(f"residual must be [B, 1, H, W] or [B, H, W]; got {tuple(residual.shape)}")
    B, H, W = residual.shape
    if B < 2:
        # Cov undefined for B<2
        return residual.new_tensor(0.0, requires_grad=True)

    flat = residual.reshape(B, -1)  # [B, H*W]
    sampled = flat[:, subsample_indices]  # [B, K]
    K = sampled.shape[1]

    # Center
    sampled_c = sampled - sampled.mean(dim=0, keepdim=True)
    # Empirical covariance K×K
    cov = (sampled_c.t() @ sampled_c) / max(B - 1, 1)  # [K, K]

    eye = torch.eye(K, device=cov.device, dtype=cov.dtype)
    # Negative log-det (we want to MAXIMISE log det -> minimise negative)
    sign, logabs = torch.linalg.slogdet(delta * eye + cov)
    # sign should be +1 for PSD + δ·I; if not, fallback to 0 (numerical issue)
    if not torch.all(sign > 0):
        return residual.new_tensor(0.0, requires_grad=True)
    return -logabs


__all__ = [
    "pinball_loss",
    "pinball_multi_tau",
    "make_stratified_subsample_indices",
    "log_det_rank_penalty",
]
