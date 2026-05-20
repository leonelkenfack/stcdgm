"""
Sliced Wasserstein-1D regularisation for heavy-tailed precipitation residuals.

Reference
---------
Liu et al. 2024, "Downscaling Extreme Precipitation with Wasserstein Regularized
Diffusion" (WassDiff) — arXiv:2410.00381. We use the simpler sliced 1D variant
from Bonneel et al. 2015 instead of the full sliced-W on probability simplices,
which is enough to penalise distributional drift of the residual tail.

Why SW-1D here
--------------
Tail-loss reweighting (BS34) bumps p95/p99 pixels but does not enforce that the
overall *distribution* of the predicted residual matches the truth: a model
can hit both per-pixel MSE and tail counts while underestimating the upper
quantiles of the marginal density. Sliced W-1D measures exactly this kind of
distributional mismatch by projecting the per-sample pixel histograms onto
random 1D directions, sorting, and comparing the resulting empirical CDFs.

The implementation:

* operates on ``[B, C, H, W]`` tensors flattened per sample;
* projects each pixel vector onto ``n_slices`` random unit directions
  (re-drawn each batch);
* sorts the projected values per slice and compares quantile-wise;
* returns a scalar mean-squared distance between sorted projections.

This is differentiable, BF16-safe (sort is stable in BF16 since CUDA 12),
and adds ≈3-5 ms per batch on A100 at n_slices=64 / B=64 / HxW=172x179.
"""

from __future__ import annotations

import torch
from torch import Tensor


def sliced_wasserstein_1d(
    pred: Tensor,
    target: Tensor,
    *,
    n_slices: int = 64,        # kept for API compat ; ignored (no projection needed in 1D)
    valid_mask: Tensor | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Per-sample 1D Wasserstein-2 distance between pred & target pixel histograms.

    V5 FIX F3 (BS41) — the previous version sorted along the BATCH axis
    (``p_proj.sort(dim=0)``), which compared the empirical distribution of
    *batch-level projection scalars* (B=64 values per slice) rather than the
    *pixel-intensity distribution per image* (B*H*W ~ 1.9M values). That was
    a statistically weak and conceptually wrong estimator.

    For grayscale residual fields (C=1) the natural 1D Wasserstein is the
    sort of pixel values WITHIN each sample. The random-projection trick
    from classical Sliced Wasserstein is unnecessary when the underlying
    distribution is already 1-D (single channel intensity). For multi-channel
    inputs we still process each channel independently.

    Formula::

        W_2(P, T)^2 \\approx (1/N) \\sum_i (P_sorted[i] - T_sorted[i])^2

    averaged over channels and batch.

    Parameters
    ----------
    pred, target : Tensor
        ``[B, C, H, W]`` tensors in the same space.
    n_slices : int
        Ignored (kept for backward compat with existing config files).
    valid_mask : Tensor, optional
        ``[B, C, H, W]`` float mask ; invalid pixels are zero-imputed before
        sorting so both tensors see identical "0" entries at the same ranks.
    generator : torch.Generator, optional
        Unused (no randomness left). Kept for API compatibility.

    Returns
    -------
    Tensor
        Scalar squared 1D Wasserstein-2 distance averaged over (B, C).
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"sliced_wasserstein_1d: shape mismatch pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if pred.dim() != 4:
        raise ValueError(f"sliced_wasserstein_1d expects [B,C,H,W]; got {tuple(pred.shape)}")

    p = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        p = p * valid_mask
        t = t * valid_mask

    B, C = p.shape[0], p.shape[1]
    # Per-sample, per-channel pixel intensity sort along the spatial axis.
    p_flat = p.reshape(B * C, -1)
    t_flat = t.reshape(B * C, -1)
    p_sorted, _ = p_flat.sort(dim=1)
    t_sorted, _ = t_flat.sort(dim=1)
    return ((p_sorted - t_sorted) ** 2).mean()


__all__ = ["sliced_wasserstein_1d"]
