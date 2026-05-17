"""
FACL — Fourier Amplitude + Correlation Loss for skillful precipitation diffusion.

Reference
---------
Yang et al. 2024, "Fourier Amplitude and Correlation Loss: Beyond Using L2 Loss
for Skillful Precipitation Nowcasting" — NeurIPS 2024 (arXiv:2410.23159).

Why FACL here
-------------
The plain EDM L2 loss penalises pixel-space residuals but is blind to whether
the model reproduces the right *spatial spectrum* of the target. On
precipitation, RAPSD/Pearson are jointly governed by mid-frequency power, which
L2 routinely undershoots (mode-averaging towards smooth fields). FACL adds two
parameter-free regularisers:

* **Fourier Amplitude Loss (FAL)** — penalises |F(pred)| ≠ |F(target)| (drives
  the radial power spectrum towards the truth, recovering RAPSD distance).
* **Fourier Correlation Loss (FCL)** — penalises phase / sign disagreement
  (lifts Pearson correlation directly, since it acts on the complex inner
  product of the two spectra).

Both terms are differentiable, BF16-compatible (rfft2 is well-supported), and
model-agnostic — the function operates on (pred, target) tensors of the same
shape ``[B, C, H, W]`` and ignores NaN/inf via a finite mask.

The training integration adds
    L_total = L_edm + lambda_facl * facl_loss(D_y, target_clean)
on top of the standard EDM denoiser output ``D_y`` (see
``train_epoch_stage2_cached`` integration in ``two_stage.py``).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _zero_nans(x: Tensor) -> Tensor:
    """Replace NaN/Inf by 0 in a way that is BF16-safe and keeps grad flow."""
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def facl_loss(
    pred: Tensor,
    target: Tensor,
    *,
    alpha_amplitude: float = 0.5,
    beta_correlation: float = 0.5,
    valid_mask: Tensor | None = None,
    eps: float = 1e-8,
) -> Tensor:
    """Compute FACL = α · FAL + β · FCL.

    Parameters
    ----------
    pred, target : Tensor
        ``[B, C, H, W]`` tensors in the same space (log1p mm/day for our use).
    alpha_amplitude : float
        Weight of the Fourier amplitude loss term (FAL).
    beta_correlation : float
        Weight of the Fourier correlation loss term (FCL).
    valid_mask : Tensor, optional
        ``[B, C, H, W]`` float mask, 1.0 where the pixel is valid. NaN pixels
        are zero-imputed before the FFT so they do not propagate; the FFT
        itself is global per channel.
    eps : float
        Numerical floor for the norm denominators of FCL.

    Returns
    -------
    Tensor
        Scalar loss. Already mean-reduced over batch and frequency bins.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"facl_loss: shape mismatch pred={tuple(pred.shape)} target={tuple(target.shape)}"
        )
    if pred.dim() != 4:
        raise ValueError(f"facl_loss expects [B, C, H, W]; got {tuple(pred.shape)}")

    # Zero-impute invalid pixels (NaN/Inf) so the FFT sees a well-defined input.
    p = _zero_nans(pred)
    t = _zero_nans(target)
    if valid_mask is not None:
        p = p * valid_mask
        t = t * valid_mask

    # Real-input 2D FFT — output ``[B, C, H, W//2+1]`` complex.
    # 'ortho' norm makes the loss scale-invariant w.r.t. grid size.
    Fp = torch.fft.rfft2(p, norm="ortho")
    Ft = torch.fft.rfft2(t, norm="ortho")

    # FAL : amplitude MSE.
    amp_p = Fp.abs()
    amp_t = Ft.abs()
    fal = ((amp_p - amp_t) ** 2).mean()

    # FCL : 1 - normalised real inner product (cosine over complex spectra).
    # We use the real part of the conjugate product so phase mismatch is
    # penalised. Per-sample to avoid one sample dominating the batch norm.
    B = Fp.shape[0]
    inner = (Fp.conj() * Ft).real.flatten(start_dim=1).sum(dim=1)   # [B]
    norm_p = amp_p.flatten(start_dim=1).pow(2).sum(dim=1).sqrt()    # [B]
    norm_t = amp_t.flatten(start_dim=1).pow(2).sum(dim=1).sqrt()    # [B]
    cos = inner / (norm_p * norm_t + eps)                            # [B]
    fcl = (1.0 - cos).mean()

    return alpha_amplitude * fal + beta_correlation * fcl


def facl_components(
    pred: Tensor,
    target: Tensor,
    *,
    valid_mask: Tensor | None = None,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor]:
    """Return (FAL, FCL) separately — useful for logging during training."""
    if pred.shape != target.shape or pred.dim() != 4:
        raise ValueError("facl_components expects matching [B,C,H,W] tensors.")
    p = _zero_nans(pred)
    t = _zero_nans(target)
    if valid_mask is not None:
        p = p * valid_mask
        t = t * valid_mask
    Fp = torch.fft.rfft2(p, norm="ortho")
    Ft = torch.fft.rfft2(t, norm="ortho")
    amp_p = Fp.abs()
    amp_t = Ft.abs()
    fal = ((amp_p - amp_t) ** 2).mean()
    inner = (Fp.conj() * Ft).real.flatten(start_dim=1).sum(dim=1)
    norm_p = amp_p.flatten(start_dim=1).pow(2).sum(dim=1).sqrt()
    norm_t = amp_t.flatten(start_dim=1).pow(2).sum(dim=1).sqrt()
    cos = inner / (norm_p * norm_t + eps)
    fcl = (1.0 - cos).mean()
    return fal, fcl


__all__ = ["facl_loss", "facl_components"]
