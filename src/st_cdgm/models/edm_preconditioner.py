"""
EDM (Elucidated Diffusion Models) preconditioning, after Karras et al. 2022.

Reference
---------
Karras, T., Aittala, M., Aila, T., Laine, S. (2022).
"Elucidating the Design Space of Diffusion-Based Generative Models." NeurIPS 2022.
arXiv:2206.00364, Section 5 + Table 1.

Why EDM here
------------
The ST-CDGM paper (oracle.tex) targets log1p-transformed precipitation
residuals on top of a bicubic baseline. The marginal distribution of the
residual is heavy-tailed but the bulk variance is small (sigma_data ~= 0.1
empirically), an order of magnitude smaller than image-domain DDPM defaults
(~0.5). Standard DDPM+epsilon-prediction therefore injects relative noise
that overwhelms the signal at training time, which we observed as
sigma_r ~= 9 (ensemble overdispersed by 9x) on Sprint 4.

EDM solves this by (i) parameterising the model as a denoiser D(x; sigma, c)
with explicit data-variance preconditioning, (ii) sampling sigma from a
log-normal prior at training so the network sees the full noise spectrum
in proportion to the loss it minimises, and (iii) using a deterministic
Heun ODE sampler at inference (Algorithm 2 of the paper) which converges
in 18 steps for image-class problems.

Notation
--------
sigma_data : empirical std of the *target* (the residual we are diffusing).
sigma      : noise level injected into the target during the forward pass.
sigma_min  : smallest noise level used at inference (sets the final
             reconstruction quality, default 0.002 from Karras Table 5).
sigma_max  : largest noise level (default 80.0 from Karras Table 5).
rho        : warps the inference sigma schedule (default 7.0).
P_mean     : mean of log(sigma) at training (default -1.2).
P_std      : std of log(sigma) at training (default 1.2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


# >>> BS34_TAIL_LOSS — tail-aware MSE config (Ravuri 2021, WassDiff 2024).
@dataclass
class TailWeightConfig:
    """Threshold-weighted denoising loss for heavy-tailed targets.

    The weight applied to ``(D - x0)²`` at each pixel is::

        w(x) = 1 + (w95 - 1) · 1[x > τ95] + (w99 - w95) · 1[x > τ99]

    where ``x`` is the FULL HR field reconstructed in log1p(mm/day) space
    (``baseline_log + μ_HR + target``). Defaults match Bénin/West African
    daily CHIRPS climatology; recompute via
    ``scripts/measure_chirps_quantiles.py`` after data swap.

    References
    ----------
    Ravuri et al. 2021, *Nature* (DGMR) — intensity-weighted nowcasting loss.
    Liu et al. 2024, arXiv:2410.00381 (WassDiff) — extreme-precipitation
    diffusion regularisation; this is the simplified weight-only ablation.
    """

    enabled: bool = False                 # OFF by default — opt-in via YAML
    tau95_mmday: float = 15.0             # log1p applied internally
    tau99_mmday: float = 35.0
    weight_p95: float = 5.0
    weight_p99: float = 10.0


@dataclass
class EDMConfig:
    """EDM hyper-parameters. Defaults are Karras 2022 Table 5 (CIFAR-10)
    except sigma_data, which MUST be calibrated to the target dataset.

    For the precipitation residual setting we expect sigma_data ~= 0.1
    after log1p; the user calibrates this with
    ``scripts/measure_residual_std.py`` before training and overrides
    via the YAML config.
    """

    sigma_data: float = 0.1
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    P_mean: float = -1.2
    P_std: float = 1.2
    # Stochastic sampling (set S_churn>0 for stochastic Heun)
    S_churn: float = 0.0
    S_tmin: float = 0.0
    S_tmax: float = float("inf")
    S_noise: float = 1.0
    # >>> BS34_TAIL_LOSS — optional tail-weighting (None = legacy MSE).
    tail_weight: TailWeightConfig | None = None


def compute_preconditioning(
    sigma: Tensor,
    sigma_data: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Karras 2022 Eq. 7. Returns (c_skip, c_out, c_in, c_noise).

    Parameters
    ----------
    sigma : Tensor
        Noise level. Shape ``[B]`` or ``[B, 1, 1, 1]``. Must be positive.
    sigma_data : float
        Empirical std of the target distribution.

    Returns
    -------
    c_skip, c_out, c_in : Tensor
        Same shape as ``sigma`` (broadcast-ready against ``[B, C, H, W]``).
    c_noise : Tensor
        Shape ``[B]`` — feeds the UNet timestep embedding.

    Notes
    -----
    The preconditioner enforces unit variance at the network input
    (``c_in``) and unit variance of the *training target*
    ``(D - x_clean) / c_out``, which decouples the network's learning
    signal from sigma. This is the property that makes EDM converge in
    far fewer steps than DDPM on small datasets.
    """
    if sigma.ndim == 1:
        sigma_b = sigma.view(-1, 1, 1, 1)
    else:
        sigma_b = sigma

    sigma_data_sq = sigma_data ** 2
    denom_sq = sigma_b ** 2 + sigma_data_sq

    c_skip = sigma_data_sq / denom_sq
    c_out = sigma_b * sigma_data / denom_sq.sqrt()
    c_in = 1.0 / denom_sq.sqrt()
    # c_noise is fed to the UNet timestep embedding as a continuous scalar.
    c_noise = (sigma_b.flatten() if sigma_b.ndim > 1 else sigma_b).log() / 4.0

    return c_skip, c_out, c_in, c_noise


def lambda_weight(sigma: Tensor, sigma_data: float) -> Tensor:
    """Karras 2022 Eq. 8 — EDM training-loss weight λ(σ).

    The full training loss is

        L = E_{σ, n, y0} [ λ(σ) · ‖D(y0+n; σ, c) - y0‖² ]

    where λ(σ) = (σ² + σ_data²) / (σ · σ_data)². This counter-balances
    the σ-scaled output of c_out so each noise level contributes roughly
    equally to the gradient.
    """
    if sigma.ndim == 1:
        sigma = sigma.view(-1, 1, 1, 1)
    sigma_data_sq = sigma_data ** 2
    return (sigma ** 2 + sigma_data_sq) / (sigma * sigma_data) ** 2


def sample_training_sigma(
    batch_size: int,
    P_mean: float,
    P_std: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Karras 2022 Eq. 9 — log-normal sigma prior used at training time.

    ``ln(σ) ~ N(P_mean, P_std²)``. Returns a 1-D tensor of size
    ``batch_size``, strictly positive.
    """
    log_sigma = torch.randn(
        batch_size, device=device, dtype=dtype, generator=generator
    ) * P_std + P_mean
    return log_sigma.exp()


def karras_sigma_schedule(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """Karras 2022 Eq. 5 — discretised sigma schedule for the inference
    sampler. Returns ``[num_steps + 1]`` with a final 0 appended (the
    sampler iterates from index 0 to ``num_steps - 1`` and ends at 0).

    ``rho=7`` warps the schedule so most steps land in the noisy half,
    which Karras §5 shows minimises sampling error for fixed step count.
    """
    if num_steps < 2:
        raise ValueError(f"num_steps must be >= 2, got {num_steps}")
    i = torch.arange(num_steps, device=device, dtype=dtype)
    inv_rho = 1.0 / rho
    sigma = (
        sigma_max ** inv_rho
        + (i / (num_steps - 1)) * (sigma_min ** inv_rho - sigma_max ** inv_rho)
    ) ** rho
    # Append a trailing zero so step (num_steps - 1) lands at sigma=0
    return torch.cat([sigma, sigma.new_zeros(1)])


def stochastic_churn(
    sigma_i: Tensor,
    *,
    S_churn: float,
    S_tmin: float,
    S_tmax: float,
    num_steps: int,
) -> Tensor:
    """Compute the per-step churn factor gamma_i for stochastic Heun
    sampling (Karras Algo 2 line 5).

    ``gamma_i = min(S_churn / N, sqrt(2) - 1)`` if ``S_tmin <= sigma_i <= S_tmax``
    else 0. This injects noise in the middle range of the trajectory
    without disturbing low-sigma final steps.

    Returns a scalar tensor matching ``sigma_i`` device/dtype.
    """
    if not (S_churn > 0):
        return sigma_i.new_zeros(())
    in_range = (sigma_i >= S_tmin) & (sigma_i <= S_tmax)
    gamma_max = min(S_churn / num_steps, math.sqrt(2.0) - 1.0)
    return torch.where(in_range, sigma_i.new_full((), gamma_max), sigma_i.new_zeros(()))
