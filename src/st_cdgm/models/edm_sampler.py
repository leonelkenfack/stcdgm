"""
Heun (2nd-order) deterministic + stochastic sampler for EDM diffusion,
Karras 2022 Algorithm 2.

The sampler calls a user-supplied ``denoiser_fn(x, sigma)`` which is the
preconditioned EDM model D(x; sigma, c). Conditioning is captured by the
caller via closure.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .edm_preconditioner import EDMConfig, karras_sigma_schedule, stochastic_churn


@torch.no_grad()
def heun_sample(
    denoiser_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    shape: tuple,
    cfg: EDMConfig,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    init_noise: Optional[Tensor] = None,
) -> Tensor:
    """Karras 2022 Algorithm 2 — Heun's 2nd-order ODE sampler with
    optional stochastic churn.

    Parameters
    ----------
    denoiser_fn : callable
        ``denoiser_fn(x_noisy, sigma) -> D(x; sigma, c)``. Must be a
        closure that already binds the conditioning. ``sigma`` is a
        scalar tensor on the same device as ``x_noisy``.
    shape : tuple
        Output shape ``(B, C, H, W)``.
    cfg : EDMConfig
        Hyperparameters (sigma_min, sigma_max, rho, S_churn, …).
    num_steps : int
        Number of sigma steps. 18 is Karras default for image generation;
        15 is used by the ST-CDGM paper.
    device, dtype : torch
        Output device/dtype.
    generator : optional torch.Generator
        For reproducible ensemble sampling.
    init_noise : optional Tensor
        If provided, used as ``x_0 ~ N(0, sigma_max²)``. Otherwise drawn
        from ``torch.randn``. Useful for fixed-seed ablations.

    Returns
    -------
    Tensor of shape ``shape`` — the denoised sample at sigma=0.

    Notes
    -----
    Heun's method is a 2nd-order predictor-corrector: at each step we
    take an Euler half-step using d_i = (x - D(x))/sigma, then refine by
    averaging d_i with d_i' computed at the predicted point. The corrector
    is skipped at the final step (sigma_next = 0) because the slope is
    undefined there. With stochastic churn (S_churn > 0) we also inject
    noise scaled to lift sigma_i to sigma_hat = sigma_i * (1 + gamma_i)
    before each step.
    """
    sigmas = karras_sigma_schedule(
        num_steps, cfg.sigma_min, cfg.sigma_max, cfg.rho,
        device=device, dtype=dtype,
    )

    if init_noise is None:
        x = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
    else:
        x = init_noise.to(device=device, dtype=dtype)
    # Scale to sigma_max — x_0 ~ N(0, sigma_max² I).
    x = x * sigmas[0]

    for i in range(num_steps):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Stochastic churn (line 5)
        gamma_i = stochastic_churn(
            sigma_i,
            S_churn=cfg.S_churn,
            S_tmin=cfg.S_tmin,
            S_tmax=cfg.S_tmax,
            num_steps=num_steps,
        )
        sigma_hat = sigma_i * (1.0 + gamma_i)
        if gamma_i > 0:
            eps = torch.randn(*shape, device=device, dtype=dtype, generator=generator) * cfg.S_noise
            x = x + (sigma_hat ** 2 - sigma_i ** 2).sqrt() * eps

        # Euler step (line 7)
        D_x = denoiser_fn(x, sigma_hat)
        d_i = (x - D_x) / sigma_hat
        x_pred = x + (sigma_next - sigma_hat) * d_i

        # Heun correction (line 9), skipped if sigma_next == 0
        if sigma_next > 0:
            D_x_next = denoiser_fn(x_pred, sigma_next)
            d_i_prime = (x_pred - D_x_next) / sigma_next
            x = x + (sigma_next - sigma_hat) * 0.5 * (d_i + d_i_prime)
        else:
            x = x_pred

    return x


# >>> V5 — Track A4 : Restart sampling (Xu et al. NeurIPS 2023, arXiv:2306.14878).
@torch.no_grad()
def restart_sample(
    denoiser_fn: Callable[[Tensor, Tensor], Tensor],
    *,
    shape: tuple,
    cfg: EDMConfig,
    num_steps: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
    init_noise: Optional[Tensor] = None,
    restart_cycles: int = 2,
    restart_inner_steps: int = 3,
    restart_sigma_threshold: float = 0.1,
) -> Tensor:
    """Restart sampling on top of Heun ODE — alternates ODE backward with
    bursts of forward noise to escape solver bias in mid-sigma regimes.

    Algorithm (Xu 2023, simplified for EDM Heun base) :
      1. Run a standard Heun trajectory from ``sigma_max`` down to a chosen
         pivot ``sigma_restart`` (here pivot = step where sigma crosses
         ``restart_sigma_threshold * sigma_max`` from above).
      2. Inject noise of magnitude ``sigma_restart`` (forward SDE).
      3. Re-run Heun from there down to 0.
      4. Repeat (2-3) ``restart_cycles`` times.

    Empirically (Xu 2023 Table 3) Restart adds 1–2% FID gain over standard
    Heun for the same compute, by reducing the discretisation bias
    accumulated in the high-sigma half of the trajectory. On climate
    downscaling we expect ↑ Pearson + ↑ spread (better ensemble calibration)
    at the cost of ``(1 + restart_cycles * restart_inner_steps / num_steps)``
    extra forward passes.

    Compatible with the same ``denoiser_fn`` closure as ``heun_sample``.
    """
    sigmas = karras_sigma_schedule(
        num_steps, cfg.sigma_min, cfg.sigma_max, cfg.rho,
        device=device, dtype=dtype,
    )

    if init_noise is None:
        x = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
    else:
        x = init_noise.to(device=device, dtype=dtype)
    x = x * sigmas[0]

    # Find the index where sigma first drops below the pivot threshold.
    pivot_sigma = float(restart_sigma_threshold) * float(cfg.sigma_max)
    pivot_idx = int((sigmas <= pivot_sigma).nonzero(as_tuple=False)[0].item())
    pivot_idx = max(1, min(pivot_idx, num_steps - 1))

    def _heun_segment(x_in: Tensor, start: int, end: int) -> Tensor:
        """Run Heun from sigmas[start] down to sigmas[end] (end ≤ num_steps)."""
        x_cur = x_in
        for i in range(start, end):
            sigma_i = sigmas[i]
            sigma_next = sigmas[i + 1]
            gamma_i = stochastic_churn(
                sigma_i,
                S_churn=cfg.S_churn,
                S_tmin=cfg.S_tmin,
                S_tmax=cfg.S_tmax,
                num_steps=num_steps,
            )
            sigma_hat = sigma_i * (1.0 + gamma_i)
            if gamma_i > 0:
                eps = torch.randn(
                    *shape, device=device, dtype=dtype, generator=generator
                ) * cfg.S_noise
                x_cur = x_cur + (sigma_hat ** 2 - sigma_i ** 2).sqrt() * eps
            D_x = denoiser_fn(x_cur, sigma_hat)
            d_i = (x_cur - D_x) / sigma_hat
            x_pred = x_cur + (sigma_next - sigma_hat) * d_i
            if sigma_next > 0:
                D_x_next = denoiser_fn(x_pred, sigma_next)
                d_i_prime = (x_pred - D_x_next) / sigma_next
                x_cur = x_cur + (sigma_next - sigma_hat) * 0.5 * (d_i + d_i_prime)
            else:
                x_cur = x_pred
        return x_cur

    # Phase 1 : full backward to the pivot.
    x = _heun_segment(x, 0, pivot_idx)

    # Phase 2..N : restart cycles. Each cycle adds forward noise then re-runs
    # the tail of the Heun schedule. The inner step count overrides the
    # default tail length when restart_inner_steps > 0, to balance compute.
    for c in range(int(restart_cycles)):
        # Inject noise scaled to bring sigma back up to sigmas[pivot_idx].
        eps = torch.randn(*shape, device=device, dtype=dtype, generator=generator)
        sigma_pivot = sigmas[pivot_idx]
        # x is currently at sigma ≈ 0 (or sigmas[num_steps-1]); we lift it
        # back to sigma_pivot. Variance preserving lift: x_new = x + s_pivot*eps.
        # (Xu 2023 §3 — additive Gaussian preserves marginals up to first order.)
        x = x + sigma_pivot * eps
        # Re-run the tail with optionally tighter step density.
        tail_end = min(num_steps, pivot_idx + max(int(restart_inner_steps), 1))
        # If restart_inner_steps < (num_steps - pivot_idx), we still iterate
        # over the existing schedule indices (no resample), which keeps the
        # noise schedule consistent with training.
        x = _heun_segment(x, pivot_idx, tail_end if tail_end <= num_steps else num_steps)

    return x

