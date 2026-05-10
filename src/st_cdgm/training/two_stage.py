"""
Two-Stage training utilities for ST-CDGM (hyperplan v2.0).

This module provides:

* :func:`stage1_compute_loss` — masked MSE between μ_HR (regression head
  output) and the log1p residual ``log1p(HR) - log1p(baseline)``, plus the
  RCN reconstruction loss and the DAGMA acyclicity penalty (with linear
  warmup over the first epochs). Stage 1 trains: encoder + RCN +
  GraphToGridDecoder.

* :func:`freeze_stage1` / :func:`unfreeze_stage1` — flips ``requires_grad``
  on the Stage 1 modules + sets them to ``eval()``. Used between stages.

* :func:`calibrate_sigma_data_two_stage` — runs the frozen Stage 1 over the
  training set and computes the empirical std of the diffusion residual
  ``δ_target = log1p(HR) - log1p(baseline) - μ_HR`` to recalibrate EDM's
  ``sigma_data``. Mandatory before launching Stage 2.

* :func:`causal_ablation_check` — validates Objective O6 on the regression
  head: compares ``μ_HR(A_dag_real)`` to ``μ_HR(A_dag := 0)`` over a sample
  set. Returns the mean absolute delta and a pass/fail flag.

References
----------
- Mardani et al. 2024, "CorrDiff/ResDiff" (arXiv:2309.15214) — the
  sequential two-stage methodology.
- Karras et al. 2022, "EDM" (arXiv:2206.00364) — sigma_data preconditioning.
- oracle.tex §sec:arch:rcn — DAG detachment requirement preserved.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------
# Stage 1 loss
# ---------------------------------------------------------------------


def gamma_dag_warmup(epoch: int, max_value: float, warmup_epochs: int = 5) -> float:
    """Linear warmup of the DAGMA penalty weight (paper §sec:arch:loss).

    Returns 0 at epoch 0 and ``max_value`` at ``warmup_epochs``. Linear
    in between.
    """
    if epoch >= warmup_epochs:
        return float(max_value)
    return float(max_value) * (epoch / max(1, warmup_epochs))


def stage1_compute_loss(
    *,
    mu_HR: Tensor,
    target_residual: Tensor,
    rcn_reconstruction_loss: Optional[Tensor] = None,
    dagma_loss: Optional[Tensor] = None,
    dag_l1_loss: Optional[Tensor] = None,
    dag_prior_loss: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
    lambda_reg: float = 1.0,
    beta_rec: float = 0.05,
    gamma_dag: float = 0.10,
    lambda_l1: float = 0.01,
    lambda_dag_prior: float = 0.0,
) -> Tuple[Tensor, dict]:
    """Stage 1 composite loss.

    L_stage1 = lambda_reg · MSE(mu_HR, target_residual)
              + beta_rec · L_rec
              + gamma_dag · L_dag                           (DAGMA acyclicity)
              + lambda_l1 · L_l1_dag                        (sparsity)
              + lambda_dag_prior · MSE(A_masked, prior)     (anti-collapse)

    The ``dag_prior_loss`` term gives ``A_dag`` a positive supervision
    target (the physically-motivated prior) so it does not collapse to
    zero under the combined L1 + DAGMA centripetal pressure when the
    SCM gradient path is detached. Mirrors the legacy ``train_epoch``
    wiring at ``training_loop.py:1002-1004``.

    Both ``mu_HR`` and ``target_residual`` are expected in log1p space.

    Parameters
    ----------
    mu_HR : Tensor
        Regression head output, ``[B, 1, H, W]``.
    target_residual : Tensor
        ``log1p(HR) - log1p(baseline)``, same shape.
    rcn_reconstruction_loss : Tensor, optional
        Scalar L_rec from the RCN (driver reconstruction).
    dagma_loss : Tensor, optional
        Scalar h_DAGMA(A) acyclicity penalty.
    dag_l1_loss : Tensor, optional
        Scalar ||A||_1 sparsity term.
    valid_mask : Tensor, optional
        Bool mask of finite pixels in ``target_residual`` (handles NaN
        ocean voids identically to the diffusion path).
    lambda_reg, beta_rec, gamma_dag, lambda_l1 : float
        Loss weights. Defaults match paper Tab.2 except ``lambda_reg=1``
        (was implicit in the diffusion path's ``lambda_gen``).

    Returns
    -------
    (loss_total, components) where ``components`` is a dict of scalars
    (Python floats) for logging.
    """
    if valid_mask is None:
        valid_mask = torch.isfinite(target_residual)

    # Replace NaN with 0 to keep autograd safe; mask is the truth source.
    target_clean = torch.where(valid_mask, target_residual, torch.zeros_like(target_residual))

    sq_err = (mu_HR - target_clean) ** 2
    n_valid = valid_mask.float().sum().clamp(min=1.0)
    mse_reg = (sq_err * valid_mask.float()).sum() / n_valid

    components = {"loss_reg": float(mse_reg.detach().item())}

    loss_total = lambda_reg * mse_reg
    if rcn_reconstruction_loss is not None:
        loss_total = loss_total + beta_rec * rcn_reconstruction_loss
        components["loss_rec"] = float(rcn_reconstruction_loss.detach().item())
    if dagma_loss is not None:
        loss_total = loss_total + gamma_dag * dagma_loss
        components["loss_dag"] = float(dagma_loss.detach().item())
    if dag_l1_loss is not None:
        loss_total = loss_total + lambda_l1 * dag_l1_loss
        components["loss_l1"] = float(dag_l1_loss.detach().item())
    if dag_prior_loss is not None and lambda_dag_prior > 0.0:
        loss_total = loss_total + lambda_dag_prior * dag_prior_loss
        components["loss_dag_prior"] = float(dag_prior_loss.detach().item())

    components["loss_total"] = float(loss_total.detach().item())
    return loss_total, components


# ---------------------------------------------------------------------
# Stage 1 → Stage 2 transition
# ---------------------------------------------------------------------


def freeze_stage1(*modules: nn.Module) -> None:
    """Set all parameters of the given modules to ``requires_grad=False``
    and switch them to ``eval()``.

    Used at the start of Stage 2 to make the diffusion training target
    stationary. The regression head, RCN cell, and encoder must all be
    frozen so that ``μ_HR`` and the residual decomposition do not drift
    while the diffusion U-Net learns.

    Idempotent: calling on already-frozen modules is a no-op.
    """
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()


def unfreeze_stage1(*modules: nn.Module) -> None:
    """Inverse of :func:`freeze_stage1`. Re-enables training on the
    Stage 1 modules. Used for Ablation A (let stage 2 fine-tune through)
    or for resumption from a partial run."""
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            p.requires_grad_(True)
        m.train()


@contextmanager
def stage1_inference_mode(*modules: nn.Module):
    """Context manager: temporarily put modules in ``eval()`` and wrap
    in ``torch.no_grad()``. Useful when computing μ_HR for Stage 2 forward
    passes (not for training Stage 1)."""
    saved = []
    for m in modules:
        if m is None:
            continue
        saved.append((m, m.training))
        m.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for m, was_training in saved:
            if was_training:
                m.train()


# ---------------------------------------------------------------------
# σ_data calibration
# ---------------------------------------------------------------------


@torch.no_grad()
def calibrate_sigma_data_two_stage(
    *,
    encoder: nn.Module,
    rcn_runner,  # RCNSequenceRunner — typed loosely to avoid circular import
    regression_head: nn.Module,
    data_loader: Iterable,
    iterate_batches_fn,  # callable(loader, builder, device) -> generator
    builder,
    device: torch.device,
    max_samples: int = 200,
    verbose: bool = True,
) -> dict:
    """Compute the empirical std of the *Stage 2 diffusion target*
    ``δ_target = log1p(HR) - log1p(baseline) - μ_HR`` over up to
    ``max_samples`` samples.

    This must run AFTER Stage 1 has converged (or hit early stop) and
    BEFORE Stage 2 begins. The returned ``sigma_data`` value is used to
    re-instantiate the EDMConfig (Karras 2022 Eq. 7).

    Returns
    -------
    dict with keys:
        sigma_data : float (recommended new value)
        mean       : float (residual mean, should be ~0)
        std        : float (= sigma_data)
        min, max   : float (extremes for sanity check)
        n_pixels   : int (sample count used for std)
    """
    encoder.eval()
    regression_head.eval()
    if hasattr(rcn_runner, "cell"):
        rcn_runner.cell.eval()

    # Welford for numerical stability over many pixels
    n = 0
    mean = 0.0
    M2 = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    samples_seen = 0

    for converted_batches in iterate_batches_fn(data_loader, builder, device):
        for batch in converted_batches:
            if samples_seen >= max_samples:
                break

            lr_data = batch["lr"].to(device)
            target_residual = batch["residual"][-1].to(device)  # log1p space
            if target_residual.dim() == 3:
                target_residual = target_residual.unsqueeze(0)

            H_init = encoder.init_state(batch["hetero"]).to(device)
            drivers = [lr_data[t] for t in range(lr_data.shape[0])]
            seq_out = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
            H_T = seq_out.states[-1]

            mu_HR = regression_head(H_T)
            if mu_HR.shape != target_residual.shape:
                # Resize defensively (CNN output shapes may vary by pixel)
                mu_HR = F.interpolate(
                    mu_HR, size=target_residual.shape[-2:],
                    mode="bilinear", align_corners=False,
                )

            delta_target = target_residual - mu_HR
            valid = torch.isfinite(delta_target)
            flat = delta_target[valid].detach().cpu().to(torch.float64).numpy().ravel()

            if flat.size > 0:
                # Subsample for speed (every 1024-th pixel)
                for x in flat[::1024]:
                    n += 1
                    delta = x - mean
                    mean += delta / n
                    M2 += delta * (x - mean)
                minimum = float(min(minimum, flat.min()))
                maximum = float(max(maximum, flat.max()))

            samples_seen += 1

            if verbose and (samples_seen % 25 == 0):
                _running_std = (M2 / max(n - 1, 1)) ** 0.5 if n > 1 else 0.0
                print(
                    f"  calibrate σ_data: {samples_seen}/{max_samples} samples, "
                    f"running σ ≈ {_running_std:.5f}",
                    flush=True,
                )

        if samples_seen >= max_samples:
            break

    if n < 2:
        raise RuntimeError(
            "calibrate_sigma_data_two_stage: not enough valid pixels — "
            "verify pipeline and Stage 1 outputs."
        )

    var = M2 / (n - 1)
    std = float(var ** 0.5)

    if verbose:
        print(f"\n📐 calibrate σ_data summary")
        print(f"   pixels examined : {n}")
        print(f"   residual mean   : {mean:+.6f}")
        print(f"   residual std    : {std:.6f}")
        print(f"   min / max       : {minimum:+.4f} / {maximum:+.4f}")
        print(f"   → σ_data = {std:.6f} (was Stage 1 init)")

    return {
        "sigma_data": std,
        "mean": float(mean),
        "std": std,
        "min": minimum,
        "max": maximum,
        "n_pixels": int(n),
    }


# ---------------------------------------------------------------------
# Causal ablation check (Objective O6 gate)
# ---------------------------------------------------------------------


@torch.no_grad()
def causal_ablation_check(
    *,
    encoder: nn.Module,
    rcn_runner,
    rcn_cell: nn.Module,  # the underlying RCNCell with .A_dag
    regression_head: nn.Module,
    data_loader: Iterable,
    iterate_batches_fn,
    builder,
    device: torch.device,
    n_samples: int = 100,
    threshold: float = 0.05,
    verbose: bool = True,
) -> dict:
    """Validate (O3): μ_HR(A_dag) ≠ μ_HR(A_dag := 0).

    For each of ``n_samples`` LR drivers, run the full encoder + RCN +
    regression_head twice — once with the learned A_dag and once with
    A_dag temporarily zeroed out. Compute the mean absolute delta in
    log1p space.

    Convention: the ratio ``Δ / signal = mean(|μ_real - μ_zero|) /
    mean(|μ_real|)`` measures how much the causal DAG matters relative
    to the magnitude of the prediction itself. Threshold 0.05 is the
    Gemini-recommended floor for "the DAG is effectively load-bearing".

    Returns
    -------
    dict with keys:
        mean_delta, mean_signal, ratio, passes, n_samples
    """
    encoder.eval()
    regression_head.eval()
    rcn_cell.eval()

    deltas: list[float] = []
    signals: list[float] = []
    samples_seen = 0

    A_dag_param = rcn_cell.A_dag

    for converted_batches in iterate_batches_fn(data_loader, builder, device):
        for batch in converted_batches:
            if samples_seen >= n_samples:
                break

            lr_data = batch["lr"].to(device)
            H_init = encoder.init_state(batch["hetero"]).to(device)
            drivers = [lr_data[t] for t in range(lr_data.shape[0])]

            # Real A_dag run
            seq_out_real = rcn_runner.run(
                H_init, drivers, reconstruction_sources=None
            )
            H_T_real = seq_out_real.states[-1]
            mu_real = regression_head(H_T_real)

            # Zero A_dag run — temporarily replace the parameter, then restore
            saved_A = A_dag_param.data.clone()
            try:
                A_dag_param.data.zero_()
                seq_out_zero = rcn_runner.run(
                    H_init, drivers, reconstruction_sources=None
                )
                H_T_zero = seq_out_zero.states[-1]
                mu_zero = regression_head(H_T_zero)
            finally:
                A_dag_param.data.copy_(saved_A)

            delta = (mu_real - mu_zero).abs().mean().item()
            signal = mu_real.abs().mean().item()
            deltas.append(delta)
            signals.append(signal)
            samples_seen += 1

            if verbose and (samples_seen % 20 == 0):
                _ratio = (
                    (sum(deltas) / len(deltas))
                    / max(sum(signals) / len(signals), 1e-12)
                )
                print(
                    f"  ablation : {samples_seen}/{n_samples}, "
                    f"running Δ/sig ≈ {_ratio:.4f}",
                    flush=True,
                )

        if samples_seen >= n_samples:
            break

    mean_delta = sum(deltas) / max(len(deltas), 1)
    mean_signal = sum(signals) / max(len(signals), 1)
    ratio = mean_delta / max(mean_signal, 1e-12)
    passes = ratio > threshold

    if verbose:
        print(f"\n🧪 Causal ablation check (n={samples_seen}):")
        print(f"   mean |μ_real - μ_zero| = {mean_delta:.6f}")
        print(f"   mean |μ_real|          = {mean_signal:.6f}")
        print(f"   ratio Δ/signal         = {ratio:.4f}  (threshold {threshold})")
        print(f"   verdict                = {'✓ PASS' if passes else '✗ FAIL — DAG decorative'}")

    return {
        "mean_delta": float(mean_delta),
        "mean_signal": float(mean_signal),
        "ratio": float(ratio),
        "threshold": float(threshold),
        "passes": bool(passes),
        "n_samples": int(samples_seen),
    }


# ---------------------------------------------------------------------
# BS32b — pre-cache Stage 1 outputs + thin Stage 2 training loop
# ---------------------------------------------------------------------


@torch.no_grad()
def precompute_stage1_outputs(
    *,
    encoder: nn.Module,
    rcn_runner,
    regression_head: nn.Module,
    train_dataset,
    iterate_batches_fn,
    device: torch.device,
    dag_variants: Sequence[str] = ("normal",),
    existing_cache: Optional[dict] = None,
) -> dict:
    """Iterate ``train_dataset`` once, run Stage 1 forward per sample,
    and stack outputs into a dict of CPU tensors.

    Stage 1 modules must be frozen (``requires_grad=False``, ``eval()``).
    Result keys: ``mu_HR``, ``baseline_log``, ``delta_target``,
    ``valid_mask``. Each tensor is shape ``(N, C, H, W)``. ``valid_mask``
    is ``bool`` (``True`` where the target is finite).

    BS35-CONTRASTIVE — ``dag_variants`` lets the caller request additional
    Stage 1 forwards with the RCN ``A_dag`` perturbed at inference time
    (``set_dag_ablation_mode``). For every non-``"normal"`` variant
    ``v`` listed, the result also contains a ``mu_HR_<v>`` tensor of the
    same shape. ``"normal"`` is always materialized as ``mu_HR`` and
    drives ``delta_target = target - mu_HR``.

    ``existing_cache``: if provided and already contains a variant
    (``mu_HR`` for ``"normal"`` or ``mu_HR_<v>`` otherwise), that
    variant is **not recomputed** — the cached tensors are reused.
    This lets a contrastive run extend a legacy cache (only ``mu_HR``)
    by computing solely the missing ablated variants.

    BS32b — eliminates per-batch Stage 1 forward (encoder + 16-step RCN
    + regression_head) which on the production training loop is run
    32×-64× per logical batch and re-runs the same deterministic forward
    every epoch. Pre-caching collapses 14k×N_epochs forwards to 14k×1.

    Parameters
    ----------
    encoder, rcn_runner, regression_head : torch.nn.Module
        Frozen Stage 1 modules.
    train_dataset : torch.utils.data.IterableDataset (or anything iterable)
        Yields the same dict samples the legacy DataLoader yielded.
    iterate_batches_fn : callable(sample, builder, device) -> dict
        The notebook's ``convert_sample_to_batch`` (we accept it as a
        callable rather than importing — it's defined per-notebook).
    device : torch.device
        Where Stage 1 forwards run.

    Returns
    -------
    dict with keys ``mu_HR``, ``baseline_log``, ``delta_target``,
    ``valid_mask`` — all stacked CPU tensors of shape ``(N, C, H, W)``.
    """
    encoder.eval()
    if hasattr(rcn_runner, "cell"):
        rcn_runner.cell.eval()
    regression_head.eval()

    rcn_cell = rcn_runner.cell if hasattr(rcn_runner, "cell") else None
    can_ablate = rcn_cell is not None and hasattr(rcn_cell, "set_dag_ablation_mode")

    variants: list[str] = []
    seen: set[str] = set()
    for v in dag_variants:
        v = str(v).lower()
        if v not in seen:
            variants.append(v)
            seen.add(v)
    if "normal" not in variants:
        variants.insert(0, "normal")

    def _key_for(v: str) -> str:
        return "mu_HR" if v == "normal" else f"mu_HR_{v}"

    out: dict = {}
    if existing_cache is not None:
        for k in ("baseline_log", "delta_target", "valid_mask"):
            if k in existing_cache:
                out[k] = existing_cache[k]
        for v in variants:
            k = _key_for(v)
            if k in existing_cache:
                out[k] = existing_cache[k]

    needed_variants = [v for v in variants if _key_for(v) not in out]

    has_targets = all(k in out for k in ("baseline_log", "delta_target", "valid_mask"))
    if not needed_variants and has_targets:
        print(
            f"  precompute Stage 1 : tous les variants {variants!r} "
            f"déjà présents dans le cache → skip",
            flush=True,
        )
        return out

    if not needed_variants and not has_targets:
        needed_variants = ["normal"]

    if not can_ablate and any(v != "normal" for v in needed_variants):
        raise RuntimeError(
            "precompute_stage1_outputs: dag_variants demande une ablation "
            "(non-'normal') mais rcn_runner.cell n'expose pas "
            "set_dag_ablation_mode. Patch BS35 manquant ?"
        )

    import time as _t

    delta_target_for_normal: Optional[list[Tensor]] = None

    for variant in needed_variants:
        if can_ablate:
            try:
                rcn_cell.set_dag_ablation_mode(variant, seed=42)
            except Exception as _e:
                raise RuntimeError(
                    f"precompute_stage1_outputs: échec set_dag_ablation_mode({variant!r}): {_e}"
                ) from _e

        mu_list: list[Tensor] = []
        base_list: list[Tensor] = []
        delta_list: list[Tensor] = []
        mask_list: list[Tensor] = []

        t0 = _t.time()
        last_print = t0
        n_seen = 0

        for sample in train_dataset:
            batch = iterate_batches_fn(sample)
            lr_data = batch["lr"].to(device)
            target = batch["residual"][-1].to(device)
            if target.dim() == 3:
                target = target.unsqueeze(0)

            baseline_t = batch.get("baseline")
            if baseline_t is not None:
                baseline_t = baseline_t[-1].to(device)
                if baseline_t.dim() == 3:
                    baseline_t = baseline_t.unsqueeze(0)

            H_init = encoder.init_state(batch["hetero"]).to(device)
            drivers = [lr_data[t] for t in range(lr_data.shape[0])]
            seq = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
            H_T = seq.states[-1]
            mu_HR = regression_head(H_T)
            if mu_HR.shape != target.shape:
                mu_HR = F.interpolate(
                    mu_HR, size=target.shape[-2:],
                    mode="bilinear", align_corners=False,
                )

            baseline_log = baseline_t if baseline_t is not None else torch.zeros_like(target)
            valid_mask = torch.isfinite(target)
            delta_target = target - mu_HR

            mu_HR = torch.nan_to_num(mu_HR, nan=0.0, posinf=0.0, neginf=0.0)
            baseline_log = torch.nan_to_num(baseline_log, nan=0.0, posinf=0.0, neginf=0.0)
            delta_target = torch.nan_to_num(delta_target, nan=0.0, posinf=0.0, neginf=0.0)

            mu_list.append(mu_HR.detach().squeeze(0).cpu())
            base_list.append(baseline_log.detach().squeeze(0).cpu())
            delta_list.append(delta_target.detach().squeeze(0).cpu())
            mask_list.append(valid_mask.detach().squeeze(0).cpu())

            n_seen += 1
            now = _t.time()
            if (now - last_print) >= 5.0:
                print(
                    f"  precompute Stage 1 [{variant}] : {n_seen} samples "
                    f"| {now-t0:.0f}s ({(now-t0)/n_seen:.2f}s/sample)",
                    flush=True,
                )
                last_print = now

        print(
            f"  precompute Stage 1 [{variant}] : {n_seen} samples | "
            f"total {_t.time()-t0:.0f}s "
            f"({(_t.time()-t0)/max(1,n_seen):.2f}s/sample)",
            flush=True,
        )

        out[_key_for(variant)] = (
            torch.stack(mu_list, dim=0) if mu_list else torch.empty(0)
        )
        if variant == "normal":
            out["baseline_log"] = (
                torch.stack(base_list, dim=0) if base_list else torch.empty(0)
            )
            out["delta_target"] = (
                torch.stack(delta_list, dim=0) if delta_list else torch.empty(0)
            )
            out["valid_mask"] = (
                torch.stack(mask_list, dim=0) if mask_list else torch.empty(0)
            )
            delta_target_for_normal = delta_list

    if can_ablate:
        try:
            rcn_cell.set_dag_ablation_mode("normal", seed=42)
        except Exception:
            pass

    return out


def train_epoch_stage2_cached(
    *,
    diffusion_decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    cached_dataloader,
    device: torch.device,
    use_amp: bool = True,
    gradient_clipping: Optional[float] = None,
    log_every: int = 20,
    verbose: bool = True,
    lambda_contrastive_dag: float = 0.0,
    contrastive_dag_margin: float = 0.02,
    contrastive_dag_interval: int = 4,
    ablated_mu_key: str = "mu_HR_zero",
    # BS37 (Sprint B / V3) — EMA des poids Stage 2.
    # Si ``ema_model`` est fourni, on met a jour ses parametres apres
    # chaque optimizer.step() avec :
    #   ema_p <- decay * ema_p + (1 - decay) * live_p
    # Standard CorrDiff/EDM : decay 0.9999 (averaging effectif sur ~10000 steps).
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.9999,
) -> dict:
    """Thin Stage 2 training loop that consumes a pre-cached dataset.

    BS32b — the diffusion forward+backward runs at the *full batch
    size* of the dataloader, not bs=1 in a per-micro loop.

    BS35-CONTRASTIVE — when ``lambda_contrastive_dag > 0`` and the
    batch contains an ablated mu_HR (key ``ablated_mu_key``, default
    ``"mu_HR_zero"``), we add a margin loss every
    ``contrastive_dag_interval`` batches:

        L_c = lambda * max(0, margin - (loss_zero - loss_real))

    where ``loss_real = compute_loss_edm(delta, mu_HR_real, ...)``
    drives the gradient and ``loss_zero = compute_loss_edm(delta,
    mu_HR_dagless, ...)`` is computed under no_grad on the same
    (delta_target, baseline_log, sigma sample) batch. The margin
    forces Stage 2 to perform measurably better when conditioned on
    the causally-informed mu_HR. The diagnostic
    ``dag_sensitivity = loss_zero - loss_real`` is averaged and
    returned in the result dict.

    Expects ``cached_dataloader`` to yield dicts with at least
    ``mu_HR``, ``baseline_log``, ``delta_target``. The contrastive
    branch is silently skipped when the ablated key is missing.
    """
    from st_cdgm.training.training_loop import (
        resolve_train_amp_mode,
        _train_autocast,
    )

    diffusion_decoder.train()
    amp_mode = resolve_train_amp_mode(device, use_amp)
    scaler = torch.amp.GradScaler(enabled=(amp_mode == "cuda_fp16"))

    contrastive_active = bool(lambda_contrastive_dag > 0.0)
    interval = max(1, int(contrastive_dag_interval))

    ema_active = ema_model is not None
    ema_steps = 0
    if ema_active:
        ema_model.eval()
        for _p in ema_model.parameters():
            _p.requires_grad_(False)

    if verbose:
        print(
            f"\n📚 Stage 2 epoch (cached) | amp={amp_mode} "
            f"| contrastive_dag={'on' if contrastive_active else 'off'}"
            + (
                f" (lambda={lambda_contrastive_dag}, margin={contrastive_dag_margin},"
                f" every {interval} batches, key={ablated_mu_key!r})"
                if contrastive_active
                else ""
            )
            + (f" | EMA on (decay={ema_decay})" if ema_active else " | EMA off"),
            flush=True,
        )

    total_loss = 0.0
    total_contrastive = 0.0
    total_dag_sensitivity = 0.0
    n_contrastive = 0
    n_batches = 0
    contrastive_skipped_missing_key = 0

    for batch_idx, batch in enumerate(cached_dataloader):
        mu_HR = batch["mu_HR"].to(device, non_blocking=True)
        baseline_log = batch["baseline_log"].to(device, non_blocking=True)
        delta_target = batch["delta_target"].to(device, non_blocking=True)

        do_contrastive = (
            contrastive_active and (batch_idx % interval == 0)
        )
        mu_HR_ablated: Optional[Tensor] = None
        if do_contrastive:
            if ablated_mu_key in batch:
                mu_HR_ablated = batch[ablated_mu_key].to(device, non_blocking=True)
            else:
                contrastive_skipped_missing_key += 1
                do_contrastive = False

        optimizer.zero_grad(set_to_none=True)

        with _train_autocast(amp_mode):
            loss_real = diffusion_decoder.compute_loss_edm(
                target=delta_target,
                conditioning=None,
                conditioning_spatial=None,
                mu_HR=mu_HR,
                baseline_log=baseline_log,
            )

            loss_contrast_value = torch.tensor(
                0.0, device=device, dtype=loss_real.dtype
            )
            dag_sens_step: Optional[float] = None

            if do_contrastive and mu_HR_ablated is not None:
                with torch.no_grad():
                    loss_zero = diffusion_decoder.compute_loss_edm(
                        target=delta_target,
                        conditioning=None,
                        conditioning_spatial=None,
                        mu_HR=mu_HR_ablated,
                        baseline_log=baseline_log,
                    )
                margin_t = torch.as_tensor(
                    contrastive_dag_margin,
                    device=device,
                    dtype=loss_real.dtype,
                )
                gap = loss_zero.detach() - loss_real
                loss_contrast_value = lambda_contrastive_dag * torch.clamp(
                    margin_t - gap, min=0.0
                )
                dag_sens_step = float((loss_zero - loss_real).detach().item())
                total_dag_sensitivity += dag_sens_step
                total_contrastive += float(loss_contrast_value.detach().item())
                n_contrastive += 1

            loss_total = loss_real + loss_contrast_value

        if amp_mode == "cuda_fp16":
            scaler.scale(loss_total).backward()
        else:
            loss_total.backward()

        if gradient_clipping is not None and gradient_clipping > 0:
            if amp_mode == "cuda_fp16":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                diffusion_decoder.parameters(), gradient_clipping
            )

        if amp_mode == "cuda_fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # BS37 — EMA update (in-place on ema_model parameters AND buffers).
        # Buffers (running stats des GroupNorm si non-affine, etc.) sont
        # copies tels quels du live model, pas moyennes — c'est la convention
        # Karras EDM2 (Karras et al. 2024 Appendix B).
        if ema_active:
            with torch.no_grad():
                for ep, lp in zip(
                    ema_model.parameters(), diffusion_decoder.parameters()
                ):
                    ep.data.mul_(ema_decay).add_(lp.data, alpha=1.0 - ema_decay)
                for eb, lb in zip(
                    ema_model.buffers(), diffusion_decoder.buffers()
                ):
                    eb.data.copy_(lb.data)
            ema_steps += 1

        total_loss += float(loss_real.detach().item())
        n_batches += 1

        if verbose and (batch_idx == 0 or (batch_idx + 1) % log_every == 0):
            extra = ""
            if dag_sens_step is not None:
                extra = (
                    f" | L_contrast={float(loss_contrast_value.detach()):.5f}"
                    f" | dag_sens={dag_sens_step:+.5f}"
                )
            print(
                f"  S2 batch {batch_idx + 1} | loss_diff={float(loss_real.detach()):.5f}{extra}",
                flush=True,
            )

    avg_dag_sens = (
        total_dag_sensitivity / max(1, n_contrastive) if n_contrastive > 0 else 0.0
    )
    avg_contrast = (
        total_contrastive / max(1, n_contrastive) if n_contrastive > 0 else 0.0
    )

    if verbose and contrastive_active:
        if n_contrastive == 0:
            warn = ""
            if contrastive_skipped_missing_key > 0:
                warn = (
                    f" (⚠ {contrastive_skipped_missing_key} batches skipped: "
                    f"clé {ablated_mu_key!r} absente du batch)"
                )
            print(f"  contrastive_dag : aucun pas exécuté{warn}", flush=True)
        else:
            verdict = (
                "OK (DAG conditionne S2)"
                if avg_dag_sens >= contrastive_dag_margin
                else (
                    "marge non atteinte"
                    if avg_dag_sens > 0.0
                    else "DAG ignoré par S2"
                )
            )
            print(
                f"  contrastive_dag : {n_contrastive} steps | "
                f"avg_loss_contrast={avg_contrast:.5f} | "
                f"avg_dag_sensitivity={avg_dag_sens:+.5f} → {verdict}",
                flush=True,
            )

    return {
        "loss_diff": total_loss / max(1, n_batches),
        "n_batches": n_batches,
        "loss_contrastive_dag": avg_contrast,
        "dag_sensitivity": avg_dag_sens,
        "n_contrastive_steps": n_contrastive,
        "ema_active": ema_active,
        "ema_steps": ema_steps,
        "ema_decay": ema_decay if ema_active else None,
    }


__all__ = [
    "gamma_dag_warmup",
    "stage1_compute_loss",
    "freeze_stage1",
    "unfreeze_stage1",
    "stage1_inference_mode",
    "calibrate_sigma_data_two_stage",
    "causal_ablation_check",
    "precompute_stage1_outputs",
    "train_epoch_stage2_cached",
]
