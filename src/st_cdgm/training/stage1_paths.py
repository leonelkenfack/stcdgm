"""Variant-aware Stage 1 helpers for two-stage ST-CDGM training.

The causal variant predicts ``mu_HR`` through encoder -> RCN ->
GraphToGridDecoder.  The non-causal CorrDiff baseline predicts ``mu_HR``
directly from the LR driver grid through ``RegressionMeanPredictor`` and must
not touch DAGMA / A_dag code paths.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Iterable, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from .two_stage import causal_ablation_check, stage1_compute_loss


def _cfg_get(obj, key: str, default=None):
    """Read ``key`` from dict/OmegaConf/object without tying callers to one type."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(obj, key, default)


def resolve_run_variant(config) -> str:
    """Return the normalized two-stage run variant.

    Supported variants are ``"causal"`` and ``"noncausal"``. Unknown values
    fail closed to ``"causal"`` so existing causal notebooks keep their legacy
    behavior.
    """
    two_stage = _cfg_get(config, "two_stage", {}) or {}
    variant = str(_cfg_get(two_stage, "run_variant", "causal")).lower()
    return variant if variant in {"causal", "noncausal"} else "causal"


def _autocast_context(device: torch.device, use_amp: bool):
    if not use_amp:
        return nullcontext()
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _as_batched_hr(x: Tensor) -> Tensor:
    return x.unsqueeze(0) if x.dim() == 3 else x


def batch_lr_grid_last(
    batch: dict,
    *,
    builder=None,
    lr_shape: Optional[Sequence[int]] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Extract the last LR timestep as ``(B, C, H_lr, W_lr)``.

    Notebook batches may carry either the raw pipeline tensor
    ``(T, C, H, W)`` or the RCN node tensor ``(T, N_lr, C)``.  The latter is
    converted back to a grid using ``builder.lr_shape``.
    """
    lr = batch.get("lr_grid", batch["lr"])
    if device is not None:
        lr = lr.to(device)

    if lr.dim() == 5:  # (B, T, C, H, W)
        return lr[:, -1].contiguous()

    if lr.dim() == 4:
        # Raw sequence (T, C, H, W) or already batched (B, C, H, W).
        shape = tuple(int(v) for v in (lr_shape or getattr(builder, "lr_shape", ()) or ()))
        if shape and tuple(lr.shape[-2:]) == shape:
            if lr.shape[0] > 1 and lr.shape[1] != shape[0]:
                return lr[-1].unsqueeze(0).contiguous()
            return lr.contiguous()
        if shape and tuple(lr.shape[2:]) == shape:
            return lr.contiguous()
        # Fallback: interpret as raw sequence.
        return lr[-1].unsqueeze(0).contiguous()

    if lr.dim() == 3:  # (T, N_lr, C)
        shape = tuple(int(v) for v in (lr_shape or getattr(builder, "lr_shape", ()) or ()))
        if len(shape) != 2:
            raise ValueError(
                "batch_lr_grid_last needs builder.lr_shape or lr_shape to "
                f"reconstruct node LR tensors; got lr shape {tuple(lr.shape)}"
            )
        h, w = shape
        last = lr[-1]
        n_lr, c_lr = last.shape
        if n_lr != h * w:
            raise ValueError(f"Expected N_lr={h*w}, got {n_lr}")
        return last.transpose(0, 1).reshape(1, c_lr, h, w).contiguous()

    raise ValueError(f"Unsupported LR tensor shape {tuple(lr.shape)}")


@torch.no_grad()
def predict_mu_hr(
    batch: dict,
    *,
    variant: str,
    encoder=None,
    rcn_runner=None,
    regression_head,
    builder=None,
    device: torch.device,
    target_shape: Optional[Sequence[int]] = None,
) -> Tensor:
    """Predict ``mu_HR`` for one converted notebook batch.

    ``variant="causal"`` uses encoder + RCN. ``variant="noncausal"`` uses the
    direct LR-grid regression baseline and never touches ``A_dag``.
    """
    variant = str(variant).lower()
    if variant == "noncausal":
        lr_grid = batch_lr_grid_last(batch, builder=builder, device=device)
        mu_hr = regression_head(lr_grid)
    elif variant == "causal":
        if encoder is None or rcn_runner is None:
            raise ValueError("causal predict_mu_hr requires encoder and rcn_runner")
        lr_data = batch["lr"].to(device)
        h_init = encoder.init_state(batch["hetero"]).to(device)
        drivers = [lr_data[t] for t in range(lr_data.shape[0])]
        seq_out = rcn_runner.run(h_init, drivers, reconstruction_sources=None)
        mu_hr = regression_head(seq_out.states[-1])
    else:
        raise ValueError(f"Unknown run variant {variant!r}")

    if target_shape is not None and tuple(mu_hr.shape[-2:]) != tuple(target_shape):
        mu_hr = F.interpolate(mu_hr, size=tuple(target_shape), mode="bilinear", align_corners=False)
    return mu_hr


def train_epoch_stage1_noncausal(
    *,
    regression_head,
    optimizer,
    data_loader: Iterable,
    device: torch.device,
    builder=None,
    lambda_reg: float = 1.0,
    gradient_clipping: Optional[float] = None,
    log_interval: int = 20,
    use_amp: bool = True,
    verbose: bool = True,
) -> dict:
    """Train non-causal Stage 1: LR grid -> ``mu_HR``.

    This is the CorrDiff-vanilla baseline path. It intentionally has no RCN,
    no DAGMA loss, no L1 DAG loss, and no O3 gate.
    """
    regression_head.train()
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    total_loss = 0.0
    total_reg = 0.0
    n_batches = 0
    n_micros = 0

    for batch_idx, batch in enumerate(data_loader):
        batches = batch if isinstance(batch, list) else [batch]
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_reg = 0.0

        for micro in batches:
            target = _as_batched_hr(micro["residual"][-1].to(device))
            with _autocast_context(device, use_amp):
                lr_grid = batch_lr_grid_last(micro, builder=builder, device=device)
                mu_hr = regression_head(lr_grid)
                if mu_hr.shape != target.shape:
                    mu_hr = F.interpolate(mu_hr, size=target.shape[-2:], mode="bilinear", align_corners=False)
                loss_total, components = stage1_compute_loss(
                    mu_HR=mu_hr,
                    target_residual=target,
                    lambda_reg=lambda_reg,
                )
                loss_for_backward = loss_total / max(len(batches), 1)

            if scaler.is_enabled():
                scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()
            step_loss += components["loss_total"]
            step_reg += components["loss_reg"]
            n_micros += 1

        if gradient_clipping is not None and gradient_clipping > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(regression_head.parameters(), gradient_clipping)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += step_loss / max(len(batches), 1)
        total_reg += step_reg / max(len(batches), 1)
        n_batches += 1

        if verbose and (batch_idx == 0 or (batch_idx + 1) % log_interval == 0):
            print(
                f"  S1 noncausal batch {batch_idx + 1} | "
                f"loss={step_loss / max(len(batches), 1):.5f}",
                flush=True,
            )

    return {
        "loss": total_loss / max(1, n_batches),
        "loss_reg": total_reg / max(1, n_batches),
        "n_batches": n_batches,
        "n_micros": n_micros,
    }


@torch.no_grad()
def calibrate_sigma_data_variant(
    *,
    variant: str,
    regression_head,
    data_loader: Iterable,
    iterate_batches_fn,
    builder,
    device: torch.device,
    encoder=None,
    rcn_runner=None,
    max_samples: int = 200,
    verbose: bool = True,
) -> dict:
    """Compute std of ``target_residual - mu_HR`` for causal or non-causal S1."""
    regression_head.eval()
    if encoder is not None:
        encoder.eval()
    if rcn_runner is not None and hasattr(rcn_runner, "cell"):
        rcn_runner.cell.eval()

    n = 0
    mean = 0.0
    m2 = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    samples_seen = 0

    for converted_batches in iterate_batches_fn(data_loader, builder, device):
        for batch in converted_batches:
            if samples_seen >= max_samples:
                break
            target = _as_batched_hr(batch["residual"][-1].to(device))
            mu_hr = predict_mu_hr(
                batch,
                variant=variant,
                encoder=encoder,
                rcn_runner=rcn_runner,
                regression_head=regression_head,
                builder=builder,
                device=device,
                target_shape=target.shape[-2:],
            )
            delta = target - mu_hr
            vals = delta[torch.isfinite(delta)].detach().float().flatten()
            if vals.numel() == 0:
                continue
            stride = max(1, vals.numel() // 1024)
            vals = vals[::stride].cpu()
            for value in vals:
                x = float(value)
                n += 1
                diff = x - mean
                mean += diff / n
                m2 += diff * (x - mean)
                minimum = min(minimum, x)
                maximum = max(maximum, x)
            samples_seen += 1
        if samples_seen >= max_samples:
            break

    if n < 2:
        raise RuntimeError("calibrate_sigma_data_variant: not enough valid pixels")
    std = (m2 / (n - 1)) ** 0.5
    if verbose:
        print(f"  sigma_data[{variant}] over {samples_seen} samples: std={std:.5f}, mean={mean:.5f}")
    return {
        "sigma_data": float(std),
        "mean": float(mean),
        "std": float(std),
        "min": float(minimum),
        "max": float(maximum),
        "n_pixels": int(n),
    }


@torch.no_grad()
def validate_stage1_gate(
    *,
    variant: str,
    encoder=None,
    rcn_runner=None,
    rcn_cell=None,
    regression_head=None,
    data_loader=None,
    iterate_batches_fn=None,
    builder=None,
    device: torch.device,
    n_samples: int = 100,
    threshold: float = 0.05,
    abort_if_fail: bool = True,
) -> dict:
    """Run the causal O3 gate only for causal models."""
    if str(variant).lower() == "noncausal":
        print("  O3 gate skipped: run_variant='noncausal' has no DAG by design.")
        return {
            "passes": True,
            "ratio": 0.0,
            "threshold": float(threshold),
            "n_samples": 0,
            "skipped": True,
            "reason": "noncausal_run_variant",
        }

    report = causal_ablation_check(
        encoder=encoder,
        rcn_runner=rcn_runner,
        rcn_cell=rcn_cell,
        regression_head=regression_head,
        data_loader=data_loader,
        iterate_batches_fn=iterate_batches_fn,
        builder=builder,
        device=device,
        n_samples=n_samples,
        threshold=threshold,
    )
    if not report["passes"] and abort_if_fail:
        raise RuntimeError(
            f"Causal ablation FAILED (ratio={report['ratio']:.4f} < {report['threshold']})."
        )
    return report


@torch.no_grad()
def precompute_stage1_outputs_variant(
    *,
    variant: str,
    regression_head,
    train_dataset,
    iterate_batches_fn,
    device: torch.device,
    encoder=None,
    rcn_runner=None,
    builder=None,
    dag_variants: Sequence[str] = ("normal",),
    existing_cache: Optional[dict] = None,
) -> dict:
    """Cache Stage 1 outputs for Stage 2, respecting causal/non-causal variant."""
    if str(variant).lower() == "causal":
        from .two_stage import precompute_stage1_outputs

        return precompute_stage1_outputs(
            encoder=encoder,
            rcn_runner=rcn_runner,
            regression_head=regression_head,
            train_dataset=train_dataset,
            iterate_batches_fn=iterate_batches_fn,
            device=device,
            dag_variants=dag_variants,
            existing_cache=existing_cache,
        )

    out = {}
    if existing_cache is not None:
        for key in ("mu_HR", "baseline_log", "delta_target", "valid_mask"):
            if key in existing_cache:
                out[key] = existing_cache[key]
    if all(k in out for k in ("mu_HR", "baseline_log", "delta_target", "valid_mask")):
        print("  precompute Stage 1 [noncausal]: cache already complete")
        return out

    regression_head.eval()
    mu_list: list[Tensor] = []
    base_list: list[Tensor] = []
    delta_list: list[Tensor] = []
    mask_list: list[Tensor] = []

    for sample in train_dataset:
        batch = iterate_batches_fn(sample)
        target = _as_batched_hr(batch["residual"][-1].to(device))
        baseline = batch.get("baseline")
        if baseline is not None:
            baseline = _as_batched_hr(baseline[-1].to(device))
        else:
            baseline = torch.zeros_like(target)
        mu_hr = predict_mu_hr(
            batch,
            variant="noncausal",
            regression_head=regression_head,
            builder=builder,
            device=device,
            target_shape=target.shape[-2:],
        )
        valid_mask = torch.isfinite(target)
        delta_target = target - mu_hr

        mu_list.append(torch.nan_to_num(mu_hr, nan=0.0).squeeze(0).cpu())
        base_list.append(torch.nan_to_num(baseline, nan=0.0).squeeze(0).cpu())
        delta_list.append(torch.nan_to_num(delta_target, nan=0.0).squeeze(0).cpu())
        mask_list.append(valid_mask.squeeze(0).cpu())

    out["mu_HR"] = torch.stack(mu_list, dim=0) if mu_list else torch.empty(0)
    out["baseline_log"] = torch.stack(base_list, dim=0) if base_list else torch.empty(0)
    out["delta_target"] = torch.stack(delta_list, dim=0) if delta_list else torch.empty(0)
    out["valid_mask"] = torch.stack(mask_list, dim=0) if mask_list else torch.empty(0)
    return out


__all__ = [
    "resolve_run_variant",
    "batch_lr_grid_last",
    "predict_mu_hr",
    "train_epoch_stage1_noncausal",
    "calibrate_sigma_data_variant",
    "validate_stage1_gate",
    "precompute_stage1_outputs_variant",
]
