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


# ---------------------------------------------------------------------
# Phase E (post-V5-mini, 2026-06-09) — TailStratifiedSampler
# ---------------------------------------------------------------------


def compute_sample_max_values(
    dataset,
    field_key: str = "residual",
    max_samples: Optional[int] = None,
) -> "list[float]":
    """Pré-calcule le max HR par échantillon du dataset (one-shot).

    Sert d'input au :class:`TailStratifiedSampler` pour stratifier les
    batches selon le quantile de précipitation par jour. À appeler une
    seule fois au setup du training, le résultat se cache facilement.

    Parameters
    ----------
    dataset : Iterable indexable
        Doit supporter ``dataset[i]`` et ``len(dataset)``. Chaque échantillon
        doit être un dict contenant ``field_key``.
    field_key : str
        Clé du tenseur HR cible dans le sample. Défaut ``"residual"`` (la
        cible Stage 1 en log1p space pour le pipeline V5).
    max_samples : int, optional
        Limite le nombre d'échantillons inspectés (utile pour subset eval).

    Returns
    -------
    list[float] — un max par échantillon, dans l'ordre des indices.

    Notes
    -----
    - Coût : O(N) forward dataset, dominé par le data loading (CPU-friendly).
    - Pour un dataset de 365 jours, ~10-20s sur CPU.
    - Le résultat peut être pickle/cache pour les runs ultérieurs.
    """
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    maxes: "list[float]" = []
    for i in range(n):
        sample = dataset[i]
        field = sample[field_key]
        if isinstance(field, list):
            # seq_len list of tensors — prendre le dernier (cible HR)
            field = field[-1]
        field_t = torch.as_tensor(field) if not isinstance(field, Tensor) else field
        # nanmax safe : NaN ignorés (pixels océan)
        valid = torch.isfinite(field_t)
        if valid.any():
            maxes.append(float(field_t[valid].max().item()))
        else:
            maxes.append(0.0)
    return maxes


class TailStratifiedSampler(torch.utils.data.Sampler):
    """Sampler stratifié garantissant ``tail_fraction`` extrêmes par batch.

    Cible la pathologie #1 (tail truncation) observée en V5-mini : sur 365
    jours d'entraînement, ~18 jours contiennent les extrêmes (top 5 %), donc
    la plupart des batches aléatoires n'en contiennent aucun → le modèle
    n'apprend pas dessus → RX1day bias -6.66 mm, F1-p99 = 0.512.

    Stratégie : pré-calculer les jours avec ``max_HR ≥ P95(train_set)``,
    puis pour chaque batch tirer ``int(batch_size * tail_fraction)`` jours
    de ce stratum et le reste du body.

    Parameters
    ----------
    sample_max_values : Sequence[float]
        Max HR par échantillon (cf :func:`compute_sample_max_values`).
    p95_threshold : float
        Seuil pour le stratum "tail". Typiquement ``np.percentile(values, 95)``.
    tail_fraction : float
        Proportion de chaque batch tirée du tail stratum. Défaut 0.30
        (= 30 %, recommandé par §12.6 du rapport multi-agents).
    batch_size : int
        Taille de batch à utiliser avec un ``BatchSampler`` wrapper.
    num_samples : int, optional
        Nombre total d'indices à émettre par epoch. Défaut len(values).
    seed : int, optional
        Graine pour la reproductibilité. Défaut None (aléatoire).

    Notes
    -----
    - À utiliser via ``DataLoader(dataset, sampler=..., batch_size=...)`` ou
      avec un ``BatchSampler`` wrapper si on veut le control fin du batching.
    - Coût compute : 0 % par step (overhead Python negligible).
    - Le sampler tire **avec remise** dans chaque stratum → un même jour
      extrême peut apparaître plusieurs fois dans un epoch. Trade-off
      accepté : sur 18 jours de tail, 30 % * batch_size = 2-3 = inévitable.
    - **NE PAS** utiliser le sampler stratifié pour la calibration
      ``calibrate_sigma_data_variant`` — celle-ci doit voir la distribution
      réelle (loader uniforme).

    References
    ----------
    Lin et al. 2017, *Focal Loss for Dense Object Detection*, ICCV — cadre
    général de re-weighting pour classes déséquilibrées.

    Examples
    --------
    >>> maxes = [0.5, 0.3, 1.2, 0.4, 0.8, 1.5, 0.2, 0.9]   # 8 samples
    >>> thr = 1.0
    >>> sampler = TailStratifiedSampler(maxes, thr, tail_fraction=0.5,
    ...                                  batch_size=4, num_samples=20, seed=42)
    >>> indices = list(sampler)
    >>> len(indices)
    20
    """

    def __init__(
        self,
        sample_max_values: Sequence[float],
        p95_threshold: float,
        tail_fraction: float = 0.30,
        batch_size: int = 8,
        num_samples: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        # NOTE : torch >= 2.0 supprime data_source du parent Sampler.__init__.
        # On n'appelle plus super().__init__() : la classe Sampler de base est
        # essentiellement abstraite, l'instance se construit comme un object.
        if not (0.0 < tail_fraction <= 1.0):
            raise ValueError(
                f"tail_fraction doit être dans (0, 1], reçu {tail_fraction}"
            )
        if batch_size < 2:
            raise ValueError(f"batch_size doit être >= 2, reçu {batch_size}")

        self.sample_max_values = list(sample_max_values)
        self.p95_threshold = float(p95_threshold)
        self.tail_fraction = float(tail_fraction)
        self.batch_size = int(batch_size)

        # Partitionnement en deux strates
        self.tail_idx = [
            i for i, v in enumerate(self.sample_max_values) if v >= self.p95_threshold
        ]
        self.body_idx = [
            i for i, v in enumerate(self.sample_max_values) if v < self.p95_threshold
        ]

        if not self.tail_idx:
            raise ValueError(
                f"Aucun échantillon avec max >= p95_threshold={p95_threshold}. "
                "Vérifier la cohérence des valeurs / seuil."
            )
        if not self.body_idx:
            raise ValueError(
                "Aucun échantillon body (tous extrêmes ?). "
                "Vérifier la cohérence des valeurs / seuil."
            )

        self.n_tail_per_batch = max(1, int(round(self.batch_size * self.tail_fraction)))
        self.n_body_per_batch = self.batch_size - self.n_tail_per_batch
        self.num_samples = num_samples if num_samples is not None else len(self.sample_max_values)

        # Reproductibilité optionnelle
        self.seed = seed
        self._rng = None
        if seed is not None:
            import random as _r
            self._rng = _r.Random(seed)

    def __iter__(self):
        import random
        rng = self._rng if self._rng is not None else random
        n_batches = max(1, self.num_samples // self.batch_size)
        for _ in range(n_batches):
            tail = rng.choices(self.tail_idx, k=self.n_tail_per_batch)
            body = rng.choices(self.body_idx, k=self.n_body_per_batch)
            batch = tail + body
            # Avec ou sans graine, on shuffle in-place
            if self._rng is not None:
                self._rng.shuffle(batch)
            else:
                random.shuffle(batch)
            for idx in batch:
                yield idx

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size

    def __repr__(self) -> str:
        return (
            f"TailStratifiedSampler(n_tail={len(self.tail_idx)}, "
            f"n_body={len(self.body_idx)}, batch_size={self.batch_size}, "
            f"tail_fraction={self.tail_fraction:.2f} -> "
            f"{self.n_tail_per_batch}/{self.batch_size} tail per batch)"
        )


__all__ = [
    "resolve_run_variant",
    "batch_lr_grid_last",
    "predict_mu_hr",
    "train_epoch_stage1_noncausal",
    "calibrate_sigma_data_variant",
    "validate_stage1_gate",
    "precompute_stage1_outputs_variant",
    "compute_sample_max_values",
    "TailStratifiedSampler",
    # Dual-Path Phase 6
    "train_epoch_dualpath_phase1",
    "train_epoch_dualpath_phase2",
    "train_epoch_dualpath_phase3",
    "predict_mu_hr_dualpath",
]


# =============================================================================
# Dual-Path Stage 1 — Phase 6 training functions
# =============================================================================

def train_epoch_dualpath_phase1(
    *,
    dual_path,
    optimizer,
    data_loader: Iterable,
    device: torch.device,
    builder=None,
    gradient_clipping: Optional[float] = 1.0,
    log_interval: int = 30,
    use_amp: bool = True,
    verbose: bool = True,
) -> dict:
    """Phase I: Train Path B alone. Loss = MSE(μ_B, HR_true).

    Path A (encoder/RCN/head) must be frozen by the caller before invoking
    this function. The gate is not used in this phase.
    """
    dual_path.path_b.train()
    dual_path.gate.eval()   # gate unused — frozen is fine too
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(data_loader):
        batches = batch if isinstance(batch, list) else [batch]
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        for micro in batches:
            target = _as_batched_hr(micro["residual"][-1].to(device))
            valid  = torch.isfinite(target)
            if not valid.any():
                continue

            with _autocast_context(device, use_amp):
                lr_grid = batch_lr_grid_last(micro, builder=builder, device=device)
                lr_safe = torch.nan_to_num(lr_grid, nan=0.0)
                mu_B = dual_path.path_b(lr_safe)
                if mu_B.shape != target.shape:
                    mu_B = F.interpolate(mu_B, size=target.shape[-2:],
                                         mode="bilinear", align_corners=False)
                loss = F.mse_loss(mu_B[valid], target[valid])
                (loss / max(len(batches), 1)).backward()
            step_loss += loss.item()

        if gradient_clipping:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dual_path.path_b.parameters(), gradient_clipping)

        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

        step_loss /= max(len(batches), 1)
        total_loss += step_loss
        n_batches += 1

        if verbose and (batch_idx == 0 or (batch_idx + 1) % log_interval == 0):
            print(f"  [6A batch {batch_idx+1}] loss_B={step_loss:.5f}", flush=True)

    return {"loss_B": total_loss / max(1, n_batches), "n_batches": n_batches}


def train_epoch_dualpath_phase2(
    *,
    dual_path,
    optimizer,
    data_loader: Iterable,
    device: torch.device,
    encoder,
    rcn_runner,
    regression_head,
    builder=None,
    lambda_div: float = 0.5,
    gradient_clipping: Optional[float] = 1.0,
    log_interval: int = 30,
    use_amp: bool = True,
    verbose: bool = True,
) -> dict:
    """Phase II: Train gate only. Path A and Path B are frozen by caller.

    Loss = MSE(μ_total, HR) + λ_div · diversity_loss(gate)
    """
    dual_path.path_b.eval()
    dual_path.gate.train()
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    total_loss = 0.0
    total_div  = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(data_loader):
        batches = batch if isinstance(batch, list) else [batch]
        optimizer.zero_grad(set_to_none=True)
        step_loss = step_div = 0.0

        for micro in batches:
            target = _as_batched_hr(micro["residual"][-1].to(device))
            valid  = torch.isfinite(target)
            if not valid.any():
                continue

            with torch.no_grad():
                # Path A forward (frozen)
                lr_data = micro["lr"].to(device)
                h_init  = encoder.init_state(micro["hetero"]).to(device)
                drivers = [lr_data[t] for t in range(lr_data.shape[0])]
                seq_out = rcn_runner.run(h_init, drivers, reconstruction_sources=None)
                H_T     = seq_out.states[-1]
                mu_A    = regression_head(H_T)
                if mu_A.dim() == 3:
                    mu_A = mu_A.unsqueeze(0)
                # Path B forward (frozen)
                lr_grid = batch_lr_grid_last(micro, builder=builder, device=device)
                lr_safe = torch.nan_to_num(lr_grid, nan=0.0)
                mu_B    = dual_path.path_b(lr_safe)

            with _autocast_context(device, use_amp):
                mu_total, gate = dual_path.gate(mu_A, mu_B)
                if mu_total.shape != target.shape:
                    mu_total = F.interpolate(mu_total, size=target.shape[-2:],
                                              mode="bilinear", align_corners=False)
                loss_mse = F.mse_loss(mu_total[valid], target[valid])
                loss_div = dual_path.gate.diversity_loss(gate)
                loss     = loss_mse + lambda_div * loss_div
                (loss / max(len(batches), 1)).backward()
            step_loss += loss_mse.item()
            step_div  += loss_div.item()

        if gradient_clipping:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(dual_path.gate.parameters(), gradient_clipping)

        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

        step_loss /= max(len(batches), 1)
        step_div  /= max(len(batches), 1)
        total_loss += step_loss
        total_div  += step_div
        n_batches  += 1

        if verbose and (batch_idx == 0 or (batch_idx + 1) % log_interval == 0):
            print(f"  [6B batch {batch_idx+1}] loss={step_loss:.5f}  div={step_div:.5f}  "
                  f"gate_mean={gate.mean().item():.3f}", flush=True)

    return {
        "loss_total": total_loss / max(1, n_batches),
        "loss_div":   total_div  / max(1, n_batches),
        "n_batches":  n_batches,
    }


def train_epoch_dualpath_phase3(
    *,
    dual_path,
    optimizer,
    data_loader: Iterable,
    device: torch.device,
    encoder,
    rcn_runner,
    regression_head,
    rcn_cell=None,
    builder=None,
    lambda_causal: float = 1.0,
    lambda_div:    float = 0.5,
    gradient_clipping: Optional[float] = 1.0,
    log_interval:  int  = 30,
    use_amp:       bool = True,
    verbose:       bool = True,
) -> dict:
    """Phase III: Joint fine-tune. A_dag must be frozen by caller.

    Loss = λ_causal·MSE(μ_A, HR) + (1−λ_causal)·MSE(μ_total, HR)
           + λ_div·diversity_loss(gate)
    """
    encoder.train()
    if hasattr(rcn_runner, "cell"):
        rcn_runner.cell.train()
    if rcn_cell is not None:
        rcn_cell.train()
    regression_head.train()
    dual_path.train()
    scaler = torch.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    total_main  = 0.0
    total_caus  = 0.0
    total_div   = 0.0
    n_batches   = 0

    for batch_idx, batch in enumerate(data_loader):
        batches = batch if isinstance(batch, list) else [batch]
        optimizer.zero_grad(set_to_none=True)
        s_main = s_caus = s_div = 0.0

        for micro in batches:
            target = _as_batched_hr(micro["residual"][-1].to(device))
            valid  = torch.isfinite(target)
            if not valid.any():
                continue

            with _autocast_context(device, use_amp):
                # Path A forward (backbone unfrozen, A_dag frozen by caller)
                lr_data = micro["lr"].to(device)
                h_init  = encoder.init_state(micro["hetero"]).to(device)
                drivers = [lr_data[t] for t in range(lr_data.shape[0])]
                seq_out = rcn_runner.run(h_init, drivers, reconstruction_sources=None)
                H_T     = seq_out.states[-1]
                mu_A    = regression_head(H_T)
                if mu_A.dim() == 3:
                    mu_A = mu_A.unsqueeze(0)
                # Path B + Gate
                lr_grid  = batch_lr_grid_last(micro, builder=builder, device=device)
                lr_safe  = torch.nan_to_num(lr_grid, nan=0.0)
                mu_total, mu_B, gate = dual_path(lr_safe, mu_A)
                if mu_total.shape != target.shape:
                    mu_total = F.interpolate(mu_total, size=target.shape[-2:],
                                              mode="bilinear", align_corners=False)
                    mu_A = F.interpolate(mu_A, size=target.shape[-2:],
                                          mode="bilinear", align_corners=False)
                loss_main = F.mse_loss(mu_total[valid], target[valid])
                loss_caus = F.mse_loss(mu_A[valid],     target[valid])
                loss_div  = dual_path.gate.diversity_loss(gate)
                loss = ((1.0 - lambda_causal) * loss_main
                        + lambda_causal        * loss_caus
                        + lambda_div           * loss_div)
                (loss / max(len(batches), 1)).backward()
            s_main += loss_main.item()
            s_caus += loss_caus.item()
            s_div  += loss_div.item()

        if gradient_clipping:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            # Clip all trainable params together
            all_params = (
                list(encoder.parameters())
                + list(rcn_runner.parameters() if hasattr(rcn_runner, "parameters")
                       else (rcn_cell.parameters() if rcn_cell else []))
                + list(regression_head.parameters())
                + list(dual_path.parameters())
            )
            torch.nn.utils.clip_grad_norm_(all_params, gradient_clipping)

        if scaler.is_enabled():
            scaler.step(optimizer); scaler.update()
        else:
            optimizer.step()

        s_main /= max(len(batches), 1)
        s_caus /= max(len(batches), 1)
        s_div  /= max(len(batches), 1)
        total_main += s_main
        total_caus += s_caus
        total_div  += s_div
        n_batches  += 1

        if verbose and (batch_idx == 0 or (batch_idx + 1) % log_interval == 0):
            print(
                f"  [6C batch {batch_idx+1}] main={s_main:.5f}  causal={s_caus:.5f}  "
                f"div={s_div:.5f}  λ_c={lambda_causal:.3f}  "
                f"gate={gate.mean().item():.3f}", flush=True
            )

    return {
        "loss_main":   total_main / max(1, n_batches),
        "loss_causal": total_caus / max(1, n_batches),
        "loss_div":    total_div  / max(1, n_batches),
        "n_batches":   n_batches,
    }


@torch.no_grad()
def predict_mu_hr_dualpath(
    batch: dict,
    *,
    encoder,
    rcn_runner,
    regression_head,
    dual_path,
    builder=None,
    device: torch.device,
    target_shape: Optional[Sequence[int]] = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Inference with Dual-Path Stage 1.

    Returns
    -------
    mu_A     : [B,1,H,W]  causal prediction
    mu_B     : [B,1,H,W]  spatial CNN prediction
    mu_total : [B,1,H,W]  fused prediction
    gate     : [B,1,H,W]  gate values ∈ (0,1)
    """
    encoder.eval()
    if hasattr(rcn_runner, "cell"):
        rcn_runner.cell.eval()
    regression_head.eval()
    dual_path.eval()

    lr_data = batch["lr"].to(device)
    h_init  = encoder.init_state(batch["hetero"]).to(device)
    drivers = [lr_data[t] for t in range(lr_data.shape[0])]
    seq_out = rcn_runner.run(h_init, drivers, reconstruction_sources=None)
    H_T     = seq_out.states[-1]
    mu_A    = regression_head(H_T)
    if mu_A.dim() == 3:
        mu_A = mu_A.unsqueeze(0)

    lr_grid  = batch_lr_grid_last(batch, builder=builder, device=device)
    lr_safe  = torch.nan_to_num(lr_grid, nan=0.0)
    mu_total, mu_B, gate = dual_path(lr_safe, mu_A)

    if target_shape is not None:
        ts = tuple(target_shape)
        mu_A     = F.interpolate(mu_A,     size=ts, mode="bilinear", align_corners=False)
        mu_B     = F.interpolate(mu_B,     size=ts, mode="bilinear", align_corners=False)
        mu_total = F.interpolate(mu_total, size=ts, mode="bilinear", align_corners=False)
        gate     = F.interpolate(gate,     size=ts, mode="bilinear", align_corners=False)

    return mu_A, mu_B, mu_total, gate
