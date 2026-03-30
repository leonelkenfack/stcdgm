"""
Module 7 – Évaluation et interprétabilité (XAI) pour l’architecture ST-CDGM.

Ce module propose des fonctions pour :
  * effectuer une inférence auto-régressive avec génération multi-échantillons,
  * calculer des métriques de précision (MSE, MAE) et de réalisme (histogrammes, CRPS placeholder),
  * visualiser et exporter le DAG appris.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.graph_builder import HeteroGraphBuilder
from ..models.causal_rcn import RCNCell, RCNSequenceRunner
from ..models.diffusion_decoder import CausalDiffusionDecoder, DiffusionOutput
from ..models.intelligible_encoder import IntelligibleVariableEncoder

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dépendance optionnelle
    raise ImportError(
        "Le module evaluation_xai nécessite seaborn et matplotlib "
        "(pip install seaborn matplotlib)."
    ) from exc


@dataclass
class InferenceResult:
    """
    Résultat d'une inférence auto-régressive multi-échantillons.
    """

    generations: List[List[DiffusionOutput]]  # [time][sample]
    states: List[Tensor]
    dag_matrices: List[Tensor]


# ---------------------------------------------------------------------------
# Phase 1 / 5 — Inférence centralisée (bicubique, masque NaN)
# ---------------------------------------------------------------------------


def resize_tensor_bicubic_nonneg(
    x: Tensor,
    size: Tuple[int, int],
    *,
    clamp_min: float = 0.0,
) -> Tensor:
    """
    Redimensionnement spatial bicubique + clamp (précip / champs positifs).
    Remplace l'interpolation bilinéaire (passe-bas) du pipeline legacy.
    """
    if x.dim() != 4:
        raise ValueError(f"Attendu [B,C,H,W], obtenu {tuple(x.shape)}")
    y = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
    return torch.clamp(y, min=clamp_min)


def resize_diffusion_output_to_spatial(
    out: DiffusionOutput,
    spatial: Tuple[int, int],
    *,
    clamp_min: float = 0.0,
) -> DiffusionOutput:
    """Aligne t_min / t_mean / t_max / residual sur la grille cible."""
    Ht, Wt = spatial

    def _resize(t: Tensor) -> Tensor:
        if t.dim() != 4:
            return t
        if t.shape[-2:] == (Ht, Wt):
            return t
        return resize_tensor_bicubic_nonneg(t, (Ht, Wt), clamp_min=clamp_min)

    res = _resize(out.residual)
    base = out.baseline
    if base is not None and base.dim() == 4 and base.shape[-2:] != (Ht, Wt):
        base = resize_tensor_bicubic_nonneg(base, (Ht, Wt), clamp_min=clamp_min)
    return DiffusionOutput(
        residual=res,
        baseline=base,
        t_min=_resize(out.t_min),
        t_mean=_resize(out.t_mean),
        t_max=_resize(out.t_max),
    )


def convert_sample_to_batch(
    sample: dict,
    builder: HeteroGraphBuilder,
    device: torch.device,
) -> dict:
    """Construit lr, residual, baseline, hetero depuis un échantillon dataset."""
    lr_seq = sample["lr"]
    seq_len = lr_seq.shape[0]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    dynamic_features = {node_type: lr_nodes_steps[0] for node_type in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_features).to(device)
    return {
        "lr": lr_tensor,
        "residual": sample["residual"],
        "baseline": sample.get("baseline"),
        "hetero": hetero,
    }


def extract_target_baseline_and_mask(
    batch: dict,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Cible HR, baseline et masque de validité (True = donnée observée).

    Phase 5.1 : le masque est extrait **avant** nan_to_num.
    """
    target_residual = batch["residual"][-1].to(device)
    baseline_tensor = (
        batch["baseline"][-1]
        if batch.get("baseline") is not None
        else torch.zeros_like(target_residual)
    )
    baseline_tensor = baseline_tensor.to(device)
    full_target = baseline_tensor + target_residual

    valid_mask = ~torch.isnan(full_target)

    baseline_tensor = torch.nan_to_num(baseline_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    full_target = torch.nan_to_num(full_target, nan=0.0, posinf=0.0, neginf=0.0)

    b = baseline_tensor.unsqueeze(0) if baseline_tensor.dim() == 3 else baseline_tensor
    t = full_target.unsqueeze(0) if full_target.dim() == 3 else full_target
    m = valid_mask.unsqueeze(0) if valid_mask.dim() == 3 else valid_mask
    return t, b, m


@torch.no_grad()
def run_st_cdgm_inference(
    sample: dict,
    *,
    builder: HeteroGraphBuilder,
    encoder: IntelligibleVariableEncoder,
    rcn_runner: RCNSequenceRunner,
    diffusion: CausalDiffusionDecoder,
    device: torch.device,
    num_samples: int,
    num_steps: int,
    scheduler_type: str = "ddpm",
    apply_constraints: bool = False,
    use_log1p_inverse: bool = False,
    cfg_scale: float = 0.0,
) -> Tuple[List[DiffusionOutput], Tensor, Tensor, Tensor, Tensor]:
    """
    Inférence complète encodeur → RCN → diffusion (multi-échantillons).

    Retourne ``samples_out, target_batch, baseline_batch, dag_last, mask_batch``.
    """
    batch = convert_sample_to_batch(sample, builder, device)
    lr_data = batch["lr"].to(device)
    target_batch, baseline_batch, mask_batch = extract_target_baseline_and_mask(batch, device)

    H_init = encoder.init_state(batch["hetero"]).to(device)
    drivers = [lr_data[t] for t in range(lr_data.shape[0])]
    seq_output = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
    conditioning = encoder.project_state_tensor(seq_output.states[-1]).to(device)
    dag_last = seq_output.dag_matrices[-1].detach().cpu()

    samples_out: List[DiffusionOutput] = []
    for _ in range(num_samples):
        generated = diffusion.sample(
            conditioning,
            num_steps=num_steps,
            scheduler_type=scheduler_type,
            apply_constraints=apply_constraints,
            baseline=baseline_batch,
            cfg_scale=cfg_scale,
        )
        if generated.t_mean.shape != target_batch.shape:
            generated = resize_diffusion_output_to_spatial(
                generated,
                (target_batch.shape[-2], target_batch.shape[-1]),
                clamp_min=0.0,
            )
        generated.t_mean = torch.nan_to_num(generated.t_mean, nan=0.0, posinf=0.0, neginf=0.0)
        if use_log1p_inverse:
            generated.t_mean = torch.expm1(generated.t_mean)
        samples_out.append(generated)

    return samples_out, target_batch.cpu(), baseline_batch.cpu(), dag_last, mask_batch.cpu()


def autoregressive_inference(
    *,
    rcn_runner: RCNSequenceRunner,
    diffusion_decoder: CausalDiffusionDecoder,
    initial_state: Tensor,
    drivers: Sequence[Tensor],
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
) -> InferenceResult:
    """
    Déroule le modèle de manière auto-régressive et génère plusieurs échantillons HR.

    Parameters
    ----------
    initial_state :
        État initial H(0) [q, N, hidden_dim].
    drivers :
        Séquence de forçages externes [T][N, driver_dim].
    num_samples :
        Nombre d'échantillons diffusion à générer par pas de temps.
    """
    H_t = initial_state
    generations: List[List[DiffusionOutput]] = []
    states: List[Tensor] = []
    dag_mats: List[Tensor] = []

    for driver in drivers:
        cell: RCNCell = rcn_runner.cell
        H_t, _, A_masked = cell(H_t, driver)
        states.append(H_t)
        dag_mats.append(A_masked)

        conditioning = H_t.mean(dim=0).unsqueeze(0)  # [1, hidden_dim, H, W] placeholder
        step_outputs: List[DiffusionOutput] = []
        for _ in range(num_samples):
            out = diffusion_decoder.sample(conditioning, generator=generator)
            step_outputs.append(out)

        generations.append(step_outputs)

    return InferenceResult(generations=generations, states=states, dag_matrices=dag_mats)


# ---------------------------------------------------------------------------
# Métriques de précision et de réalisme
# ---------------------------------------------------------------------------

def compute_mse(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> float:
    """Calcule MSE avec gestion des NaN."""
    if mask is None:
        mask = ~torch.isnan(target) & ~torch.isnan(pred)
    pred_valid = pred[mask]
    target_valid = target[mask]
    if pred_valid.numel() == 0:
        return 0.0
    return torch.mean((pred_valid - target_valid) ** 2).item()


def compute_mae(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> float:
    """Calcule MAE avec gestion des NaN."""
    if mask is None:
        mask = ~torch.isnan(target) & ~torch.isnan(pred)
    pred_valid = pred[mask]
    target_valid = target[mask]
    if pred_valid.numel() == 0:
        return 0.0
    return torch.mean(torch.abs(pred_valid - target_valid)).item()


def compute_histogram_distance(pred: Tensor, target: Tensor, bins: int = 50) -> float:
    """
    Distance simple entre histogrammes (L1) pour évaluer le réalisme.
    """
    pred_np = pred.detach().cpu().numpy().ravel()
    target_np = target.detach().cpu().numpy().ravel()
    hist_pred, bin_edges = np.histogram(pred_np, bins=bins, density=True)
    hist_target, _ = np.histogram(target_np, bins=bin_edges, density=True)
    distance = np.sum(np.abs(hist_pred - hist_target)) * (bin_edges[1] - bin_edges[0])
    return float(distance)


def compute_crps_pixel_map(pred_stack: Tensor, target: Tensor) -> Tensor:
    """
    Carte CRPS pixel (formule énergétique) : pred_stack [N,C,H,W], target [C,H,W] ou [1,C,H,W].
    """
    if target.dim() == 4:
        target = target.squeeze(0)
    tgt = target[0] if target.dim() == 3 else target
    x = pred_stack[:, 0] if pred_stack.dim() == 4 and pred_stack.shape[1] >= 1 else pred_stack
    n = x.shape[0]
    if n < 1:
        raise ValueError("Ensemble vide")
    term1 = (x - tgt.unsqueeze(0)).abs().mean(dim=0)
    if n < 2:
        return term1
    term2 = (x.unsqueeze(0) - x.unsqueeze(1)).abs().mean(dim=(0, 1)) * 0.5
    return term1 - term2


def compute_spread_skill_ratio(pred_std_map: np.ndarray, err_map: np.ndarray, valid_mask: np.ndarray) -> float:
    """Ratio moyen spread / erreur sur pixels valides (objectif ~1 calibration)."""
    m = valid_mask & np.isfinite(pred_std_map) & np.isfinite(err_map)
    if not np.any(m):
        return float("nan")
    spread = pred_std_map[m].mean()
    skill = err_map[m].mean()
    if skill < 1e-12:
        return float("nan")
    return float(spread / skill)


def compute_rapsd_numpy(field: np.ndarray) -> np.ndarray:
    """RAPSD 2D (numpy) pour une carte [H,W]."""
    fft2 = np.fft.fft2(field)
    power = np.abs(np.fft.fftshift(fft2)) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(np.int64)
    radial_sum = np.bincount(r.ravel(), power.ravel())
    radial_cnt = np.bincount(r.ravel())
    valid = radial_cnt > 0
    out = np.zeros_like(radial_sum, dtype=np.float64)
    out[valid] = radial_sum[valid] / radial_cnt[valid]
    return out


def compute_crps(
    samples: Sequence[Tensor],
    target: Tensor,
    *,
    max_ensemble_members: Optional[int] = None,
) -> float:
    """
    Calcule le CRPS (Continuous Ranked Probability Score) pour un ensemble d'échantillons.
    Formule quadratique en la taille d'ensemble : plafonner ``max_ensemble_members`` (premiers membres) sur CPU.

    Phase 4.1: Improved CRPS implementation.
    """
    if len(samples) == 0:
        return float("nan")
    if max_ensemble_members is not None and len(samples) > max_ensemble_members:
        samples = list(samples)[:max_ensemble_members]
    stack = torch.stack(samples, dim=0)  # [ensemble, C, H, W]
    target = target.unsqueeze(0)
    term1 = torch.abs(stack - target).mean(dim=0)
    pairwise = torch.abs(stack.unsqueeze(0) - stack.unsqueeze(1)).mean(dim=(0, 1))
    crps = (term1 - 0.5 * pairwise).mean().item()
    return float(crps)


def compute_fss(pred: Tensor, target: Tensor, threshold: float, window_size: int = 9) -> float:
    """
    Calcule le Fraction Skill Score (FSS) pour l'évaluation spatiale.
    
    Phase 4.1: FSS measures spatial forecast skill for binary events.
    
    Parameters
    ----------
    pred : Tensor
        Prédiction [C, H, W] ou [H, W]
    target : Tensor
        Cible [C, H, W] ou [H, W]
    threshold : float
        Seuil pour binariser les champs
    window_size : int
        Taille de la fenêtre de voisinage pour le calcul (doit être impair)
    
    Returns
    -------
    float
        FSS score (0-1, 1 = perfect forecast)
    
    References
    ----------
    - Roberts & Lean (2008): "Scale-Selective Verification of Rainfall Accumulations
      from High-Resolution Forecasts of Convective Events"
    """
    # Ensure 2D tensors
    if pred.dim() == 3:
        pred = pred[0]  # Take first channel
    if target.dim() == 3:
        target = target[0]
    
    # Binarize fields based on threshold
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()
    
    # Compute fraction of events in windows using convolution
    # Create averaging kernel
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device, dtype=pred.dtype) / (window_size ** 2)
    pred_binary_expanded = pred_binary.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    target_binary_expanded = target_binary.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    pred_frac = torch.nn.functional.conv2d(pred_binary_expanded, kernel, padding=window_size//2).squeeze()
    target_frac = torch.nn.functional.conv2d(target_binary_expanded, kernel, padding=window_size//2).squeeze()
    
    # Compute MSE of fractions
    mse_frac = torch.mean((pred_frac - target_frac) ** 2).item()
    
    # Compute MSE of fractions under random forecast (reference)
    pred_ref = torch.mean(pred_binary).item()
    target_ref = torch.mean(target_binary).item()
    mse_ref = pred_ref ** 2 + target_ref ** 2
    
    # FSS = 1 - MSE_frac / MSE_ref
    if mse_ref == 0:
        return 1.0  # Perfect forecast (both fields have no events or all events)
    fss = 1.0 - (mse_frac / mse_ref)
    return float(max(0.0, fss))  # Clamp to [0, 1]


def compute_f1_extremes(
    pred: Tensor,
    target: Tensor,
    threshold_percentiles: Sequence[float] = [95.0, 99.0],
) -> Dict[str, float]:
    """
    Phase C4: Compute F1 score for extreme events at different percentile thresholds.
    
    This metric is crucial for evaluating the model's performance on extreme events,
    which are often the most important for climate applications.
    
    Parameters
    ----------
    pred : Tensor
        Predicted field [C, H, W] or [H, W]
    target : Tensor
        Target field [C, H, W] or [H, W]
    threshold_percentiles : Sequence[float]
        Percentiles to use as thresholds for extreme events (default: [95, 99])
    
    Returns
    -------
    Dict[str, float]
        Dictionary mapping percentile threshold to F1 score
        Example: {"p95": 0.85, "p99": 0.72}
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Remove NaN/Inf if present
    valid_mask = torch.isfinite(pred_flat) & torch.isfinite(target_flat)
    pred_valid = pred_flat[valid_mask]
    target_valid = target_flat[valid_mask]
    
    if pred_valid.numel() == 0:
        return {f"p{p}": 0.0 for p in threshold_percentiles}
    
    results = {}
    
    for percentile in threshold_percentiles:
        # Compute threshold based on target distribution
        threshold = torch.quantile(target_valid, percentile / 100.0)
        
        # Binary classification: extreme (1) vs non-extreme (0)
        pred_binary = (pred_valid >= threshold).float()
        target_binary = (target_valid >= threshold).float()
        
        # Compute True Positives, False Positives, False Negatives
        tp = (pred_binary * target_binary).sum().item()
        fp = (pred_binary * (1 - target_binary)).sum().item()
        fn = ((1 - pred_binary) * target_binary).sum().item()
        
        # Compute Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[f"p{int(percentile)}"] = f1
    
    return results


def compute_wasserstein_distance(samples: Sequence[Tensor], target: Tensor, num_projections: int = 50) -> float:
    """
    Calcule la distance de Wasserstein (Sliced Wasserstein) pour comparaison de distributions.
    
    Phase 4.1: Sliced Wasserstein distance for high-dimensional distributions.
    Uses random projections to approximate Wasserstein-2 distance.
    
    Parameters
    ----------
    samples : Sequence[Tensor]
        Ensemble de prédictions [C, H, W] ou [H, W] chacune
    target : Tensor
        Cible [C, H, W] ou [H, W]
    num_projections : int
        Nombre de projections aléatoires pour l'approximation
    
    Returns
    -------
    float
        Distance de Wasserstein approximée
    """
    if len(samples) == 0:
        return float("nan")
    
    # Flatten spatial dimensions
    samples_flat = [s.flatten() if s.dim() > 1 else s for s in samples]
    target_flat = target.flatten() if target.dim() > 1 else target
    
    device = target_flat.device
    dim = target_flat.shape[0]
    
    # Compute Sliced Wasserstein distance via random projections
    wasserstein_distances = []
    for _ in range(num_projections):
        # Random projection direction
        direction = torch.randn(dim, device=device)
        direction = direction / (torch.norm(direction) + 1e-8)
        
        # Project samples and target
        samples_proj = torch.stack([torch.dot(s, direction) for s in samples_flat])
        target_proj = torch.dot(target_flat, direction).unsqueeze(0)
        
        # Sort projections
        samples_sorted, _ = torch.sort(samples_proj)
        target_sorted, _ = torch.sort(target_proj)
        
        # Compute Wasserstein-2 distance for 1D projections
        # Average over number of samples for stability
        if len(samples_sorted) > 1:
            # Interpolate target to match samples length
            indices = torch.linspace(0, len(target_sorted) - 1, len(samples_sorted), device=device).long()
            target_interp = target_sorted[indices]
        else:
            target_interp = target_sorted
        
        w2 = torch.mean((samples_sorted - target_interp) ** 2)
        wasserstein_distances.append(w2.item())
    
    return float(np.mean(wasserstein_distances))


def compute_energy_score(samples: Sequence[Tensor], target: Tensor) -> float:
    """
    Calcule l'Energy Score pour évaluer la cohérence multivariée de l'ensemble.
    
    Phase 4.1: Energy Score is a proper scoring rule for ensemble forecasts.
    
    Parameters
    ----------
    samples : Sequence[Tensor]
        Ensemble de prédictions [C, H, W] ou [H, W] chacune
    target : Tensor
        Observation cible [C, H, W] ou [H, W]
    
    Returns
    -------
    float
        Energy Score (lower is better)
    
    References
    ----------
    - Gneiting et al. (2007): "Strictly proper scoring rules, prediction, and estimation"
    """
    if len(samples) == 0:
        return float("nan")
    
    # Flatten if needed
    samples_flat = [s.flatten() if s.dim() > 1 else s for s in samples]
    target_flat = target.flatten() if target.dim() > 1 else target
    
    stack = torch.stack(samples_flat, dim=0)  # [ensemble_size, dim]
    
    # Term 1: Average distance from samples to target
    term1 = torch.mean(torch.norm(stack - target_flat.unsqueeze(0), dim=1)).item()
    
    # Term 2: Average pairwise distance within ensemble
    # Compute all pairwise distances efficiently
    pairwise_distances = torch.norm(stack.unsqueeze(1) - stack.unsqueeze(0), dim=2)
    # Average upper triangle (excluding diagonal)
    n = len(samples)
    term2 = torch.sum(torch.triu(pairwise_distances, diagonal=1)) / (n * (n - 1) / 2) if n > 1 else 0.0
    term2 = term2.item()
    
    # Energy Score = term1 - 0.5 * term2
    energy_score = term1 - 0.5 * term2
    return float(energy_score)


def _prepare_field(field: Tensor) -> Tensor:
    """
    Mise en forme standard pour le calcul du spectre (2D).
    """
    if field.dim() == 4:
        field = field.mean(dim=0)
    if field.dim() == 3:
        return field
    raise ValueError(f"Champ inattendu de forme {tuple(field.shape)} pour le calcul spectral.")


def compute_power_spectrum(field: Tensor) -> Tensor:
    """
    Calcule le spectre de puissance moyen (modulus squared de la FFT 2D).
    """
    prepared = _prepare_field(field)
    centered = prepared - prepared.mean()
    fft = torch.fft.rfft2(centered, dim=(-2, -1))
    power = (fft.real ** 2 + fft.imag ** 2)
    return power.mean(dim=0)


def compute_spectrum_distance(pred: Tensor, target: Tensor) -> float:
    """
    Compare les spectres de puissance (L1 moyen).
    """
    pred_spec = compute_power_spectrum(pred)
    target_spec = compute_power_spectrum(target)
    return torch.mean(torch.abs(pred_spec - target_spec)).item()


def compute_temporal_variance_metrics(
    predictions: Sequence[Tensor],
    targets: Sequence[Tensor],
) -> Dict[str, float]:
    """
    Compare la variabilité temporelle des prédictions et des cibles.

    Calcule la variance le long de la dimension "temps" (échantillons ordonnés),
    puis compare les champs de variance (RMSE et corrélation de Pearson).

    Parameters
    ----------
    predictions : Sequence[Tensor]
        Liste de tenseurs [C, H, W] ou [1, C, H, W] (un par pas de temps / échantillon).
    targets : Sequence[Tensor]
        Liste de tenseurs de même forme que predictions.

    Returns
    -------
    Dict[str, float]
        {"temporal_var_rmse": float, "temporal_var_corr": float}
        Si N < 2, les valeurs sont float("nan").
    """
    nan_result = {"temporal_var_rmse": float("nan"), "temporal_var_corr": float("nan")}
    if len(predictions) < 2 or len(targets) < 2 or len(predictions) != len(targets):
        return nan_result

    # Stack: ensure [N, C, H, W]
    pred_list = [p.squeeze(0) if p.dim() == 4 else p for p in predictions]
    tgt_list = [t.squeeze(0) if t.dim() == 4 else t for t in targets]
    pred_stack = torch.stack(pred_list, dim=0)
    target_stack = torch.stack(tgt_list, dim=0)

    # Variance along dim=0 -> [C, H, W]
    var_pred = pred_stack.var(dim=0)
    var_target = target_stack.var(dim=0)

    # Flatten for scalar metrics
    vp = var_pred.flatten()
    vt = var_target.flatten()

    # RMSE between variance maps
    rmse = torch.sqrt(torch.mean((vp - vt) ** 2)).item()

    # Pearson correlation
    vp_c = vp - vp.mean()
    vt_c = vt - vt.mean()
    eps = 1e-8
    num = (vp_c * vt_c).sum()
    denom = torch.sqrt((vp_c ** 2).sum()) * torch.sqrt((vt_c ** 2).sum()) + eps
    corr = (num / denom).item() if denom.item() > eps else 0.0

    return {"temporal_var_rmse": rmse, "temporal_var_corr": corr}


@dataclass
class MetricReport:
    mse: float
    mae: float
    hist_distance: float
    crps: float
    spectrum_distance: float
    baseline_mse: Optional[float] = None
    baseline_mae: Optional[float] = None
    # Phase 4.1: Advanced metrics
    fss: Optional[float] = None
    wasserstein_distance: Optional[float] = None
    energy_score: Optional[float] = None
    # Phase C4: F1 scores for extreme events
    f1_extremes: Optional[Dict[str, float]] = None
    # Phase 1 / 8 — métriques probabilistes (membre unique + CRPS spatial)
    mse_single_member: Optional[float] = None
    mae_single_member: Optional[float] = None
    corr_single_member: Optional[float] = None
    crps_spatial_mean: Optional[float] = None
    spectral_distance_rapsd: Optional[float] = None
    spread_skill_ratio: Optional[float] = None
    crps_p99: Optional[float] = None
    primary_kpi: str = "crps"


def evaluate_metrics(
    samples: Sequence[DiffusionOutput],
    target: Tensor,
    baseline: Optional[Tensor] = None,
    *,
    compute_advanced: bool = True,
    fss_threshold: Optional[float] = None,
    fss_window_size: int = 9,
    include_f1_extremes: bool = True,
    f1_percentiles: Sequence[float] = [95.0, 99.0],
    use_mean_aggregation: bool = False,
    valid_mask: Optional[Tensor] = None,
    crps_max_ensemble_members: Optional[int] = None,
) -> MetricReport:
    """
    Métriques à partir des échantillons. Par défaut : **membre unique** (Phase 1) pour MSE/MAE/spectre ;
    CRPS et métriques d'ensemble utilisent tout l'ensemble. Option ``use_mean_aggregation=True`` : ancien comportement (moyenne).
    ``valid_mask`` [H,W] bool True = pixel valide (Phase 5).
    ``crps_max_ensemble_members`` : borne la taille d'ensemble pour CRPS / carte CRPS (coût O(n²)) ; None = pas de borne.
    """
    if len(samples) == 0:
        raise ValueError("La liste d'échantillons ne doit pas être vide.")
    stacked_means = torch.stack([sample.t_mean for sample in samples], dim=0)
    pred_ensemble_mean = stacked_means.mean(dim=0)
    pred_primary = pred_ensemble_mean if use_mean_aggregation else stacked_means[0]

    mask_t = valid_mask
    if mask_t is not None and mask_t.dim() == 3:
        mask_t = mask_t.squeeze(0)
    if mask_t is not None and mask_t.dim() == 3:
        mask_t = mask_t[0]

    mse = compute_mse(pred_primary, target, mask=None)
    mae = compute_mae(pred_primary, target, mask=None)
    hist_distance = compute_histogram_distance(pred_primary, target)
    if (
        crps_max_ensemble_members is not None
        and crps_max_ensemble_members > 0
        and len(samples) > crps_max_ensemble_members
    ):
        samples_crps = list(samples)[: crps_max_ensemble_members]
    else:
        samples_crps = list(samples)
    crps = compute_crps([sample.t_mean for sample in samples_crps], target)
    spectrum = compute_spectrum_distance(pred_primary, target)

    # --- Phase 1 / 8 : membre unique + CRPS spatial + RAPSD ---
    pred_single = stacked_means[0]
    mse_single = mae_single = corr_single = None
    crps_spatial_mean = spectral_dist_rapsd = spread_skill = crps_p99 = None
    try:
        ps_full = stacked_means.detach().cpu()
        ps = torch.stack([sample.t_mean for sample in samples_crps], dim=0).detach().cpu()
        tg = target.detach().cpu()
        if tg.dim() == 3:
            tg = tg.unsqueeze(0)
        crps_map = compute_crps_pixel_map(ps, tg)
        crps_spatial_mean = float(crps_map.mean().item())
        crps_p99 = float(torch.quantile(crps_map.flatten(), 0.99).item())

        p0 = pred_single.detach().cpu().numpy()
        t0 = tg.squeeze(0).numpy()
        if p0.ndim == 3:
            p0 = p0[0]
        if t0.ndim == 3:
            t0 = t0[0]
        rp = compute_rapsd_numpy(p0)
        rt = compute_rapsd_numpy(t0)
        nbin = min(len(rp), len(rt))
        spectral_dist_rapsd = float(np.mean((rp[:nbin] - rt[:nbin]) ** 2))

        mse_single = float(compute_mse(pred_single, target, mask=None))
        mae_single = float(compute_mae(pred_single, target, mask=None))
        pf = p0.flatten()
        tf = t0.flatten()
        valid = np.isfinite(pf) & np.isfinite(tf)
        if np.sum(valid) > 2:
            corr_single = float(np.corrcoef(pf[valid], tf[valid])[0, 1])

        if mask_t is not None:
            m_np = mask_t.detach().cpu().numpy().astype(bool)
            if m_np.ndim == 3:
                m_np = m_np[0]
            err_abs = np.abs(p0 - t0)
            std_map = ps_full[:, 0].numpy().std(axis=0) if ps_full.shape[1] >= 1 else np.zeros_like(p0)
            spread_skill = compute_spread_skill_ratio(std_map, err_abs, m_np)
            mse = float(np.mean(((p0 - t0) ** 2)[m_np]))
            mae = float(np.mean(err_abs[m_np]))
    except Exception as ex:
        warnings.warn(f"Métriques probabilistes étendues: {ex}")

    baseline_mse = baseline_mae = None
    baseline_tensor = baseline
    if baseline_tensor is None and samples[0].baseline is not None:
        baseline_tensor = samples[0].baseline
    if baseline_tensor is not None:
        baseline_mse = compute_mse(baseline_tensor, target)
        baseline_mae = compute_mae(baseline_tensor, target)

    fss_val = None
    wasserstein_val = None
    energy_score_val = None

    if compute_advanced:
        try:
            if fss_threshold is not None:
                fss_val = compute_fss(pred_primary, target, threshold=fss_threshold, window_size=fss_window_size)
            wasserstein_val = compute_wasserstein_distance([sample.t_mean for sample in samples], target)
            energy_score_val = compute_energy_score([sample.t_mean for sample in samples], target)
        except Exception as e:
            warnings.warn(f"Failed to compute advanced metrics: {e}")

    f1_extremes_val = None
    if include_f1_extremes:
        try:
            f1_extremes_val = compute_f1_extremes(pred_primary, target, threshold_percentiles=f1_percentiles)
        except Exception as e:
            warnings.warn(f"Failed to compute F1 extremes: {e}")

    return MetricReport(
        mse=mse,
        mae=mae,
        hist_distance=hist_distance,
        crps=crps,
        spectrum_distance=spectrum,
        baseline_mse=baseline_mse,
        baseline_mae=baseline_mae,
        fss=fss_val,
        wasserstein_distance=wasserstein_val,
        energy_score=energy_score_val,
        f1_extremes=f1_extremes_val,
        mse_single_member=mse_single,
        mae_single_member=mae_single,
        corr_single_member=corr_single,
        crps_spatial_mean=crps_spatial_mean,
        spectral_distance_rapsd=spectral_dist_rapsd,
        spread_skill_ratio=spread_skill,
        crps_p99=crps_p99,
        primary_kpi="crps",
    )


def plot_probabilistic_dashboard_3x3(
    tgt_display: np.ndarray,
    pred_single: np.ndarray,
    err_display: np.ndarray,
    pred_std: np.ndarray,
    crps_map: np.ndarray,
    mask_display: np.ndarray,
    rapsd_pred: np.ndarray,
    rapsd_tgt: np.ndarray,
    spread_skill_ratio_val: float,
    spearman_rho: float,
    *,
    title: str = "Dashboard probabiliste ST-CDGM",
) -> plt.Figure:
    """
    Grille 3×3 : cible, prédiction, erreur masquée, écart-type, CRPS, masque, spread-skill, RAPSD, calibration.
    """
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    ax = axes.flatten()
    im0 = ax[0].imshow(tgt_display, origin="lower", cmap="Blues")
    ax[0].set_title("Cible HR (blanc = manquant)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(pred_single, origin="lower", cmap="Blues")
    ax[1].set_title("Prédiction (1 membre)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(err_display, origin="lower", cmap="hot")
    ax[2].set_title("|Erreur| masquée")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)
    im3 = ax[3].imshow(pred_std, origin="lower", cmap="plasma")
    ax[3].set_title("Écart-type intra-ensemble")
    plt.colorbar(im3, ax=ax[3], fraction=0.046)
    im4 = ax[4].imshow(crps_map, origin="lower", cmap="YlOrRd")
    ax[4].set_title("CRPS pixel")
    plt.colorbar(im4, ax=ax[4], fraction=0.046)
    im5 = ax[5].imshow(mask_display.astype(float), origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax[5].set_title("Masque valide")
    plt.colorbar(im5, ax=ax[5], fraction=0.046)
    ax[6].scatter(pred_std.flatten()[:: max(1, pred_std.size // 5000)], err_display.flatten()[:: max(1, err_display.size // 5000)], alpha=0.15, s=1, c="steelblue")
    mx = float(np.nanmax([np.nanmax(pred_std), np.nanmax(err_display)]))
    ax[6].plot([0, mx], [0, mx], "r--", label="1:1")
    ax[6].set_xlabel("Spread")
    ax[6].set_ylabel("|Erreur|")
    ax[6].set_title("Spread–Skill")
    ax[6].legend()
    rad = np.arange(len(rapsd_pred))
    ax[7].loglog(rad[1:], np.maximum(rapsd_pred[1:], 1e-20), label="Préd")
    ax[7].loglog(rad[1:], np.maximum(rapsd_tgt[1:], 1e-20), label="Cible")
    ax[7].set_title("RAPSD")
    ax[7].legend()
    ax[8].axis("off")
    ax[8].text(
        0.1,
        0.7,
        f"Spread/Skill ≈ {spread_skill_ratio_val:.3f}\nSpearman ρ (dispersion vs |y|) ≈ {spearman_rho:.3f}",
        fontsize=12,
        family="monospace",
        transform=ax[8].transAxes,
    )
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def spearman_dispersion_intensity(
    pred_std: np.ndarray,
    target_abs: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    """Corrélation rang dispersion vs intensité observée (Phase 7.3), sans scipy."""
    ps = pred_std[valid_mask].flatten()
    ta = target_abs[valid_mask].flatten()
    if ps.size < 10:
        return float("nan")

    def _rank(a: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(a))

    rp = _rank(ps)
    rt = _rank(ta)
    return float(np.corrcoef(rp, rt)[0, 1])


# ---------------------------------------------------------------------------
# Visualisation et export du DAG
# ---------------------------------------------------------------------------

def plot_dag_heatmap(A_matrix: Tensor, var_names: Sequence[str], *, output_path: Optional[Path] = None) -> None:
    """
    Trace une heatmap du DAG appris.
    """
    A_np = A_matrix.detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        A_np,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=var_names,
        yticklabels=var_names,
        center=0.0,
    )
    plt.title("Matrice DAG Apprise")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def export_dag_to_csv(A_matrix: Tensor, var_names: Sequence[str], output_path: Path) -> None:
    """
    Exporte la matrice DAG en CSV.
    """
    A_np = A_matrix.detach().cpu().numpy()
    df = pd.DataFrame(A_np, index=var_names, columns=var_names)
    df.to_csv(output_path, index=True)


def export_dag_to_json(A_matrix: Tensor, var_names: Sequence[str], output_path: Path) -> None:
    """
    Exporte la matrice DAG en JSON (liste d'arêtes).
    """
    A_np = A_matrix.detach().cpu().numpy()
    edges = []
    for i, src in enumerate(var_names):
        for j, dst in enumerate(var_names):
            weight = float(A_np[i, j])
            if weight != 0.0:
                edges.append({"source": src, "target": dst, "weight": weight})
    
    with open(output_path, "w") as f:
        json.dump({"edges": edges, "variables": list(var_names)}, f, indent=2)


def compute_structural_hamming_distance(A_pred: Tensor, A_true: Tensor, threshold: float = 0.0) -> int:
    """
    Calcule la Structural Hamming Distance (SHD) entre deux DAGs.
    
    Phase 4.2: SHD measures the number of edge additions, deletions, or reversals
    needed to transform the predicted DAG into the true DAG.
    
    Parameters
    ----------
    A_pred : Tensor
        Matrice DAG prédite [q, q]
    A_true : Tensor
        Matrice DAG de référence [q, q]
    threshold : float
        Seuil pour binariser les matrices (0.0 = strict, >0 pour seuiller)
    
    Returns
    -------
    int
        Structural Hamming Distance (nombre d'erreurs d'arêtes)
    
    References
    ----------
    - Tsamardinos et al. (2006): "The max-min hill-climbing Bayesian network structure
      learning algorithm"
    """
    # Convert to numpy and binarize
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = A_true.detach().cpu().numpy().copy()
    
    # Apply threshold to binarize (optional)
    if threshold > 0:
        A_pred_binary = (np.abs(A_pred_np) > threshold).astype(int)
        A_true_binary = (np.abs(A_true_np) > threshold).astype(int)
    else:
        # Use non-zero as threshold
        A_pred_binary = (A_pred_np != 0).astype(int)
        A_true_binary = (A_true_np != 0).astype(int)
    
    # Ensure diagonal is zero (no self-loops in DAGs)
    np.fill_diagonal(A_pred_binary, 0)
    np.fill_diagonal(A_true_binary, 0)
    
    # Count differences
    # SHD = number of edges in pred but not in true +
    #       number of edges in true but not in pred +
    #       number of reversed edges
    
    # Find edges present in each DAG
    pred_edges = set()
    true_edges = set()
    
    q = A_pred_binary.shape[0]
    for i in range(q):
        for j in range(q):
            if A_pred_binary[i, j] != 0:
                pred_edges.add((i, j))
            if A_true_binary[i, j] != 0:
                true_edges.add((i, j))
    
    # Count edge additions (in pred but not in true)
    additions = pred_edges - true_edges
    
    # Count edge deletions (in true but not in pred)
    deletions = true_edges - pred_edges
    
    # Count reversals: edge (i,j) in pred and (j,i) in true (or vice versa)
    reversals = 0
    for edge in additions:
        if (edge[1], edge[0]) in deletions:
            # This edge is reversed
            reversals += 1
            additions.discard(edge)
            deletions.discard((edge[1], edge[0]))
    
    # SHD = additions + deletions + reversals
    shd = len(additions) + len(deletions) + reversals
    
    return int(shd)

