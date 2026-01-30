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
from torch import Tensor

from ..models.causal_rcn import RCNCell, RCNSequenceRunner
from ..models.diffusion_decoder import CausalDiffusionDecoder, DiffusionOutput

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


def compute_crps(samples: Sequence[Tensor], target: Tensor) -> float:
    """
    Calcule le CRPS (Continuous Ranked Probability Score) pour un ensemble d'échantillons.
    
    Phase 4.1: Improved CRPS implementation.
    """
    if len(samples) == 0:
        return float("nan")
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


def evaluate_metrics(
    samples: Sequence[DiffusionOutput],
    target: Tensor,
    baseline: Optional[Tensor] = None,
    *,
    compute_advanced: bool = True,
    fss_threshold: Optional[float] = None,
    fss_window_size: int = 9,
    compute_f1_extremes: bool = True,  # Phase C4: Compute F1 for extreme events
    f1_percentiles: Sequence[float] = [95.0, 99.0],  # Phase C4: Percentiles for F1
) -> MetricReport:
    """
    Calcule un ensemble de métriques à partir des échantillons générés.
    
    Phase 4.1: Now includes advanced metrics (FSS, Wasserstein, Energy Score).
    
    Parameters
    ----------
    samples : Sequence[DiffusionOutput]
        Ensemble d'échantillons générés
    target : Tensor
        Cible [C, H, W] ou [H, W]
    baseline : Optional[Tensor]
        Baseline optionnel pour comparaison
    compute_advanced : bool
        Si True, calcule les métriques avancées (FSS, Wasserstein, Energy Score)
    fss_threshold : Optional[float]
        Seuil pour le calcul du FSS (si None, FSS n'est pas calculé)
    fss_window_size : int
        Taille de fenêtre pour le FSS (doit être impair)
    """
    if len(samples) == 0:
        raise ValueError("La liste d'échantillons ne doit pas être vide.")
    stacked_means = torch.stack([sample.t_mean for sample in samples], dim=0)
    pred_mean = stacked_means.mean(dim=0)

    mse = compute_mse(pred_mean, target)
    mae = compute_mae(pred_mean, target)
    hist_distance = compute_histogram_distance(pred_mean, target)
    crps = compute_crps([sample.t_mean for sample in samples], target)
    spectrum = compute_spectrum_distance(pred_mean, target)

    baseline_mse = baseline_mae = None
    baseline_tensor = baseline
    if baseline_tensor is None and samples[0].baseline is not None:
        baseline_tensor = samples[0].baseline
    if baseline_tensor is not None:
        baseline_mse = compute_mse(baseline_tensor, target)
        baseline_mae = compute_mae(baseline_tensor, target)

    # Phase 4.1: Compute advanced metrics if requested
    fss_val = None
    wasserstein_val = None
    energy_score_val = None
    
    if compute_advanced:
        try:
            # Compute FSS if threshold is provided
            if fss_threshold is not None:
                fss_val = compute_fss(pred_mean, target, threshold=fss_threshold, window_size=fss_window_size)
            
            # Compute Wasserstein distance
            wasserstein_val = compute_wasserstein_distance([sample.t_mean for sample in samples], target)
            
            # Compute Energy Score
            energy_score_val = compute_energy_score([sample.t_mean for sample in samples], target)
        except Exception as e:
            # If advanced metrics fail, continue with basic metrics
            import warnings
            warnings.warn(f"Failed to compute advanced metrics: {e}")
    
    # Phase C4: Compute F1 scores for extreme events
    f1_extremes_val = None
    if compute_f1_extremes:
        try:
            f1_extremes_val = compute_f1_extremes(pred_mean, target, threshold_percentiles=f1_percentiles)
        except Exception as e:
            import warnings
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
    )


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

    df = pd.DataFrame(edges)
    df.to_json(output_path, orient="records", indent=2)

