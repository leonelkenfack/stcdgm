"""Phase 6 — Comparaison standardisée à trois variantes pour Oracle V5.

Script de comparaison équitable entre trois variantes du modèle sur le même
jeu de test, avec les mêmes seeds d'évaluation et le même protocole. Produit
un tableau JSON + Markdown imprimable des métriques cibles du trilemme.

Les trois variantes
-------------------
1. ``corrdiff_noncausal`` : modèle entraîné avec ``run_variant="noncausal"``.
   Le DAG est forcé à zéro pendant l'entraînement → équivalent CorrDiff
   générique. Sert de baseline non causale.

2. ``oracle_v5_full`` : V5-mini complet (skip-connection + tail S1 + tail S2
   + P3-lite + prior physique + DAG appris). C'est la variante à promouvoir.

3. ``oracle_v5_ablated`` : V5-mini avec ``A_dag := 0`` forcé à l'inférence
   uniquement (le checkpoint reste celui de oracle_v5_full). Mesure
   l'apport effectif du chemin causal sur la prédiction terminale
   (différent de Δ_O3 qui mesure sur μ_HR seul).

Métriques calculées
-------------------
- Pearson global + per-sample
- RMSE / MAE
- Spread (ensemble std) + Spread/RMSE ratio (calibration probabiliste)
- p95 / p99 (Pearson sur les centiles élevés)
- RAPSD distance (cohérence spectrale)
- F1-p95 / F1-p99 (si module dispo)
- CRPS proxy (Continuous Ranked Probability Score sur l'ensemble)

Si ``cfg.v5.skip_connection.enabled`` :
- α moyen (gate du skip) — préservation O3
- Δ_O3 ratio (entre full et ablated sur μ_HR)

Usage
-----
.. code-block:: bash

    python -m scripts.eval_three_variants \\
        --ckpt-noncausal ckpt_noncausal/epoch_last.pth \\
        --ckpt-v5       ckpt_v5/epoch_last.pth \\
        --out           results/v5_three_variants.json \\
        --num-batches   16 \\
        --num-samples   16 \\
        --seed          42

Sortie : ``results/v5_three_variants.json`` + ``.md`` imprimable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation, NaN-safe, global flatten."""
    a = a.flatten().float()
    b = b.flatten().float()
    mask = torch.isfinite(a) & torch.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a * b).sum() / denom)


def _rmse(a: torch.Tensor, b: torch.Tensor) -> float:
    sq = (a - b).pow(2)
    return float(sq.nanmean().sqrt())


def _mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().nanmean())


def _pearson_at_percentile(pred: torch.Tensor, target: torch.Tensor, q: float) -> float:
    """Pearson restreint aux pixels dépassant le centile q de target."""
    flat_t = target.flatten().float()
    flat_p = pred.flatten().float()
    mask = torch.isfinite(flat_t) & torch.isfinite(flat_p)
    flat_t, flat_p = flat_t[mask], flat_p[mask]
    if flat_t.numel() < 100:
        return float("nan")
    thr = torch.quantile(flat_t, q)
    sel = flat_t > thr
    if sel.sum() < 10:
        return float("nan")
    return _pearson(flat_p[sel], flat_t[sel])


def _f1_at_percentile(pred: torch.Tensor, target: torch.Tensor, q: float) -> float:
    """F1 binaire pour les événements dépassant le centile q (calculé sur target)."""
    flat_t = target.flatten().float()
    flat_p = pred.flatten().float()
    mask = torch.isfinite(flat_t) & torch.isfinite(flat_p)
    flat_t, flat_p = flat_t[mask], flat_p[mask]
    if flat_t.numel() < 100:
        return float("nan")
    thr = torch.quantile(flat_t, q).item()
    yt = (flat_t > thr).float()
    yp = (flat_p > thr).float()
    tp = (yt * yp).sum().item()
    fp = ((1 - yt) * yp).sum().item()
    fn = (yt * (1 - yp)).sum().item()
    if tp + fp + fn == 0:
        return float("nan")
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    if precision + recall < 1e-12:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _spread(ensemble: torch.Tensor) -> float:
    """Écart-type moyen sur l'ensemble (axe 0 = samples)."""
    if ensemble.dim() < 4 or ensemble.shape[0] < 2:
        return float("nan")
    return float(ensemble.std(dim=0).nanmean())


def _crps_proxy(ensemble: torch.Tensor, target: torch.Tensor) -> float:
    """CRPS estimateur Hersbach (proxy)."""
    if ensemble.dim() < 4 or ensemble.shape[0] < 2:
        return float("nan")
    K = ensemble.shape[0]
    # CRPS ≈ (1/K) Σ |x_k - y| - (1/2K²) Σ_{j,k} |x_j - x_k|
    abs_diff_target = (ensemble - target.unsqueeze(0)).abs().nanmean(dim=0)
    pairs = (ensemble.unsqueeze(0) - ensemble.unsqueeze(1)).abs()
    abs_diff_pairs = pairs.nanmean(dim=(0, 1)) * 0.5
    crps = (abs_diff_target - abs_diff_pairs).nanmean()
    return float(crps)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def evaluate_variant(
    variant_name: str,
    predictions_ensemble: torch.Tensor,    # [K, B, 1, H, W]
    targets: torch.Tensor,                  # [B, 1, H, W]
    mu_HR: torch.Tensor | None = None,      # [B, 1, H, W] for Δ_O3
    mu_HR_ablated: torch.Tensor | None = None,  # [B, 1, H, W] (A_dag=0 inference)
) -> Dict[str, Any]:
    """Compute all metrics for one variant."""
    ens_mean = predictions_ensemble.nanmean(dim=0)

    m: Dict[str, Any] = {
        "variant": variant_name,
        "n_test_batches": int(targets.shape[0]),
        "n_ensemble": int(predictions_ensemble.shape[0]),
        "pearson_global": _pearson(ens_mean, targets),
        "rmse": _rmse(ens_mean, targets),
        "mae": _mae(ens_mean, targets),
        "spread_mean": _spread(predictions_ensemble),
        "pearson_p95": _pearson_at_percentile(ens_mean, targets, 0.95),
        "pearson_p99": _pearson_at_percentile(ens_mean, targets, 0.99),
        "f1_p95": _f1_at_percentile(ens_mean, targets, 0.95),
        "f1_p99": _f1_at_percentile(ens_mean, targets, 0.99),
        "crps_proxy": _crps_proxy(predictions_ensemble, targets),
    }

    # Spread/RMSE ratio
    if m["rmse"] > 1e-12 and not (m["spread_mean"] != m["spread_mean"]):
        m["spread_rmse_ratio"] = m["spread_mean"] / m["rmse"]
    else:
        m["spread_rmse_ratio"] = float("nan")

    # Δ_O3 si on a μ_HR full + ablated
    if mu_HR is not None and mu_HR_ablated is not None:
        mse_full = (mu_HR - targets).pow(2).nanmean()
        mse_abl = (mu_HR_ablated - targets).pow(2).nanmean()
        if mse_full > 1e-12:
            m["delta_o3_ratio"] = float((mse_abl - mse_full) / mse_full)
        else:
            m["delta_o3_ratio"] = float("nan")

    return m


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--ckpt-noncausal", type=Path, required=True,
        help="Path to noncausal checkpoint (.pth)",
    )
    parser.add_argument(
        "--ckpt-v5", type=Path, required=True,
        help="Path to Oracle V5 checkpoint (.pth)",
    )
    parser.add_argument(
        "--out", type=Path, required=True,
        help="Output JSON file (a .md sibling is also written)",
    )
    parser.add_argument(
        "--num-batches", type=int, default=16,
        help="Number of evaluation batches",
    )
    parser.add_argument(
        "--num-samples", type=int, default=16,
        help="Number of stochastic samples per batch (ensemble size)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for eval reproducibility",
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/training_config.yaml"),
        help="Path to CONFIG YAML (for model rebuild)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 6 — Comparaison trois variantes V5")
    print("=" * 70)
    print(f"  noncausal ckpt : {args.ckpt_noncausal}")
    print(f"  V5 ckpt        : {args.ckpt_v5}")
    print(f"  batches × K    : {args.num_batches} × {args.num_samples}")
    print(f"  seed           : {args.seed}")
    print()

    # NOTE :
    # Ce script est un *squelette* opérationnel. Le chargement effectif des
    # checkpoints et la boucle de sampling nécessitent les helpers du repo
    # (``convert_sample_to_batch``, ``rebuild_model_from_ckpt``,
    # ``build_two_stage_inputs``, ``sample_once_edm``). Ils dépendent de
    # l'état runtime du notebook (CONFIG, builder, dataset). Pour éviter
    # une duplication fragile, le harness délègue à une fonction
    # ``evaluate_checkpoint_three_variants`` à appeler depuis un notebook
    # (cf. docstring du module).

    print("⚠ Pour l'exécution complète, importer ce module depuis un notebook")
    print("   ayant déjà chargé CONFIG, dataset et builders. Voir la docstring.")
    print()
    print("Workflow recommandé :")
    print("  1. Charger CONFIG, dataset, builder dans le notebook")
    print("  2. Charger les deux checkpoints (noncausal et V5)")
    print("  3. Pour chaque variante : générer ensemble K samples × batches")
    print("  4. Appeler evaluate_variant(name, ensemble, targets, mu_HR, ...)")
    print("  5. Sauvegarder résultats via save_results(...)")

    return 0


def save_results(metrics_list: list, out_path: Path) -> None:
    """Sauvegarde JSON + Markdown."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {"variants": metrics_list, "schema_version": "v5.1"},
            f, ensure_ascii=False, indent=2,
        )

    # Markdown imprimable
    md_path = out_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Comparaison trois variantes V5\n\n")
        f.write("| Métrique | " + " | ".join(m["variant"] for m in metrics_list) + " |\n")
        f.write("|---|" + "---|" * len(metrics_list) + "\n")
        for key in [
            "pearson_global", "rmse", "mae", "spread_mean", "spread_rmse_ratio",
            "pearson_p95", "pearson_p99", "f1_p95", "f1_p99", "crps_proxy",
            "delta_o3_ratio",
        ]:
            row = f"| {key} | "
            for m in metrics_list:
                v = m.get(key, "—")
                if isinstance(v, float):
                    row += f"{v:.4f} | " if not (v != v) else "NaN | "
                else:
                    row += f"{v} | "
            f.write(row + "\n")
    print(f"✓ Résultats sauvegardés : {out_path} (+ {md_path.name})")


__all__ = ["evaluate_variant", "save_results", "main"]


if __name__ == "__main__":
    sys.exit(main())
