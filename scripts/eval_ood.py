"""Phase 7 — Évaluation hors-distribution (OOD) sur scénario futur.

Mesure la dégradation des métriques sous changement de régime climatique
pour valider l'hypothèse causale d'Oracle V5 : un modèle structurellement
causal devrait mieux résister au changement de distribution qu'un modèle
purement corrélatif (CorrDiff noncausal).

Protocole
---------
1. **Train** : les deux variantes (V5 et noncausal) sont entraînées sur
   la **période historique uniquement** (1950-2014 typiquement). Pas de
   contamination par les données futures.

2. **Test in-distribution (ID)** : évaluation sur le tail de la période
   historique (2015-2024) — distribution train-équivalente.

3. **Test out-of-distribution (OOD)** : évaluation sur **ACCESS-CM2
   SSP5-8.5 fin de siècle (2080-2100)**. Distribution clairement décalée.

4. **Métrique clé** : *dégradation OOD relative*

   .. math::

       \\Delta_{\\mathrm{OOD}} = \\frac{\\mathrm{Pearson}_{\\mathrm{ID}}
                                   - \\mathrm{Pearson}_{\\mathrm{OOD}}}
                                  {\\mathrm{Pearson}_{\\mathrm{ID}}}

   **Hypothèse à valider** : ``Δ_OOD(V5) < Δ_OOD(noncausal)``.

Sources données SSP5-8.5
-------------------------
- ESGF (Earth System Grid Federation) : recherche
  ``project=CMIP6, experiment_id=ssp585, source_id=ACCESS-CM2, variable=pr``.
- Période : 2080-2100 monthly ou daily.
- Région : NZ (lat -47:-34, lon 165:179).

Fallback (si données HR SSP5-8.5 indisponibles)
-----------------------------------------------
Si aucune cible HR n'est disponible pour SSP5-8.5, on évalue la **cohérence
physique** des prédictions :
- RAPSD (spectre spatial) doit rester dans la fourchette historique.
- Distribution des centiles (p50, p95, p99) doit montrer un *shift haut*
  cohérent avec l'attendu (climat plus humide en SSP5-8.5 sur la NZ).
- Pattern spatial (corrélation des cartes mean prédictions vs climatologie
  historique) — Oracle doit conserver une structure géographique cohérente.

Usage
-----
.. code-block:: bash

    python -m scripts.eval_ood \\
        --ckpt-noncausal ckpt_noncausal/epoch_last.pth \\
        --ckpt-v5       ckpt_v5/epoch_last.pth \\
        --id-data       data/test_id.nc \\
        --ood-data      data/access_cm2_ssp585_nz.nc \\
        --out           results/v5_ood.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import numpy as np


# ---------------------------------------------------------------------------
# Métriques OOD
# ---------------------------------------------------------------------------


def _pearson_safe(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().float()
    b = b.flatten().float()
    mask = torch.isfinite(a) & torch.isfinite(b)
    if mask.sum() < 2:
        return float("nan")
    a, b = a[mask], b[mask]
    a, b = a - a.mean(), b - b.mean()
    denom = (a.norm() * b.norm()).clamp(min=1e-12)
    return float((a * b).sum() / denom)


def compute_ood_degradation(
    pearson_id: float, pearson_ood: float
) -> float:
    """Δ_OOD = (Pearson_ID - Pearson_OOD) / Pearson_ID.

    Une valeur faible (proche de 0) signifie que le modèle conserve
    sa performance sur OOD ; une valeur élevée signifie qu'il
    s'effondre.
    """
    if not (pearson_id == pearson_id) or pearson_id < 1e-6:
        return float("nan")
    return float((pearson_id - pearson_ood) / pearson_id)


def evaluate_dataset(
    name: str,
    predictions_ensemble: torch.Tensor,  # [K, B, 1, H, W]
    targets: Optional[torch.Tensor],     # [B, 1, H, W] or None (fallback mode)
) -> Dict[str, Any]:
    """Compute ID or OOD metrics for a given dataset.

    En mode *fallback* (``targets is None``), seuls les diagnostics de
    cohérence physique sont calculés (distribution des centiles et
    spread spectral).
    """
    ens_mean = predictions_ensemble.nanmean(dim=0)
    metrics: Dict[str, Any] = {"dataset": name, "n_samples": int(predictions_ensemble.shape[1])}

    if targets is not None:
        metrics["pearson_global"] = _pearson_safe(ens_mean, targets)
        metrics["rmse"] = float((ens_mean - targets).pow(2).nanmean().sqrt())
        metrics["mae"] = float((ens_mean - targets).abs().nanmean())

    # Quantiles prédits (diagnostics physiques, valides même en fallback)
    flat_pred = ens_mean.flatten()
    flat_pred = flat_pred[torch.isfinite(flat_pred)]
    if flat_pred.numel() > 0:
        for q_pct in [50, 95, 99]:
            metrics[f"pred_p{q_pct}"] = float(
                torch.quantile(flat_pred, q_pct / 100.0)
            )

    # RAPSD distance si target dispo
    # (omis ici par simplicité ; appelez compute_rapsd_distance du repo
    # depuis le notebook si nécessaire)

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt-noncausal", type=Path, required=True)
    parser.add_argument("--ckpt-v5", type=Path, required=True)
    parser.add_argument(
        "--id-data", type=Path, required=True,
        help="Dataset NetCDF in-distribution (typically 2015-2024 historical)",
    )
    parser.add_argument(
        "--ood-data", type=Path, required=True,
        help="Dataset NetCDF OOD (typically ACCESS-CM2 SSP5-8.5 2080-2100)",
    )
    parser.add_argument(
        "--ood-target", type=Path, default=None,
        help="Optional HR target for OOD (fallback if missing : physical "
             "coherence diagnostics only)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-batches", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 7 — Évaluation OOD (changement de régime climatique)")
    print("=" * 70)
    print(f"  ID dataset  : {args.id_data}")
    print(f"  OOD dataset : {args.ood_data}")
    print(f"  OOD target  : {args.ood_target or '(fallback physical coherence)'}")
    print()

    print("⚠ Script à invoquer depuis un notebook ayant CONFIG, dataset, builder.")
    print()
    print("Workflow recommandé :")
    print("  1. Charger CONFIG et les 2 checkpoints (noncausal, V5)")
    print("  2. Pour chaque variante :")
    print("     a. Évaluer sur ID (test historique récent)")
    print("     b. Évaluer sur OOD (SSP5-8.5 ou fallback)")
    print("  3. Calculer Δ_OOD = (Pearson_ID - Pearson_OOD) / Pearson_ID")
    print("  4. Comparer Δ_OOD(V5) vs Δ_OOD(noncausal)")
    print("  5. Sauvegarder via save_ood_results(...)")

    return 0


def save_ood_results(
    metrics_v5_id: Dict[str, Any],
    metrics_v5_ood: Dict[str, Any],
    metrics_noncausal_id: Dict[str, Any],
    metrics_noncausal_ood: Dict[str, Any],
    out_path: Path,
) -> None:
    """Sauvegarde les métriques OOD + tableau comparatif."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    delta_v5 = compute_ood_degradation(
        metrics_v5_id.get("pearson_global", float("nan")),
        metrics_v5_ood.get("pearson_global", float("nan")),
    )
    delta_noncausal = compute_ood_degradation(
        metrics_noncausal_id.get("pearson_global", float("nan")),
        metrics_noncausal_ood.get("pearson_global", float("nan")),
    )

    payload = {
        "v5": {
            "in_distribution": metrics_v5_id,
            "ood": metrics_v5_ood,
            "delta_ood": delta_v5,
        },
        "noncausal": {
            "in_distribution": metrics_noncausal_id,
            "ood": metrics_noncausal_ood,
            "delta_ood": delta_noncausal,
        },
        "verdict": {
            "delta_ood_v5": delta_v5,
            "delta_ood_noncausal": delta_noncausal,
            "v5_wins_ood": delta_v5 < delta_noncausal,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_path = out_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Évaluation OOD — Oracle V5 vs CorrDiff noncausal\n\n")
        f.write("| Métrique | V5 (ID) | V5 (OOD) | noncausal (ID) | noncausal (OOD) |\n")
        f.write("|---|---|---|---|---|\n")
        for k in ["pearson_global", "rmse", "mae", "pred_p50", "pred_p95", "pred_p99"]:
            f.write(
                f"| {k} | "
                f"{metrics_v5_id.get(k, '—')} | "
                f"{metrics_v5_ood.get(k, '—')} | "
                f"{metrics_noncausal_id.get(k, '—')} | "
                f"{metrics_noncausal_ood.get(k, '—')} |\n"
            )
        f.write("\n## Dégradation OOD\n\n")
        f.write(f"- **Δ_OOD (V5)**         : `{delta_v5:.4f}`\n")
        f.write(f"- **Δ_OOD (noncausal)**  : `{delta_noncausal:.4f}`\n")
        if delta_v5 < delta_noncausal:
            f.write("\n✓ **V5 résiste mieux à OOD** — hypothèse causale validée.\n")
        else:
            f.write("\n⚠ V5 ne montre pas d'avantage OOD net. À discuter (trilemme).\n")

    print(f"✓ Résultats OOD sauvegardés : {out_path}")
    print(f"  Δ_OOD (V5)         : {delta_v5:.4f}")
    print(f"  Δ_OOD (noncausal)  : {delta_noncausal:.4f}")


__all__ = [
    "compute_ood_degradation",
    "evaluate_dataset",
    "save_ood_results",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
