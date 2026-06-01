"""Phase 8 — Protocole d'intervention do(·) pour démontrer la capacité unique d'Oracle V5.

Trois interventions physiques standardisées sont appliquées aux drivers
basse résolution d'un échantillon, et la réponse du modèle est confrontée
au signe et à la magnitude *physiquement attendus*.

Les trois interventions
-----------------------
1. ``do(q_850 += 20%)`` — augmentation de l'humidité de basse couche.
   *Effet attendu* : la précipitation locale doit ↑ (atmosphère plus humide).

2. ``do(T_850 += 3°C)`` — réchauffement de la troposphère basse.
   *Effet attendu* : la précipitation devrait ↑ globalement (Clausius-Clapeyron,
   atmosphère plus chargée en vapeur saturante) ; effet local complexe sur les
   contrastes (intensification convective).

3. ``do(psl -= 10 hPa)`` — chute de pression au niveau de la mer.
   *Effet attendu* : système dépressionnaire → précipitation ↑ (advection
   humide convergente).

Métrique de qualité d'intervention
----------------------------------
Pour chaque intervention :

.. math::

    Q_{\\mathrm{int}}(v) = \\mathrm{sign}(\\overline{\\Delta\\mathrm{pred}})
                     \\stackrel{?}{=} \\mathrm{sign}_{\\mathrm{phys}}(v)

Q_int global = fraction des interventions où le signe correspond
(``Q_int ∈ [0, 1]``).

Capacité unique
---------------
- ``CorrDiff noncausal`` peut techniquement répondre (on lui injecte les
  drivers modifiés), mais sa réponse est **corrélative** (selon les
  co-occurrences vues à l'entraînement), donc peut être incohérente avec
  l'attendu physique.

- ``Oracle V5`` répond **structurellement** : l'intervention est propagée
  à travers les arêtes du DAG ``q_850 → ... → pr_HR``, et la réponse est
  par construction cohérente avec la structure causale apprise.

Usage
-----
.. code-block:: bash

    python -m scripts.intervention_test \\
        --ckpt-noncausal ckpt_noncausal/epoch_last.pth \\
        --ckpt-v5       ckpt_v5/epoch_last.pth \\
        --out           results/v5_intervention.json \\
        --n-samples     8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch


# ---------------------------------------------------------------------------
# Tableau des interventions standardisées
# ---------------------------------------------------------------------------

INTERVENTIONS: List[Dict[str, Any]] = [
    {
        "name": "do_humidity_plus_20pct",
        "variable_idx": None,         # à résoudre depuis CONFIG.data.lr_variables
        "variable_name": "q_850",
        "delta_type": "multiplicative",
        "delta_value": 1.20,           # ×1.20 sur l'humidité 850 hPa
        "expected_sign": +1,           # précipitation doit ↑
        "physical_justification": (
            "Humidité de basse couche ↑ → vapeur condensable disponible ↑ → "
            "précipitation locale ↑ (loi de Clausius-Clapeyron, processus "
            "convectif et stratiforme)."
        ),
    },
    {
        "name": "do_temp_plus_3K",
        "variable_idx": None,
        "variable_name": "t_850",
        "delta_type": "additive_celsius",
        "delta_value": +3.0,           # K
        "expected_sign": +1,           # globalement la précip ↑
        "physical_justification": (
            "Température troposphère basse ↑3K → capacité de vapeur ↑ (~21%) "
            "→ intensification précipitation, surtout extrêmes (Trenberth 2003)."
        ),
    },
    {
        "name": "do_psl_minus_10hPa",
        "variable_idx": None,
        "variable_name": "psl",        # peut ne pas être dans lr_variables
        "delta_type": "additive_hPa",
        "delta_value": -10.0,          # hPa
        "expected_sign": +1,
        "physical_justification": (
            "Pression au niveau de la mer ↓10 hPa → système dépressionnaire "
            "→ advection humide convergente → précipitation ↑."
        ),
    },
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def resolve_variable_indices(lr_variables: List[str]) -> List[Dict[str, Any]]:
    """Résout les indices des variables dans le tenseur LR.

    Adapte les interventions à l'ordre effectif des variables dans la
    config (``CONFIG.data.lr_variables``). Les variables absentes du LR
    sont marquées ``variable_idx=None`` et seront skipées.
    """
    resolved = []
    for spec in INTERVENTIONS:
        copy = dict(spec)
        try:
            copy["variable_idx"] = lr_variables.index(spec["variable_name"])
        except ValueError:
            copy["variable_idx"] = None
        resolved.append(copy)
    return resolved


def apply_intervention(
    lr_batch: torch.Tensor,
    spec: Dict[str, Any],
    *,
    standardization: Dict[str, Dict[str, float]] | None = None,
) -> torch.Tensor:
    """Applique une intervention sur un tenseur LR ``[T, C, H, W]`` ou ``[B, T, C, H, W]``.

    Cas du tenseur déjà standardisé : on doit dé-standardiser, appliquer
    l'intervention en unités physiques, puis re-standardiser. Le dictionnaire
    ``standardization`` fournit ``{variable_name: {"mean": ..., "std": ...}}``.

    Si ``standardization`` est ``None``, l'intervention est appliquée
    directement dans l'espace standardisé (moins physique mais utilisable
    pour des tests qualitatifs).
    """
    if spec["variable_idx"] is None:
        return lr_batch.clone()  # variable absente du LR

    out = lr_batch.clone()
    idx = spec["variable_idx"]

    # Cible : le canal idx du tenseur LR.
    # Tensor shape :
    #   [T, C, H, W]   → lr_batch[:, idx, :, :]
    #   [B, T, C, H, W] → lr_batch[:, :, idx, :, :]
    if out.dim() == 4:
        channel = out[:, idx, :, :]
    elif out.dim() == 5:
        channel = out[:, :, idx, :, :]
    else:
        raise ValueError(f"Unsupported LR shape: {tuple(out.shape)}")

    # Si standardisation fournie, dé-standardiser
    mean = std = None
    if standardization is not None and spec["variable_name"] in standardization:
        s = standardization[spec["variable_name"]]
        mean, std = float(s["mean"]), float(s["std"])
        channel_phys = channel * std + mean
    else:
        channel_phys = channel

    # Appliquer
    if spec["delta_type"] == "multiplicative":
        channel_phys = channel_phys * spec["delta_value"]
    elif spec["delta_type"] in ("additive_celsius", "additive_hPa", "additive"):
        channel_phys = channel_phys + spec["delta_value"]
    else:
        raise ValueError(f"Unknown delta_type: {spec['delta_type']}")

    # Re-standardiser si nécessaire
    if mean is not None and std is not None:
        channel_new = (channel_phys - mean) / max(std, 1e-12)
    else:
        channel_new = channel_phys

    if out.dim() == 4:
        out[:, idx, :, :] = channel_new
    else:
        out[:, :, idx, :, :] = channel_new

    return out


def evaluate_intervention(
    model_predict,   # callable batch -> ensemble [K, B, 1, H, W]
    batch_normal: Dict[str, Any],
    spec: Dict[str, Any],
    standardization: Dict[str, Dict[str, float]] | None = None,
) -> Dict[str, Any]:
    """Évalue une intervention sur un échantillon.

    Returns
    -------
    dict avec :
    - delta_pred_mean : moyenne spatiale Δ_pred
    - delta_pred_max  : max spatial Δ_pred
    - sign_predicted  : signe de delta_pred_mean
    - sign_expected   : signe attendu physiquement
    - match           : bool sign_predicted == sign_expected
    """
    if spec["variable_idx"] is None:
        return {
            "intervention": spec["name"],
            "variable": spec["variable_name"],
            "skipped": True,
            "reason": f"variable {spec['variable_name']} not in LR",
        }

    # Prédiction sans intervention
    pred_normal = model_predict(batch_normal)
    pred_normal_mean = pred_normal.nanmean(dim=0)  # ensemble mean → [B, 1, H, W]

    # Batch intervenue
    batch_int = {k: v for k, v in batch_normal.items()}
    batch_int["lr"] = apply_intervention(batch_normal["lr"], spec, standardization=standardization)

    pred_int = model_predict(batch_int)
    pred_int_mean = pred_int.nanmean(dim=0)

    delta = pred_int_mean - pred_normal_mean
    delta_mean = float(delta.nanmean())
    delta_max = float(delta.abs().nanmean())  # magnitude moyenne
    sign_pred = int(1 if delta_mean > 0 else (-1 if delta_mean < 0 else 0))

    return {
        "intervention": spec["name"],
        "variable": spec["variable_name"],
        "delta_pred_mean": delta_mean,
        "delta_pred_magnitude": delta_max,
        "sign_predicted": sign_pred,
        "sign_expected": int(spec["expected_sign"]),
        "match": sign_pred == int(spec["expected_sign"]),
        "physical_justification": spec["physical_justification"],
        "skipped": False,
    }


def compute_q_int(intervention_results: List[Dict[str, Any]]) -> float:
    """Q_int = fraction des interventions où le signe correspond à l'attendu."""
    used = [r for r in intervention_results if not r.get("skipped")]
    if not used:
        return float("nan")
    return sum(1 for r in used if r["match"]) / len(used)


def save_intervention_results(
    results_v5: List[Dict[str, Any]],
    results_noncausal: List[Dict[str, Any]],
    out_path: Path,
) -> None:
    """Sauvegarde JSON + Markdown."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    q_v5 = compute_q_int(results_v5)
    q_nc = compute_q_int(results_noncausal)

    payload = {
        "v5": {"interventions": results_v5, "Q_int": q_v5},
        "noncausal": {"interventions": results_noncausal, "Q_int": q_nc},
        "verdict": {
            "Q_int_v5": q_v5,
            "Q_int_noncausal": q_nc,
            "v5_wins_intervention": q_v5 > q_nc,
        },
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    md_path = out_path.with_suffix(".md")
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Phase 8 — Protocole d'intervention\n\n")
        f.write("## Résultats détaillés\n\n")
        for label, res in [("V5", results_v5), ("noncausal", results_noncausal)]:
            f.write(f"### {label}\n\n")
            f.write("| Intervention | Δ_pred mean | sign pred | sign attendu | match |\n")
            f.write("|---|---|---|---|---|\n")
            for r in res:
                if r.get("skipped"):
                    f.write(f"| {r['intervention']} | skipped | — | — | — |\n")
                else:
                    f.write(
                        f"| {r['intervention']} | "
                        f"{r['delta_pred_mean']:.4f} | "
                        f"{r['sign_predicted']:+d} | "
                        f"{r['sign_expected']:+d} | "
                        f"{'✓' if r['match'] else '✗'} |\n"
                    )
            f.write("\n")
        f.write("\n## Q_int\n\n")
        f.write(f"- V5         : `Q_int = {q_v5:.3f}`\n")
        f.write(f"- noncausal  : `Q_int = {q_nc:.3f}`\n")
        if q_v5 > q_nc:
            f.write("\n✓ Oracle V5 démontre une **meilleure cohérence causale**.\n")
        else:
            f.write("\n⚠ Pas de gain net sur Q_int. Argument à nuancer (capacité corrélative noncausal suffit).\n")

    print(f"✓ Résultats intervention sauvegardés : {out_path}")
    print(f"  Q_int (V5)         : {q_v5:.3f}")
    print(f"  Q_int (noncausal)  : {q_nc:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ckpt-noncausal", type=Path, required=True)
    parser.add_argument("--ckpt-v5", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=8,
                        help="Number of stochastic samples per intervention")
    parser.add_argument("--n-batches", type=int, default=4,
                        help="Number of test batches to average over")
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 8 — Protocole d'intervention do(·)")
    print("=" * 70)
    for spec in INTERVENTIONS:
        print(f"  • {spec['name']}: {spec['variable_name']} "
              f"{spec['delta_type']} {spec['delta_value']} "
              f"(attendu : sign = {spec['expected_sign']:+d})")
    print()

    print("⚠ Script à invoquer depuis un notebook ayant CONFIG + checkpoints.")
    print()
    print("Workflow :")
    print("  1. Charger CONFIG (pour lr_variables order)")
    print("  2. resolve = resolve_variable_indices(CONFIG.data.lr_variables)")
    print("  3. Pour chaque variant (V5, noncausal) et chaque batch :")
    print("       results = [evaluate_intervention(model.predict, batch, spec, std)")
    print("                  for spec in resolve]")
    print("  4. save_intervention_results(results_v5, results_noncausal, out_path)")

    return 0


__all__ = [
    "INTERVENTIONS",
    "resolve_variable_indices",
    "apply_intervention",
    "evaluate_intervention",
    "compute_q_int",
    "save_intervention_results",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
