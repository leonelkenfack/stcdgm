"""Compare Phase 6-12 evaluation outputs between V5-mini baseline and post-finetune (Phase G).

Lit les JSONs d'évaluation (aligned_metrics, probabilistic_metrics, phase8/11/12)
des deux checkpoints (V5-mini baseline + post-Phase F fine-tune) et produit un
rapport markdown avec les delta par métrique, marqueurs de gain/régression, et
récapitulatif final.

Usage
-----
.. code-block:: bash

    python -m scripts.compare_eval_results \\
        --baseline-dir /content/drive/.../ckpt_v2_corrdiff_normal_baseline \\
        --finetuned-dir /content/drive/.../ckpt_v2_corrdiff_normal_finetuned \\
        --results-baseline /content/drive/.../results/v5_evaluation_baseline \\
        --results-finetuned /content/drive/.../results/v5_evaluation_finetuned \\
        --out results/v5_evaluation/comparison_finetuned_vs_v5mini.md

Si certains JSONs sont absents (ex: Phase 8 a été skippée d'un côté), le script
saute la section correspondante avec un avertissement.

References
----------
- architecture_journey.md §11.1 (chiffres baseline V5-mini)
- architecture_journey.md §12.8 (cibles d'amélioration post-Phase F)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------
# Définition des métriques à comparer (référence §11.1, §12.8)
# --------------------------------------------------------------

# (json_file_pattern, json_path, label, direction, target_post_phase_f)
# direction: "down" = lower is better, "up" = higher is better
METRIC_SPECS: List[Tuple[str, str, str, str, Optional[float]]] = [
    # --- Phase 6 (in-distribution) ---
    ("final_validation_metrics.json", "rmse", "RMSE log1p (ID)", "down", 0.115),
    ("final_validation_metrics.json", "mae", "MAE log1p (ID)", "down", 0.055),
    ("final_validation_metrics.json", "pearson_corr.global", "Pearson global (ID)", "up", 0.845),
    ("final_validation_metrics.json", "rapsd_distance", "RAPSD distance (ID)", "down", 215.0),
    ("final_validation_metrics.json", "f1_extremes.p95", "F1-p95 (ID)", "up", 0.685),
    ("final_validation_metrics.json", "f1_extremes.p99", "F1-p99 (ID)", "up", 0.620),
    ("final_validation_metrics.json", "mu_HR_ablation.delta_signal_ratio_avg", "μ_HR ablation Δ/signal", "up", 0.890),

    # --- Phase 7 ID (ACCESS-CM2) aligned + probabilistic ---
    ("aligned_metrics_ACCESS-CM2_v5.json", "indices.rx1day_bias", "RX1day bias ID (mm/jour)", "down_abs", -2.0),
    ("aligned_metrics_ACCESS-CM2_v5.json", "indices.cdd_bias", "CDD bias ID (jours)", "down_abs", None),
    ("aligned_metrics_ACCESS-CM2_v5.json", "indices.r10day_bias", "R10mm bias ID (jours)", "down_abs", None),
    ("aligned_metrics_ACCESS-CM2_v5.json", "psd_distance", "PSD distance ID", "down", 0.25),
    ("probabilistic_metrics_ACCESS-CM2_v5.json", "crps_model_global_mm", "CRPS ID (mm/jour)", "down", 0.325),
    ("probabilistic_metrics_ACCESS-CM2_v5.json", "crps_skill_score", "CRPS-SS ID", "up", 0.88),
    ("probabilistic_metrics_ACCESS-CM2_v5.json", "spread_skill_ratio", "spread/skill ID", "up", 0.30),
    ("probabilistic_metrics_ACCESS-CM2_v5.json", "rmse_global_mm", "RMSE ID (mm/jour)", "down", 1.15),

    # --- Phase 7 OOD EC-Earth3 ---
    ("aligned_metrics_EC-Earth3_v5.json", "indices.rx1day_bias", "RX1day bias OOD-EC (mm/jour)", "down_abs", -3.0),
    ("aligned_metrics_EC-Earth3_v5.json", "psd_distance", "PSD distance OOD-EC", "down", 0.29),
    ("probabilistic_metrics_EC-Earth3_v5.json", "crps_model_global_mm", "CRPS OOD-EC (mm/jour)", "down", 0.37),
    ("probabilistic_metrics_EC-Earth3_v5.json", "crps_skill_score", "CRPS-SS OOD-EC", "up", 0.83),

    # --- Phase 7 OOD NorESM2-MM ---
    ("aligned_metrics_NorESM2-MM_v5.json", "indices.rx1day_bias", "RX1day bias OOD-NorESM (mm/jour)", "down_abs", -2.5),
    ("aligned_metrics_NorESM2-MM_v5.json", "psd_distance", "PSD distance OOD-NorESM", "down", 0.28),
    ("probabilistic_metrics_NorESM2-MM_v5.json", "crps_model_global_mm", "CRPS OOD-NorESM (mm/jour)", "down", 0.38),
    ("probabilistic_metrics_NorESM2-MM_v5.json", "crps_skill_score", "CRPS-SS OOD-NorESM", "up", 0.85),

    # --- Phase 8 (interpretability) ---
    ("phase8_interpretability.json", "Q_int.V5", "Q_int (ORACLE, /2)", "up", 1.0),
    ("phase8_interpretability.json", "ablation_A_dag_delta_signal_ratio", "Ablation A_dag Δ/signal", "up", 0.10),
    ("phase8_interpretability.json", "sensitivities.V5.q_850", "Sensibilité q_850 (ORACLE)", "up", 0.18),
    ("phase8_interpretability.json", "sensitivities.V5.t_850", "Sensibilité t_850 (ORACLE)", "up", None),
    ("phase8_interpretability.json", "sensitivities.V5.w_850", "Sensibilité w_850 (ORACLE)", "up", None),

    # --- Phase 11 (causal advanced) ---
    ("causal_advanced_results.json", "Q_phys", "Q_phys (DAG vs G_phys)", "up", 0.70),
    ("causal_advanced_results.json", "n_correct_sign", "Nb arêtes signe correct", "up", 4),
]


# --------------------------------------------------------------
# Utils
# --------------------------------------------------------------


def _nested_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Récupère une valeur depuis un dict imbriqué via 'a.b.c'."""
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _find_json(roots: List[Path], filename: str) -> Optional[Path]:
    """Cherche le fichier JSON dans les roots fournis (récursif limité à 3 niveaux)."""
    for root in roots:
        if not root.exists():
            continue
        # Cherche directement
        candidate = root / filename
        if candidate.exists():
            return candidate
        # Cherche dans les sous-dossiers
        for level1 in root.iterdir():
            if level1.is_dir():
                candidate = level1 / filename
                if candidate.exists():
                    return candidate
                for level2 in level1.iterdir():
                    if level2.is_dir():
                        candidate = level2 / filename
                        if candidate.exists():
                            return candidate
    return None


def _format_value(v: Any, decimals: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int,)):
        return str(v)
    try:
        return f"{float(v):.{decimals}f}"
    except (TypeError, ValueError):
        return str(v)


def _direction_check(
    baseline: Optional[float], finetuned: Optional[float], direction: str
) -> Tuple[str, Optional[float]]:
    """Returns (status_emoji, delta_relative_pct).

    status: ✓ = improved, ✗ = degraded, = = no significant change, ? = N/A.
    """
    if baseline is None or finetuned is None:
        return "?", None
    if direction == "down":
        delta = finetuned - baseline
        rel = (delta / abs(baseline)) * 100 if abs(baseline) > 1e-9 else 0.0
        if delta < -abs(baseline) * 0.01:
            return "✓", rel
        elif delta > abs(baseline) * 0.01:
            return "✗", rel
        return "=", rel
    elif direction == "up":
        delta = finetuned - baseline
        rel = (delta / abs(baseline)) * 100 if abs(baseline) > 1e-9 else 0.0
        if delta > abs(baseline) * 0.01:
            return "✓", rel
        elif delta < -abs(baseline) * 0.01:
            return "✗", rel
        return "=", rel
    elif direction == "down_abs":
        # Magnitude doit diminuer (e.g. bias rapproché de 0)
        delta = abs(finetuned) - abs(baseline)
        rel = (delta / abs(baseline)) * 100 if abs(baseline) > 1e-9 else 0.0
        if delta < -abs(baseline) * 0.01:
            return "✓", rel
        elif delta > abs(baseline) * 0.01:
            return "✗", rel
        return "=", rel
    return "?", None


# --------------------------------------------------------------
# Main comparison logic
# --------------------------------------------------------------


def build_comparison_report(
    baseline_roots: List[Path],
    finetuned_roots: List[Path],
    out_path: Path,
) -> Dict[str, Any]:
    """Construit le rapport markdown et le sauvegarde."""

    lines: List[str] = [
        "# Comparaison post-Phase F vs baseline V5-mini",
        "",
        "**Date** : Phase G — éval post-fine-tune Bundle B + CASTLE + G_phys",
        "**Source** : compare_eval_results.py",
        "",
        "Comparaison métrique par métrique entre le checkpoint V5-mini baseline ",
        "(figé en §10.8 de architecture_journey.md) et le checkpoint post-fine-tune ",
        "(Phase F, branche `two-stage-causal`).",
        "",
        "Légende :",
        "- **✓** : amélioration significative (> 1 % en magnitude relative)",
        "- **✗** : régression significative",
        "- **=** : changement négligeable",
        "- **?** : métrique manquante d'un côté",
        "",
        "## Récapitulatif par phase",
        "",
        "| Métrique | Baseline V5-mini | Post-finetune | Δ relatif (%) | Direction | Verdict | Cible §12.8 |",
        "|---|---|---|---|---|---|---|",
    ]

    n_improved = 0
    n_degraded = 0
    n_neutral = 0
    n_missing = 0
    summary_per_phase: Dict[str, Dict[str, int]] = {}
    raw_data: List[Dict[str, Any]] = []

    for filename, key_path, label, direction, target in METRIC_SPECS:
        baseline_path = _find_json(baseline_roots, filename)
        finetuned_path = _find_json(finetuned_roots, filename)

        b_val = f_val = None
        if baseline_path:
            try:
                b_data = json.loads(baseline_path.read_text(encoding="utf-8"))
                b_val = _nested_get(b_data, key_path)
            except Exception:
                pass
        if finetuned_path:
            try:
                f_data = json.loads(finetuned_path.read_text(encoding="utf-8"))
                f_val = _nested_get(f_data, key_path)
            except Exception:
                pass

        try:
            b_val = float(b_val) if b_val is not None else None
        except (TypeError, ValueError):
            b_val = None
        try:
            f_val = float(f_val) if f_val is not None else None
        except (TypeError, ValueError):
            f_val = None

        status, rel = _direction_check(b_val, f_val, direction)

        if status == "✓":
            n_improved += 1
        elif status == "✗":
            n_degraded += 1
        elif status == "=":
            n_neutral += 1
        else:
            n_missing += 1

        # Phase tag
        if "ACCESS-CM2" in filename or "EC-Earth3" in filename or "NorESM2" in filename:
            phase_tag = "P7"
        elif "phase8" in filename:
            phase_tag = "P8"
        elif "causal_advanced" in filename:
            phase_tag = "P11"
        elif "final_validation" in filename:
            phase_tag = "P6"
        else:
            phase_tag = "??"
        summary_per_phase.setdefault(phase_tag, {"+": 0, "-": 0, "=": 0, "?": 0})
        if status == "✓":
            summary_per_phase[phase_tag]["+"] += 1
        elif status == "✗":
            summary_per_phase[phase_tag]["-"] += 1
        elif status == "=":
            summary_per_phase[phase_tag]["="] += 1
        else:
            summary_per_phase[phase_tag]["?"] += 1

        rel_str = f"{rel:+.2f}" if rel is not None else "—"
        target_str = _format_value(target)
        lines.append(
            f"| {label} | {_format_value(b_val)} | {_format_value(f_val)} | {rel_str} | {direction} | {status} | {target_str} |"
        )
        raw_data.append({
            "label": label, "baseline": b_val, "finetuned": f_val,
            "direction": direction, "status": status, "delta_rel_pct": rel,
            "target_phase_f": target, "phase": phase_tag,
        })

    lines += [
        "",
        "## Bilan global",
        "",
        f"- **Améliorations** (✓) : {n_improved}",
        f"- **Régressions** (✗) : {n_degraded}",
        f"- **Neutres** (=) : {n_neutral}",
        f"- **Manquantes** (?) : {n_missing}",
        "",
        "## Détail par phase",
        "",
        "| Phase | ✓ | ✗ | = | ? |",
        "|---|---|---|---|---|",
    ]
    for phase, counts in sorted(summary_per_phase.items()):
        lines.append(f"| {phase} | {counts['+']} | {counts['-']} | {counts['=']} | {counts['?']} |")

    lines += [
        "",
        "## Interprétation",
        "",
        "Si **≥ 70 %** des métriques montrent une amélioration et **≤ 10 %** une ",
        "régression, Phase F est validée et on peut passer à la rédaction du mémoire.",
        "",
        "Si **régression sur Pearson global ID** : signe que le fine-tune a sur-spécialisé ",
        "sur les extrêmes au détriment de la moyenne. Réduire `lambda_pinball` à 0.10 et ",
        "ré-essayer 10 epochs supplémentaires.",
        "",
        "Si **Q_phys n'a pas progressé** : signe que CASTLE + masque physique n'ont pas ",
        "produit l'effet attendu. Vérifier `lambda_dag_prior` (essayer 0.10) et l'init du ",
        "`A_dag` (peut être trop loin du masque physique pour converger).",
        "",
        "Si **CRPS-SS dégrade** : signe que `sigma_data` n'a pas été correctement recalibrée. ",
        "Re-run `calibrate_sigma_data_variant` avec `max_samples=1000`.",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")

    # Aussi un JSON pour analyses programmatiques ultérieures
    json_out = out_path.with_suffix(".json")
    json_out.write_text(
        json.dumps({
            "summary": {
                "improved": n_improved, "degraded": n_degraded,
                "neutral": n_neutral, "missing": n_missing,
            },
            "per_phase": summary_per_phase,
            "rows": raw_data,
        }, indent=2, default=str),
        encoding="utf-8",
    )

    return {
        "improved": n_improved, "degraded": n_degraded,
        "neutral": n_neutral, "missing": n_missing,
        "out_md": str(out_path), "out_json": str(json_out),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, action="append", default=[],
                        help="Répertoire(s) contenant les JSONs baseline V5-mini. "
                              "Peut être passé plusieurs fois.")
    parser.add_argument("--finetuned-dir", type=Path, action="append", default=[],
                        help="Répertoire(s) contenant les JSONs post-finetune. "
                              "Peut être passé plusieurs fois.")
    parser.add_argument("--results-baseline", type=Path, action="append", default=[],
                        help="Dossiers results/v5_evaluation pour le baseline.")
    parser.add_argument("--results-finetuned", type=Path, action="append", default=[],
                        help="Dossiers results/v5_evaluation pour le post-finetune.")
    parser.add_argument("--out", type=Path, default=Path("results/v5_evaluation/comparison_finetuned_vs_v5mini.md"),
                        help="Chemin du rapport markdown de sortie.")
    args = parser.parse_args()

    baseline_roots = args.baseline_dir + args.results_baseline
    finetuned_roots = args.finetuned_dir + args.results_finetuned

    if not baseline_roots:
        print("[ERREUR] Pas de --baseline-dir / --results-baseline spécifié")
        sys.exit(1)
    if not finetuned_roots:
        print("[ERREUR] Pas de --finetuned-dir / --results-finetuned spécifié")
        sys.exit(1)

    print(f"Baseline roots : {[str(r) for r in baseline_roots]}")
    print(f"Finetuned roots : {[str(r) for r in finetuned_roots]}")

    result = build_comparison_report(baseline_roots, finetuned_roots, args.out)

    print()
    print("=" * 60)
    print(f"Comparaison écrite : {result['out_md']}")
    print(f"JSON brut         : {result['out_json']}")
    print()
    print(f"Améliorations  : {result['improved']}")
    print(f"Régressions    : {result['degraded']}")
    print(f"Neutres        : {result['neutral']}")
    print(f"Manquantes     : {result['missing']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
