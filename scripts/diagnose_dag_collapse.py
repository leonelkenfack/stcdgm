"""
Diagnostic post-hoc : lit un (ou plusieurs) checkpoints two-stage,
extrait ``A_dag`` depuis ``rcn_cell_state_dict``, et calcule des
métriques de santé du DAG ainsi qu'un ratio O3 approximatif quand un
batch d'évaluation est fourni.

Usage
-----
    # Inspection d'un seul checkpoint
    python scripts/diagnose_dag_collapse.py path/to/epoch_last.pth

    # Trajectoire sur plusieurs epochs (CSV)
    python scripts/diagnose_dag_collapse.py path/to/ckpt_dir --trajectory

    # Inclure le calcul du ratio O3 (nécessite un dataset chargeable)
    python scripts/diagnose_dag_collapse.py path/to/epoch_last.pth --o3 --config config/training_config.yaml

Sorties
-------
- ``||A||_F``, ``max|A|``, ``||A||_1``, sparsité (% < 1e-3)
- Distance au prior physique (MSE + corrélation Pearson)
- Top-3 valeurs propres de ``A⊙A`` (mesure spectrale DAGMA)
- Ratio O3 ``mean(|μ(A) − μ(0)|) / mean(|μ(A)|)`` si ``--o3`` fourni

Aucune écriture de fichier sauf en mode ``--trajectory`` (CSV).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

import torch


def _extract_a_dag(ckpt_path: Path) -> torch.Tensor:
    """Charge un checkpoint et retourne A_dag masquée (diagonale = 0)."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    rcn_state = ck.get("rcn_cell_state_dict") or ck.get("rcn_state_dict")
    if rcn_state is None:
        raise RuntimeError(
            f"Aucune clé rcn_cell_state_dict dans {ckpt_path}. "
            f"Clés disponibles : {list(ck.keys())[:10]}"
        )
    if "A_dag" not in rcn_state:
        candidates = [k for k in rcn_state if "A_dag" in k]
        if not candidates:
            raise RuntimeError(
                f"A_dag absent du state_dict. Clés : {list(rcn_state.keys())[:20]}"
            )
        a = rcn_state[candidates[0]]
    else:
        a = rcn_state["A_dag"]
    a = a.float()
    a = a - torch.diag(torch.diagonal(a))
    return a


def _load_dag_prior(config_path: Optional[Path]) -> Optional[torch.Tensor]:
    if config_path is None or not config_path.exists():
        return None
    try:
        import yaml
    except ImportError:
        print("⚠️  PyYAML non disponible — ignore le prior.")
        return None
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    prior = cfg.get("loss", {}).get("dag_prior")
    if prior is None:
        return None
    return torch.tensor(prior, dtype=torch.float32)


def _format_health(a: torch.Tensor, prior: Optional[torch.Tensor] = None) -> dict:
    """Métriques scalaires sur A_dag."""
    a_abs = a.abs()
    eig = torch.linalg.eigvals(a * a).real.abs().sort(descending=True).values
    rep = {
        "shape": tuple(a.shape),
        "norm_F": float(a.norm().item()),
        "max_abs": float(a_abs.max().item()),
        "sum_abs (L1)": float(a_abs.sum().item()),
        "sparsity_lt_1e-3": float((a_abs < 1e-3).float().mean().item()),
        "mean_abs": float(a_abs.mean().item()),
        "spectral_radius_AoA": float(eig[0].item()),
        "top3_eig_AoA": [float(x) for x in eig[:3].tolist()],
    }
    if prior is not None and prior.shape == a.shape:
        diff = (a - prior)
        rep["MSE_to_prior"] = float((diff ** 2).mean().item())
        flat_a = a.reshape(-1)
        flat_p = prior.reshape(-1)
        if flat_a.std() > 1e-9 and flat_p.std() > 1e-9:
            corr = ((flat_a - flat_a.mean()) * (flat_p - flat_p.mean())).mean() / (
                flat_a.std() * flat_p.std() + 1e-12
            )
            rep["corr_with_prior"] = float(corr.item())
    return rep


def _print_report(label: str, rep: dict) -> None:
    print(f"=== {label} ===")
    for k, v in rep.items():
        if isinstance(v, list):
            v_fmt = ", ".join(f"{x:.4f}" for x in v)
            print(f"  {k:24s} = [{v_fmt}]")
        elif isinstance(v, tuple):
            print(f"  {k:24s} = {v}")
        elif isinstance(v, float):
            print(f"  {k:24s} = {v:.6f}")
        else:
            print(f"  {k:24s} = {v}")
    a_norm = rep.get("norm_F", 0.0)
    if a_norm < 0.01:
        print("  🚨 COLLAPSED  (||A||_F < 0.01) — DAG décoratif, O3 gate va échouer.")
    elif a_norm < 0.1:
        print("  ⚠️  AT-RISK    (||A||_F < 0.1) — surveiller la trajectoire.")
    else:
        print("  ✅ HEALTHY    (||A||_F ≥ 0.1)")
    print()


def _o3_ratio(
    ckpt_path: Path,
    config_path: Path,
    n_samples: int = 50,
) -> Optional[dict]:
    """Calcule le ratio O3 sur N batches de val. Best-effort : retourne
    ``None`` si la pipeline ne peut pas être instanciée hors Colab."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        import yaml
        from st_cdgm.models.causal_rcn import RCNCell, RCNSequenceRunner
        from st_cdgm.models.regression_head import GraphToGridDecoder
        from st_cdgm.models.intelligible_encoder import (
            IntelligibleVariableEncoder,
            IntelligibleVariableConfig,
        )
    except Exception as e:
        print(f"⚠️  Impossible d'importer les modules pour O3 : {e}")
        return None

    print("ℹ️  Calcul O3 hors-ligne non implémenté (nécessite dataset + builder).")
    print("    Utiliser ``causal_ablation_check`` dans la cellule TWO_STAGE_TRAINING_LOOP.")
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Diagnostic du collapse DAG")
    ap.add_argument("path", type=Path, help="Checkpoint .pth ou répertoire")
    ap.add_argument("--config", type=Path, default=Path("config/training_config.yaml"))
    ap.add_argument(
        "--trajectory",
        action="store_true",
        help="Trace ||A||_F par epoch (CSV) si --path est un répertoire",
    )
    ap.add_argument("--o3", action="store_true", help="Calcule aussi le ratio O3 (best-effort)")
    args = ap.parse_args()

    prior = _load_dag_prior(args.config)
    if prior is not None:
        print(f"📐 Prior chargé depuis {args.config} (shape {tuple(prior.shape)})\n")

    if args.path.is_dir():
        ckpts: List[Path] = sorted(args.path.glob("epoch_*.pth"))
        if not ckpts:
            ckpts = sorted(args.path.glob("*.pth"))
        if not ckpts:
            print(f"Aucun checkpoint dans {args.path}")
            return 1
    else:
        ckpts = [args.path]

    rows = []
    for ck in ckpts:
        try:
            a = _extract_a_dag(ck)
        except Exception as e:
            print(f"⚠️  {ck.name} : {e}")
            continue
        rep = _format_health(a, prior=prior)
        rep["ckpt"] = ck.name
        _print_report(ck.name, rep)
        rows.append(rep)

    if args.trajectory and rows:
        out_csv = args.path / "dag_trajectory.csv" if args.path.is_dir() else Path("dag_trajectory.csv")
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ckpt", "norm_F", "max_abs", "L1", "sparsity",
                             "MSE_to_prior", "corr_with_prior"])
            for r in rows:
                writer.writerow([
                    r["ckpt"], r["norm_F"], r["max_abs"], r["sum_abs (L1)"],
                    r["sparsity_lt_1e-3"],
                    r.get("MSE_to_prior", ""), r.get("corr_with_prior", ""),
                ])
        print(f"📊 Trajectoire écrite : {out_csv}")

    if args.o3 and ckpts:
        _o3_ratio(ckpts[-1], args.config)

    return 0


if __name__ == "__main__":
    sys.exit(main())
