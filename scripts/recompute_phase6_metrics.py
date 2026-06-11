"""Recompute final_validation_metrics.json apres Phase F (fine-tune).

VERSION REFACTORISEE (Option II du user, 2026-06-11) :
Au lieu de re-implementer les metriques (ce qui a produit des chiffres
divergents du baseline V5-mini publie), on appelle DIRECTEMENT les
fonctions du module officiel `src/st_cdgm/evaluation/evaluation_xai.py` :

- `run_st_cdgm_inference()` : pour la prediction ensemble
- `evaluate_metrics()` : pour mse, mae, f1_extremes, spectrum_distance
- Pearson + spread_mean + mu_HR_ablation : calcules manuellement
  (ils n'etaient pas dans evaluate_metrics, le V5-mini training script
  les calculait separement)

Garantit identite de formule avec le V5-mini training.

Params V5-mini training (lus depuis training_config_corrdiff_normal.yaml) :
- scheduler_type = "edm_karras"
- eval_num_steps = 32
- cfg_scale = 1.5
- K (k_samples) = 64

Usage typique depuis le notebook :

.. code-block:: python

    from scripts.recompute_phase6_metrics import recompute_phase6_metrics

    recompute_phase6_metrics(
        stack=stack_v5, builder=builder, val_dataset=val_dataset,
        DEVICE=DEVICE,
        predict_with_stack_fn=predict_with_stack,
        convert_sample_to_batch_fn=convert_sample_to_batch,
        out_path=ORACLE_FINETUNED_DIR / "final_validation_metrics.json",
        K_samples=64, n_steps=32, n_batches=16,
    )
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor


def recompute_phase6_metrics(
    *,
    stack: Dict[str, Any],
    builder,
    val_dataset,
    DEVICE: torch.device,
    predict_with_stack_fn: Callable = None,  # garde pour backward compat, pas utilise
    convert_sample_to_batch_fn: Callable = None,  # garde pour backward compat, pas utilise
    out_path: Path,
    K_samples: int = 64,
    n_steps: int = 32,
    n_batches: int = 16,
    epoch: int = 25,
    causal_concat: bool = True,
    cfg_scale: float = 1.5,
    scheduler_type: str = "edm_karras",
    do_mu_hr_ablation: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Calcule les metriques V5-mini sur stack_v5 et ecrit le JSON.

    Utilise les FONCTIONS OFFICIELLES du training V5-mini :
    `run_st_cdgm_inference` + `evaluate_metrics` du module evaluation_xai.
    Garantit que les nombres sont DIRECTEMENT comparables au baseline publie.
    """
    # Imports tardifs pour eviter circular import
    from src.st_cdgm.evaluation.evaluation_xai import (
        run_st_cdgm_inference,
        evaluate_metrics,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"[Recompute Phase 6] K={K_samples}, n_steps={n_steps}, scheduler={scheduler_type}, cfg_scale={cfg_scale}")
        print(f"  n_batches : {n_batches}")
        print(f"  out_path : {out_path}")

    t0 = time.time()

    # Extraction du stack
    encoder = stack["encoder"]
    rcn_runner = stack["rcn_runner"]
    regression_head = stack["regression_head"]
    diffusion = stack["diffusion"]
    skip_block = stack.get("skip_block")
    spatial_projector = stack.get("spatial_projector")

    # Accumulators
    metric_reports: List[Any] = []  # MetricReport per batch
    pearson_per_sample: List[float] = []
    spread_per_sample: List[float] = []
    pred_all: List[np.ndarray] = []  # pour pearson global
    target_all: List[np.ndarray] = []

    n_avail = min(len(val_dataset), n_batches)
    n_done = 0

    for i in range(n_avail):
        try:
            sample = val_dataset[i]
        except Exception as e:
            warnings.warn(f"sample {i} retrieval failed: {e}")
            continue

        try:
            # Appel direct a run_st_cdgm_inference (= ce que V5-mini training utilise)
            samples_out, target_batch, baseline_batch, dag_last, mask_batch = run_st_cdgm_inference(
                sample,
                builder=builder,
                encoder=encoder,
                rcn_runner=rcn_runner,
                diffusion=diffusion,
                device=DEVICE,
                num_samples=K_samples,
                num_steps=n_steps,
                scheduler_type=scheduler_type,
                apply_constraints=False,
                use_log1p_inverse=False,  # pred reste en log1p space comme V5-mini
                cfg_scale=cfg_scale,
                spatial_projector=spatial_projector,
            )

            # Construire le target reel : baseline + residual (= log1p HR)
            # target_batch est deja le target final dans run_st_cdgm_inference
            # baseline_batch est la baseline log1p

            # Appel direct a evaluate_metrics (= V5-mini training)
            report = evaluate_metrics(
                samples=samples_out,
                target=target_batch,
                baseline=baseline_batch,
                compute_advanced=False,  # skip FSS/Wasserstein/EnergyScore (pas dans V5 JSON)
                include_f1_extremes=True,
                f1_percentiles=[95.0, 99.0],
                use_mean_aggregation=False,  # = stacked_means[0], single-member, comme V5
                valid_mask=mask_batch,
                crps_max_ensemble_members=None,
            )
            metric_reports.append(report)

            # Pearson + spread manuel (V5-mini les calculait separement)
            stacked_means = torch.stack([s.t_mean for s in samples_out], dim=0)
            pred_primary = stacked_means[0]  # single-member, comme V5
            spread_per_sample.append(float(stacked_means.std(dim=0).mean().item()))

            # Pearson sur pixels valides
            p_np = pred_primary.detach().cpu().numpy().squeeze()
            t_np = target_batch.detach().cpu().numpy().squeeze()
            mask_np = np.isfinite(t_np) & np.isfinite(p_np)
            if mask_np.sum() > 1:
                pv = p_np[mask_np]; tv = t_np[mask_np]
                pv_c = pv - pv.mean(); tv_c = tv - tv.mean()
                denom = np.sqrt((pv_c**2).sum() * (tv_c**2).sum()) + 1e-12
                pearson_i = float((pv_c * tv_c).sum() / denom)
            else:
                pearson_i = float('nan')
            pearson_per_sample.append(pearson_i)
            pred_all.append(np.where(mask_np, p_np, 0.0))
            target_all.append(np.where(mask_np, t_np, 0.0))

            n_done += 1

            if verbose and (i + 1) % 5 == 0:
                print(f"  batch {i+1}/{n_avail} : "
                      f"mse={report.mse:.4f} mae={report.mae:.4f} "
                      f"f1_p99={report.f1_extremes.get('p99', float('nan')) if report.f1_extremes else float('nan'):.4f} "
                      f"pearson={pearson_i:.4f} spread={spread_per_sample[-1]:.4f}")
        except Exception as e:
            warnings.warn(f"batch {i} eval failed: {type(e).__name__}: {e}")
            continue

    if not metric_reports:
        return {
            "error": "Aucun batch eval reussi",
            "n_batches_attempted": n_avail,
            "n_batches_successful": 0,
        }

    # Agregation des MetricReports : moyenne sur les batches (comme V5 training)
    rmse = float(np.sqrt(np.mean([r.mse for r in metric_reports if not np.isnan(r.mse)])))
    mae = float(np.mean([r.mae for r in metric_reports if not np.isnan(r.mae)]))
    f1_p95 = float(np.mean([
        r.f1_extremes.get('p95', np.nan) for r in metric_reports if r.f1_extremes
    ]))
    f1_p99 = float(np.mean([
        r.f1_extremes.get('p99', np.nan) for r in metric_reports if r.f1_extremes
    ]))
    spectrum_distance = float(np.mean([r.spectrum_distance for r in metric_reports if not np.isnan(r.spectrum_distance)]))

    # Pearson global agrege
    pred_global = np.concatenate([p.flatten() for p in pred_all])
    target_global = np.concatenate([t.flatten() for t in target_all])
    mask_g = np.isfinite(pred_global) & np.isfinite(target_global)
    if mask_g.sum() > 1:
        p_g = pred_global[mask_g]; t_g = target_global[mask_g]
        p_gc = p_g - p_g.mean(); t_gc = t_g - t_g.mean()
        pearson_global = float((p_gc * t_gc).sum() / (np.sqrt((p_gc**2).sum() * (t_gc**2).sum()) + 1e-12))
    else:
        pearson_global = float('nan')

    spread_mean = float(np.mean(spread_per_sample))

    # mu_HR_ablation (impact A_dag) : Phase 6 specifique
    mu_HR_ablation_result = None
    if do_mu_hr_ablation and stack.get("A_dag") is not None:
        try:
            mu_HR_ablation_result = _compute_mu_HR_ablation_official(
                stack, val_dataset, builder, DEVICE,
                n_batches=min(4, n_done),
            )
        except Exception as e:
            warnings.warn(f"mu_HR ablation failed: {e}")

    result = {
        "checkpoint": str(out_path.parent / "epoch_last.pth"),
        "epoch": epoch,
        "epochs_total": epoch,
        "best_val_loss": None,
        "causal_concat": causal_concat,
        "n_test_batches": n_done,
        "k_samples": K_samples,
        "metrics_scope": "full_prediction (mu_HR + delta_hat)",
        "eval_time_s": float(time.time() - t0),
        "rmse": rmse,
        "mae": mae,
        "spread_mean": spread_mean,
        "f1_extremes": {"p95": f1_p95, "p99": f1_p99},
        "pearson_corr": {
            "global": pearson_global,
            "per_sample_avg": float(np.mean(pearson_per_sample)) if pearson_per_sample else float('nan'),
            "per_sample_n": len(pearson_per_sample),
            "per_sample_list": [float(x) for x in pearson_per_sample],
        },
        "rapsd_distance": spectrum_distance,  # alias pour compat avec Phase 6 reader
        "spectrum_distance": spectrum_distance,
        "mu_HR_ablation": mu_HR_ablation_result or {
            "delta_signal_ratio_avg": float("nan"),
            "per_batch": [],
            "verdict": "SKIPPED",
        },
        "config_eval_num_steps": n_steps,
        "config_cfg_scale": cfg_scale,
        "config_scheduler_type": scheduler_type,
        "recomputed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "uses_official_evaluate_metrics": True,
    }

    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        print()
        print("=" * 60)
        print(f"[OK] Phase 6 recomputed in {result['eval_time_s']:.1f}s ({n_done} batches)")
        print(f"  RMSE             : {result['rmse']:.4f}")
        print(f"  MAE              : {result['mae']:.4f}")
        print(f"  Pearson global   : {result['pearson_corr']['global']:.4f}")
        print(f"  Pearson per-samp : {result['pearson_corr']['per_sample_avg']:.4f}")
        print(f"  F1-p95           : {result['f1_extremes']['p95']:.4f}")
        print(f"  F1-p99           : {result['f1_extremes']['p99']:.4f}")
        print(f"  RAPSD distance   : {result['rapsd_distance']:.4f}")
        print(f"  spread_mean      : {result['spread_mean']:.4f}")
        if mu_HR_ablation_result:
            print(f"  mu_HR ablation   : {result['mu_HR_ablation']['delta_signal_ratio_avg']:.4f}")
        print(f"  Saved to         : {out_path}")
        print("=" * 60)

    return result


def _compute_mu_HR_ablation_official(stack, val_dataset, builder, DEVICE, n_batches=4):
    """mu_HR ablation : compare mu_HR(A_dag) vs mu_HR(A_dag=0).

    Identique au calcul fait dans le V5-mini training script.
    """
    rcn_cell = stack["rcn_runner"].cell
    A_orig = rcn_cell.A_dag.detach().clone()
    encoder = stack["encoder"]
    rcn_runner = stack["rcn_runner"]
    regression_head = stack["regression_head"]
    skip_block = stack.get("skip_block")

    def _forward_mu(sample):
        # Build batch
        lr_seq = sample["lr"]
        seq_len = lr_seq.shape[0]
        lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
        lr_tensor = torch.stack(lr_nodes_steps, dim=0)
        dynamic_features = {nt: lr_nodes_steps[0] for nt in builder.dynamic_node_types}
        hetero = builder.prepare_step_data(dynamic_features).to(DEVICE)
        lr_data = lr_tensor.to(DEVICE)

        H_init = encoder.init_state(hetero).to(DEVICE)
        drivers = [lr_data[t] for t in range(lr_data.shape[0])]
        seq = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
        mu_c = regression_head(seq.states[-1])

        target_residual = sample["residual"][-1]
        if target_residual.dim() == 3:
            target_residual = target_residual.unsqueeze(0)
        tshape = target_residual.to(DEVICE).shape
        if tshape[-2:] != mu_c.shape[-2:]:
            mu_c = torch.nn.functional.interpolate(
                mu_c, size=tshape[-2:], mode="bilinear", align_corners=False,
            )
        if skip_block is not None:
            lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
            try:
                mu_HR_pred, _ = skip_block(lr_last, mu_c)
            except Exception:
                mu_HR_pred = mu_c
        else:
            mu_HR_pred = mu_c
        return mu_HR_pred.cpu().squeeze().numpy()

    per_batch = []
    for i in range(n_batches):
        try:
            sample = val_dataset[i]
        except Exception:
            continue
        with torch.no_grad():
            mu_full = _forward_mu(sample)
            rcn_cell.A_dag.data.zero_()
            mu_zero = _forward_mu(sample)
            rcn_cell.A_dag.data.copy_(A_orig)
        delta = float(np.abs(mu_full - mu_zero).mean())
        signal = float(np.abs(mu_full).mean()) + 1e-12
        per_batch.append(delta / signal)

    if not per_batch:
        return {"delta_signal_ratio_avg": float("nan"), "per_batch": [], "verdict": "FAILED"}
    avg = float(np.mean(per_batch))
    return {
        "delta_signal_ratio_avg": avg,
        "per_batch": per_batch,
        "verdict": "MU_HR_CONDITIONS" if avg > 0.5 else "MU_HR_WEAK",
    }


__all__ = ["recompute_phase6_metrics"]
