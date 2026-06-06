"""Patch Phase 7 (Cell 8) : conserve l'ensemble complet, calcule CRPS / RMSE /
spread / CRPS-SS / rank histogram, ecrit un sidecar probabilistic_metrics_<GCM>_<variant>.json
sans toucher au vendored run_aligned_eval. Bump K=4 -> K=12.
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NB = Path("st_cdgm_v5_evaluation.ipynb")
with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak10")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #10 cree : {BACKUP}")

NEW_CELL = '''# ============================================================
# Phase 7 : runs OOD reels avec run_aligned_eval (vendored Rampal)
# + sidecar probabilistic_metrics : CRPS, RMSE, spread, CRPS-SS, rank histogram
# Sortie 1 : aligned_metrics_<GCM>_<variant>.json   (indices climatiques)
# Sortie 2 : probabilistic_metrics_<GCM>_<variant>.json (ensemble-based)
# ============================================================
import json
import time
import torch
import numpy as np
from st_cdgm.evaluation.aligned_eval import run_aligned_eval

# Parametres
N_TIMES_OOD = 365
K_SAMPLES_OOD = 12          # bump 4 -> 12 pour CRPS empirique non bruite
N_STEPS_DIFF = 18

# --- Metriques probabilistes ---------------------------------------------

def _crps_empirical_fast(samples, obs):
    """CRPS empirique vectorise via tri (O(K log K) par point).

    samples : (K, ...) array, ensemble
    obs     : (...,) array, observation
    return  : (...,) array, CRPS par point
    Formule : E|X - y| - 0.5 * E|X - X'|, avec
              0.5 * (1/K^2) * sum_ij |xi - xj| = (1/K^2) * sum_k (2k - K - 1) * x_(k)
    """
    K = samples.shape[0]
    term1 = np.nanmean(np.abs(samples - obs[None]), axis=0)
    s = np.sort(samples, axis=0)
    k_idx = np.arange(1, K + 1).reshape((K,) + (1,) * (s.ndim - 1)).astype(np.float64)
    weights = 2.0 * k_idx - K - 1.0
    term2 = np.sum(weights * s, axis=0) / (K * K)
    return term1 - term2


def _crps_clim_per_pixel(truth):
    """CRPS de la climato empirique (distribution par pixel sur l'axe temps).

    Pour X, X' iid ~ distribution-truth(h,w) et y ~ idem :
        CRPS_clim(h,w) = E|X - y| - 0.5 * E|X - X'| = 0.5 * E|X - X'|
    (car E|X-Y| = E|X-X'| pour des copies iid).
    """
    T = truth.shape[0]
    t_sorted = np.sort(truth, axis=0)
    k_idx = np.arange(1, T + 1).reshape((T, 1, 1)).astype(np.float64)
    weights = 2.0 * k_idx - T - 1.0
    return np.sum(weights * t_sorted, axis=0) / (T * T)


def _rank_histogram(samples, truth):
    """Histogramme de Talagrand : rang de truth parmi les K samples (K+1 bins)."""
    K = samples.shape[0]
    rank = (samples < truth[None]).sum(axis=0).astype(np.int64)
    hist, _ = np.histogram(rank.flatten(), bins=np.arange(K + 2) - 0.5)
    return hist.astype(int).tolist()


def probabilistic_metrics(ens_log1p, truth_log1p):
    """Calcule toutes les metriques probabilistes apres conversion log1p -> mm/jour.

    ens_log1p   : (K, T, H, W) ensemble en espace log1p
    truth_log1p : (T, H, W) verite en espace log1p
    """
    ens = np.expm1(np.clip(ens_log1p.astype(np.float64), 0.0, None))
    truth = np.expm1(np.clip(truth_log1p.astype(np.float64), 0.0, None))

    pred_mean = ens.mean(axis=0)                         # (T, H, W)
    err2 = (pred_mean - truth) ** 2
    rmse_global = float(np.sqrt(np.nanmean(err2)))
    rmse_map_t = np.sqrt(np.nanmean(err2, axis=0))       # (H, W)

    ens_var = ens.var(axis=0)                            # (T, H, W)
    spread_global = float(np.sqrt(np.nanmean(ens_var)))
    spread_skill_ratio = float(spread_global / max(rmse_global, 1e-9))

    crps_model = _crps_empirical_fast(ens, truth)        # (T, H, W)
    crps_model_global = float(np.nanmean(crps_model))

    crps_clim_map = _crps_clim_per_pixel(truth)          # (H, W)
    crps_clim_global = float(np.nanmean(crps_clim_map))

    crps_ss = 1.0 - crps_model_global / max(crps_clim_global, 1e-9)

    hist = _rank_histogram(ens, truth)
    K = int(ens.shape[0])
    expected_per_bin = float(truth.size / (K + 1))
    chi2_uniform = float(sum((c - expected_per_bin) ** 2 / expected_per_bin for c in hist))

    return {
        "K_samples": K,
        "n_times": int(ens.shape[1]),
        "grid": [int(truth.shape[-2]), int(truth.shape[-1])],
        "rmse_global_mm": rmse_global,
        "rmse_map_mean_mm": float(np.nanmean(rmse_map_t)),
        "rmse_map_max_mm": float(np.nanmax(rmse_map_t)),
        "spread_global_mm": spread_global,
        "spread_skill_ratio": spread_skill_ratio,
        "crps_model_global_mm": crps_model_global,
        "crps_clim_global_mm": crps_clim_global,
        "crps_skill_score": float(crps_ss),
        "rank_histogram": hist,
        "rank_histogram_bins": list(range(len(hist))),
        "rank_histogram_chi2_vs_uniform": chi2_uniform,
        "_caveat": (
            "CRPS_clim computed from test-truth empirical distribution per pixel "
            "(includes the day under evaluation; slight optimistic bias for T~365). "
            "spread_skill_ratio ~1 = well-calibrated, <1 = under-dispersive, >1 = over-dispersive."
        ),
    }


# --- Collection ensemble complet ----------------------------------------

def collect_predictions_for_gcm(stack, gcm_tag, n_times=N_TIMES_OOD, K=K_SAMPLES_OOD):
    """Genere ensemble (K, T, H, W) + truth (T, H, W) + times."""
    lr_path, hr_path, in_dist = GCM_REGISTRY[gcm_tag]
    pipe = make_pipeline(lr_path, hr_path)
    ds = pipe.build_sequence_dataset(
        seq_len=int(CONFIG.data.seq_len), stride=1, as_torch=True,
    )

    ens_log, truths_log, times_list = [], [], []
    it = iter(ds)
    for i in range(n_times):
        try:
            sample = next(it)
        except StopIteration:
            break
        batch = convert_sample_to_batch(sample, builder, DEVICE)
        ens = predict_with_stack(stack, batch, K=K, n_steps=N_STEPS_DIFF)  # (K, B=1, 1, H, W)
        # Reduit a (K, H, W) en droppant B et C
        while ens.dim() > 3:
            ens = ens.squeeze(1)
        ens_arr = ens.cpu().numpy()                      # (K, H, W)
        truth = (batch["baseline"][-1] + batch["residual"][-1]).cpu()
        truth_arr = truth.squeeze().numpy()              # (H, W)

        ens_log.append(ens_arr)
        truths_log.append(truth_arr)
        times_list.append(np.datetime64("1986-01-01") + np.timedelta64(i, "D"))

        if (i + 1) % 50 == 0:
            print(f"  [{gcm_tag} / {stack['variant']}] {i+1}/{n_times} pas evalues")

    ens_full = np.stack(ens_log, axis=1)                 # (K, T, H, W)
    truths = np.stack(truths_log, axis=0)                # (T, H, W)
    preds_mean = ens_full.mean(axis=0)                   # (T, H, W) pour run_aligned_eval
    times = np.array(times_list, dtype="datetime64[D]")
    return preds_mean, truths, times, ens_full


# --- Boucle principale --------------------------------------------------

print("=" * 70)
print(f"Phase 7 : 6 runs OOD ({N_TIMES_OOD} pas x K={K_SAMPLES_OOD} x {N_STEPS_DIFF} EDM steps)")
print(f"         + sidecar probabilistic_metrics par run")
print("=" * 70)
print()

all_results = {}      # aligned (Rampal)
all_prob = {}         # probabilistic (sidecar)
t_global = time.time()

for stack_name, stack, ckpt_dir in [("V5", stack_v5, V5_DIR),
                                      ("Noncausal", stack_nc, NONCAUSAL_DIR)]:
    for gcm_tag in ["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]:
        run_label = f"{stack_name}_{gcm_tag}"
        t0 = time.time()
        print(f"\\n=== RUN {run_label} ===")

        if not GCM_REGISTRY[gcm_tag][0].exists():
            print(f"  [SKIP] {GCM_REGISTRY[gcm_tag][0]} absent")
            continue

        try:
            preds, truths, times, ens_full = collect_predictions_for_gcm(stack, gcm_tag)
        except Exception as e:
            print(f"  [ERREUR] {type(e).__name__}: {e}")
            continue

        if len(preds) == 0:
            print(f"  [SKIP] Pas de samples retournes")
            continue

        # 1. Vendored Rampal indices
        out_path = ckpt_dir / f"aligned_metrics_{gcm_tag}_{stack_name.lower()}.json"
        try:
            run_aligned_eval(
                pred_fields=preds, truth_fields=truths, times=times,
                out_path=str(out_path),
                gcm=gcm_tag, run_variant=stack_name.lower(),
                in_distribution=GCM_REGISTRY[gcm_tag][2],
                space="log1p", k_samples=K_SAMPLES_OOD,
            )
            with open(out_path, "r", encoding="utf-8") as f:
                all_results[run_label] = json.load(f)
        except Exception as e:
            print(f"  [ERREUR run_aligned_eval] {type(e).__name__}: {e}")

        # 2. Sidecar probabiliste
        prob_path = ckpt_dir / f"probabilistic_metrics_{gcm_tag}_{stack_name.lower()}.json"
        try:
            prob = probabilistic_metrics(ens_full, truths)
            prob.update({
                "gcm": gcm_tag,
                "run_variant": stack_name.lower(),
                "in_distribution": GCM_REGISTRY[gcm_tag][2],
            })
            prob_path.write_text(json.dumps(prob, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            all_prob[run_label] = prob
            print(f"  [OK] aligned + probabilistic ({(time.time()-t0):.1f}s)")
            print(f"       CRPS={prob['crps_model_global_mm']:.3f}mm  "
                  f"CRPS-SS={prob['crps_skill_score']:+.3f}  "
                  f"RMSE={prob['rmse_global_mm']:.3f}mm  "
                  f"spread/skill={prob['spread_skill_ratio']:.3f}")
        except Exception as e:
            print(f"  [ERREUR probabilistic_metrics] {type(e).__name__}: {e}")

print()
print(f"[OK] Phase 7 terminee en {(time.time()-t_global)/60:.1f} min")

# --- Recapitulatif ------------------------------------------------------

print()
print("=" * 70)
print("RECAPITULATIF Delta_OOD (indices Rampal + probabiliste)")
print("=" * 70)
for variant in ["V5", "Noncausal"]:
    print(f"\\n--- {variant} ---")
    in_dist = all_results.get(f"{variant}_ACCESS-CM2", {})
    in_prob = all_prob.get(f"{variant}_ACCESS-CM2", {})
    if in_dist:
        psd_in = in_dist.get("psd_distance")
        crps_in = in_prob.get("crps_model_global_mm")
        crpsss_in = in_prob.get("crps_skill_score")
        if psd_in is not None:
            print(f"  ID  ACCESS-CM2  PSD={psd_in:.4f}  CRPS={crps_in:.3f}mm  CRPS-SS={crpsss_in:+.3f}")
    for ood in ["EC-Earth3", "NorESM2-MM"]:
        r = all_results.get(f"{variant}_{ood}", {})
        p = all_prob.get(f"{variant}_{ood}", {})
        if r and p:
            print(f"  OOD {ood:<12s} PSD={r.get('psd_distance'):.4f}  "
                  f"CRPS={p.get('crps_model_global_mm'):.3f}mm  "
                  f"CRPS-SS={p.get('crps_skill_score'):+.3f}  "
                  f"spread/skill={p.get('spread_skill_ratio'):.3f}")

# --- Sauvegarde recap ---------------------------------------------------

recap_path = RESULTS_DIR / "phase7_ood_aligned.json"
recap_path.write_text(json.dumps({
    "n_times": N_TIMES_OOD, "K_samples": K_SAMPLES_OOD, "n_steps": N_STEPS_DIFF,
    "runs": all_results,
    "probabilistic": all_prob,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(f"\\n[OK] Recap sauvegarde : {recap_path}")'''

nb["cells"][8]["source"] = [ln + "\n" for ln in NEW_CELL.split("\n")[:-1]] + [NEW_CELL.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Cell 8 reecrit : K={12}, sidecar probabilistic_metrics, CRPS/RMSE/spread/rank-histo")
print(f"     {len(nb['cells'])} cellules")
