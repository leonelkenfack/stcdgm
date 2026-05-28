"""Validate the causal-vs-noncausal deliverables without a GPU.

1. All three notebooks parse as JSON.
2. The non-causal training notebook carries the right config + cache overrides.
3. Dry-run the comparison notebook's quant + qual logic on SYNTHETIC metrics
   JSON + eval_samples.npz, to catch logic bugs before Colab.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# --- 1. Notebooks parse ---
for nbname in [
    "st_cdgm_training_evaluation.ipynb",
    "st_cdgm_noncausal_training.ipynb",
    "st_cdgm_causal_vs_noncausal_comparison.ipynb",
]:
    nb = json.loads((ROOT / nbname).read_text(encoding="utf-8"))
    print(f"[OK] {nbname}: {len(nb['cells'])} cells")

# --- 2. Non-causal notebook overrides ---
nc = (ROOT / "st_cdgm_noncausal_training.ipynb").read_text(encoding="utf-8")
assert "training_config_noncausal.yaml" in nc, "noncausal config override missing"
assert "stage1_cache_noncausal.pt" in nc, "noncausal cache override missing"
assert "BS43 NON-CAUSAL TRAINING NOTEBOOK" in nc, "runbook comment not updated"
assert "train_epoch_stage1_noncausal" in nc, "true noncausal Stage 1 loop missing"
assert "BS35 causal DAG ablation skipped" in nc, "noncausal DAG ablation skip missing"
print("[OK] noncausal notebook: config + cache + comment overrides present")

# --- 3. Dry-run comparison logic on synthetic artifacts ---
with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
    cd = Path(td) / "ckpt_v2_corrdiff_normal"
    nd = Path(td) / "ckpt_noncausal"
    cd.mkdir(parents=True)
    nd.mkdir(parents=True)

    def fv(pearson, rmse, rapsd):
        return {
            "pearson_corr": {"global": pearson, "per_sample_avg": pearson + 0.01},
            "rmse": rmse, "mae": rmse * 0.5, "spread_mean": 0.04,
            "rapsd_distance": rapsd,
            "mu_HR_ablation": {"delta_signal_ratio_avg": 1.07},
        }

    def dm(crps, ssr, lhd, rapsd):
        return {"spread_skill_ratio": ssr, "crps_gaussian": crps,
                "intensity_hist_distance_L1": lhd, "rapsd_distance": rapsd}

    (cd / "final_validation_metrics.json").write_text(json.dumps(fv(0.815, 0.130, 258)))
    (cd / "domain_metrics.json").write_text(json.dumps(dm(0.049, 0.32, 0.071, 258)))
    (nd / "final_validation_metrics.json").write_text(json.dumps(fv(0.808, 0.133, 270)))
    (nd / "domain_metrics.json").write_text(json.dumps(dm(0.051, 0.35, 0.078, 270)))

    # Synthetic aligned cGAN metrics across GCMs (in-dist + OOD), per variant.
    # Make causal degrade LESS in OOD (smaller bias growth) to exercise the logic.
    def aligned(gcm, variant):
        ood = gcm != "ACCESS-CM2"
        deg = (0.5 if variant == "noncausal" else 0.2) if ood else 0.0  # noncausal worse OOD
        return {
            "gcm": gcm, "run_variant": variant, "in_distribution": not ood,
            "n_days": 1000, "k_samples": 16,
            "indices": {
                "cdd_bias": 0.10 + deg, "cdd_pred": 4.0, "cdd_truth": 3.9,
                "rx1day_bias": 0.5 + deg, "r10day_bias": 1.0 + deg,
                "DJF_rainfall_bias": 0.05 + deg, "JJA_rainfall_bias": 0.04 + deg,
            },
            "psd_distance": 0.02 + 0.5 * deg,
        }
    for d, rv in [(cd, "causal"), (nd, "noncausal")]:
        for gcm in ["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]:
            (d / f"aligned_metrics_{gcm}_{rv}.json").write_text(json.dumps(aligned(gcm, rv)))

    H, W = 32, 33
    for d, rv in [(cd, "causal"), (nd, "noncausal")]:
        np.savez_compressed(
            d / "eval_samples.npz",
            target=np.random.rand(4, 1, H, W).astype("float32"),
            pred_full=np.random.rand(4, 1, H, W).astype("float32"),
            pred_std=np.random.rand(4, 1, H, W).astype("float32") * 0.04,
            valid_mask=np.ones((4, 1, H, W), "float32"),
            mu_HR=np.random.rand(4, 1, H, W).astype("float32") * 0.05,
            run_variant=np.array(rv),
        )

    # Extract + run the comparison notebook's code cells in a shared namespace.
    nb = json.loads((ROOT / "st_cdgm_causal_vs_noncausal_comparison.ipynb").read_text(encoding="utf-8"))
    ns = {"CAUSAL_DIR": str(cd), "NONCAUSAL_DIR": str(nd)}
    import matplotlib
    matplotlib.use("Agg")  # headless: plt.show() is a no-op, no crash
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        exec(compile(src, f"<{cell['id']}>", "exec"), ns)
    print("[OK] comparison notebook: quant + qual cells executed on synthetic data")

print("\nALL VALIDATION CHECKS PASSED")
