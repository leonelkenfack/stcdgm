"""Patch : ajoute Phase 9 (climate stds), Phase 10 (extreme bias maps),
Phase 11 (DAG physique + PSE), Phase 12 (Integrated Gradients).
Amende aussi Phase 7 (Cell 8) pour stocker phase7_arrays en memoire.
8 nouvelles cellules inserees avant la synthese finale.
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak11")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #11 cree : {BACKUP}")

# ============================================================
# Step 1 : Amend Cell 8 (Phase 7) to populate phase7_arrays
# ============================================================
cell8_src = "".join(nb["cells"][8]["source"])

# Add phase7_arrays init after all_prob declaration
OLD_INIT = "all_prob = {}         # probabilistic (sidecar)"
NEW_INIT = ("all_prob = {}         # probabilistic (sidecar)\n"
            "phase7_arrays = {}    # arrays in memory pour Phases 9-12")
if OLD_INIT in cell8_src and "phase7_arrays = {}" not in cell8_src:
    cell8_src = cell8_src.replace(OLD_INIT, NEW_INIT)

# Add array storage just before the [OK] aligned + probabilistic print
OLD_PRINT = '            all_prob[run_label] = prob\n            print(f"  [OK] aligned + probabilistic'
NEW_BLOCK = ('            all_prob[run_label] = prob\n'
             '            phase7_arrays[run_label] = {\n'
             '                "pred_mean": preds.astype("float32"),\n'
             '                "truth": truths.astype("float32"),\n'
             '                "ens": ens_full.astype("float32"),\n'
             '                "times": times,\n'
             '            }\n'
             '            print(f"  [OK] aligned + probabilistic')
if OLD_PRINT in cell8_src and 'phase7_arrays[run_label] = {' not in cell8_src:
    cell8_src = cell8_src.replace(OLD_PRINT, NEW_BLOCK)

nb["cells"][8]["source"] = [l + "\n" for l in cell8_src.split("\n")[:-1]] + [cell8_src.split("\n")[-1]]
print("[OK] Cell 8 amende : phase7_arrays peuple dans la boucle")

# ============================================================
# Step 2 : Build 8 new cells (4 MD + 4 code)
# ============================================================

CELL_PHASE9_MD = """---

## Phase 9 — Diagnostics climate-standard

Quatre figures attendues par la communaute climate downscaling (CorrDiff, Rampal cGAN, IPCC AR6 Ch. 11) :

1. **Q-Q plot tail** par (variant × GCM) sur jours pluvieux — fidelite distributionnelle dans la queue
2. **Return-period Gumbel** sur max domain-wide journaliers — extrapolation des extremes
3. **Reliability diagram** pour P(precip > {1, 10, 50} mm) — calibration probabiliste
4. **Fractions Skill Score (FSS)** vs taille de voisinage — echelle a laquelle le modele devient skillful

Lit `phase7_arrays` (rempli par Phase 7).
"""

CELL_PHASE9_CODE = '''# Phase 9 — Diagnostics climate-standard
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

PHASE9_DIR = RESULTS_DIR / "phase9_climate_standards"
PHASE9_DIR.mkdir(parents=True, exist_ok=True)

if "phase7_arrays" not in dir() or not phase7_arrays:
    print("[ERREUR] phase7_arrays non disponible — relance Phase 7 d'abord.")
else:
    print(f"phase7_arrays : {len(phase7_arrays)} runs disponibles")

    # === 1. Q-Q plot tail par (variant, GCM) ===========================
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    quantiles = np.linspace(0.5, 0.9999, 200)
    for r, variant in enumerate(["V5", "Noncausal"]):
        for c, gcm in enumerate(["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]):
            ax = axes[r, c]
            run_label = f"{variant}_{gcm}"
            if run_label not in phase7_arrays:
                ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{variant} / {gcm}"); ax.axis("off"); continue
            d = phase7_arrays[run_label]
            pred_mm = np.expm1(np.clip(d["pred_mean"].astype(np.float64), 0, None))
            truth_mm = np.expm1(np.clip(d["truth"].astype(np.float64), 0, None))
            pred_wet = pred_mm[pred_mm > 1.0]; truth_wet = truth_mm[truth_mm > 1.0]
            if pred_wet.size == 0 or truth_wet.size == 0:
                ax.text(0.5, 0.5, "(no wet)", ha="center"); continue
            qp = np.quantile(pred_wet, quantiles); qt = np.quantile(truth_wet, quantiles)
            ax.loglog(qt, qp, marker=".", linestyle="", color="steelblue", alpha=0.5, markersize=4)
            lo = max(min(qt.min(), qp.min()), 0.1); hi = max(qt.max(), qp.max())
            ax.loglog([lo, hi], [lo, hi], "k--", linewidth=1, label="1:1")
            id_tag = "ID" if gcm == "ACCESS-CM2" else "OOD"
            ax.set_title(f"{variant} / {gcm} ({id_tag})", fontsize=10)
            ax.set_xlabel("Truth quantile (mm/jour)")
            ax.set_ylabel("Pred quantile (mm/jour)")
            ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=7)
    plt.suptitle("Q-Q plots queue des intensites (jours pluvieux >1mm)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(PHASE9_DIR / "01_qq_tail.png", dpi=120, bbox_inches="tight"); plt.close()
    print("[OK] Figure 1 : Q-Q tail")

    # === 2. Return-period Gumbel sur max domain-wide ====================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for c, gcm in enumerate(["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]):
        ax = axes[c]
        for variant, color in [("V5", "steelblue"), ("Noncausal", "orange")]:
            run_label = f"{variant}_{gcm}"
            if run_label not in phase7_arrays: continue
            d = phase7_arrays[run_label]
            pred_mm = np.expm1(np.clip(d["pred_mean"].astype(np.float64), 0, None))
            pred_max_t = np.nanmax(pred_mm.reshape(pred_mm.shape[0], -1), axis=1)
            ps = np.sort(pred_max_t); n = len(ps)
            F = np.arange(1, n + 1) / (n + 1)
            rp = -np.log(-np.log(F))
            ax.plot(rp, ps, color=color, marker=".", linestyle="-",
                     label=f"{variant} pred", alpha=0.8, markersize=3)
        d_truth = phase7_arrays.get(f"V5_{gcm}") or phase7_arrays.get(f"Noncausal_{gcm}")
        if d_truth is not None:
            truth_mm = np.expm1(np.clip(d_truth["truth"].astype(np.float64), 0, None))
            tmax = np.nanmax(truth_mm.reshape(truth_mm.shape[0], -1), axis=1)
            ts = np.sort(tmax); F2 = np.arange(1, len(ts) + 1) / (len(ts) + 1)
            rp2 = -np.log(-np.log(F2))
            ax.plot(rp2, ts, color="black", marker=".", linestyle="-",
                     label="Truth", linewidth=2, markersize=3)
        id_tag = "ID" if gcm == "ACCESS-CM2" else "OOD"
        ax.set_xlabel("Gumbel reduced variate")
        ax.set_ylabel("Max precip domain (mm/jour)")
        ax.set_title(f"{gcm} ({id_tag})", fontsize=10)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    plt.suptitle("Return-period (Gumbel) — daily domain-max precip", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(PHASE9_DIR / "02_return_period.png", dpi=120, bbox_inches="tight"); plt.close()
    print("[OK] Figure 2 : Return-period Gumbel")

    # === 3. Reliability diagrams (calibration) ==========================
    try:
        from sklearn.calibration import calibration_curve
    except ImportError:
        calibration_curve = None
        print("[WARN] sklearn non installe — reliability skippe")

    if calibration_curve is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for c, thr in enumerate([1.0, 10.0, 50.0]):
            ax = axes[c]
            for variant, color in [("V5", "steelblue"), ("Noncausal", "orange")]:
                obs_all, pred_all = [], []
                for gcm in ["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]:
                    run_label = f"{variant}_{gcm}"
                    if run_label not in phase7_arrays: continue
                    d = phase7_arrays[run_label]
                    ens_mm = np.expm1(np.clip(d["ens"].astype(np.float64), 0, None))
                    truth_mm = np.expm1(np.clip(d["truth"].astype(np.float64), 0, None))
                    fp = (ens_mm > thr).mean(axis=0).flatten()
                    obs = (truth_mm > thr).astype(int).flatten()
                    rng = np.random.default_rng(42)
                    if fp.size > 500_000:
                        idx = rng.choice(fp.size, 500_000, replace=False)
                        fp = fp[idx]; obs = obs[idx]
                    obs_all.append(obs); pred_all.append(fp)
                if not obs_all: continue
                obs_arr = np.concatenate(obs_all); pred_arr = np.concatenate(pred_all)
                if obs_arr.sum() < 10:
                    continue  # trop peu d'evenements positifs
                try:
                    fp_, mp_ = calibration_curve(obs_arr, pred_arr, n_bins=10, strategy="uniform")
                    ax.plot(mp_, fp_, marker="o", color=color, label=variant, linewidth=2)
                except Exception as e:
                    print(f"  [WARN] calib_curve thr={thr} {variant}: {e}")
            ax.plot([0, 1], [0, 1], "k--", label="perfect")
            ax.set_xlabel("Forecast probability")
            ax.set_ylabel("Observed frequency")
            ax.set_title(f"P(precip > {thr:.0f} mm/jour)")
            ax.legend(); ax.grid(alpha=0.3)
        plt.suptitle("Reliability diagrams (wet-day calibration)", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(PHASE9_DIR / "03_reliability.png", dpi=120, bbox_inches="tight"); plt.close()
        print("[OK] Figure 3 : Reliability")

    # === 4. Fractions Skill Score vs scale ==============================
    try:
        from scipy.ndimage import uniform_filter
    except ImportError:
        uniform_filter = None

    if uniform_filter is not None:
        def _fss(p_bin, t_bin, n):
            Pf = uniform_filter(p_bin.astype(float), size=n)
            Tf = uniform_filter(t_bin.astype(float), size=n)
            num = ((Pf - Tf) ** 2).mean()
            denom = (Pf ** 2).mean() + (Tf ** 2).mean()
            return 1.0 - num / max(denom, 1e-12)

        scales = [1, 3, 5, 11, 21, 41]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for c, thr in enumerate([1.0, 10.0, 50.0]):
            ax = axes[c]
            for variant, color in [("V5", "steelblue"), ("Noncausal", "orange")]:
                fss_by_scale = []
                for n in scales:
                    vals = []
                    for gcm in ["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]:
                        run_label = f"{variant}_{gcm}"
                        if run_label not in phase7_arrays: continue
                        d = phase7_arrays[run_label]
                        pred_mm = np.expm1(np.clip(d["pred_mean"].astype(np.float64), 0, None))
                        truth_mm = np.expm1(np.clip(d["truth"].astype(np.float64), 0, None))
                        rng = np.random.default_rng(42)
                        n_days = pred_mm.shape[0]
                        idx = rng.choice(n_days, min(30, n_days), replace=False)
                        for t in idx:
                            v = _fss((pred_mm[t] > thr), (truth_mm[t] > thr), n)
                            if np.isfinite(v): vals.append(v)
                    fss_by_scale.append(np.mean(vals) if vals else np.nan)
                ax.plot(scales, fss_by_scale, marker="o", color=color, label=variant)
            ax.set_xlabel("Neighborhood window (pixels)")
            ax.set_ylabel("FSS")
            ax.set_title(f"thr = {thr:.0f} mm/jour")
            ax.legend(); ax.grid(alpha=0.3)
            ax.set_ylim(0, 1)
        plt.suptitle("Fractions Skill Score vs scale", fontsize=12, y=1.01)
        plt.tight_layout()
        plt.savefig(PHASE9_DIR / "04_fss_vs_scale.png", dpi=120, bbox_inches="tight"); plt.close()
        print("[OK] Figure 4 : FSS vs scale")

    print(f"\\n[OK] Phase 9 — figures dans {PHASE9_DIR}")
'''

CELL_PHASE10_MD = """---

## Phase 10 — Cartes spatiales des biais d'indices d'extremes

Quatre cartes 2D par GCM × variant : **RX1day, CDD, R95p, R10mm** (definitions ETCCDI/WMO).

Standard IPCC AR6 Ch. 11. La Phase 7 livre seulement les moyennes globales — ici on visualise la *structure spatiale* du biais.
"""

CELL_PHASE10_CODE = '''# Phase 10 — Cartes spatiales des biais d'indices d'extremes
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

PHASE10_DIR = RESULTS_DIR / "phase10_extreme_bias_maps"
PHASE10_DIR.mkdir(parents=True, exist_ok=True)

if "phase7_arrays" not in dir() or not phase7_arrays:
    print("[ERREUR] phase7_arrays non disponible.")
else:
    def _indices_per_pixel(pr):
        """pr: (T, H, W) en mm/jour. Returns dict de (H, W) maps."""
        T = pr.shape[0]
        rx1day = np.nanmax(pr, axis=0)
        r10mm = (pr >= 10.0).sum(axis=0).astype(np.float32)
        wet = pr[pr >= 1.0]
        thr95 = float(np.percentile(wet, 95)) if wet.size > 0 else 0.0
        r95p = np.where(pr >= thr95, pr, 0.0).sum(axis=0).astype(np.float32)
        dry = (pr < 1.0).astype(np.int8)
        cdd = np.zeros((pr.shape[1], pr.shape[2]), dtype=np.float32)
        run = np.zeros_like(cdd)
        for t in range(T):
            run = np.where(dry[t] == 1, run + 1, 0)
            cdd = np.maximum(cdd, run)
        return {"RX1day": rx1day, "R10mm": r10mm, "R95p": r95p, "CDD": cdd}

    indices_names = ["RX1day", "CDD", "R10mm", "R95p"]
    units = {"RX1day": "mm/jour", "CDD": "jours", "R10mm": "jours", "R95p": "mm"}

    for variant in ["V5", "Noncausal"]:
        fig, axes = plt.subplots(4, 3, figsize=(13, 16))
        any_data = False
        for r_idx, idx_name in enumerate(indices_names):
            for c_idx, gcm in enumerate(["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]):
                ax = axes[r_idx, c_idx]
                run_label = f"{variant}_{gcm}"
                if run_label not in phase7_arrays:
                    ax.text(0.5, 0.5, "(no data)", ha="center", va="center", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([]); continue
                d = phase7_arrays[run_label]
                pred_mm = np.expm1(np.clip(d["pred_mean"].astype(np.float64), 0, None))
                truth_mm = np.expm1(np.clip(d["truth"].astype(np.float64), 0, None))
                ip = _indices_per_pixel(pred_mm); it = _indices_per_pixel(truth_mm)
                bias = ip[idx_name] - it[idx_name]
                vmax = max(np.nanmax(np.abs(bias)), 1e-6)
                im = ax.imshow(bias, cmap="RdBu_r",
                                norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax))
                id_tag = "ID" if gcm == "ACCESS-CM2" else "OOD"
                ax.set_title(f"{idx_name} — {gcm} ({id_tag})\\n"
                              f"mean bias = {np.nanmean(bias):+.2f} {units[idx_name]}",
                              fontsize=9)
                plt.colorbar(im, ax=ax, shrink=0.7)
                ax.set_xticks([]); ax.set_yticks([])
                any_data = True
        if any_data:
            plt.suptitle(f"Cartes de biais des indices d'extremes — {variant}",
                          fontsize=13, y=1.0)
            plt.tight_layout()
            plt.savefig(PHASE10_DIR / f"extreme_bias_{variant}.png", dpi=120,
                        bbox_inches="tight")
            print(f"[OK] Figure {variant} : 4 indices x 3 GCMs")
        plt.close()

    print(f"\\n[OK] Phase 10 dans {PHASE10_DIR}")
'''

CELL_PHASE11_MD = """---

## Phase 11 — Causalite forte : DAG physique + Path-Specific Effects

Au-dela de la heatmap A_dag de Phase 8, deux tests rigoureux :

1. **DAG physique vs appris (Q_phys)** — comparaison du DAG appris a un DAG attendu construit a partir de la physique atmospherique (couplage vertical descendant GP250 → GP500 → GP850 → SP_HR). Mesure la *coherence physique* du DAG appris.
2. **Path-Specific Effects** — pour chaque arete forte du DAG, on zere uniquement cette arete (les autres restent intactes) et on mesure le Δ sur la prediction. Identifie quelles aretes portent vraiment le signal causal.

Si Q_phys est haut + PSE concentre sur les aretes physiquement attendues → le DAG appris **fonctionne comme un SCM**, pas comme une regularisation cosmetique.
"""

CELL_PHASE11_CODE = '''# Phase 11 — DAG physique + Path-Specific Effects
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

PHASE11_DIR = RESULTS_DIR / "phase11_causal_advanced"
PHASE11_DIR.mkdir(parents=True, exist_ok=True)

if stack_v5["A_dag"] is None:
    print("[SKIP] A_dag indisponible.")
else:
    A_learned = stack_v5["A_dag"].numpy().copy()
    np.fill_diagonal(A_learned, 0)
    n_vars = A_learned.shape[0]
    var_labels = ["GP850_spat", "GP850->GP500", "GP500_spat",
                  "GP500->GP250", "GP250_spat", "SP_HR"][:n_vars]

    # === 1. DAG physique attendu (couplage vertical descendant + meta-paths) ===
    G_phys = np.zeros((n_vars, n_vars))
    L2I = {l: i for i, l in enumerate(var_labels)}
    def add(src, tgt, s):
        if src in L2I and tgt in L2I:
            G_phys[L2I[src], L2I[tgt]] = s
    add("GP250_spat", "GP500_spat", +1)
    add("GP500_spat", "GP850_spat", +1)
    add("GP850_spat", "SP_HR", +1)
    add("GP850->GP500", "GP500_spat", +1)
    add("GP500->GP250", "GP250_spat", +1)

    A_sign = np.sign(A_learned); G_sign = np.sign(G_phys)
    mask = G_sign != 0
    matches = int(((A_sign == G_sign) & mask).sum())
    n_phys = int(mask.sum())
    Q_phys = matches / max(n_phys, 1)
    n_extra = int(((G_sign == 0) & (np.abs(A_learned) > 0.05)).sum())
    print(f"Q_phys = {Q_phys:.3f}  ({matches}/{n_phys} signes attendus corrects)")
    print(f"Aretes 'extra' apprises non prevues par G_phys : {n_extra}")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    im0 = axes[0].imshow(A_learned, cmap="RdBu_r",
                          vmin=-np.abs(A_learned).max(), vmax=np.abs(A_learned).max())
    axes[0].set_title("DAG appris (V5)"); plt.colorbar(im0, ax=axes[0], shrink=0.7)
    im1 = axes[1].imshow(G_phys, cmap="RdBu_r", vmin=-1.5, vmax=1.5)
    axes[1].set_title("DAG physique attendu"); plt.colorbar(im1, ax=axes[1], shrink=0.7)
    conf = np.zeros_like(A_learned)
    conf[(A_sign == G_sign) & mask] = +1
    conf[(A_sign != G_sign) & mask] = -1
    conf[(G_sign == 0) & (np.abs(A_learned) > 0.05)] = -0.4
    im2 = axes[2].imshow(conf, cmap="RdYlGn", vmin=-1.5, vmax=1.5)
    axes[2].set_title(f"Confusion (Q_phys = {Q_phys:.2f})\\nvert=correct  rouge=signe inverse  orange=extra")
    plt.colorbar(im2, ax=axes[2], shrink=0.7)
    for a in axes:
        a.set_xticks(range(n_vars)); a.set_xticklabels(var_labels, rotation=45, ha="right", fontsize=8)
        a.set_yticks(range(n_vars)); a.set_yticklabels(var_labels, fontsize=8)
    plt.suptitle("Comparaison DAG appris vs DAG physique", fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(PHASE11_DIR / "01_dag_physical_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[OK] Figure 1 : DAG appris vs physique")

    # === 2. Path-Specific Effects via edge-masking ======================
    rcn_cell = stack_v5["rcn_runner"].cell
    A_orig = rcn_cell.A_dag.detach().clone()

    edge_strength = np.abs(A_learned)
    edges = [(i, j, edge_strength[i, j]) for i in range(n_vars) for j in range(n_vars)
             if i != j and edge_strength[i, j] > 0.05]
    edges.sort(key=lambda x: -x[2])
    K_edges = min(10, len(edges))
    print(f"PSE sur les {K_edges} aretes les plus fortes...")

    sample_pse = next(iter(test_dataset))
    batch_pse = convert_sample_to_batch(sample_pse, builder, DEVICE)
    with torch.no_grad():
        pred_full = predict_with_stack(stack_v5, batch_pse, K=2, n_steps=18).nanmean(0).squeeze().numpy()

    pse_results = []
    for (i, j, strength) in edges[:K_edges]:
        rcn_cell.A_dag.data.copy_(A_orig)
        rcn_cell.A_dag.data[i, j] = 0.0
        with torch.no_grad():
            pred_cut = predict_with_stack(stack_v5, batch_pse, K=2, n_steps=18).nanmean(0).squeeze().numpy()
        delta = pred_full - pred_cut
        is_phys = bool(G_phys[i, j] != 0)
        pse_results.append({
            "src": var_labels[i], "tgt": var_labels[j],
            "edge_strength": float(strength),
            "pse_magnitude": float(np.abs(delta).mean()),
            "pse_mean": float(np.nanmean(delta)),
            "is_physical_edge": is_phys,
        })
    rcn_cell.A_dag.data.copy_(A_orig)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    labels = [f"{r['src']}\\n→{r['tgt']}" for r in pse_results]
    pse_mag = [r["pse_magnitude"] for r in pse_results]
    bar_colors = ["mediumseagreen" if r["is_physical_edge"] else "steelblue" for r in pse_results]
    ax.bar(range(len(labels)), pse_mag, color=bar_colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|Delta_pred| moyen (mm/jour log1p)")
    ax.set_title(f"Path-Specific Effects — {K_edges} aretes plus fortes\\n"
                  "vert = arete dans G_phys, bleu = arete additionnelle apprise")
    ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PHASE11_DIR / "02_path_specific_effects.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure 2 : PSE sur {K_edges} aretes")

    # === Save JSON ======================================================
    out = {
        "Q_phys": float(Q_phys),
        "n_physical_edges": n_phys, "n_correct_sign": matches,
        "n_extra_learned_above_0.05": n_extra,
        "G_phys": G_phys.tolist(), "A_learned": A_learned.tolist(),
        "var_labels": var_labels,
        "path_specific_effects": pse_results,
    }
    (PHASE11_DIR / "causal_advanced_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[OK] JSON : {PHASE11_DIR / 'causal_advanced_results.json'}")
'''

CELL_PHASE12_MD = """---

## Phase 12 — Attribution rigoureuse : Integrated Gradients

Integrated Gradients (Sundararajan 2017) : sensibilite *path-integrated* de la prediction par rapport au LR, depuis une *baseline climatologique* (et non zero, qui est non-physique). Satisfait l'axiome de completude : la somme des attributions egale f(x) − f(baseline).

Protocole Mamalakis 2022 ("Carefully Choose the Baseline") :
- Baseline = moyenne LR sur ~30 jours du test set
- N_steps = 32 pour la quadrature de Riemann
- Implementation manuelle (pas de dependance externe Captum)

Compare V5 vs Noncausal : ratio > 1 sur une variable = V5 l'exploite davantage que la baseline.
"""

CELL_PHASE12_CODE = '''# Phase 12 — Integrated Gradients (manual, Mamalakis 2022 protocol)
import numpy as np
import matplotlib.pyplot as plt
import torch
import json

PHASE12_DIR = RESULTS_DIR / "phase12_integrated_gradients"
PHASE12_DIR.mkdir(parents=True, exist_ok=True)


def _forward_scalar(stack, batch):
    """Scalar output (mean abs of mu_HR) pour IG."""
    enc, rcn, rh, skip = stack["encoder"], stack["rcn_runner"], stack["regression_head"], stack["skip_block"]
    lr = batch["lr"]
    H_init = enc.init_state(batch["hetero"]).to(DEVICE)
    drivers = [lr[t] for t in range(lr.shape[0])]
    seq = rcn.run(H_init, drivers, reconstruction_sources=None)
    mu_c = rh(seq.states[-1])
    tshape = batch["residual"][-1].to(DEVICE).shape
    if tshape[-2:] != mu_c.shape[-2:]:
        mu_c = torch.nn.functional.interpolate(mu_c, size=tshape[-2:], mode="bilinear", align_corners=False)
    if skip is not None:
        lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
        mu, _ = skip(lr_last, mu_c)
    else:
        mu = mu_c
    return mu.abs().mean()


def integrated_gradients(stack, batch, baseline_lr, n_steps=32):
    """Sundararajan 2017 : IG = (x - baseline) * mean_alpha grad f(baseline + alpha*(x-baseline))."""
    x = batch["lr"].clone().detach()
    baseline = baseline_lr.to(x.device).detach()
    alphas = torch.linspace(0.0, 1.0, n_steps, device=x.device)
    total_grad = torch.zeros_like(x)
    for a in alphas:
        x_interp = (baseline + a * (x - baseline)).detach().requires_grad_(True)
        b2 = dict(batch); b2["lr"] = x_interp
        y = _forward_scalar(stack, b2)
        g = torch.autograd.grad(y, x_interp, retain_graph=False, create_graph=False)[0]
        total_grad = total_grad + g.detach()
    avg_grad = total_grad / n_steps
    return ((x - baseline) * avg_grad).cpu().numpy()


# === Baseline climato : moyenne LR sur ~30 jours du test set =============
print("Construction de la baseline climato (LR mean sur 30 jours)...")
N_clim = 30
lr_accum = None; count = 0
it_clim = iter(test_dataset)
for k in range(N_clim):
    try:
        s = next(it_clim)
    except StopIteration:
        break
    b = convert_sample_to_batch(s, builder, DEVICE)
    if lr_accum is None:
        lr_accum = b["lr"].detach().clone().cpu()
    else:
        lr_accum = lr_accum + b["lr"].detach().clone().cpu()
    count += 1
baseline_lr = (lr_accum / count).to(DEVICE)
print(f"  baseline : shape={tuple(baseline_lr.shape)}, mean={baseline_lr.mean().item():.4f}, "
      f"std={baseline_lr.std().item():.4f}, n_samples={count}")

# === IG pour V5 vs Noncausal sur le meme sample ==========================
ig_results = {"V5": {}, "Noncausal": {}}
sample_ig = next(iter(test_dataset))
batch_ig = convert_sample_to_batch(sample_ig, builder, DEVICE)
lr_vars = list(CONFIG.data.lr_variables)

for stack_name, stack in [("V5", stack_v5), ("Noncausal", stack_nc)]:
    print(f"  Computing IG for {stack_name} (n_steps=32)...")
    ig = integrated_gradients(stack, batch_ig, baseline_lr, n_steps=32)
    # Aggregate to per-variable score (canal = derniere dim pour graph, ou 2e pour grid)
    if ig.ndim == 3:        # (T, N, C) graphe
        ig_per_var = np.abs(ig).mean(axis=(0, 1))
    elif ig.ndim == 4:      # (T, C, H, W) grid
        ig_per_var = np.abs(ig).mean(axis=(0, 2, 3))
    elif ig.ndim == 5:      # (B, T, C, H, W)
        ig_per_var = np.abs(ig).mean(axis=(0, 1, 3, 4))
    else:
        ig_per_var = np.abs(ig).reshape(ig.shape[-1], -1).mean(axis=-1)
    for i, v in enumerate(lr_vars):
        if i < len(ig_per_var):
            ig_results[stack_name][v] = float(ig_per_var[i])
        else:
            ig_results[stack_name][v] = 0.0

# === Plot ================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(lr_vars)); w = 0.35
v5_vals = [ig_results["V5"].get(v, 0) for v in lr_vars]
nc_vals = [ig_results["Noncausal"].get(v, 0) for v in lr_vars]
axes[0].bar(x - w/2, v5_vals, w, label="V5", color="steelblue")
axes[0].bar(x + w/2, nc_vals, w, label="Noncausal", color="orange")
axes[0].set_xticks(x); axes[0].set_xticklabels(lr_vars, rotation=45, ha="right", fontsize=7)
axes[0].set_ylabel("|IG| moyen par variable")
axes[0].set_title("Integrated Gradients par variable LR\\n(baseline = climato moyenne)")
axes[0].legend(); axes[0].grid(alpha=0.3)

ratios = [v5_vals[i] / max(nc_vals[i], 1e-12) for i in range(len(lr_vars))]
axes[1].bar(x, ratios, color="purple", alpha=0.7)
axes[1].axhline(1.0, color="red", linestyle="--", label="ratio=1")
axes[1].set_xticks(x); axes[1].set_xticklabels(lr_vars, rotation=45, ha="right", fontsize=7)
axes[1].set_ylabel("Ratio IG V5 / Noncausal")
axes[1].set_title("Ratio IG — > 1 = V5 exploite davantage cette variable")
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(PHASE12_DIR / "01_integrated_gradients.png", dpi=120, bbox_inches="tight")
plt.close()
print("[OK] Figure : Integrated Gradients par variable")

(PHASE12_DIR / "ig_results.json").write_text(
    json.dumps(ig_results, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] JSON : {PHASE12_DIR / 'ig_results.json'}")
'''


def _make_cell(cell_type, source_str):
    lines = source_str.split("\n")
    if len(lines) > 1:
        source_list = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    else:
        source_list = [source_str]
    cell = {"cell_type": cell_type, "metadata": {}, "source": source_list}
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


new_cells = [
    _make_cell("markdown", CELL_PHASE9_MD),
    _make_cell("code",     CELL_PHASE9_CODE),
    _make_cell("markdown", CELL_PHASE10_MD),
    _make_cell("code",     CELL_PHASE10_CODE),
    _make_cell("markdown", CELL_PHASE11_MD),
    _make_cell("code",     CELL_PHASE11_CODE),
    _make_cell("markdown", CELL_PHASE12_MD),
    _make_cell("code",     CELL_PHASE12_CODE),
]

# Insert before cell 11 (existing synthese MD)
INSERT_AT = 11
nb["cells"] = nb["cells"][:INSERT_AT] + new_cells + nb["cells"][INSERT_AT:]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] 8 cellules inserees (Phase 9 a Phase 12) avant la synthese")
print(f"     Total : {len(nb['cells'])} cellules")
