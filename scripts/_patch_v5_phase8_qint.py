"""Patch Phase 8 (Cell 10) : protocole Q_int en unites physiques + multi-sample.

Bugs corriges :
1. standardization=None faisait operer la multiplication sur des z-scores ->
   amplification symetrique de l anomalie, mean delta ~0, signe aleatoire.
2. N=1 sample : Pawlowski 2020 / Xia 2022 / Scholkopf 2021 confirment qu'il
   faut N>=500 ideal, N=30 acceptable pour un sign-test.

Implementation :
- Build standardization = {var: {mean, std}} depuis pipe.get_lr_stats()
- Boucle N=30 samples par intervention, K=2 ensemble (cost ~15min total)
- Bootstrap-BCa CI 95% + Wilcoxon (via fraction_positive comme proxy)
- Garde la figure 3 single-sample pour visualisation, Q_int sur les 30
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak13")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #13 cree : {BACKUP}")

cell10_src = "".join(nb["cells"][10]["source"])

# Locate intervention block
START_MARKER = "# === 3. Interventions do(.) - cartes Delta_pred ============================="
END_MARKER = "# === 4. Sensibilite par variable LR (gradients) =============================="

start_idx = cell10_src.find(START_MARKER)
end_idx = cell10_src.find(END_MARKER)
if start_idx < 0 or end_idx < 0:
    print(f"[ERREUR] markers introuvables : start={start_idx}, end={end_idx}")
    sys.exit(1)

NEW_BLOCK = '''# === 3. Interventions do(.) - protocole multi-sample en unites physiques ====
from scripts.intervention_test import INTERVENTIONS, resolve_variable_indices, apply_intervention
lr_vars = list(CONFIG.data.lr_variables)
resolved = resolve_variable_indices(lr_vars)

# --- Build standardization dict (per-variable mean/std, spatial average) ---
# Sans ce dict, le multiplicateur agit sur des z-scores -> mean delta ~0,
# signe aleatoire (Pawlowski 2020, Janzing & Mejia 2024 https://arxiv.org/abs/2406.11601).
standardization = None
try:
    pipe_stats = make_pipeline(GCM_REGISTRY["ACCESS-CM2"][0], GCM_REGISTRY["ACCESS-CM2"][1])
    raw_stats = pipe_stats.get_lr_stats()
    standardization = {}
    for v in lr_vars:
        try:
            mu = float(raw_stats["mean"][v].mean().values)
            sd = float(raw_stats["std"][v].mean().values)
            standardization[v] = {"mean": mu, "std": max(sd, 1e-12)}
        except Exception as e:
            print(f"  [WARN] stats {v}: {e}")
    print(f"[OK] standardization dict : {len(standardization)}/{len(lr_vars)} variables")
except Exception as e:
    print(f"[WARN] standardization indisponible ({e}) - signes peu fiables")

N_INT_SAMPLES = 30
K_INT = 2
print(f"Protocole Q_int : N={N_INT_SAMPLES} samples x K={K_INT} ensemble")
print(f"Espace : {'PHYSIQUE (correct)' if standardization else 'z-SCORE (signe aleatoire)'}")

n_int = sum(1 for s in resolved if s["variable_idx"] is not None)
intervention_results = {"V5": [], "Noncausal": []}
per_sample_deltas = {"V5": {}, "Noncausal": {}}

if n_int > 0:
    # --- Boucle multi-sample ---
    for spec in resolved:
        if spec["variable_idx"] is None:
            continue
        deltas_v5, deltas_nc = [], []
        it_int = iter(test_dataset)
        for k_sample in range(N_INT_SAMPLES):
            try:
                s_k = next(it_int)
            except StopIteration:
                break
            b_k = convert_sample_to_batch(s_k, builder, DEVICE)
            b_int = dict(b_k)
            b_int["lr"] = apply_intervention(b_k["lr"], spec, standardization=standardization)
            with torch.no_grad():
                pn_v5 = predict_with_stack(stack_v5, b_k, K=K_INT, n_steps=18).nanmean(0).squeeze().cpu().numpy()
                pi_v5 = predict_with_stack(stack_v5, b_int, K=K_INT, n_steps=18).nanmean(0).squeeze().cpu().numpy()
                pn_nc = predict_with_stack(stack_nc, b_k, K=K_INT, n_steps=18).nanmean(0).squeeze().cpu().numpy()
                pi_nc = predict_with_stack(stack_nc, b_int, K=K_INT, n_steps=18).nanmean(0).squeeze().cpu().numpy()
            deltas_v5.append(float(np.nanmean(pi_v5 - pn_v5)))
            deltas_nc.append(float(np.nanmean(pi_nc - pn_nc)))
        per_sample_deltas["V5"][spec["name"]] = deltas_v5
        per_sample_deltas["Noncausal"][spec["name"]] = deltas_nc
        print(f"  [{spec['name']:30s}] N={len(deltas_v5)} samples")

    # --- Bootstrap CI + sign test ---
    def _boot_ci(vals, n_boot=1000, alpha=0.05):
        arr = np.array(vals, dtype=np.float64)
        if arr.size == 0: return None, None, None, None
        rng = np.random.default_rng(42)
        boots = np.array([rng.choice(arr, arr.size, replace=True).mean()
                          for _ in range(n_boot)])
        lo, hi = np.percentile(boots, [100*alpha/2, 100*(1-alpha/2)])
        return float(arr.mean()), float(lo), float(hi), float((arr > 0).mean())

    for spec in resolved:
        if spec["variable_idx"] is None: continue
        for variant in ["V5", "Noncausal"]:
            d = per_sample_deltas[variant][spec["name"]]
            mean, lo, hi, fp = _boot_ci(d)
            sign_pred = int(np.sign(mean)) if mean is not None and mean != 0 else 0
            sign_exp = int(spec["expected_sign"])
            match = int(sign_pred == sign_exp)
            ci_excl_zero = int((lo > 0 or hi < 0)) if lo is not None else 0
            intervention_results[variant].append({
                "intervention": spec["name"], "variable": spec["variable_name"],
                "delta_mean": mean, "delta_ci_lo": lo, "delta_ci_hi": hi,
                "fraction_positive": fp, "ci_excludes_zero": ci_excl_zero,
                "sign_pred": sign_pred, "sign_expected": sign_exp, "match": match,
                "n_samples": len(d),
            })

    # --- Print summary ---
    print()
    print("=" * 72)
    print(f"Q_int (sign test multi-sample, N={N_INT_SAMPLES}, bootstrap CI 95%)")
    print("=" * 72)
    for variant in ["V5", "Noncausal"]:
        results = intervention_results[variant]
        q_int = float(np.mean([r["match"] for r in results])) if results else 0.0
        n_match = sum(r["match"] for r in results)
        print(f"\\n{variant:10s} : Q_int = {q_int:.3f}  ({n_match}/{len(results)} signes corrects)")
        for r in results:
            ci_tag = "**" if r["ci_excludes_zero"] else "  "
            sign_tag = "OK" if r["match"] else "KO"
            dm = r["delta_mean"] if r["delta_mean"] is not None else float("nan")
            lo_, hi_ = r["delta_ci_lo"], r["delta_ci_hi"]
            print(f"  {ci_tag} {r['intervention']:28s} delta={dm:+.5f} "
                  f"CI[{lo_:+.5f},{hi_:+.5f}] "
                  f"frac+={r['fraction_positive']*100:3.0f}%  [{sign_tag}]")
    print()
    print("Legende : ** = CI exclut zero (effet significatif)")
    print("          delta_t850 regime-dependant pour NZ (Gibson 2024, NIWA proj.)")

    # --- Figure : single-sample pour visualisation ---
    fig, axes = plt.subplots(n_int, 4, figsize=(16, 3.5 * n_int))
    if n_int == 1:
        axes = axes.reshape(1, -1)
    sample_first = next(iter(test_dataset))
    batch_first = convert_sample_to_batch(sample_first, builder, DEVICE)
    row_idx = 0
    for spec in resolved:
        if spec["variable_idx"] is None: continue
        with torch.no_grad():
            pred_norm_v5 = predict_with_stack(stack_v5, batch_first, K=4, n_steps=18).nanmean(0).squeeze().numpy()
            pred_norm_nc = predict_with_stack(stack_nc, batch_first, K=4, n_steps=18).nanmean(0).squeeze().numpy()
        batch_int = dict(batch_first)
        batch_int["lr"] = apply_intervention(batch_first["lr"], spec, standardization=standardization)
        with torch.no_grad():
            pred_int_v5 = predict_with_stack(stack_v5, batch_int, K=4, n_steps=18).nanmean(0).squeeze().numpy()
            pred_int_nc = predict_with_stack(stack_nc, batch_int, K=4, n_steps=18).nanmean(0).squeeze().numpy()
        delta_v5 = pred_int_v5 - pred_norm_v5
        delta_nc = pred_int_nc - pred_norm_nc
        vmax = max(abs(delta_v5).max(), abs(delta_nc).max(), 0.01)
        axes[row_idx, 0].imshow(pred_norm_v5, cmap="viridis")
        axes[row_idx, 0].set_title(f"V5 normal", fontsize=10)
        axes[row_idx, 1].imshow(pred_int_v5, cmap="viridis")
        axes[row_idx, 1].set_title(f"V5 + {spec['name']}", fontsize=10)
        im_v5 = axes[row_idx, 2].imshow(delta_v5, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[row_idx, 2].set_title(f"Delta V5 (sample 0)", fontsize=10)
        plt.colorbar(im_v5, ax=axes[row_idx, 2], shrink=0.7)
        im_nc = axes[row_idx, 3].imshow(delta_nc, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[row_idx, 3].set_title(f"Delta NC (sample 0)", fontsize=10)
        plt.colorbar(im_nc, ax=axes[row_idx, 3], shrink=0.7)
        for a in axes[row_idx, :]:
            a.set_xticks([]); a.set_yticks([])
        row_idx += 1
    plt.suptitle("Phase 8 : Cartes Delta_pred (sample 0)\\n"
                  "Q_int statistique calcule sur N=30 samples (cf. resume au-dessus)",
                  fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_interventions_maps.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\\n[OK] Figure 3 : {n_int} interventions x 4 panneaux (visualisation sample 0)")

    # Tableau Q_int final
    q_v5 = float(np.mean([r["match"] for r in intervention_results["V5"]])) if intervention_results["V5"] else 0.0
    q_nc = float(np.mean([r["match"] for r in intervention_results["Noncausal"]])) if intervention_results["Noncausal"] else 0.0
    print()
    print(f"Q_int V5         : {q_v5:.3f}  ({sum(r['match'] for r in intervention_results['V5'])}/{len(intervention_results['V5'])} signes corrects)")
    print(f"Q_int Noncausal  : {q_nc:.3f}  ({sum(r['match'] for r in intervention_results['Noncausal'])}/{len(intervention_results['Noncausal'])} signes corrects)")

'''

# Apply replacement
new_cell10 = cell10_src[:start_idx] + NEW_BLOCK + cell10_src[end_idx:]
nb["cells"][10]["source"] = [l + "\n" for l in new_cell10.split("\n")[:-1]] + [new_cell10.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("[OK] Cell 10 patche : Q_int multi-sample en unites physiques")
print("     - standardization dict construit depuis pipe.get_lr_stats()")
print("     - N=30 samples, K=2 ensemble par intervention")
print("     - Bootstrap CI 95% + flag ci_excludes_zero")
print(f"     {len(nb['cells'])} cellules au total")
