"""Build st_cdgm_causal_vs_noncausal_comparison.ipynb (pure post-hoc analysis).

No model is rebuilt or re-sampled: the notebook loads the JSON metrics and the
eval_samples.npz exported by each training notebook (BS43 cell), then produces a
quantitative comparison table + qualitative plots (maps, RAPSD, PDF, spread).
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "st_cdgm_causal_vs_noncausal_comparison.ipynb"

cells = []


def md(cid, lines):
    cells.append({"cell_type": "markdown", "metadata": {}, "id": cid, "source": lines})


def code(cid, srcstr):
    ast.parse(srcstr)  # fail fast on syntax error
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "id": cid, "source": srcstr.splitlines(keepends=True),
    })


md("cmp_title", [
    "# Comparaison CAUSAL vs NON-CAUSAL — ST-CDGM vs CorrDiff vanilla\n",
    "\n",
    "Analyse **purement post-hoc** : ne reconstruit ni ne re-echantillonne aucun modele.\n",
    "Charge les artefacts produits par les deux notebooks d'entrainement :\n",
    "- **causal** : `ckpt_v2_corrdiff_normal/` (Pearson 0.815)\n",
    "- **non-causal** : `ckpt_noncausal/` (= CorrDiff vanilla)\n",
    "\n",
    "Pour chaque modele on lit `final_validation_metrics.json`, `domain_metrics.json` et\n",
    "`eval_samples.npz`. La comparaison isole l'apport du **DAG causal** : le Stage 2\n",
    "(diffusion) est identique, seul le predicteur de moyenne differe.\n",
    "\n",
    "**A lire avec la nuance O3** : ceci compare deux modeles entraines separement\n",
    "(in-distribution). L'avantage attendu du causal est surtout en **OOD (CMIP6)**.\n",
])

setup = r'''# Setup : chemins + chargeurs. Override via globals() avant execution.
import json
import numpy as np
from pathlib import Path

CAUSAL_DIR = Path(str(globals().get(
    "CAUSAL_DIR", "/content/drive/MyDrive/climate_data/ckpt_v2_corrdiff_normal")))
# NONCAUSAL_DIR : Drive (Colab) sinon fallback sur ./non_causal/ (analyse hors-ligne).
_default_nc = "/content/drive/MyDrive/climate_data/ckpt_noncausal"
if not Path(_default_nc).exists() and Path("non_causal").exists():
    _default_nc = "non_causal"
NONCAUSAL_DIR = Path(str(globals().get("NONCAUSAL_DIR", _default_nc)))

print("Causal dir   :", CAUSAL_DIR)
print("Non-causal   :", NONCAUSAL_DIR)

def _load_json(p):
    p = Path(p)
    if not p.exists():
        print(f"[warn] absent : {p}")
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] echec lecture {p}: {e}")
        return {}

def _flatten(d, prefix=""):
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = float(v)
    return out

causal_fv = _load_json(CAUSAL_DIR / "final_validation_metrics.json")
causal_dm = _load_json(CAUSAL_DIR / "domain_metrics.json")
noncausal_fv = _load_json(NONCAUSAL_DIR / "final_validation_metrics.json")
noncausal_dm = _load_json(NONCAUSAL_DIR / "domain_metrics.json")

print("\nCles chargees :")
print("  causal final_validation    :", "OK" if causal_fv else "MANQUANT")
print("  causal domain              :", "OK" if causal_dm else "MANQUANT")
print("  noncausal final_validation :", "OK" if noncausal_fv else "MANQUANT (entrainer d'abord)")
print("  noncausal domain           :", "OK" if noncausal_dm else "MANQUANT (entrainer d'abord)")
'''
code("cmp_setup", setup)

quant = r'''# === COMPARAISON QUANTITATIVE ===
HEADLINE = [
    ("Pearson global",        "fv", "pearson_corr.global",        "up"),
    ("Pearson par-ech.",      "fv", "pearson_corr.per_sample_avg","up"),
    ("RMSE",                  "fv", "rmse",                       "down"),
    ("MAE",                   "fv", "mae",                        "down"),
    ("Spread (ens.)",         "fv", "spread_mean",                "info"),
    ("RAPSD distance",        "fv", "rapsd_distance",             "down"),
    ("Spread-skill ratio",    "dm", "spread_skill_ratio",         "to1"),
    ("CRPS (gaussien)",       "dm", "crps_gaussian",              "down"),
    ("Hist. distance (~LHD)", "dm", "intensity_hist_distance_L1", "down"),
    ("mu_HR abl. delta/sig",  "fv", "mu_HR_ablation.delta_signal_ratio_avg", "info"),
]

c_fv, c_dm = _flatten(causal_fv), _flatten(causal_dm)
n_fv, n_dm = _flatten(noncausal_fv), _flatten(noncausal_dm)

def _val(model, source, key):
    if model == "causal":
        d = c_fv if source == "fv" else c_dm
    else:
        d = n_fv if source == "fv" else n_dm
    return d.get(key, None)

def _winner(cv, nv, sense):
    if cv is None or nv is None:
        return "-"
    if sense == "up":
        return "causal" if cv > nv else ("non-causal" if nv > cv else "=")
    if sense == "down":
        return "causal" if cv < nv else ("non-causal" if nv < cv else "=")
    if sense == "to1":
        return "causal" if abs(cv - 1) < abs(nv - 1) else ("non-causal" if abs(nv - 1) < abs(cv - 1) else "=")
    return "-"

rows = []
print("=" * 92)
print("COMPARAISON QUANTITATIVE  (causal = ST-CDGM | non-causal = CorrDiff vanilla)")
print("=" * 92)
print(f"{'Metrique':<24}{'causal':>12}{'non-causal':>14}{'delta(nc-c)':>13}{'sens':>7}{'avantage':>13}")
print("-" * 92)
for label, source, key, sense in HEADLINE:
    cv = _val("causal", source, key)
    nv = _val("noncausal", source, key)
    delta = (nv - cv) if (cv is not None and nv is not None) else None
    win = _winner(cv, nv, sense)
    cv_s = f"{cv:.4f}" if isinstance(cv, float) else "-"
    nv_s = f"{nv:.4f}" if isinstance(nv, float) else "-"
    d_s = f"{delta:+.4f}" if isinstance(delta, float) else "-"
    print(f"{label:<24}{cv_s:>12}{nv_s:>14}{d_s:>13}{sense:>7}{win:>13}")
    rows.append(dict(metric=label, causal=cv, noncausal=nv, delta=delta, sense=sense, advantage=win))
print("-" * 92)

wins_c = sum(1 for r in rows if r["advantage"] == "causal")
wins_n = sum(1 for r in rows if r["advantage"] == "non-causal")
print(f"\nBilan skill : causal gagne {wins_c} metrique(s), non-causal {wins_n}.")
print("Rappel : une quasi-egalite in-distribution est ATTENDUE ; l'apport causal se")
print("mesure surtout en OOD (CMIP6) + interpretabilite (DAG) + garantie O3 (deja OK).")

try:
    out = NONCAUSAL_DIR.parent / "causal_vs_noncausal_table.json"
    out.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
    print(f"\nTable sauvegardee : {out}")
except Exception as e:
    print(f"[warn] sauvegarde table echouee : {e}")
'''
code("cmp_quant", quant)

qual = r'''# === COMPARAISON QUALITATIVE (depuis eval_samples.npz, aucun modele recharge) ===
import numpy as np
import matplotlib.pyplot as plt

def _load_npz(d):
    p = Path(d) / "eval_samples.npz"
    if not p.exists():
        print(f"[warn] absent : {p}")
        return None
    return np.load(p, allow_pickle=False)

cz = _load_npz(CAUSAL_DIR)
nz = _load_npz(NONCAUSAL_DIR)

if cz is None or nz is None:
    print("\n[info] eval_samples.npz manquant pour un des deux modeles.")
    print("       Lancer les deux notebooks d'entrainement (cellule BS43) d'abord.")
else:
    def rapsd(field):
        f = np.nan_to_num(np.asarray(field, dtype=np.float64), nan=0.0)
        F = np.fft.fftshift(np.fft.fft2(f))
        psd = np.abs(F) ** 2
        h, w = psd.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.indices((h, w))
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)
        tbin = np.bincount(r.ravel(), psd.ravel())
        nr = np.bincount(r.ravel())
        return tbin / np.maximum(nr, 1)

    def _sq(a):
        a = np.asarray(a)
        return a[:, 0] if (a.ndim == 4 and a.shape[1] == 1) else a

    tgt, cpred, npred = _sq(cz["target"]), _sq(cz["pred_full"]), _sq(nz["pred_full"])
    N = int(min(tgt.shape[0], cpred.shape[0], npred.shape[0], 3))

    # 1) Cartes spatiales : cible | causal | non-causal | erreurs
    fig, axes = plt.subplots(N, 5, figsize=(18, 3.4 * N))
    if N == 1:
        axes = axes[None, :]
    titles = ["Cible", "Causal", "Non-causal", "Err. causal", "Err. non-causal"]
    for i in range(N):
        vmax = float(np.nanpercentile(tgt[i], 99)) or 1.0
        ims = [tgt[i], cpred[i], npred[i], cpred[i] - tgt[i], npred[i] - tgt[i]]
        for j, (ax, im) in enumerate(zip(axes[i], ims)):
            cmap = "RdBu_r" if j >= 3 else "viridis"
            vm = vmax if j < 3 else (float(np.nanpercentile(np.abs(im), 99)) or 1.0)
            vmin = 0 if j < 3 else -vm
            hh = ax.imshow(im, cmap=cmap, vmin=vmin, vmax=vm)
            if i == 0:
                ax.set_title(titles[j], fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(hh, ax=ax, fraction=0.046)
        axes[i, 0].set_ylabel(f"ech. {i}", fontsize=10)
    fig.suptitle("Cartes spatiales (log1p mm/jour)", y=1.002, fontsize=13)
    plt.tight_layout(); plt.show()

    # 2) RAPSD overlay (moyenne sur N echantillons)
    plt.figure(figsize=(7, 5))
    for arr, lab, c in [(tgt, "Cible", "k"), (cpred, "Causal", "C0"), (npred, "Non-causal", "C1")]:
        sp = np.mean([rapsd(arr[i]) for i in range(N)], axis=0)
        plt.loglog(np.arange(1, len(sp)), sp[1:], label=lab, color=c)
    plt.xlabel("nombre d'onde radial"); plt.ylabel("puissance")
    plt.title("RAPSD - fidelite spectrale"); plt.legend(); plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout(); plt.show()

    # 3) PDF / histogramme d'intensite
    plt.figure(figsize=(7, 5))
    lo = float(min(tgt.min(), cpred.min(), npred.min()))
    hi = float(np.nanpercentile(tgt, 99.9))
    bins = np.linspace(lo, hi if hi > lo else lo + 1.0, 80)
    for arr, lab, c in [(tgt, "Cible", "k"), (cpred, "Causal", "C0"), (npred, "Non-causal", "C1")]:
        plt.hist(arr.ravel(), bins=bins, density=True, histtype="step", label=lab, color=c)
    plt.yscale("log"); plt.xlabel("intensite (log1p mm/jour)"); plt.ylabel("densite")
    plt.title("Distribution d'intensite (queue = extremes)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    # 4) Cartes de spread (calibration d'ensemble)
    if "pred_std" in cz.files and "pred_std" in nz.files:
        cstd, nstd = _sq(cz["pred_std"]), _sq(nz["pred_std"])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, im, lab in [(axes[0], cstd[0], "Spread causal"), (axes[1], nstd[0], "Spread non-causal")]:
            hh = ax.imshow(im, cmap="magma"); ax.set_title(lab); ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(hh, ax=ax, fraction=0.046)
        plt.tight_layout(); plt.show()
    print("Comparaison qualitative terminee.")
'''
code("cmp_qual", qual)

aligned = r'''# === COMPARAISON ALIGNEE cGAN + OOD (CDD / Rx1Day / R10 / saisonnier / PSD) ===
# Charge les aligned_metrics_<gcm>_<variant>.json produits par la cellule
# d'eval alignee (BS44) de chaque notebook d'entrainement, pour TOUS les GCM :
#   - in-distribution : ACCESS-CM2 (GCM d'entrainement)
#   - OOD            : EC-Earth3, NorESM2-MM (autres GCM => derive climatique)
# Protocole = identique au cGAN de Rampal (fonctions vendorisees).
GCMS = list(globals().get("GCMS", ["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]))
IN_DIST_GCM = str(globals().get("IN_DIST_GCM", "ACCESS-CM2"))

# Indices alignes a comparer (sens : |biais| plus petit = mieux ; PSD plus bas = mieux).
ALIGNED_KEYS = [
    ("CDD bias",        "indices.cdd_bias"),
    ("Rx1Day bias",     "indices.rx1day_bias"),
    ("R10 bias",        "indices.r10day_bias"),
    ("DJF rain bias",   "indices.DJF_rainfall_bias"),
    ("JJA rain bias",   "indices.JJA_rainfall_bias"),
    ("PSD distance",    "psd_distance"),
]

def _aligned_path(model_dir, gcm, variant):
    return Path(model_dir) / f"aligned_metrics_{gcm}_{variant}.json"

def _get_nested(d, dotted):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return float(cur) if isinstance(cur, (int, float)) and not isinstance(cur, bool) else None

aligned = {"causal": {}, "noncausal": {}}
for variant, mdir in [("causal", CAUSAL_DIR), ("noncausal", NONCAUSAL_DIR)]:
    for gcm in GCMS:
        aligned[variant][gcm] = _load_json(_aligned_path(mdir, gcm, variant))

_any = any(aligned[v][g] for v in aligned for g in GCMS)
if not _any:
    print("[info] Aucun aligned_metrics_*.json trouve.")
    print("       Lancer la cellule BS44 (eval alignee cGAN) dans les deux notebooks")
    print("       d'entrainement, pour ACCESS-CM2 + EC-Earth3 + NorESM2-MM.")
else:
    # 1) Table par GCM : causal vs non-causal sur chaque indice aligne.
    for gcm in GCMS:
        tag = "in-dist" if gcm == IN_DIST_GCM else "OOD"
        cj, nj = aligned["causal"].get(gcm, {}), aligned["noncausal"].get(gcm, {})
        if not cj and not nj:
            continue
        print("=" * 78)
        print(f"GCM = {gcm}  [{tag}]   (protocole cGAN/Rampal)")
        print("=" * 78)
        print(f"{'Indice':<16}{'causal':>14}{'non-causal':>14}{'|c|<|nc| ?':>14}")
        print("-" * 78)
        for label, key in ALIGNED_KEYS:
            cv, nv = _get_nested(cj, key), _get_nested(nj, key)
            if cv is None and nv is None:
                continue
            # biais : on compare les valeurs absolues (sauf PSD deja >=0)
            better = "-"
            if cv is not None and nv is not None:
                cc = abs(cv) if "bias" in key else cv
                nn = abs(nv) if "bias" in key else nv
                better = "causal" if cc < nn else ("non-causal" if nn < cc else "=")
            cv_s = f"{cv:+.4f}" if isinstance(cv, float) else "-"
            nv_s = f"{nv:+.4f}" if isinstance(nv, float) else "-"
            print(f"{label:<16}{cv_s:>14}{nv_s:>14}{better:>14}")
        print()

    # 2) DEGRADATION in-dist -> OOD : le coeur de l'argument causal.
    #    Pour chaque modele et chaque indice : |biais OOD| - |biais in-dist|.
    #    Le modele le plus ROBUSTE est celui qui se degrade le MOINS en OOD.
    ood_gcms = [g for g in GCMS if g != IN_DIST_GCM]
    if ood_gcms:
        print("=" * 78)
        print("DEGRADATION in-dist -> OOD   (|biais OOD| - |biais in-dist| ; plus bas = +robuste)")
        print("=" * 78)
        print(f"{'Indice':<16}{'causal deg.':>16}{'non-causal deg.':>18}{'+ robuste':>12}")
        print("-" * 78)
        robust_c = robust_n = 0
        for label, key in ALIGNED_KEYS:
            def _deg(variant):
                base = _get_nested(aligned[variant].get(IN_DIST_GCM, {}), key)
                if base is None:
                    return None
                degs = []
                for g in ood_gcms:
                    ov = _get_nested(aligned[variant].get(g, {}), key)
                    if ov is None:
                        continue
                    a = abs if "bias" in key else (lambda x: x)
                    degs.append(a(ov) - a(base))
                return float(np.mean(degs)) if degs else None
            dc, dn = _deg("causal"), _deg("noncausal")
            win = "-"
            if dc is not None and dn is not None:
                win = "causal" if dc < dn else ("non-causal" if dn < dc else "=")
                robust_c += int(win == "causal"); robust_n += int(win == "non-causal")
            dc_s = f"{dc:+.4f}" if isinstance(dc, float) else "-"
            dn_s = f"{dn:+.4f}" if isinstance(dn, float) else "-"
            print(f"{label:<16}{dc_s:>16}{dn_s:>18}{win:>12}")
        print("-" * 78)
        print(f"\nRobustesse OOD : causal +robuste sur {robust_c} indice(s), non-causal sur {robust_n}.")
        print("=> Si causal se degrade MOINS en OOD, c'est la preuve quantitative de")
        print("   l'apport du DAG (generalisation sous derive climatique CMIP6).")

    # 3) Sauvegarde de la synthese.
    try:
        out = NONCAUSAL_DIR.parent / "aligned_ood_comparison.json"
        out.write_text(json.dumps(aligned, indent=2, default=str), encoding="utf-8")
        print(f"\nSynthese sauvegardee : {out}")
    except Exception as e:
        print(f"[warn] sauvegarde echouee : {e}")
'''
code("cmp_aligned_ood", aligned)

md("cmp_interp", [
    "## Comment lire cette comparaison\n",
    "\n",
    "**Quantitatif** — un quasi-match in-distribution (Pearson/RMSE/CRPS proches) est le\n",
    "resultat ATTENDU : le DAG ne change que la *facon* de produire la moyenne, pas le\n",
    "budget Stage 2. Si causal ~= non-causal ici, c'est **normal et suffisant** ; la valeur\n",
    "du causal n'est pas le skill brut.\n",
    "\n",
    "**Qualitatif** — surveiller les **cartes d'erreur** (ou chaque modele se trompe), la\n",
    "**RAPSD** (hautes frequences) et la **queue de la PDF** (extremes).\n",
    "\n",
    "**Ce que cette comparaison ne montre PAS** — l'avantage causal attendu est en **OOD\n",
    "(CMIP6, derive climatique)** : robustesse + interpretabilite (DAG auditable) + garantie\n",
    "O3 (deja confirmee, ratio 0.74). Pour le chiffrer, refaire cette comparaison sur le\n",
    "test set OOD.\n",
])

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
text = json.dumps(nb, ensure_ascii=False, indent=1)
json.loads(text)  # validate
OUT.write_text(text, encoding="utf-8")
print(f"Wrote {OUT} ({len(cells)} cells) - all code cells syntax-validated")
