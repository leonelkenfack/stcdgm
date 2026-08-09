"""Génère `st_cdgm_stage1_spec.ipynb` — CAHIER DES CHARGES de l'étage 1 pour V8.

Ce notebook ne diagnostique AUCUN modèle existant. V5 est abandonné, son
skip-block n'existe pas dans V8, et son décodeur est remplacé (V7-M1/M4).
Diagnostiquer un checkpoint mort serait de l'archéologie.

Il établit à la place, À PARTIR DES DONNÉES SEULES, les spécifications
que l'étage 1 de V8 devra respecter, puis fournit un HARNAIS réutilisable
qui teste n'importe quel étage 1 futur contre ces spécifications.

  SPEC-1  Résolution du décodeur : quelle grille intermédiaire faut-il pour
          ne pas jeter de signal récupérable ? (V8 ne le spécifie pas)
  SPEC-2  Biais de Jensen : quantifié + formule de correction validée
  SPEC-3  Convention d'évaluation : ne pas mesurer les extrêmes sur la
          moyenne d'ensemble
  HARNAIS audit_stage1(mu, target, baseline) -> conforme / non conforme,
          dont le test décisif de prédictibilité résiduelle

Ne requiert NI checkpoint NI torch_geometric. Tourne sur CPU ou T4.
Sortie : path_c_plus/scripts/st_cdgm_stage1_spec.ipynb
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "path_c_plus" / "scripts" / "st_cdgm_stage1_spec.ipynb"


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": t.splitlines(keepends=True)}


C0 = """# Étage 1 de V8 — cahier des charges mesuré

**Ce notebook ne diagnostique aucun modèle existant.** V5 est abandonné : son
`ConditionalSkipBlock` n'existe pas dans V8, et son décodeur est remplacé
(héritage V7-M1/M4). Diagnostiquer ses checkpoints serait de l'archéologie.

Il établit **à partir des données seules** les spécifications que l'étage 1
de V8 devra respecter, puis fournit un **harnais réutilisable** qui testera
n'importe quel étage 1 futur.

### Déjà acquis sur CPU (`results/diag_stage1*.json`, §9 de `architecture_v8_design.md`)

| Mesure | Valeur |
|---|---|
| Prédictibilité depuis le champ grande échelle | ρ² = 0,85 à **13 km** ; 0,57 à 9 km |
| Variance HR sous 48 km | **1,0 %** → la MSE est structurellement aveugle aux extrêmes |
| Biais de Jensen (log1p → mm) | 10,6 % à 1 mm/j → **18,5 %** sur le bin le plus intense |
| Skip-block responsable des extrêmes | **RÉFUTÉ** (à 48 km, α n'a aucun effet) |
| CDD expliqué par le seuillage de la moyenne | **RÉFUTÉ** |

### Le manque que ce notebook comble

V8 §C9 spécifie « décodeur HR avec encodage positionnel + statiques CTX +
features câblées-HR », mais **aucune résolution intermédiaire**. C'est
pourtant le goulot mesuré (43×45 ≈ 48 km chez V5, alors que le signal est
récupérable jusqu'à 13 km). SPEC-1 fournit le chiffre.

> ⚠️ NorESM2-MM est un holdout pré-enregistré. Ce notebook ne l'ouvre jamais.
"""

C1 = """# >>> Cell 1 : Bootstrap (léger — ni torch_geometric, ni diffusers)
import os, sys, subprocess
from pathlib import Path

GIT_URL, GIT_BRANCH = "https://github.com/leonelkenfack/stcdgm.git", "four-node-causal"
REPO_DIR, DRIVE_ROOT = "/content/climate_data", "/content/drive/MyDrive/climate_data"

if not Path("/content/drive").exists():
    from google.colab import drive; drive.mount("/content/drive")
if not Path(REPO_DIR).exists():
    subprocess.check_call(["git", "clone", "--depth=50", "-b", GIT_BRANCH, GIT_URL, REPO_DIR])
else:
    subprocess.check_call(["git", "-C", REPO_DIR, "pull", "origin", GIT_BRANCH])
for p in (REPO_DIR, str(Path(REPO_DIR) / "src")):
    if p not in sys.path: sys.path.insert(0, p)
os.chdir(REPO_DIR)

try:
    import xarray, h5netcdf, netCDF4  # noqa: F401
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           "xarray", "h5netcdf", "netcdf4", "cftime"])

import numpy as np, torch, torch.nn.functional as F, xarray as xr, json
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| torch", torch.__version__)
"""

C2 = """# >>> Cell 2 : Données (HR uniquement ; aucun checkpoint requis)
FORBIDDEN = "NorESM2"          # holdout pré-enregistré
def guard(p):
    assert FORBIDDEN not in str(p), f"STOP — {p} touche le holdout NorESM2-MM."
    return p

CANDIDATES = [f"{DRIVE_ROOT}/data/train/pr_ACCESS-CM2_hist.nc",
              f"{DRIVE_ROOT}/pr_ACCESS-CM2_hist.nc",
              "data/raw/train/pr_ACCESS-CM2_hist.nc"]
HR_PATH = next((guard(p) for p in CANDIDATES if Path(p).exists()), None)
assert HR_PATH, f"pr_ACCESS-CM2_hist.nc introuvable. Testé : {CANDIDATES}"
print("HR :", HR_PATH)

N_DAYS = 3000                                   # T4 : ~1 min de lecture
rng = np.random.default_rng(0)
ds = xr.open_dataset(HR_PATH, decode_times=False)
idx = np.sort(rng.choice(ds.sizes["time"], size=min(N_DAYS, ds.sizes["time"]), replace=False))
x = np.nan_to_num(ds["pr"].isel(time=idx).values.astype(np.float32), nan=0.0)
ds.close()
H, W = x.shape[-2], x.shape[-1]
xt = torch.from_numpy(x).unsqueeze(1)          # [N,1,H,W], reutilise par SPEC-1/2/3
PIX_KM = 12.0
print(f"x = {x.shape}  ({PIX_KM} km/pixel, domaine {H*PIX_KM:.0f}x{W*PIX_KM:.0f} km)")

clim99 = np.quantile(x, 0.99, axis=0)
truth99 = x > clim99[None]

def f1_p99(pred):
    pe = pred > clim99[None]
    tp = float((pe & truth99).sum()); fp = float((pe & ~truth99).sum()); fn = float((~pe & truth99).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)

SPEC = {}
"""

C3 = """# >>> Cell 3 : SPEC-1 — quelle résolution intermédiaire le décodeur doit-il avoir ?
# Un décodeur qui CONSTRUIT sur une grille (h,w) puis interpole vers HR ne peut,
# au mieux, restituer que la vérité vue à cette grille. On borne donc le plafond
# atteignable POUR CHAQUE grille candidate.
# xt defini en Cell 2
grids = [(23, 26), (43, 45), (57, 60), (86, 90), (114, 119), (172, 179)]
rows = []
for (gh, gw) in grids:
    lo = F.adaptive_avg_pool2d(xt, (gh, gw))
    up = F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False).squeeze(1).numpy()
    eff_km = PIX_KM * H / gh
    rows.append(dict(grid=f"{gh}x{gw}", eff_km=round(eff_km, 1), f1_ceiling=f1_p99(up)))
    print(f"  grille {gh:>3d}x{gw:<3d}  ≈{eff_km:5.1f} km   plafond F1@p99 = {rows[-1]['f1_ceiling']:.4f}")
SPEC["S1_resolution"] = rows

plt.figure(figsize=(6.5, 4))
plt.plot([r["eff_km"] for r in rows], [r["f1_ceiling"] for r in rows], "o-")
plt.axvline(48, color="r", ls="--", label="43x45 (V5) ≈ 48 km")
plt.axhline(0.512, color="grey", ls=":", label="Oracle V5 mesuré = 0.512")
plt.gca().invert_xaxis(); plt.xlabel("résolution effective de construction (km)")
plt.ylabel("plafond F1@p99"); plt.legend(); plt.grid(alpha=.3)
plt.title("SPEC-1 : plafond atteignable selon la grille du décodeur")
plt.tight_layout(); plt.savefig("results/spec1_resolution.png", dpi=130); plt.show()

target = next((r for r in rows if r["f1_ceiling"] >= 0.80), rows[-1])
print(f"\\n=== SPEC-1 ===\\n  Pour un plafond >= 0.80, il faut une grille "
      f"{target['grid']} (≈{target['eff_km']} km).")
print("  V8 §C9 ne specifie AUCUNE resolution intermediaire -> a inscrire.")
SPEC["S1_recommended_grid"] = target
"""

C4 = """# >>> Cell 4 : SPEC-2 — biais de Jensen et sa correction
# L'etage 1 minimise une MSE sur log1p => il apprend E[log1p(x)|y], pas E[x|y].
# expm1(E[log1p x]) sous-estime. Correction lognormale : expm1(mu + s^2/2).
lo = F.adaptive_avg_pool2d(xt, (23, 26))
b = F.interpolate(lo, size=(H, W), mode="bilinear", align_corners=False).squeeze(1).numpy()
xv, bv = x.ravel(), b.ravel()
qs = np.unique(np.quantile(bv, np.linspace(0, 1, 13)))
bid = np.clip(np.digitize(bv, qs[1:-1]), 0, len(qs) - 2)

rows, naive_err, corr_err = [], [], []
for k in range(len(qs) - 1):
    s = bid == k
    if s.sum() < 500: continue
    xs = xv[s]; L = np.log1p(xs)
    true_m = float(xs.mean())
    naive  = float(np.expm1(L.mean()))
    corr   = float(np.expm1(L.mean() + 0.5 * L.var()))     # correction lognormale
    rows.append(dict(bin=k, b_lo=float(qs[k]), b_hi=float(qs[k+1]), n=int(s.sum()),
                     true=true_m, naive=naive, corrected=corr,
                     bias_naive_pct=100*(true_m-naive)/max(true_m,1e-9),
                     bias_corr_pct=100*(true_m-corr)/max(true_m,1e-9)))
    naive_err.append(abs(rows[-1]["bias_naive_pct"])); corr_err.append(abs(rows[-1]["bias_corr_pct"]))

print(f"{'bin baseline':>20s} {'vrai':>9s} {'naif':>9s} {'corrige':>9s} {'b.naif':>8s} {'b.corr':>8s}")
for r in rows:
    print(f"{r['b_lo']:8.2f}-{r['b_hi']:8.2f} {r['true']:9.3f} {r['naive']:9.3f} "
          f"{r['corrected']:9.3f} {r['bias_naive_pct']:7.1f}% {r['bias_corr_pct']:7.1f}%")
print(f"\\n=== SPEC-2 ===")
print(f"  biais absolu moyen  naif = {np.mean(naive_err):5.2f} %   corrige = {np.mean(corr_err):5.2f} %")
print("  => appliquer expm1(mu + s^2/2) au retour en mm, ou entrainer en mm.")
SPEC["S2_jensen"] = dict(bins=rows, mean_abs_bias_naive=float(np.mean(naive_err)),
                         mean_abs_bias_corrected=float(np.mean(corr_err)))
"""

C5 = """# >>> Cell 5 : SPEC-3 — ne pas mesurer les extremes sur la moyenne d'ensemble
# Le code actuel (eval_metrics_dual_convention.py:215) calcule sur
# members_mm.mean(dim=0) : un champ lisse une SECONDE fois.
NS = 400                                        # sous-echantillon : 64 membres x 3000 j = 44 Gio
xs_, cl_, tr_ = x[:NS], clim99, truth99[:NS]
mu = F.interpolate(F.adaptive_avg_pool2d(xt[:NS], (86, 90)), size=(H, W),
                   mode="bilinear", align_corners=False).squeeze(1).numpy()
sig = float(np.std(xs_ - mu))
rng2 = np.random.default_rng(42)

def f1_sub(pred):
    pe = pred > cl_[None]
    tp = float((pe & tr_).sum()); fp = float((pe & ~tr_).sum()); fn = float((~pe & tr_).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)

for K in (12, 64):
    mem = np.stack([np.maximum(mu + rng2.normal(0, sig, mu.shape).astype(np.float32), 0)
                    for _ in range(K)])
    f_mean = f1_sub(mem.mean(axis=0)); f_memb = float(np.mean([f1_sub(m) for m in mem]))
    f_q90 = f1_sub(np.quantile(mem, 0.90, axis=0)); del mem
    print(f"  K={K:>2d} | moyenne d'ensemble {f_mean:.4f} | par membre {f_memb:.4f} "
          f"| quantile 90% {f_q90:.4f}")
    SPEC[f"S3_K{K}"] = dict(ens_mean=f_mean, per_member=f_memb, q90=f_q90)
print("\\n=== SPEC-3 ===")
print("  Si 'par membre' >> 'moyenne d'ensemble', la metrique penalise la")
print("  convention d'evaluation, pas le modele. Rapporter les DEUX.")
"""

C6 = """# >>> Cell 6 : HARNAIS — a appliquer a TOUT etage 1 futur (V8 inclus)
def audit_stage1(mu_log, target_log, baseline_log, lr_fields=None, name="stage1"):
    \"\"\"Teste un etage 1 contre les specifications mesurees.

    mu_log/target_log/baseline_log : [N,H,W] en espace log1p.
    lr_fields : [N,C,h,w] optionnel -> active le test decisif de
                predictibilite residuelle.
    \"\"\"
    to_mm = lambda z: np.expm1(np.clip(z, -20, 20))
    x_mm  = to_mm(baseline_log + target_log)
    mu_mm = to_mm(baseline_log + mu_log)
    rep = {"name": name}

    # --- C1 biais conditionnel (Jensen residuel) --------------------------
    q = np.unique(np.quantile(mu_mm, np.linspace(0, 1, 11)))
    bid = np.clip(np.digitize(mu_mm.ravel(), q[1:-1]), 0, len(q) - 2)
    xr_, mr_ = x_mm.ravel(), mu_mm.ravel()
    bias = [100*(xr_[bid==k].mean()-mr_[bid==k].mean())/max(xr_[bid==k].mean(),1e-9)
            for k in range(len(q)-1) if (bid==k).sum() > 500]
    rep["C1_max_conditional_bias_pct"] = float(np.max(np.abs(bias)))
    rep["C1_pass"] = rep["C1_max_conditional_bias_pct"] < 5.0

    # --- C2 resolution effective (spectre) --------------------------------
    def rapsd(f):
        f = f - f.mean(axis=(-2, -1), keepdims=True)
        P = np.abs(np.fft.rfft2(f)) ** 2
        h, w = f.shape[-2], f.shape[-1]
        ky = np.fft.fftfreq(h)[:, None]; kx = np.fft.rfftfreq(w)[None, :]
        kr = np.sqrt(ky**2 + kx**2); nb = 30
        kb = np.clip((kr / kr.max() * nb).astype(int), 0, nb - 1)
        return np.array([P[..., kb==i].mean() if (kb==i).any() else np.nan for i in range(nb)])
    rt, rm = rapsd(target_log), rapsd(mu_log)
    ratio = rm / np.maximum(rt, 1e-30)
    half = np.argmax(ratio < 0.5) if (ratio < 0.5).any() else len(ratio)-1
    rep["C2_effective_res_km"] = float(PIX_KM * mu_log.shape[-1] / max(half, 1) / 2)
    rep["C2_pass"] = rep["C2_effective_res_km"] <= 20.0

    # --- C3 predictibilite residuelle (LE test decisif) -------------------
    if lr_fields is not None:
        import torch.nn as nn
        r = x_mm - mu_mm
        Y = F.interpolate(torch.from_numpy(lr_fields).float(),
                          size=r.shape[-2:], mode="bilinear", align_corners=False)
        R = torch.from_numpy(r).float().unsqueeze(1)
        ntr = int(0.7 * len(r))
        net = nn.Sequential(nn.Conv2d(Y.shape[1], 48, 3, padding=1), nn.GELU(),
                            nn.Conv2d(48, 48, 3, padding=1), nn.GELU(),
                            nn.Conv2d(48, 1, 3, padding=1)).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4)
        for _ in range(30):
            perm = torch.randperm(ntr)
            for j in range(0, ntr, 8):
                i2 = perm[j:j+8]
                loss = ((net(Y[i2].to(DEVICE)) - R[i2].to(DEVICE))**2).mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pr = net(Y[ntr:].to(DEVICE)).cpu().numpy().squeeze(1)
        rte = r[ntr:]
        r2 = 1 - ((rte-pr)**2).sum() / max(((rte-rte.mean())**2).sum(), 1e-9)
        rep["C3_residual_R2"] = float(r2)
        rep["C3_pass"] = r2 < 0.02
    return rep

print("=== HARNAIS PRET ===")
print("  audit_stage1(mu_log, target_log, baseline_log, lr_fields=None)")
print("  C1 biais conditionnel < 5 %      | C2 resolution effective <= 20 km")
print("  C3 R2 residuel < 0.02  <-- LE test decisif : l'etage 1 detruit-il")
print("     de l'information recuperable depuis y ?")
"""

C7 = """# >>> Cell 7 : Ce qu'il faut inscrire dans V8
json.dump(SPEC, open("results/stage1_spec.json", "w"), indent=2, default=float)
g = SPEC["S1_recommended_grid"]
print("=" * 66)
print("SPECIFICATIONS ETAGE 1 POUR V8")
print("=" * 66)
print(f"\\nSPEC-1  grille intermediaire du decodeur >= {g['grid']} (≈{g['eff_km']} km)")
print(f"        plafond F1@p99 correspondant : {g['f1_ceiling']:.3f}")
_v5 = next((r for r in SPEC["S1_resolution"] if r["grid"] == "43x45"), None)
if _v5:
    print(f"        (V5 : 43x45 ≈ 48 km -> plafond {_v5['f1_ceiling']:.3f} ; mesure 0.512)")
    print(f"        => l'ecart {_v5['f1_ceiling']:.3f} -> 0.512 n'est PAS de la resolution :")
    print(f"           le modele n'atteint pas son propre plafond architectural.")
print(f"\\nSPEC-2  corriger le retour log1p -> mm : expm1(mu + s^2/2)")
print(f"        biais moyen {SPEC['S2_jensen']['mean_abs_bias_naive']:.1f} % -> "
      f"{SPEC['S2_jensen']['mean_abs_bias_corrected']:.1f} %")
print(f"\\nSPEC-3  rapporter le F1@p99 PAR MEMBRE, pas seulement sur la moyenne")
print(f"\\nSPEC-4  critere d'acceptation : R2 residuel < 0.02 (harnais Cell 6)")
print("\\n=> ECRIT results/stage1_spec.json")
print("=" * 66)
"""

cells = [md(C0), code(C1), code(C2), code(C3), code(C4), code(C5), code(C6), code(C7)]
nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.11"}, "accelerator": "GPU"},
      "nbformat": 4, "nbformat_minor": 5}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"écrit : {OUT}  ({len(cells)} cellules)")
