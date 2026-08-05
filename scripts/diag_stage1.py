"""Diagnostics CPU, INDEPENDANTS DU MODELE, sur l'etage 1 d'Oracle.

Trois questions tranchees :
  D1  Ecart de Jensen : entrainer en log1p puis revenir en mm biaise-t-il la queue ?
  D2  Spectre de predictibilite rho^2(k) : a quelle echelle la moyenne
      conditionnelle DOIT-elle s'effondrer ?
  D3  Plafond de F1@p99 : quel score un predicteur PARFAIT MAIS LISSE
      peut-il atteindre ? (+ mecanique du biais CDD)

N'ouvre QUE ACCESS-CM2 (train). NorESM2-MM = holdout, jamais touche.
"""
import numpy as np, xarray as xr, torch, torch.nn.functional as F, json, sys

RNG = np.random.default_rng(0)
NT = 4000          # jours echantillonnes pour les stats
NT_SPEC = 400      # jours pour les spectres (plus couteux)
OUT = {}

print("[1/5] chargement HR pr (ACCESS-CM2 hist)...", flush=True)
ds = xr.open_dataset("data/raw/train/pr_ACCESS-CM2_hist.nc", decode_times=False)
T = ds.sizes["time"]
idx = np.sort(RNG.choice(T, size=min(NT, T), replace=False))
x = ds["pr"].isel(time=idx).values.astype(np.float32)   # [NT,172,179] mm/j
ds.close()
print("   HR", x.shape, "min/max", float(np.nanmin(x)), float(np.nanmax(x)), flush=True)

valid = np.isfinite(x).all(axis=0)      # pixels valides sur tout l'echantillon
print("   pixels valides :", int(valid.sum()), "/", valid.size, flush=True)
x = np.nan_to_num(x, nan=0.0)

# ---- baseline = coarsen HR -> grille LR 23x26 (area) puis upsample bilineaire
print("[2/5] construction de la baseline (coarsen 23x26 -> bilinear 172x179)...", flush=True)
xt = torch.from_numpy(x).unsqueeze(1)                       # [NT,1,172,179]
lr = F.adaptive_avg_pool2d(xt, (23, 26))                    # grille LR
b = F.interpolate(lr, size=(172, 179), mode="bilinear", align_corners=False)
b = b.squeeze(1).numpy().astype(np.float32)
del xt
print("   baseline", b.shape, flush=True)

m = valid[None, :, :] & np.ones((x.shape[0], 1, 1), bool)
xv, bv = x[m], b[m]                                          # vecteurs aplatis
print("   echantillons (pixel x jour) :", xv.size, flush=True)

# =====================================================================
# D1 — ECART DE JENSEN
#   Leur perte etage 1 : MSE sur log1p. Donc mu ~ E[log1p(x)|y].
#   Reconstruction : expm1(E[log1p x|y])  vs  vraie moyenne E[x|y].
#   On conditionne sur la baseline b (proxy du conditionnement y).
#   => borne INFERIEURE de l'ecart (un conditionnement plus riche reduit
#      la variance conditionnelle donc l'ecart).
# =====================================================================
print("[3/5] D1 ecart de Jensen...", flush=True)
qs = np.quantile(bv, np.linspace(0, 1, 21))
qs = np.unique(qs)
bin_id = np.clip(np.digitize(bv, qs[1:-1]), 0, len(qs) - 2)
rows = []
for k in range(len(qs) - 1):
    s = bin_id == k
    n = int(s.sum())
    if n < 500:
        continue
    xs = xv[s]
    true_mean = float(xs.mean())                      # E[x | bin]
    log_mean = float(np.log1p(xs).mean())             # E[log1p x | bin]
    recon = float(np.expm1(log_mean))                 # ce que donne un MSE-log
    rows.append(dict(bin=k, n=n, b_lo=float(qs[k]), b_hi=float(qs[k + 1]),
                     true_mean=true_mean, recon_mean=recon,
                     gap=true_mean - recon,
                     gap_pct=100.0 * (true_mean - recon) / max(true_mean, 1e-9),
                     cond_std=float(xs.std())))
OUT["D1_jensen_bins"] = rows

# agrege global + sur la queue
def jensen_on(mask, label):
    xs = xv[mask]
    if xs.size < 100:
        return None
    tm, rc = float(xs.mean()), float(np.expm1(np.log1p(xs).mean()))
    return dict(label=label, n=int(xs.size), true_mean=tm, recon_mean=rc,
                gap=tm - rc, gap_pct=100.0 * (tm - rc) / max(tm, 1e-9))

p90, p99, p999 = np.quantile(xv, [0.90, 0.99, 0.999])
OUT["D1_jensen_global"] = [
    jensen_on(np.ones_like(xv, bool), "tous"),
    jensen_on(xv > p90, "x > p90"),
    jensen_on(xv > p99, "x > p99"),
    jensen_on(xv > p999, "x > p99.9"),
]

# =====================================================================
# D2 — SPECTRE DE PREDICTIBILITE rho^2(k)
#   fraction de variance de x, a l'echelle k, explicable par le champ
#   grande echelle b. rho^2 -> 0 => la moyenne conditionnelle DOIT
#   s'effondrer a cette echelle (ce n'est pas un defaut du modele).
# =====================================================================
print("[4/5] D2 spectre de predictibilite...", flush=True)
sel = np.sort(RNG.choice(x.shape[0], size=min(NT_SPEC, x.shape[0]), replace=False))
X = np.log1p(x[sel]); B = np.log1p(b[sel])          # espace d'entrainement
X = X - X.mean(axis=(1, 2), keepdims=True)
B = B - B.mean(axis=(1, 2), keepdims=True)
FX = np.fft.rfft2(X); FB = np.fft.rfft2(B)
H, W = X.shape[1], X.shape[2]
ky = np.fft.fftfreq(H)[:, None]; kx = np.fft.rfftfreq(W)[None, :]
kr = np.sqrt(ky ** 2 + kx ** 2)
nb = 40
kbin = np.clip((kr / kr.max() * nb).astype(int), 0, nb - 1)
Sxx = np.zeros(nb); Sbb = np.zeros(nb); Sxb = np.zeros(nb, complex)
for i in range(nb):
    msk = kbin == i
    if not msk.any():
        continue
    Sxx[i] = float(np.mean(np.abs(FX[:, msk]) ** 2))
    Sbb[i] = float(np.mean(np.abs(FB[:, msk]) ** 2))
    Sxb[i] = complex(np.mean(FX[:, msk] * np.conj(FB[:, msk])))
coh = np.abs(Sxb) ** 2 / np.maximum(Sxx * Sbb, 1e-30)
# echelle physique : 12 km par pixel HR
scale_km = np.array([(12.0 * W) / max(i, 0.5) / nb * 0.5 for i in range(nb)])
OUT["D2_spectrum"] = [dict(bin=i, k_rel=float(i / nb), scale_km=float(scale_km[i]),
                           rho2=float(coh[i]), psd_x=float(Sxx[i]))
                      for i in range(nb) if Sxx[i] > 0]

# =====================================================================
# D3 — PLAFOND DE F1@p99 POUR UN PREDICTEUR PARFAIT MAIS LISSE
#   + mecanique du biais CDD
# =====================================================================
print("[5/5] D3 plafond F1@p99 + CDD...", flush=True)
clim99 = np.quantile(x, 0.99, axis=0)                # seuil p99 par pixel
truth_ex = (x > clim99[None]) & valid[None]

def f1_of(pred):
    pe = (pred > clim99[None]) & valid[None]
    tp = float((pe & truth_ex).sum()); fp = float((pe & ~truth_ex).sum())
    fn = float((~pe & truth_ex).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)

def blur(a, sig):
    if sig <= 0:
        return a
    r = int(3 * sig); g = np.exp(-0.5 * (np.arange(-r, r + 1) / sig) ** 2); g /= g.sum()
    t = torch.from_numpy(a).unsqueeze(1)
    kx_ = torch.from_numpy(g.astype(np.float32)).view(1, 1, 1, -1)
    ky_ = torch.from_numpy(g.astype(np.float32)).view(1, 1, -1, 1)
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), kx_)
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), ky_)
    return t.squeeze(1).numpy()

res = [dict(predictor="verite exacte (sigma=0)", f1=f1_of(x))]
for sig in [1, 2, 4, 8]:
    res.append(dict(predictor=f"verite LISSEE sigma={sig}px ({sig*12} km)", f1=f1_of(blur(x, sig))))
res.append(dict(predictor="baseline (LR upsample)", f1=f1_of(b)))
OUT["D3_f1_ceiling"] = res

# --- mecanique CDD : seuiller une moyenne conditionnelle
occ = (xv > 1.0)
inten = xv[occ].mean() if occ.any() else 0.0
OUT["D3_cdd"] = dict(
    p_wet_global=float(occ.mean()),
    mean_wet_intensity=float(inten),
    implied_conditional_mean=float(occ.mean() * inten),
    note="si p*E[x|humide] < 1 mm/j, seuiller la moyenne conditionnelle compte le jour SEC",
)
# fraction de cas ou la baseline (lisse) rate un jour humide
miss = float(((bv <= 1.0) & (xv > 1.0)).sum()) / max(float((xv > 1.0).sum()), 1.0)
OUT["D3_cdd"]["frac_wet_days_missed_by_smooth_pred"] = miss

with open("results/diag_stage1.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("\n=== ECRIT results/diag_stage1.json ===", flush=True)
