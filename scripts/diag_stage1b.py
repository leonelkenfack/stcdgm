"""Confirmation : isoler la contribution de CHAQUE source de lissage au F1@p99.

Modele du pipeline Oracle :
    mu_HR = b + alpha * ( LP_sigma(x) - b )
ou LP_sigma = lissage du a la grille intermediaire 43x45 + bilineaire du decodeur,
et alpha = gate du ConditionalSkipBlock.

On balaie (sigma, alpha) et on compare aux valeurs MESUREES :
    Oracle 0.5123   CorrDiff 0.5504
"""
import numpy as np, xarray as xr, torch, torch.nn.functional as F, json

RNG = np.random.default_rng(0); NT = 4000
ds = xr.open_dataset("data/raw/train/pr_ACCESS-CM2_hist.nc", decode_times=False)
idx = np.sort(RNG.choice(ds.sizes["time"], size=NT, replace=False))
x = np.nan_to_num(ds["pr"].isel(time=idx).values.astype(np.float32), nan=0.0)
ds.close()

xt = torch.from_numpy(x).unsqueeze(1)
b = F.interpolate(F.adaptive_avg_pool2d(xt, (23, 26)), size=(172, 179),
                  mode="bilinear", align_corners=False).squeeze(1).numpy()
clim99 = np.quantile(x, 0.99, axis=0); truth = x > clim99[None]

def f1(pred):
    pe = pred > clim99[None]
    tp = float((pe & truth).sum()); fp = float((pe & ~truth).sum()); fn = float((~pe & truth).sum())
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)

def blur(a, s):
    if s <= 0: return a
    r = int(3 * s); g = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2); g /= g.sum()
    t = torch.from_numpy(a).unsqueeze(1)
    kxx = torch.from_numpy(g.astype(np.float32)).view(1, 1, 1, -1)
    kyy = torch.from_numpy(g.astype(np.float32)).view(1, 1, -1, 1)
    t = F.conv2d(F.pad(t, (r, r, 0, 0), mode="replicate"), kxx)
    t = F.conv2d(F.pad(t, (0, 0, r, r), mode="replicate"), kyy)
    return t.squeeze(1).numpy()

OUT = {"measured": {"oracle_f1p99": 0.5123, "corrdiff_f1p99": 0.5504}}

# --- grille (sigma, alpha) ---
grid = []
for sig in [0, 1, 2, 3, 4, 6, 8]:
    xb = blur(x, sig)
    for al in [1.0, 0.88, 0.6]:
        pred = b + al * (xb - b)
        grid.append(dict(sigma_px=sig, scale_km=sig * 12, alpha=al, f1=f1(pred)))
        print(f"sigma={sig:>2d}px ({sig*12:>3d} km)  alpha={al:4.2f}  F1@p99={grid[-1]['f1']:.4f}", flush=True)
OUT["grid"] = grid

# --- fraction de variance HR au-dessus de 48 km (ce que la grille 43x45 ne peut pas porter)
sel = np.sort(RNG.choice(NT, size=400, replace=False))
X = np.log1p(x[sel]); X = X - X.mean(axis=(1, 2), keepdims=True)
P = np.abs(np.fft.rfft2(X)) ** 2
H, W = X.shape[1], X.shape[2]
ky = np.fft.fftfreq(H)[:, None]; kx = np.fft.rfftfreq(W)[None, :]
lam_km = 12.0 / np.maximum(np.sqrt(ky ** 2 + kx ** 2), 1e-9)   # longueur d'onde
tot = float(P.sum())
for cut in [96, 48, 24, 12]:
    frac = float(P[:, lam_km < cut].sum()) / tot
    OUT[f"var_frac_below_{cut}km"] = frac
    print(f"  fraction de variance HR aux echelles < {cut:>3d} km : {100*frac:5.1f} %", flush=True)

# --- CDD : mecanique par pixel (regions seches)
wet = x > 1.0
p_wet = wet.mean(axis=0)
inten = np.where(wet, x, np.nan)
mean_wet = np.nanmean(inten, axis=0)
cond_mean = p_wet * np.nan_to_num(mean_wet)
OUT["cdd_pixels_cond_mean_below_1mm"] = float((cond_mean < 1.0).mean())
OUT["cdd_p_wet_p10"] = float(np.quantile(p_wet, 0.10))
OUT["cdd_cond_mean_p10"] = float(np.quantile(cond_mean, 0.10))
print(f"  pixels ou E[x|.] < 1 mm/j (=> comptes secs) : {100*OUT['cdd_pixels_cond_mean_below_1mm']:.1f} %", flush=True)
print(f"  decile sec : P(humide)={OUT['cdd_p_wet_p10']:.3f}  moyenne cond.={OUT['cdd_cond_mean_p10']:.3f} mm/j", flush=True)

json.dump(OUT, open("results/diag_stage1b.json", "w"), indent=2)
print("\n=== ECRIT results/diag_stage1b.json ===")
