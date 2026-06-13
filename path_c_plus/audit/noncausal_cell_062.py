# >>> BS42 — DOMAIN-ALIGNED METRICS (CorrDiff / Rampal style, precipitation)
# Recompute the metrics the precipitation-downscaling literature actually
# reports (CRPS, spread-skill ratio, intensity-histogram distance), reusing the
# tensors already produced by the FINAL_VALIDATION cell. Pearson/RMSE kept only
# as SECONDARY references. CorrDiff (=ResDiff, Mardani 2024) reports CRPS/MAE/
# spectra, never Pearson; Rampal-style precip intercomparison uses MAE, LHD,
# RALSD/RAPSD, spread-skill. We align on those.
import math as _m42
import json as _j42
from pathlib import Path as _P42

print("=" * 72)
print("DOMAIN-ALIGNED METRICS (CorrDiff / Rampal - precipitation)")
print("=" * 72)

_dm = {}

# 1. Spread-skill ratio (ensemble calibration; target ~1.0).
try:
    _ssr = float(_spread) / float(_rmse) if (_rmse == _rmse and _rmse > 0) else float("nan")
except Exception:
    _ssr = float("nan")
_dm["spread_skill_ratio"] = _ssr

# 2. CRPS (Gaussian closed-form from ensemble mean + std). Gneiting & Raftery 2007.
#    CRPS(N(mu,s),y)=s[w(2Phi(w)-1)+2phi(w)-1/sqrt(pi)], w=(y-mu)/s.
_crps = float("nan")
try:
    _mu_c = _pred_full[_valid].double()
    _sd_c = _pred_std[_valid].double().clamp_min(1e-6)
    _y_c = _targets[_valid].double()
    _w_c = (_y_c - _mu_c) / _sd_c
    _Phi = 0.5 * (1.0 + torch.erf(_w_c / _m42.sqrt(2.0)))
    _phi = torch.exp(-0.5 * _w_c * _w_c) / _m42.sqrt(2.0 * _m42.pi)
    _crps_pix = _sd_c * (_w_c * (2.0 * _Phi - 1.0) + 2.0 * _phi - 1.0 / _m42.sqrt(_m42.pi))
    _crps = float(_crps_pix.mean().item())
except Exception as _e:
    print(f"[warn] CRPS failed: {_e}")
_dm["crps_gaussian"] = _crps

# 3. Intensity-histogram distance (proxy LHD): L1 between normalized value
#    histograms (log1p mm/day). 0 = identical marginal distribution.
_lhd = float("nan")
try:
    _p_h = _pred_full[_valid].double().flatten()
    _t_h = _targets[_valid].double().flatten()
    _lo = float(torch.minimum(_p_h.min(), _t_h.min()).item())
    _hi = float(torch.maximum(_p_h.max(), _t_h.max()).item())
    if _hi > _lo:
        _hp = torch.histc(_p_h.float(), bins=100, min=_lo, max=_hi)
        _ht = torch.histc(_t_h.float(), bins=100, min=_lo, max=_hi)
        _hp = _hp / _hp.sum().clamp_min(1.0)
        _ht = _ht / _ht.sum().clamp_min(1.0)
        _lhd = float((_hp - _ht).abs().sum().item())
except Exception as _e:
    print(f"[warn] histogram distance failed: {_e}")
_dm["intensity_hist_distance_L1"] = _lhd

# Reuse FINAL_VALIDATION metrics as references.
_dm["rapsd_distance"] = float(_rapsd_d) if _rapsd_d is not None else None
_dm["rmse_secondary"] = float(_rmse)
_dm["mae_secondary"] = float(_mae)
_dm["spread_mean"] = float(_spread)
_dm["pearson_global_secondary"] = float(_corr_global) if _corr_global == _corr_global else None

print(f"  Spread-skill ratio     : {_ssr:.4f}   (target ~1.0 ; <1 = under-dispersed)")
print(f"  CRPS (Gaussian approx) : {_crps:.6f}  (lower=better ; log1p mm/day)")
print(f"  Hist. distance (~LHD)  : {_lhd:.4f}   (0 = identical intensity dist.)")
print(f"  RAPSD / RALSD distance : {_dm['rapsd_distance']}")
print(f"  -- secondary (not headline) --")
print(f"  RMSE / MAE / Spread    : {_rmse:.4f} / {_mae:.4f} / {_spread:.4f}")
print(f"  Pearson (global)       : {_dm['pearson_global_secondary']}")
print()
print("  [note] Rx1Day / CDD (temporal indices) NOT computed here: they need a")
print("         continuous per-pixel time series, absent from the batched eval.")

try:
    _ckpt_dir = _P42(str(globals().get("CKPT_SAVE_DIR", "results")))
    _ckpt_dir.mkdir(parents=True, exist_ok=True)
    _out_path = _ckpt_dir / "domain_metrics.json"
    _out_path.write_text(_j42.dumps(_dm, indent=2), encoding="utf-8")
    print(f"\nDomain metrics saved: {_out_path}")
except Exception as _e:
    print(f"[warn] saving domain_metrics.json failed: {_e}")
