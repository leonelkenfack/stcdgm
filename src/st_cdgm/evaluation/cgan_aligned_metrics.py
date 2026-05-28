"""cGAN-aligned evaluation metrics (Rampal 2024/2025).

These functions are **vendored verbatim** (semantics-preserving) from the cGAN
repository ``downscaling/src/analyse_experiments_src.py`` (class ``ValidationMetric``)
and ``downscaling/src/post_process_funcs.py`` (``psd``), so that the ST-CDGM /
Oracle model can be evaluated with the *exact same protocol* as the cGAN
baseline on the shared ACCESS-CM2 -> New Zealand 12 km benchmark.

The cGAN reports climate indices, NOT Pearson / F1:
  - CDD            : max consecutive dry days per year (pr <= thresh)
  - Rx1Day         : annual maximum 1-day precipitation
  - DJF / JJA mean : seasonal mean rainfall
  - R10            : count of days with pr > 10 mm per year
  - PSD            : radially-binned 2D power spectral density (172x179 grid)

All operate on raw mm/day precipitation (NOT log1p) in an ``xarray.Dataset``
with a ``pr`` variable and a ``time`` dimension plus ``lat/lon`` (or
``latitude/longitude``). Convert the model's log1p output with ``expm1`` first.

Dependencies: numpy, xarray, scipy. dask is optional (used only if the input
is dask-backed).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

try:
    import xarray as xr
except Exception as _e:  # pragma: no cover
    xr = None

from scipy.stats import binned_statistic


# ---------------------------------------------------------------------------
# Climate indices — vendored from cGAN ValidationMetric (semantics identical)
# ---------------------------------------------------------------------------
def consecutive_dry_days(ds, thresh: float = 1.0):
    """Max number of consecutive dry days (pr <= thresh) within each year.

    Verbatim port of ``ValidationMetric.consecutive_dry_days``.
    Returns a Dataset with variable ``cdd`` (dims: year, lat, lon).
    """
    def find_consecutive_true(arr):
        if ((arr.max() == 1) & (arr.min() == 0)) | (arr.min() == 1):
            arr = np.asarray(arr)
            idx = np.flatnonzero(
                np.concatenate(([arr[0]], arr[:-1] != arr[1:], [True]))
            )
            z = np.diff(idx)[::2]
            return np.max(z, axis=0)
        return 0.0

    test_data = ds.pr
    try:
        test_data = test_data.stack(z=["lat", "lon"]).dropna("z")
        _spatial = "latlon"
    except Exception:
        test_data = test_data.stack(z=["latitude", "longitude"]).dropna("z")
        _spatial = "latlon2"

    bool_arr = (test_data <= thresh).astype("int")
    consec = xr.apply_ufunc(
        find_consecutive_true,
        bool_arr.groupby("time.year"),
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        output_dtypes=[int],
        vectorize=True,
        dask="parallelized",
    )
    # compute() only if dask-backed; harmless otherwise.
    if hasattr(consec.data, "compute"):
        consec = consec.compute()
    consec = consec.unstack()
    try:
        consec = consec.reindex(lat=sorted(consec.lat.values))
    except Exception:
        consec = consec.reindex(longitude=sorted(consec.longitude.values))
    return consec.to_dataset().rename({"pr": "cdd"})


def rx1day(ds, thresh: float = 1.0):
    """Annual maximum 1-day precipitation. Verbatim port (``thresh`` unused, as
    in the original — kept for signature parity)."""
    return ds.groupby("time.year").max().rename({"pr": "rx1day"})


def seasonal_rainfall(ds):
    """DJF and JJA seasonal-mean rainfall. Verbatim port."""
    output = ds.groupby("time.season").mean()
    o1 = output.sel(season="DJF").drop_vars("season").rename({"pr": "DJF_rainfall"})
    o2 = output.sel(season="JJA").drop_vars("season").rename({"pr": "JJA_rainfall"})
    return xr.merge([o1, o2])


def r10day(ds):
    """Count of days with pr > 10 mm per year. Verbatim port."""
    return (ds > 10).groupby("time.year").sum().rename({"pr": "r10day"})


def psd(y: np.ndarray, bins: Optional[np.ndarray] = None,
        nx: int = 172, ny: int = 179):
    """Radially-binned 2D power spectral density. Verbatim port of
    ``post_process_funcs.psd`` (grid hard-coded 172x179 in the original).

    Parameters
    ----------
    y : np.ndarray
        ``(time, lat, lon)`` field (raw mm/day).
    bins : np.ndarray
        Wavenumber bin edges (default ``np.arange(0, 0.52, 0.02)``).
    nx, ny : int
        Grid dims used to build the frequency grid (default 172, 179).

    Returns
    -------
    np.ndarray
        ``(time, K)`` binned PSD (mean per wavenumber bin), time-averaged
        downstream by the caller.
    """
    if bins is None:
        bins = np.arange(0, 0.52, 0.02)
    y = np.nan_to_num(np.asarray(y, dtype=np.float64), nan=0.0)
    ffts = np.fft.fft2(y)
    ffts = np.fft.fftshift(abs(ffts) ** 2)
    freq = np.fft.fftshift(np.fft.fftfreq(nx))
    freq2 = np.fft.fftshift(np.fft.fftfreq(ny))
    kx, ky = np.meshgrid(freq, freq2)
    kx, ky = kx.T, ky.T
    kr = np.sqrt(kx.ravel() ** 2 + ky.ravel() ** 2)
    out = np.array([
        binned_statistic(kr, values=ffts[i].ravel(),
                         statistic="mean", bins=bins).statistic
        for i in range(ffts.shape[0])
    ])
    return out  # (time, n_bins-1)


# ---------------------------------------------------------------------------
# High-level comparison helpers
# ---------------------------------------------------------------------------
def _domain_mean(da) -> float:
    """Mean over all non-time dims, NaN-safe -> python float."""
    return float(np.nanmean(np.asarray(da.values, dtype=np.float64)))


def compute_aligned_indices(pred_ds, truth_ds, thresh: float = 1.0) -> Dict:
    """Compute the cGAN index suite for prediction and truth, plus biases.

    Both inputs are ``xarray.Dataset`` with a ``pr`` variable (raw mm/day),
    ``time``, and ``lat/lon`` (or ``latitude/longitude``). Returns a dict of
    domain-mean index values for pred & truth and the bias (pred - truth).
    """
    out: Dict[str, float] = {}
    index_fns = {
        "cdd": (consecutive_dry_days, dict(thresh=thresh), "cdd"),
        "rx1day": (rx1day, dict(thresh=thresh), "rx1day"),
        "r10day": (r10day, {}, "r10day"),
    }
    for name, (fn, kw, var) in index_fns.items():
        try:
            p = fn(pred_ds, **kw)[var]
            t = fn(truth_ds, **kw)[var]
            pv, tv = _domain_mean(p), _domain_mean(t)
            out[f"{name}_pred"] = pv
            out[f"{name}_truth"] = tv
            out[f"{name}_bias"] = pv - tv
        except Exception as e:
            out[f"{name}_error"] = str(e)

    # seasonal (two variables in one call)
    try:
        ps = seasonal_rainfall(pred_ds)
        ts = seasonal_rainfall(truth_ds)
        for var in ("DJF_rainfall", "JJA_rainfall"):
            pv, tv = _domain_mean(ps[var]), _domain_mean(ts[var])
            out[f"{var}_pred"] = pv
            out[f"{var}_truth"] = tv
            out[f"{var}_bias"] = pv - tv
    except Exception as e:
        out["seasonal_error"] = str(e)

    return out


def psd_distance(pred_da, truth_da, bins: Optional[np.ndarray] = None,
                 nx: int = 172, ny: int = 179) -> float:
    """Mean-squared distance between time-averaged log-PSD of pred vs truth.

    ``pred_da`` / ``truth_da`` are ``(time, lat, lon)`` arrays or DataArrays
    (raw mm/day). Lower = better spectral fidelity.
    """
    def _arr(x):
        return np.asarray(x.values if hasattr(x, "values") else x, dtype=np.float64)

    pp = psd(_arr(pred_da), bins=bins, nx=nx, ny=ny).mean(axis=0)
    tt = psd(_arr(truth_da), bins=bins, nx=nx, ny=ny).mean(axis=0)
    eps = 1e-12
    lp = np.log10(np.clip(pp, eps, None))
    lt = np.log10(np.clip(tt, eps, None))
    m = np.isfinite(lp) & np.isfinite(lt)
    if not m.any():
        return float("nan")
    return float(np.mean((lp[m] - lt[m]) ** 2))


__all__ = [
    "consecutive_dry_days", "rx1day", "seasonal_rainfall", "r10day", "psd",
    "compute_aligned_indices", "psd_distance",
]
