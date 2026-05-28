"""Bridge model predictions -> cGAN-aligned climate indices, and save JSON.

Usage (in a notebook, after the model is built/loaded and predictions over a
*continuous time series* have been collected):

    from st_cdgm.evaluation.aligned_eval import run_aligned_eval
    run_aligned_eval(
        pred_fields=pred_list,    # list/array of HR fields, one per timestep
        truth_fields=truth_list,  # same, ground truth
        times=times,              # 1-D array of datetime64, len == n timesteps
        out_path="ckpt_xxx/aligned_metrics_ACCESS-CM2_causal.json",
        gcm="ACCESS-CM2", run_variant="causal", in_distribution=True,
        space="log1p",            # fields are log1p(mm/day) -> expm1 applied
    )

The heavy lifting (CDD / Rx1Day / R10 / seasonal / PSD) is delegated to
``cgan_aligned_metrics`` (vendored verbatim from Rampal's cGAN), so the protocol
is byte-identical to the cGAN baseline on the shared ACCESS-CM2 -> NZ benchmark.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

try:
    import xarray as xr
except Exception:  # pragma: no cover
    xr = None

from .cgan_aligned_metrics import compute_aligned_indices, psd_distance


def _to_THW(fields) -> np.ndarray:
    """Coerce a list/array of fields to a clean (T, H, W) float64 array."""
    a = np.asarray([np.asarray(f) for f in fields], dtype=np.float64) \
        if isinstance(fields, (list, tuple)) else np.asarray(fields, dtype=np.float64)
    a = np.squeeze(a)
    if a.ndim == 4 and a.shape[1] == 1:   # (T,1,H,W)
        a = a[:, 0]
    if a.ndim != 3:
        raise ValueError(f"expected (T,H,W) after squeeze, got {a.shape}")
    return a


def _pr_dataset(arr_THW: np.ndarray, times) -> "xr.Dataset":
    """Build an xarray Dataset with var ``pr`` (mm/day), dims time/lat/lon.

    lat/lon are integer indices (the cGAN indices need real *time* for the
    year/season groupby, but only the grid *shape* for CDD/Rx1Day/R10; PSD uses
    the shape too). Real geo-coords are not required for these indices.
    """
    if xr is None:
        raise ImportError("xarray is required for run_aligned_eval")
    T, H, W = arr_THW.shape
    times = np.asarray(times)
    if times.shape[0] != T:
        raise ValueError(f"len(times)={times.shape[0]} != T={T}")
    return xr.Dataset(
        {"pr": (("time", "lat", "lon"), arr_THW.astype("float32"))},
        coords={"time": times, "lat": np.arange(H), "lon": np.arange(W)},
    )


def run_aligned_eval(
    *,
    pred_fields,
    truth_fields,
    times,
    out_path: Union[str, Path],
    gcm: str,
    run_variant: str,
    in_distribution: bool,
    space: str = "log1p",
    thresh: float = 1.0,
    k_samples: Optional[int] = None,
    psd_nx: int = 172,
    psd_ny: int = 179,
) -> dict:
    """Compute the cGAN-aligned index suite + PSD distance and save JSON.

    Parameters
    ----------
    pred_fields, truth_fields :
        HR fields over a continuous time series (T,H,W) or (T,1,H,W).
    times :
        datetime64 array of length T (drives year/season groupby).
    space :
        "log1p" (default) -> ``expm1`` is applied to recover mm/day ;
        "raw" -> used as-is.
    thresh :
        Dry/wet-day threshold (mm/day). cGAN default = 1.0.

    Returns
    -------
    dict (also written to ``out_path``).
    """
    pred = _to_THW(pred_fields)
    truth = _to_THW(truth_fields)
    if space == "log1p":
        pred = np.expm1(pred)
        truth = np.expm1(truth)
    elif space != "raw":
        raise ValueError("space must be 'log1p' or 'raw'")
    # Negative precip is unphysical (sampling noise) -> clip at 0.
    pred = np.clip(pred, 0.0, None)
    truth = np.clip(truth, 0.0, None)

    pred_ds = _pr_dataset(pred, times)
    truth_ds = _pr_dataset(truth, times)

    indices = compute_aligned_indices(pred_ds, truth_ds, thresh=thresh)
    psd_d = psd_distance(pred, truth, nx=psd_nx, ny=psd_ny)

    result = {
        "gcm": gcm,
        "run_variant": run_variant,
        "in_distribution": bool(in_distribution),
        "protocol": "cgan_rampal_vendored",
        "thresh_mm": thresh,
        "n_days": int(pred.shape[0]),
        "grid": [int(pred.shape[1]), int(pred.shape[2])],
        "k_samples": k_samples,
        "indices": indices,
        "psd_distance": psd_d,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    return result


__all__ = ["run_aligned_eval"]
