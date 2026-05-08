"""
Download ERA5 reanalysis predictors for Bénin (1974-2011 daily) — Phase 0-A.

Why
---
Closing the paper-gap requires reanalysis-quality conditioning rather
than free-running GCM (ACCESS-CM2). The ST-CDGM paper trains on ERA5
1974-2011 (oracle.tex L1503); we currently train on ACCESS-CM2 which
has documented zero/negative day-to-day correlation with CHIRPS over
West Africa (CMIP6 evaluation studies, MDPI Atmosphere 2023).

What this downloads
-------------------
* Pressure levels (850, 500, 250 hPa):
  - u, v        (zonal/meridional wind)
  - t           (air temperature)
  - q           (specific humidity)
  - z           (geopotential)

* Single-level surface fields:
  - sp          (surface pressure)
  - 2t          (2 m air temperature)
  - tcwv        (total column water vapour)

* Time: daily averages 1974-01-01 → 2011-12-31 (full paper window).
* Region: lat ∈ [4, 14], lon ∈ [-2, 6] (Bénin domain + ~200 km halo
  matching the LR grid 23×26 in oracle.tex Tab.1).

The output is a single netCDF ready to drop into
``config/training_config.yaml`` as
``data.lr_path: data/raw/train/predictor_ERA5_hist.nc``.

Prerequisites
-------------
1. ECMWF CDS account (https://cds.climate.copernicus.eu, free).
2. ``~/.cdsapirc`` configured per
   https://cds.climate.copernicus.eu/how-to-api .
3. ``pip install cdsapi xarray netcdf4 pandas``.

Estimated size : ~10 GB raw download → ~2 GB after daily aggregation
                  and trimming.
Estimated time : 6-12 h depending on CDS queue (it queues by request,
                  so we issue per-year requests to parallelise the
                  queue and resume on failure).

Usage
-----
    python scripts/download_era5_benin.py \\
        --out data/raw/train/predictor_ERA5_hist.nc \\
        --years 1974-2011

    # Dry-run: print the requests but do not submit
    python scripts/download_era5_benin.py --dry-run --years 1974-1975

Resume
------
If a year has already been fetched into ``--cache-dir``, it is skipped.
Re-running after Ctrl-C continues where it stopped.

References
----------
- Hersbach et al. 2020, *QJRMS* (ERA5 reanalysis description).
- Aich et al. 2024, arXiv:2404.14416 (ESM-vs-reanalysis bias diagnosis;
  motivation for ERA5 conditioning).
- ST-CDGM oracle.tex L1503 (training data spec).
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("era5-download")


# ---------------------------------------------------------------------
# Configuration — matches oracle.tex Sec. "Predictors and target"
# ---------------------------------------------------------------------

# Domain (lat_min, lat_max, lon_min, lon_max). Centred on Bénin with a
# halo to match the paper's 23x26 LR grid at 0.25° (≈ 5.75x6.5 deg).
BBOX = {
    "north": 14.0,
    "south": 4.0,
    "west": -2.0,
    "east": 6.0,
}

PRESSURE_LEVELS = ["850", "500", "250"]
PLEV_VARIABLES = [
    "u_component_of_wind",
    "v_component_of_wind",
    "temperature",
    "specific_humidity",
    "geopotential",
]

SLEV_VARIABLES = [
    "surface_pressure",
    "2m_temperature",
    "total_column_water_vapour",
]

# 24 hourly steps → daily mean post-fetch
HOURS = [f"{h:02d}:00" for h in range(0, 24)]


# ---------------------------------------------------------------------
# Per-year request submission
# ---------------------------------------------------------------------

def _build_request_plev(year: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "format": "netcdf",
        "variable": PLEV_VARIABLES,
        "pressure_level": PRESSURE_LEVELS,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": HOURS,
        "area": [BBOX["north"], BBOX["west"], BBOX["south"], BBOX["east"]],
    }


def _build_request_slev(year: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "format": "netcdf",
        "variable": SLEV_VARIABLES,
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": HOURS,
        "area": [BBOX["north"], BBOX["west"], BBOX["south"], BBOX["east"]],
    }


def fetch_year(client, year: int, cache_dir: Path, dry_run: bool) -> tuple[Path, Path]:
    plev_path = cache_dir / f"era5_plev_{year}.nc"
    slev_path = cache_dir / f"era5_slev_{year}.nc"

    if not plev_path.exists():
        req = _build_request_plev(year)
        if dry_run:
            LOG.info("[DRY-RUN] plev %d → %s", year, plev_path.name)
        else:
            LOG.info("Fetching ERA5 pressure levels %d ...", year)
            client.retrieve(
                "reanalysis-era5-pressure-levels",
                req,
                str(plev_path),
            )
    else:
        LOG.info("plev %d cached, skip", year)

    if not slev_path.exists():
        req = _build_request_slev(year)
        if dry_run:
            LOG.info("[DRY-RUN] slev %d → %s", year, slev_path.name)
        else:
            LOG.info("Fetching ERA5 single levels %d ...", year)
            client.retrieve(
                "reanalysis-era5-single-levels",
                req,
                str(slev_path),
            )
    else:
        LOG.info("slev %d cached, skip", year)

    return plev_path, slev_path


# ---------------------------------------------------------------------
# Post-processing : daily mean + variable rename + concat
# ---------------------------------------------------------------------

def daily_aggregate(year: int, plev_path: Path, slev_path: Path):
    """Returns one xarray.Dataset for the given year, daily-aggregated.

    Variables are renamed to match the codebase convention used by
    ``predictor_ACCESS-CM2_hist.nc`` (e.g. u850, v850, t850, q850, z850,
    u500, ..., t2m, sp, tcwv). This means a swap of ``data.lr_path`` is
    sufficient — no rename required downstream.
    """
    import xarray as xr

    LOG.info("Aggregating year %d ...", year)
    ds_plev = xr.open_dataset(plev_path)
    ds_slev = xr.open_dataset(slev_path)

    # Daily mean over time axis. CDS files use 'time' or 'valid_time'.
    time_dim = "valid_time" if "valid_time" in ds_plev.dims else "time"

    ds_plev = ds_plev.resample({time_dim: "1D"}).mean()
    ds_slev = ds_slev.resample({time_dim: "1D"}).mean()

    # Rename pressure-level variables: e.g. (level=850, var='u') → 'u850'.
    plev_renamed = xr.Dataset(coords=ds_plev.coords)
    for var in ["u", "v", "t", "q", "z"]:
        if var not in ds_plev:
            continue
        for lev in [int(l) for l in PRESSURE_LEVELS]:
            arr = ds_plev[var].sel(pressure_level=lev, drop=True)
            plev_renamed[f"{var}{lev}"] = arr
    plev_renamed = plev_renamed.drop_vars(
        [c for c in ("pressure_level",) if c in plev_renamed.coords],
        errors="ignore",
    )

    slev_renamed = xr.Dataset(coords=ds_slev.coords)
    rename_map = {"sp": "sp", "t2m": "t2m", "tcwv": "tcwv"}
    for src, dst in rename_map.items():
        if src in ds_slev:
            slev_renamed[dst] = ds_slev[src]

    merged = xr.merge([plev_renamed, slev_renamed])
    if time_dim != "time":
        merged = merged.rename({time_dim: "time"})
    return merged


def concat_years(years_data, out_path: Path) -> None:
    import xarray as xr

    LOG.info("Concatenating %d years and writing %s ...", len(years_data), out_path)
    full = xr.concat(years_data, dim="time")
    full = full.sortby("time")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = {v: {"zlib": True, "complevel": 4} for v in full.data_vars}
    full.to_netcdf(out_path, encoding=encoding)
    size_mb = out_path.stat().st_size / 1e6
    LOG.info("Wrote %s (%.0f MB, %d time steps)",
             out_path.name, size_mb, full.sizes["time"])


# ---------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------

def parse_year_range(spec: str) -> list[int]:
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in spec.split(",")]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/train/predictor_ERA5_hist.nc"),
        help="Output netCDF path (default mirrors ACCESS-CM2 location).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/raw/train/era5_cache"),
        help="Per-year intermediate netCDFs; allows resume after Ctrl-C.",
    )
    parser.add_argument(
        "--years",
        default="1974-2011",
        help="Year range (e.g. '1974-2011') or comma list (e.g. '1990,2000').",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the requests but do not submit to CDS.",
    )
    parser.add_argument(
        "--skip-aggregate",
        action="store_true",
        help="Only download per-year files; skip the merge/aggregate step.",
    )
    args = parser.parse_args()

    years = parse_year_range(args.years)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Years: %s (%d)", years[0] if len(years) == 1 else f"{years[0]}-{years[-1]}", len(years))
    LOG.info("Output: %s", args.out)
    LOG.info("BBox: lat[%g,%g] lon[%g,%g]",
             BBOX["south"], BBOX["north"], BBOX["west"], BBOX["east"])

    # Lazy import so --dry-run works without cdsapi installed.
    if not args.dry_run:
        try:
            import cdsapi  # type: ignore
        except ImportError:
            LOG.error("cdsapi not installed. Run: pip install cdsapi")
            return 1
        client = cdsapi.Client()
    else:
        client = None

    fetched = []
    for year in years:
        try:
            plev_path, slev_path = fetch_year(client, year, args.cache_dir, args.dry_run)
            fetched.append((year, plev_path, slev_path))
        except Exception as exc:
            LOG.error("Year %d failed: %s — continuing", year, exc)
            continue

    if args.dry_run or args.skip_aggregate:
        LOG.info("Dry-run / skip-aggregate done; %d year-requests staged.", len(fetched))
        return 0

    years_data = []
    for year, plev_path, slev_path in fetched:
        try:
            ds = daily_aggregate(year, plev_path, slev_path)
            years_data.append(ds)
        except Exception as exc:
            LOG.error("Aggregation %d failed: %s", year, exc)

    if not years_data:
        LOG.error("No years aggregated successfully — abort.")
        return 1

    concat_years(years_data, args.out)
    LOG.info("Done. Update training_config.yaml: data.lr_path: %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
