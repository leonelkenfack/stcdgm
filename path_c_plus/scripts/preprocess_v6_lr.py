"""
V6 MVP — preprocessing for augmented LR features + V6 wind nodes (U850, V850).

Produces a NetCDF augmented LR dataset that the V6 pipeline will consume in
``Stage 1.A REFAIT`` (11-node graph + 21 LR variables). The 6 obligatory
features come directly from the Climat expert (V6 plan §1.2). The optional
``theta_w_850`` (§4.4) is also computed.

Inputs
------
- ``lr_path``  : input LR dataset (NetCDF, ACCESS-CM2 or other GCM)
- ``static_hr_path`` : HR static dataset (topography, land-sea mask)

Outputs
-------
- ``out_lr_augmented`` : NetCDF with original LR vars + 6 V6 features
- ``out_static_hr_v6`` : NetCDF with HR static + ``mask_ocean`` + ``grad_orog_HR``

V6 features computed
--------------------
1. ``w_700``          : 0.571·w_850 + 0.429·w_500 (Holton 2004 interp pondérée pression)
2. ``theta_e_850``    : Bolton 1980 simplified, q observed
3. ``theta_e_500``    : idem 500 hPa
4. ``mucape_proxy``   : theta_e_850 − theta_e_500
5. ``T_850_minus_T_500`` : LR-pure static stability (replaces ancient foehn_proxy
                          which had target leakage)
6. ``u850_grad_orog_HR`` : foehn dynamique LR×HR coherent —
                          ``u850_HR_bilin · ∇h_HR_4km`` with ``mask_ocean``
                          (Elvidge-Renfrew 2016 BAMS 97:455)
7. (optional §4.4) ``theta_w_850`` : wet-bulb potential temperature
                                    (Browning 2004 QJRMS warm conveyor belt)

References
----------
- Plan : path_c_plus/audit/PLAN_V6_BOOST_UNET.md §1.2
- Climat verbatim rondes 1-4
- Phase 8 Cell 5 (original formulas for w_700 / theta_e / mucape) ported here.

Usage
-----
    python path_c_plus/scripts/preprocess_v6_lr.py \\
        --lr-path /content/drive/.../lr_ACCESS-CM2.nc \\
        --static-hr-path /content/drive/.../static_HR.nc \\
        --out-lr-augmented /content/drive/.../lr_ACCESS-CM2_v6.nc \\
        --out-static-hr-v6 /content/drive/.../static_HR_v6.nc \\
        [--theta-w-850]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


# --------------------------------------------------------------------------- #
# Physical constants
# --------------------------------------------------------------------------- #
EPS = 1e-6
CP = 1004.5         # J/(kg·K)
LV = 2.501e6        # J/kg (latent heat of vaporization at 0°C)
RD = 287.04         # J/(kg·K) dry gas constant
RV = 461.5          # J/(kg·K) water vapor gas constant
KAPPA = RD / CP     # ≈ 0.286


# --------------------------------------------------------------------------- #
# Feature 1 — w_700 (Holton 2004 interp pondérée pression)
# --------------------------------------------------------------------------- #
def compute_w_700(w_850: xr.DataArray, w_500: xr.DataArray) -> xr.DataArray:
    """Holton 2004 weighted interpolation in pressure :
        w_700 = 0.571 · w_850 + 0.429 · w_500
    """
    w_700 = 0.571 * w_850 + 0.429 * w_500
    w_700.attrs = {
        "long_name": "vertical_velocity_at_700hPa_interpolated",
        "units": "Pa s-1",
        "interpolation": "0.571*w_850 + 0.429*w_500 (Holton 2004 linear in pressure)",
        "source": "V6 MVP preprocess",
    }
    return w_700


# --------------------------------------------------------------------------- #
# Feature 2-3 — theta_e at 850, 500 hPa (Bolton 1980 simplified, observed q)
# --------------------------------------------------------------------------- #
def _theta_e_bolton_1980(
    T_K: xr.DataArray, q: xr.DataArray, p_hPa: float
) -> xr.DataArray:
    """Equivalent potential temperature, Bolton (1980) simplified with observed q.

    Eq 39, with mixing ratio r ≈ q / (1 − q) :
        T_L = 1 / (1/(T − 55) − ln(RH)/2840) + 55
        θ_e = T (1000/p)^(0.2854·(1 − 0.28·r)) · exp((3376/T_L − 2.54)·r·(1 + 0.81·r))
    """
    # Clip q at a PHYSICALLY realistic upper bound (~0.05 in stratosphere/tropics)
    # rather than (1-EPS), to avoid exp() overflow on aberrant GCM values.
    # Atmospheric specific humidity is < 0.04 in 99.9% of cases.
    Q_MAX_PHYSICAL = 0.05
    q_safe = q.clip(min=EPS, max=Q_MAX_PHYSICAL)
    r = q_safe / (1.0 - q_safe)
    # Saturation vapor pressure (Wexler 1976 approximation for simplicity)
    es = 6.112 * np.exp(17.67 * (T_K - 273.15) / (T_K - 29.65))
    # Actual vapor pressure from mixing ratio
    e = r * p_hPa / (0.622 + r)
    RH = (e / es).clip(min=EPS, max=1.0)
    # Lifting condensation level temperature (Bolton Eq 22)
    T_L = 1.0 / (1.0 / (T_K - 55.0) - np.log(RH) / 2840.0) + 55.0
    # Theta_e (Bolton Eq 39 simplified) — additional safety cap at 500K
    # (physical theta_e is < 400K in all atmospheric conditions, so 500K = ample margin)
    theta_e = (
        T_K
        * (1000.0 / p_hPa) ** (0.2854 * (1.0 - 0.28 * r))
        * np.exp((3376.0 / T_L - 2.54) * r * (1.0 + 0.81 * r))
    )
    theta_e = theta_e.clip(min=200.0, max=500.0)
    return theta_e


def compute_theta_e_at(
    T_K: xr.DataArray, q: xr.DataArray, p_hPa: float, level_label: str
) -> xr.DataArray:
    theta_e = _theta_e_bolton_1980(T_K, q, p_hPa)
    theta_e.attrs = {
        "long_name": f"equivalent_potential_temperature_{level_label}",
        "units": "K",
        "formula": "Bolton 1980 Eq 39 simplified with observed q",
        "p_hPa": p_hPa,
        "source": "V6 MVP preprocess",
    }
    return theta_e


# --------------------------------------------------------------------------- #
# Feature 4 — mucape_proxy
# --------------------------------------------------------------------------- #
def compute_mucape_proxy(theta_e_850: xr.DataArray, theta_e_500: xr.DataArray) -> xr.DataArray:
    mucape = theta_e_850 - theta_e_500
    mucape.attrs = {
        "long_name": "MUCAPE_proxy",
        "units": "K",
        "formula": "theta_e_850 - theta_e_500",
        "source": "V6 MVP preprocess (proxy CAPE Climat verbatim)",
    }
    return mucape


# --------------------------------------------------------------------------- #
# Feature 5 — T_850 minus T_500 (LR-pure static stability)
# --------------------------------------------------------------------------- #
def compute_T_diff(T_850: xr.DataArray, T_500: xr.DataArray) -> xr.DataArray:
    out = T_850 - T_500
    out.attrs = {
        "long_name": "T_850_minus_T_500_static_stability_LR_pure",
        "units": "K",
        "formula": "T_850 - T_500",
        "note": "Climat ronde 2 — replaces ancient foehn_proxy=T_850-T_surf_HR (target leakage)",
        "source": "V6 MVP preprocess",
    }
    return out


# --------------------------------------------------------------------------- #
# Feature 6 — u850·∇h_HR (foehn dynamique, résolution-cohérent)
# --------------------------------------------------------------------------- #
def compute_grad_orography_HR(orog_HR: xr.DataArray, mask_ocean: xr.DataArray) -> xr.DataArray:
    """Compute ∇h_HR magnitude with ocean mask applied.

    Climat ronde 4 : ``mask_ocean`` (where land=1, ocean=0) avoids aberrant
    gradient at coast.

    UNIT WARNING : if coords are in degrees, the output is in m / deg (not
    m / m). For a physical gradient (dimensionless slope), the caller should
    multiply by ~1 / (111000 m / deg) for lat ; lon scaling depends on cos(lat).
    The pipeline normalizes features downstream so this scaling is absorbed,
    BUT do not interpret raw values as ``tan(slope)`` without conversion.
    """
    # Use xarray differentiate (central differences in lat/lon)
    # Assume orog_HR has lat, lon dims with metres values.
    # mask_ocean broadcast : 1 on land, 0 on ocean.
    # We compute grad on the masked orography (ocean→0 so coastal grad is dominated by land).
    orog_land = orog_HR * mask_ocean
    dh_dlat = orog_land.differentiate("lat") if "lat" in orog_land.dims else orog_land.differentiate("y")
    dh_dlon = orog_land.differentiate("lon") if "lon" in orog_land.dims else orog_land.differentiate("x")
    grad_mag = np.sqrt(dh_dlat ** 2 + dh_dlon ** 2) * mask_ocean
    grad_mag.attrs = {
        "long_name": "magnitude_orography_gradient_HR_masked_ocean",
        "units": "m / deg if coords in degrees ELSE m / grid_unit",
        "physical_unit_note": "Multiply by ~1/111000 (m/deg) to get tan(slope) for lat. Pipeline normalizes.",
        "source": "V6 MVP preprocess — Climat ronde 4 mask_ocean fix",
    }
    return grad_mag


def compute_u850_grad_orog_HR(
    u850_LR: xr.DataArray,
    grad_orog_HR: xr.DataArray,
    target_hr_shape: tuple[int, int] | None = None,
) -> xr.DataArray:
    """Resolution-coherent foehn dynamique :
        u850_HR_bilin (interp 4km) · ∇h_HR_4km

    Climat ronde 4 : ``u850_HR_bilin · ∇h_HR_4km``, sinon scale mismatch
    (Elvidge-Renfrew 2016 BAMS 97:455).

    Parameters
    ----------
    u850_LR : DataArray
        LR u-wind at 850 hPa.
    grad_orog_HR : DataArray
        HR orography gradient magnitude (with ocean masked).
    target_hr_shape : (H, W) optional override; otherwise inferred from grad_orog_HR.
    """
    # Bilinear interp u850_LR to HR grid (using grad_orog_HR's coords as target).
    u850_HR = u850_LR.interp_like(grad_orog_HR, method="linear")
    out = u850_HR * grad_orog_HR
    out.attrs = {
        "long_name": "u850_HR_bilin_times_grad_orog_HR_4km_foehn_dynamique",
        "units": "m/s · (orography grad units)",
        "formula": "(u850 bilin-interp HR) · ∇h_HR_4km",
        "reference": "Elvidge-Renfrew 2016 BAMS 97:455 — resolution-coherent",
        "source": "V6 MVP preprocess",
    }
    return out


# --------------------------------------------------------------------------- #
# Optional Feature 7 (§4.4) — theta_w_850 (Browning 2004 warm conveyor belt)
# --------------------------------------------------------------------------- #
def compute_theta_w_850(T_K: xr.DataArray, q: xr.DataArray) -> xr.DataArray:
    """Wet-bulb potential temperature at 850 hPa.

    Davies-Jones 2008 approximation : theta_w ≈ theta_e − A · (1 − exp(-B·r))
    with A ≈ 36, B ≈ 100 (simplified). Suffices as proxy.
    """
    p_hPa = 850.0
    theta_e = _theta_e_bolton_1980(T_K, q, p_hPa)
    q_safe = q.clip(min=EPS, max=1.0 - EPS)
    r = q_safe / (1.0 - q_safe)
    theta_w = theta_e - 36.0 * (1.0 - np.exp(-100.0 * r))
    theta_w.attrs = {
        "long_name": "wet_bulb_potential_temperature_850hPa",
        "units": "K",
        "formula": "Davies-Jones 2008 simplified approx of theta_w from theta_e",
        "reference": "Browning 2004 QJRMS warm conveyor belt",
        "source": "V6 MVP preprocess optional §4.4",
    }
    return theta_w


# --------------------------------------------------------------------------- #
# Region masks (West / East / North / South NZ) — for r_φ broadcast (S1.1)
# --------------------------------------------------------------------------- #
def build_nz_region_masks(static_hr: xr.Dataset, n_regions: int = 4) -> np.ndarray:
    """Build 4 regional masks (West, East, North, South) for r_φ broadcast.

    Simple geographic partition over NIWA-REMS NZ 172×179 grid based on the
    median latitude / longitude. A more refined Trenberth partition can be
    plugged here later (V6.1).

    Returns
    -------
    np.ndarray
        Shape [n_regions, H, W] with values in {0, 1} — disjoint masks summing
        to 1 everywhere on land (ocean handled by mask_ocean separately).
    """
    # Get lat/lon dims (or fallback to y/x)
    lat_name = "lat" if "lat" in static_hr.dims else "y"
    lon_name = "lon" if "lon" in static_hr.dims else "x"
    H = static_hr.sizes[lat_name]
    W = static_hr.sizes[lon_name]

    lat_vals = static_hr[lat_name].values
    lon_vals = static_hr[lon_name].values

    lat_med = np.median(lat_vals)
    lon_med = np.median(lon_vals)

    masks = np.zeros((n_regions, H, W), dtype=np.float32)
    LAT_grid, LON_grid = np.meshgrid(lat_vals, lon_vals, indexing="ij")
    # Convention : NZ is in S hemisphere, lat is negative → North = lat > median
    is_north = LAT_grid > lat_med
    is_west = LON_grid < lon_med
    # West Coast = SW (Southern Alps West)
    masks[0] = (~is_north) & is_west
    # East Coast = SE
    masks[1] = (~is_north) & (~is_west)
    # Northland = N
    masks[2] = is_north & (~is_west)
    # NW (Northland West side)
    masks[3] = is_north & is_west
    return masks.astype(np.float32)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def main(args: argparse.Namespace) -> None:
    print(f"[V6 preproc] Loading LR : {args.lr_path}")
    ds_lr = xr.open_dataset(args.lr_path)
    print(f"[V6 preproc] LR variables : {list(ds_lr.data_vars)[:10]}{'...' if len(ds_lr.data_vars) > 10 else ''}")

    print(f"[V6 preproc] Loading static HR : {args.static_hr_path}")
    ds_static_hr = xr.open_dataset(args.static_hr_path)

    # Resolve variable names robustly (different GCMs use different names)
    var_map = {
        "w_850": args.w_850_name,
        "w_500": args.w_500_name,
        "T_850": args.t_850_name,
        "T_500": args.t_500_name,
        "q_850": args.q_850_name,
        "q_500": args.q_500_name,
        "u_850": args.u_850_name,
        "v_850": args.v_850_name,
        "orog": args.orog_name,
    }
    for canonical, name in var_map.items():
        if name not in ds_lr.data_vars and name not in ds_static_hr.data_vars:
            raise RuntimeError(f"Variable {canonical!r}={name!r} not found in inputs.")

    # ----- Compute V6 features -----
    print("[V6 preproc] Feature 1 : w_700")
    w_700 = compute_w_700(ds_lr[args.w_850_name], ds_lr[args.w_500_name])

    print("[V6 preproc] Feature 2 : theta_e_850")
    theta_e_850 = compute_theta_e_at(
        ds_lr[args.t_850_name], ds_lr[args.q_850_name], p_hPa=850.0, level_label="850hPa"
    )

    print("[V6 preproc] Feature 3 : theta_e_500")
    theta_e_500 = compute_theta_e_at(
        ds_lr[args.t_500_name], ds_lr[args.q_500_name], p_hPa=500.0, level_label="500hPa"
    )

    print("[V6 preproc] Feature 4 : mucape_proxy")
    mucape = compute_mucape_proxy(theta_e_850, theta_e_500)

    print("[V6 preproc] Feature 5 : T_850 − T_500")
    t_diff = compute_T_diff(ds_lr[args.t_850_name], ds_lr[args.t_500_name])

    # ----- mask_ocean + ∇h_HR (HR static) -----
    print("[V6 preproc] mask_ocean (Climat ronde 4)")
    if args.land_sea_mask_name in ds_static_hr.data_vars:
        mask_ocean = ds_static_hr[args.land_sea_mask_name].astype("float32")
    else:
        # Fallback : threshold orography > 0 as land
        print(f"  WARN : '{args.land_sea_mask_name}' not in static_HR — deriving from orog>0")
        mask_ocean = (ds_static_hr[args.orog_name] > 0.0).astype("float32")
    mask_ocean.attrs = {"long_name": "land_mask_1land_0ocean", "source": "V6 MVP preproc"}

    print("[V6 preproc] ∇h_HR with mask_ocean")
    grad_orog_HR = compute_grad_orography_HR(ds_static_hr[args.orog_name], mask_ocean)

    print("[V6 preproc] Feature 6 : u850_HR_bilin · ∇h_HR_4km (Climat ronde 4)")
    u850_grad = compute_u850_grad_orog_HR(ds_lr[args.u_850_name], grad_orog_HR)

    # ----- Optional Feature 7 : theta_w_850 -----
    theta_w_850 = None
    if args.theta_w_850:
        print("[V6 preproc] Feature 7 (optional §4.4) : theta_w_850")
        theta_w_850 = compute_theta_w_850(ds_lr[args.t_850_name], ds_lr[args.q_850_name])

    # ----- Region masks (West/East/North/South NZ) for r_φ broadcast -----
    print("[V6 preproc] Region masks for r_φ (S1.1 StructuredResidualHead)")
    region_masks = build_nz_region_masks(ds_static_hr)
    region_masks_da = xr.DataArray(
        region_masks,
        dims=("region", *ds_static_hr[args.orog_name].dims),
        attrs={
            "long_name": "regional_masks_for_r_phi_broadcast",
            "regions": "West/East/North/NW NZ (4 quadrants by median lat/lon)",
            "source": "V6 MVP preproc S1.1",
        },
    )

    # ----- Write LR augmented (original + 6 V6 features) -----
    ds_lr_aug = ds_lr.copy()
    ds_lr_aug["w_700"] = w_700
    ds_lr_aug["theta_e_850"] = theta_e_850
    ds_lr_aug["theta_e_500"] = theta_e_500
    ds_lr_aug["mucape_proxy"] = mucape
    ds_lr_aug["T_850_minus_T_500"] = t_diff
    ds_lr_aug["u850_grad_orog_HR_LR_repr"] = u850_grad.coarsen(
        # Decimate HR back to LR for storage (we'll re-interp at use-time)
        # NOTE: u850·∇h_HR is HR-resolution feature; we store the LR
        # representation for compactness — the model will use the LR repr
        # and resolve grad at HR via the existing static_HR_v6 dataset.
        # Actually, to keep it simple, we save it as HR feature in the static
        # dataset instead — see below.
        # Here we just keep a placeholder; the real HR feature is in static_HR_v6.
    ).mean() if False else u850_grad.interp_like(ds_lr[args.t_850_name], method="linear")
    ds_lr_aug["u850_grad_orog_HR_LR_repr"].attrs = {
        "long_name": "u850_grad_orog_HR_resampled_back_to_LR_for_pipeline_storage",
        "note": "Original HR resolution conserved in static_HR_v6",
    }
    if theta_w_850 is not None:
        ds_lr_aug["theta_w_850"] = theta_w_850

    print(f"[V6 preproc] LR augmented variables : {list(ds_lr_aug.data_vars)[:15]}...")
    print(f"  Total LR vars : {len(ds_lr_aug.data_vars)} (original {len(ds_lr.data_vars)} + V6)")

    # ----- Write static_HR_v6 (original + mask_ocean + grad_orog_HR + region_masks) -----
    ds_static_hr_v6 = ds_static_hr.copy()
    ds_static_hr_v6["mask_ocean"] = mask_ocean
    ds_static_hr_v6["grad_orog_HR"] = grad_orog_HR
    ds_static_hr_v6["u850_grad_orog_HR_4km"] = u850_grad  # HR resolution preserved
    ds_static_hr_v6["region_masks_v6"] = region_masks_da

    # ----- Save outputs -----
    out_lr = Path(args.out_lr_augmented)
    out_static = Path(args.out_static_hr_v6)
    out_lr.parent.mkdir(parents=True, exist_ok=True)
    out_static.parent.mkdir(parents=True, exist_ok=True)

    print(f"[V6 preproc] Writing {out_lr}")
    ds_lr_aug.to_netcdf(out_lr)
    print(f"[V6 preproc] Writing {out_static}")
    ds_static_hr_v6.to_netcdf(out_static)
    print("[V6 preproc] DONE")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lr-path", required=True, help="Input LR NetCDF (per GCM)")
    p.add_argument("--static-hr-path", required=True, help="HR static dataset")
    p.add_argument("--out-lr-augmented", required=True, help="Output LR augmented NetCDF")
    p.add_argument("--out-static-hr-v6", required=True, help="Output static HR V6 NetCDF")
    # Variable name overrides (sensible defaults)
    p.add_argument("--w-850-name", default="w_850")
    p.add_argument("--w-500-name", default="w_500")
    p.add_argument("--t-850-name", default="T_850")
    p.add_argument("--t-500-name", default="T_500")
    p.add_argument("--q-850-name", default="q_850")
    p.add_argument("--q-500-name", default="q_500")
    p.add_argument("--u-850-name", default="u_850")
    p.add_argument("--v-850-name", default="v_850")
    p.add_argument("--orog-name", default="orog")
    p.add_argument("--land-sea-mask-name", default="sftlf")
    p.add_argument(
        "--theta-w-850", action="store_true",
        help="Also compute optional feature theta_w_850 (§4.4 Browning 2004)",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
