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
# Feature 4 — conditional_instability (P3 fix, audit Climat V6')
#   theta_e_850 − theta_e*_sat_500 (saturated equivalent potential temp at 500)
#   Remplace l'ancien mucape_proxy = theta_e_850 − theta_e_500 qui était mal
#   défini (θe500 avec q observé → valeurs négatives sur profil AR typique).
#   Convention : > 0 = instabilité conditionnelle (parcelle 850 saturée plus
#   chaude que l'environnement saturé à 500).
# --------------------------------------------------------------------------- #
def _q_saturation(T_K: xr.DataArray, p_hPa: float) -> xr.DataArray:
    """Saturation specific humidity (kg/kg) at temperature T and pressure p.

    es via Wexler/Bolton approximation (consistent with _theta_e_bolton_1980),
    q_sat = 0.622·es / (p − 0.378·es).
    """
    es = 6.112 * np.exp(17.67 * (T_K - 273.15) / (T_K - 29.65))  # hPa
    q_sat = 0.622 * es / (p_hPa - 0.378 * es)
    return q_sat.clip(min=EPS, max=0.05)


def compute_conditional_instability(
    T_850: xr.DataArray, q_850: xr.DataArray, T_500: xr.DataArray
) -> xr.DataArray:
    """P3 (V6') : θe_850(q observé) − θe*_500(q saturé).

    θe* (saturated equivalent potential temperature) évalue l'environnement à
    500 hPa comme s'il était saturé — le critère standard d'instabilité
    conditionnelle (θe_low > θe*_mid ⇒ instable pour une parcelle saturée).
    """
    theta_e_850 = _theta_e_bolton_1980(T_850, q_850, 850.0)
    q_sat_500 = _q_saturation(T_500, 500.0)
    theta_e_star_500 = _theta_e_bolton_1980(T_500, q_sat_500, 500.0)
    out = theta_e_850 - theta_e_star_500
    out.attrs = {
        "long_name": "conditional_instability_thetae850_minus_thetaestar500",
        "units": "K",
        "formula": "theta_e_850(q_obs) - theta_e_star_500(q_sat(T_500))",
        "note": "P3 fix audit Climat V6' — remplace mucape_proxy (mal defini)",
        "source": "V6' preprocess",
    }
    return out


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
EARTH_M_PER_DEG = 111_000.0  # metres per degree of latitude


def compute_grad_orography_components_HR(
    orog_HR: xr.DataArray, mask_ocean: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """P1 (V6') : SIGNED orography gradient components ∂h/∂x, ∂h/∂y in m/m.

    - ∂h/∂y (northward) = differentiate(lat) / 111000
    - ∂h/∂x (eastward)  = differentiate(lon) / (111000 · cos(lat))
      → le facteur cos(lat) corrige l'anisotropie (~37% à 43°S sinon)

    Retourne les COMPOSANTES SIGNÉES (pas la magnitude) — le signe distingue
    soulèvement amont vs subsidence foehn aval (audit Climat V6').
    """
    orog_land = orog_HR * mask_ocean
    lat_name = "lat" if "lat" in orog_land.dims else "y"
    lon_name = "lon" if "lon" in orog_land.dims else "x"

    dh_dlat = orog_land.differentiate(lat_name)   # m / deg
    dh_dlon = orog_land.differentiate(lon_name)   # m / deg

    # Convert to physical slope (m/m)
    dh_dy = (dh_dlat / EARTH_M_PER_DEG) * mask_ocean
    coslat = np.cos(np.deg2rad(orog_land[lat_name]))
    dh_dx = (dh_dlon / (EARTH_M_PER_DEG * coslat)) * mask_ocean

    dh_dx.attrs = {
        "long_name": "signed_eastward_orography_slope_HR",
        "units": "m/m (dimensionless slope)",
        "note": "cos(lat) applied — P1 fix audit Climat V6'",
    }
    dh_dy.attrs = {
        "long_name": "signed_northward_orography_slope_HR",
        "units": "m/m (dimensionless slope)",
    }
    return dh_dx, dh_dy


def compute_wind_dot_grad_orog_HR(
    u850_LR: xr.DataArray,
    v850_LR: xr.DataArray,
    dh_dx: xr.DataArray,
    dh_dy: xr.DataArray,
) -> xr.DataArray:
    """P1 (V6') : SIGNED foehn/uplift proxy — (u,v)·∇h en HR.

        w_orog ≈ u850·∂h/∂x + v850·∂h/∂y     [m/s]

    C'est la vitesse verticale orographique linéaire (Smith & Barstad 2004) :
    > 0 = soulèvement forcé (upwind), < 0 = subsidence foehn (downwind).
    Remplace l'ancien u850·|∇h| qui perdait le signe (soulèvement côte Ouest
    et subsidence Est donnaient la même valeur — audit Climat V6').

    Resolution-coherent : u/v interpolés bilinéairement sur la grille HR avant
    le produit (Elvidge-Renfrew 2016 BAMS 97:455).
    """
    u_HR = u850_LR.interp_like(dh_dx, method="linear")
    v_HR = v850_LR.interp_like(dh_dy, method="linear")
    w_orog = u_HR * dh_dx + v_HR * dh_dy
    w_orog.attrs = {
        "long_name": "signed_orographic_vertical_velocity_wind_dot_grad_h",
        "units": "m/s",
        "formula": "u850_HR_bilin * dh/dx + v850_HR_bilin * dh/dy (signed)",
        "reference": "Smith-Barstad 2004 ; Elvidge-Renfrew 2016 BAMS 97:455",
        "note": "P1 fix audit Climat V6' — remplace u850*|grad_h| (signe perdu)",
        "source": "V6' preprocess",
    }
    return w_orog


def compute_ivt_persistence_72h(
    q_850: xr.DataArray, u_850: xr.DataArray, v_850: xr.DataArray,
    time_dim: str = "time",
) -> xr.DataArray:
    """Bonus V6' (audit Climat, validation finale) : persistance AR.

    Proxy IVT bas-niveau = q850·|V850| (kg/kg · m/s), moyenné sur 72h
    (3 pas quotidiens, fenêtre right-aligned = ne voit que le passé — pas de
    fuite future). La DURÉE du stalling des ARs domine l'accumulation des
    extrêmes West Coast (Weather Clim. Extremes 2024).
    """
    ivt_proxy = q_850 * np.sqrt(u_850 ** 2 + v_850 ** 2)
    ivt_72h = ivt_proxy.rolling({time_dim: 3}, min_periods=1).mean()
    ivt_72h.attrs = {
        "long_name": "low_level_moisture_flux_72h_mean_AR_persistence_proxy",
        "units": "kg/kg * m/s",
        "formula": "rolling_mean_3d( q850 * sqrt(u850^2+v850^2) ), right-aligned (past-only)",
        "source": "V6' preprocess — audit Climat validation finale",
    }
    return ivt_72h


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

    print("[V6' preproc] Feature 4 : conditional_instability (P3 fix)")
    cond_instab = compute_conditional_instability(
        ds_lr[args.t_850_name], ds_lr[args.q_850_name], ds_lr[args.t_500_name]
    )

    print("[V6 preproc] Feature 5 : T_850 − T_500")
    t_diff = compute_T_diff(ds_lr[args.t_850_name], ds_lr[args.t_500_name])

    # ----- mask_ocean + ∇h_HR (HR static) — P2 fix sftlf -----
    print("[V6' preproc] mask_ocean (P2 fix : normalisation sftlf %)")
    if args.land_sea_mask_name in ds_static_hr.data_vars:
        mask_ocean = ds_static_hr[args.land_sea_mask_name].astype("float32")
        # P2 (audit Climat V6') : CMIP sftlf est en % (0-100), pas en fraction.
        if float(mask_ocean.max()) > 1.5:
            print(f"  P2 fix : sftlf max = {float(mask_ocean.max()):.1f} > 1.5 → division par 100")
            mask_ocean = mask_ocean / 100.0
        mask_ocean = mask_ocean.clip(min=0.0, max=1.0)
    else:
        # Fallback : threshold orography > 0 as land
        print(f"  WARN : '{args.land_sea_mask_name}' not in static_HR — deriving from orog>0")
        mask_ocean = (ds_static_hr[args.orog_name] > 0.0).astype("float32")
    mask_ocean.attrs = {"long_name": "land_mask_1land_0ocean_fraction", "source": "V6' preproc (P2 fix)"}

    print("[V6' preproc] ∂h/∂x, ∂h/∂y signés avec cos(lat) (P1 fix)")
    dh_dx, dh_dy = compute_grad_orography_components_HR(ds_static_hr[args.orog_name], mask_ocean)

    print("[V6' preproc] Feature 6 : (u,v)·∇h signé — vitesse verticale orographique (P1 fix)")
    w_orog = compute_wind_dot_grad_orog_HR(
        ds_lr[args.u_850_name], ds_lr[args.v_850_name], dh_dx, dh_dy
    )

    # ----- Bonus V6' : persistance AR (IVT proxy 72h) -----
    ivt_72h = None
    if not args.no_ivt_72h:
        print("[V6' preproc] Bonus : IVT proxy moyenné 72h (persistance AR)")
        time_dim = "time" if "time" in ds_lr.dims else list(ds_lr.dims)[0]
        ivt_72h = compute_ivt_persistence_72h(
            ds_lr[args.q_850_name], ds_lr[args.u_850_name], ds_lr[args.v_850_name],
            time_dim=time_dim,
        )

    # P4 (audit Climat V6') : theta_w_850 DROPPÉ — l'approximation était fausse
    # de +14 K vs Davies-Jones 2008 exact. La fonction reste dans le module
    # pour référence mais n'est plus appelée.

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

    # ----- Write LR augmented (original + V6' features) -----
    ds_lr_aug = ds_lr.copy()
    ds_lr_aug["w_700"] = w_700
    ds_lr_aug["theta_e_850"] = theta_e_850
    ds_lr_aug["theta_e_500"] = theta_e_500
    ds_lr_aug["conditional_instability"] = cond_instab       # P3 fix (ex mucape_proxy)
    ds_lr_aug["T_850_minus_T_500"] = t_diff
    # w_orog est en résolution HR ; on stocke la représentation LR pour le
    # pipeline (re-interp bilinéaire cohérente au chargement). La version HR
    # exacte est conservée dans static_HR_v6 (variable temporelle → gros ;
    # on garde LR ici, le modèle voit déjà ∇h via le conditioning statique).
    ds_lr_aug["w_orog_signed_LR_repr"] = w_orog.interp_like(
        ds_lr[args.t_850_name], method="linear"
    )
    ds_lr_aug["w_orog_signed_LR_repr"].attrs = dict(w_orog.attrs)
    ds_lr_aug["w_orog_signed_LR_repr"].attrs["note"] = (
        "LR representation of signed (u,v)·∇h — P1 fix. HR slope components in static_HR_v6."
    )
    if ivt_72h is not None:
        ds_lr_aug["ivt_persistence_72h"] = ivt_72h           # Bonus V6'

    print(f"[V6' preproc] LR augmented variables : {list(ds_lr_aug.data_vars)[:15]}...")
    print(f"  Total LR vars : {len(ds_lr_aug.data_vars)} (original {len(ds_lr.data_vars)} + V6')")

    # ----- Write static_HR_v6 (original + mask_ocean + slope components + region_masks) -----
    ds_static_hr_v6 = ds_static_hr.copy()
    ds_static_hr_v6["mask_ocean"] = mask_ocean
    ds_static_hr_v6["dh_dx_signed"] = dh_dx                  # P1 : composantes signées m/m
    ds_static_hr_v6["dh_dy_signed"] = dh_dy
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
        "--no-ivt-72h", action="store_true",
        help="Skip the AR-persistence bonus feature (IVT proxy 72h rolling mean)",
    )
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    main(args)
