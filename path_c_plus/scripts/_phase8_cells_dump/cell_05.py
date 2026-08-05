# === Cell 5 : Augmented LR features (w_700, θ_e_850, θ_e_500, MUCAPE proxy) ===
# Pre-compute offline once, save as NetCDF for fast loading during training.
# Formulas validated by Expert Climat :
#   w_700 = 0.571*w_850 + 0.429*w_500   (linear interpolation in pressure, Holton 2004)
#   θ_e   = θ * exp(L_v * q_sat / (c_p * T))   (Bolton 1980, simplified)
#   MUCAPE_proxy = θ_e_850 - θ_e_500   (static instability indicator, Emanuel 1994)
# Constants : R_d = 287, c_p = 1005, L_v = 2.5e6, R_v = 461.5

_t0 = time.time()

if AUGMENTED_LR_PATH.exists():
    print(f'[Cell 5] augmented LR already exists : {AUGMENTED_LR_PATH}')
    ds_lr_aug = xr.open_dataset(str(AUGMENTED_LR_PATH), engine='h5netcdf')
    print(f'[Cell 5] variables : {list(ds_lr_aug.data_vars)}')
else:
    print(f'[Cell 5] computing augmented features from {LR_RAW_PATH}...')
    ds_lr = xr.open_dataset(str(LR_RAW_PATH), engine='h5netcdf')
    print(f'[Cell 5] LR variables available : {list(ds_lr.data_vars)}')

    # ----- 1. w_700 by linear interpolation in pressure -----
    if 'w_850' in ds_lr.data_vars and 'w_500' in ds_lr.data_vars:
        w_700 = 0.571 * ds_lr['w_850'] + 0.429 * ds_lr['w_500']
        w_700.attrs['long_name'] = 'vertical_velocity_at_700hPa (interpolated)'
        w_700.attrs['units'] = 'Pa s-1'
        w_700.attrs['interpolation'] = '0.571*w_850 + 0.429*w_500 (linear in pressure)'
        print(f'[Cell 5] w_700 computed : shape {w_700.shape} range [{float(w_700.min()):.4f}, {float(w_700.max()):.4f}]')
    else:
        raise RuntimeError('w_850 and w_500 required for w_700 interpolation')

    # ----- 2. θ_e_850 and θ_e_500 via Bolton 1980 -----
    # θ = T * (1000/p)^(R_d/c_p)
    # q_sat via Magnus-Tetens : e_s = 6.112 * exp(17.67*(T-273.15)/(T-29.65))
    # θ_e = θ * exp(L_v * q_sat(T,p) / (c_p * T))
    R_d = 287.0
    c_p = 1005.0
    L_v = 2.5e6
    eps = 0.622  # R_d / R_v

    def _theta_e_bolton(T_K, p_hPa):
        """Bolton 1980 θ_e. T in Kelvin, p in hPa."""
        theta = T_K * (1000.0 / p_hPa) ** (R_d / c_p)
        e_s   = 6.112 * np.exp(17.67 * (T_K - 273.15) / (T_K - 29.65))
        q_sat = eps * e_s / (p_hPa - (1 - eps) * e_s)
        return theta * np.exp(L_v * q_sat / (c_p * T_K))

    if 't_850' in ds_lr.data_vars and 'q_850' in ds_lr.data_vars:
        T_850 = ds_lr['t_850']  # Kelvin (ACCESS-CM2 standard)
        theta_e_850 = xr.apply_ufunc(_theta_e_bolton, T_850, 850.0, dask='allowed')
        theta_e_850.attrs['long_name'] = 'equivalent_potential_temperature_850hPa (Bolton 1980)'
        theta_e_850.attrs['units'] = 'K'
        print(f'[Cell 5] θ_e_850 computed : range [{float(theta_e_850.min()):.1f}, {float(theta_e_850.max()):.1f}] K')
    else:
        raise RuntimeError('t_850 required for θ_e')

    if 't_500' in ds_lr.data_vars and 'q_500' in ds_lr.data_vars:
        T_500 = ds_lr['t_500']
        theta_e_500 = xr.apply_ufunc(_theta_e_bolton, T_500, 500.0, dask='allowed')
        theta_e_500.attrs['long_name'] = 'equivalent_potential_temperature_500hPa (Bolton 1980)'
        theta_e_500.attrs['units'] = 'K'
        print(f'[Cell 5] θ_e_500 computed : range [{float(theta_e_500.min()):.1f}, {float(theta_e_500.max()):.1f}] K')

    # ----- 3. MUCAPE proxy = θ_e_850 - θ_e_500 -----
    mucape_proxy = theta_e_850 - theta_e_500
    mucape_proxy.attrs['long_name'] = 'MUCAPE_proxy (theta_e_850 - theta_e_500)'
    mucape_proxy.attrs['units'] = 'K'
    mucape_proxy.attrs['interpretation'] = 'positive = convectively unstable'
    print(f'[Cell 5] MUCAPE proxy : range [{float(mucape_proxy.min()):.2f}, {float(mucape_proxy.max()):.2f}] K')

    # ----- Build augmented LR dataset (15 original + 4 new = 19 variables) -----
    ds_lr_aug = ds_lr.copy()
    ds_lr_aug['w_700']        = w_700
    ds_lr_aug['theta_e_850']  = theta_e_850
    ds_lr_aug['theta_e_500']  = theta_e_500
    ds_lr_aug['mucape_proxy'] = mucape_proxy

    # Save
    ds_lr_aug.to_netcdf(str(AUGMENTED_LR_PATH), engine='h5netcdf')
    ds_lr.close()
    print(f'[Cell 5] augmented LR saved : {AUGMENTED_LR_PATH}')
    print(f'[Cell 5] total variables = {len(ds_lr_aug.data_vars)} (original 15 + 4 new)')

print(f'[Cell 5] done in {time.time()-_t0:.1f}s')