# === Cell 3 : Land mask depuis static dataset (multi-source fallback) ===
# Approche : static_predictors fournit orog/he/vegt. On essaie lsm/sftlf/vegt/orog dans l'ordre.

_t0 = time.time()
if not STATIC_PATH.exists():
    raise FileNotFoundError(f'Static dataset not on Drive : {STATIC_PATH}')
ds_static = xr.open_dataset(str(STATIC_PATH), engine='h5netcdf')
print(f'[Cell 3] static variables : {list(ds_static.data_vars)}')

land_mask = None
land_mask_source = None

# Strategy 1 : explicit lsm/sftlf/landfrac
for cand in ('lsm', 'land_sea_mask', 'landfrac', 'land_mask', 'sftlf'):
    if cand in ds_static.data_vars:
        arr = ds_static[cand].values.squeeze()
        if arr.ndim != 2: continue
        thr = 50.0 if arr.max() > 1.5 else 0.5
        land_mask = arr >= thr
        land_mask_source = f'{cand} (threshold {thr})'
        break

# Strategy 2 : vegetation type
if land_mask is None and 'vegt' in ds_static.data_vars:
    vegt = ds_static['vegt'].values.squeeze()
    if vegt.ndim == 2:
        _unique = np.unique(vegt[np.isfinite(vegt)])
        print(f'[Cell 3] vegt unique values = {_unique}')
        land_mask = (vegt > 0) & (vegt != 17) & np.isfinite(vegt)
        land_mask_source = 'vegt (0 and 17 = water)'

# Strategy 3 : orography (safe NaN handling per Climat)
if land_mask is None and 'orog' in ds_static.data_vars:
    orog = ds_static['orog'].values.squeeze()
    if orog.ndim == 2:
        print(f'[Cell 3] orog range : [{np.nanmin(orog):.2f}, {np.nanmax(orog):.2f}] m')
        if np.isnan(orog).any():
            land_mask = np.isfinite(orog) & (orog > -0.5)
            land_mask_source = 'orog : isfinite & > -0.5 m'
        else:
            land_mask = orog > 0.5
            land_mask_source = 'orog > 0.5 m'

if land_mask is None:
    raise RuntimeError(f'No land/sea variable found in static : {list(ds_static.data_vars)}')

n_land = int(land_mask.sum())
n_total = int(land_mask.size)
print(f'[Cell 3] land_mask source : {land_mask_source}')
print(f'[Cell 3] land pixels = {n_land} / {n_total} ({100*n_land/n_total:.1f}%)')
print(f'[Cell 3] expected for NZ : 22-45% (varies with bbox size)')

# Sanity HR shape match
_hr_ds = xr.open_dataset(str(HR_RAW_PATH), engine='h5netcdf')
_pr_var = 'pr' if 'pr' in _hr_ds.data_vars else list(_hr_ds.data_vars)[0]
assert _hr_ds[_pr_var].shape[-2:] == land_mask.shape, 'HR grid mismatch'
_hr_ds.close()

np.save(LAND_MASK_PATH, land_mask)
ds_static.close()
print(f'[Cell 3] saved : {LAND_MASK_PATH}  ({time.time()-_t0:.1f}s)')