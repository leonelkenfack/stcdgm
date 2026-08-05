# === Cell 4 : Climatology p95/p99 per-pixel (ETCCDI Zhang 2011) ===
# Per-pixel quantile on wet days >= 1 mm/day, training period only.

_t0 = time.time()
_hr_ds = xr.open_dataset(str(HR_RAW_PATH), engine='h5netcdf')
_pr_var = 'pr' if 'pr' in _hr_ds.data_vars else list(_hr_ds.data_vars)[0]
_hr = _hr_ds[_pr_var]
_time_var = _hr.dims[0]
_hr_train = _hr.sel({_time_var: slice(K9_DATES['train'][0], K9_DATES['train'][1])})
_hr_train_np = _hr_train.values.astype(np.float32)
print(f'[Cell 4] train slice = {_hr_train_np.shape}')

H, W = _hr_train_np.shape[1], _hr_train_np.shape[2]
clim_p95 = np.full((H, W), np.nan, dtype=np.float32)
clim_p99 = np.full((H, W), np.nan, dtype=np.float32)
n_wet = np.zeros((H, W), dtype=np.int32)

for i in range(H):
    for j in range(W):
        if not land_mask[i, j]:
            continue
        px = _hr_train_np[:, i, j]
        px_finite = px[np.isfinite(px)]
        wet = px_finite[px_finite >= WET_DAY_THRESHOLD_MM]
        n_wet[i, j] = wet.size
        if wet.size >= 30:
            clim_p95[i, j] = float(np.quantile(wet, 0.95))
            clim_p99[i, j] = float(np.quantile(wet, 0.99))

_n_valid = int(np.isfinite(clim_p99).sum())
print(f'[Cell 4] land pixels with valid p99 = {_n_valid} / {n_land}')
print(f'[Cell 4] mean wet days per land pixel = {n_wet[land_mask].mean():.0f}')
print(f'[Cell 4] clim_p99 range = [{np.nanmin(clim_p99):.2f}, {np.nanmax(clim_p99):.2f}] mm/day  mean = {np.nanmean(clim_p99):.2f}')

np.savez(CLIM_PATH, clim_p95=clim_p95, clim_p99=clim_p99, n_wet=n_wet,
          land_mask=land_mask, wet_threshold_mm=WET_DAY_THRESHOLD_MM,
          train_start=K9_DATES['train'][0], train_end=K9_DATES['train'][1])
del _hr_train_np
_hr_ds.close()
print(f'[Cell 4] saved : {CLIM_PATH}  ({time.time()-_t0:.1f}s)')