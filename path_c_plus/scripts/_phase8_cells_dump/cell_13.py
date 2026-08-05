# === Cell 13 : Metriques (3 conventions F1, CSI, SEDI, FSS, CRPS + indices climatiques) ===
from scipy.ndimage import uniform_filter

# Load climatology + land_mask for ETCCDI eval
_clim = np.load(CLIM_PATH)
clim_p99_np = _clim['clim_p99']
clim_p95_np = _clim['clim_p95']
land_mask_np = _clim['land_mask']

# ----- F1 / CSI / SEDI helper functions -----
def f1_pooled(pred, target, percentile=99.0):
    """Pooled full-grid, valid-only filter, strict > (Phase 6 audit fix)."""
    pf = pred.flatten(); tf = target.flatten()
    v = torch.isfinite(pf) & torch.isfinite(tf)
    pv = pf[v]; tv = tf[v]
    if tv.numel() < 100: return float('nan')
    thr = float(torch.quantile(tv, percentile / 100.0))
    tp = ((pv > thr) & (tv > thr)).sum().item()
    fp = ((pv > thr) & (tv <= thr)).sum().item()
    fn = ((pv <= thr) & (tv > thr)).sum().item()
    if (tp + fp) == 0 or (tp + fn) == 0: return 0.0
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

def f1_etccdi_per_pixel(pred_mm, target_mm, land_mask, clim_p99):
    """ETCCDI per-pixel : seuil per-pixel = clim_p99[i,j]. Land only."""
    N = pred_mm.shape[0]
    pred_np = pred_mm.cpu().numpy().reshape(N, *clim_p99.shape)
    target_np = target_mm.cpu().numpy().reshape(N, *clim_p99.shape)
    valid_pix = land_mask & np.isfinite(clim_p99)
    if not valid_pix.any(): return {'f1': float('nan'), 'tp': 0, 'fp': 0, 'fn': 0}
    thr = clim_p99[None, :, :]
    pred_bin = pred_np >= thr
    target_bin = target_np >= thr
    valid_bcast = valid_pix[None, :, :]
    tp = int((pred_bin & target_bin & valid_bcast).sum())
    fp = int((pred_bin & ~target_bin & valid_bcast).sum())
    fn = int((~pred_bin & target_bin & valid_bcast).sum())
    if (tp + fp) == 0 or (tp + fn) == 0: return {'f1': 0.0, 'tp': tp, 'fp': fp, 'fn': fn}
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    return {'f1': 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0, 'tp': tp, 'fp': fp, 'fn': fn}

def csi_sedi(pred_bin, target_bin):
    tp = int((pred_bin & target_bin).sum())
    fp = int((pred_bin & ~target_bin).sum())
    fn = int((~pred_bin & target_bin).sum())
    tn = int((~pred_bin & ~target_bin).sum())
    csi = tp / max(tp + fp + fn, 1)
    h = tp / max(tp + fn, 1); f = fp / max(fp + tn, 1)
    eps = 1e-12
    h = max(min(h, 1 - eps), eps); f = max(min(f, 1 - eps), eps)
    num = np.log(f) - np.log(h) - np.log(1 - f) + np.log(1 - h)
    den = np.log(f) + np.log(h) + np.log(1 - f) + np.log(1 - h)
    sedi = float(num / den) if den != 0 else float('nan')
    return csi, sedi

def fss_neighborhood(pred_bin, target_bin, n=9, mode='reflect'):
    P = pred_bin.astype(np.float32); Q = target_bin.astype(np.float32)
    Pf = np.stack([uniform_filter(P[i], size=n, mode=mode) for i in range(P.shape[0])])
    Qf = np.stack([uniform_filter(Q[i], size=n, mode=mode) for i in range(Q.shape[0])])
    mse  = float(((Pf - Qf) ** 2).mean())
    norm = float((Pf ** 2 + Qf ** 2).mean())
    return 1.0 - mse / max(norm, 1e-12)

# ----- Compute metrics on Phase 8 (live N=64 K=128 ensemble) -----
print('=' * 78)
print('Phase 8 BS30 metrics on TEST split')
print('=' * 78)

f1_p99_pool = f1_pooled(pred_full_log, target_full_log, 99.0)
f1_p95_pool = f1_pooled(pred_full_log, target_full_log, 95.0)
print(f'  F1@p99 (pooled, log1p)    = {f1_p99_pool:.4f}')
print(f'  F1@p95 (pooled, log1p)    = {f1_p95_pool:.4f}')

etccdi_p99 = f1_etccdi_per_pixel(pred_mm, target_mm, land_mask_np, clim_p99_np)
etccdi_p95 = f1_etccdi_per_pixel(pred_mm, target_mm, land_mask_np, clim_p95_np)
print(f'  F1@p99 (ETCCDI per-pixel) = {etccdi_p99["f1"]:.4f}  (TP={etccdi_p99["tp"]} FP={etccdi_p99["fp"]} FN={etccdi_p99["fn"]})')
print(f'  F1@p95 (ETCCDI per-pixel) = {etccdi_p95["f1"]:.4f}')

# CSI / SEDI / FSS at ETCCDI p99 threshold
pred_np = pred_mm.cpu().numpy().reshape(test_target.shape[0], H_HR, W_HR)
target_np = target_mm.cpu().numpy().reshape(test_target.shape[0], H_HR, W_HR)
valid_pix = land_mask_np & np.isfinite(clim_p99_np)
pred_bin = (pred_np >= clim_p99_np[None]) & valid_pix[None]
target_bin = (target_np >= clim_p99_np[None]) & valid_pix[None]
csi, sedi = csi_sedi(pred_bin, target_bin)
fss_9 = fss_neighborhood(pred_bin, target_bin, n=9, mode='reflect')
fss_25 = fss_neighborhood(pred_bin, target_bin, n=25, mode='reflect')
fss_51 = fss_neighborhood(pred_bin, target_bin, n=51, mode='reflect')
print(f'  CSI@p99 = {csi:.4f}  SEDI@p99 = {sedi:.4f}')
print(f'  FSS (n=9)={fss_9:.4f}  (n=25)={fss_25:.4f}  (n=51)={fss_51:.4f}')

# RMSE, MAE, Pearson on log1p full pred
_valid = torch.isfinite(target_full_log)
pf = pred_full_log[_valid]; tf = target_full_log[_valid]
rmse = float(((pf - tf) ** 2).mean().sqrt())
mae  = float((pf - tf).abs().mean())
pf_c = pf - pf.mean(); tf_c = tf - tf.mean()
pearson_global = float((pf_c * tf_c).sum() / ((pf_c.norm() * tf_c.norm()).clamp_min(1e-12)))
per_sample = []
for i in range(pred_full_log.shape[0]):
    vi = _valid[i]
    if vi.sum() < 2: continue
    pi = pred_full_log[i][vi]; ti = target_full_log[i][vi]
    pic = pi - pi.mean(); tic = ti - ti.mean()
    c = float((pic * tic).sum() / ((pic.norm() * tic.norm()).clamp_min(1e-12)))
    per_sample.append(c)
pearson_ps = float(np.mean(per_sample)) if per_sample else float('nan')
print(f'  RMSE = {rmse:.4f}  MAE = {mae:.4f}  Pearson_global = {pearson_global:.4f}  Pearson_ps = {pearson_ps:.4f}')

# Climate indices on the 64 batches (note : for full 730-day indices, would need full test pass)
# For SMOKE, we just compute on the sampled batches as proxy
pred_mm_np = pred_mm.cpu().numpy()
target_mm_np = target_mm.cpu().numpy()
# RX1day (annual max per pixel) -- with only N batches it's approximated
rx1d_pred   = float(np.nanmean(pred_mm_np.max(axis=0)[land_mask_np]))
rx1d_target = float(np.nanmean(target_mm_np.max(axis=0)[land_mask_np]))
rx1d_bias = rx1d_pred - rx1d_target

# R10mm count (per pixel sum) and CDD
r10_pred   = float(np.nanmean((pred_mm_np >= 10).sum(axis=0)[land_mask_np]))
r10_target = float(np.nanmean((target_mm_np >= 10).sum(axis=0)[land_mask_np]))
r10_bias = r10_pred - r10_target

cdd_pred   = float(np.nanmean((pred_mm_np < 1).sum(axis=0)[land_mask_np]))
cdd_target = float(np.nanmean((target_mm_np < 1).sum(axis=0)[land_mask_np]))
cdd_bias = cdd_pred - cdd_target

print(f'  Rx1day_bias = {rx1d_bias:+.3f} mm  R10_bias = {r10_bias:+.3f}  CDD_bias = {cdd_bias:+.3f}')

phase8_metrics = {
    'F1_p99_pooled': f1_p99_pool,
    'F1_p95_pooled': f1_p95_pool,
    'F1_p99_etccdi': etccdi_p99['f1'],
    'F1_p95_etccdi': etccdi_p95['f1'],
    'CSI_p99': csi, 'SEDI_p99': sedi,
    'FSS_p99_n9': fss_9, 'FSS_p99_n25': fss_25, 'FSS_p99_n51': fss_51,
    'RMSE': rmse, 'MAE': mae,
    'Pearson_global': pearson_global, 'Pearson_per_sample': pearson_ps,
    'Rx1day_bias': rx1d_bias, 'R10_bias': r10_bias, 'CDD_bias': cdd_bias,
    'alpha_final': alpha_final,
    'best_ema_decay': best_decay,
    'n_batches_eval': int(test_pred.shape[0]),
    'k_samples': K_SAMPLES,
    'n_steps_diff': N_STEPS_DIFF,
}
print(f'[Cell 13] Phase 8 metrics computed')
