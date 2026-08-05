"""Apply remaining expert recommendations from L99 OVERTHINK XRAY CHAINLOGIC validation:

1. Stage 1 Cell 8 : log dual_path gate.mean() per epoch (Math + Climat anti-dead-encoder)
2. Stage 2 Cell 11 : log FAL + FCL separately via facl_components (Math + ML I1 stricter)
3. Cell 13 eval : add drizzle_bias metric (Climat I4 -- anti tail-weight over-correction)
4. spectral_loss.py : cos.clamp(-1, 1) defensive (IA D7)
"""
import json
import re
from pathlib import Path

nb_path = Path('path_c_plus/scripts/_phase8_from_scratch.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))


def _splitlines_preserve(src):
    lines = src.split('\n')
    return [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])


def find_cell(nb, marker):
    for c in nb['cells']:
        s = ''.join(c.get('source', []))
        if marker in s:
            return c, s
    return None, None


# ==================================================================
# Patch 1: Cell 8 Stage 1 -- log dual_path gate.mean() per epoch
# Math expert: "Surveiller `gate` du dual_path (risque dead encoder
# si dual_path bypasse mu_A)"
# ==================================================================
c8, c8_src = find_cell(nb, '# === Cell 8 : Stage 1 training')

# Append gate logging in epoch_losses + epoch print
OLD_EP_LOSSES = """    epoch_losses = {'mse': [], 'dag': [], 'l1': [], 'r10mm': [], 'rx1day': [], 'cdd': [], 'cc': [], 'causal_frac': []}"""
NEW_EP_LOSSES = """    epoch_losses = {'mse': [], 'dag': [], 'l1': [], 'r10mm': [], 'rx1day': [], 'cdd': [], 'cc': [], 'causal_frac': [], 'gate_mean': []}"""
if OLD_EP_LOSSES in c8_src and "'gate_mean'" not in c8_src:
    c8_src = c8_src.replace(OLD_EP_LOSSES, NEW_EP_LOSSES, 1)
    print('Cell 8: added gate_mean to epoch_losses dict')

# Append gate.mean() to per-batch log right after causal_frac is computed
OLD_CAUSAL_BLOCK = """        with torch.no_grad():
            _muA_n = float(mu_A.detach().pow(2).mean().sqrt())
            _muB_n = float(mu_B.detach().pow(2).mean().sqrt())
            epoch_losses['causal_frac'].append(_muA_n / (_muA_n + _muB_n + 1e-8))"""
NEW_CAUSAL_BLOCK = """        with torch.no_grad():
            _muA_n = float(mu_A.detach().pow(2).mean().sqrt())
            _muB_n = float(mu_B.detach().pow(2).mean().sqrt())
            epoch_losses['causal_frac'].append(_muA_n / (_muA_n + _muB_n + 1e-8))
            # Math expert monitor : gate.mean() == ratio of Path A (causal) vs Path B (LR-direct)
            # in dual_path fusion. If <0.05 -> dead encoder (struct_W1/W2 not learning).
            epoch_losses['gate_mean'].append(float(gate.detach().mean()))"""
if OLD_CAUSAL_BLOCK in c8_src and "'gate_mean'" not in c8_src.split('epoch_losses[')[5:][0]:
    c8_src = c8_src.replace(OLD_CAUSAL_BLOCK, NEW_CAUSAL_BLOCK, 1)
    print('Cell 8: added gate.mean() per-batch logging')

# Add to epoch print line
OLD_PRINT = """    print(f'[ep{ep}/{STAGE1_EPOCHS}] mse={avg["mse"]:.4f} dag={avg["dag"]:.4f} l1={avg["l1"]:.4f} '
          f'r10={avg["r10mm"]:.4f} rx1d={avg["rx1day"]:.4f} cdd={avg["cdd"]:.4f} '
          f'(phys={phys_active} wd_dag={lam_dag_warm:.3f} gate={_current_gate:.2f}) '
          f'A_dag.abs(): mean={_a_norm:.4f} max={_a_max:.4f} h(A)={_h_A_dag:.3e} '
          f'causal_frac={_avg_causal_frac:.3f} '
          f'time={ep_time:.0f}s n_batches={_n_batches_done}')"""

NEW_PRINT = """    _avg_gate_mean = float(np.mean(epoch_losses['gate_mean'])) if epoch_losses.get('gate_mean') else float('nan')
    print(f'[ep{ep}/{STAGE1_EPOCHS}] mse={avg["mse"]:.4f} dag={avg["dag"]:.4f} l1={avg["l1"]:.4f} '
          f'r10={avg["r10mm"]:.4f} rx1d={avg["rx1day"]:.4f} cdd={avg["cdd"]:.4f} '
          f'(phys={phys_active} wd_dag={lam_dag_warm:.3f} gate_sched={_current_gate:.2f}) '
          f'A_dag.abs(): mean={_a_norm:.4f} max={_a_max:.4f} h(A)={_h_A_dag:.3e} '
          f'causal_frac={_avg_causal_frac:.3f} dual_gate={_avg_gate_mean:.3f} '
          f'time={ep_time:.0f}s n_batches={_n_batches_done}')"""

if OLD_PRINT in c8_src:
    c8_src = c8_src.replace(OLD_PRINT, NEW_PRINT, 1)
    print('Cell 8: added dual_gate to epoch print')

compile(c8_src, 'cell8', 'exec')
c8['source'] = _splitlines_preserve(c8_src)


# ==================================================================
# Patch 2: Cell 11 Stage 2 -- use facl_components for separate FAL/FCL logging
# ==================================================================
c11, c11_src = find_cell(nb, '# === Cell 11 : Stage 2 training')

# Replace facl_loss with facl_components for split logging
OLD_FACL_CALL = """            facl_warmup_factor = min(1.0, float(ep) / max(1, FACL_WARMUP_EPOCHS))
            loss_facl = facl_loss(
                D_y.float(), delta_target_used.float(),
                alpha_amplitude=LAMBDA_FACL_FAL * facl_warmup_factor,
                beta_correlation=LAMBDA_FACL_FCL * facl_warmup_factor,
            )"""

NEW_FACL_CALL = """            facl_warmup_factor = min(1.0, float(ep) / max(1, FACL_WARMUP_EPOCHS))
            # Math + ML expert : use facl_components for separate FAL+FCL logging
            # to enable diagnostic if either dominates. Reconstruct loss_facl manually.
            _fal_val, _fcl_val = facl_components(D_y.float(), delta_target_used.float())
            loss_facl = (LAMBDA_FACL_FAL * facl_warmup_factor) * _fal_val \\
                      + (LAMBDA_FACL_FCL * facl_warmup_factor) * _fcl_val"""

if OLD_FACL_CALL in c11_src:
    c11_src = c11_src.replace(OLD_FACL_CALL, NEW_FACL_CALL, 1)
    print('Cell 11: replaced facl_loss with facl_components for split logging')
elif 'facl_components' in c11_src:
    print('Cell 11: already uses facl_components (skip)')

# Update import to include facl_components
OLD_IMPORT = """from st_cdgm.training.spectral_loss import facl_loss"""
NEW_IMPORT = """from st_cdgm.training.spectral_loss import facl_loss, facl_components"""
if OLD_IMPORT in c11_src and 'facl_components' not in c11_src.split('import')[1].split('\n')[0]:
    c11_src = c11_src.replace(OLD_IMPORT, NEW_IMPORT, 1)
    print('Cell 11: added facl_components to import')

# Add fal + fcl to losses_log dict
OLD_LOG_INIT = """    losses_log = {'edm': [], 'disp': [], 'alpha_reg': [], 'total': [], 'facl': []}"""
NEW_LOG_INIT = """    losses_log = {'edm': [], 'disp': [], 'alpha_reg': [], 'total': [], 'facl': [], 'fal': [], 'fcl': []}"""
if OLD_LOG_INIT in c11_src and "'fal':" not in c11_src:
    c11_src = c11_src.replace(OLD_LOG_INIT, NEW_LOG_INIT, 1)
    print('Cell 11: added fal+fcl to losses_log dict')

# Add per-batch append for fal + fcl
OLD_LOG_APPEND = """        losses_log['edm'].append(float(loss_edm.detach()))
        losses_log['disp'].append(float(loss_disp.detach()))
        losses_log['alpha_reg'].append(float(loss_alpha.detach()))
        losses_log['facl'].append(float(loss_facl.detach()))
        losses_log['total'].append(float(loss_total.detach()))"""

NEW_LOG_APPEND = """        losses_log['edm'].append(float(loss_edm.detach()))
        losses_log['disp'].append(float(loss_disp.detach()))
        losses_log['alpha_reg'].append(float(loss_alpha.detach()))
        losses_log['facl'].append(float(loss_facl.detach()))
        losses_log['fal'].append(float(_fal_val.detach()))
        losses_log['fcl'].append(float(_fcl_val.detach()))
        losses_log['total'].append(float(loss_total.detach()))"""

if OLD_LOG_APPEND in c11_src and "losses_log['fal']" not in c11_src:
    c11_src = c11_src.replace(OLD_LOG_APPEND, NEW_LOG_APPEND, 1)
    print('Cell 11: added per-batch FAL+FCL logging')

# Update epoch print to include fal + fcl
m = re.search(r"print\(f'\[ep\{ep\}/\{STAGE2_EPOCHS\}\][^']+?\)", c11_src)
if m:
    old_p = m.group(0)
    if 'fal=' not in old_p and 'facl=' in old_p:
        new_p = old_p.replace("facl={np.mean(losses_log['facl']", "fal={np.mean(losses_log['fal'][-n_batches_seen:]):.4f} fcl={np.mean(losses_log['fcl'][-n_batches_seen:]):.4f} facl={np.mean(losses_log['facl']")
        c11_src = c11_src.replace(old_p, new_p, 1)
        print('Cell 11: added fal+fcl separate to epoch print')

compile(c11_src, 'cell11', 'exec')
c11['source'] = _splitlines_preserve(c11_src)


# ==================================================================
# Patch 3: Cell 13 eval -- add drizzle_bias metric (Climat I4)
# ==================================================================
c13, c13_src = find_cell(nb, '# === Cell 13 : Metriques')

# Add drizzle_bias computation before the final print
# drizzle_bias = (frac of pred wet days > 1mm) - (frac of true wet days > 1mm)
# Anti tail-weight over-correction : if positive and large, model is wet-biased.

OLD_PRINT_LINE = """    print(f'  Rx1day_bias = {rx1d_bias:+.3f} mm  R10_bias = {r10_bias:+.3f}  CDD_bias = {cdd_bias:+.3f}')"""

NEW_PRINT_BLOCK = """    # Climat expert I4 : drizzle_bias = fraction (pred wet > 1mm) - fraction (true wet > 1mm).
    # Anti tail-weight over-correction monitor. Positive and large => model over-predicts.
    # WET_DAY_THRESHOLD_MM = 1.0 (ETCCDI standard, log1p threshold = log1p(1.0+0.01) = 0.6981).
    _WET_LOG1P = float(np.log1p(WET_DAY_THRESHOLD_MM + PRECIP_DELTA)) if 'PRECIP_DELTA' in dir() else float(np.log1p(WET_DAY_THRESHOLD_MM + 0.01))
    pred_wet_frac = float((pred_full_log[valid_mask_eval] > _WET_LOG1P).float().mean().item()) if hasattr(pred_full_log, 'float') else float((pred_full_log[valid_mask_eval] > _WET_LOG1P).mean())
    targ_wet_frac = float((target_full_log[valid_mask_eval] > _WET_LOG1P).float().mean().item()) if hasattr(target_full_log, 'float') else float((target_full_log[valid_mask_eval] > _WET_LOG1P).mean())
    drizzle_bias = pred_wet_frac - targ_wet_frac

    print(f'  Rx1day_bias = {rx1d_bias:+.3f} mm  R10_bias = {r10_bias:+.3f}  CDD_bias = {cdd_bias:+.3f}')
    print(f'  drizzle_bias = {drizzle_bias:+.4f} (pred_wet={pred_wet_frac:.3f} vs targ_wet={targ_wet_frac:.3f})')"""

# Only apply if the variables exist in scope. We need to verify the eval cell exposes
# pred_full_log, target_full_log, valid_mask_eval. Otherwise skip with warning.
if OLD_PRINT_LINE in c13_src and 'drizzle_bias' not in c13_src:
    # Try to apply -- but might have variable mismatches. Use a safer try/except wrapping.
    SAFE_PRINT_BLOCK = """    # Climat expert I4 : drizzle_bias = fraction (pred wet > 1mm) - fraction (true wet > 1mm).
    # Anti tail-weight over-correction monitor. Positive => model over-predicts wet days.
    try:
        _wet_thr_log1p = float(np.log1p(WET_DAY_THRESHOLD_MM + 0.01))  # PRECIP_DELTA = 0.01
        # Best-effort : look for the pred/target tensors used in eval, log1p space.
        _pred = pred_full_log if 'pred_full_log' in dir() else (
            pred_full if 'pred_full' in dir() else None)
        _targ = target_full_log if 'target_full_log' in dir() else (
            target_full if 'target_full' in dir() else None)
        if _pred is not None and _targ is not None:
            _mask = valid_mask_eval if 'valid_mask_eval' in dir() else None
            if _mask is None and hasattr(_targ, 'isfinite'):
                _mask = _targ.isfinite() if hasattr(_targ, 'isfinite') else np.isfinite(_targ)
            if hasattr(_pred, 'cpu'):  # torch tensor
                _pred_np = _pred.detach().cpu().numpy()
                _targ_np = _targ.detach().cpu().numpy() if hasattr(_targ, 'cpu') else np.asarray(_targ)
                _mask_np = _mask.detach().cpu().numpy() if hasattr(_mask, 'cpu') else np.asarray(_mask)
            else:
                _pred_np = np.asarray(_pred); _targ_np = np.asarray(_targ); _mask_np = np.asarray(_mask)
            _mask_bool = _mask_np.astype(bool)
            if _mask_bool.sum() > 100:
                _pred_wet = float((_pred_np[_mask_bool] > _wet_thr_log1p).mean())
                _targ_wet = float((_targ_np[_mask_bool] > _wet_thr_log1p).mean())
                drizzle_bias = _pred_wet - _targ_wet
                print(f'  drizzle_bias = {drizzle_bias:+.4f} (pred_wet={_pred_wet:.3f} vs targ_wet={_targ_wet:.3f})')
            else:
                drizzle_bias = float('nan')
                print(f'  drizzle_bias : skipped (mask too sparse)')
        else:
            drizzle_bias = float('nan')
            print(f'  drizzle_bias : skipped (pred/target vars not in scope)')
    except Exception as _e:
        drizzle_bias = float('nan')
        print(f'  drizzle_bias : skipped ({type(_e).__name__}: {_e})')"""

    NEW_LINE_WITH_DRIZZLE = OLD_PRINT_LINE + "\n" + SAFE_PRINT_BLOCK
    c13_src = c13_src.replace(OLD_PRINT_LINE, NEW_LINE_WITH_DRIZZLE, 1)
    print('Cell 13: added drizzle_bias (Climat I4)')

    # Also add to metrics return dict
    OLD_DICT_ENTRY = """'Rx1day_bias': rx1d_bias, 'R10_bias': r10_bias, 'CDD_bias': cdd_bias,"""
    NEW_DICT_ENTRY = """'Rx1day_bias': rx1d_bias, 'R10_bias': r10_bias, 'CDD_bias': cdd_bias,
        'drizzle_bias': drizzle_bias if 'drizzle_bias' in dir() else float('nan'),"""
    if OLD_DICT_ENTRY in c13_src:
        c13_src = c13_src.replace(OLD_DICT_ENTRY, NEW_DICT_ENTRY, 1)
        print('Cell 13: added drizzle_bias to metrics dict')

compile(c13_src, 'cell13', 'exec')
c13['source'] = _splitlines_preserve(c13_src)


# Save notebook
nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print(f'\nsaved notebook')


# ==================================================================
# Patch 4: spectral_loss.py -- cos.clamp(-1, 1) defensive (IA D7)
# ==================================================================
sl_path = Path('src/st_cdgm/training/spectral_loss.py')
sl_src = sl_path.read_text(encoding='utf-8')

OLD_COS = """    cos = inner / (norm_p * norm_t + eps)                            # [B]
    fcl = (1.0 - cos).mean()"""
NEW_COS = """    cos = inner / (norm_p * norm_t + eps)                            # [B]
    cos = cos.clamp(-1.0, 1.0)   # IA expert D7 defensive : avoid 1-cos overshoot on near-zero spectra
    fcl = (1.0 - cos).mean()"""

if OLD_COS in sl_src and 'cos.clamp' not in sl_src:
    sl_src = sl_src.replace(OLD_COS, NEW_COS, 1)
    sl_path.write_text(sl_src, encoding='utf-8')
    print('spectral_loss.py: added cos.clamp(-1, 1) defensive')
elif 'cos.clamp' in sl_src:
    print('spectral_loss.py: already has cos.clamp (skip)')


# Final sanity scan
nb2 = json.loads(nb_path.read_text(encoding='utf-8'))
c8_s = next(''.join(c.get('source', [])) for c in nb2['cells'] if '# === Cell 8 : Stage 1 training' in ''.join(c.get('source', [])))
c11_s = next(''.join(c.get('source', [])) for c in nb2['cells'] if '# === Cell 11 : Stage 2 training' in ''.join(c.get('source', [])))
c13_s = next(''.join(c.get('source', [])) for c in nb2['cells'] if '# === Cell 13 : Metriques' in ''.join(c.get('source', [])))

checks = {
    'C8 gate_mean in epoch_losses': "'gate_mean'" in c8_s,
    'C8 gate.mean() per batch': 'epoch_losses[\'gate_mean\'].append(float(gate.detach().mean()))' in c8_s,
    'C8 dual_gate in print': 'dual_gate=' in c8_s,
    'C11 facl_components import': 'facl_components' in c11_s,
    'C11 _fal_val, _fcl_val': '_fal_val, _fcl_val' in c11_s,
    'C11 fal+fcl in losses_log': "'fal': []" in c11_s,
    'C11 fal+fcl per-batch': "losses_log['fal']" in c11_s,
    'C13 drizzle_bias': 'drizzle_bias' in c13_s,
}

print('\n=== SANITY ===')
all_ok = True
for label, ok in checks.items():
    status = 'OK' if ok else 'MISSING'
    print(f'  {status:7s} : {label}')
    if not ok: all_ok = False

print(f'\nspectral_loss.py cos.clamp: {"OK" if "cos.clamp" in sl_path.read_text(encoding="utf-8") else "MISSING"}')

if not all_ok:
    raise RuntimeError('Some checks failed')
print('\nALL CHECKS PASS')
