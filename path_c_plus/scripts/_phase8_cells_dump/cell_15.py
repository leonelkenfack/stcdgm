# === Cell 15 : 3-way comparison + JSON publication-ready ===

print('=' * 100)
print('PHASE 8 vs noncausal v4 vs V5 -- comparison (apples-to-apples where possible)')
print('=' * 100)
print()
print(f'{"Metric":<28}{"Phase 8":>12}{"V5_causal":>14}{"noncausal v4":>16}{"Note":>20}')
print('-' * 100)

ref_v5 = ref_jsons.get('V5_causal', {})
ref_nc = ref_jsons.get('noncausal_v4', {})

def _get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d: d = d[k]
        else: return default
    return d if d is not None else default

rows = [
    ('F1@p99 (pooled)',      phase8_metrics['F1_p99_pooled'], _get(ref_v5, 'f1_extremes', 'p99'), _get(ref_nc, 'f1_extremes', 'p99'), 'log1p, ours vs JSON*'),
    ('F1@p95 (pooled)',      phase8_metrics['F1_p95_pooled'], _get(ref_v5, 'f1_extremes', 'p95'), _get(ref_nc, 'f1_extremes', 'p95'), 'log1p, ours vs JSON*'),
    ('F1@p99 (ETCCDI)',      phase8_metrics['F1_p99_etccdi'], None, None, 'per-pixel land only'),
    ('CSI@p99',              phase8_metrics['CSI_p99'],       None, None, 'ETCCDI threshold'),
    ('SEDI@p99',             phase8_metrics['SEDI_p99'],      None, None, 'ETCCDI threshold'),
    ('FSS@n=9',              phase8_metrics['FSS_p99_n9'],    None, None, 'reflect bord'),
    ('FSS@n=25',             phase8_metrics['FSS_p99_n25'],   None, None, ''),
    ('FSS@n=51',             phase8_metrics['FSS_p99_n51'],   None, None, ''),
    ('RMSE',                 phase8_metrics['RMSE'],          _get(ref_v5, 'rmse'), _get(ref_nc, 'rmse'), 'log1p space'),
    ('MAE',                  phase8_metrics['MAE'],           _get(ref_v5, 'mae'), _get(ref_nc, 'mae'), 'log1p space'),
    ('Pearson global',       phase8_metrics['Pearson_global'],_get(ref_v5, 'pearson_corr', 'global'), _get(ref_nc, 'pearson_corr', 'global'), ''),
    ('Pearson per-sample',   phase8_metrics['Pearson_per_sample'], _get(ref_v5, 'pearson_corr', 'per_sample_avg'), _get(ref_nc, 'pearson_corr', 'per_sample_avg'), ''),
    ('Rx1day_bias (mm)',     phase8_metrics['Rx1day_bias'],   None, None, 'mm/day, our N batches'),
    ('R10_bias (count)',     phase8_metrics['R10_bias'],      None, None, ''),
    ('CDD_bias (count)',     phase8_metrics['CDD_bias'],      None, None, ''),
]

def _fmt(x):
    if x is None: return '--'
    if isinstance(x, float): return f'{x:+.4f}' if abs(x) > 1 else f'{x:.4f}'
    return str(x)

for label, p8, v5, nc, note in rows:
    print(f'{label:<28}{_fmt(p8):>12}{_fmt(v5):>14}{_fmt(nc):>16}{note:>20}')

print('-' * 100)
print('*JSON references in Convention B (zero+pooled, NOT ETCCDI). For true apples-to-apples,')
print(' those models must be re-evaluated in this notebook\'s convention.')
print()

# Compute deltas Phase 8 vs noncausal where comparable
deltas_vs_nc = {}
for label, p8, _v5, nc, _note in rows:
    if nc is not None and isinstance(p8, (int, float)) and isinstance(nc, (int, float)):
        deltas_vs_nc[label] = p8 - nc

print('Phase 8 vs noncausal v4 (positive = Phase 8 better, except RMSE/MAE/biases where lower is better) :')
for k, v in deltas_vs_nc.items():
    print(f'  {k:<28} delta = {v:+.4f}')

# Final JSON
final_results = {
    'phase': 'phase8_from_scratch',
    'pre_registration': PRE_REG_RECORD,
    'smoke_mode': SMOKE_MODE,
    'phase8_metrics': phase8_metrics,
    'reference_jsons': ref_jsons,
    'deltas_vs_noncausal_v4': deltas_vs_nc,
    'stat_tests': stat_tests,
    'caveats': [
        'Reference JSONs (V5, noncausal) are in Convention B (zero+pooled). Phase 8 reports both pooled (apples-to-apples) and ETCCDI per-pixel.',
        'Climate indices (Rx1day, R10, CDD) computed on N_TEST_BATCHES batches only. For full 730-day indices, need a second eval pass.',
        'Paired permutation test vs noncausal/V5 requires re-eval of those models in this same convention -- TODO.',
        'alpha learned = {:.4f}, regularised to prevent collapse to 0.'.format(alpha_final),
        'Best EMA decay selected via val RMSE sweep : {}'.format(best_decay),
    ],
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}

FINAL_RESULTS.parent.mkdir(parents=True, exist_ok=True)
FINAL_RESULTS.write_text(json.dumps(final_results, indent=2, default=str), encoding='utf-8')
print()
print(f'Saved : {FINAL_RESULTS}')

try:
    from google.colab import files
    files.download(str(FINAL_RESULTS))
except Exception:
    pass

# Save training history too
TRAINING_HISTORY.write_text(json.dumps({
    'stage1_history': stage1_history,
    'stage2_history': stage2_history,
}, indent=2, default=str), encoding='utf-8')
print(f'Saved : {TRAINING_HISTORY}')

print()
print('=' * 100)
print('Phase 8 notebook complete.')
print('Next steps :')
print('  1. Validate SMOKE_MODE=True run produces reasonable numbers (~15 min compute)')
print('  2. Set SMOKE_MODE=False, restart kernel, full run (~45-55h on A100 Pro+)')
print('  3. For true apples-to-apples vs noncausal/V5 : re-eval those checkpoints with this')
print('     notebook\'s exact protocol (replace mu_total computation, keep same eval cells).')
print('=' * 100)
