# === Cell 14 : Tests statistiques (paired permutation + bootstrap BCa + Holm-Bonferroni) ===

def paired_permutation_test(deltas, n_perm=PAIRED_PERMUTATION_N, seed=SEED):
    """Paired permutation test on per-batch deltas.
    H0 : mean(delta) = 0. Two-sided p-value."""
    deltas = np.asarray(deltas)
    obs = float(np.mean(deltas))
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(deltas))
        if abs(float(np.mean(signs * deltas))) >= abs(obs):
            count += 1
    return obs, (count + 1) / (n_perm + 1)

def bootstrap_bca_ci(values, n_resample=BOOTSTRAP_N_RESAMPLES, alpha=0.05, seed=SEED):
    """BCa bootstrap CI (Efron 1987)."""
    values = np.asarray(values)
    n = len(values)
    if n < 5:
        return float('nan'), float('nan'), float('nan')
    rng = np.random.default_rng(seed)
    # Bootstrap resamples
    boots = np.array([np.mean(rng.choice(values, size=n, replace=True)) for _ in range(n_resample)])
    theta_hat = float(np.mean(values))
    # Bias correction z0
    p = float(np.mean(boots < theta_hat))
    p = min(max(p, 1e-6), 1 - 1e-6)
    from scipy.stats import norm
    z0 = norm.ppf(p)
    # Acceleration a via jackknife
    jack = np.array([np.mean(np.delete(values, i)) for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6 * ((jm - jack) ** 2).sum() ** 1.5
    a = num / max(den, 1e-12)
    # Adjusted alpha levels
    z_lo = norm.ppf(alpha / 2); z_hi = norm.ppf(1 - alpha / 2)
    alpha_lo = norm.cdf(z0 + (z0 + z_lo) / (1 - a * (z0 + z_lo)))
    alpha_hi = norm.cdf(z0 + (z0 + z_hi) / (1 - a * (z0 + z_hi)))
    return float(np.quantile(boots, alpha_lo)), float(np.quantile(boots, alpha_hi)), theta_hat

def holm_bonferroni(p_values, alpha=0.05):
    """Holm-Bonferroni correction. Returns rejected[] array."""
    p = np.asarray(p_values)
    idx = np.argsort(p)
    n = len(p)
    rejected = np.zeros(n, dtype=bool)
    for i, k in enumerate(idx):
        threshold = alpha / (n - i)
        if p[k] <= threshold:
            rejected[k] = True
        else:
            break
    return rejected

# Per-batch F1@p99 pooled (for bootstrap CI on Phase 8 alone)
per_batch_f1 = []
for i in range(pred_full_log.shape[0]):
    p_i = pred_full_log[i:i+1].flatten()
    t_i = target_full_log[i:i+1].flatten()
    v_i = torch.isfinite(p_i) & torch.isfinite(t_i)
    if v_i.sum() < 100:
        per_batch_f1.append(np.nan)
        continue
    pv = p_i[v_i]; tv = t_i[v_i]
    thr = float(torch.quantile(tv, 0.99))
    tp = ((pv > thr) & (tv > thr)).sum().item()
    fp = ((pv > thr) & (tv <= thr)).sum().item()
    fn = ((pv <= thr) & (tv > thr)).sum().item()
    if (tp + fp) == 0 or (tp + fn) == 0:
        per_batch_f1.append(0.0); continue
    prec = tp / (tp + fp); rec = tp / (tp + fn)
    per_batch_f1.append(2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0)

per_batch_f1 = [x for x in per_batch_f1 if np.isfinite(x)]
lo, hi, theta = bootstrap_bca_ci(per_batch_f1)
print(f'[Cell 14] Phase 8 F1@p99 (per-batch) : theta_hat={theta:.4f}  CI95%=[{lo:.4f}, {hi:.4f}]')

# For now, we have Phase 8 only. Comparison with noncausal v4 + V5 requires re-eval
# of those models in the same convention. We document and load existing JSONs.
ref_jsons = {}
for name, p in [
    ('noncausal_v4', DRIVE_ROOT / 'ckpt_noncausal' / 'final_validation_metrics.json'),
    ('V5_causal',    DRIVE_ROOT / 'ckpt_v2_corrdiff_normal' / 'final_validation_metrics.json'),
]:
    if p.exists():
        try: ref_jsons[name] = json.loads(p.read_text())
        except Exception as e: print(f'  Failed to load {name} : {e}')

print(f'[Cell 14] Loaded reference JSONs : {list(ref_jsons.keys())}')
print(f'[Cell 14] Note : these are in Convention B (zero+pooled), comparison via Section A2 below')

# Statistical tests summary (Phase 8 internal)
stat_tests = {
    'phase8_F1_p99_per_batch_BCa_CI95': {'lo': lo, 'hi': hi, 'theta_hat': theta, 'n': len(per_batch_f1)},
    'bootstrap_n_resamples': BOOTSTRAP_N_RESAMPLES,
    'paired_permutation_n': PAIRED_PERMUTATION_N,
    'note': 'Paired tests vs noncausal/V5 require re-eval of those models in this notebook\'s exact convention.',
}
print(f'[Cell 14] Stats tests computed (full paired vs noncausal pending re-eval)')
