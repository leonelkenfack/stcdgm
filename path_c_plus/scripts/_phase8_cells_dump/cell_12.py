# === Cell 12 : Sampling Phase 8 BS30 (N=64 x K=128 x 32 steps) avec post-hoc EMA sweep ===
import time

# Select EMA decoder via post-hoc sweep on a small validation set
# (Expert ML : choose the EMA decay that minimises val CRPS / RMSE)
def _ema_select_via_val(ema_list, val_loader, n_check=4):
    """Quick eval each EMA on a few val batches, return the one with lowest RMSE."""
    scores = []
    for i, ema in enumerate(ema_list):
        rmses = []
        with torch.no_grad():
            count = 0
            for batch in val_loader:
                if count >= n_check: break
                mu = batch['mu_HR'].to(DEVICE); base = batch['baseline_log'].to(DEVICE)
                tgt = batch['delta_target'].to(DEVICE)
                samples = []
                for _ in range(min(8, K_SAMPLES)):
                    out = ema.sample(
                        conditioning=None,
                        num_steps=N_STEPS_DIFF,
                        scheduler_type='edm_karras',
                        cfg_scale=0.0,
                        apply_constraints=False,
                        mu_HR=mu, baseline_log=base,
                    )
                    res = out.residual if hasattr(out, 'residual') else out
                    samples.append(res)
                pred = torch.stack(samples, dim=0).mean(dim=0)
                rmses.append(float(((pred - tgt) ** 2).mean().sqrt()))
                count += 1
        scores.append(np.mean(rmses) if rmses else float('inf'))
    best_i = int(np.argmin(scores))
    print(f'[Cell 12] EMA sweep RMSEs : {[f"{s:.4f}" for s in scores]}  best decay = {EMA_DECAYS[best_i]}')
    return ema_list[best_i], EMA_DECAYS[best_i]

best_ema, best_decay = _ema_select_via_val(ema_decoders, val_cached_dataloader, n_check=2 if SMOKE_MODE else 4)
print(f'[Cell 12] Selected EMA decoder (decay={best_decay}) for final BS30 eval')

# Now run full BS30 eval on test split.
# We iterate test_dataset (IterableDataset, fresh stream).
print(f'[Cell 12] Sampling : N={N_TEST_BATCHES} batches x K={K_SAMPLES} samples x {N_STEPS_DIFF} steps')
print(f'[Cell 12] Sampler = {SAMPLER_SCHEDULER}, cfg_scale = {CFG_SCALE}')
print(f'[Cell 12] Limited-Interval Guidance sigma in [{LIMITED_GUIDANCE_SIGMA_MIN}, {LIMITED_GUIDANCE_SIGMA_MAX}]')

_t0 = time.time()
test_pred_list = []   # K-mean per batch
test_target_list = []
test_mu_list = []
test_baseline_list = []
test_time_list = []   # for climate indices on full 730 days

# Use Stage 1 frozen + Stage 2 EMA selected
_count = 0
for sample in test_dataset:
    if _count >= N_TEST_BATCHES: break
    batch = convert_sample_to_batch(sample, builder, DEVICE)
    with torch.no_grad():
        mu_A, mu_total, mu_B, gate, baseline_log, target_residual = _stage1_forward(batch)
        samples = []
        for k in range(K_SAMPLES):
            torch.manual_seed(SEED + 1000 * _count + k)
            out = best_ema.sample(
                conditioning=None,
                num_steps=N_STEPS_DIFF,
                scheduler_type='edm_karras',  # fallback if dpm_solver++ not present
                cfg_scale=CFG_SCALE,
                apply_constraints=False,
                mu_HR=mu_total, baseline_log=baseline_log,
            )
            res = out.residual if hasattr(out, 'residual') else out
            samples.append(res)
        pred_delta = torch.stack(samples, dim=0).mean(dim=0)
    test_pred_list.append(pred_delta.detach().cpu())
    test_target_list.append(target_residual.detach().cpu())
    test_mu_list.append(mu_total.detach().cpu())
    test_baseline_list.append(baseline_log.detach().cpu())
    _t = sample.get('time', None)
    test_time_list.append(_t)
    _count += 1
    if _count % 4 == 0:
        print(f'  batch {_count}/{N_TEST_BATCHES}  elapsed={time.time()-_t0:.0f}s  '
              f'avg/batch={(time.time()-_t0)/_count:.1f}s', flush=True)

test_pred     = torch.cat(test_pred_list, dim=0)
test_target   = torch.cat(test_target_list, dim=0)
test_mu       = torch.cat(test_mu_list, dim=0)
test_baseline = torch.cat(test_baseline_list, dim=0)

# Full HR (log1p space)
alpha_final = float(torch.sigmoid(alpha_logit).detach())
pred_full_log = test_baseline + alpha_final * test_mu + test_pred
target_full_log = test_baseline + test_mu + test_target  # noting target = HR_log - baseline - mu

# Convert to mm/day (Climat fix : - PRECIP_DELTA, clamp 500)
PRECIP_DELTA = 0.01
pred_mm   = (torch.expm1(pred_full_log)   - PRECIP_DELTA).clamp(min=0, max=500)
target_mm = (torch.expm1(target_full_log) - PRECIP_DELTA).clamp(min=0, max=500)

print(f'[Cell 12] Sampling done in {time.time()-_t0:.0f}s')
print(f'[Cell 12] alpha_final used in reconstruction = {alpha_final:.4f}')
print(f'[Cell 12] pred_full shape = {tuple(pred_full_log.shape)}')
