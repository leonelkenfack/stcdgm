# === Cell 9 : Precompute mu_total cache (Stage 1 FROZEN forward) ===
# One-time pass on the train_dataset to cache (mu_total, baseline_log, delta_target).
# Used by Stage 2 training (cached -> 70x faster than recomputing Stage 1 each batch).
import time

if STAGE1_CACHE_PATH.exists() and not SMOKE_MODE:
    print(f'[Cell 9] Cache exists, loading : {STAGE1_CACHE_PATH}')
    cache = torch.load(STAGE1_CACHE_PATH, map_location='cpu', weights_only=False)
    cache = {k: v.contiguous().clone() for k, v in cache.items()}
else:
    print(f'[Cell 9] Precomputing mu_total cache from train_dataset (SMOKE={SMOKE_MODE})...')
    _t0 = time.time()
    mu_list, base_list, delta_list, valid_list = [], [], [], []
    _count = 0
    for sample in train_dataset:
        batch = convert_sample_to_batch(sample, builder, DEVICE)
        with torch.no_grad():
            mu_A, mu_total, mu_B, gate, baseline_log, target_residual = _stage1_forward(batch)
            delta_target = target_residual - mu_total
            valid = torch.isfinite(target_residual)
        mu_list.append(mu_total.squeeze(0).cpu())
        base_list.append(baseline_log.squeeze(0).cpu())
        delta_list.append(torch.nan_to_num(delta_target, nan=0.0).squeeze(0).cpu())
        valid_list.append(valid.squeeze(0).cpu())
        _count += 1
        if SMOKE_MODE and _count >= 30: break
        if _count % 500 == 0:
            print(f'  cached {_count} samples ({(time.time()-_t0)/60:.1f} min)')
    cache = {
        'mu_HR':        torch.stack(mu_list, dim=0),
        'baseline_log': torch.stack(base_list, dim=0),
        'delta_target': torch.stack(delta_list, dim=0),
        'valid_mask':   torch.stack(valid_list, dim=0),
    }
    torch.save(cache, STAGE1_CACHE_PATH)
    print(f'[Cell 9] Cache built in {time.time()-_t0:.1f}s : {STAGE1_CACHE_PATH}')

# Integrity check (Expert IA from previous audit)
for k in ('mu_HR', 'baseline_log', 'delta_target'):
    if not torch.isfinite(cache[k]).all():
        raise RuntimeError(f'cache[{k!r}] contains NaN/Inf -- delete and rerun')

N_CACHE = int(cache['mu_HR'].shape[0])
print(f'[Cell 9] Cache shapes :')
for k, v in cache.items():
    print(f'  {k} = {tuple(v.shape)}')
print(f'[Cell 9] Total cached samples : {N_CACHE}')

# Cached dataset for Stage 2
class _CachedDataset(torch.utils.data.Dataset):
    def __init__(self, cache, indices=None):
        self.mu_HR = cache['mu_HR']
        self.baseline_log = cache['baseline_log']
        self.delta_target = cache['delta_target']
        self.valid_mask = cache['valid_mask']
        self.indices = indices if indices is not None else list(range(len(self.mu_HR)))
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        idx = self.indices[i]
        return {
            'mu_HR': self.mu_HR[idx],
            'baseline_log': self.baseline_log[idx],
            'delta_target': self.delta_target[idx],
            'valid_mask': self.valid_mask[idx],
        }

_N_TRAIN_S2 = int(N_CACHE * 0.9)
train_cached = _CachedDataset(cache, list(range(_N_TRAIN_S2)))
val_cached   = _CachedDataset(cache, list(range(_N_TRAIN_S2, N_CACHE)))
train_cached_dataloader = torch.utils.data.DataLoader(
    train_cached, batch_size=STAGE2_BATCH_SIZE, shuffle=True,
    num_workers=0, pin_memory=False, drop_last=False,
)
val_cached_dataloader = torch.utils.data.DataLoader(
    val_cached, batch_size=STAGE2_BATCH_SIZE, shuffle=False,
    num_workers=0, pin_memory=False, drop_last=False,
)
print(f'[Cell 9] Stage 2 dataloaders : train={len(train_cached)}  val={len(val_cached)}  bs={STAGE2_BATCH_SIZE}')
