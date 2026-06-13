# Option C notebook implementation plan

Total cells planned : ~14 code cells + 1 markdown intro.

## Cell map

| # | Type | Content | Source | Status |
|---|---|---|---|---|
| 0 | md | Intro + structure + compute budget | NEW | ✅ DONE |
| 1 | code | Bootstrap Colab + git sync | A1 cell 1 (adapted) | ✅ DONE |
| 2 | code | Load CorrDiff Normal V2 config + Path C+ hyperparam override + ORACLE_FULL_DIR | NEW | ✅ DONE |
| 3 | code | Pipeline + K9 temporal split + builder + train/val/OOD datasets + iterate_batches helper | training_eval.ipynb cell 14 + A1 cell 3 (K9) | ⏳ TODO |
| 4 | code | Stack constructors : encoder, rcn_cell, rcn_runner, regression_head, diffusion, skip_block (from-scratch fresh init per seed) | training_eval.ipynb cells 24-30 | ⏳ TODO |
| 5 | code | Helpers import (Q_phys variants + projection hook + check_pc4_pc13_gate + compute_h1_verdict + paired_t_h2_h5 + holm_bonferroni + stamp_option_c_json) | path_c_plus.scripts.option_c_helpers | ⏳ TODO |
| 6 | code | Seed loop : for seed in [42,7,123]: skip if results.json exists; else Stage 1 (15 epochs train_epoch_stage1) + Stage 2 (200 epochs train_epoch_stage2) with atomic per-epoch checkpoint save | training_eval.ipynb cells 31-32 + A1 multi-seed wrapping | ⏳ TODO |
| 7 | code | Per-seed Q_phys eval (4 variants + skeleton F1 + phys_mag_gained + projection log save) | A1 cell 6 | ⏳ TODO |
| 8 | code | Per-seed FINAL_VALIDATION : produce `final_validation_metrics.json` | noncausal cell 61 (inline-port) | ⏳ TODO |
| 9 | code | Per-seed BS42 DOMAIN : produce `domain_metrics.json` | noncausal cell 62 (inline-port) | ⏳ TODO |
| 10 | code | Per-seed BS43 EVAL_SAMPLES : produce `eval_samples.npz` | noncausal cell 63 (inline-port) | ⏳ TODO |
| 11 | code | Per-seed loop over 3 GCMs (ACCESS-CM2 + EC-Earth3 + NorESM2-MM) : BS45 dataloader + BS44 aligned eval -> `aligned_metrics_<GCM>_causal.json` | noncausal cells 64+65 (inline-port) | ⏳ TODO |
| 12 | code | Cross-seed H1 verdict : compute_h1_verdict on q_phys_continuous_final per seed; output verdict + BCa CI + Student-t + Cohen's d + sparse_recovery flag | option_c_helpers.compute_h1_verdict | ⏳ TODO |
| 13 | code | Cross-seed H2-H5 paired tests vs noncausal : load_noncausal_baseline_metrics + paired_t_h2_h5 for each (MAE, Pearson, CRPS, F1@p95, F1@p99, CDD, Rx1Day, RAPSD) + Holm-Bonferroni correction | option_c_helpers + custom mapping | ⏳ TODO |
| 14 | code | Aggregate JSON save + export to results/oracle_* (renamed cell 67 mirror) | noncausal cell 67 (rename targets) | ⏳ TODO |

## Key implementation details

### Cell 3 (pipeline)

- Construct `NetCDFDataPipeline(..., train_start_date="1980-01-01", train_end_date="2009-12-31", val_start_date="2010-01-01", val_end_date="2011-12-31", test_start_date="2012-01-01", test_end_date="2013-12-31", temporal_holdout_start_date="2014-01-01", temporal_holdout_end_date="2014-12-31")`
- `pipeline.build_sequence_dataset(split="train", stride=2, as_torch=True)` → train_dataset
- `pipeline.build_sequence_dataset(split="val", stride=2, as_torch=True)` → val_dataset
- Wrap in DataLoaders with batch_size=64, micro=16, num_workers=GPU_PROFILE["num_workers"]
- Define `iterate_batches(loader)` helper (yields dicts with lr/residual/baseline/hetero)
- Build builder = HeteroGraphBuilder(...)

### Cell 4 (stack constructors)

- Define `build_fresh_stack(seed)` function that returns `(stack, rcn_cell, load_audit)`:
  - `torch.manual_seed(seed); np.random.seed(seed)`
  - Build encoder, rcn_cell, rcn_runner, regression_head, diffusion, skip_block (per training_eval cells 24-30)
  - load_audit = {} (no warm-start, so empty)
  - Return stack dict + rcn_cell + load_audit
- This function will be called per-seed in cell 6

### Cell 5 (helpers import + globals)

```python
from path_c_plus.scripts.option_c_helpers import (
    compute_q_phys_binary, compute_q_phys_adaptive, compute_q_phys_continuous,
    compute_phys_mag_gained, compute_skeleton_f1,
    install_projection_hook, check_pc4_pc13_gate,
    compute_h1_verdict, load_noncausal_baseline_metrics,
    paired_t_h2_h5, holm_bonferroni_h1_h5, stamp_option_c_json,
)
from src.st_cdgm.training.physics_prior import build_physical_mask
from src.st_cdgm.training.training_loop import train_epoch_stage1, train_epoch_stage2
from src.st_cdgm.training.two_stage import freeze_stage1, gamma_dag_warmup, stage1_compute_loss
G_phys = build_physical_mask(num_vars=NUM_VARS)
G_phys_np = G_phys.numpy()
SEEDS = [42, 7, 123]
K9_DATES = {"train": ["1980-01-01","2009-12-31"], "val": ["2010-01-01","2011-12-31"],
            "test": ["2012-01-01","2013-12-31"], "holdout": ["2014-01-01","2014-12-31"]}
```

### Cell 6 (seed training loop)

For each seed in SEEDS:
- Check if `oracle_full/seed_<n>/results.json` exists → skip with log
- Else:
  - stack, rcn_cell, load_audit = build_fresh_stack(seed)
  - PC4+PC13 gate (will pass trivially since from-scratch)
  - projection_log = install_projection_hook(rcn_cell)
  - A_dag_initial = rcn_cell.A_dag.detach().cpu().numpy()
  - Stage 1 : 15 epochs of train_epoch_stage1 with per-epoch checkpoint save (epoch_last.pth atomic)
  - After Stage 1 : freeze Stage 1 modules
  - Stage 2 : 200 epochs of train_epoch_stage2 with per-epoch checkpoint save
  - A_dag_final = rcn_cell.A_dag.detach().cpu().numpy()
  - Save A_dag_trajectory + projection_log
  - results = {seed, a_dag_initial, a_dag_final, projection_log, ...}
  - results_path.write_text(json.dumps(results)) → triggers next cells

### Cell 7 (per-seed Q_phys eval)

For each seed in SEEDS:
- Load A_dag from oracle_full/seed_<n>/stage2_last.pth
- Compute Q_phys binary / adaptive / continuous / pre_spectral
- Compute phys_mag_gained vs A_dag_initial
- Compute skeleton F1
- Compute n_extra_edges
- Append to per-seed results.json

### Cells 8-11 (eval mirror noncausal)

Inline-port the 5 eval cells from noncausal_training.ipynb (61, 62, 63, 64, 65) with :
- `_run_variant44 = "causal"` (instead of "noncausal")
- Output paths : `oracle_full/seed_<n>/...` (instead of CKPT_SAVE_DIR)
- For cell 64+65 : loop over `EVAL_GCM_TAG ∈ {ACCESS-CM2, EC-Earth3, NorESM2-MM}`
- Files produced :
  - `oracle_full/seed_<n>/final_validation_metrics.json`
  - `oracle_full/seed_<n>/domain_metrics.json`
  - `oracle_full/seed_<n>/eval_samples.npz`
  - `oracle_full/seed_<n>/aligned_metrics_<GCM>_causal.json` × 3

### Cell 12 (cross-seed H1 verdict)

- Collect q_phys_continuous_final per seed from oracle_full/seed_<n>/results.json
- Collect n_extra_edges_final per seed
- Collect q_phys_continuous_collapsed_final per seed
- verdict = compute_h1_verdict(q_cont_per_seed, n_extra_per_seed, collapsed_per_seed)
- Save `oracle_full/h1_verdict.json`

### Cell 13 (H2-H5 cross-seed vs noncausal)

- Load noncausal baseline : `baseline = load_noncausal_baseline_metrics(CKPT_NONCAUSAL_DIR)`
- For each metric (H2 MAE, H3 CDD bias, H4 spread/RMSE, H5 OOD CRPS) :
  - oracle_per_seed = [json.load(oracle_full/seed_<n>/<metric_json>)[metric_key] for seed in SEEDS]
  - noncausal_value = baseline[<metric_json>][metric_key]
  - result_h2 = paired_t_h2_h5(oracle_per_seed, noncausal_value, label=..., delta_threshold=..., direction=...)
- p_values = {H2: result_h2.p_value, H3: result_h3.p_value, ..., H1: from cell 12}
- holm = holm_bonferroni_h1_h5(p_values, alpha=0.05)
- Save `oracle_full/h2_h5_paired_tests.json` + `oracle_full/holm_bonferroni.json`

### Cell 14 (aggregate + export)

- aggregate = {verdict_h1, h1_diagnostics, h2_h5_paired_tests, holm_bonferroni, seeds, ...}
- aggregate = stamp_option_c_json(aggregate, seeds_lineage=SEEDS, k9_dates=K9_DATES, pathcplus_hyperparams=PATHCPLUS_HYPERPARAM_OVERRIDES, pre_registration_commit=commit_sha, is_aggregate=True)
- Save `oracle_full/oracle_full_aggregate.json`
- Export to results/ (rename targets to oracle_metrics.json + oracle_*_ablation.json to NOT clobber noncausal baseline)

## Critical risks

1. **Stage 2 200 epochs on A100 = 25-40h per seed × 3 = 75-120h**. Must implement robust resume per-epoch.
2. **K9 split with stride=2** : ~6927 samples / epoch. Must verify NetCDFDataPipeline can handle this with the K9 fix.
3. **Encoder migration ON from-scratch** : shouldn't fire (no V5_DIR warm-start), but PC13 check is still wired as safety net.
4. **DataLoader iterate_batches** : training_eval.ipynb has a custom helper. Must port it OR import from a shared module.
5. **GCM OOD K23** : the noncausal pipeline uses SAME mean/std (ACCESS-CM2) for OOD GCMs. K23 audit flagged this as questionable. Option C inherits this limitation. Document in thesis caveat.
6. **Holm-Bonferroni on n=3** : with only 3 seeds, paired t-test power is low. Holm correction across 5 hypotheses with k=5 multiplier makes it even more conservative. Math Prof may flag this.
