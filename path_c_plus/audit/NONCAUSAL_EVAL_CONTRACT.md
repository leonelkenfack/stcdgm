# NONCAUSAL_EVAL_CONTRACT

Reference contract extracted from `st_cdgm_noncausal_training.ipynb` (cells 014, 061, 062, 063, 064, 065, 067) for reuse with `run_variant="causal"`.

---

## Cell 014 — BS31b CONFIG SELECT

**Produces:** in-memory `CONFIG` (OmegaConf) merged from `config/training_config.yaml` + override.

**Requires:** env `CONFIG_FILE` (optional), files in `config/`.

**Run-variant sensitivity:** default override is `training_config_noncausal.yaml` (=> `two_stage.run_variant=noncausal`). For Option C, set `os.environ["CONFIG_FILE"]="training_config_causal.yaml"` BEFORE this cell, or directly point to the V5/causal override.

**Key parameters used downstream:**
- `CONFIG.diffusion.eval_num_steps` (default 18 causal / 30 noncausal)
- `CONFIG.diffusion.scheduler_type` (default `edm_karras` causal / `dpm_solver++` noncausal)
- `CONFIG.diffusion.cfg_scale` (default 0.0)
- `CONFIG.diffusion.unet_kwargs.block_out_channels`
- `CONFIG.checkpoint.save_dir` -> `CKPT_SAVE_DIR`
- `CONFIG.two_stage.run_variant` -> read by `resolve_run_variant`

---

## Cell 061 — FINAL_VALIDATION (BS30)

**Produces:** `{CKPT_SAVE_DIR}/final_validation_metrics.json`

**Requires (globals in scope):**
- `CONFIG`, `DEVICE`, `builder`, `val_dataloader`, `iterate_batches`
- `diffusion` (mandatory), `encoder`, `rcn_cell`, `rcn_runner`, `regression_head`, `spatial_projector`, `hr_ident_head` (any can be `None` in noncausal; required for causal)
- Helper `_persist_load_state_dict` (from cell 47, BS22/23/24)
- Imports: `st_cdgm.evaluation.compute_f1_extremes`, `compute_spectrum_distance`, `st_cdgm.evaluation.two_stage_inference.build_two_stage_inputs`, `st_cdgm.training.stage1_paths.resolve_run_variant`
- Checkpoint at `CONFIG.checkpoint.save_dir / "epoch_last.pth"` (fallback `epoch_best.pth`)

**Sampling protocol:**
- `K_SAMPLES = int(globals().get("K_SAMPLES_OVERRIDE", 64))` (ensemble size)
- `N_TEST_BATCHES = 16`
- `N_INTERVENTION = 4` (μ_HR ablation batches)
- `num_steps`: `CONFIG.diffusion.eval_num_steps`, default 18 if `causal_concat` else 30
- `scheduler_type`: `CONFIG.diffusion.scheduler_type`, default `edm_karras` if causal else `dpm_solver++`
- `cfg_scale`: `CONFIG.diffusion.cfg_scale`, default 0.0
- `apply_constraints=False`
- EMA weights preferred unless `BS41_FORCE_LIVE_INFERENCE=True`

**Computes Q_phys?** NO. Q_phys / hard physical constraints are NOT computed here (`apply_constraints=False`).

**μ_HR ablation?** YES, but ONLY when `_causal_concat=True` AND `_mu_HR is not None`. Compares `_sample_once(μ_HR)` vs `_sample_once(0)` → `Δ/signal` ratio over `N_INTERVENTION=4` batches; verdicts: `MU_HR_IGNORED` (<0.1%), `WEAK` (<1%), `MU_HR_CONDITIONS` (>=1%).

**Metrics computed on `_pred_full = _pred_mean + _mu_HR` (BS31f "full prediction"):**
RMSE, MAE, spread, Pearson global, Pearson per-sample avg, F1@p95, F1@p99, RAPSD distance (on sample 0 only), shortcut diagnostic (residual collapse probe).

**JSON top-level schema (`final_validation_metrics.json`):**
```
checkpoint, epoch, epochs_total, best_val_loss, causal_concat,
n_test_batches, k_samples, metrics_scope, eval_time_s,
rmse, mae, spread_mean,
f1_extremes: {f1_p95.0, f1_p99.0, ...},
pearson_corr: {global, per_sample_avg, per_sample_n, per_sample_list[<=64]},
rapsd_distance,
mu_HR_ablation: {delta_signal_ratio_avg, per_batch[], verdict},
shortcut_diagnostic: {norm_output, norm_mu_HR, norm_target,
                       norm_output_minus_mu_HR, norm_target_minus_mu_HR,
                       shortcut_ratio, verdict},
config_eval_num_steps, config_cfg_scale, config_block_out_channels
```

**Variables produced for downstream cells:** `_ckpt_dir`, `_pred_mean`, `_pred_std`, `_pred_full`, `_targets`, `_valid`, `_mu_concat`, `_rmse`, `_mae`, `_spread`, `_corr_global`, `_rapsd_d`, `_build_inputs`, `_sample_once`, `K_SAMPLES`.

---

## Cell 062 — BS42 DOMAIN-ALIGNED METRICS

**Produces:** `{CKPT_SAVE_DIR}/domain_metrics.json`

**Requires (globals from cell 061):** `_pred_full`, `_pred_std`, `_targets`, `_valid`, `_rmse`, `_mae`, `_spread`, `_corr_global`, `_rapsd_d`, `CKPT_SAVE_DIR`. Imports `torch`.

**Metrics computed:**
1. Spread-skill ratio `_spread / _rmse` (target ~1.0)
2. CRPS Gaussian closed-form (Gneiting & Raftery 2007) on `(μ=_pred_full, σ=_pred_std, y=_targets)` in log1p mm/day space
3. Intensity-histogram L1 distance (proxy LHD), 100 bins between joint [min,max]
4. RAPSD/RALSD (reused from cell 061)
5. Secondary refs: RMSE, MAE, spread_mean, Pearson global

**NOT computed here:** Rx1Day, CDD (require continuous per-pixel time series → done in BS44). F1@p95/p99 are NOT recomputed (they were done in cell 061).

**JSON schema (`domain_metrics.json`):**
```
spread_skill_ratio, crps_gaussian, intensity_hist_distance_L1,
rapsd_distance, rmse_secondary, mae_secondary, spread_mean,
pearson_global_secondary
```

**Run-variant sensitivity:** none directly — operates on whatever tensors cell 061 produced.

---

## Cell 063 — BS43 EXPORT EVAL SAMPLES

**Produces:** `{_ckpt_dir or CKPT_SAVE_DIR}/eval_samples.npz`

**Requires (globals):** `_targets`, `_pred_full`, `_pred_std`, `_valid`, `_mu_concat` (optional), `CONFIG`.

**Payload (npz):**
- `target` (N≤8, …), `pred_full`, `pred_std`, `valid_mask` (float32), `mu_HR` (if `_mu_concat` exists), `run_variant` (scalar unicode array)

**Run-variant sensitivity:** writes `run_variant = CONFIG.two_stage.run_variant` (default `"causal"` if missing). For causal run this becomes `"causal"`.

---

## Cell 064 — BS45 GCM DATALOADER

**Produces:** globals `ALIGNED_EVAL_LOADER`, `EVAL_GCM_TAG`, `EVAL_IN_DIST` (consumed by BS44).

**Requires:** `st_cdgm.data.pipeline.NetCDFDataPipeline`; pipeline kwargs read from globals (with defaults): `DATA_ROOT`, `STATIC_PATH`, `MEAN_PATH`, `STD_PATH`, `SEQ_LEN=16`, `BASELINE_STRATEGY="hr_smoothing"`, `BASELINE_FACTOR=4`, `NORMALIZE=False`, `NAN_FILL_STRATEGY="zero"`, `PRECIPITATION_DELTA=0.01`, `LR_VARIABLES`, `HR_VARIABLES`, `STATIC_VARIABLES`, `BATCH_SIZE=64`, `NUM_WORKERS=0`, `PIN_MEMORY=False`.

**GCM registry (relative to `DATA_ROOT`):**
| Tag | LR path | HR path | in_dist |
|---|---|---|---|
| ACCESS-CM2 | `train/predictor_ACCESS-CM2_hist.nc` | `train/pr_ACCESS-CM2_hist.nc` | True |
| EC-Earth3 | `test/EC-Earth3_histupdated_compressed.nc` | `test/EC-Earth3_historical_precip_compressed.nc` | False |
| NorESM2-MM | `test/NorESM2-MM_histupdated_compressed.nc` | `test/NorESM2-MM_historical_precip_compressed.nc` | False |

**Normalisation for OOD GCMs:** NO separate stats are loaded. The cell reuses the SAME `MEAN_PATH` / `STD_PATH` from training (K23 stats). The drop-in assumption is that the 15 LR variables (t/u/v/w/q × 850/500/250) and the LR 23×26 / HR 172×179 grid match exactly, so ACCESS-CM2 training means/stds apply. The pipeline is built with `means_path=_mean45 if exists else None` — same for both in-dist and OOD.

**Dataset settings:** `stride=1`, `drop_last=True`, `training=False`, `as_torch=True`. DataLoader has no shuffle (IterableDataset preserves temporal order — critical for CDD/Rx1Day).

**Run-variant sensitivity:** none (data is the same for both variants).

---

## Cell 065 — BS44 ALIGNED METRICS cGAN

**Produces:** `{_ckpt_dir}/aligned_metrics_{EVAL_GCM_TAG}_{run_variant}.json`

**Requires (globals):** `ALIGNED_EVAL_LOADER` (or `val_dataloader`), `iterate_batches`, `builder`, `DEVICE`, `_build_inputs`, `_sample_once`, `CONFIG`, `_ckpt_dir`/`CKPT_SAVE_DIR`. Imports `st_cdgm.evaluation.aligned_eval.run_aligned_eval`.

**Key parameters:**
- `ALIGNED_K_SAMPLES = min(K_SAMPLES, 16)` (ensemble for residual mean)
- `space="log1p"`, `thresh=1.0` (mm/day in physical space after expm1)
- Per-batch reconstruction: `full_log = baseline_log + μ_HR + δ̂_mean`, same for truth (with `_tgt` instead of `_ens`)
- Time axis: real `batch["time"]` if available, else synthetic daily `datetime64[D]`

**Rampal-style metrics produced** (via `run_aligned_eval`): CDD, Rx1Day, R10, seasonal indices, PSD (RALSD) — all in the JSON under `indices` + `psd_distance`. The cell prints `*_bias` indices and `psd_distance`.

**Run-variant sensitivity:** `_run_variant44 = CONFIG.two_stage.run_variant` is written into the JSON filename AND passed to `run_aligned_eval`. Reconstruction handles `_muHR is None` (noncausal) by substituting 0.0.

---

## Cell 067 — BS40 EXPORT V4 ARTIFACTS

**Produces (copies):**
- `results/v4_metrics.json`        ← `{CKPT_SAVE_DIR}/final_validation_metrics.json`
- `results/v4_bs35_ablation.json`  ← `ablation_suite_{RUN_VARIANT}.json` (if exists)

**Requires:** globals `CKPT_SAVE_DIR`, `RUN_VARIANT`.

**Run-variant sensitivity:** filename `v4_*` is HARD-CODED → for Option C causal run, rename targets to `oracle_metrics.json` / `oracle_bs35_ablation.json` (or `v5_*`) to avoid clobbering the noncausal baseline already in `results/`.

---

## Build order (call sequence + required scope)

1. **Cell 014** — sets `CONFIG`. Prerequisite globals: `os.environ["CONFIG_FILE"]` (optional).
2. **Training cells** (not in scope of this audit) must produce: `DEVICE`, `builder`, `val_dataloader`, `iterate_batches`, `encoder`, `rcn_cell`, `rcn_runner`, `diffusion`, `regression_head`, `spatial_projector`, `hr_ident_head`, `_persist_load_state_dict`, `CKPT_SAVE_DIR`, `RUN_VARIANT`, and the checkpoint file `epoch_last.pth`.
3. **Cell 061 FINAL_VALIDATION** — exports `final_validation_metrics.json`; leaves in scope: `_ckpt_dir, _pred_mean, _pred_std, _pred_full, _targets, _valid, _mu_concat, _rmse, _mae, _spread, _corr_global, _rapsd_d, _build_inputs, _sample_once, K_SAMPLES`.
4. **Cell 062 BS42** — needs (3) globals; exports `domain_metrics.json`.
5. **Cell 063 BS43** — needs (3) globals; exports `eval_samples.npz`.
6. **Cell 064 BS45** — needs `DATA_ROOT` + pipeline globals + `EVAL_GCM_TAG`; sets `ALIGNED_EVAL_LOADER`. Run THREE TIMES with `EVAL_GCM_TAG ∈ {ACCESS-CM2, EC-Earth3, NorESM2-MM}`, each followed by cell 065.
7. **Cell 065 BS44** — needs (3) helpers `_build_inputs`, `_sample_once` + `ALIGNED_EVAL_LOADER` from (6); exports `aligned_metrics_{GCM}_{variant}.json` (one per GCM).
8. **Cell 067 BS40** — copy final + ablation JSON into `results/`. For causal: rewrite target filenames to avoid overwriting noncausal artifacts.

## Run-variant switch checklist (noncausal → causal)

- Cell 014: set `CONFIG_FILE` to a causal override so `CONFIG.two_stage.run_variant == "causal"`, which flips `causal_concat=True`, `eval_num_steps=18`, `scheduler=edm_karras`, and activates the μ_HR ablation in cell 061.
- Ensure `encoder`, `rcn_cell`, `rcn_runner`, `regression_head` are NOT None (built by training).
- Cell 067: rename copy targets from `v4_*` to `oracle_*` / `v5_*` to preserve baseline outputs.
- Cell 065: filename auto-discriminates via `_run_variant44` — no manual change needed.
