# Council patch log — Option C notebook v1.1

Notebook : `path_c_plus/scripts/st_cdgm_path_c_option_c.ipynb`
Date : 2026-06-13
Status : **GO** (after council patches applied)

## Reviewer verdicts (pre-patch)

| Reviewer | Verdict | Blockers |
|---|---|---|
| AI Engineer | GO-WITH-CHANGE | 3 |
| Mathematics Prof | GO-WITH-CHANGE | 2 |
| Data Scientist | GO-WITH-CHANGE | 2 |
| Reviewer / SRE | GO-WITH-CHANGE | 4 |

11 blockers total, 4 convergences. Patched in single sweep.

## Patches applied

### Cell 0 (markdown)
- DS #2 : tightened budget claim 90-150h → 85-120h within PC14 #8 cap, with Option C-bis amendment guidance if exceeded.

### Cell 5 (helpers)
- **SRE B1** : `_atomic_save_pth` now fsyncs the tmp file body before `os.replace` AND fsyncs the parent directory entry after rename. Mirrors noncausal commit f76e8f5. Prevents zero-byte ckpt artifacts on Colab Drive FUSE kill.
- **DS #6** : added `PATHCPLUS_LOCKED_COMMIT = "78a3783"` constant (A1 H1_PASS commit). Distinct from `PRE_REGISTRATION_COMMIT` (current HEAD).

### Cell 6 (seed loop)
- **SRE O4** : orphan tmp cleanup at top of `train_one_seed` (deletes `epoch_last.pth.XXX.tmp` from prior killed sessions).
- **AI Eng B3** : rebuild `train_dataset_seed = pipeline.build_sequence_dataset(...)` per seed inside `train_one_seed`, replacing the cell-3 global (which is an IterableDataset and gets partially drained across seeds).
- **AI Eng B1+B3** : SAGEConv lazy-param materialization block now runs for BOTH fresh runs AND resume (`s1_from == 0 or resume is not None`).
- **AI Eng risk #2** : `train_epoch_stage1(lambda_dag_prior=...)` now reads `sched["lambda_dag_prior"]` (consistent with `HP_OPTION_C`) instead of `ts_cfg.stage1.lambda_dag_prior`.
- **SRE B3** : BS32b cache (`{seed_dir}/stage1_cache.pt`) is DROPPED if `s1_from < S1_EPOCHS` on resume. Prevents Stage 2 from training against stale mu_HR / baseline_log from a partial Stage 1.
- **SRE B2 + AI Eng B2 + SRE B4** : master loop completely rewritten.
  - Split skip-gate : `train_done = results.json exists`, `eval_done = 5 eval files exist`.
  - `_maybe_eval(seed)` called INSIDE the per-seed loop right after training succeeds → kernel kill between seeds preserves done-seeds' eval JSONs.
  - try/except wraps `train_one_seed(_seed)` : O3 failure or any RuntimeError writes `PROTOCOL_FAILURE.json` and continues to the next seed. One failed seed no longer kills the whole 3-seed run.
  - `LIVE_SEED_CONTEXT` removed (dead).

### Cell 8 (full eval pipeline)
- **AI Eng B1** : SAGEConv lazy-param materialization forced via one no-grad forward BEFORE `_persist_load_state_dict` writes encoder weights. Otherwise eval-time encoder runs with fresh (random) weights and every eval JSON would be wrong.
- **SRE I3** : per-file skip for `final_validation_metrics.json` / `domain_metrics.json` / `eval_samples.npz`. Avoids ~50min wasted re-sampling on partial re-entry.
- **DS B1 + Math Prof B2** : inline CRPS Gaussian (closed-form, log1p space, ensemble mean+std) computed during the BS44 per-GCM loop. The CRPS value is written to each `aligned_metrics_<GCM>_causal.json` under key `crps_gaussian_log1p`. Cell 10 reads it for the OOD H5 endpoint.

### Cell 9 (H1 verdict)
- **Math Prof B1** : `random_null` is no longer hard-coded 0.083. Recomputed at runtime from `G_phys` :
  `random_null = 0.5 * n_phys_edges / (NUM_VARS * (NUM_VARS - 1))`.
  Notebook prints both the computed value and the pre-reg 0.083, with a warning if they differ by > 0.01.
- **Math Prof I6** : Hedges' g added alongside Cohen's d. Small-sample bias correction `J = 1 - 3/(4·df - 1) ≈ 0.571` for n=3.
- Cohen's d note added : "uses sd_floor=0.05 → d is a CONSERVATIVE LOWER BOUND".

### Cell 10 (H2-H5 tests)
- **DS B1 + Math Prof B2** : H5 endpoint switched from `domain_metrics.crps_gaussian` (in-distribution ACCESS-CM2) to mean of `aligned_metrics_<GCM>.crps_gaussian_log1p` over `OOD_GCMS = ["EC-Earth3", "NorESM2-MM"]`. Per pre-registration H5 spec.
- Noncausal H5 also reads OOD CRPS via `_dig(baseline, "aligned_metrics_per_gcm", _gcm, "crps_gaussian_log1p")`. If absent (current `ckpt_noncausal/` doesn't have it), H5 returns NaN → Holm-Bonferroni still uses k=4 (over-corrects, conservative).
- **Math Prof I3** : `metric_transforms` dict added documenting `abs(.)` for H3 and `abs(1 - .)` for H4.
- **Math Prof I5** : `h2_h5_caveats` dict added with explicit notes on :
  - noncausal-as-zero-variance constant (test_type = one-sample t, df=2)
  - H5 endpoint clarification (OOD averaged over 2 GCMs)
  - H5 skip behavior if noncausal OOD CRPS missing
  - Holm family size k=4 decision rationale
  - K23 OOD limitation

## Items deliberately NOT patched (acknowledged risks)

- **AI Eng risk : `g_phys_alpha` dead in this training path.** `train_epoch_stage1` doesn't read it; only `finetune_bundle_b` does. Path C+ Option C trains via `train_epoch_stage1`, so `g_phys_alpha=0.25` has no effect. Kept in `PATHCPLUS_HYPERPARAM_OVERRIDES` for documentation/parity with A1 stamp.
- **DS #3 : probe stack in cell 5 wastes ~30s.** Kept for correctness (needs `num_vars` from a real encoder build).
- **DS #4 : `spatial_projector` unused in causal_concat mode.** Architecturally correct (concat conditioning bypasses cross-attention); the projector is in checkpoints but receives no gradient. Cosmetic.

## Validation

- All 12 cells parse without syntax errors (Python AST validated).
- All notebook imports resolve against `src/st_cdgm/` and `path_c_plus/scripts/option_c_helpers.py`.
- Total notebook size : ~110k chars (was ~75k pre-patch, +35k from council fixes).

## Go-no-go

**GO** for the 85-120h A100 Colab Pro+ run, 3 seeds [42, 7, 123], with Option C as the primary H1-H5 test per PC14 #1.

Pre-launch checklist (operator-side) :
- Verify `ckpt_noncausal/` is at epoch=200 (already confirmed earlier).
- Optional : re-eval `ckpt_noncausal/` with the patched BS44 inline CRPS to populate noncausal OOD CRPS so H5 doesn't degrade to NaN. If skipped, H5 is reported SKIPPED in the aggregate.
- Set `K_SAMPLES_OVERRIDE` global if you want fewer than K=64 ensemble samples to save eval time.
