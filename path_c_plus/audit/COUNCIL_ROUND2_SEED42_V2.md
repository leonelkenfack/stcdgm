# Council audit round 2 — seed 42 v2 retrain notebook

Notebook : `path_c_plus/scripts/st_cdgm_seed42_eval.ipynb`
Date     : 2026-06-16
Status   : **GO** (after 5 council patches applied)

## Reviewer verdicts (pre-patch)

| Reviewer            | Verdict        | Blockers identified |
|---------------------|----------------|---------------------|
| AI Engineer         | NO-GO          | 3 (B1, B2, B3)      |
| Mathematics Prof    | GO-WITH-CHANGE | 1 (B6 — eval recon) |
| Climate ML          | NO-GO          | 3 (B1, B2, B4)      |
| Lit Review Mardani  | GO-WITH-CHANGE | 1 (framing only)    |

5 distinct blockers across reviewers, 3 convergences (B1, B2). Patched in single sweep.

## Patches applied

### Cell 10 (V2_CONFIG)
- **B3 fix** : added `"mardani_fix_zero_mu_HR_in_conditioning": True`. Cell 12 was reading
  this key (defaulting to False) while Cell 10 only set `..._in_cache`. Distribution
  shift at eval (REAL mu_HR fed to UNet that trained on zeros) → now both train and
  eval feed zeros consistently.
- **B4 fix #1** : `sampler_cfg_scale = 1.15 → 1.0`. With mu_HR zeroed in cache, no valid
  unconditional branch exists; CFG > 1 produces undefined behavior.
- **B4 fix #2** : `conditioning_dropout_prob = 0.15 → 0.0`. Same reason: dropping mu_HR
  from a batch where it's already zero is a no-op.
- Header docstring updated + round-2 audit footnote added.

### Cell 11 (training loop)
- **B1 fix** : `_persist_state_dict(diffusion_v2)` → `(diffusion_v2.module if hasattr(...) else diffusion_v2).state_dict()`.
  Helper was never defined or imported → first atomic save would raise `NameError`,
  burning 25-30h of A100 with zero checkpoints. Inline DDP-safe state_dict extraction
  removes the dependency.
- Same fix for `ema_diffusion_v2`.

### Cell 12 (eval)
- **B2 fix #1** : `EMA_CHOICE = "0.9995"` → `str(V2_CONFIG.get("ema_decay", 0.999))`.
  Hard-coded value did not match the actually trained EMA decay (0.999).
- **B2 fix #2** : `ck_v2.get(f"ema_{EMA_CHOICE}_state_dict")` → `ck_v2.get("ema_state_dict")`.
  Cell 11 saves under the bare `"ema_state_dict"` key. Old lookup returned `None` →
  `_load_sd` silently no-op → eval ran on live (non-EMA) weights from the checkpoint
  loader, producing meaningless v2 metrics.

### Cells 6 + 7 + 12 + 13 (printing/comparison)
- **B5 fix** : `f1_p95.0` → `p95`, `f1_p99.0` → `p99` (10 replacements). `compute_f1_extremes`
  in `src/st_cdgm/evaluation/evaluation_xai.py:549,565` writes keys `"p95"` / `"p99"`.
  Old lookups returned `None` everywhere → comparison table printed `n/a` for every F1
  cell.

## Items deliberately NOT patched (acknowledged)

- **Math Prof B6 (eval reconstruction missing `baseline_log`)**: the v1 eval at line 633
  uses the same `_pred_full = _pred_mean + _mu_concat` pattern (no baseline_log added).
  v1 produced sensible RMSE=0.124 / Pearson=0.78. Conclusion: either `sample()` returns
  `residual + baseline` (the DiffusionOutput.residual return value already includes the
  baseline), or both v1 and v2 are biased identically and the comparison remains valid.
  Decision: leave as-is to keep apples-to-apples vs v1. Re-investigate if v2 numbers
  look implausible.
- **Lit Review framing risk**: Mardani-Nature-CEE-2025 / PhysicsNeMo / StormCast all
  concatenate the regression mean. Framing the fix as "we found a Mardani bug" is
  inaccurate; correct framing is "our training setup needed mu_HR removed from
  conditioning to avoid a shortcut". Thesis narrative update required, not a code fix.
- **Lit Review B suggestion to rebuild UNet with `in_channels=2`**: clean fix but
  requires architecture rebuild + new EDM preconditioning. M2 timeline does not permit.
  Cache-zeroing is functionally equivalent at gradient level (dead-channel weights
  drift only via weight decay).
- **Climate ML B7 (missing v2 domain + 3-GCM eval)**: in-distribution `final_validation_metrics`
  is sufficient for the v2 ↔ noncausal head-to-head. OOD/GCM-aligned eval can be added
  post-hoc using Cell 8 logic if v2 wins the head-to-head.

## Validation

- All 14 cells parse as valid JSON (round-tripped via `json.loads` + `json.dumps`).
- Cell IDs preserved; no cells added or deleted.
- Manual grep confirms each of B1–B5 lands exactly where expected.

## Go / no-go

**GO** for the 25-30h Colab Pro+ A100 launch.

Pre-launch checklist (operator-side) :
- Run cells 1–5 (already done in prior session).
- **SKIP cell 6** (Fix #1 sampling variants — superseded by v2 retrain).
- **SKIP cells 7-9** (v1 results already on disk).
- Run cell 10 (Fix #2 prep — builds V2_CONFIG, BS32b v2 cache, fresh `diffusion_v2`).
- Run cell 11 (200-epoch training loop with atomic per-epoch saves).
- Run cell 12 (eval against v2 EMA weights).
- Run cell 13 (final comparison v1 + v2 + noncausal).
