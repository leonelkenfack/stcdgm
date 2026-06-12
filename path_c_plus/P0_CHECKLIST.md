# P0 Fixes Checklist — 22 commits planifiés

**Stratégie** : 1 fix = 1 commit. Traçabilité maximale, `git bisect`-ready.

**Ordre** : dépendances en priorité (un fix doit pouvoir être testé indépendamment).

## Batch A — Quick wins config/docs (commits 1-6)

Ces fixes touchent à config/YAML/scripts isolés, faible risque.

| # | Commit | Issue | Files | Effort |
|---|---|---|---|---|
| 1 | `chore: pin package versions (K27)` | K27 | `requirements.txt` | 30 min |
| 2 | `config: add temporal split fields (K8)` | K8 | `config/training_config.yaml` + helper | 1h |
| 3 | `feat: add seed parameter to recompute_phase6 (K16)` | K16 | `scripts/recompute_phase6_metrics.py` | 30 min |
| 4 | `fix: physical_prior_loss without /off_diag normalization (I1)` | I1 | `src/st_cdgm/training/physics_prior.py:201` | 30 min |
| 5 | `feat: assert causal_concat consistency in eval (K1)` | K1 | `scripts/recompute_phase6_metrics.py` | 1h |
| 6 | `feat: warn if epochs_total < epochs_max (K30)` | K30 | `scripts/finetune_stage1_bundle_b.py` | 30 min |

## Batch B — Training loop critical fixes (commits 7-12)

Cœur du training, plus de risque mais petits diffs.

| # | Commit | Issue | Files | Effort |
|---|---|---|---|---|
| 7 | `fix: wire dag_grad_gate in finetune training loop (consensus 1.1)` | §1.1 | `scripts/finetune_stage1_bundle_b.py:151` | 30 min |
| 8 | `fix: DAGMA s param dynamic from Gershgorin (consensus 1.2)` | §1.2 | `scripts/finetune_stage1_bundle_b.py:258` | 1h |
| 9 | `fix: replace biased sigma_data with calibrate_sigma_data_variant (consensus 1.3)` | §1.3 | `scripts/finetune_stage1_bundle_b.py:671-725` | 2h |
| 10 | `fix: RNG state save/restore on resume (consensus 1.4)` | §1.4 | `scripts/finetune_stage1_bundle_b.py:592-614` | 1h |
| 11 | `fix: strict=True default on resume (J18)` | J18 | `scripts/finetune_stage1_bundle_b.py:546-555` | 30 min |
| 12 | `feat: clip_grad_norm on all param_groups (J12)` | J12 | `src/st_cdgm/training/training_loop.py:1494-1500` | 1h |

## Batch C — Architecture fixes (commits 13-17)

Modifs `src/st_cdgm/models/`, plus de risque, demandent tests rigoureux.

| # | Commit | Issue | Files | Effort |
|---|---|---|---|---|
| 13 | `fix: eager state_adapter init in RegressionMeanPredictor (J1)` | J1 | `src/st_cdgm/models/regression_mean_predictor.py:186-190` | 1h |
| 14 | `fix: HeteroConv aggr=cat + per-target Linear (consensus 1.6 + J4)` | §1.6 + J4 | `src/st_cdgm/models/intelligible_encoder.py:92-130` | 4h |
| 15 | `fix: per-metapath LayerNorm (J3)` | J3 | `src/st_cdgm/models/intelligible_encoder.py:97-100` | 1h |
| 16 | `fix: ConditionalSkipBlock proper LR grid input (J8)` | J8 | `scripts/finetune_stage1_bundle_b.py:227-229` + `src/st_cdgm/models/skip_direct.py` | 2h |
| 17 | `fix: CASTLE H_t detach (I11)` | I11 | `src/st_cdgm/models/causal_rcn.py:755-761` | 30 min |

## Batch D — Eval pipeline integrity (commits 18-22)

Eval metrics, comparaison Oracle vs CorrDiff, statistical validity.

| # | Commit | Issue | Files | Effort |
|---|---|---|---|---|
| 18 | `feat: CFG assertion in edm_karras path (J29)` | J29 | `src/st_cdgm/models/diffusion_decoder.py` | 1h |
| 19 | `fix: F1 threshold per-pixel climatology (K2 + I8)` | K2 + I8 | `src/st_cdgm/evaluation/evaluation_xai.py:531-533` | 4h |
| 20 | `fix: pred_full vs target space alignment (K3)` | K3 | `scripts/recompute_phase6_metrics.py:239` | 2h |
| 21 | `fix: temporal split (not random) for time series (K9 + 1.5)` | K9 + §1.5 | `_cell43.py:38-40` + helper | 1 jour |
| 22 | `fix: normalization stats train-period only (K5)` | K5 | `src/st_cdgm/data/pipeline.py:647-649` | 2h |

## Tests par batch

Après chaque batch, tests à exécuter :

```bash
# Batch A : pas de test runtime, juste validation YAML/imports
python -c "from src.st_cdgm.training.physics_prior import physical_prior_loss; print('OK')"

# Batch B : training loop smoke
pytest path_c_plus/tests/test_training_loop_p0_fixes.py -v

# Batch C : architecture forward pass
pytest path_c_plus/tests/test_architecture_p0_fixes.py -v

# Batch D : eval pipeline + statistical
pytest path_c_plus/tests/test_eval_pipeline_p0_fixes.py -v

# After all 22 commits :
pytest path_c_plus/tests/ -v
```

## Estimation totale

- Batch A : ~4-6 heures dev
- Batch B : ~5-7 heures dev
- Batch C : ~8-12 heures dev
- Batch D : ~9-13 heures dev + 1 jour pour Batch D-21 (temporal split)
- **Total** : 3-5 jours dev intensif (sans compute)

Avec tests + debug imprévu : **5-7 jours dev** réalistes.

## Validation finale Phase 0.2

Avant de passer à Phase 0.3 (smoke test) puis Phase A0' :

- [ ] 22 commits sur branch `four-node-causal`
- [ ] Tous les tests pytest verts (path_c_plus/tests/)
- [ ] Encoder representation CI corr < 0.5 (pre-flight Phase 0.3)
- [ ] Smoke test 5-epoch end-to-end OK sur T4
- [ ] `git log --oneline four-node-causal ^two-stage-causal | wc -l` ≥ 23 (22 fixes + commit 0)
- [ ] README + HYPERPLAN + LAUNCH_COLAB_A100 + PRE_REGISTRATION commités

## Progress tracking

Mettre à jour cette section après chaque commit :

```
[ ] 1.  K27 — Pin package versions
[ ] 2.  K8  — Temporal split fields config
[ ] 3.  K16 — Seed param recompute_phase6
[ ] 4.  I1  — physical_prior_loss /off_diag fix
[ ] 5.  K1  — causal_concat consistency assert
[ ] 6.  K30 — epochs_total warn
[ ] 7.  §1.1 — dag_grad_gate wired
[ ] 8.  §1.2 — DAGMA s dynamic
[ ] 9.  §1.3 — sigma_data unbiased
[ ] 10. §1.4 — RNG state save/restore
[ ] 11. J18  — strict=True resume
[ ] 12. J12  — clip_grad_norm all groups
[ ] 13. J1   — state_adapter eager
[ ] 14. §1.6 — HeteroConv aggr=cat
[ ] 15. J3   — per-metapath LayerNorm
[ ] 16. J8   — skip_block LR grid
[ ] 17. I11  — CASTLE H_t detach
[ ] 18. J29  — CFG assertion edm_karras
[ ] 19. K2   — F1 per-pixel climatology
[ ] 20. K3   — pred_full vs target
[ ] 21. K9   — temporal split
[ ] 22. K5   — normalization train-only
```
