# Hyperplan Path C+ — Version consensus FINALE (3 experts, audit 100% du code)

**Statut** : Plan révisé après audit indépendant de 100% du code par un Professeur de Mathématiques, un Ingénieur IA senior, et un Data Scientist senior.

**Date** : 2026-06-12
**Branch cible** : `four-node-causal` (à créer depuis `two-stage-causal`)

**Total findings** : **95 issues identifiés** (10 consensus initial + 16 math prof I1-I16 + 42 AI engineer J1-J42 + 37 data scientist K1-K37), dont **22 P0/CRITICAL** qui invalident la validité des résultats actuels.

---

## ⚠️ ALERTE MÉTHODOLOGIQUE MAJEURE

Trois découvertes du second tour d'audit, prises ensemble, **remettent en cause la validité scientifique du baseline V5-mini lui-même** :

1. **K1** (data scientist) : Les JSONs Oracle ET CorrDiff montrent `causal_concat: true`. Cela signifie que la baseline CorrDiff a été évaluée avec le **chemin causal-concat** (incluant l'injection de `mu_HR` causal), pas avec le sampling CorrDiff vanilla. **La comparaison Oracle vs CorrDiff est apples-to-mangoes** — les deux modèles utilisent le même code path causal au moment de l'évaluation.

2. **J29** (AI engineer) : `_sample_edm_karras` n'implémente PAS le classifier-free guidance. Tous les configs spécifiant `scheduler_type: "edm_karras"` AND `cfg_scale: 1.5` (= `training_config_corrdiff_normal.yaml`) ont vu leur cfg_scale **silencieusement ignoré**. **Les nombres Pearson 0.815 ne sont pas reproductibles** à partir des configs publiés tels quels.

3. **K30** (data scientist) : Le JSON Phase 6 montre `epoch: 200, epochs_total: 10`. Le modèle a tourné **seulement 10 epochs sur les 200 configurés**. Avec ~1000 optimizer steps pour un UNet 50M params, le modèle est **sévèrement sous-entraîné**. Les chiffres reportés ne représentent pas la convergence du modèle.

**Implication** : avant tout retrain Path C+, il faut établir un baseline V5-mini correctement entraîné ET correctement évalué. Le baseline actuel ne peut pas servir de référence statistique.

---

## Résumé exécutif

Le plan original Path C+ proposait un refactor 6→4 nœuds + PCMCI init + Stage 1+2 retrain en 18-25h Colab sur 3 semaines. **L'audit exhaustif a identifié 95 issues totaux dont 22 P0** invalidant la baseline et compromettant le retrain.

### Verdict des experts (round 2)

| Expert | Verdict | Nouveaux findings |
|---|---|---|
| **Math Professor** | Plan mathématiquement incomplet + bugs formules | 16 nouveaux (I1-I16) |
| **AI Engineer** | Plan techniquement intenable + 6 P0 archi/training | 42 nouveaux (J1-J42) |
| **Data Scientist** | Plan statistiquement non-publiable + baseline invalide | 37 nouveaux (K1-K37) |

### Décision consensus FINALE

**Avant Path C+**, exécuter dans cet ordre strict :

1. **Phase 0** (semaine 1) : Correction des 22 bugs P0 + définition train/val/test temporal split + audit causal_concat
2. **Phase A0'** (semaine 2) : **RE-EVALUATION** du baseline V5-mini avec protocole corrigé (CFG fix, causal_concat aligné, n_batches=64)
3. **Phase A0''** (semaine 2) : **RE-TRAINING** V5-mini complet (200 epochs réels, pas 10) pour avoir une vraie baseline
4. **Phase A1** (semaine 3) : 6-node minimal fix avec dag_grad_gate wired + HeteroConv fix + LayerNorm separation + struct_W properly initialized
5. **Decision gate** : si A1 atteint Q_phys ≥ 0.65 avec baseline corrigée → publication
6. **Phase C+** (semaines 4-8) : si A1 échoue → refactor 4-nœuds avec corrections math/AI/DS

---

## 1. Findings consensus consolidé (95 issues)

### 1.1 Issues bloquants identifiés au premier round (consensus §1.1-1.10)

| # | Issue | File | Severity |
|---|---|---|---|
| 1.1 | `dag_grad_gate` jamais wired | `finetune_stage1_bundle_b.py` | P0 |
| 1.2 | DAGMA `s=1.0` hardcodé | `finetune_stage1_bundle_b.py:258` | P0 |
| 1.3 | Sigma_data biased Jensen | `finetune_stage1_bundle_b.py:701-707` | P0 |
| 1.4 | RNG state non sauvé sur resume | `finetune_stage1_bundle_b.py:592-614` | P1 |
| 1.5 | Val/train boundary undefined | `_cell43.py:22-25` | P0 |
| 1.6 | HeteroConv `aggr="sum"` partagé | `intelligible_encoder.py:92` | P0 |
| 1.7 | `project_dag_floor(prior=G_phys)` écrase PCMCI | `finetune_stage1_bundle_b.py:338` | P1 |
| 1.8 | KKT λ ratio violé | Plusieurs | P1 |
| 1.9 | RAPSD sur 1 sample | `recompute_phase6_metrics.py:328-329` | P1 |
| 1.10 | n_batches=16, no CI | `recompute_phase6_metrics.py:70` | P1 |

### 1.2 Math prof additions (16 nouveaux)

| # | Issue | File | Severity |
|---|---|---|---|
| I1 | `physical_prior_loss` divise par N(N-1)=30, pas N_phys=5 → λ_phys effectif /30 | `physics_prior.py:201` | P0 |
| I2 | GRU `H_prev` non utilisé, hidden input est H_hat_tensor (SCM output) | `causal_rcn.py:464-481` | Documentation P2 |
| I3 | `_sample_edm_ode` mix DDPM epsilon + EDM schedule → ODE incorrecte | `diffusion_decoder.py:1086-1114` | P2 legacy |
| I4 | `high_k_rapsd_loss` log-power vs log-amplitude inconsistency | `two_stage.py:345-349` | P1 |
| I5 | Clausius-Clapeyron rate appliquée en log1p sans correction Jacobian | `two_stage.py:239-256` | P1 |
| I6 | FACL formule OK pour C=1 mais conflate channels-frequencies si C>1 | `spectral_loss.py:105-109` | P3 |
| I7 | `sliced_wasserstein_1d` ignore `n_slices` et `generator` silencieusement | `wasserstein_reg.py:38-102` | P2 |
| I8 | F1 threshold global (pooled), pas per-sample | `evaluation_xai.py:531-533` | P0 |
| I9 | 3 conventions FFT différentes (fft2/rfft2, norm/none, shift/no-shift) | Multi-file | P0 |
| I10 | `project_dag_spectral` Gershgorin overconservative vs vrai spectral radius | `causal_rcn.py:594-597` | P1 |
| I11 | CASTLE loss gradient leak through H_t (pas detach) | `causal_rcn.py:755-761` | P0 |
| I12 | CFG dropout adjusts delta_target mais ignore baseline_log → CFG non true-uncond | `two_stage.py:1229-1238` | P1 |
| I13 | `compute_crps` pairwise estimateur biased (M-1)/M | `evaluation_xai.py:425-428` | P2 |
| I14 | `compute_spectrum_distance` (rfft2, DC-removed) vs `compute_rapsd_numpy` (fft2, no DC) | `evaluation_xai.py:684-690` | P0 |
| I15 | `restart_sample` Xu 2023 variance lift incorrect quand tail_end < num_steps | `edm_sampler.py:205-208` | P1 |
| I16 | PSD `cgan_aligned_metrics` lat/lon meshgrid convention | `cgan_aligned_metrics.py:128-133` | P3 |

### 1.3 AI engineer additions (42 nouveaux)

**Architecture (J1-J11)** :
- J1 : `RegressionMeanPredictor._state_adapter` créé lazy → jamais dans optimizer → jamais entraîné (P0)
- J2 : `GraphToGridDecoder.grid_queries` silent dim mismatch sur YAML override (P1)
- J3 : `IntelligibleVariableEncoder.layer_norm` shared across all metapaths (P1)
- J4 : Encoder outputs collapse to per-target-type tensors → q "variables" réduit à 3 tensors uniques (P0)
- J5 : Batch=1 pooling broadcast breaks micro-batching > 1 (P0)
- J6 : DAG token broadcast bypasses conditioning_dropout gradient (P1)
- J7 : `extract_target_stats` Python loop force CUDA syncs (P2)
- J8 : `ConditionalSkipBlock` fed 2-D node tensor → silencieusement bypassed à chaque batch (P0)
- J9 : Tail-weighted MSE overflow en bf16 → hard raise (P1)
- J10 : `unet_in_channels = in_ch + 2` assume mu_HR/baseline 1-channel (P2)
- J11 : `cross_attention_dim` mismatch entre `conditioning_dim` et YAML override (P2)

**Training loop (J12-J22)** :
- J12 : `clip_grad_norm_` enumère seulement 3 modules → CASTLE/ident/projector unclipped (P1)
- J13 : `.item()` sur bf16 tensor → 3 digits précision dans les logs (P1)
- J14 : DDP `no_sync` skipped quand `torch.compile` wraps DDP (P2)
- J15 : NaN-skip branch ne projette pas A_dag (P2)
- J16 : Optimizer load mis-assign Adam moments sur structural param_group diff (P1)
- J17 : ZÉRO LR scheduler pendant 25-epoch fine-tune (P1)
- J18 : `strict=False` sur resume silencieusement avale 6→4 mismatch (P0)
- J19 : Pas de `cuda.synchronize()` avant save → race (P2)
- J20 : Bundle-B fine-tune n'utilise pas `_grad_comp` → λ values pas comparables au main loop (P1)
- J21 : EMA `warmup_steps=0` default pollue shadow avec init noise (P2)
- J22 : `dag_grad_gate` persistent buffer → restauré on resume, fight cold-start scripts (P1)

**Multi-GPU/DDP (J23-J27)** :
- J23 : `find_unused_parameters=True` default → ~30% DDP overhead (P2)
- J24 : `MASTER_PORT=12355` hard default → collision (P2)
- J25 : `ShardedIterableDataset` shards modulo-rank sans shuffle entre epochs (P1)
- J26 : `wrap_models_for_notebook` utilise DataParallel deprecated (P2)
- J27 : `cleanup_ddp` sans barrier avant destroy (P2)

**Configs (J28-J35)** :
- J28 : `steps: 1000` ignoré sous `edm_karras` (P3 orphan)
- **J29** : `cfg_scale` ignoré dans `_sample_edm_karras` — **INVALIDE V4 Pearson 0.815** (P0)
- J30 : `cfg=1.0` + dropout=0.13 → 13% wasted dropout-branch training (P2)
- J31 : `huber+cosine` silently falls through to MSE (P1)
- J32 : `norm_num_groups` override chain dépend de merge depth (P3)
- J33 : `cfg_scale=1.0` traité comme CFG actif → wasted forward (P2)
- J34 : EMA config sans base → override-only (P3)
- J35 : V3 vs V5 même decay=0.9999 mais 2.5× different total steps (P2)

**Notebooks (J36-J42)** :
- J36 : SSD copy re-stat à chaque reload (P3)
- J37 : `_safe_load` swallows missing-key → random-init eval (P1)
- J38 : `compute_validation_loss` peut tourner DDPM 1000 steps par sample (P1)
- J39 : deepcopy sur val-improve eat Colab RAM (P1)
- J40 : Hardcoded `N_VAL_SAMPLES=24` ≠ cell 5 `n_batches` (P1)
- J41 : Pas de git SHA logging on bootstrap (P2)
- J42 : Hardcoded `causal_concat=True` (P2)

### 1.4 Data scientist additions (37 nouveaux)

**Méthodologie & comparaison (K1-K3)** :
- **K1** : Oracle ET CorrDiff JSONs montrent `causal_concat: true` — comparaison **apples-to-mangoes** (P0 ALERTE)
- K2 : F1 threshold global (pooled all 16 batches), pas climatologie per-pixel (P0)
- K3 : `pred_full = pred_mean + mu_HR` mais comparé à `_targets = residual[-1]` seul → double-count (P0)

**Stats & validité (K4, K9-K16)** :
- K4 : RAPSD 1 sample (confirmé) — CV ~40% (P0)
- K9 : Random (non-temporal) train/val split — autocorrélation lag-1 ≈ 0.3-0.5 (P0)
- K10 : Noncausal éval sur causal_concat path → mismatch avec son training (P1)
- K11 : Phase 7 K=12 vs Phase 6 K=64 → CRPS pas comparable (P1)
- K12 : Phase 7 n_steps=18 vs Phase 6 n_steps=32 → quality difference confounded (P1)
- K13 : CRPS clim inclut le jour évalué → +0.27% bias (P2)
- K14 : Rank histogram chi²=5.4M (catastrophic) — non flaggué dans la thèse (P1)
- K15 : CRPS clim = test truth → skill score circular (P1)
- **K16** : ZÉRO seed dans eval → results non reproductibles run-to-run (P0)

**Données & leakage (K5-K9, K17-K20)** :
- K5 : Normalization stats sur full dataset (train+val+test) (P0 data leak)
- K6 : `hr_smoothing` baseline sur full period (P1)
- K7 : NaN fill spatial mean sur full dataset (P2)
- **K8** : ZÉRO temporal split boundary dans config (P0)
- K9 : Random split sur time series (autocorrélation) (P0)
- K17 : `val_dataset[i]` fail sur IterableDataset (P1)
- K18 : Static tensor shared sans detach documenté (P2)
- K19 : Only last time step utilisé pour target (P2)
- K20 : Graph initialisé avec t=0 features seulement (P1)

**Baseline V5-mini invalidé (K21-K23, K30-K33)** :
- K21 : CDD bias +17.95 jours (140% over-prediction) non-rapporté (P1)
- K22 : PSD distance V5: 0.001 vs Noncausal: 0.447 (450×) inexpliqué (P1)
- K23 : Phase 7 protocole différent de Phase 6 (P1)
- **K30** : `epoch: 200, epochs_total: 10` → modèle sous-entraîné (P0)
- K31 : Spread/RMSE=0.41 non flaggé comme calibration failure (P1)
- K32 : mu_HR ablation delta/signal > 100% → unphysical (P1)
- K33 : `baseline_factor=4` ≠ vrai LR/HR ratio 7.5 (P1)

**Reproductibilité (K24-K29, K34-K37)** :
- K24 : Val split bypassed pour IterableDataset → no early stopping (P1)
- K25 : Validation loss avec DDPM, final eval avec EDM (P1)
- K26 : `hflip_prob` reverse signed wind variables (P2)
- **K27** : ZÉRO version pinning packages (P0)
- K28 : Lazy NetCDF stream sur NFS = non-deterministic order (P2)
- K29 : `per_sample_n: 16` → CI ±0.032 sur Pearson, diff Oracle vs CorrDiff = 0.006 inside (P0)
- K34 : `nan_to_num` sur mu_HR silencieusement discard NaN failures (P1)
- K35 : ZÉRO GPU AMP smoke test (P0)
- K36 : ZÉRO ensemble diversity test (P1)
- K37 : log1p back-transformation undocumented (P1)

### 1.5 Synthèse sévérité

| Severity | Count | Action requise |
|---|---|---|
| **P0** (showstopper) | 22 | MUST fix avant tout retrain |
| **P1** (degrade results) | 47 | Fix avant Phase A1 |
| **P2** (perf/UX) | 21 | Fix avant publication |
| **P3** (cosmetic/docs) | 5 | Optional |
| **TOTAL** | **95** | |

---

## 2. Décision préalable obligatoire MISE À JOUR

### 2.1 Arbre de décision révisé

```
                  ┌─────────────────────────┐
                  │ Phase 0 : pre-flight    │
                  │ FIX TOUS LES 22 P0      │
                  │ + temporal split        │
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ Phase A0' : RE-EVAL     │
                  │ baseline V5-mini avec   │
                  │ protocole corrigé       │
                  │ (CFG fix, causal_concat │
                  │  aligned, n=64)         │
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ Phase A0'' : RE-TRAIN   │
                  │ V5-mini 200 epochs      │
                  │ (épargne K30 sous-      │
                  │  entraînement)          │
                  │ × 3 seeds               │
                  └────────────┬────────────┘
                               │
                  ┌────────────▼────────────┐
                  │ Phase A1 : minimal fix  │
                  │ 6-node avec :           │
                  │ - dag_grad_gate wired   │
                  │ - HeteroConv fix        │
                  │ - LayerNorm per-var     │
                  │ - struct_W properly     │
                  │   initialized           │
                  │ - state_adapter eager   │
                  │ - skip_block fixed      │
                  │ × 3 seeds               │
                  └────────────┬────────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
       Q_phys ≥ 0.65 ?                Q_phys < 0.65 ?
                │                             │
                ▼                             ▼
       ┌─────────────────┐         ┌─────────────────────┐
       │ PUBLISH         │         │ Phase C+ complet    │
       │ (minimal fix +  │         │ refactor 4-node     │
       │ corrected base) │         │ avec TOUS les fixes │
       └─────────────────┘         └─────────────────────┘
```

---

## 3. Hyperplan révisé — 5 macro-phases

### MACRO-PHASE I : Pre-flight + fix 22 P0 (2 semaines)

#### Phase 0.1 — Setup (1 jour)

```bash
git checkout -b four-node-causal
# documenter dans docs/
docs/hyperplan_path_c_plus_consensus.md  # CE document
docs/pre_registration_pathC_plus.md       # NOUVEAU (DS K-prereg)
docs/p0_fix_checklist.md                  # NOUVEAU (22 items)
docs/test_coverage_gaps.md                # NOUVEAU (17 gaps DS)
docs/ai_eng_full_audit.md                 # généré par audit
```

#### Phase 0.2 — Fix les 22 issues P0 (5-7 jours)

| Priorité | Fix | Files | Effort |
|---|---|---|---|
| 1 | **K1** : vérifier `causal_concat` cohérent avec model | recompute_phase6 + assert | 1h |
| 2 | **K8** : définir `train_period`, `val_period`, `test_period` | training_config.yaml | 1h |
| 3 | **K5** : normalisation stats sur train period only | pipeline.py:647-649 | 2h |
| 4 | **K9** : temporal split (pas random) | _cell43.py:38-40 | 1 jour |
| 5 | **K16** : seed param dans recompute_phase6 | recompute_phase6_metrics.py | 30 min |
| 6 | **K27** : pin versions diffusers, torch, numpy | requirements.txt | 30 min |
| 7 | **K30** : assert `epochs_total == epochs_max` ou flag undertrain | training_loop.py | 30 min |
| 8 | **K35** : add GPU AMP smoke test | tests/test_gpu_amp_smoke.py | 1 jour |
| 9 | **K3** : clarifier pred_full vs target space | recompute_phase6_metrics.py:239 | 2h |
| 10 | **K2** : F1 threshold per-pixel climatology | evaluation_xai.py:531-533 | 4h |
| 11 | **J1** : `_state_adapter` eager init | regression_mean_predictor.py:186-190 | 1h |
| 12 | **J4** : encoder per-target Linear, pas shared | intelligible_encoder.py:118-130 | 4h |
| 13 | **J5** : assert B==1 ou fix _assign_default_batch | intelligible_encoder.py | 2h |
| 14 | **J8** : LR grid conversion before skip_block | finetune_stage1_bundle_b.py:227-229 | 2h |
| 15 | **J18** : `strict=True` resume default | finetune_stage1_bundle_b.py:546-555 | 30 min |
| 16 | **J29** : implement CFG dans `_sample_edm_karras` OU assert | edm_sampler.py + diffusion_decoder.py | 1 jour |
| 17 | **§1.1** : wire `dag_grad_gate` | finetune_stage1_bundle_b.py:151 | 30 min |
| 18 | **§1.2** : DAGMA s = Gershgorin + margin | finetune_stage1_bundle_b.py:258 | 30 min |
| 19 | **§1.3** : appeler `calibrate_sigma_data_variant` | finetune_stage1_bundle_b.py:671-725 | 1h |
| 20 | **§1.5** : audit val_dataset definition | _cell43.py, notebook | 2h |
| 21 | **§1.6** : HeteroConv `aggr="cat"` + per-target Linear | intelligible_encoder.py:92 | 4h |
| 22 | **I1** : `physical_prior_loss` sans division | physics_prior.py:201 | 30 min |

**Total Phase 0.2** : ~7 jours dev + tests unitaires pour chaque fix.

#### Phase 0.3 — Unit tests + faithfulness pre-flight (1 jour)

```python
# tests/test_p0_fixes.py
def test_K1_causal_concat_consistency():
    """Vérifie que causal_concat in JSON == model causal_concat attribute."""
    ...

def test_K5_normalization_train_only():
    """Vérifie que stats normalisation utilisent seulement train period."""
    ...

def test_K9_temporal_split_no_overlap():
    """Vérifie zero overlap train/val temporal."""
    ...

def test_K16_seed_reproducibility():
    """Vérifie même seed → mêmes métriques."""
    ...

def test_J1_state_adapter_in_optimizer():
    """Vérifie _state_adapter dans optimizer.param_groups."""
    ...

def test_J4_encoder_no_shared_params():
    """Vérifie 4 nodes uniques, pas collapse."""
    ...

def test_J29_cfg_assertion():
    """Vérifie raise si edm_karras + cfg > 1."""
    ...

def test_encoder_faithfulness_corr_lt_0_5():
    """Pre-flight encoder representation CI test."""
    ...
```

#### Phase 0.4 — Decision gate I → A0'

**Go criteria** :
- ✅ Tous les 22 P0 fixes commit + unit tests verts
- ✅ Encoder faithfulness corr < 0.5
- ✅ Temporal split documenté dans YAML
- ✅ Version pinning requirements.txt commit

---

### MACRO-PHASE II : RE-EVALUATION + RE-TRAIN baseline V5-mini (2 semaines)

**Justification** : K1 + K30 + J29 ensemble invalident la baseline V5-mini actuelle. **Sans baseline propre, aucune comparaison n'est défendable.**

#### Phase A0' — RE-EVAL baseline V5-mini avec protocole corrigé (1 jour, 10h Colab)

Reload checkpoint actuel mais avec :
- CFG correctement géré (J29 fix : implémenté ou assert)
- causal_concat consistant entre Oracle et CorrDiff (K1)
- n_batches=64 (pas 16) (K4, 1.10)
- Seed=42 fixé (K16)
- F1 threshold per-pixel climatology (K2)
- RAPSD sur tous les samples (K4, 1.9)
- Bootstrap CI 95% (K29)

**Output** : `results/v5_baseline_corrected_eval/` avec nouveaux chiffres.

**Critère décision** : si les chiffres corrigés montrent que V5-mini est **statistiquement non distingable de CorrDiff** (CI overlap), la motivation pour Path C+ devient encore plus forte (le baseline n'a pas validement battu CorrDiff).

#### Phase A0'' — RE-TRAIN V5-mini complet (3 jours setup + 20-25h Colab × 3 seeds = 60-75h)

Le modèle a tourné seulement 10/200 epochs (K30). Pour avoir une vraie baseline scientifique :
- Re-train V5-mini 200 epochs sur ACCESS-CM2 1980-2011 (avec temporal split K9)
- 3 seeds = [42, 7, 123]
- Avec tous les P0 fixes appliqués (sauf 4-node refactor)
- Sigma_data unbiased, CFG correctement géré, etc.

**Compute** : 20-25h Colab × 3 seeds = 60-75h total.

**Critère décision** : avec V5-mini correctement entraîné et évalué, on a un baseline scientifiquement défendable.

---

### MACRO-PHASE III : Phase A1 — 6-node minimal fix (1 semaine)

Sur la nouvelle baseline V5-mini corrigée, tester l'alternative cheap consensus.

#### Phase A1 — 3 seeds × 25 epochs FT (3 jours, 30-40h Colab)

**Changements code (au-delà des P0 Phase 0.2)** :
- ✅ `dag_grad_gate` ramped 0→1 epoch 5-20 (§1.1 fix)
- ✅ HeteroConv `aggr="cat"` + per-target Linear (§1.6 fix)
- ✅ Per-metapath LayerNorm (J3 fix)
- ✅ `_state_adapter` eager init (J1 fix)
- ✅ `skip_block` proper LR grid input (J8 fix)
- ✅ Sigma_data unbiased (§1.3 fix)
- ✅ DAGMA `s` dynamic (§1.2 fix)
- ✅ `project_dag_floor` désactivé après epoch 50 (§1.7 fix)
- ✅ λ_l1 constant à 0.040 (§1.8 fix)
- ✅ `physical_prior_loss` sans division (I1 fix)
- ✅ CASTLE `H_t.detach()` (I11 fix)
- ✅ `clip_grad_norm_` sur tous les param_groups (J12 fix)
- ✅ LR cosine schedule + warmup (J17 fix)

**Instrumentation** (10 items obligatoires) :
1. `g_A_dag.norm() / g_driver_enc.norm()` per step
2. `||A_dag||_F`, `nnz`, asymmetry per epoch
3. KKT residual per edge type per step
4. Sigma-binned EDM loss (5 bins)
5. NaN counter per loss component
6. Encoder representation CI corr per epoch
7. `dag_grad_gate` actual value per epoch
8. Q_phys with bootstrap CI per epoch
9. `edge_gate` parameter value
10. `effective_amp_dtype` per epoch (J13 fix)

**Décision A1** :

| Outcome | Q_phys (mean 3 seeds) | RMSE change vs corrected baseline | Décision |
|---|---|---|---|
| **SUCCESS** | ≥ 0.65 ± 0.10 | ≤ +5% | STOP. Publish A1 + corrected baseline |
| **PARTIAL** | 0.50 - 0.65 | ≤ +5% | Continue vers C+ B1 (PCMCI seul) |
| **FAIL** | < 0.50 | ≤ +5% | C+ full justifié |
| **DIVERGE** | n/a | > +5% ou NaN | Investigation 2j, possibly back to A0'' |

---

### MACRO-PHASE IV : Phase C+ refactor (si A1 fail) (4 semaines)

#### Phase B0 — PCMCI préliminaires avec corrections (3 jours, 4-8h CPU)

Modifications vs plan original (math prof + AI eng) :
1. **Detrending stationnarité** (math prof R1) : per-pixel linear detrend avant spatial mean
2. **CI test approprié** : CMIknn pour SP_HR (non-Gaussian)
3. **Bootstrap stability** (math prof V2 + DS R10) : 10 bootstrap resamples
4. **Spatial reduction sound** : column-wise PCMCI puis majority vote (AI eng 2.3)
5. **Acyclicity guard** (math prof C4) : `np.linalg.eigvals` check
6. **PAG → DAG projection** si bidirected (AI eng 2.3)

#### Phase B1 — Architectural refactor 6→4 (3-4 jours dev)

Avec tous les fixes (Phase 0.2 + A1 + PCMCI init), refactor :
- `physics_prior.py` : 4 nœuds, EXPECTED_EDGES = 3
- `intelligible_encoder.py` : node_states + edge_features dict (J4 fix)
- `causal_rcn.py` : A_dag 4×4, struct_W 4 (cold-start accepted, J20 fix), edge_gate learnable (AI eng 2.2)
- `stage1_paths.py` : `predict_mu_hr` consomme le dict
- **Backward compat flag** : `CONFIG.architecture.num_dag_nodes` pour rollback

#### Phase B2-B3 — Loss + smoke test (1 jour + 3h Colab)

**Hyperparamètres corrigés** (math prof + AI eng + DS) :
```python
DEFAULT_HYPERPARAMS_4NODE = {
    "lr_main": 1e-4,
    "lr_castle": 1e-4,
    "lr_a_dag": 5e-5,
    
    # L1 CONSTANT au-dessus borne KKT (4-node: λ_phys/6 = 0.067)
    "lambda_l1_const": 0.040,
    "lambda_l1_late": 0.035,
    
    # Physical prior strong (avec /off_diag SUPPRIMÉ via I1 fix)
    "lambda_dag_prior": 0.40,
    "g_phys_alpha": 0.25,
    
    # DAG gate ramp 0→1 epoch 5-20
    "dag_gate_warmup_start_epoch": 5,
    "dag_gate_warmup_end_epoch": 20,
    
    "lambda_castle_epoch_lt_10": 0.05,
    "lambda_castle_epoch_ge_10": 0.10,
    "castle_expansion": 2,
    
    # DAGMA s dynamic
    "gamma_dag_max": 0.10,
    
    # LR schedule (J17 fix)
    "lr_warmup_epochs": 5,
    "lr_decay_type": "cosine",
}
```

#### Phase B4 — Stage 1 full retrain 4-node (1 jour setup + 30-48h Colab × 3 seeds)

Critères validation :
- Q_phys mean ± std 3 seeds
- RMSE val ≤ corrected baseline + 5%
- A_dag asymmetry > 0.20
- spectral_radius < 0.95

#### Phase B5 — Stage 2 EDM retrain (1 jour + 8-24h Colab)

**Stratégie** (AI eng 3.2) : fine-tune V5-mini Stage 2, pas from-scratch.

**Recalibrer P_mean ET P_std** (Karras 2024, math prof R4) :
```python
edm_config.P_mean = math.log(sigma_data_new) - 0.4
edm_config.P_std = 1.2
```

EMA buffer reset (AI eng 3.3).

#### Phase B6 — Evaluation refactor + run (2 jours + 12h Colab)

**Nouvelles métriques** :
- Q_phys avec bootstrap CI 95% (DS §6 + math prof V5)
- Skeleton F1 + MEC F1 (math prof T5)
- Paired Wilcoxon V5-mini-corrected vs V6-4node vs CorrDiff-corrected
- Multiple testing correction (BH FDR)
- Cohen's d effect sizes
- **Temporal holdout OOD** : ACCESS-CM2 2014 (K23 fix)

---

### MACRO-PHASE V : Documentation + statistical analysis (3 jours)

Pre-registration completed (DS §5). Bootstrap CIs sur toutes métriques. Paired tests. Effect sizes.

**Pre-registered hypothesis** (DS §5) :
- H1 (primary) : Q_phys ≥ 0.65
- H2 : Pearson within 2pp de corrected baseline
- H3 : CDD bias < 10 jours
- H4 : Spread/RMSE ≥ 0.60
- H5 : OOD CRPS skill within 5pp de in-dist

---

## 4. Timeline révisée FINALE

| Sem | Phase | Activité | Compute |
|---|---|---|---|
| **1** | I.0 | Setup + fix 22 P0 issues + unit tests | 0 |
| **2** | I.0 | Suite fixes + faithfulness pre-flight + temporal split | 0 |
| **3** | II.A0' | RE-EVAL baseline V5-mini avec corrections | 10h Colab |
| **3-5** | II.A0'' | RE-TRAIN V5-mini 200 epochs × 3 seeds | **60-75h Colab** |
| **6** | III.A1 | 6-node minimal fix × 3 seeds + analyse | 30-40h Colab |
| **7** | DECISION | A1 SUCCESS ? → STOP & publish OR → C+ continue | 0 |
| **8-9** | IV.B0 | PCMCI préliminaires avec corrections | 4-8h CPU |
| **9-10** | IV.B1 | Refactor architectural 6→4 + tests | 0 |
| **10** | IV.B2-3 | Loss reformulation + smoke test | 3h Colab |
| **10-12** | IV.B4 | Stage 1 retrain 4-node × 3 seeds | 30-48h Colab |
| **12-13** | IV.B5 | Stage 2 retrain (fine-tune ou scratch) | 8-24h Colab |
| **13-14** | IV.B6 | Eval refactor + full run × 3 seeds | 12h Colab |
| **14-15** | V.Z | Stat analysis + documentation | 0 |

### Calendrier réaliste FINAL

- **Si A1 SUCCESS** : **6-7 semaines**, ~110h Colab. Publication avec corrected baseline.
- **Si Path C+ complet** : **14-15 semaines** (3.5 mois), ~180-220h Colab. Publication full.

**Plan original** : 3 semaines, 18-25h. **Sous-estimé d'un facteur 5-7×** quand on tient compte des corrections nécessaires.

---

## 5. Pre-registration des hypothèses (DS §5)

**Document à signer + commit à `docs/pre_registration_pathC_plus.md` avant Phase A0'.**

### H1 — Primary causal efficacy
**Métrique** : Q_phys
**Seuil** : ≥ 0.65 mean over 3 seeds, bootstrap CI(95%, 1000 resamples sur 64 samples)
**Rejection** : Q_phys < 0.55 → Path C+ non validé

### H2 — Predictive skill non-degradation
**Métrique** : Pearson per-sample mean
**Seuil** : ≥ 0.810 (corrected baseline CI lower bound)
**Test** : Paired bootstrap, α=0.05

### H3 — CDD bias reduction
**Métrique** : `|cdd_pred - cdd_truth|` sur 365 jours
**Seuil** : < 10 jours (50% reduction vs current +17.95)

### H4 — Ensemble calibration improvement
**Métrique** : Spread/RMSE
**Seuil** : ≥ 0.60 (vs current 0.41)

### H5 — OOD generalization
**Métrique** : CRPS skill score par GCM, LOO climatology
**Seuil** : |CRPS_OOD - CRPS_in_dist| < 0.05

### Règles pre-registration
- 5 hypothèses pre-registered. H1 = primary.
- Checkpoint selection : LAST epoch (pas best), OU epoch 25 fixé
- 3 seeds, mean ± SD, pas de cherry-picking
- Negative results MUST be reported
- FDR : Holm-Bonferroni sur 5 hypothèses
- All claims include bootstrap CI 95%
- Cohen's d reported pour chaque comparaison

---

## 6. Test coverage gaps (DS §4)

17 gaps de tests à combler avant publication :

1. Temporal train/val split — pas de test
2. Normalization stats train-only — pas de test
3. F1 per-pixel vs global threshold — pas de test
4. `causal_concat` flag consistency — pas de test
5. `pred_full` formula correctness — pas de test
6. RAPSD multi-sample — pas de test
7. Ensemble diversity (spread > 0) — pas de test
8. Full pipeline integration (real data) — pas de test
9. `baseline_factor` vs LR/HR ratio — pas de test
10. Seed determinism across restarts — pas de test
11. IterableDataset non-indexability — pas de test
12. GPU AMP smoke test — pas de test
13. CRPS correctness — pas de test
14. cftime calendar handling — pas de test
15. log1p back-transformation — pas de test
16. PCMCI init → project_dag_floor interaction — pas de test
17. n_batches cap effect on variance — pas de test

**Add to Phase 0.3** : créer tests pour les 17 gaps avant tout retrain.

---

## 7. Statistical validity assessment des claims actuels (DS §3)

| Claim actuel | Verdict | Raison |
|---|---|---|
| Oracle Pearson 0.825 vs CorrDiff 0.819 | **NOT SIGNIFICANT** | n=16, CI ±0.032, diff=0.006 |
| Oracle RMSE 0.124 vs CorrDiff 0.130 | **Marginal** | 4%, no CI |
| F1-p99 0.512 vs 0.550 (CorrDiff wins!) | **INVALID** | Global threshold + n=16 |
| RAPSD 241.8 vs 270.1 | **NOT VALID** | 1 sample, CV ~40% |
| PSD V5: 0.001 vs Noncausal: 0.447 | **Different metric** | K22 |
| CRPS skill V5: 0.895 vs 0.826 | **Defensible mais biased** | K13 |
| mu_HR ablation MU_HR_CONDITIONS | **NOT INTERPRETABLE** | K32 |
| CDD bias not flagged | **Unreported failure** | K21 |

**Conclusion** : aucun des claims actuels n'est statistiquement défendable avec les standards de publication NeurIPS/ICML. Tous ont besoin de re-évaluation avec n=64, bootstrap CI, paired tests, FDR correction.

---

## 8. Risques résiduels actualisés

| Risque | Probabilité | Impact | Mitigation |
|---|---|---|---|
| Corrected baseline V5-mini montre que V5 ≈ CorrDiff | 0.40 | Motivation Path C+ moins claire | Plus de motivation pour publier le négatif |
| RE-train V5-mini 60-75h dépasse budget Colab | 0.50 | Compute additional | Split sur plusieurs sessions, budget 30% buffer |
| Quelqu'un trouve 96ème issue critique | 0.30 | Re-plan needed | Allouer buffer 1 semaine |
| Path A1 SUCCESS mais réviseur conteste statistical methodology | 0.40 | Revisions needed | Pre-registration + multi-seed + bootstrap CI mitigent |
| PCMCI bootstrap < 0.5 | 0.30 | Path C+ B0 fail | Fallback G_phys init seul |
| Stage 2 EDM diverge post P_mean recalibration | 0.30 | +24h Colab | Sigma-binned loss instrumentation detect early |
| Compute total > 220h | 0.50 | Calendar slip | Accept, budget 30% buffer |

---

## 9. Conclusion

Le baseline V5-mini actuel n'est **pas scientifiquement défendable** pour servir de référence comparative — il combine :
- Comparison fraud (K1: causal_concat=true des deux côtés)
- Modèle sous-entraîné (K30: 10/200 epochs)
- CFG silencieusement ignoré (J29)
- Sigma_data biased (1.3)
- Random temporal split (K9)
- Normalization data leak (K5)
- Pas de seeds reproductibles (K16)
- n=16 inside noise floor (1.10, K29)
- RAPSD 1 sample (1.9, K4)

**Le plan révisé** :
1. Corrige les 22 P0 (semaines 1-2)
2. RE-évalue ET re-entraîne V5-mini correctement (semaines 2-5)
3. Teste Phase A1 minimal fix (semaine 6)
4. Décide Path C+ ou publication selon A1

Total : 6-7 semaines minimum si A1 réussit, 14-15 semaines si Path C+ full. Compute : 110-220h Colab T4.

**Investissement justifié si** : tu veux un résultat scientifiquement défendable face à un reviewer NeurIPS/JGR rigoureux. **Non justifié si** : objectif court terme avec acceptation des limitations actuelles (et dans ce cas, le mémoire/thèse doit explicitement reconnaître les limitations 1-22 P0).

---

## Sources audit (3 experts)

- **Math Professor** : 16 nouveaux issues I1-I16, focus formules + KKT + projections graphes
- **AI Engineer** : 42 nouveaux issues J1-J42, focus architecture + training + DDP + configs + notebooks (peer-reviewed: Karras 2022/2024, Mardani 2023, Bello 2022)
- **Data Scientist** : 37 nouveaux issues K1-K37, focus data integrity + splits + leakage + reproducibility + statistical validity (peer-reviewed: NeurIPS Reproducibility Checklist 2024, Leutbecher & Palmer 2008, Ferro 2014, ETCCDI WMO)

**Document version** : 2.0 FINAL — 2026-06-12
**Statut** : Consensus 3 experts post-100%-codebase-audit
**Next action** : Phase 0.1 setup (1 jour), commit avec ce hyperplan + pre-registration draft.
