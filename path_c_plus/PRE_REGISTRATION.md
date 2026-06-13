# Pre-Registration Plan — Path C+ Retrain

**Statut** : Draft, à signer avant tout training Phase A1+
**Auteur** : Leonel KENFACK (M2)
**Date** : 2026-06-12
**Document à committer avant Phase A0''**

## Engagement

Je m'engage à respecter les hypothèses pré-enregistrées ci-dessous **avant** d'exécuter
les expériences Path C+. Aucune modification des seuils de décision ne sera faite
post-hoc en fonction des résultats observés. Les résultats négatifs seront rapportés
avec la même rigueur que les résultats positifs.

## H1 (PRIMARY) — Causal efficacy

**Statement** : Q_phys (causal intervention score) pour Path C+ 4-node strictement supérieur à V5-mini baseline corrigé.

**Métrique** :
```
Q_phys = mean(|mu_HR(A_real) - mu_HR(A_zeroed)|) / mean(|mu_HR(A_real)|)
```
calculé sur N=64 validation samples, avec bootstrap CI 95% (1000 resamples).

**Threshold** :
- **SUCCESS** : Q_phys mean ≥ 0.65 (3 seeds), CI ne overlap pas avec baseline corrigé CI
- **PARTIAL** : 0.50 ≤ Q_phys < 0.65
- **FAILURE** : Q_phys < 0.50

**Action si rejection** :
- PARTIAL : continue vers Path C+ B1 (PCMCI seul, sans 4-node refactor)
- FAILURE : abandon Path C+, retour Path A1 minimal fix

## H2 (SECONDARY) — Predictive skill non-degradation

**Statement** : Path C+ Pearson per-sample mean ≥ baseline corrigé Pearson - 0.02 (within 2pp).

**Métrique** : Pearson per-sample sur 64 validation samples, K=64 ensemble mean.

**Threshold** : Pearson mean ≥ baseline_corrected_Pearson_CI_lower_bound - 0.02

**Test statistique** : Paired bootstrap test (Path C+ vs baseline corrigé), α=0.05.

**Action si rejection** : flag as trade-off, document in thesis.

## H3 (SECONDARY) — CDD bias reduction

**Statement** : Path C+ CDD bias < 10 jours (vs current +17.95).

**Métrique** :
```
CDD_bias = |CDD_pred - CDD_truth| averaged over 365 days
```
Computed in physical mm/day space (after log1p back-transformation, K37 fix verified).

**Threshold** : CDD_bias < 10 jours (50% reduction baseline +17.95).

## H4 (SECONDARY) — Ensemble calibration improvement

**Statement** : Path C+ Spread/RMSE ratio > 0.60 (vs current 0.41).

**Métrique** :
```
Spread/RMSE = spread_global_mm / rmse_global_mm
```
Over N≥365 days, K≥32 members. With rank histogram chi² < 1M.

**Threshold** : Spread/RMSE ≥ 0.60.

## H5 (SECONDARY) — OOD generalization

**Statement** : Path C+ CRPS skill score sur EC-Earth3 et NorESM2-MM within 5pp de in-distribution ACCESS-CM2.

**Métrique** : CRPS skill score per GCM, avec **leave-one-out climatology** (K13 fix).

**Threshold** : |CRPS_skill(OOD) - CRPS_skill(in-dist)| < 0.05.

## Clauses PC1-PC4 (DS Round-9, post smoke #2 FAIL diagnosis)

Ces clauses sont des **clarifications de protocole**, pas des changements de seuil.
Elles formalisent les conditions découvertes par la corruption du smoke ckpt.

### PC1 — Fresh checkpoint mandatory

L'initialisation A0'' utilise EXCLUSIVEMENT `V5_DIR/epoch_last.pth`.
Aucun reuse de SMOKE_DIR/ ou de checkpoint dérivé d'un run smoke.
Toute violation invalide le seed avant unblinding.

Détails : voir `path_c_plus/FRESH_CHECKPOINT_PROTOCOL.txt`.

### PC2 — Partial success reporting category

Si le bootstrap CI 95% (BCa, 3 seeds, 1000 resamples) pour Q_phys tombe dans
**[0.50, 0.65)**, le résultat est rapporté comme :

> "PARTIAL — insufficient to claim H1, sufficient to motivate
> architectural refinement"

Cette catégorie était implicite, maintenant explicite. Ne déplace PAS le seuil H1.

### PC3 — Gradient diagnostics as secondary outcome

`median(A_dag.grad.norm())` à travers les training steps est un secondary outcome
pré-enregistré.

Si Q_phys FAIL mais gradient norm est ~0 :
- Failure mode = "optimization failure" (signal ne flow pas vers A_dag)
- Remediation path = revoir gate ramping / encoder representation

Si Q_phys FAIL mais gradient norm > 1e-3 :
- Failure mode = "causal structure unlearnable from observational data"
- Remediation path = architectural revision OR accept negative result

Ces deux failure modes nécessitent des actions DIFFÉRENTES en post-hoc.

### PC4 — Audit gate (BLOCKING)

Avant qu'un run A0'' compte pour H1 evaluation, le `load_audit` dict capturé
au load V5-mini ckpt DOIT satisfaire :

```python
load_audit["modules_loaded_fallback"] == []  # ZERO module en strict=False
load_audit["encoder_state_dict_n_keys"] > 0   # ckpt encoder non-vide
load_audit["modules_loaded_strict"] >= {"encoder", "rcn_cell",
                                         "regression_head", "diffusion"}
```

Si `diffusion` est dans `modules_loaded_fallback` (comme en smoke #2),
le run est **AUTOMATIQUEMENT EXCLU** de l'analyse H1.

Ceci empêche la répétition du failure mode smoke #2 (Q_phys 0.40 → 0.40
sous diffusion partiellement random).

### PC5 — H1 metric is the CONTINUOUS magnitude ratio, not sign-binary

(Ajouté après le post-mortem du smoke #3 — council ARTEFACT verdict, 2026-06-13)

The H1 primary endpoint is the CONTINUOUS Q_phys defined as :

```python
Q_phys_cont = sum(|A_dag[i,j]| for (i,j) where sign(A_dag[i,j]) == sign(G_phys[i,j]) and G_phys[i,j] != 0)
              / sum(|A_dag[i,j]| for all off-diagonal (i,j))
```

Range [0, 1]. Gaming-resistant : un modèle avec 5/5 edges sign-correct à
magnitude 0.011 entouré de 7 edges spurious à magnitude 0.17 donne
`Q_phys_cont ≈ 0.04` (et NON 1.0 comme le métric binary).

Le métric `Q_phys_binary` (sign-count avec threshold) est conservé comme
**diagnostic secondaire** dans les JSONs, mais ne sert PAS à l'acceptation/
rejet de H1. Toute claim H1 dans le mémoire / paper utilise `Q_phys_cont`.

H1 acceptance threshold (locked here, before A0''): `Q_phys_cont ≥ 0.50`
sur la moyenne des 3 seeds, avec BCa 95% CI lower bound > baseline_cont
(baseline_cont = Q_phys_cont du V5-mini band-diagonal = 0.04).

### PC6 — Band-diagonal n_extra_edges as pre-registered secondary outcome

(Ajouté après le post-mortem du smoke #3 — DS recommendation)

Define :
```python
n_extra_edges := count of |A_dag[i,j]| > 0.05 where (i,j) is off-diagonal
                                                AND G_phys[i,j] == 0
```

Threshold 0.05 = 5× le smoke threshold 0.01 pour évacuer le bruit.

`n_extra_edges` est reporté pour chaque seed dans le JSON A0''. La claim H1 +
n_extra_edges interagissent ainsi :

| Q_phys_cont | n_extra_edges | Reporting |
|---|---|---|
| ≥ 0.50 | ≤ 2 | "H1 met : interventional + sparse structural recovery" |
| ≥ 0.50 | ≥ 3 | "H1 met under primary endpoint, structural sparsity NOT achieved (n_extra_edges = X). Causal-recovery claim restricted to interventional sign-consistency; structural claim withheld." |
| < 0.50 | any | "H1 failed; see PC3 to disambiguate optimization-vs-architecture failure" |

### PC7 — Smoke ≠ A0'' regime gap (acknowledged)

(Ajouté après le post-mortem du smoke #3 — council unanime)

Les smoke tests (#1, #2, #3, #4) tournent depuis le ckpt V5-mini avec
`diffusion` en strict=False (V5-mini précède J29). C'est une **violation
PC4 par construction** ; les smokes sont **EXEMPTÉS de PC4** car ils sont
diagnostiques, pas evidentiels.

En conséquence, **les résultats Q_phys des smokes ne sont PAS prédictifs
des résultats Q_phys de A0''**. Le smoke valide :
1. Que le code tourne (no exception)
2. Que `A_dag` bouge (norm delta > 0.05) — proxy pour "le signal d'apprentissage
   passe"
3. Que les diagnostics post-mortem sont calibrés (Q_phys_cont, projection log,
   per-epoch trajectory)

Le smoke ne valide PAS :
1. Que Q_phys_cont ≥ 0.50 sera atteint en A0'' (régime différent)
2. Que le band-diagonal pattern sera cassé en A0''
3. Que H1 sera accepté

Les JSONs des smokes portent `"schema_version": "path-c-plus-smoke-X-batch-Y"`
et `"valid_for_analysis": false` (cf. `stamp_batch_d_json`). Les JSONs A0''
porteront `"schema_version": "path-c-plus-batch-D-v1"` et
`"valid_for_analysis": true`.

### PC8 — Effect-size threshold for H1 (locked before A0'')

(Ajouté après le post-mortem du smoke #3 — DS Holm-Bonferroni pre-lock)

Path C+ est déclaré EFFECTIF sur H1 ssi **TOUS** les critères suivants sont
atteints conjointement :

1. **Mean** : `Q_phys_cont` moyen sur 3 seeds ≥ 0.50
2. **CI non-overlap** : BCa 95% CI lower bound de `Q_phys_cont` (3-seed)
   strictement supérieur au BCa 95% CI upper bound du baseline V5-mini
   (Q_phys_cont = 0.04)
3. **Effect size** : Cohen's d ≥ 0.8 entre le 3-seed Path C+ et le V5-mini
   single-point baseline (treating baseline_var = 0 → d = mean_diff / sd_pathcplus)

Le test sur Q_phys_cont est combiné avec H2-H5 sous Holm-Bonferroni (k=5)
avec alpha=0.05. Le ranking p-value sera fait avant l'unblinding des A0'' JSONs.

**Q_phys_binary** (la métrique du smoke #3) reste un diagnostic secondaire,
reporté avec son threshold (0.01 ou 0.3·max), mais ne contribue PAS au
test statistique H1.

## Tombstone des résultats historiques pré-K1

DS Round-2 Condition A : tous les résultats V5-mini produits avant le K1 fix
sont méthodologiquement compromis (Oracle ET CorrDiff avaient causal_concat=True).

Action commit-bloquante avant Batch D :
- Script `path_c_plus/scripts/_tombstone_legacy_jsons.py` applique
  `"schema_version": "legacy-pre-K1"` et `"valid_for_analysis": false`
  à TOUS les `results/v5_evaluation/*.json` produits par smoke #1 et smoke #2.
- Documenter ces JSONs comme exclus de l'analyse H1 dans le mémoire.

## Règles de reporting

### Multi-seed
- 3 seeds : [42, 7, 123]
- Toute claim statistique = mean ± SD across seeds + bootstrap CI 95%
- **PAS de cherry-picking** : pas de "best seed" reporting

### Checkpoint selection
- Pas d'early stopping pour métriques finales
- **LAST epoch** OR pre-specified `epoch=200` (Stage 1) / `epoch=50` (Stage 2 finetune)
- Le critère de sélection est fixé AVANT training

### Statistical reporting obligatoire
Pour chaque comparaison reportée :
- Mean ± SD across 3 seeds
- Bootstrap CI 95% (1000 resamples)
- Paired test (Wilcoxon signed-rank) avec p-value
- Cohen's d effect size
- **Multiple testing correction** : Holm-Bonferroni sur les 5 hypothèses

### Negative results
Si H1 ne passe pas :
- Reporter explicitement Q_phys = X.XX (CI 95% [a, b])
- Discuter pourquoi (architectural? data limitation? optimization?)
- Comparer avec literature (Cachay 2021 Q_phys, etc.)
- Ne pas re-spinner positivement

### Forbidden practices
- ❌ Modifier les seuils H1-H5 post-hoc
- ❌ Sélectionner le "best seed" si SD trop élevée
- ❌ Reporter sans CI ni effect size
- ❌ Comparer Phase 6 vs Phase 7 sans flag mismatched protocol (K23)
- ❌ Rapporter F1-p99 avec global threshold (K2 fix doit être appliqué)

## Validation pre-flight checklist

Avant de lancer Phase A0'' (premier training propre) :

- [ ] Tous les 22 P0 fixes commitis sur branch
- [ ] Tests unitaires verts
- [ ] Encoder representation CI corr < 0.5
- [ ] Temporal split documenté dans `training_config.yaml`
- [ ] Pre-registration commit avec ce document signé
- [ ] V5-mini baseline checkpoint sauvegardé en lecture seule
- [ ] Plan de compute Colab : sessions planifiées, budgets confirmés

## Signature

```
Date signature : _______________
Étudiant : Leonel KENFACK
Branche cible : four-node-causal
Commit hash @ signature : _______________
```

## Versions et amendements

| Version | Date | Changement | Approbation |
|---|---|---|---|
| 1.0 | 2026-06-12 | Draft initial | (pending) |

Toute modification post-signature doit être tracée ici avec justification scientifique.
