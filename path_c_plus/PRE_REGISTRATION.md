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
