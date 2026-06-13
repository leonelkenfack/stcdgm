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

**Note d'endpoint-drift** (post-mortem smoke #3, DS audit) :
l'endpoint H1 originellement défini dans `HYPERPLAN.md §H1` était
l'intervention ratio :
```
Q_phys_interv = mean(|mu_HR(A_real) - mu_HR(A_zeroed)|) / mean(|mu_HR(A_real)|)
```
PC5 redéfinit l'endpoint en magnitude ratio structural (sur `A_dag`)
pour fermer la voie de gaming exposée par smoke #3 (sign-correct edges
à magnitude juste au-dessus du threshold). Le `Q_phys_interv` reste
reporté en **diagnostic tertiaire** dans le JSON A0'' (champ
`q_phys_interventional`), mais ne participe PAS au test statistique H1.
Pas de cherry-picking entre les deux : si le mémoire cite Q_phys, il
s'agit toujours de `Q_phys_cont` (PC5).

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

**PC8 follow-ups (Batch F-bis, 2026-06-13)** :

4. **sd floor + small-sample fallback** : avec n=3 seeds, Cohen's d peut
   diverger si les 3 seeds collapsent sur une A_dag identique
   (`sd_pathcplus → 0` → `d → ∞`). Floor explicite :
   ```python
   sd_pathcplus = max(sd_pathcplus_observed, 0.05)
   ```
   En parallèle du BCa CI, rapporter aussi un **Student-t one-sided CI
   (df=2, α=0.05)** sur Q_phys_cont. Pour passer H1, exiger que **LES DEUX
   bornes inférieures** (BCa AND Student-t) > baseline_cont upper CI.
   BCa avec n=3 est mathématiquement instable ; le Student-t one-sided
   sert de garde-fou conservatif.

5. **Random-null check** : la baseline V5-mini Q_phys_cont = 0.04 est
   *band-diagonal-induced*, pas random. Pour éviter qu'un Path C+ ré-arrange
   simplement le band-diagonal et "batte" 0.04 sans recovery causale,
   le critère H1 exige aussi de dépasser le null analytique :
   ```
   E[Q_phys_cont | A iid Normal(0,sigma)] = 0.5 × (n_phys / n_off_diag)
                                          = 0.5 × 5/30
                                          ≈ 0.083
   ```
   Donc Path C+ doit satisfaire `Q_phys_cont > 0.083` AVANT même de
   comparer au baseline. Avec le threshold PC5 (≥ 0.50), cette condition
   est automatiquement remplie, mais elle est listée ici pour fermer la
   voie d'un Path C+ borderline (e.g., Q_phys_cont = 0.06) qui passerait
   "> 0.04" sans dépasser le null random.

### PC5-bis — Collapse tie-break (added by council ter, 2026-06-13)

Si `compute_q_phys_continuous` retourne `(0.0, collapsed=True)` pour
**≥ 2 des 3 seeds** A0'', le test statistique H1 est **REMPLACÉ** par
un rapport de "protocol failure" et un re-tune est requis avant relancer
A0''. Un outcome COLLAPSE ne compte PAS comme `Q_phys_cont < 0.50`
contre H1 — c'est un mode d'échec différent (la matrice s'est effondrée
à zéro, pas "le modèle a échoué à recouvrer la structure").

Justification : collapse = pathologie optimization (λ_l1 trop fort, ou
sigma_data calib divergente). Compter cela comme "H1 failed" mélangerait
deux modes d'échec orthogonaux et empêcherait l'identification de la
cause racine.

### PC7-bis — Code-path identity acknowledged (added by council ter)

Précision sur la portée PC7 (smoke ≠ A0'') :

**Le smoke utilise** `scripts/finetune_stage1_bundle_b.finetune_bundle_b`
(Bundle B fine-tune, gate via `schedule_lambdas()` auto-scale ramp).

**A0'' utilise** `src/st_cdgm/training/training_loop.train_epoch_stage1`
(from-scratch, static `dag_grad_gate_value=1.0`).

Le code path est DIFFÉRENT. La fix Batch F-1 (gate auto-scale dans
`schedule_lambdas`) **n'a aucun effet sur A0''**. Le smoke ne valide PAS
le chemin code A0''. Le smoke valide uniquement :
1. Que le pipeline Bundle B exécute end-to-end avec les hyperparams Path C+.
2. Que les diagnostics post-mortem (Q_phys variants, projection hook,
   per-epoch trajectory) sont calibrés.

Le smoke ne porte **aucun poids inférentiel** pour H1. Toute évidence H1
repose exclusivement sur les 3 seeds A0'' rapportés au Chapitre 4.

### PC9 — Smoke triage protocol (added by council ter)

Si smoke #4 retourne :

- **`FAIL_NO_GAIN`** (phys_mag_gained ≤ 0.003) :
  - (a) Inspection diagnostique du `set_dag_grad_gate` path + signe `G_phys`
        avec **UN** re-run smoke.
  - (b) Si cause identifiée, appliquer le fix et relancer smoke **UNE FOIS**.
  - (c) Si pas de cause identifiée OU smoke #5 FAIL aussi, A0'' n'est PAS
        lancé et le projet pivote vers Path A1.
  - **AUCUN hyperparameter tuning autorisé entre smokes sans nouvel
    amendement PC.**

- **`FAIL_COLLAPSE`** :
  - λ_l1 peut être réduit **UNE FOIS** à la valeur pré-spécifiée 0.02
    (smoke actuel = 0.04).
  - Si collapse récidive sur le re-smoke, même pivot Path A1.

Ce protocole pré-enregistré ferme la porte au HARKing post-smoke
("on a tuné les hyperparams jusqu'à ce que ça passe").

### PC13 — Warm-start FT with NEW Path C+ params (locked BEFORE A1 launch retry, 2026-06-13)

PC4 blocks strict=False fallback to catch the smoke #2 failure mode (silent
re-init of existing weights when keys SHOULD have matched). But a legitimate
warm-start FT scenario also produces `strict=False`-required loads :
the V5_DIR ckpt was trained BEFORE Path C+ fixes (§1.6, J3, J1, J29) which
ADDED new parameters to the encoder, regression_head, and diffusion modules.
Loading V5_DIR into the post-fix model produces `missing_keys` for those
NEW params -- which is methodologically OK because the FT loop will train
them from random init.

PC13 distinguishes the two cases :

- **PC4 BLOCKING (unchanged)** : `unexpected_keys` non-empty OR shape mismatch.
  These indicate genuine semantic incompatibility -> AUTOMATIC EXCLUSION.

- **PC13 SOFT-PASS** : the strict=False fallback is allowed IF AND ONLY IF :
  1. `unexpected_keys == []` (no orphan ckpt weights)
  2. ALL `missing_keys` are in the pre-registered Path C+ NEW-params allowlist :
     - `metapath_convs.*` (§1.6 per-metapath convs, after legacy migration)
     - `layer_norms.*` (J3 per-metapath layer norms, after legacy migration)
     - `_state_adapter.*` (J1 eager init in regression_head)
     - `edge_gate.*` (J20 learnable edge gate, if added)
     - any other field explicitly listed below before A1 launch

  The JSON records `pc13_softpass_reason` with the matched-against-allowlist
  evidence + the explicit ckpt commit hash that justifies the warm-start.

- **PC13 HARD BLOCK** : any other strict=False outcome (e.g., legacy keys
  outside the allowlist, or shape mismatch) raises `RuntimeError`.

PC13 PRESERVES the PC4 filet (smoke #2 still blocked) while authorizing
the legitimate FT scenario. Without PC13, PC4 would mechanically force a
from-scratch retrain (A0'' scope) for any new Path C+ commit -- defeating
the cheap-FT advantage of A1.

**Allowlist for A1 launch (locked 2026-06-13, commit c7b65ee or successor) :**
```
[
  "metapath_convs",  # encoder §1.6 (after legacy migration)
  "layer_norms",     # encoder J3   (after legacy migration)
  "_state_adapter",  # regression_head J1
]
```
The diffusion module (J29 cfg_scale handling) does NOT add params -- J29 is
a runtime check, not a weight. So diffusion strict=False is NOT PC13-eligible
and remains PC4-blocking.

### PC12 — M2 reporting floor (locked BEFORE A1 launch, 2026-06-13)

Lors de la validation conseil du notebook A1 (commit fb7c7af), le DS a
identifié que cell 7 introduisait un seuil 0.30 comme "M2 defendable bar"
**non pré-enregistré**. Risque HARKing : importer le smoke-pass-gate 0.30
dans le verdict A1 après avoir vu les résultats smoke.

PC12 résout l'ambiguïté **AVANT** que A1 soit lancé :

1. **PC5 H1 acceptance threshold = 0.50** (locké en PC5, INCHANGÉ).
   `Q_phys_cont ≥ 0.50 + tous les gates PC8` → H1 ACCEPTED.

2. **0.30 est un REPORTING FLOOR, pas une acceptance threshold.**
   Quand `0.30 ≤ Q_phys_cont < 0.50`, le résultat est rapporté dans la
   section "Discussion / Limitations" du mémoire comme :
   > "Path C+ A1 achieves Q_phys_continuous = X (8.X× baseline V5-mini
   > of 0.04, p < α paired Wilcoxon) but does NOT meet the pre-registered
   > H1 acceptance threshold of 0.50. Interventional sign-consistency is
   > demonstrated; H1 is not statistically accepted."

3. **Le label de verdict pour ce cas est `BELOW_H1_ABOVE_FLOOR`** (pas
   `M2_PASS` qui implique acceptance). Aucune claim H1 dans le mémoire
   ne peut citer ce résultat comme "Path C+ effective". Le mémoire peut
   citer ce résultat comme "preliminary evidence" mais doit explicitement
   noter "H1 not met" dans la même phrase.

4. **Quand `Q_phys_cont < 0.30`**, le verdict est `FAIL` (pas de claim
   évidentielle possible).

5. **Le seuil 0.30 est lockée AVANT smoke-#5/A1-launch** au commit qui
   ajoute cette section. Aucune modification post-A1 n'est autorisée
   sans amendement PC ultérieur signé avant ré-analyse.

Cette amendement préserve la défendabilité M2 tout en empêchant le
relâchement post-hoc des critères PC5. Le mémoire peut faire des
claims "preliminary" sous PC12 mais pas des claims "H1 accepted".

### PC10 — Lineage snapshot integrity (added by council ter)

Le champ `path_c_plus_full_fix_lineage` dans les JSONs A0'' est un
**deep copy** de `PATH_C_PLUS_ALL_FIXES` au moment du stamp (via
`copy.deepcopy`). La constante module est gelée au commit hash de lancement
d'A0'', enregistré dans le champ `pre_registration_commit`.

Ceci empêche qu'une mutation post-A0'' de la constante module ne change
rétroactivement ce que les JSONs A0'' rapportent comme "tous les fixes
appliqués". Auditabilité long-terme : un reviewer en 2030 peut prendre
le commit du JSON, vérifier que le lineage matche `PATH_C_PLUS_ALL_FIXES`
à ce commit-hash spécifique.

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
| 1.1 | 2026-06-13 | PC5-PC8 ajoutés post-mortem smoke #3 (council ARTEFACT verdict, commit b5c57b8, ref `project_smoke_artefact.md`). Endpoint H1 redéfini d'interventional (HYPERPLAN §H1 = `mean(\|mu_HR(A_real)-mu_HR(A_zeroed)\|)/mean(\|mu_HR(A_real)\|)`) → structural magnitude-ratio (PC5). Justification : smoke #3 a montré que le binary metric peut être gamé sous projection (3 edges TP à 0.011 = threshold + 1%). L'interventional Q_phys reste reporté comme diagnostic tertiaire à A0''. Tightening only (criteria stricter, never looser). | Council DS sign-off pending |
| 1.2 | 2026-06-13 | Batch F post-validation : PC8 sd floor (0.05) + Student-t fallback CI, PC8 random-null check (0.083), PC5 endpoint-drift note. PC7 tombstone tagging hardened : smoke JSONs ne portent PLUS le marker `fixes_applied` (réservé A0''). | Council Math/AI/DS sign-off pending |
| 1.3 | 2026-06-13 | Council ter follow-up post-blindspots : PC5-bis (collapse tie-break), PC7-bis (code-path identity explicit, smoke ne pré-valide PAS A0''), PC9 (smoke triage protocol — anti-HARKing), PC10 (lineage deep-copy integrity). Phys_mag_gained threshold lowered 0.005 -> 0.003 (Math Prof + AI Eng convergence : 55% du bandwidth max au lieu de 91%). | Council Math/AI/DS sign-off pending |
| 1.4 | 2026-06-13 | Pre-A1 council validation : PC12 (M2 reporting floor 0.30 explicitement séparé de PC5 H1 acceptance 0.50, anti-HARKing). Notebook A1 patché : PC4 BLOCKING gate dans run_a1_seed (raise si diffusion strict=False), K9 temporal split dates passés à NetCDFDataPipeline + split="train"/"val" (DS flag : sans ça val ⊂ train). BCa CI clip z0 + a1/a2 pour n=3 dégénéré (Math Prof). | Council Math/AI/DS sign-off pending |
| 1.5 | 2026-06-13 | A1 launch retry : PC4 trop strict pour le scénario warm-start FT car §1.6/J3/J1 ajoutent NEW params absents du V5_DIR ckpt original. PC13 ajouté : SOFT-PASS pour strict=False ssi unexpected_keys=[] AND missing_keys ⊆ allowlist {metapath_convs, layer_norms, _state_adapter}. Diffusion strict=False reste PC4-bloquant (J29 = runtime check, pas un weight). Préserve filet smoke #2. | Council Math/AI/DS sign-off pending |

Toute modification post-signature doit être tracée ici avec justification scientifique.
