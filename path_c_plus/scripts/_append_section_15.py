"""Append section 15 to architecture_journey.md."""
from pathlib import Path

SECTION = r"""

## 15. Bilan complet seed 42 v1 (avec conditioning μ_HR) et changements appliqués pour v2 (2026-06-16)

Cette section archive les métriques complètes du checkpoint seed 42 v1 (Stage 1 + Stage 2 avec `causal_concat=True` et μ_HR dans le canal de conditioning du UNet, design CorrDiff Nature CEE 2025 canonique) tel qu'évalué à la fin de §14, et formalise pourquoi la version v2 (Mardani fix au niveau cache) a été lancée pour remplacer le conditioning total μ_HR. Les valeurs ci-dessous sont reprises verbatim des JSON `oracle_full/seed_42/{summary,domain_metrics,final_validation_metrics}.json`.

### 15.1 Stage 1 seed 42 v1 — métriques de récupération causale

| Métrique Stage 1 | Valeur seed 42 v1 |
|---|---|
| `q_phys_binary_final` | **1.0** (toutes les arêtes physiques récupérées avec le bon signe) |
| `q_phys_adaptive_final` | **1.0** |
| `q_phys_continuous_final` | **0.99984** |
| `q_phys_collapsed_final` | False |
| `skeleton_f1_final` | **1.0** |
| `n_extra_edges_final` | **0** (zéro arête spurieuse → sparse PASS) |
| `phys_mag_gained_final` | 4.4115 |
| `ablation_report.passes` / `ratio` / `threshold` | 1.0 / 1.0 / 0.05 |
| `best_s1_val_mse` | 0.04696 |
| `sigma_data_post_s1` | 0.18006 |
| `sigma_min_post_s1` | 0.00360 |
| `projection_log` n_pre / n_post / n_floor | 0 / 0 / 0 (aucune projection spectrale nécessaire) |

`A_dag_final` matérialise exactement le schéma quasi-géostrophique descendant (Holton & Hakim Ch. 6) :
- GP250 → GP500 : poids 0.9747
- GP500 → GP850 : poids 0.9736
- GP850 → SP_HR : poids 0.9492
- Méta-paths GP250 → SP_HR (0.9392) et GP500 → SP_HR via GP850 (0.9391)
- Toutes les autres entrées hors-diagonale au-dessous de 1.2e-4 (effectivement zéro)

Hyperparamètres figés (`pathcplus_hyperparams`) : `lambda_dag_prior=0.40`, `lambda_l1_start=0.04 → end=0.005`, `g_phys_alpha=0.25`, gate warmup désactivé. Pre-registration commit : `d4a4457`. Run type : `seed_42`, K9 temporal split (train 1980-2009, val 2010-11, test 2012-13, holdout 2014).

**Verdict Stage 1** : `H1_PASS_SPARSE` empirique sur seed 42 — récupération structurelle parfaite avec interprétabilité maximale. Ce résultat est ce que la v2 doit **préserver bit-exact** via `freeze_stage1`.

### 15.2 Stage 2 seed 42 v1 — métriques de skill prédictif (avec μ_HR en conditioning)

`final_validation_metrics.json` (16 batches val, K=64 ensembles, eval_time=2331 s) :

| Métrique | seed 42 v1 | Noncausal v4 (ckpt_v2_corrdiff_normal) | Δ relatif |
|---|---|---|---|
| RMSE | **0.14137** | 0.12430 | **+13.7 %** (pire) |
| MAE | **0.06971** | ~0.060 | **+16 %** (pire) |
| Pearson global | 0.76658 | 0.835 | -8.2 % |
| **Pearson per-sample avg** | **0.76181** (n=16) | **0.83440** | **-8.7 %** (pire) |
| Spread mean | 0.04654 | 0.0512 | -9 % |
| F1 @ p95 | **0.58106** | 0.65020 | **-10.6 %** (pire) |
| F1 @ p99 | **0.44705** | 0.51230 | **-12.7 %** (pire) |
| RAPSD distance | 307.78 | ~250 (ordre) | +23 % (pire) |

`domain_metrics.json` (full validation set) :

| Métrique domaine | seed 42 v1 | Noncausal v4 | Δ |
|---|---|---|---|
| Spread-Skill Ratio (SSR) | **0.32924** | 0.41160 | **-20 %** (sous-dispersé) |
| CRPS-Gaussian | 0.05409 | ~0.045 | +20 % (pire) |
| Intensity hist L1 distance | 0.10352 | ~0.085 | +22 % (pire) |
| RMSE secondary | 0.14137 | 0.12430 | +13.7 % |
| Pearson global secondary | 0.76658 | 0.835 | -8.2 % |

**Tous les points du Pareto sont strictement dominés par noncausal v4** — sauf l'axe interprétabilité où seed 42 v1 a Q_phys=0.9998 alors que noncausal a Q_phys structurellement 0.

### 15.3 Diagnostic μ_HR — pourquoi la v1 a régressé : copy-mode confirmé

Les deux diagnostics post-eval intégrés au notebook révèlent le mécanisme :

```
"mu_HR_ablation": {
    "delta_signal_ratio_avg": 1.02213,
    "verdict": "MU_HR_CONDITIONS"
},
"shortcut_diagnostic": {
    "shortcut_ratio": 1.18219,
    "verdict": "REFINEMENT_OK"
}
```

Lecture :

1. **`mu_HR_ablation.delta_signal_ratio_avg = 1.022` (102 %)** — quand on remplace `μ_HR_real` par zéro à l'inférence, la prédiction change de **102 % du signal** — c'est-à-dire que μ_HR constitue **à lui seul ~100 % du signal de prédiction**. La diffusion ne fait pratiquement plus que recopier μ_HR avec un bruit additif. C'est le **copy-mode collapse** anticipé par le Math Prof : la diffusion réduit son problème à `D_θ(x_noisy, μ_HR, baseline_log) ≈ μ_HR + ε` au lieu d'apprendre une décomposition causale.

2. **`shortcut_diagnostic.shortcut_ratio = 1.182` ≥ 1 → `REFINEMENT_OK`** — la prédiction complète a un signal *plus grand* que la baseline seule (≈ +18 %), donc le résidu de diffusion contribue quand même. Mais ce contribution se fait au-dessus d'un μ_HR rigide qui a déjà fixé la quasi-totalité du signal — d'où la régression sur RMSE/Pearson.

3. **`B₁ rank-5` (Math Prof)** : `A_dag` est creuse (5 arêtes physiques sur 30 entrées possibles). μ_HR, calculé comme propagation linéaire le long de `A_dag`, vit donc dans un sous-espace de rang 5 du HR. La diffusion conditionnée sur μ_HR est forcée de prédire HR dans ce même sous-espace — ce qui explique pourquoi le **70-80 % du MSE est structurellement irréductible** sans intervention sur Stage 2.

C'est précisément le scénario "interprétabilité au coût de la skill" prévu en §14.9.1.

### 15.4 Pourquoi abandonner le conditioning total μ_HR

Le council 7-reviewers de §14.3 converge sur le fait que le **conditioning μ_HR canal-concat** est la cause primaire de la copy-mode. Quatre raisons cumulées pour passer à v2 :

1. **Diagnostic empirique direct** : `mu_HR_ablation = 102 %` est la signature numérique du copy-mode. Toutes les techniques alternatives (Min-SNR, multi-EMA, sampler tweaks) attaqueraient le symptôme sans toucher la cause structurelle.

2. **Argumentation Mardani (Lit Review Western)** : la version v3 arXiv de Mardani et al. (2309.15214) écrit le résidu `r = x − μ̂` mais **ne montre PAS μ̂ comme entrée du dénoiseur** dans les équations formelles. La version Nature CEE 2025 a re-introduit μ̂ comme canal de conditioning, ainsi que PhysicsNeMo `hr_mean_conditioning` et StormCast Appendix C — mais notre setup d'entraînement (capacité UNet, sigma_data calibré post-S1, residual variance) reproduit la pathologie qu'on observe. Retirer μ_HR du canal de conditioning revient au design v3 arXiv original.

3. **Bornage formel du copy-mode (Math Prof B₁ post-hoc)** : dans la configuration v1, le rank-5 de A_dag impose un plancher de MSE d'au moins **70-80 % du MSE total**. Aucune optimisation Stage 2 ne peut faire mieux que ce plancher sans changer la structure du conditioning. L'option "réduire `in_channels` à 2 = `[δ_noisy, baseline_log]`" est mathématiquement équivalente au cache-zeroing à coût-d'implémentation nul.

4. **Préservation de l'interprétabilité** : le freeze de Stage 1 garantit Q_phys=0.9998 bit-exact. Le DAG reste interprétable indépendamment de ce que le Stage 2 décide d'utiliser. μ_HR rentre toujours dans la prédiction finale par décomposition additive `HR_pred = baseline + μ_HR_real + residual_diffusion(baseline_log)` au moment de l'éval — c'est l'architecture CorrDiff canonique.

### 15.5 Changements appliqués pour v2 (récapitulatif technique)

Les modifications matérialisées dans `path_c_plus/scripts/st_cdgm_seed42_eval.ipynb` (cellules 10/11/12) et `path_c_plus/audit/COUNCIL_ROUND2_SEED42_V2.md` :

| Composant | seed 42 v1 (conditioning) | seed 42 v2 (Mardani cache fix) | Justification |
|---|---|---|---|
| **μ_HR dans UNet input** | μ_HR_real ∈ ℝ^(B,C,H,W), `in_channels=3` `[δ_noisy, μ_HR, baseline_log]` | **μ_HR = 0** dans cache BS32b, `in_channels=3` mais slice μ_HR à zéro ≡ `in_channels=2` au niveau gradient | casse le copy-mode (mu_HR_ablation 102 % → ~0 %) |
| **`delta_target` (cible Stage 2)** | `δ = HR − baseline − μ_HR_real` | **inchangé** : `δ = HR − baseline − μ_HR_real` (cloné depuis cache v1) | la décomposition causale reste correcte ; μ_HR_real reste dans la reconstruction finale |
| **Conditioning channel `baseline_log`** | présent | présent | inchangé |
| **Conditioning class-embedding** | présent (intervention encoder.encode) | présent | inchangé |
| **`freeze_stage1` (encoder + rcn_cell + regression_head)** | actif | **actif** | Q_phys préservé bit-exact, vérifié par `A_dag drift = 0.0` chaque epoch |
| **EMA** | multi-decay {0.999, 0.9995, 0.9999} sweep post-hoc | **single decay 0.999** | sweep multi-EMA inutile sans `mu_HR_zero` dans le cache (cf §15.6 ci-dessous) |
| **`conditioning_dropout_prob`** | 0.0 (jamais activé en v1) | **0.0** (no-op puisque μ_HR=0 dans cache : la branche droppée est identique à la branche gardée) | audit Climate ML round 2 |
| **`lambda_contrastive_dag`** | 0.5 (souhaité, mais cache v1 n'avait pas `mu_HR_zero` → jamais exécuté) | **0.5 nominal, effectivement 0** (cache v2 n'a pas `mu_HR_zero` non plus, et serait de toute façon = `mu_HR` = 0) | structurellement incompatible avec Mardani fix : si μ_HR=0 partout, il n'y a pas de comparaison à faire entre `μ_HR_real` et `μ_HR_zero` |
| **`sampler_scheduler`** | `edm_karras` (S_churn=40) | **`dpm_solver++`** (S_churn=10, num_steps=32, cfg_scale=1.0) | dpm_solver++ converge en 32 steps ; cfg_scale=1.0 car pas de branche uncond avec μ_HR=0 cache |
| **`K_samples` ensemble eval** | 64 | **128** | doubler l'ensemble réduit la variance Pearson per-sample |
| **`tail_p95/p99_weight`** | (8, 25) | **(4, 12)** | recommandation Climate ML : poids trop agressifs sur-pénalisent la queue et dégradent F1@p99 |
| **Checkpointing** | manuel (Stage 1) + autosave Stage 2 | **atomique per-epoch** (fsync + os.replace + dir fsync + orphan cleanup) | mirror du pattern noncausal cell 48, survie aux kills FUSE Drive |

Le notebook a perdu sa cellule Fix #1 (cell-6-fix1-sampling, sampling multi-variants) en round 2 du council : la pathologie est jugée d'origine training, pas sampling, donc Fix #2 attaque la racine directement.

### 15.6 Pourquoi `contrastive_dag` est resté silencieusement OFF

Le cache BS32b v1 a été construit avec `precompute_stage1_outputs(..., dag_variants=["normal"])` — seul `μ_HR_real` est stocké, pas `μ_HR_zero`. La fonction `train_epoch_stage2_cached` attend la clé `mu_HR_zero` dans le batch pour calculer la loss contrastive ; en son absence elle skip l'étape (warning `"contrastive_dag : aucun pas exécuté"`).

Conséquence cumulative en v2 :

- Le cache v2 hérite des 4 clés v1 (`mu_HR`, `baseline_log`, `delta_target`, `valid_mask`) avec `mu_HR` zeroé → toujours pas de `mu_HR_zero`.
- Même si on l'avait calculé, avec Mardani fix `μ_HR_zero = μ_HR_real = 0` → loss contrastive identiquement nulle.
- Donc `dag_sensitivity` reste structurellement à 0 dans toute la v2.

**Cela n'affecte pas l'interprétabilité** (Q_phys = f(A_dag), A_dag frozen) ; cela affecte la métrique secondaire `dag_sensitivity` qui mesurait dans la v1 la sensibilité comportementale de la diffusion à μ_HR. Le mémoire doit présenter cette métrique comme "non applicable en v2 par design", pas comme "DAG ignoré".

### 15.7 Estimation des métriques attendues v2 (pré-launch)

Plancher théorique Math Prof (`Var(HR) = Var(baseline) + Var(μ_HR) + Var(δ)`) : avec μ_HR retiré du conditioning mais conservé dans la reconstruction finale, le Pearson_PS noncausal (0.834) est le **plafond architectural** que v2 peut approcher sans le dépasser strictement. Distribution probable des résultats v2 vs noncausal v4 :

| Métrique | seed 42 v1 | Cible v2 (non-infériorité) | Plafond v2 |
|---|---|---|---|
| RMSE | 0.1414 | < 0.1305 (within +5 %) | ≈ 0.124 (noncausal) |
| Pearson_PS | 0.762 | > 0.824 (within −1.2 %) | ≈ 0.834 |
| F1 @ p99 | 0.447 | > 0.481 (within −6 %) | ≈ 0.512 |
| SSR | 0.329 | ∈ [0.4, 1.6] (calibration) | — |
| RAPSD | 307.78 | < 0.20 (echelle relative noncausal) | — |
| **Q_phys** | **0.9998** | **0.9998** (bit-exact, frozen) | **0.9998** ✓ |
| **dag_sensitivity** | — (v1 contrastive_dag déjà OFF) | **0.0 par design** | **0.0** |

Probabilité de résultat défendable au M2 estimée à **~95 %** (cf §14.8). Le scénario à risque est P(v2 ≈ v1) ≈ 5 % où la Mardani fix n'aurait pas suffi à casser le copy-mode — ce que les logs détaillés `mu_HR.absmax = 0` à chaque batch et la décroissance attendue de `loss_diff` 0.5-1.5 → 0.05-0.20 doivent détecter dès les 20 premières epochs.

### 15.8 Note pour la rédaction du mémoire

Le narratif à présenter en chapitre Résultats :

1. **Stage 1** : récupération causale parfaite, Q_phys=0.9998, sparse FAIL avorté (n_extra=0).
2. **Stage 2 v1** (avec conditioning total μ_HR) : régression −9 à −20 % sur les métriques standard vs noncausal v4, **diagnostiquée comme copy-mode** par `mu_HR_ablation=102 %`.
3. **Stage 2 v2** (Mardani cache fix) : retour au design v3 arXiv pré-conditioning total ; séparation propre entre interprétabilité (Stage 1, Q_phys) et skill (Stage 2, diffusion). Q_phys préservé bit-exact (A_dag frozen). Si v2 atteint la zone de non-infériorité vs noncausal, le mémoire peut affirmer **"interprétabilité gagnée sans coût significatif de skill"**. Sinon, le narratif scientifique reste **"interprétabilité × skill : Pareto frontier identifiée, choix tactique pour le M2, ouverture à article de recherche sur 3 seeds avec auxiliary residual head r_φ(H)"**.
4. **Limitations** : pas d'évaluation OOD GCM en v2 (in-distribution validation seulement, ACCESS-CM2 val K9), pas de seeds 7 et 123 (reportés à l'article), `dag_sensitivity` non rapportée (non applicable par design Mardani fix).
"""

p = Path(r"c:/Users/reall/Desktop/climate_data/architecture_journey.md")
with p.open("a", encoding="utf-8") as f:
    f.write(SECTION)
print(f"Appended {len(SECTION)} chars to {p}")
print(f"New total lines: {sum(1 for _ in p.open(encoding='utf-8'))}")
