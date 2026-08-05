# Journal complet des échecs depuis P0 → seed 42 v2 (2025-01 → 2026-06-16)

Ce document recense **chronologiquement** tous les échecs constatés dans le projet
ST-CDGM / ORACLE, leur cause racine confirmée, et la leçon transférable pour la
prochaine architecture. Pas de narratif, pas de spéculation — uniquement les faits
documentés dans `architecture_journey.md`, les commits git, et les memory files.

Total : **47 échecs distincts** répartis en 6 classes.

---

## CLASSE A — Échecs architecturaux (le modèle ne fonctionne pas)

| # | Modèle/Phase | Date | Cause racine confirmée | Leçon |
|---|---|---|---|---|
| A1 | Approches déterministes P0 (RF, XGBoost, LSTM, U-Net) | 2025 début | Lissage des extrêmes ; pas génératif ; pas de quantification d'incertitude | **Tout modèle déterministe perd sur F1@p99** pour la précipitation |
| A2 | HG-SCM (Heterogeneous Graph SCM) intégré | 2025-09 | Papier d'origine = classification de nœuds discrète, pas downscaling continu. Greffe inter-domaine sans bénéfice | **Ne pas greffer un framework d'un domaine à un autre** sans validation théorique |
| A3 | CorrDiff Normal V1 (mars 2026) | 2026-03 | μ_HR en conditioning channel → copy-mode shortcut dès les premiers epochs | **Tout conditioning par concaténation channel-wise crée un shortcut** si la cible est un résidu corrélé au conditioning |
| A4 | BS31a-e shortcut fix attempts | 2026-05 | 5 variantes de fix tentées : Min-SNR, tail_weight, contrastive_dag, spectral loss, AdaGN. Aucune n'a fermé le shortcut | **Optimiser le sampler ou la loss ne casse pas un shortcut architectural** |
| A5 | V5 (full retrain avec F1-F4 fixes) | 2026-05-17 → 21 | Pearson 0.090 cible jamais atteint ; F1-p99 regression confirmée vs V4 | **Augmenter la capacité du UNet (V4→V5) sans toucher au conditioning n'apporte pas de gain** |
| A6 | V5-mini (P1+P2+P3-lite+P5+A1) | 2026-06-05 | Gain Spread/RMSE +30% mais F1-p99 -3% vs V4 | **Le gain spread/skill peut masquer une régression sur les extrêmes** |
| A7 | Bundle B + CASTLE + masque physique | 2026-06-09 | Smoke #1, #2 et #3 montrent que CASTLE seul n'attaque pas la rigidité μ_HR | **CASTLE résout les pathologies Stage 1 sans toucher au shortcut Stage 2** |
| A8 | Smoke test #3 (Bundle B 10ep) | 2026-06-13 | PASS verdict était un **artefact** : `project_dag_floor min_norm=0.10` injectait Q_phys=1.0 mécaniquement ; A_dag norm FLAT 0.530→0.526 | **Toujours vérifier que A_dag norm/asymmetry a vraiment bougé** avant de croire un PASS |
| A9 | Path C+ A1 (3 seeds × 25ep Bundle B FT) | 2026-06-13 | H1_PASS_INTERVENTIONAL_ONLY mais sparse FAIL (n_extra=7 sur 3 seeds) | **Le band-diagonal attractor est structurel** au 6-node architecture, pas un bug d'optimisation |
| A10 | Path C+ Option C 3 seeds | 2026-06-14 | Avorté après seed 42 (budget Colab Pro+ épuisé, ~35h consommées sur 85-120h estimées) | **Estimer le compute en multipliant par 1.5-2 pour Colab Drive I/O** |
| A11 | Seed 42 v1 (Stage 2 avec conditioning μ_HR) | 2026-06-15 | RMSE +13.7%, Pearson -8.7%, F1@p99 -12.7% vs noncausal v4. `mu_HR_ablation=102%` → **copy-mode confirmé** | **μ_HR rigide en conditioning + delta_target=HR-baseline-μ_HR → la diffusion se réduit à copier μ_HR** |
| A12 | Seed 42 v2 (Mardani fix : μ_HR=0 en cache) | 2026-06-17 | RMSE +18% vs noncausal v4. v2 légèrement PIRE que v1. Pearson 0.7405 (-2.8% vs v1) | **Retirer μ_HR du conditioning casse le copy-mode mais ne récupère pas la skill** — le plateau B₁ rank-5 persiste |

---

## CLASSE B — Échecs de diagnostic (PASS qui étaient des artefacts)

| # | Diagnostic | Date | Cause racine | Leçon |
|---|---|---|---|---|
| B1 | Smoke #3 PASS = artefact | 2026-06-13 | `project_dag_floor` injectait 0.011 sur les 5 positions physiques + binary threshold Q_phys ≥ 0.01 | **Ne jamais utiliser un threshold qui peut être atteint par injection** |
| B2 | `shortcut_diagnostic.verdict = REFINEMENT_OK` v1 trompeur | 2026-06-15 | Ratio 1.18 ≥ 1 mais simultanément `mu_HR_ablation = 102%` → diagnostic contradictoire interprété optimistically en v1 | **2 verdicts contradictoires = vérifier les 2 mécanismes, ne pas faire confiance au plus optimiste** |
| B3 | 5 pathologies Stage 1 non détectées avant Phase 6 | 2026-06-09 | Tail truncation, cécité humidité, DAG collapse uniforme, Q_int=0, FSS@50mm — tous masqués par les métriques standards | **RMSE/Pearson seuls cachent les pathologies structurelles** |
| B4 | `dag_grad_gate` reaching only 0.33 max | 2026-06-13 | `DEFAULT_HYPERPARAMS` figeait `dag_gate_warmup_start_epoch=5, end=20`, défait l'auto-scale | **Vérifier que les gates atteignent vraiment leur valeur cible** sur le nombre d'epochs effectif |
| B5 | Cohen's d=9.6 sans gain de skill | 2026-06-13 | sd floor = 0.05 active → d est borné inférieurement, masque la similarité entre 3 seeds (sd=0.00017) | **Cohen's d avec sd floor est un lower bound**, pas une mesure d'effet réelle |
| B6 | Multi-EMA sweep sans calibration | 2026-06-15 | EMA_CHOICE="0.9995" hard-codé alors que seul decay=0.999 entraîné | **Tout sweep post-hoc multi-EMA nécessite que les decays soient effectivement entraînés** |

---

## CLASSE C — Bugs d'implémentation (corrigés post-hoc, souvent coûteux)

| # | Bug | Cellule/Fichier | Coût évité | Leçon |
|---|---|---|---|---|
| C1 | `_persist_state_dict` undefined | cell 11 v2 retrain | 25-30h gaspillées si non détecté (NameError sur 1er save) | **Toujours linter le code avant un long run** |
| C2 | EMA key mismatch `ema_state_dict` vs `ema_{EMA_CHOICE}_...` | cell 12 v2 eval | Eval silencieuse sur non-EMA → métriques fausses | **Un `.get(key, default=None)` qui returne None ne crashe pas mais corrompt silencieusement** |
| C3 | Mardani key mismatch `_in_cache` vs `_in_conditioning` | cell 10 vs cell 12 | Distribution shift train ↔ eval (train fed 0, eval fed real μ_HR) | **Un seul nom de config par feature, partagé entre train et eval** |
| C4 | CFG + conditioning_dropout incohérent avec Mardani fix | V2_CONFIG | cfg_scale=1.15 sans branche uncond → comportement undefined | **Vérifier la cohérence de configuration entre composants interdépendants** |
| C5 | F1 keys `f1_p95.0` vs `p95` (10 mismatches) | cells 6, 7, 12, 13 | Comparaison silencieuse retournant n/a partout | **Aligner immédiatement les clés JSON avec la fonction qui les écrit** |
| C6 | `_stat()` torch.quantile > 16M elements | cell 10 logs | Crash sur cache 168M éléments | **torch.quantile a une limite hard à ~16M**, sous-échantillonner systématiquement |
| C7 | Duplicate `_stat` body après patch | cell 10 | Double exécution → 2e crash | **Toujours vérifier que les patches en `replace()` ne laissent pas de leftovers** |
| C8 | DPM-Solver++ sigma schedule incompatible avec EDM training | cell 12 v2 eval | RMSE=87.7, Pearson=-0.62 → 50 min eval gaspillées | **JAMAIS mélanger samplers** — utiliser exactement le même que la loss d'entraînement |
| C9 | Comma after `#` in dict literal | cell 10 V2_CONFIG | SyntaxError immédiat | **`#` mange tout jusqu'au newline, mettre la virgule AVANT** |
| C10 | Conditioning_dropout no-op avec Mardani fix | cell 11 | Dropped branch = kept branch → CFG break | **Identifier les configurations devenues no-op après une autre modification** |
| C11 | `contrastive_dag` silencieusement OFF | cell 10 + train_epoch | Cache v1/v2 sans `mu_HR_zero` key → loss skip toute la training | **Tout composant de loss qui peut "skip silently" doit raise un warning visible AU PRE-RUN** |
| C12 | Cellule Fix #1 sampling jamais nécessaire | cell-6-fix1-sampling | 2-3h calcul mort si exécutée | **Supprimer le code mort avant launch, pas après** |
| C13 | `_pred_full = _pred_mean + _mu_concat` sans baseline | cell 8 v1, cell 12 v2 | Math Prof flagué en round 1, deferred, peut-être source du -3% v2 vs v1 | **Implémenter immédiatement les fixes mathématiques validés** |

---

## CLASSE D — Échecs méthodologiques

| # | Échec | Cause | Leçon |
|---|---|---|---|
| D1 | Single seed evaluation (seed 42) | Budget Colab épuisé → impossible de faire 3 seeds | **1 seed n'est jamais publication-quality** ; toujours réserver 3× budget |
| D2 | Pivot métrique pendant le run (Pearson/F1 → CRPS/SSR/RAPSD) | V5 a abandonné Pearson après l'avoir manqué | **Figer les métriques AVANT le run, pas pendant** |
| D3 | Pas d'OOD eval en v2 (seulement ACCESS-CM2 val K9) | Cell 12 n'inclut pas domain_metrics + 3 GCM aligned | **OOD est non-négociable** pour un mémoire en downscaling climatique |
| D4 | `dag_sensitivity` non rapportée car cache sans `mu_HR_zero` | precompute avec `dag_variants=["normal"]` seul | **Pre-compute TOUTES les variants requises** par les losses configurées |
| D5 | 7-reviewer council activé tardivement (post-A1) | Compute déjà engagé sur des décisions sub-optimales | **Council BEFORE compute, pas après** |
| D6 | RAPSD/CRPS calculés sur batch subsets seulement | Pour gagner du temps eval | **Sub-sampling brisée la comparaison apples-to-apples** vs noncausal full |

---

## CLASSE E — Échecs compute / opérationnels

| # | Échec | Cause | Leçon |
|---|---|---|---|
| E1 | Path C+ Option C estimé 85-120h, réalisé seed 42 seul (~35h) | Drive I/O stalls + epoch timing instable | **Estimer 1.5-2× le compute prévu sur Colab Pro+** |
| E2 | Atomic checkpointing absent → kills Colab perdent epochs | Pattern noncausal cell 48 pas mirroré dans Stage 2 | **Atomic save + fsync + os.replace + dir fsync + orphan cleanup obligatoire** sur Colab |
| E3 | Round 1 council laissé passer 5 blockers du v2 notebook | Round 1 = 4 reviewers, pas assez de couverture code | **Council code-level ET threshold-level OBLIGATOIRES** |
| E4 | v2 eval initial avec dpm_solver++ a tourné 50+ min avant détection garbage | Pas de sanity check pré-eval | **Run sanity check sur 1 batch avant l'eval full** |
| E5 | Notebook a grossi de 75k → 110k chars (council patches) | Patches incrémentaux sans refactor | **Refactor obligatoire dès que les patches dépassent 30%** |

---

## CLASSE F — Échecs conceptuels (le DAG ne suffit pas, seul)

| # | Échec conceptuel | Confirmation empirique | Leçon |
|---|---|---|---|
| F1 | « DAG → μ_HR → conditioning » crée un copy-mode | mu_HR_ablation=102% en v1 | **Le DAG ne doit PAS être l'unique vecteur d'information vers la diffusion** |
| F2 | Bias floor B₁ rank-5 ≈ 70-80% du MSE | Math Prof analytique + v2 empirique (Pearson plafond ≈ noncausal) | **Un DAG creux à 5 arêtes ne suffit pas à représenter un champ HR continu** sans tête auxiliaire |
| F3 | Stage 1 frozen → impossible d'améliorer la skill au-delà du plancher B₁ | v2 prouve : skill identique à v1 même avec Mardani fix | **Si Stage 1 doit rester frozen, alors Stage 2 a besoin d'une voie d'expression hors-DAG** |
| F4 | Auxiliary residual head r_φ(H) (Math Prof) jamais implémentée | Recommandée en §14.3 mais pas dans le code | **La recommandation théorique la plus solide est restée non-testée** — c'est le dette critique |
| F5 | Interprétabilité Q_phys=0.998 ne convertit pas en skill prédictif | v1 et v2 confirment : Q_phys élevé n'est pas corrélé à RMSE/Pearson | **Q_phys et skill sont 2 axes orthogonaux du Pareto** — gagner sur un n'implique pas gagner sur l'autre |
| F6 | μ_HR rigide impose un sous-espace de rang 5 du HR | Math Prof B₁ : Var(HR_pred) ≤ Var(baseline) + 5·Var(μ_HR_chan) | **La capacité du UNet ne compte pas si l'input est bottlenecked à rang 5** |
| F7 | 6-node band-diagonal attractor persiste malgré 22 P0 fixes | A1 sparse FAIL avec n_extra=7 reproductible 3 seeds | **L'architecture 6-node a un biais inductif structurel qu'aucun reg term ne résout** |

---

## CE QUI FONCTIONNE (validé empiriquement, à conserver)

| ✓ | Composant | Évidence |
|---|---|---|
| W1 | **Stage 1 DAG learning** (encoder GNN + RCN cell + DAGMA + regression head) | Q_phys=0.9998, skeleton F1=1.0, n_extra=0 sur seed 42 ; A_dag = schéma QG exact (GP250→GP500→GP850→SP_HR) |
| W2 | **Noncausal CorrDiff v4** (RMSE 0.124, Pearson 0.834) | C'est le SOTA à battre — toutes nos variants causales sont en dessous |
| W3 | **EDM sampler `edm_karras` Heun ODE** | Stable, reproductible, ne déstabilise pas le sampling (vs dpm_solver++) |
| W4 | **Atomic checkpointing per-epoch** (fsync + os.replace + dir fsync) | Survie aux kills Colab FUSE, validé sur 200 epochs v2 retrain |
| W5 | **7-reviewer multi-language council** (EN + 中文 + 日本語 + 한국어) | Capture techniques absentes de la littérature anglophone (Dispersive Loss, FSS pivot, AdaGN-everywhere) |
| W6 | **Pre-registration commit + locked hyperparams** | Bloque le HARKing, valide A1 H1 acceptance |
| W7 | **K9 temporal split** (train 1980-2009 / val 2010-11 / test 2012-13 / holdout 2014) | Prévention de la fuite temporelle |
| W8 | **`freeze_stage1` API** (encoder + rcn_cell + regression_head) | A_dag drift = 0.0 bit-exact sur 200 epochs prouvé |
| W9 | **Pipeline BS32b cache** (mu_HR, baseline_log, delta_target, valid_mask) | Permet train Stage 2 sans recomputer Stage 1 |
| W10 | **Mardani fix au cache level** | Casse le copy-mode (mu_HR_ablation 102% → ~0) — succès partiel |

---

## LEÇONS TRANSFÉRABLES (pour la prochaine architecture)

1. **Le DAG est nécessaire mais NON SUFFISANT pour battre noncausal v4.** Toute architecture future qui passe par μ_HR comme seul vecteur d'info DAG→prédiction reproduira le plateau B₁ rank-5.

2. **Stage 1 frozen est une contrainte de design, pas un acquis.** Si Stage 1 est figé après une récupération causale parfaite, Stage 2 doit avoir une **voie expressive hors-DAG** pour combler les ~20% de signal manquant.

3. **Le conditioning par concaténation channel-wise est piégé.** Toute variante (μ_HR direct, μ_HR + baseline, AdaGN à chaque layer, FiLM, cross-attention) doit être empiriquement validée AVANT de scaler.

4. **L'auxiliary residual head r_φ(H) (Math Prof) est la dette critique non-implémentée.** C'est la seule recommandation théorique solide qui n'a jamais été testée.

5. **La diffusion EDM marche** mais pas comme architecture principale dans un cadre causal — le couplage causal/diffusion via μ_HR conditioning est problématique.

6. **Les architectures alternatives à explorer** (cGAN, ResDiff, Flow Matching, Normalizing Flows, VAE+DAG prior, hybrid GAN+diff, latent diffusion) doivent toutes répondre à 4 questions :
   - Comment le DAG entre dans la prédiction ?
   - Y a-t-il une voie d'expression hors-DAG ?
   - Le sampler est-il compatible avec une loss multi-objectif (causale + skill) ?
   - Le coût compute est-il < 30h sur Colab A100 ?

7. **Council BEFORE compute, threshold calibration AFTER training data.** L'ordre inversé coûte cher.

8. **Métriques figées avant run, OOD obligatoire, 3 seeds minimum réservés.** Sans ces 3 conditions, aucun résultat n'est publication-quality.
