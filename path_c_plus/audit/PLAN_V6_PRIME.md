# PLAN V6′ (V6-prime) — Pivot conditioning full-LR + fixes physiques

**Date :** 2026-06-30
**Remplace :** `PLAN_V6_BOOST_UNET.md` (V6 MVP), rejeté par l'audit indépendant du 2026-06-30
**Base inchangée :** protocole 9-node seed 42 (`st_cdgm_path_c_option_c_9node.ipynb`), Stage 1 intact
**Cible :** battre `noncausal_v4` — métrique co-primaire per-gridpoint ETCCDI (voir §5)

---

## 0) Pourquoi V6 a été rejeté (résumé de l'audit indépendant 5 experts)

Verdicts : ML REJETTE (7%), Math REJETTE (10-15%), IA REWORK (15%), Recherche 25-30%, Climat 35%.
Le ~57% des rondes précédentes était du wishful thinking — aucun mécanisme n'avait été simulé.

**Résultats numériques de l'audit (simulations sur le code réel) :**

| Constat | Preuve |
|---|---|
| Le "plateau rank-5" (F2/F6 d'ECHECS_ET_LECONS) est un théorème **faux** pour décodeurs non-linéaires | Toy diffusion EDM : conditioning rang-5 → rang de sortie 62-64/64, Var ratio 1.10. Le bruit multi-step brise le plateau de variance |
| Le vrai goulot est **informationnel**, pas de rang | Même toy : F1@p99 = 0.053 (conditioning pauvre) vs **0.639** (conditioning full-info) |
| `r_φ(H)` est mort-né | Affine-linéaire, sortie ∈ span{4 masques régionaux} → **rang spatial = 4** (SVD). R² ≤ 14.3% en oracle. Reproduit la pathologie qu'il devait guérir |
| Log-det cassé 3× | (1) appliqué à l'erreur `D_y − target` → récompense les GROSSES erreurs ; (2) K=384 > B=128 → 257 valeurs propres aveugles ; (3) `slogdet` crashe sous bf16 AMP |
| Pinball multi-τ inopérant | Tête unique → quantile effectif unique τ*=0.813 ; déplacement +0.009σ ; ratio gradient 1.25% |
| Bugs physiques preproc | θ_w faux de +14 K ; `u·|∇h|` perd le signe amont/aval ; `sftlf` en % → gradient ×100 ; mucape_proxy mal défini |
| Notebook inexécutable | `compute_g_phys` inexistant, `schedule_lambdas` mauvais module, imports cassés, TypeError Cell 11, contradiction gel-2k vs gate-epoch-5 |

---

## 1) Le changement structural V6′ : Stage 2 voit le LR complet

### 1.1 — Diagnostic racine

Le Stage 2 causal actuel (`causal_concat=True`) ne voit que :
```
UNet_in = [c_in·y_noisy, mu_HR, baseline_log]        # 3 canaux
```
- Le mu_HR causal vient du `GraphToGridDecoder` (cross-attention depuis ~8 tokens de graphe) — spatialement pauvre
- Le mu_HR noncausal vient d'un **UNet de régression complet sur les 15 champs LR** (`RegressionMeanPredictor`) — spatialement riche
- CorrDiff (Mardani), StormCast (NVIDIA) et Rampal 2025 (NIWA) conditionnent tous leur étage génératif **sur les champs LR bruts** — aucun leader ne s'impose le goulot actuel

Les 12 échecs Stage 2 historiques (A3-A12 d'ECHECS_ET_LECONS) sont des symptômes de ce choix.

### 1.2 — Le fix (~2h de code)

```
UNet_in = [c_in·y_noisy, mu_HR, baseline_log, LR_1..LR_21 upsamplés]   # 24 canaux
```

- `CausalDiffusionDecoder.__init__` : `unet_in_channels = in_channels + 2 + n_lr_channels` (nouveau param opt-in `lr_conditioning_channels: int = 0` ; défaut 0 = comportement actuel bit-identique)
- `forward_edm` / `sample` / `compute_loss_edm` : accepter `lr_fields: Optional[Tensor]` `[B, 21, H_HR, W_HR]`, concaténer si fourni
- **Cache BS32b** : stocker le LR du dernier pas de temps en résolution native `[N, 21, 23, 26]` (~700 MB fp32 pour 14k samples) et upsampler bilinéairement au batch-time → pas d'explosion mémoire
- mu_HR **reste dans le concat** → l'attribution causale par ablation (`mu_HR_ablation`) est préservée, et on peut mesurer précisément ce que la voie causale ajoute par-dessus le LR brut

### 1.3 — Ce que ça garantit

L'information du Stage 2 devient un **sur-ensemble strict** de celle du noncausal_v4 (qui atteint 0.550) ET de celle du V5_causal. Le modèle ne peut être structurellement en déficit d'information sur aucun des deux.

---

## 2) Stage 1 : conservé avec fixes physiques (audit Climat)

Ajouts conservés : nœuds `U850`, `V850` (`extended_v6_wind=True`) + features LR Climat. **Corrections obligatoires :**

| # | Fix | Détail |
|---|---|---|
| P1 | **Produit scalaire signé** | Remplacer `u850·|∇h|` par `(u850·∂h/∂x + v850·∂h/∂y)` — le signe distingue soulèvement côte Ouest vs subsidence foehn Est. V850 est disponible. Ajouter le facteur cos(lat) sur ∂/∂lon |
| P2 | **Fix sftlf** | Normaliser le masque land-sea : si max > 1.5, diviser par 100 (CMIP fournit des %) |
| P3 | **Proxy convectif correct** | Remplacer `mucape_proxy = θe850 − θe500` par `θe850 − θe*_sat500` (saturated equivalent potential temperature à 500) et renommer `conditional_instability` |
| P4 | **Dropper θ_w_850** | L'approximation est fausse de +14 K vs Davies-Jones exact. Dropper (ou implémenter Davies-Jones 2008 complet en V6′.1) |
| P5 | `w_700`, `T_850−T_500` | Information nouvelle nulle (combinaisons linéaires d'entrées présentes). **Conservés** car déjà codés et inoffensifs, mais dé-priorisés dans le narratif |

`λ_l1` ×0.7, `λ_dag_prior=0.40`, safeguards DAG : inchangés (comme V6).

---

## 3) Stage 2 : ce qu'on DROPPE (prouvé inopérant ou nuisible)

| Composant V6 | Sort | Raison (audit) |
|---|---|---|
| `r_φ(H)` StructuredResidualHead | **DROPPÉ** | Rang spatial 4 — ne peut pas influencer F1@p99 per-pixel. Le module et `cache_h_t_pooled` restent dans le repo (opt-in, inutilisés) |
| Log-det rank penalty | **DROPPÉ** | Appliqué à la mauvaise quantité, K>B casse l'estimateur, crash bf16. Le "plateau de rang" qu'il visait n'existe pas (théorème réfuté) |
| Pinball multi-τ tête unique | **DROPPÉ pour V6′.0** | Inopérant à λ=0.02 ; incohérent multi-τ sur sortie unique. Alternative propre (têtes quantiles séparées façon Q-SRDRN 2026) = V6′.1 si V6′.0 échoue sur les extrêmes |
| Warmup r_φ, monitoring r_φ, gates §3.3 | **DROPPÉS** | Sans objet (r_φ droppé). Le gate ‖r_φ‖/‖μ‖ était de toute façon trivialement satisfiable (faux PASS) |

**Le Stage 2 V6′ = diffusion EDM du 9-node seed 42, strictement identique, + 21 canaux LR dans le concat.** Un seul changement, mesurable par une seule ablation.

---

## 4) Fixes code P0 (audit IA — le notebook ne tournait pas)

1. Imports : ajouter `REPO_DIR/src` au `sys.path` (pattern du notebook 9-node)
2. `compute_g_phys` n'existe pas → écrire la fonction (extension du G_phys 9-node avec les arêtes U850/V850, dans `option_c_helpers.py`) ou l'inliner
3. `schedule_lambdas` : importer depuis `scripts/finetune_stage1_bundle_b.py` ou porter dans `option_c_helpers.py`
4. Cell 11 : signature `compute_rphi_attribution_pct` → sans objet (r_φ droppé) ; remplacer par l'ablation `mu_HR_ablation` + ablation canal LR
5. `existing_cache` : préserver la clé `lr_fields` au resume (même bug que `H_T_pooled`)
6. Compléter les placeholders `...` (data_loader, iterate_batches_fn, sample_v6) en portant depuis le notebook 9-node — **obligatoire avant tout run**
7. SMOKE : mesurer réellement (pas de valeurs codées en dur)

---

## 5) Métrique cible amendée (audit Recherche + Climat)

Le F1@p99 **pooled** est dominé par les pixels West Coast (gradient 400→8000 mm/an) — il mesure surtout la reproduction de la climatologie orographique. Le standard du domaine (ETCCDI, NIWA/Rampal 2025) est **per-gridpoint**.

**Amendement DÉCLARÉ du pré-enregistrement (avant tout run, anti-HARKing) :**

| Rang | Métrique | Seuils |
|---|---|---|
| **Co-primaire 1** | F1@p99 **per-gridpoint ETCCDI** (Convention A) | battre noncausal_v4 = 0.816 ; V5 fait déjà 0.841 → **ne pas régresser sous 0.841** |
| **Co-primaire 2** | F1@p99 **pooled** (Convention B) | PASS_MINIMAL 0.480 / PASS_TARGET 0.550 / PASS_STRONG 0.580 |
| Secondaires | CRPS, RAPSD, Rx1day_bias, FSS, Pearson, RMSE | non-régression vs V5 (RMSE ≤ +5%, Pearson ≥ −3%) |
| OOD | EC-Earth3 F1@p99 per-gridpoint + pooled | pas d'effondrement (pooled ≥ 0.40) |

Nouveau fichier : `V6_PRIME_seuils_preregistered.json` (commit + hash avant run).

---

## 6) Ablations pré-enregistrées (attribution propre)

Un seul changement Stage 2 → attribution simple :

1. **A1 — mu_HR ablation** (existant) : mu_HR→0 à l'inférence. Mesure ce que la voie causale ajoute par-dessus le LR brut. *C'est LE test du narratif causal.*
2. **A2 — LR-channels ablation** : lr_fields→0 à l'inférence. Mesure ce que le LR brut ajoute par-dessus mu_HR (attendu : beaucoup).
3. **A3 — Stage 1 features** : comparaison V6′ (21 vars, 8 nœuds) vs re-run 15 vars/6 nœuds si budget (sinon reporté).

Si A1 montre ≈ 0 : conclusion honnête "le DAG contribue à l'interprétabilité (Q_phys, do-tests) mais pas à la skill" — publiable comme negative result rigoureux (GMD/AIES, cf. audit Recherche).

---

## 7) Budget compute & séquence

| Étape | Durée A100 |
|---|---|
| Fixes code P0 + preproc (P1-P4) | local, 0 GPU |
| Préproc V6′ features (4 GCMs si OOD) | 3-5h |
| Stage 1.A refait (21 vars, 8 nœuds, λ_l1 ×0.7) | ~15h |
| O3 gate + freeze + cache BS32b (+ lr_fields) | ~1h |
| SMOKE Stage 2 (24 canaux, mesures réelles) | 2h |
| Stage 2 FULL (~150 ep) | 8-10h |
| Eval in-distrib + ablations A1/A2 + OOD EC-Earth3 | ~3h |
| **Total** | **~32-36h** |

Kill-switch inchangé : si V6′ < V5 sur les DEUX co-primaires → stop, publier V5 + audit comme negative result.

---

## 8) Probabilités attendues (à re-estimer par l'équipe d'audit)

- Recherche (audit) : ~55-60% avec ce pivot
- Justification structurelle : le conditioning devient un sur-ensemble strict de celui du baseline qui atteint déjà 0.550 ; la même architecture (CorrDiff-style) est celle des leaders publiés
- Risque principal restant : le Stage 2 pourrait ignorer mu_HR une fois le LR disponible (copy du pattern noncausal) → c'est précisément ce que l'ablation A1 mesurera honnêtement

---

## 9) Ce que V6′ N'EST PAS

- Pas un abandon du causal : Stage 1 DAG + Q_phys + do(SST+2K) restent ; l'ablation A1 quantifie honnêtement leur contribution
- Pas une réécriture : UN changement structural Stage 2 + fixes de bugs
- Pas un changement de métrique caché : l'amendement per-gridpoint est déclaré et commité AVANT tout run

---

## 10) VALIDATION FINALE — 2ème passage équipe d'audit indépendante (2026-06-30)

**Verdict unanime : GO (avec modifs chirurgicales, toutes intégrées ci-dessous).**

| Expert | Verdict V6′ | P(battre noncausal sur ≥1 co-primaire) |
|---|---|---|
| ML | GO-AVEC-MODIFS | ~75% |
| Math | GO-AVEC-MODIFS | ~80% |
| Recherche | GO | ~75% |
| IA | CODE-FAISABLE (2 corrections) | ~55% |
| Climat | GO-AVEC-MODIFS | ~80% |

Médiane ~75% (vs 7-35% pour V6 rejeté). Décomposition consensuelle : per-gridpoint ≥ 0.841 (non-régression) ~70-80% ; pooled ≥ 0.550 ~35-55%.

### 10.1 — Modifs OBLIGATOIRES intégrées (consensus 4/5)

**M1 — Normalisation des 21 canaux LR (ML + Math + IA + Climat) :**
- z-score **par canal**, statistiques **figées sur le train du GCM d'entraînement** (ACCESS-CM2)
- **PAS de re-normalisation per-GCM à l'inférence OOD** (Climat : sinon on masque les biais moyens et on fausse le test OOD)
- Stats stockées dans le checkpoint ET le cache, **une seule clé de config train/éval** (leçon C3)
- Concat brut interdit : des champs en hPa/Pa·s à côté de log1p détruiraient le conditionnement (IA)
- L'ablation A2 "LR→0" = **zéro post-normalisation** (= moyenne du canal), sinon champ hors-distribution (Math)

**M2 — Patcher TOUS les sites de concat en un commit (IA) :**
≥5 sites construisent l'input UNet : `forward_edm` (L440-478), `sample` (L767), `_sample_edm_karras` (L979-1027), `_sample_dpm_solver` (L1219, L1258) + chemins contrastif et cond_dropout dans `train_epoch_stage2_cached`. Un commit, puis re-grep de vérification (règle callsites).

**M3 — Parité inférence (ML) :** `sample()` doit recevoir les vrais `lr_fields` — assert + sanity-eval 1 batch avant full run (leçon E4). `existing_cache` doit préserver `lr_fields` au resume.

**M4 — PAS de dropout de canaux LR pour forcer l'usage de mu_HR (Math) :** ce serait un biais méthodologique (A1 mesurerait un artefact d'entraînement, pas une contribution informationnelle).

**M5 — Interprétation A1 pré-déclarée (Math) :** attendu A1 ≈ 0-2%. mu_HR n'est non-redondant que par sa **mémoire temporelle** (le RCN voit toute la séquence T_seq ; le cache LR ne stocke que le dernier pas). À documenter dans le pré-enregistrement comme l'interprétation correcte — pas de sur-vente du narratif causal.

**M6 — Monitorer A2 dès le SMOKE (Recherche) :** vérifier que le denoiser n'ignore pas les 21 canaux dès le smoke 2h, pas seulement en éval finale. + Ablation A2 **par canal** (Climat : détecter les canaux-béquilles).

**M7 — Upsample batch-time obligatoire (IA) :** pré-computer HR = 36 GB (exclu). `F.interpolate` [B,21,23,26]→[B,21,172,179] ≈ quelques ms sur A100. `conv_in` diffusers accepte in_channels=24 sans autre changement (+~9k params).

### 10.2 — Risque n°1 identifié (ML) + décision

`delta_target = HR − baseline − mu_HR` force la diffusion à d'abord **annuler** l'erreur d'un mu_HR spatialement pauvre avant d'ajouter du détail — handicap que le noncausal n'a pas. 

**Décision V6′.0 :** cible résiduelle INCHANGÉE (principe "un seul changement attribuable"). L'alternative `delta = HR − baseline` (mu_HR conditioning seul) est **pré-enregistrée comme variante V6′.1** à déclencher si le SMOKE ou l'éval montre le symptôme (D_y anti-corrélé à mu_HR = la diffusion passe sa capacité à annuler mu_HR).

### 10.3 — Option quasi-gratuite (Climat)

Persistance AR : `IVT moyenné 72h` en feature LR additionnelle — si <1h de preproc, l'inclure en V6′.0 ; sinon V6′.1. Interpréter l'OOD NorESM2 avec prudence sans SST Tasman (V6′.1).

### 10.4 — Trade-off OOD assumé (Climat, avis franc)

La diffusion full-LR apprendra l'empreinte de biais d'ACCESS-CM2 que le goulot graphe filtrait de facto. Trois atténuants : (1) le noncausal full-info ne s'effondre pas en OOD dans nos propres chiffres ; (2) à 25→4 km le mapping est dominé par le forçage orographique quasi-déterministe ; (3) le gate OOD EC-Earth3 est pré-enregistré. **La valeur du causal est diagnostique (A1, do-tests), pas protectrice** — à assumer dans le narratif.
