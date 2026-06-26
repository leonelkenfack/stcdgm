# Plan figures du mémoire — V5 / Oracle

État au 2026-06-26. Inventaire complet → catégorisation A/B/C → actions concrètes.

---

## Légende

- **Catégorie A** = Figure de la littérature (autre article) → je donne source + numéro
- **Catégorie B** = Figure de NOTRE travail → notebook d'évaluation unique `_v5_figures_export.ipynb`
- **Catégorie C** = Schéma conceptuel TikZ / autre → instructions de production

---

## CHAPITRE 1 — Contexte et verrous

### F1.1 — Paradigme deux étages CorrDiff (origine)
- **Catégorie A** (littérature)
- **Source** : Mardani et al. (2024) *"Residual Corrective Diffusion Modeling for Km-Scale Atmospheric Downscaling"*, **Nature Communications Earth & Environment** 6:124 (2025) — arXiv:2309.15214
- **Figure à extraire** : Fig. 1 (Architecture overview du paradigme deux étages : regression + diffusion)
- **Emplacement mémoire** : chap 1, section "Le paradigme deux étages : CorrDiff"
- **État actuel** : la caption existe (`chapitre_1.tex:281`) mais sans inclusion graphique. Il faut télécharger la figure du papier.
- **Action user** : extraire Fig. 1 de l'article, sauvegarder dans `results/literature_figures/mardani_corrdiff_fig1.png`

### F1.2 — Trilemme du downscaling génératif
- **Catégorie C** (schéma conceptuel TikZ)
- **Emplacement mémoire** : chap 1, section "Verrous scientifiques et trilemme"
- **État actuel** : caption existe (`chapitre_1.tex:350`), TikZ déjà retiré pour économie. À recréer en TikZ propre.
- **Action production** : TikZ triangle stabilité ↔ extrêmes ↔ robustesse, avec positions cGAN / CorrDiff / Oracle / idéal
- **Contenu** : voir `chapitre_4.tex` (Fig. trilemme supprimée) — réintégrer simplifié

### F1.3 — Cartouche conceptuel Oracle
- **Catégorie C** (schéma TikZ simple)
- **Emplacement** : chap 1, fin de chapitre
- **Action** : TikZ rectangle "version causalisée de CorrDiff" (déjà décrit dans caption `chapitre_1.tex:360`)

---

## CHAPITRE 2 — État de l'art

### F2.1 — Frise chronologique downscaling profond
- **Catégorie C** (TikZ timeline)
- **Source** : à composer à partir des références de chap 2 (Vandal 2017 → CorrDiff 2024 → GenCast 2024 → Rampal 2024)
- **Emplacement** : chap 2, début
- **Action** : TikZ timeline horizontale, 5-6 dates clés

### F2.2 — Architecture CorrDiff détaillée
- **Catégorie A** (littérature)
- **Source** : Mardani et al. (2024) **Nature Comm. EE**
- **Figure à extraire** : Fig. 2 (UNet regression + EDM diffusion architecture)
- **Action user** : extraire Fig. 2, sauvegarder dans `results/literature_figures/mardani_corrdiff_fig2.png`

### F2.3 — Tableau confrontation approches × verrous
- **Catégorie C** (tableau LaTeX, déjà fait)
- **Emplacement** : chap 2, section synthèse comparative
- **État** : OK (caption `chapitre_2.tex:361`)

### F2.4 — Cartouche Oracle vs littérature
- **Catégorie C** (TikZ — déjà décrit)
- **État** : OK

---

## CHAPITRE 3 — Architecture Oracle

### F3.1 — Tableau notations principales
- **Catégorie C** (tableau LaTeX)
- **État** : OK

### F3.2 — Vue d'ensemble Oracle en 4 blocs
- **Catégorie C** (TikZ)
- **État** : OK (existant)

### F3.3 — Graphe atmosphérique hétérogène (4 types de nœuds)
- **Catégorie C** (TikZ)
- **État** : OK

### F3.4 — Tableau traçabilité composants
- **Catégorie C** (tableau LaTeX)
- **État** : OK

---

## CHAPITRE 4 — Expérimentation et résultats

### F4.1 — Pipeline architectural Oracle
- **Catégorie C** (TikZ existant, OK)
- **État** : OK (`chapitre_4.tex:199`)

### F4.2 ★ NOUVEAU — Prédiction journalière 4 jours (V5 causal vs noncausal)
- **Catégorie B** (notre travail, à produire)
- **Description** : grille 4×2 (4 jours en lignes, V5 et noncausal en colonnes) + 1 colonne pour la vérité ERA5 = grille 4×3 OU 4×4 avec biais
- **Emplacement** : chap 4, section "Performance en distribution" (après tableau résultats)
- **Notebook** : `_v5_figures_export.ipynb` → cellule "Daily 4-day comparison"
- **Données** : 4 jours d'événements intéressants (1 stratiforme, 1 convectif, 1 extrême, 1 jour sec) sur split test 2012-2013
- **Format** : cartes 172×179 NIWA-REMS NZ, échelle commune log(1+mm), même colorbar

### F4.3 ★ NOUVEAU — OOD : prédictions EC-Earth3 + NorESM2-MM
- **Catégorie B**
- **Description** : 2 lignes (EC-Earth3, NorESM2) × 3 colonnes (V5, noncausal, vérité)
- **Emplacement** : chap 4, section "Robustesse inter-GCM"
- **Notebook** : `_v5_figures_export.ipynb` → cellule "OOD predictions"
- **Données** : 1 événement par GCM sur la période historique

### F4.4 ★ NOUVEAU — OOD : distribution shift histogrammes
- **Catégorie B**
- **Description** : histogramme précipitations train ACCESS-CM2 vs eval EC-Earth3 vs NorESM2, échelle log-log, queue p99 zoomée
- **Emplacement** : chap 4, section "Robustesse inter-GCM"
- **Notebook** : `_v5_figures_export.ipynb` → cellule "OOD distribution shift"
- **Métrique** : Wasserstein-1 distance reportée dans le titre/caption

### F4.5 — Spectres RAPSD
- **Catégorie B** (existant : `phase8_figures/07_rapsd_comparison.png`)
- **État** : actuellement référence texte uniquement (j'ai retiré l'inclusion). À réinclure si on a la place.
- **Action notebook** : régénérer + nettoyer

### F4.6 — Diagramme de fiabilité
- **Catégorie B** (existant : `phase9_climate_standards/03_reliability.png`)
- **État** : actuellement référence texte uniquement
- **Action notebook** : régénérer

### F4.7 — Carte interventions q+20%, t+3K
- **Catégorie B** (existant : `phase8_figures/03_interventions_maps.png`)
- **État** : OK, inclus (`chapitre_4.tex:526`)

### F4.8 — Biais RX1day Oracle (+ baseline)
- **Catégorie B** (existants : `phase10_extreme_bias_maps/extreme_bias_ORACLE.png` + `_CorrDiff.png`)
- **État** : 1/2 inclus (Oracle), CorrDiff référencé en texte
- **Action** : éventuellement réinclure CorrDiff côte à côte

### F4.9 ★ NOUVEAU — DAG matrix Adag (heatmap)
- **Catégorie B** (existant : `phase8_figures/01_dag_oracle.png`)
- **État** : actuellement référence texte uniquement
- **Action notebook** : régénérer propre + à réinclure (centrale pour la causalité)

### F4.10 ★ NOUVEAU — Comparaison DAG appris vs DAG physique
- **Catégorie B** (existant : `phase11_causal_advanced/01_dag_physical_comparison.png`)
- **Emplacement** : chap 4, section "Diagnostics causaux"
- **Action notebook** : régénérer

### F4.11 ★ NOUVEAU — Integrated Gradients (explicabilité)
- **Catégorie B** (existant : `phase12_integrated_gradients/01_integrated_gradients.png`)
- **Emplacement** : chap 4, nouvelle sous-section "Explicabilité par integrated gradients"
- **Action notebook** : régénérer + caption explicative

### F4.12 — Sensibilités par variable (bar chart)
- **Catégorie B** (existant : `phase8_figures/04_sensitivity_per_variable.png`)
- **Emplacement** : annexe IV.B (sensibilités directes)
- **État** : référence texte uniquement
- **Action** : OK en référence

### F4.13 — Ablation A_dag (delta MSE)
- **Catégorie B** (existant : `phase8_figures/05_ablation_A_dag.png`)
- **Emplacement** : chap 4 (test O3)
- **Action** : pas critique, garder en référence si manque de place

### F4.14 — QQ plot tail
- **Catégorie B** (existant : `phase9_climate_standards/01_qq_tail.png`)
- **Emplacement** : chap 4 (calibration probabiliste OU annexe)

### F4.15 — Return period curves
- **Catégorie B** (existant : `phase9_climate_standards/02_return_period.png`)
- **Emplacement** : chap 4 (indices climatiques)

### F4.16 — FSS vs scale
- **Catégorie B** (existant : `phase9_climate_standards/04_fss_vs_scale.png`)
- **Emplacement** : chap 4 (performance en distribution)
- **État** : référencé en texte uniquement

### F4.17 — Histogramme des intensités log-log
- **Catégorie B** (existant : `phase8_figures/08_intensity_histogram.png`)
- **Emplacement** : annexe IV.E
- **État** : référencé en texte uniquement

### F4.18 ★ NOUVEAU — Comparaison spatiale Oracle vs CorrDiff (1 jour)
- **Catégorie B** (existant : `phase8_figures/06_spatial_comparison.png`)
- **Emplacement** : chap 4 (performance en distribution, complément F4.2)
- **Action notebook** : régénérer

---

## RÉCAPITULATIF — Actions à mener

### Catégorie A — Figures littérature à extraire (action USER)

| Réf | Article | Figure | Sauver dans |
|---|---|---|---|
| F1.1 | Mardani 2024 *Nature Comm. EE* 6:124 (arXiv:2309.15214) | Fig. 1 (paradigme 2 étages) | `results/literature_figures/mardani_corrdiff_fig1.png` |
| F2.2 | Mardani 2024 (idem) | Fig. 2 (architecture détaillée) | `results/literature_figures/mardani_corrdiff_fig2.png` |
| F2.1 (optionnel) | Rampal 2024 *GMD* 17:3873 (NowcastNet-NZ) | Fig. 1 (architecture) | `results/literature_figures/rampal_nowcastnet_fig1.png` |
| F2.1 (optionnel) | Harris 2022 *JAMES* 14:e2022MS003120 | Fig. d'archi cGAN | `results/literature_figures/harris_cgan_fig.png` |

### Catégorie B — Notebook unique d'évaluation `_v5_figures_export.ipynb` (action MOI/USER sur Colab)

À créer dans `path_c_plus/scripts/`. Cellules :

| Cell | Figure | PNG sortie |
|---|---|---|
| 1 | Bootstrap (Drive, git, deps) | — |
| 2 | Config + load V5 + load noncausal checkpoints | — |
| 3 | F4.2 prédictions 4 jours côte à côte V5/noncausal/ERA5 | `figures_export/F4_2_daily_4days.png` |
| 4 | F4.3 OOD prédictions EC-Earth3 + NorESM2 | `figures_export/F4_3_ood_predictions.png` |
| 5 | F4.4 OOD distribution shift histogrammes + Wasserstein | `figures_export/F4_4_ood_distribution_shift.png` |
| 6 | F4.5 RAPSD spectra | `figures_export/F4_5_rapsd.png` |
| 7 | F4.6 Reliability diagram | `figures_export/F4_6_reliability.png` |
| 8 | F4.9 DAG matrix heatmap | `figures_export/F4_9_dag_matrix.png` |
| 9 | F4.10 DAG appris vs DAG physique | `figures_export/F4_10_dag_vs_physical.png` |
| 10 | F4.11 Integrated Gradients | `figures_export/F4_11_integrated_gradients.png` |
| 11 | F4.18 Comparaison spatiale 1 jour | `figures_export/F4_18_spatial_compare.png` |
| 12 | Manifest JSON listant toutes les figures exportées | `figures_export/manifest.json` |

### Catégorie C — Schémas conceptuels TikZ (action MOI dans LaTeX)

| Réf | Schéma | Emplacement |
|---|---|---|
| F1.2 | Trilemme triangle (stabilité, extrêmes, robustesse) | chap 1 |
| F1.3 | Cartouche Oracle = causalised CorrDiff | chap 1 fin |
| F2.1 | Frise chronologique 2017→2024 | chap 2 début |

---

## Décision attendue de l'utilisateur

1. **Quelles figures de la catégorie A** vas-tu extraire toi-même (vs. me les fournir si tu as accès aux PDFs) ?
2. **Le notebook `_v5_figures_export.ipynb`** : je le crée maintenant comme squelette (cellules avec TODO d'évaluation), ou tu préfères que j'attende d'avoir les checkpoints disponibles ?
3. **Catégorie C** : je code les 3 TikZ maintenant (~30 min), ou plus tard ?
4. **Priorité** : si tu dois choisir 3 figures NOUVELLES essentielles, ce serait : F4.2 (prédiction 4j), F4.3+F4.4 (OOD), F4.11 (IG) ? Confirme.
