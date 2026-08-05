# Triage du rapport d'évaluation — Oracle (V5)

> Généré en réponse au rapport d'évaluation du mémoire. **Oracle = V5** (checkpoint
> `ckpt_v2_corrdiff_normal`). Le « 9 nœuds » = `oracle_9node/seed_42`.
> Classe chaque critique : **TEXTE** (correction LaTeX seule) / **DONNÉES EXISTANTES**
> (déjà sur disque, aucun run) / **RUN** (inférence courte requise).

## Verdict global
Sur les 17 points, **aucun ne requiert de ré-entraînement**. Deux points seulement
justifient un run d'**inférence** (~30 min A100), tous deux couverts par le notebook
`path_c_plus/scripts/st_cdgm_review_ablation_qint.ipynb`. Le reste est du texte ou des
données déjà produites.

## Fautes de fond (§2)

| # | Point | Classe | Détail / source |
|---|---|---|---|
| 2.1 | OOD → inter-GCM historique | **TEXTE** | Vérifié : `results/v5_evaluation/phase7_ood_aligned.json` — EC-Earth3/NorESM2 sont `in_distribution:false` mais **régime historique**, pas SSP. Les CRPS (0,249 / 0,319 / 0,342) matchent le Tableau VIII. Reformuler résumé/abstract. |
| 2.2 | Ablation two-stage sans DAG | **RUN** | Notebook, Cellule 7. 4 variantes (`normal`/`random`/`zero`/`noncausal`) sur même sous-ensemble ID. Isole two-stage vs topologie fine. |
| 2.3 | Variante 9 nœuds non documentée | **DONNÉES EXISTANTES** | `results/9node-seed42/` : matrice `A_dag_final` (9×9) déjà sur disque. Généré : `annex_dag_heatmap.png` + `annex_9node_summary.json` (Q_phys=0,9997 reproduit, 10/10 arêtes, n_extra=0). **Prêt pour l'annexe.** |
| 2.4 | Un seul seed | **TEXTE** | 3–5 seeds = ré-entraînement (hors budget 40 min). Écrire la limite explicitement à côté des tableaux. |
| 2.5 | K différent entre tableaux | **TEXTE** | K=64 (déterministe) / K=12 (probabiliste) documentés dans les JSON. Ajouter note sous tableau. |
| 2.6 | Q_int portée trop étroite | **RUN** | Notebook, Cellule 9. Passe à 6 perturbations (signes opposés + échelle). |
| 2.7 | DEM absent = comparaison neutre extrêmes | **TEXTE** | Reformuler : comparaison « à information d'entrée égale et déficitaire ». |
| 2.8 | Points mineurs (trilemme, réf [19], jours secs) | **TEXTE** | Renvoi DAGMA [19]→[1] ; préciser LR vs HR pour jours secs. |

## Fautes de forme (§3) et langue (§4)
**Toutes TEXTE** (LaTeX) : sommaire chap. IV désaligné, figures 12/13/14 à refaire,
liste d'acronymes, notations σ_data/A_dag uniformes, « 40 % » espace insécable,
guillemets français, néologismes à définir. Aucune donnée manquante.

## Ce qui a été produit (aucun run)
- `results/9node-seed42/annex_dag_heatmap.png` — heatmap A_dag 9×9, arêtes physiques encadrées.
- `results/9node-seed42/annex_9node_summary.json` — Q_phys recalculés + table arête-par-arête + métriques + protocole (K9 split, hyperparams, 215 époques).

## Ce qui nécessite un run (notebook, ~30 min A100)
`path_c_plus/scripts/st_cdgm_review_ablation_qint.ipynb` :
1. **Ablation DAG 4-way** (§2.2) → `review_ablation_dag.json` + `.png`.
2. **Q_int élargi 6 perturbations** (§2.6) → `review_qint_extended.json` + `.png`.

Bootstrap = verbatim du notebook d'éval (mêmes checkpoints, mêmes helpers probabilistes
→ chiffres directement comparables au mémoire).
