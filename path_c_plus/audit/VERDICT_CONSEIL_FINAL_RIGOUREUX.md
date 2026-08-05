# Verdict consolidé du conseil — rigoureux, sans hand-waving
Date : 2026-06-17

Ce document remplace `CONSEIL_VERDICT_FINAL.md` qui contenait des probabilités inventées.

## Mandat du conseil

Trois agents avec mandat strict :
1. **Math** : démontrer formellement quels mécanismes échappent au paradoxe bypass + copy-mode + plateau B₁ rank-r sous loss MSE
2. **Code-archéologue** : cataloguer EXHAUSTIVEMENT toutes les architectures testées dans le repo + leur résultat empirique
3. **ClimatML littérature** : trouver tous les papers 2023-2026 sur DAG/causal + downscaling précipitation avec **résultats chiffrés publiés**

Aucun agent n'avait droit aux probabilités inventées. Citations arXiv obligatoires.

---

## Verdict math (agent `a5bccf56`)

**Formalisation rigoureuse du paradoxe** :
- Bypass = `∃ décomposition f_θ = h_θ(skip channels) + ε(A)` avec `∂ε/∂A = 0`
- Copy-mode = `D_θ(x, σ→0) → μ_HR` asymptotiquement
- Plateau B₁ rank-r = `Var(HR_pred) ≤ Var(baseline) + rank(A)·σ²` borné par construction

**Verdict sur 5 mécanismes candidats** :

| Mécanisme | Résiste bypass ? | Évite copy-mode ? | Plateau B₁ ? |
|---|---|---|---|
| DPS / energy guidance (M6) | partiel (sampler-only) | NON | persiste |
| Hard projection P_A · g_θ | OUI par construction | **NON — force le copy-mode** | persiste (strictement pire que v2) |
| Régularisation gradient implicite | local seulement (Goodhart) | NON | inchangé |
| Two-path TPH-DAG (M7) | **NON — α-collapse EST le bypass** | hérite du copy-mode de HR_direct | converge vers Var(HR_direct) |
| r_φ(H) auxiliary head | NON (DPI: r_φ ne peut créer info > H) | non adressé | inchangé |

**Lemme transversal** : sous loss MSE, le plateau B₁ rank-r est **structurel**. Aucun des mécanismes 1-5 ne peut le lever sans casser (C2) ou la loss MSE elle-même.

**Verdict final math** :
> *"The bypass/copy-mode duality under MSE is, to my knowledge, an open problem for generative downscaling — no published architecture proves an escape."*

**Inconnues explicites (3 directions non-analysées par le math)** :
1. Projection non-linéaire `P_A(x) = A · NL(Aᵀx)` — pas de verdict closed-form
2. DPS avec énergie non-quadratique (OT entropy-reg vers span(A)) — DPS Thm 1 ne s'applique que pour log-concave
3. Loss non-MSE (score-matching + rank-promoting reg sur covariance) — aucun travail théorique connu

---

## Verdict code-archéologue (agent `a7e48286`)

**17 variants distincts testés depuis mars 2026** documentés avec commit SHA + métriques.

**Le verdict empirique brut** :

| Variant | Pearson | F1@p99 | Verdict |
|---|---|---|---|
| Noncausal vanilla CorrDiff v4 | **0.819** | **0.550** | **MEILLEUR SUR F1@p99** |
| V5-mini (causal + DAG + Bundle B) | 0.825 | 0.512 | +0.6% Pearson = bruit ; F1@p99 −7.4% |
| Seed42 v1 (causal_concat) | 0.762 | 0.447 | copy-mode confirmé μ_HR_ablation=102% |
| Seed42 v2 (Mardani fix) | 0.740 | non rapporté | plateau B₁ rank-5 confirmé |
| Sprint 4 (cross-attn DAG) | < 0.1 | 0.023 | bypass total |

**Le fait critique caché dans V5-mini** :

V5-mini contient déjà un `ConditionalSkipBlock A1` = skip-connection gate `α(LR)·CRM + (1-α)·direct`. **C'est exactement la topologie M7 TPH-DAG.** Diagnostic documenté : *"shorts extremes unintentionally"* — α apprend à shorter Path causal sur les extrêmes, ce qui dégrade F1@p99.

**Cela confirme empiriquement la prédiction math** : α-collapse = bypass = perte de skill sur extrêmes.

**Mécanismes NON testés dans le repo (catalogue exhaustif)** :
- Auxiliary residual head r_φ(H) (dette F4)
- Mardani fix + r_φ
- Normalizing flows
- Flow matching
- Latent diffusion (éliminé par taille grille)
- VAE + DAG prior
- Hard projection sur span(A_dag) (éliminé par math)
- INR / coordinate MLP
- cGAN hybrides
- Patch-based diffusion
- Focal loss F1
- CFG complet (jamais activé)

---

## Verdict ClimatML littérature (agent `a275937b`)

**Aucun paper publié 2023-2026 ne rapporte simultanément** :
- DAG/A_dag binaire injecté dans downscaling précipitation génératif
- Métriques RMSE + Pearson + CSI@p99 + RAPSD + SSR
- Delta favorable vs noncausal comparable

**Les 4 papers "causal" les plus cités se décomposent à l'inspection** :

| Paper | Mécanisme "causal" réel | Résultat |
|---|---|---|
| DTCA (arXiv 2410.13314) | masque temporel autoregressif (PAS de DAG) | +15% CSI heavy sur Swedish radar |
| RainSeer (arXiv 2510.02414) | masque temporel + radar reflectivity (PAS de DAG appris) | NSE 0.934 sur RAIN-F mais ablation : retirer "causal attention" coûte 0.004 RMSE (= ~rien) |
| CaPaint (NeurIPS 2024, arXiv 2409.19608) | **front-door adjustment** : inpainting des patches environnementaux uniquement | **+6.1% MAE +7.7% MSE sur SEVIR** (modeste mais réel) |
| NuwaDynamics (ICLR 2024) | back-door adjustment via augmentation | +4-22% MAE/MSE |

**Verdict ClimatML final** :
> *"The published literature does NOT contain an empirically-validated DAG-injection architecture for precipitation downscaling that simultaneously reports RMSE, Pearson, F1@p99, RAPSD, and SSR with deltas over a comparable non-causal baseline."*

**Une seule mécanique a un précédent empirique défendable** : front-door inpainting (CaPaint). Mais sur SEVIR (pas ACCESS-CM2), gains modestes (~6%), aucune métrique d'extrême rapportée.

---

## Convergence des 3 verdicts

| Question | Math | Code-archéo | ClimatML |
|---|---|---|---|
| Existe-t-il mécanisme qui échappe au paradoxe ? | NON sous MSE | NON empiriquement (17 variants) | NON publié |
| M7 TPH-DAG (two-path gate) ? | bypass formel | déjà testé (ConditionalSkipBlock A1) → shorts extremes | aucun précédent publié |
| DAG bat noncausal sur F1@p99 ? | Non démontrable | NON (V5-mini 0.512 < noncausal 0.550) | NON rapporté nulle part |
| Existe-t-il une voie ouverte ? | 3 inconnues théoriques | r_φ + 9 autres jamais testés | CaPaint front-door + Causal-Adapter |

**Conclusion convergente** : la combinaison "DAG causal + downscaling génératif + battre noncausal sur F1@p99" est :
- **Mathématiquement non démontrée** (open problem sous MSE)
- **Empiriquement non réalisée** (0/17 variants chez toi, 0/N publications)
- **Littérature absente** (positive)

Ton résultat négatif est cohérent avec le silence de la littérature.

---

## Options honnêtes (sans probabilité inventée)

### Option A — Accepter le verdict, arrêter

Q_phys = 0.9998 est réel et défendable comme **résultat d'interprétabilité**. Mais comme **résultat de skill**, le DAG ne bat pas noncausal et probablement ne le battra pas avec les mécanismes connus.

- Compute : 0h
- Risque : 0
- Coût opportunité : accepter que l'objectif "skill amélioré par causalité" n'a pas de solution actuelle

### Option B — Pari recherche front-door inpainting (CaPaint-style)

Le seul mécanisme avec **précédent empirique de gain** (CaPaint +6.1% MAE sur SEVIR) et **résistance au bypass démontrée** (math : front-door évite la formulation MSE-bypass).

- Compute : 30-50h (estimation, jamais testé sur ACCESS-CM2)
- Risque : élevé (transfer de SEVIR à ACCESS-CM2 non garanti, métriques d'extrême non rapportées dans CaPaint)
- Bénéfice attendu (cité CaPaint) : +6-8% MAE sur métriques de centre, INCONNU sur F1@p99
- Pas de probabilité chiffrée fournie par le conseil

### Option C — Pari recherche Causal-Adapter pattern

Frozen backbone + contrastive token loss conçu pour empêcher le bypass. Démontré sur T2I (ADNI MRI, Pendulum), **jamais testé sur précipitation**.

- Compute : 25-40h
- Risque : très élevé (transfer de domaine T2I → climat non documenté)
- Bénéfice : inconnu
- Pas de probabilité chiffrée fournie

### Option D — Abandon MSE, pari recherche fondamentale

Score-matching avec régularisateur rank-promoting + loss non-quadratique. Aucun travail théorique publié. Pure exploration.

- Compute : indéterminé
- Risque : maximal
- Bénéfice : potentiellement majeur si succès, mais sans aucun précédent

### Option E — Implémenter r_φ(H) auxiliary head (dette F4)

Le seul mécanisme dans la liste des "jamais testés chez toi" qui correspond à une **dette Math Prof documentée**. **Mais** : le math agent dit que r_φ ne peut PAS créer d'info nouvelle au-delà de H (data-processing inequality).

- Compute : 25-30h
- Math agent verdict : "by DPI cannot create information; at best decorative"
- Justification de tenter quand même : empiriquement non testé, le DPI argument suppose `I(H; μ_HR) ≈ H(μ_HR)` qui n'est qu'une borne supérieure
- Mais 0 garantie de succès

---

## Ma recommandation honnête

Je ne peux pas te donner de probabilité chiffrée. Mes estimations précédentes (50%, 70%, 80%) étaient inventées.

**Ce que les évidences disent** :
- Aucune voie n'a de garantie math + empirique + littéraire de succès
- Toutes les voies non-encore-testées sont des paris

**Si tu veux un pari rationnel sur les évidences disponibles** : Option B (CaPaint front-door) — c'est la SEULE qui a (a) un précédent empirique de gain (modeste mais réel) et (b) une explication math de pourquoi ça évite le bypass.

**Si tu veux le risque minimal** : Option A. Accepter l'évidence convergente que ce problème n'a pas de solution connue.

**Si tu veux explorer une dette documentée chez toi** : Option E (r_φ aux head). Pas de garantie math, mais c'est la dette F4 jamais réglée.

C'est à toi de décider. Je ne te ferai pas perdre plus de temps avec des proba hand-wavées.

---

## Documents annexes
- `path_c_plus/audit/ECHECS_ET_LECONS.md` — 47 échecs initiaux
- Verdict math complet : agent `a5bccf56` output
- Catalogue 17 variants : agent `a7e48286` output
- Revue littérature 2023-2026 : agent `a275937b` output
