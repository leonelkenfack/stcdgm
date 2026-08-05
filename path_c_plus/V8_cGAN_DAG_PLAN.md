# V8 — cGAN résiduel end-to-end + DAG-régularisateur physique (DAG gelé)

> **Statut : BROUILLON DE TRAVAIL.** Aucun run. Base à soumettre au conseil des 5 experts *avant* exécution. Probabilités/budgets = estimations à challenger.
>
> Hypothèse : « on garde le cGAN ». Branche GAN du duel encore ouvert *diffusion-WassDiff vs GAN-intensité*.
>
> **Aligné sur le code de référence** : `downscaling/src/gan.py` (Rampal 2025, JAMES) — architecture résiduelle, **entraînement end-to-end (UN seul stage)**, contrainte d'intensité sur la sortie. **Corrige** ma v1 qui importait à tort la structure figée 2-stages de CorrDiff (diffusion).

---

## 0. Faits qui contraignent le design (non négociables)

1. **La régression V6′ n'était PAS la taxe causale.** Verdict conseil 2026-07-05 (7 experts) : c'était **Stage 1 coupé à 30 ép + décodeur restructuré**. La machinerie causale était *inerte* (A1=+0.6 %, A2=+0.05 %). → *Garder le UNet-moyenne fort ; ne pas bottlenecker.*
2. **DPI** : `I(HR; h(LR)|LR)=0` → une représentation déterministe ne peut qu'**égaler** CorrDiff in-dist. → *Full-LR alimente toujours le générateur.*
3. **F1@p99-sur-moyenne est aveugle au générateur stochastique** (`eval:215 .mean(dim=0)`). → *Juger extrêmes sur CRPS / Rx1day / return levels / RAPSD, PAS F1-moyenne.*
4. **Tail-weighting diffusion saturé** (weight_p99=25) → Rx1day −21 %. → *Contrainte d'intensité sur la SORTIE (ce que le code Rampal fait : MSE sur `reduce_max`+`reduce_mean`).*
5. **Preuve domaine (NZ 2025)** : cGAN-intensité = biais **0.0 %** sur signal-CC extrêmes ; diffusion **−5.3 %**. → *argument #1 du GAN.*
6. **Tension fondamentale irréductible (verdict V7)** : on ne peut PAS avoir « A_dag identifié/load-bearing » ET « l'ablater ne coûte rien ». Le **prior-lock qui gèle A_dag est CE qui achète l'OOD** (×5.83 contraint vs ×13.84 libre). → **Le DAG gelé est une FEATURE OOD, pas un bug.**
7. **Historique collapses tail-weighting** (Phase 8). → *oversampling p95+, pas reweighting brutal ; garde-fous.*

**Principe directeur (séparation des rôles) :**
- **Skill/moyenne** → UNet régression, co-entraîné (hérité CorrDiff/Rampal).
- **Extrêmes/stochastique** → générateur résiduel + `L_intensity` sur la sortie.
- **Physique/OOD/interprétabilité** → **DAG standalone gelé** (Stage 0), conditionne la branche résiduelle *hors du chemin de skill*.

**Cible pré-enregistrée : PARITÉ ID (Δ dans le bruit inter-seed ~1e-3), PAS amélioration.** Gain visé = ~2–8 % de coût ID assumé contre ~60 % de gain OOD prouvé. « Battre le F1-ID par la causalité » = **mort-né, abandonné**.

---

## 1. Architecture (résiduelle, end-to-end)

```
Stage 0 (séparé, CPU) ─────────────────────────────────────────────
   DAG standalone : fit Â*, K*  →  GELÉS      (prior structurel auditable)

Entraînement principal — UN SEUL train_step (façon gan.py:241) ─────
        LR complet (22 ch) + orographie HR + statiques
                          │
        ┌─────────────────┼──────────────────────┐
        ▼                 ▼                        │ Â* gelé
  ┌───────────┐   ┌───────────────┐                │
  │ UNet μ    │   │ Générateur G   │◄── z ~ N(0,I) │ branche causale :
  │  → μ_HR   │   │  → résidu r    │                │ +γ·Σ_u Â*[u,v]·(K*_{u→v}*x_u)
  └─────┬─────┘   └───────┬────────┘◄───────────────┘  (conditionne G/features)
        │  x̂ = μ_HR + r   │
        └────────┬────────┘
                 ▼
     Discriminateur D(r | μ_HR, cond)  →  L_adv (WGAN-GP)
                 +  L_content(r)  +  L_intensity(x̂)
```

Les trois (UNet, G, D) sont mis à jour **dans le même train_step** (co-entraînement), exactement comme `gan.py:256-283`. Flag `train_unet` permet de geler le UNet en option (ablation). Le résidu `r_gt = HR − μ_HR` est recalculé à la volée depuis le UNet courant.

### 1.1 UNet régression (μ_HR)
- Réutilise `RegressionMeanPredictor`. Entrée full-LR (DPI-safe). **Arête orographique HR au décodeur** `w_orog=(U,V)·∇h_HR` (scaffold DUR, jamais annelé — verdict M3).
- ⚠️ **Ne pas re-régresser V6′** : budget d'entraînement plein (pas 30 ép tronquées). Le piège n°1 du verdict = « budget, pas archi ».

### 1.2 Générateur résiduel G
- Entrée `[μ_HR, cond, z]`, sortie r. `x̂ = μ_HR + r`. ResGAN (Rampal).
- Injection z multi-échelle (anti noise-collapse).

### 1.3 Discriminateur D
- Conditionnel `D(r | μ_HR, cond)`, WGAN-GP (gradient penalty, `gan.py:83`). PatchGAN / multi-échelle.

### 1.4 DAG (Stage 0, gelé) — conditionnement option B sur champs bruts
- `Â*[u,v]` pré-entraîné standalone (§3), **gelé**. Conditionne via **branche résiduelle sur les champs d'entrée** (portable, indépendant de l'archi UNet) :
  `H' = backbone(full-LR) + γ·Σ_u Â*[u,v]·(K*_{u→v} * x_u)`, γ annelé 0→γ_max.
- `‖K*_{u→v}‖₁=1`. Backbone full-LR toujours présent → **DPI-safe**.
- **Gelé = assumé une feature OOD** (verdict #6), pas un défaut. On NE prétend PAS le découvrir.

---

## 2. Pertes (toutes dans le même step)

| Perte | Rôle | Agit sur | Réf code |
|-------|------|----------|----------|
| `L_adv` | réalisme | r / x̂ | WGAN critic `gan.py:37-44` |
| `L_content` | ancrage | r (MSE résidu) | `gan.py:182` |
| **`L_intensity`** | **extrêmes** | **x̂ = μ+r** | MSE sur `reduce_max`+`reduce_mean` `gan.py:464-480` |
| `L_gp` | Lipschitz D | — | `gan.py:83` |
| `L_spectral` (opt) | RAPSD | x̂ | hérité |

- `L_intensity` validée par le code de référence : `|max(HR) − max(x̂)|²  +  |mean(HR) − mean(x̂)|²`, pondérée `intensity_weight`. **C'est la brique que la diffusion ne fait pas naturellement.**
- Oversampling p95+ (PAS reweighting brutal — collapse Phase 8).

---

## 3. Stage 0 — DAG standalone (CPU, AVANT tout GPU)

Fit d'un module structurel autoportant, indépendant de la loss downscaling :

```
L_fit(Â) =  ‖ x_v − Σ_{u∈pa(v)} Â[u,v]·(K_{u→v}*x_u) ‖²        (reconstruction structurelle)
          + λ_sparse‖Â‖₁ + λ_acy·h(Â)   (NOTEARS h=tr(e^{A∘A})−n)
          + λ_prior‖Â − G_phys‖²   (FORT ; ancre la physique)
```

**Gate CPU AVANT de conditionner** (évite de conditionner sur du bruit) :
- cosinus inter-seed des déviations `Δ=Â−αG_phys ≥ 0.5` (identifiabilité, pas bruit de seed),
- acyclicité `h(Â)≈0`, convergence `‖ΔÂ‖` sous seuil,
- **E3 corruption-prior** OK.
- Si gate échoue → `Â = prior pur`, lecture seule, **on ne conditionne pas** (skill V5 préservée).

⚠️ **Biais mono-GCM (Rosenfeld)** : `L_fit` sur ACCESS-CM2 seul encode le biais ACCESS → `λ_prior` fort maintenu. Claim = *« physique imposée, cohérente-data »*, jamais *« découverte »*.

---

## 4. Protocole d'évaluation PRÉ-ENREGISTRÉ — **OOD = gate NO-GO**

> Verdict V7 : *« pas de protocole OOD → NO-GO, sinon V7 est infalsifiable. »*

- **Cible ID = PARITÉ** avec V5 (Δ dans le bruit inter-seed). Primaire = **CRPS / FSS / spectral / extrêmes-par-membre**. F1-sur-moyenne = garde-fou aveugle, PAS cible.
- **OOD (co-primaire, bloquant)** : **train-froid/test-chaud** (pseudo-réalité 140 ans CCAM, standard NIWA) **+ GCM hold-out** (EC-Earth3/NorESM2). Ordre pré-déclaré : `dégradation(V8) < dégradation(CorrDiff)`.
- **Baseline équitable** : CorrDiff reçoit les **mêmes 22 canaux** (piège du 3-way confondu).
- **E1** : ablation branche causale (γ=0) → `skill(γ=0) ≈ skill(γ>0)` attendu (DAG = feature OOD, pas béquille skill).
- **E3** : corruption prior (déjà en gate Stage 0).
- **E5** : interventions par arête, signatures physiques pré-déclarées (le vrai argument causal honnête).
- **Validation d'identification sur SCM synthétique à arêtes connues** (exigé par 2 experts, absent des plans précédents).
- Métriques : toutes dans `eval_metrics_dual_convention.py` (CRPS fair, SSR, Rx1day/Rx5day bias, qbias p99/p99.9, RAPSD, FSS@p99, GEV return levels, **skill stratifié easterly/ex-TC** = trou ouvert littérature NZ).

---

## 5. MVP obligatoire AVANT full run — gates de découplage (verdict M-a)

> *« Ne jamais bundler décodeur+M2+annealing+budget dans un run = re-confondre comme V6′. »*

- **A** = V5/CorrDiff ckpt (coût 0, référence).
- **B** = UNet μ + G résiduel **sans DAG, sans L_intensity**, budget **plein** (≥ budget V5). **Gate dur : si B n'atteint pas la parité RMSE/Pearson V5 → STOP** (l'archi GAN de base est cassée ou budget insuffisant).
- **C** = B + `L_intensity`. **Question décisive : ferme-t-il le Rx1day −21 % ?** C'est le seul but du GAN.
- **D** = C + DAG gelé (γ ramp) — **seulement si C réussit**. Mesurer `skill(γ)` ET `cosinus inter-seed(Â)`.
- **Micro-ablation conditioning (désaccord non tranché)** : G conditionné sur **22 LR bruts** vs **canaux filtrés-DAG**. Le Climat montre que les canaux bruts *spurious* dégradent l'OOD ×7.81 ; ML/Recherche veulent le spread riche. **Trancher par OOD, pas par principe.**

**Gate full run V8** : `C : Rx1day_bias ∈ [−8 %,+5 %]` ET `RMSE(C) ≤ RMSE(V5)·(1+ε)` (ε≈0.08, coût ID assumé).

---

## 6. Risques & mitigations

| Risque | Prob | Mitigation |
|--------|------|------------|
| Instabilité GAN / mode collapse | Élevée | EMA, TTUR, R1/GP, warmup-G, monitoring variance inter-membres |
| Re-régression V6′ (budget tronqué) | **Élevée** | budget plein obligatoire ; gate B ; ne pas bundler |
| `L_intensity` → artefacts | Moyenne | quantile-soft + max ; `L_spectral` |
| Â périmé (gelé) | Faible | opère sur champs bruts (moins sensible dérive) ; warm-start lent optionnel |
| Biais mono-GCM | Connue | λ_prior fort ; claim honnête |
| Collapse tail (Phase 8) | Connue | oversampling ≠ reweighting ; floor |

---

## 7. Budget compute (estimation)
- Stage 0 DAG : **CPU**, quelques h (colle au hardware).
- MVP A/B/C/D : ~8–12 h A100.
- Full end-to-end : ~15–25 h (co-entraînement UNet+G+D).
- 2e seed CorrDiff (ensemble poolé par membres) : ~35 h, indépendant.

---

## 8. Narratif honnête
- ✅ « GAN + contrainte d'intensité sur la SORTIE → corrige le biais de queue que la diffusion sous-estime (NZ 2025) ».
- ✅ « DAG = physique IMPOSÉE, gelée, auditable et intervenable (E5) — **feature OOD assumée, pas découverte causale** ».
- ✅ « Préserve la réponse-CC des extrêmes en OOD là où la diffusion dégrade » — **ssi E4/OOD le montre**.
- ✅ Cible = **parité ID + gain OOD**, coût ID ~2–8 % assumé.
- ❌ JAMAIS : DAG bat CorrDiff in-dist ; A_dag data-driven/découvert ; causalité = source de skill.

---

## 9. Questions ouvertes pour le conseil
1. `L_intensity` : max+mean (code Rampal) suffit-il, ou ajouter quantile-matching p99.9 / EVT-GPD ?
2. Conditioning de G : **22 LR bruts vs canaux filtrés-DAG** → micro-ablation OOD (désaccord non tranché).
3. `train_unet` : co-entraîner (défaut Rampal) vs geler μ après warmup ?
4. Discriminateur mono vs multi-échelle ; conditionnement complet vs sous-ensemble.
5. **Branche GAN (ce plan) vs branche diffusion-WassDiff** — comparer par sims coordonnées avant de committer ~20 h.
```
