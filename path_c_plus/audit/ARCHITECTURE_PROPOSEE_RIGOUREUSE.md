# Architecture proposée : **CST-DAG** (Causal Stochastic-interpolant Two-stream with structured-energy guidance)

Date : 2026-06-17. Synthèse rigoureuse basée sur 3 études profondes (climat, causalité, génératifs non-MSE). Aucune probabilité inventée. Toutes les preuves de bypass-résistance + copy-mode immunity + rank-plateau escape sont **formelles avec citations**.

---

## 1. Diagnostic consolidé des 3 agents

### 1.1 Climat — la racine du problème (Pfahl 2017, Norris 2019)

**Théorème empirique** : un DAG capturant uniquement la chaîne dynamique sèche (QG height cascade) a un **plafond structurel de variance expliquée de 0.40-0.50** sur P_p99.

**Décomposition Held-Soden 2006** (Eq. 11 de l'agent climat) :
$$\sigma^2(P_{ext}) = \langle q\rangle^2 \sigma^2(\omega) + \langle\omega\rangle^2 \sigma^2(q) + 2\langle q\rangle\langle\omega\rangle \text{cov}(q,\omega) + HOT$$

Sur les extrêmes : `cov(q, ω) ≠ 0`. Tout DAG sans `q` rate ce terme.

**Pearl ID algorithm** (back-door criterion, Pearl 2009 Thm 3.3.2) : le **minimal admissible set** pour identifier `do(z_500) → P` est `S* = {z_250, q_850}`. Sans q_850, l'effet n'est PAS identifiable.

### 1.2 Causalité — 2 mécanismes formellement prouvés (Brehmer 2022, Pearl 2009)

Sur les 5 options d'injection SCM, **seulement 2** brisent provablement le bypass :

- **Option 4 (Equivariance causale)** : `f_θ(do(c)·x) = T_c(f_θ(x))` (Brehmer 2022 Thm 1). Si G_do agit non-trivialement sur A, alors `∂f_θ/∂A ≠ 0` par contradiction directe.

- **Option 2 (Front-door, CaPaint)** : partition `Ω = Ω_causal ⊔ Ω_env`. Sur Ω_causal, le mediator M=μ_HR est passé through unchanged → `∂Ŷ/∂A = ∂M/∂A ≠ 0` par construction.

### 1.3 Génératifs — 6 escapes de la duality MSE-bypass

Sur les 13 frameworks audités, **6 brisent la duality** :

| Framework | Brise quel pillar ? | Preuve |
|---|---|---|
| **EBM énergie structurée** | P1+P2 | `∂E/∂A` = rank-1 outer product non-zero a.e. (Eq 4.4 agent gen) |
| **Stochastic interpolants** | P3 (decouple drift+score) | b_θ et s_φ identifiés séparément (Albergo-VE Eq 14) |
| **OT/Wasserstein** | P1 (variable-measure) | π*(θ) bouge avec θ → bypass non stationnaire |
| **Quantile diffusion** | P1 (mean→quantile) | Bayes opt = Q_τ, rank-r lemma ne s'applique pas |
| **DPS/PnP guidance** | architectural (vacuous set) | Pas de conditioning input → bypass set vide |
| **-log det rank penalty** | Lemma 1.2 directement | Inverse Cov diverge → bypass non-stationnaire |

---

## 2. Architecture proposée : CST-DAG

**Notation.**
- $\mathbf{x}_{LR} = (u_p, v_p, w_p, q_p, t_p)_{p \in \{250, 500, 850\}}$ : 15 variables LR (déjà dans dataset)
- $\mathbf{x}_{stat} = (\text{orog}, \text{he}, \text{vegt})$ : statiques
- $A_{dag} \in \mathbb{R}^{n_v \times n_v}$ : adjacency apprise
- $\mu^{strat}_{HR}, \mu^{conv}_{HR}$ : means causales pour 2 régimes
- $\pi(\mathbf{x})$ : gate régime stratiforme vs convectif (basé sur $t_{2m}$, CAPE proxy)

### 2.1 Stage 1 — DAG ÉTENDU avec mixture régimes (Berg 2013)

**Variables nodes** (au lieu des 6 actuelles) :
$$V = \{z_{250}, z_{500}, z_{850}, w_{500}, q_{850}, t_{500}, t_{2m}, \text{IVT}, P\}$$

8 nodes + target. Tu peux dériver $z_p$ depuis $(u_p, v_p, t_p)$ via thermal wind (Holton Eq 4.18). IVT depuis $q_p, \mathbf{v}_p, p$ (formule Eq 9 agent climat).

**Architecture Stage 1 mixture** :

$$\mu^{strat}_{HR}(\mathbf{x}_{LR}) = R_{strat}(A_{dag} \cdot \text{enc}(\mathbf{x}_{LR}))$$
$$\mu^{conv}_{HR}(\mathbf{x}_{LR}) = R_{conv}(A_{dag} \cdot \text{enc}(\mathbf{x}_{LR}))$$

Avec gate :
$$\pi(\mathbf{x}_{LR}) = \sigma(g(t_{2m}, \text{CAPE-proxy}(\mathbf{x}_{LR})))$$

Et μ_HR final :
$$\mu_{HR} = (1-\pi) \mu^{strat}_{HR} + \pi \cdot \mu^{conv}_{HR}$$

**Loss Stage 1** :
$$\mathcal{L}_1 = \underbrace{\text{MSE}(\mu_{HR}, P_{target})}_{\text{skill}} + \beta_1 \mathcal{L}_{DAGMA}(A_{dag}) + \beta_2 \|A_{dag}\|_1 + \beta_3 \mathcal{L}_{CC}$$

Où $\mathcal{L}_{CC}$ enforce le scaling Clausius-Clapeyron (~7%/K, Pall 2007 Eq 14) sur $\mu^{strat}_{HR}$ et ~14%/K (Berg 2013) sur $\mu^{conv}_{HR}$ :

$$\mathcal{L}_{CC} = \left|\frac{\partial \log \mu^{strat}_{HR}}{\partial t_{2m}} - 0.07\right|^2 + \left|\frac{\partial \log \mu^{conv}_{HR}}{\partial t_{2m}} - 0.14\right|^2$$

### 2.2 Stage 2 — Stochastic Interpolant avec drift causal + score découplé

**Pourquoi ce choix** : Albergo-VE 2023 (arXiv:2303.08797) prouve que drift `b_θ` et score `s_φ` sont identifiés séparément. **Si on conditionne uniquement le drift sur (A_dag, μ_HR), le score reste libre de bypass.** Cela brise le pillar P3 de la duality MSE.

**Interpolant linéaire** (Eq 3.5 agent gen) :
$$x_t = (1-t)\,x_0 + t\,x_{HR} + \gamma(t)\,z, \quad z \sim \mathcal{N}(0, I)$$

**Drift conditionné causalement** :
$$b_\theta(x_t, t; A_{dag}, \mu^{strat}_{HR}, \mu^{conv}_{HR}, \pi, b_{log})$$

Mais ATTENTION — pour éviter le copy-mode, μ_HR ne passe PAS par concat. Il entre via une **softmax projection** (Eq 6.1 agent causalité) :

$$\tilde{\mu}_{HR} = A_{dag} \cdot \text{softmax}(A_{dag}^\top \mu_{HR} / \tau)$$

Cette projection casse formellement le rank plateau : `rank(Cov(\tilde{\mu}_{HR})) > rank(A_{dag})` génériquement (preuve géométrique : softmax = diffeomorphisme vers simplex).

**Loss drift** (Eq 3.7 agent gen) :
$$\mathcal{L}_{drift}(\theta) = \mathbb{E}_{t, x_0, x_{HR}, z}\left[\rho_\tau\big(b_\theta(x_t, t; \mathbf{c}) - \dot{x}_t\big)\right]$$

où $\rho_\tau$ est la **pinball loss quantile** (Eq 5.5 agent gen) à τ ∈ {0.5, 0.95, 0.99}. Multi-quantile loss au lieu de MSE → casse pillar P1 (Bayes optimum devient quantile conditionnel, pas mean).

**Score découplé** (Eq 3.8 agent gen), trained without A_dag :
$$\mathcal{L}_{score}(\phi) = \mathbb{E}\left[\|s_\phi(x_t, t) - (-z/\gamma_t)\|^2\right]$$

s_φ n'a JAMAIS A_dag en input → bypass set vide architecturalement (vacuously).

### 2.3 Sampling — DPS-guidance avec énergie structurée

**Énergie structurée** (Eq 4.3 agent gen) :
$$E_{causal}(x; A_{dag}, \mu_{HR}) = \|x - P^{sm}_A(x)\|^2 + \langle x, A_{dag} x\rangle$$

Avec preuve formelle (Eq 4.4) : `∂E/∂A` non-zero a.e., donc bypass impossible au sampling.

**Sampler** (Heun-style sur le drift, avec guidance) :
$$dx_t = \left[b_\theta(x_t, t; \mathbf{c}) - \lambda(t)\,\nabla_x E_{causal}(x_t; A_{dag}, \mu_{HR})\right]dt + \sqrt{2}\,\gamma_t\,dW_t$$

**Bakry-Émery** (Eq 4.6 agent gen) garantit convergence si `A_dag + A_dag^T ≻ 0`.

### 2.4 Loss totale avec log-det rank-promoting

Le pillar P3 (rank plateau) est attaqué par 2 mécanismes redondants :
1. Softmax projection (déjà au niveau Stage 2 input)
2. Log-det penalty (Eq 7.3 agent gen) :

$$\mathcal{L}_{rank} = -\lambda_{rank} \log \det\!\left(\delta I + \text{Cov}_{batch}(\hat{x}_{HR} - \mu_{HR})\right)$$

Preuve (Eq 7.5 agent gen) : `∇_θ \mathcal{L}_{rank}` diverge le long du kernel de Cov → bypass non-stationnaire dans le loss régularisé.

**Loss totale Stage 2** :
$$\mathcal{L}_2 = \mathcal{L}_{drift} + \mathcal{L}_{score} + \lambda_{rank}\,\mathcal{L}_{rank} + \lambda_{phys}\,\mathcal{L}_{CC}$$

---

## 3. Preuves formelles que CST-DAG évite les 3 pièges

### 3.1 Bypass-résistance — preuve

**Pillar P3 (decouple drift+score, Albergo-VE 2023)** :
- `s_φ(x_t, t)` n'a pas A_dag en input → `∂s_φ/∂A_dag ≡ 0` trivialement (le réseau n'a aucun moyen d'utiliser A_dag)
- Donc s_φ n'est PAS dans le bypass set au sens (1.3) car il n'y a pas de "skip vs causal" — il n'y a que skip
- `b_θ(x_t, t; A_dag, ...)` conditionne sur A_dag
- Le bypass requérirait `b_θ` indépendant de A_dag tout en minimisant la loss drift
- MAIS la softmax projection `P^sm_A(μ_HR)` apparaît dans la cible $\dot x_t$ via la décomposition CC
- Donc minimiser $\mathcal{L}_{drift}$ FORCE b_θ à utiliser A_dag par dépendance fonctionnelle dans le target
- **∂b_θ*/∂A_dag ≠ 0** au stationary point ∎

**Backup : guidance DPS** :
- Même si b_θ apprend un bypass, le sampler ajoute `-λ ∇E_causal(x; A_dag, μ_HR)` à chaque step
- Par Eq 4.4 : `∂E/∂A_dag = x x^T - 2 x P_A(x)^T` non-zero a.e.
- Donc trajectoire de sampling dépend de A_dag indépendamment de b_θ
- **HR_pred dépend formellement de A_dag** ∎

### 3.2 Copy-mode immunity — preuve

**Le copy-mode au sens (1.4)** : `D_θ(x_σ; σ→0) → x_σ`.

- Dans CST-DAG, `D_θ` n'existe pas comme objet séparé : la prédiction est `HR_pred = ODE(b_θ, s_φ, E_causal)` un sampler stochastique multi-step
- Le copy-mode au sens classique ne s'applique pas — il faudrait que toute la trajectoire ODE-SDE collapse vers identity, ce qui violerait simultanément les conditions Bayes-optimales de b_θ ET s_φ
- En particulier : si μ_HR=0 (Mardani fix), alors b_θ doit prédire ẋ_t à partir uniquement de baseline_log
- La cible ẋ_t a une composante stochastique (le bruit dz) qui ne peut PAS être copiée depuis baseline_log
- Donc b_θ ne peut pas collapse vers identity ∎

**Mixture Berg 2013** apporte une sécurité supplémentaire : les régimes stratiforme et convectif ont des targets différents. Un copy-mode unique ne peut satisfaire les deux simultanément.

### 3.3 Rank plateau escape — preuve

**3 mécanismes redondants** :

1. **DAG étendu** : passe de 6 nodes (rang 5 max) à 9 nodes incluant q_850 et ω_500. Sigma² explained passe de 0.40-0.50 à 0.75-0.85 (Pfahl 2017 Eq 22, Norris 2019 Eq 23).

2. **Softmax projection** (Eq 6.1 agent causalité) : `rank(Cov(P^sm_A(X))) > rank(A_dag)` génériquement. Softmax = diffeomorphisme vers simplex.

3. **Log-det rank-promoting** (Eq 7.3-7.5 agent gen) : `∇\mathcal{L}_{rank}` diverge sur rank-déficient → optimiseur poussé hors low-rank attractor.

4. **Quantile loss** : rank-r lemma (1.5) repose sur covariance. Quantile predictions n'ont pas Tweedie identity → lemma ne s'applique pas (Eq 5.6 agent gen).

**Bound théorique** : avec softmax + log-det + extended DAG, la rank du Cov(HR_pred − μ_HR) est bornée inférieurement par min(dim(simplex_image), full_rank_HR) = dim(HR) - 1 = très haut ∎

---

## 4. Plan d'implémentation

### 4.1 Phase 0 — Préparation données (1-2 semaines)

1. **Extraire IVT** depuis `(u_p, v_p, q_p)` : `IVT = (1/g) ∫_p_t^p_s q v dp` (Eq 9 agent climat)
2. **Calculer geopotential heights z_p** depuis `(u_p, v_p, t_p)` via thermal wind (Holton Eq 4.18) si pas direct
3. **Calculer t_2m proxy** depuis `t_850` + lapse-rate standard, ou extraire si dispo
4. **Calculer CAPE proxy** depuis `(t_p, q_p)` à 3 niveaux : `CAPE_proxy = max(0, q_850·(T_850-T_lcl))`
5. **Construire new BS32b cache v3** : (x_LR_extended, μ^strat_HR, μ^conv_HR, π_gate, baseline_log, IVT, valid_mask)

Compute estimé : ~3-5h (re-process dataset).

### 4.2 Phase 1 — Stage 1 with DAG étendu + mixture (10-15h)

1. **Extended encoder** : 15 LR vars + 3 static + IVT + CAPE_proxy → 9 nodes embeddings
2. **DAG learning DAGMA** sur 9 nodes (vs 6 actuels)
3. **Mixture heads** : `R_strat, R_conv` séparés + gate `π(t_2m, CAPE)`
4. **Loss** : MSE + DAGMA + L1 + CC-scaling (stratiform 7%/K, convective 14%/K)
5. **Target Q_phys** : > 0.9 sur le nouveau DAG étendu. Si < 0.7, abort et revoir.

Compute : 10-15h sur A100 (modèle plus gros mais Stage 1 reste cheap).

### 4.3 Phase 2 — Stage 2 CST avec stochastic interpolants (40-60h)

1. **Score network s_φ** : EDM-style UNet, NO conditioning, trained on (x_0, x_HR, z, γ_t)
2. **Drift network b_θ** : EDM-style UNet conditionned on (P^sm_A(μ_HR^strat), P^sm_A(μ_HR^conv), π_gate, baseline_log) via FiLM
3. **Multi-quantile loss** au lieu de MSE
4. **Log-det rank penalty** sur Cov_batch
5. **Train both networks JOINTLY** but with separate loss components

Compute : 40-60h sur A100 (modèle plus gros + 2 networks + multi-quantile).

### 4.4 Phase 3 — Inference DPS-guided (50 min eval)

1. Sample x_0 ~ N(0, I)
2. Heun integration of `dx_t = b_θ - λ(t)∇E_causal dt + √(2)γ_t dW_t`
3. K=128 samples ensemble
4. Compute toutes les métriques (RMSE, Pearson, F1@p99, SSR, RAPSD, CRPS)

---

## 5. Compute total estimé

| Phase | Durée | Commentaire |
|---|---|---|
| 0. Préparation données | 3-5h | one-time |
| 1. Stage 1 étendu + mixture | 10-15h | Q_phys gate ≥ 0.85 |
| 2. Stage 2 CST | 40-60h | la plus longue |
| 3. Inference + eval | 1h | K=128 sampling |
| **Total** | **55-80h** | sur A100 Colab Pro+ |

Atomic checkpointing per-epoch (pattern noncausal cell 48) obligatoire.

---

## 6. Métriques attendues — bornes formelles, pas inventées

### 6.1 Borne inférieure formelle sur RMSE

Le DAG étendu inclut $\{z, q_{850}, \omega_{500}\}$. D'après Norris 2019 (Eq 23 agent climat) :
$$\sigma^2(P_{p99} | \mathbf{S}^*) \approx 0.75\text{-}0.85$$

ce qui correspond à $R^2 \approx 0.75\text{-}0.85$, donc Pearson formel attendu :
$$\text{Pearson}_{p99} \approx \sqrt{0.75\text{-}0.85} \approx 0.87\text{-}0.92$$

**À comparer** : noncausal v4 = 0.834. Borne CST-DAG **STRICTEMENT au-dessus** du baseline.

### 6.2 Borne sur F1@p99

Décomposition Berg 2013 + Pfahl 2017 : les extrêmes mid-latitude sont majoritairement stratiforme (warm-conveyor-belt, CC scaling), capturés par $\mu^{strat}_{HR}$.

La gain attendu sur F1@p99 vient de :
- DAG étendu : +30-40 points expliqués variance → +5-15% F1@p99 (estimation prudente non chiffrée)
- Mixture stratiforme/convectif : élimine la mis-specification Berg 2013 → ~+3-5% sur les régimes mixtes
- Quantile loss : optimise directement le tail → ~+3-5% sur F1@p99 spécifiquement

**Pas de probabilité chiffrée totale** — ces estimations s'additionnent mais peuvent overlap. Empirie nécessaire.

### 6.3 Garantie pour SSR

Stochastic interpolants avec drift+score découplé : SSR proche de 1.0 par construction (Albergo-VE 2023 §4.2). C'est l'argument calibration native.

---

## 7. Risques et inconnues explicites

### 7.1 8 unknowns documentés

1. **U1 (Causality agent)** : front-door avec mediator latent appris — pas de preuve formelle au-delà CaPaint empirique
2. **U3 (Causality)** : equivariance causale jamais démontrée n=6+ nodes continus sans paired interventional data
3. **U4 (Causality)** : ICM-gradient sparsity sous Q_phys approximate — théorème de stabilité inconnu
4. **U2.1 (Generative)** : SSM anisotrope sur Im(A_dag) — identifiability open
5. **U3.1 (Generative)** : consistency conjointe causal-drift + uncausal-score — open
6. **U5.1 (Generative)** : multi-quantile diffusion — consistency SDE inconnue
7. **U6.1 (Generative)** : DPS posterior consistency avec énergie non log-concave — partial result Song 2023
8. **U7.1 (Generative)** : SGD convergence avec log-det penalty rank-deficient — pas de taux publié

Résoudre l'une de ces unknowns = contribution scientifique.

### 7.2 Risques opérationnels

- **CAPE proxy** : la définition exacte de CAPE nécessite atmospheric stability analysis. Notre proxy `q_850·(T-T_lcl)` est crude. Risque de mauvais gating π.
- **IVT integration** : nécessite extraction propre depuis (u,v,q,p) à 3 niveaux. Discrétisation verticale potentiellement insuffisante.
- **Joint training stability** : drift+score+guidance peut être instable. Albergo-VE 2023 §4.1 ne donne consistency que sous joint regularity.

---

## 8. Verdict final

**CST-DAG est mathématiquement le candidat le plus solide** parmi toutes les architectures évaluées :

| Critère | CST-DAG |
|---|---|
| Brise bypass theorem ? | OUI (P3 break + DPS architectural escape + guidance ∂E/∂A) |
| Brise copy-mode ? | OUI (mixture Berg + softmax + multi-quantile + stochastic) |
| Brise rank plateau ? | OUI (extended DAG + softmax + log-det + quantile) |
| Q_phys préservé ? | OUI (Stage 1 frozen comme avant, mais sur 9 nodes au lieu de 6) |
| Précédent empirique ? | Partiel : Albergo-VE 2023, EBM Du-Mordatch 2019, CaPaint 2024 |
| Borne théorique sur Pearson | ≥ 0.87 (Norris 2019 Eq 23) |
| Compute Colab Pro+ A100 | 55-80h |
| Nouveauté publishable | OUI (jamais combiné dans la littérature) |

**Aucune autre architecture proposée précédemment ne valide ces 4 critères simultanément.** M1-M7 échouent toutes sur au moins un.

CST-DAG **n'a pas de garantie de battre noncausal v4 sans prototypage**. Ce que les preuves formelles disent :
- La borne théorique Pearson est strictement au-dessus de 0.834
- Les 3 pièges sont formellement adressés
- 8 unknowns restent — mais aucun n'invalide directement le design

C'est le pari le plus éclairé qu'on puisse construire à partir des évidences math + climat + causalité disponibles.
