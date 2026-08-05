# Phase 0 — Spec d'implémentation du DAG 9-node (option C)

**Statut :** spec figée, **non lancée**. Gate d'entrée = robustesse de (C) confirmée
(`smoke_9node_extended_dag.ipynb`, cellule probe réel multi-seed : `ROBUSTE = OUI`,
i.e. `ΔF1@p99 mean − std > 0`). Tant que ce gate n'est pas vert, **ne pas engager les ~17h**.

**Source de vérité du contenu :** verdict du smoke 9-node
(`GO_OPTION_C` ; A PASS ΔF1@p99=+0.250, B PASS Q_phys_cont=0.682 / skeleton F1=1.0 / 0 arête parasite,
C PASS directionnel). La structure cible (10 arêtes physiques) y est déjà validée comme apprenable.

---

## 0. Résumé exécutif

| Item | 6-node (actuel) | 9-node (cible) |
|---|---|---|
| Nœuds DAG | 6 | 9 |
| Métapaths encoder | 5 (+ static SP_HR) | 8 (+ static SP_HR) |
| Arêtes physiques (`EXPECTED_EDGES`) | 5 | 10 |
| Canaux LR | 15 | 16 (+ IVT) |
| `q_850`, `w_500` | déjà natifs (idx 9, 7) | **réutilisés** (pas de calcul) |
| IVT | absent | **calculé** depuis q·(u,v) sur 3 niveaux |

**Travail réel = 5 chantiers** : (1) calcul IVT dans le pipeline, (2) **routage de features
par nœud** (le point critique), (3) `graph_builder.py` (3 node types + 5 arêtes),
(4) `config/training_config.yaml` (métapaths, `dag_prior` 9×9, dims), (5) `physics_prior.py`
(`VAR_LABELS` 9, `EXPECTED_EDGES` 10).

Compute : Phase 0 data-prep ≈ 5h (dont calcul + re-normalisation IVT) + Stage 1 retrain 9-node
≈ 12h. Puis re-run smoke pinball CDPM-min sur le nouveau DAG avant tout commit.

---

## 1. Convention canonique des 9 nœuds (À RESPECTER PARTOUT)

Ordre des canaux de `A_dag` = ordre des `encoder_configs` (métapaths) **puis** le nœud statique.
Cet ordre DOIT être identique dans : `config.encoder.metapaths`, `physics_prior.VAR_LABELS`,
`config.loss.dag_prior` (lignes/colonnes), et l'assemblage runtime des canaux.

```
idx  label            origine
 0   GP850_spat       métapath GP850_spat_adj
 1   GP850->GP500     métapath GP850_to_GP500
 2   GP500_spat       métapath GP500_spat_adj
 3   GP500->GP250     métapath GP500_to_GP250
 4   GP250_spat       métapath GP250_spat_adj
 5   Q850             métapath Q850_spat_adj      (NOUVEAU)
 6   W500             métapath W500_spat_adj      (NOUVEAU)
 7   IVT              métapath IVT_spat_adj       (NOUVEAU)
 8   SP_HR            nœud statique (toujours en dernier)
```

> **Risque #1 (bloquant) — ordre static vs métapaths.** Aujourd'hui SP_HR est à l'index 5
> (5 métapaths + static). En insérant 3 métapaths AVANT le static, SP_HR passe de l'index 5
> à l'index 8. Toute matrice/figure/threshold qui hardcode « index 5 = SP_HR » casse.
> **Action obligatoire :** après construction, asserter
> `VAR_LABELS == [cfg.name-dérivé for cfg in encoder_configs] + ["SP_HR"]` au runtime
> (échec bruyant, pas silencieux — cohérent avec `allow_missing_metapaths: false`).

---

## 2. Chantier 1 — Calcul de l'IVT (data-prep)

`w_500` et `q_850` sont **déjà** des variables LR natives ACCESS-CM2
(`config.data.lr_variables`, idx 7 et 9). Seul l'IVT est dérivé.

**Définition (transport intégré de vapeur, magnitude) :**

\[
\mathrm{IVT} = \frac{1}{g}\left\| \sum_{\ell \in \{850,500,250\}} q_\ell\,(u_\ell, v_\ell)\,\Delta p_\ell \right\|
= \frac{1}{g}\sqrt{\Big(\textstyle\sum_\ell q_\ell u_\ell \Delta p_\ell\Big)^2 + \Big(\textstyle\sum_\ell q_\ell v_\ell \Delta p_\ell\Big)^2}
\]

- Niveaux disponibles : 850 / 500 / 250 hPa. Poids de couche \( \Delta p_\ell \) (trapèze) :
  Δp₈₅₀ ≈ (1000−675)·100 Pa, Δp₅₀₀ ≈ (675−375)·100, Δp₂₅₀ ≈ (375−100)·100
  (bornes d'intégration documentées dans le code ; ce sont des **proxies** — l'IVT « vrai »
  exige plus de niveaux, mais le ceiling-probe a validé que ce proxy 3-niveaux porte de l'info de queue).
- `g = 9.80665`.
- Indices canaux (ordre `lr_variables`) : u = {0,1,2}, v = {3,4,5}, q = {9,10,11} pour {850,500,250}.

**Où le calculer :** dans `src/st_cdgm/data/pipeline.py`, au moment où le tenseur LR est
assemblé (avant normalisation), ajouter IVT comme **16ᵉ canal** :

```python
# pipeline.py — après lecture des canaux LR bruts, avant normalize
def _append_ivt(lr_raw, var_index):  # lr_raw: [C=15, H, W] en unités physiques
    g = 9.80665
    dp = {"850": 325e2, "500": 300e2, "250": 275e2}
    iu = iv = 0.0
    for lvl, w in dp.items():
        q = lr_raw[var_index[f"q_{lvl}"]]
        iu = iu + q * lr_raw[var_index[f"u_{lvl}"]] * w
        iv = iv + q * lr_raw[var_index[f"v_{lvl}"]] * w
    ivt = torch.sqrt(iu**2 + iv**2) / g
    return torch.cat([lr_raw, ivt[None]], dim=0)  # [16, H, W]
```

> **Risque #2 — normalisation.** `config.data.normalize: true` + `mean_*.nc`/`std_*.nc` sont
> calculés sur les 15 variables existantes. L'IVT a une **échelle et une distribution
> (lourde-queue, positive)** différentes. Options, par ordre de préférence :
> 1. Calculer IVT AVANT normalisation, puis recalculer mean/std sur 16 canaux sur la fenêtre
>    train (1980-2009) uniquement (respecte le fix K5 anti-fuite). **+~1h** de data-prep.
> 2. Normaliser l'IVT séparément (log1p puis z-score) car positif lourde-queue — plus propre
>    physiquement, mais ajoute un cas spécial dans le pipeline.
>
> **Recommandation : option 1** (cohérence avec le reste du pipeline) + clip à p99.9 pour borner la queue.

**Sanity data-prep (avant retrain) :** `IVT ≥ 0` partout ; `corr(IVT, pr_HR_tail) > corr(q_850, pr_HR_tail)`
sur un échantillon (l'IVT doit être ≥ aussi informatif que q_850 seul, sinon le proxy est cassé).

---

## 3. Chantier 2 — Routage de features par nœud (POINT DE DESIGN CRITIQUE)

**Constat (cf. `two_stage_inference.py:25-28`) :** actuellement tous les `dynamic_node_types`
reçoivent le **même** tenseur LR complet ; la spécialisation est uniquement topologique.

Si Q850/W500/IVT reçoivent eux aussi les 16 canaux complets, ils sont **redondants** avec les
nœuds GP existants et l'extension n'apporte PAS l'info de queue que le ceiling-probe (A) a mesurée.
→ Il faut **router des sous-ensembles de canaux par nœud** :

```python
# Mapping node_type -> indices de canaux LR (16 canaux après IVT)
NODE_CHANNELS = {
    "GP850": list(range(15)),     # legacy : tous les champs (rétro-compat)
    "GP500": list(range(15)),
    "GP250": list(range(15)),
    "Q850":  [9, 10, 11],         # q_850, q_500, q_250 (humidité)
    "W500":  [6, 7, 8],           # w_850, w_500, w_250 (vitesse verticale)
    "IVT":   [15],                # canal IVT calculé
}
```

> **Risque #3 — `in_channels` des SAGEConv.** L'encoder infère `in_channels=(-1,-1)`
> (auto). Des node types à dimensionnalité différente (15 vs 3 vs 1) sont supportés par
> `metapath_convs` (ModuleDict par métapath, §1.6), **à condition** que chaque métapath ne
> mélange pas des sources de dims différentes. Comme les 3 nouveaux métapaths sont
> `spat_adj` (source == target), c'est homogène par métapath → OK. **Vérifier** que
> `lr_grid_to_nodes` est appelé par sous-ensemble de canaux pour les nouveaux nœuds
> (adapter la boucle d'assemblage des `dynamic_features`, en training ET en inférence).

**Points de modif identifiés :** l'assemblage `dynamic_features = {node_type: ...}` doit être
généralisé via `NODE_CHANNELS` à **deux endroits** :
`src/st_cdgm/evaluation/two_stage_inference.py:27` et l'équivalent dans la boucle de training
(`src/st_cdgm/training/training_loop.py`, chemin qui appelle `prepare_step_data` /
`inject_dynamic_features`). Grep `dynamic_node_types` et `inject_dynamic_features` pour les recenser.

---

## 4. Chantier 3 — `graph_builder.py`

Dans `HeteroGraphBuilder` (`src/st_cdgm/models/graph_builder.py`) :

1. **Node types** (`__init__`, ~ligne 83) :
   ```python
   self.dynamic_node_types = ["GP850"]
   if self.include_mid_layer:
       self.dynamic_node_types.extend(["GP500", "GP250"])
   if self.extended_9node:                       # NOUVEAU flag
       self.dynamic_node_types.extend(["Q850", "W500", "IVT"])
   ```
2. **Arêtes spatiales** (`build`, ~ligne 128) : ajouter `spat_adj` pour Q850/W500/IVT
   (réutiliser `self._spatial_edge_index.clone()`).
3. **Arêtes dirigées physiques** (les 5 nouvelles de §5) — elles passent par le **DAG appris**
   `A_dag`, PAS par des `edge_index` topologiques fixes. Donc dans le builder on ajoute seulement
   les `spat_adj` des nouveaux nœuds ; les arêtes `GP850->Q850`, `Q850->IVT`, etc. sont portées
   par `G_phys` + `A_dag` (cf. §5). **Ne pas** créer d'`edge_index` pour ces 5 arêtes.
4. `_validate_edge_ranges` reste valable (vérifie bornes par type).

> **Risque #4 — backward-compat des checkpoints.** Ajouter des node types = nouvelles clés
> dans `metapath_convs` / `layer_norms`. Le retrain 9-node est **from-scratch** (pas de
> warm-start depuis un ckpt 6-node), donc PC4/PC13 (`option_c_helpers.check_pc4_pc13_gate`)
> ne s'applique pas. Si un jour warm-start : les 3 nouveaux métapaths seraient des
> `missing_keys` → à ajouter à `PC13_NEW_PARAMS_ALLOWLIST`.

---

## 5. Chantier 4 — `config/training_config.yaml`

### 5.1 Métapaths (3 ajouts, après `GP250_spat_adj`, AVANT toute logique static)

```yaml
encoder:
  metapaths:
    # ... 5 métapaths existants inchangés ...
    - name: "Q850_spat_adj"
      src: "Q850"
      relation: "spat_adj"
      target: "Q850"
      pool: "mean"
    - name: "W500_spat_adj"
      src: "W500"
      relation: "spat_adj"
      target: "W500"
      pool: "mean"
    - name: "IVT_spat_adj"
      src: "IVT"
      relation: "spat_adj"
      target: "IVT"
      pool: "mean"
```

### 5.2 `dag_prior` → 9×9 (signe/magnitude des 10 arêtes physiques, α=0.2)

Ordre lignes/colonnes = §1. Arêtes (src→tgt) = 5 anciennes + 5 nouvelles :
`GP250→GP500, GP500→GP850, GP850→SP_HR, [GP850→GP500]meta→GP500, [GP500→GP250]meta→GP250`
**+** `GP850→Q850, GP500→W500, Q850→IVT, W500→SP_HR, IVT→SP_HR`.

```yaml
loss:
  lambda_dag_prior: 0.40          # valeur Path C+ (option_c_helpers), PAS 0.005
  dag_prior:   # 9×9, idx: 0 GP850s 1 GP850>500 2 GP500s 3 GP500>250 4 GP250s 5 Q850 6 W500 7 IVT 8 SP_HR
    - [0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.2]   # GP850 -> Q850, SP_HR
    - [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # GP850->GP500 (meta) -> GP500
    - [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0]   # GP500 -> GP850, W500
    - [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]   # GP500->GP250 (meta) -> GP250
    - [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # GP250 -> GP500
    - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0]   # Q850 -> IVT
    - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]   # W500 -> SP_HR
    - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2]   # IVT -> SP_HR
    - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # SP_HR : rien (prior fixe)
```

> Note : c'est un **prior signé** (magnitude α=0.2). Les valeurs exactes (descente verticale
> 250→500→850→SP_HR) suivent la même convention QG que le 6-node. La direction
> `GP500→W500` / `GP850→Q850` traduit « la dynamique synoptique force humidité/ascendance »,
> et `Q850/W500/IVT→SP_HR` ferme la chaîne humide vers la précip (décomposition Held-Soden).

### 5.3 Dimensions auto-calculées

```yaml
diffusion:
  unet_kwargs:
    projection_class_embeddings_input_dim: 1152   # 128 × 9 (était 768 = 128×6). "auto-aligné runtime"
rcn:
  driver_dim: 16          # 15 → 16 (IVT). "Overridden at runtime by data sample shape" → vérifier override
  reconstruction_dim: 16
```

> Vérifier que `projection_class_embeddings_input_dim` est bien recalculé au runtime
> (`= hidden_dim × n_metapaths_effectifs`). Le commentaire config dit « auto-aligné » ;
> si le hardcode 768 sert de fallback, le passer à 1152.

---

## 6. Chantier 5 — `physics_prior.py`

```python
VAR_LABELS: List[str] = [
    "GP850_spat", "GP850->GP500", "GP500_spat", "GP500->GP250", "GP250_spat",
    "Q850", "W500", "IVT",           # NOUVEAUX (idx 5,6,7)
    "SP_HR",                          # static, repoussé à idx 8
]

EXPECTED_EDGES: List[Tuple[str, str, int]] = [
    ("GP250_spat", "GP500_spat", +1),
    ("GP500_spat", "GP850_spat", +1),
    ("GP850_spat", "SP_HR", +1),
    ("GP850->GP500", "GP500_spat", +1),
    ("GP500->GP250", "GP250_spat", +1),
    # --- chaîne humide (Held-Soden) ---
    ("GP850_spat", "Q850", +1),
    ("GP500_spat", "W500", +1),
    ("Q850", "IVT", +1),
    ("W500", "SP_HR", +1),
    ("IVT", "SP_HR", +1),
]
```

- `build_physical_mask(num_vars=9)` par défaut.
- Mettre à jour le docstring `>>> G.shape … (9,9), 10.0, 10`.
- `physical_prior_loss` inchangé (déjà `normalize=False`, scaling I1 OK).

> Le smoke 9-node récupère ces 10 arêtes avec **0 arête parasite** (`n_extra=0`,
> seuil 0.114) et `Q_phys_cont=0.682` → les hyperparams DAGMA Path C+
> (`lambda_dag_prior=0.40`, `g_phys_alpha=0.25`, `lambda_l1` 0.04→0.005) sont les bons
> points de départ (cf. `PATHCPLUS_HYPERPARAM_OVERRIDES`). Ne PAS repartir des valeurs
> V5-mini (0.005 / 0.001) qui collapsent.

---

## 7. Plan d'exécution & gates

| Étape | Durée | Gate de sortie |
|---|---|---|
| G0. Robustesse smoke (C) | ~10 min | `ROBUSTE = OUI` sinon STOP |
| 1. IVT pipeline + re-norm 16 canaux | ~2h | sanity IVT≥0 ; corr(IVT,tail) ≥ corr(q850,tail) |
| 2. Routage features par nœud | ~1h | test unitaire : 9 node types peuplés, dims {15,3,1} OK |
| 3. graph_builder + config + physics_prior | ~1h | assert ordre `VAR_LABELS` == runtime ; `build_physical_mask(9).sum()==10` |
| 4. Stage 1 retrain 9-node (3 seeds) | ~12h | `Q_phys_cont ≥ 0.50`, `n_extra<3`, pas de collapse, intervention test PASS |
| 5. Re-run smoke pinball CDPM-min sur DAG 9-node | ~30 min | les 3 axes (tail RMSE, μ_HR usage, rank escape) PASS |
| 6. Commit CDPM-min full | — | seulement si 4+5 PASS |

**Causal-ablation (`config.two_stage.causal_ablation`, `abort_if_fail: true`)** reste actif :
si le 9-node ne conditionne pas effectivement μ_HR, le run s'arrête → pas de gaspillage.

---

## 8. Risques consolidés

1. **Ordre static/métapaths** (§1) — bloquant, assertion runtime obligatoire.
2. **Normalisation IVT** (§2) — recalcul mean/std 16 canaux sur fenêtre train (anti-fuite K5).
3. **Routage features par nœud** (§3) — sans ça l'extension est topologique-seulement = inutile.
4. **`in_channels` hétérogènes** par node type — homogène par métapath (spat_adj) → OK, à tester.
5. **`projection_class_embeddings_input_dim`** 768→1152 — vérifier auto vs hardcode.
6. **(C) marginal** — le ΔF1@p99 réel (+0.05) est faible ; A+B portent la décision. Si le
   retrain Stage 1 ne dépasse pas le 6-node sur F1@p99 validation, **fallback option B**
   (6-node + CDPM-min) ou pivot métriques FSS/CSI/SEDI (leçon §14.5).

---

## 9. Fichiers touchés (récapitulatif)

| Fichier | Modif |
|---|---|
| `src/st_cdgm/data/pipeline.py` | calcul IVT (16ᵉ canal) + re-normalisation |
| `src/st_cdgm/models/graph_builder.py` | 3 node types + 3 `spat_adj` (flag `extended_9node`) |
| `src/st_cdgm/training/training_loop.py` | routage `NODE_CHANNELS` dans l'assemblage `dynamic_features` |
| `src/st_cdgm/evaluation/two_stage_inference.py` | idem routage (ligne 27) |
| `src/st_cdgm/training/physics_prior.py` | `VAR_LABELS` (9), `EXPECTED_EDGES` (10), `num_vars=9` |
| `config/training_config.yaml` | 3 métapaths, `dag_prior` 9×9, `lambda_dag_prior=0.40`, dims |

*(Spec exempte d'analyse — schéma `phase0-spec-9node-v1`, comme les autres docs d'audit.)*
