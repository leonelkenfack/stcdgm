# PLAN V6 — MVP révisé final — Mise à jour du 9-node seed 42

**Base :** `path_c_plus/scripts/st_cdgm_path_c_option_c_9node.ipynb` (V5_causal seed 42 Path C+ Option C).

**Contrainte (NB user) :** maintenir ce qui est fait dans 9-node seed 42. Petites corrections / ajouts uniquement. Garder tout ce qui est déjà dessus.

**Cible :** battre `noncausal_v4` F1@p99 = 0.550 (Convention B pooled).

**Status :** plan révisé après **4 rondes** de consultation des 5 experts (ML, Math, Recherche, IA, Climat). MVP révisé final validé **unanimement (5/5 GO)**.

**Probabilité moyenne battre noncausal F1@p99=0.550 : ~57%**
- ML : 55-60% | Math : 52-58% | Recherche : 55-65% | IA : 55-60% | Climat : 58-62%

**Compute total estimé : ~32h** (préproc 3-5h + Stage 1.A 15h + SMOKE 4h + Stage 2 8-10h + validation 2h)

**Pivot vs versions précédentes :** Abandonné le UNet refiner boost (V6 initial — sur-ingénié, Phase 6 v2). Retenu : ajouts chirurgicaux ciblés sur F4 (dette `r_φ(H)` Math Prof) + plateau B₁ rank-5 (F2/F3/F6 dans `ECHECS_ET_LECONS.md`).

---

## 1) Ajouts au Stage 1.A — REFAIT (~15h)

### 1.1 — Nœuds à ajouter au graphe (9 → 11 dyn)

| Nœud | Variable physique | Source |
|---|---|---|
| `u850` | Composante zonale du vent à 850 hPa | Climat verbatim ronde 1 : *"u850/v850 → précipitation orographique locale"* |
| `v850` | Composante méridienne du vent à 850 hPa | Idem |

**Graphe final : 11 nœuds dynamiques + SP_HR static**
- Existant : GP850, GP500, GP250, Q850, W500, IVT
- V6 ajout : U850, V850

**Décisions MVP (économie compute) :**
- ~~SST_TASMAN nœud~~ : **droppé** (Climat ronde 3 : "acceptable MVP sans SST" — retour V6.1 si signal insuffisant)
- ~~IVT_U, IVT_V vectoriels~~ : **droppés** (économie, IVT magnitude déjà présente)

### 1.2 — Features LR à ajouter (15 → 21 vars, +1 optionnelle = 22)

| Feature | Définition | Source | Statut |
|---|---|---|---|
| `w_700` | `0.571·w_850 + 0.429·w_500` (interp pondérée pression, Holton 2004) | Climat verbatim | obligatoire |
| `theta_e_850` | Bolton 1980 simplifié, q observé, à 850 hPa | Climat verbatim | obligatoire |
| `theta_e_500` | Bolton 1980 simplifié, à 500 hPa | Climat verbatim | obligatoire |
| `mucape_proxy` | `theta_e_850 − theta_e_500` | Climat verbatim | obligatoire |
| `T_850 − T_500` | Stabilité statique LR-pure | Climat ronde 2 — remplace ancien `foehn_proxy = T_850 − T_surf_HR` (fuite cible→entrée) | obligatoire |
| `u850_HR_bilin · ∇h_HR_4km` (avec **mask_ocean**) | Forçage orographique dynamique, résolution-cohérent | Climat ronde 2+3 — Elvidge-Renfrew 2016 *BAMS* 97:455 ; `mask_ocean` (Climat ronde 4) évite ∇h aberrant côtier | obligatoire |
| `theta_w_850` | Wet-bulb potential temperature à 850 hPa | **Climat ronde 4** — Browning 2004 *QJRMS* "warm conveyor belt", meilleur proxy instabilité convective NZ pré-frontale | **OPTIONNEL (§4.4)** |

### 1.3 — Arêtes prior G_phys 11×11 (étendues)

Ajouts à `G_phys` (initialisation prior pour A_dag, A_dag reste appris) :

| Arête | Justification physique |
|---|---|
| `U850 → SP_HR` | Vent zonal bas niveau → forçage orographique West Coast NZ |
| `V850 → SP_HR` | Vent méridien bas niveau → ARs N→S transpacifiques |
| `U850 → IVT` | u850 contribue au transport zonal de vapeur |
| `V850 → IVT` | v850 contribue au transport méridien |

Bilan : G_phys passe ~10 arêtes (9-node) → ~14 arêtes (11-node V6).

### 1.4 — Hyperparam adaptation Stage 1.A (Climat ronde 3)

| Hyperparam | 9-node seed 42 | 11-node V6 | Raison |
|---|---|---|---|
| `λ_l1` (sparsité A_dag) | 0.04→0.005 cosine | **0.028→0.0035** (×0.7) | Passage 9→11 nœuds : ‖A‖₀ attendu monte ~22→33 arêtes Trenberth-pondérées. Garder λ_l1 inchangé → sur-sparsification. Climat ronde 3 : *"λ_l1 doit baisser ~30%"* |
| `λ_dag_prior` | 0.40 | 0.40 maintenu | MVP simplicité (option B/F : rescale Math 0.24 disponible en V6.1) |
| `γ_dag` (DAGMA) | identique | identique | Climat : "stable" |
| `dag_grad_gate` | identique | identique | Inchangé |
| `dag_floor_min_norm` | 0.10 | 0.10 | Inchangé |
| `abort_on_collapse` | True | True | Inchangé |
| `dag_spectral_projection` | True | True | Inchangé |

**Tout le reste du protocole Stage 1.A reste strictement identique à `st_cdgm_path_c_option_c_9node.ipynb` seed 42.**

---

## 2) Ajouts au Stage 2 — 3 ajouts orthogonaux (~8-10h)

### 2.1 — `r_φ(H)` auxiliary residual head AVEC structure spatiale (F4 dette critique)

**Justification (citation `ECHECS_ET_LECONS.md` F4) :** *"Recommandée en §14.3 mais pas dans le code. La recommandation théorique la plus solide est restée non-testée — c'est la dette critique."*

F3 prouve formellement : *"Stage 1 frozen → impossible d'améliorer la skill au-delà du plancher B₁ rank-5. Stage 2 a besoin d'une voie expressive hors-DAG"*. `r_φ(H)` est exactement cette voie.

**Architecture (IA ronde 4 — structure spatiale obligatoire pour éviter collapse F4) :**

```python
class StructuredResidualHead(nn.Module):
    """r_φ(H) avec structure spatiale (project + broadcast régional + conv 1×1).

    Flatten MLP simple → collapse historique (cf. memoire project_mc2rd_dead.md).
    Solution IA : projeter H_T par node embedding + broadcast régional via
    masks Trenberth (West Coast, East Coast, Northland, South) + conv 1×1.

    Shapes :
      H_T          : [B, num_vars=11, hidden=128]
      h_proj       : [B, 11, emb_dim=16]   (project chaque node)
      region_masks : [n_regions=4, H_HR=172, W_HR=179]
      h_spatial    : [B, 11*4*emb_dim=704, H_HR, W_HR]   (broadcast spatial)
      r            : [B, 1, H_HR, W_HR]
    """
    def __init__(self, num_vars=11, hidden=128, emb_dim=16,
                 region_masks=None, n_regions=4):
        super().__init__()
        self.num_vars = num_vars
        self.n_regions = n_regions
        self.emb_dim = emb_dim
        self.node_proj = nn.Linear(hidden, emb_dim)
        self.register_buffer("region_masks", region_masks)  # [n_regions, H_HR, W_HR]
        in_channels = num_vars * n_regions * emb_dim
        self.refine = nn.Conv2d(in_channels, 1, kernel_size=1)
        # Init anti-collapse (IA ronde 4)
        nn.init.normal_(self.refine.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.refine.bias)

    def forward(self, H_T):  # [B, num_vars, hidden]
        B = H_T.shape[0]
        # 1. Project chaque node embedding
        h_proj = self.node_proj(H_T)  # [B, num_vars, emb_dim]
        # 2. Broadcast spatial via masks régionaux (n=num_vars, r=regions, e=emb_dim)
        #    region_masks : [r, H, W]  → broadcast par node ET par emb
        h_spatial = torch.einsum('bne,rhw->bnrehw', h_proj, self.region_masks)
        # 3. Flatten en canaux pour conv 1×1
        h_spatial = h_spatial.reshape(B, self.num_vars * self.n_regions * self.emb_dim,
                                       *self.region_masks.shape[-2:])
        # 4. Conv 1×1 final
        r = self.refine(h_spatial)  # [B, 1, H_HR, W_HR]
        return r

# Sortie finale Stage 2 (parallèle à μ_HR) :
hr_predicted = baseline_log + μ_HR + D_y + λ_r · r_φ(H_T.detach())
                                              # ^^^ contribution résiduelle hors-DAG
```

**Warmup procedure (IA ronde 4) :**
- Init : `conv1x1.weight ~ N(0, 1e-3)` (déjà dans code ci-dessus)
- **Gel `r_φ` pendant 2k steps Stage 2** (entraînement MSE EDM pure)
- Puis dégel avec **ramp linéaire `λ_r : 0.05 → 0.20` sur 5k steps**
- Coût : 0 (juste un scheduler)

### 2.2 — Pinball loss multi-τ (P1 MSE-bypass)

**Justification (CST-DAG §2.2) :** *"Multi-quantile loss au lieu de MSE → casse pillar P1 (Bayes optimum devient quantile conditionnel, pas mean). Rank-r lemma ne s'applique pas (Eq 5.6 agent gen)"*.

```python
def pinball_loss(pred, target, tau):
    diff = target - pred
    return torch.mean(torch.maximum(tau * diff, (tau - 1) * diff))

# Loss totale Stage 2 :
L_total = L_edm + λ_pinball · sum(pinball_loss(D_y + μ_HR, target, tau) 
                                   for tau in [0.5, 0.95, 0.99])
```

**Contrainte Math (ronde 4) :** `λ_pinball ≤ 0.1 · λ_MSE`. Sinon dérive Heun ODE déterministe (asymétrie gradient).

**Calibration p99 sur val set AVANT compute Stage 2 full (Recherche + ML, 1h smoke).**

### 2.3 — Log-det rank-promoting penalty (P3 plateau B₁)

**Justification (CST-DAG §2.4) :** *"`L_rank = −λ · log det(δI + Cov_batch(x_HR_pred − μ_HR))`. Preuve : `∇L_rank` diverge le long du kernel de Cov → bypass non-stationnaire dans le loss régularisé"*.

**Implémentation MVP (Math + IA contrainte mémoire) :**

```python
# Subsample 256-512 pixels stratifiés par région NZ (West, East, North, South)
# Cov 256×256 (au lieu de 30000×30000 impraticable, 8 GB)
K_SUBSAMPLE = 384  # entre 256 et 512
N_REGIONS = 4  # West, East, North, South NZ
PIXELS_PER_REGION = K_SUBSAMPLE // N_REGIONS

def log_det_rank_penalty(x_HR_pred, mu_HR, region_indices_stratified):
    residual = (x_HR_pred - mu_HR).reshape(B, -1)
    # subsample stratifié
    sampled = residual[:, region_indices_stratified]  # [B, K_SUBSAMPLE]
    cov = sampled.T @ sampled / max(B - 1, 1)
    return -torch.logdet(δ * torch.eye(K_SUBSAMPLE, device=cov.device) + cov)

# batch_size ≥ 128 OBLIGATOIRE (sinon Cov dégénère, log-det → -∞)
```

**Recompute des indices subsample toutes les `N=500 steps`** (IA).

### 2.4 — Softmax projection — **DROPPÉE** (Math + IA + ML)

**Raison :** Math preuve formelle ronde 4 — *"Si rank(A_dag)=5 (Trenberth-like), alors A_dag · softmax(·) ∈ Im(A_dag), dim ≤ 5. **Ne casse PAS le rank plateau**, le déplace seulement vers le simplex. Erreur conceptuelle."*

IA ronde 4 : *"Incompatible avec `causal_concat=True` actuel — risque double-projection (Stage 1 a déjà dag_floor)."*

---

## 3) Garde-fous OBLIGATOIRES (7 — consensus 3+/5)

### 3.1 — Freeze A_dag pendant Stage 2 (ML ronde 4)

A_dag reste **figé** tel qu'appris en Stage 1.A. **Jamais touché pendant Stage 2**. Évite co-adaptation `r_φ ↔ A_dag` → attribution causale ininterprétable.

Mécanisme : `freeze_stage1(encoder, rcn_runner.cell, regression_head)` (fonction existante) couvre déjà A_dag. À VÉRIFIER que `requires_grad=False` est bien propagé à A_dag (buffer vs param — IA ronde 3).

### 3.2 — r_φ warmup procedure (IA ronde 4)

Détails dans §2.1 ci-dessus. Init `N(0, 1e-3)` + gel 2k steps + ramp `0.05→0.2` sur 5k steps.

### 3.3 — Monitoring live `‖r_φ‖ / ‖μ_θ‖` (Math + ML + IA)

À logger chaque epoch Stage 2 :

| Métrique | Seuil | Action |
|---|---|---|
| `‖r_φ(H)‖₂ / ‖μ_θ‖₂` dès epoch 5 | ≥ 0.05 | sinon **RED FLAG** (r_φ collapse) — debug obligatoire |
| Contribution r_φ au gain F1 (en ablation) | ≤ 40% | si > 40% → **REJET du run** (ML : "MVP techniquement gagnant mais scientifiquement vide") |
| rank(H + r_φ) per epoch | augmente | ML : "monitorer rank pour log-det vs r_φ pas hijacking" |

### 3.4 — SMOKE 4h OBLIGATOIRE avant full run (IA ronde 4)

Tester l'interaction `r_φ + causal_concat` (jamais testée ensemble) :

```
SMOKE 4h :
  ├─ 1h : pinball calibration p99 sur val set
  ├─ 2h : test r_φ + causal_concat — vérifier H_T garde signal causal
  │       après projection node-embedding (pas de leak/wash-out)
  ├─ 30min : log-det subsample stabilité (Cov 384×384 conditioning)
  └─ 30min : abort gates check (red flags ne se déclenchent pas faux positif)

SMOKE PASS si :
  - pinball gradient borné (||∇|| < 1.0 partout)
  - r_φ output non-zero après dégel (‖r_φ‖/‖μ_θ‖ > 0.02 fin smoke)
  - log-det finite, batch ≥ 128 OK
  - abort gates silencieux

SMOKE FAIL → diagnostic + fix + re-smoke. PAS de full run avant SMOKE PASS.
```

### 3.5 — Pré-enregistrement seuils PASS/FAIL (Recherche)

Commit JSON + commit hash **AVANT** tout run Stage 2 full. Anti-HARKing (Kerr 1998 *Pers. Soc. Psychol. Rev.* 2:196).

**Seuils proposés (tous sur Convention B pooled, standard littérature cGAN) :**

| Critère | Valeur |
|---|---|
| `F1@p99_convB ≥ 0.480` | PASS minimal (battre V5_causal seed 42 = 0.453) |
| `F1@p99_convB ≥ 0.550` | PASS CIBLE (égale noncausal_v4) |
| `F1@p99_convB ≥ 0.580` | PASS STRONG (bat noncausal +5%) |
| `RMSE_convB` dégradation ≤ 5% vs V5_causal | non-régression |
| `Pearson_global_convB` dégradation ≤ 3% vs V5_causal | non-régression |
| OOD EC-Earth3 `F1@p99` ≥ 0.40 | robustesse OOD (sinon FAIL) |
| Ablation r_φ contribution | ≤ 40% du gain F1 (sinon REJET, ML) |

### 3.6 — Calibration pinball val set (Recherche, ML)

1h smoke (inclus dans §3.4). Vérifier que `τ=0.99` calibration est stable (pas de gradient explosion sur events rares).

### 3.7 — OOD EC-Earth3 OBLIGATOIRE (Climat + Recherche)

**Climat ronde 4 :** *"EC-Earth3 plus discriminant : meilleur jet subtropical SH + ARs Tasman (Reid et al. 2021 *Weather Clim. Extremes*); NorESM2 a biais SST Pacifique sud trop froid qui masque advection."*

Eval OOD complet sur EC-Earth3 = condition validation. NorESM2 optionnel.

---

## 4) Ajouts optionnels (~+2-4h compute, ROI publication massif)

### 4.1 — CRPS multi-échelle (Gneiting & Raftery 2007)

Métrique probabiliste honnête pour ensembles diffusion. Recherche : *"sans ça causal reste cosmétique"*. ~30min compute.

### 4.2 — Scénario interventionnel test-time `do(SST+2K)` (Recherche ronde 4)

Test causal réel : perturber SST de +2K à l'inférence, mesurer réponse F1@p99 et Rx1day. Si réponse cohérente avec littérature (Clausius-Clapeyron ~7%/K, Pall 2007 *Nature*), le DAG appris a une vraie sémantique causale. ~1h compute.

### 4.3 — `mask_ocean` pour `∇h` (Climat ronde 4)

Évite gradient topographique aberrant en zones côtières. Intégré dans préproc §1.2. ~0h coût (préproc).

### 4.4 — `θ_w_850` (theta-w humide, Browning 2004 *QJRMS*)

Meilleur proxy instabilité convective NZ pré-frontale (warm conveyor belt). Intégré dans préproc §1.2. ~0h coût (préproc).

---

## 5) Pipeline d'exécution séquentiel obligatoire

```
[0] Pré-enregistrement seuils PASS/FAIL (15 min)
    └─ commit JSON dans path_c_plus/audit/V6_MVP_seuils_preregistered.json
       + commit hash

[1] Préproc V6 features Climat (~3-5h)
    ├─ Calcul des 7 features LR (w_700, θ_e_*, θ_w_850, mucape, T_850−T_500, u850·∇h)
    ├─ mask_ocean appliqué
    ├─ Construction nœuds U850, V850 dans dataset LR
    └─ Sauvegarde lr_v6_augmented.nc

[2] Stage 1.A REFAIT 11-node (~15h A100)
    ├─ HeteroGraphBuilder étendu (11 dyn nodes)
    ├─ λ_l1 0.028→0.0035 (×0.7 du 9-node)
    ├─ Reste protocole strictement identique 9-node seed 42
    ├─ Stage 1.A safeguards activés (dag_floor, abort_on_collapse, spectral_proj)
    ├─ A_dag_final + Q_phys + skel_F1 doivent passer le PASS minimal
    └─ checkpoint stage1A_v6_11node.pth

[3] causal_ablation_check (O3 gate, identique 9-node) — abort si fail

[4] freeze_stage1(encoder, rcn_cell.A_dag, regression_head)
    └─ VÉRIFIER A_dag.requires_grad == False explicitement (IA)

[5] SMOKE 4h Stage 2 (§3.4 OBLIGATOIRE)
    ├─ Pinball calibration p99
    ├─ r_φ + causal_concat interaction test
    ├─ Log-det subsample stabilité
    ├─ Abort gates check
    ├─ PASS → continuer [6]
    └─ FAIL → diagnostic + fix + re-smoke

[6] Stage 2 FULL ~8-10h
    ├─ optimizer AdamW sur (diffusion.parameters() + r_φ.parameters())
    ├─ gradient_clip_norm = 1.0 (ML — anti pinball instabilité)
    ├─ batch_size ≥ 128 (OBLIGATOIRE pour log-det)
    ├─ Loss = L_edm + λ_pinball·L_pinball + λ_rank·L_logdet + λ_r·MSE(r_φ contribution)
    ├─ r_φ warmup : gel 2k steps + ramp λ_r 0.05→0.2 sur 5k steps
    ├─ Monitoring live ‖r_φ‖/‖μ_θ‖ + rank(H+r_φ) (§3.3)
    ├─ Abort si red flags
    └─ EMA BS37 decay 0.9999 (inchangé)

[7] Validation (~2-3h)
    ├─ Eval ACCESS-CM2 test split (Convention A + Convention B)
    ├─ Ablation r_φ (mesurer attribution réelle — REJET si > 40%)
    ├─ Eval OOD EC-Earth3 complet
    ├─ CRPS multi-échelle (§4.1)
    ├─ do(SST+2K) test interventionnel (§4.2)
    └─ Comparaison vs seuils pré-enregistrés [0]
```

**Compute total : ~28-32h** (32h avec ajouts optionnels CRPS + do(SST+2K)).

---

## 6) Architecture finale détaillée

### 6.1 — Vue d'ensemble

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  INPUTS V6 MVP (par échantillon temporel)               │
│                                                                         │
│  LR brute : [T_seq, B, C_LR=21, H_LR, W_LR]                             │
│      C_LR = 15 NONCAUSAL_15_VARS + 6 features Climat                    │
│             + 7e feature optionnelle (θ_w_850 si activé)                │
│                                                                         │
│  HR cible : [T_seq, B, 1, H_HR=172, W_HR=179]   log1p(mm/d)             │
│  Baseline : [T_seq, B, 1, H_HR, W_HR]            log1p baseline         │
│  Static HR : [B, C_static, H_HR, W_HR]            topo + mask_ocean     │
│                                                                         │
│  Hetero graph : 11 dynamic + SP_HR                                      │
│      Existant : GP850, GP500, GP250, Q850, W500, IVT                    │
│      V6 ajout : U850, V850                                              │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1.A REFAIT — protocole IDENTIQUE 9-node seed 42                  │
│                     (modulo λ_l1 baissé 30% pour 11-node)               │
│                                                                         │
│  1. encoder = IntelligibleVariableEncoder(11 metapaths, hidden, cond)   │
│       in  : HeteroData (11 dyn nodes, edges spat/vert/statiques)        │
│       out : H_0                                                         │
│                                                                         │
│  2. rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval)           │
│       rcn_cell : RCNCell(num_vars=11, hidden, driver_dim, dropout)      │
│                  A_dag ∈ R^{11×11} appris (Trenberth-prior G_phys 11×11)│
│       in  : H_0, drivers=[lr_grid_t for t in T_seq]                     │
│       out : H_T ∈ R^{B, 11, hidden_dim}                                 │
│                                                                         │
│  3. regression_head = GraphToGridDecoder(d_model, hr_h, hr_w, …)        │
│       in  : H_T                                                         │
│       out : mu_HR ∈ R^{B, 1, H_HR, W_HR}    log1p(mm/d) résidu          │
│                                                                         │
│  Loss Stage 1.A = identique 9-node + λ_l1 baissé 30% pour 11-node       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ├── mu_HR
                              └── H_T (export pour Stage 2 — r_φ)
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│         freeze_stage1(encoder, rcn_runner.cell, regression_head)        │
│         + VÉRIF A_dag.requires_grad == False explicitement              │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2 V6 MVP — diffusion EDM identique + 3 ajouts orthogonaux        │
│                                                                         │
│   ┌── ENTRÉES ──────────────────────────────────────────────────────┐  │
│   │  mu_HR ∈ [B, 1, H_HR, W_HR]   (Stage 1.A output, gelé)          │  │
│   │  baseline_log ∈ [B, 1, H_HR, W_HR]                              │  │
│   │  delta_target = target_residual − mu_HR                         │  │
│   │  H_T.detach() ∈ [B, 11, hidden]  (Stage 1.A latent pré-decoder) │  │
│   │  region_masks ∈ [4, H_HR, W_HR]  (Trenberth West/East/N/S NZ)   │  │
│   └─────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│   ┌── DIFFUSION FORWARD ───────────────────────────────────────────┐   │
│   │  σ ~ LogNormal(P_mean, P_std)                                   │   │
│   │  y_noisy = delta_target + σ · noise                             │   │
│   │  unet_in = [c_in · y_noisy, mu_HR, baseline_log]  (causal_concat)│  │
│   │  D_y = diffusion.forward_edm(y_noisy, σ, mu_HR, baseline_log)   │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│   ┌── AJOUT 1 : r_φ(H) AUXILIARY RESIDUAL HEAD (F4 dette) ───────┐    │
│   │  r_φ_out = StructuredResidualHead(H_T.detach(), region_masks)  │    │
│   │          = conv1×1(broadcast_régional(project_node(H_T)))       │    │
│   │  Init : conv1×1.weight ~ N(0, 1e-3)                             │    │
│   │  Warmup : gel 2k steps, puis ramp λ_r 0.05→0.2 sur 5k steps    │    │
│   │  Output : contribution résiduelle parallèle à mu_HR             │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│   ┌── COMBINAISON FINALE (parallèle, pas concat) ─────────────────┐    │
│   │  hr_log_pred = baseline_log + mu_HR + D_y + λ_r · r_φ_out      │    │
│   │  (r_φ est une VOIE HORS-DAG additive — F3 satisfaite)          │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│   ┌── LOSSES (3 termes orthogonaux) ──────────────────────────────┐    │
│   │  L_edm     = E[λ(σ)·‖D_y − delta_target‖²]   (EDM Karras)     │    │
│   │  L_pinball = Σ_τ∈{0.5,0.95,0.99} pinball(τ, hr_log_pred, target)│   │
│   │              avec λ_pinball ≤ 0.1 · λ_edm                       │    │
│   │  L_logdet  = −λ_rank · logdet(δI + Cov_subsample_384(           │    │
│   │              hr_log_pred − mu_HR))                              │    │
│   │              subsample stratifié par région NZ                  │    │
│   │  L_total   = L_edm + λ_pinball·L_pinball + λ_rank·L_logdet     │    │
│   └────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│   Optimization :                                                        │
│     optimizer AdamW sur (diffusion.params + r_φ.params)                 │
│     gradient_clip_norm = 1.0                                            │
│     batch_size ≥ 128 OBLIGATOIRE                                        │
│     EMA BS37 decay 0.9999 (inchangé)                                    │
│     conditioning_dropout_prob = 0.0 (inchangé 9-node)                   │
│                                                                         │
│   Monitoring live :                                                     │
│     ‖r_φ‖/‖μ_θ‖ ≥ 0.05 dès epoch 5 (red flag sinon)                    │
│     rank(H + r_φ) per epoch                                             │
│     Abort gates §3.3 actifs                                             │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  INFÉRENCE & VALIDATION                                                 │
│                                                                         │
│  diffusion.sample(edm_karras, num_steps=32, cfg_scale=1.0)              │
│  hr_log_final = D_y + mu_HR + baseline_log + λ_r · r_φ_out              │
│  hr_mm        = expm1(hr_log_final)                                     │
│                                                                         │
│  Évaluation :                                                           │
│    ├─ ACCESS-CM2 test (Conv A + Conv B)                                 │
│    ├─ OOD EC-Earth3 obligatoire (§3.7)                                  │
│    ├─ CRPS multi-échelle (§4.1)                                         │
│    ├─ do(SST+2K) test causal (§4.2)                                     │
│    ├─ Ablation r_φ (REJET si > 40% gain)                                │
│    └─ Comparaison vs seuils pré-enregistrés (§3.5)                      │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 — Module par module

Légende : **NOUVEAU V6**, **CODE modifié**, **CODE inchangé, données étendues**, **CODE et données inchangés**

| Bloc | Statut V6 MVP | Source |
|---|---|---|
| Préproc LR augmenté | **NOUVEAU V6** — script `scripts/preprocess_v6_lr.py` : 7 features Climat + mask_ocean | nouveau |
| Pipeline LR (`NetCDFDataPipeline`) | **CODE modifié** : `lr_variables` 15 → 21 | [pipeline.py] |
| `HeteroGraphBuilder` | **CODE modifié** : +`U850`, `V850` aux types de nœuds dynamiques. Arêtes spatiales auto-bouclées. Arêtes prior G_phys §1.3 | [graph_builder.py](src/st_cdgm/models/graph_builder.py) |
| `IntelligibleVariableEncoder` | **CODE inchangé, données étendues** : `num_vars=11` | [intelligible_encoder.py](src/st_cdgm/models/intelligible_encoder.py) |
| `RCNCell` + `RCNSequenceRunner` | **CODE inchangé, données étendues** : `num_vars=11`, `A_dag ∈ R^{11×11}`. **Export `H_T`** pré-regression_head | [causal_rcn.py](src/st_cdgm/models/causal_rcn.py) |
| Schedules DAG (`γ_dag`, `dag_grad_gate`, `dag_floor_min_norm=0.10`, `abort_on_collapse`, `dag_spectral_projection`) | **CODE et données inchangés** | [training_loop.py](src/st_cdgm/training/training_loop.py) |
| `λ_l1` | **VALEUR modifiée** : 0.04→0.005 cosine devient 0.028→0.0035 (×0.7) pour 11-node (Climat) | training_loop.py |
| `λ_dag_prior` | **CODE et données inchangés** : 0.40 maintenu (MVP simplicité) | training_loop.py |
| `causal_ablation_check` (O3 gate) | **CODE et données inchangés** | [two_stage.py:753](src/st_cdgm/training/two_stage.py#L753) |
| `GraphToGridDecoder` (regression_head) | **CODE et données inchangés** | [regression_head.py](src/st_cdgm/models/regression_head.py) |
| `freeze_stage1` | **CODE et données inchangés** + vérif explicite A_dag.requires_grad=False | [two_stage.py freeze_stage1] |
| **`StructuredResidualHead`** | **NOUVEAU V6** — `src/st_cdgm/models/stage2_residual_head.py` (project node + broadcast régional Trenberth + conv 1×1) | — |
| **Pinball loss multi-τ** | **NOUVEAU V6** — `src/st_cdgm/training/queue_losses.py` | — |
| **Log-det rank penalty subsample stratifié** | **NOUVEAU V6** — `src/st_cdgm/training/rank_losses.py` | — |
| **r_φ warmup scheduler** | **NOUVEAU V6** — gel 2k + ramp 5k dans `train_epoch_stage2_cached` | two_stage.py |
| `precompute_stage1_outputs` (BS32b cache) | **CODE modifié (petit)** : ajout `H_T` au cache pour Stage 2 r_φ | [two_stage.py:881](src/st_cdgm/training/two_stage.py#L881) |
| `CausalDiffusionDecoder` (`causal_concat=True`) | **CODE et données inchangés** | [diffusion_decoder.py](src/st_cdgm/models/diffusion_decoder.py) |
| `train_epoch_stage2_cached` | **CODE modifié (petit)** : ajouter loss terms + r_φ warmup + monitoring live | [two_stage.py:1072](src/st_cdgm/training/two_stage.py#L1072) |
| EMA BS37 (decay 0.9999) | **CODE et données inchangés** | two_stage.py |
| **Monitoring live r_φ + rank** | **NOUVEAU V6** — `src/st_cdgm/evaluation/refiner_monitor.py` | — |
| **OOD eval EC-Earth3** | **NOUVEAU V6** — script `scripts/eval_v6_ood_ecearth3.py` | nouveau |
| **CRPS multi-échelle** (optionnel) | **NOUVEAU V6** — `src/st_cdgm/evaluation/crps_multiscale.py` | nouveau |
| **do(SST+2K) test causal** (optionnel) | **NOUVEAU V6** — `scripts/eval_v6_intervention_sst.py` | nouveau |
| **Pré-enregistrement seuils** | **NOUVEAU V6** — `path_c_plus/audit/V6_MVP_seuils_preregistered.json` | nouveau |

### 6.3 — Tensor shapes complets

| Étape | Tensor | Shape | Espace |
|---|---|---|---|
| Input LR | `lr` | `[T_seq, B, C_LR=21, H_LR, W_LR]` | normalisé z-score |
| Input HR cible | `target` | `[T_seq, B, 1, 172, 179]` | log1p(mm/d) |
| Static HR | `static_HR` | `[B, C_static, 172, 179]` (incl. mask_ocean) | brut |
| Hetero graph | `data` | HeteroData 11 dyn nodes | — |
| Encoder out | `H_0` | `[B, 11, hidden_dim]` | latent |
| **RCN export (A1)** | `H_T = seq_out.states[-1]` | `[B, 11, hidden_dim]` | latent |
| RegressionHead out | `mu_HR` | `[B, 1, 172, 179]` | log1p résidu |
| **r_φ output (NEW)** | `r_φ_out = StructuredResidualHead(H_T)` | `[B, 1, 172, 179]` | log1p contribution |
| Stage 2 EDM out | `D_y` | `[B, 1, 172, 179]` | log1p résidu |
| **HR final (NEW combo)** | `hr_log = D_y + mu_HR + baseline_log + λ_r · r_φ_out` | `[B, 1, 172, 179]` | log1p(mm/d) |
| mm/d final | `hr_mm = expm1(hr_log)` | `[B, 1, 172, 179]` | mm/d |
| **Log-det subsample** | `cov_sub` | `[K=384, K=384]` (au lieu de 30k×30k) | covariance batch |

---

## 7) Critères de succès / décision (pré-enregistrés)

À commit dans `path_c_plus/audit/V6_MVP_seuils_preregistered.json` AVANT lancement Stage 2 full.

| Critère | Seuil | Action |
|---|---|---|
| F1@p99 Conv B (pooled) | ≥ 0.480 | PASS minimal (bat V5_causal seed 42 = 0.453) |
| F1@p99 Conv B (pooled) | ≥ 0.550 | **PASS CIBLE** (égale noncausal_v4) |
| F1@p99 Conv B (pooled) | ≥ 0.580 | PASS STRONG (bat noncausal +5%) |
| RMSE Conv B | dégradation ≤ 5% vs V5_causal | non-régression |
| Pearson global Conv B | dégradation ≤ 3% vs V5_causal | non-régression |
| OOD EC-Earth3 F1@p99 | ≥ 0.40 | robustesse OOD (sinon FAIL) |
| Ablation r_φ contribution | ≤ 40% du gain F1 | si > 40% : REJET (gain vide scientifiquement, ML) |
| ‖r_φ‖/‖μ_θ‖ epoch 5 | ≥ 0.05 | sinon RED FLAG (collapse) |
| do(SST+2K) réponse | ~7%/K (Clausius-Clapeyron, Pall 2007) | validation causale réelle |

**Probabilité moyenne battre noncausal F1@p99=0.550 (5/5 experts) : ~57%.**

---

## 8) Ce que ce plan N'EST PAS

- Ce n'est PAS un retour à V6 boost UNet refiner (abandonné — sur-ingénié, Phase 6 v2)
- Ce n'est PAS une re-construction from-scratch (cf. memory `feedback_dont_rebuild_from_scratch_inherit_safeguards`)
- Ce n'est PAS une modification du protocole Stage 1.A 9-node seed 42 (modulo λ_l1 -30% pour 11-node)
- Ce n'est PAS un changement du protocole Stage 2 EDM (modulo 3 ajouts orthogonaux + 7 garde-fous)
- Ce n'est PAS un changement de l'éval `_eval_3way_dual_convention.ipynb` (modulo slot Phase 8 → V6 MVP)

---

## 9) Références scientifiques

| Référence | Contribution au plan |
|---|---|
| **F4 ECHECS_ET_LECONS.md** | r_φ(H) = dette critique non testée — Math Prof §14.3 |
| Math Prof §14.3 | r_φ(H) auxiliary head spec original |
| Holton 2004 | w_700 interp pondérée pression |
| Bolton 1980 | θ_e formule simplifiée |
| Browning 2004 *QJRMS* | θ_w_850 warm conveyor belt (optionnel §4.4) |
| Elvidge & Renfrew 2016 *BAMS* 97:455 | u850·∇h_HR résolution-cohérent |
| Friedman 2001 *Ann. Stat.* | Boosting (réf cadre théorique r_φ) |
| He 2016 ResNet | Residual learning (réf B2 formulation) |
| Cover & Thomas 2.8.1 | Data-processing inequality (H_T info vs μ_HR) |
| Koenker-Bassett 1978 | Pinball loss |
| Gneiting-Raftery 2007 *JASA* | CRPS (§4.1 optionnel) |
| Reid 2021 *Weather Clim. Extremes* | EC-Earth3 OOD discriminant pour ARs Tasman |
| Salinger 2013 *Int. J. Climatol.* | u850/v850 ARs NZ orographique |
| Henderson-Thompson 1999 *Weather and Climate* | Rx1day West Coast >80% stratiforme orographique |
| Pall 2007 *Nature* | Clausius-Clapeyron 7%/K (validation do(SST+2K)) |
| Mardani 2024 *Nature Comm. Earth & Environ.* | CorrDiff cascade diffusion (réf architecture) |
| Rampal 2024 *Geosci. Model Dev.* 17:3873 | NowcastNet-NZ benchmark (réf NZ-spécifique) |
| Kerr 1998 *Pers. Soc. Psychol. Rev.* 2:196 | HARKing (justification §3.5 pré-enregistrement) |
| Wolpert 1992 *Neural Networks* | Stacking (rebaptisation paradigm r_φ) |

---

## 10) Décision attendue

Plan validé unanimement (5/5 GO) après 4 rondes de consultation experts.
Probabilité moyenne battre noncausal F1@p99=0.550 : **~57%**.

Compute estimé : **~32h** (avec ajouts optionnels CRPS + do(SST+2K)).

**Prêt pour la phase d'implémentation (S0 → S7) dès validation finale user.**
