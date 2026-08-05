# Phase 8 — ST-CDGM **from-scratch** intégrant 10+ fixes experts

Architecture from-scratch validée par 5 experts (ML/Math/Recherche/IA/Climat). Vise à **battre noncausal v4 sur la majorité des métriques**.

## Probabilités empiriques estimées (basées sur V5 mesuré)

| Métrique | Proba battre noncausal | Justification |
|----------|------------------------|---------------|
| Pearson, RMSE, MAE, CRPS, RAPSD | **70-85%** | V5 le fait déjà ; Min-SNR + features physiques aident |
| SSR (calibration) | 50-60% | Multi-EMA + cond_dropout + Dispersive Loss |
| **F1@p99** | **10-30%** | Verdict expert (17/17 variants échec, 0 précédent publié) |
| CSI/SEDI/FSS | 40-60% | Suivent F1 ; FSS tolère décalages spatiaux |
| Indices climatiques (CDD, R10, RX1) | 60-75% | V5 déjà mieux sur CDD/R10 ; tail_weight raffine |

## Architecture (fixes intégrés)

### Stage 1 (causal mean predictor, from-scratch)
- 15 features existantes + **w_700** (interp pondérée 0.571·w_850 + 0.429·w_500, Holton 2004)
- + **θ_e_850, θ_e_500** (Bolton 1980)
- + **MUCAPE proxy = θ_e_850 − θ_e_500** (remplace CAPE 2-niveaux trop grossier, Emanuel 1994)
- Encoder + RCN + DAG learnable + dual_path Path B UNet
- **Losses physiques différentiables corrigées** :
  - `L_R10mm = MSE(Σ_t σ(5·(expm1(x)-10)), Σ_t 1[expm1(x)>10])` ← seuil en mm/day après expm1
  - `L_Rx1day = LSE_T=5(expm1(pred)) approx max` ← LogSumExp stable
  - `L_CDD = MSE(σ(5·(1-expm1(x))), 1[expm1(x)<1])`
  - `L_CC = autograd((∂μ/∂T_850)·σ_T/σ_μ − 0.07)²` ← adimensionné, autograd obligatoire
  - Pondération : 0.20·R10 + 0.15·Rx1 + 0.10·CDD + 0.05·CC, warmup epoch 10+

### Stage 2 (diffusion EDM)
- UNet 4 niveaux [128,256,256,256] CorrDiff Normal (~50M params)
- **Conditioning** : `causal_concat=True` (fallback documenté — AdaGN reporté à V9 cf. réserve IA)
- **Min-SNR-γ=5** weighting EDM (Hang ICCV 2023, arXiv 2303.09556)
- **Tail_weight (4, 12)** au lieu de (8, 25) — Climate ML
- **Dispersive Loss** λ=0.25 (corrigé vs 0.05 sous-dosé) mid-block hook (He&Wang 2025, arXiv 2506.09027)
- **conditioning_dropout p=0.13** (Ho&Salimans 2022)
- **Multi-EMA** {0.999, 0.9995, 0.9999} avec post-hoc sweep (Karras 2024, arXiv 2312.02696)
- **α appris ∈ [0,1]** avec régularisation `L_α = 0.1·(α-0.5)² + 1.0·max(0, 0.3-α)²` (évite α→0 collapse)

### Sampling
- `dpm_solver++` 32 steps (Lu 2022, arXiv 2211.01095)
- **Limited-Interval Guidance** σ ∈ [0.05, 1.0] (Kynkäänniemi 2024, arXiv 2404.07724)
- cfg_scale 1.0-1.5

### Eval protocol (publication-ready)
- **N_BATCHES = 64**, **K_SAMPLES = 128** (CI ±0.014)
- 3 conventions F1@p99 : pooled full-grid (vs noncausal), ETCCDI per-pixel land, land-only pooled
- CSI@p99, SEDI@p99, FSS (n=9, n=25, n=51 px)
- CRPS gaussian, rank histogram, RMSE, MAE, Pearson global+per-sample
- **Indices climatiques sur 730 jours ENTIERS** (rx1day, CDD, R10mm, DJF/JJA) — ETCCDI Zhang 2011
- μ_HR ablation (causalité opérationnelle), Q_phys (interprétabilité), α appris final
- **Paired permutation test** n=10000 (Phase 8 vs noncausal sur mêmes batches)
- **Bootstrap CI 95% BCa** n=1000 (Math)
- **Holm-Bonferroni** correction multi-tests
- **Pre-enregistrement** : git rev-parse HEAD logged dans JSON output

## Coût compute estimé (A100 Pro+)
- Stage 1 : ~6-7h, 15 epochs
- Stage 2 : ~22-27h, 200 epochs (cached)
- Eval BS30 unifié + comparaison 3-way : ~15h
- **Total : ~45-55h, soit 2-3 sessions Pro+ avec checkpoint/resume**