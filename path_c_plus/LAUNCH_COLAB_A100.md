# Lancement Path C+ sur Colab A100 — Guide concret

**Hardware target** : Google Colab Pro+ A100 (40GB ou 80GB selon disponibilité)
**Date** : 2026-06-12

## Pourquoi Colab A100 (vs Lambda/Vast)

Choix utilisateur : workflow Colab déjà rodé, Drive intégré, pas de nouveau compte à créer.

**Tradeoffs acceptés** :
- Colab disconnect risk (~12h max session) — mitigé par resume logic robuste (Phase 0 fix §1.4)
- A100 availability variable — peut tomber sur V100 si A100 indispo
- Pricing par compute units (~$1.30/h équivalent A100 80GB)

## Modifications config A100 vs T4

Le code restera **T4-compatible** (dev/debug). Les optimisations A100 sont des **flags activables** dans la config.

### Fichier : `config/training_config.yaml` — section `training`

```yaml
training:
  # Ces 4 lignes sont les changements A100 (sans casser T4 compat)
  num_workers: 8                # ÉTAIT 0 ; T4 supporte 2-4, A100 8-12
  pin_memory: true              # AJOUTER ; A100 + NVMe local sur Colab
  amp_dtype: "bfloat16"         # AJOUTER ; T4 forcera fp16 automatiquement
  use_amp: true                 # déjà OK
  
  # batch_size scaling automatique :
  batch_size_t4: 8              # T4 16GB limit
  batch_size_a100_40: 32        # A100 40GB
  batch_size_a100_80: 64        # A100 80GB
  
  # torch.compile (A100 only — T4 sm_75 a moins de gain)
  compile:
    enabled: true
    diffusion_mode: "default"   # éviter inductor pad_mm bug
    encoder_mode: "default"
    rcn_mode: "default"
```

### Detection automatique GPU + batch_size

Helper à ajouter dans le bootstrap :

```python
# path_c_plus/scripts/gpu_detect.py
import torch

def detect_gpu_profile():
    if not torch.cuda.is_available():
        return {"profile": "cpu", "batch_size": 4}
    
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    sm = (props.major, props.minor)
    
    if sm[0] >= 8 and vram_gb >= 70:
        # A100 80GB ou H100
        return {"profile": "a100_80", "batch_size": 64, "amp_dtype": "bfloat16"}
    elif sm[0] >= 8 and vram_gb >= 35:
        # A100 40GB
        return {"profile": "a100_40", "batch_size": 32, "amp_dtype": "bfloat16"}
    elif sm == (7, 5):
        # T4
        return {"profile": "t4", "batch_size": 8, "amp_dtype": "float16"}
    else:
        # V100, P100, etc.
        return {"profile": "other", "batch_size": 16, "amp_dtype": "float16"}
```

## Workflow Colab A100 par phase

### Phase A0' — RE-EVAL baseline V5-mini corrigé

**Notebook** : `path_c_plus/scripts/colab_a100_phase_a0_reeval.ipynb`

**Compute attendu** : ~5-6h sur A100 (≈ 65-80 units Colab)

```python
# Cell 1 : Bootstrap (clone repo, install deps)
!git clone -b four-node-causal https://github.com/leonelkenfack/stcdgm.git
%cd stcdgm
!pip install -e . -q

# Cell 2 : Detect GPU + load config
import sys; sys.path.insert(0, 'src')
from path_c_plus.scripts.gpu_detect import detect_gpu_profile
GPU_PROFILE = detect_gpu_profile()
print(f"Profile: {GPU_PROFILE}")

# Cell 3 : Eval V5-mini avec protocole corrigé (3 seeds)
from scripts.recompute_phase6_metrics import recompute_phase6_metrics
for seed in [42, 7, 123]:
    result = recompute_phase6_metrics(
        stack=stack_v5_baseline,
        builder=builder,
        val_dataset=val_dataset_temporal_held_out,  # K9 fix : temporal split
        DEVICE=DEVICE,
        convert_sample_to_batch_fn=convert_sample_to_batch,
        out_path=Path(f"results/baseline_corrected/seed_{seed}/metrics.json"),
        run_variant="causal",
        K_samples=64,
        n_steps=32,
        n_batches=64,  # K4 fix : pas 16
        seed=seed,     # K16 fix
        cfg_scale=1.5,
        scheduler_type="edm_karras",
    )

# Cell 4 : Bootstrap CI 95% sur les 3 seeds
from path_c_plus.scripts.stats_utils import bootstrap_ci_3seeds
ci_results = bootstrap_ci_3seeds("results/baseline_corrected/")
```

### Phase A0'' — RE-TRAIN V5-mini propre 3 seeds

**Notebook** : `path_c_plus/scripts/colab_a100_phase_a0_retrain.ipynb`

**Compute attendu** : 5-8h × 3 seeds = 15-24h A100 (≈ 195-310 units)

Stratégie : 1 seed = 1 session Colab (≤ 12h max session). 3 sessions séquentielles.

```python
# Per session:
# Cell 1 : Bootstrap + load Phase 0 fixed code
# Cell 2 : Verify P0 fixes applied (run pre-flight tests)
!pytest path_c_plus/tests/ -v

# Cell 3 : Resume-aware training (RNG state, atomic save, fsync)
from src.st_cdgm.training.two_stage import train_v5_mini_full
result = train_v5_mini_full(
    seed=SEED_FOR_THIS_SESSION,
    epochs=200,
    batch_size=GPU_PROFILE["batch_size"],
    amp_dtype=GPU_PROFILE["amp_dtype"],
    use_torch_compile=True,
    ckpt_save_dir=Path(f"/content/drive/MyDrive/climate_data/v5mini_corrected/seed_{SEED}"),
    save_every=5,
    fsync_every_save=True,  # Colab Drive
)
```

### Phase A1 — 6-node minimal fix

**Notebook** : `path_c_plus/scripts/colab_a100_phase_a1_minimal.ipynb`

**Compute attendu** : 2.5-5h × 3 seeds = 7.5-15h A100 (≈ 100-195 units)

3 sessions Colab séquentielles, 1 seed/session.

### Phase B0 — PCMCI offline (CPU)

**Important** : PCMCI est **CPU-bound**, A100 ne sert à rien ici. Utiliser Colab session CPU normale.

**Compute attendu** : 4-8h CPU (gratuit ou Pro $10/mois)

### Phase B1-B6 — Path C+ full

Si A1 fail, on continue. Compute attendu : 30-60h A100 total sur 3-4 semaines.

## Persistance & resume — critique pour Colab

Toutes les phases utilisent le pattern de persistance déjà éprouvé dans `finetune_stage1_bundle_b.py` (§1.4 fixé) :

```
ORACLE_FINETUNED_DIR/
  ├── epoch_inprogress.pth      # atomic write via tmp + os.replace + fsync
  ├── epoch_finetuned.pth        # final after convergence
  ├── epoch_last.pth             # alias = epoch_finetuned
  ├── training_history.json      # per-epoch logs
  └── rng_state.pkl              # torch + numpy + python random (Phase 0 §1.4 fix)
```

**Disconnect detection** :
- Si `epoch_inprogress.pth` existe au démarrage → resume depuis cet epoch
- RNG state restauré → batches identiques au run interrompu
- Persistance toutes les 5 epochs (compromis I/O vs resilience)

## Monitoring obligatoire pendant runs

Per Phase 0 instrumentation list :

```python
# Logging toutes les N steps :
metrics_log = {
    "epoch": epoch,
    "step": step,
    "loss_data": loss_data.item(),
    "loss_dag": loss_dag.item(),
    "loss_l1": loss_l1.item(),
    "loss_phys": loss_phys.item(),
    "loss_castle": loss_castle.item(),
    "A_dag_frobenius": float(rcn_cell.A_dag.norm()),
    "A_dag_asymmetry": float((A - A.T).norm() / (A.norm() + 1e-9)),
    "A_dag_nnz_above_0.01": int((A.abs() > 0.01).sum()),
    "Q_phys_running": q_phys_weighted(A.cpu().numpy(), G_phys.numpy()),
    "spectral_radius": float(torch.linalg.eigvals((A**2).abs()).abs().max()),
    "gradient_ratio_g_A_dag_to_g_driver_enc": g_A / max(g_drv, 1e-8),  # detect decorative
    "kkt_residual_phys_edges": kkt_resid_phys,
    "kkt_residual_non_phys_edges": kkt_resid_non_phys,
    "loss_edm_bin0_low_sigma": ...,  # 5 sigma bins
    "loss_edm_bin1": ..., ...,
    "dag_grad_gate_value": float(rcn_cell.dag_grad_gate),
    "amp_effective_dtype": str(autocast_dtype),
    "vram_used_gb": torch.cuda.memory_allocated() / 1024**3,
}
```

Logged to `training_history.json` + W&B (optional) + TensorBoard (optional).

## Décision gates avant chaque phase

| Gate | Critère GO | Action si NO-GO |
|---|---|---|
| G0 : Phase 0 → A0' | All P0 fixes + unit tests pass + encoder CI corr < 0.5 | HeteroConv refactor obligatoire |
| G1 : A0' → A0'' | Noise floor CIs documented | n/a (always proceed) |
| G2 : A0'' → A1 | V5-mini retrain converged (loss plateau) | Investigate divergence |
| G3 : A1 → publish OR C+ | Q_phys ≥ 0.65 mean 3 seeds AND RMSE ≤ +5% | Proceed to Path C+ |
| G4 : B0 → B1 | PCMCI bootstrap Jaccard > 0.80 | Fallback G_phys init seul |
| G5 : B3 smoke → B4 | Q_phys epoch 5 > init + 0.05, edge_gate < 0.8 | Debug 1-2j |
| G6 : B4 → B5 | Q_phys final 3 seeds ≥ 0.55, RMSE ≤ +10% | Abandon C+, fallback A1 |
| G7 : B5 → B6 | Stage 2 val loss < baseline + 5% | Re-tune P_mean/P_std |

## Compute budget tracking

Fichier `path_c_plus/compute_log.csv` à maintenir :

```csv
date,phase,seed,gpu_profile,start_time,end_time,hours,units_consumed,checkpoint_size_gb,status
2026-06-14,A0_reeval,42,a100_40,10:00,15:30,5.5,72,2.1,SUCCESS
...
```

## Backup strategy

- Drive : `/content/drive/MyDrive/climate_data/v6_path_c_plus/` — checkpoints
- GitHub : push branch `four-node-causal` après chaque phase
- Local : `git pull` régulier vers la machine de dev

## Troubleshooting Colab A100

| Problème | Cause probable | Solution |
|---|---|---|
| "GPU not available" au launch | A100 quota peak hours | Attendre ou se reconnecter en off-peak (3am UTC) |
| Disconnect après 11h | Session timeout Pro+ | Pas grave : resume logic + RNG state |
| OOM au batch_size=64 | A100 40GB pas 80GB | Auto-detect dans gpu_detect.py → bs=32 |
| Drive sync slow | FUSE Colab issue | `pin_memory=True` aide partiellement |
| BF16 NaN sur slogdet | DAGMA M-matrix degenerate | Fallback FP32 sur slogdet (§1.2 fix) |
| `torch.compile` cache miss | Premier run | Compile 5-10 min, normal |

## Pre-launch checklist

Avant de lancer Phase A0' :

- [ ] Branch `four-node-causal` créée et pushée
- [ ] 22 P0 fixes commits sur la branch
- [ ] Unit tests verts : `pytest path_c_plus/tests/`
- [ ] `path_c_plus/PRE_REGISTRATION.md` signé et commité
- [ ] Compute budget Colab vérifié (Pro+ active ou crédits suffisants)
- [ ] Backup Drive folder `/content/drive/MyDrive/climate_data/v6_path_c_plus/` créé
- [ ] V5-mini baseline checkpoint sauvegardé (lecture seule, jamais écrasé)
- [ ] Smoke test 5-epoch sur T4 OK (validate code works before A100 spend)
