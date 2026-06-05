"""Patch st_cdgm_v5_evaluation.ipynb :
- Ajoute Cell 2.5 : bootstrap autonome (CONFIG, DEVICE, builder, convert, dataset ACCESS)
- Reecrit Phase 7 (Cell 6 -> 7 apres insertion) : VRAI test OOD EC-Earth3 + NorESM2-MM
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

NB = Path("st_cdgm_v5_evaluation.ipynb")
with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak4")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #4 cree : {BACKUP}")

# ─────────────────────────────────────────────────────────────
# NOUVELLE CELLULE 2.5 — Bootstrap autonome (CONFIG, builder, etc.)
# ─────────────────────────────────────────────────────────────
BOOTSTRAP_MD = """---

## Bootstrap autonome — charge CONFIG / builder / DEVICE / dataset

Cette cellule rend le notebook **autonome** : plus besoin d'exécuter les cellules 14-30 de `st_cdgm_training_evaluation.ipynb` avant. Elle :

1. Charge `CONFIG` depuis `config/training_config.yaml` (+ override `training_config_corrdiff_normal.yaml` si dispo)
2. Définit `DEVICE`, `lr_shape`, `hr_shape`
3. Construit le **`NetCDFDataPipeline` ACCESS-CM2** (training data, pour Phase 6 in-distribution)
4. Définit `builder` (`HeteroGraphBuilder`)
5. Définit `convert_sample_to_batch`
6. Construit `test_dataset` (premier `next(iter(...))` pour smoke test)"""

BOOTSTRAP_CODE = '''# ============================================================
# Bootstrap autonome — Phase 6/7/8 prerequis
# Pas besoin de st_cdgm_training_evaluation cellules 14-30.
# ============================================================
import os
import torch
from pathlib import Path
from omegaconf import OmegaConf

# 1. CONFIG (base + override corrdiff_normal si dispo)
_base = Path("config/training_config.yaml")
_override = Path("config/training_config_corrdiff_normal.yaml")
CONFIG = OmegaConf.load(_base)
if _override.exists():
    CONFIG = OmegaConf.merge(CONFIG, OmegaConf.load(_override))
    print(f"[OK] CONFIG = base + override (corrdiff_normal)")
else:
    print(f"[OK] CONFIG = base seule")

# 2. DEVICE + dimensions
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr_shape = tuple(CONFIG.graph.lr_shape)
hr_shape = tuple(CONFIG.graph.hr_shape)
print(f"[OK] DEVICE = {DEVICE}")
print(f"[OK] lr_shape = {lr_shape}, hr_shape = {hr_shape}")

# 3. Resolution des paths donnees (Drive si Colab, local sinon)
ON_COLAB = "google.colab" in sys.modules or Path("/content").exists()
if ON_COLAB:
    DATA_ROOT = Path("/content/drive/MyDrive/climate_data/data")
    if not DATA_ROOT.exists():
        # Fallback SSD local si pas synchronise sur Drive
        DATA_ROOT = Path("/content/data_local")
else:
    DATA_ROOT = Path("data")

DATA_RAW = DATA_ROOT / "raw"

# Paths ACCESS-CM2 (in-distribution training)
LR_PATH_TRAIN     = DATA_RAW / "train" / "predictor_ACCESS-CM2_hist.nc"
HR_PATH_TRAIN     = DATA_RAW / "train" / "pr_ACCESS-CM2_hist.nc"
STATIC_PATH       = DATA_RAW / "static_predictors" / "ERA5_eval_ccam_12km.198110_NZ_Invariant.nc"
MEAN_PATH         = DATA_RAW / "normalization_coefs" / "mean_1974_2011.nc"
STD_PATH          = DATA_RAW / "normalization_coefs" / "std_1974_2011.nc"

# Paths OOD
LR_PATH_ECEARTH   = DATA_RAW / "test" / "EC-Earth3_histupdated_compressed.nc"
HR_PATH_ECEARTH   = DATA_RAW / "test" / "EC-Earth3_historical_precip_compressed.nc"
LR_PATH_NORESM    = DATA_RAW / "test" / "NorESM2-MM_histupdated_compressed.nc"
HR_PATH_NORESM    = DATA_RAW / "test" / "NorESM2-MM_historical_precip_compressed.nc"

for label, p in [("ACCESS-CM2 LR", LR_PATH_TRAIN), ("ACCESS-CM2 HR", HR_PATH_TRAIN),
                  ("EC-Earth3 LR", LR_PATH_ECEARTH), ("EC-Earth3 HR", HR_PATH_ECEARTH),
                  ("NorESM2-MM LR", LR_PATH_NORESM), ("NorESM2-MM HR", HR_PATH_NORESM),
                  ("Static", STATIC_PATH), ("Mean", MEAN_PATH), ("Std", STD_PATH)]:
    status = "OK" if p.exists() else "MISSING"
    print(f"  [{status}] {label:<18s} {p}")

# 4. Construction du pipeline ACCESS-CM2 (in-distribution, pour Phase 6)
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder

def make_pipeline(lr_path, hr_path):
    """Construit un NetCDFDataPipeline aligne sur CONFIG mais avec paths customs."""
    return NetCDFDataPipeline(
        lr_path=str(lr_path),
        hr_path=str(hr_path),
        static_path=str(STATIC_PATH) if STATIC_PATH.exists() else None,
        seq_len=int(CONFIG.data.seq_len),
        baseline_strategy=str(CONFIG.data.baseline_strategy),
        baseline_factor=int(CONFIG.data.baseline_factor),
        target_transform=str(CONFIG.data.get("target_transform", "log1p")),
        normalize=bool(CONFIG.data.normalize),
        nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
        precipitation_delta=float(CONFIG.data.get("precipitation_delta", 0.01)),
        lr_variables=list(CONFIG.data.lr_variables),
        hr_variables=list(CONFIG.data.hr_variables),
        static_variables=list(CONFIG.data.static_variables) if STATIC_PATH.exists() else None,
        means_path=str(MEAN_PATH) if MEAN_PATH.exists() else None,
        stds_path=str(STD_PATH) if STD_PATH.exists() else None,
        eager_load_datasets=bool(CONFIG.data.get("eager_load_datasets", False)),
    )

pipeline = make_pipeline(LR_PATH_TRAIN, HR_PATH_TRAIN)
print(f"[OK] Pipeline ACCESS-CM2 cree")

# 5. Builder
builder = HeteroGraphBuilder(
    lr_shape=lr_shape,
    hr_shape=hr_shape,
    static_dataset=pipeline.get_static_dataset(),
    include_mid_layer=bool(CONFIG.graph.include_mid_layer),
)
print(f"[OK] Builder cree : {len(builder.dynamic_node_types)} types dynamiques + {len(builder.static_node_types)} statiques")

# 6. convert_sample_to_batch (copie depuis training_evaluation cell 40)
def convert_sample_to_batch(sample, builder, device):
    lr_seq = sample["lr"]
    seq_len = lr_seq.shape[0]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    dynamic_features = {nt: lr_nodes_steps[0] for nt in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_features).to(device)
    batch = {
        "lr": lr_tensor,
        "residual": sample["residual"],
        "baseline": sample.get("baseline"),
        "hetero": hetero,
    }
    return batch

# 7. Dataset ACCESS pour Phase 6 (in-distribution evaluation)
test_dataset = pipeline.build_sequence_dataset(
    seq_len=int(CONFIG.data.seq_len),
    stride=int(CONFIG.data.stride),
    as_torch=True,
)

# Smoke test
sample = next(iter(test_dataset))
print(f"[OK] test_dataset (ACCESS-CM2) : sample keys = {list(sample.keys())}")
print(f"     lr shape = {tuple(sample['lr'].shape)}, residual shape = {tuple(sample['residual'].shape)}")

# Update RCN_DRIVER_DIM in CONFIG si dimension auto-detectee differente
_runtime_driver_dim = int(sample["lr"].shape[1])
if _runtime_driver_dim != int(CONFIG.rcn.driver_dim):
    print(f"[INFO] Override CONFIG.rcn.driver_dim {CONFIG.rcn.driver_dim} -> {_runtime_driver_dim}")
    CONFIG.rcn.driver_dim = _runtime_driver_dim
    CONFIG.rcn.reconstruction_dim = _runtime_driver_dim

print()
print("[OK] Bootstrap complet :")
print(f"     - CONFIG, DEVICE, lr_shape, hr_shape charges")
print(f"     - pipeline, builder, convert_sample_to_batch definis")
print(f"     - test_dataset (ACCESS-CM2 in-dist) pret")
print()
print("Prochaines cellules : Phase 6 (in-dist), Phase 7 (OOD EC-Earth3 + NorESM2-MM), Phase 8 (intervention)")
'''

bootstrap_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [ln + "\n" for ln in BOOTSTRAP_MD.split("\n")[:-1]] + [BOOTSTRAP_MD.split("\n")[-1]],
}

bootstrap_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [ln + "\n" for ln in BOOTSTRAP_CODE.split("\n")[:-1]] + [BOOTSTRAP_CODE.split("\n")[-1]],
}

# Insertion APRES Cell 2 (paths), AVANT Cell 3 (MD Phase 6)
# Position d'insertion : 3 (decale tout ce qui suit)
nb["cells"].insert(3, bootstrap_md_cell)
nb["cells"].insert(4, bootstrap_code_cell)

# Maintenant Phase 7 (qui etait Cell 6) est devenue Cell 8.
# On reecrit la Cell 8 (code) avec un VRAI test OOD.
# Cells: 0=MD intro, 1=setup, 2=paths, 3=bootstrap MD, 4=bootstrap code, 5=MD Phase 6, 6=code Phase 6, 7=MD Phase 7, 8=code Phase 7, ...

PHASE7_REAL = '''# ============================================================
# Phase 7 : EVALUATION OOD REELLE
# Compare V5 vs Noncausal sur 2 GCM differents (EC-Earth3 + NorESM2-MM)
# vs leur baseline in-distribution (ACCESS-CM2).
#
# C'est un test OOD INTER-GCM : le modele a ete entraine sur ACCESS-CM2,
# on l'evalue sur EC-Earth3 et NorESM2-MM (jamais vus). Les biais GCM-
# specifiques de chaque modele source diffusent dans la distribution
# des predicteurs, donc c'est un vrai shift de distribution.
#
# Pour chaque (modele, dataset) on calcule Pearson, RMSE, MAE, Spread.
# Δ_OOD = (Pearson_ACCESS - Pearson_OOD) / Pearson_ACCESS
# Verdict : V5 wins OOD si Δ_OOD(V5) < Δ_OOD(Noncausal).
# ============================================================

import json
import time
import torch
import numpy as np

# Helpers d'evaluation rapide
@torch.no_grad()
def quick_eval(stack, dataset, n_batches=8, K=4, n_steps=18, device=None):
    """Evalue un stack sur un dataset, retourne dict de metriques."""
    if device is None:
        device = DEVICE
    encoder = stack["encoder"]
    rcn_runner = stack["rcn_runner"]
    regression_head = stack["regression_head"]
    diffusion = stack["diffusion"]
    skip_block = stack["skip_block"]
    encoder.eval(); rcn_runner.cell.eval(); regression_head.eval(); diffusion.eval()

    preds_mean, targets, spreads = [], [], []
    sample_iter = iter(dataset)
    for b_idx in range(n_batches):
        try:
            sample = next(sample_iter)
        except StopIteration:
            break
        batch = convert_sample_to_batch(sample, builder, device)
        lr_data = batch["lr"].to(device)
        target_residual = batch["residual"][-1].to(device)
        if target_residual.dim() == 3:
            target_residual = target_residual.unsqueeze(0)
        baseline_t = batch["baseline"][-1].to(device)
        if baseline_t.dim() == 3:
            baseline_t = baseline_t.unsqueeze(0)
        baseline_log = torch.nan_to_num(baseline_t, nan=0.0)

        # Stage 1
        H_init = encoder.init_state(batch["hetero"]).to(device)
        drivers = [lr_data[t] for t in range(lr_data.shape[0])]
        seq_out = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
        mu_HR_causal = regression_head(seq_out.states[-1])
        if mu_HR_causal.shape[-2:] != target_residual.shape[-2:]:
            mu_HR_causal = torch.nn.functional.interpolate(
                mu_HR_causal, size=target_residual.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        if skip_block is not None:
            lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
            mu_HR, _ = skip_block(lr_last, mu_HR_causal)
        else:
            mu_HR = mu_HR_causal
        mu_HR = torch.nan_to_num(mu_HR, nan=0.0)

        # Stage 2 — ensemble K samples
        ens = []
        for _ in range(K):
            out = diffusion.sample(
                conditioning=None, num_steps=n_steps,
                scheduler_type="edm_karras", apply_constraints=False,
                mu_HR=mu_HR, baseline_log=baseline_log,
            )
            residual = out.residual if hasattr(out, "residual") else out
            ens.append((baseline_log + mu_HR + residual).cpu())
        ens = torch.stack(ens, dim=0)  # [K, B, 1, H, W]
        preds_mean.append(ens.nanmean(0))
        spreads.append(ens.std(0).nanmean().item())
        targets.append((baseline_log + target_residual).cpu())

    preds = torch.cat(preds_mean, dim=0)
    tgts = torch.cat(targets, dim=0)

    # Metriques
    flat_p = preds.flatten().float()
    flat_t = tgts.flatten().float()
    mask = torch.isfinite(flat_p) & torch.isfinite(flat_t)
    flat_p, flat_t = flat_p[mask], flat_t[mask]
    p_c = flat_p - flat_p.mean()
    t_c = flat_t - flat_t.mean()
    pearson = float((p_c * t_c).sum() / ((p_c.norm() * t_c.norm()).clamp(min=1e-12)))
    rmse = float((flat_p - flat_t).pow(2).mean().sqrt())
    mae = float((flat_p - flat_t).abs().mean())
    spread = float(np.mean(spreads))

    # F1 extremes (p99)
    thr = torch.quantile(flat_t, 0.99).item()
    yp = (flat_p > thr).float()
    yt = (flat_t > thr).float()
    tp = (yp * yt).sum().item()
    fp = (yp * (1-yt)).sum().item()
    fn = ((1-yp) * yt).sum().item()
    f1_p99 = 2*tp / (2*tp + fp + fn + 1e-12)

    return {
        "pearson": pearson, "rmse": rmse, "mae": mae,
        "spread": spread, "spread_rmse": spread / max(rmse, 1e-12),
        "f1_p99": f1_p99, "n_batches_used": len(preds_mean),
    }


# === 1. Construction des 3 pipelines / datasets ===
print("Construction des 3 datasets (1 in-dist + 2 OOD)...")
print()

# In-distribution : ACCESS-CM2 (deja construit dans le bootstrap)
ds_access = test_dataset
print(f"[OK] ACCESS-CM2 (in-dist) : pipeline deja construit")

# OOD #1 : EC-Earth3
if LR_PATH_ECEARTH.exists() and HR_PATH_ECEARTH.exists():
    pipe_ecearth = make_pipeline(LR_PATH_ECEARTH, HR_PATH_ECEARTH)
    ds_ecearth = pipe_ecearth.build_sequence_dataset(
        seq_len=int(CONFIG.data.seq_len),
        stride=int(CONFIG.data.stride),
        as_torch=True,
    )
    print(f"[OK] EC-Earth3 (OOD #1) : pipeline + dataset crees")
else:
    ds_ecearth = None
    print(f"[SKIP] EC-Earth3 absent")

# OOD #2 : NorESM2-MM
if LR_PATH_NORESM.exists() and HR_PATH_NORESM.exists():
    pipe_noresm = make_pipeline(LR_PATH_NORESM, HR_PATH_NORESM)
    ds_noresm = pipe_noresm.build_sequence_dataset(
        seq_len=int(CONFIG.data.seq_len),
        stride=int(CONFIG.data.stride),
        as_torch=True,
    )
    print(f"[OK] NorESM2-MM (OOD #2) : pipeline + dataset crees")
else:
    ds_noresm = None
    print(f"[SKIP] NorESM2-MM absent")

print()

# === 2. Chargement V5 + Noncausal stacks (si pas deja fait) ===
if "stack_v5" not in globals() or "stack_nc" not in globals():
    print("Chargement des stacks V5 + Noncausal (peut prendre 2-3 min)...")
    # On reutilise build_stack_from_ckpt si deja defini (Phase 8 a tourne)
    # sinon on doit l'importer ici
    if "build_stack_from_ckpt" not in globals():
        # Inline minimal du loader
        from st_cdgm.models import (
            IntelligibleVariableEncoder, IntelligibleVariableConfig,
            GraphToGridDecoder, RCNCell, RCNSequenceRunner,
            CausalDiffusionDecoder,
        )
        try:
            from st_cdgm.models import ConditionalSkipBlock
            SKIP_AVAILABLE = True
        except ImportError:
            SKIP_AVAILABLE = False
        from st_cdgm.models.edm_preconditioner import EDMConfig

        def build_stack_from_ckpt(ckpt_path, variant_name):
            print(f"  [{variant_name}] Loading {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
            enc_cfg = IntelligibleVariableConfig(
                num_variables=len(CONFIG.data.lr_variables),
                hidden_dim=CONFIG.encoder.hidden_dim,
                conditioning_dim=CONFIG.encoder.conditioning_dim,
                num_dag_tokens=int(CONFIG.encoder.get("num_dag_tokens", 2)),
                causal_conditioning=bool(CONFIG.encoder.get("causal_conditioning", True)),
            )
            encoder = IntelligibleVariableEncoder(enc_cfg).to(DEVICE)
            rcn_cell = RCNCell(
                num_variables=enc_cfg.num_variables,
                hidden_dim=CONFIG.rcn.hidden_dim,
                driver_dim=CONFIG.rcn.driver_dim,
                reconstruction_dim=CONFIG.rcn.reconstruction_dim,
                dropout=CONFIG.rcn.dropout,
            ).to(DEVICE)
            rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.detach_interval)
            regression_head = GraphToGridDecoder(
                d_model=CONFIG.encoder.hidden_dim,
                hr_h=CONFIG.graph.hr_shape[0], hr_w=CONFIG.graph.hr_shape[1],
            ).to(DEVICE)
            edm_cfg = EDMConfig.from_yaml_dict(CONFIG.diffusion.get("edm", {}))
            diffusion = CausalDiffusionDecoder(
                in_channels=CONFIG.diffusion.in_channels,
                conditioning_dim=CONFIG.diffusion.conditioning_dim,
                height=CONFIG.diffusion.height, width=CONFIG.diffusion.width,
                scheduler_type=CONFIG.diffusion.scheduler_type,
                causal_concat=True,
                edm_config=edm_cfg,
                unet_kwargs=dict(CONFIG.diffusion.unet_kwargs),
            ).to(DEVICE)
            for name, mod in [("encoder", encoder), ("rcn_cell", rcn_cell),
                              ("regression_head", regression_head), ("diffusion", diffusion)]:
                key = f"{name}_state_dict"
                if key in ckpt:
                    mod.load_state_dict(ckpt[key])
            skip_block = None
            if SKIP_AVAILABLE and "skip_block_state_dict" in ckpt:
                skip_block = ConditionalSkipBlock(
                    lr_channels=len(CONFIG.data.lr_variables),
                    hr_shape=tuple(CONFIG.graph.hr_shape),
                ).to(DEVICE)
                skip_block.load_state_dict(ckpt["skip_block_state_dict"])
            return {"encoder": encoder, "rcn_runner": rcn_runner,
                    "regression_head": regression_head, "diffusion": diffusion,
                    "skip_block": skip_block, "variant": variant_name}

    stack_v5 = build_stack_from_ckpt(V5_DIR / "epoch_last.pth", "V5")
    stack_nc = build_stack_from_ckpt(NONCAUSAL_DIR / "epoch_last.pth", "Noncausal")
else:
    print("[OK] Stacks deja charges en memoire (Phase 8 a deja tourne)")
print()

# === 3. Eval sur les 3 datasets ===
N_BATCHES_OOD = 4   # 4 batches x K=4 = 16 samples par (stack, dataset) — boost si tu as du temps
K_OOD = 4

results_all = {}

for ds_name, ds in [("ACCESS-CM2_in_dist", ds_access),
                     ("EC-Earth3_OOD", ds_ecearth),
                     ("NorESM2-MM_OOD", ds_noresm)]:
    if ds is None:
        continue
    print(f"=== Eval sur {ds_name} ({N_BATCHES_OOD} batches x K={K_OOD}) ===")
    for stack_name, stack in [("V5", stack_v5), ("Noncausal", stack_nc)]:
        t0 = time.time()
        m = quick_eval(stack, ds, n_batches=N_BATCHES_OOD, K=K_OOD, n_steps=18)
        dt = time.time() - t0
        key = f"{stack_name}_{ds_name}"
        results_all[key] = m
        print(f"  {stack_name:<10s} : Pearson={m['pearson']:+.4f}  RMSE={m['rmse']:.4f}  "
              f"MAE={m['mae']:.4f}  Spread/RMSE={m['spread_rmse']:.3f}  F1-p99={m['f1_p99']:.3f}  ({dt:.0f}s)")
    print()

# === 4. Calcul Delta OOD ===
def delta_ood(in_dist_pearson, ood_pearson):
    if in_dist_pearson < 1e-6:
        return float("nan")
    return (in_dist_pearson - ood_pearson) / in_dist_pearson

print("=" * 70)
print("RECAPITULATIF — Delta_OOD = (Pearson_in - Pearson_ood) / Pearson_in")
print("=" * 70)
ood_results = {}
for stack_name in ["V5", "Noncausal"]:
    in_dist = results_all.get(f"{stack_name}_ACCESS-CM2_in_dist", {}).get("pearson", float("nan"))
    if in_dist != in_dist:
        continue
    print(f"\n{stack_name} :")
    print(f"  Pearson in-dist (ACCESS-CM2) : {in_dist:.4f}")
    for ds_name in ["EC-Earth3_OOD", "NorESM2-MM_OOD"]:
        ood = results_all.get(f"{stack_name}_{ds_name}", {}).get("pearson", float("nan"))
        if ood == ood:
            d = delta_ood(in_dist, ood)
            ood_results[f"{stack_name}_{ds_name}_delta"] = d
            print(f"  Delta_OOD ({ds_name}) : {d:+.4f}  (Pearson OOD = {ood:.4f})")

# Verdict
print()
print("=" * 70)
print("VERDICT")
print("=" * 70)
for ds_name in ["EC-Earth3_OOD", "NorESM2-MM_OOD"]:
    d_v5 = ood_results.get(f"V5_{ds_name}_delta")
    d_nc = ood_results.get(f"Noncausal_{ds_name}_delta")
    if d_v5 is not None and d_nc is not None:
        winner = "V5" if d_v5 < d_nc else "Noncausal"
        print(f"  {ds_name:<20s} : Delta V5={d_v5:+.4f} vs Noncausal={d_nc:+.4f}  -> {winner} resiste mieux")

# === 5. Sauvegarde ===
out = RESULTS_DIR / "phase7_ood_real.json"
out.write_text(json.dumps({
    "datasets": {
        "ACCESS-CM2 (in-dist, training)": str(LR_PATH_TRAIN),
        "EC-Earth3 (OOD #1)": str(LR_PATH_ECEARTH),
        "NorESM2-MM (OOD #2)": str(LR_PATH_NORESM),
    },
    "n_batches": N_BATCHES_OOD,
    "K_samples": K_OOD,
    "metrics_per_run": results_all,
    "delta_ood_per_pair": ood_results,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print()
print(f"[OK] Phase 7 OOD reel sauvegarde : {out}")
'''

phase7_md_new = """---

## Phase 7 — Évaluation OOD **réelle** sur EC-Earth3 et NorESM2-MM

Test **inter-GCM** : on évalue V5-mini et Noncausal — entraînés sur ACCESS-CM2 — sur deux autres modèles climat globaux (EC-Earth3 et NorESM2-MM) jamais vus à l'entraînement. C'est du vrai changement de distribution (les biais GCM-spécifiques diffèrent entre modèles sources).

**3 datasets** :
- ACCESS-CM2 (in-distribution, 1960-2014)
- EC-Earth3 (OOD #1, 1986-2005)
- NorESM2-MM (OOD #2, 1986-2005)

**Métriques** : Pearson, RMSE, MAE, Spread/RMSE, F1-p99 sur chaque (modèle × dataset).

**Δ_OOD** = `(Pearson_in - Pearson_ood) / Pearson_in` pour chaque modèle et chaque OOD.

**Hypothèse à valider** : `Δ_OOD(V5) < Δ_OOD(Noncausal)` sur au moins un des deux OOD."""

phase7_md_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [ln + "\n" for ln in phase7_md_new.split("\n")[:-1]] + [phase7_md_new.split("\n")[-1]],
}

phase7_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [ln + "\n" for ln in PHASE7_REAL.split("\n")[:-1]] + [PHASE7_REAL.split("\n")[-1]],
}

# Apres insertion bootstrap (Cell 3+4), les cells de Phase 7 etaient cells 7+8 -> maintenant 7+8 (decalees de 2 chacune)
# Verifions positions actuelles
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if "Phase 7" in src[:100]:
        print(f"  Cell {i} contient 'Phase 7' (type={c['cell_type']})")

# Find Phase 7 MD and code cells and replace
phase7_md_idx = None
phase7_code_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    if c["cell_type"] == "markdown" and "Phase 7" in src and "Évaluation OOD" in src:
        phase7_md_idx = i
    if c["cell_type"] == "code" and "OOD test" in src and phase7_code_idx is None:
        phase7_code_idx = i

if phase7_md_idx is not None:
    nb["cells"][phase7_md_idx] = phase7_md_cell
    print(f"[OK] Phase 7 MD remplace (Cell {phase7_md_idx})")
if phase7_code_idx is not None:
    nb["cells"][phase7_code_idx] = phase7_code_cell
    print(f"[OK] Phase 7 code remplace par eval OOD reel (Cell {phase7_code_idx})")

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print(f"[OK] Notebook final : {NB} ({len(nb['cells'])} cellules)")
