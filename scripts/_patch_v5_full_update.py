"""Patch complet st_cdgm_v5_evaluation.ipynb :
- Insertion bootstrap autonome (cells 3+4 nouvelles)
- Phase 7 reecrite : 4 runs OOD reels (V5+NC x EC-Earth3+NorESM2-MM) avec run_aligned_eval
- Phase 8 reecrite : interpretabilite visuelle (DAG, interventions, sensibilite, alpha)
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak5")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #5 cree : {BACKUP}")


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in text.split("\n")[:-1]] + [text.split("\n")[-1]],
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in text.split("\n")[:-1]] + [text.split("\n")[-1]],
    }


# ───────────────────────────────────────────────────────────────────
# CELLULE BOOTSTRAP AUTONOME (insertion apres Cell 2 — paths)
# ───────────────────────────────────────────────────────────────────
BOOTSTRAP_MD = """---

## Bootstrap autonome — CONFIG / pipeline / stacks

Cette cellule rend le notebook entièrement **autonome** : plus besoin d'exécuter les cellules du notebook training avant.

Elle :
1. Charge `CONFIG` (base + override `corrdiff_normal`)
2. Définit `DEVICE`, `lr_shape`, `hr_shape`
3. Construit le `NetCDFDataPipeline` ACCESS-CM2 (in-distribution)
4. Définit `builder` et `convert_sample_to_batch`
5. Charge les **deux stacks complets** : V5 et Noncausal (encoder + RCN + regression_head + skip_block + diffusion)
6. Définit `predict_with_stack()` réutilisé par toutes les phases

Coût : ~3 min (chargement Drive + checkpoints)"""

BOOTSTRAP_CODE = '''# ============================================================
# Bootstrap autonome — charge CONFIG + pipelines + stacks
# ============================================================
import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf

ON_COLAB = "google.colab" in sys.modules or Path("/content").exists()

# 1. CONFIG
_base = Path("config/training_config.yaml")
_override = Path("config/training_config_corrdiff_normal.yaml")
CONFIG = OmegaConf.load(_base)
if _override.exists():
    CONFIG = OmegaConf.merge(CONFIG, OmegaConf.load(_override))
    print("[OK] CONFIG = base + corrdiff_normal override")

# 2. DEVICE / shapes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr_shape = tuple(CONFIG.graph.lr_shape)
hr_shape = tuple(CONFIG.graph.hr_shape)
print(f"[OK] DEVICE={DEVICE}  lr_shape={lr_shape}  hr_shape={hr_shape}")

# 3. Resolution paths
if ON_COLAB:
    DATA_ROOT = Path("/content/drive/MyDrive/climate_data/data")
    if not DATA_ROOT.exists():
        DATA_ROOT = Path("/content/data_local")
else:
    DATA_ROOT = Path("data")
RAW = DATA_ROOT / "raw"

LR_PATH_ACCESS   = RAW / "train" / "predictor_ACCESS-CM2_hist.nc"
HR_PATH_ACCESS   = RAW / "train" / "pr_ACCESS-CM2_hist.nc"
STATIC_PATH      = RAW / "static_predictors" / "ERA5_eval_ccam_12km.198110_NZ_Invariant.nc"
MEAN_PATH        = RAW / "normalization_coefs" / "mean_1974_2011.nc"
STD_PATH         = RAW / "normalization_coefs" / "std_1974_2011.nc"

LR_PATH_ECEARTH  = RAW / "test" / "EC-Earth3_histupdated_compressed.nc"
HR_PATH_ECEARTH  = RAW / "test" / "EC-Earth3_historical_precip_compressed.nc"
LR_PATH_NORESM   = RAW / "test" / "NorESM2-MM_histupdated_compressed.nc"
HR_PATH_NORESM   = RAW / "test" / "NorESM2-MM_historical_precip_compressed.nc"

GCM_REGISTRY = {
    "ACCESS-CM2":  (LR_PATH_ACCESS,  HR_PATH_ACCESS,  True),
    "EC-Earth3":   (LR_PATH_ECEARTH, HR_PATH_ECEARTH, False),
    "NorESM2-MM":  (LR_PATH_NORESM,  HR_PATH_NORESM,  False),
}
for tag, (lr, hr, in_d) in GCM_REGISTRY.items():
    ok = "OK" if lr.exists() and hr.exists() else "MISSING"
    print(f"  [{ok}] {tag:<14s} in_dist={in_d}")

# 4. Helper pipeline + builder
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder

def make_pipeline(lr_path, hr_path):
    return NetCDFDataPipeline(
        lr_path=str(lr_path), hr_path=str(hr_path),
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

pipeline_access = make_pipeline(LR_PATH_ACCESS, HR_PATH_ACCESS)
builder = HeteroGraphBuilder(
    lr_shape=lr_shape, hr_shape=hr_shape,
    static_dataset=pipeline_access.get_static_dataset(),
    include_mid_layer=bool(CONFIG.graph.include_mid_layer),
)
print(f"[OK] Builder cree ({len(builder.dynamic_node_types)} dyn + {len(builder.static_node_types)} static)")

def convert_sample_to_batch(sample, builder, device):
    lr_seq = sample["lr"]
    seq_len = lr_seq.shape[0]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    dynamic_features = {nt: lr_nodes_steps[0] for nt in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_features).to(device)
    return {"lr": lr_tensor, "residual": sample["residual"],
            "baseline": sample.get("baseline"), "hetero": hetero}

# 5. Smoke test dataset
test_dataset = pipeline_access.build_sequence_dataset(
    seq_len=int(CONFIG.data.seq_len), stride=int(CONFIG.data.stride), as_torch=True,
)
sample = next(iter(test_dataset))
_runtime_dim = int(sample["lr"].shape[1])
if _runtime_dim != int(CONFIG.rcn.driver_dim):
    CONFIG.rcn.driver_dim = _runtime_dim
    CONFIG.rcn.reconstruction_dim = _runtime_dim
    print(f"[INFO] CONFIG.rcn.driver_dim aligne sur runtime: {_runtime_dim}")

# 6. Chargement des 2 stacks complets
from st_cdgm.models import (
    IntelligibleVariableEncoder, IntelligibleVariableConfig,
    GraphToGridDecoder, RCNCell, RCNSequenceRunner,
    CausalDiffusionDecoder,
)
from st_cdgm.models.edm_preconditioner import EDMConfig
try:
    from st_cdgm.models import ConditionalSkipBlock
    SKIP_AVAILABLE = True
except ImportError:
    SKIP_AVAILABLE = False
    ConditionalSkipBlock = None

def build_stack(ckpt_path, name):
    print(f"  [{name}] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    enc_cfg = IntelligibleVariableConfig(
        num_variables=len(CONFIG.data.lr_variables),
        hidden_dim=CONFIG.encoder.hidden_dim,
        conditioning_dim=CONFIG.encoder.conditioning_dim,
        num_dag_tokens=int(CONFIG.encoder.get("num_dag_tokens", 2)),
        causal_conditioning=bool(CONFIG.encoder.get("causal_conditioning", True)),
    )
    enc = IntelligibleVariableEncoder(enc_cfg).to(DEVICE)
    rcn_cell = RCNCell(
        num_variables=enc_cfg.num_variables,
        hidden_dim=CONFIG.rcn.hidden_dim,
        driver_dim=CONFIG.rcn.driver_dim,
        reconstruction_dim=CONFIG.rcn.reconstruction_dim,
        dropout=CONFIG.rcn.dropout,
    ).to(DEVICE)
    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.detach_interval)
    rh = GraphToGridDecoder(
        d_model=CONFIG.encoder.hidden_dim,
        hr_h=CONFIG.graph.hr_shape[0], hr_w=CONFIG.graph.hr_shape[1],
    ).to(DEVICE)
    edm_cfg = EDMConfig.from_yaml_dict(CONFIG.diffusion.get("edm", {}))
    diff = CausalDiffusionDecoder(
        in_channels=CONFIG.diffusion.in_channels,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        height=CONFIG.diffusion.height, width=CONFIG.diffusion.width,
        scheduler_type=CONFIG.diffusion.scheduler_type, causal_concat=True,
        edm_config=edm_cfg, unet_kwargs=dict(CONFIG.diffusion.unet_kwargs),
    ).to(DEVICE)
    for n, m in [("encoder", enc), ("rcn_cell", rcn_cell),
                  ("regression_head", rh), ("diffusion", diff)]:
        k = f"{n}_state_dict"
        if k in ckpt:
            m.load_state_dict(ckpt[k])
    skip = None
    if SKIP_AVAILABLE and "skip_block_state_dict" in ckpt:
        skip = ConditionalSkipBlock(
            lr_channels=len(CONFIG.data.lr_variables),
            hr_shape=tuple(CONFIG.graph.hr_shape),
        ).to(DEVICE)
        skip.load_state_dict(ckpt["skip_block_state_dict"])
        print(f"  [{name}] [+] skip_block ({skip.num_params()} params)")
    enc.eval(); rcn_cell.eval(); rh.eval(); diff.eval()
    if skip is not None:
        skip.eval()
    # Recupere A_dag depuis le RCN cell pour interpretabilite
    A_dag = None
    if hasattr(rcn_cell, "A_dag"):
        A_dag = rcn_cell.A_dag.detach().cpu().clone()
    return {"encoder": enc, "rcn_runner": rcn_runner, "regression_head": rh,
            "diffusion": diff, "skip_block": skip, "A_dag": A_dag, "variant": name}

print()
print("Chargement des stacks...")
t0 = time.time()
stack_v5 = build_stack(V5_DIR / "epoch_last.pth", "V5")
stack_nc = build_stack(NONCAUSAL_DIR / "epoch_last.pth", "Noncausal")
print(f"[OK] 2 stacks charges en {time.time()-t0:.1f}s")

# 7. Predict generique
@torch.no_grad()
def predict_with_stack(stack, batch, K=4, n_steps=32):
    enc, rcn, rh, diff, skip = (stack["encoder"], stack["rcn_runner"],
                                  stack["regression_head"], stack["diffusion"],
                                  stack["skip_block"])
    lr = batch["lr"].to(DEVICE)
    H_init = enc.init_state(batch["hetero"]).to(DEVICE)
    drivers = [lr[t] for t in range(lr.shape[0])]
    seq = rcn.run(H_init, drivers, reconstruction_sources=None)
    H_T = seq.states[-1]
    mu_c = rh(H_T)
    tshape = batch["residual"][-1].to(DEVICE).shape
    if tshape[-2:] != mu_c.shape[-2:]:
        mu_c = torch.nn.functional.interpolate(
            mu_c, size=tshape[-2:], mode="bilinear", align_corners=False,
        )
    if skip is not None:
        lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
        mu, _ = skip(lr_last, mu_c)
    else:
        mu = mu_c
    mu = torch.nan_to_num(mu, nan=0.0)
    bl = batch["baseline"][-1].to(DEVICE)
    if bl.dim() == mu.dim() - 1:
        bl = bl.unsqueeze(0)
    bl = torch.nan_to_num(bl, nan=0.0)
    ens = []
    for _ in range(K):
        o = diff.sample(
            conditioning=None, num_steps=n_steps,
            scheduler_type="edm_karras", apply_constraints=False,
            mu_HR=mu, baseline_log=bl,
        )
        r = o.residual if hasattr(o, "residual") else o
        ens.append((bl + mu + r).cpu())
    return torch.stack(ens, dim=0)

print()
print("Bootstrap autonome complet. Variables disponibles :")
print(f"  CONFIG, DEVICE, builder, convert_sample_to_batch, predict_with_stack")
print(f"  stack_v5, stack_nc, GCM_REGISTRY, make_pipeline")
print(f"  test_dataset (ACCESS-CM2 in-dist)")
'''

# ───────────────────────────────────────────────────────────────────
# PHASE 7 OOD REELLE — 4 runs avec run_aligned_eval
# ───────────────────────────────────────────────────────────────────
PHASE7_MD = """---

## Phase 7 — OOD réel sur EC-Earth3 + NorESM2-MM (4 runs alignés cGAN)

**Test inter-GCM** : on évalue V5-mini et Noncausal — entraînés sur ACCESS-CM2 — sur deux autres GCM (EC-Earth3 et NorESM2-MM) jamais vus à l'entraînement.

**Protocole** : pour chaque (variant × GCM), on utilise `st_cdgm.evaluation.aligned_eval.run_aligned_eval` (vendorisé depuis le cGAN de Rampal). Produit des fichiers `aligned_metrics_<GCM>_<variant>.json` contenant :
- **CDD** (Consecutive Dry Days) — fréquence des séquences sèches
- **Rx1Day** — précipitation maximale quotidienne
- **R10** — nombre de jours > 10 mm/j
- **Saisonnier** (DJF/JJA) — moyennes saisonnières
- **PSD** — Power Spectral Density (structure spectrale)
- + Pearson, RMSE, MAE comme références secondaires

**4 runs au total** : V5×EC-Earth3, V5×NorESM2-MM, NC×EC-Earth3, NC×NorESM2-MM. Plus l'in-dist ACCESS pour les deux pour la baseline.

**Δ_OOD** = `(Pearson_in - Pearson_ood) / Pearson_in` pour chaque (variant, GCM).

Coût estimé : ~3-5 min par run × 4 runs ≈ **15-25 min** (selon K_SAMPLES et longueur du sample subset)."""

PHASE7_CODE = '''# ============================================================
# Phase 7 : 4 runs OOD reels avec run_aligned_eval
# Sortie : aligned_metrics_<GCM>_<variant>.json pour chaque combo
# ============================================================
import json
import time
import torch
import numpy as np
from st_cdgm.evaluation.aligned_eval import run_aligned_eval

# Parametres
N_TIMES_OOD = 365          # Nombre de pas de temps a evaluer par GCM (continuous)
K_SAMPLES_OOD = 4          # Ensemble size par snapshot
N_STEPS_DIFF = 18          # Pas EDM (18 = defaut Karras)

def collect_predictions_for_gcm(stack, gcm_tag, n_times=N_TIMES_OOD, K=K_SAMPLES_OOD):
    """Genere predictions HR + collecte la verite terrain et les timestamps pour run_aligned_eval."""
    lr_path, hr_path, in_dist = GCM_REGISTRY[gcm_tag]
    pipe = make_pipeline(lr_path, hr_path)
    ds = pipe.build_sequence_dataset(
        seq_len=int(CONFIG.data.seq_len),
        stride=1,             # stride=1 pour ALIGNED metrics (CDD, Rx1Day exigent une serie continue)
        as_torch=True,
    )

    preds_log, truths_log = [], []
    times_list = []

    it = iter(ds)
    for i in range(n_times):
        try:
            sample = next(it)
        except StopIteration:
            break
        batch = convert_sample_to_batch(sample, builder, DEVICE)
        ens = predict_with_stack(stack, batch, K=K, n_steps=N_STEPS_DIFF)
        pred_mean = ens.nanmean(0)  # [B=1, 1, H, W]
        truth = (batch["baseline"][-1] + batch["residual"][-1]).cpu()
        if truth.dim() == 3:
            truth = truth.unsqueeze(0)
        preds_log.append(pred_mean.squeeze().numpy())
        truths_log.append(truth.squeeze().numpy())
        # Time : approximer depuis l'index dans la sequence (xarray-friendly)
        times_list.append(np.datetime64(f"1986-01-01") + np.timedelta64(i, "D"))

        if (i + 1) % 50 == 0:
            print(f"  [{gcm_tag} / {stack['variant']}] {i+1}/{n_times} pas evalues")

    return np.array(preds_log), np.array(truths_log), np.array(times_list, dtype="datetime64[D]")


# Boucle 4 runs : V5 x {EC-Earth3, NorESM2-MM} + NC x {EC-Earth3, NorESM2-MM}
print("=" * 70)
print(f"Phase 7 : 4 runs OOD ({N_TIMES_OOD} pas x K={K_SAMPLES_OOD} x {N_STEPS_DIFF} EDM steps)")
print("=" * 70)
print()

all_results = {}
t_global = time.time()

for stack_name, stack, ckpt_dir in [("V5", stack_v5, V5_DIR),
                                      ("Noncausal", stack_nc, NONCAUSAL_DIR)]:
    for gcm_tag in ["ACCESS-CM2", "EC-Earth3", "NorESM2-MM"]:
        run_label = f"{stack_name}_{gcm_tag}"
        t0 = time.time()
        print(f"\\n=== RUN {run_label} ===")

        if not GCM_REGISTRY[gcm_tag][0].exists():
            print(f"  [SKIP] {GCM_REGISTRY[gcm_tag][0]} absent")
            continue

        try:
            preds, truths, times = collect_predictions_for_gcm(stack, gcm_tag)
        except Exception as e:
            print(f"  [ERREUR] {type(e).__name__}: {e}")
            continue

        if len(preds) == 0:
            print(f"  [SKIP] Pas de samples retournes")
            continue

        # Appel run_aligned_eval
        out_path = ckpt_dir / f"aligned_metrics_{gcm_tag}_{stack_name.lower()}.json"
        try:
            run_aligned_eval(
                pred_fields=preds, truth_fields=truths, times=times,
                out_path=str(out_path),
                gcm=gcm_tag,
                run_variant=stack_name.lower(),
                in_distribution=GCM_REGISTRY[gcm_tag][2],
                space="log1p",
                k_samples=K_SAMPLES_OOD,
            )
            with open(out_path, "r", encoding="utf-8") as f:
                all_results[run_label] = json.load(f)
            print(f"  [OK] {(time.time()-t0):.1f}s -> {out_path}")
        except Exception as e:
            print(f"  [ERREUR run_aligned_eval] {type(e).__name__}: {e}")

print()
print(f"[OK] Phase 7 terminee en {(time.time()-t_global)/60:.1f} min")

# Synthese Delta_OOD
print()
print("=" * 70)
print("RECAPITULATIF — Delta_OOD")
print("=" * 70)
for variant in ["V5", "Noncausal"]:
    in_dist = all_results.get(f"{variant}_ACCESS-CM2", {})
    if not in_dist:
        continue
    pearson_in = in_dist.get("pearson", {}).get("global", None) or in_dist.get("pearson_global")
    if pearson_in is None:
        continue
    print(f"\\n{variant} (in-dist Pearson = {pearson_in:.4f}) :")
    for ood in ["EC-Earth3", "NorESM2-MM"]:
        r = all_results.get(f"{variant}_{ood}", {})
        if not r:
            continue
        pearson_ood = r.get("pearson", {}).get("global", None) or r.get("pearson_global")
        if pearson_ood is None:
            continue
        delta = (pearson_in - pearson_ood) / max(pearson_in, 1e-6)
        print(f"  {ood:<15s} Pearson_OOD={pearson_ood:.4f}  Delta_OOD={delta:+.4f} ({delta*100:+.1f}%)")

# Sauvegarde recap
recap_path = RESULTS_DIR / "phase7_ood_aligned.json"
recap_path.write_text(json.dumps({
    "n_times": N_TIMES_OOD, "K_samples": K_SAMPLES_OOD, "n_steps": N_STEPS_DIFF,
    "runs": all_results,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(f"\\n[OK] Recap sauvegarde : {recap_path}")
'''

# ───────────────────────────────────────────────────────────────────
# PHASE 8 INTERPRETABILITE VISUELLE
# ───────────────────────────────────────────────────────────────────
PHASE8_MD = """---

## Phase 8 — Interprétabilité visuelle de la causalité

Cette phase **génère un maximum de visualisations** pour démontrer l'apport interprétable du chemin causal d'Oracle V5 vs le baseline Noncausal. Toutes les figures sont sauvegardées en PNG dans `RESULTS_DIR/phase8_figures/`.

### 8 visualisations produites

1. **DAG appris V5** — heatmap de la matrice `A_dag` 6×6 (qui cause quoi)
2. **Distribution α (gate skip-connection)** — histogramme de la fraction du chemin causal
3. **Interventions `do(·)` — cartes Δ_pred** — 3 interventions × 2 variants = 6 cartes côte à côte
4. **Sensibilité par variable LR** — gradient |∂pr_HR/∂var_LR| par variable (15 panneaux)
5. **Ablation A_dag** — μ_HR avec A_dag réel vs A_dag=0 (cartes side-by-side)
6. **Comparaison spatiale V5 vs Noncausal** — same input, deux sorties côte à côte + différence
7. **Spectre radial (RAPSD)** — courbes V5 vs Noncausal vs vérité
8. **Histogramme intensités** — distribution des prédictions vs vérité (queue lourde)

### Tableaux

- Q_int par modèle (V5, Noncausal) sur 3 interventions standardisées
- Magnitude et signe Δ_pred par intervention

Coût estimé : ~5-8 min sur A100 (peu de samples, beaucoup de figures)."""

PHASE8_CODE = '''# ============================================================
# Phase 8 : Interpretabilite visuelle
# Genere 8 figures PNG + tableaux Q_int + sauvegarde JSON.
# ============================================================
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from pathlib import Path

FIG_DIR = RESULTS_DIR / "phase8_figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Figures -> {FIG_DIR}")

# === 1. DAG appris V5 (heatmap A_dag) =======================================
if stack_v5["A_dag"] is not None:
    A = stack_v5["A_dag"].numpy()
    # Mask diagonale (no self-loop par convention)
    A_disp = A.copy()
    np.fill_diagonal(A_disp, 0)

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    im = ax.imshow(A_disp, cmap="RdBu_r", vmin=-np.abs(A_disp).max(), vmax=np.abs(A_disp).max())
    ax.set_title("DAG appris (V5) — matrice d'adjacence A_dag\\n(diagonale masquee)", fontsize=11)
    # Labels : par convention, lignes = sources, colonnes = cibles
    var_labels = ["GP850_spat", "GP850→GP500", "GP500_spat", "GP500→GP250", "GP250_spat", "SP_HR"]
    ax.set_xticks(range(len(var_labels))); ax.set_xticklabels(var_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(var_labels))); ax.set_yticklabels(var_labels)
    ax.set_xlabel("Cible (effet)")
    ax.set_ylabel("Source (cause)")
    for i in range(A_disp.shape[0]):
        for j in range(A_disp.shape[1]):
            if i != j:
                ax.text(j, i, f"{A_disp[i,j]:.2f}", ha="center", va="center",
                        color="white" if abs(A_disp[i,j]) > 0.3 else "black", fontsize=8)
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_dag_v5.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("[OK] Figure 1 : DAG V5")
else:
    print("[SKIP] Figure 1 : A_dag indisponible")

# === 2. Distribution alpha (gate skip-connection) ==========================
if stack_v5["skip_block"] is not None:
    alphas = []
    it = iter(test_dataset)
    skip = stack_v5["skip_block"]
    rh = stack_v5["regression_head"]
    enc = stack_v5["encoder"]
    rcn = stack_v5["rcn_runner"]
    with torch.no_grad():
        for i in range(min(32, 100)):
            try:
                s = next(it)
            except StopIteration:
                break
            b = convert_sample_to_batch(s, builder, DEVICE)
            lr = b["lr"].to(DEVICE)
            H_init = enc.init_state(b["hetero"]).to(DEVICE)
            drivers = [lr[t] for t in range(lr.shape[0])]
            seq = rcn.run(H_init, drivers, reconstruction_sources=None)
            mu_c = rh(seq.states[-1])
            tshape = b["residual"][-1].to(DEVICE).shape
            if tshape[-2:] != mu_c.shape[-2:]:
                mu_c = torch.nn.functional.interpolate(mu_c, size=tshape[-2:], mode="bilinear", align_corners=False)
            lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
            _, a = skip(lr_last, mu_c)
            alphas.append(a.cpu().numpy().flatten())
    alphas = np.concatenate(alphas) if alphas else np.array([])
    if alphas.size > 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.hist(alphas, bins=30, color="steelblue", edgecolor="black", alpha=0.75)
        ax.axvline(0.6, color="red", linestyle="--", label="alpha_floor=0.6")
        ax.axvline(alphas.mean(), color="green", linestyle="-", label=f"moyenne={alphas.mean():.3f}")
        ax.set_xlabel("alpha (fraction chemin causal)")
        ax.set_ylabel("Frequence")
        ax.set_title(f"Distribution alpha (skip-connection gate) — n={len(alphas)} samples\\n"
                      f"alpha > 0.6 sur {(alphas > 0.6).mean()*100:.1f}% des cas (preservation O3)")
        ax.legend()
        ax.set_xlim(0, 1)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "02_alpha_distribution.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"[OK] Figure 2 : alpha mean={alphas.mean():.3f}, prop>0.6={((alphas>0.6).mean()*100):.1f}%")
else:
    print("[SKIP] Figure 2 : skip_block absent")

# === 3. Interventions do(.) - cartes Delta_pred =============================
from scripts.intervention_test import INTERVENTIONS, resolve_variable_indices, apply_intervention
lr_vars = list(CONFIG.data.lr_variables)
resolved = resolve_variable_indices(lr_vars)

intervention_results = {"V5": [], "Noncausal": []}
sample_for_int = next(iter(test_dataset))
batch_normal = convert_sample_to_batch(sample_for_int, builder, DEVICE)

n_int = sum(1 for s in resolved if s["variable_idx"] is not None)
if n_int > 0:
    fig, axes = plt.subplots(n_int, 4, figsize=(16, 3.5 * n_int))
    if n_int == 1:
        axes = axes.reshape(1, -1)

    row_idx = 0
    for spec in resolved:
        if spec["variable_idx"] is None:
            continue
        # Predictions normales
        pred_norm_v5 = predict_with_stack(stack_v5, batch_normal, K=4, n_steps=18).nanmean(0).squeeze().numpy()
        pred_norm_nc = predict_with_stack(stack_nc, batch_normal, K=4, n_steps=18).nanmean(0).squeeze().numpy()

        # Predictions intervenues
        batch_int = dict(batch_normal)
        batch_int["lr"] = apply_intervention(batch_normal["lr"], spec, standardization=None)
        pred_int_v5 = predict_with_stack(stack_v5, batch_int, K=4, n_steps=18).nanmean(0).squeeze().numpy()
        pred_int_nc = predict_with_stack(stack_nc, batch_int, K=4, n_steps=18).nanmean(0).squeeze().numpy()

        delta_v5 = pred_int_v5 - pred_norm_v5
        delta_nc = pred_int_nc - pred_norm_nc

        # Sauvegarde des resultats numeriques
        intervention_results["V5"].append({
            "intervention": spec["name"], "variable": spec["variable_name"],
            "delta_mean": float(np.nanmean(delta_v5)), "delta_std": float(np.nanstd(delta_v5)),
            "sign_pred": int(np.sign(np.nanmean(delta_v5))), "sign_expected": int(spec["expected_sign"]),
            "match": int(np.sign(np.nanmean(delta_v5))) == int(spec["expected_sign"]),
        })
        intervention_results["Noncausal"].append({
            "intervention": spec["name"], "variable": spec["variable_name"],
            "delta_mean": float(np.nanmean(delta_nc)), "delta_std": float(np.nanstd(delta_nc)),
            "sign_pred": int(np.sign(np.nanmean(delta_nc))), "sign_expected": int(spec["expected_sign"]),
            "match": int(np.sign(np.nanmean(delta_nc))) == int(spec["expected_sign"]),
        })

        # Plot : pred normale | pred intervenue | delta V5 | delta NC
        vmin, vmax = -max(abs(delta_v5).max(), abs(delta_nc).max(), 0.01), max(abs(delta_v5).max(), abs(delta_nc).max(), 0.01)

        axes[row_idx, 0].imshow(pred_norm_v5, cmap="viridis")
        axes[row_idx, 0].set_title(f"V5 normal", fontsize=10)

        axes[row_idx, 1].imshow(pred_int_v5, cmap="viridis")
        axes[row_idx, 1].set_title(f"V5 + {spec['name']}", fontsize=10)

        im_v5 = axes[row_idx, 2].imshow(delta_v5, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        delta_mean_v5 = np.nanmean(delta_v5)
        sign_v5 = "+" if delta_mean_v5 > 0 else "-"
        axes[row_idx, 2].set_title(f"Delta V5 (mean={delta_mean_v5:+.4f})", fontsize=10)
        plt.colorbar(im_v5, ax=axes[row_idx, 2], shrink=0.7)

        im_nc = axes[row_idx, 3].imshow(delta_nc, cmap="RdBu_r", vmin=vmin, vmax=vmax)
        delta_mean_nc = np.nanmean(delta_nc)
        axes[row_idx, 3].set_title(f"Delta NC (mean={delta_mean_nc:+.4f})", fontsize=10)
        plt.colorbar(im_nc, ax=axes[row_idx, 3], shrink=0.7)

        for a in axes[row_idx, :]:
            a.set_xticks([]); a.set_yticks([])

        row_idx += 1

    plt.suptitle("Phase 8 : Reponses des modeles aux interventions do(.)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_interventions_maps.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure 3 : {n_int} interventions x 4 panneaux")

    # Tableau Q_int
    q_v5 = np.mean([r["match"] for r in intervention_results["V5"]])
    q_nc = np.mean([r["match"] for r in intervention_results["Noncausal"]])
    print()
    print(f"Q_int V5         : {q_v5:.3f}  ({sum(r['match'] for r in intervention_results['V5'])}/{len(intervention_results['V5'])} signes corrects)")
    print(f"Q_int Noncausal  : {q_nc:.3f}  ({sum(r['match'] for r in intervention_results['Noncausal'])}/{len(intervention_results['Noncausal'])} signes corrects)")

# === 4. Sensibilite par variable LR (gradients) ==============================
print()
print("Computation de la sensibilite par variable...")
sensitivities = {"V5": {}, "Noncausal": {}}
sample_sens = next(iter(test_dataset))
batch_sens = convert_sample_to_batch(sample_sens, builder, DEVICE)

for stack_name, stack in [("V5", stack_v5), ("Noncausal", stack_nc)]:
    enc, rcn, rh, skip = stack["encoder"], stack["rcn_runner"], stack["regression_head"], stack["skip_block"]
    lr = batch_sens["lr"].clone().to(DEVICE).requires_grad_(True)
    H_init = enc.init_state(batch_sens["hetero"]).to(DEVICE)
    drivers = [lr[t] for t in range(lr.shape[0])]
    seq = rcn.run(H_init, drivers, reconstruction_sources=None)
    mu_c = rh(seq.states[-1])
    tshape = batch_sens["residual"][-1].to(DEVICE).shape
    if tshape[-2:] != mu_c.shape[-2:]:
        mu_c = torch.nn.functional.interpolate(mu_c, size=tshape[-2:], mode="bilinear", align_corners=False)
    if skip is not None:
        lr_last = drivers[-1] if drivers[-1].dim() == 4 else drivers[-1].unsqueeze(0)
        mu, _ = skip(lr_last, mu_c)
    else:
        mu = mu_c
    target_scalar = mu.abs().sum()
    target_scalar.backward()
    # Gradient absolu moyenne dans le temps et dans l'espace LR
    grad = lr.grad.detach().abs().mean(dim=(0, 2, 3))  # [C_lr] = sensitivity per variable
    for i, v in enumerate(lr_vars):
        sensitivities[stack_name][v] = float(grad[i].cpu())

# Plot par variable
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
x = np.arange(len(lr_vars))
w = 0.35
v5_vals = [sensitivities["V5"][v] for v in lr_vars]
nc_vals = [sensitivities["Noncausal"][v] for v in lr_vars]
axes[0].bar(x - w/2, v5_vals, w, label="V5", color="steelblue")
axes[0].bar(x + w/2, nc_vals, w, label="Noncausal", color="orange")
axes[0].set_xticks(x); axes[0].set_xticklabels(lr_vars, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("|d mu_HR / d var_LR| moyenne")
axes[0].set_title("Sensibilite de mu_HR par variable LR")
axes[0].legend()
axes[0].grid(alpha=0.3)

# Ratio V5/NC pour voir ou la causalite change la sensibilite
ratios = [v5_vals[i] / max(nc_vals[i], 1e-12) for i in range(len(lr_vars))]
axes[1].bar(x, ratios, color="purple", alpha=0.7)
axes[1].axhline(1.0, color="red", linestyle="--", label="ratio=1 (egal)")
axes[1].set_xticks(x); axes[1].set_xticklabels(lr_vars, rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("Ratio sensibilite V5 / NC")
axes[1].set_title("Ratio des sensibilites — > 1 = V5 utilise plus cette variable")
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "04_sensitivity_per_variable.png", dpi=120, bbox_inches="tight")
plt.close()
print("[OK] Figure 4 : sensibilites par variable")

# === 5. Ablation A_dag (mu_HR avec vs sans DAG) ============================
if stack_v5["A_dag"] is not None:
    print()
    print("Ablation A_dag sur V5...")
    rcn_cell = stack_v5["rcn_runner"].cell
    A_orig = rcn_cell.A_dag.detach().clone()
    sample_abl = next(iter(test_dataset))
    batch_abl = convert_sample_to_batch(sample_abl, builder, DEVICE)

    with torch.no_grad():
        # mu_HR avec A_dag normal
        mu_full = predict_with_stack(stack_v5, batch_abl, K=1, n_steps=18).nanmean(0).squeeze().numpy()
        # Ablation : A_dag := 0
        rcn_cell.A_dag.data.zero_()
        mu_ablated = predict_with_stack(stack_v5, batch_abl, K=1, n_steps=18).nanmean(0).squeeze().numpy()
        # Restaurer
        rcn_cell.A_dag.data.copy_(A_orig)

    delta_ablation = mu_full - mu_ablated
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(mu_full, cmap="viridis")
    axes[0].set_title("V5 avec A_dag appris")
    axes[1].imshow(mu_ablated, cmap="viridis")
    axes[1].set_title("V5 avec A_dag = 0 (ablation)")
    vmax = max(abs(delta_ablation).max(), 1e-3)
    im = axes[2].imshow(delta_ablation, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    delta_signal_ratio = float(np.abs(delta_ablation).mean() / max(np.abs(mu_full).mean(), 1e-12))
    axes[2].set_title(f"Delta (Delta/signal = {delta_signal_ratio:.2%})")
    plt.colorbar(im, ax=axes[2], shrink=0.7)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    plt.suptitle("Ablation A_dag sur V5 — la difference quantifie l'apport du DAG", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_ablation_A_dag.png", dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[OK] Figure 5 : Delta/signal = {delta_signal_ratio:.2%}")

# === 6. Comparaison spatiale V5 vs Noncausal (meme input) ==================
sample_comp = next(iter(test_dataset))
batch_comp = convert_sample_to_batch(sample_comp, builder, DEVICE)
with torch.no_grad():
    pred_v5 = predict_with_stack(stack_v5, batch_comp, K=4, n_steps=18).nanmean(0).squeeze().numpy()
    pred_nc = predict_with_stack(stack_nc, batch_comp, K=4, n_steps=18).nanmean(0).squeeze().numpy()
truth = (batch_comp["baseline"][-1] + batch_comp["residual"][-1]).cpu().squeeze().numpy()
diff = pred_v5 - pred_nc

fig, axes = plt.subplots(2, 2, figsize=(11, 9))
vmin, vmax = np.nanmin([pred_v5, pred_nc, truth]), np.nanmax([pred_v5, pred_nc, truth])
axes[0,0].imshow(truth, cmap="viridis", vmin=vmin, vmax=vmax)
axes[0,0].set_title("Verite terrain (HR cible)")
axes[0,1].imshow(pred_v5, cmap="viridis", vmin=vmin, vmax=vmax)
axes[0,1].set_title("V5 prediction")
axes[1,0].imshow(pred_nc, cmap="viridis", vmin=vmin, vmax=vmax)
axes[1,0].set_title("Noncausal prediction")
diff_max = max(abs(diff).max(), 1e-3)
im = axes[1,1].imshow(diff, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
axes[1,1].set_title(f"V5 - Noncausal\\n(mean abs = {np.abs(diff).mean():.4f})")
plt.colorbar(im, ax=axes[1,1], shrink=0.7)
for a in axes.flatten():
    a.set_xticks([]); a.set_yticks([])
plt.suptitle("Comparaison spatiale V5 vs Noncausal (meme input)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_spatial_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("[OK] Figure 6 : comparaison spatiale")

# === 7. Spectre radial RAPSD ================================================
def radial_power_spectrum(field):
    """RAPSD 1D : moyenne radiale de la PSD 2D."""
    f = np.fft.fft2(field)
    psd2d = np.abs(f) ** 2
    H, W = field.shape
    cy, cx = H // 2, W // 2
    Y, X = np.indices(field.shape)
    R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2).astype(int)
    R_max = min(cx, cy)
    radial_mean = np.zeros(R_max)
    for r in range(R_max):
        mask = R == r
        if mask.sum() > 0:
            radial_mean[r] = psd2d[mask].mean()
    return radial_mean

psd_v5 = radial_power_spectrum(pred_v5)
psd_nc = radial_power_spectrum(pred_nc)
psd_truth = radial_power_spectrum(truth)
k = np.arange(1, len(psd_v5) + 1)

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.loglog(k, psd_truth[:len(k)], label="Verite", color="black", linewidth=2)
ax.loglog(k, psd_v5[:len(k)], label="V5", color="steelblue", linewidth=1.5)
ax.loglog(k, psd_nc[:len(k)], label="Noncausal", color="orange", linewidth=1.5)
ax.set_xlabel("Nombre d'onde radial k")
ax.set_ylabel("PSD radiale")
ax.set_title("Spectre radial (RAPSD) — fidelite des structures spatiales par echelle")
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(FIG_DIR / "07_rapsd_comparison.png", dpi=120, bbox_inches="tight")
plt.close()
print("[OK] Figure 7 : RAPSD")

# === 8. Histogramme intensites ==============================================
fig, ax = plt.subplots(1, 1, figsize=(9, 5))
bins = np.linspace(0, max(np.nanmax(truth), np.nanmax(pred_v5), np.nanmax(pred_nc)), 60)
ax.hist(truth.flatten(), bins=bins, alpha=0.6, label="Verite", color="black", density=True)
ax.hist(pred_v5.flatten(), bins=bins, alpha=0.5, label="V5", color="steelblue", density=True)
ax.hist(pred_nc.flatten(), bins=bins, alpha=0.5, label="Noncausal", color="orange", density=True)
ax.set_yscale("log")
ax.set_xlabel("Intensite (log1p mm/jour)")
ax.set_ylabel("Densite (log)")
ax.set_title("Distribution des intensites (queue lourde des extremes)")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "08_intensity_histogram.png", dpi=120, bbox_inches="tight")
plt.close()
print("[OK] Figure 8 : histogramme intensites")

# === Sauvegarde JSON de tous les resultats =================================
results_full = {
    "interventions": intervention_results,
    "Q_int": {
        "V5": float(np.mean([r["match"] for r in intervention_results["V5"]])) if intervention_results["V5"] else None,
        "Noncausal": float(np.mean([r["match"] for r in intervention_results["Noncausal"]])) if intervention_results["Noncausal"] else None,
    },
    "sensitivities": sensitivities,
    "ablation_A_dag_delta_signal_ratio": delta_signal_ratio if stack_v5["A_dag"] is not None else None,
    "alpha_distribution": {
        "mean": float(alphas.mean()) if stack_v5["skip_block"] is not None and len(alphas) > 0 else None,
        "std": float(alphas.std()) if stack_v5["skip_block"] is not None and len(alphas) > 0 else None,
        "prop_above_floor": float((alphas > 0.6).mean()) if stack_v5["skip_block"] is not None and len(alphas) > 0 else None,
    },
    "figures": sorted([str(p.name) for p in FIG_DIR.glob("*.png")]),
}
(RESULTS_DIR / "phase8_interpretability.json").write_text(
    json.dumps(results_full, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
)
print()
print("=" * 70)
print("Phase 8 terminee — 8 figures + JSON resultats")
print("=" * 70)
print(f"Q_int V5         : {results_full['Q_int']['V5']}")
print(f"Q_int Noncausal  : {results_full['Q_int']['Noncausal']}")
print(f"Figures dans : {FIG_DIR}")
print(f"JSON : {RESULTS_DIR / 'phase8_interpretability.json'}")
'''


# ───────────────────────────────────────────────────────────────────
# APPLICATION DU PATCH
# ───────────────────────────────────────────────────────────────────

# Construction de la nouvelle liste de cellules
new_cells = []

# Garde cells 0, 1, 2 (intro, bootstrap, paths)
new_cells.extend(nb["cells"][:3])

# Insertion : bootstrap autonome (MD + code)
new_cells.append(md(BOOTSTRAP_MD))
new_cells.append(code(BOOTSTRAP_CODE))

# Garde cells 3, 4 (Phase 6 MD + code)
new_cells.extend(nb["cells"][3:5])

# Remplace Phase 7 (etait cells 5+6) par version reelle OOD
new_cells.append(md(PHASE7_MD))
new_cells.append(code(PHASE7_CODE))

# Remplace Phase 8 (etait cells 7+8+9) par version interpretabilite visuelle
new_cells.append(md(PHASE8_MD))
new_cells.append(code(PHASE8_CODE))

# Garde la synthese finale (cells 10+11)
new_cells.extend(nb["cells"][10:12])

nb["cells"] = new_cells

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print(f"[OK] Notebook reorganise : {NB} ({len(nb['cells'])} cellules)")
print()
print("Structure finale :")
for i, c in enumerate(nb["cells"]):
    src = "".join(c.get("source", []))
    head = src.split("\\n")[0][:90].strip()
    print(f"  [{i:2d}] {c['cell_type']:8s} : {head}")
