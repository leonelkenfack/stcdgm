"""
Build the V6' (V6-prime) FULL notebook — `st_cdgm_v6_prime_seed42.ipynb`.

FULL implementation per the audit recommendations (no simplified alternative) :
  1. Stage 1 = 11-node graph (9-node + U850/V850 wind nodes)
     - extended_v6_wind=True in the builder
     - 2 injected metapaths (U850_spat, V850_spat) -> num_vars 9 -> 11
     - per-node channel routing (U850<-u_*, V850<-v_*)
     - G_phys 11x11 via physics_prior.VAR_LABELS_V6 / EXPECTED_EDGES_V6
     - LR drivers = 21 vars (15 base + 6 climat features)
  2. Stage 2 = full-LR conditioning (22 channels : 21 obligatory + IVT-72h bonus)
  3. End-to-end metric cells (Convention A per-gridpoint + Convention B pooled)
     via the shared module eval_metrics_dual_convention.py
  4. Pre-registered ablations A1 (mu_HR->0) / A2 (LR->0) wired + evaluated

All P0 fixes from the IA audit applied (sys.path src/, real schedule_lambdas,
real build_physical_mask, convert_sample_to_batch lr_grid, no placeholders).

Output : path_c_plus/scripts/st_cdgm_v6_prime_seed42.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "path_c_plus" / "scripts" / "st_cdgm_v6_prime_seed42.ipynb"


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": text.splitlines(keepends=True)}


CELL_0 = """# V6' (V6-prime) FULL — Pivot full-LR + 11-node wind graph (ST-CDGM Path C+)

**Validé 5/5 par l'audit indépendant (2026-06-30).** Médiane P(battre noncausal sur ≥1 co-primaire) ≈ **75%** (V6 MVP rejeté ≈ 15%).

## Deux changements, tous deux recommandés par l'audit
1. **Stage 2 full-LR conditioning** (LE pivot) : `[y_noisy, mu_HR, baseline]` → `+ 22 champs LR`. Le vrai goulot était informationnel (toy diffusion : F1@p99 0.053 pauvre vs **0.639** full-info). Pattern CorrDiff/StormCast/Rampal 2025.
2. **Stage 1 : 11-node** (9-node + U850/V850 wind nodes, recommandation Climat) + 21 LR drivers (15 base + 6 features Climat).

## Wiring 11-node (complet)
- `extended_v6_wind=True` → builder ajoute U850, V850
- 2 metapaths injectés (U850_spat, V850_spat) → num_vars 9 → **11**
- routing canaux : U850 ← u_850/500/250, V850 ← v_850/500/250
- **G_phys 11×11** : `physics_prior.VAR_LABELS_V6` / `EXPECTED_EDGES_V6` (14 arêtes : 10 humide-QG + 4 vent U850/V850→IVT/SP_HR)

## Garde-fous pré-enregistrés (`V6_PRIME_seuils_preregistered.json`)
- **M1** normalisation LR z-score figée train ; **M2** concat centralisé ; **M3** parité inférence (sample lève si lr_fields=None)
- **A1** (mu_HR→0) / **A2** (LR→0, post-norm) monitorées **dès le SMOKE** (M6)
- Métrique **co-primaire** : per-gridpoint ETCCDI (battre 0.816, ne pas régresser < V5=0.841) + pooled (0.550)
- Ablations A1/A2 câblées end-to-end (Cell 11)
"""


CELL_1 = """# >>> Cell 1 : Bootstrap Colab + git sync (P0 fix : sys.path src/)
import os, sys, subprocess
from pathlib import Path

GIT_URL = "https://github.com/leonelkenfack/stcdgm.git"
GIT_BRANCH = "four-node-causal"
REPO_DIR = "/content/climate_data"
DRIVE_ROOT = "/content/drive/MyDrive/climate_data"

if not Path("/content/drive").exists():
    from google.colab import drive
    drive.mount("/content/drive")

if not Path(REPO_DIR).exists():
    subprocess.check_call(["git", "clone", "--depth=200", "-b", GIT_BRANCH, GIT_URL, REPO_DIR])
else:
    subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "origin"])
    subprocess.check_call(["git", "-C", REPO_DIR, "checkout", GIT_BRANCH])
    subprocess.check_call(["git", "-C", REPO_DIR, "pull", "origin", GIT_BRANCH])

# P0 FIX (audit IA) : add BOTH repo root (path_c_plus, scripts) AND repo/src (st_cdgm)
for _p in (REPO_DIR, str(Path(REPO_DIR) / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for pkg in ["torch_geometric", "diffusers", "omegaconf", "scipy"]:
    try: __import__(pkg)
    except Exception: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_sha = subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "HEAD"]).decode().strip()
print(f"[Cell 1] Bootstrap OK — commit {_sha[:8]} — sys.path has src/ (P0 fix)")
"""


CELL_2 = """# >>> Cell 2 : Config + V6' constants (11-node Stage 1, 22-ch Stage 2 cond)
import torch, numpy as np
from omegaconf import OmegaConf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE_MODE = False   # True → few epochs/batches for the mandatory smoke

from st_cdgm.v6_constants import (
    NONCAUSAL_15_VARS, V6_LR_VARS_OBLIGATORY, V6_LR_VARS_FULL,
    V6_NODE_CHANNEL_ROUTING, V6_NUM_ENCODER_VARS, assert_lr_vars_match,
)

USE_IVT_72H_BONUS = True
# Stage 1 drivers = full augmented LR (21 obligatory, + IVT-72h bonus if enabled)
STAGE1_LR_VARS = list(V6_LR_VARS_FULL) if USE_IVT_72H_BONUS else list(V6_LR_VARS_OBLIGATORY)
# Stage 2 conditioning = same set (full LR)
STAGE2_COND_LR_VARS = list(STAGE1_LR_VARS)
LR_COND_CHANNELS = len(STAGE2_COND_LR_VARS)
print(f"[Cell 2] Stage 1 LR vars = {len(STAGE1_LR_VARS)} (drivers)  |  "
      f"Stage 2 LR conditioning channels = {LR_COND_CHANNELS}")

# --- OOD parameterization (fix ML : évite le re-run manuel skippable) -------
# Pour l'OOD, changer UNIQUEMENT GCM_ID — les stats de normalisation restent
# celles d'ACCESS-CM2 (fichiers explicites, cf. Cell 3). PAS de re-norm per-GCM.
GCM_ID = "ACCESS-CM2"           # "EC-Earth3" pour le run OOD
IS_OOD_RUN = GCM_ID != "ACCESS-CM2"

CONFIG = OmegaConf.load("config/training_config.yaml")
CONFIG = OmegaConf.merge(CONFIG, OmegaConf.load("config/training_config_corrdiff_normal.yaml"))
OmegaConf.set_struct(CONFIG, False)
CONFIG.data.lr_variables = STAGE1_LR_VARS   # 11-node Stage 1 sees the full LR

EXTENDED_9NODE = True
EXTENDED_V6_WIND = True   # <<< FULL V6 : U850/V850 as graph nodes

SEEDS = [42]
K9_DATES = {"train": ("1980-01-01","2009-12-31"), "val": ("2010-01-01","2011-12-31"),
            "test": ("2012-01-01","2013-12-31"), "holdout": ("2014-01-01","2014-12-31")}
print("[Cell 2] V6' FULL config ready (11-node + full-LR conditioning)")
"""


CELL_3 = """# >>> Cell 3 : Pipeline + 11-node builder + metapath injection + routing
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder
from omegaconf import OmegaConf as _OC

LR_PATH_V6  = f"{DRIVE_ROOT}/lr_{GCM_ID}_v6.nc"     # from preprocess_v6_lr.py
HR_PATH     = f"{DRIVE_ROOT}/hr_NIWA-REMS.nc"
STATIC_PATH = f"{DRIVE_ROOT}/static_HR_v6.nc"
# --- P1-C fix (audit IA) + fix OOD (audit ML) --------------------------------
# Les anciens means/stds (15 vars) ne couvrent PAS les 22 vars V6' → KeyError.
# Protocole : on génère UNE FOIS des stats v6 explicites sur la fenêtre train
# d'ACCESS-CM2, sauvegardées sur Drive, et TOUT run (train ET OOD EC-Earth3)
# les charge explicitement. Jamais de fallback silencieux None (qui ferait
# recalculer les stats sur le GCM OOD = fuite de re-normalisation per-GCM).
import xarray as _xr
MEANS_PATH_V6 = f"{DRIVE_ROOT}/train/means_ACCESS-CM2_v6.nc"
STDS_PATH_V6  = f"{DRIVE_ROOT}/train/stds_ACCESS-CM2_v6.nc"

if not (Path(MEANS_PATH_V6).exists() and Path(STDS_PATH_V6).exists()):
    if IS_OOD_RUN:
        raise FileNotFoundError(
            f"OOD run ({GCM_ID}) : les stats train ACCESS-CM2 v6 sont OBLIGATOIRES "
            f"({MEANS_PATH_V6}). Lancer d'abord le run in-distribution qui les génère. "
            f"Recalculer les stats sur {GCM_ID} masquerait les biais moyens (audit ML)."
        )
    print("[Cell 3] Génération des stats train v6 (une fois) ...")
    _ds_access = _xr.open_dataset(f"{DRIVE_ROOT}/lr_ACCESS-CM2_v6.nc")
    _tr = _ds_access.sel(time=slice(K9_DATES["train"][0], K9_DATES["train"][1]))
    _tr = _tr[STAGE1_LR_VARS]
    _tr.mean(dim="time").to_netcdf(MEANS_PATH_V6)
    _tr.std(dim="time").to_netcdf(STDS_PATH_V6)
    _ds_access.close()
    print(f"[Cell 3] stats v6 sauvegardées : {MEANS_PATH_V6}")
# Vérifie que les stats couvrent bien les 22 vars (fail loud, pas de KeyError tardif)
_m_check = _xr.open_dataset(MEANS_PATH_V6)
_missing_stats = [v for v in STAGE1_LR_VARS if v not in _m_check.data_vars]
_m_check.close()
assert not _missing_stats, f"Stats v6 incomplètes, vars manquantes : {_missing_stats}"

pipeline = NetCDFDataPipeline(
    lr_path=LR_PATH_V6, hr_path=HR_PATH,
    static_path=STATIC_PATH if Path(STATIC_PATH).exists() else None,
    seq_len=int(CONFIG.data.seq_len),
    baseline_strategy=str(CONFIG.data.baseline_strategy),
    baseline_factor=int(CONFIG.data.baseline_factor),
    normalize=bool(CONFIG.data.normalize),
    nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
    precipitation_delta=float(CONFIG.data.precipitation_delta),
    lr_variables=STAGE1_LR_VARS,
    hr_variables=list(CONFIG.data.hr_variables),
    static_variables=list(CONFIG.data.static_variables) if CONFIG.data.get("static_variables") else [],
    means_path=MEANS_PATH_V6,   # EXPLICITE — jamais None (M1 + fix OOD)
    stds_path=STDS_PATH_V6,
    train_start_date=K9_DATES["train"][0], train_end_date=K9_DATES["train"][1],
    val_start_date=K9_DATES["val"][0],     val_end_date=K9_DATES["val"][1],
    test_start_date=K9_DATES["test"][0],   test_end_date=K9_DATES["test"][1],
    temporal_holdout_start_date=K9_DATES["holdout"][0],
    temporal_holdout_end_date=K9_DATES["holdout"][1],
)

# --- 11-node builder : 9-node + U850/V850 wind nodes -----------------------
builder = HeteroGraphBuilder(
    lr_shape=tuple(CONFIG.graph.lr_shape), hr_shape=tuple(CONFIG.graph.hr_shape),
    static_dataset=pipeline.get_static_dataset(), include_mid_layer=True,
    extended_9node=True, extended_v6_wind=True,
)
print(f"[Cell 3] builder dynamic nodes = {builder.dynamic_node_types}")
assert "U850" in builder.dynamic_node_types and "V850" in builder.dynamic_node_types

# --- Inject humid (9-node) + wind (V6) spatial metapaths -------------------
_existing = {m.name for m in CONFIG.encoder.metapaths}
_new_mps = [
    {"name": "Q850_spat", "src": "Q850", "relation": "spat_adj", "target": "Q850", "pool": "mean"},
    {"name": "W500_spat", "src": "W500", "relation": "spat_adj", "target": "W500", "pool": "mean"},
    {"name": "IVT_spat",  "src": "IVT",  "relation": "spat_adj", "target": "IVT",  "pool": "mean"},
    {"name": "U850_spat", "src": "U850", "relation": "spat_adj", "target": "U850", "pool": "mean"},
    {"name": "V850_spat", "src": "V850", "relation": "spat_adj", "target": "V850", "pool": "mean"},
]
for _m in _new_mps:
    if _m["name"] not in _existing:
        CONFIG.encoder.metapaths.append(_OC.create(_m))
print(f"[Cell 3] metapaths -> {[m.name for m in CONFIG.encoder.metapaths]}")

# --- Per-node channel routing (V6) -----------------------------------------
_LR_VARS = list(CONFIG.data.lr_variables)
_VI = {v: i for i, v in enumerate(_LR_VARS)}
def _idx(names): return [_VI[v] for v in names if v in _VI]
_ROUTE_IDX = {node: _idx(chans) for node, chans in V6_NODE_CHANNEL_ROUTING.items()}
_IVT_LEVELS = [lev for lev in ("850","500","250")
               if f"q_{lev}" in _VI and f"u_{lev}" in _VI and f"v_{lev}" in _VI]
print(f"[Cell 3] routing idx = {_ROUTE_IDX} | IVT levels = {_IVT_LEVELS}")

def _compute_ivt_nodes(lr0):
    acc = None
    for lev in _IVT_LEVELS:
        q = lr0[:, _VI[f"q_{lev}"]]; u = lr0[:, _VI[f"u_{lev}"]]; v = lr0[:, _VI[f"v_{lev}"]]
        term = q * torch.sqrt(u*u + v*v + 1e-12)
        acc = term if acc is None else acc + term
    if acc is None: acc = torch.zeros(lr0.shape[0], device=lr0.device, dtype=lr0.dtype)
    acc = (acc - acc.mean()) / (acc.std() + 1e-6)
    return acc.unsqueeze(1)

def convert_sample_to_batch(sample, builder, device):
    lr_seq = sample["lr"]                       # [seq, C_LR, lat, lon]
    seq_len = lr_seq.shape[0]
    lr_nodes = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes, dim=0)    # RCN drivers (full LR)
    lr0 = lr_nodes[0]
    _ivt = _compute_ivt_nodes(lr0)
    dyn = {}
    for nt in builder.dynamic_node_types:
        if nt in _ROUTE_IDX and _ROUTE_IDX[nt]:
            dyn[nt] = lr0[:, _ROUTE_IDX[nt]]     # Q850/W500/U850/V850 <- routed channels
        elif nt == "IVT":
            dyn[nt] = _ivt
        else:
            dyn[nt] = lr0                        # GP850/GP500/GP250 <- full LR
    hetero = builder.prepare_step_data(dyn).to(device)
    return {
        "lr": lr_tensor, "residual": sample["residual"],
        "baseline": sample.get("baseline"), "hetero": hetero,
        "time": sample.get("time"),
        "lr_grid": lr_seq,   # V6' : raw LR conditioning grid [seq, C_LR, H_LR, W_LR]
    }

def iterate_batches_v6(ds, builder, device):
    for s in ds:
        yield [convert_sample_to_batch(s, builder, device)]

train_dataset = pipeline.build_sequence_dataset(split="train", seq_len=int(CONFIG.data.seq_len), stride=int(CONFIG.data.stride), as_torch=True)
val_dataset   = pipeline.build_sequence_dataset(split="val",   seq_len=int(CONFIG.data.seq_len), stride=int(CONFIG.data.stride), as_torch=True)
test_dataset  = pipeline.build_sequence_dataset(split="test",  seq_len=int(CONFIG.data.seq_len), stride=int(CONFIG.data.stride), as_torch=True)
print("[Cell 3] datasets ready (11-node, lr_grid conditioning attached)")
"""


CELL_4 = """# >>> Cell 4 : Stack (encoder 11 vars + RCN 11x11 + regression_head + diffusion V6')
from st_cdgm.models.intelligible_encoder import IntelligibleVariableEncoder, IntelligibleVariableConfig
from st_cdgm.models.causal_rcn import RCNCell, RCNSequenceRunner
from st_cdgm.models.regression_head import GraphToGridDecoder
from st_cdgm.models import CausalDiffusionDecoder
from st_cdgm.models.edm_preconditioner import EDMConfig

RCN_DRIVER_DIM = None

def build_fresh_stack(seed: int):
    global RCN_DRIVER_DIM
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    allowed = set(builder.dynamic_node_types) | set(builder.static_node_types)
    enc_cfgs = [IntelligibleVariableConfig(name=m.name, meta_path=(m.src, m.relation, m.target),
                                            pool=m.get("pool", "mean"))
                for m in CONFIG.encoder.metapaths if m.src in allowed and m.target in allowed]
    # static SP_HR conditioning variable (as in 9-node _build_encoder)
    if pipeline.get_static_dataset() is not None:
        enc_cfgs.append(IntelligibleVariableConfig(name="static",
            meta_path=("SP_HR", "causes", "GP850"), pool="mean"))
    encoder = IntelligibleVariableEncoder(configs=enc_cfgs,
        hidden_dim=int(CONFIG.encoder.hidden_dim),
        conditioning_dim=int(CONFIG.encoder.conditioning_dim)).to(DEVICE)
    num_vars = len(enc_cfgs)
    assert num_vars == V6_NUM_ENCODER_VARS, f"num_vars={num_vars} != {V6_NUM_ENCODER_VARS} (11)"
    print(f"   encoder : {num_vars} intelligible variables (11-node V6)")

    # P0-A fix (audit IA) : lr_grid_to_nodes exige un tenseur 3D [C, H, W] —
    # un zeros 2D (lr_shape seul) levait ValueError. Probe avec les 22 canaux.
    _probe = builder.lr_grid_to_nodes(
        torch.zeros(len(STAGE1_LR_VARS), *tuple(CONFIG.graph.lr_shape))
    )
    RCN_DRIVER_DIM = _probe.shape[-1]
    rcn_cell = RCNCell(num_vars=num_vars, hidden_dim=int(CONFIG.rcn.hidden_dim),
        driver_dim=RCN_DRIVER_DIM, reconstruction_dim=RCN_DRIVER_DIM,
        dropout=float(CONFIG.rcn.dropout)).to(DEVICE)
    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get("detach_interval"))
    print(f"   rcn_cell A_dag shape = {tuple(rcn_cell.A_dag.shape)} (target 11x11)")

    rh = CONFIG.two_stage.regression_head
    regression_head = GraphToGridDecoder(d_model=int(rh.d_model),
        hr_h=int(CONFIG.diffusion.height), hr_w=int(CONFIG.diffusion.width),
        intermediate_h=int(rh.intermediate_h), intermediate_w=int(rh.intermediate_w),
        n_heads=int(rh.n_heads), refine_channels=int(rh.refine_channels),
        output_channels=1).to(DEVICE)

    UNET = OmegaConf.to_container(CONFIG.diffusion.unet_kwargs, resolve=True)
    for k in ("down_block_types", "up_block_types"):
        if k in UNET and isinstance(UNET[k], list): UNET[k] = tuple(UNET[k])
    UNET["projection_class_embeddings_input_dim"] = num_vars * int(CONFIG.diffusion.conditioning_dim)
    diffusion = CausalDiffusionDecoder(
        in_channels=1, conditioning_dim=int(CONFIG.diffusion.conditioning_dim),
        height=int(CONFIG.diffusion.height), width=int(CONFIG.diffusion.width),
        unet_kwargs=UNET, scheduler_type="edm_karras",
        edm_config=EDMConfig.from_yaml_dict(CONFIG.diffusion.get("edm", {})),
        causal_concat=True,
        lr_conditioning_channels=LR_COND_CHANNELS,   # <<< V6' PIVOT
    ).to(DEVICE)
    print(f"   diffusion conv_in in_channels = {diffusion.unet.conv_in.in_channels} (=3+{LR_COND_CHANNELS})")
    return dict(encoder=encoder, rcn_cell=rcn_cell, rcn_runner=rcn_runner,
                regression_head=regression_head, diffusion=diffusion, num_vars=num_vars)

print("[Cell 4] build_fresh_stack (11-node + lr_conditioning) defined")
"""


CELL_5 = """# >>> Cell 5 : Helpers (REAL imports) + G_phys 11x11 + schedule
import copy, subprocess as _sp
from st_cdgm.training.training_loop import train_epoch_stage1
from st_cdgm.training.two_stage import (
    freeze_stage1, causal_ablation_check, precompute_stage1_outputs, train_epoch_stage2_cached,
)
from st_cdgm.training.physics_prior import build_physical_mask, VAR_LABELS_V6, EXPECTED_EDGES_V6
from path_c_plus.scripts.option_c_helpers import PATHCPLUS_HYPERPARAM_OVERRIDES
from scripts.finetune_stage1_bundle_b import schedule_lambdas, DEFAULT_HYPERPARAMS

from st_cdgm.v6_constants import V6_LAMBDA_L1_START, V6_LAMBDA_L1_END

HP = copy.deepcopy(DEFAULT_HYPERPARAMS)
HP.update({
    "lambda_dag_prior": PATHCPLUS_HYPERPARAM_OVERRIDES["lambda_dag_prior"],  # 0.40
    # P2-i fix (audit IA) : le plan V6' §2 spécifie λ_l1 ×0.7 (adaptation
    # 11-node) — utiliser les constantes V6, pas les overrides 9-node.
    # (Audit Math : strictement inutile car gradient L1 par-entrée, mais
    # inoffensif — on suit le plan pré-enregistré.)
    "lambda_l1_start":  V6_LAMBDA_L1_START,   # 0.028 (9-node : 0.04)
    "lambda_l1_end":    V6_LAMBDA_L1_END,     # 0.0035 (9-node : 0.005)
    "g_phys_alpha":     PATHCPLUS_HYPERPARAM_OVERRIDES["g_phys_alpha"],
    "dag_gate_warmup_start_epoch": PATHCPLUS_HYPERPARAM_OVERRIDES["dag_gate_warmup_start_epoch"],
    "dag_gate_warmup_end_epoch":   PATHCPLUS_HYPERPARAM_OVERRIDES["dag_gate_warmup_end_epoch"],
})

# G_phys 11x11 (V6 : 9-node humid-QG + 4 wind edges U850/V850 -> IVT/SP_HR)
G_phys = build_physical_mask(num_vars=11, var_labels=VAR_LABELS_V6,
                             expected_edges=EXPECTED_EDGES_V6).to(DEVICE)
print(f"[Cell 5] G_phys 11x11 : |edges|={int((G_phys != 0).sum())} (expect 14)")
assert G_phys.shape == (11, 11)

PRE_REG_COMMIT = _sp.check_output(["git","-C",REPO_DIR,"rev-parse","--short","HEAD"]).decode().strip()
print(f"[Cell 5] pre-registration commit = {PRE_REG_COMMIT}")
print("[Cell 5] thresholds : path_c_plus/audit/V6_PRIME_seuils_preregistered.json")
"""


CELL_6 = """# >>> Cell 6 : Stage 1 (11-node) training from scratch (seed 42)
import torch.nn.functional as F
SEED = 42
S1_EPOCHS = 3 if SMOKE_MODE else 75

stack = build_fresh_stack(SEED)
encoder, rcn_cell = stack["encoder"], stack["rcn_cell"]
rcn_runner, regression_head = stack["rcn_runner"], stack["regression_head"]
diffusion = stack["diffusion"]

opt_s1 = torch.optim.AdamW(
    list(encoder.parameters()) + list(rcn_cell.parameters()) + list(regression_head.parameters()),
    lr=float(CONFIG.training.learning_rate), weight_decay=1e-4)
ts = CONFIG.two_stage.stage1
for ep in range(S1_EPOCHS):
    sch = schedule_lambdas(ep, S1_EPOCHS, HP)
    m = train_epoch_stage1(
        encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
        optimizer=opt_s1, data_loader=iterate_batches_v6(train_dataset, builder, DEVICE),
        device=DEVICE, epoch_idx=ep,
        lambda_reg=float(ts.lambda_reg), beta_rec=float(ts.beta_rec),
        gamma_dag_max=float(ts.gamma_dag_max), gamma_dag_warmup_epochs=int(ts.gamma_dag_warmup_epochs),
        lambda_l1=float(sch["lambda_l1"]), lambda_dag_prior=float(sch["lambda_dag_prior"]),
        dag_prior=G_phys, dag_grad_gate_value=float(sch["dag_grad_gate"]),
        abort_on_collapse=True, collapse_threshold=0.05,
        dag_floor_projection=True, dag_floor_min_norm=0.10,
        gradient_clipping=CONFIG.training.gradient_clipping)
    if (ep+1) % 5 == 0 or ep == 0:
        print(f"  S1 ep{ep+1}/{S1_EPOCHS} loss={m['loss_total']:.4f} A_dag_norm={float(rcn_cell.A_dag.norm()):.3f}")
print("[Cell 6] Stage 1 (11-node) trained")
"""


CELL_7 = """# >>> Cell 7 : O3 gate + freeze + A_dag freeze verification
# P0-B fix (audit IA) : causal_ablation_check appelle
# iterate_batches_fn(data_loader, builder, device) — 3 arguments positionnels.
# iterate_batches_v6 a exactement cette signature → le passer DIRECTEMENT.
ablation = causal_ablation_check(
    encoder=encoder, rcn_runner=rcn_runner, rcn_cell=rcn_cell,
    regression_head=regression_head, data_loader=val_dataset,
    iterate_batches_fn=iterate_batches_v6,
    builder=builder, device=DEVICE,
    n_samples=int(CONFIG.two_stage.causal_ablation.n_samples),
    threshold=float(CONFIG.two_stage.causal_ablation.threshold))
print(f"[Cell 7] O3 gate ratio={ablation['ratio']:.4f} passes={ablation['passes']}")
if not ablation["passes"]:
    raise RuntimeError("O3 gate FAILED — DAG decorative, abort Stage 2")

freeze_stage1(encoder, rcn_runner.cell, regression_head)
assert not rcn_cell.A_dag.requires_grad, "A_dag NOT frozen"
print("[Cell 7] Stage 1 frozen (encoder + rcn_cell incl. A_dag + regression_head)")
"""


CELL_8 = """# >>> Cell 8 : BS32b cache WITH lr_fields (full-LR conditioning)
cache = precompute_stage1_outputs(
    encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
    train_dataset=train_dataset,
    iterate_batches_fn=lambda s: convert_sample_to_batch(s, builder, DEVICE),
    device=DEVICE, dag_variants=["normal"], cache_lr_fields=True)
assert "lr_fields" in cache, "V6' cache MUST contain lr_fields"
print(f"[Cell 8] cache keys = {sorted(cache.keys())}")
print(f"[Cell 8] lr_fields cached shape = {tuple(cache['lr_fields'].shape)} (native res)")

from torch.utils.data import DataLoader as _DL, Dataset as _DS
class _CondDS(_DS):
    def __init__(self, c):
        self.mu=c["mu_HR"]; self.base=c["baseline_log"]; self.delta=c["delta_target"]
        self.mask=c["valid_mask"]; self.lr=c["lr_fields"]
    def __len__(self): return self.mu.shape[0]
    def __getitem__(self, i):
        return {"mu_HR": self.mu[i], "baseline_log": self.base[i], "delta_target": self.delta[i],
                "valid_mask": self.mask[i], "lr_fields": self.lr[i]}
BATCH = 32 if SMOKE_MODE else 64
cached_loader = _DL(_CondDS(cache), batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
print(f"[Cell 8] cached_loader ready (batch={BATCH}, {len(cached_loader)} batches)")
"""


CELL_9 = """# >>> Cell 9 : SMOKE Stage 2 + A2 monitor (M6) BEFORE full run
import copy, torch.nn.functional as F
diffusion_smoke = copy.deepcopy(diffusion)
opt_smoke = torch.optim.AdamW(diffusion_smoke.parameters(), lr=float(CONFIG.two_stage.stage2.lr), weight_decay=1e-4)
for ep in range(3):
    m = train_epoch_stage2_cached(diffusion_decoder=diffusion_smoke, optimizer=opt_smoke,
        cached_dataloader=cached_loader, device=DEVICE, use_amp=True, gradient_clipping=1.0, log_every=50)
    print(f"SMOKE ep{ep+1} loss_diff={m['loss_diff']:.4f}  corr(D_y,mu_HR)={m.get('corr_dy_mu', float('nan')):+.3f}")
    # Declencheur V6'.1 pre-enregistre (audit ML) : si la diffusion ANNULE
    # mu_HR (anti-copy A3/A11), corr(D_y, mu_HR) devient fortement negative.
    _c = m.get("corr_dy_mu", float("nan"))
    if _c == _c and _c < -0.3:
        raise RuntimeError(
            f"SMOKE ABORT : corr(D_y, mu_HR) = {_c:.3f} < -0.3 — la diffusion "
            "depense sa capacite a annuler mu_HR. Basculer sur la variante "
            "V6'.1 pre-enregistree (delta = HR - baseline, mu_HR conditioning seul).")

diffusion_smoke.eval()
def _val_loss(zero_lr=False, n=5):
    tot, k = 0.0, 0
    with torch.no_grad():
        for b in cached_loader:
            mu=b["mu_HR"].to(DEVICE); bl=b["baseline_log"].to(DEVICE); dt=b["delta_target"].to(DEVICE)
            lr=b["lr_fields"].to(DEVICE)
            if lr.shape[-2:] != dt.shape[-2:]:
                lr = F.interpolate(lr, size=dt.shape[-2:], mode="bilinear", align_corners=False)
            if zero_lr: lr = torch.zeros_like(lr)
            tot += float(diffusion_smoke.compute_loss_edm(target=dt, mu_HR=mu, baseline_log=bl, lr_fields=lr)); k += 1
            if k >= n: break
    return tot / max(1, k)
l_on, l_off = _val_loss(False), _val_loss(True)
print(f"[SMOKE A2] loss(LR on)={l_on:.4f} loss(LR->0)={l_off:.4f} degradation={100*(l_off-l_on)/max(1e-6,l_on):+.1f}%")
print("  ✓ denoiser USES LR" if l_off > l_on*1.02 else "  ⚠ A2 <2% — investigate before full run")
"""


CELL_10 = """# >>> Cell 10 : Stage 2 FULL (full-LR conditioning) + EMA + persist
import copy, os
S2_EPOCHS = 5 if SMOKE_MODE else 150
ema = copy.deepcopy(diffusion).eval()
for p in ema.parameters(): p.requires_grad_(False)
opt_s2 = torch.optim.AdamW(diffusion.parameters(), lr=float(CONFIG.two_stage.stage2.lr), weight_decay=1e-4)
for ep in range(S2_EPOCHS):
    m = train_epoch_stage2_cached(diffusion_decoder=diffusion, optimizer=opt_s2,
        cached_dataloader=cached_loader, device=DEVICE, use_amp=True, gradient_clipping=1.0,
        log_every=50, ema_model=ema, ema_decay=0.9999)
    if (ep+1) % 10 == 0 or ep == 0:
        print(f"[S2 ep{ep+1}/{S2_EPOCHS}] loss_diff={m['loss_diff']:.5f}")
save_dir = f"{DRIVE_ROOT}/oracle_v6_prime/seed_42"; os.makedirs(save_dir, exist_ok=True)
torch.save({"diffusion_state_dict": diffusion.state_dict(), "ema_state_dict": ema.state_dict(),
            "encoder_state_dict": encoder.state_dict(), "rcn_cell_state_dict": rcn_cell.state_dict(),
            "regression_head_state_dict": regression_head.state_dict(),
            "lr_conditioning_channels": LR_COND_CHANNELS, "stage2_cond_lr_vars": STAGE2_COND_LR_VARS,
            "A_dag_final": rcn_cell.A_dag.detach().cpu().numpy()},
           f"{save_dir}/v6_prime_seed42.pth")
print(f"[Cell 10] saved → {save_dir}/v6_prime_seed42.pth")
"""


CELL_11 = """# >>> Cell 11 : Eval FULL TEST SPLIT + ablations A1/A2 — END-TO-END
# P0 fix (audits Recherche + ML) : verdict co-primaire sur le TEST SPLIT
# COMPLET (2 ans), pas 16 echantillons (~0.16 evenement p99/pixel sur 16 jours
# = statistiquement indefini, incomparable aux references).
# Protocole : VERDICT = full split, K=64. ATTRIBUTION A1/A2 = full split, K=16
# (3 conditions au MEME K -> differences comparables, compute maitrise).
import torch.nn.functional as F, numpy as np, json
from st_cdgm.evaluation.eval_metrics_dual_convention import evaluate_ensemble

K_VERDICT   = 8 if SMOKE_MODE else 64
K_ABLATION  = 4 if SMOKE_MODE else 16
NUM_STEPS   = 32
EVAL_BATCH  = 16   # sampling batch size (memoire)

# --- climatology per-pixel thresholds (Convention A, ETCCDI) ---------------
CLIM_PATH = f"{DRIVE_ROOT}/oracle_9node/seed_42/phase8/clim_p95_p99.npz"
_alt = f"{DRIVE_ROOT}/ckpt_v2_corrdiff_normal/clim_p95_p99.npz"
_cp = CLIM_PATH if Path(CLIM_PATH).exists() else _alt
_clim = np.load(_cp)
# P1-D fix (audit IA) : .to(DEVICE) — mismatch device CPU/CUDA sinon
clim_p99 = torch.from_numpy(_clim["clim_p99"].astype(np.float32)).to(DEVICE)
clim_p95 = torch.from_numpy(_clim["clim_p95"].astype(np.float32)).to(DEVICE)
print(f"[Cell 11] climatology loaded from {_cp}")

# --- materialise test conditioning on the FULL split ------------------------
test_cache = precompute_stage1_outputs(
    encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
    train_dataset=test_dataset,
    iterate_batches_fn=lambda s: convert_sample_to_batch(s, builder, DEVICE),
    device=DEVICE, dag_variants=["normal"], cache_lr_fields=True)

N_TOTAL = test_cache["mu_HR"].shape[0]
N_TEST = min(16, N_TOTAL) if SMOKE_MODE else N_TOTAL   # FULL split hors smoke
print(f"[Cell 11] test samples : {N_TEST} / {N_TOTAL}")
mu_all    = test_cache["mu_HR"][:N_TEST]
base_all  = test_cache["baseline_log"][:N_TEST]
delta_all = test_cache["delta_target"][:N_TEST]
lr_all    = test_cache["lr_fields"][:N_TEST]

def sample_ensemble(zero_mu=False, zero_lr=False, K=None):
    \"\"\"Batched ensemble sampler avec commutateurs A1/A2 (M3 : lr_fields requis).
    cfg_scale=0.0 (P2-ii audit IA) : semantique conditioned-only identique a
    1.0 sur edm_karras, mais evite tout double-forward CFG.\"\"\"
    if K is None: K = K_VERDICT
    ema.eval()
    members = []
    with torch.no_grad():
        for k in range(K):
            torch.manual_seed(1000 + k)
            chunks = []
            for i0 in range(0, N_TEST, EVAL_BATCH):
                sl = slice(i0, min(i0 + EVAL_BATCH, N_TEST))
                mu_ = mu_all[sl].to(DEVICE); bl_ = base_all[sl].to(DEVICE)
                lr_ = lr_all[sl].to(DEVICE)
                if lr_.shape[-2:] != mu_.shape[-2:]:
                    lr_ = F.interpolate(lr_, size=mu_.shape[-2:], mode="bilinear", align_corners=False)
                if zero_mu: mu_ = torch.zeros_like(mu_)
                if zero_lr: lr_ = torch.zeros_like(lr_)   # A2 = zero post-norm
                o = ema.sample(conditioning=None, num_steps=NUM_STEPS,
                               scheduler_type="edm_karras", cfg_scale=0.0,
                               mu_HR=mu_, baseline_log=bl_, lr_fields=lr_)
                chunks.append(o.residual.cpu())
            members.append(torch.cat(chunks, dim=0))
    return torch.stack(members, 0)   # [K, N_TEST, 1, H, W] sur CPU

mu_t, base_t, delta_t = mu_all.to(DEVICE), base_all.to(DEVICE), delta_all.to(DEVICE)
def M(ens):
    return evaluate_ensemble(ens.to(DEVICE), mu_t, base_t, delta_t, clim_p99, clim_p95)

print(f"[Cell 11] VERDICT sampling (full split, K={K_VERDICT}) ...")
res_full = M(sample_ensemble(K=K_VERDICT))
print(f"[Cell 11] ATTRIBUTION sampling (K={K_ABLATION} x 3 conditions) ...")
res_refA  = M(sample_ensemble(K=K_ABLATION))
res_a1    = M(sample_ensemble(zero_mu=True, K=K_ABLATION))
res_a2    = M(sample_ensemble(zero_lr=True, K=K_ABLATION))

results = {
    "F1_p99_pergrid": res_full["conv_A_F1p99"],
    "F1_p99_pooled":  res_full["conv_B_F1p99"],
    "RMSE": res_full["rmse"], "Pearson": res_full["pearson_global"],
    "Rx1day_bias": res_full["rx1day_bias"], "RAPSD": res_full["rapsd_distance"],
    "n_test": int(N_TEST), "K_verdict": int(K_VERDICT), "K_ablation": int(K_ABLATION),
    # Attribution : triple au MEME K (comparabilite interne, semantique A1
    # INFORMATIONNELLE : conditioning ablate, mu reel conserve en recomposition
    # — pre-declare dans le prereg)
    "ref_KA_F1_pooled":    res_refA["conv_B_F1p99"],
    "A1_mu_off_F1_pooled": res_a1["conv_B_F1p99"],
    "A2_lr_off_F1_pooled": res_a2["conv_B_F1p99"],
    "ref_KA_F1_pergrid":    res_refA["conv_A_F1p99"],
    "A1_mu_off_F1_pergrid": res_a1["conv_A_F1p99"],
    "A2_lr_off_F1_pergrid": res_a2["conv_A_F1p99"],
}
print(); print("=== V6' RESULTS (full test split) ===")
for k, v in results.items(): print(f"  {k:26s} = {v}")
print(); print(f"  A1 (causal) contribution pooled  = {results['ref_KA_F1_pooled']-results['A1_mu_off_F1_pooled']:+.4f}")
print(f"  A2 (full-LR) contribution pooled = {results['ref_KA_F1_pooled']-results['A2_lr_off_F1_pooled']:+.4f}")
"""


CELL_12 = """# >>> Cell 12 : Verdict vs seuils pre-enregistres + gate OOD
import json, os
seuils = json.load(open("path_c_plus/audit/V6_PRIME_seuils_preregistered.json"))
thr_pg = seuils["targets_to_beat"]["co_primary_1_per_gridpoint"]
thr_pl = seuils["targets_to_beat"]["co_primary_2_pooled"]

# --- P0 (audit Recherche) : les references per-gridpoint 0.841/0.816 ont ete
# calculees avec l'ANCIENNE metrique Conv A (quantile scalaire, buggee).
# Elles doivent etre RECOMPUTEES avec la metrique corrigee (broadcast per-pixel)
# via _eval_3way_dual_convention re-execute. Ce notebook cherche le fichier de
# references recomputees ; sinon le verdict per-gridpoint est marque STALE.
RECOMPUTED_REFS = f"{DRIVE_ROOT}/oracle_v6_prime/recomputed_pergrid_references.json"
if Path(RECOMPUTED_REFS).exists():
    _refs = json.load(open(RECOMPUTED_REFS))
    ref_pg_noncausal = float(_refs["noncausal_v4_F1p99_pergrid"])
    ref_pg_v5        = float(_refs["v5_causal_F1p99_pergrid"])
    refs_status = "RECOMPUTED"
else:
    ref_pg_noncausal = float(thr_pg["noncausal_v4"])
    ref_pg_v5        = float(thr_pg["v5_causal_seed42"])
    refs_status = "STALE_OLD_METRIC"
    print("  !! References per-gridpoint NON recomputees avec la metrique corrigee")
    print("  !! -> verdict co-primaire 1 = provisoire. Re-executer le 3-way eval")
    print(f"  !! et sauvegarder {RECOMPUTED_REFS}")

pg, pl = results["F1_p99_pergrid"], results["F1_p99_pooled"]
v_pg = "PASS" if (pg >= ref_pg_noncausal and pg >= ref_pg_v5) else "FAIL"
if refs_status != "RECOMPUTED":
    v_pg = f"{v_pg}_PROVISOIRE_REFS_STALE"
v_pl = ("PASS_STRONG" if pl >= thr_pl["PASS_STRONG"] else "PASS_TARGET" if pl >= thr_pl["PASS_TARGET"]
        else "PASS_MINIMAL" if pl >= thr_pl["PASS_MINIMAL"] else "FAIL")

# --- Gate OOD (P1 audit Recherche + fix ML) : le verdict n'est FINAL qu'avec
# l'OOD EC-Earth3 complete. Le run OOD = relancer CE notebook avec
# GCM_ID="EC-Earth3" (Cell 2) — il ecrira son propre verdict ; ce champ trace.
OOD_VERDICT_PATH = f"{DRIVE_ROOT}/oracle_v6_prime/seed_42/v6_prime_verdict_EC-Earth3.json"
ood_status = "DONE" if Path(OOD_VERDICT_PATH).exists() else "PENDING"

verdict = {
    "gcm": GCM_ID,
    "per_gridpoint": v_pg, "pooled": v_pl,
    "pergrid_refs_status": refs_status,
    "at_least_one_coprimary": (v_pg.startswith("PASS") or v_pl != "FAIL"),
    "ood_EC-Earth3": ood_status if not IS_OOD_RUN else "THIS_IS_THE_OOD_RUN",
    "final": (ood_status == "DONE" and refs_status == "RECOMPUTED") if not IS_OOD_RUN else True,
    "results": results,
}
print("=== V6' VERDICT ===")
print(f"  per-gridpoint : {v_pg}  (F1={pg:.4f} vs noncausal {ref_pg_noncausal} / V5 {ref_pg_v5} [{refs_status}])")
print(f"  pooled        : {v_pl}  (F1={pl:.4f} vs target {thr_pl['PASS_TARGET']})")
print(f"  OOD EC-Earth3 : {verdict['ood_EC-Earth3']}")
print(f"  FINAL         : {verdict['final']}  (exige OOD DONE + refs RECOMPUTED)")

os.makedirs(f"{DRIVE_ROOT}/oracle_v6_prime/seed_42", exist_ok=True)
_out = (f"{DRIVE_ROOT}/oracle_v6_prime/seed_42/v6_prime_verdict_{GCM_ID}.json"
        if IS_OOD_RUN else f"{DRIVE_ROOT}/oracle_v6_prime/seed_42/v6_prime_verdict.json")
json.dump(verdict, open(_out, "w"), indent=2)
print(f"[Cell 12] verdict saved -> {_out}")
"""


def build():
    cells = [md(CELL_0), code(CELL_1), code(CELL_2), code(CELL_3), code(CELL_4),
             code(CELL_5), code(CELL_6), code(CELL_7), code(CELL_8), code(CELL_9),
             code(CELL_10), code(CELL_11), code(CELL_12)]
    return {"cells": cells,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python", "version": "3.11"},
                         "v6_prime_metadata": {
                             "experiment_id": "V6_PRIME_seed42_fullLR_11node",
                             "plan": "path_c_plus/audit/PLAN_V6_PRIME.md",
                             "pre_registration": "path_c_plus/audit/V6_PRIME_seuils_preregistered.json",
                             "structural_changes": ["Stage 2 full-LR conditioning (3->3+22 channels)",
                                                     "Stage 1 11-node (U850/V850 wind graph nodes, G_phys 11x11)"],
                             "audit_median_p_beat_noncausal": 0.75}},
            "nbformat": 4, "nbformat_minor": 5}


def main():
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"V6' FULL notebook written : {OUT} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
