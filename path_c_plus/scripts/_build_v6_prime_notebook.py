"""
Build the V6' (V6-prime) notebook — `st_cdgm_v6_prime_seed42.ipynb`.

Design decision (integration reality, documented for the user)
--------------------------------------------------------------
The encoder variables (num_vars, hence G_phys size) are driven by
CONFIG.encoder.metapaths, NOT by builder.dynamic_node_types. Adding U850/V850
as *graph nodes* would require new metapaths + an 11x11 G_phys — a topology
change with runtime risk.

BUT the audit's core insight is INFORMATIONAL : Stage 2 must see the LR fields.
And U850/V850 are ALREADY in the 21 LR channels that condition Stage 2. So the
wind information reaches Stage 2 through the full-LR conditioning regardless of
graph topology.

Therefore V6'.0 = the cleanest, most attributable, lowest-risk pivot :
  - Stage 1 : IDENTICAL to the proven 9-node seed 42 (num_vars=9, G_phys 9x9,
    15 base LR vars). The existing trained checkpoint can even be REUSED — no
    Stage-1 retrain needed for the pivot's attribution.
  - Stage 2 : retrained with full-LR conditioning (21 channels = 15 base + 6
    climat features), the SINGLE structural change.

The U850/V850 graph-node variant + climat features in the RCN drivers is
deferred to V6'.1 (requires the metapath config schema).

This keeps the pre-registered ablations clean :
  A1 (mu_HR -> 0) : contribution of the causal Stage 1
  A2 (lr_fields -> 0) : contribution of the full-LR conditioning (the pivot)

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
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


CELL_0 = """# V6' (V6-prime) — Pivot conditioning full-LR (ST-CDGM Path C+ Option C)

**Validé 5/5 par l'audit indépendant (2026-06-30).** Médiane P(battre noncausal sur ≥1 co-primaire) ≈ **75%** (V6 MVP rejeté était à ~15%).

## Le changement (un seul, attribuable)
Le vrai goulot du Stage 2 causal était **informationnel** : il ne voyait que `[y_noisy, mu_HR, baseline]` (3 canaux) là où le noncausal voit l'information LR complète. Toy diffusion (audit Math) : F1@p99 = 0.053 conditioning pauvre vs **0.639** full-info.

**V6'.0** : Stage 2 conditionné sur les **21 champs LR** (15 base + 6 features Climat), upsamplés bilinéairement.
```
UNet_in = [c_in·y_noisy, mu_HR, baseline_log, LR_1..LR_21]   # 3 → 24 canaux
```
Pattern CorrDiff / StormCast / Rampal 2025 — sur-ensemble strict de l'information du baseline (0.550) ET du causal.

## Design V6'.0 (décision d'intégration, cf. `_build_v6_prime_notebook.py`)
- **Stage 1 : IDENTIQUE au 9-node seed 42** (num_vars=9, G_phys 9×9, 15 vars). Réutilisable tel quel → **le checkpoint Stage 1 existant peut être rechargé** (pas de retrain Stage 1 pour l'attribution du pivot).
- **Stage 2 : retrain avec conditioning full-LR** (21 canaux) — LE changement.
- Nœuds-graphe U850/V850 + climat dans les drivers RCN → **V6'.1** (U850/V850 sont déjà dans les 21 canaux LR côté conditioning).

## Garde-fous (pré-enregistrés — `V6_PRIME_seuils_preregistered.json`)
- **M1** normalisation LR z-score figée train (pipeline K5) ; A2 = zéro post-normalisation
- **M3** parité inférence : `sample()` lève si `lr_fields=None` sur decoder LR-conditionné
- **A1** (mu_HR→0) / **A2** (LR→0) monitorées **dès le SMOKE**
- Métrique **co-primaire** : per-gridpoint ETCCDI (battre 0.816, ne pas régresser sous V5=0.841) + pooled (0.550)
"""


CELL_1 = """# >>> Cell 1 : Bootstrap Colab + git sync (V6' — P0 fix sys.path src/)
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

# P0 FIX (audit IA) : internal modules do `from st_cdgm...` — add BOTH repo root
# (for `path_c_plus`, `scripts`) AND repo/src (for `st_cdgm`).
for _p in (REPO_DIR, str(Path(REPO_DIR) / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for pkg in ["torch_geometric", "diffusers", "omegaconf"]:
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_sha = subprocess.check_output(["git", "-C", REPO_DIR, "rev-parse", "HEAD"]).decode().strip()
print(f"[Cell 1] Bootstrap OK — commit {_sha[:8]} — sys.path has src/ (P0 fix)")
"""


CELL_2 = """# >>> Cell 2 : Config + V6' constants + LR conditioning setup
import torch, numpy as np
from omegaconf import OmegaConf

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SMOKE_MODE = False   # set True for the mandatory 4h smoke (fewer epochs/batches)

from st_cdgm.v6_constants import (
    NONCAUSAL_15_VARS, V6_LR_VARS_OBLIGATORY, V6_LR_VARS_FULL,
    V6_PRIME_LR_CONDITIONING_CHANNELS, assert_lr_vars_match,
)

# --- V6'.0 : Stage 1 uses the ORIGINAL 15 vars (9-node identical) ----------
#     Stage 2 conditioning uses the FULL 21 (obligatory) LR fields.
STAGE1_LR_VARS = list(NONCAUSAL_15_VARS)          # 15 — Stage 1 unchanged
STAGE2_COND_LR_VARS = list(V6_LR_VARS_OBLIGATORY)  # 21 — full-LR conditioning
USE_IVT_72H_BONUS = True
if USE_IVT_72H_BONUS:
    STAGE2_COND_LR_VARS = list(V6_LR_VARS_FULL)    # 22 (+ ivt_persistence_72h)
LR_COND_CHANNELS = len(STAGE2_COND_LR_VARS)
print(f"[Cell 2] Stage 1 LR vars = {len(STAGE1_LR_VARS)} (9-node identical)")
print(f"[Cell 2] Stage 2 LR conditioning channels = {LR_COND_CHANNELS}")

# --- Base config (9-node seed 42) ------------------------------------------
CONFIG = OmegaConf.load("config/training_config.yaml")
_corr = OmegaConf.load("config/training_config_corrdiff_normal.yaml")
CONFIG = OmegaConf.merge(CONFIG, _corr)
OmegaConf.set_struct(CONFIG, False)
CONFIG.data.lr_variables = STAGE1_LR_VARS   # Stage 1 = 15 vars (unchanged)

EXTENDED_9NODE = True   # 9-node graph exactly as seed 42

SEEDS = [42]
K9_DATES = {
    "train":   ("1980-01-01", "2009-12-31"),
    "val":     ("2010-01-01", "2011-12-31"),
    "test":    ("2012-01-01", "2013-12-31"),
    "holdout": ("2014-01-01", "2014-12-31"),
}
print("[Cell 2] V6' config ready")
"""


CELL_3 = """# >>> Cell 3 : Pipeline + 9-node builder + datasets + convert_sample_to_batch
#     Stage 1 pipeline = 15 vars. Stage 2 LR-conditioning fields come from the
#     SAME augmented LR NetCDF (produced by preprocess_v6_lr.py) but a SEPARATE
#     channel selection (STAGE2_COND_LR_VARS).
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder

LR_PATH_V6   = f"{DRIVE_ROOT}/lr_ACCESS-CM2_v6.nc"   # from preprocess_v6_lr.py
HR_PATH      = f"{DRIVE_ROOT}/hr_NIWA-REMS.nc"
STATIC_PATH  = f"{DRIVE_ROOT}/static_HR_v6.nc"
MEANS_PATH   = f"{DRIVE_ROOT}/train/means_ACCESS-CM2.nc"
STDS_PATH    = f"{DRIVE_ROOT}/train/stds_ACCESS-CM2.nc"

# Stage 1 pipeline (15 vars, 9-node)
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
    means_path=MEANS_PATH if Path(MEANS_PATH).exists() else None,
    stds_path=STDS_PATH if Path(STDS_PATH).exists() else None,
    train_start_date=K9_DATES["train"][0], train_end_date=K9_DATES["train"][1],
    val_start_date=K9_DATES["val"][0],     val_end_date=K9_DATES["val"][1],
    test_start_date=K9_DATES["test"][0],   test_end_date=K9_DATES["test"][1],
    temporal_holdout_start_date=K9_DATES["holdout"][0],
    temporal_holdout_end_date=K9_DATES["holdout"][1],
)

# SEPARATE pipeline handle for the Stage-2 LR conditioning fields (21/22 vars).
# M1 : same normalization convention (train-frozen z-score via means/stds path
# if present, else K5 train-window stats). The conditioning fields are the
# SAME physical variables, just a wider channel selection.
pipeline_cond = NetCDFDataPipeline(
    lr_path=LR_PATH_V6, hr_path=HR_PATH,
    static_path=STATIC_PATH if Path(STATIC_PATH).exists() else None,
    seq_len=int(CONFIG.data.seq_len),
    baseline_strategy=str(CONFIG.data.baseline_strategy),
    baseline_factor=int(CONFIG.data.baseline_factor),
    normalize=bool(CONFIG.data.normalize),
    nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
    precipitation_delta=float(CONFIG.data.precipitation_delta),
    lr_variables=STAGE2_COND_LR_VARS,
    hr_variables=list(CONFIG.data.hr_variables),
    static_variables=[],
    means_path=MEANS_PATH if Path(MEANS_PATH).exists() else None,
    stds_path=STDS_PATH if Path(STDS_PATH).exists() else None,
    train_start_date=K9_DATES["train"][0], train_end_date=K9_DATES["train"][1],
    val_start_date=K9_DATES["val"][0],     val_end_date=K9_DATES["val"][1],
    test_start_date=K9_DATES["test"][0],   test_end_date=K9_DATES["test"][1],
    temporal_holdout_start_date=K9_DATES["holdout"][0],
    temporal_holdout_end_date=K9_DATES["holdout"][1],
)

builder = HeteroGraphBuilder(
    lr_shape=tuple(CONFIG.graph.lr_shape),
    hr_shape=tuple(CONFIG.graph.hr_shape),
    static_dataset=pipeline.get_static_dataset(),
    include_mid_layer=True,
    extended_9node=True,       # 9-node EXACTLY as seed 42
    extended_v6_wind=False,    # V6'.0 : no graph-node change (U850/V850 reach
                               # Stage 2 via LR conditioning instead)
)
print(f"[Cell 3] builder dynamic nodes = {builder.dynamic_node_types}")

train_dataset = pipeline.build_sequence_dataset(split="train", seq_len=int(CONFIG.data.seq_len), stride=int(CONFIG.data.stride), as_torch=True)
val_dataset   = pipeline.build_sequence_dataset(split="val",   seq_len=int(CONFIG.data.seq_len), stride=int(CONFIG.data.stride), as_torch=True)
test_dataset  = pipeline.build_sequence_dataset(split="test",  seq_len=int(CONFIG.data.seq_len), stride=int(CONFIG.data.stride), as_torch=True)
# Conditioning fields datasets (aligned splits, same seq/stride)
train_cond = pipeline_cond.build_sequence_dataset(split="train", seq_len=int(CONFIG.data.seq_len), stride=int(CONFIG.data.stride), as_torch=True)

RCN_DRIVER_DIM = None  # set from first sample in Cell 4

# --- 9-node metapath routing helpers (ported from 9-node notebook) ---------
def _resolve_channel_idx(varlist, names):
    return [varlist.index(n) for n in names if n in varlist]
_Q_IDX = _resolve_channel_idx(STAGE1_LR_VARS, ["Q850"])
_W_IDX = _resolve_channel_idx(STAGE1_LR_VARS, ["W500"])
def _compute_ivt_nodes(lr0):
    # IVT proxy from Q + winds present in the 15 base vars (as in 9-node nb).
    return lr0  # routed to IVT node; encoder metapath handles projection

def convert_sample_to_batch(sample, builder, device, sample_cond=None):
    \"\"\"9-node convert + V6' : attach raw LR conditioning grid.\"\"\"
    lr_seq = sample["lr"]                     # [seq, 15, lat, lon]  (Stage 1)
    seq_len = lr_seq.shape[0]
    lr_nodes = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes, dim=0)
    lr0 = lr_nodes[0]
    if EXTENDED_9NODE:
        _ivt = _compute_ivt_nodes(lr0)
        dyn = {}
        for nt in builder.dynamic_node_types:
            if nt == "Q850":   dyn[nt] = lr0[:, _Q_IDX] if _Q_IDX else lr0
            elif nt == "W500": dyn[nt] = lr0[:, _W_IDX] if _W_IDX else lr0
            elif nt == "IVT":  dyn[nt] = _ivt
            else:              dyn[nt] = lr0
    else:
        dyn = {nt: lr0 for nt in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dyn).to(device)
    out = {
        "lr": lr_tensor,
        "residual": sample["residual"],
        "baseline": sample.get("baseline"),
        "hetero": hetero,
        "time": sample.get("time"),
    }
    # V6' : raw LR conditioning grid [seq, 21/22, lat, lon] (native res).
    # M1 : already z-scored by pipeline_cond (train-frozen stats).
    if sample_cond is not None:
        out["lr_grid"] = sample_cond["lr"]     # [seq, C_cond, H_LR, W_LR]
    return out

def iterate_batches_v6(ds_main, ds_cond, builder, device):
    \"\"\"Zip Stage-1 samples with the aligned conditioning-field samples.\"\"\"
    it_cond = iter(ds_cond)
    for s in ds_main:
        sc = next(it_cond)
        yield [convert_sample_to_batch(s, builder, device, sample_cond=sc)]

print("[Cell 3] datasets + convert_sample_to_batch (V6' lr_grid) ready")
"""


CELL_4 = """# >>> Cell 4 : Stack constructor (encoder + RCN + regression_head + diffusion V6')
from st_cdgm.models.intelligible_encoder import IntelligibleVariableEncoder, IntelligibleVariableConfig
from st_cdgm.models.causal_rcn import RCNCell, RCNSequenceRunner
from st_cdgm.models.regression_head import GraphToGridDecoder
from st_cdgm.models import CausalDiffusionDecoder
from st_cdgm.models.edm_preconditioner import EDMConfig

def build_fresh_stack(seed: int):
    torch.manual_seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    allowed = set(builder.dynamic_node_types) | set(builder.static_node_types)
    enc_cfgs = [IntelligibleVariableConfig(name=m.name, meta_path=(m.src, m.relation, m.target),
                                            pool=m.get("pool", "mean"))
                for m in CONFIG.encoder.metapaths if m.src in allowed and m.target in allowed]
    encoder = IntelligibleVariableEncoder(configs=enc_cfgs,
        hidden_dim=int(CONFIG.encoder.hidden_dim),
        conditioning_dim=int(CONFIG.encoder.conditioning_dim)).to(DEVICE)
    num_vars = len(enc_cfgs)

    global RCN_DRIVER_DIM
    _probe = builder.lr_grid_to_nodes(torch.zeros(tuple(CONFIG.graph.lr_shape)))
    RCN_DRIVER_DIM = _probe.shape[-1]
    rcn_cell = RCNCell(num_vars=num_vars, hidden_dim=int(CONFIG.rcn.hidden_dim),
        driver_dim=RCN_DRIVER_DIM, reconstruction_dim=RCN_DRIVER_DIM,
        dropout=float(CONFIG.rcn.dropout)).to(DEVICE)
    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get("detach_interval"))

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
        lr_conditioning_channels=LR_COND_CHANNELS,   # <<< V6' PIVOT (0 → 21/22)
    ).to(DEVICE)
    print(f"   diffusion conv_in in_channels = {diffusion.unet.conv_in.in_channels} "
          f"(= 3 + {LR_COND_CHANNELS} LR)")
    return dict(encoder=encoder, rcn_cell=rcn_cell, rcn_runner=rcn_runner,
                regression_head=regression_head, diffusion=diffusion, num_vars=num_vars)

print("[Cell 4] build_fresh_stack (V6' lr_conditioning_channels) defined")
"""


CELL_5 = """# >>> Cell 5 : Helpers (REAL imports — P0 fix) + G_phys 9-node + schedule
import copy, subprocess as _sp
from st_cdgm.training.training_loop import train_epoch_stage1
from st_cdgm.training.two_stage import (
    freeze_stage1, causal_ablation_check,
    precompute_stage1_outputs, train_epoch_stage2_cached,
)
from st_cdgm.training.physics_prior import (
    build_physical_mask, VAR_LABELS_9NODE, EXPECTED_EDGES_9NODE,
)
from path_c_plus.scripts.option_c_helpers import PATHCPLUS_HYPERPARAM_OVERRIDES
# P0 fix (audit IA) : schedule_lambdas lives in scripts.finetune_stage1_bundle_b
from scripts.finetune_stage1_bundle_b import schedule_lambdas, DEFAULT_HYPERPARAMS

HP = copy.deepcopy(DEFAULT_HYPERPARAMS)
HP.update({
    "lambda_dag_prior": PATHCPLUS_HYPERPARAM_OVERRIDES["lambda_dag_prior"],
    "lambda_l1_start":  PATHCPLUS_HYPERPARAM_OVERRIDES["lambda_l1_start"],
    "lambda_l1_end":    PATHCPLUS_HYPERPARAM_OVERRIDES["lambda_l1_end"],
    "g_phys_alpha":     PATHCPLUS_HYPERPARAM_OVERRIDES["g_phys_alpha"],
    "dag_gate_warmup_start_epoch": PATHCPLUS_HYPERPARAM_OVERRIDES["dag_gate_warmup_start_epoch"],
    "dag_gate_warmup_end_epoch":   PATHCPLUS_HYPERPARAM_OVERRIDES["dag_gate_warmup_end_epoch"],
})

# G_phys 9-node — IDENTICAL to seed 42 (V6'.0 keeps Stage 1 unchanged)
G_phys = build_physical_mask(num_vars=9, var_labels=VAR_LABELS_9NODE,
                             expected_edges=EXPECTED_EDGES_9NODE).to(DEVICE)
print(f"[Cell 5] G_phys 9-node : |edges|={int((G_phys != 0).sum())}")

PRE_REG_COMMIT = _sp.check_output(["git","-C",REPO_DIR,"rev-parse","--short","HEAD"]).decode().strip()
print(f"[Cell 5] pre-registration commit = {PRE_REG_COMMIT}")
print(f"[Cell 5] pre-registered thresholds : path_c_plus/audit/V6_PRIME_seuils_preregistered.json")
"""


CELL_6 = """# >>> Cell 6 : Stage 1 (9-node seed 42 IDENTICAL) — or reuse existing checkpoint
#     V6'.0 : Stage 1 is bit-identical to seed 42, so you can REUSE the trained
#     checkpoint if available (skip ~15h). Set REUSE_STAGE1_CKPT accordingly.
import torch.nn.functional as F

SEED = 42
S1_EPOCHS = 3 if SMOKE_MODE else 75
REUSE_STAGE1_CKPT = f"{DRIVE_ROOT}/oracle_9node/seed_42/epoch_last.pth"

stack = build_fresh_stack(SEED)
encoder, rcn_cell = stack["encoder"], stack["rcn_cell"]
rcn_runner, regression_head = stack["rcn_runner"], stack["regression_head"]
diffusion = stack["diffusion"]

_reused = False
if Path(REUSE_STAGE1_CKPT).exists():
    print(f"[Cell 6] Reusing Stage 1 checkpoint (V6'.0 identical Stage 1) : {REUSE_STAGE1_CKPT}")
    _ck = torch.load(REUSE_STAGE1_CKPT, map_location=DEVICE, weights_only=False)
    from st_cdgm.training.two_stage import _persist_load_state_dict  # noqa
    try:
        _persist_load_state_dict(encoder, _ck.get("encoder_state_dict"))
        _persist_load_state_dict(rcn_cell, _ck.get("rcn_cell_state_dict"))
        _persist_load_state_dict(regression_head, _ck.get("regression_head_state_dict"))
        _reused = True
        print("[Cell 6] Stage 1 weights loaded — SKIP Stage 1 retrain")
    except Exception as e:
        print(f"[Cell 6] reuse failed ({e}) — will train Stage 1 from scratch")

if not _reused:
    opt_s1 = torch.optim.AdamW(
        list(encoder.parameters()) + list(rcn_cell.parameters()) + list(regression_head.parameters()),
        lr=float(CONFIG.training.learning_rate), weight_decay=1e-4)
    ts = CONFIG.two_stage.stage1
    for ep in range(S1_EPOCHS):
        sch = schedule_lambdas(ep, S1_EPOCHS, HP)
        m = train_epoch_stage1(
            encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
            optimizer=opt_s1, data_loader=iterate_batches_v6(train_dataset, train_cond, builder, DEVICE),
            device=DEVICE, epoch_idx=ep,
            lambda_reg=float(ts.lambda_reg), beta_rec=float(ts.beta_rec),
            gamma_dag_max=float(ts.gamma_dag_max), gamma_dag_warmup_epochs=int(ts.gamma_dag_warmup_epochs),
            lambda_l1=float(sch["lambda_l1"]), lambda_dag_prior=float(sch["lambda_dag_prior"]),
            dag_prior=G_phys, dag_grad_gate_value=float(sch["dag_grad_gate"]),
            abort_on_collapse=True, collapse_threshold=0.05,
            dag_floor_projection=True, dag_floor_min_norm=0.10,
            gradient_clipping=CONFIG.training.gradient_clipping)
        print(f"  S1 ep {ep+1}/{S1_EPOCHS} loss={m['loss_total']:.4f} A_dag_norm={float(rcn_cell.A_dag.norm()):.3f}")
print("[Cell 6] Stage 1 ready")
"""


CELL_7 = """# >>> Cell 7 : O3 gate + freeze + A_dag freeze verification
ablation = causal_ablation_check(
    encoder=encoder, rcn_runner=rcn_runner, rcn_cell=rcn_cell,
    regression_head=regression_head, data_loader=val_dataset,
    iterate_batches_fn=lambda ds: iterate_batches_v6(ds, train_cond, builder, DEVICE),
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


CELL_8 = """# >>> Cell 8 : BS32b cache WITH lr_fields (V6' full-LR conditioning)
cache = precompute_stage1_outputs(
    encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
    train_dataset=zip(train_dataset, train_cond),   # (main, cond) pairs
    iterate_batches_fn=lambda pair: convert_sample_to_batch(
        pair[0], builder, DEVICE, sample_cond=pair[1]),
    device=DEVICE, dag_variants=["normal"],
    cache_lr_fields=True,     # <<< V6' : cache the raw LR conditioning grid
)
assert "lr_fields" in cache, "V6' cache MUST contain lr_fields"
print(f"[Cell 8] cache keys = {sorted(cache.keys())}")
print(f"[Cell 8] lr_fields cached shape = {tuple(cache['lr_fields'].shape)} "
      f"(native res, upsampled at batch-time)")

from torch.utils.data import DataLoader as _DL, Dataset as _DS
class _V6CondDataset(_DS):
    def __init__(self, c):
        self.mu=c["mu_HR"]; self.base=c["baseline_log"]
        self.delta=c["delta_target"]; self.mask=c["valid_mask"]; self.lr=c["lr_fields"]
    def __len__(self): return self.mu.shape[0]
    def __getitem__(self, i):
        return {"mu_HR": self.mu[i], "baseline_log": self.base[i],
                "delta_target": self.delta[i], "valid_mask": self.mask[i],
                "lr_fields": self.lr[i]}
BATCH = 32 if SMOKE_MODE else 128    # >=128 not required here (no log-det) but keep CorrDiff-scale
cached_loader = _DL(_V6CondDataset(cache), batch_size=BATCH, shuffle=True, num_workers=0, drop_last=True)
print(f"[Cell 8] cached_loader ready (batch={BATCH}, {len(cached_loader)} batches)")
"""


CELL_9 = """# >>> Cell 9 : SMOKE Stage 2 (A2 monitored — M6) BEFORE full run
#     Verify (a) the LR conditioning trains, (b) A2 (LR->0) degrades the loss
#     (denoiser actually USES the LR channels). If A2 shows ~0 effect at smoke,
#     the pivot isn't landing — stop and diagnose.
import copy
diffusion_smoke = copy.deepcopy(diffusion)
opt_smoke = torch.optim.AdamW(diffusion_smoke.parameters(), lr=float(CONFIG.two_stage.stage2.lr), weight_decay=1e-4)
SMOKE_EP = 3
for ep in range(SMOKE_EP):
    m = train_epoch_stage2_cached(
        diffusion_decoder=diffusion_smoke, optimizer=opt_smoke,
        cached_dataloader=cached_loader, device=DEVICE, use_amp=True,
        gradient_clipping=1.0, log_every=50)
    print(f"SMOKE ep{ep+1} loss_diff={m['loss_diff']:.4f}")

# A2 monitor : compare val loss with lr_fields present vs zeroed (post-norm 0)
diffusion_smoke.eval()
def _val_loss(zero_lr=False):
    tot, n = 0.0, 0
    with torch.no_grad():
        for b in cached_loader:
            mu=b["mu_HR"].to(DEVICE); bl=b["baseline_log"].to(DEVICE)
            dt=b["delta_target"].to(DEVICE); lr=b["lr_fields"].to(DEVICE)
            if lr.shape[-2:] != dt.shape[-2:]:
                lr = F.interpolate(lr, size=dt.shape[-2:], mode="bilinear", align_corners=False)
            if zero_lr: lr = torch.zeros_like(lr)   # A2 : post-norm zero
            l = diffusion_smoke.compute_loss_edm(target=dt, mu_HR=mu, baseline_log=bl, lr_fields=lr)
            tot += float(l); n += 1
            if n >= 5: break
    return tot / max(1, n)
l_full, l_zero = _val_loss(False), _val_loss(True)
print(f"[SMOKE A2] loss(LR on)={l_full:.4f}  loss(LR->0)={l_zero:.4f}  "
      f"degradation={100*(l_zero-l_full)/max(1e-6,l_full):+.1f}%")
if l_zero <= l_full * 1.02:
    print("  ⚠ A2 shows <2% effect at smoke — denoiser may ignore LR. Investigate before full run.")
else:
    print("  ✓ denoiser USES the LR conditioning (A2 degradation > 2%)")
"""


CELL_10 = """# >>> Cell 10 : Stage 2 FULL (full-LR conditioning) + EMA
import copy
S2_EPOCHS = 5 if SMOKE_MODE else 150
ema = copy.deepcopy(diffusion).eval()
for p in ema.parameters(): p.requires_grad_(False)
opt_s2 = torch.optim.AdamW(diffusion.parameters(), lr=float(CONFIG.two_stage.stage2.lr), weight_decay=1e-4)

for ep in range(S2_EPOCHS):
    m = train_epoch_stage2_cached(
        diffusion_decoder=diffusion, optimizer=opt_s2, cached_dataloader=cached_loader,
        device=DEVICE, use_amp=True, gradient_clipping=1.0, log_every=50,
        ema_model=ema, ema_decay=0.9999)
    if (ep+1) % 10 == 0 or ep == 0:
        print(f"[S2 ep{ep+1}/{S2_EPOCHS}] loss_diff={m['loss_diff']:.5f}")

# Persist
import os
save_dir = f"{DRIVE_ROOT}/oracle_v6_prime/seed_42"; os.makedirs(save_dir, exist_ok=True)
torch.save({"diffusion_state_dict": diffusion.state_dict(),
            "ema_state_dict": ema.state_dict(),
            "lr_conditioning_channels": LR_COND_CHANNELS,
            "stage2_cond_lr_vars": STAGE2_COND_LR_VARS},
           f"{save_dir}/stage2_v6_prime.pth")
print(f"[Cell 10] Stage 2 saved → {save_dir}/stage2_v6_prime.pth")
"""


CELL_11 = """# >>> Cell 11 : Eval + ablations A1 (mu_HR->0) / A2 (LR->0) — pre-registered
#     Uses ema weights. Sampling MUST pass lr_fields (M3 parity — sample() will
#     raise otherwise). This cell computes BOTH co-primaries (per-gridpoint +
#     pooled) and the two ablations. Metric helpers reused from the 3-way eval.
from st_cdgm.evaluation.v6_extras import crps_multiscale
import json

K_SAMPLES = 8 if SMOKE_MODE else 64
NUM_STEPS = 32

# Materialise a small set of test conditioning tensors (mu_HR, baseline, lr_fields)
test_cache = precompute_stage1_outputs(
    encoder=encoder, rcn_runner=rcn_runner, regression_head=regression_head,
    train_dataset=zip(test_dataset, pipeline_cond.build_sequence_dataset(
        split="test", seq_len=int(CONFIG.data.seq_len), stride=int(CONFIG.data.stride), as_torch=True)),
    iterate_batches_fn=lambda pair: convert_sample_to_batch(pair[0], builder, DEVICE, sample_cond=pair[1]),
    device=DEVICE, dag_variants=["normal"], cache_lr_fields=True)

def sample_v6(mu, bl, lr, zero_mu=False, zero_lr=False, K=K_SAMPLES):
    \"\"\"Ensemble sampler with A1/A2 ablation switches (M3 : lr_fields required).\"\"\"
    mu_ = torch.zeros_like(mu) if zero_mu else mu
    lr_ = torch.zeros_like(lr) if zero_lr else lr   # A2 : post-norm zero
    outs = []
    for k in range(K):
        torch.manual_seed(1000 + k)
        o = ema.sample(conditioning=None, num_steps=NUM_STEPS, scheduler_type="edm_karras",
                       cfg_scale=1.0, mu_HR=mu_, baseline_log=bl, lr_fields=lr_)
        outs.append(o.residual)
    return torch.stack(outs, 0)   # [K, B, 1, H, W]

# NOTE : full F1@p99 per-gridpoint + pooled computation reuses the metric code
# from _eval_3way_dual_convention.ipynb (Convention A + B). Here we wire the
# sampler + ablations ; import the metric functions from that eval module or
# paste the two convention cells. Placeholder verdict assembly below.
print("[Cell 11] sample_v6 wired (A1/A2 switches). "
      "Plug the Conv-A (per-gridpoint) + Conv-B (pooled) metric cells from "
      "_eval_3way_dual_convention.ipynb to compute F1@p99, then fill results below.")

# results = { 'F1_p99_pergrid': ..., 'F1_p99_pooled': ..., 'A1_mu_off': ..., 'A2_lr_off': ... }
"""


CELL_12 = """# >>> Cell 12 : Verdict vs pre-registered thresholds + OOD EC-Earth3
#     Fill `results` from Cell 11, then this cell renders PASS/FAIL against the
#     frozen thresholds and writes the final JSON.
seuils = json.load(open("path_c_plus/audit/V6_PRIME_seuils_preregistered.json"))
thr_pooled = seuils["targets_to_beat"]["co_primary_2_pooled"]
thr_pg = seuils["targets_to_beat"]["co_primary_1_per_gridpoint"]

def verdict(results):
    pg = results.get("F1_p99_pergrid", 0.0)
    pooled = results.get("F1_p99_pooled", 0.0)
    v_pg = "PASS" if (pg >= thr_pg["noncausal_v4"] and pg >= thr_pg["v5_causal_seed42"]) else "FAIL"
    v_pooled = ("PASS_STRONG" if pooled >= thr_pooled["PASS_STRONG"]
                else "PASS_TARGET" if pooled >= thr_pooled["PASS_TARGET"]
                else "PASS_MINIMAL" if pooled >= thr_pooled["PASS_MINIMAL"] else "FAIL")
    a1 = results.get("A1_mu_off"); a2 = results.get("A2_lr_off")
    return {"per_gridpoint": v_pg, "pooled": v_pooled,
            "A1_mu_HR_contribution": a1, "A2_lr_contribution": a2,
            "at_least_one_coprimary": (v_pg == "PASS" or v_pooled != "FAIL")}

# Example (fill from Cell 11):
# print(json.dumps(verdict(results), indent=2))
print("[Cell 12] verdict() ready. Run after Cell 11 results are computed.")
print("[Cell 12] OOD EC-Earth3 : re-run Cells 3/8/11 with LR_PATH_V6 -> EC-Earth3 v6 NetCDF.")
"""


def build():
    cells = [md(CELL_0), code(CELL_1), code(CELL_2), code(CELL_3), code(CELL_4),
             code(CELL_5), code(CELL_6), code(CELL_7), code(CELL_8), code(CELL_9),
             code(CELL_10), code(CELL_11), code(CELL_12)]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "v6_prime_metadata": {
                "experiment_id": "V6_PRIME_seed42_fullLR_conditioning",
                "plan": "path_c_plus/audit/PLAN_V6_PRIME.md",
                "pre_registration": "path_c_plus/audit/V6_PRIME_seuils_preregistered.json",
                "structural_change": "Stage 2 conditioning 3 -> 3+21/22 channels (full-LR)",
                "design_note": "V6'.0 keeps Stage 1 9-node identical (reusable ckpt); "
                               "U850/V850 graph nodes deferred to V6'.1 (already in LR conditioning)",
                "audit_median_p_beat_noncausal": 0.75,
            },
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


def main():
    nb = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"V6' notebook written : {OUT} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
