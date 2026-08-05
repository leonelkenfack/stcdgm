"""Build a dedicated notebook : Phase 6 re-evaluation on existing checkpoints.

Produces st_cdgm_phase6_finetuned_rerun.ipynb which :
- Loads CONFIG, builds the val_dataset (in-distribution ACCESS-CM2)
- Charges the two Oracle stacks : baseline V5-mini (V5_DIR) and fine-tuned (ORACLE_FINETUNED_DIR)
- Re-runs Phase 6 eval with the EXACT Cell 61 protocol on both, producing apples-to-apples metrics
- Loads CorrDiff baseline JSON (already produced with the same protocol)
- Displays a 3-way comparison table : baseline V5-mini vs FT V5-mini vs CorrDiff

The notebook never modifies V5_DIR or NONCAUSAL_DIR. All outputs go to
results/phase6_rerun/.
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def code_cell(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]],
    }


def md_cell(src):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]],
    }


# Read source cells from v5_evaluation to vendor the bootstrap verbatim
nb_src = json.load(open("st_cdgm_v5_evaluation.ipynb", encoding="utf-8"))
SRC_BOOTSTRAP = "".join(nb_src["cells"][1]["source"])


CELL0_MD = """# Re-evaluation Phase 6 — version fine-tunee (Oracle FT)

Notebook dedie qui :
1. Charge les checkpoints existants (baseline V5-mini, Oracle fine-tune, CorrDiff baseline)
2. Recompute Phase 6 sur baseline et FT avec le **MEME protocole** (Cell 61 du training noncausal)
   pour comparaison apples-to-apples
3. Lit la baseline CorrDiff existante (deja produite par le meme protocole Cell 61)
4. Affiche le tableau comparatif **baseline V5-mini vs FT V5-mini vs CorrDiff**

Protocole d'evaluation : K=64 samples, n_steps=32, scheduler edm_karras, cfg_scale=1.5,
apply_constraints=False (= `final_validation_metrics.json` du training original).

Aucune modification du notebook requise : il est dedie a cet usage.
"""


CELL1_CODE = SRC_BOOTSTRAP


CELL2_CODE = '''# === Paths config — dedie Phase 6 re-evaluation ===
from pathlib import Path

# Oracle baseline V5-mini (NE PAS ECRIRE — lecture seule)
V5_DIR = Path("/content/drive/MyDrive/climate_data/ckpt_v2_corrdiff_normal")

# Oracle fine-tune (lit/ecrit ici)
ORACLE_FINETUNED_DIR = Path("/content/drive/MyDrive/climate_data/oracle_finetuned")

# CorrDiff baseline (lecture seule)
NONCAUSAL_DIR = Path("/content/drive/MyDrive/climate_data/ckpt_noncausal")

# Sortie : nouveau dossier pour ne JAMAIS toucher aux baselines
RERUN_DIR = Path("/content/drive/MyDrive/climate_data/results/phase6_rerun")
RERUN_DIR.mkdir(parents=True, exist_ok=True)

# === Verification existence des checkpoints ===
print("=" * 70)
print("Inventaire des checkpoints")
print("=" * 70)

checks = [
    ("Oracle baseline V5-mini", V5_DIR / "epoch_last.pth"),
    ("Oracle fine-tuned",       ORACLE_FINETUNED_DIR / "epoch_finetuned.pth"),
    ("CorrDiff baseline",       NONCAUSAL_DIR / "epoch_last.pth"),
]
all_ok = True
for label, p in checks:
    if p.exists():
        size_gb = p.stat().st_size / 1e9
        print(f"  [OK]    {label:30s} {p.name:30s} ({size_gb:.2f} GB)")
    else:
        all_ok = False
        print(f"  [MANQUE] {label:30s} {p}")

if not all_ok:
    print()
    print("[ERREUR] Au moins un checkpoint manque. Verifie les paths ou lance d abord Phase F.")

# JSON CorrDiff baseline (deja produit par le training Cell 61 protocol)
nc_baseline_json = NONCAUSAL_DIR / "final_validation_metrics.json"
if nc_baseline_json.exists():
    print(f"\\n  [OK] CorrDiff baseline JSON : {nc_baseline_json.name}")
else:
    print(f"\\n  [WARN] CorrDiff baseline JSON absent — comparaison incomplete")

print()
print(f"Sortie : {RERUN_DIR}")
'''


CELL3_CODE = '''# === Bootstrap CONFIG + dataset + builder (in-distribution ACCESS-CM2 uniquement) ===
import os, sys, time, json, shutil
import torch
import numpy as np
from omegaconf import OmegaConf

ON_COLAB = "google.colab" in sys.modules or Path("/content").exists()

# 1. CONFIG (base + override corrdiff_normal si dispo)
_base = Path("config/training_config.yaml")
_override = Path("config/training_config_corrdiff_normal.yaml")
CONFIG = OmegaConf.load(_base)
if _override.exists():
    CONFIG = OmegaConf.merge(CONFIG, OmegaConf.load(_override))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr_shape = tuple(CONFIG.graph.lr_shape)
hr_shape = tuple(CONFIG.graph.hr_shape)
print(f"[OK] DEVICE={DEVICE}  lr_shape={lr_shape}  hr_shape={hr_shape}")

# 2. DATA_ROOT detection
DATA_ROOT_LOCAL = Path("data/raw")
DATA_ROOT_DRIVE = Path("/content/drive/MyDrive/climate_data/data")
_DATA_ROOT_LOCAL_SSD = Path("/content/data_local")

if ON_COLAB and DATA_ROOT_DRIVE.parent.parent.exists():
    DATA_ROOT = DATA_ROOT_DRIVE
else:
    DATA_ROOT = DATA_ROOT_LOCAL
DATA_ROOT.mkdir(parents=True, exist_ok=True)

# 3. SSD copy (Drive -> /content/data_local pour I/O rapide)
if ON_COLAB and DATA_ROOT == DATA_ROOT_DRIVE:
    _files_to_copy = [
        ("train/predictor_ACCESS-CM2_hist.nc",   "predictor_ACCESS-CM2_hist.nc"),
        ("train/pr_ACCESS-CM2_hist.nc",          "pr_ACCESS-CM2_hist.nc"),
        ("static_predictors/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc",
         "ERA5_eval_ccam_12km.198110_NZ_Invariant.nc"),
        ("normalization_coefs/mean_1974_2011.nc", "mean_1974_2011.nc"),
        ("normalization_coefs/std_1974_2011.nc",  "std_1974_2011.nc"),
    ]
    _ssd_train = _DATA_ROOT_LOCAL_SSD / "train"
    _ssd_static = _DATA_ROOT_LOCAL_SSD / "static_predictors"
    _ssd_norm = _DATA_ROOT_LOCAL_SSD / "normalization_coefs"
    for _d in (_ssd_train, _ssd_static, _ssd_norm):
        _d.mkdir(parents=True, exist_ok=True)
    for _rel, _name in _files_to_copy:
        _src = DATA_ROOT_DRIVE / _rel
        if "train/" in _rel:
            _dst = _ssd_train / _name
        elif "static_predictors/" in _rel:
            _dst = _ssd_static / _name
        else:
            _dst = _ssd_norm / _name
        if not _src.exists():
            continue
        if _dst.exists() and _dst.stat().st_size == _src.stat().st_size:
            continue
        shutil.copy2(_src, _dst)
    DATA_ROOT = _DATA_ROOT_LOCAL_SSD
    print(f"[INFO] DATA_ROOT redirige vers SSD : {_DATA_ROOT_LOCAL_SSD}")

def _relocate(p):
    if not p:
        return p
    s = str(p)
    if s.startswith("data/raw/"):
        return str(DATA_ROOT / s[len("data/raw/"):])
    return s

for _key in ("lr_path", "hr_path", "static_path"):
    if CONFIG.data.get(_key):
        CONFIG.data[_key] = _relocate(CONFIG.data[_key])

LR_PATH = str(CONFIG.data.lr_path)
HR_PATH = str(CONFIG.data.hr_path)
STATIC_PATH = str(CONFIG.data.static_path) if CONFIG.data.get("static_path") else None
MEAN_PATH = str(DATA_ROOT / "normalization_coefs" / "mean_1974_2011.nc")
STD_PATH = str(DATA_ROOT / "normalization_coefs" / "std_1974_2011.nc")

# 4. Pipeline + builder
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder

pipeline_access = NetCDFDataPipeline(
    lr_path=LR_PATH, hr_path=HR_PATH,
    static_path=STATIC_PATH if STATIC_PATH and Path(STATIC_PATH).exists() else None,
    seq_len=int(CONFIG.data.seq_len),
    baseline_strategy=str(CONFIG.data.baseline_strategy),
    baseline_factor=int(CONFIG.data.baseline_factor),
    target_transform=str(CONFIG.data.get("target_transform", "log1p")),
    normalize=bool(CONFIG.data.normalize),
    nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
    precipitation_delta=float(CONFIG.data.get("precipitation_delta", 0.01)),
    lr_variables=list(CONFIG.data.lr_variables),
    hr_variables=list(CONFIG.data.hr_variables),
    static_variables=list(CONFIG.data.static_variables) if STATIC_PATH and Path(STATIC_PATH).exists() else None,
    means_path=MEAN_PATH if Path(MEAN_PATH).exists() else None,
    stds_path=STD_PATH if Path(STD_PATH).exists() else None,
    eager_load_datasets=bool(CONFIG.data.get("eager_load_datasets", False)),
)

builder = HeteroGraphBuilder(
    lr_shape=lr_shape, hr_shape=hr_shape,
    static_dataset=pipeline_access.get_static_dataset(),
    include_mid_layer=bool(CONFIG.graph.include_mid_layer),
)
print(f"[OK] Builder cree ({len(builder.dynamic_node_types)} dyn + {len(builder.static_node_types)} static)")

def convert_sample_to_batch(sample, builder, device):
    """Conversion sample -> batch dict, identique a v5_evaluation."""
    lr_seq = sample["lr"]
    seq_len = lr_seq.shape[0]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    dynamic_features = {nt: lr_nodes_steps[0] for nt in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_features).to(device)
    return {"lr": lr_tensor, "residual": sample["residual"],
            "baseline": sample.get("baseline"), "hetero": hetero}

# 5. Materialize val_dataset
print()
print("[Dataset] Construction du val_dataset (in-distribution ACCESS-CM2)...")
test_dataset_iter = pipeline_access.build_sequence_dataset(
    seq_len=int(CONFIG.data.seq_len),
    stride=int(CONFIG.data.stride),
    as_torch=True,
)
sample0 = next(iter(test_dataset_iter))
_runtime_dim = int(sample0["lr"].shape[1])
if _runtime_dim != int(CONFIG.rcn.driver_dim):
    CONFIG.rcn.driver_dim = _runtime_dim
    CONFIG.rcn.reconstruction_dim = _runtime_dim
    print(f"[INFO] CONFIG.rcn.driver_dim -> {_runtime_dim}")

# 16 batches Phase 6 + petit buffer
N_VAL_SAMPLES = 24
import itertools as _it
_val_samples = [sample0] + list(_it.islice(pipeline_access.build_sequence_dataset(
    seq_len=int(CONFIG.data.seq_len),
    stride=int(CONFIG.data.stride),
    as_torch=True,
), N_VAL_SAMPLES - 1))

from torch.utils.data import Dataset as _TorchDataset
class _MapStyleListDataset(_TorchDataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]

val_dataset = _MapStyleListDataset(_val_samples)
print(f"[OK] val_dataset materialise : {len(val_dataset)} samples")
print(f"     residual shape : {tuple(sample0['residual'].shape)}")
'''


CELL4_CODE = '''# === Helper build_stack (= v5_evaluation Cell 4) + chargement 2 stacks Oracle ===
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
    """Charge un stack Oracle depuis un checkpoint. Identique a v5_evaluation Cell 4."""
    print(f"  [{name}] {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    allowed_nodes = set(builder.dynamic_node_types + builder.static_node_types)
    encoder_configs = []
    for _mp in CONFIG.encoder.metapaths:
        _src, _rel, _tgt = _mp.src, _mp.relation, _mp.target
        if _src in allowed_nodes and _tgt in allowed_nodes:
            encoder_configs.append(IntelligibleVariableConfig(
                name=_mp.name,
                meta_path=(_src, _rel, _tgt),
                pool=_mp.get("pool", "mean"),
            ))
    if pipeline_access.get_static_dataset() is not None:
        encoder_configs.append(IntelligibleVariableConfig(
            name="static", meta_path=("SP_HR", "causes", "GP850"), pool="mean",
        ))

    enc = IntelligibleVariableEncoder(
        configs=encoder_configs,
        hidden_dim=CONFIG.encoder.hidden_dim,
        conditioning_dim=CONFIG.encoder.conditioning_dim,
    ).to(DEVICE)
    num_vars = len(encoder_configs)

    rcn_cell = RCNCell(
        num_vars=num_vars,
        hidden_dim=CONFIG.rcn.hidden_dim,
        driver_dim=int(CONFIG.rcn.driver_dim),
        reconstruction_dim=int(CONFIG.rcn.reconstruction_dim),
        dropout=CONFIG.rcn.dropout,
    ).to(DEVICE)
    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get("detach_interval"))

    rh = GraphToGridDecoder(
        d_model=CONFIG.encoder.hidden_dim,
        hr_h=CONFIG.graph.hr_shape[0], hr_w=CONFIG.graph.hr_shape[1],
    ).to(DEVICE)

    edm_cfg = EDMConfig.from_yaml_dict(CONFIG.diffusion.get("edm", {}))
    _unet_kwargs = OmegaConf.to_container(CONFIG.diffusion.unet_kwargs, resolve=True)
    for _k in ("down_block_types", "up_block_types"):
        if _k in _unet_kwargs and isinstance(_unet_kwargs[_k], list):
            _unet_kwargs[_k] = tuple(_unet_kwargs[_k])

    hr_channels = int(sample0["residual"].shape[1])

    diff = CausalDiffusionDecoder(
        in_channels=hr_channels,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        height=int(CONFIG.diffusion.height),
        width=int(CONFIG.diffusion.width),
        unet_kwargs=_unet_kwargs,
        scheduler_type=str(CONFIG.diffusion.scheduler_type),
        use_gradient_checkpointing=bool(CONFIG.diffusion.get("use_gradient_checkpointing", False)),
        conv_padding_mode=str(CONFIG.diffusion.get("conv_padding_mode", "zeros")),
        anti_checkerboard=bool(CONFIG.diffusion.get("anti_checkerboard", False)),
        edm_config=edm_cfg,
        causal_concat=True,
    ).to(DEVICE)

    def _safe_load(name_, module):
        key = f"{name_}_state_dict"
        if key not in ckpt:
            print(f"  [{name_}] [WARN] {key} absent")
            return False
        sd = ckpt[key]
        if sd is None or not hasattr(sd, "items"):
            print(f"  [{name_}] [WARN] {key} invalid")
            return False
        stripped = {}
        for kk, vv in sd.items():
            new_k = kk
            for p in ("_orig_mod.", "module."):
                if new_k.startswith(p):
                    new_k = new_k[len(p):]
            stripped[new_k] = vv
        try:
            module.load_state_dict(stripped, strict=False)
            return True
        except Exception as e:
            print(f"  [{name_}] [ERREUR] {type(e).__name__}: {e}")
            return False

    for n, m in [("encoder", enc), ("rcn_cell", rcn_cell),
                  ("regression_head", rh), ("diffusion", diff)]:
        _safe_load(n, m)

    skip = None
    if (SKIP_AVAILABLE and "skip_block_state_dict" in ckpt
            and ckpt["skip_block_state_dict"] is not None):
        skip = ConditionalSkipBlock(
            lr_channels=len(CONFIG.data.lr_variables),
            hr_shape=tuple(CONFIG.graph.hr_shape),
        ).to(DEVICE)
        try:
            skip.load_state_dict(ckpt["skip_block_state_dict"], strict=False)
        except Exception as e:
            print(f"  [{name}] [WARN] skip_block load failed: {e}")
            skip = None
    enc.eval(); rcn_cell.eval(); rh.eval(); diff.eval()
    if skip is not None:
        skip.eval()
    A_dag = rcn_cell.A_dag.detach().cpu().clone() if hasattr(rcn_cell, "A_dag") else None
    return {"encoder": enc, "rcn_runner": rcn_runner, "regression_head": rh,
            "diffusion": diff, "skip_block": skip, "A_dag": A_dag, "variant": name}


print()
print("Chargement des stacks Oracle...")
print()
t0 = time.time()
stack_baseline = build_stack(V5_DIR / "epoch_last.pth", "Oracle-baseline")
print()
stack_ft = build_stack(ORACLE_FINETUNED_DIR / "epoch_finetuned.pth", "Oracle-FT")
print()
print(f"[OK] 2 stacks Oracle charges en {time.time()-t0:.1f}s")
'''


CELL5_CODE = '''# === Re-evaluation Phase 6 sur baseline + FT avec le MEME protocole (Cell 61) ===
from scripts.recompute_phase6_metrics import recompute_phase6_metrics

# Params : EXACTEMENT ceux du training Cell 61 / final_validation_metrics.json
EVAL_PARAMS = dict(
    K_samples=64,        # ensemble size
    n_steps=32,          # diffusion steps
    n_batches=16,        # test batches
    scheduler_type="edm_karras",
    cfg_scale=1.5,
    run_variant="causal",
)

# Cache : si le JSON existe deja dans RERUN_DIR, skip recompute. Force re-eval
# en mettant FORCE_RECOMPUTE = True.
FORCE_RECOMPUTE = False

results_jsons = {}

for label, stack, ckpt_label in [
    ("baseline", stack_baseline, "Oracle-baseline (V5-mini publie)"),
    ("ft",       stack_ft,       "Oracle-FT (post-Phase F)"),
]:
    out_path = RERUN_DIR / f"phase6_{label}_metrics.json"
    if out_path.exists() and not FORCE_RECOMPUTE:
        print(f"\\n[SKIP] {ckpt_label} -> cache deja present : {out_path.name}")
        print(f"       FORCE_RECOMPUTE=True pour re-evaluer")
        results_jsons[label] = json.loads(out_path.read_text(encoding="utf-8"))
        continue

    print()
    print("#" * 76)
    print(f"#  RE-EVAL : {ckpt_label}")
    print("#" * 76)
    t0 = time.time()
    try:
        result = recompute_phase6_metrics(
            stack=stack, builder=builder, val_dataset=val_dataset,
            DEVICE=DEVICE,
            convert_sample_to_batch_fn=convert_sample_to_batch,
            out_path=out_path,
            **EVAL_PARAMS,
            verbose=True,
        )
        results_jsons[label] = result
        print(f"\\n[OK] {ckpt_label} : {time.time()-t0:.0f}s")
    except Exception as e:
        print(f"\\n[ERREUR] {ckpt_label} : {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

# === CorrDiff baseline : load existing (deja produit par meme protocole) ===
nc_path = NONCAUSAL_DIR / "final_validation_metrics.json"
if nc_path.exists():
    results_jsons["corrdiff"] = json.loads(nc_path.read_text(encoding="utf-8"))
    print(f"\\n[OK] CorrDiff baseline charge depuis {nc_path.name}")
    print(f"     (deja produit par le training noncausal avec le meme protocole Cell 61)")
else:
    print(f"\\n[WARN] {nc_path} absent - comparaison sans CorrDiff")

print()
print("=" * 76)
print(f"[OK] Re-evaluation terminee. {len(results_jsons)} JSONs disponibles")
print("=" * 76)
'''


CELL6_CODE = '''# === Tableau comparatif : baseline V5-mini vs FT V5-mini vs CorrDiff baseline ===

def _get(d, *path, default=None):
    """Lecture safe dans dict imbrique."""
    if d is None:
        return default
    v = d
    for k in path:
        if not isinstance(v, dict) or k not in v:
            return default
        v = v[k]
    return v

m_baseline = results_jsons.get("baseline")
m_ft       = results_jsons.get("ft")
m_corrdiff = results_jsons.get("corrdiff")

if m_baseline is None or m_ft is None:
    print("[ERREUR] baseline ou FT manquant - relance Cell 5")
else:
    metrics_def = [
        # (label, json path, direction)
        ("Pearson global",     ("pearson_corr", "global"),         "haut"),
        ("Pearson per-sample", ("pearson_corr", "per_sample_avg"), "haut"),
        ("RMSE",               ("rmse",),                            "bas"),
        ("MAE",                ("mae",),                             "bas"),
        ("Spread (ens std)",   ("spread_mean",),                     "calib"),
        ("F1-p95",             ("f1_extremes", "p95"),               "haut"),
        ("F1-p99",             ("f1_extremes", "p99"),               "haut"),
        ("RAPSD distance",     ("rapsd_distance",),                  "bas"),
        ("mu_HR ablation",     ("mu_HR_ablation", "delta_signal_ratio_avg"), "haut"),
    ]

    rows = []
    for label, path, direction in metrics_def:
        v_base = _get(m_baseline, *path)
        v_ft   = _get(m_ft, *path)
        v_corr = _get(m_corrdiff, *path)
        rows.append((label, v_base, v_ft, v_corr, direction))

    print()
    print("=" * 116)
    print(f"{'Metrique':<22} {'Baseline V5-mini':>18} {'FT V5-mini':>14} {'CorrDiff':>12} {'Delta FT-base':>16} {'sens':>8} {'gagnant':>14}")
    print("-" * 116)

    summary = {}
    for label, v_base, v_ft, v_corr, direction in rows:
        if v_base is None or v_ft is None:
            print(f"{label:<22} {'n/a':>18} {'n/a':>14}")
            continue
        delta = v_ft - v_base
        # Determine winner FT vs baseline V5-mini
        if abs(delta) < (abs(v_base) + 1e-9) * 0.01:
            winner_ft_vs_base = "egal"
        elif direction == "haut" and delta > 0:
            winner_ft_vs_base = "FT mieux"
        elif direction == "bas" and delta < 0:
            winner_ft_vs_base = "FT mieux"
        elif direction == "calib":
            winner_ft_vs_base = "FT mieux" if abs(v_ft) < abs(v_base) else "Base mieux"
        else:
            winner_ft_vs_base = "Base mieux"
        v_corr_str = f"{v_corr:>12.4f}" if v_corr is not None else f"{'n/a':>12}"
        print(f"{label:<22} {v_base:>18.4f} {v_ft:>14.4f} {v_corr_str} {delta:>+16.4f} {direction:>8} {winner_ft_vs_base:>14}")
        summary[label] = {
            "baseline_v5": v_base,
            "ft_v5": v_ft,
            "corrdiff": v_corr,
            "delta_ft_vs_base": delta,
            "direction": direction,
            "winner_ft_vs_base": winner_ft_vs_base,
        }

    out_summary = RERUN_DIR / "phase6_rerun_comparison.json"
    out_summary.write_text(json.dumps({
        "protocol": "Cell 61 training_noncausal (K=64, n_steps=32, edm_karras, cfg_scale=1.5)",
        "n_test_batches": _get(m_ft, "n_test_batches"),
        "eval_time_s_baseline": _get(m_baseline, "eval_time_s"),
        "eval_time_s_ft": _get(m_ft, "eval_time_s"),
        "comparison": summary,
        "raw": {
            "baseline_v5_mini": m_baseline,
            "ft_v5_mini": m_ft,
            "corrdiff_baseline": m_corrdiff,
        },
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print()
    print("=" * 116)
    print(f"[OK] Comparaison sauvegardee : {out_summary}")
    print(f"     JSONs bruts : {RERUN_DIR}/phase6_*.json")
    print("=" * 116)

    # Bilan rapide
    n_ft_wins = sum(1 for s in summary.values() if s["winner_ft_vs_base"] == "FT mieux")
    n_base_wins = sum(1 for s in summary.values() if s["winner_ft_vs_base"] == "Base mieux")
    n_egal = sum(1 for s in summary.values() if s["winner_ft_vs_base"] == "egal")
    print()
    print(f"Bilan FT vs baseline V5-mini : {n_ft_wins} FT mieux | {n_egal} egal | {n_base_wins} Base mieux")

    # Diagnostics FT
    sc_v = _get(m_ft, "shortcut_diagnostic", "verdict")
    sc_r = _get(m_ft, "shortcut_diagnostic", "shortcut_ratio")
    if sc_v and sc_v != "N/A" and sc_r is not None:
        print(f"Shortcut diagnostic FT : {sc_v} (ratio={sc_r:.4f})")

    mu_v = _get(m_ft, "mu_HR_ablation", "verdict")
    mu_a = _get(m_ft, "mu_HR_ablation", "delta_signal_ratio_avg")
    if mu_v and mu_v != "N/A" and mu_a is not None:
        print(f"mu_HR ablation FT      : {mu_v} (delta/signal={mu_a*100:.3f}%)")
'''


nb = {
    "cells": [
        md_cell(CELL0_MD),
        code_cell(CELL1_CODE),
        code_cell(CELL2_CODE),
        code_cell(CELL3_CODE),
        code_cell(CELL4_CODE),
        code_cell(CELL5_CODE),
        code_cell(CELL6_CODE),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("st_cdgm_phase6_finetuned_rerun.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Notebook ecrit : {out} ({out.stat().st_size/1024:.1f} KB, {len(nb['cells'])} cells)")
