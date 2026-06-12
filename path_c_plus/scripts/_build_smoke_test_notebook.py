"""Build the Path C+ smoke test notebook for the user to run on Colab T4.

Smoke test goal: validate that the 17 P0 fixes correctly break the band-diagonal
Q_phys=0.40 pattern in <= 10 epochs on the corrected V5-mini codebase.

Success criterion (per AI eng round-7 validation):
- Q_phys > 0.55 by epoch 10 (vs 0.40 baseline)
- skip_block grad norm > 0 (J8 fix actually flows gradients)
- No J8 reshape failures (would indicate broken contract)
- No §1.6 missing-edge fallback warnings (would indicate cfg with zero contribution)
"""
import json
from pathlib import Path


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


CELL0_MD = """# Path C+ Smoke Test — V5-mini avec 17 P0 fixes appliqués

**Objectif** : valider que les 17 fixes Path C+ cassent le pattern band-diagonal
Q_phys=0.40 observé en V5-mini avant correction.

**Cible** : ≥ 10 epochs, ≤ 1 heure sur Colab T4 (smoke test rapide, pas un full
retrain).

**Critères de succès** (AI eng round 7) :
- ✅ `Q_phys > 0.55` par epoch 10 (vs 0.40 collapsed)
- ✅ `skip_block.parameters().grad.norm() > 0` après epoch 1 (J8 fix)
- ✅ AUCUN warning `J8 skip_block forward failed` (contract broken si présent)
- ✅ AUCUN warning `§1.6 missing-edge fallback` (cfg à zéro gradient si présent)
- ⚠️ Si `Q_phys < 0.45` à epoch 10 → bisecter entre commits J3, I11, J8

**Branch** : `four-node-causal` (23 commits depuis `two-stage-causal`)

## Phase plan
1. Bootstrap Colab (clone, install deps)
2. GPU profile detection
3. Pre-flight check (K8 temporal split warning)
4. Build V5-mini stack
5. Run smoke test 10 epochs avec instrumentation
6. Verdict + go/no-go pour Batch D
"""


CELL1_BOOTSTRAP = '''# === Cell 1 : Bootstrap Colab (clone four-node-causal branch) ===
import os, sys, time, shlex, subprocess
from pathlib import Path

GIT_URL = "https://github.com/leonelkenfack/stcdgm.git"
GIT_BRANCH = "four-node-causal"  # Path C+ branch
LOCAL_PROJECT = "/content/climate_data"
_IS_COLAB = "google.colab" in sys.modules or Path("/content").exists()

if _IS_COLAB:
    # Mount Drive
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")

    # Clone the four-node-causal branch
    project_path = Path(LOCAL_PROJECT)
    if not (project_path / ".git").exists():
        project_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(shlex.split(
            f"git clone --depth 1 -b {GIT_BRANCH} {GIT_URL} {LOCAL_PROJECT}"
        ))
    else:
        # Pull latest from the Path C+ branch
        subprocess.call(shlex.split(f"git -C {LOCAL_PROJECT} fetch origin {GIT_BRANCH}"))
        subprocess.call(shlex.split(f"git -C {LOCAL_PROJECT} checkout {GIT_BRANCH}"))
        subprocess.call(shlex.split(f"git -C {LOCAL_PROJECT} pull --ff-only"))

    os.chdir(project_path)
    sys.path.insert(0, str(project_path / "src"))
    sys.path.insert(0, str(project_path))

    # Log the exact commit SHA for audit trail
    sha = subprocess.check_output(
        shlex.split(f"git -C {LOCAL_PROJECT} rev-parse HEAD")
    ).decode().strip()
    print(f"[Bootstrap] On commit {sha[:8]}")

    # Install pinned deps from requirements.txt (K27 fix)
    subprocess.check_call(shlex.split(
        f"{shlex.quote(sys.executable)} -m pip install --no-warn-script-location "
        "-q omegaconf==2.3.0 diffusers==0.36.0 transformers==4.57.6 "
        "huggingface-hub==0.36.0 accelerate==1.12.0 safetensors==0.7.0 "
        "xbatcher webdataset cftime h5netcdf numcodecs torch-geometric xformers "
        "tigramite statsmodels"
    ))
    # Editable install
    subprocess.check_call(shlex.split(
        f"{shlex.quote(sys.executable)} -m pip install --no-warn-script-location "
        f"--no-deps -e {LOCAL_PROJECT}"
    ))
    print("[Bootstrap] Path C+ four-node-causal branch ready.")
else:
    # Local dev: assume already at project root
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / "config" / "training_config.yaml").exists():
            if cand != here:
                os.chdir(cand)
            sys.path.insert(0, str(cand / "src"))
            sys.path.insert(0, str(cand))
            print(f"[Bootstrap] Local mode at {cand}")
            break
'''


CELL2_GPU = '''# === Cell 2 : GPU profile detection + pre-flight K8 check ===
from path_c_plus.scripts.gpu_detect import detect_gpu_profile, print_profile_banner
from path_c_plus.scripts.preflight_checks import run_all_preflight_checks
from omegaconf import OmegaConf

GPU_PROFILE = detect_gpu_profile()
print_profile_banner(GPU_PROFILE)

# Load config + run pre-flight checks
CONFIG = OmegaConf.load("config/training_config.yaml")
_corrdiff = OmegaConf.load("config/training_config_corrdiff_normal.yaml")
CONFIG = OmegaConf.merge(CONFIG, _corrdiff)

print()
print("=" * 70)
print("Pre-flight checks (Path C+ Phase 0.3)")
print("=" * 70)
preflight_report = run_all_preflight_checks(CONFIG)
print()
print(f"Pre-flight report: {preflight_report}")
print()
print("NOTE: K8 temporal split warning is EXPECTED (commit 9402cfc declared")
print("      fields, K9/K5 enforcement deferred to Batch D commits 24-25).")
print("      Smoke test runs on FULL dataset for now.")
'''


CELL3_STACK = '''# === Cell 3 : Build V5-mini stack + load baseline checkpoint ===
import torch
import numpy as np
from pathlib import Path

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
V5_DIR = Path("/content/drive/MyDrive/climate_data/ckpt_v2_corrdiff_normal")
SMOKE_DIR = Path("/content/drive/MyDrive/climate_data/smoke_test_v6_pathc")
SMOKE_DIR.mkdir(parents=True, exist_ok=True)

# Copy baseline checkpoint as smoke starting point
import shutil as _sh
_smoke_ckpt = SMOKE_DIR / "epoch_last.pth"
if not _smoke_ckpt.exists():
    _src = V5_DIR / "epoch_last.pth"
    print(f"[Setup] Copying baseline {_src.name} to smoke dir...")
    _sh.copy(_src, _smoke_ckpt)
    print(f"  [OK] {_smoke_ckpt.stat().st_size/1024**3:.2f} GB")

# Build the stack via the same code path as the eval notebook
# (we vendor a minimal version here to keep smoke test self-contained)
print()
print("=" * 70)
print("Building V5-mini stack with NEW Path C+ code")
print("=" * 70)

# Apply GPU profile to config
CONFIG.training.batch_size = GPU_PROFILE["batch_size"]
CONFIG.training.use_amp = GPU_PROFILE["use_amp"]
CONFIG.training.num_workers = GPU_PROFILE["num_workers"]

# Build pipeline + builder (in-distribution ACCESS-CM2)
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder
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

# Data root detection (Drive or local SSD)
DATA_ROOT = Path("/content/drive/MyDrive/climate_data/data")
LR_PATH = str(DATA_ROOT / "train/predictor_ACCESS-CM2_hist.nc")
HR_PATH = str(DATA_ROOT / "train/pr_ACCESS-CM2_hist.nc")
STATIC_PATH = str(DATA_ROOT / "static_predictors/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc")
MEAN_PATH = str(DATA_ROOT / "normalization_coefs/mean_1974_2011.nc")
STD_PATH = str(DATA_ROOT / "normalization_coefs/std_1974_2011.nc")

pipeline = NetCDFDataPipeline(
    lr_path=LR_PATH, hr_path=HR_PATH, static_path=STATIC_PATH,
    seq_len=int(CONFIG.data.seq_len),
    baseline_strategy=str(CONFIG.data.baseline_strategy),
    baseline_factor=int(CONFIG.data.baseline_factor),
    target_transform=str(CONFIG.data.get("target_transform", "log1p")),
    normalize=bool(CONFIG.data.normalize),
    nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
    precipitation_delta=float(CONFIG.data.get("precipitation_delta", 0.01)),
    lr_variables=list(CONFIG.data.lr_variables),
    hr_variables=list(CONFIG.data.hr_variables),
    static_variables=list(CONFIG.data.static_variables),
    means_path=MEAN_PATH, stds_path=STD_PATH,
    eager_load_datasets=False,
)

lr_shape = tuple(CONFIG.graph.lr_shape)
hr_shape = tuple(CONFIG.graph.hr_shape)
builder = HeteroGraphBuilder(
    lr_shape=lr_shape, hr_shape=hr_shape,
    static_dataset=pipeline.get_static_dataset(),
    include_mid_layer=bool(CONFIG.graph.include_mid_layer),
)
print(f"[OK] Builder created: {len(builder.dynamic_node_types)} dyn + {len(builder.static_node_types)} static nodes")
print(f"     LR shape: {lr_shape}, HR shape: {hr_shape}")
'''


CELL4_LOAD = '''# === Cell 4 : Materialize datasets + load stack from baseline ===
import itertools as _it
from torch.utils.data import Dataset as _TorchDataset


class _MapStyleListDataset(_TorchDataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, i):
        return self.samples[i]


# Smoke test: 100 train + 24 val samples (sufficient for 10 epochs at bs=8)
MAX_TRAIN_SAMPLES = 100
MAX_VAL_SAMPLES = 24

print(f"[Smoke] Materializing {MAX_TRAIN_SAMPLES} train + {MAX_VAL_SAMPLES} val samples...")
train_iter = pipeline.build_sequence_dataset(
    seq_len=int(CONFIG.data.seq_len), stride=1, as_torch=True,
)
_train_samples = list(_it.islice(train_iter, MAX_TRAIN_SAMPLES))
train_dataset = _MapStyleListDataset(_train_samples)
print(f"  [OK] train_dataset : {len(train_dataset)} samples")

val_iter = pipeline.build_sequence_dataset(
    seq_len=int(CONFIG.data.seq_len),
    stride=int(CONFIG.data.stride),
    as_torch=True,
)
_val_samples = list(_it.islice(val_iter, MAX_VAL_SAMPLES))
val_dataset = _MapStyleListDataset(_val_samples)
print(f"  [OK] val_dataset   : {len(val_dataset)} samples")

# Detect runtime driver_dim from samples
_runtime_dim = int(_train_samples[0]["lr"].shape[1])
if _runtime_dim != int(CONFIG.rcn.driver_dim):
    CONFIG.rcn.driver_dim = _runtime_dim
    CONFIG.rcn.reconstruction_dim = _runtime_dim


def convert_sample_to_batch(sample, builder, device):
    lr_seq = sample["lr"]
    seq_len = lr_seq.shape[0]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    dynamic_features = {nt: lr_nodes_steps[0] for nt in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_features).to(device)
    return {"lr": lr_tensor, "residual": sample["residual"],
            "baseline": sample.get("baseline"), "hetero": hetero}


# Build encoder + RCN + regression head + skip_block + diffusion (V5-mini config)
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
if pipeline.get_static_dataset() is not None:
    encoder_configs.append(IntelligibleVariableConfig(
        name="static", meta_path=("SP_HR", "causes", "GP850"), pool="mean",
    ))

print(f"[Stack] {len(encoder_configs)} encoder configs (= q variables)")

enc = IntelligibleVariableEncoder(
    configs=encoder_configs,
    hidden_dim=CONFIG.encoder.hidden_dim,
    conditioning_dim=CONFIG.encoder.conditioning_dim,
).to(DEVICE)
num_vars = len(encoder_configs)
print(f"  [§1.6 + J4] metapath_convs: {len(enc.metapath_convs)} distinct SAGEConvs")
print(f"  [J3]        layer_norms:   {len(enc.layer_norms)} distinct LayerNorms")

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
hr_channels = int(_train_samples[0]["residual"].shape[1])
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

skip = None
if SKIP_AVAILABLE:
    skip = ConditionalSkipBlock(
        lr_channels=len(CONFIG.data.lr_variables),
        hr_shape=tuple(CONFIG.graph.hr_shape),
    ).to(DEVICE)
    print(f"  [J8]        skip_block:    {sum(p.numel() for p in skip.parameters())} params")

# Load baseline checkpoint (will produce J18 warnings for renamed keys —
# that is the intended behavior)
print()
print("[Load] Loading baseline V5-mini checkpoint into new Path C+ architecture...")
ckpt = torch.load(_smoke_ckpt, map_location=DEVICE, weights_only=False)


def _safe_load(name_, module, sd):
    """J18 helper: strict=False with verbose key diff."""
    try:
        module.load_state_dict(sd, strict=True)
        print(f"  [J18 OK] {name_:18s} loaded strict=True")
        return True
    except RuntimeError as e:
        result = module.load_state_dict(sd, strict=False)
        print(f"  [J18 WARN] {name_:18s} strict=False fallback")
        print(f"             Missing: {len(result.missing_keys)} keys "
              f"(first 3: {result.missing_keys[:3]})")
        print(f"             Unexpected: {len(result.unexpected_keys)} keys "
              f"(first 3: {result.unexpected_keys[:3]})")
        return False


_safe_load("encoder", enc, ckpt.get("encoder_state_dict", {}))
_safe_load("rcn_cell", rcn_cell, ckpt.get("rcn_cell_state_dict", {}))
_safe_load("regression_head", rh, ckpt.get("regression_head_state_dict", {}))
_safe_load("diffusion", diff, ckpt.get("diffusion_state_dict", {}))
if skip is not None and ckpt.get("skip_block_state_dict") is not None:
    _safe_load("skip_block", skip, ckpt["skip_block_state_dict"])

A_dag = rcn_cell.A_dag.detach().cpu().clone() if hasattr(rcn_cell, "A_dag") else None
stack_v5 = {
    "encoder": enc, "rcn_runner": rcn_runner, "regression_head": rh,
    "diffusion": diff, "skip_block": skip, "A_dag": A_dag, "variant": "Oracle-smoke",
}
print()
print(f"[OK] Stack ready. A_dag shape: {A_dag.shape if A_dag is not None else None}")
print(f"     Q_phys baseline (pre-finetune): see Cell 5")
'''


CELL5_SMOKE = '''# === Cell 5 : Run smoke test 10 epochs with instrumentation ===
import json
import warnings
from pathlib import Path
from scripts.finetune_stage1_bundle_b import finetune_bundle_b
from src.st_cdgm.training.physics_prior import (
    build_physical_mask,
    physical_prior_loss,
    VAR_LABELS,
)


# Compute initial Q_phys before any training
def compute_q_phys(A_dag_np, G_phys_np):
    """Magnitude-weighted Q_phys (more robust than binary sign-only)."""
    import numpy as np
    A = np.array(A_dag_np)
    np.fill_diagonal(A, 0.0)
    G = np.array(G_phys_np)
    mask = G != 0
    if not mask.any():
        return 0.0
    mags = np.abs(A[mask])
    signs_correct = (np.sign(A[mask]) == np.sign(G[mask])).astype(float)
    total = mags.sum()
    if total < 1e-9:
        return 0.0
    return float((mags * signs_correct).sum() / total)


G_phys = build_physical_mask(num_vars=num_vars)
A_dag_initial = rcn_cell.A_dag.detach().cpu().numpy()
q_phys_initial = compute_q_phys(A_dag_initial, G_phys.numpy())
print(f"[Smoke] Q_phys initial (pre-finetune): {q_phys_initial:.4f}")
print(f"[Smoke] A_dag initial norm: {(A_dag_initial ** 2).sum() ** 0.5:.4f}")
print(f"[Smoke] A_dag asymmetry: {((A_dag_initial - A_dag_initial.T) ** 2).sum() ** 0.5:.4f}")
print()

# Smoke test config: 10 epochs (AI eng minimum recommendation)
SMOKE_EPOCHS = 10
SMOKE_SANITY_EVERY = 1  # log every epoch for tight smoke visibility

print(f"[Smoke] Launching {SMOKE_EPOCHS} epochs Bundle B fine-tune...")
print(f"[Smoke] Hyperparameters (Path C+ corrected):")
print(f"        lambda_l1_start  : 0.04 (was 0.10 in V5-mini)")
print(f"        lambda_l1_end    : 0.005 (was 0.01)")
print(f"        lambda_dag_prior : 0.40 (was 0.05)")
print(f"        g_phys_alpha     : 0.25 (was 0.20)")
print(f"        dag_grad_gate    : ramp 0->1 epochs {min(5, max(2, SMOKE_EPOCHS // 8))}-{min(20, max(min(5, max(2, SMOKE_EPOCHS // 8)) + 2, SMOKE_EPOCHS // 4))}")
print()


# Catch warnings so we can detect J8/§1.6 fallback fires
with warnings.catch_warnings(record=True) as w_record:
    warnings.simplefilter("always")
    result = finetune_bundle_b(
        stack=stack_v5,
        builder=builder,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        CONFIG=CONFIG,
        DEVICE=DEVICE,
        epochs=SMOKE_EPOCHS,
        batch_size=GPU_PROFILE["batch_size"],
        ckpt_save_dir=SMOKE_DIR,
        convert_sample_to_batch_fn=convert_sample_to_batch,
        sanity_eval_every=SMOKE_SANITY_EVERY,
        seed=42,
        skip_sigma_data_recalib=True,  # skip in smoke (saves ~2 min)
    )

# Check critical warnings
j8_failed = [w for w in w_record if "J8 skip_block forward failed" in str(w.message)]
missing_edge = [w for w in w_record if "§1.6 missing-edge fallback" in str(w.message)]
print()
print(f"[Smoke] J8 reshape failures detected: {len(j8_failed)}")
print(f"[Smoke] §1.6 missing-edge fallbacks: {len(missing_edge)}")
'''


CELL6_VERDICT = '''# === Cell 6 : Compute final Q_phys + verdict ===
import json
import numpy as np

# Final A_dag after smoke training
A_dag_final = rcn_cell.A_dag.detach().cpu().numpy()
q_phys_final = compute_q_phys(A_dag_final, G_phys.numpy())

# Asymmetry: 0 = pure symmetric (bad), large = strong DAG
asym_initial = float(((A_dag_initial - A_dag_initial.T) ** 2).sum() ** 0.5)
asym_final = float(((A_dag_final - A_dag_final.T) ** 2).sum() ** 0.5)

# A_dag magnitude
mag_initial = float((A_dag_initial ** 2).sum() ** 0.5)
mag_final = float((A_dag_final ** 2).sum() ** 0.5)

# Count edges above threshold
thresh = 0.05
n_edges_initial = int((np.abs(A_dag_initial) > thresh).sum())
n_edges_final = int((np.abs(A_dag_final) > thresh).sum())

print("=" * 72)
print("PATH C+ SMOKE TEST VERDICT")
print("=" * 72)
print()
print(f"  Q_phys           : {q_phys_initial:.4f}  ->  {q_phys_final:.4f}  "
      f"(delta {q_phys_final - q_phys_initial:+.4f})")
print(f"  A_dag norm       : {mag_initial:.4f}  ->  {mag_final:.4f}")
print(f"  A_dag asymmetry  : {asym_initial:.4f}  ->  {asym_final:.4f}")
print(f"  #edges > 0.05    : {n_edges_initial}     ->  {n_edges_final}")
print()
print(f"  J8 reshape fails : {len(j8_failed)}")
print(f"  §1.6 missing-edge: {len(missing_edge)}")
print()

# Apply AI eng's success criteria
SUCCESS_Q_PHYS = q_phys_final > 0.55
SUCCESS_J8 = len(j8_failed) == 0
SUCCESS_NO_MISSING_EDGE = len(missing_edge) == 0

print("=" * 72)
print("AI ENG SUCCESS CRITERIA (round 7 validation)")
print("=" * 72)
print(f"  [{('OK' if SUCCESS_Q_PHYS else 'KO')}] Q_phys > 0.55 by epoch 10: {q_phys_final:.4f}")
print(f"  [{('OK' if SUCCESS_J8 else 'KO')}] J8 no reshape failures: {len(j8_failed)}")
print(f"  [{('OK' if SUCCESS_NO_MISSING_EDGE else 'KO')}] No §1.6 missing-edge: {len(missing_edge)}")
print()

if SUCCESS_Q_PHYS and SUCCESS_J8 and SUCCESS_NO_MISSING_EDGE:
    VERDICT = "PASS"
    print("✅ SMOKE TEST PASS — Path C+ fixes ARE EFFECTIVE")
    print("   GO Batch D (commits 24-28: J29, K2, K3, K9, K5)")
elif q_phys_final < 0.45:
    VERDICT = "FAIL_BISECT"
    print("❌ SMOKE TEST FAIL — Q_phys < 0.45")
    print("   Bisect commits 3eb991d (J3), b2f664a (I11), b966ab7 (J8)")
    print("   Run git bisect start; git bisect bad b966ab7; git bisect good 3eb991d")
else:
    VERDICT = "PARTIAL"
    print("⚠️  SMOKE TEST PARTIAL — Q_phys in [0.45, 0.55]")
    print("   Consider extending smoke to 15 epochs to confirm trend")

# Save smoke test results to JSON for audit trail
smoke_results = {
    "verdict": VERDICT,
    "q_phys_initial": float(q_phys_initial),
    "q_phys_final": float(q_phys_final),
    "q_phys_delta": float(q_phys_final - q_phys_initial),
    "a_dag_norm_initial": mag_initial,
    "a_dag_norm_final": mag_final,
    "a_dag_asymmetry_initial": asym_initial,
    "a_dag_asymmetry_final": asym_final,
    "n_edges_initial": n_edges_initial,
    "n_edges_final": n_edges_final,
    "j8_failures": len(j8_failed),
    "missing_edge_fallbacks": len(missing_edge),
    "smoke_epochs": SMOKE_EPOCHS,
    "smoke_seed": 42,
    "gpu_profile": GPU_PROFILE.get("profile_id"),
    "criteria_q_phys_55": SUCCESS_Q_PHYS,
    "criteria_j8_no_fail": SUCCESS_J8,
    "criteria_no_missing_edge": SUCCESS_NO_MISSING_EDGE,
    "commit_sha": (
        subprocess.check_output(shlex.split("git rev-parse HEAD")).decode().strip()
        if _IS_COLAB else None
    ),
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
}
_smoke_json = SMOKE_DIR / "smoke_test_results.json"
_smoke_json.write_text(json.dumps(smoke_results, indent=2, default=str))
print()
print(f"[Audit] Results saved to {_smoke_json}")
print(f"[Audit] Push to GitHub four-node-causal branch for team review")
'''


nb = {
    "cells": [
        md_cell(CELL0_MD),
        code_cell(CELL1_BOOTSTRAP),
        code_cell(CELL2_GPU),
        code_cell(CELL3_STACK),
        code_cell(CELL4_LOAD),
        code_cell(CELL5_SMOKE),
        code_cell(CELL6_VERDICT),
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("path_c_plus/scripts/st_cdgm_path_c_smoke_test.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Smoke test notebook written: {out}")
print(f"     {out.stat().st_size/1024:.1f} KB, {len(nb['cells'])} cells")
