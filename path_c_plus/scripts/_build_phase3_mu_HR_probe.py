"""Builder script for phase3_mu_HR_probe.ipynb.

Generates a standalone Jupyter notebook to probe mu_HR variants on a frozen
2-stage ST-CDGM checkpoint (9-node seed42, with 6-node fallback) WITHOUT
any retraining. Produces a verdict on whether mu_HR's direction is correct
and exploitable for AdaLN warm-start.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "phase3_mu_HR_probe.ipynb"


def code_cell(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def md_cell(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True),
    }


# ---------------------------------------------------------------------------
# Cell 0 : title markdown
# ---------------------------------------------------------------------------
CELL_TITLE = """# Phase 3 -- mu_HR direction probe (no retraining)

**Objective.** On a frozen ST-CDGM 2-stage checkpoint (9-node seed42, 6-node
fallback), probe 5 reconstructions of mu_HR (0x, 1x, 2x, 5x, oracle alpha*)
to answer one question:

> **Is the DIRECTION of mu_HR correct and exploitable?**

We never retrain. We sample the diffusion UNet once per batch with the real
mu_HR (since training used `mardani_fix_zero_mu_HR_in_conditioning=True`,
the UNet has only seen zeros in its mu_HR channel; the only thing that
matters for the residual decomposition is `delta = HR - baseline - mu_HR`).
We then extract `delta_pred = pred_ref - baseline - mu_HR` and recompose
five variants by varying the mu_HR coefficient in the reconstruction.

**Verdicts**
- **A.** alpha_opt_mean > 0.8 AND F1@p99(oracle) > 0.512 -> direction correct,
  magnitude insufficient -> **AdaLN warm-start viable**.
- **B.** F1@p99(5x) > F1@p99(1x) -> scaling helps -> **AdaLN warm-start**
  (boost signal).
- **C.** F1@p99(oracle) <= 0.512 -> mu_HR has wrong direction -> deeper issue.
- **D.** F1@p99(ablation) > F1@p99(1x) -> mu_HR hurts -> mu_HR is biased,
  must be pruned.

**Reference metrics (current 9-node seed42)**
- RMSE = 0.14135, Pearson = 0.7667, F1@p99 = 0.4531
- Noncausal v4 (CorrDiff Normal V2): F1@p99 = 0.512
"""

# ---------------------------------------------------------------------------
# Cell 1 : Bootstrap
# ---------------------------------------------------------------------------
CELL_BOOTSTRAP = """# >>> Cell 1 : Bootstrap (Colab clone, sys.path, deps)
import os, sys, subprocess, shlex
from pathlib import Path

IN_COLAB = 'google.colab' in sys.modules
GIT_URL = 'https://github.com/leonelkenfack/stcdgm.git'
GIT_BRANCH = 'four-node-causal'

if IN_COLAB:
    from google.colab import drive
    if not os.path.ismount('/content/drive'):
        drive.mount('/content/drive', force_remount=False)
    DRIVE_ROOT = Path('/content/drive/MyDrive/climate_data')
    REPO_DIR = Path('/content/climate_data')
    if not (REPO_DIR / '.git').exists():
        subprocess.run(
            shlex.split(f'git clone --depth 200 -b {GIT_BRANCH} {GIT_URL} {REPO_DIR}'),
            check=True,
        )
    else:
        subprocess.run(
            shlex.split(f'git -C {REPO_DIR} fetch --depth=200 origin {GIT_BRANCH}'),
            check=True,
        )
        subprocess.run(
            shlex.split(f'git -C {REPO_DIR} reset --hard origin/{GIT_BRANCH}'),
            check=True,
        )
    os.chdir(str(REPO_DIR))
else:
    DRIVE_ROOT = Path('c:/Users/reall/Desktop/climate_data')
    REPO_DIR = DRIVE_ROOT

if str(REPO_DIR / 'src') not in sys.path:
    sys.path.insert(0, str(REPO_DIR / 'src'))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

# Idempotent dep install (skip if everything already imports)
_NEED_INSTALL = False
try:
    import torch_geometric  # noqa
    import cftime           # noqa
    import h5netcdf         # noqa
    import xbatcher         # noqa
    import diffusers        # noqa
    from omegaconf import OmegaConf  # noqa
except ImportError as _e:
    print(f'Missing dep -> batch install : {_e}')
    _NEED_INSTALL = True

if _NEED_INSTALL:
    EXTRA_DEPS = [
        'omegaconf==2.3.0', 'hydra-core==1.3.2', 'diffusers==0.36.0',
        'transformers==4.57.6', 'accelerate==1.12.0', 'huggingface-hub==0.36.0',
        'safetensors==0.7.0', 'xbatcher', 'webdataset', 'cftime', 'h5netcdf',
        'numcodecs', 'torch-geometric', 'einops', 'scipy', 'h5py', 'netCDF4',
        'xarray', 'dask', 'zarr',
    ]
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-warn-script-location', '-q']
        + EXTRA_DEPS,
        check=True,
    )

print(f'DRIVE_ROOT = {DRIVE_ROOT}')
print(f'REPO_DIR   = {REPO_DIR}')
print(f'IN_COLAB   = {IN_COLAB}')
"""

# ---------------------------------------------------------------------------
# Cell 2 : Imports + paths + constants
# ---------------------------------------------------------------------------
CELL_IMPORTS = """# >>> Cell 2 : Imports + paths + constants
import json
import time
import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# ST-CDGM internals (same set as st_cdgm_seed42_eval.ipynb)
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder
from st_cdgm.models.intelligible_encoder import (
    IntelligibleVariableEncoder,
    IntelligibleVariableConfig,
    SpatialConditioningProjector,
)
from st_cdgm.models.causal_rcn import RCNCell, RCNSequenceRunner
from st_cdgm.models.diffusion_decoder import CausalDiffusionDecoder
from st_cdgm.models.regression_head import GraphToGridDecoder
from st_cdgm.models.edm_preconditioner import EDMConfig
from st_cdgm.evaluation import compute_f1_extremes
from st_cdgm.training.two_stage import freeze_stage1

# Optional helpers (build_two_stage_inputs may not exist on all branches)
try:
    from st_cdgm.evaluation.two_stage_inference import build_two_stage_inputs
    _HAS_BUILD_INPUTS = True
except ImportError:
    build_two_stage_inputs = None
    _HAS_BUILD_INPUTS = False

try:
    from st_cdgm.training.stage1_paths import resolve_run_variant
except ImportError:
    def resolve_run_variant(cfg):  # fallback
        return 'causal'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'DEVICE = {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU    = {torch.cuda.get_device_name(0)}')

# Sampling / batch constants
N_STEPS = 18           # EDM Heun steps (idem eval officielle)
K_SAMPLES = 4          # samples par batch (probe -> small K, deterministic-ish mean)
BATCH_SIZE = 4         # batches du DataLoader
N_PROBE_BATCHES = 16   # combien de batches traiter (~64 samples)

# Reference metrics
REF_RMSE          = 0.14135   # 9-node seed42 current
REF_PEARSON       = 0.7667    # 9-node seed42 current
REF_F1P99         = 0.4531    # 9-node seed42 current
NONCAUSAL_F1P99   = 0.512     # noncausal v4 target (the bar to beat)

# Variants tested
VARIANT_NAMES = ['ablation_0x', 'ref_1x', 'scale_2x', 'scale_5x', 'oracle']

print(f'\\nProbe config :')
print(f'  N_STEPS         = {N_STEPS}')
print(f'  K_SAMPLES       = {K_SAMPLES}')
print(f'  BATCH_SIZE      = {BATCH_SIZE}')
print(f'  N_PROBE_BATCHES = {N_PROBE_BATCHES}')
print(f'  variants        = {VARIANT_NAMES}')
print(f'  References      : F1@p99 current={REF_F1P99}, noncausal={NONCAUSAL_F1P99}')
"""

# ---------------------------------------------------------------------------
# Cell 3 : Checkpoint resolution + stack builder
# ---------------------------------------------------------------------------
CELL_STACK = """# >>> Cell 3 : Checkpoint resolution + build_fresh_stack()
# Tries 9-node seed42 first, falls back to 6-node if missing.

CKPT_9NODE = DRIVE_ROOT / 'oracle_9node' / 'seed_42' / 'epoch_last.pth'
CKPT_6NODE = DRIVE_ROOT / 'oracle_full'  / 'seed_42' / 'epoch_last.pth'

if CKPT_9NODE.exists():
    CKPT_PATH = CKPT_9NODE
    STACK_TAG = '9node'
elif CKPT_6NODE.exists():
    CKPT_PATH = CKPT_6NODE
    STACK_TAG = '6node_fallback'
    print('[warn] 9-node checkpoint missing, falling back to 6-node seed42')
else:
    raise FileNotFoundError(
        f'No checkpoint found. Tried:\\n  {CKPT_9NODE}\\n  {CKPT_6NODE}'
    )
print(f'CKPT_PATH = {CKPT_PATH}')
print(f'STACK_TAG = {STACK_TAG}')

# Output dir for probe results
OUT_DIR = CKPT_PATH.parent / 'phase3_probe'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / 'probe_metrics.json'
OUT_PNG_BAR = OUT_DIR / 'probe_f1_bar.png'
OUT_PNG_ALPHA_HIST = OUT_DIR / 'probe_alpha_hist.png'
OUT_PNG_ALPHA_SCATTER = OUT_DIR / 'probe_alpha_scatter.png'
print(f'OUT_DIR = {OUT_DIR}')

# ---- CONFIG load ----
CONFIG = OmegaConf.load(str(REPO_DIR / 'config' / 'training_config.yaml'))
_corrdiff_path = REPO_DIR / 'config' / 'training_config_corrdiff_normal.yaml'
if _corrdiff_path.exists():
    CONFIG = OmegaConf.merge(CONFIG, OmegaConf.load(str(_corrdiff_path)))


def _load_sd(m, sd):
    \"\"\"Tolerant state-dict loader (handles _orig_mod / DataParallel wrappers).\"\"\"
    if m is None or sd is None:
        return 0
    base = m.module if hasattr(m, 'module') and not hasattr(m, '_orig_mod') else m
    base = getattr(base, '_orig_mod', base)
    stripped = {k.replace('_orig_mod.', ''): v for k, v in sd.items()}
    target = base.state_dict()
    matched = {}
    for tk in target.keys():
        ntk = tk.replace('_orig_mod.', '')
        if ntk not in stripped:
            continue
        v = stripped[ntk]
        try:
            ls = tuple(target[tk].shape) if hasattr(target[tk], 'shape') else None
        except (RuntimeError, ValueError):
            ls = None
        cs = tuple(v.shape) if hasattr(v, 'shape') else None
        if ls is not None and cs is not None and ls != cs:
            continue
        matched[tk] = v
    base.load_state_dict(matched, strict=False)
    return len(matched)
"""

# ---------------------------------------------------------------------------
# Cell 4 : Data pipeline + val loader
# ---------------------------------------------------------------------------
CELL_DATA = """# >>> Cell 4 : Data pipeline + K9 val split + builder + iterate_batches
import xarray as xr
from torch.utils.data import DataLoader as _DataLoader

DATA_ROOT_DRIVE = DRIVE_ROOT / 'data'
DATA_ROOT_SSD = Path('/content/data_local')
DATA_ROOT = DATA_ROOT_SSD if (DATA_ROOT_SSD / 'train').exists() else DATA_ROOT_DRIVE
print(f'DATA_ROOT = {DATA_ROOT}')

LR_PATH = str(DATA_ROOT / 'train' / 'predictor_ACCESS-CM2_hist.nc')
HR_PATH = str(DATA_ROOT / 'train' / 'pr_ACCESS-CM2_hist.nc')
_static_p = DATA_ROOT / 'static_predictors' / 'ERA5_eval_ccam_12km.198110_NZ_Invariant.nc'
STATIC_PATH = str(_static_p) if _static_p.exists() else None
_mean_p = DATA_ROOT / 'normalization_coefs' / 'mean_1974_2011.nc'
_std_p = DATA_ROOT / 'normalization_coefs' / 'std_1974_2011.nc'
MEAN_PATH = str(_mean_p) if _mean_p.exists() else None
STD_PATH = str(_std_p) if _std_p.exists() else None

assert Path(LR_PATH).exists(), f'LR missing : {LR_PATH}'
assert Path(HR_PATH).exists(), f'HR missing : {HR_PATH}'

K9_DATES = {
    'train':   ['1980-01-01', '2009-12-31'],
    'val':     ['2010-01-01', '2011-12-31'],
    'test':    ['2012-01-01', '2013-12-31'],
    'holdout': ['2014-01-01', '2014-12-31'],
}

SEQ_LEN = int(CONFIG.data.seq_len)
_default_lr = ['q_500', 'q_850', 'u_500', 'u_850', 'v_500', 'v_850', 't_500', 't_850']
_default_hr = ['pr']
LR_VARIABLES = list(CONFIG.data.lr_variables) if CONFIG.data.get('lr_variables') else _default_lr
HR_VARIABLES = list(CONFIG.data.hr_variables) if CONFIG.data.get('hr_variables') else _default_hr
STATIC_VARIABLES = (
    list(CONFIG.data.static_variables) if CONFIG.data.get('static_variables')
    else (['orog', 'he', 'vegt'] if STATIC_PATH else None)
)

_lr_avail = set(xr.open_dataset(LR_PATH).data_vars)
_hr_avail = set(xr.open_dataset(HR_PATH).data_vars)
if not set(LR_VARIABLES).issubset(_lr_avail):
    LR_VARIABLES = [v for v in LR_VARIABLES if v in _lr_avail] or sorted(_lr_avail)[:8]
if not set(HR_VARIABLES).issubset(_hr_avail):
    HR_VARIABLES = [sorted(_hr_avail)[0]]

pipeline = NetCDFDataPipeline(
    lr_path=LR_PATH, hr_path=HR_PATH, static_path=STATIC_PATH,
    seq_len=SEQ_LEN,
    baseline_strategy=str(CONFIG.data.baseline_strategy),
    baseline_factor=int(CONFIG.data.baseline_factor),
    normalize=bool(CONFIG.data.normalize),
    nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
    precipitation_delta=float(CONFIG.data.precipitation_delta),
    lr_variables=LR_VARIABLES, hr_variables=HR_VARIABLES,
    static_variables=STATIC_VARIABLES,
    means_path=MEAN_PATH if (MEAN_PATH and os.path.exists(MEAN_PATH)) else None,
    stds_path=STD_PATH if (STD_PATH and os.path.exists(STD_PATH)) else None,
    train_start_date=K9_DATES['train'][0], train_end_date=K9_DATES['train'][1],
    val_start_date=K9_DATES['val'][0], val_end_date=K9_DATES['val'][1],
    test_start_date=K9_DATES['test'][0], test_end_date=K9_DATES['test'][1],
    temporal_holdout_start_date=K9_DATES['holdout'][0],
    temporal_holdout_end_date=K9_DATES['holdout'][1],
)

val_dataset = pipeline.build_sequence_dataset(
    split='val', seq_len=SEQ_LEN, stride=int(CONFIG.data.stride), as_torch=True,
)
sample = next(iter(val_dataset))

PIN_MEMORY = bool(torch.cuda.is_available())
val_dataloader = _DataLoader(
    val_dataset, shuffle=False,
    batch_size=BATCH_SIZE, num_workers=0, pin_memory=PIN_MEMORY,
    collate_fn=lambda x: x,
)

lr_shape = tuple(CONFIG.graph.lr_shape)
hr_shape = tuple(CONFIG.graph.hr_shape)
builder = HeteroGraphBuilder(
    lr_shape=lr_shape, hr_shape=hr_shape,
    static_dataset=pipeline.get_static_dataset(),
    include_mid_layer=CONFIG.graph.include_mid_layer,
)


def convert_sample_to_batch(sample, builder, device):
    lr_seq = sample['lr']
    seq_len = lr_seq.shape[0]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    dynamic_features = {nt: lr_nodes_steps[0] for nt in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_features).to(device)
    return {
        'lr': lr_tensor,
        'residual': sample['residual'],
        'baseline': sample.get('baseline'),
        'hetero': hetero,
        'time': sample.get('time'),
    }


def iterate_batches(dataloader, builder, device):
    for batch_list in dataloader:
        if not isinstance(batch_list, list):
            batch_list = [batch_list]
        yield [convert_sample_to_batch(s, builder, device) for s in batch_list]


RCN_DRIVER_DIM = sample['lr'].shape[1]
hr_channels = sample['residual'].shape[1]
print(f'\\nval_dataset OK -- LR={tuple(sample[\"lr\"].shape)}, '
      f'residual={tuple(sample[\"residual\"].shape)}')
print(f'RCN_DRIVER_DIM={RCN_DRIVER_DIM}, hr_channels={hr_channels}')
"""

# ---------------------------------------------------------------------------
# Cell 5 : build_fresh_stack + load checkpoint
# ---------------------------------------------------------------------------
CELL_BUILD = """# >>> Cell 5 : build_fresh_stack + load checkpoint

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Encoder configs from CONFIG.encoder.metapaths
allowed_nodes = set(builder.dynamic_node_types) | set(builder.static_node_types)
encoder_configs = []
for _mp in CONFIG.encoder.metapaths:
    if _mp.src in allowed_nodes and _mp.target in allowed_nodes:
        encoder_configs.append(IntelligibleVariableConfig(
            name=_mp.name,
            meta_path=(_mp.src, _mp.relation, _mp.target),
            pool=_mp.get('pool', 'mean'),
        ))
if pipeline.get_static_dataset() is not None:
    encoder_configs.append(IntelligibleVariableConfig(
        name='static',
        meta_path=('SP_HR', 'causes', 'GP850'),
        pool='mean',
    ))
encoder = IntelligibleVariableEncoder(
    configs=encoder_configs,
    hidden_dim=int(CONFIG.encoder.hidden_dim),
    conditioning_dim=int(CONFIG.encoder.conditioning_dim),
).to(DEVICE)
num_vars = len(encoder_configs)
print(f'num_vars (causal nodes) = {num_vars}')

rcn_cell = RCNCell(
    num_vars=num_vars,
    hidden_dim=int(CONFIG.rcn.hidden_dim),
    driver_dim=RCN_DRIVER_DIM,
    reconstruction_dim=RCN_DRIVER_DIM,
    dropout=float(CONFIG.rcn.dropout),
).to(DEVICE)
rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get('detach_interval'))

rh_cfg = CONFIG.two_stage.regression_head
regression_head = GraphToGridDecoder(
    d_model=int(rh_cfg.d_model),
    hr_h=int(CONFIG.diffusion.height), hr_w=int(CONFIG.diffusion.width),
    intermediate_h=int(rh_cfg.intermediate_h), intermediate_w=int(rh_cfg.intermediate_w),
    n_heads=int(rh_cfg.n_heads), refine_channels=int(rh_cfg.refine_channels),
    output_channels=1,
).to(DEVICE)

UNET_KWARGS = OmegaConf.to_container(CONFIG.diffusion.unet_kwargs, resolve=True)
for _k in ('down_block_types', 'up_block_types'):
    if _k in UNET_KWARGS and isinstance(UNET_KWARGS[_k], list):
        UNET_KWARGS[_k] = tuple(UNET_KWARGS[_k])
UNET_KWARGS['projection_class_embeddings_input_dim'] = (
    num_vars * int(CONFIG.diffusion.conditioning_dim)
)

_edm_cfg_raw = CONFIG.diffusion.get('edm', {})
_edm_config = EDMConfig.from_yaml_dict(_edm_cfg_raw)

diffusion = CausalDiffusionDecoder(
    in_channels=hr_channels,
    conditioning_dim=int(CONFIG.diffusion.conditioning_dim),
    height=int(CONFIG.diffusion.height), width=int(CONFIG.diffusion.width),
    num_diffusion_steps=int(CONFIG.diffusion.steps),
    unet_kwargs=UNET_KWARGS,
    use_gradient_checkpointing=bool(CONFIG.diffusion.get('use_gradient_checkpointing', False)),
    scheduler_type=CONFIG.diffusion.get('scheduler_type', 'edm_karras'),
    conv_padding_mode=CONFIG.diffusion.get('conv_padding_mode', 'zeros'),
    anti_checkerboard=bool(CONFIG.diffusion.get('anti_checkerboard', False)),
    edm_config=_edm_config,
    causal_concat=True,
).to(DEVICE)

_spatial_target_shape = tuple(CONFIG.diffusion.get('spatial_target_shape', [6, 7]))
spatial_projector = SpatialConditioningProjector(
    num_vars=num_vars,
    hidden_dim=int(CONFIG.rcn.hidden_dim),
    conditioning_dim=int(CONFIG.diffusion.conditioning_dim),
    lr_shape=lr_shape,
    target_shape=_spatial_target_shape,
).to(DEVICE)

# ---- pre-warm SAGEConv lazy params with one batch ----
print('[pre-warm] materializing SAGEConv lazy params...')
encoder.train(); rcn_cell.train(); regression_head.train()
with torch.no_grad():
    for _conv in iterate_batches(val_dataloader, builder, DEVICE):
        for _b in _conv:
            _H = encoder.init_state(_b['hetero']).to(DEVICE)
            _lr = _b['lr'].to(DEVICE)
            _drv = [_lr[t] for t in range(_lr.shape[0])]
            _seq = rcn_runner.run(_H, _drv, reconstruction_sources=None)
            _ = regression_head(_seq.states[-1])
            break
        break
print('[pre-warm] done')

# ---- Load checkpoint ----
print(f'\\n[load] {CKPT_PATH}')
ck = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
ts_state = ck.get('two_stage') or {}
print(f'  stage1_epoch_done : {ts_state.get(\"stage1_epoch_done\")}')
print(f'  stage2_epoch_done : {ts_state.get(\"stage2_epoch_done\")}')
print(f'  sigma_data        : {ts_state.get(\"sigma_data\")}')

n_enc = _load_sd(encoder, ck.get('encoder_state_dict'))
n_rcn = _load_sd(rcn_cell, ck.get('rcn_cell_state_dict'))
n_rh  = _load_sd(regression_head, ck.get('regression_head_state_dict'))
n_sp  = _load_sd(spatial_projector, ck.get('spatial_projector_state_dict'))
print(f'  encoder={n_enc}, rcn={n_rcn}, regression_head={n_rh}, projector={n_sp} keys loaded')

# Prefer EMA diffusion weights if available (matches eval protocol)
ema_sd = ck.get('diffusion_ema_state_dict')
if ema_sd is not None:
    n_d = _load_sd(diffusion, ema_sd)
    print(f'  diffusion (EMA) = {n_d} keys')
else:
    n_d = _load_sd(diffusion, ck.get('diffusion_state_dict'))
    print(f'  diffusion (raw) = {n_d} keys')

# Restore EDM sigma_data if saved (must match training)
if ts_state.get('sigma_data') is not None:
    diffusion.edm_config = EDMConfig(
        sigma_data=float(ts_state['sigma_data']),
        sigma_min=float(ts_state.get('sigma_min', max(1e-4, float(ts_state['sigma_data']) * 0.02))),
        sigma_max=float(CONFIG.diffusion.edm.sigma_max),
        rho=float(CONFIG.diffusion.edm.rho),
        P_mean=float(CONFIG.diffusion.edm.P_mean),
        P_std=float(CONFIG.diffusion.edm.P_std),
    )
    print(f'  EDM sigma_data restored = {ts_state[\"sigma_data\"]}')

# Freeze Stage 1 + eval mode everywhere
freeze_stage1(encoder, rcn_runner.cell, regression_head)
encoder.eval(); rcn_runner.cell.eval(); regression_head.eval()
spatial_projector.eval(); diffusion.eval()
print('\\n[load] stack ready -- frozen, eval mode')

# Sanity : check UNet input channels (3 = [delta, mu_HR_slot, baseline_log])
_diff_core = getattr(diffusion, '_orig_mod', diffusion)
_diff_core = getattr(_diff_core, 'module', _diff_core)
print(f'\\nUNet in_channels   = {_diff_core.unet.config.in_channels}')
print(f'causal_concat flag = {bool(getattr(_diff_core, \"causal_concat\", False))}')
"""

# ---------------------------------------------------------------------------
# Cell 6 : Probe helpers + run_probe_variants
# ---------------------------------------------------------------------------
CELL_PROBE = r"""# >>> Cell 6 : Run probe variants
#
# Core trick of the Mardani fix (mardani_fix_zero_mu_HR_in_conditioning=True):
#   - During training : UNet input = [delta_noisy, ZEROS, baseline_log]
#   - Target          : delta = HR - baseline - mu_HR     (mu_HR enters only here)
#   - At sampling     : we pass mu_HR_real so the residual decomposition works:
#                       pred_ref = baseline + mu_HR + delta_UNet
#
# So : delta_pred = pred_ref - baseline - mu_HR  -> the "pure UNet delta"
# We can then recompose any variant: pred(alpha) = baseline + alpha * mu_HR + delta_pred
# Per-sample oracle alpha* :
#     alpha_opt = <mu_HR, HR_true - baseline - delta_pred> / ||mu_HR||^2

RUN_VARIANT = resolve_run_variant(CONFIG)
print(f'RUN_VARIANT = {RUN_VARIANT}')

_diff_core = getattr(diffusion, '_orig_mod', diffusion)
_diff_core = getattr(_diff_core, 'module', _diff_core)
_causal_concat = bool(getattr(_diff_core, 'causal_concat', False))
assert _causal_concat, 'This probe assumes causal_concat=True (Mardani fix path)'

EVAL_SCHEDULER = str(CONFIG.diffusion.get('scheduler_type', 'edm_karras'))
EVAL_CFG_SCALE = 1.0  # edm_karras has no CFG implemented; honest setting
print(f'sampler: scheduler={EVAL_SCHEDULER}, cfg_scale={EVAL_CFG_SCALE}, n_steps={N_STEPS}')


def _build_inputs_for_batch(batch):
    # Return cond, mu_HR, baseline_log, HR_target_residual for one converted batch.
    # Falls back to manual extraction if build_two_stage_inputs is missing.
    if _HAS_BUILD_INPUTS:
        return build_two_stage_inputs(
            batch, variant=RUN_VARIANT,
            regression_head=regression_head,
            encoder=encoder if RUN_VARIANT == 'causal' else None,
            rcn_runner=rcn_runner if RUN_VARIANT == 'causal' else None,
            builder=builder, device=DEVICE,
        )
    # Manual fallback : replicate the 2-stage forward
    target = batch['residual'][-1].to(DEVICE)
    if target.dim() == 3:
        target = target.unsqueeze(0)
    baseline = batch.get('baseline')
    if baseline is not None:
        baseline_log = baseline[-1].to(DEVICE)
        if baseline_log.dim() == 3:
            baseline_log = baseline_log.unsqueeze(0)
    else:
        baseline_log = torch.zeros_like(target)
    # Stage 1 forward to produce mu_HR
    H = encoder.init_state(batch['hetero']).to(DEVICE)
    lr_seq = batch['lr'].to(DEVICE)
    drivers = [lr_seq[t] for t in range(lr_seq.shape[0])]
    seq_out = rcn_runner.run(H, drivers, reconstruction_sources=None)
    h_last = seq_out.states[-1]
    mu_hr = regression_head(h_last)
    mu_hr = torch.nan_to_num(mu_hr, nan=0.0, posinf=0.0, neginf=0.0)
    baseline_log = torch.nan_to_num(baseline_log, nan=0.0, posinf=0.0, neginf=0.0)
    return None, mu_hr, baseline_log, target


@torch.no_grad()
def _sample_once(cond, mu_HR, baseline_log):
    # One EDM Heun sample with the real mu_HR (training-time conditioning path).
    out = _diff_core.sample(
        conditioning=cond,
        mu_HR=mu_HR,
        baseline_log=baseline_log,
        scheduler_type=EVAL_SCHEDULER,
        num_steps=N_STEPS,
        cfg_scale=EVAL_CFG_SCALE,
        apply_constraints=False,
    )
    return out.residual


def _compute_alpha_opt(mu_HR, residual_to_explain, eps=1e-8):
    # alpha* per sample : <mu_HR, residual_to_explain> / ||mu_HR||^2, clamped [0,10].
    num = (mu_HR * residual_to_explain).sum(dim=[1, 2, 3])
    den = mu_HR.pow(2).sum(dim=[1, 2, 3]) + eps
    alpha = num / den
    return alpha.clamp(0.0, 10.0)


def run_probe_variants(n_batches=N_PROBE_BATCHES, k_samples=K_SAMPLES):
    # Iterate val_loader, run K samples per batch, build 5 reconstructions.
    # Returns a dict storing per-variant predictions + targets and per-batch
    # alpha_opt statistics.
    print(f'\n[probe] starting -- n_batches={n_batches}, k_samples={k_samples}')

    storage = {
        v: {'preds': [], 'mu_HR_list': [], 'targets_full': []}
        for v in VARIANT_NAMES
    }
    # Common buffers
    all_targets_HR = []        # baseline + mu_HR + target_residual  (= HR_true_log)
    all_mu_HR = []
    all_baseline = []
    all_delta_pred = []
    all_alpha_opt = []         # per sample
    per_batch_diagnostics = []

    t0 = time.time()
    count = 0
    with torch.no_grad():
        for converted in iterate_batches(val_dataloader, builder, DEVICE):
            for batch in converted:
                if count >= n_batches:
                    break
                cond, mu_HR, baseline_log, tgt_residual = _build_inputs_for_batch(batch)
                # K independent samples -> ensemble mean (reduces sampling noise)
                pred_refs = torch.stack(
                    [_sample_once(cond, mu_HR, baseline_log) for _ in range(k_samples)],
                    dim=0,
                ).mean(dim=0)
                # pred_refs is the model's residual estimate -> reconstruction in log space
                # The model's output (.residual) in the causal_concat path is the FULL HR
                # field reconstruction = baseline + mu_HR + delta_UNet.
                # See diffusion_decoder.sample(): when causal_concat, it returns the
                # reconstructed field. delta_pred = pred_refs - baseline - mu_HR.
                delta_pred = pred_refs - baseline_log - mu_HR
                # Sanity : delta_pred should be O(target_residual.std()), not O(0).
                # If output were already a "delta", subtracting baseline+mu_HR twice
                # would explode it. We log this once.
                if count == 0:
                    print(f'  [diag] pred_ref mean/std  : {pred_refs.mean():.4f} / {pred_refs.std():.4f}')
                    print(f'  [diag] baseline mean/std  : {baseline_log.mean():.4f} / {baseline_log.std():.4f}')
                    print(f'  [diag] mu_HR mean/std     : {mu_HR.mean():.4f} / {mu_HR.std():.4f}')
                    print(f'  [diag] delta_pred m/s     : {delta_pred.mean():.4f} / {delta_pred.std():.4f}')
                    print(f'  [diag] target_res m/s     : {tgt_residual.mean():.4f} / {tgt_residual.std():.4f}')

                # HR_true_log = baseline + mu_HR + target_residual
                HR_true = baseline_log + mu_HR + tgt_residual

                # Variants -- each is HR_pred_log
                pred_ablation = baseline_log + 0.0      * mu_HR + delta_pred  # 0x
                pred_ref_1x   = baseline_log + 1.0      * mu_HR + delta_pred  # 1x (baseline)
                pred_2x       = baseline_log + 2.0      * mu_HR + delta_pred  # 2x
                pred_5x       = baseline_log + 5.0      * mu_HR + delta_pred  # 5x

                # Oracle per-sample alpha*
                residual_true = HR_true - baseline_log - delta_pred
                alpha_opt = _compute_alpha_opt(mu_HR, residual_true)  # [B]
                alpha_b = alpha_opt.view(-1, 1, 1, 1)
                pred_oracle = baseline_log + alpha_b * mu_HR + delta_pred

                # Store
                storage['ablation_0x']['preds'].append(pred_ablation.cpu())
                storage['ref_1x'     ]['preds'].append(pred_ref_1x.cpu())
                storage['scale_2x'   ]['preds'].append(pred_2x.cpu())
                storage['scale_5x'   ]['preds'].append(pred_5x.cpu())
                storage['oracle'     ]['preds'].append(pred_oracle.cpu())
                for v in VARIANT_NAMES:
                    storage[v]['targets_full'].append(HR_true.cpu())
                    storage[v]['mu_HR_list'].append(mu_HR.cpu())

                all_targets_HR.append(HR_true.cpu())
                all_mu_HR.append(mu_HR.cpu())
                all_baseline.append(baseline_log.cpu())
                all_delta_pred.append(delta_pred.cpu())
                all_alpha_opt.append(alpha_opt.cpu())

                # Per-batch diagnostics
                mu_norm = float(mu_HR.norm().item())
                pred_norm = float(pred_refs.norm().item())
                per_batch_diagnostics.append({
                    'batch': count,
                    'alpha_opt_mean': float(alpha_opt.mean().item()),
                    'alpha_opt_std': float(alpha_opt.std().item()),
                    'mu_HR_norm': mu_norm,
                    'pred_ref_norm': pred_norm,
                    'mu_contribution': float(mu_norm / max(pred_norm, 1e-8)),
                })

                count += 1
                if count % 4 == 0 or count == 1:
                    print(f'  batch {count}/{n_batches}  '
                          f'alpha_mean={alpha_opt.mean().item():+.3f}  '
                          f'alpha_std={alpha_opt.std().item():.3f}  '
                          f'mu_contrib={per_batch_diagnostics[-1]["mu_contribution"]:.3f}')
            if count >= n_batches:
                break
    elapsed = time.time() - t0
    print(f'[probe] done in {elapsed:.1f}s, total={count} batches')

    return {
        'variants': storage,
        'all_targets_HR': torch.cat(all_targets_HR, dim=0),
        'all_mu_HR': torch.cat(all_mu_HR, dim=0),
        'all_baseline': torch.cat(all_baseline, dim=0),
        'all_delta_pred': torch.cat(all_delta_pred, dim=0),
        'all_alpha_opt': torch.cat(all_alpha_opt, dim=0),
        'per_batch_diagnostics': per_batch_diagnostics,
        'n_samples_total': count * BATCH_SIZE,
        'elapsed_s': elapsed,
    }


probe_result = run_probe_variants()
print(f'\n[probe] HR_true tensor shape : {tuple(probe_result["all_targets_HR"].shape)}')
print(f'[probe] alpha_opt overall    : mean={probe_result["all_alpha_opt"].mean():.4f}, '
      f'std={probe_result["all_alpha_opt"].std():.4f}, '
      f'min={probe_result["all_alpha_opt"].min():.4f}, '
      f'max={probe_result["all_alpha_opt"].max():.4f}')
"""

# ---------------------------------------------------------------------------
# Cell 7 : compute_all_metrics
# ---------------------------------------------------------------------------
CELL_METRICS = r"""# >>> Cell 7 : Compute metrics per variant

def _pearson(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    am = a - a.mean()
    bm = b - b.mean()
    num = (am * bm).sum()
    den = torch.sqrt((am * am).sum() * (bm * bm).sum() + eps)
    return float((num / den).item())


def compute_all_metrics(probe_result):
    targets = probe_result['all_targets_HR']        # HR_true_log [N,1,H,W]
    mu_HR_all = probe_result['all_mu_HR']
    valid = torch.isfinite(targets)

    metrics_by_variant = {}
    for v in VARIANT_NAMES:
        preds = torch.cat(probe_result['variants'][v]['preds'], dim=0)
        # Mask invalids identically
        p = torch.where(valid, preds, torch.zeros_like(preds))
        t = torch.where(valid, targets, torch.zeros_like(targets))
        # Scalar metrics on valid mask
        diff = (p - t)[valid]
        rmse = float((diff.pow(2).mean()).sqrt().item())
        mae = float(diff.abs().mean().item())
        pear = _pearson(p[valid], t[valid])
        # F1 extremes (use the masked tensors -- compute_f1_extremes handles NaNs internally)
        f1 = compute_f1_extremes(p, t, threshold_percentiles=[95.0, 99.0])
        # mu_HR contribution to this variant's reconstruction
        if v == 'ablation_0x':
            mu_contrib = 0.0
        elif v == 'oracle':
            # alpha varies per sample -> use average effective magnitude
            alphas = probe_result['all_alpha_opt'].view(-1, 1, 1, 1)
            scaled_mu_norm = (alphas * mu_HR_all).norm().item()
            mu_contrib = float(scaled_mu_norm / max(preds.norm().item(), 1e-8))
        else:
            k = {'ref_1x': 1.0, 'scale_2x': 2.0, 'scale_5x': 5.0}[v]
            scaled_mu_norm = (k * mu_HR_all).norm().item()
            mu_contrib = float(scaled_mu_norm / max(preds.norm().item(), 1e-8))

        metrics_by_variant[v] = {
            'rmse': rmse,
            'mae': mae,
            'pearson': pear,
            'f1_p95': f1.get('p95', float('nan')),
            'f1_p99': f1.get('p99', float('nan')),
            'mu_HR_contribution': mu_contrib,
        }

    # Global alpha stats
    alpha = probe_result['all_alpha_opt']
    metrics_by_variant['_alpha_opt'] = {
        'mean': float(alpha.mean().item()),
        'std': float(alpha.std().item()),
        'min': float(alpha.min().item()),
        'max': float(alpha.max().item()),
        'median': float(alpha.median().item()),
        'q25': float(alpha.quantile(0.25).item()),
        'q75': float(alpha.quantile(0.75).item()),
    }
    return metrics_by_variant


METRICS = compute_all_metrics(probe_result)

print('=' * 86)
print(f'{"Variant":14s} {"RMSE":>9s} {"MAE":>9s} {"Pearson":>9s} '
      f'{"F1@p95":>9s} {"F1@p99":>9s} {"mu_contrib":>11s}')
print('-' * 86)
for v in VARIANT_NAMES:
    m = METRICS[v]
    print(f'{v:14s} {m["rmse"]:9.4f} {m["mae"]:9.4f} {m["pearson"]:9.4f} '
          f'{m["f1_p95"]:9.4f} {m["f1_p99"]:9.4f} {m["mu_HR_contribution"]:11.4f}')
print('-' * 86)
print(f'REF  9-node    : RMSE={REF_RMSE:.4f}  Pearson={REF_PEARSON:.4f}  F1@p99={REF_F1P99:.4f}')
print(f'NONCAUSAL v4   :                                F1@p99={NONCAUSAL_F1P99:.4f}')
print('-' * 86)
a = METRICS['_alpha_opt']
print(f'alpha_opt      : mean={a["mean"]:.4f} std={a["std"]:.4f} '
      f'median={a["median"]:.4f} q25={a["q25"]:.4f} q75={a["q75"]:.4f} '
      f'min={a["min"]:.4f} max={a["max"]:.4f}')
"""

# ---------------------------------------------------------------------------
# Cell 8 : Verdict + JSON save
# ---------------------------------------------------------------------------
CELL_VERDICT = r"""# >>> Cell 8 : Automatic verdict + save JSON

def decide_verdict(metrics):
    m_abl    = metrics['ablation_0x']
    m_ref    = metrics['ref_1x']
    m_2x     = metrics['scale_2x']
    m_5x     = metrics['scale_5x']
    m_oracle = metrics['oracle']
    a        = metrics['_alpha_opt']

    notes = []
    verdict = None

    # VERDICT D : mu_HR hurts (must come first -- if 0x beats 1x, all else moot)
    if m_abl['f1_p99'] > m_ref['f1_p99'] + 0.005:
        verdict = 'D'
        notes.append(
            f'F1@p99(0x)={m_abl["f1_p99"]:.4f} > F1@p99(1x)={m_ref["f1_p99"]:.4f} (delta>=+0.5pt) '
            '-> mu_HR HURTS the prediction. mu_HR is biased; consider pruning it from training.'
        )

    # VERDICT C : oracle cannot beat noncausal -> wrong direction
    if verdict is None and m_oracle['f1_p99'] <= NONCAUSAL_F1P99:
        verdict = 'C'
        notes.append(
            f'F1@p99(oracle)={m_oracle["f1_p99"]:.4f} <= noncausal F1@p99={NONCAUSAL_F1P99} '
            '-> even the per-sample optimal scaling of mu_HR cannot reach the noncausal bar. '
            'mu_HR does not point in the right DIRECTION. Deeper Stage-1 issue.'
        )

    # VERDICT A : direction correct + magnitude underestimated
    if verdict is None and a['mean'] > 0.8 and m_oracle['f1_p99'] > NONCAUSAL_F1P99:
        verdict = 'A'
        notes.append(
            f'alpha_opt_mean={a["mean"]:.3f} > 0.8 AND F1@p99(oracle)={m_oracle["f1_p99"]:.4f} '
            f'> {NONCAUSAL_F1P99} -> direction is correct, magnitude is insufficient. '
            'AdaLN warm-start should let Stage 2 internally re-scale mu_HR.'
        )

    # VERDICT B : naive 5x scaling already helps
    if verdict is None and m_5x['f1_p99'] > m_ref['f1_p99'] + 0.005:
        verdict = 'B'
        notes.append(
            f'F1@p99(5x)={m_5x["f1_p99"]:.4f} > F1@p99(1x)={m_ref["f1_p99"]:.4f} '
            '-> naive 5x scaling already helps -> AdaLN warm-start should boost the signal.'
        )

    if verdict is None:
        verdict = 'INCONCLUSIVE'
        notes.append(
            'No criterion was triggered cleanly. Inspect alpha_opt distribution and '
            'oracle metrics manually; consider running more batches.'
        )

    return verdict, notes


verdict, verdict_notes = decide_verdict(METRICS)

print('\n')
print('#' * 86)
print('#' + ' ' * 84 + '#')
print('#' + f'   VERDICT  :  {verdict}'.ljust(84) + '#')
print('#' + ' ' * 84 + '#')
print('#' * 86)
print()
for note in verdict_notes:
    print(f'  >> {note}')
print()

# Summary one-line decision support
print('Decision summary :')
print(f'  F1@p99  0x   = {METRICS["ablation_0x"]["f1_p99"]:.4f}')
print(f'  F1@p99  1x   = {METRICS["ref_1x"     ]["f1_p99"]:.4f}  (= current)')
print(f'  F1@p99  2x   = {METRICS["scale_2x"   ]["f1_p99"]:.4f}')
print(f'  F1@p99  5x   = {METRICS["scale_5x"   ]["f1_p99"]:.4f}')
print(f'  F1@p99 ORACLE= {METRICS["oracle"     ]["f1_p99"]:.4f}')
print(f'  noncausal v4 = {NONCAUSAL_F1P99:.4f}')
print(f'  alpha_opt mean / median = {METRICS["_alpha_opt"]["mean"]:.3f} / '
      f'{METRICS["_alpha_opt"]["median"]:.3f}')

# ---- Save JSON ----
payload = {
    'checkpoint': str(CKPT_PATH),
    'stack_tag': STACK_TAG,
    'eval_config': {
        'n_steps': N_STEPS,
        'k_samples': K_SAMPLES,
        'batch_size': BATCH_SIZE,
        'n_probe_batches': N_PROBE_BATCHES,
        'scheduler': EVAL_SCHEDULER,
        'cfg_scale': EVAL_CFG_SCALE,
    },
    'references': {
        'rmse_current': REF_RMSE,
        'pearson_current': REF_PEARSON,
        'f1_p99_current': REF_F1P99,
        'f1_p99_noncausal_v4': NONCAUSAL_F1P99,
    },
    'metrics_by_variant': METRICS,
    'per_batch_diagnostics': probe_result['per_batch_diagnostics'],
    'n_samples_total': int(probe_result['n_samples_total']),
    'elapsed_s': float(probe_result['elapsed_s']),
    'verdict': verdict,
    'verdict_notes': verdict_notes,
}

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(payload, indent=2, default=str), encoding='utf-8')
print(f'\n[save] {OUT_JSON}')
"""

# ---------------------------------------------------------------------------
# Cell 9 : Visualizations
# ---------------------------------------------------------------------------
CELL_VIZ = r"""# >>> Cell 9 : Visualizations -- bar chart, alpha hist, alpha scatter

import matplotlib
import matplotlib.pyplot as plt

# ---- (1) Bar chart : F1@p99 per variant ----
fig, ax = plt.subplots(figsize=(8.5, 4.5))
xs = np.arange(len(VARIANT_NAMES))
ys = [METRICS[v]['f1_p99'] for v in VARIANT_NAMES]
colors = ['#bbbbbb', '#4477aa', '#88aacc', '#ccaa66', '#22aa44']
ax.bar(xs, ys, color=colors, edgecolor='black', linewidth=0.8)
ax.axhline(REF_F1P99, color='steelblue', linestyle='--', linewidth=1.2,
           label=f'9-node current = {REF_F1P99:.4f}')
ax.axhline(NONCAUSAL_F1P99, color='crimson', linestyle='--', linewidth=1.2,
           label=f'noncausal v4 = {NONCAUSAL_F1P99:.4f}')
for x, y in zip(xs, ys):
    ax.text(x, y + 0.005, f'{y:.4f}', ha='center', fontsize=9)
ax.set_xticks(xs)
ax.set_xticklabels(VARIANT_NAMES, rotation=15)
ax.set_ylabel('F1 @ p99 (extremes)')
ax.set_title(f'mu_HR probe -- F1@p99 by variant ({STACK_TAG}, verdict={verdict})')
ax.set_ylim(0, max(max(ys), NONCAUSAL_F1P99) * 1.15)
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_PNG_BAR, dpi=110)
plt.show()
print(f'[save] {OUT_PNG_BAR}')

# ---- (2) Histogram : alpha_opt distribution ----
alpha = probe_result['all_alpha_opt'].numpy()
fig, ax = plt.subplots(figsize=(8.0, 4.0))
ax.hist(alpha, bins=40, color='#4477aa', edgecolor='black', alpha=0.85)
ax.axvline(1.0, color='black', linestyle='-',  linewidth=1.2, label='alpha = 1 (current 1x)')
ax.axvline(2.0, color='#888888', linestyle=':',  linewidth=1.0, label='alpha = 2')
ax.axvline(5.0, color='#888888', linestyle='-.', linewidth=1.0, label='alpha = 5')
ax.axvline(alpha.mean(), color='crimson', linestyle='--', linewidth=1.5,
           label=f'mean = {alpha.mean():.3f}')
ax.axvline(np.median(alpha), color='darkgreen', linestyle='--', linewidth=1.5,
           label=f'median = {np.median(alpha):.3f}')
ax.set_xlabel('alpha_opt (per-sample oracle scaling)')
ax.set_ylabel('count')
ax.set_title(f'mu_HR probe -- alpha_opt distribution (N={len(alpha)})')
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_PNG_ALPHA_HIST, dpi=110)
plt.show()
print(f'[save] {OUT_PNG_ALPHA_HIST}')

# ---- (3) Scatter : alpha_opt vs mu_HR contribution per batch ----
diags = probe_result['per_batch_diagnostics']
xs = np.array([d['mu_contribution'] for d in diags])
ys = np.array([d['alpha_opt_mean'] for d in diags])
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sc = ax.scatter(xs, ys, c=ys, cmap='viridis', s=70, edgecolor='black')
ax.axhline(1.0, color='black', linestyle='-',  linewidth=1.0, alpha=0.7)
ax.axhline(METRICS['_alpha_opt']['mean'], color='crimson', linestyle='--',
           linewidth=1.0, alpha=0.8, label=f'mean alpha_opt = {METRICS["_alpha_opt"]["mean"]:.3f}')
ax.set_xlabel('mu_HR contribution to pred_ref (||mu_HR|| / ||pred_ref||)')
ax.set_ylabel('alpha_opt per-batch mean')
ax.set_title('mu_HR probe -- alpha_opt vs mu_HR contribution (per batch)')
ax.legend()
ax.grid(alpha=0.3)
plt.colorbar(sc, ax=ax, label='alpha_opt')
fig.tight_layout()
fig.savefig(OUT_PNG_ALPHA_SCATTER, dpi=110)
plt.show()
print(f'[save] {OUT_PNG_ALPHA_SCATTER}')

print()
print('=' * 86)
print(f'PROBE COMPLETE -- verdict = {verdict}')
print(f'JSON       : {OUT_JSON}')
print(f'Bar chart  : {OUT_PNG_BAR}')
print(f'Alpha hist : {OUT_PNG_ALPHA_HIST}')
print(f'Scatter    : {OUT_PNG_ALPHA_SCATTER}')
print('=' * 86)
"""


# ---------------------------------------------------------------------------
# Assemble notebook
# ---------------------------------------------------------------------------
nb = {
    "cells": [
        md_cell(CELL_TITLE),
        code_cell(CELL_BOOTSTRAP),
        code_cell(CELL_IMPORTS),
        code_cell(CELL_STACK),
        code_cell(CELL_DATA),
        code_cell(CELL_BUILD),
        code_cell(CELL_PROBE),
        code_cell(CELL_METRICS),
        code_cell(CELL_VERDICT),
        code_cell(CELL_VIZ),
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {NB_PATH}  ({NB_PATH.stat().st_size} bytes, {len(nb['cells'])} cells)")
