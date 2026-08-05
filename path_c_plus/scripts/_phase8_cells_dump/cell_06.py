# === Cell 6 : Pipeline + dataloaders avec features augmentées ===
from torch.utils.data import DataLoader as _DataLoader, IterableDataset
from st_cdgm.data.pipeline import NetCDFDataPipeline
from st_cdgm.models.graph_builder import HeteroGraphBuilder
from path_c_plus.scripts.option_c_helpers import PATHCPLUS_HYPERPARAM_OVERRIDES

# Load config
CONFIG = OmegaConf.load('config/training_config.yaml')
_corrdiff = OmegaConf.load('config/training_config_corrdiff_normal.yaml')
CONFIG = OmegaConf.merge(CONFIG, _corrdiff)
CONFIG.training.batch_size  = 1
CONFIG.training.use_amp     = True
CONFIG.training.num_workers = 0
ts_cfg = CONFIG.two_stage
ts_cfg.stage1['lambda_dag_prior'] = LAMBDA_DAG_PRIOR
ts_cfg.stage1['g_phys_alpha']     = G_PHYS_ALPHA
OmegaConf.set_struct(CONFIG, False)

# Add augmented LR variables (4 new : w_700, theta_e_850, theta_e_500, mucape_proxy)
ORIGINAL_LR_VARS  = list(CONFIG.data.lr_variables)
AUGMENTED_LR_VARS = ORIGINAL_LR_VARS + ['w_700', 'theta_e_850', 'theta_e_500', 'mucape_proxy']
CONFIG.data.lr_variables = AUGMENTED_LR_VARS
print(f'[Cell 6] LR variables : {len(AUGMENTED_LR_VARS)} (original 15 + 4 augmented)')
print(f'[Cell 6] new variables : w_700, theta_e_850, theta_e_500, mucape_proxy')

# Add 9-node metapaths (existing + new ones for augmented features)
for _m in [
    {'name': 'Q850', 'src': 'Q850', 'relation': 'causes', 'target': 'GP850', 'pool': 'mean'},
    {'name': 'W500', 'src': 'W500', 'relation': 'causes', 'target': 'GP500', 'pool': 'mean'},
    {'name': 'IVT',  'src': 'IVT',  'relation': 'causes', 'target': 'GP850', 'pool': 'mean'},
]:
    if _m['name'] not in {mm.name for mm in CONFIG.encoder.metapaths}:
        CONFIG.encoder.metapaths.append(OmegaConf.create(_m))

# Build pipeline using AUGMENTED LR file (with w_700, theta_e, MUCAPE)
SEQ_LEN = int(CONFIG.data.seq_len)
pipeline = NetCDFDataPipeline(
    lr_path=str(AUGMENTED_LR_PATH), hr_path=str(HR_RAW_PATH),
    static_path=str(STATIC_PATH) if STATIC_PATH.exists() else None,
    seq_len=SEQ_LEN, baseline_strategy=str(CONFIG.data.baseline_strategy),
    baseline_factor=int(CONFIG.data.baseline_factor),
    normalize=bool(CONFIG.data.normalize),
    nan_fill_strategy=str(CONFIG.data.nan_fill_strategy),
    precipitation_delta=float(CONFIG.data.precipitation_delta),
    lr_variables=AUGMENTED_LR_VARS,
    hr_variables=list(CONFIG.data.hr_variables),
    static_variables=list(CONFIG.data.static_variables) if CONFIG.data.get('static_variables') else [],
    means_path=str(DATA_ROOT / 'train' / 'means_ACCESS-CM2.nc') if (DATA_ROOT / 'train' / 'means_ACCESS-CM2.nc').exists() else None,
    stds_path=str(DATA_ROOT / 'train' / 'stds_ACCESS-CM2.nc') if (DATA_ROOT / 'train' / 'stds_ACCESS-CM2.nc').exists() else None,
    train_start_date=K9_DATES['train'][0], train_end_date=K9_DATES['train'][1],
    val_start_date=K9_DATES['val'][0],     val_end_date=K9_DATES['val'][1],
    test_start_date=K9_DATES['test'][0],   test_end_date=K9_DATES['test'][1],
    temporal_holdout_start_date=K9_DATES['holdout'][0],
    temporal_holdout_end_date=K9_DATES['holdout'][1],
)

train_dataset = pipeline.build_sequence_dataset(split='train', seq_len=SEQ_LEN, stride=int(CONFIG.data.stride), as_torch=True)
val_dataset   = pipeline.build_sequence_dataset(split='val',   seq_len=SEQ_LEN, stride=int(CONFIG.data.stride), as_torch=True)
test_dataset  = pipeline.build_sequence_dataset(split='test',  seq_len=SEQ_LEN, stride=int(CONFIG.data.stride), as_torch=True)

train_dataloader = _DataLoader(train_dataset, batch_size=1, num_workers=0, pin_memory=True,
                                collate_fn=lambda x: x, shuffle=False)
val_dataloader   = _DataLoader(val_dataset,   batch_size=1, num_workers=0, pin_memory=True,
                                collate_fn=lambda x: x, shuffle=False)

# Graph builder (9-node)
lr_shape = tuple(CONFIG.graph.lr_shape); hr_shape = tuple(CONFIG.graph.hr_shape)
builder = HeteroGraphBuilder(lr_shape=lr_shape, hr_shape=hr_shape,
                              static_dataset=pipeline.get_static_dataset(),
                              include_mid_layer=CONFIG.graph.include_mid_layer,
                              extended_9node=True)
H_HR, W_HR = int(hr_shape[0]), int(hr_shape[1])

# Convert sample to batch (mirrors phase6 pattern)
_LR_VARS = AUGMENTED_LR_VARS
_VI = {v: i for i, v in enumerate(_LR_VARS)}
_Q_IDX = [_VI[v] for v in ('q_850','q_500','q_250') if v in _VI]
_W_IDX = [_VI[v] for v in ('w_850','w_500','w_250') if v in _VI]
_IVT_LEVELS = [lev for lev in ('850','500','250')
               if f'q_{lev}' in _VI and f'u_{lev}' in _VI and f'v_{lev}' in _VI]

def _compute_ivt_nodes(lr0):
    acc = None
    for lev in _IVT_LEVELS:
        q = lr0[:, [_VI[f'q_{lev}']]]; u = lr0[:, [_VI[f'u_{lev}']]]; v = lr0[:, [_VI[f'v_{lev}']]]
        term = q * torch.sqrt(u*u + v*v + 1e-12)
        acc = term if acc is None else acc + term
    if acc is None: acc = lr0[:, 0:1] * 0.0
    return acc / (len(_IVT_LEVELS) + 1e-8)

def _ensure_2d(t): return t.unsqueeze(-1) if t.dim() == 1 else t

def convert_sample_to_batch(sample, builder, device):
    lr_seq = sample['lr']; seq_len = lr_seq.shape[0]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    lr0 = lr_nodes_steps[0]
    _ivt = _compute_ivt_nodes(lr0)
    dyn = {}
    for nt in builder.dynamic_node_types:
        if nt == 'Q850':   dyn[nt] = _ensure_2d(lr0[:, _Q_IDX] if _Q_IDX else lr0)
        elif nt == 'W500': dyn[nt] = _ensure_2d(lr0[:, _W_IDX] if _W_IDX else lr0)
        elif nt == 'IVT':  dyn[nt] = _ensure_2d(_ivt)
        else:              dyn[nt] = _ensure_2d(lr0)
    hetero = builder.prepare_step_data(dyn).to(device)
    return {'lr': lr_tensor, 'lr_grid': lr_seq, 'residual': sample['residual'],
            'baseline': sample.get('baseline'), 'hetero': hetero, 'time': sample.get('time')}

_probe = next(iter(train_dataset))
C_LR = _probe['lr'].shape[1]
print(f'[Cell 6] LR channels detected = {C_LR} (expected {len(AUGMENTED_LR_VARS)})')
print(f'[Cell 6] HR shape = ({H_HR}, {W_HR})')
print(f'[Cell 6] dynamic node types = {builder.dynamic_node_types}')
print(f'[Cell 6] Pipeline + dataloaders ready')
