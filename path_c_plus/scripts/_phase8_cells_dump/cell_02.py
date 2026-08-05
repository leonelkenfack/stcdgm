# === Cell 2 : Constants ===
DRIVE_ROOT = Path('/content/drive/MyDrive/climate_data')
DATA_ROOT  = DRIVE_ROOT / 'data'
HR_RAW_PATH = DATA_ROOT / 'train' / 'pr_ACCESS-CM2_hist.nc'
LR_RAW_PATH = DATA_ROOT / 'train' / 'predictor_ACCESS-CM2_hist.nc'
STATIC_PATH = DATA_ROOT / 'static_predictors' / 'ERA5_eval_ccam_12km.198110_NZ_Invariant.nc'

# Phase 8 output dir
OUT_DIR = DRIVE_ROOT / 'oracle_9node' / 'phase8_from_scratch'
OUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_STAGE1_LAST  = OUT_DIR / 'stage1_last.pth'
CKPT_STAGE1_BEST  = OUT_DIR / 'stage1_best.pth'
STAGE1_CACHE_PATH = OUT_DIR / 'stage1_cache_mu_total.pt'
CKPT_STAGE2_LAST  = OUT_DIR / 'stage2_last.pth'
CKPT_STAGE2_BEST  = OUT_DIR / 'stage2_best.pth'
LAND_MASK_PATH    = OUT_DIR / 'land_mask_nz.npy'
CLIM_PATH         = OUT_DIR / 'clim_train_p95_p99.npz'
AUGMENTED_LR_PATH = OUT_DIR / 'lr_augmented_features.nc'  # w_700, theta_e, MUCAPE
TRAINING_HISTORY  = OUT_DIR / 'training_history.json'
FINAL_RESULTS     = OUT_DIR / 'phase8_final_results.json'

# Reference checkpoints to compare against (existing models)
REF_CKPT_NONCAUSAL = DRIVE_ROOT / 'ckpt_noncausal'
REF_CKPT_V5_CAUSAL = DRIVE_ROOT / 'ckpt_v2_corrdiff_normal'

# ============== SMOKE MODE FLAG (Expert team validation step) ==============
# When True : 2-3 epochs, N=2 batches, K=4 samples, N_STEPS=8 -> ~15 min total
# When False : full protocol -> ~45-55h
SMOKE_MODE = True

# Reproducibility
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# K9 temporal split (used by all phases)
K9_DATES = {
    'train':   ['1980-01-01', '2009-12-31'],   # 30 years
    'val':     ['2010-01-01', '2011-12-31'],   # 2 years
    'test':    ['2012-01-01', '2013-12-31'],   # 2 years -> 730 days for climate indices
    'holdout': ['2014-01-01', '2014-12-31'],   # 1 year (out-of-distribution warm year)
}

# Stage 1 hyperparams (Path C+ Option C proven on seed_42)
STAGE1_EPOCHS    = 15
STAGE1_LR        = 1e-3
LAMBDA_DAG_PRIOR = 0.40
LAMBDA_L1_START  = 0.04
LAMBDA_L1_END    = 0.005
G_PHYS_ALPHA     = 0.25
# Physical loss weights (warmup applied after epoch 10)
LAMBDA_R10MM   = 0.20
LAMBDA_RX1DAY  = 0.15
LAMBDA_CDD     = 0.10
LAMBDA_CC      = 0.05
PHYS_LOSS_WARMUP_EPOCH = 10

# Stage 2 hyperparams
STAGE2_EPOCHS         = 200
STAGE2_LR             = 2e-4
STAGE2_BATCH_SIZE     = 64
STAGE2_WEIGHT_DECAY   = 1e-4
STAGE2_GRADIENT_CLIP  = 1.0
MIN_SNR_GAMMA         = 5.0
TAIL_WEIGHT_P95       = 4.0    # (Climate ML : 8 -> 4 less aggressive)
TAIL_WEIGHT_P99       = 12.0   # (Climate ML : 25 -> 12)
DISPERSIVE_LAMBDA     = 0.25   # (Expert ML : raised from 0.05 sous-dosé to paper's recommended range)
DISPERSIVE_TAU        = 0.5    # Kernel temperature
COND_DROPOUT_P        = 0.13   # CFG compatibility (CorrDiff Nature CEE 2025)
EMA_DECAYS            = [0.999, 0.9995, 0.9999]   # Multi-EMA EDM2 Karras 2024
ALPHA_TARGET          = 0.5   # Regularization target for learned alpha
ALPHA_FLOOR           = 0.3   # Penalty if alpha < 0.3 (prevent collapse)
LAMBDA_ALPHA_REG      = 0.1
BETA_ALPHA_FLOOR      = 1.0
SIGMA_DATA_NEW        = 0.193  # Phase 6 dualpath recalibrated

# BS30 eval protocol (publication-ready)
N_TEST_BATCHES = 64    # Math expert : N=64 -> CI +/-0.014
K_SAMPLES      = 128   # >> CorrDiff Mardani (32)
N_STEPS_DIFF   = 32    # dpm_solver++ converges at 32 NFE
WET_DAY_THRESHOLD_MM = 1.0   # ETCCDI standard

# Sampling
SAMPLER_SCHEDULER       = 'dpm_solver++'
CFG_SCALE               = 1.0
LIMITED_GUIDANCE_SIGMA_MIN = 0.05   # Kynkäänniemi NeurIPS 2024 (calibrated for sigma_data=0.1)
LIMITED_GUIDANCE_SIGMA_MAX = 1.0

# Bootstrap + statistical tests
BOOTSTRAP_N_RESAMPLES = 1000   # Math : n_boot=1000
PAIRED_PERMUTATION_N = 10000   # Math : n_perm=10000 for p<=0.001
BOOTSTRAP_METHOD     = 'BCa'   # Bias-corrected accelerated (Efron 1987)

print(f'[Cell 2] DEVICE = {DEVICE}')
print(f'[Cell 2] DRIVE_ROOT = {DRIVE_ROOT}')
print(f'[Cell 2] OUT_DIR = {OUT_DIR}')
print(f'[Cell 2] Stage 1 : {STAGE1_EPOCHS} epochs, LR {STAGE1_LR}')
print(f'[Cell 2] Stage 2 : {STAGE2_EPOCHS} epochs, LR {STAGE2_LR}, batch {STAGE2_BATCH_SIZE}')
print(f'[Cell 2] Min-SNR γ = {MIN_SNR_GAMMA}, tail_weight ({TAIL_WEIGHT_P95}, {TAIL_WEIGHT_P99})')
print(f'[Cell 2] Multi-EMA decays = {EMA_DECAYS}')
print(f'[Cell 2] BS30 eval : N={N_TEST_BATCHES} batches x K={K_SAMPLES} samples x {N_STEPS_DIFF} steps')

# Pre-registration record (Expert Recherche)
PRE_REG_RECORD = {
    'phase': 'phase8_from_scratch',
    'git_sha': _git_sha,
    'git_branch': GIT_BRANCH,
    'seed': SEED,
    'k9_dates': K9_DATES,
    'stage1_hyperparams': {
        'epochs': STAGE1_EPOCHS, 'lr': STAGE1_LR,
        'lambda_dag_prior': LAMBDA_DAG_PRIOR,
        'lambda_l1_start': LAMBDA_L1_START, 'lambda_l1_end': LAMBDA_L1_END,
        'g_phys_alpha': G_PHYS_ALPHA,
        'physical_loss_weights': {
            'R10mm': LAMBDA_R10MM, 'Rx1day': LAMBDA_RX1DAY,
            'CDD': LAMBDA_CDD, 'CC': LAMBDA_CC,
        },
        'phys_warmup_epoch': PHYS_LOSS_WARMUP_EPOCH,
    },
    'stage2_hyperparams': {
        'epochs': STAGE2_EPOCHS, 'lr': STAGE2_LR,
        'batch_size': STAGE2_BATCH_SIZE, 'weight_decay': STAGE2_WEIGHT_DECAY,
        'min_snr_gamma': MIN_SNR_GAMMA,
        'tail_weight': [TAIL_WEIGHT_P95, TAIL_WEIGHT_P99],
        'dispersive_lambda': DISPERSIVE_LAMBDA,
        'cond_dropout_p': COND_DROPOUT_P,
        'ema_decays': EMA_DECAYS,
        'alpha_reg': {'target': ALPHA_TARGET, 'floor': ALPHA_FLOOR,
                       'lambda': LAMBDA_ALPHA_REG, 'beta_floor': BETA_ALPHA_FLOOR},
        'sigma_data': SIGMA_DATA_NEW,
    },
    'eval_protocol': {
        'n_batches': N_TEST_BATCHES, 'k_samples': K_SAMPLES,
        'n_steps_diff': N_STEPS_DIFF,
        'sampler': SAMPLER_SCHEDULER, 'cfg_scale': CFG_SCALE,
        'limited_guidance': [LIMITED_GUIDANCE_SIGMA_MIN, LIMITED_GUIDANCE_SIGMA_MAX],
        'bootstrap_n': BOOTSTRAP_N_RESAMPLES,
        'paired_permutation_n': PAIRED_PERMUTATION_N,
    },
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}
if SMOKE_MODE:
    print('[Cell 2] *** SMOKE_MODE active : overriding hyperparams for fast smoke ***')
    STAGE1_EPOCHS = 2
    STAGE2_EPOCHS = 2
    N_TEST_BATCHES = 2
    K_SAMPLES = 4
    N_STEPS_DIFF = 8
    BOOTSTRAP_N_RESAMPLES = 50
    PAIRED_PERMUTATION_N = 500
    print(f'[Cell 2] SMOKE : Stage1={STAGE1_EPOCHS}ep  Stage2={STAGE2_EPOCHS}ep  '
          f'N={N_TEST_BATCHES} K={K_SAMPLES} steps={N_STEPS_DIFF}')

print(f'[Cell 2] Pre-registration record created (SHA {_git_sha[:8]})')
print(f'[Cell 2] SMOKE_MODE = {SMOKE_MODE}')