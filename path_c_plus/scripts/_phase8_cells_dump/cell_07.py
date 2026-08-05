# === Cell 7 : Stage 1 build from-scratch + physical losses corrigées ===
from st_cdgm.models.dual_path_stage1 import DualPathPredictor
from st_cdgm.training.stage1_paths import batch_lr_grid_last
from st_cdgm.models.intelligible_encoder import IntelligibleVariableEncoder, IntelligibleVariableConfig
from st_cdgm.models.causal_rcn import RCNCell, RCNSequenceRunner
from st_cdgm.models.regression_head import GraphToGridDecoder

# Build modules from-scratch (NO checkpoint load).
_metapath_configs = [
    IntelligibleVariableConfig(name=m.name, meta_path=(m.src, m.relation, m.target), pool='mean')
    for m in CONFIG.encoder.metapaths
]
encoder = IntelligibleVariableEncoder(
    configs=_metapath_configs,
    hidden_dim=int(CONFIG.encoder.hidden_dim),
    conditioning_dim=int(CONFIG.encoder.conditioning_dim),
).to(DEVICE)
num_vars = len(_metapath_configs)

_lr_nodes = builder.lr_grid_to_nodes(_probe['lr'][0])
rcn_driver_dim = _lr_nodes.shape[-1]
rcn_cell = RCNCell(
    num_vars=num_vars, hidden_dim=int(CONFIG.rcn.hidden_dim),
    driver_dim=rcn_driver_dim, reconstruction_dim=rcn_driver_dim,
    dropout=float(CONFIG.rcn.dropout),
).to(DEVICE)
rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get('detach_interval'))

rh_cfg = CONFIG.two_stage.regression_head
regression_head = GraphToGridDecoder(
    d_model=int(rh_cfg.d_model), hr_h=H_HR, hr_w=W_HR,
    intermediate_h=int(rh_cfg.intermediate_h), intermediate_w=int(rh_cfg.intermediate_w),
    n_heads=int(rh_cfg.n_heads), refine_channels=int(rh_cfg.refine_channels),
    output_channels=1,
).to(DEVICE)

dual_path = DualPathPredictor(
    in_channels=C_LR, base_ch=48, hr_h=H_HR, hr_w=W_HR,
    gate_max_mean=0.40, path_b_kind='unet',
    path_b_unet_channels=(32, 64, 128),
    path_b_unet_lr_shape=(23, 26),
).to(DEVICE)

# Learnable alpha for Stage 2 reconstruction (HR = baseline + alpha * mu_HR + delta)
# Initialised at sigmoid(0) = 0.5 (Math expert : prevents alpha->0 collapse)
alpha_logit = torch.nn.Parameter(torch.tensor(0.0, device=DEVICE))

print(f'[Cell 7] Stage 1 modules built from-scratch')
print(f'[Cell 7]   encoder params  : {sum(p.numel() for p in encoder.parameters()):,}')
print(f'[Cell 7]   rcn_cell params : {sum(p.numel() for p in rcn_cell.parameters()):,}')
print(f'[Cell 7]   reg_head params : {sum(p.numel() for p in regression_head.parameters()):,}')
print(f'[Cell 7]   dual_path params: {sum(p.numel() for p in dual_path.parameters()):,}')
print(f'[Cell 7]   alpha init      : {torch.sigmoid(alpha_logit).item():.4f}')

# ===========================================================================
# Physical losses (differentiable, in mm/day space after expm1 — Math+Climat fix)
# Pipeline applies log1p(x + PRECIPITATION_DELTA), so to recover mm/day :
#   x_mm = expm1(x_log1p) - PRECIPITATION_DELTA
# PRECIPITATION_DELTA = 0.01 (from pipeline.py)
# ===========================================================================
PRECIP_DELTA = 0.01

def _to_mm_day(x_log1p):
    """Convert log1p(pr + delta) back to mm/day, clamped to [0, 500] (Climat)."""
    return (torch.expm1(x_log1p) - PRECIP_DELTA).clamp(min=0.0, max=500.0)

def loss_R10mm(pred_log1p, target_log1p, valid_mask=None, steepness=5.0):
    """Differentiable approx of R10mm (count of days >= 10 mm/day).
    Steepness=5 (Climat raised from 2). Operates in mm/day space."""
    pred_mm = _to_mm_day(pred_log1p)
    target_mm = _to_mm_day(target_log1p)
    # Sigmoid approx of 1[x >= 10]
    pred_soft   = torch.sigmoid(steepness * (pred_mm - 10.0))
    target_soft = torch.sigmoid(steepness * (target_mm - 10.0))
    if valid_mask is not None:
        pred_soft   = pred_soft   * valid_mask
        target_soft = target_soft * valid_mask
    # Sum over time dimension if present, else over batch
    pred_count   = pred_soft.sum(dim=tuple(range(1, pred_soft.dim())))    # per-batch count
    target_count = target_soft.sum(dim=tuple(range(1, target_soft.dim())))
    return ((pred_count - target_count) ** 2).mean()

def loss_Rx1day(pred_log1p, target_log1p, T=5.0):
    """Differentiable approx of annual max (Rx1day).
    Use LogSumExp_T / T (numerically stable approx of max)."""
    pred_mm   = _to_mm_day(pred_log1p)
    target_mm = _to_mm_day(target_log1p)
    # LSE_T(x) = (1/T) * log(sum exp(T*x))  -> approaches max(x) as T -> inf
    flat_pred   = pred_mm.flatten(start_dim=1)
    flat_target = target_mm.flatten(start_dim=1)
    lse_pred   = torch.logsumexp(T * flat_pred,   dim=1) / T
    lse_target = torch.logsumexp(T * flat_target, dim=1) / T
    return ((lse_pred - lse_target) ** 2).mean()

def loss_CDD(pred_log1p, target_log1p, steepness=5.0):
    """Differentiable approx of CDD (count of dry days, x < 1 mm/day)."""
    pred_mm   = _to_mm_day(pred_log1p)
    target_mm = _to_mm_day(target_log1p)
    # Sigmoid approx of 1[x < 1]
    pred_soft   = torch.sigmoid(steepness * (1.0 - pred_mm))
    target_soft = torch.sigmoid(steepness * (1.0 - target_mm))
    pred_count   = pred_soft.sum(dim=tuple(range(1, pred_soft.dim())))
    target_count = target_soft.sum(dim=tuple(range(1, target_soft.dim())))
    return ((pred_count - target_count) ** 2).mean()

def loss_Clausius_Clapeyron(mu_HR_pred, T_850_batch, target_log1p, mode='extreme'):
    """Clausius-Clapeyron loss : (d mu / d T) - rate * mu = 0.
    Math : autograd, NOT finite differences. Adimensionned.
    Climat : mode='extreme' (rate=0.07 for P>P95), mode='mean' (rate=0.05)."""
    if not T_850_batch.requires_grad:
        T_850_batch = T_850_batch.detach().requires_grad_(True)
    # We need mu_HR_pred to depend on T_850 for autograd to work.
    # Caller is responsible for this dependency. Here we just compute the loss.
    rate = 0.07 if mode == 'extreme' else 0.05
    try:
        grad = torch.autograd.grad(
            outputs=mu_HR_pred.sum(),
            inputs=T_850_batch,
            create_graph=True,
            retain_graph=True,
        )[0]
    except RuntimeError:
        # If T_850 was not part of the graph, return zero loss (safe fallback)
        return torch.tensor(0.0, device=mu_HR_pred.device, requires_grad=False)
    # Adimensionner par sigma de chaque variable (Math)
    sigma_T = T_850_batch.std().clamp_min(1e-6)
    sigma_mu = mu_HR_pred.std().clamp_min(1e-6)
    grad_adim = grad * sigma_T / sigma_mu
    target_grad = rate * mu_HR_pred / sigma_mu
    return ((grad_adim - target_grad) ** 2).mean()

# Alpha regularization (Math : prevent alpha -> 0 collapse, cf. MC2RD failure)
def loss_alpha_reg(alpha):
    target_loss = LAMBDA_ALPHA_REG * (alpha - ALPHA_TARGET) ** 2
    floor_loss  = BETA_ALPHA_FLOOR * (torch.relu(ALPHA_FLOOR - alpha) ** 2)
    return target_loss + floor_loss

print('[Cell 7] Physical losses defined :')
print('  loss_R10mm        (steepness=5, in mm/day space)')
print('  loss_Rx1day       (LogSumExp_T=5)')
print('  loss_CDD          (steepness=5)')
print('  loss_Clausius_Clapeyron (autograd, mode=extreme)')
print('  loss_alpha_reg    (lambda=0.1, beta_floor=1.0 for alpha < 0.3)')
