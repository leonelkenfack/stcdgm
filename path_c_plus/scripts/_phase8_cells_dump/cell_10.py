# === Cell 10 : Stage 2 build (UNet 50M EDM + Dispersive hook + multi-EMA) ===
import copy
from st_cdgm.models import CausalDiffusionDecoder
from st_cdgm.models.edm_preconditioner import EDMConfig
from omegaconf import OmegaConf as _OC

# Probe HR channels
_probe = next(iter(val_dataset))
hr_channels = int(_probe['residual'].shape[1])

UNET_KW = _OC.to_container(CONFIG.diffusion.unet_kwargs, resolve=True)
for _k in ('down_block_types', 'up_block_types'):
    if _k in UNET_KW and isinstance(UNET_KW[_k], list):
        UNET_KW[_k] = tuple(UNET_KW[_k])
UNET_KW['projection_class_embeddings_input_dim'] = num_vars * int(CONFIG.diffusion.conditioning_dim)

edm_cfg = EDMConfig.from_yaml_dict(CONFIG.diffusion.get('edm', {}))

def _build_stage2_decoder():
    d = CausalDiffusionDecoder(
        in_channels=hr_channels,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        height=int(CONFIG.diffusion.height), width=int(CONFIG.diffusion.width),
        unet_kwargs=UNET_KW,
        scheduler_type=str(CONFIG.diffusion.scheduler_type),
        use_gradient_checkpointing=True,
        conv_padding_mode=str(CONFIG.diffusion.get('conv_padding_mode', 'zeros')),
        anti_checkerboard=bool(CONFIG.diffusion.get('anti_checkerboard', False)),
        edm_config=edm_cfg, causal_concat=True,
    ).to(DEVICE)
    d.edm_config.sigma_data = float(SIGMA_DATA_NEW)
    return d

# Live decoder (the one being trained)
diff_decoder = _build_stage2_decoder()
n_params = sum(p.numel() for p in diff_decoder.parameters())
print(f'[Cell 10] Stage 2 decoder built : {n_params:,} params (sigma_data={SIGMA_DATA_NEW})')

# Multi-EMA models (Expert ML : 3 EMA decays + post-hoc sweep at eval)
ema_decoders = []
for decay in EMA_DECAYS:
    ema = _build_stage2_decoder()
    ema.load_state_dict(diff_decoder.state_dict())
    ema.eval()
    for p in ema.parameters(): p.requires_grad_(False)
    ema._ema_decay = decay
    ema._ema_step_counter = 0
    ema_decoders.append(ema)
print(f'[Cell 10] Multi-EMA decoders built : decays = {EMA_DECAYS}')

# ---- Dispersive Loss hook on UNet mid-block (Wang & He 2025) ----
# Captures mid-block features for the dispersive penalty term.
dispersive_features = {'mid': None}

def _dispersive_hook(module, inputs, output):
    # Module output is a Tensor (mid-block residual). Store a detached-noncausal handle.
    if isinstance(output, tuple):
        feat = output[0]
    else:
        feat = output
    dispersive_features['mid'] = feat

# Attach hook to UNet mid_block (diffusers UNet2DConditionModel exposes .mid_block)
try:
    _mid_block = diff_decoder.unet.mid_block
    _disp_handle = _mid_block.register_forward_hook(_dispersive_hook)
    print(f'[Cell 10] Dispersive Loss hook registered on mid_block (lambda={DISPERSIVE_LAMBDA}, tau={DISPERSIVE_TAU})')
except AttributeError:
    print(f'[Cell 10] WARNING : unet.mid_block not found, Dispersive Loss disabled')
    _disp_handle = None

def loss_dispersive(features, tau=DISPERSIVE_TAU):
    """Dispersive Loss (Wang & He 2025, arXiv 2506.09027).
    Encourages batch-wise feature diversity via contrastive repulsion.
    L = -log(E[K(z_i, z_j) / tau])  where K is a Gaussian kernel.
    """
    if features is None or features.dim() < 2:
        return torch.tensor(0.0, device=DEVICE)
    B = features.shape[0]
    if B < 2:
        return torch.tensor(0.0, device=DEVICE)
    flat = features.flatten(start_dim=1)
    # Pairwise distance (cosine-like via normalized dot product)
    flat = flat / (flat.norm(dim=1, keepdim=True) + 1e-8)
    sim = (flat @ flat.T) / tau   # [B, B]
    # Off-diagonal repulsion : we want low pairwise similarity
    mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
    off_diag = sim[mask]
    # -log(softmax-like form) : encourages off-diag to be small
    return torch.logsumexp(off_diag.view(B, B-1), dim=1).mean()

# ---- Min-SNR-gamma weighting (Hang ICCV 2023) ----
def _min_snr_weight(sigma, sigma_data, gamma):
    """w_min_snr(sigma) = min(SNR, gamma) / SNR  where SNR = sigma_d^2 / sigma^2.
    Final weight = w_min_snr * lambda_karras (combined per Math expert)."""
    snr = (sigma_data / sigma.clamp_min(1e-8)) ** 2
    return (torch.clamp(snr, max=gamma) / snr).view(-1, 1, 1, 1)

# ---- Tail weight (4, 12) on extreme pixels ----
def _tail_weight(hr_log_recon, tau95_mm=15.0, tau99_mm=35.0, w95=TAIL_WEIGHT_P95, w99=TAIL_WEIGHT_P99):
    import math
    tau95 = math.log1p(tau95_mm)
    tau99 = math.log1p(tau99_mm)
    return (1.0 + (w95 - 1.0) * (hr_log_recon > tau95).float()
                + (w99 - w95) * (hr_log_recon > tau99).float())

print(f'[Cell 10] Min-SNR weighting ready (gamma={MIN_SNR_GAMMA})')
print(f'[Cell 10] Tail weight ready ({TAIL_WEIGHT_P95}, {TAIL_WEIGHT_P99}) on pixels >15/>35 mm/day')
