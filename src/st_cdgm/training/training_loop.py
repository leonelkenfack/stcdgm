"""
Module 6 – Boucle d'entraînement pour l'architecture ST-CDGM.

Ce module assemble les pertes (diffusion, reconstruction, NO TEARS) et fournit
une routine d'entraînement par epoch qui enchaîne les modules précédents :
encodeur de variables intelligibles, RCN et décodeur de diffusion.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from ..models.causal_rcn import RCNSequenceRunner
from ..models.diffusion_decoder import CausalDiffusionDecoder
from ..models.intelligible_encoder import IntelligibleVariableEncoder


def _train_autocast(amp_mode: str):
    """
    Mixed precision context for train_epoch: CUDA FP16 (with GradScaler) or CPU BF16 (no scaler).
    amp_mode: "none" | "cuda_fp16" | "cpu_bf16"
    """
    if amp_mode == "cuda_fp16":
        return torch.cuda.amp.autocast()
    if amp_mode == "cpu_bf16":
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def _eager_core(m):
    """
    Strip both DDP and torch.compile (``_orig_mod``) wrappers and return the
    eager-mode submodule. Use whenever a forward must run *outside* the main
    ``compute_loss`` path with tensors built locally — those tensors can have
    different strides than what compile traced, which fires the TensorMatch
    guard and is silently swallowed by the surrounding ``except Exception``.
    Safe when neither wrapper is present (returns ``m`` unchanged).
    """
    if m is None:
        return None
    if isinstance(m, DDP):
        m = m.module
    return getattr(m, "_orig_mod", m)


def resolve_train_amp_mode(device: torch.device, use_amp: bool) -> str:
    """Même logique que le début de ``train_epoch`` (pour métrique RAPSD fin d’époque)."""
    if not use_amp:
        return "none"
    if device.type == "cuda" and torch.cuda.is_available():
        return "cuda_fp16"
    if device.type == "cpu":
        bf16_ok = getattr(torch.cpu, "is_bf16_supported", None)
        if bf16_ok is not None and bf16_ok():
            return "cpu_bf16"
    return "none"


def loss_reconstruction(
    pred: Optional[Tensor],
    target: Optional[Tensor],
    loss_type: str = "mse",  # Phase D4: Loss type ("mse", "cosine", "mse+cosine")
) -> Tensor:
    """
    Perte de reconstruction L_rec.
    Phase D4: Supports multiple loss types (MSE, cosine similarity, or combined).
    
    Parameters
    ----------
    pred : Optional[Tensor]
        Predicted reconstruction
    target : Optional[Tensor]
        Target reconstruction
    loss_type : str
        Phase D4: Type of loss ("mse", "cosine", "mse+cosine")
    
    Returns
    -------
    Tensor
        Reconstruction loss (0 if pred or target is None)
    """
    if pred is None or target is None:
        return torch.zeros((), device=pred.device if pred is not None else "cpu")
    
    # Phase D4: Cosine similarity loss
    if loss_type == "cosine":
        # Flatten for cosine similarity computation
        pred_flat = pred.flatten(start_dim=1)  # [batch, features]
        target_flat = target.flatten(start_dim=1)  # [batch, features]
        
        # Compute cosine similarity: cos(θ) = (A·B) / (||A|| ||B||)
        dot_product = (pred_flat * target_flat).sum(dim=1)  # [batch]
        pred_norm = pred_flat.norm(dim=1)  # [batch]
        target_norm = target_flat.norm(dim=1)  # [batch]
        
        # Avoid division by zero
        epsilon = 1e-8
        cosine_sim = dot_product / (pred_norm * target_norm + epsilon)
        
        # Convert similarity to loss: 1 - cosine_similarity
        # Cosine sim ranges from -1 to 1, we want loss from 0 to 2
        loss = (1.0 - cosine_sim).mean()
        return loss
    
    # Phase D4: Combined MSE + Cosine
    elif loss_type == "mse+cosine":
        mse_loss = nn.functional.mse_loss(pred, target)
        
        # Cosine similarity component
        pred_flat = pred.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)
        dot_product = (pred_flat * target_flat).sum(dim=1)
        pred_norm = pred_flat.norm(dim=1)
        target_norm = target_flat.norm(dim=1)
        epsilon = 1e-8
        cosine_sim = dot_product / (pred_norm * target_norm + epsilon)
        cosine_loss = (1.0 - cosine_sim).mean()
        
        # Combine with equal weights
        return 0.5 * mse_loss + 0.5 * cosine_loss
    
    # Default: MSE loss
    else:  # loss_type == "mse"
        return nn.functional.mse_loss(pred, target)


def compute_rapsd_loss(
    pred: Tensor,
    target: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Perte spectrale radiale (RAPSD) — compare log-puissance entre prédiction et cible.
    pred, target : [B, C, H, W]
    """
    if pred.shape != target.shape:
        raise ValueError(f"RAPSD: shapes {pred.shape} vs {target.shape}")
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    B, C, H, W = pred.shape
    losses: List[Tensor] = []
    for b in range(B):
        for c in range(C):
            p = pred[b, c]
            t = target[b, c]
            fft_p = torch.fft.fftshift(torch.fft.fft2(p))
            fft_t = torch.fft.fftshift(torch.fft.fft2(t))
            psd_p = torch.abs(fft_p) ** 2
            psd_t = torch.abs(fft_t) ** 2
            cy, cx = H // 2, W // 2
            y_idx = torch.arange(H, device=pred.device, dtype=torch.float32) - cy
            x_idx = torch.arange(W, device=pred.device, dtype=torch.float32) - cx
            yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
            r = torch.sqrt(xx ** 2 + yy ** 2).long().clamp(min=0)
            max_r = int(r.max().item()) + 1
            rapsd_p = torch.zeros(max_r, device=pred.device)
            rapsd_t = torch.zeros(max_r, device=pred.device)
            counts = torch.zeros(max_r, device=pred.device)
            rf = r.flatten()
            rapsd_p.scatter_add_(0, rf, psd_p.flatten())
            rapsd_t.scatter_add_(0, rf, psd_t.flatten())
            counts.scatter_add_(0, rf, torch.ones_like(rf, dtype=torch.float32))
            valid = counts > 0
            rapsd_p[valid] /= counts[valid]
            rapsd_t[valid] /= counts[valid]
            log_ratio = torch.log(rapsd_p[valid] + eps) - torch.log(rapsd_t[valid] + eps)
            losses.append((log_ratio ** 2).mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def compute_rapsd_spectral_value_no_grad(
    diffusion_decoder: CausalDiffusionDecoder,
    target: Tensor,
    conditioning: Tensor,
    *,
    amp_mode: str = "none",
) -> Tensor:
    """
    Une passe bruit → ε_θ → x̂₀ puis perte RAPSD, sans gradient.
    Utilisé en fin d’époque (métrique) pour éviter FFT/scatter dans la boucle batch.
    """
    noise_sp = torch.randn_like(target)
    bs = target.shape[0]
    timesteps_sp = torch.randint(
        0,
        diffusion_decoder.scheduler.num_train_timesteps,
        (bs,),
        device=target.device,
        dtype=torch.long,
    )
    noisy_sp = diffusion_decoder.scheduler.add_noise(target, noise_sp, timesteps_sp)
    with _train_autocast(amp_mode):
        noise_pred_sp = diffusion_decoder.forward(noisy_sp, timesteps_sp, conditioning)
    pred_x0 = diffusion_decoder.predict_x0_from_epsilon(noisy_sp, noise_pred_sp, timesteps_sp)
    return compute_rapsd_loss(pred_x0, target)


def prepare_target_and_conditioning_for_metric(
    batch: Dict[str, Any],
    encoder: IntelligibleVariableEncoder,
    rcn_runner: RCNSequenceRunner,
    *,
    device: torch.device,
    residual_key: str = "residual",
    batch_index_key: str = "batch_index",
    conditioning_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    amp_mode: str = "none",
) -> Optional[Tuple[Tensor, Tensor]]:
    """
    Reproduit le chemin encodeur → RCN → conditioning et la cible HR du train_epoch,
    pour un seul micro-batch (métrique RAPSD en fin d’époque).
    """
    lr_data: Tensor = batch["lr"].to(device)
    target_data: Tensor = batch.get(residual_key, batch.get("hr")).to(device)

    if torch.isnan(target_data).any():
        nan_fill = torch.nanmean(target_data).item()
        if not math.isfinite(nan_fill):
            nan_fill = 0.0
        target_data = torch.nan_to_num(target_data, nan=nan_fill)
    if torch.isinf(target_data).any():
        valid_mean = target_data[~(torch.isnan(target_data) | torch.isinf(target_data))].mean().item() if target_data.numel() > 0 else 0.0
        target_data = torch.nan_to_num(target_data, nan=valid_mean, posinf=valid_mean, neginf=valid_mean)
    if torch.isnan(lr_data).any():
        nan_fill = torch.nanmean(lr_data).item()
        if not math.isfinite(nan_fill):
            nan_fill = 0.0
        lr_data = torch.nan_to_num(lr_data, nan=nan_fill)

    hetero_data = batch["hetero"]
    with _train_autocast(amp_mode):
        H_init = encoder.init_state(hetero_data).to(device)
    drivers = [lr_data[t] for t in range(lr_data.shape[0])]
    with _train_autocast(amp_mode):
        seq_output = rcn_runner.run(H_init, drivers, reconstruction_sources=drivers)

    H_condition = seq_output.states[-1]
    batch_index = batch.get(batch_index_key)
    if batch_index is not None:
        batch_index = batch_index.to(device)
    if conditioning_fn is None:
        conditioning = encoder.project_state_tensor(H_condition, batch_index=batch_index)
    else:
        conditioning = conditioning_fn(H_condition, batch_index)
    conditioning = conditioning.to(device)

    if torch.isnan(conditioning).any() or torch.isinf(conditioning).any():
        return None

    target = target_data[-1]
    if target.dim() == 3:
        target = target.unsqueeze(0)
    elif target.dim() != 4:
        return None

    valid_mask_ts = batch.get("valid_mask")
    if valid_mask_ts is not None:
        vm = valid_mask_ts[-1].to(device=device)
        if vm.dtype != torch.bool:
            vm = vm > 0.5
        while vm.dim() < target.dim():
            vm = vm.unsqueeze(0)
        vm = vm.expand_as(target)
        target = target.clone().masked_fill(~vm, float("nan"))

    return target, conditioning


def compute_rapsd_metric_from_batch(
    *,
    encoder: IntelligibleVariableEncoder,
    rcn_runner: RCNSequenceRunner,
    diffusion_decoder: CausalDiffusionDecoder,
    batch: Dict[str, Any],
    device: torch.device,
    residual_key: str = "residual",
    batch_index_key: str = "batch_index",
    conditioning_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    amp_mode: str = "none",
    verbose: bool = False,
) -> Optional[float]:
    """
    Calcule la métrique RAPSD (scalaire) sur un batch, sans gradient.
    À appeler une fois par époque après ``train_epoch`` (ex. premier batch du loader).
    """
    # Strip both DDP and torch.compile wrappers so the eager-mode submodules
    # are exposed for the RAPSD metric forward (avoids the same TensorMatch
    # guard failure described in the precip/contrastive blocks).
    enc = _eager_core(encoder)
    rcn_cell = _eager_core(rcn_runner.cell)
    diff = _eager_core(diffusion_decoder)

    was_enc_train = enc.training
    was_rcn_train = rcn_cell.training
    was_diff_train = diff.training
    enc.eval()
    rcn_cell.eval()
    diff.eval()
    # Swap the runner's cell to the eager-mode core for the duration of this
    # call so ``rcn_runner.run(...)`` inside ``prepare_target_and_conditioning_for_metric``
    # does NOT re-enter the compiled graph with locally built tensors. Restored
    # in ``finally`` regardless of exceptions.
    _saved_cell = rcn_runner.cell
    rcn_runner.cell = rcn_cell
    try:
        prepared = prepare_target_and_conditioning_for_metric(
            batch,
            enc,
            rcn_runner,
            device=device,
            residual_key=residual_key,
            batch_index_key=batch_index_key,
            conditioning_fn=conditioning_fn,
            amp_mode=amp_mode,
        )
        if prepared is None:
            return None
        target, conditioning = prepared
        if target.shape[1] != diff.in_channels:
            if verbose:
                print(f"[RAPSD metric] Channel mismatch: target {target.shape[1]} vs UNet {diff.in_channels}")
            return None
        val = compute_rapsd_spectral_value_no_grad(diff, target, conditioning, amp_mode=amp_mode)
        return float(val.item())
    finally:
        rcn_runner.cell = _saved_cell
        if was_enc_train:
            enc.train()
        if was_rcn_train:
            rcn_cell.train()
        if was_diff_train:
            diff.train()


def loss_diffusion(
    decoder: CausalDiffusionDecoder,
    target: Tensor,
    conditioning: Tensor,
    *,
    use_focal_loss: bool = False,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    conditioning_spatial: Optional[Tensor] = None,
) -> Tensor:
    """
    Perte de diffusion L_gen en déléguant à CausalDiffusionDecoder.
    """
    return decoder.compute_loss(
        target,
        conditioning,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        conditioning_spatial=conditioning_spatial,
    )


def loss_no_tears(A_masked: Tensor) -> Tensor:
    """
    Implémentation de la contrainte NO TEARS : tr(e^{A∘A}) - q.
    
    Note: Cette méthode est instable et O(q³). Utilisez loss_dagma() pour de meilleures performances.
    """
    A_squared = torch.mul(A_masked, A_masked)
    matrix_exp = torch.matrix_exp(A_squared)
    trace = torch.trace(matrix_exp)
    return trace - A_masked.shape[0]


def compute_divergence(field: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tensor:
    """
    Phase 3.3: Compute divergence of a 2D vector field using finite differences.
    
    For a field [u, v] with shape [batch, 2, H, W], computes div = ∂u/∂x + ∂v/∂y
    
    Parameters
    ----------
    field : Tensor
        Vector field [batch, 2, H, W] or [batch, channels, H, W] where first 2 channels are (u, v)
    dx : float
        Spatial step in x-direction (longitude)
    dy : float
        Spatial step in y-direction (latitude)
    
    Returns
    -------
    Tensor
        Divergence field [batch, H, W]
    """
    if field.shape[1] < 2:
        return torch.zeros(field.shape[0], field.shape[2], field.shape[3], device=field.device)
    
    u = field[:, 0:1, :, :]  # [batch, 1, H, W]
    v = field[:, 1:2, :, :]  # [batch, 1, H, W]
    
    # Compute gradients using central differences
    # ∂u/∂x using central difference
    u_pad = nn.functional.pad(u, (1, 1, 0, 0), mode='replicate')
    du_dx = (u_pad[:, :, :, 2:] - u_pad[:, :, :, :-2]) / (2 * dx)
    
    # ∂v/∂y using central difference
    v_pad = nn.functional.pad(v, (0, 0, 1, 1), mode='replicate')
    dv_dy = (v_pad[:, :, 2:, :] - v_pad[:, :, :-2, :]) / (2 * dy)
    
    # Divergence
    divergence = du_dx.squeeze(1) + dv_dy.squeeze(1)  # [batch, H, W]
    return divergence


def compute_vorticity(field: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tensor:
    """
    Phase 3.3: Compute vorticity (curl) of a 2D vector field using finite differences.
    
    For a field [u, v] with shape [batch, 2, H, W], computes vorticity = ∂v/∂x - ∂u/∂y
    
    Parameters
    ----------
    field : Tensor
        Vector field [batch, 2, H, W] or [batch, channels, H, W] where first 2 channels are (u, v)
    dx : float
        Spatial step in x-direction (longitude)
    dy : float
        Spatial step in y-direction (latitude)
    
    Returns
    -------
    Tensor
        Vorticity field [batch, H, W]
    """
    if field.shape[1] < 2:
        return torch.zeros(field.shape[0], field.shape[2], field.shape[3], device=field.device)
    
    u = field[:, 0:1, :, :]  # [batch, 1, H, W]
    v = field[:, 1:2, :, :]  # [batch, 1, H, W]
    
    # Compute gradients using central differences
    # ∂v/∂x
    v_pad = nn.functional.pad(v, (1, 1, 0, 0), mode='replicate')
    dv_dx = (v_pad[:, :, :, 2:] - v_pad[:, :, :, :-2]) / (2 * dx)
    
    # ∂u/∂y
    u_pad = nn.functional.pad(u, (0, 0, 1, 1), mode='replicate')
    du_dy = (u_pad[:, :, 2:, :] - u_pad[:, :, :-2, :]) / (2 * dy)
    
    # Vorticity
    vorticity = dv_dx.squeeze(1) - du_dy.squeeze(1)  # [batch, H, W]
    return vorticity


def loss_physical(output: Tensor, target: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tensor:
    """
    Phase 3.3: Compute physical loss L_phy enforcing divergence and vorticity constraints.
    
    Penalizes divergence and vorticity errors between output and target fields.
    For climate fields, divergence should be small (mass conservation) and vorticity
    should match the target (circulation conservation).
    
    Parameters
    ----------
    output : Tensor
        Predicted field [batch, channels, H, W]
    target : Tensor
        Target field [batch, channels, H, W]
    dx : float
        Spatial step in x-direction
    dy : float
        Spatial step in y-direction
    
    Returns
    -------
    Tensor
        Physical loss (divergence + vorticity errors)
    """
    # Compute divergence for both output and target
    div_output = compute_divergence(output, dx=dx, dy=dy)
    div_target = compute_divergence(target, dx=dx, dy=dy)
    
    # Divergence error (should be close to zero for mass conservation)
    div_loss = nn.functional.mse_loss(div_output, div_target)
    
    # Compute vorticity for both output and target
    vort_output = compute_vorticity(output, dx=dx, dy=dy)
    vort_target = compute_vorticity(target, dx=dx, dy=dy)
    
    # Vorticity error (should match target circulation)
    vort_loss = nn.functional.mse_loss(vort_output, vort_target)
    
    # Combined physical loss
    return div_loss + vort_loss


def loss_precip_physical(
    x0_pred: Tensor,
    target: Tensor,
    *,
    w_positivity: float = 1.0,
    w_mass: float = 0.1,
    w_quantile: float = 0.2,
    quantiles: Tuple[float, float] = (0.95, 0.99),
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """
    Sprint 3: precipitation-specific physical loss.

    The divergence/vorticity penalty in :func:`loss_physical` is defined for
    wind vector fields and degenerates for the 1-channel precipitation
    target (``compute_divergence`` returns zeros when ``field.shape[1] < 2``),
    which is why ``lambda_phy`` was inert in the precipitation config.
    This helper provides meaningful, differentiable physical constraints
    that *do* apply to a scalar precipitation field:

    1. **Positivity**: penalise negative predicted precipitation via a soft
       hinge ``mean(relu(-x0_pred))``. Precipitation is non-negative by
       definition; this acts as a prior that regularises extrapolated
       residuals under climate drift.
    2. **Global mass conservation**: align the spatial mean of the
       prediction with that of the target over valid pixels. Prevents
       the decoder from introducing a global bias on top of the bicubic
       baseline.
    3. **Tail quantile matching**: square-difference between matched
       quantiles (p95, p99) of the prediction and the target, exactly the
       tail ORACLE cares about. This is the loss that directly targets
       ``F1_p95`` / ``F1_p99``.

    ``x0_pred`` is typically recovered inside the training loop from the
    predicted ε and the current noisy sample via
    :func:`CausalDiffusionDecoder.predict_x0_from_epsilon`, so the gradient
    flows back through the UNet — unlike the target-only fallback in
    :func:`loss_physical`, which had zero gradient.

    All components are reduced to a single scalar loss.
    """
    if valid_mask is None:
        valid_mask = torch.isfinite(target) & torch.isfinite(x0_pred)
    else:
        valid_mask = valid_mask & torch.isfinite(target) & torch.isfinite(x0_pred)
    if not valid_mask.any():
        return torch.zeros((), device=x0_pred.device, dtype=x0_pred.dtype)

    xp = x0_pred[valid_mask]
    xt = target[valid_mask]

    # 1. Positivity (soft hinge on negatives).
    l_pos = nn.functional.relu(-xp).mean() if w_positivity > 0.0 else xp.new_zeros(())

    # 2. Global mass conservation: match spatial means.
    if w_mass > 0.0:
        l_mass = (xp.mean() - xt.mean()) ** 2
    else:
        l_mass = xp.new_zeros(())

    # 3. Quantile matching — per-quantile square error.
    if w_quantile > 0.0 and xp.numel() > 0:
        q_tensor = torch.tensor(quantiles, device=xp.device, dtype=xp.dtype)
        q_pred = torch.quantile(xp, q_tensor)
        q_true = torch.quantile(xt, q_tensor)
        l_quant = ((q_pred - q_true) ** 2).mean()
    else:
        l_quant = xp.new_zeros(())

    return w_positivity * l_pos + w_mass * l_mass + w_quantile * l_quant


def loss_dagma(
    A_masked: Tensor,
    s: float = 1.0,
    add_l1_regularization: bool = False,  # Phase D3: Add L1 regularization for sparsity
    l1_weight: float = 0.01,  # Phase D3: Weight for L1 regularization
) -> Tensor:
    """
    Implémentation de la contrainte DAGMA (DAG via la méthode d'augmentation du log-déterminant).
    
    Plus stable et efficace que NO TEARS. Utilise la formule:
    h(W) = -log det(sI - W∘W) + d log s
    
    Phase D3: Enhanced with numerical stability improvements and optional L1 regularization.
    
    Parameters
    ----------
    A_masked : Tensor
        Matrice DAG avec diagonale masquée [q, q]
    s : float
        Paramètre de régularisation (par défaut 1.0). Doit être > rho(A_masked ∘ A_masked)
        où rho est le rayon spectral.
    add_l1_regularization : bool
        Phase D3: If True, add L1 regularization for sparser DAGs
    l1_weight : float
        Phase D3: Weight for L1 regularization term
    
    Returns
    -------
    Tensor
        Valeur de la contrainte DAGMA (doit être > 0 pour un DAG valide)
    
    References
    ----------
    - Bello et al. (2022): "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization"
    """
    q = A_masked.shape[0]
    device = A_masked.device
    dtype = A_masked.dtype
    
    A_clipped = torch.clamp(A_masked, min=-10.0, max=10.0)
    
    # W∘W (Hadamard square, element-wise)
    W_squared = torch.mul(A_clipped, A_clipped)  # [q, q]
    
    # s must exceed the spectral radius of W_squared for M to be positive-definite.
    # Gershgorin bound: rho(W²) <= max row-sum of |W²|.  This is tight and O(q²).
    gershgorin_bound = W_squared.abs().sum(dim=1).max().item()
    s_safe = max(s, gershgorin_bound + 0.1)
    
    # M = sI - W∘W  (M-matrix, positive-definite when s > rho(W²))
    sI = s_safe * torch.eye(q, device=device, dtype=dtype)
    M = sI - W_squared  # [q, q]
    
    eps = 1e-7
    M = M + eps * torch.eye(q, device=device, dtype=dtype)
    
    try:
        log_det_M = torch.logdet(M)
        if not torch.isfinite(log_det_M):
            M = M + 1e-5 * torch.eye(q, device=device, dtype=dtype)
            log_det_M = torch.logdet(M)
    except RuntimeError:
        return torch.tensor(float(q), device=device, dtype=dtype, requires_grad=True)
    
    # h(W) = -log det(sI - W∘W) + d log s
    h_W = -log_det_M + q * math.log(s_safe + eps)
    
    if add_l1_regularization:
        l1_term = l1_weight * A_masked.abs().sum()
        h_W = h_W + l1_term
    
    if not torch.isfinite(h_W):
        return torch.tensor(float(q), device=device, dtype=dtype, requires_grad=True)
    
    return h_W


@dataclass
class TrainingStepResult:
    """
    Résultats agrégés d'une étape (batch) d'entraînement.
    """

    loss_total: float
    loss_gen: float
    loss_rec: float
    loss_dag: float


def train_epoch(
    *,
    encoder: IntelligibleVariableEncoder,
    rcn_runner: RCNSequenceRunner,
    diffusion_decoder: CausalDiffusionDecoder,
    optimizer: torch.optim.Optimizer,
    data_loader: Iterable[Dict[str, Tensor]],
    lambda_gen: float,
    beta_rec: float,
    gamma_dag: float,
    conditioning_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    device: torch.device,
    gradient_clipping: Optional[float] = None,
    batch_index_key: str = "batch_index",
    residual_key: str = "residual",
    log_interval: int = 10,
    verbose: bool = True,
    dag_method: str = "dagma",  # "dagma" or "no_tears"
    dagma_s: float = 1.0,  # Parameter for DAGMA constraint
    lambda_phy: float = 0.0,  # Phase 3.3: Weight for physical loss (divergence + vorticity)
    dx: float = 1.0,  # Phase 3.3: Spatial step in x-direction (longitude)
    dy: float = 1.0,  # Phase 3.3: Spatial step in y-direction (latitude)
    use_predicted_output: bool = False,  # Phase B2: Use predictions for physical loss (expensive)
    physical_sample_interval: int = 10,  # Phase B2: Sample predictions every N batches
    physical_num_steps: int = 15,  # Phase B2: EDM sampling steps for physical loss
    use_amp: bool = True,  # Phase C1: CUDA FP16+GradScaler, or CPU bfloat16 autocast if supported
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,  # Phase C2: LR scheduler
    use_focal_loss: bool = False,  # Phase D1: Use focal loss for diffusion
    focal_alpha: float = 1.0,  # Phase D1: Focal loss alpha
    focal_gamma: float = 2.0,  # Phase D1: Focal loss gamma (higher = more focus on hard pixels)
    extreme_weight_factor: float = 0.0,  # Phase D2: Weight factor for extreme events (0 = disabled)
    extreme_percentiles: List[float] = None,  # Phase D2: Percentiles for extreme events
    reconstruction_loss_type: str = "mse",  # Phase D4: Loss type for reconstruction ("mse", "cosine", "mse+cosine")
    use_spectral_loss: bool = False,
    lambda_spectral: float = 0.0,
    conditioning_dropout_prob: float = 0.0,
    lambda_dag_prior: float = 0.0,
    dag_prior: Optional[Tensor] = None,
    spatial_projector: Optional[nn.Module] = None,
    # Phase DAG-decouple
    dag_warmup_scale: float = 1.0,          # multiplier in [0, 1] applied to gamma_dag this epoch
    dag_l1_regularization: bool = False,    # forwarded to loss_dagma
    dag_l1_weight: float = 0.01,            # forwarded to loss_dagma
    dag_spectral_projection: bool = True,   # hard-project A_dag after each optimizer.step()
    dag_spectral_max_radius: float = 0.95,  # target spectral radius bound for rho(A ∘ A)
    # Sprint 2: HR identifiability head (predicts summary stats of the HR
    # target from the pooled causal state). When a head is provided,
    # L_hr_ident is added to the total loss scaled by ``beta_hr_ident``,
    # and its gradient flows back through the encoder + RCN, pulling A_dag
    # toward representations that actually know about precipitation extremes.
    hr_ident_head: Optional[nn.Module] = None,
    beta_hr_ident: float = 0.0,
    # Sprint 3: precipitation-specific physical loss (positivity + global
    # mass + quantile matching). Differentiable through the UNet via
    # x0-from-epsilon, unlike the target-only ``lambda_phy`` fallback.
    lambda_precip_phy: float = 0.0,
    precip_phy_weights: Tuple[float, float, float] = (1.0, 0.1, 0.2),
    # Sprint 4: contrastive DAG-conditioning loss. For every
    # ``contrastive_dag_interval``-th optimizer step, compute an *extra*
    # UNet forward with the DAG-token portion of ``conditioning_spatial``
    # zeroed, and add a margin loss forcing the real-A_dag epsilon
    # prediction to beat the DAG-blind one by at least
    # ``contrastive_dag_margin``. This is what makes the UNet *use* the
    # DAG tokens rather than marginalising them out. Only active when the
    # spatial projector is causal-aware (``CausalConditioningProjector``).
    lambda_contrastive_dag: float = 0.0,
    contrastive_dag_margin: float = 0.02,
    contrastive_dag_interval: int = 4,
) -> Dict[str, float]:
    """
    Entraîne les modules sur une epoch complète.

    ``use_spectral_loss`` / ``lambda_spectral`` : conservés pour compatibilité ; la perte RAPSD
    n'est plus appliquée dans la boucle batch (voir ``log_spectral_metric_each_epoch`` + ``compute_rapsd_metric_from_batch``).
    
    Parameters
    ----------
    log_interval : int
        Affiche les logs tous les N batches (par défaut 10).
    verbose : bool
        Si True, affiche des logs détaillés (par défaut True).
    """
    encoder.train()
    rcn_runner.cell.train()
    diffusion_decoder.train()
    
    # Phase C1: Mixed precision — CUDA FP16 + GradScaler, or CPU BF16 (no scaler)
    scaler = None
    amp_mode = "none"
    if use_amp:
        if device.type == "cuda" and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler

            scaler = GradScaler()
            amp_mode = "cuda_fp16"
        elif device.type == "cpu":
            bf16_ok = getattr(torch.cpu, "is_bf16_supported", None)
            if bf16_ok is not None and bf16_ok():
                amp_mode = "cpu_bf16"
            elif verbose:
                print("[WARN] CPU bfloat16 not supported; training in FP32")
        elif verbose:
            print(f"[WARN] AMP not used: device type {device.type}")
    
    # Phase D2: Initialize extreme percentiles if not provided
    if extreme_percentiles is None:
        extreme_percentiles = [95.0, 99.0]

    total_loss = 0.0
    total_gen = 0.0
    total_rec = 0.0
    total_dag = 0.0
    total_phy = 0.0  # Phase 3.3: Physical loss accumulator
    total_contrastive = 0.0  # Sprint 4: contrastive DAG loss accumulator
    total_dag_sensitivity = 0.0  # Sprint 4: avg (MSE_zero - MSE_real)
    num_contrastive_steps = 0
    num_batches = 0
    # Sprint 4 / C4: epoch-end failure counters so silent ``except Exception``
    # blocks no longer hide repeated failures past the first batch. Each entry
    # records the last error message and a hit count.
    silent_failures: Dict[str, Tuple[int, str]] = {
        "physical_sample": (0, ""),
        "precip_phy": (0, ""),
        "hr_ident": (0, ""),
        "contrastive_dag": (0, ""),
    }
    
    epoch_start_time = time.time()
    
    if verbose:
        print(f"\n{'='*80}")
        print("START EPOCH")
        print(f"{'='*80}")
        print("Config:")
        print(f"   - Device: {device}")
        print(f"   - AMP mode: {amp_mode}")
        print(f"   - Lambda (gen): {lambda_gen}")
        print(f"   - Beta (rec): {beta_rec}")
        print(f"   - Gamma (DAG): {gamma_dag}")
        print(f"   - Gradient clipping: {gradient_clipping}")
        print(f"   - Log interval: {log_interval}")
        print(f"{'='*80}\n")

    for batch_idx, batch in enumerate(data_loader):
        # Support batch_size > 1: batch can be a list of batch dicts (gradient accumulation)
        batches = batch if isinstance(batch, list) else [batch]
        batch_start_time = time.time()
        
        optimizer.zero_grad()
        step_loss_total = 0.0
        step_loss_gen = 0.0
        step_loss_rec = 0.0
        step_loss_dag = 0.0
        step_loss_phy = 0.0
        
        for micro_idx, batch in enumerate(batches):
            _do_timing = verbose and batch_idx == 0 and micro_idx == 0

            if _do_timing:
                print(f"Batch {batch_idx + 1}" + (f" (n={len(batches)})" if len(batches) > 1 else "") + ":")
                print(f"   - Keys: {list(batch.keys())}")
            
            lr_data: Tensor = batch["lr"].to(device)      # [seq_len, N, features_lr]
            target_data: Tensor = batch.get(residual_key, batch.get("hr")).to(device)  # [seq_len, channels, H, W]
            hetero_data = batch["hetero"]
        
            # NaN/Inf input sanitization (cadenced to first micro-batch of every 20th optimizer step)
            if batch_idx % 20 == 0 and micro_idx == 0:
                _tensors_to_check = {
                    "target_data": target_data,
                    "lr_data": lr_data,
                }
                for _name, _t in _tensors_to_check.items():
                    if not torch.isfinite(_t).all():
                        if _name == "target_data":
                            _nan_mask = torch.isnan(target_data)
                            if _nan_mask.any():
                                nan_fill = torch.nanmean(target_data).item()
                                if not math.isfinite(nan_fill):
                                    nan_fill = 0.0
                                target_data = torch.nan_to_num(target_data, nan=nan_fill)
                            _inf_mask = torch.isinf(target_data)
                            if _inf_mask.any():
                                _valid_mask = torch.isfinite(target_data)
                                valid_mean = target_data[_valid_mask].mean().item() if _valid_mask.any() else 0.0
                                target_data = torch.nan_to_num(
                                    target_data,
                                    nan=valid_mean,
                                    posinf=valid_mean,
                                    neginf=valid_mean,
                                )
                        elif _name == "lr_data":
                            _nan_mask = torch.isnan(lr_data)
                            if _nan_mask.any():
                                nan_fill = torch.nanmean(lr_data).item()
                                if not math.isfinite(nan_fill):
                                    nan_fill = 0.0
                                lr_data = torch.nan_to_num(lr_data, nan=nan_fill)

            if _do_timing:
                print(f"   - LR data shape: {lr_data.shape}")
                print(f"   - Target data shape: {target_data.shape}")
                print(f"   - Sequence length: {lr_data.shape[0]}")

            # Encoder step
            if _do_timing:
                encoder_time = time.time()
            with _train_autocast(amp_mode):
                H_init = encoder.init_state(hetero_data).to(device)
            if _do_timing:
                encoder_time = time.time() - encoder_time
                print(f"   - H_init shape: {H_init.shape}")
                print(f"   - Encoder time: {encoder_time:.4f}s")

            # RCN step
            if _do_timing:
                rcn_time = time.time()
            drivers = [lr_data[t] for t in range(lr_data.shape[0])]
            # Note: ``reconstruction_sources`` is the *input* of the recon
            # decoder, not its target. The decoder is a
            # ``Linear(num_vars * hidden_dim, driver_dim)`` so it must be fed
            # the hidden state ``H`` (shape ``[q, N, hidden]``), not the
            # drivers themselves (``[N, driver_dim]``). Passing ``None`` here
            # makes the cell fall back to ``H_prev``, which is what the
            # decoder was sized for; the loss is then computed against
            # ``driver_step`` further down. Previously this passed
            # ``reconstruction_sources=drivers`` and crashed with
            # ``mat1 [N, driver_dim] vs mat2 [num_vars*hidden, driver_dim]``.
            with _train_autocast(amp_mode):
                seq_output = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
            if _do_timing:
                rcn_time = time.time() - rcn_time
                print(f"   - Number of states: {len(seq_output.states)}")
                print(f"   - Number of reconstructions: {len(seq_output.reconstructions)}")
                print(f"   - Number of DAG matrices: {len(seq_output.dag_matrices)}")
                print(f"   - RCN time: {rcn_time:.4f}s")

            # Loss computation (reconstruction and DAG)
            loss_rec_value = torch.tensor(0.0, device=device)
            loss_dag_value = torch.tensor(0.0, device=device)
            num_reconstructions = 0
            with _train_autocast(amp_mode):
                for recon, driver_step in zip(
                    seq_output.reconstructions,
                    drivers,
                ):
                    if recon is not None:
                        num_reconstructions += 1
                        loss_rec_value = loss_rec_value + beta_rec * loss_reconstruction(
                            recon, driver_step, loss_type=reconstruction_loss_type
                        )

                # A_masked is the same learned adjacency at every timestep;
                # compute the DAG penalty **once** (no n_dag_steps multiplier —
                # the old code effectively multiplied gamma_dag by seq_len,
                # which made the penalty dominate the diffusion objective in
                # gradient norm). Scaling is now governed entirely by
                # ``gamma_dag * dag_warmup_scale``, which lets the caller run
                # a warm-up curriculum where the DAG penalty starts at 0 and
                # ramps up as the diffusion branch stabilises.
                A_masked_0 = seq_output.dag_matrices[0]
                gamma_eff = gamma_dag * dag_warmup_scale
                if dag_method == "dagma":
                    loss_dag_value = gamma_eff * loss_dagma(
                        A_masked_0,
                        s=dagma_s,
                        add_l1_regularization=dag_l1_regularization,
                        l1_weight=dag_l1_weight,
                    )
                else:
                    loss_dag_value = gamma_eff * loss_no_tears(A_masked_0)

                if lambda_dag_prior > 0.0 and dag_prior is not None:
                    _prior = dag_prior.to(device=A_masked_0.device, dtype=A_masked_0.dtype)
                    loss_dag_value = loss_dag_value + lambda_dag_prior * nn.functional.mse_loss(A_masked_0, _prior)
        
            if _do_timing:
                print(f"   - Reconstructions computed: {num_reconstructions}/{len(seq_output.reconstructions)}")

            H_condition = seq_output.states[-1]
            batch_index = batch.get(batch_index_key)
            if batch_index is not None:
                batch_index = batch_index.to(device)
            if conditioning_fn is None:
                conditioning = encoder.project_state_tensor(H_condition, batch_index=batch_index)
            else:
                conditioning = conditioning_fn(H_condition, batch_index)
            conditioning = conditioning.to(device)

            conditioning_spatial = None
            if spatial_projector is not None:
                # Sprint 2: CausalConditioningProjector also needs A_dag so the
                # UNet can attend to the current DAG topology. We detect it
                # via duck-typing on ``num_dag_tokens`` — if the projector has
                # this attribute, it's causal-aware and we pass A_masked.
                _proj_target = _eager_core(spatial_projector)
                if hasattr(_proj_target, "num_dag_tokens"):
                    # A_masked is attached in the last cell output; we pass
                    # it so the DAG-token gradient can flow back to A_dag
                    # (combined with dag_grad_gate this is what lets the
                    # diffusion loss actually train the DAG).
                    _A = seq_output.dag_matrices[-1]
                    conditioning_spatial = spatial_projector(
                        H_condition, _A, batch_index=batch_index
                    ).to(device)
                else:
                    conditioning_spatial = spatial_projector(
                        H_condition, batch_index=batch_index
                    ).to(device)

            _dropout = conditioning_dropout_prob > 0.0 and torch.rand(1, device=device).item() < conditioning_dropout_prob
            if _dropout:
                conditioning = torch.zeros_like(conditioning)
                if conditioning_spatial is not None:
                    conditioning_spatial = torch.zeros_like(conditioning_spatial)
        
            if _do_timing:
                print(f"   - Conditioning shape: {conditioning.shape}")

            target = target_data[-1]  # Should be [channels, H, W] or [H, W, channels]
            # Ensure target has shape [batch, channels, H, W]
            if target.dim() == 3:
                # Check if it's [channels, H, W] or [H, W, channels]
                # UNet expects [batch, channels, H, W]
                # If first dim is very large, it might be [H*W, channels] or similar
                # For now, assume [channels, H, W] and add batch dim
                target = target.unsqueeze(0)
            elif target.dim() == 4:
                # Already has batch dimension, use as is
                pass
            else:
                raise ValueError(f"Unexpected target shape: {target.shape}, expected [channels, H, W] or [batch, channels, H, W]")
        
            if _do_timing:
                print(f"   - Target shape (after processing): {target.shape}")
        
            # Verify channel count matches UNet expectations
            if target.shape[1] != diffusion_decoder.in_channels:
                raise ValueError(
                    f"Channel mismatch: target has {target.shape[1]} channels, "
                    f"but UNet expects {diffusion_decoder.in_channels} channels. "
                    f"Target shape: {target.shape}"
                )

            valid_mask_ts = batch.get("valid_mask")
            if valid_mask_ts is not None:
                vm = valid_mask_ts[-1].to(device=device)
                if vm.dtype != torch.bool:
                    vm = vm > 0.5
                while vm.dim() < target.dim():
                    vm = vm.unsqueeze(0)
                vm = vm.expand_as(target)
                target = target.clone().masked_fill(~vm, float("nan"))
        
            # Expensive per-micro-batch diagnostics: only on first micro-batch
            nan_count = 0
            nan_ratio = 0.0
            if micro_idx == 0:
                if _do_timing:
                    print(f"   - Target stats: min={target.min().item():.6f}, max={target.max().item():.6f}, mean={target.mean().item():.6f}, std={target.std().item():.6f}")
                    _target_isfinite = torch.isfinite(target).all().item()
                    print(f"   - Target has NaN/Inf: {not _target_isfinite}")
                    print(f"   - Conditioning stats: min={conditioning.min().item():.6f}, max={conditioning.max().item():.6f}, mean={conditioning.mean().item():.6f}")
                    _cond_isfinite = torch.isfinite(conditioning).all().item()
                    print(f"   - Conditioning has NaN/Inf: {not _cond_isfinite}")

                target_abs_max = target.abs().max().item()
                if target_abs_max > 1e6:
                    if verbose:
                        print(f"[WARN] Target has very large values: max_abs={target_abs_max:.2e}")

                nan_mask = ~torch.isfinite(target)
                nan_count = nan_mask.sum().item()
                nan_ratio = nan_count / target.numel() if target.numel() > 0 else 0.0
                if nan_count > 0 and verbose and (batch_idx == 0 or batch_idx % log_interval == 0):
                    print(f"[INFO] Target contains {nan_count} NaN/Inf pixels ({nan_ratio:.2%}) - will be masked in loss")

            # Conditioning must NEVER contain NaN/Inf (critical safety check on every micro-batch)
            if not torch.isfinite(conditioning).all():
                print(f"[ERROR] Conditioning contains NaN/Inf in batch {batch_idx + 1}")
                print(f"   - NaN count: {torch.isnan(conditioning).sum().item()}")
                print(f"   - Inf count: {torch.isinf(conditioning).sum().item()}")
                print(f"   - This is a critical error, skipping batch")
                continue
        
            # Phase C1: Mixed Precision - Forward pass with autocast for entire forward
            # Diffusion loss (gère automatiquement les NaN via masquage)
            # Phase D1: Supports focal loss for focusing on hard pixels
            if _do_timing:
                diffusion_time = time.time()
            with _train_autocast(amp_mode):
                loss_gen_value = lambda_gen * loss_diffusion(
                    diffusion_decoder, target, conditioning,
                    use_focal_loss=use_focal_loss,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    conditioning_spatial=conditioning_spatial,
                )

            # RAPSD (FFT + scatter_add) : déplacé en fin d’époque — voir compute_rapsd_metric_from_batch.

            # Phase B2: Compute physical loss (divergence + vorticity)
            # Improved version that compares predictions vs target (not just target)
            loss_phy_value = torch.tensor(0.0, device=device)
            if lambda_phy > 0.0:
                # Phase B2: Option 3 (Hybrid) - Compute physical loss on predictions periodically
                # For efficiency, we only sample predictions every N batches
                # This reduces cost while still enforcing physical consistency on model outputs
                compute_physical_on_predictions = (
                    use_predicted_output and
                    batch_idx % physical_sample_interval == 0
                )
            
                if compute_physical_on_predictions:
                    # Sample a prediction from the model (expensive but accurate)
                    # Use EDM with few steps for efficiency (15-25 steps vs 1000 for DDPM)
                    with torch.no_grad():  # Don't backprop through sampling
                        try:
                            sampled_output = diffusion_decoder.sample(
                                conditioning=conditioning,
                                num_steps=physical_num_steps,
                                scheduler_type="edm",  # Use EDM for fast sampling
                                apply_constraints=True,
                            )
                            # sampled_output.residual is [batch, channels, H, W]
                            pred_residual = sampled_output.residual
                        
                            # Compare physical constraints: pred vs target
                            div_pred = compute_divergence(pred_residual, dx=dx, dy=dy)
                            div_target = compute_divergence(target, dx=dx, dy=dy)
                            vort_pred = compute_vorticity(pred_residual, dx=dx, dy=dy)
                            vort_target = compute_vorticity(target, dx=dx, dy=dy)
                        
                            # Physical loss: enforce that predictions have similar physical properties as target
                            div_error = ((div_pred - div_target) ** 2).mean()
                            vort_error = ((vort_pred - vort_target) ** 2).mean()
                            loss_phy_value = lambda_phy * (div_error + 0.1 * vort_error)
                        
                            if _do_timing:
                                print(f"   - Physical loss computed on predictions (EDM, {physical_num_steps} steps)")
                        except Exception as e:
                            # Fallback to target-only physical loss if sampling fails
                            _n, _ = silent_failures["physical_sample"]
                            silent_failures["physical_sample"] = (_n + 1, str(e))
                            if verbose and batch_idx == 0:
                                print(f"[WARN] Physical loss sampling failed: {e}, falling back to target-only")
                            div_target = compute_divergence(target, dx=dx, dy=dy)
                            vort_target = compute_vorticity(target, dx=dx, dy=dy)
                            div_penalty = (div_target ** 2).mean()
                            vort_penalty = (vort_target ** 2).mean()
                            loss_phy_value = lambda_phy * (div_penalty + 0.1 * vort_penalty)
                else:
                    # Default: compute physical loss on target only (fast, acts as regularization)
                    # This ensures target satisfies physical laws
                    div_target = compute_divergence(target, dx=dx, dy=dy)
                    vort_target = compute_vorticity(target, dx=dx, dy=dy)
                    # Penalize non-zero divergence (should be ~0 for mass conservation) and inconsistent vorticity
                    div_penalty = (div_target ** 2).mean()
                    vort_penalty = (vort_target ** 2).mean()
                    loss_phy_value = lambda_phy * (div_penalty + 0.1 * vort_penalty)
        
            if _do_timing:
                diffusion_time = time.time() - diffusion_time
                print(f"   - Diffusion time: {diffusion_time:.4f}s")
                if lambda_phy > 0.0:
                    print(f"   - Physical loss: {loss_phy_value.item():.6f}")
                if nan_count > 0:
                    print(f"   - Loss computed on {target.numel() - nan_count}/{target.numel()} valid pixels ({1.0 - nan_ratio:.2%})")

            # Sprint 3: precipitation-specific physical loss.
            # Computed via an extra UNet forward on the current noisy target
            # (not a full sampling — ~1.2x the cost of compute_loss), from
            # which we recover x0_pred = (noisy - √(1-α̅)·ε̂) / √α̅ and feed
            # the composite ``baseline + x0_pred`` to ``loss_precip_physical``.
            # Gated by ``lambda_precip_phy > 0`` and by a coarse interval to
            # keep CPU cost bounded. Unlike ``lambda_phy`` (which is zero for
            # 1-channel precipitation), this one has a non-zero gradient on
            # the UNet / encoder / RCN parameters.
            loss_precip_phy_value = torch.tensor(0.0, device=device)
            # Unwrap *both* DDP and torch.compile. The previous code only
            # unwrapped DDP, which left ``_diff_unwrapped`` as a
            # ``torch._dynamo.eval_frame.OptimizedModule`` whenever
            # ``training.compile.enabled`` was true. Calling ``.forward(...)``
            # on the compiled module from outside the main ``compute_loss``
            # entry point re-enters the compiled graph with tensors built
            # by ``torch.where`` / ``randn_like`` / ``scheduler.add_noise``
            # *here* (not by ``compute_loss``), which present different
            # strides than what was traced — the TensorMatch guard fires
            # and the surrounding try/except silently skips the loss.
            # ``_eager_core`` strips both DDP and ``_orig_mod`` so the
            # extra forward passes always hit the eager-mode forward.
            _diff_unwrapped = _eager_core(diffusion_decoder)
            if (
                lambda_precip_phy > 0.0
                and batch_idx % max(1, physical_sample_interval) == 0
                # Only fire once per outer batch (was firing on every
                # micro-batch when ``batch_size > 1`` was implemented as
                # gradient accumulation, which multiplied the cost by 48
                # on the user's setup and made batch 1 take 10+ minutes).
                and micro_idx == 0
            ):
                try:
                    with _train_autocast(amp_mode):
                        _valid = torch.isfinite(target)
                        _target_clean = torch.where(
                            _valid, target, torch.zeros_like(target)
                        )
                        _noise = torch.randn_like(_target_clean)
                        _bs = _target_clean.shape[0]
                        _ts = torch.randint(
                            0,
                            _diff_unwrapped.scheduler.num_train_timesteps,
                            (_bs,),
                            device=_target_clean.device,
                            dtype=torch.long,
                        )
                        _noisy = _diff_unwrapped.scheduler.add_noise(
                            _target_clean, _noise, _ts
                        )
                        _noise_pred = _diff_unwrapped.forward(
                            _noisy,
                            _ts,
                            conditioning,
                            conditioning_spatial=conditioning_spatial,
                        )
                        _x0_pred = _diff_unwrapped.predict_x0_from_epsilon(
                            _noisy, _noise_pred, _ts
                        )
                        # Compose with baseline when available so the loss
                        # speaks in HR composite units; otherwise stay in
                        # residual space.
                        _baseline = batch.get("baseline")
                        if _baseline is not None:
                            _baseline_t = _baseline[-1].to(device)
                            if _baseline_t.dim() == 3:
                                _baseline_t = _baseline_t.unsqueeze(0)
                            if _baseline_t.shape == _x0_pred.shape:
                                _x0_composite = _baseline_t + _x0_pred
                                _target_composite = _baseline_t + _target_clean
                            else:
                                _x0_composite = _x0_pred
                                _target_composite = _target_clean
                        else:
                            _x0_composite = _x0_pred
                            _target_composite = _target_clean
                        w_pos, w_mass, w_quant = precip_phy_weights
                        loss_precip_phy_value = lambda_precip_phy * loss_precip_physical(
                            _x0_composite,
                            _target_composite,
                            w_positivity=w_pos,
                            w_mass=w_mass,
                            w_quantile=w_quant,
                            valid_mask=_valid,
                        )
                except Exception as _pp_ex:
                    _n, _ = silent_failures["precip_phy"]
                    silent_failures["precip_phy"] = (_n + 1, str(_pp_ex))
                    if verbose and batch_idx == 0:
                        print(f"[WARN] precip physical loss skipped: {_pp_ex}")

            # Sprint 2: HR identifiability loss (optional).
            loss_hr_ident_value = torch.tensor(0.0, device=device)
            if hr_ident_head is not None and beta_hr_ident > 0.0:
                # Use a pooled causal state without the conditioning projection
                # to match the head's input dim.
                with _train_autocast(amp_mode):
                    _pool = H_condition
                    if _pool.dim() == 3:
                        # [q, N, hidden] -> [1, q, hidden]
                        _pool = _pool.mean(dim=1, keepdim=False).unsqueeze(0)
                    try:
                        pred_stats = hr_ident_head(_pool)
                        _head = _eager_core(hr_ident_head)
                        true_stats = _head.extract_target_stats(
                            target.detach(), stats=_head.stats
                        )
                        loss_hr_ident_value = beta_hr_ident * nn.functional.smooth_l1_loss(
                            pred_stats, true_stats
                        )
                    except Exception as _hr_ex:
                        _n, _ = silent_failures["hr_ident"]
                        silent_failures["hr_ident"] = (_n + 1, str(_hr_ex))
                        if verbose and batch_idx == 0:
                            print(f"[WARN] HR ident head skipped: {_hr_ex}")

            # Sprint 4: contrastive DAG-conditioning loss.
            # For every ``contrastive_dag_interval``-th batch we do an extra
            # UNet forward with the dag-token slice of ``conditioning_spatial``
            # zeroed. A margin loss then forces the epsilon prediction from
            # the real-A_dag branch to beat the DAG-blind branch by at least
            # ``contrastive_dag_margin`` MSE points on the diffusion target.
            # Gradient flows only through the real branch (eps_zero is
            # produced under ``no_grad``), so the UNet has an unambiguous
            # incentive to route A_dag information into its prediction.
            # Without this term the conditioning_dropout path trains an
            # equally good unconditional branch and the UNet learns to
            # ignore A_dag — which is exactly what the intervention test on
            # the epoch-10 Sprint 2 checkpoint reported (delta ≈ 0.0001%).
            loss_contrastive_dag_value = torch.tensor(0.0, device=device)
            # C1: per-micro DAG sensitivity scalar surfaced in the heartbeat.
            # ``mse_zero - mse_real`` is the always-meaningful diagnostic
            # (positive ⇒ DAG is conditioning the UNet); the clamp(margin - …)
            # used by the loss is zero exactly when conditioning is healthy
            # and is therefore a misleading heartbeat metric.
            _dag_sens_step: Optional[float] = None
            _can_contrast = (
                lambda_contrastive_dag > 0.0
                and conditioning_spatial is not None
                and spatial_projector is not None
                and not _dropout  # skip when dropout has already nulled the conditioning
                and batch_idx % max(1, contrastive_dag_interval) == 0
                # Only fire once per outer batch (cf. the same fix on
                # precip_phy above). Without this, batch 1 was triggering
                # 48 contrastive passes (2 extra UNet forwards each) on
                # top of 48 main forwards, blowing past 10 minutes.
                and micro_idx == 0
            )
            if _can_contrast:
                _sp_core = _eager_core(spatial_projector)
                if hasattr(_sp_core, "num_dag_tokens"):
                    try:
                        with _train_autocast(amp_mode):
                            _nT = int(_sp_core.num_dag_tokens)
                            cond_spatial_noDAG = conditioning_spatial.clone()
                            cond_spatial_noDAG[:, -_nT:, :] = 0.0

                            _valid_c = torch.isfinite(target)
                            _target_c = torch.where(
                                _valid_c, target, torch.zeros_like(target)
                            )
                            _eps_c = torch.randn_like(_target_c)
                            _bs_c = _target_c.shape[0]
                            _ts_c = torch.randint(
                                0,
                                _diff_unwrapped.scheduler.num_train_timesteps,
                                (_bs_c,),
                                device=device,
                                dtype=torch.long,
                            )
                            _noisy_c = _diff_unwrapped.scheduler.add_noise(
                                _target_c, _eps_c, _ts_c
                            )

                            # Real-A_dag prediction (gradient flows through
                            # UNet + dag_mlp + A_dag).
                            eps_real = _diff_unwrapped.forward(
                                _noisy_c,
                                _ts_c,
                                conditioning,
                                conditioning_spatial=conditioning_spatial,
                            )
                            # DAG-blind prediction (detached reference).
                            with torch.no_grad():
                                eps_zero = _diff_unwrapped.forward(
                                    _noisy_c,
                                    _ts_c,
                                    conditioning,
                                    conditioning_spatial=cond_spatial_noDAG,
                                )
                            _vc = _valid_c.to(dtype=eps_real.dtype)
                            _denom = _vc.sum().clamp(min=1.0)
                            mse_real = (((eps_real - _eps_c) ** 2) * _vc).sum() / _denom
                            mse_zero = (((eps_zero - _eps_c) ** 2) * _vc).sum() / _denom
                            # Margin loss: we want mse_zero - mse_real >= margin,
                            # i.e. the real-A_dag prediction must beat the
                            # DAG-blind one by at least ``margin`` on the
                            # diffusion target. If already satisfied → 0.
                            loss_contrastive_dag_value = (
                                lambda_contrastive_dag
                                * torch.clamp(
                                    contrastive_dag_margin - (mse_zero - mse_real),
                                    min=0.0,
                                )
                            )
                        # Diagnostic: raw sensitivity of the UNet to the DAG
                        # tokens, independent of the margin clamp. A positive
                        # value means the UNet's epsilon prediction is *more*
                        # accurate with the real DAG tokens than with them
                        # zeroed — i.e. the DAG is conditioning. Track this
                        # so the per-epoch log shows the actual conditioning
                        # health (loss_contrastive can be 0 either because
                        # the margin is met or because lambda is 0).
                        _dag_sens_step = float((mse_zero - mse_real).item())
                        total_dag_sensitivity += _dag_sens_step
                        num_contrastive_steps += 1
                    except Exception as _c_ex:
                        _n, _ = silent_failures["contrastive_dag"]
                        silent_failures["contrastive_dag"] = (_n + 1, str(_c_ex))
                        if verbose and batch_idx == 0:
                            print(
                                f"[WARN] contrastive DAG loss skipped: {_c_ex}"
                            )

            # Phase C1 + Sprint 4 fix: Compute total loss.
            #
            # Per-micro losses (gen, rec, dag, phy) accumulate ``Σ_µ(loss * 1/N)``
            # = mean over micro-batches, which is the standard gradient-accumulation
            # contract. But ``loss_precip_phy`` and ``loss_contrastive_dag`` are
            # gated to fire only at ``micro_idx == 0`` (to avoid 48× extra UNet
            # forwards on CPU). Without compensation, the ``* 1/N`` scale at
            # backward time would make their effective lambda ``λ/N`` — i.e.
            # 48× weaker than the same lambda meant under ``batch_size=1``.
            # Multiplying by ``_grad_comp = len(batches)`` here pre-cancels that
            # division so the gated losses contribute their full nominal weight.
            # ``loss_*_value`` itself stays unchanged for logging purposes.
            _grad_comp = float(len(batches))
            with _train_autocast(amp_mode):
                loss_total = (
                    loss_gen_value
                    + loss_rec_value
                    + loss_dag_value
                    + loss_phy_value
                    + loss_hr_ident_value
                    + loss_precip_phy_value * _grad_comp
                    + loss_contrastive_dag_value * _grad_comp
                )

            # Check for NaN or Inf
            if not torch.isfinite(loss_total):
                print(f"[WARN] Batch {batch_idx + 1} has invalid loss!")
                print(f"   - Loss total: {loss_total.item()}")
                print(f"   - Loss gen: {loss_gen_value.item()}")
                print(f"   - Loss rec: {loss_rec_value.item()}")
                print(f"   - Loss DAG: {loss_dag_value.item()}")
                if torch.isnan(loss_total):
                    print(f"   - Loss is NaN, skipping batch")
                    continue
                elif torch.isinf(loss_total):
                    print(f"   - Loss is Inf, skipping batch")
                    continue

            # Accumulate for logging (average over micro-batches)
            step_loss_total += loss_total.item()
            step_loss_gen += loss_gen_value.item()
            step_loss_rec += loss_rec_value.item()
            step_loss_dag += loss_dag_value.item()
            step_loss_phy += loss_phy_value.item()
            # Sprint 4: track contrastive DAG loss separately.
            if isinstance(loss_contrastive_dag_value, torch.Tensor):
                total_contrastive += float(loss_contrastive_dag_value.item())

            # Phase C1: Mixed Precision - Backward pass (scaled for gradient accumulation)
            # DDP: skip gradient sync on non-last micro-batches (no_sync) for faster accumulation
            if _do_timing:
                backward_time = time.time()
            scale = 1.0 / len(batches)  # Scale gradients for accumulation
            is_last_micro = (micro_idx == len(batches) - 1)
            ctx_enc = encoder.no_sync() if (isinstance(encoder, DDP) and not is_last_micro) else nullcontext()
            ctx_rcn = rcn_runner.cell.no_sync() if (isinstance(rcn_runner.cell, DDP) and not is_last_micro) else nullcontext()
            ctx_diff = diffusion_decoder.no_sync() if (isinstance(diffusion_decoder, DDP) and not is_last_micro) else nullcontext()
            _micro_t0 = time.time()
            with ctx_enc, ctx_rcn, ctx_diff:
                if amp_mode == "cuda_fp16":
                    scaler.scale(loss_total * scale).backward()
                else:
                    (loss_total * scale).backward()
            if _do_timing:
                backward_time = time.time() - backward_time
            # Per-micro-batch heartbeat so the user can see progress
            # inside batch_size>1 gradient-accumulation steps. Without
            # this, batch_idx==0 with batches=48 was completely silent
            # for 10+ minutes. Cheap (one print per micro_idx).
            if verbose and len(batches) > 1 and (
                batch_idx == 0
                or batch_idx % max(1, log_interval) == 0
            ):
                _mb_dt = time.time() - _micro_t0
                _suffix = ""
                # C1: surface DAG sensitivity (positive ⇒ DAG is conditioning),
                # which is informative regardless of whether the margin is met.
                # Falls back to nothing when the contrastive block didn't fire
                # this micro (gated to micro_idx==0 + interval).
                if _dag_sens_step is not None:
                    _suffix += f" | dag_sens={_dag_sens_step:+.4f}"
                if loss_precip_phy_value.item() > 0.0:
                    _suffix += f" | precip={loss_precip_phy_value.item():.4f}"
                print(
                    f"   micro {micro_idx + 1}/{len(batches)} "
                    f"loss={loss_total.item():.4f} bwd={_mb_dt:.2f}s{_suffix}",
                    flush=True,
                )

        if gradient_clipping is not None:
            clip_time = time.time()
            if amp_mode == "cuda_fp16":
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                grad_norm_rcn = torch.nn.utils.clip_grad_norm_(rcn_runner.cell.parameters(), gradient_clipping)
                grad_norm_diff = torch.nn.utils.clip_grad_norm_(diffusion_decoder.parameters(), gradient_clipping)
                grad_norm_enc = torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clipping)
            else:
                grad_norm_rcn = torch.nn.utils.clip_grad_norm_(rcn_runner.cell.parameters(), gradient_clipping)
                grad_norm_diff = torch.nn.utils.clip_grad_norm_(diffusion_decoder.parameters(), gradient_clipping)
                grad_norm_enc = torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clipping)
            clip_time = time.time() - clip_time
            
            # Vérifier les gradients après clipping pour détecter les NaN
            nan_grads_found = False
            if batch_idx % 50 == 0:
                for name, param in encoder.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"[ERROR] NaN gradient detected in encoder.{name}")
                        nan_grads_found = True
                for name, param in rcn_runner.cell.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"[ERROR] NaN gradient detected in rcn.{name}")
                        nan_grads_found = True
                for name, param in diffusion_decoder.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"[ERROR] NaN gradient detected in diffusion.{name}")
                        nan_grads_found = True
            
            if nan_grads_found:
                print(f"[WARN] NaN gradients detected after clipping - this may indicate model divergence")
            
            if verbose and (batch_idx % log_interval == 0 or batch_idx == 0):
                print(f"   - Gradient norms (clipped): RCN={grad_norm_rcn:.4f}, Diff={grad_norm_diff:.4f}, Enc={grad_norm_enc:.4f}")

        # Phase C1: Mixed Precision - Optimizer step with scaler (CUDA only)
        if amp_mode == "cuda_fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Phase DAG-decouple: hard acyclicity projection on A_dag. This is a
        # post-hoc rescaling — it never enters the loss, so it cannot fight
        # the diffusion objective in gradient space, yet guarantees the DAG
        # stays strictly acyclic (rho(A ∘ A) <= max_radius < 1, which is the
        # sufficient condition for the DAGMA M-matrix to stay positive
        # definite). Combined with a small ``gamma_dag``, this gives a
        # "strong constraint on the DAG without penalising the stats".
        if dag_spectral_projection:
            _rcn = _eager_core(rcn_runner.cell)
            _rcn.project_dag_spectral(max_radius=dag_spectral_max_radius)

        step_time = time.time() - batch_start_time

        n_micro = len(batches)
        total_loss += step_loss_total
        total_gen += step_loss_gen
        total_rec += step_loss_rec
        total_dag += step_loss_dag
        total_phy += step_loss_phy
        num_batches += 1
        
        # Logging (show average loss over micro-batches when batch_size > 1)
        if verbose and (batch_idx % log_interval == 0 or batch_idx == 0):
            print(f"\nBatch {batch_idx + 1}" + (f" (n={n_micro})" if n_micro > 1 else "") + ":")
            print(f"   - Loss total: {step_loss_total/n_micro:.6f}")
            print(f"   - Loss gen: {step_loss_gen/n_micro:.6f}")
            print(f"   - Loss rec: {step_loss_rec/n_micro:.6f}")
            print(f"   - Loss DAG: {step_loss_dag/n_micro:.6f}")
            if lambda_phy > 0.0:
                print(f"   - Loss phy: {step_loss_phy/n_micro:.6f}")
            print(f"   - Batch time: {step_time:.4f}s")
            if batch_idx == 0:
                try:
                    print(f"   - Time breakdown: Enc={encoder_time:.3f}s, RCN={rcn_time:.3f}s, "
                          f"Diff={diffusion_time:.3f}s, Backward={backward_time:.3f}s")
                    if gradient_clipping is not None:
                        print(f"   - Clip time: {clip_time:.3f}s")
                except NameError:
                    pass

    epoch_time = time.time() - epoch_start_time
    
    if num_batches == 0:
        if verbose:
            print("\n[WARN] No batches were processed in this epoch!")
        return {"loss": 0.0, "loss_gen": 0.0, "loss_rec": 0.0, "loss_dag": 0.0}

    avg_loss = total_loss / num_batches
    avg_gen = total_gen / num_batches
    avg_rec = total_rec / num_batches
    avg_dag = total_dag / num_batches
    avg_phy = total_phy / num_batches
    avg_contrastive = (
        total_contrastive / max(1, num_contrastive_steps)
        if num_contrastive_steps > 0
        else 0.0
    )
    avg_dag_sensitivity = (
        total_dag_sensitivity / max(1, num_contrastive_steps)
        if num_contrastive_steps > 0
        else 0.0
    )
    
    if verbose:
        print(f"\n{'='*80}")
        print("END EPOCH")
        print(f"{'='*80}")
        print("Results:")
        print(f"   - Num batches: {num_batches}")
        print(f"   - Temps total: {epoch_time:.2f}s ({epoch_time/60:.2f} min)")
        print(f"   - Temps moyen par batch: {epoch_time/num_batches:.4f}s")
        print("\nAverage losses:")
        print(f"   - Loss totale: {avg_loss:.6f}")
        print(f"   - Loss génération (diffusion): {avg_gen:.6f}")
        print(f"   - Loss reconstruction: {avg_rec:.6f}")
        print(f"   - Loss DAG ({dag_method.upper()}): {avg_dag:.6f}")
        if lambda_phy > 0.0:
            print(f"   - Loss physique (divergence+vorticité): {avg_phy:.6f}")
        if num_contrastive_steps > 0:
            print(
                f"   - Loss contrastive DAG: {avg_contrastive:.6f} "
                f"({num_contrastive_steps} steps, margin={contrastive_dag_margin})"
            )
            # Health indicator: positive ≈ DAG conditions the UNet, near-zero
            # ≈ UNet ignores the DAG tokens (intervention test will report
            # "non_conditioning"). Watch this trend across epochs.
            _verdict = (
                "DAG CONDITIONS"
                if avg_dag_sensitivity >= contrastive_dag_margin
                else (
                    "DAG WEAKLY CONDITIONS"
                    if avg_dag_sensitivity > 0.0
                    else "DAG IGNORED (warning)"
                )
            )
            print(
                f"   - DAG sensitivity (MSE_zero - MSE_real): "
                f"{avg_dag_sensitivity:+.6f}  → {_verdict}"
            )
        # C4: surface silent failures so they don't accumulate invisibly
        # past the first batch. Each entry is (count, last_message).
        _failed = {k: v for k, v in silent_failures.items() if v[0] > 0}
        if _failed:
            print("\nSilent failures (try/except blocks that swallowed errors):")
            for _name, (_cnt, _msg) in _failed.items():
                print(f"   - {_name}: {_cnt}× — last: {_msg[:200]}")
        print(f"{'='*80}\n")

    result = {
        "loss": avg_loss,
        "loss_gen": avg_gen,
        "loss_rec": avg_rec,
        "loss_dag": avg_dag,
    }
    if lambda_phy > 0.0:
        result["loss_phy"] = avg_phy
    if num_contrastive_steps > 0:
        result["loss_contrastive_dag"] = avg_contrastive
        result["dag_sensitivity"] = avg_dag_sensitivity
        result["num_contrastive_steps"] = float(num_contrastive_steps)
    return result

