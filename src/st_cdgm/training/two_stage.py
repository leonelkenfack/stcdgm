"""
Two-Stage training utilities for ST-CDGM (hyperplan v2.0).

This module provides:

* :func:`stage1_compute_loss` — masked MSE between μ_HR (regression head
  output) and the log1p residual ``log1p(HR) - log1p(baseline)``, plus the
  RCN reconstruction loss and the DAGMA acyclicity penalty (with linear
  warmup over the first epochs). Stage 1 trains: encoder + RCN +
  GraphToGridDecoder.

* :func:`freeze_stage1` / :func:`unfreeze_stage1` — flips ``requires_grad``
  on the Stage 1 modules + sets them to ``eval()``. Used between stages.

* :func:`calibrate_sigma_data_two_stage` — runs the frozen Stage 1 over the
  training set and computes the empirical std of the diffusion residual
  ``δ_target = log1p(HR) - log1p(baseline) - μ_HR`` to recalibrate EDM's
  ``sigma_data``. Mandatory before launching Stage 2.

* :func:`causal_ablation_check` — validates Objective O6 on the regression
  head: compares ``μ_HR(A_dag_real)`` to ``μ_HR(A_dag := 0)`` over a sample
  set. Returns the mean absolute delta and a pass/fail flag.

References
----------
- Mardani et al. 2024, "CorrDiff/ResDiff" (arXiv:2309.15214) — the
  sequential two-stage methodology.
- Karras et al. 2022, "EDM" (arXiv:2206.00364) — sigma_data preconditioning.
- oracle.tex §sec:arch:rcn — DAG detachment requirement preserved.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------
# Stage 1 loss
# ---------------------------------------------------------------------


def gamma_dag_warmup(epoch: int, max_value: float, warmup_epochs: int = 5) -> float:
    """Linear warmup of the DAGMA penalty weight (paper §sec:arch:loss).

    Returns 0 at epoch 0 and ``max_value`` at ``warmup_epochs``. Linear
    in between.
    """
    if epoch >= warmup_epochs:
        return float(max_value)
    return float(max_value) * (epoch / max(1, warmup_epochs))


def lambda_l1_cosine_anneal(
    epoch: int,
    total_epochs: int,
    lambda_start: float = 0.10,
    lambda_end: float = 0.01,
) -> float:
    """Cosine annealing schedule for the DAG L1 sparsity weight.

    Phase B (post-V5-mini, 2026-06-09) — cible la pathologie #3 :
    A_dag observé en V5-mini avec magnitudes uniformes (~0.18 partout,
    Q_phys = 0.40). La L1 constante (λ=0.01) ne permet pas aux arêtes
    survivantes de prendre des magnitudes différenciées.

    Stratégie : démarrer avec une pénalité forte (λ_start = 0.10) qui
    élimine agressivement les arêtes faibles dès les premières epochs,
    puis décroître en cosinus jusqu'à λ_end = 0.01 pour laisser les
    arêtes utiles prendre des magnitudes hiérarchiques.

    Parameters
    ----------
    epoch : int
        Epoch actuelle (0-indexed).
    total_epochs : int
        Nombre total d'epochs du fine-tune.
    lambda_start : float
        Valeur de λ_l1 à epoch 0. Recommandé : 0.10.
    lambda_end : float
        Valeur de λ_l1 à epoch ``total_epochs - 1``. Recommandé : 0.01.

    Returns
    -------
    float : la valeur de λ_l1 à utiliser pour cette epoch.

    Notes
    -----
    - À appeler en début de chaque epoch dans la boucle de training.
    - Le `gradient_clipping=1.0` recommandé en Phase B atténue le
      risque de collapse A_dag → 0 sous lambda_start élevé.
    - Si gamma_dag_warmup_epochs >= 5, la pression purement L1 à
      epoch 0 (gamma_dag = 0) peut être excessive. Mitigation :
      réduire lambda_start à 0.05 ou ramper L1 vers le haut pendant
      le warmup. Voir §12.9 de architecture_journey.md.

    Examples
    --------
    >>> [round(lambda_l1_cosine_anneal(e, 30), 4) for e in [0, 10, 20, 29]]
    [0.1, 0.0775, 0.0325, 0.01]
    """
    if total_epochs <= 1:
        return float(lambda_end)
    t = min(epoch, total_epochs - 1) / (total_epochs - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return float(lambda_end + (lambda_start - lambda_end) * cosine)


# ---------------------------------------------------------------------
# Phase C (post-V5-mini, 2026-06-09) — losses additionnelles Bundle B
# Cible pathologies #1 (tail truncation), #2 (cécité humidité),
# #4 (Q_int=0 / non-CC scaling), #5 (FSS@50mm sous-optimal).
# Voir §12.4 de architecture_journey.md.
# ---------------------------------------------------------------------


def pinball_loss(
    pred: Tensor,
    target: Tensor,
    tau: float,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Pinball (quantile regression) loss au quantile τ ∈ (0, 1).

    Pour τ = 0.95 :
        residual = target - pred
        si target > pred  →  pénalité 0.95 * residual  (sous-estimation forte)
        si target < pred  →  pénalité 0.05 * |residual| (sur-estimation faible)

    Cette asymétrie pousse pred à approcher le quantile τ de la distribution
    conditionnelle de target, plutôt que sa moyenne (que MSE optimise).

    Cible pathologie #1 (V5-mini RX1day bias -6.66 mm, F1-p99 = 0.512).

    Parameters
    ----------
    pred : Tensor
        Prédiction du modèle, ``[B, ...]``.
    target : Tensor
        Cible, même shape que ``pred``.
    tau : float
        Quantile cible. Recommandé : 0.95 ou 0.99 pour les extrêmes.
    valid_mask : Tensor, optional
        Bool mask, True = pixel valide.

    Returns
    -------
    Tensor scalaire — moyenne pondérée par ``valid_mask``.

    References
    ----------
    Koenker & Bassett 1978, *Regression Quantiles*, Econometrica 46.
    """
    residual = target - pred
    loss_per_pixel = torch.where(
        residual >= 0,
        tau * residual,
        (tau - 1.0) * residual,  # = (1-tau) * |residual| for residual < 0
    )
    if valid_mask is None:
        return loss_per_pixel.mean()
    mask_f = valid_mask.float()
    n_valid = mask_f.sum().clamp(min=1.0)
    return (loss_per_pixel * mask_f).sum() / n_valid


def clausius_clapeyron_reg(
    mu_HR: Tensor,
    lr_input: Tensor,
    t850_channel_idx: int,
    *,
    cc_rate: float = 0.07,
    weight: float = 1.0,
    create_graph: bool = False,
) -> Tensor:
    """Régularisateur Clausius-Clapeyron : ∂μ_HR/∂T_850 ≈ 0.07 · μ_HR.

    Loi physique : l'air peut contenir ~7 %/K de vapeur saturante en plus
    (équation de Clausius-Clapeyron sur la pression saturante de vapeur).
    Pour une atmosphère qui se réchauffe à humidité relative constante,
    la précipitation moyenne doit donc augmenter de ~7 %/K (Trenberth 2003).

    Cible pathologie #2 (V5-mini : do(t+3K) donne ~0 réponse, q_*
    sensitivities = 0.05-0.11 vs w_850 = 0.43) et #4 (Q_int = 0).

    Implémentation : calcule le gradient `∂(mean(mu_HR))/∂T_850` via autograd,
    puis pénalise sa MSE par rapport à `cc_rate · mu_HR_detached`.

    **IMPORTANT** : ``lr_input`` doit avoir ``requires_grad=True`` AVANT
    le passage forward qui a produit ``mu_HR``. Sinon retourne 0 silencieusement.

    Parameters
    ----------
    mu_HR : Tensor
        Sortie du Stage 1, ``[B, 1, H, W]``. Doit avoir ``requires_grad=True``.
    lr_input : Tensor
        Input LR utilisé pour produire ``mu_HR``, ``[B, C, H_lr, W_lr]``.
        Doit avoir ``requires_grad=True``.
    t850_channel_idx : int
        Index du canal T_850 dans la dimension canal de ``lr_input``.
    cc_rate : float
        Taux CC en log1p-space. Défaut 0.07 (= 7%/K, valeur physique).
    weight : float
        Poids du terme de loss (déjà appliqué ici).
    create_graph : bool
        Si True, autorise le double-backprop. Coûteux, généralement False.

    Returns
    -------
    Tensor scalaire — pénalité CC pondérée par ``weight``. 0 si autograd fail.

    Notes
    -----
    - Coût compute : +8-12 % par step (un backward additionnel).
    - À encadrer dans try/except si autograd peut échouer (in-place ops,
      compilation torch.compile, etc.) — voir wrapper dans training_loop.
    - Au démarrage du training, garder ``weight=0`` durant 5-10 epochs
      de warmup, puis activer.

    References
    ----------
    Trenberth et al. 2003, *The Changing Character of Precipitation*, BAMS.
    Held & Soden 2006, *Robust Responses of the Hydrological Cycle*, J. Climate.
    """
    if not lr_input.requires_grad:
        return mu_HR.new_zeros(())
    if t850_channel_idx < 0 or t850_channel_idx >= lr_input.size(1):
        raise ValueError(
            f"t850_channel_idx={t850_channel_idx} hors bornes pour lr_input "
            f"de shape {tuple(lr_input.shape)}"
        )

    scalar_mu = mu_HR.mean()
    grad_lr = torch.autograd.grad(
        scalar_mu, lr_input,
        create_graph=create_graph,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if grad_lr is None:
        return mu_HR.new_zeros(())

    grad_t850 = grad_lr[:, t850_channel_idx:t850_channel_idx + 1, :, :]
    if grad_t850.shape[-2:] != mu_HR.shape[-2:]:
        grad_t850 = torch.nn.functional.interpolate(
            grad_t850, size=mu_HR.shape[-2:],
            mode="bilinear", align_corners=False,
        )
    target_grad = cc_rate * mu_HR.detach()
    return weight * ((grad_t850 - target_grad) ** 2).mean()


_HIGH_K_CACHE: dict = {}


def high_k_rapsd_loss(
    pred: Tensor,
    target: Tensor,
    *,
    k_min: int = 30,
    valid_mask: Optional[Tensor] = None,
    eps: float = 1e-8,
) -> Tensor:
    """RAPSD L1 log-ratio loss restreint aux wavenumbers k ≥ k_min.

    Calcule la FFT2D radial des champs pred et target, puis pénalise les
    écarts spectraux en log-space sur la bande haute fréquence (k ≥ k_min).

    Cible pathologie #5 : V5-mini RAPSD ~3× truth à k > 50 (Phase 8.07),
    FSS@50mm = 0.69 vs CorrDiff 0.75 (Phase 9.04). Symptôme : μ_HR
    sur-injecte de l'énergie aux petites échelles (artefacts haute fréquence).

    Parameters
    ----------
    pred, target : Tensor
        Champs ``[B, 1, H, W]``, même shape.
    k_min : int
        Wavenumber minimal à inclure. Défaut 30 (sub-mésoéchelle à
        H × W = 172 × 179, correspond à ~10 km de longueur d'onde).
    valid_mask : Tensor, optional
        Mask de validité. Si fourni, applique avant FFT.
    eps : float
        Régularisation log.

    Returns
    -------
    Tensor scalaire — L1 log-ratio moyennée sur la bande k ≥ k_min.

    Notes
    -----
    - Coût compute : +10-15 % par step (RFFT2D sur ``[B, 1, H, W]``).
    - Le radial wavenumber grid est caché par shape pour économiser
      la ré-allocation (variable module-level ``_HIGH_K_CACHE``).
    - Si k_min trop grand → la fenêtre haute fréquence est vide → retour 0.

    References
    ----------
    Roberts & Lean 2008, *Scale-Selective Verification of Rainfall*, MWR.
    Rampal et al. 2024, *Reliable cGAN downscaling*.
    """
    if pred.dim() != 4 or pred.shape != target.shape:
        raise ValueError(
            f"high_k_rapsd_loss : pred {tuple(pred.shape)} ≠ target {tuple(target.shape)}, "
            "ou pas 4D [B, C, H, W]"
        )

    B, C, H, W = pred.shape
    p = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    if valid_mask is not None:
        vm = valid_mask.float()
        if vm.dim() == 2:
            vm = vm.unsqueeze(0).unsqueeze(0)
        elif vm.dim() == 3:
            vm = vm.unsqueeze(0)
        p = p * vm
        t = t * vm

    F_pred = torch.fft.rfft2(p, norm="ortho")
    F_targ = torch.fft.rfft2(t, norm="ortho")
    amp_pred = F_pred.abs().pow(2)
    amp_targ = F_targ.abs().pow(2)

    # Construction (et cache) du masque radial k ≥ k_min
    cache_key = (H, W, int(k_min), str(pred.device))
    if cache_key not in _HIGH_K_CACHE:
        ky = torch.fft.fftfreq(H, device=pred.device) * H
        kx = torch.fft.rfftfreq(W, device=pred.device) * W
        KY, KX = torch.meshgrid(ky, kx, indexing="ij")
        K = (KY.pow(2) + KX.pow(2)).sqrt()
        mask = (K >= float(k_min)).float().unsqueeze(0).unsqueeze(0)
        _HIGH_K_CACHE[cache_key] = mask
    mask_high_k = _HIGH_K_CACHE[cache_key]

    n_high = mask_high_k.sum().clamp(min=1.0)
    if n_high.item() < 1.5:
        return pred.new_zeros(())

    log_ratio = (
        torch.log(amp_pred * mask_high_k + eps)
        - torch.log(amp_targ * mask_high_k + eps)
    ) * mask_high_k
    return log_ratio.abs().sum() / (B * n_high)


def stage1_compute_loss(
    *,
    mu_HR: Tensor,
    target_residual: Tensor,
    rcn_reconstruction_loss: Optional[Tensor] = None,
    dagma_loss: Optional[Tensor] = None,
    dag_l1_loss: Optional[Tensor] = None,
    dag_prior_loss: Optional[Tensor] = None,
    valid_mask: Optional[Tensor] = None,
    lambda_reg: float = 1.0,
    beta_rec: float = 0.05,
    gamma_dag: float = 0.10,
    lambda_l1: float = 0.01,
    lambda_dag_prior: float = 0.0,
    # >>> V5 — P1 : tail-weighting sur la MSE Stage 1
    tail_weight_target: Optional[Tensor] = None,
    tail_weight_tau: Optional[float] = None,
    tail_weight_alpha: float = 0.0,
    # >>> Phase B (post-V5-mini, 2026-06-09) — exposant power-law
    # Si > 0, supersède la forme relu-au-dessus-de-tau du V5-mini par la
    # forme multiplicative `w = (1 + alpha · y)^beta`. Active uniquement
    # quand tail_weight_beta > 0 et tail_weight_target / tail_weight_tau
    # sont None. Voir §12.3 de architecture_journey.md.
    tail_weight_beta: float = 0.0,
    # >>> V5 — A1 : perte de préservation de la propriété (O3)
    o3_preserve_loss: Optional[Tensor] = None,
    lambda_o3_preserve: float = 0.0,
    # >>> Phase A (post-V5-mini, 2026-06-09) — CASTLE-style joint anchor
    # Force A_dag à représenter les vraies contributions prédictives entre
    # variables — fix le collapse à magnitude uniforme observé en V5-mini.
    # Réf : Kyono 2020 NeurIPS, arXiv:2009.13180. Voir §11.4-12.2 de
    # architecture_journey.md pour la motivation.
    castle_anchor: Optional[nn.Module] = None,
    castle_H_t: Optional[Tensor] = None,
    castle_A_dag: Optional[Tensor] = None,
    lambda_castle: float = 0.0,
    # >>> Phase C (post-V5-mini, 2026-06-09) — Bundle B losses additionnelles
    # Voir §12.4 architecture_journey.md, helpers pinball_loss,
    # clausius_clapeyron_reg, high_k_rapsd_loss définis plus haut.
    # Cible pathologies #1 (tail), #2 (humidité), #4 (Q_int), #5 (FSS@50mm).
    lambda_pinball: float = 0.0,
    pinball_taus: Tuple[float, ...] = (0.95, 0.99),
    cc_reg_loss: Optional[Tensor] = None,
    lambda_cc_reg: float = 0.0,
    lambda_spectral_highk: float = 0.0,
    k_highk_min: int = 30,
) -> Tuple[Tensor, dict]:
    """Stage 1 composite loss.

    L_stage1 = lambda_reg · MSE(mu_HR, target_residual)
              + beta_rec · L_rec
              + gamma_dag · L_dag                           (DAGMA acyclicity)
              + lambda_l1 · L_l1_dag                        (sparsity)
              + lambda_dag_prior · MSE(A_masked, prior)     (anti-collapse)

    The ``dag_prior_loss`` term gives ``A_dag`` a positive supervision
    target (the physically-motivated prior) so it does not collapse to
    zero under the combined L1 + DAGMA centripetal pressure when the
    SCM gradient path is detached. Mirrors the legacy ``train_epoch``
    wiring at ``training_loop.py:1002-1004``.

    Both ``mu_HR`` and ``target_residual`` are expected in log1p space.

    Parameters
    ----------
    mu_HR : Tensor
        Regression head output, ``[B, 1, H, W]``.
    target_residual : Tensor
        ``log1p(HR) - log1p(baseline)``, same shape.
    rcn_reconstruction_loss : Tensor, optional
        Scalar L_rec from the RCN (driver reconstruction).
    dagma_loss : Tensor, optional
        Scalar h_DAGMA(A) acyclicity penalty.
    dag_l1_loss : Tensor, optional
        Scalar ||A||_1 sparsity term.
    valid_mask : Tensor, optional
        Bool mask of finite pixels in ``target_residual`` (handles NaN
        ocean voids identically to the diffusion path).
    lambda_reg, beta_rec, gamma_dag, lambda_l1 : float
        Loss weights. Defaults match paper Tab.2 except ``lambda_reg=1``
        (was implicit in the diffusion path's ``lambda_gen``).

    Returns
    -------
    (loss_total, components) where ``components`` is a dict of scalars
    (Python floats) for logging.
    """
    if valid_mask is None:
        valid_mask = torch.isfinite(target_residual)

    # Replace NaN with 0 to keep autograd safe; mask is the truth source.
    target_clean = torch.where(valid_mask, target_residual, torch.zeros_like(target_residual))

    sq_err = (mu_HR - target_clean) ** 2

    # >>> V5 — P1 : tail-weighting de la MSE Stage 1
    # Multiplie la MSE pixel par pixel par un facteur croissant avec
    # l'intensité reconstruite. Soit on utilise un tenseur de pondération
    # externe (``tail_weight_target``), soit on calcule la pondération
    # depuis ``target_residual`` directement avec un seuil ``tau`` (log1p
    # space) et un coefficient ``alpha`` (multiplicateur additif). Effet
    # courbe en cloche sur ``alpha`` — voir cfg.v5.tail_weight_stage1.
    tail_w = None
    if tail_weight_target is not None:
        # Mode prioritaire : pondération externe (V5-mini compat)
        tail_w = tail_weight_target.detach()
    elif tail_weight_beta > 0.0 and tail_weight_alpha > 0.0:
        # Phase B : forme power-law `w = (1 + alpha · y_pos)^beta`.
        # Upweight TOUS les pixels proportionnellement à leur intensité,
        # pas juste ceux > tau. Cible la pathologie #1 (tail truncation)
        # observée en V5-mini : RX1day bias -6.66 mm, F1-p99 = 0.512.
        # alpha=0.5, beta=1.0 → pixel 10mm pèse 6×, pixel 50mm pèse 26×.
        with torch.no_grad():
            y_pos = target_clean.clamp(min=0.0)
            tail_w = (1.0 + tail_weight_alpha * y_pos).pow(tail_weight_beta)
    elif tail_weight_tau is not None and tail_weight_alpha > 0.0:
        # V5-mini legacy : forme relu-au-dessus-de-tau (forme P1 originale).
        # Préservée pour reproductibilité des runs antérieurs.
        with torch.no_grad():
            tail_w = 1.0 + tail_weight_alpha * torch.relu(target_clean - tail_weight_tau)
    if tail_w is not None:
        sq_err = sq_err * tail_w

    n_valid = valid_mask.float().sum().clamp(min=1.0)
    mse_reg = (sq_err * valid_mask.float()).sum() / n_valid

    components = {"loss_reg": float(mse_reg.detach().item())}

    loss_total = lambda_reg * mse_reg
    if rcn_reconstruction_loss is not None:
        loss_total = loss_total + beta_rec * rcn_reconstruction_loss
        components["loss_rec"] = float(rcn_reconstruction_loss.detach().item())
    if dagma_loss is not None:
        loss_total = loss_total + gamma_dag * dagma_loss
        components["loss_dag"] = float(dagma_loss.detach().item())
    if dag_l1_loss is not None:
        loss_total = loss_total + lambda_l1 * dag_l1_loss
        components["loss_l1"] = float(dag_l1_loss.detach().item())
    if dag_prior_loss is not None and lambda_dag_prior > 0.0:
        loss_total = loss_total + lambda_dag_prior * dag_prior_loss
        components["loss_dag_prior"] = float(dag_prior_loss.detach().item())
    # >>> V5 — A1 : perte de préservation (O3)
    if o3_preserve_loss is not None and lambda_o3_preserve > 0.0:
        loss_total = loss_total + lambda_o3_preserve * o3_preserve_loss
        components["loss_o3_preserve"] = float(o3_preserve_loss.detach().item())

    # >>> Phase A (post-V5-mini) — CASTLE joint prediction anchor
    # Voir CASTLEAnchor dans models/causal_rcn.py. Active uniquement si
    # les 3 arguments castle_* sont fournis et lambda_castle > 0.
    # Coût compute : ~5 % par step pour q=6 nœuds.
    if (
        castle_anchor is not None
        and castle_H_t is not None
        and castle_A_dag is not None
        and lambda_castle > 0.0
    ):
        try:
            castle_loss = castle_anchor(castle_H_t, castle_A_dag)
            loss_total = loss_total + lambda_castle * castle_loss
            components["loss_castle"] = float(castle_loss.detach().item())
        except Exception as e:
            # try/except mandatory : si CASTLE fail (shape mismatch, autograd
            # issue), on log et on continue avec les autres losses pour ne pas
            # bloquer le training. Voir §12.9 risques.
            import warnings
            warnings.warn(f"CASTLE loss skipped : {type(e).__name__}: {e}")
            components["loss_castle"] = float("nan")

    # >>> Phase C (post-V5-mini) — Pinball quantile loss
    # Cible pathologie #1 : MSE seul optimise la moyenne, pas les quantiles
    # extrêmes (queue lourde de la précipitation). Pinball à τ=0.95, 0.99
    # force pred à approcher ces quantiles → réduit RX1day bias.
    if lambda_pinball > 0.0 and len(pinball_taus) > 0:
        pb_losses: List[Tensor] = []
        for tau in pinball_taus:
            pb_losses.append(pinball_loss(mu_HR, target_clean, tau, valid_mask=valid_mask))
        pinball_term = torch.stack(pb_losses).mean()
        loss_total = loss_total + lambda_pinball * pinball_term
        components["loss_pinball"] = float(pinball_term.detach().item())

    # >>> Phase C (post-V5-mini) — Clausius-Clapeyron regularizer
    # Cible pathologies #2 (cécité humidité) et #4 (Q_int=0).
    # Le cc_reg_loss doit être pré-calculé par le caller (training_loop)
    # car il nécessite que lr_input ait requires_grad=True AVANT le forward
    # qui a produit mu_HR. Voir clausius_clapeyron_reg() au-dessus.
    if cc_reg_loss is not None and lambda_cc_reg > 0.0:
        loss_total = loss_total + lambda_cc_reg * cc_reg_loss
        components["loss_cc_reg"] = float(cc_reg_loss.detach().item())

    # >>> Phase C (post-V5-mini) — High-k spectral loss
    # Cible pathologie #5 : RAPSD overshoot à k>30, FSS@50mm sous-optimal.
    # Pénalise les écarts spectraux log-ratio sur la bande haute fréquence.
    if lambda_spectral_highk > 0.0:
        try:
            spec_loss = high_k_rapsd_loss(
                mu_HR, target_clean,
                k_min=k_highk_min,
                valid_mask=valid_mask.float() if valid_mask is not None else None,
            )
            loss_total = loss_total + lambda_spectral_highk * spec_loss
            components["loss_spec_highk"] = float(spec_loss.detach().item())
        except Exception as e:
            import warnings
            warnings.warn(f"High-k spectral loss skipped : {type(e).__name__}: {e}")
            components["loss_spec_highk"] = float("nan")

    components["loss_total"] = float(loss_total.detach().item())
    return loss_total, components


# ---------------------------------------------------------------------
# Stage 1 → Stage 2 transition
# ---------------------------------------------------------------------


def freeze_stage1(*modules: nn.Module) -> None:
    """Set all parameters of the given modules to ``requires_grad=False``
    and switch them to ``eval()``.

    Used at the start of Stage 2 to make the diffusion training target
    stationary. The regression head, RCN cell, and encoder must all be
    frozen so that ``μ_HR`` and the residual decomposition do not drift
    while the diffusion U-Net learns.

    Idempotent: calling on already-frozen modules is a no-op.
    """
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()


def unfreeze_stage1(*modules: nn.Module) -> None:
    """Inverse of :func:`freeze_stage1`. Re-enables training on the
    Stage 1 modules. Used for Ablation A (let stage 2 fine-tune through)
    or for resumption from a partial run."""
    for m in modules:
        if m is None:
            continue
        for p in m.parameters():
            p.requires_grad_(True)
        m.train()


@contextmanager
def stage1_inference_mode(*modules: nn.Module):
    """Context manager: temporarily put modules in ``eval()`` and wrap
    in ``torch.no_grad()``. Useful when computing μ_HR for Stage 2 forward
    passes (not for training Stage 1)."""
    saved = []
    for m in modules:
        if m is None:
            continue
        saved.append((m, m.training))
        m.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for m, was_training in saved:
            if was_training:
                m.train()


# ---------------------------------------------------------------------
# σ_data calibration
# ---------------------------------------------------------------------


@torch.no_grad()
def calibrate_sigma_data_two_stage(
    *,
    encoder: nn.Module,
    rcn_runner,  # RCNSequenceRunner — typed loosely to avoid circular import
    regression_head: nn.Module,
    data_loader: Iterable,
    iterate_batches_fn,  # callable(loader, builder, device) -> generator
    builder,
    device: torch.device,
    max_samples: int = 200,
    verbose: bool = True,
) -> dict:
    """Compute the empirical std of the *Stage 2 diffusion target*
    ``δ_target = log1p(HR) - log1p(baseline) - μ_HR`` over up to
    ``max_samples`` samples.

    This must run AFTER Stage 1 has converged (or hit early stop) and
    BEFORE Stage 2 begins. The returned ``sigma_data`` value is used to
    re-instantiate the EDMConfig (Karras 2022 Eq. 7).

    Returns
    -------
    dict with keys:
        sigma_data : float (recommended new value)
        mean       : float (residual mean, should be ~0)
        std        : float (= sigma_data)
        min, max   : float (extremes for sanity check)
        n_pixels   : int (sample count used for std)
    """
    encoder.eval()
    regression_head.eval()
    if hasattr(rcn_runner, "cell"):
        rcn_runner.cell.eval()

    # Welford for numerical stability over many pixels
    n = 0
    mean = 0.0
    M2 = 0.0
    minimum = float("inf")
    maximum = float("-inf")
    samples_seen = 0

    for converted_batches in iterate_batches_fn(data_loader, builder, device):
        for batch in converted_batches:
            if samples_seen >= max_samples:
                break

            lr_data = batch["lr"].to(device)
            target_residual = batch["residual"][-1].to(device)  # log1p space
            if target_residual.dim() == 3:
                target_residual = target_residual.unsqueeze(0)

            H_init = encoder.init_state(batch["hetero"]).to(device)
            drivers = [lr_data[t] for t in range(lr_data.shape[0])]
            seq_out = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
            H_T = seq_out.states[-1]

            mu_HR = regression_head(H_T)
            if mu_HR.shape != target_residual.shape:
                # Resize defensively (CNN output shapes may vary by pixel)
                mu_HR = F.interpolate(
                    mu_HR, size=target_residual.shape[-2:],
                    mode="bilinear", align_corners=False,
                )

            delta_target = target_residual - mu_HR
            valid = torch.isfinite(delta_target)
            flat = delta_target[valid].detach().cpu().to(torch.float64).numpy().ravel()

            if flat.size > 0:
                # Subsample for speed (every 1024-th pixel)
                for x in flat[::1024]:
                    n += 1
                    delta = x - mean
                    mean += delta / n
                    M2 += delta * (x - mean)
                minimum = float(min(minimum, flat.min()))
                maximum = float(max(maximum, flat.max()))

            samples_seen += 1

            if verbose and (samples_seen % 25 == 0):
                _running_std = (M2 / max(n - 1, 1)) ** 0.5 if n > 1 else 0.0
                print(
                    f"  calibrate σ_data: {samples_seen}/{max_samples} samples, "
                    f"running σ ≈ {_running_std:.5f}",
                    flush=True,
                )

        if samples_seen >= max_samples:
            break

    if n < 2:
        raise RuntimeError(
            "calibrate_sigma_data_two_stage: not enough valid pixels — "
            "verify pipeline and Stage 1 outputs."
        )

    var = M2 / (n - 1)
    std = float(var ** 0.5)

    if verbose:
        print(f"\n📐 calibrate σ_data summary")
        print(f"   pixels examined : {n}")
        print(f"   residual mean   : {mean:+.6f}")
        print(f"   residual std    : {std:.6f}")
        print(f"   min / max       : {minimum:+.4f} / {maximum:+.4f}")
        print(f"   → σ_data = {std:.6f} (was Stage 1 init)")

    return {
        "sigma_data": std,
        "mean": float(mean),
        "std": std,
        "min": minimum,
        "max": maximum,
        "n_pixels": int(n),
    }


# ---------------------------------------------------------------------
# Causal ablation check (Objective O6 gate)
# ---------------------------------------------------------------------


@torch.no_grad()
def causal_ablation_check(
    *,
    encoder: nn.Module,
    rcn_runner,
    rcn_cell: nn.Module,  # the underlying RCNCell with .A_dag
    regression_head: nn.Module,
    data_loader: Iterable,
    iterate_batches_fn,
    builder,
    device: torch.device,
    n_samples: int = 100,
    threshold: float = 0.05,
    verbose: bool = True,
) -> dict:
    """Validate (O3): μ_HR(A_dag) ≠ μ_HR(A_dag := 0).

    For each of ``n_samples`` LR drivers, run the full encoder + RCN +
    regression_head twice — once with the learned A_dag and once with
    A_dag temporarily zeroed out. Compute the mean absolute delta in
    log1p space.

    Convention: the ratio ``Δ / signal = mean(|μ_real - μ_zero|) /
    mean(|μ_real|)`` measures how much the causal DAG matters relative
    to the magnitude of the prediction itself. Threshold 0.05 is the
    Gemini-recommended floor for "the DAG is effectively load-bearing".

    Returns
    -------
    dict with keys:
        mean_delta, mean_signal, ratio, passes, n_samples
    """
    encoder.eval()
    regression_head.eval()
    rcn_cell.eval()

    deltas: list[float] = []
    signals: list[float] = []
    samples_seen = 0

    A_dag_param = rcn_cell.A_dag

    for converted_batches in iterate_batches_fn(data_loader, builder, device):
        for batch in converted_batches:
            if samples_seen >= n_samples:
                break

            lr_data = batch["lr"].to(device)
            H_init = encoder.init_state(batch["hetero"]).to(device)
            drivers = [lr_data[t] for t in range(lr_data.shape[0])]

            # Real A_dag run
            seq_out_real = rcn_runner.run(
                H_init, drivers, reconstruction_sources=None
            )
            H_T_real = seq_out_real.states[-1]
            mu_real = regression_head(H_T_real)

            # Zero A_dag run — temporarily replace the parameter, then restore
            saved_A = A_dag_param.data.clone()
            try:
                A_dag_param.data.zero_()
                seq_out_zero = rcn_runner.run(
                    H_init, drivers, reconstruction_sources=None
                )
                H_T_zero = seq_out_zero.states[-1]
                mu_zero = regression_head(H_T_zero)
            finally:
                A_dag_param.data.copy_(saved_A)

            delta = (mu_real - mu_zero).abs().mean().item()
            signal = mu_real.abs().mean().item()
            deltas.append(delta)
            signals.append(signal)
            samples_seen += 1

            if verbose and (samples_seen % 20 == 0):
                _ratio = (
                    (sum(deltas) / len(deltas))
                    / max(sum(signals) / len(signals), 1e-12)
                )
                print(
                    f"  ablation : {samples_seen}/{n_samples}, "
                    f"running Δ/sig ≈ {_ratio:.4f}",
                    flush=True,
                )

        if samples_seen >= n_samples:
            break

    mean_delta = sum(deltas) / max(len(deltas), 1)
    mean_signal = sum(signals) / max(len(signals), 1)
    ratio = mean_delta / max(mean_signal, 1e-12)
    passes = ratio > threshold

    if verbose:
        print(f"\n🧪 Causal ablation check (n={samples_seen}):")
        print(f"   mean |μ_real - μ_zero| = {mean_delta:.6f}")
        print(f"   mean |μ_real|          = {mean_signal:.6f}")
        print(f"   ratio Δ/signal         = {ratio:.4f}  (threshold {threshold})")
        print(f"   verdict                = {'✓ PASS' if passes else '✗ FAIL — DAG decorative'}")

    return {
        "mean_delta": float(mean_delta),
        "mean_signal": float(mean_signal),
        "ratio": float(ratio),
        "threshold": float(threshold),
        "passes": bool(passes),
        "n_samples": int(samples_seen),
    }


# ---------------------------------------------------------------------
# BS32b — pre-cache Stage 1 outputs + thin Stage 2 training loop
# ---------------------------------------------------------------------


@torch.no_grad()
def precompute_stage1_outputs(
    *,
    encoder: nn.Module,
    rcn_runner,
    regression_head: nn.Module,
    train_dataset,
    iterate_batches_fn,
    device: torch.device,
    dag_variants: Sequence[str] = ("normal",),
    existing_cache: Optional[dict] = None,
    cache_h_t_pooled: bool = False,
) -> dict:
    """Iterate ``train_dataset`` once, run Stage 1 forward per sample,
    and stack outputs into a dict of CPU tensors.

    Stage 1 modules must be frozen (``requires_grad=False``, ``eval()``).
    Result keys: ``mu_HR``, ``baseline_log``, ``delta_target``,
    ``valid_mask``. Each tensor is shape ``(N, C, H, W)``. ``valid_mask``
    is ``bool`` (``True`` where the target is finite).

    BS35-CONTRASTIVE — ``dag_variants`` lets the caller request additional
    Stage 1 forwards with the RCN ``A_dag`` perturbed at inference time
    (``set_dag_ablation_mode``). For every non-``"normal"`` variant
    ``v`` listed, the result also contains a ``mu_HR_<v>`` tensor of the
    same shape. ``"normal"`` is always materialized as ``mu_HR`` and
    drives ``delta_target = target - mu_HR``.

    ``existing_cache``: if provided and already contains a variant
    (``mu_HR`` for ``"normal"`` or ``mu_HR_<v>`` otherwise), that
    variant is **not recomputed** — the cached tensors are reused.
    This lets a contrastive run extend a legacy cache (only ``mu_HR``)
    by computing solely the missing ablated variants.

    BS32b — eliminates per-batch Stage 1 forward (encoder + 16-step RCN
    + regression_head) which on the production training loop is run
    32×-64× per logical batch and re-runs the same deterministic forward
    every epoch. Pre-caching collapses 14k×N_epochs forwards to 14k×1.

    Parameters
    ----------
    encoder, rcn_runner, regression_head : torch.nn.Module
        Frozen Stage 1 modules.
    train_dataset : torch.utils.data.IterableDataset (or anything iterable)
        Yields the same dict samples the legacy DataLoader yielded.
    iterate_batches_fn : callable(sample, builder, device) -> dict
        The notebook's ``convert_sample_to_batch`` (we accept it as a
        callable rather than importing — it's defined per-notebook).
    device : torch.device
        Where Stage 1 forwards run.

    Returns
    -------
    dict with keys ``mu_HR``, ``baseline_log``, ``delta_target``,
    ``valid_mask`` — all stacked CPU tensors of shape ``(N, C, H, W)``.
    """
    encoder.eval()
    if hasattr(rcn_runner, "cell"):
        rcn_runner.cell.eval()
    regression_head.eval()

    rcn_cell = rcn_runner.cell if hasattr(rcn_runner, "cell") else None
    can_ablate = rcn_cell is not None and hasattr(rcn_cell, "set_dag_ablation_mode")
    # V6 MVP — if cache_h_t_pooled, we additionally store H_T_pooled = mean(H_T, dim=N)
    # → shape [q, hidden] per sample, ~few KB per sample.
    # Used by Stage 2 r_phi(H) StructuredResidualHead (F4 dette critique).

    variants: list[str] = []
    seen: set[str] = set()
    for v in dag_variants:
        v = str(v).lower()
        if v not in seen:
            variants.append(v)
            seen.add(v)
    if "normal" not in variants:
        variants.insert(0, "normal")

    def _key_for(v: str) -> str:
        return "mu_HR" if v == "normal" else f"mu_HR_{v}"

    out: dict = {}
    if existing_cache is not None:
        for k in ("baseline_log", "delta_target", "valid_mask"):
            if k in existing_cache:
                out[k] = existing_cache[k]
        for v in variants:
            k = _key_for(v)
            if k in existing_cache:
                out[k] = existing_cache[k]

    needed_variants = [v for v in variants if _key_for(v) not in out]

    has_targets = all(k in out for k in ("baseline_log", "delta_target", "valid_mask"))
    if not needed_variants and has_targets:
        print(
            f"  precompute Stage 1 : tous les variants {variants!r} "
            f"déjà présents dans le cache → skip",
            flush=True,
        )
        return out

    if not needed_variants and not has_targets:
        needed_variants = ["normal"]

    if not can_ablate and any(v != "normal" for v in needed_variants):
        raise RuntimeError(
            "precompute_stage1_outputs: dag_variants demande une ablation "
            "(non-'normal') mais rcn_runner.cell n'expose pas "
            "set_dag_ablation_mode. Patch BS35 manquant ?"
        )

    import time as _t

    delta_target_for_normal: Optional[list[Tensor]] = None

    for variant in needed_variants:
        if can_ablate:
            try:
                rcn_cell.set_dag_ablation_mode(variant, seed=42)
            except Exception as _e:
                raise RuntimeError(
                    f"precompute_stage1_outputs: échec set_dag_ablation_mode({variant!r}): {_e}"
                ) from _e

        mu_list: list[Tensor] = []
        base_list: list[Tensor] = []
        delta_list: list[Tensor] = []
        mask_list: list[Tensor] = []
        # V6 MVP — H_T_pooled cache for r_phi (only for "normal" variant)
        h_t_pooled_list: list[Tensor] = []

        t0 = _t.time()
        last_print = t0
        n_seen = 0

        for sample in train_dataset:
            batch = iterate_batches_fn(sample)
            lr_data = batch["lr"].to(device)
            target = batch["residual"][-1].to(device)
            if target.dim() == 3:
                target = target.unsqueeze(0)

            baseline_t = batch.get("baseline")
            if baseline_t is not None:
                baseline_t = baseline_t[-1].to(device)
                if baseline_t.dim() == 3:
                    baseline_t = baseline_t.unsqueeze(0)

            H_init = encoder.init_state(batch["hetero"]).to(device)
            drivers = [lr_data[t] for t in range(lr_data.shape[0])]
            seq = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
            H_T = seq.states[-1]
            mu_HR = regression_head(H_T)
            if mu_HR.shape != target.shape:
                mu_HR = F.interpolate(
                    mu_HR, size=target.shape[-2:],
                    mode="bilinear", align_corners=False,
                )

            baseline_log = baseline_t if baseline_t is not None else torch.zeros_like(target)
            valid_mask = torch.isfinite(target)
            delta_target = target - mu_HR

            mu_HR = torch.nan_to_num(mu_HR, nan=0.0, posinf=0.0, neginf=0.0)
            baseline_log = torch.nan_to_num(baseline_log, nan=0.0, posinf=0.0, neginf=0.0)
            delta_target = torch.nan_to_num(delta_target, nan=0.0, posinf=0.0, neginf=0.0)

            mu_list.append(mu_HR.detach().squeeze(0).cpu())
            base_list.append(baseline_log.detach().squeeze(0).cpu())
            delta_list.append(delta_target.detach().squeeze(0).cpu())
            mask_list.append(valid_mask.detach().squeeze(0).cpu())

            # V6 MVP — H_T_pooled (over N nodes) for r_phi(H) auxiliary head.
            # Only stored for "normal" variant.
            # H_T shape : [q, N, hidden] -> pool over N -> [q, hidden]
            if cache_h_t_pooled and variant == "normal":
                h_t_safe = torch.nan_to_num(H_T, nan=0.0, posinf=0.0, neginf=0.0)
                h_pooled = h_t_safe.mean(dim=-2)  # [q, hidden]
                h_t_pooled_list.append(h_pooled.detach().cpu())

            n_seen += 1
            now = _t.time()
            if (now - last_print) >= 5.0:
                print(
                    f"  precompute Stage 1 [{variant}] : {n_seen} samples "
                    f"| {now-t0:.0f}s ({(now-t0)/n_seen:.2f}s/sample)",
                    flush=True,
                )
                last_print = now

        print(
            f"  precompute Stage 1 [{variant}] : {n_seen} samples | "
            f"total {_t.time()-t0:.0f}s "
            f"({(_t.time()-t0)/max(1,n_seen):.2f}s/sample)",
            flush=True,
        )

        out[_key_for(variant)] = (
            torch.stack(mu_list, dim=0) if mu_list else torch.empty(0)
        )
        if variant == "normal":
            out["baseline_log"] = (
                torch.stack(base_list, dim=0) if base_list else torch.empty(0)
            )
            out["delta_target"] = (
                torch.stack(delta_list, dim=0) if delta_list else torch.empty(0)
            )
            out["valid_mask"] = (
                torch.stack(mask_list, dim=0) if mask_list else torch.empty(0)
            )
            delta_target_for_normal = delta_list
            # V6 MVP — stack H_T_pooled if collected
            if cache_h_t_pooled and h_t_pooled_list:
                out["H_T_pooled"] = torch.stack(h_t_pooled_list, dim=0)  # [N, q, hidden]

    if can_ablate:
        try:
            rcn_cell.set_dag_ablation_mode("normal", seed=42)
        except Exception:
            pass

    return out


def train_epoch_stage2_cached(
    *,
    diffusion_decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    cached_dataloader,
    device: torch.device,
    use_amp: bool = True,
    gradient_clipping: Optional[float] = None,
    log_every: int = 20,
    verbose: bool = True,
    lambda_contrastive_dag: float = 0.0,
    contrastive_dag_margin: float = 0.02,
    contrastive_dag_interval: int = 4,
    ablated_mu_key: str = "mu_HR_zero",
    # BS37 (Sprint B / V3) — EMA des poids Stage 2.
    # Si ``ema_model`` est fourni, on met a jour ses parametres apres
    # chaque optimizer.step() avec :
    #   ema_p <- decay * ema_p + (1 - decay) * live_p
    # Standard CorrDiff/EDM : decay 0.9999 (averaging effectif sur ~10000 steps).
    ema_model: Optional[nn.Module] = None,
    ema_decay: float = 0.9999,
    ema_warmup_steps: int = 0,
    # V5 — Track B2 : conditioning_dropout pour CFG-compatibility.
    # Avec proba p le mu_HR est zeroe avant compute_loss_edm, entrainant
    # ainsi la branche "uncond" indispensable pour cfg_scale > 1 a l'inference.
    # CorrDiff utilise 0.13 (cf. Mardani 2024 §4.2). Pas le contrastive_dag
    # qui a sa propre logique d'ablation.
    conditioning_dropout_prob: float = 0.0,
    # V5 — Track C : log FACL + Sliced-W si compute_loss_edm les retourne.
    log_loss_components: bool = True,
    # ======================== V6 MVP additions ============================ #
    # All V6 params are opt-in. If left at defaults, behavior is bit-identical
    # to the 9-node seed 42 protocol (V5_causal). When provided, integrate :
    #   - r_phi(H) auxiliary residual head (F4 dette critique, S1.1 module)
    #   - pinball multi-tau loss (S1.2)
    #   - log-det rank-promoting penalty (S1.2)
    #   - r_phi warmup scheduler (gel N steps + ramp lambda_r 0->max)
    # See V6 plan §2 + §3 garde-fous.
    r_phi_module: Optional[nn.Module] = None,  # StructuredResidualHead, optimized via the same optimizer
    lambda_pinball: float = 0.0,
    pinball_taus: Sequence[float] = (0.5, 0.95, 0.99),
    lambda_logdet: float = 0.0,
    logdet_subsample_indices: Optional[Tensor] = None,
    logdet_delta: float = 1e-3,
    logdet_min_batch_size: int = 128,
    r_phi_freeze_steps: int = 2000,
    r_phi_ramp_steps: int = 5000,
    r_phi_lambda_max: float = 0.20,
    v6_global_step_start: int = 0,
) -> dict:
    """Thin Stage 2 training loop that consumes a pre-cached dataset.

    BS32b — the diffusion forward+backward runs at the *full batch
    size* of the dataloader, not bs=1 in a per-micro loop.

    BS35-CONTRASTIVE — when ``lambda_contrastive_dag > 0`` and the
    batch contains an ablated mu_HR (key ``ablated_mu_key``, default
    ``"mu_HR_zero"``), we add a margin loss every
    ``contrastive_dag_interval`` batches:

        L_c = lambda * max(0, margin - (loss_zero - loss_real))

    where ``loss_real = compute_loss_edm(delta, mu_HR_real, ...)``
    drives the gradient and ``loss_zero = compute_loss_edm(delta,
    mu_HR_dagless, ...)`` is computed under no_grad on the same
    (delta_target, baseline_log, sigma sample) batch. The margin
    forces Stage 2 to perform measurably better when conditioned on
    the causally-informed mu_HR. The diagnostic
    ``dag_sensitivity = loss_zero - loss_real`` is averaged and
    returned in the result dict.

    Expects ``cached_dataloader`` to yield dicts with at least
    ``mu_HR``, ``baseline_log``, ``delta_target``. The contrastive
    branch is silently skipped when the ablated key is missing.
    """
    from st_cdgm.training.training_loop import (
        resolve_train_amp_mode,
        _train_autocast,
    )

    diffusion_decoder.train()
    amp_mode = resolve_train_amp_mode(device, use_amp)
    scaler = torch.amp.GradScaler(enabled=(amp_mode == "cuda_fp16"))

    contrastive_active = bool(lambda_contrastive_dag > 0.0)
    interval = max(1, int(contrastive_dag_interval))
    cond_drop_active = bool(conditioning_dropout_prob > 0.0)
    cond_drop_p = float(conditioning_dropout_prob)

    ema_active = ema_model is not None
    ema_skip_warmup = max(0, int(ema_warmup_steps))
    # V5 FIX F1 (BS41) — ema_step_counter PERSISTANT sur ema_model pour eviter
    # le reset per-epoch qui rendait warmup_steps=1000 inatteignable (141
    # batches/epoch < 1000 => 0 update EMA, le shadow restait fige a son init).
    # On stocke comme attribut Python sur ema_model : conserve entre epochs
    # dans une meme session. Pour la persistance cross-session/checkpoint,
    # voir le block resume EMA dans le notebook.
    if ema_active:
        ema_model.eval()
        for _p in ema_model.parameters():
            _p.requires_grad_(False)
        # Initialise le compteur s'il n'existe pas, sinon on reprend la valeur
        # accumulee aux epochs precedentes.
        if not hasattr(ema_model, "_ema_step_counter"):
            ema_model._ema_step_counter = 0
    ema_steps = getattr(ema_model, "_ema_step_counter", 0) if ema_active else 0
    ema_steps_at_entry = ema_steps

    if verbose:
        print(
            f"\n📚 Stage 2 epoch (cached) | amp={amp_mode} "
            f"| contrastive_dag={'on' if contrastive_active else 'off'}"
            + (
                f" (lambda={lambda_contrastive_dag}, margin={contrastive_dag_margin},"
                f" every {interval} batches, key={ablated_mu_key!r})"
                if contrastive_active
                else ""
            )
            + (
                f" | EMA on (decay={ema_decay}"
                + (
                    f", warmup={ema_skip_warmup}, step_counter={ema_steps_at_entry}"
                    if ema_skip_warmup > 0 else ""
                )
                + ")"
                if ema_active else " | EMA off"
            )
            + (f" | cond_drop p={cond_drop_p}" if cond_drop_active else ""),
            flush=True,
        )

    total_loss = 0.0
    total_facl = 0.0
    total_swd = 0.0
    n_facl_steps = 0
    total_contrastive = 0.0
    total_dag_sensitivity = 0.0
    n_contrastive = 0
    n_batches = 0
    contrastive_skipped_missing_key = 0
    n_cond_dropped = 0

    # ====================== V6 MVP — init metrics ======================== #
    v6_active = (r_phi_module is not None) or (lambda_pinball > 0.0) or (lambda_logdet > 0.0)
    v6_metrics = {
        "total_pinball": 0.0,
        "total_logdet": 0.0,
        "total_r_phi_norm_ratio": 0.0,  # mean ‖r_phi‖₂ / ‖mu_HR‖₂ per batch
        "n_r_phi_batches": 0,
        "lambda_r_last": 0.0,
        "v6_global_step": int(v6_global_step_start),
    }
    if v6_active:
        from st_cdgm.training.queue_losses import (
            pinball_multi_tau as _v6_pinball_multi_tau,
            log_det_rank_penalty as _v6_log_det_rank_penalty,
        )

    for batch_idx, batch in enumerate(cached_dataloader):
        mu_HR = batch["mu_HR"].to(device, non_blocking=True)
        baseline_log = batch["baseline_log"].to(device, non_blocking=True)
        delta_target = batch["delta_target"].to(device, non_blocking=True)
        # V6 MVP — H_T_pooled used by r_phi if present in cache
        h_t_pooled = batch.get("H_T_pooled")
        if h_t_pooled is not None:
            h_t_pooled = h_t_pooled.to(device, non_blocking=True)

        do_contrastive = (
            contrastive_active and (batch_idx % interval == 0)
        )
        mu_HR_ablated: Optional[Tensor] = None
        if do_contrastive:
            if ablated_mu_key in batch:
                mu_HR_ablated = batch[ablated_mu_key].to(device, non_blocking=True)
            else:
                contrastive_skipped_missing_key += 1
                do_contrastive = False

        # >>> V5 — Track B2 : conditioning_dropout pour CFG-compatibility.
        # On droppe mu_HR au niveau du sample (Bernoulli p) avant la loss EDM.
        # baseline_log reste intact car il porte le seul signal LR utile dans
        # la branche uncond. Quand do_contrastive=True le contrastive utilise
        # mu_HR_ablated separement — pas de double-drop.
        #
        # V5 FIX F2 (BS41) — quand on zero mu_HR on doit AUSSI ajuster la cible :
        # delta_target = (HR - baseline) - mu_HR_real. Si on prive le modele
        # de mu_HR_real (entree=0) sans modifier la cible, on lui demande de
        # predire une quantite qui depend de l'inconnue mu_HR_real -> tache
        # imitable + gradients degeneres sur 13% du batch (pollue le live et
        # l'EMA). Le fix : pour les samples drop-es, target_used = delta + mu_HR
        # (= HR - baseline). Le modele apprend alors la VRAIE branche
        # unconditional : "predire le residu apres baseline sans Stage 1".
        # Reference : Ho & Salimans 2022 (CFG) §3.2 — la branche uncond
        # doit predire une cible independante du conditionnement droppe.
        mu_HR_used = mu_HR
        delta_target_used = delta_target
        if cond_drop_active and not do_contrastive:
            B = mu_HR.shape[0]
            mask = (torch.rand(B, device=device) < cond_drop_p).view(B, 1, 1, 1)
            if mask.any():
                mu_HR_used = torch.where(mask, torch.zeros_like(mu_HR), mu_HR)
                # CRITIQUE : ajuste la cible de la branche unconditional.
                delta_target_used = torch.where(
                    mask, delta_target + mu_HR, delta_target
                )
                n_cond_dropped += int(mask.sum().item())

        optimizer.zero_grad(set_to_none=True)

        with _train_autocast(amp_mode):
            _loss_out = diffusion_decoder.compute_loss_edm(
                target=delta_target_used,
                conditioning=None,
                conditioning_spatial=None,
                mu_HR=mu_HR_used,
                baseline_log=baseline_log,
                return_components=log_loss_components,
            )
            if log_loss_components and isinstance(_loss_out, tuple):
                loss_real, loss_components = _loss_out
                facl_val = loss_components.get("facl", None)
                swd_val = loss_components.get("swd", None)
                if facl_val is not None and torch.is_tensor(facl_val):
                    total_facl += float(facl_val.item())
                if swd_val is not None and torch.is_tensor(swd_val):
                    total_swd += float(swd_val.item())
                n_facl_steps += 1
            else:
                loss_real = _loss_out

            loss_contrast_value = torch.tensor(
                0.0, device=device, dtype=loss_real.dtype
            )
            dag_sens_step: Optional[float] = None

            if do_contrastive and mu_HR_ablated is not None:
                with torch.no_grad():
                    loss_zero = diffusion_decoder.compute_loss_edm(
                        target=delta_target,
                        conditioning=None,
                        conditioning_spatial=None,
                        mu_HR=mu_HR_ablated,
                        baseline_log=baseline_log,
                    )
                margin_t = torch.as_tensor(
                    contrastive_dag_margin,
                    device=device,
                    dtype=loss_real.dtype,
                )
                gap = loss_zero.detach() - loss_real
                loss_contrast_value = lambda_contrastive_dag * torch.clamp(
                    margin_t - gap, min=0.0
                )
                dag_sens_step = float((loss_zero - loss_real).detach().item())
                total_dag_sensitivity += dag_sens_step
                total_contrastive += float(loss_contrast_value.detach().item())
                n_contrastive += 1

            loss_total = loss_real + loss_contrast_value

            # ================== V6 MVP — pinball + log-det + r_phi =========
            if v6_active and log_loss_components and isinstance(_loss_out, tuple):
                _, _v6_components = _loss_out
                D_y = _v6_components.get("D_y")
                target_clean_v6 = _v6_components.get("target_clean")
                v_mask = _v6_components.get("valid_mask")

                # --- r_phi contribution with warmup schedule ---------------
                r_phi_pred: Optional[Tensor] = None
                lam_r_eff = 0.0
                if r_phi_module is not None and h_t_pooled is not None:
                    step = v6_metrics["v6_global_step"]
                    if step < r_phi_freeze_steps:
                        # Freeze : compute r_phi but mask it to 0 contribution
                        # (still want gradients flowing? No — fully frozen)
                        with torch.no_grad():
                            r_phi_pred = r_phi_module(h_t_pooled)
                        lam_r_eff = 0.0
                    else:
                        r_phi_pred = r_phi_module(h_t_pooled)
                        # Linear ramp 0 -> r_phi_lambda_max over r_phi_ramp_steps
                        ramp_frac = float(step - r_phi_freeze_steps) / max(1, r_phi_ramp_steps)
                        lam_r_eff = min(r_phi_lambda_max, r_phi_lambda_max * ramp_frac)
                    v6_metrics["lambda_r_last"] = float(lam_r_eff)

                # --- Pinball loss on D_y (predicts delta_target) -----------
                if lambda_pinball > 0.0 and D_y is not None and target_clean_v6 is not None:
                    # Optionally include r_phi in the prediction
                    pred_for_pinball = D_y
                    if r_phi_pred is not None and lam_r_eff > 0.0:
                        pred_for_pinball = D_y + lam_r_eff * r_phi_pred
                    loss_pinball = _v6_pinball_multi_tau(
                        pred_for_pinball, target_clean_v6, taus=pinball_taus
                    )
                    loss_total = loss_total + lambda_pinball * loss_pinball
                    v6_metrics["total_pinball"] += float(loss_pinball.detach().item())

                # --- Log-det rank penalty on residual = (D_y + r_phi - delta_target)
                if (
                    lambda_logdet > 0.0
                    and D_y is not None
                    and target_clean_v6 is not None
                    and logdet_subsample_indices is not None
                    and D_y.shape[0] >= logdet_min_batch_size
                ):
                    pred_full = D_y
                    if r_phi_pred is not None and lam_r_eff > 0.0:
                        pred_full = D_y + lam_r_eff * r_phi_pred
                    residual_v6 = pred_full - target_clean_v6
                    loss_logdet = _v6_log_det_rank_penalty(
                        residual_v6,
                        logdet_subsample_indices.to(D_y.device),
                        delta=logdet_delta,
                    )
                    loss_total = loss_total + lambda_logdet * loss_logdet
                    v6_metrics["total_logdet"] += float(loss_logdet.detach().item())

                # --- Monitoring : ‖r_phi‖ / ‖mu_HR‖ ratio per batch --------
                if r_phi_pred is not None:
                    with torch.no_grad():
                        rn = float(r_phi_pred.norm(p=2).item())
                        mn = float(mu_HR_used.norm(p=2).item())
                        if mn > 1e-8:
                            v6_metrics["total_r_phi_norm_ratio"] += rn / mn
                            v6_metrics["n_r_phi_batches"] += 1

        if amp_mode == "cuda_fp16":
            scaler.scale(loss_total).backward()
        else:
            loss_total.backward()

        if gradient_clipping is not None and gradient_clipping > 0:
            if amp_mode == "cuda_fp16":
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                diffusion_decoder.parameters(), gradient_clipping
            )

        if amp_mode == "cuda_fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # BS37 — EMA update (in-place on ema_model parameters AND buffers).
        # Buffers (running stats des GroupNorm si non-affine, etc.) sont
        # copies tels quels du live model, pas moyennes — c'est la convention
        # Karras EDM2 (Karras et al. 2024 Appendix B).
        # V5 FIX F1 (BS41) — ema_steps est maintenant le compteur GLOBAL repris
        # depuis ema_model._ema_step_counter (persiste entre epochs). warmup
        # est respecte si total accumule > seuil.
        if ema_active and (ema_steps >= ema_skip_warmup or ema_skip_warmup == 0):
            with torch.no_grad():
                for ep, lp in zip(
                    ema_model.parameters(), diffusion_decoder.parameters()
                ):
                    ep.data.mul_(ema_decay).add_(lp.data, alpha=1.0 - ema_decay)
                for eb, lb in zip(
                    ema_model.buffers(), diffusion_decoder.buffers()
                ):
                    eb.data.copy_(lb.data)
        if ema_active:
            ema_steps += 1

        total_loss += float(loss_real.detach().item())
        n_batches += 1

        # V6 MVP — increment global step (used by r_phi warmup schedule)
        if v6_active:
            v6_metrics["v6_global_step"] += 1

        if verbose and (batch_idx == 0 or (batch_idx + 1) % log_every == 0):
            extra = ""
            if dag_sens_step is not None:
                extra = (
                    f" | L_contrast={float(loss_contrast_value.detach()):.5f}"
                    f" | dag_sens={dag_sens_step:+.5f}"
                )
            print(
                f"  S2 batch {batch_idx + 1} | loss_diff={float(loss_real.detach()):.5f}{extra}",
                flush=True,
            )

    # V5 FIX F1 (BS41) — persiste le compteur EMA sur l'objet ema_model pour
    # que le prochain appel reprenne ce nombre cumule (au lieu de remettre a 0).
    if ema_active:
        ema_model._ema_step_counter = int(ema_steps)
        ema_updates_this_epoch = max(0, ema_steps - max(ema_steps_at_entry, ema_skip_warmup))

    avg_dag_sens = (
        total_dag_sensitivity / max(1, n_contrastive) if n_contrastive > 0 else 0.0
    )
    avg_contrast = (
        total_contrastive / max(1, n_contrastive) if n_contrastive > 0 else 0.0
    )
    avg_facl = total_facl / max(1, n_facl_steps) if n_facl_steps > 0 else 0.0
    avg_swd = total_swd / max(1, n_facl_steps) if n_facl_steps > 0 else 0.0

    if verbose and (avg_facl > 0 or avg_swd > 0):
        print(
            f"  V5 loss components : avg_FACL={avg_facl:.5f} | "
            f"avg_SW1={avg_swd:.5f}",
            flush=True,
        )
    if verbose and cond_drop_active and n_cond_dropped > 0:
        print(
            f"  cond_drop : {n_cond_dropped} mu_HR samples zeroed (p={cond_drop_p})",
            flush=True,
        )

    if verbose and contrastive_active:
        if n_contrastive == 0:
            warn = ""
            if contrastive_skipped_missing_key > 0:
                warn = (
                    f" (⚠ {contrastive_skipped_missing_key} batches skipped: "
                    f"clé {ablated_mu_key!r} absente du batch)"
                )
            print(f"  contrastive_dag : aucun pas exécuté{warn}", flush=True)
        else:
            verdict = (
                "OK (DAG conditionne S2)"
                if avg_dag_sens >= contrastive_dag_margin
                else (
                    "marge non atteinte"
                    if avg_dag_sens > 0.0
                    else "DAG ignoré par S2"
                )
            )
            print(
                f"  contrastive_dag : {n_contrastive} steps | "
                f"avg_loss_contrast={avg_contrast:.5f} | "
                f"avg_dag_sensitivity={avg_dag_sens:+.5f} → {verdict}",
                flush=True,
            )

    out_metrics = {
        "loss_diff": total_loss / max(1, n_batches),
        "n_batches": n_batches,
        "loss_contrastive_dag": avg_contrast,
        "dag_sensitivity": avg_dag_sens,
        "n_contrastive_steps": n_contrastive,
        "ema_active": ema_active,
        "ema_steps": ema_steps,
        "ema_updates_this_epoch": int(ema_updates_this_epoch) if ema_active else 0,
        "ema_decay": ema_decay if ema_active else None,
        # V5 — Track C : loss components for monitoring.
        "loss_facl": avg_facl,
        "loss_swd": avg_swd,
        # V5 — Track B2 : conditioning_dropout stats.
        "cond_drop_active": cond_drop_active,
        "cond_drop_prob": cond_drop_p,
        "n_samples_cond_dropped": n_cond_dropped,
    }
    # V6 MVP — append V6 metrics (only meaningful if v6_active)
    if v6_active:
        nb = max(1, n_batches)
        nr = max(1, v6_metrics["n_r_phi_batches"])
        out_metrics["v6"] = {
            "active": True,
            "avg_loss_pinball": v6_metrics["total_pinball"] / nb,
            "avg_loss_logdet": v6_metrics["total_logdet"] / nb,
            "avg_r_phi_norm_ratio": v6_metrics["total_r_phi_norm_ratio"] / nr,
            "lambda_r_last": v6_metrics["lambda_r_last"],
            "global_step": v6_metrics["v6_global_step"],
        }
    else:
        out_metrics["v6"] = {"active": False}
    return out_metrics


__all__ = [
    "gamma_dag_warmup",
    "stage1_compute_loss",
    "freeze_stage1",
    "unfreeze_stage1",
    "stage1_inference_mode",
    "calibrate_sigma_data_two_stage",
    "causal_ablation_check",
    "precompute_stage1_outputs",
    "train_epoch_stage2_cached",
]
