"""
Module 5 – Décodeur de diffusion conditionnel pour ST-CDGM.

Ce module encapsule un UNet conditionnel (diffusers) et fournit des utilitaires
pour calculer la perte de diffusion, appliquer les contraintes physiques et
échantillonner des sorties haute résolution conditionnées par l'état causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from diffusers import DDPMScheduler, UNet2DConditionModel
except ImportError as exc:  # pragma: no cover - dépendance optionnelle
    raise ImportError(
        "Le module diffusion_decoder nécessite la bibliothèque `diffusers` "
        "(pip install diffusers accelerate)."
    ) from exc

# Phase E1: Check if DPM-Solver++ is available (might be in newer versions of diffusers)
try:
    from diffusers import DPMSolverMultistepScheduler
    HAS_DPM_SOLVER = True
except ImportError:
    HAS_DPM_SOLVER = False


def _replace_conv_transpose_with_resize(module: nn.Module) -> None:
    """Remplace ConvTranspose2d par Upsample + Conv2d (réduit artefacts checkerboard)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose2d):
            st = child.stride[0]
            if st >= 2 and child.stride[0] == child.stride[1]:
                repl = nn.Sequential(
                    nn.Upsample(scale_factor=float(st), mode="nearest"),
                    nn.Conv2d(
                        child.in_channels,
                        child.out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=child.bias is not None,
                    ),
                )
                if child.bias is not None:
                    nn.init.zeros_(repl[1].bias)
                nn.init.kaiming_normal_(repl[1].weight, nonlinearity="relu")
                setattr(module, name, repl)
            else:
                _replace_conv_transpose_with_resize(child)
        else:
            _replace_conv_transpose_with_resize(child)


@dataclass
class DiffusionOutput:
    """
    Résultat d'un échantillonnage de diffusion.
    """

    residual: Tensor
    baseline: Optional[Tensor]
    t_min: Tensor
    t_mean: Tensor
    t_max: Tensor

    @property
    def composite(self) -> Tensor:
        """Retourne le champ reconstruit (concaténé) [B,3,H,W]."""
        return torch.cat([self.t_min, self.t_mean, self.t_max], dim=1)


class CausalDiffusionDecoder(nn.Module):
    """
    Décodeur de diffusion conditionnel pour générer les champs HR.
    """

    def __init__(
        self,
        in_channels: int,
        conditioning_dim: int,
        height: int,
        width: int,
        *,
        num_diffusion_steps: int = 1000,
        unet_kwargs: Optional[dict] = None,
        scheduler_type: str = "ddpm",
        use_gradient_checkpointing: bool = False,
        conv_padding_mode: str = "zeros",
        anti_checkerboard: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conditioning_dim = conditioning_dim
        self.height = height
        self.width = width
        self.num_diffusion_steps = num_diffusion_steps
        self.scheduler_type = scheduler_type  # "ddpm" or "edm"

        self._condition_adapter: Optional[Callable[[Tensor], Tensor]] = None

        unet_kwargs = unet_kwargs or {}
        self.unet = UNet2DConditionModel(
            # NOTE: use (height, width) for non-square grids; passing an int makes Diffusers assume square inputs.
            sample_size=(height, width),
            in_channels=in_channels,
            out_channels=in_channels,
            cross_attention_dim=conditioning_dim,
            **unet_kwargs,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_steps)

        if conv_padding_mode == "replicate":
            for m in self.unet.modules():
                if isinstance(m, nn.Conv2d):
                    m.padding_mode = "replicate"
        if anti_checkerboard:
            _replace_conv_transpose_with_resize(self.unet)
        
        # Phase C3: Gradient checkpointing support
        # Reduces memory usage by ~50% but increases computation time by ~20-30%
        # Trade-off: Use when memory is limited or to allow larger batch sizes
        self._gradient_checkpointing_enabled = False
        if use_gradient_checkpointing:
            self.enable_gradient_checkpointing()

    def enable_gradient_checkpointing(self) -> None:
        """
        Active le gradient checkpointing en délégant au UNet diffusers
        sous-jacent. Sans cette méthode, ``__init__`` levait
        ``AttributeError: 'CausalDiffusionDecoder' object has no attribute
        'enable_gradient_checkpointing'`` quand ``use_gradient_checkpointing=True``.

        Le UNet de diffusers expose ``enable_gradient_checkpointing()`` qui
        active ``gradient_checkpointing=True`` sur tous les blocs supportant
        l'API (down/up/mid). En fallback (versions très anciennes), on
        positionne directement l'attribut sur les blocs.
        """
        if hasattr(self.unet, "enable_gradient_checkpointing"):
            self.unet.enable_gradient_checkpointing()
        else:
            # Fallback : poser l'attribut directement (les blocs diffusers
            # le testent via ``getattr(self, "gradient_checkpointing", False)``).
            for _module in self.unet.modules():
                if hasattr(_module, "gradient_checkpointing"):
                    _module.gradient_checkpointing = True
        self._gradient_checkpointing_enabled = True

    def disable_gradient_checkpointing(self) -> None:
        """Symétrique de ``enable_gradient_checkpointing`` pour pouvoir
        repasser en mode plein-mémoire (ex. avant un export ONNX)."""
        if hasattr(self.unet, "disable_gradient_checkpointing"):
            self.unet.disable_gradient_checkpointing()
        else:
            for _module in self.unet.modules():
                if hasattr(_module, "gradient_checkpointing"):
                    _module.gradient_checkpointing = False
        self._gradient_checkpointing_enabled = False

    def _pool_conditioning(self, conditioning: Tensor) -> Tensor:
        """Flatten conditioning [B, seq, dim] -> [B, seq*dim] for FiLM class_labels."""
        return conditioning.flatten(start_dim=1)

    def forward(
        self,
        noisy_sample: Tensor,
        timestep: Tensor,
        conditioning: Tensor,
        conditioning_spatial: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Passe avant du UNet (prédiction du bruit).

        Parameters
        ----------
        conditioning : Tensor
            Global conditioning ``[B, num_vars, dim]`` — flattened for FiLM
            ``class_labels``.
        conditioning_spatial : Optional[Tensor]
            Spatial tokens ``[B, num_tokens, dim]`` for cross-attention
            (``encoder_hidden_states``).  Falls back to *conditioning* when
            ``None``.
        """
        conditioning = self._prepare_conditioning(conditioning)
        class_labels = self._pool_conditioning(conditioning)
        hidden_states = conditioning_spatial if conditioning_spatial is not None else conditioning
        output = self.unet(
            sample=noisy_sample,
            timestep=timestep,
            encoder_hidden_states=hidden_states,
            class_labels=class_labels,
        )
        return output.sample

    def compute_loss(
        self,
        target: Tensor,
        conditioning: Tensor,
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        conditioning_spatial: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calcule la perte de diffusion (MSE entre bruit réel et prédit).
        Gère les NaN dans le target en utilisant un masque (standard pour données climatiques).
        """
        # Vérifier le conditioning (ne doit pas contenir de NaN/Inf)
        if not torch.isfinite(conditioning).all():
            raise ValueError(
                f"Conditioning contains NaN/Inf: NaN={torch.isnan(conditioning).sum().item()}, "
                f"Inf={torch.isinf(conditioning).sum().item()}, "
                f"shape={conditioning.shape}, "
                f"stats: min={conditioning.min().item():.6f}, max={conditioning.max().item():.6f}"
            )
        
        # Créer un masque pour les valeurs valides dans le target
        # Les NaN peuvent représenter des masques géographiques (océan, etc.)
        valid_mask = torch.isfinite(target)
        nan_count = (~valid_mask).sum().item()
        total_pixels = target.numel()
        
        # Si tous les pixels sont NaN, retourner une loss par défaut
        if not valid_mask.any():
            return torch.tensor(0.0, device=target.device, requires_grad=True)
        
        # Remplacer temporairement les NaN par 0 pour add_noise
        # (les NaN se propagent dans noisy_sample, on les masquera après)
        target_clean = target.clone()
        target_clean[~valid_mask] = 0.0
        
        noise = torch.randn_like(target_clean)
        batch_size = target_clean.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (batch_size,),
            device=target_clean.device,
            dtype=torch.long,
        )
        noisy_sample = self.scheduler.add_noise(target_clean, noise, timesteps)
        
        # Le masque reste valide car add_noise préserve la structure
        # (les NaN dans target_clean deviennent des valeurs, mais on utilise le masque original)
        # En fait, on doit recréer le masque car add_noise peut changer les valeurs
        # Mais comme on a remplacé les NaN par 0, le masque original reste valide
        
        noise_pred = self.forward(noisy_sample, timesteps, conditioning, conditioning_spatial=conditioning_spatial)
        
        # Vérifier que noise_pred ne contient pas de NaN/Inf (problème du modèle)
        if not torch.isfinite(noise_pred).all():
            raise ValueError(
                f"noise_pred contains NaN/Inf after UNet forward: "
                f"NaN={torch.isnan(noise_pred).sum().item()}, "
                f"Inf={torch.isinf(noise_pred).sum().item()}, "
                f"shape={noise_pred.shape}, "
                f"stats: min={noise_pred.min().item():.6f}, max={noise_pred.max().item():.6f}"
            )
        
        # Calculer la loss uniquement sur les pixels valides
        # Utiliser le masque pour filtrer les pixels NaN
        
        # Phase 3.2: Min-SNR γ-weighting for better training stability (optional)
        # This helps with training stability by downweighting high SNR timesteps
        try:
            # Access alphas_cumprod from scheduler
            if hasattr(self.scheduler, 'alphas_cumprod'):
                alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]  # [batch_size]
                # Expand to match noise_pred shape for broadcasting
                for _ in range(len(noise_pred.shape) - len(alphas_cumprod.shape)):
                    alphas_cumprod = alphas_cumprod.unsqueeze(-1)
                
                # SNR = alpha^2 / (1 - alpha^2) = alpha^2 / sigma^2
                snr = alphas_cumprod / (1.0 - alphas_cumprod + 1e-8)
                # Min-SNR weighting: weight = min(SNR, 5.0) / SNR
                # This downweights very high SNR timesteps (typically > 5.0)
                min_snr_weight = torch.clamp(snr / 5.0, max=1.0)
                
                # Apply weighting only to valid pixels
                noise_error = noise_pred[valid_mask] - noise[valid_mask]
                weight_expanded = min_snr_weight.expand_as(noise_pred)[valid_mask]
                mse_error = noise_error ** 2
                weighted_error = weight_expanded * mse_error
                
                # Phase D1: Apply focal loss weighting if enabled
                # Focal loss focuses on hard pixels (high error) for better learning
                if use_focal_loss:
                    # Normalize error to [0, 1] range for focal weighting
                    # Use relative error: normalize by max error in batch
                    error_normalized = mse_error / (mse_error.max() + 1e-8)
                    # Focal weight: (error_normalized)^gamma
                    # Higher gamma = more focus on hard pixels
                    focal_weight = (error_normalized ** focal_gamma)
                    # Apply focal weighting
                    weighted_error = focal_alpha * focal_weight * weighted_error
                
                loss = weighted_error.mean()
            else:
                # Fallback to standard MSE if alphas_cumprod not available
                noise_error = noise_pred[valid_mask] - noise[valid_mask]
                mse_error = noise_error ** 2
                
                # Phase D1: Apply focal loss if enabled (even without Min-SNR)
                if use_focal_loss:
                    error_normalized = mse_error / (mse_error.max() + 1e-8)
                    focal_weight = (error_normalized ** focal_gamma)
                    loss = focal_alpha * (focal_weight * mse_error).mean()
                else:
                    loss = mse_error.mean()
        except (AttributeError, IndexError, RuntimeError):
            # Fallback to standard MSE on any error
            noise_error = noise_pred[valid_mask] - noise[valid_mask]
            mse_error = noise_error ** 2
            
            # Phase D1: Apply focal loss if enabled
            if use_focal_loss:
                error_normalized = mse_error / (mse_error.max() + 1e-8)
                focal_weight = (error_normalized ** focal_gamma)
                loss = focal_alpha * (focal_weight * mse_error).mean()
            else:
                loss = mse_error.mean()
        
        # Vérifier la loss finale
        if not torch.isfinite(loss):
            raise ValueError(
                f"Loss is NaN/Inf: loss={loss.item()}, "
                f"valid_pixels={valid_mask.sum().item()}/{total_pixels}, "
                f"noise_pred_stats: min={noise_pred[valid_mask].min().item():.6f}, max={noise_pred[valid_mask].max().item():.6f}"
            )
        
        return loss

    def predict_x0_from_epsilon(
        self,
        noisy_sample: Tensor,
        epsilon_pred: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """DDPM : estimation x0 depuis ε_θ (pour perte spectrale RAPSD)."""
        a = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return (noisy_sample - (1.0 - a).sqrt() * epsilon_pred) / a.sqrt().clamp(min=1e-8)

    @staticmethod
    def apply_physical_constraints(raw_output: Tensor, use_soft: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Applique les contraintes physiques : T_min <= T <= T_max.
        
        Phase 3.3: Uses soft constraints (Softplus/Mish) instead of hard ReLU
        for better gradient flow and training stability.
        
        Parameters
        ----------
        raw_output : Tensor
            Raw output [batch, 3, H, W] with channels (T_min, Δ1, Δ2)
        use_soft : bool
            If True, use soft constraints (Softplus). If False, use hard ReLU.
        
        Returns
        -------
        Tuple of (t_min, t, t_max) tensors
        """
        if raw_output.shape[1] != 3:
            raise ValueError(
                "La sortie brute doit avoir exactement 3 canaux (T_min, Δ1, Δ2)."
            )

        t_min = raw_output[:, 0:1, :, :]
        delta_1 = raw_output[:, 1:2, :, :]
        delta_2 = raw_output[:, 2:3, :, :]

        # Phase 3.3: Use soft constraints instead of hard ReLU
        if use_soft:
            # Softplus: smooth approximation of ReLU, better gradients
            # f(x) = log(1 + exp(x)) / beta, where beta controls sharpness
            softplus = nn.Softplus(beta=1.0)
            t = t_min + softplus(delta_1)
            t_max = t + softplus(delta_2)
        else:
            # Original hard constraints
            t = t_min + torch.relu(delta_1)
            t_max = t + torch.relu(delta_2)

        return t_min, t, t_max

    def sample(
        self,
        conditioning: Tensor,
        *,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        baseline: Optional[Tensor] = None,
        apply_constraints: bool = True,
        scheduler_type: Optional[str] = None,
        cfg_scale: float = 0.0,
        conditioning_spatial: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        """
        Génère une sortie par diffusion conditionnée.
        
        Phase 3.2: Supports both DDPM and EDM (ODE-based) sampling.
        EDM uses fewer steps (15-50) via ODE solver for faster generation.
        """
        conditioning = self._prepare_conditioning(conditioning)
        
        # Use provided scheduler_type or default to instance setting
        scheduler_type = scheduler_type or getattr(self, 'scheduler_type', 'ddpm')
        
        # Phase 3.2: Use EDM ODE solver if requested
        # Phase E1: Use DPM-Solver++ if requested (faster than EDM)
        if scheduler_type == "edm":
            # EDM ODE solver with fewer steps
            num_steps = num_steps or 25  # Default to 25 steps for EDM (vs 1000 for DDPM)
            return self._sample_edm_ode(
                conditioning=conditioning,
                num_steps=num_steps,
                generator=generator,
                baseline=baseline,
                apply_constraints=apply_constraints,
                conditioning_spatial=conditioning_spatial,
            )
        elif scheduler_type == "dpm_solver" or scheduler_type == "dpm_solver++":
            if not HAS_DPM_SOLVER:
                raise ImportError(
                    "DPM-Solver++ is not available. Please update diffusers: "
                    "pip install --upgrade diffusers"
                )
            num_steps = num_steps or 15  # Default to 15 steps for DPM-Solver++ (vs 25 for EDM)
            return self._sample_dpm_solver(
                conditioning=conditioning,
                num_steps=num_steps,
                generator=generator,
                baseline=baseline,
                apply_constraints=apply_constraints,
                conditioning_spatial=conditioning_spatial,
                cfg_scale=cfg_scale,
            )
        
        # Original DDPM sampling
        scheduler = self.scheduler
        inference_steps = num_steps or getattr(scheduler, "num_inference_steps", None)
        if inference_steps is None:
            inference_steps = self.num_diffusion_steps
        max_train = int(
            getattr(scheduler.config, "num_train_timesteps", self.num_diffusion_steps)
        )
        inference_steps = max(1, min(int(inference_steps), max_train))
        scheduler.set_timesteps(inference_steps, device=conditioning.device)

        sample = torch.randn(
            conditioning.shape[0],
            self.in_channels,
            self.height,
            self.width,
            device=conditioning.device,
            generator=generator,
        )

        class_labels = self._pool_conditioning(conditioning)
        hidden_states = conditioning_spatial if conditioning_spatial is not None else conditioning
        for t in scheduler.timesteps:
            noise_pred_cond = self.unet(
                sample=sample,
                timestep=t,
                encoder_hidden_states=hidden_states,
                class_labels=class_labels,
            ).sample
            if cfg_scale and cfg_scale > 0.0:
                null_hs = torch.zeros_like(hidden_states)
                null_cl = torch.zeros_like(class_labels)
                noise_pred_uncond = self.unet(
                    sample=sample,
                    timestep=t,
                    encoder_hidden_states=null_hs,
                    class_labels=null_cl,
                ).sample
                model_output = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                model_output = noise_pred_cond
            # CFG safety net: with an under-trained unconditional branch
            # (early epochs, small conditioning_dropout_prob), the
            # extrapolation ``u + s·(c - u)`` can overshoot and saturate
            # in bf16/fp16. Clamp to a wide finite range so the scheduler
            # step does not consume a NaN that then propagates to every
            # subsequent step of the reverse process.
            model_output = torch.nan_to_num(
                model_output, nan=0.0, posinf=1e4, neginf=-1e4
            )
            sample = scheduler.step(model_output, t, sample).prev_sample

        residual = sample
        if baseline is not None:
            if baseline.shape != residual.shape:
                raise ValueError(
                    f"Baseline shape {tuple(baseline.shape)} incompatible avec résidu {tuple(residual.shape)}."
                )
            composite = baseline + residual
        else:
            composite = residual

        # Physical constraints are only defined for the special 3-channel representation:
        # (T_min, Δ1, Δ2) → (T_min, T_mean, T_max). For any other channel count,
        # we skip constraints and return the first channel for all three outputs.
        if composite.shape[1] == 3:
            if apply_constraints:
                # Phase 3.3: Use soft constraints by default
                t_min, t_mean, t_max = self.apply_physical_constraints(composite, use_soft=True)
            else:
                t_min, t_mean, t_max = (
                    composite[:, 0:1, :, :],
                    composite[:, 1:2, :, :],
                    composite[:, 2:3, :, :],
                )
        else:
            t_min = t_mean = t_max = composite[:, 0:1, :, :]
        if residual.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"Sortie diffusion {tuple(residual.shape[-2:])} != ({self.height}, {self.width})"
            )
        return DiffusionOutput(residual=residual, baseline=baseline, t_min=t_min, t_mean=t_mean, t_max=t_max)

    def _sample_edm_ode(
        self,
        conditioning: Tensor,
        num_steps: int = 25,
        generator: Optional[torch.Generator] = None,
        baseline: Optional[Tensor] = None,
        apply_constraints: bool = True,
        conditioning_spatial: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        """
        Phase 3.2: EDM (Elucidated Diffusion Models) sampling using ODE solver.
        
        Uses Euler method to solve the probability flow ODE with fewer steps (15-50)
        instead of the full 1000-step DDPM process.
        
        Parameters
        ----------
        conditioning : Tensor
            Conditioning tensor
        num_steps : int
            Number of ODE steps (15-50 recommended)
        generator : Optional[torch.Generator]
            Random number generator
        baseline : Optional[Tensor]
            Baseline to add to residual
        apply_constraints : bool
            Whether to apply physical constraints
        
        Returns
        -------
        DiffusionOutput
            Generated sample
        """
        device = conditioning.device
        
        # Initialize with noise
        sample = torch.randn(
            conditioning.shape[0],
            self.in_channels,
            self.height,
            self.width,
            device=device,
            generator=generator,
        )
        
        # Create time schedule for ODE (from t=1.0 to t=0.0)
        # Using EDM parameterization: sigma(t) = t
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        dt = 1.0 / num_steps
        
        class_labels = self._pool_conditioning(conditioning)
        hidden_states = conditioning_spatial if conditioning_spatial is not None else conditioning
        for i in range(num_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            
            scheduler_timestep = (1.0 - t_current) * self.num_diffusion_steps
            scheduler_timestep = scheduler_timestep.long().clamp(0, self.num_diffusion_steps - 1)
            
            with torch.no_grad():
                noise_pred = self.unet(
                    sample=sample,
                    timestep=scheduler_timestep.expand(sample.shape[0]),
                    encoder_hidden_states=hidden_states,
                    class_labels=class_labels,
                ).sample
            
            # EDM ODE step: dx/dt = -sigma * sigma' * score
            # Simplified Euler step: x_{t+dt} = x_t - dt * sigma(t) * sigma'(t) * score
            # For linear schedule: sigma(t) = t, so sigma'(t) = 1
            sigma = t_current
            score = -noise_pred / (sigma + 1e-8)  # Convert noise prediction to score
            
            # Euler step
            sample = sample + dt * sigma * score
        
        residual = sample
        if baseline is not None:
            if baseline.shape != residual.shape:
                raise ValueError(
                    f"Baseline shape {tuple(baseline.shape)} incompatible avec résidu {tuple(residual.shape)}."
                )
            composite = baseline + residual
        else:
            composite = residual

        if composite.shape[1] == 3:
            if apply_constraints:
                # Phase 3.3: Use soft constraints by default
                t_min, t_mean, t_max = self.apply_physical_constraints(composite, use_soft=True)
            else:
                t_min, t_mean, t_max = (
                    composite[:, 0:1, :, :],
                    composite[:, 1:2, :, :],
                    composite[:, 2:3, :, :],
                )
        else:
            t_min = t_mean = t_max = composite[:, 0:1, :, :]
        return DiffusionOutput(residual=residual, baseline=baseline, t_min=t_min, t_mean=t_mean, t_max=t_max)
    
    def _sample_dpm_solver(
        self,
        conditioning: Tensor,
        num_steps: int = 15,
        generator: Optional[torch.Generator] = None,
        baseline: Optional[Tensor] = None,
        apply_constraints: bool = True,
        conditioning_spatial: Optional[Tensor] = None,
        cfg_scale: float = 0.0,
    ) -> DiffusionOutput:
        """
        Phase E1: DPM-Solver++ sampling for ultra-fast inference.
        
        DPM-Solver++ is a high-order solver that can achieve high-quality results
        in 15-20 steps (compared to 25-50 for EDM and 1000 for DDPM).
        
        Parameters
        ----------
        conditioning : Tensor
            Conditioning tensor
        num_steps : int
            Number of sampling steps (15-20 recommended for DPM-Solver++)
        generator : Optional[torch.Generator]
            Random number generator
        baseline : Optional[Tensor]
            Baseline to add to residual
        apply_constraints : bool
            Whether to apply physical constraints
        
        Returns
        -------
        DiffusionOutput
            Generated sample
        
        References
        ----------
        - Lu et al. (2022): "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
        """
        if not HAS_DPM_SOLVER:
            raise ImportError(
                "DPM-Solver++ is not available. Please update diffusers: "
                "pip install --upgrade diffusers"
            )
        
        device = conditioning.device
        batch_size = conditioning.shape[0]
        
        # Initialize with noise
        sample = torch.randn(
            batch_size,
            self.in_channels,
            self.height,
            self.width,
            device=device,
            generator=generator,
        )
        
        # Create DPM-Solver scheduler (configured for fast sampling)
        dpm_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            algorithm_type="dpmsolver++",  # Use DPM-Solver++ algorithm
            solver_order=2,  # Second-order solver for balance of speed and quality
            use_karras_sigmas=True,  # Karras noise schedule for better quality
        )
        dpm_scheduler.set_timesteps(num_steps, device=device)

        # Sprint 3: classifier-free guidance for DPM-Solver++.
        # The DDPM branch already supported CFG, but the production config
        # uses DPM-Solver++ at inference — CFG was silently ignored, which
        # made ``cfg_scale`` in the YAML a no-op. We replicate the same
        # dual-stream (class_labels + encoder_hidden_states) null-branch
        # logic here. Training already drops both streams jointly under
        # ``conditioning_dropout_prob`` (see training_loop.py), so the
        # model has learned an unconditional branch we can legitimately
        # extrapolate away from.
        use_cfg = bool(cfg_scale) and cfg_scale > 0.0
        if use_cfg:
            prepared_cond = self._prepare_conditioning(conditioning)
            class_labels_cond = self._pool_conditioning(prepared_cond)
            hidden_cond = (
                conditioning_spatial
                if conditioning_spatial is not None
                else prepared_cond
            )
            null_hidden = torch.zeros_like(hidden_cond)
            null_class = torch.zeros_like(class_labels_cond)

        # Sampling loop with DPM-Solver++
        for t in dpm_scheduler.timesteps:
            cond_out = self.forward(
                sample, t, conditioning,
                conditioning_spatial=conditioning_spatial,
            )
            if use_cfg:
                uncond_out = self.unet(
                    sample=sample,
                    timestep=t,
                    encoder_hidden_states=null_hidden,
                    class_labels=null_class,
                ).sample
                model_output = uncond_out + cfg_scale * (cond_out - uncond_out)
            else:
                model_output = cond_out
            # CFG safety net: same reasoning as the DDPM branch above —
            # under-trained uncond + high cfg_scale can saturate; clamp to
            # a finite range so DPM-Solver++ never ingests a NaN.
            model_output = torch.nan_to_num(
                model_output, nan=0.0, posinf=1e4, neginf=-1e4
            )
            sample = dpm_scheduler.step(model_output, t, sample, return_dict=False)[0]
        
        residual = sample
        if baseline is not None:
            if baseline.shape != residual.shape:
                raise ValueError(
                    f"Baseline shape {tuple(baseline.shape)} incompatible avec résidu {tuple(residual.shape)}."
                )
            composite = baseline + residual
        else:
            composite = residual
        
        if composite.shape[1] == 3:
            if apply_constraints:
                t_min, t_mean, t_max = self.apply_physical_constraints(composite, use_soft=True)
            else:
                t_min, t_mean, t_max = (
                    composite[:, 0:1, :, :],
                    composite[:, 1:2, :, :],
                    composite[:, 2:3, :, :],
                )
        else:
            t_min = t_mean = t_max = composite[:, 0:1, :, :]
        
        return DiffusionOutput(residual=residual, baseline=baseline, t_min=t_min, t_mean=t_mean, t_max=t_max)

    def reconstruct_from_residual(
        self,
        residual: Tensor,
        *,
        baseline: Optional[Tensor] = None,
        apply_constraints: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Combine un résidu prédictif avec un baseline et applique les contraintes physiques.
        """
        if baseline is not None:
            if baseline.shape != residual.shape:
                raise ValueError(
                    f"Baseline shape {tuple(baseline.shape)} incompatible avec résidu {tuple(residual.shape)}."
                )
            composite = baseline + residual
        else:
            composite = residual
        if composite.shape[1] == 3:
            if apply_constraints:
                # Phase 3.3: Use soft constraints by default
                return self.apply_physical_constraints(composite, use_soft=True)
            return (
                composite[:, 0:1, :, :],
                composite[:, 1:2, :, :],
                composite[:, 2:3, :, :],
            )
        t = composite[:, 0:1, :, :]
        return (t, t, t)

    def set_condition_adapter(self, adapter: Optional[Callable[[Tensor], Tensor]]) -> None:
        """
        Définit un adaptateur appliqué sur le tenseur de conditionnement avant le UNet.
        """
        self._condition_adapter = adapter

    def _prepare_conditioning(self, conditioning: Tensor) -> Tensor:
        if self._condition_adapter is not None:
            conditioning = self._condition_adapter(conditioning)
        if conditioning.dim() != 3:
            raise ValueError(
                f"Le conditionnement doit avoir la forme [batch, sequence, dim], obtenu {tuple(conditioning.shape)}."
            )
        return conditioning

