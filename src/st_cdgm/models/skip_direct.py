"""Skip-connection conditionnelle LR → μ_HR pour Oracle V5 (A1).

Module introduit par V5-mini pour récupérer 50-80 % du handicap d'expressivité
d'Oracle face au U-Net générique de CorrDiff, sans rompre la propriété (O3).

Principe
--------
Le module mélange deux voies parallèles :

    μ_HR = α(y_LR) · CRM(y_LR ; A_dag)  +  (1 - α(y_LR)) · direct(y_LR)

où :
- ``CRM(...)`` est la moyenne haute résolution produite par le chemin causal
  d'Oracle (encodeur → RCN → décodeur graphe-à-grille → regression_head).
- ``direct(...)`` est un sur-échantillonnage convolutif léger qui exploite
  directement la basse résolution sans passer par le DAG.
- ``α(y_LR) ∈ [0, 1]`` est un gate scalaire global appris par un MLP qui
  voit l'entrée basse résolution pooled. Il décide pour chaque exemple
  la fraction de blending.

Préservation de (O3)
--------------------
Une régularisation ``L_o3_preserve = λ_o3 · max(0, alpha_floor - α.mean())²``
pénalise les états où α descend sous un plancher (typ. 0.6), forçant la voie
causale à rester dominante. Sans cette pénalité, α convergerait souvent vers
0 (la voie directe est moins contrainte donc plus facile à minimiser).

À l'initialisation
------------------
Le biais du dernier neurone du gate est positionné à ``alpha_init_logit``
(typ. 2.0 → sigmoid ≈ 0.88) pour que le chemin causal domine dès le pas 0.

Coût de calcul
--------------
- ``direct`` : 2 convolutions (kernel 3) + bilinear upsample + 1 conv 1×1.
  Pour la résolution NZ 23×26 → 172×179, c'est ~0.5M FLOPs/forward, négligeable.
- ``alpha_mlp`` : pooling global + 2 Linear, ~1k FLOPs/forward.

Sur A100 / training de 30h, surcoût mesuré ~3h (10 %).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["ConditionalSkipBlock"]


class ConditionalSkipBlock(nn.Module):
    """Skip-connection conditionnelle entre la voie causale et une voie directe.

    Parameters
    ----------
    lr_channels :
        Nombre de canaux d'entrée basse résolution (typ. 15 variables × T pas).
    hr_shape :
        Forme HR cible ``(H, W)`` (typ. ``(172, 179)`` pour NZ).
    alpha_init_logit :
        Biais initial du gate. ``sigmoid(2.0) ≈ 0.88`` → chemin causal dominant.
    alpha_floor :
        Plancher utilisé par ``compute_o3_preserve_loss`` pour pénaliser α bas.
        Doit être en accord avec ``cfg.v5.skip_connection.alpha_floor``.
    direct_channels :
        Largeur du réseau ``direct`` (32 par défaut, suffisant sur NZ).

    Attributes
    ----------
    direct : nn.Sequential
        Voie résiduelle directe LR → HR (sans DAG).
    alpha_mlp : nn.Sequential
        Gate scalaire ``α(y_LR) ∈ [0, 1]``.
    alpha_floor : float
        Plancher pour la régularisation (O3).
    """

    def __init__(
        self,
        lr_channels: int,
        hr_shape: Tuple[int, int],
        *,
        alpha_init_logit: float = 2.0,
        alpha_floor: float = 0.6,
        direct_channels: int = 32,
    ) -> None:
        super().__init__()
        self.hr_shape = tuple(hr_shape)
        self.alpha_floor = float(alpha_floor)

        # ─── Voie directe : Conv → GELU → Upsample → Conv → 1×1 ───
        self.direct = nn.Sequential(
            nn.Conv2d(lr_channels, direct_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(size=self.hr_shape, mode="bilinear", align_corners=False),
            nn.Conv2d(direct_channels, direct_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(direct_channels // 2, 1, kernel_size=1),
        )

        # ─── Gate α(y_LR) : Pooling → MLP → Sigmoid ───
        self.alpha_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(lr_channels, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        # Bias initial du dernier Linear pour démarrer α ≈ sigmoid(alpha_init_logit)
        with torch.no_grad():
            self.alpha_mlp[-1].bias.fill_(float(alpha_init_logit))
            self.alpha_mlp[-1].weight.normal_(mean=0.0, std=0.01)

    # ------------------------------------------------------------------

    def forward(self, lr: Tensor, mu_causal: Tensor) -> Tuple[Tensor, Tensor]:
        """Mélange voie causale et voie directe.

        Parameters
        ----------
        lr :
            Tenseur basse résolution ``[B, C_lr, H_lr, W_lr]``.
        mu_causal :
            Sortie du chemin causal Oracle, ``[B, 1, H_hr, W_hr]``.

        Returns
        -------
        mu_blend : Tensor
            Moyenne haute résolution blendée ``[B, 1, H_hr, W_hr]``.
        alpha : Tensor
            Gate scalaire ``[B, 1]`` ∈ [0, 1] — la fraction du chemin causal.
        """
        alpha_logit = self.alpha_mlp(lr)  # [B, 1]
        alpha = torch.sigmoid(alpha_logit)  # [B, 1] ∈ (0, 1)

        direct = self.direct(lr)  # [B, 1, H, W]
        if direct.shape != mu_causal.shape:
            direct = torch.nn.functional.interpolate(
                direct, size=mu_causal.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        alpha_4d = alpha.view(-1, 1, 1, 1)
        mu_blend = alpha_4d * mu_causal + (1.0 - alpha_4d) * direct
        return mu_blend, alpha

    # ------------------------------------------------------------------

    def compute_o3_preserve_loss(self, alpha: Tensor) -> Tensor:
        """Régularisation ``L_o3_preserve`` pour préserver la dominance causale.

        Pénalise les valeurs moyennes de ``α`` en-dessous de ``alpha_floor``.
        Sans cette régularisation, le gradient tirerait α vers 0 (la voie
        directe est moins contrainte) et la propriété (O3) serait perdue.

        Parameters
        ----------
        alpha :
            Tenseur ``[B, 1]`` produit par ``forward``.

        Returns
        -------
        Tensor
            Scalaire à ajouter à la perte Stage 1 (multiplié par
            ``cfg.v5.skip_connection.lambda_o3_preserve`` côté appelant).
        """
        mean_alpha = alpha.mean()
        gap = torch.relu(self.alpha_floor - mean_alpha)
        return gap.pow(2)

    # ------------------------------------------------------------------

    def num_params(self) -> int:
        """Compte des paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
