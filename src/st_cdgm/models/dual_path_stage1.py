"""
Dual-Path Stage 1 for ST-CDGM.

Combines:
  Path A (causal, frozen): DAG → GNN → RCN → GraphToGridDecoder → μ_A
  Path B (new, spatial):   LR grid → CNN×3 + upsample → μ_B

Fusion (pixel-wise gate):
  g = σ(Conv([μ_A, μ_B]))   [init bias=-2 → g≈0.12 → μ_total ≈ μ_A]
  μ_total = g · μ_B + (1-g) · μ_A

Causal safety:
  - A_dag.requires_grad = False at all times
  - diversity_loss prevents gate collapse (mean(g) > gate_max_mean)
  - Q_phys preserved: d(μ_total)/d(A_dag) = (1-g) · d(μ_A)/d(A_dag) ≠ 0

Why Path B predicts HR directly (not residual):
  The SRE trained on HR - μ_A.detach() collapsed to 0 because
  E[HR - μ_A | LR] ≈ 0 when μ_A already captures all LR-correlated signal.
  Path B avoids this by predicting HR_true directly → SNR ≈ 4 vs 0.67.

References:
  - Geneva & Zabaras 2020 (multi-fidelity dual-path, PNAS)
  - Harder et al. 2023 (physics-constrained CNN, JMLR)
  - Mardani et al. 2024 / CorrDiff (direct Stage 1 regression, arXiv:2309.15214)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _gn_groups(channels: int, target: int = 8) -> int:
    g = target
    while channels % g != 0 and g > 1:
        g //= 2
    return g


class _GNConvBlock(nn.Module):
    """Conv2d(3×3) + GroupNorm + GELU."""

    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(_gn_groups(c_out), c_out),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Path B — Spatial CNN
# ---------------------------------------------------------------------------

class PathBCNN(nn.Module):
    """Spatial CNN: LR grid [B,C,H_lr,W_lr] → μ_B [B,1,H_hr,W_hr].

    ~530k params at base_ch=32. Predicts HR directly (MSE(μ_B, HR_true)),
    NOT a residual, so gradient is proportional to the full HR signal.

    Upsampling path (NZ domain):
      (23,26) → block1 → (44,46) → block2 → (86,90) → block3 → (172,179)

    Parameters
    ----------
    in_channels : int   LR channels (C_LR = 15)
    base_ch : int       Base feature width. ~530k at 32.
    hr_h, hr_w : int    HR output size (172, 179 for NZ domain).
    """

    def __init__(
        self,
        in_channels: int = 15,
        base_ch: int = 32,
        hr_h: int = 172,
        hr_w: int = 179,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_ch, 3, padding=1),
            nn.GELU(),
        )

        self.block1 = _GNConvBlock(base_ch, base_ch * 2)
        self.up1    = nn.Upsample(size=(44, 46), mode="bilinear", align_corners=False)

        self.block2 = _GNConvBlock(base_ch * 2, base_ch * 2)
        self.up2    = nn.Upsample(size=(86, 90), mode="bilinear", align_corners=False)

        self.block3 = _GNConvBlock(base_ch * 2, base_ch)
        self.up3    = nn.Upsample(size=(hr_h, hr_w), mode="bilinear", align_corners=False)

        # Normal init (NOT zero-init) — direct HR prediction, not residual
        self.head = nn.Conv2d(base_ch, 1, 3, padding=1)

    def forward(self, lr_grid: Tensor) -> Tensor:
        """
        Parameters
        ----------
        lr_grid : [B, C_LR, H_lr, W_lr]  NaN-free (fill ocean with 0 before call)

        Returns
        -------
        Tensor [B, 1, H_hr, W_hr]
        """
        x = self.stem(lr_grid)
        x = self.up1(self.block1(x))
        x = self.up2(self.block2(x))
        x = self.up3(self.block3(x))
        return self.head(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Fusion Gate
# ---------------------------------------------------------------------------

class FusionGate(nn.Module):
    """Pixel-wise learnable gate between μ_A (causal) and μ_B (spatial).

    g = σ( Conv1×1( GELU( Conv3×3( [μ_A, μ_B] ) ) ) )
    μ_total = g · μ_B + (1 − g) · μ_A

    Initialization:
      conv_out weights = 0, bias = gate_init_bias
      → g ≈ σ(gate_init_bias) ≈ 0.12 at start → μ_total ≈ 0.88·μ_A

    Diversity constraint (Q_phys preservation):
      diversity_loss = relu(mean(g) − gate_max_mean)²
      → penalises gate when it exceeds gate_max_mean on average
      → causal branch retains at least (1 − gate_max_mean) weight globally

    Parameters
    ----------
    hidden_ch : int      Conv intermediate channels.
    gate_init_bias : float  Initial output logit (σ(−2) ≈ 0.12).
    gate_max_mean : float   Diversity constraint threshold (default 0.50).
    """

    def __init__(
        self,
        hidden_ch: int = 16,
        gate_init_bias: float = -2.0,
        gate_max_mean: float = 0.50,
    ) -> None:
        super().__init__()
        self.gate_max_mean = float(gate_max_mean)

        self.conv_in  = nn.Conv2d(2, hidden_ch, 3, padding=1)
        self.conv_out = nn.Conv2d(hidden_ch, 1, 1)

        nn.init.zeros_(self.conv_out.weight)
        nn.init.constant_(self.conv_out.bias, gate_init_bias)

    def forward(self, mu_A: Tensor, mu_B: Tensor) -> tuple[Tensor, Tensor]:
        """Fuse μ_A and μ_B.

        Inputs to the gate use .detach() so the gate learns a selector
        (where is μ_B better?) without creating implicit gradient paths
        through the gate's CNN for μ_A and μ_B separately.
        Gradients for μ_A still flow via the (1-g)·μ_A term in μ_total.

        Parameters
        ----------
        mu_A : [B, 1, H, W]
        mu_B : [B, 1, H, W]

        Returns
        -------
        mu_total : [B, 1, H, W]
        gate : [B, 1, H, W]  ∈ (0, 1)
        """
        x = torch.cat([mu_A.detach(), mu_B.detach()], dim=1)  # [B,2,H,W]
        g = torch.sigmoid(self.conv_out(F.gelu(self.conv_in(x))))  # [B,1,H,W]
        mu_total = g * mu_B + (1.0 - g) * mu_A
        return mu_total, g

    def diversity_loss(self, gate: Tensor) -> Tensor:
        """L_div = relu(mean(g) - gate_max_mean)²  (scalar)."""
        return torch.relu(gate.mean() - self.gate_max_mean).pow(2)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# DualPathPredictor
# ---------------------------------------------------------------------------

class DualPathPredictor(nn.Module):
    """Combined Dual-Path Stage 1 — wraps PathBCNN + FusionGate.

    Path A (causal) is external: it is computed upstream and passed as mu_A.

    Parameters
    ----------
    in_channels : int      LR grid channels (15).
    base_ch : int          PathBCNN base width (~530k at 32).
    hr_h, hr_w : int       HR output size.
    gate_max_mean : float  Diversity constraint (default 0.50).
    gate_init_bias : float Gate init logit (default −2.0 → g≈0.12).
    """

    def __init__(
        self,
        in_channels: int = 15,
        base_ch: int = 32,
        hr_h: int = 172,
        hr_w: int = 179,
        gate_max_mean: float = 0.50,
        gate_init_bias: float = -2.0,
    ) -> None:
        super().__init__()
        self.path_b = PathBCNN(
            in_channels=in_channels,
            base_ch=base_ch,
            hr_h=hr_h,
            hr_w=hr_w,
        )
        self.gate = FusionGate(
            gate_init_bias=gate_init_bias,
            gate_max_mean=gate_max_mean,
        )

    def forward(
        self, lr_grid: Tensor, mu_A: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        lr_grid : [B, C_LR, H_lr, W_lr]  NaN-free LR grid
        mu_A    : [B, 1, H_hr, W_hr]      causal prediction from Path A

        Returns
        -------
        mu_total : [B, 1, H_hr, W_hr]
        mu_B     : [B, 1, H_hr, W_hr]
        gate     : [B, 1, H_hr, W_hr]  ∈ (0, 1)
        """
        mu_B = self.path_b(lr_grid)
        mu_total, gate = self.gate(mu_A, mu_B)
        return mu_total, mu_B, gate

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def causal_frac(self, mu_A: Tensor, mu_B: Tensor) -> float:
        """||μ_A||_rms / (||μ_A||_rms + ||μ_B||_rms) — causal dominance metric."""
        n_A = mu_A.detach().pow(2).mean().sqrt().item()
        n_B = mu_B.detach().pow(2).mean().sqrt().item()
        return n_A / (n_A + n_B + 1e-8)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def path_b_params(self):
        return self.path_b.parameters()

    def gate_params(self):
        return self.gate.parameters()

    def all_params(self):
        return self.parameters()
