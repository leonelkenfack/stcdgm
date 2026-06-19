"""
Spatial Residual Encoder (SRE) — ST-CDGM dual-branch Stage 1.

Adds a spatially-aware branch to Stage 1 that processes the full LR field
(grid, not node-pooled) and produces a spatial residual δ conditioned on the
DAG causal embedding H_T via Adaptive Layer Normalization (AdaLN).

Architecture (expert consensus Phase 5, 2026-06-19):
  - Branch A (existing): DAG + GNN + RCN + GraphToGridDecoder → μ_HR_causal
  - Branch B (this module): SRE(lr_field, H_T) → δ_spatial
  - Stage 1 output: μ_HR_total = μ_HR_causal + δ_spatial  (same log1p space)

Causal safety constraints (Expert 2, arxiv 2409.19608 / 2510.02117):
  1. Zero-init output head + learnable out_gain (init 0.1) → δ≈0 at t=0,
     preserving μ_HR_causal dominance during warm-up.
  2. AdaLN conditioning injected at EVERY conv block (not once) — DAG
     embedding H_T controls which spatial patterns the SRE emphasises.
  3. ≤200k params to prevent SRE from becoming a standalone downscaler.
  4. Training order: freeze DAG/GNN/RCN/head first, train SRE on the
     detached residual (HR_true − μ_causal.detach()) before joint fine-tuning.
  5. Loss: MSE(μ_total, HR_true) + λ_causal·MSE(μ_causal, HR_true)
     with λ_causal schedule 1.0 → 0.3 (preserves causal identifiability).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class AdaLNConvBlock(nn.Module):
    """Conv block with Adaptive Layer Normalization from causal conditioning.

    Follows DiT-style AdaLN-Zero (Peebles & Xie 2023): modulation weights
    are zero-initialised so the block starts as an identity transform.

    Parameters
    ----------
    c_in, c_out : int
        Input/output channel counts.
    d_cond : int
        Dimension of the conditioning vector (= encoder hidden_dim, 128).
    """

    def __init__(self, c_in: int, c_out: int, d_cond: int) -> None:
        super().__init__()
        # GroupNorm without affine — scale/shift come entirely from AdaLN
        n_groups = 8
        while c_in % n_groups != 0 and n_groups > 1:
            n_groups //= 2
        self.norm = nn.GroupNorm(n_groups, c_in, affine=False)

        # Zero-init modulation → γ=0, β=0 at startup (identity scale)
        self.modulation = nn.Linear(d_cond, 2 * c_in)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x    : [B, c_in, H, W]
        cond : [B, d_cond]
        """
        gamma, beta = self.modulation(cond).chunk(2, dim=-1)   # [B, c_in] each
        x_mod = self.norm(x) * (1.0 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.conv(x_mod)


class SpatialResidualEncoder(nn.Module):
    """Spatial branch for ST-CDGM Stage 1 — produces δ_spatial in log1p space.

    Parameters
    ----------
    in_channels : int
        Number of LR input channels (C_lr).
    d_cond : int
        Conditioning dim = encoder hidden_dim (default 128).
    base_ch : int
        Base feature channels. Total params ≈ 190k at base_ch=32.
    hr_h, hr_w : int
        Target HR output dimensions (172 × 179 for NZ domain).
    out_gain : float
        Initial value of the learnable output gain scalar. Combined with
        zero-init head this ensures δ ≈ 0 at initialisation.
    """

    def __init__(
        self,
        in_channels: int,
        d_cond: int = 128,
        base_ch: int = 32,
        hr_h: int = 172,
        hr_w: int = 179,
        out_gain: float = 0.1,
    ) -> None:
        super().__init__()
        self.hr_h = hr_h
        self.hr_w = hr_w

        # Stem: raw LR channels → feature map (no AdaLN yet)
        self.stem = nn.Conv2d(in_channels, base_ch, 3, padding=1)

        # Three AdaLN conv blocks, each followed by 2× bilinear upsampling.
        # LR (22,23) → (44,46) → (86,90) → (172,179)
        self.block1 = AdaLNConvBlock(base_ch, base_ch * 2, d_cond)
        self.up1 = nn.Upsample(size=(44, 46), mode="bilinear", align_corners=False)

        self.block2 = AdaLNConvBlock(base_ch * 2, base_ch * 2, d_cond)
        self.up2 = nn.Upsample(size=(86, 90), mode="bilinear", align_corners=False)

        self.block3 = AdaLNConvBlock(base_ch * 2, base_ch, d_cond)
        self.up3 = nn.Upsample(size=(hr_h, hr_w), mode="bilinear", align_corners=False)

        # Zero-init output head → δ ≈ 0 at initialisation
        self.out_head = nn.Sequential(nn.Conv2d(base_ch, 1, 3, padding=1))
        nn.init.zeros_(self.out_head[0].weight)
        nn.init.zeros_(self.out_head[0].bias)

        # Learnable gain: starts small, grows as SRE learns the spatial residual
        self.out_gain = nn.Parameter(torch.tensor(float(out_gain)))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pool_cond(self, H_T: Tensor) -> Tensor:
        """Pool H_T to a [B, d_cond] conditioning vector.

        Accepts H_T in any of the shapes produced by RCNSequenceRunner:
          [B, q, N, d]  →  mean over q and N  →  [B, d]
          [B, N, d]     →  mean over N         →  [B, d]
          [B, d]        →  identity             →  [B, d]
        """
        if H_T.dim() == 4:      # [B, q, N, d]
            return H_T.mean(dim=(1, 2))
        if H_T.dim() == 3:      # [B, N, d]
            return H_T.mean(dim=1)
        if H_T.dim() == 2:      # [B, d] — already pooled
            return H_T
        raise ValueError(f"SRE._pool_cond: unexpected H_T shape {tuple(H_T.shape)}")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, lr_field: Tensor, H_T: Tensor) -> Tensor:
        """Compute spatial residual δ_spatial.

        Parameters
        ----------
        lr_field : Tensor [B, C_lr, H_lr, W_lr]
            Full LR GCM field in the same normalised/log1p space as Stage 1.
            Must be NaN-free (fill ocean pixels with 0 before calling).
        H_T : Tensor
            RCN causal latent state — shape [B, q, N, d], [B, N, d], or [B, d].

        Returns
        -------
        Tensor [B, 1, H_hr, W_hr]
            Spatial residual in log1p space. Approximately 0 at initialisation.
        """
        cond = self._pool_cond(H_T)                       # [B, d_cond]
        x = self.stem(lr_field)                            # [B, base_ch, H_lr, W_lr]
        x = self.up1(self.block1(x, cond))                # [B, base_ch*2, 44, 46]
        x = self.up2(self.block2(x, cond))                # [B, base_ch*2, 86, 90]
        x = self.up3(self.block3(x, cond))                # [B, base_ch, H_hr, W_hr]
        return self.out_gain * self.out_head(x)            # [B, 1, H_hr, W_hr]

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
