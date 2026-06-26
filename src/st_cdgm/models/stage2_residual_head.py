"""
Stage 2 Auxiliary Residual Head r_φ(H) — V6 MVP.

Implements the F4 critical debt from ``path_c_plus/audit/ECHECS_ET_LECONS.md``
(Math Prof §14.3 r_φ(H) auxiliary residual head, never tested before V6).

Purpose
-------
F3 prouve formellement : "Stage 1 frozen → impossible d'améliorer la skill au-delà
du plancher B₁ rank-5. Stage 2 a besoin d'une voie expressive hors-DAG". Cette
voie hors-DAG est exactement ``r_φ(H)``.

The module takes the RCN hidden state ``H_T ∈ ℝ^{[B, q, N, hidden]}`` (output
of ``RCNSequenceRunner.run().states[-1]``) and produces a residual contribution
``r_φ ∈ ℝ^{[B, 1, H_HR, W_HR]}`` that is ADDED IN PARALLEL to ``μ_HR`` at
inference, providing a non-DAG expressive channel.

Architecture (IA ronde 4 — structure spatiale obligatoire vs collapse historique
project_mc2rd_dead.md) :

  H_T (B, q, N, hidden)
    │
    ├── 1) spatial pool over N nodes (mean) → (B, q, hidden)
    ├── 2) per-node projection → (B, q, emb_dim)
    ├── 3) broadcast via region masks (West, East, North, South NZ) → (B, q, n_regions, emb_dim, H_HR, W_HR)
    ├── 4) flatten channels → (B, q*n_regions*emb_dim, H_HR, W_HR)
    └── 5) conv 1×1 → (B, 1, H_HR, W_HR)

Anti-collapse safeguards
-----------------------
- ``refine`` conv 1×1 init : ``N(0, 1e-3)`` weights, zero bias
- Warmup procedure handled externally (see ``train_epoch_stage2_cached`` V6)
  - Freeze 2k steps, then ramp ``λ_r`` 0.05 → 0.20 over 5k steps

References
----------
- F4 in ECHECS_ET_LECONS.md — dette critique non-testée
- IA ronde 4 (V6 plan §2.1) — broadcast régional via masks Trenberth obligatoire
- Math ronde 4 — bonded risk r_φ→0 reste ~15% avec cette structure (vs 70% sans)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class StructuredResidualHead(nn.Module):
    """r_φ(H) — auxiliary residual head with regional spatial broadcast.

    Parameters
    ----------
    num_vars : int
        Number of variables in the causal graph. V6 MVP = 8 dynamic node types
        (6 from extended_9node : GP850/500/250, Q850, W500, IVT + 2 from
        extended_v6_wind : U850, V850). The "11-node" naming in the plan refers
        to a different spec. Pass ``len(builder.dynamic_node_types)`` to be safe.
    hidden : int
        Hidden dimension of H_T (must match ``encoder.hidden_dim``, default 128).
    emb_dim : int
        Per-node embedding dimension after projection. Default 16.
    n_regions : int
        Number of regional masks. Default 4 (West/East/North/South NZ).
    hr_h, hr_w : int
        Target HR grid dimensions (172, 179 for NIWA-REMS NZ).
    init_std : float
        Std for refine conv1x1 weight init. Default 1e-3 (anti-collapse).
    """

    def __init__(
        self,
        num_vars: int = 8,
        hidden: int = 128,
        emb_dim: int = 16,
        n_regions: int = 4,
        hr_h: int = 172,
        hr_w: int = 179,
        init_std: float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.hidden = hidden
        self.emb_dim = emb_dim
        self.n_regions = n_regions
        self.hr_h = hr_h
        self.hr_w = hr_w

        self.node_proj = nn.Linear(hidden, emb_dim)

        # region_masks must be loaded externally via set_region_masks().
        # Shape : [n_regions, hr_h, hr_w] with values in [0, 1].
        self.register_buffer(
            "region_masks", torch.zeros(n_regions, hr_h, hr_w), persistent=True
        )

        in_channels = num_vars * n_regions * emb_dim
        self.refine = nn.Conv2d(in_channels, 1, kernel_size=1)
        # Anti-collapse init (IA ronde 4)
        nn.init.normal_(self.refine.weight, mean=0.0, std=init_std)
        nn.init.zeros_(self.refine.bias)

    @torch.no_grad()
    def set_region_masks(self, masks: Tensor) -> None:
        """Load region masks from external (preproc) source.

        Parameters
        ----------
        masks : Tensor
            Shape ``[n_regions, hr_h, hr_w]``, values in [0, 1].
        """
        if masks.shape != (self.n_regions, self.hr_h, self.hr_w):
            raise ValueError(
                f"masks shape {tuple(masks.shape)} != expected "
                f"({self.n_regions}, {self.hr_h}, {self.hr_w})"
            )
        if masks.min() < 0.0 or masks.max() > 1.0:
            raise ValueError("region_masks must have values in [0, 1]")
        self.region_masks.copy_(masks.to(self.region_masks.device, self.region_masks.dtype))

    def forward(self, H_T: Tensor) -> Tensor:
        """Compute r_φ(H_T).

        Accepts three input formats :
          - ``[B, q, hidden]``    — already-pooled (V6 BS32b cache yields this)
          - ``[B, q, N, hidden]`` — batched full (encoder/RCN runtime output)
          - ``[q, N, hidden]``    — single-sample full

        ``q`` = num_vars, ``N`` = num_nodes_lr. Pre-pooled input skips the
        internal mean(dim=N).

        Returns
        -------
        Tensor
            Residual contribution of shape ``[B, 1, hr_h, hr_w]``.
        """
        if H_T.dim() == 3:
            # Can be either [B, q, hidden] (pre-pooled) or [q, N, hidden] (single full).
            # Disambiguate by checking middle dim against num_vars : if middle == q,
            # it's pre-pooled [B, q, hidden]. Otherwise treat as [q, N, hidden].
            if H_T.shape[1] == self.num_vars and H_T.shape[-1] == self.hidden:
                # Already-pooled [B, q, hidden]
                h_pooled = H_T
            elif H_T.shape[0] == self.num_vars and H_T.shape[-1] == self.hidden:
                # Single-sample full [q, N, hidden] -> add batch + pool over N
                h_pooled = H_T.unsqueeze(0).mean(dim=2)  # [1, q, hidden]
            else:
                raise ValueError(
                    f"H_T 3D shape {tuple(H_T.shape)} ambiguous : "
                    f"expected [B, q={self.num_vars}, hidden={self.hidden}] "
                    f"or [q={self.num_vars}, N, hidden={self.hidden}]"
                )
        elif H_T.dim() == 4:
            # [B, q, N, hidden] — pool over N
            B_, q_, N_, h_ = H_T.shape
            if q_ != self.num_vars:
                raise ValueError(f"H_T q={q_} != num_vars={self.num_vars}")
            if h_ != self.hidden:
                raise ValueError(f"H_T hidden={h_} != self.hidden={self.hidden}")
            h_pooled = H_T.mean(dim=2)  # [B, q, hidden]
        else:
            raise ValueError(
                f"H_T expected 3D [B,q,hidden] or [q,N,hidden] "
                f"or 4D [B,q,N,hidden]; got {tuple(H_T.shape)}"
            )

        B, q, hidden = h_pooled.shape
        if q != self.num_vars:
            raise ValueError(f"h_pooled q={q} != num_vars={self.num_vars}")
        if hidden != self.hidden:
            raise ValueError(f"h_pooled hidden={hidden} != self.hidden={self.hidden}")

        # 2) Per-node projection → [B, q, emb_dim]
        h_proj = self.node_proj(h_pooled)

        # 3) Broadcast via region masks
        #    Einsum 'bne,rhw->bnrehw' :
        #       b=batch, n=num_vars(q), e=emb_dim, r=n_regions, h=H_HR, w=W_HR
        h_spatial = torch.einsum(
            "bne,rhw->bnrehw", h_proj, self.region_masks
        )  # [B, q, n_regions, emb_dim, H_HR, W_HR]

        # 4) Flatten q·n_regions·emb_dim into channels
        h_flat = h_spatial.reshape(
            B, q * self.n_regions * self.emb_dim, self.hr_h, self.hr_w
        )

        # 5) Conv 1×1 → [B, 1, H_HR, W_HR]
        r = self.refine(h_flat)
        return r

    def num_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["StructuredResidualHead"]
