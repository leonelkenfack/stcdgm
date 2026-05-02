"""
Regression head for ST-CDGM Two-Stage Causal Architecture.

Maps the RCN causal latent state ``H_T ∈ ℝ^{[B, q, N, d]}`` to the
high-resolution residual mean ``μ_HR ∈ ℝ^{[B, 1, H_HR, W_HR]}``.

Design (per Gemini Pro Deep Research validated 2026-05-02, hyperplan v2.0)
---------------------------------------------------------------------------

The decoder is **structurally load-bearing** for the causal DAG: its only
input is ``H_T`` which depends on ``A_dag`` through the RCN dynamics. Setting
``A_dag := 0`` changes ``H_T`` and therefore changes ``μ_HR`` — there is no
alternative input path. This guarantees ``Objective O6 (intervention)`` of
the paper at the architecture level, not merely as an optimisation outcome.

Memory considerations
---------------------

A naive cross-attention from H_T tokens (≈3000 tokens) to a 172×179 query
grid (≈30k queries) at d=128, 4 heads, batch=16 produces ~24 GB of attention
weights — would not fit on A100 80GB.

We therefore cross-attend at an *intermediate* resolution (``intermediate_h``,
``intermediate_w``), and bilinearly upsample + refine via a shallow CNN to
the full HR grid. Default intermediate is 43×45 ≈ 2× LR resolution. Memory
budget: ~1.5 GB for attention scores at batch 16. Total decoder param count
is ~670 k — strictly separate from the diffusion U-Net.

The grid queries are learnable parameters (a positional embedding) so the
decoder is free to discover its own coordinate system that best aligns with
the H_T topology.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class GraphToGridDecoder(nn.Module):
    """Cross-attention decoder mapping causal graph state to HR grid.

    Pipeline:
      1. Flatten ``H_T`` to a sequence of ``q × N`` tokens, each in
         ``ℝ^{d_model}``.
      2. Cross-attention: queries are a learnable
         ``[intermediate_h × intermediate_w, d_model]`` grid; keys and
         values are the H_T tokens.
      3. Reshape attention output to ``[d_model, intermediate_h,
         intermediate_w]`` and refine via a shallow CNN with bilinear
         upsampling to the target HR resolution.

    Parameters
    ----------
    d_model : int
        Hidden dimension of H_T (must match ``encoder.hidden_dim``).
    hr_h, hr_w : int
        Target HR grid dimensions (e.g. 172, 179 for the NZ domain).
    intermediate_h, intermediate_w : int
        Intermediate grid for cross-attention. Memory-quality trade-off.
        Default 43×45 ≈ 2× LR resolution → 1935 queries.
    n_heads : int
        Number of attention heads. ``d_model`` must be divisible by it.
    refine_channels : int
        Number of channels in the first refinement conv. Halves at each
        upsampling level until the final 1-channel projection.
    output_channels : int
        Output channels (1 for precipitation residual mean).
    """

    def __init__(
        self,
        d_model: int = 128,
        hr_h: int = 172,
        hr_w: int = 179,
        intermediate_h: int = 43,
        intermediate_w: int = 45,
        n_heads: int = 4,
        refine_channels: int = 64,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )

        self.d_model = d_model
        self.hr_h = hr_h
        self.hr_w = hr_w
        self.intermediate_h = intermediate_h
        self.intermediate_w = intermediate_w
        self.output_channels = output_channels

        # Learnable query grid at intermediate resolution.
        # Initialised small so cross-attention starts close to identity in
        # output statistics — avoids early instability.
        n_queries = intermediate_h * intermediate_w
        self.grid_queries = nn.Parameter(
            torch.randn(1, n_queries, d_model) * 0.02
        )

        # Cross-attention: queries attend to H_T tokens.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

        # Choose an interim spatial step ~halfway between intermediate and HR.
        # 43→86, 45→90 ≈ HR/2; bilinear keeps it parameterless.
        interim_h = (hr_h // 2) + (hr_h % 2)
        interim_w = (hr_w // 2) + (hr_w % 2)

        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, refine_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(interim_h, interim_w), mode="bilinear", align_corners=False),
            nn.Conv2d(
                refine_channels, refine_channels // 2, kernel_size=3, padding=1
            ),
            nn.GELU(),
            nn.Upsample(size=(hr_h, hr_w), mode="bilinear", align_corners=False),
            nn.Conv2d(refine_channels // 2, output_channels, kernel_size=3, padding=1),
        )

    def forward(self, H_T: Tensor) -> Tensor:
        """Apply cross-attention + CNN refinement.

        Parameters
        ----------
        H_T : Tensor
            Causal latent state. Accepted shapes:
            ``[q, N, d_model]`` (single sample, RCN native output) or
            ``[B, q, N, d_model]`` (batched).

        Returns
        -------
        Tensor
            ``μ_HR`` of shape ``[B, output_channels, hr_h, hr_w]``.
            Always batched, even when input was a single sample.
        """
        if H_T.dim() == 3:
            H_T = H_T.unsqueeze(0)
        if H_T.dim() != 4:
            raise ValueError(
                f"GraphToGridDecoder expects H_T with 3 or 4 dims, "
                f"got {H_T.dim()} dims of shape {tuple(H_T.shape)}"
            )

        B, q, N, d = H_T.shape
        if d != self.d_model:
            raise ValueError(
                f"H_T last dim {d} != d_model {self.d_model}"
            )

        # Flatten q × N into token sequence: [B, q*N, d_model]
        tokens = H_T.reshape(B, q * N, d)

        # Expand queries for the batch
        queries = self.grid_queries.expand(B, -1, -1)

        # Cross-attention; attention scores shape [B, n_heads, n_queries, q*N]
        # are released as soon as cross_attn returns — only attn_out is kept.
        attn_out, _ = self.cross_attn(query=queries, key=tokens, value=tokens)

        # Residual + LayerNorm
        grid_features = self.norm(queries + attn_out)

        # Reshape to spatial grid: [B, d_model, intermediate_h, intermediate_w]
        grid_features = grid_features.transpose(1, 2).contiguous().view(
            B, self.d_model, self.intermediate_h, self.intermediate_w
        )

        # CNN refine + upsample to HR
        mu_HR = self.upsample(grid_features)

        return mu_HR

    def num_params(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["GraphToGridDecoder"]
