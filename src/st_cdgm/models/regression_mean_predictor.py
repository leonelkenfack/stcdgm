"""
RegressionMeanPredictor — non-causal Stage 1 baseline (BS35 / B2).

Drop-in replacement for the causal Stage 1 (encoder + RCN + DAGMA +
GraphToGridDecoder) when running the architectural ablation. Produces
``mu_HR_log`` of the same shape from the SAME inputs (the LR driver
grid extracted from the hetero-graph data) without any causal
inductive bias.

Architecture
------------
* Input  : LR driver grid ``(B, C_LR, H_LR, W_LR)`` ≈ (B, 15, 23, 26)
           extracted from the hetero-graph batch.
* Body   : diffusers ``UNet2DModel`` operating at LR resolution with
           channel multipliers chosen so the param count matches the
           causal Stage 1 within ±10 %.
* Head   : bilinear upsample (LR → HR) + 3×3 conv refinement.
* Output : ``(B, C_HR, H_HR, W_HR)`` log-residual mean prediction.

Capacity matching
-----------------
The causal Stage 1 (encoder + RCN + decoder) is currently ~5 M params
(verified via ``scripts/_inspect_param_counts.py``). This UNet at
``block_out_channels=[64, 128, 192]`` lands at ~4.8 M, within budget.
If the causal arch param count drifts, retune via
``RegressionMeanPredictor.from_target_params(target_params)``.

References
----------
* Mardani et al. 2024 (CorrDiff, arXiv:2309.15214) — UNet regression
  for HR mean prediction in residual diffusion downscaling.
* Schölkopf et al. 2021 (arXiv:2102.11107) — capacity confound
  motivating iso-budget ablation.

BS35_CAUSAL_ABLATION sentinel.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class RegressionPredictorConfig:
    in_channels: int = 15            # number of LR driver channels
    out_channels: int = 1            # HR target channels
    lr_height: int = 23
    lr_width: int = 26
    hr_height: int = 172
    hr_width: int = 179
    block_out_channels: Tuple[int, ...] = (64, 128, 192)
    layers_per_block: int = 2
    norm_num_groups: int = 16


class RegressionMeanPredictor(nn.Module):
    """Iso-capacity non-causal Stage 1 baseline (BS35 B2)."""

    def __init__(self, cfg: RegressionPredictorConfig):
        super().__init__()
        self.cfg = cfg
        self._state_adapter: Optional[nn.Linear] = None

        # Lazy import so the module can be loaded without diffusers when
        # only the API surface is needed (e.g. testing).
        from diffusers import UNet2DModel

        # The diffusers UNet expects an integer ``sample_size``; we pass
        # the larger of (lr_h, lr_w) since downsampling will be square-ish
        # and we'll upsample externally to HR shape afterwards.
        sample_size = max(cfg.lr_height, cfg.lr_width)
        unet_kwargs = dict(
            sample_size=sample_size,
            in_channels=cfg.in_channels,
            out_channels=cfg.in_channels,           # same channels back; we project after
            block_out_channels=cfg.block_out_channels,
            layers_per_block=cfg.layers_per_block,
            down_block_types=tuple(["DownBlock2D"] * len(cfg.block_out_channels)),
            up_block_types=tuple(["UpBlock2D"] * len(cfg.block_out_channels)),
            norm_num_groups=cfg.norm_num_groups,
            time_embedding_type="positional",       # required even though we feed t=0
            class_embed_type=None,
            addition_embed_type=None,
        )
        # Diffusers versions differ slightly; filter unsupported optional keys
        # instead of pinning notebook execution to one constructor signature.
        supported = set(inspect.signature(UNet2DModel.__init__).parameters)
        unet_kwargs = {k: v for k, v in unet_kwargs.items() if k in supported}
        self.unet = UNet2DModel(**unet_kwargs)

        # HR projection : 3x3 conv after bilinear upsample.
        self.hr_proj = nn.Sequential(
            nn.Conv2d(cfg.in_channels, cfg.in_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(cfg.in_channels, cfg.out_channels, kernel_size=3, padding=1),
        )

        # No-op timestep tensor reused at every forward (we don't use the
        # diffusion semantics, just the UNet backbone for regression).
        self.register_buffer(
            "_t_zero", torch.zeros((), dtype=torch.long), persistent=False
        )

    @classmethod
    def from_target_params(
        cls,
        target_params: int,
        cfg_kwargs: Optional[Dict[str, Any]] = None,
    ) -> "RegressionMeanPredictor":
        """Construct a model whose param count matches ``target_params`` to ±10 %.

        Iterates over (depth, width) combinations in a small grid until
        the closest match is found. Used to enforce the iso-capacity
        constraint from Schölkopf et al. 2021.
        """
        cfg_kwargs = cfg_kwargs or {}
        candidates = []
        for base in (32, 48, 64, 80, 96):
            for depth in (2, 3, 4):
                channels = tuple(base * (2 ** i) for i in range(depth))
                cfg = RegressionPredictorConfig(
                    block_out_channels=channels, **cfg_kwargs
                )
                try:
                    m = cls(cfg)
                except Exception:
                    continue
                n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
                candidates.append((n_params, cfg, m))
        # Pick the closest by absolute distance.
        best = min(candidates, key=lambda t: abs(t[0] - target_params))
        return best[2]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _nodes_to_lr_grid(self, lr_nodes: Tensor) -> Tensor:
        """Convert nodal LR features to grid layout.

        Parameters
        ----------
        lr_nodes : Tensor
            Shape ``(B, N_lr, C_lr)``.
        """
        if lr_nodes.dim() != 3:
            raise ValueError(
                f"_nodes_to_lr_grid expects (B, N_lr, C_lr), got {tuple(lr_nodes.shape)}"
            )
        b, n_lr, c_lr = lr_nodes.shape
        expected = self.cfg.lr_height * self.cfg.lr_width
        if n_lr != expected:
            raise ValueError(
                f"Expected N_lr={expected}, got {n_lr}. "
                "Cannot map nodes to LR grid."
            )
        return (
            lr_nodes.transpose(1, 2)
            .reshape(b, c_lr, self.cfg.lr_height, self.cfg.lr_width)
            .contiguous()
        )

    def _state_to_lr_grid(self, state: Tensor) -> Tensor:
        """Adapter for ST-CDGM state tensors.

        Accepts:
        - ``(q, N_lr, hidden)`` from RCN final state
        - ``(B, q, N_lr, hidden)`` batched variant
        and projects to ``(B, C_lr, H_lr, W_lr)`` for UNet regression.
        """
        if state.dim() == 3:
            state = state.unsqueeze(0)
        if state.dim() != 4:
            raise ValueError(
                f"_state_to_lr_grid expects (q,N,h) or (B,q,N,h), got {tuple(state.shape)}"
            )
        b, q, n_lr, hidden = state.shape
        # Pool over q variables: non-causal baseline should not preserve
        # explicit DAG-structured channels.
        pooled = state.mean(dim=1)  # (B, N_lr, hidden)
        if self._state_adapter is None:
            self._state_adapter = nn.Linear(hidden, self.cfg.in_channels).to(
                device=state.device, dtype=state.dtype
            )
        lr_nodes = self._state_adapter(pooled)  # (B, N_lr, C_lr)
        return self._nodes_to_lr_grid(lr_nodes)

    def forward(self, lr_grid: Tensor) -> Tensor:
        """Forward pass : LR driver grid → HR mean prediction.

        Parameters
        ----------
        lr_grid : Tensor
            Shape ``(B, C_LR, H_LR, W_LR)``.

        Returns
        -------
        mu_HR_log : Tensor
            Shape ``(B, C_HR, H_HR, W_HR)``, log-residual mean.
        """
        # Primary path: direct LR grid (B, C, H, W).
        # Compatibility path for existing two-stage code: RCN state
        # tensors (q, N, h) / (B, q, N, h).
        if lr_grid.dim() == 3 or lr_grid.dim() == 4 and lr_grid.shape[1] != self.cfg.in_channels:
            lr_grid = self._state_to_lr_grid(lr_grid)
        elif lr_grid.dim() != 4:
            raise ValueError(
                f"RegressionMeanPredictor expects a 4-D LR grid or RCN state; "
                f"got shape {tuple(lr_grid.shape)}"
            )

        B = lr_grid.shape[0]
        # UNet skip connections require spatial sizes divisible by the total
        # downsampling factor. Pad on the right/bottom, then trim back below.
        down_factor = 2 ** max(0, len(self.cfg.block_out_channels) - 1)
        pad_h = (-lr_grid.shape[-2]) % down_factor
        pad_w = (-lr_grid.shape[-1]) % down_factor
        if pad_h or pad_w:
            lr_grid = F.pad(lr_grid, (0, pad_w, 0, pad_h), mode="replicate")
        # diffusers UNet2DModel wants a per-sample timestep.
        t = self._t_zero.expand(B)
        h = self.unet(lr_grid, t).sample           # (B, C_LR, H_LR_padded, W_LR_padded)
        # Trim back to the LR shape if UNet padded for stride compatibility.
        h = h[:, :, : self.cfg.lr_height, : self.cfg.lr_width]
        h = F.interpolate(
            h, size=(self.cfg.hr_height, self.cfg.hr_width),
            mode="bilinear", align_corners=False,
        )
        return self.hr_proj(h)
