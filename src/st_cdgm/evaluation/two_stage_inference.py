"""Shared two-stage inference helpers for notebooks and evaluation scripts."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor

from ..models.graph_builder import HeteroGraphBuilder
from ..training.stage1_paths import predict_mu_hr


def build_two_stage_batch(
    sample: dict,
    builder: HeteroGraphBuilder,
    device: torch.device,
) -> dict:
    """Build a notebook-style batch while preserving the raw LR grid.

    ``lr`` remains the node tensor used by the causal RCN path. ``lr_grid`` is
    the raw ``(T,C,H,W)`` tensor used by the non-causal regression baseline.
    """
    lr_grid = sample["lr"]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_grid[t]) for t in range(lr_grid.shape[0])]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    dynamic_features = {node_type: lr_nodes_steps[0] for node_type in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_features).to(device)
    return {
        "lr": lr_tensor,
        "lr_grid": lr_grid,
        "residual": sample["residual"],
        "baseline": sample.get("baseline"),
        "hetero": hetero,
        "time": sample.get("time"),
    }


@torch.no_grad()
def build_two_stage_inputs(
    batch: dict,
    *,
    variant: str,
    regression_head,
    device: torch.device,
    encoder=None,
    rcn_runner=None,
    builder: Optional[HeteroGraphBuilder] = None,
) -> tuple[None, Tensor, Tensor, Tensor]:
    """Return ``conditioning, mu_HR, baseline_log, target_residual``.

    The conditioning value is currently ``None`` for causal-concat EDM. It is
    kept in the return signature to match existing notebook helpers.
    """
    target = batch["residual"][-1].to(device)
    if target.dim() == 3:
        target = target.unsqueeze(0)

    baseline = batch.get("baseline")
    if baseline is not None:
        baseline_log = baseline[-1].to(device)
        if baseline_log.dim() == 3:
            baseline_log = baseline_log.unsqueeze(0)
    else:
        baseline_log = torch.zeros_like(target)

    mu_hr = predict_mu_hr(
        batch,
        variant=variant,
        encoder=encoder,
        rcn_runner=rcn_runner,
        regression_head=regression_head,
        builder=builder,
        device=device,
        target_shape=target.shape[-2:],
    )

    mu_hr = torch.nan_to_num(mu_hr, nan=0.0, posinf=0.0, neginf=0.0)
    baseline_log = torch.nan_to_num(baseline_log, nan=0.0, posinf=0.0, neginf=0.0)
    return None, mu_hr, baseline_log, target


@torch.no_grad()
def sample_once_edm(
    diffusion,
    *,
    mu_HR: Tensor,
    baseline_log: Tensor,
    scheduler_type: str,
    num_steps: int,
    cfg_scale: float = 1.0,
    apply_constraints: bool = False,
):
    """Sample one residual field from a causal-concat diffusion decoder."""
    return diffusion.sample(
        conditioning=None,
        mu_HR=mu_HR,
        baseline_log=baseline_log,
        scheduler_type=scheduler_type,
        num_steps=num_steps,
        cfg_scale=cfg_scale,
        apply_constraints=apply_constraints,
    ).residual


__all__ = [
    "build_two_stage_batch",
    "build_two_stage_inputs",
    "sample_once_edm",
]
