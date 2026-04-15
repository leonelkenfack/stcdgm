"""
Utility script to load a model checkpoint.

This module provides functions to load ST-CDGM model checkpoints and restore
the full training state (models, optimizer, config, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import torch

from st_cdgm import (
    IntelligibleVariableEncoder,
    IntelligibleVariableConfig,
    RCNCell,
    RCNSequenceRunner,
    CausalDiffusionDecoder,
    SpatialConditioningProjector,
    CausalConditioningProjector,
)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
    return_full_state: bool = False,
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file (.pt)
    device : torch.device
        Device to load models onto
    return_full_state : bool
        If True, return optimizer state and full config
    
    Returns
    -------
    Dict containing:
        - encoder: IntelligibleVariableEncoder
        - rcn_cell: RCNCell
        - rcn_runner: RCNSequenceRunner
        - diffusion_decoder: CausalDiffusionDecoder
        - config: DictConfig (if available)
        - metrics: dict (if available)
        - optimizer_state_dict: dict (if return_full_state=True)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract components
    encoder_state = checkpoint["encoder_state_dict"]
    rcn_state = checkpoint.get("rcn_state_dict") or checkpoint.get("rcn_cell_state_dict")
    if rcn_state is None:
        raise KeyError("Checkpoint missing rcn_state_dict or rcn_cell_state_dict")
    diffusion_state = checkpoint["diffusion_state_dict"]
    config = checkpoint.get("config", {})
    metrics = checkpoint.get("metrics", {})
    
    # Reconstruct models from config
    # This assumes config contains the necessary info to rebuild models
    if not config:
        raise ValueError("Checkpoint must contain config to reconstruct models")
    
    # Build encoder
    encoder_cfg = config.get("encoder", {})
    meta_configs = [
        IntelligibleVariableConfig(
            name=mp.get("name", ""),
            meta_path=(mp["src"], mp["relation"], mp["target"]),
            pool=mp.get("pool", "mean"),
        )
        for mp in encoder_cfg.get("metapaths", [])
    ]
    encoder = IntelligibleVariableEncoder(
        configs=meta_configs,
        hidden_dim=encoder_cfg.get("hidden_dim", 128),
        conditioning_dim=encoder_cfg.get("conditioning_dim", 128),
    )
    encoder.load_state_dict(encoder_state)
    encoder.to(device)
    encoder.eval()

    _proj_expected = len(encoder.configs) * encoder.conditioning_dim
    
    # Build RCN
    rcn_cfg = config.get("rcn", {})
    rcn_cell = RCNCell(
        num_vars=len(encoder.configs),
        hidden_dim=rcn_cfg.get("hidden_dim", 128),
        driver_dim=rcn_cfg.get("driver_dim", 8),
        reconstruction_dim=rcn_cfg.get("reconstruction_dim", 8),
        dropout=rcn_cfg.get("dropout", 0.0),
    )
    rcn_cell.load_state_dict(rcn_state)
    rcn_cell.to(device)
    rcn_cell.eval()
    rcn_runner = RCNSequenceRunner(
        rcn_cell,
        detach_interval=rcn_cfg.get("detach_interval", None),
    )
    
    # Build diffusion decoder
    diffusion_cfg = config.get("diffusion", {})
    unet_kwargs = dict(diffusion_cfg.get("unet_kwargs", {})) or dict(
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        mid_block_type="UNetMidBlock2D",
        norm_num_groups=8,
        class_embed_type="projection",
        projection_class_embeddings_input_dim=640,
        resnet_time_scale_shift="scale_shift",
        attention_head_dim=32,
        only_cross_attention=[False, True],
    )
    _proj_ckpt = unet_kwargs.get("projection_class_embeddings_input_dim")
    unet_kwargs["projection_class_embeddings_input_dim"] = _proj_expected
    if _proj_ckpt is not None and int(_proj_ckpt) != int(_proj_expected):
        print(
            f"⚠ projection_class_embeddings_input_dim: config/checkpoint had {_proj_ckpt}, "
            f"rebuilt U-Net uses {_proj_expected} (len(encoder.configs) × conditioning_dim). "
            "Old checkpoints trained with a mismatched FiLM width may fail strict load."
        )
    diffusion = CausalDiffusionDecoder(
        in_channels=diffusion_cfg.get("in_channels", 1),
        conditioning_dim=diffusion_cfg.get("conditioning_dim", 128),
        height=diffusion_cfg.get("height", 172),
        width=diffusion_cfg.get("width", 179),
        num_diffusion_steps=diffusion_cfg.get("steps", 1000),
        unet_kwargs=unet_kwargs,
        use_gradient_checkpointing=diffusion_cfg.get("use_gradient_checkpointing", False),
        scheduler_type=diffusion_cfg.get("scheduler_type", "ddpm"),
        conv_padding_mode=diffusion_cfg.get("conv_padding_mode", "zeros"),
        anti_checkerboard=diffusion_cfg.get("anti_checkerboard", False),
    )
    diffusion.load_state_dict(diffusion_state)
    diffusion.to(device)
    diffusion.eval()

    # Sprint 1 / B4 + Sprint 2 fix: rebuild the conditioning projector when
    # the checkpoint contains its state dict. We detect whether the saved
    # state belongs to a ``CausalConditioningProjector`` (Sprint 2) by
    # looking for ``dag_mlp`` keys; otherwise we fall back to the Sprint 1
    # ``SpatialConditioningProjector``.
    spatial_projector = None
    spatial_state = (
        checkpoint.get("spatial_projector_state_dict")
        or checkpoint.get("spatial_projector_state")
    )
    if spatial_state is not None:
        graph_cfg = config.get("graph", {})
        encoder_cfg = config.get("encoder", {}) if isinstance(config, dict) else {}
        is_causal_proj = any("dag_mlp" in k for k in spatial_state.keys())
        common_kwargs = dict(
            num_vars=len(encoder.configs),
            hidden_dim=rcn_cfg.get("hidden_dim", 128),
            conditioning_dim=diffusion_cfg.get("conditioning_dim", 128),
            lr_shape=tuple(graph_cfg.get("lr_shape", [23, 26])),
            target_shape=tuple(diffusion_cfg.get("spatial_target_shape", [6, 7])),
        )
        if is_causal_proj:
            spatial_projector = CausalConditioningProjector(
                **common_kwargs,
                num_dag_tokens=int(encoder_cfg.get("num_dag_tokens", 1)),
            )
        else:
            spatial_projector = SpatialConditioningProjector(**common_kwargs)
        spatial_projector.load_state_dict(spatial_state)
        spatial_projector.to(device)
        spatial_projector.eval()

    result = {
        "encoder": encoder,
        "rcn_cell": rcn_cell,
        "rcn_runner": rcn_runner,
        "diffusion_decoder": diffusion,
        "spatial_projector": spatial_projector,
        "config": config,
        "metrics": metrics,
    }
    
    if return_full_state and "optimizer_state_dict" in checkpoint:
        result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
    
    print("✓ Checkpoint loaded successfully")
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and verify model checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    
    print("\nCheckpoint Summary:")
    print(f"  Config: {checkpoint['config'] is not None}")
    print(f"  Metrics: {list(checkpoint['metrics'].keys())}")
    print(f"  Models loaded on: {device}")

