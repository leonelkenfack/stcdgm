"""
Utility script to save a model checkpoint with metadata.

Usage:
    python scripts/save_model.py \
        --encoder model_encoder.pt \
        --rcn model_rcn.pt \
        --diffusion model_diffusion.pt \
        --output models/checkpoint.pt \
        --config config/training_config.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf


def save_model_checkpoint(
    encoder_path: Path,
    rcn_path: Path,
    diffusion_path: Path,
    output_path: Path,
    config_path: Optional[Path] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Save a combined model checkpoint with metadata."""
    
    # Load model states
    encoder_state = torch.load(encoder_path, map_location="cpu")
    rcn_state = torch.load(rcn_path, map_location="cpu")
    diffusion_state = torch.load(diffusion_path, map_location="cpu")
    
    # Load config if provided
    config = None
    if config_path and config_path.exists():
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
    
    # Create checkpoint
    checkpoint = {
        "encoder_state_dict": encoder_state,
        "rcn_state_dict": rcn_state,
        "diffusion_state_dict": diffusion_state,
        "config": config,
        "metrics": metrics or {},
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    
    # Save metadata as JSON
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "encoder_path": str(encoder_path),
        "rcn_path": str(rcn_path),
        "diffusion_path": str(diffusion_path),
        "config_path": str(config_path) if config_path else None,
        "metrics": metrics,
        "timestamp": checkpoint["timestamp"],
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Checkpoint saved: {output_path}")
    print(f"✓ Metadata saved: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save model checkpoint")
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--rcn", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None, help="JSON file with metrics")
    
    args = parser.parse_args()
    
    metrics = None
    if args.metrics and args.metrics.exists():
        with open(args.metrics) as f:
            metrics = json.load(f)
    
    save_model_checkpoint(
        args.encoder, args.rcn, args.diffusion,
        args.output, args.config, metrics
    )

