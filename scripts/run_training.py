"""
Script d'exécution pour l'entraînement ST-CDGM avec checkpointing et callbacks.

Ce script lance l'entraînement complet avec support pour:
- Checkpointing automatique
- Early Stopping
- LR Scheduling
- Sauvegarde des modèles
- Logging structuré

Usage:
    # Avec Docker:
    docker-compose exec st-cdgm-training python scripts/run_training.py \
        --config config/training_config.yaml

    # Directement:
    python scripts/run_training.py \
        --config config/training_config.yaml \
        --checkpoint_dir models \
        --save_every 5
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add workspace to path for imports
workspace_path = Path(__file__).parent.parent
if str(workspace_path) not in sys.path:
    sys.path.insert(0, str(workspace_path))
if str(workspace_path / "src") not in sys.path:
    sys.path.insert(0, str(workspace_path / "src"))

from ops.train_st_cdgm import main as train_main
from st_cdgm import (
    NetCDFDataPipeline,
    HeteroGraphBuilder,
    IntelligibleVariableConfig,
    IntelligibleVariableEncoder,
    RCNCell,
    RCNSequenceRunner,
    CausalDiffusionDecoder,
    train_epoch,
)
from st_cdgm.training import EarlyStopping

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


def save_checkpoint(
    encoder: IntelligibleVariableEncoder,
    rcn_cell: RCNCell,
    diffusion_decoder: CausalDiffusionDecoder,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> Path:
    """Save model checkpoint with metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_best:
        checkpoint_path = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}_{timestamp}.pt"
    
    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "rcn_state_dict": rcn_cell.state_dict(),
        "diffusion_state_dict": diffusion_decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": timestamp,
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Save metadata as JSON
    metadata_path = checkpoint_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump({
            "epoch": epoch,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "timestamp": timestamp,
            "is_best": is_best,
        }, f, indent=2)
    
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    encoder: IntelligibleVariableEncoder,
    rcn_cell: RCNCell,
    diffusion_decoder: CausalDiffusionDecoder,
    optimizer: Optional[optim.Optimizer] = None,
) -> tuple[int, dict]:
    """Load model checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    rcn_cell.load_state_dict(checkpoint["rcn_state_dict"])
    diffusion_decoder.load_state_dict(checkpoint["diffusion_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    
    print(f"✓ Checkpoint loaded: {checkpoint_path} (epoch {epoch})")
    return epoch, metrics


def run_training_with_checkpoints(
    cfg: DictConfig,
    checkpoint_dir: Path,
    save_every: int = 5,
    max_checkpoints: int = 5,
    resume_from: Optional[Path] = None,
) -> None:
    """Run training with checkpointing and callbacks."""
    
    print("=" * 80)
    print("ST-CDGM Training with Checkpointing")
    print("=" * 80)
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Save Every: {save_every} epochs")
    print(f"Max Checkpoints: {max_checkpoints}")
    print("=" * 80)
    
    # Import training setup from train_st_cdgm
    # We'll need to refactor train_st_cdgm to expose setup functions
    # For now, we'll call the main function but with checkpointing logic
    
    # This is a simplified version - in production, we'd refactor train_st_cdgm
    # to separate setup from training loop
    
    device = torch.device(cfg.training.device)
    
    # Setup models (same as train_st_cdgm)
    from ops.train_st_cdgm import _build_encoder, _iterate_batches
    
    pipeline = NetCDFDataPipeline(
        lr_path=cfg.data.lr_path,
        hr_path=cfg.data.hr_path,
        static_path=cfg.data.static_path,
        seq_len=cfg.data.seq_len,
        baseline_strategy=cfg.data.baseline_strategy,
        baseline_factor=cfg.data.baseline_factor,
        normalize=cfg.data.normalize,
    )
    
    from torch.utils.data import DataLoader
    dataset = pipeline.build_sequence_dataset(
        seq_len=cfg.data.seq_len,
        stride=cfg.data.stride,
        as_torch=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2,
        persistent_workers=True,
    )
    
    builder = HeteroGraphBuilder(
        lr_shape=tuple(cfg.graph.lr_shape),
        hr_shape=tuple(cfg.graph.hr_shape),
        static_dataset=pipeline.get_static_dataset(),
        static_variables=cfg.graph.static_variables,
        include_mid_layer=cfg.graph.include_mid_layer,
    )
    
    encoder = _build_encoder(cfg.encoder).to(device)
    rcn_cell = RCNCell(
        num_vars=len(cfg.encoder.metapaths),
        hidden_dim=cfg.rcn.hidden_dim,
        driver_dim=cfg.rcn.driver_dim,
        reconstruction_dim=cfg.rcn.reconstruction_dim,
        dropout=cfg.rcn.dropout,
    ).to(device)
    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=cfg.rcn.detach_interval)
    
    diffusion = CausalDiffusionDecoder(
        in_channels=cfg.diffusion.in_channels,
        conditioning_dim=cfg.diffusion.conditioning_dim,
        height=cfg.diffusion.height,
        width=cfg.diffusion.width,
        num_diffusion_steps=cfg.diffusion.steps,
        use_gradient_checkpointing=cfg.diffusion.get("use_gradient_checkpointing", False),
    ).to(device)
    
    # Compile models if enabled
    if cfg.training.compile.get("enabled", False):
        if hasattr(torch, 'compile'):
            compile_mode_rcn = cfg.training.compile.get("rcn_mode", "reduce-overhead")
            compile_mode_diffusion = cfg.training.compile.get("diffusion_mode", "max-autotune")
            compile_mode_encoder = cfg.training.compile.get("encoder_mode", "reduce-overhead")
            
            try:
                rcn_cell = torch.compile(rcn_cell, mode=compile_mode_rcn)
                rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=cfg.rcn.detach_interval)
                print(f"✓ RCN cell compiled with torch.compile (mode: {compile_mode_rcn})")
            except Exception as e:
                print(f"⚠ torch.compile for RCN cell failed: {e}")
            
            try:
                diffusion = torch.compile(diffusion, mode=compile_mode_diffusion)
                print(f"✓ Diffusion decoder compiled with torch.compile (mode: {compile_mode_diffusion})")
            except Exception as e:
                print(f"⚠ torch.compile for diffusion decoder failed: {e}")
            
            try:
                encoder = torch.compile(encoder, mode=compile_mode_encoder, fullgraph=False)
                print(f"✓ Encoder compiled with torch.compile (mode: {compile_mode_encoder})")
            except Exception as e:
                print(f"⚠ torch.compile for encoder failed: {e}")
    
    params = list(encoder.parameters()) + list(rcn_cell.parameters()) + list(diffusion.parameters())
    optimizer = optim.Adam(params, lr=cfg.training.lr)
    
    # Setup LR scheduler
    scheduler = None
    if cfg.training.lr_scheduler.get("enabled", False):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=cfg.training.lr_scheduler.get("mode", "min"),
            factor=cfg.training.lr_scheduler.get("factor", 0.5),
            patience=cfg.training.lr_scheduler.get("patience", 3),
            min_lr=cfg.training.lr_scheduler.get("min_lr", 1e-7),
        )
        print("✓ LR Scheduler enabled")
    
    # Setup Early Stopping
    early_stopping = None
    if cfg.training.early_stopping.get("enabled", False):
        early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping.get("patience", 7),
            min_delta=cfg.training.early_stopping.get("min_delta", 0.0),
            restore_best=cfg.training.early_stopping.get("restore_best", True),
            verbose=True,
        )
        print("✓ Early Stopping enabled")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    if resume_from is not None:
        start_epoch, metrics = load_checkpoint(
            resume_from, encoder, rcn_cell, diffusion, optimizer
        )
        best_loss = metrics.get("loss", float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop with checkpointing
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, cfg.training.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{cfg.training.epochs}")
        print(f"{'='*80}")
        
        batch_iter = _iterate_batches(dataloader, builder, device)
        
        # Get training configuration for train_epoch
        metrics = train_epoch(
            encoder=encoder,
            rcn_runner=rcn_runner,
            diffusion_decoder=diffusion,
            optimizer=optimizer,
            data_loader=batch_iter,
            lambda_gen=cfg.loss.lambda_gen,
            beta_rec=cfg.loss.beta_rec,
            gamma_dag=cfg.loss.gamma_dag,
            conditioning_fn=None,
            device=device,
            gradient_clipping=cfg.training.gradient_clipping,
            dag_method=cfg.loss.get("dag_method", "dagma"),
            dagma_s=cfg.loss.get("dagma_s", 1.0),
            lambda_phy=cfg.loss.get("lambda_phy", 0.0),
            use_predicted_output=cfg.training.physical_loss.get("use_predicted_output", False),
            physical_sample_interval=cfg.training.physical_loss.get("physical_sample_interval", 10),
            physical_num_steps=cfg.training.physical_loss.get("physical_num_steps", 15),
            use_amp=cfg.training.get("use_amp", True),
            scheduler=scheduler,
            use_focal_loss=cfg.loss.get("use_focal_loss", False),
            focal_alpha=cfg.loss.get("focal_alpha", 1.0),
            focal_gamma=cfg.loss.get("focal_gamma", 2.0),
            extreme_weight_factor=cfg.loss.get("extreme_weight_factor", 0.0),
            extreme_percentiles=cfg.loss.get("extreme_percentiles", [95.0, 99.0]),
            reconstruction_loss_type=cfg.loss.get("reconstruction_loss_type", "mse"),
        )
        
        current_loss = metrics["loss"]
        
        # Update LR scheduler
        if scheduler is not None:
            scheduler.step(current_loss)
        
        # Early stopping check
        if early_stopping is not None:
            # For early stopping, we'd need validation loss
            # For now, use training loss (not ideal, but functional)
            if early_stopping(current_loss, rcn_cell):  # Use rcn_cell as model proxy
                print("Early stopping triggered!")
                break
        
        # Save checkpoint
        is_best = current_loss < best_loss
        if is_best:
            best_loss = current_loss
        
        if (epoch + 1) % save_every == 0 or is_best:
            checkpoint_path = save_checkpoint(
                encoder, rcn_cell, diffusion, optimizer,
                epoch + 1, metrics, checkpoint_dir, is_best=is_best
            )
            
            # Clean up old checkpoints (keep only last N)
            if not is_best:  # Don't delete best model
                checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                if len(checkpoints) > max_checkpoints:
                    for old_checkpoint in checkpoints[:-max_checkpoints]:
                        old_checkpoint.unlink()
                        old_checkpoint.with_suffix(".json").unlink(missing_ok=True)
                        print(f"  Deleted old checkpoint: {old_checkpoint.name}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train ST-CDGM model with checkpointing and callbacks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default="config/training_config.yaml",
        help="Path to Hydra config file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default="models",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Load Hydra config
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Use Hydra to load config
    # Since we're calling from outside Hydra, we need to use OmegaConf directly
    cfg = OmegaConf.load(args.config)
    
    # Convert to DictConfig for compatibility
    cfg = OmegaConf.structured(cfg) if OmegaConf.is_config(cfg) else cfg
    
    run_training_with_checkpoints(
        cfg,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        max_checkpoints=args.max_checkpoints,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()

