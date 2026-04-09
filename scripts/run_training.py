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
import os
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
    SpatialConditioningProjector,
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
    
    n_threads = max(1, (os.cpu_count() or 1) // 2)
    torch.set_num_threads(n_threads)
    print(f"[PERF] torch.set_num_threads({n_threads})")

    device = torch.device(cfg.training.device)
    
    # Setup models (same as train_st_cdgm)
    from ops.train_st_cdgm import build_encoder_for_graph, _iterate_batches
    
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

    sample_for_shapes = next(iter(dataset))
    rcn_driver_dim = sample_for_shapes["lr"].shape[1]
    hr_channels = sample_for_shapes["residual"].shape[1]
    print(f"[DIM] Inferred from data: rcn_driver_dim={rcn_driver_dim}, hr_channels={hr_channels}")

    dataset = pipeline.build_sequence_dataset(
        seq_len=cfg.data.seq_len,
        stride=cfg.data.stride,
        as_torch=True,
    )
    num_workers = int(cfg.training.get("num_workers", 0))
    dataloader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    
    builder = HeteroGraphBuilder(
        lr_shape=tuple(cfg.graph.lr_shape),
        hr_shape=tuple(cfg.graph.hr_shape),
        static_dataset=pipeline.get_static_dataset(),
        static_variables=cfg.graph.static_variables,
        include_mid_layer=cfg.graph.include_mid_layer,
    )
    
    encoder = build_encoder_for_graph(cfg.encoder, builder).to(device)
    rcn_cell = RCNCell(
        num_vars=len(encoder.configs),
        hidden_dim=cfg.rcn.hidden_dim,
        driver_dim=rcn_driver_dim,
        reconstruction_dim=rcn_driver_dim,
        dropout=cfg.rcn.dropout,
    ).to(device)
    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=cfg.rcn.detach_interval)
    
    unet_kwargs = dict(cfg.diffusion.unet_kwargs) if cfg.diffusion.get("unet_kwargs") else dict(
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
    unet_kwargs["projection_class_embeddings_input_dim"] = (
        len(encoder.configs) * encoder.conditioning_dim
    )
    diffusion = CausalDiffusionDecoder(
        in_channels=hr_channels,
        conditioning_dim=cfg.diffusion.conditioning_dim,
        height=cfg.diffusion.height,
        width=cfg.diffusion.width,
        num_diffusion_steps=cfg.diffusion.steps,
        unet_kwargs=unet_kwargs,
        scheduler_type=cfg.diffusion.get("scheduler_type", "ddpm"),
        use_gradient_checkpointing=cfg.diffusion.get("use_gradient_checkpointing", False),
        conv_padding_mode=cfg.diffusion.get("conv_padding_mode", "zeros"),
        anti_checkerboard=cfg.diffusion.get("anti_checkerboard", False),
    ).to(device)
    
    # Compile models if enabled
    compile_cfg = cfg.training.get("compile", {}) or {}
    if compile_cfg.get("enabled", False):
        if hasattr(torch, 'compile'):
            compile_mode_rcn = compile_cfg.get("rcn_mode", "reduce-overhead")
            compile_mode_diffusion = compile_cfg.get("diffusion_mode", "max-autotune")
            compile_mode_encoder = compile_cfg.get("encoder_mode", "reduce-overhead")
            _cuda = torch.cuda.is_available()

            try:
                rcn_cell = torch.compile(rcn_cell, mode=compile_mode_rcn)
                rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=cfg.rcn.detach_interval)
                print(f"✓ RCN cell compiled with torch.compile (mode: {compile_mode_rcn})")
            except Exception as e:
                print(f"⚠ torch.compile for RCN cell failed: {e}")

            try:
                if _cuda:
                    diffusion = torch.compile(diffusion, mode=compile_mode_diffusion)
                    print(f"✓ Diffusion decoder compiled with torch.compile (mode: {compile_mode_diffusion})")
                else:
                    print("⚠ Skipping torch.compile for diffusion decoder (CUDA not available)")
            except Exception as e:
                print(f"⚠ torch.compile for diffusion decoder failed: {e}")
            
            try:
                encoder = torch.compile(encoder, mode=compile_mode_encoder, fullgraph=False)
                print(f"✓ Encoder compiled with torch.compile (mode: {compile_mode_encoder})")
            except Exception as e:
                print(f"⚠ torch.compile for encoder failed: {e}")
    
    spatial_projector = SpatialConditioningProjector(
        num_vars=len(encoder.configs),
        hidden_dim=cfg.rcn.hidden_dim,
        conditioning_dim=cfg.diffusion.conditioning_dim,
        lr_shape=tuple(cfg.graph.lr_shape),
        target_shape=tuple(cfg.diffusion.get("spatial_target_shape", [6, 7])),
    ).to(device)

    params = (
        list(encoder.parameters()) + list(rcn_cell.parameters())
        + list(diffusion.parameters()) + list(spatial_projector.parameters())
    )
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
            use_spectral_loss=cfg.loss.get("use_spectral_loss", False),
            lambda_spectral=cfg.loss.get("lambda_spectral", 0.0),
            conditioning_dropout_prob=cfg.diffusion.get("conditioning_dropout_prob", 0.0),
            lambda_dag_prior=cfg.loss.get("lambda_dag_prior", 0.0),
            dag_prior=torch.tensor(cfg.loss.dag_prior, dtype=torch.float32) if cfg.loss.get("dag_prior") else None,
            spatial_projector=spatial_projector,
        )

        if cfg.loss.get("log_spectral_metric_each_epoch", False):
            from st_cdgm.training.training_loop import (
                compute_rapsd_metric_from_batch,
                resolve_train_amp_mode,
            )

            amp_m = resolve_train_amp_mode(device, cfg.training.get("use_amp", True))
            try:
                metric_iter = _iterate_batches(dataloader, builder, device)
                batch0 = next(metric_iter)
                rapsd_v = compute_rapsd_metric_from_batch(
                    encoder=encoder,
                    rcn_runner=rcn_runner,
                    diffusion_decoder=diffusion,
                    batch=batch0,
                    device=device,
                    amp_mode=amp_m,
                )
                if rapsd_v is not None:
                    print(f"[Epoch {epoch + 1}] RAPSD metric (epoch end): {rapsd_v:.6f}")
            except StopIteration:
                pass
            except Exception as e:
                print(f"[WARN] RAPSD epoch metric failed: {e}")
        
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

