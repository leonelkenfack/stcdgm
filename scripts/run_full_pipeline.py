"""
Script orchestrateur pour exécuter le pipeline complet ST-CDGM.

Ce script exécute dans l'ordre:
1. Preprocessing (NetCDF → Zarr/WebDataset)
2. Training (avec checkpointing)
3. Evaluation (sur le modèle entraîné)

Usage:
    python scripts/run_full_pipeline.py \
        --lr_path data/raw/lr.nc \
        --hr_path data/raw/hr.nc \
        --config config/training_config.yaml \
        --format zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add workspace to path
workspace_path = Path(__file__).parent.parent
if str(workspace_path) not in sys.path:
    sys.path.insert(0, str(workspace_path))
if str(workspace_path / "src") not in sys.path:
    sys.path.insert(0, str(workspace_path / "src"))

from scripts.run_preprocessing import main as preprocess_main
from scripts.run_training import run_training_with_checkpoints
from scripts.run_evaluation import run_evaluation
from omegaconf import OmegaConf


def run_full_pipeline(
    lr_path: Path,
    hr_path: Path,
    config_path: Path,
    *,
    static_path: Path | None = None,
    format: str = "zarr",
    checkpoint_dir: Path = Path("models"),
    results_dir: Path = Path("results"),
    skip_preprocessing: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False,
) -> None:
    """Run the complete ST-CDGM pipeline."""
    
    print("=" * 80)
    print("ST-CDGM Full Pipeline")
    print("=" * 80)
    print(f"LR Path: {lr_path}")
    print(f"HR Path: {hr_path}")
    print(f"Config: {config_path}")
    print(f"Format: {format}")
    print("=" * 80)
    
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Step 1: Preprocessing
    processed_dir = Path("data/processed")
    if not skip_preprocessing:
        print("\n" + "=" * 80)
        print("STEP 1: Preprocessing")
        print("=" * 80)
        
        # Prepare preprocessing arguments
        preprocess_args = [
            "--lr_path", str(lr_path),
            "--hr_path", str(hr_path),
            "--format", format,
            "--output_dir", str(processed_dir),
            "--seq_len", str(cfg.data.get("seq_len", 10)),
            "--baseline_strategy", cfg.data.get("baseline_strategy", "hr_smoothing"),
        ]
        
        if static_path:
            preprocess_args.extend(["--static_path", str(static_path)])
        if cfg.data.get("normalize", False):
            preprocess_args.append("--normalize")
        
        # Run preprocessing (we'll need to modify to accept args)
        # For now, call the main function directly
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "run_preprocessing.py")] + preprocess_args,
            check=True,
        )
        print("✓ Preprocessing completed")
    else:
        print("\n⏭ Skipping preprocessing (using existing data)")
    
    # Step 2: Training
    if not skip_training:
        print("\n" + "=" * 80)
        print("STEP 2: Training")
        print("=" * 80)
        
        run_training_with_checkpoints(
            cfg,
            checkpoint_dir=checkpoint_dir,
            save_every=cfg.checkpoint.get("save_every", 5),
            max_checkpoints=cfg.checkpoint.get("max_checkpoints", 5),
        )
        print("✓ Training completed")
    else:
        print("\n⏭ Skipping training")
    
    # Step 3: Evaluation
    if not skip_evaluation:
        print("\n" + "=" * 80)
        print("STEP 3: Evaluation")
        print("=" * 80)
        
        # Find best model checkpoint
        best_checkpoint = checkpoint_dir / "best_model.pt"
        if not best_checkpoint.exists():
            # Find latest checkpoint
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if checkpoints:
                best_checkpoint = checkpoints[-1]
            else:
                print("⚠ No checkpoint found, skipping evaluation")
                return
        
        run_evaluation(
            checkpoint_path=best_checkpoint,
            lr_path=lr_path,
            hr_path=hr_path,
            output_dir=results_dir / "evaluation",
            static_path=static_path,
            num_samples=cfg.evaluation.get("num_samples", 10),
            num_inference_steps=25,
            scheduler_type=cfg.diffusion.get("scheduler_type", "edm"),
            seq_len=cfg.data.get("seq_len", 6),
            device=cfg.training.get("device", "cuda"),
            compute_f1_extremes=cfg.evaluation.get("compute_f1_extremes", True),
            save_visualizations=cfg.evaluation.get("save_visualizations", True),
        )
        print("✓ Evaluation completed")
    else:
        print("\n⏭ Skipping evaluation")
    
    print("\n" + "=" * 80)
    print("✓ Full pipeline completed successfully!")
    print("=" * 80)
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Results: {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete ST-CDGM pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--lr_path", type=Path, required=True)
    parser.add_argument("--hr_path", type=Path, required=True)
    parser.add_argument("--static_path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default="config/training_config.yaml")
    parser.add_argument("--format", type=str, choices=["zarr", "webdataset"], default="zarr")
    parser.add_argument("--checkpoint_dir", type=Path, default="models")
    parser.add_argument("--results_dir", type=Path, default="results")
    parser.add_argument("--skip_preprocessing", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        lr_path=args.lr_path,
        hr_path=args.hr_path,
        config_path=args.config,
        static_path=args.static_path,
        format=args.format,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        skip_preprocessing=args.skip_preprocessing,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation,
    )


if __name__ == "__main__":
    main()

