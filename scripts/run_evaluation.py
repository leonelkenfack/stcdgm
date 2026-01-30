"""
Script d'évaluation pour tester un modèle entraîné ST-CDGM.

Ce script charge un modèle sauvegardé, exécute l'inférence et calcule les métriques
d'évaluation (MSE, MAE, CRPS, FSS, F1 extremes, etc.).

Usage:
    # Avec Docker:
    docker-compose exec st-cdgm-training python scripts/run_evaluation.py \
        --checkpoint models/best_model.pt \
        --data_dir data/processed \
        --output_dir results/evaluation

    # Directement:
    python scripts/run_evaluation.py \
        --checkpoint models/best_model.pt \
        --lr_path data/raw/lr.nc \
        --hr_path data/raw/hr.nc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add workspace to path for imports
workspace_path = Path(__file__).parent.parent
if str(workspace_path) not in sys.path:
    sys.path.insert(0, str(workspace_path))
if str(workspace_path / "src") not in sys.path:
    sys.path.insert(0, str(workspace_path / "src"))

from st_cdgm import (
    NetCDFDataPipeline,
    HeteroGraphBuilder,
    IntelligibleVariableConfig,
    IntelligibleVariableEncoder,
    RCNCell,
    RCNSequenceRunner,
    CausalDiffusionDecoder,
)
from st_cdgm.evaluation import (
    autoregressive_inference,
    evaluate_metrics,
    MetricReport,
)
# Note: load_model will be imported if needed


def run_evaluation(
    checkpoint_path: Path,
    lr_path: Path,
    hr_path: Path,
    output_dir: Path,
    *,
    static_path: Optional[Path] = None,
    num_samples: int = 10,
    num_inference_steps: int = 25,
    scheduler_type: str = "edm",
    seq_len: int = 6,
    device: str = "cuda",
    compute_f1_extremes: bool = True,
    f1_percentiles: Sequence[float] = [95.0, 99.0],
    save_visualizations: bool = True,
) -> None:
    """Run evaluation on a trained model."""
    
    print("=" * 80)
    print("ST-CDGM Model Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"LR Path: {lr_path}")
    print(f"HR Path: {hr_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    
    # Load model
    print("\nLoading model...")
    from scripts.load_model import load_checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path, device=device_obj)
    
    # Setup data pipeline
    print("\nSetting up data pipeline...")
    pipeline = NetCDFDataPipeline(
        lr_path=str(lr_path),
        hr_path=str(hr_path),
        static_path=str(static_path) if static_path else None,
        seq_len=seq_len,
        normalize=True,
    )
    
    # Get test data (last sequence)
    dataset = pipeline.build_sequence_dataset(seq_len=seq_len, stride=1, as_torch=True)
    # Get a test sample (could be extended to use validation set)
    test_sample = next(iter(dataset))
    
    encoder = checkpoint_data["encoder"]
    rcn_runner = checkpoint_data["rcn_runner"]
    diffusion_decoder = checkpoint_data["diffusion_decoder"]
    config = checkpoint_data["config"]
    
    # Setup graph builder
    graph_cfg = config.get("graph", {})
    builder = HeteroGraphBuilder(
        lr_shape=tuple(graph_cfg.get("lr_shape", [23, 26])),
        hr_shape=tuple(graph_cfg.get("hr_shape", [172, 179])),
        static_dataset=pipeline.get_static_dataset(),
        static_variables=graph_cfg.get("static_variables", []),
        include_mid_layer=graph_cfg.get("include_mid_layer", False),
    )
    
    # Run inference
    print(f"\nRunning inference ({num_samples} samples, {num_inference_steps} steps, {scheduler_type} scheduler)...")
    
    # Prepare input data
    lr_seq = test_sample["lr"]  # [seq_len, channels, lat, lon]
    target = test_sample["hr"][-1]  # Last timestep [channels, H, W]
    baseline = test_sample.get("baseline", None)
    
    # Convert to batch format
    from ops.train_st_cdgm import _convert_sample_to_batch
    batch = _convert_sample_to_batch(
        test_sample, builder, device_obj
    )
    
    # Run autoregressive inference
    inference_results = autoregressive_inference(
        encoder=encoder,
        rcn_runner=rcn_runner,
        diffusion_decoder=diffusion_decoder,
        lr_sequence=batch["lr"],
        hetero_graph=batch["hetero"],
        num_samples=num_samples,
        num_steps=num_inference_steps,
        scheduler_type=scheduler_type,
        baseline=baseline,
        device=device_obj,
    )
    
    # Compute metrics
    print("\nComputing metrics...")
    target_tensor = torch.from_numpy(target).unsqueeze(0).to(device_obj)  # [1, C, H, W]
    if baseline is not None:
        baseline_tensor = torch.from_numpy(baseline).unsqueeze(0).to(device_obj)
    else:
        baseline_tensor = None
    
    metrics = evaluate_metrics(
        samples=inference_results.samples,
        target=target_tensor,
        baseline=baseline_tensor,
        compute_advanced=True,
        compute_f1_extremes=compute_f1_extremes,
        f1_percentiles=list(f1_percentiles),
    )
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    results_dict = {
        "mse": metrics.mse,
        "mae": metrics.mae,
        "hist_distance": metrics.hist_distance,
        "crps": metrics.crps,
        "spectrum_distance": metrics.spectrum_distance,
        "fss": metrics.fss,
        "wasserstein_distance": metrics.wasserstein_distance,
        "energy_score": metrics.energy_score,
        "f1_extremes": metrics.f1_extremes,
        "baseline_mse": metrics.baseline_mse,
        "baseline_mae": metrics.baseline_mae,
    }
    
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Results Summary")
    print("=" * 80)
    print(f"MSE: {metrics.mse:.6f}")
    print(f"MAE: {metrics.mae:.6f}")
    print(f"CRPS: {metrics.crps:.6f}")
    if metrics.fss is not None:
        print(f"FSS: {metrics.fss:.6f}")
    if metrics.f1_extremes is not None:
        print(f"F1 Extremes:")
        for threshold, f1_score in metrics.f1_extremes.items():
            print(f"  {threshold}: {f1_score:.4f}")
    print("=" * 80)
    
    # Save visualizations if requested
    if save_visualizations:
        print("\nGenerating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot sample predictions
        pred_mean = inference_results.samples[0].t_mean.cpu().numpy()
        
        # Simple visualization (can be extended)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(target[0], cmap='viridis')
        axes[0].set_title("Target")
        axes[1].imshow(pred_mean[0], cmap='viridis')
        axes[1].set_title("Prediction (Mean)")
        axes[2].imshow(np.abs(target[0] - pred_mean[0]), cmap='hot')
        axes[2].set_title("Absolute Error")
        
        vis_path = vis_dir / "prediction_comparison.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved: {vis_dir}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ST-CDGM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--lr_path",
        type=Path,
        required=True,
        help="Path to low-resolution NetCDF file",
    )
    parser.add_argument(
        "--hr_path",
        type=Path,
        required=True,
        help="Path to high-resolution NetCDF file",
    )
    parser.add_argument(
        "--static_path",
        type=Path,
        default=None,
        help="Path to static NetCDF file (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="results/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of diffusion steps for inference",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=["ddpm", "edm", "dpm_solver++"],
        default="edm",
        help="Diffusion scheduler type",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=6,
        help="Sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--no_f1_extremes",
        action="store_true",
        help="Disable F1 extremes computation",
    )
    parser.add_argument(
        "--no_visualizations",
        action="store_true",
        help="Disable visualization generation",
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        checkpoint_path=args.checkpoint,
        lr_path=args.lr_path,
        hr_path=args.hr_path,
        output_dir=args.output_dir,
        static_path=args.static_path,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        scheduler_type=args.scheduler_type,
        seq_len=args.seq_len,
        device=args.device,
        compute_f1_extremes=not args.no_f1_extremes,
        save_visualizations=not args.no_visualizations,
    )


if __name__ == "__main__":
    main()

