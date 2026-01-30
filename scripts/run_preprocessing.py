"""
Script d'exécution pour le preprocessing NetCDF → Zarr/WebDataset.

Ce script permet de convertir des données NetCDF en format optimisé pour l'entraînement,
soit en Zarr (accès aléatoire) soit en WebDataset (lecture séquentielle).

Usage:
    # Avec Docker:
    docker-compose exec st-cdgm-training python scripts/run_preprocessing.py \
        --lr_path data/raw/lr.nc \
        --hr_path data/raw/hr.nc \
        --format zarr \
        --output_dir data/processed

    # Directement:
    python scripts/run_preprocessing.py \
        --lr_path data/raw/lr.nc \
        --hr_path data/raw/hr.nc \
        --format webdataset \
        --output_dir data/processed
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add workspace to path for imports
workspace_path = Path(__file__).parent.parent
if str(workspace_path) not in sys.path:
    sys.path.insert(0, str(workspace_path))
if str(workspace_path / "src") not in sys.path:
    sys.path.insert(0, str(workspace_path / "src"))

from ops.preprocess_to_zarr import convert_netcdf_to_zarr
from ops.preprocess_to_shards import main as preprocess_to_shards_main


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess NetCDF data to Zarr or WebDataset format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Input data
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
        help="Path to static high-resolution NetCDF file (optional)",
    )
    
    # Output format
    parser.add_argument(
        "--format",
        type=str,
        choices=["zarr", "webdataset"],
        default="zarr",
        help="Output format: 'zarr' for random access, 'webdataset' for sequential",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Output directory for processed data",
    )
    
    # Processing options
    parser.add_argument(
        "--seq_len",
        type=int,
        default=10,
        help="Sequence length for temporal windows",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding window",
    )
    parser.add_argument(
        "--baseline_strategy",
        type=str,
        choices=["hr_smoothing", "lr_interp"],
        default="hr_smoothing",
        help="Baseline computation strategy",
    )
    parser.add_argument(
        "--baseline_factor",
        type=int,
        default=4,
        help="Smoothing factor for baseline",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Enable normalization of LR data",
    )
    
    # WebDataset specific
    parser.add_argument(
        "--shard_size",
        type=int,
        default=1000,
        help="Number of samples per shard (WebDataset only)",
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not args.lr_path.exists():
        raise FileNotFoundError(f"LR file not found: {args.lr_path}")
    if not args.hr_path.exists():
        raise FileNotFoundError(f"HR file not found: {args.hr_path}")
    if args.static_path is not None and not args.static_path.exists():
        raise FileNotFoundError(f"Static file not found: {args.static_path}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("ST-CDGM Preprocessing")
    print("=" * 80)
    print(f"Format: {args.format}")
    print(f"LR Path: {args.lr_path}")
    print(f"HR Path: {args.hr_path}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Sequence Length: {args.seq_len}")
    print("=" * 80)
    
    # Convert based on format
    if args.format == "zarr":
        print("\nConverting to Zarr format...")
        convert_netcdf_to_zarr(
            lr_path=args.lr_path,
            hr_path=args.hr_path,
            output_dir=args.output_dir,
            static_path=args.static_path,
            seq_len=args.seq_len,
            stride=args.stride,
            baseline_strategy=args.baseline_strategy,
            baseline_factor=args.baseline_factor,
            normalize=args.normalize,
        )
        print(f"\n✓ Zarr conversion complete! Output: {args.output_dir}")
        
    elif args.format == "webdataset":
        print("\nConverting to WebDataset format...")
        # Create arguments for preprocess_to_shards
        shard_args = [
            "--lr_path", str(args.lr_path),
            "--hr_path", str(args.hr_path),
            "--output_dir", str(args.output_dir),
            "--seq_len", str(args.seq_len),
            "--stride", str(args.stride),
            "--shard_size", str(args.shard_size),
            "--baseline_strategy", args.baseline_strategy,
            "--baseline_factor", str(args.baseline_factor),
        ]
        if args.static_path is not None:
            shard_args.extend(["--static_path", str(args.static_path)])
        if args.normalize:
            shard_args.append("--normalize")
        
        # Modify sys.argv for preprocess_to_shards
        original_argv = sys.argv
        sys.argv = ["preprocess_to_shards.py"] + shard_args
        try:
            preprocess_to_shards_main()
        finally:
            sys.argv = original_argv
        
        print(f"\n✓ WebDataset conversion complete! Output: {args.output_dir}")
    
    print("\n" + "=" * 80)
    print("Preprocessing completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

