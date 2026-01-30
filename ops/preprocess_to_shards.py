"""
Phase B3: Preprocessing script to convert NetCDF data to WebDataset format (TAR shards).

WebDataset stores each sample as individual files in TAR archives, optimized for
sequential reading during training. Provides 5-10x better throughput than Zarr for
sequential access patterns.

Usage:
    python ops/preprocess_to_shards.py \
        --lr_path data/raw/lr.nc \
        --hr_path data/raw/hr.nc \
        --output_dir data/webdataset \
        --seq_len 10 \
        --shard_size 1000
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

try:
    from webdataset import ShardWriter
    HAS_WEBDATASET = True
except ImportError:
    HAS_WEBDATASET = False

from st_cdgm import NetCDFDataPipeline


def create_shards(
    pipeline: NetCDFDataPipeline,
    output_dir: Path,
    seq_len: int,
    shard_size: int = 1000,
    stride: int = 1,
) -> None:
    """
    Convert NetCDF pipeline data to WebDataset TAR shards.
    
    Parameters
    ----------
    pipeline : NetCDFDataPipeline
        Initialized data pipeline
    output_dir : Path
        Output directory for shard files
    seq_len : int
        Sequence length for each sample
    shard_size : int
        Number of samples per shard (default: 1000)
    stride : int
        Stride for sequence windows (default: 1)
    """
    if not HAS_WEBDATASET:
        raise ImportError(
            "WebDataset is not installed. Install via: pip install webdataset"
        )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build dataset iterator
    dataset = pipeline.build_sequence_dataset(
        seq_len=seq_len,
        stride=stride,
        as_torch=True,
    )
    
    # Pattern for shard files: data_%06d.tar
    shard_pattern = str(output_dir / "data_%06d.tar")
    
    # Create ShardWriter
    with ShardWriter(shard_pattern, maxcount=shard_size) as writer:
        sample_idx = 0
        
        for sample in dataset:
            # Convert sample to WebDataset format
            # WebDataset expects: {"__key__": key, "__ext__": ext, ...data...}
            
            key = f"{sample_idx:08d}"
            
            # Save each tensor component as .pt files
            sample_dict = {
                "__key__": key,
            }
            
            # Save LR data: [seq_len, channels, H, W]
            if "lr" in sample:
                sample_dict["lr.pt"] = sample["lr"]
            
            # Save baseline: [seq_len, channels, H, W]
            if "baseline" in sample:
                sample_dict["baseline.pt"] = sample["baseline"]
            
            # Save residual: [seq_len, channels, H, W]
            if "residual" in sample:
                sample_dict["residual.pt"] = sample["residual"]
            
            # Save HR: [seq_len, channels, H, W]
            if "hr" in sample:
                sample_dict["hr.pt"] = sample["hr"]
            
            # Save static: [channels, H, W] (optional)
            if "static" in sample:
                sample_dict["static.pt"] = sample["static"]
            
            # Save time metadata as JSON
            if "time" in sample:
                # Convert time to serializable format
                time_data = sample["time"]
                if isinstance(time_data, np.ndarray):
                    # Convert numpy array to list
                    if time_data.dtype.kind == 'M':  # datetime64
                        time_list = [str(t) for t in time_data]
                    else:
                        time_list = time_data.tolist()
                else:
                    time_list = list(time_data)
                sample_dict["time.json"] = json.dumps(time_list).encode('utf-8')
            
            # Write sample to shard
            writer.write(sample_dict)
            sample_idx += 1
    
    # Save metadata
    metadata = {
        "num_samples": sample_idx,
        "seq_len": seq_len,
        "stride": stride,
        "shard_size": shard_size,
        "dims": {
            "time": pipeline.dims.time,
            "lr_lat": pipeline.dims.lr_lat,
            "lr_lon": pipeline.dims.lr_lon,
            "hr_lat": pipeline.dims.hr_lat,
            "hr_lon": pipeline.dims.hr_lon,
        },
        "lr_shape": list(pipeline.lr_dataset.dims.values()),
        "hr_shape": list(pipeline.hr_dataset.dims.values()),
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Created {sample_idx} samples in shards at {output_dir}")
    print(f"✓ Metadata saved to {metadata_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert NetCDF data to WebDataset TAR shards"
    )
    parser.add_argument("--lr_path", type=str, required=True, help="Path to LR NetCDF file")
    parser.add_argument("--hr_path", type=str, required=True, help="Path to HR NetCDF file")
    parser.add_argument("--static_path", type=str, default=None, help="Path to static NetCDF file (optional)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for shards")
    parser.add_argument("--seq_len", type=int, default=10, help="Sequence length (default: 10)")
    parser.add_argument("--stride", type=int, default=1, help="Stride for windows (default: 1)")
    parser.add_argument("--shard_size", type=int, default=1000, help="Samples per shard (default: 1000)")
    parser.add_argument("--baseline_strategy", type=str, default="hr_smoothing", help="Baseline strategy")
    parser.add_argument("--baseline_factor", type=float, default=2.0, help="Baseline factor")
    parser.add_argument("--normalize", action="store_true", help="Normalize LR data")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    print(f"Initializing data pipeline...")
    pipeline = NetCDFDataPipeline(
        lr_path=args.lr_path,
        hr_path=args.hr_path,
        static_path=args.static_path,
        seq_len=args.seq_len,
        baseline_strategy=args.baseline_strategy,
        baseline_factor=args.baseline_factor,
        normalize=args.normalize,
    )
    
    # Create shards
    print(f"Creating WebDataset shards...")
    create_shards(
        pipeline=pipeline,
        output_dir=Path(args.output_dir),
        seq_len=args.seq_len,
        shard_size=args.shard_size,
        stride=args.stride,
    )
    
    print("✓ Conversion complete!")


if __name__ == "__main__":
    main()

