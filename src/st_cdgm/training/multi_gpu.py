"""
Multi-GPU training utilities for ST-CDGM.

Provides functions for setting up and managing distributed training with
DistributedDataParallel (DDP) across multiple GPUs.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialize the distributed process group for DDP.
    
    Parameters
    ----------
    rank : int
        Rank of the current process (GPU ID).
    world_size : int
        Total number of processes (GPUs).
    backend : str, optional
        Backend for distributed communication ('nccl' for GPUs, 'gloo' for CPU).
    """
    os.environ["MASTER_ADDR"] = os.environ.get("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.environ.get("MASTER_PORT", "12355")
    
    if not dist.is_initialized():
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        print(f"[Rank {rank}] Initialized DDP process group (world_size={world_size})")


def cleanup_ddp() -> None:
    """
    Clean up the distributed process group.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
        print("Cleaned up DDP process group")


def wrap_model_ddp(
    model: torch.nn.Module,
    device_ids: list[int],
    find_unused_parameters: bool = False,
) -> DDP:
    """
    Wrap a model with DistributedDataParallel.
    
    Parameters
    ----------
    model : torch.nn.Module
        Model to wrap.
    device_ids : list[int]
        List of GPU IDs to use (typically [rank] for single GPU per process).
    find_unused_parameters : bool, optional
        Whether to find unused parameters (required for PyG HeteroData graphs).
        
    Returns
    -------
    DDP
        Wrapped model ready for distributed training.
    """
    # Move model to the appropriate device
    if len(device_ids) > 0:
        model = model.to(device_ids[0])
    
    # Wrap with DDP
    ddp_model = DDP(
        model,
        device_ids=device_ids,
        output_device=device_ids[0] if len(device_ids) > 0 else None,
        find_unused_parameters=find_unused_parameters,
    )
    
    return ddp_model


def get_rank() -> int:
    """
    Get the rank of the current process.
    
    Returns
    -------
    int
        Rank of the current process (0 if not in DDP mode).
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """
    Get the total number of processes (GPUs).
    
    Returns
    -------
    int
        Total number of processes (1 if not in DDP mode).
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """
    Check if the current process is the main process (rank 0).
    
    Returns
    -------
    bool
        True if rank 0 or not in DDP mode.
    """
    return get_rank() == 0


def setup_multi_gpu_notebook(config) -> tuple[bool, int, int]:
    """
    Setup multi-GPU configuration for notebook environment.
    
    This is a simplified version for Jupyter notebooks that doesn't use
    torch.distributed.launch. Instead, it uses DataParallel for multi-GPU.
    
    For full DDP support, training should be launched via torch.distributed.launch
    or torchrun with a separate Python script.
    
    Parameters
    ----------
    config : omegaconf.DictConfig
        Configuration object with multi_gpu settings.
        
    Returns
    -------
    tuple[bool, int, int]
        (use_multi_gpu, rank, world_size)
    """
    if not config.training.get("multi_gpu", {}).get("enabled", False):
        return False, 0, 1
    
    # In notebook environment, we can't use full DDP without proper launch
    # We'll use DataParallel instead (simpler but less efficient)
    warnings.warn(
        "Multi-GPU in notebook uses DataParallel, not DistributedDataParallel. "
        "For full DDP performance, run training from a script with torchrun."
    )
    
    gpus = config.training.multi_gpu.get("gpus", [0])
    world_size = len(gpus)
    
    return True, 0, world_size  # Notebook always acts as rank 0


def wrap_models_for_notebook(models: dict, config) -> dict:
    """
    Wrap models for multi-GPU training in notebook environment.
    
    Uses DataParallel for simplicity in notebooks. For full DDP, use a script.
    
    Parameters
    ----------
    models : dict
        Dictionary of models to wrap (e.g., {'encoder': encoder, 'rcn': rcn, ...})
    config : omegaconf.DictConfig
        Configuration with multi_gpu settings.
        
    Returns
    -------
    dict
        Dictionary of wrapped models.
    """
    if not config.training.get("multi_gpu", {}).get("enabled", False):
        return models
    
    gpus = config.training.multi_gpu.get("gpus", [0])
    
    wrapped = {}
    for name, model in models.items():
        if model is not None:
            # Use DataParallel for notebook multi-GPU
            wrapped[name] = torch.nn.DataParallel(model, device_ids=gpus)
            print(f"[MultiGPU] Wrapped {name} with DataParallel on GPUs {gpus}")
        else:
            wrapped[name] = None
    
    return wrapped
