"""
ST-CDGM training script with DistributedDataParallel (DDP).

Launch with torchrun to use all GPUs (e.g. 4 GPUs):
    torchrun --nproc_per_node=4 train_ddp.py

Requires: config/training_config.yaml, data pipeline (Zarr or NetCDF) set up.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Project root and path setup (must be before other imports)
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset

from omegaconf import OmegaConf

from st_cdgm import (
    HeteroGraphBuilder,
    IntelligibleVariableEncoder,
    RCNCell,
    RCNSequenceRunner,
    CausalDiffusionDecoder,
    train_epoch,
)
from st_cdgm.models.intelligible_encoder import IntelligibleVariableConfig
from st_cdgm.training.multi_gpu import setup_ddp, cleanup_ddp, wrap_model_ddp, get_rank


class ShardedIterableDataset(IterableDataset):
    """Wraps an IterableDataset so each DDP rank only sees 1/world_size of the data."""

    def __init__(self, dataset: IterableDataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        for i, sample in enumerate(self.dataset):
            if i % self.world_size == self.rank:
                yield sample


def _convert_sample_to_batch(sample, builder, device):
    """Convert a pipeline sample to the batch dict expected by train_epoch."""
    import torch
    lr_seq = sample["lr"]
    seq_len = lr_seq.shape[0]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    dynamic_features = {nt: lr_nodes_steps[0] for nt in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_features).to(device)
    return {
        "lr": lr_tensor,
        "residual": sample["residual"],
        "baseline": sample.get("baseline"),
        "hetero": hetero,
    }


def iterate_batches(dataloader, builder, device):
    """Yield lists of batch dicts (one list per DataLoader batch)."""
    for batch_list in dataloader:
        if not isinstance(batch_list, list):
            batch_list = [batch_list]
        converted = [_convert_sample_to_batch(s, builder, device) for s in batch_list]
        yield converted


def main():
    # Distributed setup (torchrun sets these)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        setup_ddp(rank=rank, world_size=world_size, backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Load config
    cfg = OmegaConf.load(PROJECT_ROOT / "config" / "training_config.yaml")
    CONFIG = cfg

    # Data paths (relative to project root)
    lr_path = Path(CONFIG.data.lr_path)
    hr_path = Path(CONFIG.data.hr_path)
    static_path = Path(CONFIG.data.static_path) if CONFIG.data.get("static_path") else None
    seq_len = CONFIG.data.seq_len
    stride = CONFIG.data.get("stride", 1)
    
    # Validate that data files exist (for netcdf mode)
    if rank == 0:
        if not lr_path.exists():
            raise FileNotFoundError(f"LR data file not found: {lr_path}")
        if not hr_path.exists():
            raise FileNotFoundError(f"HR data file not found: {hr_path}")
        if static_path and not static_path.exists():
            raise FileNotFoundError(f"Static data file not found: {static_path}")
    
    # Lire le format de données depuis la configuration
    dataset_format = CONFIG.data.get("dataset_format", "zarr").lower()
    zarr_dir = Path(CONFIG.data.get("zarr_dir", "data/raw/train/zarr"))
    shard_dir = Path(CONFIG.data.get("shard_dir", "data/raw/train/shards"))
    
    if rank == 0:
        print(f"📋 Format de données configuré: {dataset_format}")
    
    # Fonctions de vérification
    def check_zarr_exists():
        return (zarr_dir / "lr.zarr").exists() and (zarr_dir / "hr.zarr").exists()
    
    def check_shards_exist():
        return shard_dir.exists() and (shard_dir / "metadata.json").exists() and len(list(shard_dir.glob("*.tar"))) > 0
    
    # Fonctions de préprocessing (rank 0 seulement)
    def preprocess_to_zarr():
        if rank != 0:
            return
        import subprocess
        print("🔄 Conversion NetCDF → Zarr...")
        cmd = [
            sys.executable, "-m", "ops.preprocess_to_zarr",
            "--lr_path", str(lr_path),
            "--hr_path", str(hr_path),
            "--output_dir", str(zarr_dir),
            "--seq_len", str(seq_len),
            "--baseline_strategy", CONFIG.data.baseline_strategy,
            "--baseline_factor", str(CONFIG.data.get("baseline_factor", 4)),
            "--chunk_size_time", "100",
            "--chunk_size_lat", "64",
            "--chunk_size_lon", "64",
        ]
        if CONFIG.data.get("normalize", False):
            cmd.append("--normalize")
        if static_path:
            cmd.extend(["--static_path", str(static_path)])
        if CONFIG.data.get("lr_variables"):
            cmd.extend(["--lr_variables"] + list(CONFIG.data.lr_variables))
        if CONFIG.data.get("hr_variables"):
            cmd.extend(["--hr_variables"] + list(CONFIG.data.hr_variables))
        if CONFIG.data.get("static_variables"):
            cmd.extend(["--static_variables"] + list(CONFIG.data.static_variables))
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Conversion Zarr terminée!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de la conversion Zarr: {e}")
            print(f"   Commande: {' '.join(cmd)}")
            if world_size > 1:
                cleanup_ddp()
            sys.exit(1)
    
    def preprocess_to_shards():
        if rank != 0:
            return
        import subprocess
        print("🔄 Conversion NetCDF → WebDataset Shards...")
        cmd = [
            sys.executable, "-m", "ops.preprocess_to_shards",
            "--lr_path", str(lr_path),
            "--hr_path", str(hr_path),
            "--output_dir", str(shard_dir),
            "--seq_len", str(seq_len),
            "--baseline_strategy", CONFIG.data.baseline_strategy,
            "--baseline_factor", str(CONFIG.data.get("baseline_factor", 4)),
            "--samples_per_shard", "100",
        ]
        if CONFIG.data.get("normalize", False):
            cmd.append("--normalize")
        if static_path:
            cmd.extend(["--static_path", str(static_path)])
        if CONFIG.data.get("lr_variables"):
            cmd.extend(["--lr_variables"] + list(CONFIG.data.lr_variables))
        if CONFIG.data.get("hr_variables"):
            cmd.extend(["--hr_variables"] + list(CONFIG.data.hr_variables))
        if CONFIG.data.get("static_variables"):
            cmd.extend(["--static_variables"] + list(CONFIG.data.static_variables))
        
        try:
            subprocess.run(cmd, check=True)
            print(f"✅ Conversion Shards terminée!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Erreur lors de la conversion Shards: {e}")
            print(f"   Commande: {' '.join(cmd)}")
            if world_size > 1:
                cleanup_ddp()
            sys.exit(1)
    
    # Logique de sélection et de génération automatique
    if dataset_format == "netcdf":
        if rank == 0:
            print("✅ Mode NetCDF: Lecture directe des fichiers .nc")
        from st_cdgm.data.pipeline import NetCDFDataPipeline
        pipeline = NetCDFDataPipeline(
            lr_path=str(lr_path),
            hr_path=str(hr_path),
            static_path=str(static_path) if static_path else None,
            seq_len=seq_len,
            baseline_strategy=CONFIG.data.baseline_strategy,
            baseline_factor=CONFIG.data.get("baseline_factor", 4),
            normalize=CONFIG.data.get("normalize", False),
            target_transform=CONFIG.data.get("target_transform"),
            nan_fill_strategy=CONFIG.data.get("nan_fill_strategy", "mean"),
            precipitation_delta=CONFIG.data.get("precipitation_delta", 0.01),
            lr_variables=list(CONFIG.data.get("lr_variables", [])),
            hr_variables=list(CONFIG.data.get("hr_variables", [])),
            static_variables=list(CONFIG.data.get("static_variables", [])) if static_path else None,
        )
        
    elif dataset_format == "zarr":
        if not check_zarr_exists():
            if rank == 0:
                print(f"⚠️  Données Zarr non trouvées dans: {zarr_dir}")
                print("   Génération automatique...")
            preprocess_to_zarr()
            if world_size > 1:
                dist.barrier()  # Attendre que rank 0 finisse
        else:
            if rank == 0:
                print(f"✅ Données Zarr déjà présentes dans: {zarr_dir}")
        
        from st_cdgm.data.pipeline import ZarrDataPipeline
        pipeline = ZarrDataPipeline(zarr_dir=str(zarr_dir), seq_len=seq_len, stride=stride)
        
    elif dataset_format == "shard":
        if not check_shards_exist():
            if rank == 0:
                print(f"⚠️  Shards non trouvés dans: {shard_dir}")
                print("   Génération automatique...")
            preprocess_to_shards()
            if world_size > 1:
                dist.barrier()  # Attendre que rank 0 finisse
        else:
            if rank == 0:
                print(f"✅ Shards WebDataset déjà présents dans: {shard_dir}")
        
        from st_cdgm.data.pipeline import WebDatasetDataPipeline
        pipeline = WebDatasetDataPipeline(
            shard_dir=str(shard_dir),
            shuffle=CONFIG.data.get("shuffle", False),
            shardshuffle=100,
            shuffle_buffer_size=1000,
        )
        
    else:
        raise ValueError(f"Format de données non reconnu: {dataset_format}. Utilisez 'netcdf', 'zarr', ou 'shard'.")

    # Build dataset (before sharding for DDP, get one sample for shapes)
    base_dataset = pipeline.build_sequence_dataset(seq_len=seq_len, stride=stride, as_torch=True)
    
    # Get one sample for shapes before DDP sharding
    sample_for_shapes = next(iter(base_dataset))
    lr_shape = tuple(CONFIG.graph.lr_shape)
    hr_shape = tuple(CONFIG.graph.hr_shape)
    rcn_driver_dim = sample_for_shapes["lr"].shape[1]
    hr_channels = sample_for_shapes["residual"].shape[1]
    
    # Now create a fresh dataset for training and shard for DDP
    dataset = pipeline.build_sequence_dataset(seq_len=seq_len, stride=stride, as_torch=True)
    if world_size > 1:
        dataset = ShardedIterableDataset(dataset, rank, world_size)

    # Graph builder
    builder = HeteroGraphBuilder(
        lr_shape=lr_shape,
        hr_shape=hr_shape,
        static_dataset=pipeline.get_static_dataset(),
        include_mid_layer=CONFIG.graph.get("include_mid_layer", False),
    )

    # Encoder configs (only metapaths whose nodes exist in the graph)
    allowed_nodes = set(builder.dynamic_node_types) | set(builder.static_node_types)
    encoder_configs = [
        IntelligibleVariableConfig(name=mp.name, meta_path=(mp.src, mp.relation, mp.target), pool=mp.get("pool", "mean"))
        for mp in CONFIG.encoder.metapaths
        if mp.src in allowed_nodes and mp.target in allowed_nodes
    ]
    if pipeline.get_static_dataset() is not None:
        encoder_configs.append(
            IntelligibleVariableConfig(name="static", meta_path=("SP_HR", "causes", "GP850"), pool="mean")
        )

    # Models (on device)
    encoder = IntelligibleVariableEncoder(
        configs=encoder_configs,
        hidden_dim=CONFIG.encoder.hidden_dim,
        conditioning_dim=CONFIG.encoder.conditioning_dim,
    ).to(device)

    num_vars = len(encoder_configs)
    rcn_cell = RCNCell(
        num_vars=num_vars,
        hidden_dim=CONFIG.rcn.hidden_dim,
        driver_dim=rcn_driver_dim,
        reconstruction_dim=rcn_driver_dim,
        dropout=CONFIG.rcn.get("dropout", 0.0),
    ).to(device)

    unet_kwargs = dict(
        layers_per_block=1,
        block_out_channels=(32,),
        down_block_types=("DownBlock2D",),
        up_block_types=("UpBlock2D",),
        norm_num_groups=8,
    )
    diffusion = CausalDiffusionDecoder(
        in_channels=hr_channels,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        height=CONFIG.diffusion.height,
        width=CONFIG.diffusion.width,
        num_diffusion_steps=CONFIG.diffusion.steps,
        unet_kwargs=unet_kwargs,
    ).to(device)

    # Apply torch.compile if enabled (before DDP wrapping)
    if CONFIG.training.get("compile", {}).get("enabled", False):
        if rank == 0:
            print("🔧 Compiling models with torch.compile...")
        compile_cfg = CONFIG.training.compile
        encoder = torch.compile(encoder, mode=compile_cfg.get("encoder_mode", "default"))
        rcn_cell = torch.compile(rcn_cell, mode=compile_cfg.get("rcn_mode", "default"))
        diffusion = torch.compile(diffusion, mode=compile_cfg.get("diffusion_mode", "default"))
        if rank == 0:
            print("✅ Models compiled successfully")
    
    # Wrap with DDP
    find_unused = CONFIG.training.get("multi_gpu", {}).get("find_unused_parameters", True)
    if world_size > 1:
        encoder = wrap_model_ddp(encoder, device_ids=[local_rank], find_unused_parameters=find_unused)
        rcn_cell = wrap_model_ddp(rcn_cell, device_ids=[local_rank], find_unused_parameters=find_unused)
        diffusion = wrap_model_ddp(diffusion, device_ids=[local_rank], find_unused_parameters=find_unused)

    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get("detach_interval"))

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(rcn_cell.parameters()) + list(diffusion.parameters()),
        lr=CONFIG.training.lr,
    )

    batch_size = CONFIG.training.get("batch_size", 1)
    num_workers = 4  # Per process (4 workers × 4 GPUs = 16 total)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=num_workers > 0,
        collate_fn=lambda x: x,
    )

    history = {"loss": [], "loss_gen": [], "loss_rec": [], "loss_dag": []}

    for epoch in range(CONFIG.training.epochs):
        # Reuse the same DataLoader (no need to recreate with persistent_workers)
        data_loader = iterate_batches(train_loader, builder, device)
        metrics = train_epoch(
            encoder=encoder,
            rcn_runner=rcn_runner,
            diffusion_decoder=diffusion,
            optimizer=optimizer,
            data_loader=data_loader,
            lambda_gen=CONFIG.loss.lambda_gen,
            beta_rec=CONFIG.loss.beta_rec,
            gamma_dag=CONFIG.loss.gamma_dag,
            device=device,
            gradient_clipping=CONFIG.training.get("gradient_clipping"),
            log_interval=CONFIG.training.get("log_every", 10),
            verbose=(get_rank() == 0),
            use_amp=CONFIG.training.get("use_amp", True),
            dag_method=CONFIG.loss.get("dag_method", "dagma"),
            dagma_s=CONFIG.loss.get("dagma_s", 1.0),
        )
        for k in history:
            if k in metrics:
                history[k].append(metrics[k])

        if get_rank() == 0:
            save_dir = Path(CONFIG.checkpoint.get("save_dir", "models"))
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt = save_dir / "st_cdgm_checkpoint.pth"
            enc = encoder.module if hasattr(encoder, "module") else encoder
            rcn = rcn_cell.module if hasattr(rcn_cell, "module") else rcn_cell
            diff = diffusion.module if hasattr(diffusion, "module") else diffusion
            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": enc.state_dict(),
                "rcn_cell_state_dict": rcn.state_dict(),
                "diffusion_state_dict": diff.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, ckpt)
            print(f"Checkpoint saved: {ckpt}")

    if world_size > 1:
        cleanup_ddp()


if __name__ == "__main__":
    main()
