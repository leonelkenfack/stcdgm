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
    SpatialConditioningProjector,
    CausalConditioningProjector,
    HRTargetIdentifiabilityHead,
    train_epoch,
    compute_rapsd_metric_from_batch,
    resolve_train_amp_mode,
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
    out = {
        "lr": lr_tensor,
        "residual": sample["residual"],
        "baseline": sample.get("baseline"),
        "hetero": hetero,
    }
    if "valid_mask" in sample:
        out["valid_mask"] = sample["valid_mask"]
    return out


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

    n_threads = max(1, (os.cpu_count() or 1) // 2)
    torch.set_num_threads(n_threads)
    if rank == 0:
        print(f"[PERF] torch.set_num_threads({n_threads})")

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
        pipeline = ZarrDataPipeline(zarr_dir=str(zarr_dir))
        
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
        include_mid_layer=CONFIG.graph.get("include_mid_layer", True),
    )

    # Encoder configs (only metapaths whose nodes exist in the graph).
    # Sprint 1 / B2 fix: a silently filtered metapath degrades the DAG size
    # without warning. The 2x2 DAG observed in the current checkpoint comes
    # from here: GP500 / GP250 were missing from the builder, four of the
    # five configured metapaths were dropped, and q collapsed to 2. We now
    # fail loudly unless the user explicitly opts out via
    # ``encoder.allow_missing_metapaths: true``.
    allowed_nodes = set(builder.dynamic_node_types) | set(builder.static_node_types)
    configured_metapaths = list(CONFIG.encoder.metapaths)
    kept_metapaths = [
        mp for mp in configured_metapaths
        if mp.src in allowed_nodes and mp.target in allowed_nodes
    ]
    missing_metapaths = [
        mp.name for mp in configured_metapaths
        if mp.src not in allowed_nodes or mp.target not in allowed_nodes
    ]
    if missing_metapaths:
        allow_missing = bool(CONFIG.encoder.get("allow_missing_metapaths", False))
        msg = (
            "Encoder metapaths were silently filtered because their source or "
            "target node types are absent from the heterogeneous graph.\n"
            f"  - missing    : {missing_metapaths}\n"
            f"  - kept       : {[mp.name for mp in kept_metapaths]}\n"
            f"  - known nodes: {sorted(allowed_nodes)}\n"
            "This usually means your NetCDF/Zarr data does not contain the "
            "500 hPa / 250 hPa variables (u_500, v_500, w_500, q_500, t_500, "
            "u_250, ...) or that `graph.include_mid_layer` is false. A reduced "
            "metapath set collapses the learned DAG (q becomes smaller than "
            "the 5 variables assumed in the ORACLE paper) and silently "
            "degrades downscaling quality. Set "
            "`encoder.allow_missing_metapaths: true` in the config to opt in."
        )
        if not allow_missing:
            raise RuntimeError(msg)
        if get_rank() == 0:
            print(f"[WARN] {msg}")
    encoder_configs = [
        IntelligibleVariableConfig(name=mp.name, meta_path=(mp.src, mp.relation, mp.target), pool=mp.get("pool", "mean"))
        for mp in kept_metapaths
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
    # Sprint 1: prior-based DAG init. If the user supplied a physically
    # meaningful prior matrix under ``loss.dag_prior`` and set
    # ``rcn.init_dag_from_prior: true``, the RCNCell starts A_dag at that
    # prior (plus small noise) instead of Xavier random. We only use the
    # prior when its shape matches the actual q inferred from the graph,
    # to stay robust to configuration mismatches.
    rcn_dag_prior_tensor = None
    if CONFIG.rcn.get("init_dag_from_prior", False) and CONFIG.loss.get("dag_prior") is not None:
        prior_list = CONFIG.loss.dag_prior
        prior_tensor = torch.tensor(prior_list, dtype=torch.float32)
        if prior_tensor.shape == (num_vars, num_vars):
            rcn_dag_prior_tensor = prior_tensor
        elif get_rank() == 0:
            print(
                f"[WARN] loss.dag_prior shape {tuple(prior_tensor.shape)} "
                f"does not match num_vars={num_vars}; falling back to Xavier init."
            )

    rcn_cell = RCNCell(
        num_vars=num_vars,
        hidden_dim=CONFIG.rcn.hidden_dim,
        driver_dim=rcn_driver_dim,
        reconstruction_dim=rcn_driver_dim,
        dropout=CONFIG.rcn.get("dropout", 0.0),
        dag_prior=rcn_dag_prior_tensor,
        dag_prior_noise=CONFIG.rcn.get("init_dag_prior_noise", 0.02),
    ).to(device)

    unet_kwargs = dict(
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
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        height=CONFIG.diffusion.height,
        width=CONFIG.diffusion.width,
        num_diffusion_steps=CONFIG.diffusion.steps,
        unet_kwargs=unet_kwargs,
        scheduler_type=CONFIG.diffusion.get("scheduler_type", "ddpm"),
        use_gradient_checkpointing=CONFIG.diffusion.get("use_gradient_checkpointing", False),
        conv_padding_mode=CONFIG.diffusion.get("conv_padding_mode", "zeros"),
        anti_checkerboard=CONFIG.diffusion.get("anti_checkerboard", False),
    ).to(device)

    # Apply torch.compile if enabled (before DDP wrapping)
    compile_cfg = CONFIG.training.get("compile", {}) or {}
    if compile_cfg.get("enabled", False):
        if rank == 0:
            print("🔧 Compiling models with torch.compile...")
        encoder = torch.compile(encoder, mode=compile_cfg.get("encoder_mode", "default"))
        rcn_cell = torch.compile(rcn_cell, mode=compile_cfg.get("rcn_mode", "default"))
        if torch.cuda.is_available():
            diffusion = torch.compile(diffusion, mode=compile_cfg.get("diffusion_mode", "default"))
        elif rank == 0:
            print("⚠ Skipping torch.compile for diffusion decoder (CUDA not available)")
        if rank == 0:
            print("✅ Models compiled successfully")
    
    # Wrap with DDP
    find_unused = CONFIG.training.get("multi_gpu", {}).get("find_unused_parameters", True)
    if world_size > 1:
        encoder = wrap_model_ddp(encoder, device_ids=[local_rank], find_unused_parameters=find_unused)
        rcn_cell = wrap_model_ddp(rcn_cell, device_ids=[local_rank], find_unused_parameters=find_unused)
        diffusion = wrap_model_ddp(diffusion, device_ids=[local_rank], find_unused_parameters=find_unused)

    rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=CONFIG.rcn.get("detach_interval"))

    # Sprint 2: use the CausalConditioningProjector (DAG-token aware) when
    # ``encoder.causal_conditioning: true`` in the YAML. This is the key
    # architectural change that makes A_dag *visible* to the UNet via
    # cross-attention — the decoder learns ``f(x | H_T, A_dag)`` instead
    # of just ``f(x | H_T)``, and interventional edits of A_dag at
    # inference now directly change the tokens fed to the UNet.
    use_causal_proj = bool(CONFIG.encoder.get("causal_conditioning", False))
    spatial_target_shape = tuple(CONFIG.diffusion.get("spatial_target_shape", [6, 7]))
    if use_causal_proj:
        if get_rank() == 0:
            print("🧠 Sprint 2: using CausalConditioningProjector (A_dag tokens)")
        spatial_projector = CausalConditioningProjector(
            num_vars=len(encoder_configs),
            hidden_dim=CONFIG.rcn.hidden_dim,
            conditioning_dim=CONFIG.diffusion.conditioning_dim,
            lr_shape=tuple(CONFIG.graph.lr_shape),
            target_shape=spatial_target_shape,
            num_dag_tokens=int(CONFIG.encoder.get("num_dag_tokens", 1)),
        ).to(device)
    else:
        spatial_projector = SpatialConditioningProjector(
            num_vars=len(encoder_configs),
            hidden_dim=CONFIG.rcn.hidden_dim,
            conditioning_dim=CONFIG.diffusion.conditioning_dim,
            lr_shape=tuple(CONFIG.graph.lr_shape),
            target_shape=spatial_target_shape,
        ).to(device)

    # Sprint 2: optional HR-target identifiability head.
    hr_ident_cfg = CONFIG.loss.get("hr_ident", {}) or {}
    hr_ident_enabled = bool(hr_ident_cfg.get("enabled", False))
    beta_hr_ident = float(hr_ident_cfg.get("beta", 0.0))
    if hr_ident_enabled and beta_hr_ident > 0.0:
        hr_ident_head = HRTargetIdentifiabilityHead(
            num_vars=len(encoder_configs),
            hidden_dim=CONFIG.rcn.hidden_dim,
            stats=list(hr_ident_cfg.get("stats", ["mean", "std", "p95", "p99"])),
        ).to(device)
        if get_rank() == 0:
            print(
                f"🎯 Sprint 2: HR identifiability head enabled "
                f"(beta={beta_hr_ident}, stats={hr_ident_head.stats})"
            )
    else:
        hr_ident_head = None

    _opt_params = (
        list(encoder.parameters())
        + list(rcn_cell.parameters())
        + list(diffusion.parameters())
        + list(spatial_projector.parameters())
    )
    if hr_ident_head is not None:
        _opt_params += list(hr_ident_head.parameters())
    optimizer = torch.optim.Adam(_opt_params, lr=CONFIG.training.lr)

    batch_size = CONFIG.training.get("batch_size", 1)
    num_workers = CONFIG.training.get("num_workers", 0)  # 0 on CyVerse (limited /dev/shm)
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        collate_fn=lambda x: x,
    )

    history = {"loss": [], "loss_gen": [], "loss_rec": [], "loss_dag": []}

    # Sprint 2: dag_grad_gate warm-up schedule.
    # Gate stays at 0 during the first ``dag_gate_cold_epochs`` so the UNet
    # can stabilise against the detached A_dag (Sprint 1 behaviour), then
    # ramps linearly to ``dag_gate_max`` over ``dag_gate_ramp_epochs``,
    # then stays constant. This gives end-to-end causal training with a
    # safe cold-start and no risk of wrecking A_dag in the first epochs.
    dag_gate_cfg = CONFIG.rcn.get("dag_grad_gate", {}) or {}
    dag_gate_enabled = bool(dag_gate_cfg.get("enabled", False))
    dag_gate_cold_epochs = int(dag_gate_cfg.get("cold_epochs", 5))
    dag_gate_ramp_epochs = max(1, int(dag_gate_cfg.get("ramp_epochs", 20)))
    dag_gate_max = float(dag_gate_cfg.get("max", 1.0))

    def _dag_gate_value(epoch_idx: int) -> float:
        if not dag_gate_enabled:
            return 0.0
        if epoch_idx < dag_gate_cold_epochs:
            return 0.0
        progress = (epoch_idx - dag_gate_cold_epochs) / dag_gate_ramp_epochs
        return float(max(0.0, min(dag_gate_max, progress * dag_gate_max)))

    for epoch in range(CONFIG.training.epochs):
        # Sprint 2: set the gate *before* train_epoch so every batch this
        # epoch sees the same gate value. We poke through the DDP wrapper
        # if present, and through the torch.compile wrapper (``_orig_mod``).
        _gate_value = _dag_gate_value(epoch)
        _rcn_target = rcn_cell.module if hasattr(rcn_cell, "module") else rcn_cell
        _rcn_target = getattr(_rcn_target, "_orig_mod", _rcn_target)
        if hasattr(_rcn_target, "set_dag_grad_gate"):
            _rcn_target.set_dag_grad_gate(_gate_value)
        if get_rank() == 0 and dag_gate_enabled:
            print(f"[Epoch {epoch + 1}] dag_grad_gate = {_gate_value:.3f}")

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
            use_focal_loss=CONFIG.loss.get("use_focal_loss", False),
            focal_alpha=CONFIG.loss.get("focal_alpha", 1.0),
            focal_gamma=CONFIG.loss.get("focal_gamma", 2.0),
            extreme_weight_factor=CONFIG.loss.get("extreme_weight_factor", 0.0),
            extreme_percentiles=list(CONFIG.loss.get("extreme_percentiles", [95.0, 99.0])),
            reconstruction_loss_type=CONFIG.loss.get("reconstruction_loss_type", "mse"),
            use_spectral_loss=CONFIG.loss.get("use_spectral_loss", False),
            lambda_spectral=CONFIG.loss.get("lambda_spectral", 0.0),
            conditioning_dropout_prob=CONFIG.diffusion.get("conditioning_dropout_prob", 0.0),
            lambda_dag_prior=CONFIG.loss.get("lambda_dag_prior", 0.0),
            dag_prior=torch.tensor(CONFIG.loss.dag_prior, dtype=torch.float32) if CONFIG.loss.get("dag_prior") else None,
            spatial_projector=spatial_projector,
            hr_ident_head=hr_ident_head,
            beta_hr_ident=beta_hr_ident,
            lambda_precip_phy=float(CONFIG.loss.get("lambda_precip_phy", 0.0)),
            precip_phy_weights=tuple(
                CONFIG.loss.get("precip_phy_weights", [1.0, 0.1, 0.2])
            ),
            physical_sample_interval=int(
                CONFIG.training.get("physical_loss", {}).get("physical_sample_interval", 10)
            ),
        )
        if get_rank() == 0 and CONFIG.loss.get("log_spectral_metric_each_epoch", False):
            amp_m = resolve_train_amp_mode(device, CONFIG.training.get("use_amp", True))
            try:
                metric_iter = iterate_batches(train_loader, builder, device)
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
            except Exception as ex:
                print(f"[WARN] RAPSD epoch metric: {ex}")
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
            sp = spatial_projector.module if hasattr(spatial_projector, "module") else spatial_projector
            _ckpt_payload = {
                "epoch": epoch + 1,
                "encoder_state_dict": enc.state_dict(),
                "rcn_cell_state_dict": rcn.state_dict(),
                "diffusion_state_dict": diff.state_dict(),
                # Sprint 1 / B4 fix: persist the spatial projector so that
                # evaluation can rebuild it and feed cross-attention tokens
                # to the UNet instead of q global scalars.
                "spatial_projector_state_dict": sp.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": OmegaConf.to_container(CONFIG, resolve=True),
            }
            # Sprint 2: persist the HR identifiability head when enabled.
            if hr_ident_head is not None:
                _hrh = hr_ident_head.module if hasattr(hr_ident_head, "module") else hr_ident_head
                _ckpt_payload["hr_ident_head_state_dict"] = _hrh.state_dict()
            torch.save(_ckpt_payload, ckpt)
            print(f"Checkpoint saved: {ckpt}")

    if world_size > 1:
        cleanup_ddp()


if __name__ == "__main__":
    main()
