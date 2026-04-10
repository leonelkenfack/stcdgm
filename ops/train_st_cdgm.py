"""
Hydra-driven training script for the ST-CDGM pipeline.

This script stitches together the individual modules (data pipeline, graph builder,
intelligible encoder, causal RCN, diffusion decoder) into a full training loop.
It provides a composable configuration that can be overridden via Hydra's CLI:

Example:
    python ops/train_st_cdgm.py \
        data.lr_path=data/raw/lr.nc \
        data.hr_path=data/raw/hr.nc \
        diffusion.height=256 diffusion.width=256
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

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
    compute_rapsd_metric_from_batch,
    resolve_train_amp_mode,
)


@dataclass
class MetaPathConfig:
    name: str
    src: str
    relation: str
    target: str
    pool: str = "mean"


@dataclass
class DataConfig:
    lr_path: str = "data/raw/lr.nc"
    hr_path: str = "data/raw/hr.nc"
    static_path: Optional[str] = None
    seq_len: int = 6
    stride: int = 1
    baseline_strategy: str = "hr_smoothing"
    baseline_factor: int = 4
    normalize: bool = True
    nan_fill_strategy: str = "zero"  # "zero", "mean", or "interpolate"
    precipitation_delta: float = 0.01  # Delta pour log1p des précipitations


@dataclass
class GraphConfig:
    lr_shape: Tuple[int, int] = (23, 26)
    hr_shape: Tuple[int, int] = (172, 179)
    include_mid_layer: bool = True
    static_variables: Optional[List[str]] = None


@dataclass
class EncoderConfig:
    hidden_dim: int = 128
    conditioning_dim: int = 128
    metapaths: List[MetaPathConfig] = field(
        default_factory=lambda: [
            MetaPathConfig(name="surface", src="GP850", relation="spat_adj", target="GP850", pool="mean"),
            MetaPathConfig(name="vertical", src="GP500", relation="vert_adj", target="GP850", pool="mean"),
            MetaPathConfig(name="static", src="SP_HR", relation="causes", target="GP850", pool="mean"),
        ]
    )


@dataclass
class RCNConfig:
    hidden_dim: int = 128
    driver_dim: int = 15
    reconstruction_dim: Optional[int] = 15
    dropout: float = 0.0
    detach_interval: Optional[int] = None


@dataclass
class DiffusionConfig:
    in_channels: int = 1
    conditioning_dim: int = 128
    height: int = 172
    width: int = 179
    steps: int = 1000


@dataclass
class LossConfig:
    lambda_gen: float = 1.0
    beta_rec: float = 0.1
    gamma_dag: float = 0.1


@dataclass
class TrainingConfig:
    epochs: int = 1
    lr: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gradient_clipping: Optional[float] = 1.0
    log_every: int = 1
    num_workers: int = 0  # 0 avoids /dev/shm (CyVerse/Docker have ~64MB). Use 4-8 if shm is large


@dataclass
class STCDGMConfig:
    data: DataConfig = field(default_factory=DataConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    rcn: RCNConfig = field(default_factory=RCNConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


cs = ConfigStore.instance()
cs.store(name="st_cdgm_default", node=STCDGMConfig)


def _build_encoder(cfg: EncoderConfig) -> IntelligibleVariableEncoder:
    meta_configs = [
        IntelligibleVariableConfig(
            name=mp.name,
            meta_path=(mp.src, mp.relation, mp.target),
            pool=mp.pool,
        )
        for mp in cfg.metapaths
    ]
    return IntelligibleVariableEncoder(
        configs=meta_configs,
        hidden_dim=cfg.hidden_dim,
        conditioning_dim=cfg.conditioning_dim,
    )


def build_encoder_for_graph(cfg_encoder, builder: HeteroGraphBuilder) -> IntelligibleVariableEncoder:
    """
    Construit l'encodeur en ne gardant que les méta-chemins dont src/cible existent dans le graphe.

    Si ``graph.include_mid_layer`` est False, seuls GP850 (et SP_HR si données statiques) sont
    présents : les chemins vers GP500/GP250 doivent être ignorés pour éviter KeyError après HeteroConv.
    Aligné sur ``train_ddp.py`` et les notebooks d'inférence.
    """
    allowed_nodes = set(builder.dynamic_node_types) | set(builder.static_node_types)
    meta_configs: List[IntelligibleVariableConfig] = []
    for mp in cfg_encoder.metapaths:
        if mp.src not in allowed_nodes or mp.target not in allowed_nodes:
            continue
        pool = getattr(mp, "pool", None) or "mean"
        meta_configs.append(
            IntelligibleVariableConfig(
                name=mp.name,
                meta_path=(mp.src, mp.relation, mp.target),
                pool=pool,
            )
        )
    static_key = ("SP_HR", "causes", "GP850")
    if builder.static_dataset is not None and "SP_HR" in allowed_nodes:
        if not any(c.meta_path == static_key for c in meta_configs):
            meta_configs.append(
                IntelligibleVariableConfig(
                    name="static",
                    meta_path=static_key,
                    pool="mean",
                )
            )
    if not meta_configs:
        raise ValueError(
            "Aucun méta-chemin compatible avec le graphe. "
            f"Nœuds disponibles: {sorted(allowed_nodes)}. "
            "Utilisez graph.include_mid_layer: true si vos méta-chemins utilisent GP500/GP250, "
            "ou réduisez encoder.metapaths à la topologie présente (ex. GP850 uniquement)."
        )
    return IntelligibleVariableEncoder(
        configs=meta_configs,
        hidden_dim=cfg_encoder.hidden_dim,
        conditioning_dim=cfg_encoder.conditioning_dim,
    )


def _convert_sample_to_batch(
    sample: DictConfig,
    builder: HeteroGraphBuilder,
    device: torch.device,
) -> dict:
    """
    Convertit un échantillon du DataLoader ResDiff en dictionnaire prêt pour train_epoch.
    """
    lr_seq = sample["lr"]  # [seq_len, channels, lat, lon]
    seq_len, _, _, _ = lr_seq.shape

    lr_nodes_steps: List[torch.Tensor] = []
    for t in range(seq_len):
        lr_nodes_steps.append(builder.lr_grid_to_nodes(lr_seq[t]))
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)  # [seq_len, N_lr, channels]

    dynamic_features = {}
    for node_type in builder.dynamic_node_types:
        dynamic_features[node_type] = lr_nodes_steps[0]

    hetero = builder.prepare_step_data(dynamic_features).to(device)

    batch = {
        "lr": lr_tensor,
        "residual": sample["residual"],
        "baseline": sample.get("baseline"),
        "hetero": hetero,
    }
    if "valid_mask" in sample:
        batch["valid_mask"] = sample["valid_mask"]
    return batch


def _iterate_batches(
    dataloader: Iterable[dict],
    builder: HeteroGraphBuilder,
    device: torch.device,
) -> Iterator[dict]:
    for sample in dataloader:
        yield _convert_sample_to_batch(sample, builder, device)


@hydra.main(version_base=None, config_name="st_cdgm_default")
def main(cfg: DictConfig) -> None:
    # CPU threading: on a CPU-only host, use *all* logical cores for intra-op
    # math (BLAS/MKL/oneDNN), and a small fixed pool for inter-op scheduling.
    # OMP_NUM_THREADS / MKL_NUM_THREADS must be set before any heavy import that
    # initialises the threadpool — set them defensively here in case the user
    # didn't export them.
    ncpu = os.cpu_count() or 1
    os.environ.setdefault("OMP_NUM_THREADS", str(ncpu))
    os.environ.setdefault("MKL_NUM_THREADS", str(ncpu))
    torch.set_num_threads(ncpu)
    try:
        torch.set_num_interop_threads(2)
    except RuntimeError:
        # set_num_interop_threads can only be called once per process; ignore
        # if a parent already configured it (e.g. when re-entering main).
        pass
    print(f"[PERF] torch.set_num_threads({ncpu}), interop=2, OMP/MKL={ncpu}")

    print("===== ST-CDGM training configuration =====")
    print(OmegaConf.to_yaml(cfg))

    device = torch.device(cfg.training.device)

    pipeline = NetCDFDataPipeline(
        lr_path=cfg.data.lr_path,
        hr_path=cfg.data.hr_path,
        static_path=cfg.data.static_path,
        seq_len=cfg.data.seq_len,
        baseline_strategy=cfg.data.baseline_strategy,
        baseline_factor=cfg.data.baseline_factor,
        normalize=cfg.data.normalize,
        nan_fill_strategy=getattr(cfg.data, 'nan_fill_strategy', 'zero'),
        precipitation_delta=getattr(cfg.data, 'precipitation_delta', 0.01),
    )
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
    # DataLoader: num_workers=0 on CyVerse/Docker (limited /dev/shm)
    num_workers = getattr(cfg.training, 'num_workers', 0)
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
    _proj_in = len(encoder.configs) * encoder.conditioning_dim
    unet_kwargs["projection_class_embeddings_input_dim"] = _proj_in
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
    
    # Phase A3: Compile critical modules with torch.compile for performance
    # Reads training.compile (same layout as training_config.yaml)
    compile_cfg = cfg.training.get("compile", {}) or {}
    compile_enabled = compile_cfg.get("enabled", False)
    compile_mode_rcn = compile_cfg.get("rcn_mode", "reduce-overhead")
    compile_mode_diffusion = compile_cfg.get("diffusion_mode", "max-autotune")
    compile_mode_encoder = compile_cfg.get("encoder_mode", "reduce-overhead")
    _cuda = torch.cuda.is_available()

    if compile_enabled and hasattr(torch, 'compile'):
        print("\n" + "="*80)
        print("Phase A3: Compiling modules with torch.compile")
        print("="*80)
        
        try:
            # Compile RCN cell - Phase A2 vectorization makes this very effective
            # reduce-overhead mode reduces Python overhead (good for RCN)
            rcn_cell = torch.compile(rcn_cell, mode=compile_mode_rcn, fullgraph=False)
            # Update the runner to use the compiled cell
            rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=cfg.rcn.get('detach_interval'))
            print(f"✓ RCN cell compiled with torch.compile (mode: {compile_mode_rcn})")
            print(f"  - Vectorized GRU computation (Phase A2) is compilation-friendly")
        except Exception as e:
            print(f"⚠ torch.compile for RCN cell failed: {e}")
            print(f"  - Falling back to uncompiled RCN cell")
            import traceback
            if compile_cfg.get("verbose_errors", False):
                traceback.print_exc()
        
        try:
            # Compile encoder - test if compatible with PyG HeteroData
            # reduce-overhead is safer for PyG operations
            encoder = torch.compile(encoder, mode=compile_mode_encoder, fullgraph=False)
            print(f"✓ Encoder compiled with torch.compile (mode: {compile_mode_encoder})")
        except Exception as e:
            print(f"⚠ torch.compile for encoder failed: {e}")
            print(f"  - Encoder may use PyG operations incompatible with compile")
            print(f"  - Falling back to uncompiled encoder")
            if compile_cfg.get("verbose_errors", False):
                import traceback
                traceback.print_exc()
        
        try:
            if _cuda:
                diffusion = torch.compile(diffusion, mode=compile_mode_diffusion, fullgraph=False)
                print(f"✓ Diffusion decoder compiled with torch.compile (mode: {compile_mode_diffusion})")
            else:
                print("⚠ Skipping torch.compile for diffusion decoder (CUDA not available)")
        except Exception as e:
            print(f"⚠ torch.compile for diffusion decoder failed: {e}")
            print(f"  - Falling back to uncompiled diffusion decoder")
            if compile_cfg.get("verbose_errors", False):
                import traceback
                traceback.print_exc()
        
        print("="*80 + "\n")
    else:
        if not hasattr(torch, 'compile'):
            print("⚠ torch.compile not available (PyTorch < 2.0). Skipping compilation.")
        elif not compile_enabled:
            print("⚠ torch.compile disabled in config. Skipping compilation.")

    spatial_projector = SpatialConditioningProjector(
        num_vars=len(encoder.configs),
        hidden_dim=cfg.rcn.hidden_dim,
        conditioning_dim=cfg.diffusion.conditioning_dim,
        lr_shape=tuple(cfg.graph.lr_shape),
        target_shape=tuple(cfg.diffusion.get("spatial_target_shape", [6, 7])),
    ).to(device)

    # Phase DAG-decouple: split the optimizer into two parameter groups. The
    # DAG adjacency has a very different loss landscape (only L_dag + L_rec
    # push on it now, thanks to the ``detach_dag_in_state`` guard) and it
    # benefits from a higher LR and lighter momentum — otherwise Adam state
    # accumulated from diffusion-dominated steps bleeds into A_dag and the
    # two objectives fight each other.
    dag_params = [p for n, p in rcn_cell.named_parameters() if n == "A_dag"]
    other_rcn_params = [p for n, p in rcn_cell.named_parameters() if n != "A_dag"]
    other_params = (
        list(encoder.parameters())
        + other_rcn_params
        + list(diffusion.parameters())
        + list(spatial_projector.parameters())
    )
    dag_lr_mult = float(cfg.training.get("dag_lr_mult", 5.0))
    optimizer = torch.optim.Adam(
        [
            {"params": other_params, "lr": cfg.training.lr},
            {
                "params": dag_params,
                "lr": cfg.training.lr * dag_lr_mult,
                "betas": (0.5, 0.9),
            },
        ]
    )

    # Phase DAG-decouple: DAG curriculum. Ramp gamma_dag from 0 to 1 over
    # the first ``dag_warmup_epochs`` epochs so the diffusion branch has
    # time to find a sensible local minimum before the acyclicity penalty
    # starts tugging on the shared encoder/RCN gradients.
    dag_warmup_epochs = int(cfg.loss.get("dag_warmup_epochs", 5))
    dag_l1_reg = bool(cfg.loss.get("dag_l1_regularization", False))
    dag_l1_w = float(cfg.loss.get("dag_l1_weight", 0.01))
    dag_spectral_proj = bool(cfg.loss.get("dag_spectral_projection", True))
    dag_spectral_radius = float(cfg.loss.get("dag_spectral_max_radius", 0.95))

    for epoch in range(cfg.training.epochs):
        if dag_warmup_epochs > 0:
            dag_warmup_scale = min(1.0, (epoch + 1) / dag_warmup_epochs)
        else:
            dag_warmup_scale = 1.0

        batch_iter = _iterate_batches(dataloader, builder, device)
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
            use_focal_loss=cfg.loss.get("use_focal_loss", False),
            focal_alpha=cfg.loss.get("focal_alpha", 1.0),
            focal_gamma=cfg.loss.get("focal_gamma", 2.0),
            extreme_weight_factor=cfg.loss.get("extreme_weight_factor", 0.0),
            extreme_percentiles=list(cfg.loss.get("extreme_percentiles", [95.0, 99.0])),
            reconstruction_loss_type=cfg.loss.get("reconstruction_loss_type", "mse"),
            use_spectral_loss=cfg.loss.get("use_spectral_loss", False),
            lambda_spectral=cfg.loss.get("lambda_spectral", 0.0),
            conditioning_dropout_prob=cfg.diffusion.get("conditioning_dropout_prob", 0.0),
            lambda_dag_prior=cfg.loss.get("lambda_dag_prior", 0.0),
            dag_prior=torch.tensor(cfg.loss.dag_prior, dtype=torch.float32) if cfg.loss.get("dag_prior") else None,
            spatial_projector=spatial_projector,
            dag_warmup_scale=dag_warmup_scale,
            dag_l1_regularization=dag_l1_reg,
            dag_l1_weight=dag_l1_w,
            dag_spectral_projection=dag_spectral_proj,
            dag_spectral_max_radius=dag_spectral_radius,
        )
        if cfg.loss.get("log_spectral_metric_each_epoch", False):
            amp_m = resolve_train_amp_mode(device, cfg.training.get("use_amp", True))
            try:
                biter = _iterate_batches(dataloader, builder, device)
                batch0 = next(biter)
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
        if (epoch + 1) % cfg.training.log_every == 0:
            pretty = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"[Epoch {epoch + 1}] {pretty}")


if __name__ == "__main__":
    main()

