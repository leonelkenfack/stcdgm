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
    train_epoch,
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
    include_mid_layer: bool = False
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
    driver_dim: int = 8
    reconstruction_dim: Optional[int] = 8
    dropout: float = 0.0
    detach_interval: Optional[int] = None


@dataclass
class DiffusionConfig:
    in_channels: int = 3
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


@dataclass
class STCDGMConfig:
    data: DataConfig = DataConfig()
    graph: GraphConfig = GraphConfig()
    encoder: EncoderConfig = EncoderConfig()
    rcn: RCNConfig = RCNConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    loss: LossConfig = LossConfig()
    training: TrainingConfig = TrainingConfig()


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
    # Optimized DataLoader with parallel workers and pinned memory
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
    ).to(device)
    
    # Phase A3: Compile critical modules with torch.compile for performance
    # This provides 10-50% speedup on compatible PyTorch versions (>= 2.0)
    # The vectorized RCN (Phase A2) eliminates Python loops, making compilation more effective
    compile_enabled = cfg.get('compile', {}).get('enabled', True)
    compile_mode_rcn = cfg.get('compile', {}).get('mode_rcn', 'reduce-overhead')
    compile_mode_diffusion = cfg.get('compile', {}).get('mode_diffusion', 'max-autotune')
    compile_mode_encoder = cfg.get('compile', {}).get('mode_encoder', 'reduce-overhead')
    
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
            if cfg.get('compile', {}).get('verbose_errors', False):
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
            if cfg.get('compile', {}).get('verbose_errors', False):
                import traceback
                traceback.print_exc()
        
        try:
            # Compile diffusion decoder - max-autotune for best performance
            # Diffusion models benefit from aggressive optimizations
            diffusion = torch.compile(diffusion, mode=compile_mode_diffusion, fullgraph=False)
            print(f"✓ Diffusion decoder compiled with torch.compile (mode: {compile_mode_diffusion})")
        except Exception as e:
            print(f"⚠ torch.compile for diffusion decoder failed: {e}")
            print(f"  - Falling back to uncompiled diffusion decoder")
            if cfg.get('compile', {}).get('verbose_errors', False):
                import traceback
                traceback.print_exc()
        
        print("="*80 + "\n")
    else:
        if not hasattr(torch, 'compile'):
            print("⚠ torch.compile not available (PyTorch < 2.0). Skipping compilation.")
        elif not compile_enabled:
            print("⚠ torch.compile disabled in config. Skipping compilation.")

    params = list(encoder.parameters()) + list(rcn_cell.parameters()) + list(diffusion.parameters())
    optimizer = torch.optim.Adam(params, lr=cfg.training.lr)

    for epoch in range(cfg.training.epochs):
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
        )
        if (epoch + 1) % cfg.training.log_every == 0:
            pretty = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            print(f"[Epoch {epoch + 1}] {pretty}")


if __name__ == "__main__":
    main()

