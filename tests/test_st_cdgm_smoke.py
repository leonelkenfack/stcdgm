import itertools
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import torch
import xarray as xr

from st_cdgm import (
    RCNCell,
    RCNSequenceRunner,
    NetCDFDataPipeline,
    CausalDiffusionDecoder,
    HeteroGraphBuilder,
    IntelligibleVariableConfig,
    IntelligibleVariableEncoder,
    SpatialConditioningProjector,
    train_epoch,
)


def _create_synthetic_dataset(time_steps: int, lr_shape: tuple[int, int], hr_shape: tuple[int, int]) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    times = pd.date_range("2000-01-01", periods=time_steps, freq="D")
    lr_lat = np.linspace(-10, 10, lr_shape[0])
    lr_lon = np.linspace(0, 20, lr_shape[1])
    hr_lat = np.linspace(-10, 10, hr_shape[0])
    hr_lon = np.linspace(0, 20, hr_shape[1])

    lr_data = np.random.randn(time_steps, lr_shape[0], lr_shape[1]).astype(np.float32)
    hr_data = np.random.randn(time_steps, hr_shape[0], hr_shape[1]).astype(np.float32)
    static_data = np.random.rand(hr_shape[0], hr_shape[1]).astype(np.float32)

    lr_ds = xr.Dataset(
        {"q_850": (("time", "lat", "lon"), lr_data)},
        coords={"time": times, "lat": lr_lat, "lon": lr_lon},
    )
    hr_ds = xr.Dataset(
        {"tas": (("time", "lat", "lon"), hr_data)},
        coords={"time": times, "lat": hr_lat, "lon": hr_lon},
    )
    static_ds = xr.Dataset(
        {"orog": (("lat", "lon"), static_data)},
        coords={"lat": hr_lat, "lon": hr_lon},
    )
    return lr_ds, hr_ds, static_ds


def _convert_sample(sample: dict, builder: HeteroGraphBuilder, device: torch.device) -> dict:
    seq_len = sample["lr"].shape[0]
    lr_nodes_steps = []
    for step in range(seq_len):
        lr_nodes_steps.append(builder.lr_grid_to_nodes(sample["lr"][step]))
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)

    dynamic_feats = {node_type: lr_nodes_steps[0] for node_type in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_feats).to(device)

    batch = {
        "lr": lr_tensor,
        "residual": sample["residual"],
        "baseline": sample.get("baseline"),
        "hetero": hetero,
    }
    if "valid_mask" in sample:
        batch["valid_mask"] = sample["valid_mask"]
    return batch


def _run_one_smoke_train_epoch(tmp_path: Path, *, use_amp: bool) -> dict:
    lr_ds, hr_ds, static_ds = _create_synthetic_dataset(time_steps=6, lr_shape=(2, 3), hr_shape=(4, 6))

    lr_path = tmp_path / "lr.nc"
    hr_path = tmp_path / "hr.nc"
    static_path = tmp_path / "static.nc"
    lr_ds.to_netcdf(lr_path)
    hr_ds.to_netcdf(hr_path)
    static_ds.to_netcdf(static_path)

    pipeline = NetCDFDataPipeline(
        lr_path=lr_path,
        hr_path=hr_path,
        static_path=static_path,
        seq_len=3,
        baseline_strategy="hr_smoothing",
        baseline_factor=1,
        normalize=False,
    )
    dataset = pipeline.build_sequence_dataset(seq_len=3, as_torch=True)
    sample = next(iter(dataset))

    device = torch.device("cpu")

    builder = HeteroGraphBuilder(
        lr_shape=(2, 3),
        hr_shape=(4, 6),
        static_dataset=pipeline.get_static_dataset(),
        include_mid_layer=False,
    )

    driver_nodes = builder.lr_grid_to_nodes(sample["lr"][0])
    driver_dim = driver_nodes.shape[1]

    encoder = IntelligibleVariableEncoder(
        configs=[
            IntelligibleVariableConfig(
                name="surface",
                meta_path=("GP850", "spat_adj", "GP850"),
            ),
            IntelligibleVariableConfig(
                name="static",
                meta_path=("SP_HR", "causes", "GP850"),
            ),
        ],
        hidden_dim=32,
        conditioning_dim=32,
    ).to(device)

    rcn_cell = RCNCell(
        num_vars=2,
        hidden_dim=32,
        driver_dim=driver_dim,
        reconstruction_dim=driver_dim,
        dropout=0.0,
    ).to(device)
    rcn_runner = RCNSequenceRunner(rcn_cell)

    diffusion = CausalDiffusionDecoder(
        in_channels=1,
        conditioning_dim=32,
        height=sample["residual"].shape[-2],
        width=sample["residual"].shape[-1],
        num_diffusion_steps=50,
        unet_kwargs=dict(
            layers_per_block=1,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            mid_block_type="UNetMidBlock2D",
            norm_num_groups=8,
            class_embed_type="projection",
            projection_class_embeddings_input_dim=len(encoder.configs) * encoder.conditioning_dim,
            resnet_time_scale_shift="scale_shift",
            attention_head_dim=32,
            only_cross_attention=[False, True],
        ),
    ).to(device)

    spatial_projector = SpatialConditioningProjector(
        num_vars=2, hidden_dim=32, conditioning_dim=32,
        lr_shape=(2, 3), target_shape=(1, 2),
    ).to(device)

    params = (
        list(encoder.parameters()) + list(rcn_cell.parameters())
        + list(diffusion.parameters()) + list(spatial_projector.parameters())
    )
    optimizer = torch.optim.Adam(params, lr=1e-4)

    batch = _convert_sample(sample, builder, device)
    metrics = train_epoch(
        encoder=encoder,
        rcn_runner=rcn_runner,
        diffusion_decoder=diffusion,
        optimizer=optimizer,
        data_loader=itertools.repeat(batch, 1),
        lambda_gen=1.0,
        beta_rec=0.1,
        gamma_dag=0.1,
        conditioning_fn=None,
        device=device,
        gradient_clipping=1.0,
        use_amp=use_amp,
        spatial_projector=spatial_projector,
    )
    return metrics


def test_st_cdgm_smoke(tmp_path: Path):
    metrics = _run_one_smoke_train_epoch(tmp_path, use_amp=True)
    assert "loss" in metrics and np.isfinite(metrics["loss"])


def test_film_conditioning_ablation(tmp_path: Path):
    """Verify that FiLM conditioning is effective: output must change when conditioning is zeroed."""
    lr_ds, hr_ds, static_ds = _create_synthetic_dataset(time_steps=6, lr_shape=(2, 3), hr_shape=(4, 6))
    lr_path = tmp_path / "lr.nc"
    hr_path = tmp_path / "hr.nc"
    static_path = tmp_path / "static.nc"
    lr_ds.to_netcdf(lr_path)
    hr_ds.to_netcdf(hr_path)
    static_ds.to_netcdf(static_path)

    pipeline = NetCDFDataPipeline(
        lr_path=lr_path, hr_path=hr_path, static_path=static_path,
        seq_len=3, baseline_strategy="hr_smoothing", baseline_factor=1, normalize=False,
    )
    sample = next(iter(pipeline.build_sequence_dataset(seq_len=3, as_torch=True)))
    device = torch.device("cpu")
    _q, _cdim = 2, 32  # align with cond_real [1, 2, 32] → FiLM input q * cdim

    diffusion = CausalDiffusionDecoder(
        in_channels=1, conditioning_dim=32,
        height=sample["residual"].shape[-2], width=sample["residual"].shape[-1],
        num_diffusion_steps=50,
        unet_kwargs=dict(
            layers_per_block=1, block_out_channels=(32, 64),
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
            mid_block_type="UNetMidBlock2D", norm_num_groups=8,
            class_embed_type="projection",
            projection_class_embeddings_input_dim=_q * _cdim,
            resnet_time_scale_shift="scale_shift",
            attention_head_dim=32,
            only_cross_attention=[False, True],
        ),
    ).to(device)
    diffusion.eval()

    noisy = torch.randn(1, 1, sample["residual"].shape[-2], sample["residual"].shape[-1], device=device)
    timestep = torch.tensor([10], device=device)

    cond_real = torch.randn(1, 2, 32, device=device)
    cond_zero = torch.zeros(1, 2, 32, device=device)

    with torch.no_grad():
        out_real = diffusion(noisy, timestep, cond_real)
        out_zero = diffusion(noisy, timestep, cond_zero)

    diff = (out_real - out_zero).abs().max().item()
    assert diff > 1e-6, f"FiLM conditioning has no effect: max diff = {diff}"


def test_train_epoch_cpu_bf16_amp_when_supported(tmp_path: Path):
    """Exercises train_epoch with use_amp=True on CPU when BF16 is available (no GradScaler)."""
    bf16 = getattr(torch.cpu, "is_bf16_supported", None)
    if bf16 is None or not bf16():
        pytest.skip("CPU bfloat16 not supported on this host")
    metrics = _run_one_smoke_train_epoch(tmp_path, use_amp=True)
    assert "loss" in metrics and np.isfinite(metrics["loss"])

