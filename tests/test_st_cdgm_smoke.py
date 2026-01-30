import itertools
from pathlib import Path

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
    return batch


def test_st_cdgm_smoke(tmp_path: Path):
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
        in_channels=3,
        conditioning_dim=32,
        height=sample["residual"].shape[-2],
        width=sample["residual"].shape[-1],
        num_diffusion_steps=50,
    ).to(device)

    params = list(encoder.parameters()) + list(rcn_cell.parameters()) + list(diffusion.parameters())
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
    )

    assert "loss" in metrics and np.isfinite(metrics["loss"])

