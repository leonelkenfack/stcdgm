# ST-CDGM Quickstart

This guide summarises how to invoke the end-to-end ST-CDGM training harness that
connects the data pipeline, heterogeneous graph builder, causal RCN core, and
diffusion decoder.

## 1. Prepare the Environment

Install the project dependencies (PyTorch, PyG, diffusers, Hydra, xbatcher):

```bash
pip install -r requirements.txt
```

At minimum the following packages must be available:

```
torch, torch-geometric, diffusers, hydra-core, xarray, xbatcher, pandas
```

## 2. Provide Input Data

The training script expects:

- a low-resolution predictor dataset (`lr_path`) in NetCDF/Zarr format,
- a high-resolution target dataset (`hr_path`),
- optional static high-resolution fields (`static_path`), such as topography.

Each file must expose a common `time` coordinate. The LR predictors will be
normalised (if enabled), while the HR data are decomposed into deterministic
baselines and residual targets following the ResDiff paradigm.

## 3. Run the Hydra Training Driver

The driver lives at `ops/train_st_cdgm.py`. It exposes a Hydra configuration
which can be overridden directly from the command line.

```bash
python ops/train_st_cdgm.py \
    data.lr_path=/path/to/lr.nc \
    data.hr_path=/path/to/hr.nc \
    data.static_path=/path/to/static.nc \
    data.seq_len=6 \
    diffusion.height=128 diffusion.width=128 \
    training.epochs=5 training.lr=2e-4
```

Key configuration groups:

- `data`: controls sequence length, baseline strategy (`hr_smoothing` or `lr_interp`), and streaming stride.
- `graph`: sets LR/HR grid shapes and whether mid-level layers are included.
- `encoder`: meta-path definitions for the intelligible variables and the hidden dimension.
- `rcn`: specifies the SCM-GRU core (hidden dimension, dropout, reconstruction head).
- `diffusion`: configures the residual diffusion decoder (UNet channels, conditioning size).
- `loss`: weights for the joint loss terms (`λ_gen`, `β_rec`, `γ_dag`).
- `training`: optimiser hyperparameters and logging cadence.

The script prints the active configuration (via `OmegaConf`) at start-up, then
executes `training_loop.train_epoch` for the requested number of epochs.

## 4. Smoke Test

A synthetic smoke test is available at `tests/test_st_cdgm_smoke.py`. It creates
toy LR/HR datasets, builds all modules, and runs a single optimisation step,
verifying that no component breaks when assembled. Run it with:

```bash
pytest tests/test_st_cdgm_smoke.py
```

## 5. Next Steps

- Integrate real data paths and tune the hyperparameters per experiment.
- Hook the evaluation utilities (`evaluation_xai.py`) to generate CRPS, spectral,
  and histogram-based diagnostics, alongside DAG heatmaps for scientific review.
- Extend the Hydra configuration to include wandb logging, checkpointing, and
  multi-run sweeps.

This quickstart should help you transition from the legacy cGAN workflow to the
causal ST-CDGM pipeline while retaining reproducibility and interpretability.

