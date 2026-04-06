# Documentation complete du projet ST-CDGM

## Spatio-Temporal Causal Diffusion Generative Model

Ce document est genere automatiquement par `scripts/gen_project_complet_st_cdgm.py`.
Derniere generation : **2026-04-01 06:09 UTC**.

Les **notebooks** (.ipynb) ne sont pas embarques en integralite : seul un resume (nombre de cellules) est fourni.

---

## Structure du Projet

```
climate_data/
├── .dockerignore
├── .gitignore
├── copywritting.md
├── docker-compose.yml
├── Dockerfile
├── environment.yml
├── explicabilite.md
├── PROMPT_IMPLEMENTATION_ST_CDGM.md
├── README.md
├── report_gemini.md
├── requirements.txt
├── resume_training_from_checkpoint.ipynb
├── setup.py
├── st_cdgm_publication_figures.ipynb
├── st_cdgm_results_presentation.ipynb
├── st_cdgm_training_evaluation.ipynb
├── st_cdgm_validation_inference.ipynb
├── stats.md
├── train_ddp.py
├── config/
│   ├── docker.env
│   ├── training_config.yaml
│   └── training_config_vice.yaml
├── data/
│   ├── metadata/
│   │   ├── NorESM2-MM_histupdated_compressed.metadata.csv
│   │   └── NorESM2-MM_histupdated_compressed.metadata.json
│   └── raw/
├── docs/
│   ├── ARCHITECTURE_MODEL.md
│   ├── DOCKER_README.md
│   ├── GUIDE_PEDAGOGIQUE_ST-CDGM.md
│   ├── OPTIMISATION.md
│   ├── rapport_avantages_st_cdgm.md
│   ├── rapport_flux_scm_gru_unet.md
│   ├── rapport_optimisation_unet.md
│   ├── RAPPORT_TECHNIQUE_COMPLET.md
│   ├── research_article_st_cdgm.md
│   ├── SCRIPTS_README.md
│   └── st_cdgm_quickstart.md
├── models/
│   └── *.pth (checkpoints)
├── ops/
│   ├── preprocess_to_shards.py
│   ├── preprocess_to_zarr.py
│   └── train_st_cdgm.py
├── scripts/
│   ├── cleanup_repeated_lines.py
│   ├── gen_project_complet_st_cdgm.py
│   ├── load_model.py
│   ├── run_evaluation.py
│   ├── run_full_pipeline.py
│   ├── run_preprocessing.py
│   ├── run_training.py
│   ├── save_model.py
│   ├── sync_datastore.py
│   ├── test_installation.py
│   ├── test_pipeline.py
│   ├── validate_antismoothing.py
│   ├── validate_setup.py
│   └── vice_utils.py
├── src/
│   ├── st_cdgm/
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── netcdf_utils.py
│   │   │   └── pipeline.py
│   │   ├── evaluation/
│   │   │   ├── __init__.py
│   │   │   └── evaluation_xai.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── causal_rcn.py
│   │   │   ├── diffusion_decoder.py
│   │   │   ├── graph_builder.py
│   │   │   └── intelligible_encoder.py
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── callbacks.py
│   │   │   ├── multi_gpu.py
│   │   │   └── training_loop.py
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   └── checkpoint.py
│   │   └── __init__.py
│   └── st_cdgm.egg-info/
│       ├── dependency_links.txt
│       ├── PKG-INFO
│       ├── requires.txt
│       ├── SOURCES.txt
│       └── top_level.txt
└── tests/
    ├── __init__.py
    ├── test_corrections_antilissage.py
    ├── test_installation.py
    └── test_st_cdgm_smoke.py
```

---

## Fichiers de configuration et code source

### `config/docker.env`

```env
# Docker Environment Variables for ST-CDGM Training

# GPU Configuration
CUDA_VISIBLE_DEVICES=all
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Python Configuration
PYTHONPATH=/workspace/src:/workspace
PYTHONUNBUFFERED=1

# Default Data Paths (can be overridden)
DATA_RAW_DIR=/workspace/data/raw
DATA_PROCESSED_DIR=/workspace/data/processed
MODELS_DIR=/workspace/models
RESULTS_DIR=/workspace/results

# Training Configuration (defaults)
DEFAULT_CONFIG=training_config.yaml

# Optional: Jupyter Configuration
# JUPYTER_ENABLE_LAB=yes
# JUPYTER_PORT=8888

# Optional: Mixed Precision (default: enabled if GPU available)
USE_AMP=true

# Optional: Logging
LOG_LEVEL=INFO
LOG_DIR=/workspace/results/logs
```

---

### `config/training_config.yaml`

```yaml
# Hydra Training Configuration for ST-CDGM
# This configuration includes all new optimizations (Phase C, D, E)
#
# FOR CYVERSE VICE USERS:
# - See config/training_config_vice.yaml for VICE-optimized configuration
# - See CYVERSE_VICE_SETUP.md for installation and usage instructions
# - Data Store paths (~/data-store/home/<username>/) are slower for large files
# - Recommended: Copy data to local disk first using scripts/sync_datastore.py

defaults:
  - _self_

# Data Configuration
# 
# For CyVerse VICE users:
# - Option 1 (RECOMMENDED): Copy data to ~/climate_data/data/raw/ first
#   Use relative paths: "data/raw/your_file.nc"
# - Option 2: Use Data Store paths directly (slower):
#   "~/data-store/home/<username>/data/raw/your_file.nc"
data:
  # netcdf = lecture directe des .nc (notebooks, resume_training_from_checkpoint).
  # zarr / shard = pipelines train_ddp avec repertoires preprocesse (voir train_ddp.py).
  dataset_format: "netcdf"
  lr_path: "data/raw/train/predictor_ACCESS-CM2_hist.nc"
  hr_path: "data/raw/train/pr_ACCESS-CM2_hist.nc"
  static_path: "data/raw/static_predictors/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc"
  zarr_dir: "data/raw/train/zarr"  # Répertoire pour données Zarr préprocessées
  shard_dir: "data/raw/train/shards"  # Répertoire pour shards WebDataset
  seq_len: 8  # Increased for 256 Go RAM
  stride: 2  # Faster dataset iteration
  baseline_strategy: "hr_smoothing"  # or "lr_interp"
  baseline_factor: 4
  normalize: true
  target_transform: "log1p"  # ["log1p", "none"]
  nan_fill_strategy: "mean"  # ["zero", "mean", "interpolate"]
  precipitation_delta: 0.01  
  lr_variables: ['u_850', 'u_500', 'u_250', 'v_850', 'v_500', 'v_250', 'w_850', 'w_500', 'w_250', 'q_850', 'q_500', 'q_250', 't_850', 't_500', 't_250']
  hr_variables: ["pr"]
  static_variables: ["orog", "he", "vegt"]

# Graph Configuration
graph:
  lr_shape: [23, 26]
  hr_shape: [172, 179]
  static_variables: []
  # true = nœuds GP500/GP250 + arêtes verticales (cohérent avec encoder.metapaths)
  include_mid_layer: true

# Encoder Configuration
encoder:
  hidden_dim: 128
  conditioning_dim: 128
  metapaths:
    - name: "GP850_spat_adj"
      src: "GP850"
      relation: "spat_adj"
      target: "GP850"
      pool: "max"
    - name: "GP850_to_GP500"
      src: "GP850"
      relation: "vert_adj"
      target: "GP500"
      pool: "max"
    - name: "GP500_spat_adj"
      src: "GP500"
      relation: "spat_adj"
      target: "GP500"
      pool: "max"
    - name: "GP500_to_GP250"
      src: "GP500"
      relation: "vert_adj"
      target: "GP250"
      pool: "max"
    - name: "GP250_spat_adj"
      src: "GP250"
      relation: "spat_adj"
      target: "GP250"
      pool: "max"

# RCN Configuration
rcn:
  hidden_dim: 128
  driver_dim: 15      # overridden at runtime by data sample shape
  reconstruction_dim: 15  # overridden at runtime by data sample shape
  dropout: 0.0
  detach_interval: 4  # truncated BPTT: gradient flows back 4 steps instead of seq_len

# Diffusion Decoder Configuration
diffusion:
  in_channels: 1      # overridden at runtime by data sample shape
  conditioning_dim: 128  # must match encoder.conditioning_dim
  height: 172
  width: 179
  steps: 20
  scheduler_type: "ddpm"  # "ddpm", "edm", or "dpm_solver++"
  use_gradient_checkpointing: false
  cfg_scale: 0.2
  conditioning_dropout_prob: 0.1
  conv_padding_mode: "zeros"
  anti_checkerboard: false
  unet_kwargs:
    layers_per_block: 1
    block_out_channels: [32, 64]
    down_block_types: ["DownBlock2D", "CrossAttnDownBlock2D"]
    up_block_types: ["CrossAttnUpBlock2D", "UpBlock2D"]
    mid_block_type: "UNetMidBlock2D"
    norm_num_groups: 8
    class_embed_type: "projection"
    projection_class_embeddings_input_dim: 640
    resnet_time_scale_shift: "scale_shift"
    attention_head_dim: 32
    only_cross_attention: [false, true]

# Loss Configuration
loss:
  lambda_gen: 1.0
  beta_rec: 0.05
  gamma_dag: 0.1
  lambda_phy: 0.0
  lambda_spectral: 0.0
  use_spectral_loss: false
  log_spectral_metric_each_epoch: true
  dag_method: "dagma"
  dagma_s: 1.0
  use_focal_loss: true
  focal_alpha: 1.0
  focal_gamma: 2.0
  extreme_weight_factor: 2.0
  extreme_percentiles: [95.0, 99.0]
  dag_l1_regularization: true
  dag_l1_weight: 0.01
  reconstruction_loss_type: "mse+cosine"
  # Soft prior on DAG: encourages physically plausible edges.
  # Rows = target variable, Cols = source variable.
  # Variables: 0=GP850_spat, 1=GP850→GP500, 2=GP500_spat, 3=GP500→GP250, 4=GP250_spat
  # Positive values encourage an edge; 0 = no preference.
  lambda_dag_prior: 0.01
  dag_prior:
    - [0.0, 0.3, 0.0, 0.0, 0.0]
    - [0.3, 0.0, 0.3, 0.0, 0.0]
    - [0.0, 0.3, 0.0, 0.3, 0.0]
    - [0.0, 0.0, 0.3, 0.0, 0.3]
    - [0.0, 0.0, 0.0, 0.3, 0.0]

# Training Configuration
training:
  device: "cpu"  # "cuda" or "cpu" - use "cpu" if PyTorch not compiled with CUDA
  epochs: 10  
  batch_size: 48  # Reduce to 8-16 if OOM on CPU
  num_workers: 0  # 0 = avoid shared memory (/dev/shm). CyVerse/Docker have ~64MB shm, use 0.
  lr: 0.0002  # Increased for larger batch size
  gradient_clipping: 1.0
  log_every: 10
  # CUDA: FP16 + GradScaler. CPU: bfloat16 autocast (no GradScaler) when supported; else FP32.
  use_amp: true
  
  # Multi-GPU configuration (4 GPUs available)
  multi_gpu:
    enabled: false  # Disable when using CPU
    strategy: "ddp"  # DistributedDataParallel (better than DataParallel)
    gpus: [0, 1, 2, 3]  # Use all 4 GPUs
    find_unused_parameters: true  # Required for PyG HeteroData graphs
  early_stopping:
    enabled: false
    patience: 7
    min_delta: 0.0
    restore_best: true
  lr_scheduler:
    enabled: false
    mode: "min"  # "min" or "max"
    factor: 0.5 
    patience: 3  
    min_lr: 1e-7
  physical_loss:
    use_predicted_output: false
    physical_sample_interval: 10
    physical_num_steps: 15
  compile:
    enabled: false  # disabled on CPU (overhead > benefit); enable on CUDA
    rcn_mode: "reduce-overhead"
    diffusion_mode: "max-autotune"
    encoder_mode: "reduce-overhead"

# Model Checkpointing
checkpoint:
  enabled: true
  save_dir: "models"
  save_every: 5  
  save_best: true  
  max_checkpoints: 5  

# Evaluation Configuration
evaluation:
  enabled: true
  eval_every: 5  
  num_samples: 10
  crps_max_ensemble_members: 32
  compute_f1_extremes: true
  f1_percentiles: [95.0, 99.0]
  save_visualizations: true
  output_dir: "results"
```

---

### `config/training_config_vice.yaml`

```yaml
# Hydra Training Configuration for ST-CDGM - CyVerse VICE Environment
# 
# This configuration is optimized for CyVerse Discovery Environment (VICE).
# 
# IMPORTANT NOTES FOR VICE USERS:
# - Data Store paths (~/data-store/home/<username>/) are slower for large NetCDF files
# - RECOMMENDED: Copy data to local disk (~/climate_data/data/raw/) for better performance
# - Use scripts/sync_datastore.py to copy data from Data Store to local disk
# - Save results regularly to Data Store as VICE containers are ephemeral
# 
# For more information, see CYVERSE_VICE_SETUP.md

defaults:
  - _self_

# Data Configuration
# 
# For VICE users:
# - Option 1 (RECOMMENDED): Copy data to ~/climate_data/data/raw/ first using sync_datastore.py
#   Then use relative paths: "data/raw/your_file.nc"
# - Option 2: Use Data Store paths directly (slower but persistent):
#   "~/data-store/home/<username>/data/raw/your_file.nc"
data:
  # netcdf = lecture directe des .nc ; zarr = train_ddp avec lr.zarr / hr.zarr preprocesse
  dataset_format: "netcdf"
  # Paths relative to project root (recommended for VICE after copying to local disk)
  # Replace these with your actual file paths
  lr_path: "data/raw/predictor_ACCESS-CM2_hist.nc"  # Low-resolution input data
  hr_path: "data/raw/pr_ACCESS-CM2_hist.nc"  # High-resolution target data
  
  # Alternative: Data Store paths (slower but persistent)
  # lr_path: "~/data-store/home/<username>/data/raw/predictor_ACCESS-CM2_hist.nc"
  # hr_path: "~/data-store/home/<username>/data/raw/pr_ACCESS-CM2_hist.nc"
  
  static_path: null  # Optional static fields (e.g., topography)
  zarr_dir: "data/raw/train/zarr"  # Répertoire pour données Zarr préprocessées
  shard_dir: "data/raw/train/shards"  # Répertoire pour shards WebDataset
  seq_len: 6  # Temporal sequence length
  stride: 1  # Stride for sliding window
  baseline_strategy: "hr_smoothing"  # or "lr_interp"
  baseline_factor: 4
  normalize: true
  target_transform: "log1p"  # aligné sur training_config.yaml (v6)
  nan_fill_strategy: "mean"
  precipitation_delta: 0.01
  lr_variables: null  # Auto-detect if null
  hr_variables: null  # Auto-detect if null
  static_variables: null

# Graph Configuration
graph:
  lr_shape: [23, 26]  # Low-resolution grid shape (lat, lon)
  hr_shape: [172, 179]  # High-resolution grid shape (lat, lon)
  static_variables: []
  # true = nœuds GP500/GP250 + arêtes verticales (cohérent avec encoder.metapaths)
  include_mid_layer: true

# Encoder Configuration
encoder:
  hidden_dim: 128
  conditioning_dim: 128
  metapaths:
    - name: "GP850_spat_adj"
      src: "GP850"
      relation: "spat_adj"
      target: "GP850"
      pool: "max"
    - name: "GP850_to_GP500"
      src: "GP850"
      relation: "vert_adj"
      target: "GP500"
      pool: "max"
    - name: "GP500_spat_adj"
      src: "GP500"
      relation: "spat_adj"
      target: "GP500"
      pool: "max"
    - name: "GP500_to_GP250"
      src: "GP500"
      relation: "vert_adj"
      target: "GP250"
      pool: "max"
    - name: "GP250_spat_adj"
      src: "GP250"
      relation: "spat_adj"
      target: "GP250"
      pool: "max"

# RCN Configuration
rcn:
  hidden_dim: 128
  driver_dim: 15      # overridden at runtime by data sample shape
  reconstruction_dim: 15  # overridden at runtime by data sample shape
  dropout: 0.0
  detach_interval: 4  # truncated BPTT: gradient flows back 4 steps instead of seq_len

# Diffusion Decoder Configuration
diffusion:
  in_channels: 1      # overridden at runtime by data sample shape
  conditioning_dim: 128
  height: 172
  width: 179
  steps: 100
  scheduler_type: "ddpm"  # "ddpm", "edm", or "dpm_solver++"
  use_gradient_checkpointing: false
  cfg_scale: 0.2
  conditioning_dropout_prob: 0.1
  conv_padding_mode: "zeros"
  anti_checkerboard: false
  unet_kwargs:
    layers_per_block: 1
    block_out_channels: [32, 64]
    down_block_types: ["DownBlock2D", "CrossAttnDownBlock2D"]
    up_block_types: ["CrossAttnUpBlock2D", "UpBlock2D"]
    mid_block_type: "UNetMidBlock2D"
    norm_num_groups: 8
    class_embed_type: "projection"
    projection_class_embeddings_input_dim: 640
    resnet_time_scale_shift: "scale_shift"
    attention_head_dim: 32
    only_cross_attention: [false, true]

# Loss Configuration
loss:
  lambda_gen: 1.0
  beta_rec: 0.05
  gamma_dag: 0.1
  lambda_phy: 0.0
  lambda_spectral: 0.0
  use_spectral_loss: false
  log_spectral_metric_each_epoch: true
  dag_method: "dagma"  # "dagma" or "no_tears"
  dagma_s: 1.0
  use_focal_loss: true
  focal_alpha: 1.0
  focal_gamma: 2.0
  extreme_weight_factor: 2.0
  extreme_percentiles: [95.0, 99.0]
  dag_l1_regularization: false
  dag_l1_weight: 0.01
  reconstruction_loss_type: "mse+cosine"  # "mse", "cosine", or "mse+cosine"
  lambda_dag_prior: 0.01
  dag_prior:
    - [0.0, 0.3, 0.0, 0.0, 0.0]
    - [0.3, 0.0, 0.3, 0.0, 0.0]
    - [0.0, 0.3, 0.0, 0.3, 0.0]
    - [0.0, 0.0, 0.3, 0.0, 0.3]
    - [0.0, 0.0, 0.0, 0.3, 0.0]

# Training Configuration
# 
# For VICE users:
# - GPU availability depends on VICE configuration
# - Check GPU availability: python -c "import torch; print(torch.cuda.is_available())"
# - If GPU available, set device: "cuda" and enable mixed precision training
# - If CPU only: use_amp true enables bfloat16 autocast when supported (see main training_config)
training:
  # Device: "cuda" or "cpu"
  # In VICE, check GPU availability first before setting to "cuda"
  device: "cpu"  # Change to "cuda" if GPU available in your VICE session
  
  # CRITICAL for CyVerse: num_workers=0 to avoid "No space left on device" /dev/shm errors
  num_workers: 0  # VICE has ~64MB /dev/shm - use 0, never 20+
  
  epochs: 100  # Adjust based on your needs
  lr: 0.0001
  gradient_clipping: 1.0
  log_every: 10  # Log progress every N batches
  
  # Mixed Precision Training (recommended for GPU)
  use_amp: true  # GPU: FP16; CPU: BF16 autocast when supported
  
  # Early Stopping and LR Scheduling
  early_stopping:
    enabled: true  # Recommended for VICE (containers have time limits)
    patience: 7
    min_delta: 0.0
    restore_best: true
  lr_scheduler:
    enabled: true  # Recommended for better convergence
    mode: "min"
    factor: 0.5
    patience: 3
    min_lr: 1e-7
  
  # Physical Loss Options
  physical_loss:
    use_predicted_output: false
    physical_sample_interval: 10
    physical_num_steps: 15
  
  # torch.compile (may improve performance on GPU)
  compile:
    enabled: false  # Enable if GPU available for potential speedup
    rcn_mode: "reduce-overhead"
    diffusion_mode: "max-autotune"
    encoder_mode: "reduce-overhead"

# Model Checkpointing
# 
# IMPORTANT FOR VICE: Save checkpoints regularly as containers are ephemeral
# Use scripts/sync_datastore.py to backup checkpoints to Data Store
checkpoint:
  enabled: true
  save_dir: "models"  # Will be created in ~/climate_data/models/
  save_every: 5  # Save checkpoint every N epochs (adjust based on your needs)
  save_best: true  # Save best model based on validation loss
  max_checkpoints: 5  # Keep only last N checkpoints (to save disk space)

# Evaluation Configuration
evaluation:
  enabled: true
  eval_every: 5  # Evaluate every N epochs
  num_samples: 10  # Number of samples for evaluation metrics
  crps_max_ensemble_members: 32
  compute_f1_extremes: true
  f1_percentiles: [95.0, 99.0]
  save_visualizations: true
  output_dir: "results"  # Will be created in ~/climate_data/results/
  
  # Save results to Data Store regularly:
  # python scripts/sync_datastore.py --save-to-datastore \
  #     ~/climate_data/results/ \
  #     ~/data-store/home/<username>/st-cdgm/results/
```

---

### `docker-compose.yml`

```yaml
services:
  st-cdgm-training:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: st-cdgm-training
    working_dir: /workspace
    
    # GPU support (requires nvidia-docker or Docker with GPU runtime)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    
    # Environment variables
    environment:
      - CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}
      - PYTHONPATH=/workspace/src:/workspace
      - PYTHONUNBUFFERED=1
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
    
    # Volumes for data persistence
    volumes:
      - ./data:/workspace/data                    # Input NetCDF files and processed data
      - ./models:/workspace/models                # Saved model checkpoints
      - ./results:/workspace/results              # Evaluation results and logs
      - ./src:/workspace/src                      # Source code (bind mount)
      - ./ops:/workspace/ops                      # Operations scripts
      - ./scripts:/workspace/scripts              # Execution scripts
      - ./config:/workspace/config                # Configuration files
      - ./tests:/workspace/tests                  # Test files
    
    # Ports (optional, for Jupyter if needed)
    # ports:
    #   - "8888:8888"
    
    # Keep container running (can be overridden with command)
    stdin_open: true
    tty: true
    
    # Default command (can be overridden)
    command: /bin/bash
    
    # Restart policy
    restart: unless-stopped
    
    # Resource limits (optional, adjust based on your system)
    # deploy:
    #   resources:
    #     limits:
    #       cpus: '8'
    #       memory: 32G
```

---

### `Dockerfile`

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt setup.py ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Default command
CMD ["/bin/bash"]
```

---

### `environment.yml`

```yaml
name: st-cdgm
channels:
  - pytorch
  - conda-forge
  - defaults

dependencies:
  # Python version
  - python=3.10

  # Core scientific
  - numpy>=1.21.0
  - pandas>=1.3.0
  - scipy>=1.7.0
  - xarray>=2023.1.0

  # PyTorch (conda install is more reliable)
  - pytorch>=2.0.0
  - torchvision>=0.15.0
  - pytorch-cuda=11.8  # Ou pytorch-cpu si pas de GPU

  # Climate data processing
  - netcdf4>=1.6.0
  - h5netcdf>=1.1.0
  - dask>=2023.1.0

  # Visualization
  - matplotlib>=3.5.0
  - seaborn>=0.12.0

  # Jupyter (optionnel)
  - jupyter>=1.0.0
  - jupyterlab>=3.5.0
  - ipykernel>=6.15.0

  # Testing
  - pytest>=7.0.0

  # Pip dependencies (pas disponibles via conda ou mieux via pip)
  - pip
  - pip:
      # PyTorch Geometric
      - torch-geometric>=2.3.0
      - torch-scatter>=2.1.0
      - torch-sparse>=0.6.17
      
      # Diffusion models
      - diffusers>=0.21.0
      - accelerate>=0.20.0
      - transformers>=4.30.0
      
      # Climate data
      - xbatcher>=0.3.0
      
      # Configuration
      - hydra-core>=1.3.0
      - omegaconf>=2.3.0
      
      # Optional
      - plotly>=5.10.0
      - networkx>=2.8.0
      - tqdm>=4.64.0

# ============================================================================
# Installation avec Conda:
#   conda env create -f environment.yml
#   conda activate st-cdgm
#
# Mise à jour:
#   conda env update -f environment.yml --prune
#
# Export de l'environnement actuel:
#   conda env export > environment-frozen.yml
# ============================================================================
```

---

### `requirements.txt`

```text
# =========================
# Core scientific stack
# =========================
numpy
pandas
scipy
scikit-learn

# =========================
# Data formats & processing
# =========================
xarray
netCDF4
h5netcdf
zarr
dask[complete]
webdataset  # For shard-based data format (high-performance sequential I/O)

# =========================
# Visualization
# =========================
matplotlib
seaborn
plotly

# =========================
# PyTorch (GPU support)
# Install CPU version via: pip install -r requirements.txt
# For CUDA 11.8: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# =========================
torch>=2.1.0
torchvision
torchaudio

# PyTorch Geometric (GNNs for heterogeneous graphs)
torch-geometric
torch-scatter
torch-sparse

# =========================
# Data batching and time
# =========================
xbatcher
cftime

# =========================
# Deep Learning / Diffusion
# =========================
diffusers==0.36.0
transformers==4.57.6
accelerate==1.12.0
huggingface-hub==0.36.0
safetensors==0.7.0

# =========================
# Configuration & utilities
# =========================
hydra-core==1.3.2
omegaconf==2.3.0
pyyaml
tqdm
requests

# =========================
# Notebooks / interactive
# =========================
ipykernel
ipywidgets

# =========================
# GPU monitoring (optional)
# =========================
nvidia-ml-py

cartopy
```

---

### `setup.py`

```python
"""
Setup script for ST-CDGM package.
Allows installation in development mode with: pip install -e .
"""

from pathlib import Path
from setuptools import setup, find_packages

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, encoding="utf-8") as f:
        install_requires = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]
else:
    install_requires = []

setup(
    name="st-cdgm",
    version="0.1.0",
    description="Spatio-Temporal Causal Diffusion Generative Model",
    author="ST-CDGM Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
```

---

### `.gitignore`

```text
# Python
__pycache__/
.cursor/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
downscaling/
docs/
data/raw
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv/

# IDE
.idea/
.vscode/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Environment variables
.env
.env.local
requirements-windows.txt

# Backup files
*.bak
*.backup
*.tmp

# Large data files (optional - uncomment if you want to ignore NetCDF files)
# data/raw/*.nc
# data/raw/*.zarr

# Logs
*.log
logs/

# OS
Thumbs.db
desktop.ini

!README.md
*.md
```

---

### `.dockerignore`

```text
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/
.venv

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Git
.git/
.gitignore

# Docker
.dockerignore
docker-compose.yml
Dockerfile
*.dockerfile

# Documentation
docs/
*.md
!README.md

# Logs and temporary files
*.log
*.tmp
.cache/
.pytest_cache/
.coverage
htmlcov/

# Data (large files, mount as volume instead)
# data/raw/*.nc  # Uncomment if you don't want to include NetCDF files in image
# models/*.pt    # Uncomment if you don't want to include models in image

# Results
results/
*.png
*.jpg
*.pdf

# OS
Thumbs.db
```

---

### `data/metadata/NorESM2-MM_histupdated_compressed.metadata.json`

```json
{
  "file_info": {
    "filepath": "./NorESM2-MM_histupdated_compressed.nc",
    "filename": "NorESM2-MM_histupdated_compressed.nc",
    "file_size_bytes": 217310777,
    "file_size_mb": 207.24,
    "file_extension": ".nc",
    "file_format": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc"
  },
  "dimensions": {
    "time": {
      "size": 7300,
      "unlimited": true,
      "has_coordinates": true,
      "dtype": "object",
      "shape": [
        7300
      ],
      "attributes": {},
      "min_value": "1986-01-01 00:00:00",
      "max_value": "2005-12-31 00:00:00",
      "mean_value": "1995-12-31 12:00:00",
      "values_sample": [
        "1986-01-01 00:00:00",
        "1986-01-02 00:00:00",
        "1986-01-03 00:00:00",
        "1986-01-04 00:00:00",
        "1986-01-05 00:00:00",
        "1986-01-06 00:00:00",
        "1986-01-07 00:00:00",
        "1986-01-08 00:00:00",
        "1986-01-09 00:00:00",
        "1986-01-10 00:00:00"
      ],
      "netcdf4_id": "Dimension"
    },
    "lat": {
      "size": 23,
      "unlimited": false,
      "has_coordinates": true,
      "dtype": "float64",
      "shape": [
        23
      ],
      "attributes": {
        "standard_name": "latitude",
        "long_name": "latitude",
        "units": "degrees_north",
        "axis": "Y"
      },
      "min_value": -59.38,
      "max_value": -26.380000000000003,
      "mean_value": -42.88,
      "values_sample": [
        -59.38,
        -57.88,
        -56.38,
        -54.88,
        -53.38,
        -51.88,
        -50.38,
        -48.88,
        -47.38,
        -45.88
      ],
      "netcdf4_id": "Dimension"
    },
    "lon": {
      "size": 26,
      "unlimited": false,
      "has_coordinates": true,
      "dtype": "float64",
      "shape": [
        26
      ],
      "attributes": {
        "standard_name": "longitude",
        "long_name": "longitude",
        "units": "degrees_east",
        "axis": "X"
      },
      "min_value": 150.6,
      "max_value": 188.1,
      "mean_value": 169.35000000000002,
      "values_sample": [
        150.6,
        152.1,
        153.6,
        155.1,
        156.6,
        158.1,
        159.6,
        161.1,
        162.6,
        164.1
      ],
      "netcdf4_id": "Dimension"
    }
  },
  "coordinates": {},
  "data_variables": {
    "t_850": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": 251.9374237060547,
        "max": 299.81768798828125,
        "mean": 275.5995788574219,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 6.415265083312988,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "t_500": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": 227.02061462402344,
        "max": 275.2542419433594,
        "mean": 253.26918029785156,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 7.3385396003723145,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "t_250": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": 204.3632049560547,
        "max": 239.75775146484375,
        "mean": 222.84344482421875,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 5.005284786224365,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "u_850": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": -37.91753005981445,
        "max": 39.81140899658203,
        "mean": 7.004383563995361,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 8.753339767456055,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "u_500": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": -30.826129913330078,
        "max": 62.41065216064453,
        "mean": 14.587430000305176,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 10.458207130432129,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "u_250": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": -40.83777618408203,
        "max": 93.4031982421875,
        "mean": 25.20738983154297,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 14.893157005310059,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "v_850": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": -38.89788055419922,
        "max": 42.43639373779297,
        "mean": -0.8492357134819031,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 6.900809288024902,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "v_500": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": -48.62022399902344,
        "max": 49.010337829589844,
        "mean": -0.906661868095398,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 9.331302642822266,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "v_250": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": -65.61503601074219,
        "max": 66.41313934326172,
        "mean": -0.8343009352684021,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 13.10961627960205,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "w_850": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": -0.2237137258052826,
        "max": 0.26198911666870117,
        "mean": -0.0002054092037724331,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 0.012605391442775726,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "w_500": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": -0.38293278217315674,
        "max": 0.48701292276382446,
        "mean": -5.987402346363524e-06,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 0.02132747322320938,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "w_250": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": -0.2414090931415558,
        "max": 0.5248967409133911,
        "mean": 0.0004757193091791123,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 0.016292216256260872,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "q_850": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": 5.875585884496104e-06,
        "max": 0.017141252756118774,
        "mean": 0.004244939424097538,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 0.00196230411529541,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "q_500": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": 3.543378397807828e-06,
        "max": 0.006711352150887251,
        "mean": 0.0006539419409818947,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 0.0006003747694194317,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    },
    "q_250": {
      "dims": [
        "time",
        "lat",
        "lon"
      ],
      "shape": [
        7300,
        23,
        26
      ],
      "dtype": "float32",
      "size": 4365400,
      "nbytes": 17461600,
      "attributes": {},
      "encoding": {
        "dtype": "float32",
        "zlib": true,
        "szip": false,
        "zstd": false,
        "bzip2": false,
        "blosc": false,
        "shuffle": true,
        "complevel": 8,
        "fletcher32": false,
        "contiguous": false,
        "chunksizes": [
          1,
          23,
          26
        ],
        "preferred_chunks": {
          "time": 1,
          "lat": 23,
          "lon": 26
        },
        "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc",
        "original_shape": [
          7300,
          23,
          26
        ],
        "_FillValue": NaN
      },
      "statistics": {
        "min": 1.337454165195595e-07,
        "max": 0.0007597199873998761,
        "mean": 5.339308700058609e-05,
        "min_raw": null,
        "max_raw": null,
        "mean_raw": null,
        "std": 5.7975234085461125e-05,
        "has_nan": false,
        "nan_count": 0
      },
      "chunksizes": [
        1,
        23,
        26
      ],
      "fill_value": NaN,
      "netcdf4_info": {
        "storage": [
          1,
          23,
          26
        ],
        "endianness": "little",
        "all_attributes": {
          "_FillValue": NaN
        },
        "compression": {
          "filters": {
            "zlib": true,
            "szip": false,
            "zstd": false,
            "bzip2": false,
            "blosc": false,
            "shuffle": true,
            "complevel": 8,
            "fletcher32": false
          },
          "is_compressed": true
        },
        "datatype": {
          "name": "float32",
          "is_compound": false,
          "is_enum": false,
          "is_vlen": false,
          "is_opaque": false
        }
      }
    }
  },
  "global_attributes": {},
  "cf_standard_attributes": {},
  "encoding": {
    "unlimited_dims": "{'time'}",
    "source": "C:\\Users\\reall\\Desktop\\climate_data\\NorESM2-MM_histupdated_compressed.nc"
  },
  "file_structure": {
    "has_groups": false,
    "number_of_dimensions": 3,
    "number_of_coordinates": 3,
    "number_of_data_variables": 15,
    "total_variables": 18,
    "unlimited_dimensions": [
      "time"
    ],
    "root_path": "/"
  },
  "statistics": {},
  "cf_conventions": {}
}
```

---

### `data/metadata/NorESM2-MM_histupdated_compressed.metadata.csv`

```csv
variable_name,dims,shape,dtype,size,nbytes,stat_min,stat_max,stat_mean,stat_min_raw,stat_max_raw,stat_mean_raw,stat_std,stat_has_nan,stat_nan_count,encoding_dtype,encoding_zlib,encoding_szip,encoding_zstd,encoding_bzip2,encoding_blosc,encoding_shuffle,encoding_complevel,encoding_fletcher32,encoding_contiguous,encoding_chunksizes,encoding_preferred_chunks,encoding_source,encoding_original_shape,encoding__FillValue,fill_value,chunksizes
t_850,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,251.9374237060547,299.81768798828125,275.5995788574219,,,,6.415265083312988,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
t_500,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,227.02061462402344,275.2542419433594,253.26918029785156,,,,7.3385396003723145,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
t_250,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,204.3632049560547,239.75775146484375,222.84344482421875,,,,5.005284786224365,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
u_850,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,-37.91753005981445,39.81140899658203,7.004383563995361,,,,8.753339767456055,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
u_500,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,-30.826129913330078,62.41065216064453,14.587430000305176,,,,10.458207130432129,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
u_250,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,-40.83777618408203,93.4031982421875,25.20738983154297,,,,14.893157005310059,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
v_850,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,-38.89788055419922,42.43639373779297,-0.8492357134819031,,,,6.900809288024902,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
v_500,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,-48.62022399902344,49.010337829589844,-0.906661868095398,,,,9.331302642822266,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
v_250,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,-65.61503601074219,66.41313934326172,-0.8343009352684021,,,,13.10961627960205,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
w_850,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,-0.2237137258052826,0.26198911666870117,-0.0002054092037724331,,,,0.012605391442775726,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
w_500,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,-0.38293278217315674,0.48701292276382446,-5.987402346363524e-06,,,,0.02132747322320938,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
w_250,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,-0.2414090931415558,0.5248967409133911,0.0004757193091791123,,,,0.016292216256260872,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
q_850,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,5.875585884496104e-06,0.017141252756118774,0.004244939424097538,,,,0.00196230411529541,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
q_500,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,3.543378397807828e-06,0.006711352150887251,0.0006539419409818947,,,,0.0006003747694194317,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
q_250,"['time', 'lat', 'lon']","[7300, 23, 26]",float32,4365400,17461600,1.337454165195595e-07,0.0007597199873998761,5.339308700058609e-05,,,,5.7975234085461125e-05,False,0,float32,True,False,False,False,False,True,8,False,False,"(1, 23, 26)","{""time"": 1, ""lat"": 23, ""lon"": 26}",C:\Users\reall\Desktop\climate_data\NorESM2-MM_histupdated_compressed.nc,"(7300, 23, 26)",,,"(1, 23, 26)"
```

---

### `src/st_cdgm/__init__.py`

```python
"""
ST-CDGM: Spatio-Temporal Causal Diffusion Generative Model

Package principal pour le modèle ST-CDGM.
"""

from .models.causal_rcn import RCNCell, RCNSequenceRunner
from .models.diffusion_decoder import CausalDiffusionDecoder, DiffusionOutput
from .models.intelligible_encoder import IntelligibleVariableEncoder, IntelligibleVariableConfig, SpatialConditioningProjector
from .models.graph_builder import HeteroGraphBuilder
from .data.pipeline import NetCDFDataPipeline, ZarrDataPipeline, ResDiffIterableDataset
from .data.netcdf_utils import NetCDFToDataFrame
from .training.training_loop import (
    train_epoch,
    compute_rapsd_metric_from_batch,
    resolve_train_amp_mode,
)

__all__ = [
    # Models
    "RCNCell",
    "RCNSequenceRunner",
    "CausalDiffusionDecoder",
    "DiffusionOutput",
    "IntelligibleVariableEncoder",
    "IntelligibleVariableConfig",
    "SpatialConditioningProjector",
    "HeteroGraphBuilder",
    # Data
    "NetCDFDataPipeline",
    "ZarrDataPipeline",
    "ResDiffIterableDataset",
    "NetCDFToDataFrame",
    # Training
    "train_epoch",
    "compute_rapsd_metric_from_batch",
    "resolve_train_amp_mode",
]
```

---

### `src/st_cdgm/data/__init__.py`

```python
"""
Modules de gestion des données pour ST-CDGM.
"""

from .pipeline import (
    NetCDFDataPipeline,
    ZarrDataPipeline,
    ResDiffIterableDataset,
    WebDatasetIterableDataset,
)
from .netcdf_utils import NetCDFToDataFrame

__all__ = [
    "NetCDFDataPipeline",
    "ZarrDataPipeline",
    "ResDiffIterableDataset",
    "WebDatasetIterableDataset",
    "NetCDFToDataFrame",
]
```

---

### `src/st_cdgm/data/pipeline.py`

```python
"""
Module 1 - Data pipeline utilities for the ST-CDGM architecture.

This module prepares climate NetCDF datasets for the ST-CDGM pipeline:
  * loading and aligning LR/HR/static datasets,
  * optional normalisation and target-domain transforms,
  * construction of deterministic baselines and residual targets,
  * creation of streaming IterableDataset objects that yield
    ResDiff-style sequences (LR inputs, baselines, residuals, HR truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple
import json

import numpy as np
import pandas as pd
import xarray as xr

from .netcdf_utils import NetCDFToDataFrame

try:
    import cftime
    HAS_CFTIME = True
except ImportError:
    HAS_CFTIME = False

try:  # Optional torch dependency (required for IterableDataset)
    import torch
    from torch import Tensor
    from torch.utils.data import IterableDataset
except ImportError:  # pragma: no cover
    torch = None
    Tensor = None
    IterableDataset = None

try:
    import xbatcher
except ImportError:  # pragma: no cover
    xbatcher = None

try:
    import zarr
    HAS_ZARR = True
except ImportError:  # pragma: no cover
    HAS_ZARR = False
    zarr = None

try:
    import webdataset as wds
    HAS_WEBDATASET = True
except ImportError:  # pragma: no cover
    HAS_WEBDATASET = False
    wds = None

ArrayLike = np.ndarray
TransformFn = Callable[[xr.Dataset], xr.Dataset]


@dataclass
class GridMetadata:
    """Container describing the main dimension names used across datasets."""

    time: str
    lr_lat: str
    lr_lon: str
    hr_lat: str
    hr_lon: str


def _infer_dim(dataset: xr.Dataset, keyword: str) -> str:
    """Infer a dimension/coordinate name containing ``keyword`` (case-insensitive)."""
    keyword = keyword.lower()
    for name in dataset.dims:
        if keyword in name.lower():
            return name
    for name in dataset.coords:
        if keyword in name.lower():
            return name
    raise ValueError(f"Unable to infer dimension for '{keyword}' in dataset {list(dataset.dims)}")


def _ensure_callable_transform(
    transform: Optional[TransformFn | str],
    epsilon: float,
) -> Optional[TransformFn]:
    """Normalise transform specifications to callables."""
    if transform is None:
        return None
    if isinstance(transform, str):
        key = transform.lower()
        if key in {"log", "logarithm"}:
            return lambda ds: xr.apply_ufunc(
                lambda x: np.log(x + epsilon),
                ds,
                keep_attrs=True,
            )
        if key in {"log1p"}:
            return lambda ds: xr.apply_ufunc(
                lambda x: np.log1p(x),
                ds,
                keep_attrs=True,
            )
        raise ValueError(f"Unknown transform identifier '{transform}'.")
    if callable(transform):
        return transform
    raise TypeError("target_transform must be callable or string identifier.")


def _dataset_to_numpy(
    dataset: xr.Dataset,
    time_dim: str,
    lat_dim: str,
    lon_dim: str,
    spatial_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Convert an ``xr.Dataset`` to ``np.ndarray`` with shape (time, channel, lat, lon)."""
    # Convert to array (stacks variables into 'channel' dimension)
    array = dataset.to_array(dim="channel")
    
    # Get available dimensions after to_array
    available_dims = list(array.dims)
    
    # Check if we have the expected dimensions
    missing_dims = []
    if time_dim not in available_dims:
        missing_dims.append(time_dim)
    if lat_dim not in available_dims:
        missing_dims.append(lat_dim)
    if lon_dim not in available_dims:
        missing_dims.append(lon_dim)
    
    if missing_dims:
        # Try to find alternative dimension names
        # Sometimes dimensions might be in coordinates but not in dims
        # Or xbatcher might have renamed them
        
        # Check if dimensions exist as coordinates
        coord_dims = list(dataset.coords.keys())
        
        # Try to infer spatial dimensions from available dims
        spatial_candidates = [d for d in available_dims if d not in [time_dim, "channel"]]
        
        # If we have exactly 2 spatial candidates, use them
        if len(spatial_candidates) == 2:
            # Assume they are lat and lon (order might matter)
            inferred_lat = spatial_candidates[0]
            inferred_lon = spatial_candidates[1]
            lat_dim = inferred_lat
            lon_dim = inferred_lon
        elif "sample" in available_dims and spatial_shape is not None:
            # xbatcher has flattened spatial dims into 'sample'
            # It may have flattened lat, lon, and potentially other dims (like lev)
            lat_size, lon_size = spatial_shape
            
            # Get the current shape and find sample dimension index
            sample_idx = available_dims.index("sample")
            values = array.values
            
            # Calculate expected sample size (just lat * lon)
            expected_sample_size = lat_size * lon_size
            actual_sample_size = values.shape[sample_idx]
            
            # Check if there are additional dimensions that were flattened
            # (e.g., lev, level, depth, etc.)
            extra_dims_in_sample = actual_sample_size // expected_sample_size
            
            if extra_dims_in_sample == 1:
                # Simple case: only lat and lon were flattened
                new_shape = list(values.shape)
                new_shape[sample_idx] = lat_size
                new_shape.insert(sample_idx + 1, lon_size)
                values = values.reshape(new_shape)
                
                # Create new dimension names
                new_dims = list(available_dims)
                new_dims[sample_idx] = lat_dim
                new_dims.insert(sample_idx + 1, lon_dim)
            elif extra_dims_in_sample > 1 and actual_sample_size % expected_sample_size == 0:
                # Complex case: additional dimensions were flattened (e.g., lev)
                # We need to find what extra dimensions exist in the original dataset
                original_dims = set(dataset.dims)
                expected_dims = {time_dim, lat_dim, lon_dim}
                extra_dims_original = original_dims - expected_dims
                
                if extra_dims_original:
                    # Try to find the extra dimension that was flattened
                    # Usually it's 'lev', 'level', 'depth', etc.
                    extra_dim_name = None
                    extra_dim_size = None
                    for dim_name in ['lev', 'level', 'depth', 'z']:
                        if dim_name in dataset.dims:
                            extra_dim_name = dim_name
                            extra_dim_size = dataset.dims[dim_name]
                            break
                    
                    if extra_dim_name and extra_dim_size == extra_dims_in_sample:
                        # Reshape: sample -> extra_dim, lat, lon
                        new_shape = list(values.shape)
                        new_shape[sample_idx] = extra_dim_size
                        new_shape.insert(sample_idx + 1, lat_size)
                        new_shape.insert(sample_idx + 2, lon_size)
                        values = values.reshape(new_shape)
                        
                        # Create new dimension names
                        new_dims = list(available_dims)
                        new_dims[sample_idx] = extra_dim_name
                        new_dims.insert(sample_idx + 1, lat_dim)
                        new_dims.insert(sample_idx + 2, lon_dim)
                        
                        # After reshape, we need to average or select a level
                        # For now, let's average across the extra dimension
                        # (or we could select the first level)
                        extra_dim_idx = new_dims.index(extra_dim_name)
                        values = values.mean(axis=extra_dim_idx)
                        new_dims.pop(extra_dim_idx)
                    else:
                        # Fallback: just reshape assuming the extra dimension
                        new_shape = list(values.shape)
                        new_shape[sample_idx] = extra_dims_in_sample
                        new_shape.insert(sample_idx + 1, lat_size)
                        new_shape.insert(sample_idx + 2, lon_size)
                        values = values.reshape(new_shape)
                        
                        # Average across the first extra dimension
                        values = values.mean(axis=sample_idx)
                        new_shape.pop(sample_idx)
                        
                        # Create new dimension names
                        new_dims = list(available_dims)
                        new_dims[sample_idx] = lat_dim
                        new_dims.insert(sample_idx + 1, lon_dim)
                else:
                    raise ValueError(
                        f"Cannot determine extra dimensions. "
                        f"Expected sample size {expected_sample_size}, got {actual_sample_size}. "
                        f"Ratio: {extra_dims_in_sample}, but no extra dimensions found in dataset."
                    )
            else:
                raise ValueError(
                    f"Cannot reshape 'sample' dimension: expected size {expected_sample_size} "
                    f"(lat={lat_size} * lon={lon_size}), but got {actual_sample_size}. "
                    f"The size is not a multiple of the expected size."
                )
            
            # Recreate the DataArray with new dimensions
            # Build coordinates carefully - only use coords that match the new dimensions exactly
            new_coords = {}
            for dim in new_dims:
                # For time and channel, use coordinates from array if they match
                if dim in array.coords:
                    coord = array.coords[dim]
                    # Only use if it has the correct single dimension
                    if hasattr(coord, 'dims') and coord.dims == (dim,):
                        new_coords[dim] = coord
                # For lat and lon that were reshaped, create simple index-based coordinates
                # Don't try to use dataset.coords as they may have wrong dimensions
                elif dim == lat_dim:
                    new_coords[dim] = np.arange(lat_size)
                elif dim == lon_dim:
                    new_coords[dim] = np.arange(lon_size)
            
            # Create DataArray - xarray will validate coordinates match dimensions
            array = xr.DataArray(values, dims=new_dims, coords=new_coords if new_coords else None)
            # Update available_dims for transpose
            available_dims = new_dims
        elif "sample" in available_dims:
            raise ValueError(
                f"xbatcher has flattened spatial dimensions into 'sample', "
                f"but spatial_shape was not provided. "
                f"Available dimensions: {available_dims}."
            )
        else:
            raise ValueError(
                f"Missing dimensions: {missing_dims}. "
                f"Available dimensions: {available_dims}. "
                f"Dataset coordinates: {coord_dims}. "
                f"Cannot infer spatial dimensions automatically."
            )
    
    # Now transpose to the expected order
    try:
        array = array.transpose(time_dim, "channel", lat_dim, lon_dim)
    except ValueError as e:
        raise ValueError(
            f"Failed to transpose array. "
            f"Available dims: {available_dims}, "
            f"Requested order: ({time_dim}, channel, {lat_dim}, {lon_dim}). "
            f"Error: {e}"
        )
    
    return array.values.astype(np.float32)


class NetCDFDataPipeline:
    """
    High-level data preparation pipeline for ST-CDGM training.

    Parameters
    ----------
    lr_path :
        Path to the low-resolution (LR) dataset (predictors).
    hr_path :
        Path to the high-resolution (HR) ground-truth dataset.
    static_path :
        Optional static fields at HR resolution (topography, land-use, ...).
    seq_len :
        Sequence length (number of time steps) for iterable datasets.
    baseline_strategy :
        Strategy to build deterministic baselines. Options: ``"hr_smoothing"`` (default),
        ``"lr_interp"`` (bilinear upsampling of LR to HR grid).
    baseline_factor :
        Coarsening factor used with ``hr_smoothing`` strategy.
    target_transform :
        Optional transform applied to HR/baseline datasets (callable or "log"/"log1p").
    target_inverse_transform :
        Optional inverse transform callable (used for evaluation/export).
    normalize :
        Whether to normalise LR predictors using per-variable mean/std.
    lr_variables / hr_variables / static_variables :
        Optional variable subsets to select from the respective datasets.
    means_path / stds_path :
        Optional pre-computed statistics for LR normalisation.
    chunks :
        Optional chunk sizes passed to ``xr.open_dataset``.
    """

    def __init__(
        self,
        lr_path: str | Path,
        hr_path: str | Path,
        static_path: Optional[str | Path] = None,
        *,
        seq_len: int = 10,
        baseline_strategy: str = "hr_smoothing",
        baseline_factor: int = 4,
        target_transform: Optional[TransformFn | str] = None,
        target_inverse_transform: Optional[TransformFn] = None,
        normalize: bool = False,
        nan_fill_strategy: str = "zero",
        precipitation_delta: float = 0.01,
        lr_variables: Optional[Sequence[str]] = None,
        hr_variables: Optional[Sequence[str]] = None,
        static_variables: Optional[Sequence[str]] = None,
        means_path: Optional[str | Path] = None,
        stds_path: Optional[str | Path] = None,
        chunks: Optional[Dict[str, int]] = None,
        transform_epsilon: float = 1e-6,
    ) -> None:
        if xbatcher is None:
            raise ImportError("xbatcher is required for ST-CDGM data streaming. Install it via `pip install xbatcher`.")

        self.seq_len = seq_len
        self.baseline_strategy = baseline_strategy
        self.baseline_factor = max(1, baseline_factor)
        self.normalize = normalize
        self.nan_fill_strategy = nan_fill_strategy
        self.precipitation_delta = precipitation_delta
        self.lr_path = Path(lr_path)
        self.hr_path = Path(hr_path)
        self.static_path = Path(static_path) if static_path else None
        self.means_path = Path(means_path) if means_path else None
        self.stds_path = Path(stds_path) if stds_path else None
        self.transform_epsilon = transform_epsilon
        self._chunks = chunks

        self._target_transform = _ensure_callable_transform(target_transform, transform_epsilon)
        self._target_inverse_transform = target_inverse_transform

        # ------------------------------------------------------------------
        # Load datasets (kept both raw + working copies)
        # ------------------------------------------------------------------
        self.lr_dataset_raw = self._open_dataset(self.lr_path)
        self.hr_dataset_raw = self._open_dataset(self.hr_path)
        self.static_dataset = self._open_dataset(self.static_path) if self.static_path else None

        if self.hr_dataset_raw is None:
            raise ValueError("High-resolution dataset is required for ST-CDGM training.")

        # Select variables if requested
        if lr_variables:
            missing = set(lr_variables) - set(self.lr_dataset_raw.data_vars)
            if missing:
                raise KeyError(f"LR variables not found: {missing}")
            self.lr_dataset_raw = self.lr_dataset_raw[lr_variables]
        if hr_variables:
            missing = set(hr_variables) - set(self.hr_dataset_raw.data_vars)
            if missing:
                raise KeyError(f"HR variables not found: {missing}")
            self.hr_dataset_raw = self.hr_dataset_raw[hr_variables]
        if self.static_dataset is not None and static_variables:
            missing = set(static_variables) - set(self.static_dataset.data_vars)
            if missing:
                raise KeyError(f"Static variables not found: {missing}")
            self.static_dataset = self.static_dataset[static_variables]

        # Infer shared dimension names
        self.dims = GridMetadata(
            time=_infer_dim(self.lr_dataset_raw, "time"),
            lr_lat=_infer_dim(self.lr_dataset_raw, "lat"),
            lr_lon=_infer_dim(self.lr_dataset_raw, "lon"),
            hr_lat=_infer_dim(self.hr_dataset_raw, "lat"),
            hr_lon=_infer_dim(self.hr_dataset_raw, "lon"),
        )
        self.grid_metadata = self.dims  # backward compatibility

        # Align datasets along the shared temporal axis
        self.lr_dataset_raw, self.hr_dataset_raw = self._align_time(self.lr_dataset_raw, self.hr_dataset_raw)
        # Load into memory (chunked on failure for CyVerse Data Store / slow HDF reads)
        self.lr_dataset_raw = self._load_dataset_robust(self.lr_dataset_raw, "LR", self.lr_path)
        self.hr_dataset_raw = self._load_dataset_robust(self.hr_dataset_raw, "HR", self.hr_path)

        # Clean NaN values from datasets
        self.lr_dataset_raw = self._clean_nan_values(self.lr_dataset_raw, self.nan_fill_strategy)
        self.hr_dataset_raw = self._clean_nan_values(self.hr_dataset_raw, self.nan_fill_strategy)
        
        if self.static_dataset is not None:
            self.static_dataset = self._load_dataset_robust(self.static_dataset, "static", self.static_path)

        # Normalise LR predictors if requested
        if self.normalize:
            self.lr_dataset_normalised, self.lr_stats = self._normalise_lr_dataset(self.lr_dataset_raw)
        else:
            self.lr_dataset_normalised = self.lr_dataset_raw
            self.lr_stats: Dict[str, xr.Dataset] = {}

        # Prepare deterministic baselines and residual ground-truth
        self.baseline_raw = self._compute_baseline()
        self.hr_prepared = self._apply_target_transform(self.hr_dataset_raw)
        self.baseline_prepared = self._apply_target_transform(self.baseline_raw)
        self.residual_dataset = self.hr_prepared - self.baseline_prepared

        # Convenience handles used by downstream modules
        self.lr_dataset = self.lr_dataset_normalised
        self.hr_dataset = self.hr_prepared

        self.static_tensor_np: Optional[np.ndarray] = self._prepare_static_tensor()
        self.static_tensor_torch: Optional[Tensor] = (
            torch.from_numpy(self.static_tensor_np) if (torch is not None and self.static_tensor_np is not None) else None
        )

    # ------------------------------------------------------------------
    # Dataset opening & alignment helpers
    # ------------------------------------------------------------------
    def _open_dataset(self, path: Optional[Path]) -> Optional[xr.Dataset]:
        if path is None:
            return None
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        path_str = str(path)
        # Try engines in order (fixes "did not find a match" on CyVerse/remote paths)
        for engine in ("netcdf4", "scipy", "h5netcdf"):
            try:
                kwargs = {"chunks": self._chunks} if self._chunks and engine != "scipy" else {}
                return xr.open_dataset(path_str, engine=engine, **kwargs)
            except Exception as e:
                if engine == "h5netcdf":
                    raise
                continue
        raise RuntimeError(f"Could not open {path_str} with any NetCDF engine")

    def _load_dataset_robust(
        self, ds: xr.Dataset, name: str = "dataset", path: Optional[Path] = None
    ) -> xr.Dataset:
        """Load xarray dataset into memory. On HDF error (CyVerse Data Store), retry with alternate engines."""
        src = path
        if src is None and getattr(ds, "encoding", None):
            src = ds.encoding.get("source") if isinstance(ds.encoding, dict) else None
        if isinstance(src, list):
            src = src[0] if len(src) == 1 else None
        path = Path(src) if src else None
        try:
            return ds.load()
        except RuntimeError as e:
            err_str = str(e)
            if "HDF" not in err_str and "netCDF" not in err_str.lower():
                raise
            if path is None:
                raise RuntimeError(
                    f"Failed to load {name}: {e}. Path unknown (multi-file?). "
                    "Try: 1) Copy data to local disk (scripts/sync_datastore.py), "
                    "2) Use Zarr format (ops/preprocess_to_zarr.py)."
                ) from e
            if not path.exists():
                raise RuntimeError(
                    f"Failed to load {name}: {e}. Path not found: {path}. "
                    "Try: 1) Copy data to local disk (scripts/sync_datastore.py), "
                    "2) Use Zarr format (ops/preprocess_to_zarr.py)."
                ) from e
            # Retry with HDF5-capable engines only. Do not use scipy here: NetCDF4/HDF5 files
            # are not NetCDF3; scipy fails with a misleading "install netcdf4" message.
            try:
                ds.close()
            except Exception:
                pass
            path_str = str(path)
            time_dim = self.dims.time if (self.dims and hasattr(self.dims, "time")) else "time"
            last_err: Optional[BaseException] = None
            for engine in ("h5netcdf", "netcdf4"):
                ds_new = None
                try:
                    ds_new = xr.open_dataset(path_str, engine=engine)
                    vars_in_ds = [v for v in ds.data_vars if v in ds_new]
                    ds_new = ds_new[vars_in_ds]
                    if time_dim in ds.dims and time_dim in ds_new.dims and ds.dims[time_dim] != ds_new.dims.get(time_dim, 0):
                        ds_new = ds_new.reindex({time_dim: ds[time_dim]}, method="nearest")
                    result = ds_new.load()
                    ds_new.close()
                    return result
                except Exception as e2:
                    last_err = e2
                    if ds_new is not None:
                        try:
                            ds_new.close()
                        except Exception:
                            pass
                    continue
            raise RuntimeError(
                f"Failed to load {name}: {e}. Retries with h5netcdf and netcdf4 also failed"
                + (f" (last: {last_err})" if last_err else "")
                + ". Often caused by NFS/Data Store I/O or a truncated file. "
                "Try: 1) Copy the .nc to local fast disk and point paths there (scripts/sync_datastore.py), "
                "2) Verify the file (e.g. ncdump -h, file size), "
                "3) Use Zarr (ops/preprocess_to_zarr.py)."
            ) from (last_err if last_err else e)

    def _convert_cftime_to_datetime(self, time_values):
        """
        Convert cftime objects to pandas datetime, handling various calendar types.
        
        Args:
            time_values: Array of time values (could be cftime, datetime64, or other)
            
        Returns:
            pandas Index of datetime values
        """
        # Check if we have cftime objects
        if HAS_CFTIME and len(time_values) > 0 and isinstance(time_values[0], cftime.datetime):
            # Convert cftime to pandas datetime
            # Use the num2date approach for consistency
            datetime_list = []
            for t in time_values:
                # Convert cftime to standard datetime
                # cftime objects have year, month, day, hour, minute, second attributes
                try:
                    dt = pd.Timestamp(
                        year=t.year,
                        month=t.month,
                        day=t.day,
                        hour=t.hour,
                        minute=t.minute,
                        second=t.second
                    )
                    datetime_list.append(dt)
                except (ValueError, AttributeError):
                    # Fallback: convert to string and let pandas parse it
                    datetime_list.append(pd.Timestamp(str(t)))
            return pd.Index(datetime_list)
        else:
            # Standard datetime conversion
            try:
                return pd.Index(pd.to_datetime(time_values))
            except Exception:
                # Fallback: convert to string first
                return pd.Index(pd.to_datetime([str(t) for t in time_values]))

    def _align_time(self, lr_ds: xr.Dataset, hr_ds: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
        # Convert time coordinates to comparable datetime format
        lr_times = self._convert_cftime_to_datetime(lr_ds[self.dims.time].values)
        hr_times = self._convert_cftime_to_datetime(hr_ds[self.dims.time].values)
        
        # Find common times
        common_times = lr_times.intersection(hr_times)
        if common_times.empty:
            raise ValueError("No overlapping timestamps between LR and HR datasets.")
        
        # Get indices of common times in original datasets
        lr_indices = [i for i, t in enumerate(lr_times) if t in common_times]
        hr_indices = [i for i, t in enumerate(hr_times) if t in common_times]
        
        # Select by integer index instead of coordinate value
        lr_aligned = lr_ds.isel({self.dims.time: lr_indices})
        hr_aligned = hr_ds.isel({self.dims.time: hr_indices})
        
        return lr_aligned, hr_aligned

    #TODO: recuperer strategy depuis config.yml
    def _clean_nan_values(self, dataset: xr.Dataset, strategy: str = "zero") -> xr.Dataset:
        """
        Nettoie les valeurs NaN du dataset selon la stratégie spécifiée.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset à nettoyer
        strategy : str
            Stratégie de nettoyage : "zero" (remplacer par 0), "mean" (moyenne spatiale),
            ou "interpolate" (interpolation temporelle)
        
        Returns
        -------
        xr.Dataset
            Dataset nettoyé
        """
        if strategy == "zero":
            return dataset.fillna(0.0)
        elif strategy == "mean":
            # Remplacer par la moyenne spatiale (dimensions spatiales selon LR ou HR)
            spatial_dims = [self.dims.hr_lat, self.dims.hr_lon] if self.dims.hr_lat in dataset.dims else [self.dims.lr_lat, self.dims.lr_lon]
            return dataset.fillna(dataset.mean(dim=spatial_dims))
        elif strategy == "interpolate":
            return dataset.interpolate_na(dim=self.dims.time, method="linear")
        else:
            raise ValueError(f"Unknown nan_fill_strategy: {strategy}. Must be 'zero', 'mean', or 'interpolate'")

    def _normalise_lr_dataset(self, dataset: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, xr.Dataset]]:
        if self.means_path and self.stds_path:
            means = xr.open_dataset(self.means_path)
            stds = xr.open_dataset(self.stds_path)
        else:
            # Utiliser skipna=True pour ignorer les NaN dans le calcul des statistiques
            means = dataset.mean(dim=self.dims.time, skipna=True, keep_attrs=True)
            stds = dataset.std(dim=self.dims.time, skipna=True, keep_attrs=True)
        
        # Protection contre division par zéro avec epsilon
        epsilon = 1e-6
        stds = stds.where(stds > epsilon, other=1.0)
        normalised = (dataset - means) / stds
        
        # Remplacer les NaN résiduels par 0 (après normalisation)
        normalised = normalised.fillna(0.0)
        
        return normalised, {"mean": means, "std": stds}

    def _compute_baseline(self) -> xr.Dataset:
        if self.baseline_strategy == "lr_interp":
            mapping = {
                self.dims.lr_lat: self.hr_dataset_raw[self.dims.hr_lat],
                self.dims.lr_lon: self.hr_dataset_raw[self.dims.hr_lon],
            }
            baseline = self.lr_dataset_raw.interp(mapping, method="linear")
            baseline = baseline.rename({self.dims.lr_lat: self.dims.hr_lat, self.dims.lr_lon: self.dims.hr_lon})
            return baseline

        if self.baseline_strategy == "hr_smoothing":
            coarsen_kwargs = {
                self.dims.hr_lat: self.baseline_factor,
                self.dims.hr_lon: self.baseline_factor,
            }
            smoothed = self.hr_dataset_raw.coarsen(coarsen_kwargs, boundary="trim").mean(keep_attrs=True)
            baseline = smoothed.interp(
                {
                    self.dims.hr_lat: self.hr_dataset_raw[self.dims.hr_lat],
                    self.dims.hr_lon: self.hr_dataset_raw[self.dims.hr_lon],
                },
                method="linear",
            )
            return baseline

        raise ValueError(f"Unsupported baseline_strategy '{self.baseline_strategy}'.")

    def _detect_precipitation_vars(self, dataset: xr.Dataset) -> list[str]:
        """
        Détecte les variables de précipitation dans le dataset.
        
        Parameters
        ----------
        dataset : xr.Dataset
            Dataset à analyser
        
        Returns
        -------
        list[str]
            Liste des noms de variables de précipitation détectées
        """
        precip_keywords = ['pr', 'precip', 'precipitation', 'rain']
        return [var for var in dataset.data_vars 
                if any(kw in var.lower() for kw in precip_keywords)]

    def _apply_target_transform(self, dataset: xr.Dataset) -> xr.Dataset:
        """
        Applique la transformation cible avec détection automatique des précipitations.
        
        Si target_transform est None et que des variables de précipitation sont détectées,
        applique automatiquement log1p avec precipitation_delta.
        """
        if self._target_transform is not None:
            transformed = self._target_transform(dataset)
            if not isinstance(transformed, xr.Dataset):
                raise TypeError("target_transform must return an xarray.Dataset.")
            return transformed
        
        # Auto-détection : si précipitations et pas de transform explicite, utiliser log1p
        precip_vars = self._detect_precipitation_vars(dataset)
        if precip_vars:
            # Appliquer log1p uniquement aux variables de précipitation
            result = dataset.copy()
            for var in precip_vars:
                result[var] = xr.apply_ufunc(
                    lambda x: np.log1p(x + self.precipitation_delta),
                    dataset[var],
                    keep_attrs=True
                )
            return result
        
        return dataset

    def _prepare_static_tensor(self) -> Optional[np.ndarray]:
        if self.static_dataset is None:
            return None

        # Create a working copy
        ds = self.static_dataset.copy()

        # 1. Drop coordinate bounds if present as variables (common in climate data)
        # Find variables that look like bounds (containing 'bnds' or 'bounds')
        drop_vars = [v for v in ds.data_vars if "bnds" in str(v) or "bounds" in str(v)]
        if drop_vars:
            ds = ds.drop_vars(drop_vars)

        # 2. Squeeze singleton dimensions (e.g. depth=1, time=1)
        ds = ds.squeeze(drop=True)

        # 3. Handle potential remaining extra dimensions (e.g. depth > 1)
        # We expect only (lat, lon) to remain for static 2D fields
        expected_dims = {self.dims.hr_lat, self.dims.hr_lon}
        # Note: ds.dims might include 'time' if it wasn't squeezed out (unlikely for static)
        
        # Identify extra dimensions that are NOT lat/lon
        extra_dims = set(ds.dims) - expected_dims
        
        if extra_dims:
            # If we still have extra dims, selecting the first index is a reasonable default 
            # for "static predictors" which are typically 2D maps.
            # (e.g. selecting surface level if depth is present)
            isel_kwargs = {dim: 0 for dim in extra_dims}
            ds = ds.isel(**isel_kwargs, drop=True)

        # 4. Convert to array (stacks variables into 'channel' dimension)
        static_array = ds.to_array(dim="channel")
        
        # 5. Transpose to ensure (channel, lat, lon) order
        # We use ... to handle any edge cases, but at this point we should have 3 dims
        try:
            static_array = static_array.transpose("channel", self.dims.hr_lat, self.dims.hr_lon)
        except ValueError:
            # Fallback if strict transpose fails (should be covered by logic above, but for safety)
             static_array = static_array.transpose("channel", ...)

        return static_array.values.astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_lr_dataset(self) -> xr.Dataset:
        return self.lr_dataset

    def get_lr_stats(self) -> Dict[str, xr.Dataset]:
        return self.lr_stats

    def get_hr_dataset(self) -> xr.Dataset:
        return self.hr_dataset

    def get_baseline_dataset(self) -> xr.Dataset:
        return self.baseline_prepared

    def get_residual_dataset(self) -> xr.Dataset:
        return self.residual_dataset

    def get_static_dataset(self) -> Optional[xr.Dataset]:
        return self.static_dataset

    def get_target_inverse_transform(self) -> Optional[TransformFn]:
        return self._target_inverse_transform

    # ------------------------------------------------------------------
    # Metadata export helpers
    # ------------------------------------------------------------------
    def export_lr_metadata_to_json(self, output_file: Optional[str | Path] = None) -> str:
        converter = NetCDFToDataFrame(self.lr_path)
        return converter.export_metadata_to_json(str(output_file) if output_file else None)

    def export_lr_metadata_to_csv(self, output_file: Optional[str | Path] = None) -> str:
        converter = NetCDFToDataFrame(self.lr_path)
        return converter.export_metadata_to_csv(str(output_file) if output_file else None)

    def export_hr_metadata_to_json(self, output_file: Optional[str | Path] = None) -> str:
        converter = NetCDFToDataFrame(self.hr_path)
        return converter.export_metadata_to_json(str(output_file) if output_file else None)

    def export_hr_metadata_to_csv(self, output_file: Optional[str | Path] = None) -> str:
        converter = NetCDFToDataFrame(self.hr_path)
        return converter.export_metadata_to_csv(str(output_file) if output_file else None)

    # ------------------------------------------------------------------
    # IterableDataset factory
    # ------------------------------------------------------------------
    def build_sequence_dataset(
        self,
        *,
        seq_len: Optional[int] = None,
        stride: int = 1,
        drop_last: bool = True,
        as_torch: bool = True,
    ) -> "ResDiffIterableDataset":
        seq_len = seq_len or self.seq_len
        if as_torch and torch is None:
            raise ImportError("Torch is required to obtain PyTorch tensors. Install it via `pip install torch`.")
        if IterableDataset is None:
            raise ImportError("Torch IterableDataset is required. Install PyTorch to continue.")
        return ResDiffIterableDataset(
            lr_dataset=self.lr_dataset,
            baseline_dataset=self.baseline_prepared,
            residual_dataset=self.residual_dataset,
            hr_dataset=self.hr_dataset,
            static_tensor_np=self.static_tensor_np,
            static_tensor_torch=self.static_tensor_torch,
            dims=self.dims,
            seq_len=seq_len,
            stride=max(1, stride),
            drop_last=drop_last,
            as_torch=as_torch,
        )


class ResDiffIterableDataset(IterableDataset):
    """
    Streaming dataset yielding ResDiff-style batches for ST-CDGM training.

    Each yielded sample is a dictionary containing:
        * ``lr``        : (seq_len, channels_lr, lat_lr, lon_lr)
        * ``baseline``  : (seq_len, channels_hr, lat_hr, lon_hr)
        * ``residual``  : (seq_len, channels_hr, lat_hr, lon_hr)
        * ``hr``        : (seq_len, channels_hr, lat_hr, lon_hr)
        * ``valid_mask``: (seq_len, lat_hr, lon_hr) float32, 1.0 où HR/résidu sont finis (Phase 5.5)
        * ``static``    : (channels_static, lat_hr, lon_hr)  (optional)
        * ``time``      : sequence of timestamps
    """

    def __init__(
        self,
        *,
        lr_dataset: xr.Dataset,
        baseline_dataset: xr.Dataset,
        residual_dataset: xr.Dataset,
        hr_dataset: xr.Dataset,
        static_tensor_np: Optional[np.ndarray],
        static_tensor_torch: Optional[Tensor],
        dims: GridMetadata,
        seq_len: int,
        stride: int,
        drop_last: bool,
        as_torch: bool,
    ) -> None:
        if IterableDataset is None:
            raise ImportError("PyTorch IterableDataset unavailable. Install torch to use ResDiffIterableDataset.")

        self.seq_len = seq_len
        self.stride = max(1, stride)
        self.drop_last = drop_last
        self.as_torch = as_torch
        self.dims = dims
        self.static_tensor_np = static_tensor_np
        self.static_tensor_torch = static_tensor_torch

        overlap = max(seq_len - self.stride, 0)
        batch_kwargs = dict(
            input_dims={dims.time: seq_len},
            input_overlap={dims.time: overlap},
            preload_batch=False,
        )
        # Store original dataset shapes for potential reshaping if xbatcher flattens spatial dims
        self.lr_spatial_shape = (lr_dataset.sizes[dims.lr_lat], lr_dataset.sizes[dims.lr_lon])
        self.hr_spatial_shape = (hr_dataset.sizes[dims.hr_lat], hr_dataset.sizes[dims.hr_lon])
        
        self.lr_gen = xbatcher.BatchGenerator(lr_dataset, **batch_kwargs)
        self.baseline_gen = xbatcher.BatchGenerator(baseline_dataset, **batch_kwargs)
        self.residual_gen = xbatcher.BatchGenerator(residual_dataset, **batch_kwargs)
        self.hr_gen = xbatcher.BatchGenerator(hr_dataset, **batch_kwargs)

    def __iter__(self) -> Iterable[Dict[str, object]]:
        for lr_window, baseline_window, residual_window, hr_window in zip(
            self.lr_gen, self.baseline_gen, self.residual_gen, self.hr_gen
        ):
            if not self._window_has_required_length(lr_window):
                if self.drop_last:
                    continue
            yield self._format_sample(lr_window, baseline_window, residual_window, hr_window)

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _window_has_required_length(self, window: xr.Dataset) -> bool:
        return window.sizes.get(self.dims.time, 0) == self.seq_len

    def _format_sample(
        self,
        lr_window: xr.Dataset,
        baseline_window: xr.Dataset,
        residual_window: xr.Dataset,
        hr_window: xr.Dataset,
    ) -> Dict[str, object]:
        lr_np = _dataset_to_numpy(
            lr_window, self.dims.time, self.dims.lr_lat, self.dims.lr_lon,
            spatial_shape=self.lr_spatial_shape
        )
        baseline_np = _dataset_to_numpy(
            baseline_window, self.dims.time, self.dims.hr_lat, self.dims.hr_lon,
            spatial_shape=self.hr_spatial_shape
        )
        residual_np = _dataset_to_numpy(
            residual_window, self.dims.time, self.dims.hr_lat, self.dims.hr_lon,
            spatial_shape=self.hr_spatial_shape
        )
        hr_np = _dataset_to_numpy(
            hr_window, self.dims.time, self.dims.hr_lat, self.dims.hr_lon,
            spatial_shape=self.hr_spatial_shape
        )
        valid_np = (
            np.isfinite(residual_np).all(axis=1) & np.isfinite(hr_np).all(axis=1)
        ).astype(np.float32)

        if self.as_torch and torch is not None:
            sample = {
                "lr": torch.from_numpy(lr_np),
                "baseline": torch.from_numpy(baseline_np),
                "residual": torch.from_numpy(residual_np),
                "hr": torch.from_numpy(hr_np),
                "valid_mask": torch.from_numpy(valid_np),
                "time": lr_window[self.dims.time].values,
            }
            if self.static_tensor_torch is not None:
                sample["static"] = self.static_tensor_torch
        else:
            sample = {
                "lr": lr_np,
                "baseline": baseline_np,
                "residual": residual_np,
                "hr": hr_np,
                "valid_mask": valid_np,
                "time": lr_window[self.dims.time].values,
            }
            if self.static_tensor_np is not None:
                sample["static"] = self.static_tensor_np
        return sample


class ZarrDataPipeline:
    """
    High-level data preparation pipeline for ST-CDGM training using pre-processed Zarr data.
    
    This class reads pre-processed Zarr datasets that have already been transformed
    (normalized, baseline computed, residuals calculated) and creates IterableDatasets
    for training.
    
    Parameters
    ----------
    zarr_dir :
        Directory containing the pre-processed Zarr datasets (lr.zarr, hr.zarr,
        baseline.zarr, residual.zarr, static.zarr, metadata.json).
    """
    
    def __init__(
        self,
        zarr_dir: str | Path,
    ) -> None:
        if not HAS_ZARR:
            raise ImportError(
                "Zarr support is not available. Install zarr via `pip install zarr`."
            )
        
        zarr_dir = Path(zarr_dir)
        if not zarr_dir.exists():
            raise ValueError(f"Zarr directory does not exist: {zarr_dir}")
        
        # Load metadata
        metadata_path = zarr_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        self.zarr_dir = zarr_dir
        self.seq_len = metadata.get("seq_len", 10)
        
        # Load dimension metadata
        dims_dict = metadata.get("dims", {})
        self.dims = GridMetadata(
            time=dims_dict.get("time", "time"),
            lr_lat=dims_dict.get("lr_lat", "lat"),
            lr_lon=dims_dict.get("lr_lon", "lon"),
            hr_lat=dims_dict.get("hr_lat", "lat"),
            hr_lon=dims_dict.get("hr_lon", "lon"),
        )
        
        # Load Zarr datasets
        self.lr_dataset = xr.open_zarr(zarr_dir / "lr.zarr")
        self.hr_dataset = xr.open_zarr(zarr_dir / "hr.zarr")
        self.baseline_prepared = xr.open_zarr(zarr_dir / "baseline.zarr")
        self.residual_dataset = xr.open_zarr(zarr_dir / "residual.zarr")
        
        static_zarr_path = zarr_dir / "static.zarr"
        self.static_dataset = xr.open_zarr(static_zarr_path) if static_zarr_path.exists() else None
        
        # Prepare static tensor
        self.static_tensor_np: Optional[np.ndarray] = self._prepare_static_tensor()
        self.static_tensor_torch: Optional[Tensor] = (
            torch.from_numpy(self.static_tensor_np) if (torch is not None and self.static_tensor_np is not None) else None
        )
    
    def _prepare_static_tensor(self) -> Optional[np.ndarray]:
        """Prepare static tensor from static dataset (same logic as NetCDFDataPipeline)."""
        if self.static_dataset is None:
            return None
        
        # Create a working copy
        ds = self.static_dataset.copy()
        
        # 1. Drop coordinate bounds if present as variables
        drop_vars = [v for v in ds.data_vars if "bnds" in str(v) or "bounds" in str(v)]
        if drop_vars:
            ds = ds.drop_vars(drop_vars)
        
        # 2. Squeeze singleton dimensions
        ds = ds.squeeze(drop=True)
        
        # 3. Handle extra dimensions (expect only lat, lon)
        expected_dims = {self.dims.hr_lat, self.dims.hr_lon}
        extra_dims = set(ds.dims) - expected_dims
        
        if extra_dims:
            isel_kwargs = {dim: 0 for dim in extra_dims}
            ds = ds.isel(**isel_kwargs, drop=True)
        
        # 4. Convert to array and transpose
        static_array = ds.to_array(dim="channel")
        try:
            static_array = static_array.transpose("channel", self.dims.hr_lat, self.dims.hr_lon)
        except ValueError:
            static_array = static_array.transpose("channel", ...)
        
        return static_array.values.astype(np.float32)
    
    def build_sequence_dataset(
        self,
        *,
        seq_len: Optional[int] = None,
        stride: int = 1,
        drop_last: bool = True,
        as_torch: bool = True,
    ) -> "ResDiffIterableDataset":
        """
        Build an IterableDataset for training.
        
        Parameters
        ----------
        seq_len :
            Sequence length (defaults to the value from metadata).
        stride :
            Stride for sequence generation.
        drop_last :
            Whether to drop the last incomplete sequence.
        as_torch :
            Whether to return PyTorch tensors.
        
        Returns
        -------
        ResDiffIterableDataset
            IterableDataset yielding ResDiff-style batches.
        """
        seq_len = seq_len or self.seq_len
        if as_torch and torch is None:
            raise ImportError("Torch is required to obtain PyTorch tensors. Install it via `pip install torch`.")
        if IterableDataset is None:
            raise ImportError("Torch IterableDataset is required. Install PyTorch to continue.")
        return ResDiffIterableDataset(
            lr_dataset=self.lr_dataset,
            baseline_dataset=self.baseline_prepared,
            residual_dataset=self.residual_dataset,
            hr_dataset=self.hr_dataset,
            static_tensor_np=self.static_tensor_np,
            static_tensor_torch=self.static_tensor_torch,
            dims=self.dims,
            seq_len=seq_len,
            stride=max(1, stride),
            drop_last=drop_last,
            as_torch=as_torch,
        )
    
    def get_lr_dataset(self) -> xr.Dataset:
        """Return the low-resolution dataset."""
        return self.lr_dataset
    
    def get_hr_dataset(self) -> xr.Dataset:
        """Return the high-resolution dataset."""
        return self.hr_dataset
    
    def get_baseline_dataset(self) -> xr.Dataset:
        """Return the baseline dataset."""
        return self.baseline_prepared
    
    def get_residual_dataset(self) -> xr.Dataset:
        """Return the residual dataset."""
        return self.residual_dataset
    
    def get_static_dataset(self) -> Optional[xr.Dataset]:
        """Return the static dataset."""
        return self.static_dataset


class WebDatasetDataPipeline:
    """
    High-level data preparation pipeline for ST-CDGM training using pre-processed WebDataset shards.
    
    This class reads pre-processed TAR shards that have already been transformed
    (normalized, baseline computed, residuals calculated) and creates IterableDatasets
    for training with high I/O performance.
    
    Parameters
    ----------
    shard_dir :
        Directory containing the pre-processed shard files (*.tar) and metadata.json.
    shuffle :
        Whether to shuffle shards and samples (default: False for deterministic training).
    """
    
    def __init__(
        self,
        shard_dir: str | Path,
        *,
        shuffle: bool = False,
        shardshuffle: int = 100,
        shuffle_buffer_size: int = 1000,
    ) -> None:
        if not HAS_WEBDATASET:
            raise ImportError(
                "WebDataset support is not available. Install webdataset via `pip install webdataset`."
            )
        
        shard_dir = Path(shard_dir)
        if not shard_dir.exists():
            raise ValueError(f"Shard directory does not exist: {shard_dir}")
        
        # Load metadata
        metadata_path = shard_dir / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")
        
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        self.shard_dir = shard_dir
        self.seq_len = metadata.get("seq_len", 10)
        self.shuffle = shuffle
        self.shardshuffle = shardshuffle
        self.shuffle_buffer_size = shuffle_buffer_size
        
        # Load dimension metadata
        dims_dict = metadata.get("dims", {})
        self.dims = GridMetadata(
            time=dims_dict.get("time", "time"),
            lr_lat=dims_dict.get("lr_lat", "lat"),
            lr_lon=dims_dict.get("lr_lon", "lon"),
            hr_lat=dims_dict.get("hr_lat", "lat"),
            hr_lon=dims_dict.get("hr_lon", "lon"),
        )
        
        # Shard pattern (all .tar files in the directory)
        self.shard_pattern = str(shard_dir / "*.tar")
        
        # Static dataset is embedded in shards (not separate)
        self.static_dataset = None
    
    def build_sequence_dataset(
        self,
        *,
        seq_len: Optional[int] = None,
        stride: int = 1,
        drop_last: bool = True,
        as_torch: bool = True,
    ) -> "WebDatasetIterableDataset":
        """
        Build an IterableDataset for training from WebDataset shards.
        
        Parameters
        ----------
        seq_len :
            Sequence length (ignored, read from metadata).
        stride :
            Stride (ignored, shards contain pre-generated sequences).
        drop_last :
            Whether to drop incomplete sequences (ignored, shards contain complete sequences).
        as_torch :
            Whether to return PyTorch tensors (always True for WebDataset).
        
        Returns
        -------
        WebDatasetIterableDataset
            IterableDataset yielding ResDiff-style batches.
        """
        return WebDatasetIterableDataset(
            shard_pattern=self.shard_pattern,
            metadata_path=self.shard_dir / "metadata.json",
            shuffle=self.shuffle,
            shardshuffle=self.shardshuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
        )
    
    def get_lr_dataset(self) -> None:
        """Not available for WebDataset (data is in shards)."""
        return None
    
    def get_hr_dataset(self) -> None:
        """Not available for WebDataset (data is in shards)."""
        return None
    
    def get_baseline_dataset(self) -> None:
        """Not available for WebDataset (data is in shards)."""
        return None
    
    def get_residual_dataset(self) -> None:
        """Not available for WebDataset (data is in shards)."""
        return None
    
    def get_static_dataset(self) -> Optional[xr.Dataset]:
        """Return None (static data is embedded in shards)."""
        return None


class WebDatasetIterableDataset(IterableDataset):
    """
    Phase B3: Streaming dataset from WebDataset TAR shards.
    
    Reads samples from pre-processed TAR shards (created by preprocess_to_shards.py).
    Provides 5-10x better throughput than Zarr for sequential reading patterns.
    
    Each sample in the shard contains:
        * ``lr.pt``      : LR tensor (seq_len, channels_lr, lat_lr, lon_lr)
        * ``baseline.pt``: Baseline tensor (seq_len, channels_hr, lat_hr, lon_hr)
        * ``residual.pt``: Residual tensor (seq_len, channels_hr, lat_hr, lon_hr)
        * ``hr.pt``      : HR tensor (seq_len, channels_hr, lat_hr, lon_hr)
        * ``static.pt``  : Static tensor (channels_static, lat_hr, lon_hr) (optional)
        * ``time.json``  : Time metadata (JSON list of timestamps)
    """
    
    def __init__(
        self,
        shard_pattern: str | Path,
        *,
        metadata_path: Optional[str | Path] = None,
        shuffle: bool = False,
        shardshuffle: int = 100,
        shuffle_buffer_size: int = 1000,
    ) -> None:
        if IterableDataset is None:
            raise ImportError(
                "PyTorch IterableDataset unavailable. Install torch to use WebDatasetIterableDataset."
            )
        if not HAS_WEBDATASET:
            raise ImportError(
                "WebDataset is not installed. Install via: pip install webdataset"
            )
        
        self.shard_pattern = str(shard_pattern)
        self.metadata_path = Path(metadata_path) if metadata_path else None
        
        # Load metadata if provided
        self.metadata = {}
        if self.metadata_path and self.metadata_path.exists():
            import json
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        
        # Create WebDataset pipeline
        # Pattern: "data_{000000..000999}.tar" or "data_*.tar"
        dataset = wds.WebDataset(self.shard_pattern)
        
        # Shuffle shards if requested
        if shuffle:
            dataset = dataset.shuffle(shardshuffle)
        
        # Decode tensors and JSON
        dataset = dataset.decode("torch")  # Decode .pt files as PyTorch tensors
        
        # Map function to format samples
        dataset = dataset.map(self._format_sample)
        
        # Shuffle samples within buffer if requested
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size)
        
        self.dataset = dataset
    
    def _format_sample(self, sample: Dict) -> Dict[str, object]:
        """
        Format WebDataset sample to match ResDiffIterableDataset output format.
        """
        formatted = {}
        
        # Extract tensors (already decoded by WebDataset)
        if "lr.pt" in sample:
            formatted["lr"] = sample["lr.pt"]
        if "baseline.pt" in sample:
            formatted["baseline"] = sample["baseline.pt"]
        if "residual.pt" in sample:
            formatted["residual"] = sample["residual.pt"]
        if "hr.pt" in sample:
            formatted["hr"] = sample["hr.pt"]
        if "static.pt" in sample:
            formatted["static"] = sample["static.pt"]
        
        # Decode time metadata
        if "time.json" in sample:
            import json
            time_data = sample["time.json"]
            if isinstance(time_data, bytes):
                time_data = json.loads(time_data.decode('utf-8'))
            formatted["time"] = time_data
        
        return formatted
    
    def __iter__(self) -> Iterable[Dict[str, object]]:
        """Iterate over samples in the WebDataset."""
        return iter(self.dataset)
```

---

### `src/st_cdgm/data/netcdf_utils.py`

```python
"""
Programme pour transformer des fichiers NetCDF (.nc) en DataFrame pandas
Inspiré du code du projet downscaling
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
from dask.diagnostics import ProgressBar
from typing import Union, List, Optional, Dict, Any
import json
import os
from datetime import datetime

# Import netCDF4 pour accès direct aux métadonnées brutes
try:
    import netCDF4
    NETCDF4_AVAILABLE = True
except ImportError:
    NETCDF4_AVAILABLE = False


class NetCDFToDataFrame:
    """
    Classe pour convertir des fichiers NetCDF climatiques en DataFrame pandas
    """

    def __init__(self, filepath: Union[str, Path],
                 chunks: Optional[Dict[str, int]] = None,
                 load_in_memory: bool = False):
        """
        Initialise le convertisseur avec un fichier NetCDF

        Parameters:
        -----------
        filepath : str ou Path
            Chemin vers le fichier NetCDF
        chunks : dict, optional
            Dictionnaire pour le chargement en chunks (ex: {"time": 1000})
            Utile pour les gros fichiers
        load_in_memory : bool
            Si True, charge toutes les données en mémoire
        """
        self.filepath = filepath
        self.chunks = chunks
        self.dataset = None

        # Ouvrir le dataset
        if chunks:
            self.dataset = xr.open_dataset(filepath, chunks=chunks)
        else:
            self.dataset = xr.open_dataset(filepath)

        if load_in_memory:
            with ProgressBar():
                self.dataset = self.dataset.load()

    def format_time(self):
        """
        Formate la dimension temporelle au format datetime pandas
        Similaire à ce qui est fait dans process_input_training_data.py
        """
        if 'time' in self.dataset.coords:
            self.dataset['time'] = pd.to_datetime(
                self.dataset.time.dt.strftime("%Y-%m-%d")
            )

    def select_region(self,
                     lat_slice: Optional[slice] = None,
                     lon_slice: Optional[slice] = None,
                     lat_range: Optional[tuple] = None,
                     lon_range: Optional[tuple] = None):
        """
        Sélectionne une région géographique

        Parameters:
        -----------
        lat_slice : slice, optional
            Tranche de latitude (ex: slice(-65, -25))
        lon_slice : slice, optional
            Tranche de longitude (ex: slice(150, 220.5))
        lat_range : tuple, optional
            Tuple (lat_min, lat_max) pour créer automatiquement le slice
        lon_range : tuple, optional
            Tuple (lon_min, lon_max) pour créer automatiquement le slice
        """
        if lat_range:
            lat_slice = slice(lat_range[0], lat_range[1])
        if lon_range:
            lon_slice = slice(lon_range[0], lon_range[1])

        if lat_slice or lon_slice:
            sel_dict = {}
            if lat_slice:
                sel_dict['lat'] = lat_slice
            if lon_slice:
                sel_dict['lon'] = lon_slice
            self.dataset = self.dataset.sel(**sel_dict)

    def select_time_period(self,
                          time_slice: Optional[slice] = None,
                          start_date: Optional[str] = None,
                          end_date: Optional[str] = None):
        """
        Sélectionne une période temporelle

        Parameters:
        -----------
        time_slice : slice, optional
            Tranche temporelle (ex: slice("1975", "2014"))
        start_date : str, optional
            Date de début (format: "YYYY-MM-DD" ou "YYYY")
        end_date : str, optional
            Date de fin (format: "YYYY-MM-DD" ou "YYYY")
        """
        if start_date and end_date:
            time_slice = slice(start_date, end_date)
        elif start_date:
            time_slice = slice(start_date, None)
        elif end_date:
            time_slice = slice(None, end_date)

        if time_slice:
            self.dataset = self.dataset.sel(time=time_slice)

    def normalize_variables(self,
                           variable_names: List[str],
                           means_dataset: Optional[xr.Dataset] = None,
                           stds_dataset: Optional[xr.Dataset] = None,
                           means_filepath: Optional[str] = None,
                           stds_filepath: Optional[str] = None):
        """
        Normalise les variables climatiques

        Parameters:
        -----------
        variable_names : list
            Liste des noms de variables à normaliser
        means_dataset : xr.Dataset, optional
            Dataset contenant les moyennes
        stds_dataset : xr.Dataset, optional
            Dataset contenant les écarts-types
        means_filepath : str, optional
            Chemin vers le fichier des moyennes
        stds_filepath : str, optional
            Chemin vers le fichier des écarts-types
        """
        # Charger les moyennes et écarts-types si nécessaire
        if means_filepath:
            means_dataset = xr.open_dataset(means_filepath)
        if stds_filepath:
            stds_dataset = xr.open_dataset(stds_filepath)

        # Normaliser (comme dans prepare_training_data)
        if means_dataset and stds_dataset:
            for var in variable_names:
                if var in self.dataset.data_vars:
                    mean_val = means_dataset[var].mean() if var in means_dataset else means_dataset[var]
                    std_val = stds_dataset[var].mean() if var in stds_dataset else stds_dataset[var]
                    self.dataset[var] = (self.dataset[var] - mean_val) / std_val

    def to_dataframe(self,
                    variable_name: Optional[str] = None,
                    method: str = 'point',
                    drop_na: bool = True) -> pd.DataFrame:
        """
        Convertit le dataset en DataFrame pandas

        Parameters:
        -----------
        variable_name : str, optional
            Nom de la variable à convertir. Si None, convertit toutes les variables
        method : str
            'point' : chaque point spatial devient une colonne
            'stacked' : empile toutes les dimensions sauf time
            'mean' : moyenne spatiale par date
        drop_na : bool
            Si True, supprime les valeurs NaN

        Returns:
        --------
        pd.DataFrame
            DataFrame avec les données NetCDF
        """
        # Vérifier si la variable existe
        if variable_name:
            if variable_name not in self.dataset.data_vars:
                available_vars = list(self.dataset.data_vars.keys())
                raise KeyError(
                    f"Variable '{variable_name}' non trouvée dans le dataset. "
                    f"Variables disponibles: {available_vars}"
                )

        if method == 'point':
            # Conversion simple : chaque point spatial = colonne
            if variable_name:
                data = self.dataset[variable_name]
            else:
                # Prendre la première variable si non spécifiée
                available_vars = list(self.dataset.data_vars.keys())
                if not available_vars:
                    raise ValueError("Aucune variable trouvée dans le dataset")
                data = self.dataset[available_vars[0]]
                variable_name = available_vars[0]

            df = data.to_dataframe(name=variable_name or 'value')
            df = df.reset_index()

            if drop_na:
                df = df.dropna()
            return df

        elif method == 'stacked':
            # Empile toutes les dimensions sauf time
            if variable_name:
                data = self.dataset[variable_name]
            else:
                # Prendre la première variable si non spécifiée
                available_vars = list(self.dataset.data_vars.keys())
                if not available_vars:
                    raise ValueError("Aucune variable trouvée dans le dataset")
                data = self.dataset[available_vars[0]]
                variable_name = available_vars[0]

            # Stack les dimensions spatiales
            stacked = data.stack(point=('lat', 'lon'))
            df = stacked.to_dataframe(name=variable_name or 'value')
            df = df.reset_index()

            if drop_na:
                df = df.dropna()
            return df

        elif method == 'mean':
            # Moyenne spatiale par date
            if variable_name:
                data = self.dataset[variable_name]
            else:
                # Prendre la première variable si non spécifiée
                available_vars = list(self.dataset.data_vars.keys())
                if not available_vars:
                    raise ValueError("Aucune variable trouvée dans le dataset")
                data = self.dataset[available_vars[0]]
                variable_name = available_vars[0]

            # Moyenne sur les dimensions spatiales
            data_mean = data.mean(['lat', 'lon'])
            df = data_mean.to_pandas()
            df = df.reset_index()

            return df

        else:
            raise ValueError(f"Méthode '{method}' non reconnue. Utilisez 'point', 'stacked', ou 'mean'")

    def to_dataframe_multiple_vars(self,
                                   variable_names: List[str],
                                   method: str = 'stacked') -> pd.DataFrame:
        """
        Convertit plusieurs variables en un seul DataFrame

        Parameters:
        -----------
        variable_names : list
            Liste des noms de variables
        method : str
            Méthode de conversion (voir to_dataframe)

        Returns:
        --------
        pd.DataFrame
            DataFrame avec plusieurs variables comme colonnes
        """
        dfs = []
        for var in variable_names:
            if var in self.dataset.data_vars:
                df_var = self.to_dataframe(variable_name=var, method=method, drop_na=False)
                dfs.append(df_var)

        # Fusionner les DataFrames
        if dfs:
            df_final = dfs[0]
            for df in dfs[1:]:
                df_final = pd.merge(df_final, df, on=['time', 'lat', 'lon'], how='outer')
            return df_final
        else:
            raise ValueError("Aucune variable trouvée dans le dataset")

    def extract_all_metadata(self) -> Dict[str, Any]:
        """
        Extrait TOUTES les métadonnées possibles du fichier NetCDF sans exception

        Returns:
        --------
        dict
            Dictionnaire contenant toutes les métadonnées extraites
        """
        metadata = {
            'file_info': {},
            'dimensions': {},
            'coordinates': {},
            'data_variables': {},
            'global_attributes': {},
            'cf_standard_attributes': {},  # Attributs CF-Conventions standards
            'encoding': {},
            'file_structure': {},
            'statistics': {},
            'cf_conventions': {}
        }

        # Informations sur le fichier
        file_path = Path(self.filepath)
        metadata['file_info'] = {
            'filepath': str(self.filepath),
            'filename': file_path.name,
            'file_size_bytes': os.path.getsize(self.filepath) if os.path.exists(self.filepath) else None,
            'file_size_mb': round(os.path.getsize(self.filepath) / (1024*1024), 2) if os.path.exists(self.filepath) else None,
            'file_extension': file_path.suffix,
            'file_format': self.dataset.encoding.get('source', 'unknown') if hasattr(self.dataset, 'encoding') else 'unknown'
        }

        # Dimensions avec détails complets
        for dim_name, dim_size in self.dataset.sizes.items():
            dim_info = {
                'size': int(dim_size),
                'unlimited': False  # sera mis à jour avec netCDF4 si disponible
            }

            # Extraire les coordonnées si elles existent
            if dim_name in self.dataset.coords:
                coord = self.dataset.coords[dim_name]

                # Fonction helper pour convertir les valeurs en format sérialisable
                def safe_convert_to_value(val):
                    """Convertit une valeur en format sérialisable"""
                    if val is None:
                        return None
                    try:
                        # Essayer de convertir en float
                        return float(val)
                    except (TypeError, ValueError):
                        # Si c'est une date/temps, convertir en string
                        try:
                            return str(val)
                        except:
                            return repr(val)

                dim_info.update({
                    'has_coordinates': True,
                    'dtype': str(coord.dtype),
                    'shape': list(coord.shape) if hasattr(coord, 'shape') else None,
                    'attributes': dict(coord.attrs) if coord.attrs else {},
                })

                # Extraire min/max/mean avec gestion des erreurs
                try:
                    if hasattr(coord, 'min') and coord.size > 0:
                        dim_info['min_value'] = safe_convert_to_value(coord.min().values)
                except Exception as e:
                    dim_info['min_value_error'] = str(e)

                try:
                    if hasattr(coord, 'max') and coord.size > 0:
                        dim_info['max_value'] = safe_convert_to_value(coord.max().values)
                except Exception as e:
                    dim_info['max_value_error'] = str(e)

                try:
                    if hasattr(coord, 'mean') and coord.size > 0:
                        dim_info['mean_value'] = safe_convert_to_value(coord.mean().values)
                except Exception as e:
                    dim_info['mean_value_error'] = str(e)

                # Échantillon de valeurs avec conversion safe
                try:
                    if hasattr(coord, 'values') and len(coord.values) > 0:
                        sample = coord.values[:10]
                        # Convertir en liste avec conversion safe
                        dim_info['values_sample'] = [safe_convert_to_value(v) for v in sample]
                except Exception as e:
                    dim_info['values_sample_error'] = str(e)
            else:
                dim_info['has_coordinates'] = False

            metadata['dimensions'][dim_name] = dim_info

        # Fonction helper pour convertir les valeurs
        def safe_convert_to_value(val):
            """Convertit une valeur en format sérialisable"""
            if val is None:
                return None
            try:
                return float(val)
            except (TypeError, ValueError):
                try:
                    return str(val)
                except:
                    return repr(val)

        # Coordonnées avec tous leurs attributs
        for coord_name in self.dataset.coords:
            if coord_name not in self.dataset.dims:  # Coordonnées non-dimensionnelles
                coord = self.dataset.coords[coord_name]
                coord_info = {
                    'dims': list(coord.dims),
                    'shape': list(coord.shape),
                    'dtype': str(coord.dtype),
                    'size': int(coord.size),
                    'attributes': dict(coord.attrs)
                }

                # Extraire min/max/mean avec gestion des erreurs
                try:
                    if coord.size > 0:
                        coord_info['min_value'] = safe_convert_to_value(coord.min().values)
                except Exception as e:
                    coord_info['min_value_error'] = str(e)

                try:
                    if coord.size > 0:
                        coord_info['max_value'] = safe_convert_to_value(coord.max().values)
                except Exception as e:
                    coord_info['max_value_error'] = str(e)

                try:
                    if coord.size > 0:
                        coord_info['mean_value'] = safe_convert_to_value(coord.mean().values)
                except Exception as e:
                    coord_info['mean_value_error'] = str(e)

                metadata['coordinates'][coord_name] = coord_info

        # Variables de données avec TOUS leurs attributs
        for var_name in self.dataset.data_vars:
            var = self.dataset.data_vars[var_name]
            var_info = {
                'dims': list(var.dims),
                'shape': list(var.shape),
                'dtype': str(var.dtype),
                'size': int(var.size),
                'nbytes': int(var.nbytes) if hasattr(var, 'nbytes') else None,
                'attributes': dict(var.attrs),
                'encoding': dict(var.encoding) if hasattr(var, 'encoding') and var.encoding else {}
            }

            # Statistiques sur les données (si possible)
            def safe_float_conversion(val):
                """Convertit en float si possible, sinon retourne None"""
                try:
                    return float(val)
                except (TypeError, ValueError):
                    return None

            try:
                if var.size > 0:
                    min_val = var.min().values
                    max_val = var.max().values
                    mean_val = var.mean().values

                    var_info['statistics'] = {
                        'min': safe_float_conversion(min_val),
                        'max': safe_float_conversion(max_val),
                        'mean': safe_float_conversion(mean_val),
                        'min_raw': str(min_val) if safe_float_conversion(min_val) is None else None,
                        'max_raw': str(max_val) if safe_float_conversion(max_val) is None else None,
                        'mean_raw': str(mean_val) if safe_float_conversion(mean_val) is None else None,
                    }

                    # STD et NaN seulement pour les types numériques
                    try:
                        if np.issubdtype(var.dtype, np.number):
                            var_info['statistics']['std'] = safe_float_conversion(var.std().values) if hasattr(var, 'std') else None
                            if hasattr(var.values, '__iter__'):
                                var_info['statistics']['has_nan'] = bool(np.isnan(var.values).any())
                                var_info['statistics']['nan_count'] = int(np.isnan(var.values).sum())
                    except Exception:
                        pass
            except Exception as e:
                var_info['statistics_error'] = str(e)

            # Informations sur les chunks si disponibles
            if hasattr(var, 'encoding') and 'chunksizes' in var.encoding:
                var_info['chunksizes'] = var.encoding['chunksizes']

            # Fill value
            if hasattr(var, 'encoding') and '_FillValue' in var.encoding:
                var_info['fill_value'] = var.encoding['_FillValue']
            elif hasattr(var, 'attrs') and '_FillValue' in var.attrs:
                var_info['fill_value'] = var.attrs['_FillValue']

            metadata['data_variables'][var_name] = var_info

        # Attributs globaux
        metadata['global_attributes'] = dict(self.dataset.attrs)

        # Extraire les attributs CF-Conventions standards spécifiques
        cf_standard_attrs = ['title', 'institution', 'source_id', 'experiment_id',
                             'grid_label', 'history', 'references', 'Conventions']
        for attr in cf_standard_attrs:
            if attr in self.dataset.attrs:
                metadata['cf_standard_attributes'][attr] = self.dataset.attrs[attr]

        # Encodings du dataset
        if hasattr(self.dataset, 'encoding'):
            metadata['encoding'] = dict(self.dataset.encoding)

        # Structure du fichier
        metadata['file_structure'] = {
            'has_groups': False,  # sera mis à jour avec netCDF4
            'number_of_dimensions': len(self.dataset.dims),
            'number_of_coordinates': len(self.dataset.coords),
            'number_of_data_variables': len(self.dataset.data_vars),
            'total_variables': len(self.dataset.variables)
        }

        # Informations CF Conventions si présentes
        cf_attrs = ['Conventions', 'featureType', 'history', 'institution', 'source',
                   'references', 'comment', 'title', 'summary', 'keywords',
                   'keywords_vocabulary', 'platform', 'product_version', 'date_created']
        for attr in cf_attrs:
            if attr in self.dataset.attrs:
                metadata['cf_conventions'][attr] = self.dataset.attrs[attr]

        # Extraire les informations CF-Conventions spécifiques pour chaque variable
        for var_name in self.dataset.data_vars:
            var = self.dataset.data_vars[var_name]
            var_attrs = var.attrs

            # Attributs CF-Conventions spécifiques
            cf_var_info = {}

            # Informations sur les cell methods
            if 'cell_methods' in var_attrs:
                cf_var_info['cell_methods'] = var_attrs['cell_methods']

            # Plages de valeurs valides
            if 'valid_range' in var_attrs:
                cf_var_info['valid_range'] = var_attrs['valid_range']
            if 'valid_min' in var_attrs:
                cf_var_info['valid_min'] = var_attrs['valid_min']
            if 'valid_max' in var_attrs:
                cf_var_info['valid_max'] = var_attrs['valid_max']

            # Flags pour données catégorielles
            if 'flag_values' in var_attrs:
                cf_var_info['flag_values'] = var_attrs['flag_values']
            if 'flag_meanings' in var_attrs:
                cf_var_info['flag_meanings'] = var_attrs['flag_meanings']
            if 'flag_masks' in var_attrs:
                cf_var_info['flag_masks'] = var_attrs['flag_masks']

            # Coordonnées auxiliaires
            if 'coordinates' in var_attrs:
                cf_var_info['coordinates'] = var_attrs['coordinates']
            if 'bounds' in var_attrs:
                cf_var_info['bounds'] = var_attrs['bounds']
            if 'ancillary_variables' in var_attrs:
                cf_var_info['ancillary_variables'] = var_attrs['ancillary_variables']

            # Direction pour coordonnées verticales
            if 'positive' in var_attrs:
                cf_var_info['positive'] = var_attrs['positive']

            # Grid mapping
            if 'grid_mapping' in var_attrs:
                cf_var_info['grid_mapping'] = var_attrs['grid_mapping']

            # Formula terms pour coordonnées
            if 'formula_terms' in var_attrs:
                cf_var_info['formula_terms'] = var_attrs['formula_terms']

            # Compression et scaling
            if 'scale_factor' in var_attrs:
                cf_var_info['scale_factor'] = var_attrs['scale_factor']
            if 'add_offset' in var_attrs:
                cf_var_info['add_offset'] = var_attrs['add_offset']

            if cf_var_info:
                if 'cf_conventions_attributes' not in metadata['data_variables'][var_name]:
                    metadata['data_variables'][var_name]['cf_conventions_attributes'] = {}
                metadata['data_variables'][var_name]['cf_conventions_attributes'] = cf_var_info

        # Accès direct avec netCDF4 pour informations supplémentaires
        if NETCDF4_AVAILABLE:
            try:
                with netCDF4.Dataset(self.filepath, 'r') as nc:
                    # Vérifier les dimensions illimitées
                    for dim_name, dim in nc.dimensions.items():
                        if dim_name in metadata['dimensions']:
                            metadata['dimensions'][dim_name]['unlimited'] = dim.isunlimited()
                            metadata['dimensions'][dim_name]['netcdf4_id'] = dim.__class__.__name__

                    # Informations sur les groupes
                    metadata['file_structure']['has_groups'] = hasattr(nc, 'groups') and len(nc.groups) > 0
                    if hasattr(nc, 'groups') and len(nc.groups) > 0:
                        metadata['file_structure']['groups'] = list(nc.groups.keys())
                        # Informations détaillées sur les groupes
                        group_info = {}
                        for group_name, group in nc.groups.items():
                            group_info[group_name] = {
                                'variables': list(group.variables.keys()),
                                'dimensions': list(group.dimensions.keys()),
                                'groups': list(group.groups.keys()) if hasattr(group, 'groups') else []
                            }
                        if group_info:
                            metadata['file_structure']['groups_detail'] = group_info

                    # Informations sur les variables brutes (accès direct)
                    for var_name in nc.variables:
                        if var_name in metadata['data_variables']:
                            nc_var = nc.variables[var_name]
                            nc4_info = {
                                'storage': nc_var.chunking() if hasattr(nc_var, 'chunking') else None,
                                'endianness': nc_var.endian() if hasattr(nc_var, 'endian') else None,
                                'all_attributes': {attr: getattr(nc_var, attr) for attr in nc_var.ncattrs()}
                            }

                            # Informations sur les filtres de compression
                            try:
                                if hasattr(nc_var, 'filters'):
                                    filters = nc_var.filters()
                                    if filters:
                                        nc4_info['compression'] = {
                                            'filters': filters,
                                            'is_compressed': True
                                        }
                                    else:
                                        nc4_info['compression'] = {'is_compressed': False}
                            except Exception:
                                pass

                            # Type de données spécial
                            if hasattr(nc_var, 'datatype'):
                                dt = nc_var.datatype
                                nc4_info['datatype'] = {
                                    'name': dt.name if hasattr(dt, 'name') else str(dt),
                                    'is_compound': dt.iscompound if hasattr(dt, 'iscompound') else False,
                                    'is_enum': dt.isenum if hasattr(dt, 'isenum') else False,
                                    'is_vlen': dt.isvlen if hasattr(dt, 'isvlen') else False,
                                    'is_opaque': dt.isopaquetype if hasattr(dt, 'isopaquetype') else False
                                }

                            metadata['data_variables'][var_name]['netcdf4_info'] = nc4_info

                    # Informations sur les types de données composés et enums
                    if hasattr(nc, 'datatypes'):
                        metadata['file_structure']['datatypes'] = {}
                        for dt_name, dt in nc.datatypes.items():
                            dt_info = {
                                'name': dt_name,
                                'is_compound': dt.iscompound if hasattr(dt, 'iscompound') else False,
                                'is_enum': dt.isenum if hasattr(dt, 'isenum') else False,
                                'size': dt.size if hasattr(dt, 'size') else None
                            }
                            if hasattr(dt, 'fields') and dt.fields:
                                dt_info['fields'] = {field: {'name': field, 'dtype': str(dt.fields[field][1])}
                                                    for field in dt.fields.keys()}
                            metadata['file_structure']['datatypes'][dt_name] = dt_info

                    # Informations sur les dimensions unlimited
                    unlimited_dims = [dim for dim in nc.dimensions.values() if dim.isunlimited()]
                    if unlimited_dims:
                        metadata['file_structure']['unlimited_dimensions'] = [dim.name for dim in unlimited_dims]

                    # Informations sur les chemin des variables dans les groupes
                    if hasattr(nc, 'path'):
                        metadata['file_structure']['root_path'] = nc.path

                    # Informations supplémentaires sur la structure
                    if hasattr(nc, 'isncfile'):
                        metadata['file_structure']['is_valid_netcdf'] = nc.isncfile()
            except Exception as e:
                metadata['netcdf4_extraction_error'] = str(e)

        return metadata

    def get_info(self, detailed: bool = True):
        """
        Affiche les informations sur le dataset

        Parameters:
        -----------
        detailed : bool
            Si True, affiche toutes les métadonnées détaillées
        """
        if detailed:
            metadata = self.extract_all_metadata()

            print("=" * 80)
            print("INFORMATIONS COMPLÈTES DU FICHIER NETCDF")
            print("=" * 80)

            # Informations fichier
            print("\n📁 INFORMATIONS FICHIER")
            print("-" * 80)
            for key, value in metadata['file_info'].items():
                print(f"  {key}: {value}")

            # Dimensions
            print("\n📏 DIMENSIONS")
            print("-" * 80)
            for dim_name, dim_info in metadata['dimensions'].items():
                print(f"\n  Dimension: {dim_name}")
                for key, value in dim_info.items():
                    if key != 'values_sample' or (key == 'values_sample' and value is not None):
                        if key == 'values_sample':
                            print(f"    {key}: {value} (échantillon)")
                        else:
                            print(f"    {key}: {value}")

            # Coordonnées
            if metadata['coordinates']:
                print("\n🗺️  COORDONNÉES NON-DIMENSIONNELLES")
                print("-" * 80)
                for coord_name, coord_info in metadata['coordinates'].items():
                    print(f"\n  Coordonnée: {coord_name}")
                    for key, value in coord_info.items():
                        print(f"    {key}: {value}")

            # Variables de données
            print("\n📊 VARIABLES DE DONNÉES")
            print("-" * 80)
            for var_name, var_info in metadata['data_variables'].items():
                print(f"\n  Variable: {var_name}")
                print(f"    Dimensions: {var_info['dims']}")
                print(f"    Shape: {var_info['shape']}")
                print(f"    Type: {var_info['dtype']}")
                print(f"    Taille: {var_info['size']} éléments")
                if var_info.get('nbytes'):
                    print(f"    Mémoire: {var_info['nbytes'] / (1024*1024):.2f} MB")

                if 'statistics' in var_info:
                    print(f"    Statistiques:")
                    for stat_key, stat_value in var_info['statistics'].items():
                        if stat_value is not None:
                            print(f"      {stat_key}: {stat_value}")

                if var_info.get('attributes'):
                    print(f"    Attributs ({len(var_info['attributes'])}):")
                    for attr_key, attr_value in var_info['attributes'].items():
                        if isinstance(attr_value, (list, tuple)) and len(str(attr_value)) > 100:
                            print(f"      {attr_key}: {str(attr_value)[:100]}...")
                        else:
                            print(f"      {attr_key}: {attr_value}")

                if var_info.get('encoding'):
                    print(f"    Encodage:")
                    for enc_key, enc_value in var_info['encoding'].items():
                        print(f"      {enc_key}: {enc_value}")

                # Attributs CF-Conventions spécifiques
                if var_info.get('cf_conventions_attributes'):
                    print(f"    Attributs CF-Conventions:")
                    for cf_key, cf_value in var_info['cf_conventions_attributes'].items():
                        print(f"      {cf_key}: {cf_value}")

                # Informations netCDF4 brutes si disponibles
                if var_info.get('netcdf4_info'):
                    print(f"    Informations NetCDF4:")
                    nc4 = var_info['netcdf4_info']
                    if nc4.get('compression'):
                        print(f"      Compression: {nc4['compression']}")
                    if nc4.get('datatype'):
                        print(f"      Type de données: {nc4['datatype']}")
                    if nc4.get('endianness'):
                        print(f"      Endianness: {nc4['endianness']}")
                    if nc4.get('storage'):
                        print(f"      Chunking: {nc4['storage']}")

            # Attributs globaux
            if metadata['global_attributes']:
                print("\n🌐 ATTRIBUTS GLOBAUX")
                print("-" * 80)
                for attr_key, attr_value in metadata['global_attributes'].items():
                    if isinstance(attr_value, (list, tuple)) and len(str(attr_value)) > 200:
                        print(f"  {attr_key}: {str(attr_value)[:200]}...")
                    else:
                        print(f"  {attr_key}: {attr_value}")

            # Structure
            print("\n🏗️  STRUCTURE DU FICHIER")
            print("-" * 80)
            for key, value in metadata['file_structure'].items():
                print(f"  {key}: {value}")

            # Attributs CF-Conventions standards
            if metadata['cf_standard_attributes']:
                print("\n📋 ATTRIBUTS CF-CONVENTIONS STANDARDS")
                print("-" * 80)
                for key, value in metadata['cf_standard_attributes'].items():
                    if isinstance(value, (list, tuple)) and len(str(value)) > 200:
                        print(f"  {key}: {str(value)[:200]}...")
                    else:
                        print(f"  {key}: {value}")

            # Conventions CF (autres attributs)
            if metadata['cf_conventions']:
                print("\n📋 AUTRES CONVENTIONS CF")
                print("-" * 80)
                for key, value in metadata['cf_conventions'].items():
                    if isinstance(value, (list, tuple)) and len(str(value)) > 200:
                        print(f"  {key}: {str(value)[:200]}...")
                    else:
                        print(f"  {key}: {value}")
        else:
            # Version simple (comme avant)
            print("=== Informations du Dataset NetCDF ===")
            print(f"Fichier: {self.filepath}")
            print(f"\nDimensions: {dict(self.dataset.sizes)}")
            print(f"\nCoordonnées: {list(self.dataset.coords.keys())}")
            print(f"\nVariables: {list(self.dataset.data_vars.keys())}")
            print(f"\nAttributs: {self.dataset.attrs}")

            if 'time' in self.dataset.coords:
                print(f"\nPériode temporelle: {self.dataset.time.min().values} à {self.dataset.time.max().values}")
            if 'lat' in self.dataset.coords:
                print(f"Latitude: {float(self.dataset.lat.min().values)} à {float(self.dataset.lat.max().values)}")
            if 'lon' in self.dataset.coords:
                print(f"Longitude: {float(self.dataset.lon.min().values)} à {float(self.dataset.lon.max().values)}")

    def export_metadata_to_json(self, output_file: Optional[str] = None) -> str:
        """
        Exporte toutes les métadonnées au format JSON

        Parameters:
        -----------
        output_file : str, optional
            Chemin du fichier de sortie. Si None, utilise le nom du fichier avec extension .json

        Returns:
        --------
        str
            Chemin du fichier JSON créé
        """
        metadata = self.extract_all_metadata()

        # Convertir les types non sérialisables
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (datetime, pd.Timestamp)):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (bytes, bytearray)):
                return obj.decode('utf-8', errors='ignore')
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)

        metadata_serializable = convert_to_serializable(metadata)

        if output_file is None:
            file_path = Path(self.filepath)
            output_file = str(file_path.with_suffix('.metadata.json'))

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata_serializable, f, indent=2, ensure_ascii=False)

        return output_file

    def export_metadata_to_csv(self, output_file: Optional[str] = None) -> str:
        """
        Exporte les métadonnées pertinentes des variables au format CSV
        (Les métadonnées statiques restent dans le JSON)

        Parameters:
        -----------
        output_file : str, optional
            Chemin du fichier de sortie

        Returns:
        --------
        str
            Chemin du fichier CSV créé
        """
        metadata = self.extract_all_metadata()

        # Fonction pour convertir les structures complexes en JSON string
        def to_csv_value(value):
            """Convertit une valeur en format approprié pour CSV"""
            if value is None:
                return ''
            elif isinstance(value, (dict, list)):
                # Convertir les structures complexes en JSON string
                try:
                    return json.dumps(value, ensure_ascii=False, default=str)
                except:
                    return str(value)
            elif isinstance(value, (np.integer, np.floating)):
                return float(value)
            elif isinstance(value, (bool, np.bool_)):
                return bool(value)
            elif isinstance(value, (bytes, bytearray)):
                return value.decode('utf-8', errors='ignore')
            else:
                return str(value)

        # Créer un DataFrame avec les métadonnées pertinentes des variables (une ligne par variable)
        rows = []
        for var_name, var_info in metadata['data_variables'].items():
            row = {
                'variable_name': var_name,
                'dims': str(var_info.get('dims', '')),
                'shape': str(var_info.get('shape', '')),
                'dtype': var_info.get('dtype', ''),
                'size': var_info.get('size', ''),
                'nbytes': var_info.get('nbytes', ''),
            }

            # Ajouter les statistiques si disponibles
            if 'statistics' in var_info and var_info['statistics']:
                for stat_key, stat_value in var_info['statistics'].items():
                    row[f'stat_{stat_key}'] = to_csv_value(stat_value)

            # Ajouter les attributs principaux (métadonnées pertinentes)
            attrs = var_info.get('attributes', {})
            if attrs:
                for attr_key, attr_value in attrs.items():
                    row[f'attr_{attr_key}'] = to_csv_value(attr_value)

            # Ajouter les informations d'encodage principales
            encoding = var_info.get('encoding', {})
            if encoding:
                for enc_key, enc_value in encoding.items():
                    row[f'encoding_{enc_key}'] = to_csv_value(enc_value)

            # Ajouter les attributs CF-Conventions
            cf_attrs = var_info.get('cf_conventions_attributes', {})
            if cf_attrs:
                for cf_key, cf_value in cf_attrs.items():
                    row[f'cf_{cf_key}'] = to_csv_value(cf_value)

            # Ajouter d'autres champs pertinents si présents
            if 'fill_value' in var_info:
                row['fill_value'] = to_csv_value(var_info['fill_value'])
            if 'chunksizes' in var_info:
                row['chunksizes'] = to_csv_value(var_info['chunksizes'])
            if 'statistics_error' in var_info:
                row['statistics_error'] = to_csv_value(var_info['statistics_error'])

            rows.append(row)

        df = pd.DataFrame(rows)

        if output_file is None:
            file_path = Path(self.filepath)
            output_file = str(file_path.with_suffix('.metadata.csv'))

        # Sauvegarder avec toutes les colonnes
        df.to_csv(output_file, index=False, encoding='utf-8')
        return output_file


def process_netcdf_file(filepath: str,
                        variable_name: Optional[str] = None,
                        variable_names: Optional[List[str]] = None,
                        lat_range: Optional[tuple] = None,
                        lon_range: Optional[tuple] = None,
                        time_slice: Optional[slice] = None,
                        normalize: bool = False,
                        means_filepath: Optional[str] = None,
                        stds_filepath: Optional[str] = None,
                        method: str = 'stacked',
                        chunks: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """
    Fonction utilitaire pour traiter un fichier NetCDF et le convertir en DataFrame

    Parameters:
    -----------
    filepath : str
        Chemin vers le fichier NetCDF
    variable_name : str, optional
        Nom d'une variable à extraire
    variable_names : list, optional
        Liste de variables à extraire
    lat_range : tuple, optional
        (lat_min, lat_max) pour sélectionner une région
    lon_range : tuple, optional
        (lon_min, lon_max) pour sélectionner une région
    time_slice : slice, optional
        Tranche temporelle à sélectionner
    normalize : bool
        Si True, normalise les variables
    means_filepath : str, optional
        Chemin vers le fichier des moyennes
    stds_filepath : str, optional
        Chemin vers le fichier des écarts-types
    method : str
        Méthode de conversion ('point', 'stacked', 'mean')
    chunks : dict, optional
        Chunks pour le chargement

    Returns:
    --------
    pd.DataFrame
        DataFrame avec les données
    """
    # Initialiser le convertisseur
    converter = NetCDFToDataFrame(filepath, chunks=chunks)

    # Formater le temps
    converter.format_time()

    # Sélectionner une région si nécessaire
    if lat_range or lon_range:
        converter.select_region(lat_range=lat_range, lon_range=lon_range)

    # Sélectionner une période si nécessaire
    if time_slice:
        converter.select_time_period(time_slice=time_slice)

    # Normaliser si nécessaire
    if normalize:
        vars_to_norm = variable_names or ([variable_name] if variable_name else [])
        converter.normalize_variables(
            vars_to_norm,
            means_filepath=means_filepath,
            stds_filepath=stds_filepath
        )

    # Convertir en DataFrame
    if variable_names:
        df = converter.to_dataframe_multiple_vars(variable_names, method=method)
    elif variable_name:
        df = converter.to_dataframe(variable_name, method=method)
    else:
        df = converter.to_dataframe(method=method)

    return df


# Exemple d'utilisation
if __name__ == "__main__":
    import sys
    import os

    # Exemple 1: Traitement simple d'un fichier
    # Utiliser le fichier fourni en argument ou un fichier par défaut
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Chercher un fichier NetCDF dans le répertoire courant
        nc_files = ['./NorESM2-MM_histupdated_compressed.nc']
        if nc_files:
            filepath = nc_files[0]
            print(f"Utilisation du fichier trouvé: {filepath}")
        else:
            print("Aucun fichier NetCDF trouvé. Veuillez spécifier un fichier en argument.")
            sys.exit(1)

    # Créer un convertisseur
    converter = NetCDFToDataFrame(filepath)

    # Extraire et afficher TOUTES les métadonnées
    print("\n" + "="*80)
    print("EXTRACTION COMPLÈTE DES MÉTADONNÉES")
    print("="*80)
    converter.get_info(detailed=True)

    # Exporter les métadonnées en JSON
    print("\n" + "="*80)
    print("EXPORT DES MÉTADONNÉES")
    print("="*80)
    try:
        json_file = converter.export_metadata_to_json()
        print(f"✓ Métadonnées exportées en JSON: {json_file}")

        csv_file = converter.export_metadata_to_csv()
        print(f"✓ Résumé exporté en CSV: {csv_file}")
    except Exception as e:
        print(f"Erreur lors de l'export: {e}")

    print("\n" + "="*80)
    print("FIN DU TRAITEMENT DES MÉTADONNÉES")
    print("="*80)
    print("Les données brutes ne sont plus exportées dans un CSV combiné.")
```

---

### `src/st_cdgm/models/__init__.py`

```python
"""
Modules de modèles pour ST-CDGM.
"""

from .causal_rcn import RCNCell, RCNSequenceRunner
from .diffusion_decoder import CausalDiffusionDecoder, DiffusionOutput
from .intelligible_encoder import IntelligibleVariableEncoder, IntelligibleVariableConfig, SpatialConditioningProjector
from .graph_builder import HeteroGraphBuilder

__all__ = [
    "RCNCell",
    "RCNSequenceRunner",
    "CausalDiffusionDecoder",
    "DiffusionOutput",
    "IntelligibleVariableEncoder",
    "IntelligibleVariableConfig",
    "SpatialConditioningProjector",
    "HeteroGraphBuilder",
]
```

---

### `src/st_cdgm/models/causal_rcn.py`

```python
"""
Module 4 – Réseau causal récurrent (RCN) pour l’architecture ST-CDGM.

Ce module implémente la cellule RCN (`RCNCell`) et un utilitaire de déroulement
séquentiel (`RCNSequenceRunner`). La cellule combine un cœur causal (matrice DAG
apprenante + assignations structurelles) et une mise à jour récurrente via GRU,
suivie d’une reconstruction optionnelle pour la perte L_rec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MaskDiagonal(torch.autograd.Function):
    """
    Fonction autograd pour imposer une diagonale nulle sur la matrice DAG.
    """

    @staticmethod
    def forward(ctx, matrix: Tensor) -> Tensor:
        ctx.save_for_backward(matrix)
        out = matrix.clone()
        out.fill_diagonal_(0.0)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        grad_input = grad_output.clone()
        grad_input.fill_diagonal_(0.0)
        return (grad_input,)


class RCNCell(nn.Module):
    """
    Cellule du réseau causal récurrent.

    Parameters
    ----------
    num_vars :
        Nombre de variables intelligibles (q).
    hidden_dim :
        Dimension de l'état caché par variable.
    driver_dim :
        Dimension du forçage externe (features LR).
    reconstruction_dim :
        Dimension de la reconstruction (optionnel). Si None, la reconstruction est omise.
    activation :
        Fonction d'activation utilisée dans les MLPs d'assignation structurelle.
    dropout :
        Dropout appliqué sur les sorties GRU pour régularisation.
    """

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        driver_dim: int,
        *,
        reconstruction_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.driver_dim = driver_dim
        self.reconstruction_dim = reconstruction_dim
        self.activation = activation or nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Matrice DAG apprenable
        self.A_dag = nn.Parameter(torch.randn(num_vars, num_vars))

        # MLPs d'assignation structurelle (une par variable cible)
        self.structural_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    self.activation,
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_vars)
            ]
        )

        # Encodeur du forçage externe
        self.driver_encoder = nn.Sequential(
            nn.Linear(driver_dim, hidden_dim),
            self.activation,
        )

        # GRUCell par variable
        self.gru_cells = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_vars)]
        )

        # Embedding d'identité par variable — spécialise le driver pour chaque GRU
        self.var_embed = nn.Embedding(num_vars, hidden_dim)

        # Décodeur de reconstruction optionnel
        if self.reconstruction_dim is not None:
            self.reconstruction_decoder = nn.Linear(
                num_vars * hidden_dim, self.reconstruction_dim
            )
        else:
            self.reconstruction_decoder = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Réinitialise les paramètres internes.
        """
        nn.init.xavier_uniform_(self.A_dag)
        for mlp in self.structural_mlps:
            for layer in mlp:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        for gru in self.gru_cells:
            gru.reset_parameters()
        for layer in self.driver_encoder:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        nn.init.normal_(self.var_embed.weight, std=0.02)
        if self.reconstruction_decoder is not None:
            nn.init.xavier_uniform_(self.reconstruction_decoder.weight)
            nn.init.zeros_(self.reconstruction_decoder.bias)

    def forward(
        self,
        H_prev: Tensor,
        driver: Tensor,
        reconstruction_source: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Applique une mise à jour récurrente.

        Parameters
        ----------
        H_prev :
            État précédent [q, N, hidden_dim].
        driver :
            Forçage externe [N, driver_dim].
        reconstruction_source :
            Tenseur utilisé comme base pour la reconstruction (par défaut ``H_prev``).

        Returns
        -------
        H_next :
            Nouvel état [q, N, hidden_dim].
        reconstruction :
            Reconstruction optionnelle des features [N, reconstruction_dim].
        A_masked :
            Matrice DAG avec diagonale masquée (pour L_dag).
        """
        if H_prev.dim() != 3:
            raise ValueError("H_prev doit avoir la forme [q, N, hidden_dim].")
        q, N, hidden_dim = H_prev.shape
        if q != self.num_vars or hidden_dim != self.hidden_dim:
            raise ValueError("Dimensions de H_prev incompatibles avec la cellule RCN.")
        if driver.shape[0] != N:
            raise ValueError("driver doit partager la dimension N avec H_prev.")
        if driver.shape[1] != self.driver_dim:
            raise ValueError("Dimension du driver incompatible.")

        A_masked = MaskDiagonal.apply(self.A_dag)

        # Étape 1 : Prédiction interne via SCM (vectorisée)
        weighted_sum = torch.einsum("ik,inj->knj", A_masked, H_prev)
        H_hat = []
        for k in range(self.num_vars):
            h_k_hat = self.structural_mlps[k](weighted_sum[k])
            H_hat.append(h_k_hat)
        H_hat_tensor = torch.stack(H_hat, dim=0)

        # Étape 2 : Mise à jour par forçage externe (GRU)
        # Phase A2: Fully vectorized GRU computation - eliminates Python loop
        # Pre-encode driver once, reuse for all GRU cells
        driver_emb = self.driver_encoder(driver)  # [N, hidden_dim]
        
        # Phase A2: Vectorized GRU computation with separate parameters per variable
        # Extract all GRU parameters and process in a single batched operation
        # This eliminates the Python loop overhead and allows better GPU utilization
        
        # Prepare batched inputs: [q, N, hidden_dim]
        # driver_emb: [N, hidden_dim] -> expand to [q, N, hidden_dim]
        driver_batch = driver_emb.unsqueeze(0).expand(self.num_vars, -1, -1)  # [q, N, hidden_dim]
        var_ids = torch.arange(self.num_vars, device=driver_emb.device)
        driver_batch = driver_batch + self.var_embed(var_ids).unsqueeze(1)   # [q, N, d] + [q, 1, d]
        hidden_batch = H_hat_tensor  # [q, N, hidden_dim]
        
        # Vectorized GRU computation for all variables in parallel
        # Each GRU cell has separate parameters, so we batch the computation manually
        # by extracting weights and doing batched matrix operations
        
        # Extract weights from all GRU cells and stack them: [q, ...]
        weight_ih_list = []
        weight_hh_list = []
        bias_ih_list = []
        bias_hh_list = []
        
        for k in range(self.num_vars):
            gru_cell = self.gru_cells[k]
            # GRUCell weights: [3*hidden_dim, hidden_dim] (for input->reset/update/new)
            weight_ih_list.append(gru_cell.weight_ih)  # [3*hidden_dim, hidden_dim]
            weight_hh_list.append(gru_cell.weight_hh)  # [3*hidden_dim, hidden_dim]
            bias_ih_list.append(gru_cell.bias_ih if gru_cell.bias_ih is not None else torch.zeros(3*hidden_dim, device=driver_emb.device))
            bias_hh_list.append(gru_cell.bias_hh if gru_cell.bias_hh is not None else torch.zeros(3*hidden_dim, device=driver_emb.device))
        
        # Stack all parameters: [q, 3*hidden_dim, hidden_dim]
        W_ih_batch = torch.stack(weight_ih_list, dim=0)  # [q, 3*hidden_dim, hidden_dim]
        W_hh_batch = torch.stack(weight_hh_list, dim=0)  # [q, 3*hidden_dim, hidden_dim]
        b_ih_batch = torch.stack(bias_ih_list, dim=0)  # [q, 3*hidden_dim]
        b_hh_batch = torch.stack(bias_hh_list, dim=0)  # [q, 3*hidden_dim]
        
        # Batched matrix operations
        # Input transformation: [q, N, hidden_dim] @ [q, hidden_dim, 3*hidden_dim] -> [q, N, 3*hidden_dim]
        gi_batch = torch.bmm(driver_batch, W_ih_batch.transpose(1, 2)) + b_ih_batch.unsqueeze(1)  # [q, N, 3*hidden_dim]
        gh_batch = torch.bmm(hidden_batch, W_hh_batch.transpose(1, 2)) + b_hh_batch.unsqueeze(1)  # [q, N, 3*hidden_dim]
        
        # Split into reset, update, and new gates
        # GRU gates are ordered as: reset, update, new
        i_r, i_u, i_n = gi_batch.chunk(3, dim=2)  # Each: [q, N, hidden_dim]
        h_r, h_u, h_n = gh_batch.chunk(3, dim=2)  # Each: [q, N, hidden_dim]
        
        # GRU computations (all vectorized)
        reset_gate = torch.sigmoid(i_r + h_r)  # [q, N, hidden_dim]
        update_gate = torch.sigmoid(i_u + h_u)  # [q, N, hidden_dim]
        new_gate = torch.tanh(i_n + reset_gate * h_n)  # [q, N, hidden_dim]
        
        # Final hidden state: h_new = (1 - z) * n + z * h
        H_next_tensor = (1 - update_gate) * new_gate + update_gate * hidden_batch  # [q, N, hidden_dim]
        
        # Apply dropout in vectorized manner
        H_next_tensor = self.dropout(H_next_tensor)

        # Reconstruction facultative
        reconstruction = None
        if self.reconstruction_decoder is not None:
            # Use provided reconstruction source or default to H_prev
            source = reconstruction_source if reconstruction_source is not None else H_prev
            
            # Ensure source has shape [N, num_vars * hidden_dim]
            # If source is [q, N, hidden_dim] (like H_prev), permute and reshape
            if source.dim() == 3 and source.shape[0] == self.num_vars and source.shape[2] == self.hidden_dim:
                recon_input = source.permute(1, 0, 2).reshape(N, -1)
            elif source.dim() == 2:
                # Assume already flattened or correct shape [N, features]
                recon_input = source
            else:
                # Fallback reshape trying to preserve N
                recon_input = source.reshape(N, -1)
                
            reconstruction = self.reconstruction_decoder(recon_input)

        return H_next_tensor, reconstruction, A_masked

    def pool_state(
        self,
        H: Tensor,
        *,
        batch: Optional[Tensor] = None,
        pool: str = "mean",
    ) -> Tensor:
        """
        Agrège l'état caché sur la dimension spatiale.

        Parameters
        ----------
        H :
            Tenseur [q, N, hidden_dim].
        batch :
            Vecteur d'indices de graphe pour chaque nœud (longueur N).
        pool :
            ``"mean"`` (défaut) ou ``"max"``.

        Returns
        -------
        Tensor
            Tenseur [num_graphs, q, hidden_dim].
        """
        pool = pool.lower()
        if batch is None:
            if pool == "mean":
                return H.mean(dim=1, keepdim=False).unsqueeze(0)
            if pool == "max":
                return H.amax(dim=1, keepdim=False).unsqueeze(0)
            raise ValueError(f"Pooling '{pool}' non pris en charge sans batch.")

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled: List[Tensor] = []
        for g in range(num_graphs):
            mask = batch == g
            if not torch.any(mask):
                pooled.append(torch.zeros(self.num_vars, self.hidden_dim, device=H.device, dtype=H.dtype))
                continue
            if pool == "mean":
                pooled.append(H[:, mask, :].mean(dim=1))
            elif pool == "max":
                pooled.append(H[:, mask, :].amax(dim=1))
            else:
                raise ValueError(f"Pooling '{pool}' non pris en charge.")
        return torch.stack(pooled, dim=0)

    def dag_matrix(self, *, masked: bool = True) -> Tensor:
        """
        Retourne la matrice causale (masquée ou brute).
        """
        return MaskDiagonal.apply(self.A_dag) if masked else self.A_dag

    def _prepare_reconstruction_features(self, tensor: Tensor, N: int) -> Tensor:
        """
        Mise en forme standard des caractéristiques utilisées pour la reconstruction.
        """
        if tensor.dim() == 2:
            if tensor.size(0) != N:
                raise ValueError(
                    f"reconstruction_source attend {N} nœuds, obtenu {tensor.size(0)}."
                )
            return tensor
        if tensor.dim() == 3:
            if tensor.size(1) != N:
                raise ValueError(
                    f"reconstruction_source attend {N} nœuds sur la dimension 1, obtenu {tensor.size(1)}."
                )
            return tensor.permute(1, 0, 2).reshape(N, -1)
        raise ValueError("reconstruction_source doit être de dimension 2 ou 3.")


@dataclass
class RCNSequenceOutput:
    """
    Résultats du déroulement séquentiel de la cellule RCN.
    """

    states: List[Tensor]
    reconstructions: List[Optional[Tensor]]
    dag_matrices: List[Tensor]


class RCNSequenceRunner:
    """
    Utilitaire pour dérouler la cellule RCN sur des séquences temporelles.
    """

    def __init__(self, cell: RCNCell, *, detach_interval: Optional[int] = None) -> None:
        self.cell = cell
        self.detach_interval = detach_interval

    def run(
        self,
        H_init: Tensor,
        drivers: Sequence[Tensor],
        reconstruction_sources: Optional[Sequence[Optional[Tensor]]] = None,
    ) -> RCNSequenceOutput:
        """
        Déroule la cellule sur la séquence de drivers.

        Parameters
        ----------
        H_init :
            État initial [q, N, hidden_dim].
        drivers :
            Séquence de tenseurs [N, driver_dim] de longueur T.
        reconstruction_sources :
            Séquence optionnelle alignée sur ``drivers`` contenant les tenseurs
            utilisés pour la reconstruction (ex: features LR à reconstruire).
        """
        H_t = H_init
        states: List[Tensor] = []
        reconstructions: List[Optional[Tensor]] = []
        dag_matrices: List[Tensor] = []

        for t, driver in enumerate(drivers):
            recon_source = None
            if reconstruction_sources is not None:
                recon_source = reconstruction_sources[t]
            H_next, recon, A_masked = self.cell(H_t, driver, reconstruction_source=recon_source)
            states.append(H_next)
            reconstructions.append(recon)
            dag_matrices.append(A_masked)

            if self.detach_interval is not None and (t + 1) % self.detach_interval == 0:
                H_t = H_next.detach()
            else:
                H_t = H_next

        return RCNSequenceOutput(states=states, reconstructions=reconstructions, dag_matrices=dag_matrices)
```

---

### `src/st_cdgm/models/diffusion_decoder.py`

```python
"""
Module 5 – Décodeur de diffusion conditionnel pour ST-CDGM.

Ce module encapsule un UNet conditionnel (diffusers) et fournit des utilitaires
pour calculer la perte de diffusion, appliquer les contraintes physiques et
échantillonner des sorties haute résolution conditionnées par l'état causal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor

try:
    from diffusers import DDPMScheduler, UNet2DConditionModel
except ImportError as exc:  # pragma: no cover - dépendance optionnelle
    raise ImportError(
        "Le module diffusion_decoder nécessite la bibliothèque `diffusers` "
        "(pip install diffusers accelerate)."
    ) from exc

# Phase E1: Check if DPM-Solver++ is available (might be in newer versions of diffusers)
try:
    from diffusers import DPMSolverMultistepScheduler
    HAS_DPM_SOLVER = True
except ImportError:
    HAS_DPM_SOLVER = False


def _replace_conv_transpose_with_resize(module: nn.Module) -> None:
    """Remplace ConvTranspose2d par Upsample + Conv2d (réduit artefacts checkerboard)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose2d):
            st = child.stride[0]
            if st >= 2 and child.stride[0] == child.stride[1]:
                repl = nn.Sequential(
                    nn.Upsample(scale_factor=float(st), mode="nearest"),
                    nn.Conv2d(
                        child.in_channels,
                        child.out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=child.bias is not None,
                    ),
                )
                if child.bias is not None:
                    nn.init.zeros_(repl[1].bias)
                nn.init.kaiming_normal_(repl[1].weight, nonlinearity="relu")
                setattr(module, name, repl)
            else:
                _replace_conv_transpose_with_resize(child)
        else:
            _replace_conv_transpose_with_resize(child)


@dataclass
class DiffusionOutput:
    """
    Résultat d'un échantillonnage de diffusion.
    """

    residual: Tensor
    baseline: Optional[Tensor]
    t_min: Tensor
    t_mean: Tensor
    t_max: Tensor

    @property
    def composite(self) -> Tensor:
        """Retourne le champ reconstruit (concaténé) [B,3,H,W]."""
        return torch.cat([self.t_min, self.t_mean, self.t_max], dim=1)


class CausalDiffusionDecoder(nn.Module):
    """
    Décodeur de diffusion conditionnel pour générer les champs HR.
    """

    def __init__(
        self,
        in_channels: int,
        conditioning_dim: int,
        height: int,
        width: int,
        *,
        num_diffusion_steps: int = 1000,
        unet_kwargs: Optional[dict] = None,
        scheduler_type: str = "ddpm",
        use_gradient_checkpointing: bool = False,
        conv_padding_mode: str = "zeros",
        anti_checkerboard: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.conditioning_dim = conditioning_dim
        self.height = height
        self.width = width
        self.num_diffusion_steps = num_diffusion_steps
        self.scheduler_type = scheduler_type  # "ddpm" or "edm"

        self._condition_adapter: Optional[Callable[[Tensor], Tensor]] = None

        unet_kwargs = unet_kwargs or {}
        self.unet = UNet2DConditionModel(
            # NOTE: use (height, width) for non-square grids; passing an int makes Diffusers assume square inputs.
            sample_size=(height, width),
            in_channels=in_channels,
            out_channels=in_channels,
            cross_attention_dim=conditioning_dim,
            **unet_kwargs,
        )
        self.scheduler = DDPMScheduler(num_train_timesteps=num_diffusion_steps)

        if conv_padding_mode == "replicate":
            for m in self.unet.modules():
                if isinstance(m, nn.Conv2d):
                    m.padding_mode = "replicate"
        if anti_checkerboard:
            _replace_conv_transpose_with_resize(self.unet)
        
        # Phase C3: Gradient checkpointing support
        # Reduces memory usage by ~50% but increases computation time by ~20-30%
        # Trade-off: Use when memory is limited or to allow larger batch sizes
        self._gradient_checkpointing_enabled = False
        if use_gradient_checkpointing:
            self.enable_gradient_checkpointing()

    def _pool_conditioning(self, conditioning: Tensor) -> Tensor:
        """Flatten conditioning [B, seq, dim] -> [B, seq*dim] for FiLM class_labels."""
        return conditioning.flatten(start_dim=1)

    def forward(
        self,
        noisy_sample: Tensor,
        timestep: Tensor,
        conditioning: Tensor,
        conditioning_spatial: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Passe avant du UNet (prédiction du bruit).

        Parameters
        ----------
        conditioning : Tensor
            Global conditioning ``[B, num_vars, dim]`` — flattened for FiLM
            ``class_labels``.
        conditioning_spatial : Optional[Tensor]
            Spatial tokens ``[B, num_tokens, dim]`` for cross-attention
            (``encoder_hidden_states``).  Falls back to *conditioning* when
            ``None``.
        """
        conditioning = self._prepare_conditioning(conditioning)
        class_labels = self._pool_conditioning(conditioning)
        hidden_states = conditioning_spatial if conditioning_spatial is not None else conditioning
        output = self.unet(
            sample=noisy_sample,
            timestep=timestep,
            encoder_hidden_states=hidden_states,
            class_labels=class_labels,
        )
        return output.sample

    def compute_loss(
        self,
        target: Tensor,
        conditioning: Tensor,
        use_focal_loss: bool = False,
        focal_alpha: float = 1.0,
        focal_gamma: float = 2.0,
        conditioning_spatial: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Calcule la perte de diffusion (MSE entre bruit réel et prédit).
        Gère les NaN dans le target en utilisant un masque (standard pour données climatiques).
        """
        # Vérifier le conditioning (ne doit pas contenir de NaN/Inf)
        if not torch.isfinite(conditioning).all():
            raise ValueError(
                f"Conditioning contains NaN/Inf: NaN={torch.isnan(conditioning).sum().item()}, "
                f"Inf={torch.isinf(conditioning).sum().item()}, "
                f"shape={conditioning.shape}, "
                f"stats: min={conditioning.min().item():.6f}, max={conditioning.max().item():.6f}"
            )
        
        # Créer un masque pour les valeurs valides dans le target
        # Les NaN peuvent représenter des masques géographiques (océan, etc.)
        valid_mask = torch.isfinite(target)
        nan_count = (~valid_mask).sum().item()
        total_pixels = target.numel()
        
        # Si tous les pixels sont NaN, retourner une loss par défaut
        if not valid_mask.any():
            return torch.tensor(0.0, device=target.device, requires_grad=True)
        
        # Remplacer temporairement les NaN par 0 pour add_noise
        # (les NaN se propagent dans noisy_sample, on les masquera après)
        target_clean = target.clone()
        target_clean[~valid_mask] = 0.0
        
        noise = torch.randn_like(target_clean)
        batch_size = target_clean.shape[0]
        timesteps = torch.randint(
            0,
            self.scheduler.num_train_timesteps,
            (batch_size,),
            device=target_clean.device,
            dtype=torch.long,
        )
        noisy_sample = self.scheduler.add_noise(target_clean, noise, timesteps)
        
        # Le masque reste valide car add_noise préserve la structure
        # (les NaN dans target_clean deviennent des valeurs, mais on utilise le masque original)
        # En fait, on doit recréer le masque car add_noise peut changer les valeurs
        # Mais comme on a remplacé les NaN par 0, le masque original reste valide
        
        noise_pred = self.forward(noisy_sample, timesteps, conditioning, conditioning_spatial=conditioning_spatial)
        
        # Vérifier que noise_pred ne contient pas de NaN/Inf (problème du modèle)
        if not torch.isfinite(noise_pred).all():
            raise ValueError(
                f"noise_pred contains NaN/Inf after UNet forward: "
                f"NaN={torch.isnan(noise_pred).sum().item()}, "
                f"Inf={torch.isinf(noise_pred).sum().item()}, "
                f"shape={noise_pred.shape}, "
                f"stats: min={noise_pred.min().item():.6f}, max={noise_pred.max().item():.6f}"
            )
        
        # Calculer la loss uniquement sur les pixels valides
        # Utiliser le masque pour filtrer les pixels NaN
        
        # Phase 3.2: Min-SNR γ-weighting for better training stability (optional)
        # This helps with training stability by downweighting high SNR timesteps
        try:
            # Access alphas_cumprod from scheduler
            if hasattr(self.scheduler, 'alphas_cumprod'):
                alphas_cumprod = self.scheduler.alphas_cumprod[timesteps]  # [batch_size]
                # Expand to match noise_pred shape for broadcasting
                for _ in range(len(noise_pred.shape) - len(alphas_cumprod.shape)):
                    alphas_cumprod = alphas_cumprod.unsqueeze(-1)
                
                # SNR = alpha^2 / (1 - alpha^2) = alpha^2 / sigma^2
                snr = alphas_cumprod / (1.0 - alphas_cumprod + 1e-8)
                # Min-SNR weighting: weight = min(SNR, 5.0) / SNR
                # This downweights very high SNR timesteps (typically > 5.0)
                min_snr_weight = torch.clamp(snr / 5.0, max=1.0)
                
                # Apply weighting only to valid pixels
                noise_error = noise_pred[valid_mask] - noise[valid_mask]
                weight_expanded = min_snr_weight.expand_as(noise_pred)[valid_mask]
                mse_error = noise_error ** 2
                weighted_error = weight_expanded * mse_error
                
                # Phase D1: Apply focal loss weighting if enabled
                # Focal loss focuses on hard pixels (high error) for better learning
                if use_focal_loss:
                    # Normalize error to [0, 1] range for focal weighting
                    # Use relative error: normalize by max error in batch
                    error_normalized = mse_error / (mse_error.max() + 1e-8)
                    # Focal weight: (error_normalized)^gamma
                    # Higher gamma = more focus on hard pixels
                    focal_weight = (error_normalized ** focal_gamma)
                    # Apply focal weighting
                    weighted_error = focal_alpha * focal_weight * weighted_error
                
                loss = weighted_error.mean()
            else:
                # Fallback to standard MSE if alphas_cumprod not available
                noise_error = noise_pred[valid_mask] - noise[valid_mask]
                mse_error = noise_error ** 2
                
                # Phase D1: Apply focal loss if enabled (even without Min-SNR)
                if use_focal_loss:
                    error_normalized = mse_error / (mse_error.max() + 1e-8)
                    focal_weight = (error_normalized ** focal_gamma)
                    loss = focal_alpha * (focal_weight * mse_error).mean()
                else:
                    loss = mse_error.mean()
        except (AttributeError, IndexError, RuntimeError):
            # Fallback to standard MSE on any error
            noise_error = noise_pred[valid_mask] - noise[valid_mask]
            mse_error = noise_error ** 2
            
            # Phase D1: Apply focal loss if enabled
            if use_focal_loss:
                error_normalized = mse_error / (mse_error.max() + 1e-8)
                focal_weight = (error_normalized ** focal_gamma)
                loss = focal_alpha * (focal_weight * mse_error).mean()
            else:
                loss = mse_error.mean()
        
        # Vérifier la loss finale
        if not torch.isfinite(loss):
            raise ValueError(
                f"Loss is NaN/Inf: loss={loss.item()}, "
                f"valid_pixels={valid_mask.sum().item()}/{total_pixels}, "
                f"noise_pred_stats: min={noise_pred[valid_mask].min().item():.6f}, max={noise_pred[valid_mask].max().item():.6f}"
            )
        
        return loss

    def predict_x0_from_epsilon(
        self,
        noisy_sample: Tensor,
        epsilon_pred: Tensor,
        timesteps: Tensor,
    ) -> Tensor:
        """DDPM : estimation x0 depuis ε_θ (pour perte spectrale RAPSD)."""
        a = self.scheduler.alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        return (noisy_sample - (1.0 - a).sqrt() * epsilon_pred) / a.sqrt().clamp(min=1e-8)

    @staticmethod
    def apply_physical_constraints(raw_output: Tensor, use_soft: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Applique les contraintes physiques : T_min <= T <= T_max.
        
        Phase 3.3: Uses soft constraints (Softplus/Mish) instead of hard ReLU
        for better gradient flow and training stability.
        
        Parameters
        ----------
        raw_output : Tensor
            Raw output [batch, 3, H, W] with channels (T_min, Δ1, Δ2)
        use_soft : bool
            If True, use soft constraints (Softplus). If False, use hard ReLU.
        
        Returns
        -------
        Tuple of (t_min, t, t_max) tensors
        """
        if raw_output.shape[1] != 3:
            raise ValueError(
                "La sortie brute doit avoir exactement 3 canaux (T_min, Δ1, Δ2)."
            )

        t_min = raw_output[:, 0:1, :, :]
        delta_1 = raw_output[:, 1:2, :, :]
        delta_2 = raw_output[:, 2:3, :, :]

        # Phase 3.3: Use soft constraints instead of hard ReLU
        if use_soft:
            # Softplus: smooth approximation of ReLU, better gradients
            # f(x) = log(1 + exp(x)) / beta, where beta controls sharpness
            softplus = nn.Softplus(beta=1.0)
            t = t_min + softplus(delta_1)
            t_max = t + softplus(delta_2)
        else:
            # Original hard constraints
            t = t_min + torch.relu(delta_1)
            t_max = t + torch.relu(delta_2)

        return t_min, t, t_max

    def sample(
        self,
        conditioning: Tensor,
        *,
        num_steps: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        baseline: Optional[Tensor] = None,
        apply_constraints: bool = True,
        scheduler_type: Optional[str] = None,
        cfg_scale: float = 0.0,
        conditioning_spatial: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        """
        Génère une sortie par diffusion conditionnée.
        
        Phase 3.2: Supports both DDPM and EDM (ODE-based) sampling.
        EDM uses fewer steps (15-50) via ODE solver for faster generation.
        """
        conditioning = self._prepare_conditioning(conditioning)
        
        # Use provided scheduler_type or default to instance setting
        scheduler_type = scheduler_type or getattr(self, 'scheduler_type', 'ddpm')
        
        # Phase 3.2: Use EDM ODE solver if requested
        # Phase E1: Use DPM-Solver++ if requested (faster than EDM)
        if scheduler_type == "edm":
            # EDM ODE solver with fewer steps
            num_steps = num_steps or 25  # Default to 25 steps for EDM (vs 1000 for DDPM)
            return self._sample_edm_ode(
                conditioning=conditioning,
                num_steps=num_steps,
                generator=generator,
                baseline=baseline,
                apply_constraints=apply_constraints,
                conditioning_spatial=conditioning_spatial,
            )
        elif scheduler_type == "dpm_solver" or scheduler_type == "dpm_solver++":
            if not HAS_DPM_SOLVER:
                raise ImportError(
                    "DPM-Solver++ is not available. Please update diffusers: "
                    "pip install --upgrade diffusers"
                )
            num_steps = num_steps or 15  # Default to 15 steps for DPM-Solver++ (vs 25 for EDM)
            return self._sample_dpm_solver(
                conditioning=conditioning,
                num_steps=num_steps,
                generator=generator,
                baseline=baseline,
                apply_constraints=apply_constraints,
                conditioning_spatial=conditioning_spatial,
            )
        
        # Original DDPM sampling
        scheduler = self.scheduler
        inference_steps = num_steps or getattr(scheduler, "num_inference_steps", None)
        if inference_steps is None:
            inference_steps = self.num_diffusion_steps
        max_train = int(
            getattr(scheduler.config, "num_train_timesteps", self.num_diffusion_steps)
        )
        inference_steps = max(1, min(int(inference_steps), max_train))
        scheduler.set_timesteps(inference_steps, device=conditioning.device)

        sample = torch.randn(
            conditioning.shape[0],
            self.in_channels,
            self.height,
            self.width,
            device=conditioning.device,
            generator=generator,
        )

        class_labels = self._pool_conditioning(conditioning)
        hidden_states = conditioning_spatial if conditioning_spatial is not None else conditioning
        for t in scheduler.timesteps:
            noise_pred_cond = self.unet(
                sample=sample,
                timestep=t,
                encoder_hidden_states=hidden_states,
                class_labels=class_labels,
            ).sample
            if cfg_scale and cfg_scale > 0.0:
                null_hs = torch.zeros_like(hidden_states)
                null_cl = torch.zeros_like(class_labels)
                noise_pred_uncond = self.unet(
                    sample=sample,
                    timestep=t,
                    encoder_hidden_states=null_hs,
                    class_labels=null_cl,
                ).sample
                model_output = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                model_output = noise_pred_cond
            sample = scheduler.step(model_output, t, sample).prev_sample

        residual = sample
        if baseline is not None:
            if baseline.shape != residual.shape:
                raise ValueError(
                    f"Baseline shape {tuple(baseline.shape)} incompatible avec résidu {tuple(residual.shape)}."
                )
            composite = baseline + residual
        else:
            composite = residual

        # Physical constraints are only defined for the special 3-channel representation:
        # (T_min, Δ1, Δ2) → (T_min, T_mean, T_max). For any other channel count,
        # we skip constraints and return the first channel for all three outputs.
        if composite.shape[1] == 3:
            if apply_constraints:
                # Phase 3.3: Use soft constraints by default
                t_min, t_mean, t_max = self.apply_physical_constraints(composite, use_soft=True)
            else:
                t_min, t_mean, t_max = (
                    composite[:, 0:1, :, :],
                    composite[:, 1:2, :, :],
                    composite[:, 2:3, :, :],
                )
        else:
            t_min = t_mean = t_max = composite[:, 0:1, :, :]
        if residual.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"Sortie diffusion {tuple(residual.shape[-2:])} != ({self.height}, {self.width})"
            )
        return DiffusionOutput(residual=residual, baseline=baseline, t_min=t_min, t_mean=t_mean, t_max=t_max)

    def _sample_edm_ode(
        self,
        conditioning: Tensor,
        num_steps: int = 25,
        generator: Optional[torch.Generator] = None,
        baseline: Optional[Tensor] = None,
        apply_constraints: bool = True,
        conditioning_spatial: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        """
        Phase 3.2: EDM (Elucidated Diffusion Models) sampling using ODE solver.
        
        Uses Euler method to solve the probability flow ODE with fewer steps (15-50)
        instead of the full 1000-step DDPM process.
        
        Parameters
        ----------
        conditioning : Tensor
            Conditioning tensor
        num_steps : int
            Number of ODE steps (15-50 recommended)
        generator : Optional[torch.Generator]
            Random number generator
        baseline : Optional[Tensor]
            Baseline to add to residual
        apply_constraints : bool
            Whether to apply physical constraints
        
        Returns
        -------
        DiffusionOutput
            Generated sample
        """
        device = conditioning.device
        
        # Initialize with noise
        sample = torch.randn(
            conditioning.shape[0],
            self.in_channels,
            self.height,
            self.width,
            device=device,
            generator=generator,
        )
        
        # Create time schedule for ODE (from t=1.0 to t=0.0)
        # Using EDM parameterization: sigma(t) = t
        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)
        dt = 1.0 / num_steps
        
        class_labels = self._pool_conditioning(conditioning)
        hidden_states = conditioning_spatial if conditioning_spatial is not None else conditioning
        for i in range(num_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            
            scheduler_timestep = (1.0 - t_current) * self.num_diffusion_steps
            scheduler_timestep = scheduler_timestep.long().clamp(0, self.num_diffusion_steps - 1)
            
            with torch.no_grad():
                noise_pred = self.unet(
                    sample=sample,
                    timestep=scheduler_timestep.expand(sample.shape[0]),
                    encoder_hidden_states=hidden_states,
                    class_labels=class_labels,
                ).sample
            
            # EDM ODE step: dx/dt = -sigma * sigma' * score
            # Simplified Euler step: x_{t+dt} = x_t - dt * sigma(t) * sigma'(t) * score
            # For linear schedule: sigma(t) = t, so sigma'(t) = 1
            sigma = t_current
            score = -noise_pred / (sigma + 1e-8)  # Convert noise prediction to score
            
            # Euler step
            sample = sample + dt * sigma * score
        
        residual = sample
        if baseline is not None:
            if baseline.shape != residual.shape:
                raise ValueError(
                    f"Baseline shape {tuple(baseline.shape)} incompatible avec résidu {tuple(residual.shape)}."
                )
            composite = baseline + residual
        else:
            composite = residual

        if composite.shape[1] == 3:
            if apply_constraints:
                # Phase 3.3: Use soft constraints by default
                t_min, t_mean, t_max = self.apply_physical_constraints(composite, use_soft=True)
            else:
                t_min, t_mean, t_max = (
                    composite[:, 0:1, :, :],
                    composite[:, 1:2, :, :],
                    composite[:, 2:3, :, :],
                )
        else:
            t_min = t_mean = t_max = composite[:, 0:1, :, :]
        return DiffusionOutput(residual=residual, baseline=baseline, t_min=t_min, t_mean=t_mean, t_max=t_max)
    
    def _sample_dpm_solver(
        self,
        conditioning: Tensor,
        num_steps: int = 15,
        generator: Optional[torch.Generator] = None,
        baseline: Optional[Tensor] = None,
        apply_constraints: bool = True,
        conditioning_spatial: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        """
        Phase E1: DPM-Solver++ sampling for ultra-fast inference.
        
        DPM-Solver++ is a high-order solver that can achieve high-quality results
        in 15-20 steps (compared to 25-50 for EDM and 1000 for DDPM).
        
        Parameters
        ----------
        conditioning : Tensor
            Conditioning tensor
        num_steps : int
            Number of sampling steps (15-20 recommended for DPM-Solver++)
        generator : Optional[torch.Generator]
            Random number generator
        baseline : Optional[Tensor]
            Baseline to add to residual
        apply_constraints : bool
            Whether to apply physical constraints
        
        Returns
        -------
        DiffusionOutput
            Generated sample
        
        References
        ----------
        - Lu et al. (2022): "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models"
        """
        if not HAS_DPM_SOLVER:
            raise ImportError(
                "DPM-Solver++ is not available. Please update diffusers: "
                "pip install --upgrade diffusers"
            )
        
        device = conditioning.device
        batch_size = conditioning.shape[0]
        
        # Initialize with noise
        sample = torch.randn(
            batch_size,
            self.in_channels,
            self.height,
            self.width,
            device=device,
            generator=generator,
        )
        
        # Create DPM-Solver scheduler (configured for fast sampling)
        dpm_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.num_diffusion_steps,
            algorithm_type="dpmsolver++",  # Use DPM-Solver++ algorithm
            solver_order=2,  # Second-order solver for balance of speed and quality
            use_karras_sigmas=True,  # Karras noise schedule for better quality
        )
        dpm_scheduler.set_timesteps(num_steps, device=device)
        
        # Sampling loop with DPM-Solver++
        for t in dpm_scheduler.timesteps:
            model_output = self.forward(sample, t, conditioning, conditioning_spatial=conditioning_spatial)
            sample = dpm_scheduler.step(model_output, t, sample, return_dict=False)[0]
        
        residual = sample
        if baseline is not None:
            if baseline.shape != residual.shape:
                raise ValueError(
                    f"Baseline shape {tuple(baseline.shape)} incompatible avec résidu {tuple(residual.shape)}."
                )
            composite = baseline + residual
        else:
            composite = residual
        
        if composite.shape[1] == 3:
            if apply_constraints:
                t_min, t_mean, t_max = self.apply_physical_constraints(composite, use_soft=True)
            else:
                t_min, t_mean, t_max = (
                    composite[:, 0:1, :, :],
                    composite[:, 1:2, :, :],
                    composite[:, 2:3, :, :],
                )
        else:
            t_min = t_mean = t_max = composite[:, 0:1, :, :]
        
        return DiffusionOutput(residual=residual, baseline=baseline, t_min=t_min, t_mean=t_mean, t_max=t_max)

    def reconstruct_from_residual(
        self,
        residual: Tensor,
        *,
        baseline: Optional[Tensor] = None,
        apply_constraints: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Combine un résidu prédictif avec un baseline et applique les contraintes physiques.
        """
        if baseline is not None:
            if baseline.shape != residual.shape:
                raise ValueError(
                    f"Baseline shape {tuple(baseline.shape)} incompatible avec résidu {tuple(residual.shape)}."
                )
            composite = baseline + residual
        else:
            composite = residual
        if composite.shape[1] == 3:
            if apply_constraints:
                # Phase 3.3: Use soft constraints by default
                return self.apply_physical_constraints(composite, use_soft=True)
            return (
                composite[:, 0:1, :, :],
                composite[:, 1:2, :, :],
                composite[:, 2:3, :, :],
            )
        t = composite[:, 0:1, :, :]
        return (t, t, t)

    def set_condition_adapter(self, adapter: Optional[Callable[[Tensor], Tensor]]) -> None:
        """
        Définit un adaptateur appliqué sur le tenseur de conditionnement avant le UNet.
        """
        self._condition_adapter = adapter

    def _prepare_conditioning(self, conditioning: Tensor) -> Tensor:
        if self._condition_adapter is not None:
            conditioning = self._condition_adapter(conditioning)
        if conditioning.dim() != 3:
            raise ValueError(
                f"Le conditionnement doit avoir la forme [batch, sequence, dim], obtenu {tuple(conditioning.shape)}."
            )
        return conditioning
```

---

### `src/st_cdgm/models/graph_builder.py`

```python
"""
Module 2 – Construction du graphe hétérogène statique pour l’architecture ST-CDGM.

Ce module fournit une classe utilitaire `HeteroGraphBuilder` qui prépare un
objet `torch_geometric.data.HeteroData` en codant les relations physiques
principales : advection (spatiale), convection (verticale) et influence
statique (topographie HR vers dynamique LR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch_geometric.data import HeteroData
except ImportError as exc:  # pragma: no cover - dépendance optionnelle
    raise ImportError(
        "torch et torch_geometric sont requis pour utiliser HeteroGraphBuilder."
    ) from exc

import xarray as xr


GridShape = Tuple[int, int]


@dataclass
class GraphBuildReport:
    """
    Informations de diagnostic sur la construction du graphe.
    """

    num_nodes_lr: int
    num_nodes_hr: int
    edges_spatial: Dict[str, int]
    edges_vertical: Dict[str, int]
    edges_static: Dict[str, int]
    hr_to_lr_parent: Sequence[int]


class HeteroGraphBuilder:
    """
    Construit un graphe hétérogène statique basé sur des grilles LR/HR.

    Parameters
    ----------
    lr_shape :
        Shape (lat, lon) de la grille basse résolution.
    hr_shape :
        Shape (lat, lon) de la grille haute résolution.
    static_dataset :
        Dataset xarray contenant les variables statiques HR (topographie, etc.).
    static_variables :
        Liste de variables statiques à intégrer. Toutes si None.
    include_mid_layer :
        Contrôle la présence de la couche intermédiaire GP500/GP250.
    """

    def __init__(
        self,
        lr_shape: GridShape,
        hr_shape: GridShape,
        *,
        static_dataset: Optional[xr.Dataset] = None,
        static_variables: Optional[Sequence[str]] = None,
        include_mid_layer: bool = True,
    ) -> None:
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.static_dataset = static_dataset
        self.static_variables = static_variables
        self.include_mid_layer = include_mid_layer

        self._validate_shapes()

        self.num_nodes_lr = self.lr_shape[0] * self.lr_shape[1]
        self.num_nodes_hr = self.hr_shape[0] * self.hr_shape[1]

        self.dynamic_node_types = ["GP850"]
        if self.include_mid_layer:
            self.dynamic_node_types.extend(["GP500", "GP250"])
        
        # Static node types (always includes SP_HR if static dataset is provided)
        self.static_node_types = ["SP_HR"] if self.static_dataset is not None else []

        self._spatial_edge_index = self._build_spatial_adjacency(self.lr_shape)
        self._vertical_edge_index = self._build_vertical_edges(self.num_nodes_lr)
        self._static_edge_index = self._build_static_influence_mapping(self.lr_shape, self.hr_shape)
        self._hr_parent_index = self._static_edge_index[1].clone()

        if self.static_dataset is not None:
            self._static_features = self._extract_static_features(self.static_dataset)
        else:
            self._static_features = torch.zeros((self.num_nodes_hr, 0), dtype=torch.float32)

        self._template_cache: Optional[HeteroData] = None
        self._report_cache: Optional[GraphBuildReport] = None

    def _validate_shapes(self) -> None:
        if len(self.lr_shape) != 2 or len(self.hr_shape) != 2:
            raise ValueError("Les shapes LR et HR doivent être de longueur 2 (lat, lon).")
        if any(dim <= 0 for dim in self.lr_shape + self.hr_shape):
            raise ValueError("Toutes les dimensions de grille doivent être positives.")
        # Note: We now support non-integer ratios by using interpolation/rounding
        # The mapping will use the nearest LR node for each HR node

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def build(self) -> Tuple[HeteroData, GraphBuildReport]:
        """
        Construit et retourne l'objet HeteroData ainsi qu'un rapport de diagnostic.
        """
        data = HeteroData()

        for node_type in self.dynamic_node_types:
            data[node_type].num_nodes = self.num_nodes_lr

        data["SP_HR"].num_nodes = self.num_nodes_hr
        data["SP_HR"].x = self._static_features.clone()

        edges_spatial: Dict[str, int] = {}
        spatial_index = self._spatial_edge_index.clone()
        data["GP850", "spat_adj", "GP850"].edge_index = spatial_index
        edges_spatial["GP850"] = spatial_index.size(1)
        if self.include_mid_layer:
            data["GP500", "spat_adj", "GP500"].edge_index = spatial_index.clone()
            data["GP250", "spat_adj", "GP250"].edge_index = spatial_index.clone()
            edges_spatial["GP500"] = spatial_index.size(1)
            edges_spatial["GP250"] = spatial_index.size(1)

        edges_vertical: Dict[str, int] = {}
        if self.include_mid_layer:
            vert_edge = self._vertical_edge_index.clone()
            data["GP850", "vert_adj", "GP500"].edge_index = vert_edge
            data["GP500", "vert_adj", "GP850"].edge_index = vert_edge[[1, 0], :]

            data["GP500", "vert_adj", "GP250"].edge_index = vert_edge.clone()
            data["GP250", "vert_adj", "GP500"].edge_index = vert_edge[[1, 0], :]

            edges_vertical["GP850↔GP500"] = vert_edge.size(1) * 2
            edges_vertical["GP500↔GP250"] = vert_edge.size(1) * 2

        edges_static: Dict[str, int] = {}
        static_edge_index = self._static_edge_index.clone()
        data["SP_HR", "causes", "GP850"].edge_index = static_edge_index
        edges_static["SP_HR→GP850"] = static_edge_index.size(1)

        if self.include_mid_layer:
            data["SP_HR", "causes", "GP500"].edge_index = static_edge_index.clone()
            data["SP_HR", "causes", "GP250"].edge_index = static_edge_index.clone()
            edges_static["SP_HR→GP500"] = static_edge_index.size(1)
            edges_static["SP_HR→GP250"] = static_edge_index.size(1)

        self._assign_default_batch(data)

        report = GraphBuildReport(
            num_nodes_lr=self.num_nodes_lr,
            num_nodes_hr=self.num_nodes_hr,
            edges_spatial=edges_spatial,
            edges_vertical=edges_vertical,
            edges_static=edges_static,
            hr_to_lr_parent=self._hr_parent_index.tolist(),
        )
        self._validate_edge_ranges(data)

        self._template_cache = data.clone()
        self._report_cache = report

        return data, report

    def build_template(self) -> HeteroData:
        """Return a fresh clone of the cached template graph."""
        if self._template_cache is None:
            template, _ = self.build()
            self._template_cache = template.clone()
        return self._template_cache.clone()

    def get_report(self) -> GraphBuildReport:
        if self._report_cache is None:
            _, report = self.build()
            self._report_cache = report
        return self._report_cache

    def prepare_step_data(
        self,
        features: Dict[str, torch.Tensor],
        *,
        clone_template: bool = False,
    ) -> HeteroData:
        """
        Retourne un HeteroData prêt à l'emploi avec les features dynamiques injectées.
        
        Phase 2.1: Optimized to avoid full graph cloning by default.
        The template graph (edge_index, static features) is reused, only dynamic
        node features are updated in-place.
        
        Parameters
        ----------
        features : Dict[str, torch.Tensor]
            Dynamic node features to inject (e.g., atmospheric variables per timestep)
        clone_template : bool
            If True, creates a full deep copy of the template (slower but safe if you
            need to modify the graph structure). If False (default), reuses the template
            and only updates dynamic features in-place (faster, recommended for training).
        
        Returns
        -------
        HeteroData
            Graph with dynamic features injected
        """
        if self._template_cache is None:
            self._template_cache = self.build_template()
        
        # Phase 2.1: Only clone if explicitly requested (for backward compatibility)
        # Otherwise, reuse template and inject features in-place (much faster)
        if clone_template:
            data = self._template_cache.clone()
        else:
            data = self._template_cache
        
        self.inject_dynamic_features(data, features)
        return data

    def inject_dynamic_features(self, data: HeteroData, features: Dict[str, torch.Tensor]) -> None:
        """
        Injecte des features nodales dynamiques (par pas de temps).
        
        Phase 2.1: Modifies node features in-place. This is safe when called on a
        cloned template, or when the input tensor is a new tensor (not a view of
        previously injected data).
        
        Note: If you're reusing the same HeteroData instance (clone_template=False),
        ensure that the input tensors are new tensors, not views/slices of previously
        injected tensors, to avoid unintended side effects.
        """
        for node_type, tensor in features.items():
            if node_type not in data.node_types:
                raise KeyError(f"Node type '{node_type}' absent du graphe.")
            expected_nodes = data[node_type].num_nodes
            if tensor.shape[0] != expected_nodes:
                raise ValueError(
                    f"Features pour '{node_type}' incompatibles: "
                    f"{tensor.shape[0]} nœuds fournis, {expected_nodes} attendus."
                )
            # Phase 2.1: In-place modification - safe as long as input tensor is new
            # or we're working on a cloned template
            data[node_type].x = tensor

    def get_hr_to_lr_parent_index(self) -> torch.Tensor:
        """Retourne le mapping hr->lr (1D tensor de taille num_nodes_hr)."""
        return self._hr_parent_index.clone()

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def lr_grid_to_nodes(
        self,
        grid: torch.Tensor,
        *,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """
        Convertit un champ LR (C,H,W) ou (H,W,C) en matrice nodale [N_lr, C].
        """
        if channel_last:
            grid = grid.permute(2, 0, 1)
        if grid.dim() != 3:
            raise ValueError("Le tenseur LR doit être de dimension 3 (C,H,W).")
        c, h, w = grid.shape
        if (h, w) != self.lr_shape:
            raise ValueError(f"Shape LR attendu {self.lr_shape}, obtenu {(h, w)}")
        nodes = grid.reshape(c, -1).transpose(0, 1).contiguous()
        return nodes

    def hr_grid_to_nodes(
        self,
        grid: torch.Tensor,
        *,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """
        Convertit un champ HR (C,H,W) ou (H,W,C) en matrice nodale [N_hr, C].
        """
        if channel_last:
            grid = grid.permute(2, 0, 1)
        if grid.dim() != 3:
            raise ValueError("Le tenseur HR doit être de dimension 3 (C,H,W).")
        c, h, w = grid.shape
        if (h, w) != self.hr_shape:
            raise ValueError(f"Shape HR attendu {self.hr_shape}, obtenu {(h, w)}")
        nodes = grid.reshape(c, -1).transpose(0, 1).contiguous()
        return nodes

    def hr_nodes_to_grid(
        self,
        nodes: torch.Tensor,
        *,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """
        Convertit un tenseur nodal HR [N_hr, C] en champ grille (C,H,W) ou (H,W,C).
        """
        if nodes.dim() != 2 or nodes.size(0) != self.num_nodes_hr:
            raise ValueError(
                f"Tenseur nodal HR incompatible, attendu ({self.num_nodes_hr}, C), obtenu {tuple(nodes.shape)}"
            )
        c = nodes.size(1)
        grid = nodes.transpose(0, 1).reshape(c, self.hr_shape[0], self.hr_shape[1]).contiguous()
        if channel_last:
            grid = grid.permute(1, 2, 0)
        return grid

    def expand_lr_nodes_to_hr(self, lr_nodes: torch.Tensor) -> torch.Tensor:
        """
        Broadcast des features LR [N_lr, C] sur la grille HR via le mapping parent.
        """
        if lr_nodes.dim() != 2 or lr_nodes.size(0) != self.num_nodes_lr:
            raise ValueError(
                f"Tenseur nodal LR incompatible, attendu ({self.num_nodes_lr}, C), obtenu {tuple(lr_nodes.shape)}"
            )
        expanded = lr_nodes[self._hr_parent_index]
        return expanded.contiguous()

    def _assign_default_batch(self, data: HeteroData) -> None:
        """Initialise le vecteur batch (nécessaire pour le pooling global)."""
        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            data[node_type].batch = torch.zeros(num_nodes, dtype=torch.long)

    # ------------------------------------------------------------------
    # Construction des arêtes
    # ------------------------------------------------------------------
    @staticmethod
    def _build_spatial_adjacency(shape: GridShape) -> torch.Tensor:
        """
        Retourne l'edge_index (2, E) pour la connectivité 8-voisins.
        """
        lat, lon = shape
        indices = []
        offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        for i in range(lat):
            for j in range(lon):
                src = i * lon + j
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < lat and 0 <= nj < lon:
                        dst = ni * lon + nj
                        indices.append((src, dst))

        edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
        return edge_index

    @staticmethod
    def _build_vertical_edges(num_nodes: int) -> torch.Tensor:
        """
        Crée des arêtes verticales (mapping identique entre deux couches).
        """
        indices = torch.arange(num_nodes, dtype=torch.long)
        edge_index = torch.stack([indices, indices], dim=0)
        return edge_index

    @staticmethod
    def _build_static_influence_mapping(
        lr_shape: GridShape,
        hr_shape: GridShape,
    ) -> torch.Tensor:
        """
        Mappe chaque nœud HR vers son nœud LR parent le plus proche.
        Supporte les ratios non-entiers en utilisant une interpolation par arrondi.
        """
        # Calculate actual ratios (may be non-integer)
        ratio_lat = hr_shape[0] / lr_shape[0]
        ratio_lon = hr_shape[1] / lr_shape[1]
        
        indices = []

        for i_hr in range(hr_shape[0]):
            for j_hr in range(hr_shape[1]):
                # Map HR coordinates to LR coordinates using the ratio
                # Use rounding to find the nearest LR node
                i_lr = int(round(i_hr / ratio_lat))
                j_lr = int(round(j_hr / ratio_lon))
                
                # Clamp to valid LR indices
                i_lr = max(0, min(i_lr, lr_shape[0] - 1))
                j_lr = max(0, min(j_lr, lr_shape[1] - 1))
                
                hr_idx = i_hr * hr_shape[1] + j_hr
                lr_idx = i_lr * lr_shape[1] + j_lr
                indices.append((hr_idx, lr_idx))

        edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
        return edge_index

    # ------------------------------------------------------------------
    # Gestion des features statiques
    # ------------------------------------------------------------------
    def _extract_static_features(self, dataset: xr.Dataset) -> torch.Tensor:
        """
        Transforme les variables statiques en un tenseur torch [num_nodes_hr, num_features].
        """
        vars_to_use: Iterable[str]
        if self.static_variables is None:
            # Filter out bounds variables (common in climate data)
            vars_to_use = [
                var for var in dataset.data_vars.keys()
                if "bnds" not in str(var).lower() and "bounds" not in str(var).lower()
            ]
        else:
            vars_to_use = self.static_variables

        features = []
        for var in vars_to_use:
            if var not in dataset:
                raise KeyError(f"Variable statique '{var}' absente du dataset.")
            arr = dataset[var].values
            
            # Skip variables that don't have the expected spatial shape
            # (e.g., bounds variables, or variables with extra dimensions)
            if arr.ndim < 2:
                continue  # Skip 1D variables
            
            # Check if the last two dimensions match hr_shape
            if arr.shape[-2:] != self.hr_shape:
                # Try to squeeze singleton dimensions
                arr_squeezed = arr.squeeze()
                if arr_squeezed.ndim >= 2 and arr_squeezed.shape[-2:] == self.hr_shape:
                    arr = arr_squeezed
                else:
                    # Skip variables that don't match the expected shape
                    continue
            
            features.append(arr.reshape(-1))

        if not features:
            return torch.zeros(
                (self.hr_shape[0] * self.hr_shape[1], 0), dtype=torch.float32
            )

        stacked = np.stack(features, axis=-1)
        tensor = torch.from_numpy(stacked.astype(np.float32))
        return tensor

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_edge_ranges(data: HeteroData) -> None:
        """
        Vérifie que les indices d'arêtes sont dans les bornes pour chaque type.
        """
        for key, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = key
            src_nodes = data[src_type].num_nodes
            dst_nodes = data[dst_type].num_nodes
            if edge_index.numel() == 0:
                continue
            if edge_index.min().item() < 0:
                raise ValueError(f"Indices négatifs détectés pour {key}.")
            if edge_index[0].max().item() >= src_nodes:
                raise ValueError(f"Indice source hors bornes pour {key}.")
            if edge_index[1].max().item() >= dst_nodes:
                raise ValueError(f"Indice destination hors bornes pour {key}.")
```

---

### `src/st_cdgm/models/intelligible_encoder.py`

```python
"""
Module 3 – Encodeur de variables intelligibles via HeteroConv.

Ce module fournit une classe `IntelligibleVariableEncoder` qui agrège les
informations d'un `HeteroData` en suivant différents méta-chemins (advection,
convection, influence statique, etc.) afin de produire un état caché initial
`H(0)` pour le réseau causal récurrent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    HeteroConv,
    MessagePassing,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
)


MetaPath = Tuple[str, str, str]


@dataclass
class IntelligibleVariableConfig:
    """
    Configuration d'une variable intelligible.

    Attributes
    ----------
    name :
        Nom de la variable intelligible (ex: 'h_advection').
    meta_path :
        Méta-chemin torch_geometric (source_type, relation_type, target_type).
    conv_class :
        Classe de convolution à utiliser; défaut `SAGEConv`.
    conv_kwargs :
        Paramètres additionnels passés au constructeur de la convolution.
    pool :
        Mode de pooling à appliquer en sortie ("mean", "max", None) lors du
        calcul des états agrégés pour le conditionnement.
    """

    name: str
    meta_path: MetaPath
    conv_class: type[MessagePassing] = SAGEConv
    conv_kwargs: Optional[Dict] = None
    pool: Optional[str] = None


class IntelligibleVariableEncoder(nn.Module):
    """
    Encodeur HeteroConv produisant les variables intelligibles pour H(0).
    """

    def __init__(
        self,
        configs: Iterable[IntelligibleVariableConfig],
        hidden_dim: int,
        *,
        activation: Optional[nn.Module] = None,
        use_layer_norm: bool = True,
        default_pool: str = "mean",
        conditioning_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation or nn.ReLU()
        self.use_layer_norm = use_layer_norm
        self.default_pool = default_pool

        self.configs: List[IntelligibleVariableConfig] = list(configs)
        if not self.configs:
            raise ValueError("Au moins une configuration de variable intelligible est requise.")

        convs_dict = {}
        for cfg in self.configs:
            kwargs = {"out_channels": hidden_dim}
            if cfg.conv_class is SAGEConv:
                kwargs["in_channels"] = (-1, -1)  # auto-infer
            if cfg.conv_kwargs:
                kwargs.update(cfg.conv_kwargs)
            convs_dict[cfg.meta_path] = cfg.conv_class(**kwargs)

        self.hetero_conv = HeteroConv(convs_dict, aggr="sum")
        
        # Phase B1: Check if pyg-lib is available for Grouped GEMM optimizations
        self._check_pyg_lib_availability()

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = nn.Identity()

        if conditioning_dim is None or conditioning_dim == hidden_dim:
            self.conditioning_dim = hidden_dim
            self.conditioning_projection = nn.Identity()
        else:
            self.conditioning_dim = conditioning_dim
            self.conditioning_projection = nn.Linear(hidden_dim, conditioning_dim)

    def forward(self, data: HeteroData, *, pooled: bool = False) -> Dict[str, Tensor]:
        """
        Applique l'encodeur et retourne un dict {variable_name: embeddings}.
        Si ``pooled=True``, les embeddings sont agrégés par graphe (global pooling).
        """
        x_dict = {node_type: data[node_type].x for node_type in data.node_types}
        embeddings = self.hetero_conv(x_dict, data.edge_index_dict)

        outputs: Dict[str, Tensor] = {}
        for cfg in self.configs:
            tensor = embeddings[cfg.meta_path[-1]]
            tensor = self.layer_norm(tensor)
            tensor = self.activation(tensor)

            if pooled:
                pool_type = cfg.pool or self.default_pool
                batch_attr = getattr(data[cfg.meta_path[-1]], "batch", None)
                if batch_attr is None:
                    batch_attr = torch.zeros(tensor.size(0), dtype=torch.long, device=tensor.device)
                tensor = self._apply_pooling(tensor, batch_attr, pool_type)

            outputs[cfg.name] = tensor

        return outputs

    def init_state(self, data: HeteroData) -> Tensor:
        """
        Génère l'état initial H(0) comme tenseur [q, num_nodes, hidden_dim].
        """
        embeddings = self.forward(data, pooled=False)
        tensors = [embeddings[cfg.name] for cfg in self.configs]
        aligned = []

        for tensor, cfg in zip(tensors, self.configs):
            if tensor.dim() == 2:
                aligned.append(tensor)
            elif tensor.dim() == 3:
                aligned.append(tensor.squeeze(0))
            else:
                raise ValueError(
                    f"Embedding pour {cfg.name} a une dimension inattendue: {tensor.shape}"
                )

        stacked = torch.stack(aligned, dim=0)
        return stacked

    def pooled_state(self, data: HeteroData) -> Tensor:
        """
        Retourne un tenseur [batch, q, hidden_dim] agrégé par graphe.
        """
        pooled_embeddings = self.forward(data, pooled=True)
        tensors = []
        for cfg in self.configs:
            tensor = pooled_embeddings[cfg.name]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        stacked = torch.stack(tensors, dim=1)
        return stacked

    def project_conditioning(self, data: HeteroData) -> Tensor:
        """
        Projette l'état causal agrégé dans l'espace de conditionnement diffusion.

        Returns
        -------
        Tensor
            Tensor de forme [batch, q, conditioning_dim]
        """
        pooled = self.pooled_state(data)
        batch, q, hidden = pooled.shape
        projected = self.conditioning_projection(pooled.view(batch * q, hidden))
        projected = projected.view(batch, q, -1)
        return projected

    def project_state_tensor(
        self,
        state: Tensor,
        *,
        batch_index: Optional[Tensor] = None,
        pool: Optional[str] = None,
    ) -> Tensor:
        """
        Convertit un tenseur d'état causal H(t) en représentation cross-attention.

        Parameters
        ----------
        state :
            Tenseur de forme [q, N, hidden] (un graphe) ou [batch, q, N, hidden].
        batch_index :
            Assignation des nœuds à chaque graphe (longueur N). Optionnel si ``state``
            contient déjà une dimension batch explicite.
        pool :
            Mode de pooling à appliquer ("mean" par défaut, "max" accepté).

        Returns
        -------
        Tensor
            Tenseur [batch, q, conditioning_dim] prêt pour cross-attention.
        """
        pool_type = (pool or self.default_pool).lower()

        if state.dim() == 4:
            # state: [batch, q, N, hidden]
            if pool_type == "max":
                pooled = state.max(dim=2).values
            else:
                pooled = state.mean(dim=2)
        elif state.dim() == 3:
            # state: [q, N, hidden]
            if batch_index is None:
                if pool_type == "max":
                    pooled = state.max(dim=1).values.unsqueeze(0)
                else:
                    pooled = state.mean(dim=1).unsqueeze(0)
            else:
                num_batches = int(batch_index.max().item()) + 1
                batch_chunks = []
                for b in range(num_batches):
                    mask = batch_index == b
                    if not torch.any(mask):
                        if pool_type == "max":
                            aggregated = state.max(dim=1).values
                        else:
                            aggregated = state.mean(dim=1)
                    else:
                        selected = state[:, mask, :]
                        if pool_type == "max":
                            aggregated = selected.max(dim=1).values
                        else:
                            aggregated = selected.mean(dim=1)
                    batch_chunks.append(aggregated)
                pooled = torch.stack(batch_chunks, dim=0)
        else:
            raise ValueError(f"Tenseur d'état inattendu de forme {tuple(state.shape)}.")

        batch, q, hidden = pooled.shape
        projected = self.conditioning_projection(pooled.view(batch * q, hidden))
        projected = projected.view(batch, q, -1)
        return projected

    def reset_parameters(self) -> None:
        """
        Réinitialise les poids des convolutions.
        """
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _check_pyg_lib_availability(self) -> None:
        """
        Vérifie si pyg-lib est disponible pour les optimisations Grouped GEMM.
        Log un message informatif si disponible.
        """
        try:
            import pyg_lib
            # pyg-lib est disponible, HeteroConv bénéficiera automatiquement des optimisations
            # Pas besoin de configuration supplémentaire
            pass
        except ImportError:
            # pyg-lib n'est pas disponible, ce n'est pas critique
            # HeteroConv fonctionnera normalement sans les optimisations Grouped GEMM
            pass

    def _apply_pooling(self, tensor: Tensor, batch: Tensor, pool_type: Optional[str]) -> Tensor:
        pool_type = (pool_type or "").lower()
        if pool_type in {"mean", "avg"}:
            return global_mean_pool(tensor, batch)
        if pool_type == "max":
            return global_max_pool(tensor, batch)
        if pool_type in {"", "none", None}:
            return tensor
        raise ValueError(f"Pooling '{pool_type}' non pris en charge.")


class SpatialConditioningProjector(nn.Module):
    """Projects RCN state [q, N, hidden] into spatially-compressed conditioning tokens.

    The state is reshaped onto the LR grid, adaptively pooled to a small target
    spatial resolution, then projected to ``conditioning_dim``.  The output has
    shape ``[batch, num_vars * target_h * target_w, conditioning_dim]`` and can
    be passed directly as ``encoder_hidden_states`` to a UNet with cross-attention.
    """

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        conditioning_dim: int,
        lr_shape: Tuple[int, int],
        target_shape: Tuple[int, int] = (6, 7),
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.lr_shape = lr_shape
        self.target_shape = target_shape
        self.adaptive_pool = nn.AdaptiveAvgPool2d(target_shape)
        self.proj = nn.Linear(hidden_dim, conditioning_dim)
        self.norm = nn.LayerNorm(conditioning_dim)

    def forward(self, state: Tensor, *, batch_index: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        state : Tensor
            RCN hidden state of shape ``[q, N, hidden]``.
        batch_index : Optional[Tensor]
            Not used (single-graph path). Reserved for future multi-graph batching.

        Returns
        -------
        Tensor
            ``[batch, num_vars * th * tw, conditioning_dim]``
        """
        q, N, d = state.shape
        lat, lon = self.lr_shape
        if N != lat * lon:
            raise ValueError(
                f"SpatialConditioningProjector: expected N={lat*lon} "
                f"(lr_shape={self.lr_shape}), got N={N}."
            )
        grid = state.view(q, lat, lon, d).permute(0, 3, 1, 2)    # [q, d, lat, lon]
        pooled = self.adaptive_pool(grid)                          # [q, d, th, tw]
        th, tw = self.target_shape
        tokens = pooled.permute(0, 2, 3, 1).reshape(q * th * tw, d)  # [q*th*tw, d]
        projected = self.norm(self.proj(tokens))                   # [q*th*tw, cond_dim]
        return projected.unsqueeze(0)                              # [1, q*th*tw, cond_dim]
```

---

### `src/st_cdgm/training/__init__.py`

```python
"""
Modules d'entraînement pour ST-CDGM.
"""

from .training_loop import train_epoch
from .callbacks import EarlyStopping

__all__ = [
    "train_epoch",
    "EarlyStopping",
]
```

---

### `src/st_cdgm/training/callbacks.py`

```python
"""
Phase C2: Training callbacks for Early Stopping and Learning Rate Scheduling.

This module provides utilities for monitoring training and automatically
stopping when validation loss stops improving, as well as adaptive learning
rate scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class EarlyStopping:
    """
    Phase C2: Early stopping callback to stop training when validation loss stops improving.
    
    Monitors validation loss and stops training if no improvement is seen for
    a specified number of epochs (patience). Optionally restores the best model.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping if no improvement
    min_delta : float
        Minimum change in loss to qualify as an improvement
    mode : str
        'min' for loss (lower is better) or 'max' for metrics (higher is better)
    restore_best : bool
        If True, restores the best model weights at the end
    verbose : bool
        If True, prints messages when stopping or restoring
    """
    
    patience: int = 7
    min_delta: float = 0.0
    mode: str = "min"
    restore_best: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_weights: Optional[dict] = None
        self.early_stop = False
        
        if self.mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Parameters
        ----------
        score : float
            Current validation score (loss or metric)
        model : torch.nn.Module
            Model to monitor (weights will be saved if best)
        
        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, score)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model, score)
            if self.verbose:
                print(f"✓ EarlyStopping: New best score: {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.6f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training (patience={self.patience} reached)")
                if self.restore_best and self.best_weights is not None:
                    self._restore_checkpoint(model)
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "min":
            return current < (best - self.min_delta)
        else:  # mode == "max"
            return current > (best + self.min_delta)
    
    def _save_checkpoint(self, model: torch.nn.Module, score: float) -> None:
        """Save model weights checkpoint."""
        self.best_weights = {
            'state_dict': model.state_dict().copy(),
            'score': score,
        }
    
    def _restore_checkpoint(self, model: torch.nn.Module) -> None:
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights['state_dict'])
            if self.verbose:
                print(f"✓ EarlyStopping: Restored best model (score: {self.best_weights['score']:.6f})")
    
    def reset(self) -> None:
        """Reset early stopping state (useful for new training runs)."""
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
```

---

### `src/st_cdgm/training/training_loop.py`

```python
"""
Module 6 – Boucle d'entraînement pour l'architecture ST-CDGM.

Ce module assemble les pertes (diffusion, reconstruction, NO TEARS) et fournit
une routine d'entraînement par epoch qui enchaîne les modules précédents :
encodeur de variables intelligibles, RCN et décodeur de diffusion.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import time

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP

from ..models.causal_rcn import RCNSequenceRunner
from ..models.diffusion_decoder import CausalDiffusionDecoder
from ..models.intelligible_encoder import IntelligibleVariableEncoder


def _train_autocast(amp_mode: str):
    """
    Mixed precision context for train_epoch: CUDA FP16 (with GradScaler) or CPU BF16 (no scaler).
    amp_mode: "none" | "cuda_fp16" | "cpu_bf16"
    """
    if amp_mode == "cuda_fp16":
        return torch.cuda.amp.autocast()
    if amp_mode == "cpu_bf16":
        return torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16)
    return nullcontext()


def resolve_train_amp_mode(device: torch.device, use_amp: bool) -> str:
    """Même logique que le début de ``train_epoch`` (pour métrique RAPSD fin d’époque)."""
    if not use_amp:
        return "none"
    if device.type == "cuda" and torch.cuda.is_available():
        return "cuda_fp16"
    if device.type == "cpu":
        bf16_ok = getattr(torch.cpu, "is_bf16_supported", None)
        if bf16_ok is not None and bf16_ok():
            return "cpu_bf16"
    return "none"


def loss_reconstruction(
    pred: Optional[Tensor],
    target: Optional[Tensor],
    loss_type: str = "mse",  # Phase D4: Loss type ("mse", "cosine", "mse+cosine")
) -> Tensor:
    """
    Perte de reconstruction L_rec.
    Phase D4: Supports multiple loss types (MSE, cosine similarity, or combined).
    
    Parameters
    ----------
    pred : Optional[Tensor]
        Predicted reconstruction
    target : Optional[Tensor]
        Target reconstruction
    loss_type : str
        Phase D4: Type of loss ("mse", "cosine", "mse+cosine")
    
    Returns
    -------
    Tensor
        Reconstruction loss (0 if pred or target is None)
    """
    if pred is None or target is None:
        return torch.zeros((), device=pred.device if pred is not None else "cpu")
    
    # Phase D4: Cosine similarity loss
    if loss_type == "cosine":
        # Flatten for cosine similarity computation
        pred_flat = pred.flatten(start_dim=1)  # [batch, features]
        target_flat = target.flatten(start_dim=1)  # [batch, features]
        
        # Compute cosine similarity: cos(θ) = (A·B) / (||A|| ||B||)
        dot_product = (pred_flat * target_flat).sum(dim=1)  # [batch]
        pred_norm = pred_flat.norm(dim=1)  # [batch]
        target_norm = target_flat.norm(dim=1)  # [batch]
        
        # Avoid division by zero
        epsilon = 1e-8
        cosine_sim = dot_product / (pred_norm * target_norm + epsilon)
        
        # Convert similarity to loss: 1 - cosine_similarity
        # Cosine sim ranges from -1 to 1, we want loss from 0 to 2
        loss = (1.0 - cosine_sim).mean()
        return loss
    
    # Phase D4: Combined MSE + Cosine
    elif loss_type == "mse+cosine":
        mse_loss = nn.functional.mse_loss(pred, target)
        
        # Cosine similarity component
        pred_flat = pred.flatten(start_dim=1)
        target_flat = target.flatten(start_dim=1)
        dot_product = (pred_flat * target_flat).sum(dim=1)
        pred_norm = pred_flat.norm(dim=1)
        target_norm = target_flat.norm(dim=1)
        epsilon = 1e-8
        cosine_sim = dot_product / (pred_norm * target_norm + epsilon)
        cosine_loss = (1.0 - cosine_sim).mean()
        
        # Combine with equal weights
        return 0.5 * mse_loss + 0.5 * cosine_loss
    
    # Default: MSE loss
    else:  # loss_type == "mse"
        return nn.functional.mse_loss(pred, target)


def compute_rapsd_loss(
    pred: Tensor,
    target: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    """
    Perte spectrale radiale (RAPSD) — compare log-puissance entre prédiction et cible.
    pred, target : [B, C, H, W]
    """
    if pred.shape != target.shape:
        raise ValueError(f"RAPSD: shapes {pred.shape} vs {target.shape}")
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    B, C, H, W = pred.shape
    losses: List[Tensor] = []
    for b in range(B):
        for c in range(C):
            p = pred[b, c]
            t = target[b, c]
            fft_p = torch.fft.fftshift(torch.fft.fft2(p))
            fft_t = torch.fft.fftshift(torch.fft.fft2(t))
            psd_p = torch.abs(fft_p) ** 2
            psd_t = torch.abs(fft_t) ** 2
            cy, cx = H // 2, W // 2
            y_idx = torch.arange(H, device=pred.device, dtype=torch.float32) - cy
            x_idx = torch.arange(W, device=pred.device, dtype=torch.float32) - cx
            yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
            r = torch.sqrt(xx ** 2 + yy ** 2).long().clamp(min=0)
            max_r = int(r.max().item()) + 1
            rapsd_p = torch.zeros(max_r, device=pred.device)
            rapsd_t = torch.zeros(max_r, device=pred.device)
            counts = torch.zeros(max_r, device=pred.device)
            rf = r.flatten()
            rapsd_p.scatter_add_(0, rf, psd_p.flatten())
            rapsd_t.scatter_add_(0, rf, psd_t.flatten())
            counts.scatter_add_(0, rf, torch.ones_like(rf, dtype=torch.float32))
            valid = counts > 0
            rapsd_p[valid] /= counts[valid]
            rapsd_t[valid] /= counts[valid]
            log_ratio = torch.log(rapsd_p[valid] + eps) - torch.log(rapsd_t[valid] + eps)
            losses.append((log_ratio ** 2).mean())
    return torch.stack(losses).mean()


@torch.no_grad()
def compute_rapsd_spectral_value_no_grad(
    diffusion_decoder: CausalDiffusionDecoder,
    target: Tensor,
    conditioning: Tensor,
    *,
    amp_mode: str = "none",
) -> Tensor:
    """
    Une passe bruit → ε_θ → x̂₀ puis perte RAPSD, sans gradient.
    Utilisé en fin d’époque (métrique) pour éviter FFT/scatter dans la boucle batch.
    """
    noise_sp = torch.randn_like(target)
    bs = target.shape[0]
    timesteps_sp = torch.randint(
        0,
        diffusion_decoder.scheduler.num_train_timesteps,
        (bs,),
        device=target.device,
        dtype=torch.long,
    )
    noisy_sp = diffusion_decoder.scheduler.add_noise(target, noise_sp, timesteps_sp)
    with _train_autocast(amp_mode):
        noise_pred_sp = diffusion_decoder.forward(noisy_sp, timesteps_sp, conditioning)
    pred_x0 = diffusion_decoder.predict_x0_from_epsilon(noisy_sp, noise_pred_sp, timesteps_sp)
    return compute_rapsd_loss(pred_x0, target)


def prepare_target_and_conditioning_for_metric(
    batch: Dict[str, Any],
    encoder: IntelligibleVariableEncoder,
    rcn_runner: RCNSequenceRunner,
    *,
    device: torch.device,
    residual_key: str = "residual",
    batch_index_key: str = "batch_index",
    conditioning_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    amp_mode: str = "none",
) -> Optional[Tuple[Tensor, Tensor]]:
    """
    Reproduit le chemin encodeur → RCN → conditioning et la cible HR du train_epoch,
    pour un seul micro-batch (métrique RAPSD en fin d’époque).
    """
    lr_data: Tensor = batch["lr"].to(device)
    target_data: Tensor = batch.get(residual_key, batch.get("hr")).to(device)

    if torch.isnan(target_data).any():
        nan_fill = torch.nanmean(target_data).item()
        if not math.isfinite(nan_fill):
            nan_fill = 0.0
        target_data = torch.nan_to_num(target_data, nan=nan_fill)
    if torch.isinf(target_data).any():
        valid_mean = target_data[~(torch.isnan(target_data) | torch.isinf(target_data))].mean().item() if target_data.numel() > 0 else 0.0
        target_data = torch.nan_to_num(target_data, nan=valid_mean, posinf=valid_mean, neginf=valid_mean)
    if torch.isnan(lr_data).any():
        nan_fill = torch.nanmean(lr_data).item()
        if not math.isfinite(nan_fill):
            nan_fill = 0.0
        lr_data = torch.nan_to_num(lr_data, nan=nan_fill)

    hetero_data = batch["hetero"]
    with _train_autocast(amp_mode):
        H_init = encoder.init_state(hetero_data).to(device)
    drivers = [lr_data[t] for t in range(lr_data.shape[0])]
    with _train_autocast(amp_mode):
        seq_output = rcn_runner.run(H_init, drivers, reconstruction_sources=None)

    H_condition = seq_output.states[-1]
    batch_index = batch.get(batch_index_key)
    if batch_index is not None:
        batch_index = batch_index.to(device)
    if conditioning_fn is None:
        conditioning = encoder.project_state_tensor(H_condition, batch_index=batch_index)
    else:
        conditioning = conditioning_fn(H_condition, batch_index)
    conditioning = conditioning.to(device)

    if torch.isnan(conditioning).any() or torch.isinf(conditioning).any():
        return None

    target = target_data[-1]
    if target.dim() == 3:
        target = target.unsqueeze(0)
    elif target.dim() != 4:
        return None

    valid_mask_ts = batch.get("valid_mask")
    if valid_mask_ts is not None:
        vm = valid_mask_ts[-1].to(device=device)
        if vm.dtype != torch.bool:
            vm = vm > 0.5
        while vm.dim() < target.dim():
            vm = vm.unsqueeze(0)
        vm = vm.expand_as(target)
        target = target.clone().masked_fill(~vm, float("nan"))

    return target, conditioning


def compute_rapsd_metric_from_batch(
    *,
    encoder: IntelligibleVariableEncoder,
    rcn_runner: RCNSequenceRunner,
    diffusion_decoder: CausalDiffusionDecoder,
    batch: Dict[str, Any],
    device: torch.device,
    residual_key: str = "residual",
    batch_index_key: str = "batch_index",
    conditioning_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    amp_mode: str = "none",
    verbose: bool = False,
) -> Optional[float]:
    """
    Calcule la métrique RAPSD (scalaire) sur un batch, sans gradient.
    À appeler une fois par époque après ``train_epoch`` (ex. premier batch du loader).
    """
    enc = encoder.module if isinstance(encoder, DDP) else encoder
    rcn_cell = rcn_runner.cell.module if isinstance(rcn_runner.cell, DDP) else rcn_runner.cell
    diff = diffusion_decoder.module if isinstance(diffusion_decoder, DDP) else diffusion_decoder

    was_enc_train = enc.training
    was_rcn_train = rcn_cell.training
    was_diff_train = diff.training
    enc.eval()
    rcn_cell.eval()
    diff.eval()
    try:
        prepared = prepare_target_and_conditioning_for_metric(
            batch,
            enc,
            rcn_runner,
            device=device,
            residual_key=residual_key,
            batch_index_key=batch_index_key,
            conditioning_fn=conditioning_fn,
            amp_mode=amp_mode,
        )
        if prepared is None:
            return None
        target, conditioning = prepared
        if target.shape[1] != diff.in_channels:
            if verbose:
                print(f"[RAPSD metric] Channel mismatch: target {target.shape[1]} vs UNet {diff.in_channels}")
            return None
        val = compute_rapsd_spectral_value_no_grad(diff, target, conditioning, amp_mode=amp_mode)
        return float(val.item())
    finally:
        if was_enc_train:
            enc.train()
        if was_rcn_train:
            rcn_cell.train()
        if was_diff_train:
            diff.train()


def loss_diffusion(
    decoder: CausalDiffusionDecoder,
    target: Tensor,
    conditioning: Tensor,
    *,
    use_focal_loss: bool = False,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
    conditioning_spatial: Optional[Tensor] = None,
) -> Tensor:
    """
    Perte de diffusion L_gen en déléguant à CausalDiffusionDecoder.
    """
    return decoder.compute_loss(
        target,
        conditioning,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        conditioning_spatial=conditioning_spatial,
    )


def loss_no_tears(A_masked: Tensor) -> Tensor:
    """
    Implémentation de la contrainte NO TEARS : tr(e^{A∘A}) - q.
    
    Note: Cette méthode est instable et O(q³). Utilisez loss_dagma() pour de meilleures performances.
    """
    A_squared = torch.mul(A_masked, A_masked)
    matrix_exp = torch.matrix_exp(A_squared)
    trace = torch.trace(matrix_exp)
    return trace - A_masked.shape[0]


def compute_divergence(field: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tensor:
    """
    Phase 3.3: Compute divergence of a 2D vector field using finite differences.
    
    For a field [u, v] with shape [batch, 2, H, W], computes div = ∂u/∂x + ∂v/∂y
    
    Parameters
    ----------
    field : Tensor
        Vector field [batch, 2, H, W] or [batch, channels, H, W] where first 2 channels are (u, v)
    dx : float
        Spatial step in x-direction (longitude)
    dy : float
        Spatial step in y-direction (latitude)
    
    Returns
    -------
    Tensor
        Divergence field [batch, H, W]
    """
    if field.shape[1] < 2:
        return torch.zeros(field.shape[0], field.shape[2], field.shape[3], device=field.device)
    
    u = field[:, 0:1, :, :]  # [batch, 1, H, W]
    v = field[:, 1:2, :, :]  # [batch, 1, H, W]
    
    # Compute gradients using central differences
    # ∂u/∂x using central difference
    u_pad = nn.functional.pad(u, (1, 1, 0, 0), mode='replicate')
    du_dx = (u_pad[:, :, :, 2:] - u_pad[:, :, :, :-2]) / (2 * dx)
    
    # ∂v/∂y using central difference
    v_pad = nn.functional.pad(v, (0, 0, 1, 1), mode='replicate')
    dv_dy = (v_pad[:, :, 2:, :] - v_pad[:, :, :-2, :]) / (2 * dy)
    
    # Divergence
    divergence = du_dx.squeeze(1) + dv_dy.squeeze(1)  # [batch, H, W]
    return divergence


def compute_vorticity(field: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tensor:
    """
    Phase 3.3: Compute vorticity (curl) of a 2D vector field using finite differences.
    
    For a field [u, v] with shape [batch, 2, H, W], computes vorticity = ∂v/∂x - ∂u/∂y
    
    Parameters
    ----------
    field : Tensor
        Vector field [batch, 2, H, W] or [batch, channels, H, W] where first 2 channels are (u, v)
    dx : float
        Spatial step in x-direction (longitude)
    dy : float
        Spatial step in y-direction (latitude)
    
    Returns
    -------
    Tensor
        Vorticity field [batch, H, W]
    """
    if field.shape[1] < 2:
        return torch.zeros(field.shape[0], field.shape[2], field.shape[3], device=field.device)
    
    u = field[:, 0:1, :, :]  # [batch, 1, H, W]
    v = field[:, 1:2, :, :]  # [batch, 1, H, W]
    
    # Compute gradients using central differences
    # ∂v/∂x
    v_pad = nn.functional.pad(v, (1, 1, 0, 0), mode='replicate')
    dv_dx = (v_pad[:, :, :, 2:] - v_pad[:, :, :, :-2]) / (2 * dx)
    
    # ∂u/∂y
    u_pad = nn.functional.pad(u, (0, 0, 1, 1), mode='replicate')
    du_dy = (u_pad[:, :, 2:, :] - u_pad[:, :, :-2, :]) / (2 * dy)
    
    # Vorticity
    vorticity = dv_dx.squeeze(1) - du_dy.squeeze(1)  # [batch, H, W]
    return vorticity


def loss_physical(output: Tensor, target: Tensor, dx: float = 1.0, dy: float = 1.0) -> Tensor:
    """
    Phase 3.3: Compute physical loss L_phy enforcing divergence and vorticity constraints.
    
    Penalizes divergence and vorticity errors between output and target fields.
    For climate fields, divergence should be small (mass conservation) and vorticity
    should match the target (circulation conservation).
    
    Parameters
    ----------
    output : Tensor
        Predicted field [batch, channels, H, W]
    target : Tensor
        Target field [batch, channels, H, W]
    dx : float
        Spatial step in x-direction
    dy : float
        Spatial step in y-direction
    
    Returns
    -------
    Tensor
        Physical loss (divergence + vorticity errors)
    """
    # Compute divergence for both output and target
    div_output = compute_divergence(output, dx=dx, dy=dy)
    div_target = compute_divergence(target, dx=dx, dy=dy)
    
    # Divergence error (should be close to zero for mass conservation)
    div_loss = nn.functional.mse_loss(div_output, div_target)
    
    # Compute vorticity for both output and target
    vort_output = compute_vorticity(output, dx=dx, dy=dy)
    vort_target = compute_vorticity(target, dx=dx, dy=dy)
    
    # Vorticity error (should match target circulation)
    vort_loss = nn.functional.mse_loss(vort_output, vort_target)
    
    # Combined physical loss
    return div_loss + vort_loss


def loss_dagma(
    A_masked: Tensor,
    s: float = 1.0,
    add_l1_regularization: bool = False,  # Phase D3: Add L1 regularization for sparsity
    l1_weight: float = 0.01,  # Phase D3: Weight for L1 regularization
) -> Tensor:
    """
    Implémentation de la contrainte DAGMA (DAG via la méthode d'augmentation du log-déterminant).
    
    Plus stable et efficace que NO TEARS. Utilise la formule:
    h(W) = -log det(sI - W∘W) + d log s
    
    Phase D3: Enhanced with numerical stability improvements and optional L1 regularization.
    
    Parameters
    ----------
    A_masked : Tensor
        Matrice DAG avec diagonale masquée [q, q]
    s : float
        Paramètre de régularisation (par défaut 1.0). Doit être > rho(A_masked ∘ A_masked)
        où rho est le rayon spectral.
    add_l1_regularization : bool
        Phase D3: If True, add L1 regularization for sparser DAGs
    l1_weight : float
        Phase D3: Weight for L1 regularization term
    
    Returns
    -------
    Tensor
        Valeur de la contrainte DAGMA (doit être > 0 pour un DAG valide)
    
    References
    ----------
    - Bello et al. (2022): "DAGMA: Learning DAGs via M-matrices and a Log-Determinant Acyclicity Characterization"
    """
    q = A_masked.shape[0]
    device = A_masked.device
    dtype = A_masked.dtype
    
    A_clipped = torch.clamp(A_masked, min=-10.0, max=10.0)
    
    # W∘W (Hadamard square, element-wise)
    W_squared = torch.mul(A_clipped, A_clipped)  # [q, q]
    
    # s must exceed the spectral radius of W_squared for M to be positive-definite.
    # Gershgorin bound: rho(W²) <= max row-sum of |W²|.  This is tight and O(q²).
    gershgorin_bound = W_squared.abs().sum(dim=1).max().item()
    s_safe = max(s, gershgorin_bound + 0.1)
    
    # M = sI - W∘W  (M-matrix, positive-definite when s > rho(W²))
    sI = s_safe * torch.eye(q, device=device, dtype=dtype)
    M = sI - W_squared  # [q, q]
    
    eps = 1e-7
    M = M + eps * torch.eye(q, device=device, dtype=dtype)
    
    try:
        log_det_M = torch.logdet(M)
        if not torch.isfinite(log_det_M):
            M = M + 1e-5 * torch.eye(q, device=device, dtype=dtype)
            log_det_M = torch.logdet(M)
    except RuntimeError:
        return torch.tensor(float(q), device=device, dtype=dtype, requires_grad=True)
    
    # h(W) = -log det(sI - W∘W) + d log s
    h_W = -log_det_M + q * math.log(s_safe + eps)
    
    if add_l1_regularization:
        l1_term = l1_weight * A_masked.abs().sum()
        h_W = h_W + l1_term
    
    if not torch.isfinite(h_W):
        return torch.tensor(float(q), device=device, dtype=dtype, requires_grad=True)
    
    return h_W


@dataclass
class TrainingStepResult:
    """
    Résultats agrégés d'une étape (batch) d'entraînement.
    """

    loss_total: float
    loss_gen: float
    loss_rec: float
    loss_dag: float


def train_epoch(
    *,
    encoder: IntelligibleVariableEncoder,
    rcn_runner: RCNSequenceRunner,
    diffusion_decoder: CausalDiffusionDecoder,
    optimizer: torch.optim.Optimizer,
    data_loader: Iterable[Dict[str, Tensor]],
    lambda_gen: float,
    beta_rec: float,
    gamma_dag: float,
    conditioning_fn: Optional[Callable[[Tensor, Optional[Tensor]], Tensor]] = None,
    device: torch.device,
    gradient_clipping: Optional[float] = None,
    batch_index_key: str = "batch_index",
    residual_key: str = "residual",
    log_interval: int = 10,
    verbose: bool = True,
    dag_method: str = "dagma",  # "dagma" or "no_tears"
    dagma_s: float = 1.0,  # Parameter for DAGMA constraint
    lambda_phy: float = 0.0,  # Phase 3.3: Weight for physical loss (divergence + vorticity)
    dx: float = 1.0,  # Phase 3.3: Spatial step in x-direction (longitude)
    dy: float = 1.0,  # Phase 3.3: Spatial step in y-direction (latitude)
    use_predicted_output: bool = False,  # Phase B2: Use predictions for physical loss (expensive)
    physical_sample_interval: int = 10,  # Phase B2: Sample predictions every N batches
    physical_num_steps: int = 15,  # Phase B2: EDM sampling steps for physical loss
    use_amp: bool = True,  # Phase C1: CUDA FP16+GradScaler, or CPU bfloat16 autocast if supported
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,  # Phase C2: LR scheduler
    use_focal_loss: bool = False,  # Phase D1: Use focal loss for diffusion
    focal_alpha: float = 1.0,  # Phase D1: Focal loss alpha
    focal_gamma: float = 2.0,  # Phase D1: Focal loss gamma (higher = more focus on hard pixels)
    extreme_weight_factor: float = 0.0,  # Phase D2: Weight factor for extreme events (0 = disabled)
    extreme_percentiles: List[float] = None,  # Phase D2: Percentiles for extreme events
    reconstruction_loss_type: str = "mse",  # Phase D4: Loss type for reconstruction ("mse", "cosine", "mse+cosine")
    use_spectral_loss: bool = False,
    lambda_spectral: float = 0.0,
    conditioning_dropout_prob: float = 0.0,
    lambda_dag_prior: float = 0.0,
    dag_prior: Optional[Tensor] = None,
    spatial_projector: Optional[nn.Module] = None,
) -> Dict[str, float]:
    """
    Entraîne les modules sur une epoch complète.

    ``use_spectral_loss`` / ``lambda_spectral`` : conservés pour compatibilité ; la perte RAPSD
    n'est plus appliquée dans la boucle batch (voir ``log_spectral_metric_each_epoch`` + ``compute_rapsd_metric_from_batch``).
    
    Parameters
    ----------
    log_interval : int
        Affiche les logs tous les N batches (par défaut 10).
    verbose : bool
        Si True, affiche des logs détaillés (par défaut True).
    """
    encoder.train()
    rcn_runner.cell.train()
    diffusion_decoder.train()
    
    # Phase C1: Mixed precision — CUDA FP16 + GradScaler, or CPU BF16 (no scaler)
    scaler = None
    amp_mode = "none"
    if use_amp:
        if device.type == "cuda" and torch.cuda.is_available():
            from torch.cuda.amp import GradScaler

            scaler = GradScaler()
            amp_mode = "cuda_fp16"
        elif device.type == "cpu":
            bf16_ok = getattr(torch.cpu, "is_bf16_supported", None)
            if bf16_ok is not None and bf16_ok():
                amp_mode = "cpu_bf16"
            elif verbose:
                print("[WARN] CPU bfloat16 not supported; training in FP32")
        elif verbose:
            print(f"[WARN] AMP not used: device type {device.type}")
    
    # Phase D2: Initialize extreme percentiles if not provided
    if extreme_percentiles is None:
        extreme_percentiles = [95.0, 99.0]

    total_loss = 0.0
    total_gen = 0.0
    total_rec = 0.0
    total_dag = 0.0
    total_phy = 0.0  # Phase 3.3: Physical loss accumulator
    num_batches = 0
    
    epoch_start_time = time.time()
    
    if verbose:
        print(f"\n{'='*80}")
        print("START EPOCH")
        print(f"{'='*80}")
        print("Config:")
        print(f"   - Device: {device}")
        print(f"   - AMP mode: {amp_mode}")
        print(f"   - Lambda (gen): {lambda_gen}")
        print(f"   - Beta (rec): {beta_rec}")
        print(f"   - Gamma (DAG): {gamma_dag}")
        print(f"   - Gradient clipping: {gradient_clipping}")
        print(f"   - Log interval: {log_interval}")
        print(f"{'='*80}\n")

    for batch_idx, batch in enumerate(data_loader):
        # Support batch_size > 1: batch can be a list of batch dicts (gradient accumulation)
        batches = batch if isinstance(batch, list) else [batch]
        batch_start_time = time.time()
        
        optimizer.zero_grad()
        step_loss_total = 0.0
        step_loss_gen = 0.0
        step_loss_rec = 0.0
        step_loss_dag = 0.0
        step_loss_phy = 0.0
        
        for micro_idx, batch in enumerate(batches):
            _do_timing = verbose and batch_idx == 0 and micro_idx == 0

            if _do_timing:
                print(f"Batch {batch_idx + 1}" + (f" (n={len(batches)})" if len(batches) > 1 else "") + ":")
                print(f"   - Keys: {list(batch.keys())}")
            
            lr_data: Tensor = batch["lr"].to(device)      # [seq_len, N, features_lr]
            target_data: Tensor = batch.get(residual_key, batch.get("hr")).to(device)  # [seq_len, channels, H, W]
            hetero_data = batch["hetero"]
        
            # NaN/Inf input sanitization (cadenced to first micro-batch of every 20th optimizer step)
            if batch_idx % 20 == 0 and micro_idx == 0:
                _tensors_to_check = {
                    "target_data": target_data,
                    "lr_data": lr_data,
                }
                for _name, _t in _tensors_to_check.items():
                    if not torch.isfinite(_t).all():
                        if _name == "target_data":
                            _nan_mask = torch.isnan(target_data)
                            if _nan_mask.any():
                                nan_fill = torch.nanmean(target_data).item()
                                if not math.isfinite(nan_fill):
                                    nan_fill = 0.0
                                target_data = torch.nan_to_num(target_data, nan=nan_fill)
                            _inf_mask = torch.isinf(target_data)
                            if _inf_mask.any():
                                _valid_mask = torch.isfinite(target_data)
                                valid_mean = target_data[_valid_mask].mean().item() if _valid_mask.any() else 0.0
                                target_data = torch.nan_to_num(
                                    target_data,
                                    nan=valid_mean,
                                    posinf=valid_mean,
                                    neginf=valid_mean,
                                )
                        elif _name == "lr_data":
                            _nan_mask = torch.isnan(lr_data)
                            if _nan_mask.any():
                                nan_fill = torch.nanmean(lr_data).item()
                                if not math.isfinite(nan_fill):
                                    nan_fill = 0.0
                                lr_data = torch.nan_to_num(lr_data, nan=nan_fill)

            if _do_timing:
                print(f"   - LR data shape: {lr_data.shape}")
                print(f"   - Target data shape: {target_data.shape}")
                print(f"   - Sequence length: {lr_data.shape[0]}")

            # Encoder step
            if _do_timing:
                encoder_time = time.time()
            with _train_autocast(amp_mode):
                H_init = encoder.init_state(hetero_data).to(device)
            if _do_timing:
                encoder_time = time.time() - encoder_time
                print(f"   - H_init shape: {H_init.shape}")
                print(f"   - Encoder time: {encoder_time:.4f}s")

            # RCN step
            if _do_timing:
                rcn_time = time.time()
            drivers = [lr_data[t] for t in range(lr_data.shape[0])]
            with _train_autocast(amp_mode):
                seq_output = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
            if _do_timing:
                rcn_time = time.time() - rcn_time
                print(f"   - Number of states: {len(seq_output.states)}")
                print(f"   - Number of reconstructions: {len(seq_output.reconstructions)}")
                print(f"   - Number of DAG matrices: {len(seq_output.dag_matrices)}")
                print(f"   - RCN time: {rcn_time:.4f}s")

            # Loss computation (reconstruction and DAG)
            loss_rec_value = torch.tensor(0.0, device=device)
            loss_dag_value = torch.tensor(0.0, device=device)
            num_reconstructions = 0
            with _train_autocast(amp_mode):
                for recon, driver_step in zip(
                    seq_output.reconstructions,
                    drivers,
                ):
                    if recon is not None:
                        num_reconstructions += 1
                        loss_rec_value = loss_rec_value + beta_rec * loss_reconstruction(
                            recon, driver_step, loss_type=reconstruction_loss_type
                        )

                # A_masked is the same learned adjacency at every timestep;
                # compute the DAG penalty once and scale by the number of steps.
                A_masked_0 = seq_output.dag_matrices[0]
                n_dag_steps = len(seq_output.dag_matrices)
                if dag_method == "dagma":
                    loss_dag_value = gamma_dag * n_dag_steps * loss_dagma(A_masked_0, s=dagma_s)
                else:
                    loss_dag_value = gamma_dag * n_dag_steps * loss_no_tears(A_masked_0)

                if lambda_dag_prior > 0.0 and dag_prior is not None:
                    _prior = dag_prior.to(device=A_masked_0.device, dtype=A_masked_0.dtype)
                    loss_dag_value = loss_dag_value + lambda_dag_prior * nn.functional.mse_loss(A_masked_0, _prior)
        
            if _do_timing:
                print(f"   - Reconstructions computed: {num_reconstructions}/{len(seq_output.reconstructions)}")

            H_condition = seq_output.states[-1]
            batch_index = batch.get(batch_index_key)
            if batch_index is not None:
                batch_index = batch_index.to(device)
            if conditioning_fn is None:
                conditioning = encoder.project_state_tensor(H_condition, batch_index=batch_index)
            else:
                conditioning = conditioning_fn(H_condition, batch_index)
            conditioning = conditioning.to(device)

            conditioning_spatial = None
            if spatial_projector is not None:
                conditioning_spatial = spatial_projector(H_condition, batch_index=batch_index).to(device)

            _dropout = conditioning_dropout_prob > 0.0 and torch.rand(1, device=device).item() < conditioning_dropout_prob
            if _dropout:
                conditioning = torch.zeros_like(conditioning)
                if conditioning_spatial is not None:
                    conditioning_spatial = torch.zeros_like(conditioning_spatial)
        
            if _do_timing:
                print(f"   - Conditioning shape: {conditioning.shape}")

            target = target_data[-1]  # Should be [channels, H, W] or [H, W, channels]
            # Ensure target has shape [batch, channels, H, W]
            if target.dim() == 3:
                # Check if it's [channels, H, W] or [H, W, channels]
                # UNet expects [batch, channels, H, W]
                # If first dim is very large, it might be [H*W, channels] or similar
                # For now, assume [channels, H, W] and add batch dim
                target = target.unsqueeze(0)
            elif target.dim() == 4:
                # Already has batch dimension, use as is
                pass
            else:
                raise ValueError(f"Unexpected target shape: {target.shape}, expected [channels, H, W] or [batch, channels, H, W]")
        
            if _do_timing:
                print(f"   - Target shape (after processing): {target.shape}")
        
            # Verify channel count matches UNet expectations
            if target.shape[1] != diffusion_decoder.in_channels:
                raise ValueError(
                    f"Channel mismatch: target has {target.shape[1]} channels, "
                    f"but UNet expects {diffusion_decoder.in_channels} channels. "
                    f"Target shape: {target.shape}"
                )

            valid_mask_ts = batch.get("valid_mask")
            if valid_mask_ts is not None:
                vm = valid_mask_ts[-1].to(device=device)
                if vm.dtype != torch.bool:
                    vm = vm > 0.5
                while vm.dim() < target.dim():
                    vm = vm.unsqueeze(0)
                vm = vm.expand_as(target)
                target = target.clone().masked_fill(~vm, float("nan"))
        
            # Expensive per-micro-batch diagnostics: only on first micro-batch
            nan_count = 0
            nan_ratio = 0.0
            if micro_idx == 0:
                if _do_timing:
                    print(f"   - Target stats: min={target.min().item():.6f}, max={target.max().item():.6f}, mean={target.mean().item():.6f}, std={target.std().item():.6f}")
                    _target_isfinite = torch.isfinite(target).all().item()
                    print(f"   - Target has NaN/Inf: {not _target_isfinite}")
                    print(f"   - Conditioning stats: min={conditioning.min().item():.6f}, max={conditioning.max().item():.6f}, mean={conditioning.mean().item():.6f}")
                    _cond_isfinite = torch.isfinite(conditioning).all().item()
                    print(f"   - Conditioning has NaN/Inf: {not _cond_isfinite}")

                target_abs_max = target.abs().max().item()
                if target_abs_max > 1e6:
                    if verbose:
                        print(f"[WARN] Target has very large values: max_abs={target_abs_max:.2e}")

                nan_mask = ~torch.isfinite(target)
                nan_count = nan_mask.sum().item()
                nan_ratio = nan_count / target.numel() if target.numel() > 0 else 0.0
                if nan_count > 0 and verbose and (batch_idx == 0 or batch_idx % log_interval == 0):
                    print(f"[INFO] Target contains {nan_count} NaN/Inf pixels ({nan_ratio:.2%}) - will be masked in loss")

            # Conditioning must NEVER contain NaN/Inf (critical safety check on every micro-batch)
            if not torch.isfinite(conditioning).all():
                print(f"[ERROR] Conditioning contains NaN/Inf in batch {batch_idx + 1}")
                print(f"   - NaN count: {torch.isnan(conditioning).sum().item()}")
                print(f"   - Inf count: {torch.isinf(conditioning).sum().item()}")
                print(f"   - This is a critical error, skipping batch")
                continue
        
            # Phase C1: Mixed Precision - Forward pass with autocast for entire forward
            # Diffusion loss (gère automatiquement les NaN via masquage)
            # Phase D1: Supports focal loss for focusing on hard pixels
            if _do_timing:
                diffusion_time = time.time()
            with _train_autocast(amp_mode):
                loss_gen_value = lambda_gen * loss_diffusion(
                    diffusion_decoder, target, conditioning,
                    use_focal_loss=use_focal_loss,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                    conditioning_spatial=conditioning_spatial,
                )

            # RAPSD (FFT + scatter_add) : déplacé en fin d’époque — voir compute_rapsd_metric_from_batch.

            # Phase B2: Compute physical loss (divergence + vorticity)
            # Improved version that compares predictions vs target (not just target)
            loss_phy_value = torch.tensor(0.0, device=device)
            if lambda_phy > 0.0:
                # Phase B2: Option 3 (Hybrid) - Compute physical loss on predictions periodically
                # For efficiency, we only sample predictions every N batches
                # This reduces cost while still enforcing physical consistency on model outputs
                compute_physical_on_predictions = (
                    use_predicted_output and
                    batch_idx % physical_sample_interval == 0
                )
            
                if compute_physical_on_predictions:
                    # Sample a prediction from the model (expensive but accurate)
                    # Use EDM with few steps for efficiency (15-25 steps vs 1000 for DDPM)
                    with torch.no_grad():  # Don't backprop through sampling
                        try:
                            sampled_output = diffusion_decoder.sample(
                                conditioning=conditioning,
                                num_steps=physical_num_steps,
                                scheduler_type="edm",  # Use EDM for fast sampling
                                apply_constraints=True,
                            )
                            # sampled_output.residual is [batch, channels, H, W]
                            pred_residual = sampled_output.residual
                        
                            # Compare physical constraints: pred vs target
                            div_pred = compute_divergence(pred_residual, dx=dx, dy=dy)
                            div_target = compute_divergence(target, dx=dx, dy=dy)
                            vort_pred = compute_vorticity(pred_residual, dx=dx, dy=dy)
                            vort_target = compute_vorticity(target, dx=dx, dy=dy)
                        
                            # Physical loss: enforce that predictions have similar physical properties as target
                            div_error = ((div_pred - div_target) ** 2).mean()
                            vort_error = ((vort_pred - vort_target) ** 2).mean()
                            loss_phy_value = lambda_phy * (div_error + 0.1 * vort_error)
                        
                            if _do_timing:
                                print(f"   - Physical loss computed on predictions (EDM, {physical_num_steps} steps)")
                        except Exception as e:
                            # Fallback to target-only physical loss if sampling fails
                            if verbose:
                                print(f"[WARN] Physical loss sampling failed: {e}, falling back to target-only")
                            div_target = compute_divergence(target, dx=dx, dy=dy)
                            vort_target = compute_vorticity(target, dx=dx, dy=dy)
                            div_penalty = (div_target ** 2).mean()
                            vort_penalty = (vort_target ** 2).mean()
                            loss_phy_value = lambda_phy * (div_penalty + 0.1 * vort_penalty)
                else:
                    # Default: compute physical loss on target only (fast, acts as regularization)
                    # This ensures target satisfies physical laws
                    div_target = compute_divergence(target, dx=dx, dy=dy)
                    vort_target = compute_vorticity(target, dx=dx, dy=dy)
                    # Penalize non-zero divergence (should be ~0 for mass conservation) and inconsistent vorticity
                    div_penalty = (div_target ** 2).mean()
                    vort_penalty = (vort_target ** 2).mean()
                    loss_phy_value = lambda_phy * (div_penalty + 0.1 * vort_penalty)
        
            if _do_timing:
                diffusion_time = time.time() - diffusion_time
                print(f"   - Diffusion time: {diffusion_time:.4f}s")
                if lambda_phy > 0.0:
                    print(f"   - Physical loss: {loss_phy_value.item():.6f}")
                if nan_count > 0:
                    print(f"   - Loss computed on {target.numel() - nan_count}/{target.numel()} valid pixels ({1.0 - nan_ratio:.2%})")

            # Phase C1: Mixed Precision - Compute total loss
            with _train_autocast(amp_mode):
                loss_total = (
                    loss_gen_value
                    + loss_rec_value
                    + loss_dag_value
                    + loss_phy_value
                )
        
            # Check for NaN or Inf
            if not torch.isfinite(loss_total):
                print(f"[WARN] Batch {batch_idx + 1} has invalid loss!")
                print(f"   - Loss total: {loss_total.item()}")
                print(f"   - Loss gen: {loss_gen_value.item()}")
                print(f"   - Loss rec: {loss_rec_value.item()}")
                print(f"   - Loss DAG: {loss_dag_value.item()}")
                if torch.isnan(loss_total):
                    print(f"   - Loss is NaN, skipping batch")
                    continue
                elif torch.isinf(loss_total):
                    print(f"   - Loss is Inf, skipping batch")
                    continue

            # Accumulate for logging (average over micro-batches)
            step_loss_total += loss_total.item()
            step_loss_gen += loss_gen_value.item()
            step_loss_rec += loss_rec_value.item()
            step_loss_dag += loss_dag_value.item()
            step_loss_phy += loss_phy_value.item()

            # Phase C1: Mixed Precision - Backward pass (scaled for gradient accumulation)
            # DDP: skip gradient sync on non-last micro-batches (no_sync) for faster accumulation
            if _do_timing:
                backward_time = time.time()
            scale = 1.0 / len(batches)  # Scale gradients for accumulation
            is_last_micro = (micro_idx == len(batches) - 1)
            ctx_enc = encoder.no_sync() if (isinstance(encoder, DDP) and not is_last_micro) else nullcontext()
            ctx_rcn = rcn_runner.cell.no_sync() if (isinstance(rcn_runner.cell, DDP) and not is_last_micro) else nullcontext()
            ctx_diff = diffusion_decoder.no_sync() if (isinstance(diffusion_decoder, DDP) and not is_last_micro) else nullcontext()
            with ctx_enc, ctx_rcn, ctx_diff:
                if amp_mode == "cuda_fp16":
                    scaler.scale(loss_total * scale).backward()
                else:
                    (loss_total * scale).backward()
            if _do_timing:
                backward_time = time.time() - backward_time

        if gradient_clipping is not None:
            clip_time = time.time()
            if amp_mode == "cuda_fp16":
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                grad_norm_rcn = torch.nn.utils.clip_grad_norm_(rcn_runner.cell.parameters(), gradient_clipping)
                grad_norm_diff = torch.nn.utils.clip_grad_norm_(diffusion_decoder.parameters(), gradient_clipping)
                grad_norm_enc = torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clipping)
            else:
                grad_norm_rcn = torch.nn.utils.clip_grad_norm_(rcn_runner.cell.parameters(), gradient_clipping)
                grad_norm_diff = torch.nn.utils.clip_grad_norm_(diffusion_decoder.parameters(), gradient_clipping)
                grad_norm_enc = torch.nn.utils.clip_grad_norm_(encoder.parameters(), gradient_clipping)
            clip_time = time.time() - clip_time
            
            # Vérifier les gradients après clipping pour détecter les NaN
            nan_grads_found = False
            if batch_idx % 50 == 0:
                for name, param in encoder.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"[ERROR] NaN gradient detected in encoder.{name}")
                        nan_grads_found = True
                for name, param in rcn_runner.cell.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"[ERROR] NaN gradient detected in rcn.{name}")
                        nan_grads_found = True
                for name, param in diffusion_decoder.named_parameters():
                    if param.grad is not None and torch.isnan(param.grad).any():
                        print(f"[ERROR] NaN gradient detected in diffusion.{name}")
                        nan_grads_found = True
            
            if nan_grads_found:
                print(f"[WARN] NaN gradients detected after clipping - this may indicate model divergence")
            
            if verbose and (batch_idx % log_interval == 0 or batch_idx == 0):
                print(f"   - Gradient norms (clipped): RCN={grad_norm_rcn:.4f}, Diff={grad_norm_diff:.4f}, Enc={grad_norm_enc:.4f}")

        # Phase C1: Mixed Precision - Optimizer step with scaler (CUDA only)
        if amp_mode == "cuda_fp16":
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        step_time = time.time() - batch_start_time

        n_micro = len(batches)
        total_loss += step_loss_total
        total_gen += step_loss_gen
        total_rec += step_loss_rec
        total_dag += step_loss_dag
        total_phy += step_loss_phy
        num_batches += 1
        
        # Logging (show average loss over micro-batches when batch_size > 1)
        if verbose and (batch_idx % log_interval == 0 or batch_idx == 0):
            print(f"\nBatch {batch_idx + 1}" + (f" (n={n_micro})" if n_micro > 1 else "") + ":")
            print(f"   - Loss total: {step_loss_total/n_micro:.6f}")
            print(f"   - Loss gen: {step_loss_gen/n_micro:.6f}")
            print(f"   - Loss rec: {step_loss_rec/n_micro:.6f}")
            print(f"   - Loss DAG: {step_loss_dag/n_micro:.6f}")
            if lambda_phy > 0.0:
                print(f"   - Loss phy: {step_loss_phy/n_micro:.6f}")
            print(f"   - Batch time: {step_time:.4f}s")
            if batch_idx == 0:
                try:
                    print(f"   - Time breakdown: Enc={encoder_time:.3f}s, RCN={rcn_time:.3f}s, "
                          f"Diff={diffusion_time:.3f}s, Backward={backward_time:.3f}s")
                    if gradient_clipping is not None:
                        print(f"   - Clip time: {clip_time:.3f}s")
                except NameError:
                    pass

    epoch_time = time.time() - epoch_start_time
    
    if num_batches == 0:
        if verbose:
            print("\n[WARN] No batches were processed in this epoch!")
        return {"loss": 0.0, "loss_gen": 0.0, "loss_rec": 0.0, "loss_dag": 0.0}

    avg_loss = total_loss / num_batches
    avg_gen = total_gen / num_batches
    avg_rec = total_rec / num_batches
    avg_dag = total_dag / num_batches
    avg_phy = total_phy / num_batches
    
    if verbose:
        print(f"\n{'='*80}")
        print("END EPOCH")
        print(f"{'='*80}")
        print("Results:")
        print(f"   - Num batches: {num_batches}")
        print(f"   - Temps total: {epoch_time:.2f}s ({epoch_time/60:.2f} min)")
        print(f"   - Temps moyen par batch: {epoch_time/num_batches:.4f}s")
        print("\nAverage losses:")
        print(f"   - Loss totale: {avg_loss:.6f}")
        print(f"   - Loss génération (diffusion): {avg_gen:.6f}")
        print(f"   - Loss reconstruction: {avg_rec:.6f}")
        print(f"   - Loss DAG ({dag_method.upper()}): {avg_dag:.6f}")
        if lambda_phy > 0.0:
            print(f"   - Loss physique (divergence+vorticité): {avg_phy:.6f}")
        print(f"{'='*80}\n")

    result = {
        "loss": avg_loss,
        "loss_gen": avg_gen,
        "loss_rec": avg_rec,
        "loss_dag": avg_dag,
    }
    if lambda_phy > 0.0:
        result["loss_phy"] = avg_phy
    return result
```

---

### `src/st_cdgm/training/multi_gpu.py`

```python
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
```

---

### `src/st_cdgm/evaluation/__init__.py`

```python
"""
Modules d'évaluation et XAI pour ST-CDGM.
"""

from .evaluation_xai import (
    autoregressive_inference,
    convert_sample_to_batch,
    extract_target_baseline_and_mask,
    resize_diffusion_output_to_spatial,
    resize_tensor_bicubic_nonneg,
    run_st_cdgm_inference,
    evaluate_metrics,
    compute_crps,
    compute_crps_pixel_map,
    compute_fss,
    compute_f1_extremes,
    compute_spectrum_distance,
    compute_temporal_variance_metrics,
    compute_wasserstein_distance,
    compute_energy_score,
    compute_structural_hamming_distance,
    plot_dag_heatmap,
    plot_probabilistic_dashboard_3x3,
    export_dag_to_csv,
    export_dag_to_json,
    MetricReport,
    InferenceResult,
)

__all__ = [
    "autoregressive_inference",
    "convert_sample_to_batch",
    "extract_target_baseline_and_mask",
    "resize_diffusion_output_to_spatial",
    "resize_tensor_bicubic_nonneg",
    "run_st_cdgm_inference",
    "evaluate_metrics",
    "compute_crps",
    "compute_crps_pixel_map",
    "compute_fss",
    "compute_f1_extremes",
    "compute_spectrum_distance",
    "compute_temporal_variance_metrics",
    "compute_wasserstein_distance",
    "compute_energy_score",
    "compute_structural_hamming_distance",
    "plot_dag_heatmap",
    "plot_probabilistic_dashboard_3x3",
    "export_dag_to_csv",
    "export_dag_to_json",
    "MetricReport",
    "InferenceResult",
]
```

---

### `src/st_cdgm/evaluation/evaluation_xai.py`

```python
"""
Module 7 – Évaluation et interprétabilité (XAI) pour l’architecture ST-CDGM.

Ce module propose des fonctions pour :
  * effectuer une inférence auto-régressive avec génération multi-échantillons,
  * calculer des métriques de précision (MSE, MAE) et de réalisme (histogrammes, CRPS placeholder),
  * visualiser et exporter le DAG appris.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import json
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from ..models.graph_builder import HeteroGraphBuilder
from ..models.causal_rcn import RCNCell, RCNSequenceRunner
from ..models.diffusion_decoder import CausalDiffusionDecoder, DiffusionOutput
from ..models.intelligible_encoder import IntelligibleVariableEncoder

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - dépendance optionnelle
    raise ImportError(
        "Le module evaluation_xai nécessite seaborn et matplotlib "
        "(pip install seaborn matplotlib)."
    ) from exc


@dataclass
class InferenceResult:
    """
    Résultat d'une inférence auto-régressive multi-échantillons.
    """

    generations: List[List[DiffusionOutput]]  # [time][sample]
    states: List[Tensor]
    dag_matrices: List[Tensor]


# ---------------------------------------------------------------------------
# Phase 1 / 5 — Inférence centralisée (bicubique, masque NaN)
# ---------------------------------------------------------------------------


def resize_tensor_bicubic_nonneg(
    x: Tensor,
    size: Tuple[int, int],
    *,
    clamp_min: float = 0.0,
) -> Tensor:
    """
    Redimensionnement spatial bicubique + clamp (précip / champs positifs).
    Remplace l'interpolation bilinéaire (passe-bas) du pipeline legacy.
    """
    if x.dim() != 4:
        raise ValueError(f"Attendu [B,C,H,W], obtenu {tuple(x.shape)}")
    y = F.interpolate(x, size=size, mode="bicubic", align_corners=False)
    return torch.clamp(y, min=clamp_min)


def resize_diffusion_output_to_spatial(
    out: DiffusionOutput,
    spatial: Tuple[int, int],
    *,
    clamp_min: float = 0.0,
) -> DiffusionOutput:
    """Aligne t_min / t_mean / t_max / residual sur la grille cible."""
    Ht, Wt = spatial

    def _resize(t: Tensor) -> Tensor:
        if t.dim() != 4:
            return t
        if t.shape[-2:] == (Ht, Wt):
            return t
        return resize_tensor_bicubic_nonneg(t, (Ht, Wt), clamp_min=clamp_min)

    res = _resize(out.residual)
    base = out.baseline
    if base is not None and base.dim() == 4 and base.shape[-2:] != (Ht, Wt):
        base = resize_tensor_bicubic_nonneg(base, (Ht, Wt), clamp_min=clamp_min)
    return DiffusionOutput(
        residual=res,
        baseline=base,
        t_min=_resize(out.t_min),
        t_mean=_resize(out.t_mean),
        t_max=_resize(out.t_max),
    )


def convert_sample_to_batch(
    sample: dict,
    builder: HeteroGraphBuilder,
    device: torch.device,
) -> dict:
    """Construit lr, residual, baseline, hetero depuis un échantillon dataset."""
    lr_seq = sample["lr"]
    seq_len = lr_seq.shape[0]
    lr_nodes_steps = [builder.lr_grid_to_nodes(lr_seq[t]) for t in range(seq_len)]
    lr_tensor = torch.stack(lr_nodes_steps, dim=0)
    dynamic_features = {node_type: lr_nodes_steps[0] for node_type in builder.dynamic_node_types}
    hetero = builder.prepare_step_data(dynamic_features).to(device)
    return {
        "lr": lr_tensor,
        "residual": sample["residual"],
        "baseline": sample.get("baseline"),
        "hetero": hetero,
    }


def extract_target_baseline_and_mask(
    batch: dict,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Cible HR, baseline et masque de validité (True = donnée observée).

    Phase 5.1 : le masque est extrait **avant** nan_to_num.
    """
    target_residual = batch["residual"][-1].to(device)
    baseline_tensor = (
        batch["baseline"][-1]
        if batch.get("baseline") is not None
        else torch.zeros_like(target_residual)
    )
    baseline_tensor = baseline_tensor.to(device)
    full_target = baseline_tensor + target_residual

    valid_mask = ~torch.isnan(full_target)

    baseline_tensor = torch.nan_to_num(baseline_tensor, nan=0.0, posinf=0.0, neginf=0.0)
    full_target = torch.nan_to_num(full_target, nan=0.0, posinf=0.0, neginf=0.0)

    b = baseline_tensor.unsqueeze(0) if baseline_tensor.dim() == 3 else baseline_tensor
    t = full_target.unsqueeze(0) if full_target.dim() == 3 else full_target
    m = valid_mask.unsqueeze(0) if valid_mask.dim() == 3 else valid_mask
    return t, b, m


@torch.no_grad()
def run_st_cdgm_inference(
    sample: dict,
    *,
    builder: HeteroGraphBuilder,
    encoder: IntelligibleVariableEncoder,
    rcn_runner: RCNSequenceRunner,
    diffusion: CausalDiffusionDecoder,
    device: torch.device,
    num_samples: int,
    num_steps: int,
    scheduler_type: str = "ddpm",
    apply_constraints: bool = False,
    use_log1p_inverse: bool = False,
    cfg_scale: float = 0.0,
) -> Tuple[List[DiffusionOutput], Tensor, Tensor, Tensor, Tensor]:
    """
    Inférence complète encodeur → RCN → diffusion (multi-échantillons).

    Retourne ``samples_out, target_batch, baseline_batch, dag_last, mask_batch``.
    """
    batch = convert_sample_to_batch(sample, builder, device)
    lr_data = batch["lr"].to(device)
    target_batch, baseline_batch, mask_batch = extract_target_baseline_and_mask(batch, device)

    H_init = encoder.init_state(batch["hetero"]).to(device)
    drivers = [lr_data[t] for t in range(lr_data.shape[0])]
    seq_output = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
    conditioning = encoder.project_state_tensor(seq_output.states[-1]).to(device)
    dag_last = seq_output.dag_matrices[-1].detach().cpu()

    samples_out: List[DiffusionOutput] = []
    for _ in range(num_samples):
        generated = diffusion.sample(
            conditioning,
            num_steps=num_steps,
            scheduler_type=scheduler_type,
            apply_constraints=apply_constraints,
            baseline=baseline_batch,
            cfg_scale=cfg_scale,
        )
        if generated.t_mean.shape != target_batch.shape:
            generated = resize_diffusion_output_to_spatial(
                generated,
                (target_batch.shape[-2], target_batch.shape[-1]),
                clamp_min=0.0,
            )
        generated.t_mean = torch.nan_to_num(generated.t_mean, nan=0.0, posinf=0.0, neginf=0.0)
        if use_log1p_inverse:
            generated.t_mean = torch.expm1(generated.t_mean)
        samples_out.append(generated)

    return samples_out, target_batch.cpu(), baseline_batch.cpu(), dag_last, mask_batch.cpu()


def autoregressive_inference(
    *,
    rcn_runner: RCNSequenceRunner,
    diffusion_decoder: CausalDiffusionDecoder,
    initial_state: Tensor,
    drivers: Sequence[Tensor],
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
) -> InferenceResult:
    """
    Déroule le modèle de manière auto-régressive et génère plusieurs échantillons HR.

    Parameters
    ----------
    initial_state :
        État initial H(0) [q, N, hidden_dim].
    drivers :
        Séquence de forçages externes [T][N, driver_dim].
    num_samples :
        Nombre d'échantillons diffusion à générer par pas de temps.
    """
    H_t = initial_state
    generations: List[List[DiffusionOutput]] = []
    states: List[Tensor] = []
    dag_mats: List[Tensor] = []

    for driver in drivers:
        cell: RCNCell = rcn_runner.cell
        H_t, _, A_masked = cell(H_t, driver)
        states.append(H_t)
        dag_mats.append(A_masked)

        conditioning = H_t.mean(dim=0).unsqueeze(0)  # [1, hidden_dim, H, W] placeholder
        step_outputs: List[DiffusionOutput] = []
        for _ in range(num_samples):
            out = diffusion_decoder.sample(conditioning, generator=generator)
            step_outputs.append(out)

        generations.append(step_outputs)

    return InferenceResult(generations=generations, states=states, dag_matrices=dag_mats)


# ---------------------------------------------------------------------------
# Métriques de précision et de réalisme
# ---------------------------------------------------------------------------

def compute_mse(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> float:
    """Calcule MSE avec gestion des NaN."""
    if mask is None:
        mask = ~torch.isnan(target) & ~torch.isnan(pred)
    pred_valid = pred[mask]
    target_valid = target[mask]
    if pred_valid.numel() == 0:
        return 0.0
    return torch.mean((pred_valid - target_valid) ** 2).item()


def compute_mae(pred: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> float:
    """Calcule MAE avec gestion des NaN."""
    if mask is None:
        mask = ~torch.isnan(target) & ~torch.isnan(pred)
    pred_valid = pred[mask]
    target_valid = target[mask]
    if pred_valid.numel() == 0:
        return 0.0
    return torch.mean(torch.abs(pred_valid - target_valid)).item()


def compute_histogram_distance(pred: Tensor, target: Tensor, bins: int = 50) -> float:
    """
    Distance simple entre histogrammes (L1) pour évaluer le réalisme.
    """
    pred_np = pred.detach().cpu().numpy().ravel()
    target_np = target.detach().cpu().numpy().ravel()
    hist_pred, bin_edges = np.histogram(pred_np, bins=bins, density=True)
    hist_target, _ = np.histogram(target_np, bins=bin_edges, density=True)
    distance = np.sum(np.abs(hist_pred - hist_target)) * (bin_edges[1] - bin_edges[0])
    return float(distance)


def compute_crps_pixel_map(pred_stack: Tensor, target: Tensor) -> Tensor:
    """
    Carte CRPS pixel (formule énergétique) : pred_stack [N,C,H,W], target [C,H,W] ou [1,C,H,W].
    """
    if target.dim() == 4:
        target = target.squeeze(0)
    tgt = target[0] if target.dim() == 3 else target
    x = pred_stack[:, 0] if pred_stack.dim() == 4 and pred_stack.shape[1] >= 1 else pred_stack
    n = x.shape[0]
    if n < 1:
        raise ValueError("Ensemble vide")
    term1 = (x - tgt.unsqueeze(0)).abs().mean(dim=0)
    if n < 2:
        return term1
    term2 = (x.unsqueeze(0) - x.unsqueeze(1)).abs().mean(dim=(0, 1)) * 0.5
    return term1 - term2


def compute_spread_skill_ratio(pred_std_map: np.ndarray, err_map: np.ndarray, valid_mask: np.ndarray) -> float:
    """Ratio moyen spread / erreur sur pixels valides (objectif ~1 calibration)."""
    m = valid_mask & np.isfinite(pred_std_map) & np.isfinite(err_map)
    if not np.any(m):
        return float("nan")
    spread = pred_std_map[m].mean()
    skill = err_map[m].mean()
    if skill < 1e-12:
        return float("nan")
    return float(spread / skill)


def compute_rapsd_numpy(field: np.ndarray) -> np.ndarray:
    """RAPSD 2D (numpy) pour une carte [H,W]."""
    fft2 = np.fft.fft2(field)
    power = np.abs(np.fft.fftshift(fft2)) ** 2
    h, w = power.shape
    cy, cx = h // 2, w // 2
    y_idx, x_idx = np.indices((h, w))
    r = np.sqrt((x_idx - cx) ** 2 + (y_idx - cy) ** 2).astype(np.int64)
    radial_sum = np.bincount(r.ravel(), power.ravel())
    radial_cnt = np.bincount(r.ravel())
    valid = radial_cnt > 0
    out = np.zeros_like(radial_sum, dtype=np.float64)
    out[valid] = radial_sum[valid] / radial_cnt[valid]
    return out


def compute_crps(
    samples: Sequence[Tensor],
    target: Tensor,
    *,
    max_ensemble_members: Optional[int] = None,
) -> float:
    """
    Calcule le CRPS (Continuous Ranked Probability Score) pour un ensemble d'échantillons.
    Formule quadratique en la taille d'ensemble : plafonner ``max_ensemble_members`` (premiers membres) sur CPU.

    Phase 4.1: Improved CRPS implementation.
    """
    if len(samples) == 0:
        return float("nan")
    if max_ensemble_members is not None and len(samples) > max_ensemble_members:
        samples = list(samples)[:max_ensemble_members]
    stack = torch.stack(samples, dim=0)  # [ensemble, C, H, W]
    target = target.unsqueeze(0)
    term1 = torch.abs(stack - target).mean(dim=0)
    pairwise = torch.abs(stack.unsqueeze(0) - stack.unsqueeze(1)).mean(dim=(0, 1))
    crps = (term1 - 0.5 * pairwise).mean().item()
    return float(crps)


def compute_fss(pred: Tensor, target: Tensor, threshold: float, window_size: int = 9) -> float:
    """
    Calcule le Fraction Skill Score (FSS) pour l'évaluation spatiale.
    
    Phase 4.1: FSS measures spatial forecast skill for binary events.
    
    Parameters
    ----------
    pred : Tensor
        Prédiction [C, H, W] ou [H, W]
    target : Tensor
        Cible [C, H, W] ou [H, W]
    threshold : float
        Seuil pour binariser les champs
    window_size : int
        Taille de la fenêtre de voisinage pour le calcul (doit être impair)
    
    Returns
    -------
    float
        FSS score (0-1, 1 = perfect forecast)
    
    References
    ----------
    - Roberts & Lean (2008): "Scale-Selective Verification of Rainfall Accumulations
      from High-Resolution Forecasts of Convective Events"
    """
    # Ensure 2D tensors
    if pred.dim() == 3:
        pred = pred[0]  # Take first channel
    if target.dim() == 3:
        target = target[0]
    
    # Binarize fields based on threshold
    pred_binary = (pred >= threshold).float()
    target_binary = (target >= threshold).float()
    
    # Compute fraction of events in windows using convolution
    # Create averaging kernel
    kernel = torch.ones(1, 1, window_size, window_size, device=pred.device, dtype=pred.dtype) / (window_size ** 2)
    pred_binary_expanded = pred_binary.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    target_binary_expanded = target_binary.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    pred_frac = torch.nn.functional.conv2d(pred_binary_expanded, kernel, padding=window_size//2).squeeze()
    target_frac = torch.nn.functional.conv2d(target_binary_expanded, kernel, padding=window_size//2).squeeze()
    
    # Compute MSE of fractions
    mse_frac = torch.mean((pred_frac - target_frac) ** 2).item()
    
    # Compute MSE of fractions under random forecast (reference)
    pred_ref = torch.mean(pred_binary).item()
    target_ref = torch.mean(target_binary).item()
    mse_ref = pred_ref ** 2 + target_ref ** 2
    
    # FSS = 1 - MSE_frac / MSE_ref
    if mse_ref == 0:
        return 1.0  # Perfect forecast (both fields have no events or all events)
    fss = 1.0 - (mse_frac / mse_ref)
    return float(max(0.0, fss))  # Clamp to [0, 1]


def compute_f1_extremes(
    pred: Tensor,
    target: Tensor,
    threshold_percentiles: Sequence[float] = [95.0, 99.0],
) -> Dict[str, float]:
    """
    Phase C4: Compute F1 score for extreme events at different percentile thresholds.
    
    This metric is crucial for evaluating the model's performance on extreme events,
    which are often the most important for climate applications.
    
    Parameters
    ----------
    pred : Tensor
        Predicted field [C, H, W] or [H, W]
    target : Tensor
        Target field [C, H, W] or [H, W]
    threshold_percentiles : Sequence[float]
        Percentiles to use as thresholds for extreme events (default: [95, 99])
    
    Returns
    -------
    Dict[str, float]
        Dictionary mapping percentile threshold to F1 score
        Example: {"p95": 0.85, "p99": 0.72}
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Remove NaN/Inf if present
    valid_mask = torch.isfinite(pred_flat) & torch.isfinite(target_flat)
    pred_valid = pred_flat[valid_mask]
    target_valid = target_flat[valid_mask]
    
    if pred_valid.numel() == 0:
        return {f"p{p}": 0.0 for p in threshold_percentiles}
    
    results = {}
    
    for percentile in threshold_percentiles:
        # Compute threshold based on target distribution
        threshold = torch.quantile(target_valid, percentile / 100.0)
        
        # Binary classification: extreme (1) vs non-extreme (0)
        pred_binary = (pred_valid >= threshold).float()
        target_binary = (target_valid >= threshold).float()
        
        # Compute True Positives, False Positives, False Negatives
        tp = (pred_binary * target_binary).sum().item()
        fp = (pred_binary * (1 - target_binary)).sum().item()
        fn = ((1 - pred_binary) * target_binary).sum().item()
        
        # Compute Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[f"p{int(percentile)}"] = f1
    
    return results


def compute_wasserstein_distance(samples: Sequence[Tensor], target: Tensor, num_projections: int = 50) -> float:
    """
    Calcule la distance de Wasserstein (Sliced Wasserstein) pour comparaison de distributions.
    
    Phase 4.1: Sliced Wasserstein distance for high-dimensional distributions.
    Uses random projections to approximate Wasserstein-2 distance.
    
    Parameters
    ----------
    samples : Sequence[Tensor]
        Ensemble de prédictions [C, H, W] ou [H, W] chacune
    target : Tensor
        Cible [C, H, W] ou [H, W]
    num_projections : int
        Nombre de projections aléatoires pour l'approximation
    
    Returns
    -------
    float
        Distance de Wasserstein approximée
    """
    if len(samples) == 0:
        return float("nan")
    
    # Flatten spatial dimensions
    samples_flat = [s.flatten() if s.dim() > 1 else s for s in samples]
    target_flat = target.flatten() if target.dim() > 1 else target
    
    device = target_flat.device
    dim = target_flat.shape[0]
    
    # Compute Sliced Wasserstein distance via random projections
    wasserstein_distances = []
    for _ in range(num_projections):
        # Random projection direction
        direction = torch.randn(dim, device=device)
        direction = direction / (torch.norm(direction) + 1e-8)
        
        # Project samples and target
        samples_proj = torch.stack([torch.dot(s, direction) for s in samples_flat])
        target_proj = torch.dot(target_flat, direction).unsqueeze(0)
        
        # Sort projections
        samples_sorted, _ = torch.sort(samples_proj)
        target_sorted, _ = torch.sort(target_proj)
        
        # Compute Wasserstein-2 distance for 1D projections
        # Average over number of samples for stability
        if len(samples_sorted) > 1:
            # Interpolate target to match samples length
            indices = torch.linspace(0, len(target_sorted) - 1, len(samples_sorted), device=device).long()
            target_interp = target_sorted[indices]
        else:
            target_interp = target_sorted
        
        w2 = torch.mean((samples_sorted - target_interp) ** 2)
        wasserstein_distances.append(w2.item())
    
    return float(np.mean(wasserstein_distances))


def compute_energy_score(samples: Sequence[Tensor], target: Tensor) -> float:
    """
    Calcule l'Energy Score pour évaluer la cohérence multivariée de l'ensemble.
    
    Phase 4.1: Energy Score is a proper scoring rule for ensemble forecasts.
    
    Parameters
    ----------
    samples : Sequence[Tensor]
        Ensemble de prédictions [C, H, W] ou [H, W] chacune
    target : Tensor
        Observation cible [C, H, W] ou [H, W]
    
    Returns
    -------
    float
        Energy Score (lower is better)
    
    References
    ----------
    - Gneiting et al. (2007): "Strictly proper scoring rules, prediction, and estimation"
    """
    if len(samples) == 0:
        return float("nan")
    
    # Flatten if needed
    samples_flat = [s.flatten() if s.dim() > 1 else s for s in samples]
    target_flat = target.flatten() if target.dim() > 1 else target
    
    stack = torch.stack(samples_flat, dim=0)  # [ensemble_size, dim]
    
    # Term 1: Average distance from samples to target
    term1 = torch.mean(torch.norm(stack - target_flat.unsqueeze(0), dim=1)).item()
    
    # Term 2: Average pairwise distance within ensemble
    # Compute all pairwise distances efficiently
    pairwise_distances = torch.norm(stack.unsqueeze(1) - stack.unsqueeze(0), dim=2)
    # Average upper triangle (excluding diagonal)
    n = len(samples)
    term2 = torch.sum(torch.triu(pairwise_distances, diagonal=1)) / (n * (n - 1) / 2) if n > 1 else 0.0
    term2 = term2.item()
    
    # Energy Score = term1 - 0.5 * term2
    energy_score = term1 - 0.5 * term2
    return float(energy_score)


def _prepare_field(field: Tensor) -> Tensor:
    """
    Mise en forme standard pour le calcul du spectre (2D).
    """
    if field.dim() == 4:
        field = field.mean(dim=0)
    if field.dim() == 3:
        return field
    raise ValueError(f"Champ inattendu de forme {tuple(field.shape)} pour le calcul spectral.")


def compute_power_spectrum(field: Tensor) -> Tensor:
    """
    Calcule le spectre de puissance moyen (modulus squared de la FFT 2D).
    """
    prepared = _prepare_field(field)
    centered = prepared - prepared.mean()
    fft = torch.fft.rfft2(centered, dim=(-2, -1))
    power = (fft.real ** 2 + fft.imag ** 2)
    return power.mean(dim=0)


def compute_spectrum_distance(pred: Tensor, target: Tensor) -> float:
    """
    Compare les spectres de puissance (L1 moyen).
    """
    pred_spec = compute_power_spectrum(pred)
    target_spec = compute_power_spectrum(target)
    return torch.mean(torch.abs(pred_spec - target_spec)).item()


def compute_temporal_variance_metrics(
    predictions: Sequence[Tensor],
    targets: Sequence[Tensor],
) -> Dict[str, float]:
    """
    Compare la variabilité temporelle des prédictions et des cibles.

    Calcule la variance le long de la dimension "temps" (échantillons ordonnés),
    puis compare les champs de variance (RMSE et corrélation de Pearson).

    Parameters
    ----------
    predictions : Sequence[Tensor]
        Liste de tenseurs [C, H, W] ou [1, C, H, W] (un par pas de temps / échantillon).
    targets : Sequence[Tensor]
        Liste de tenseurs de même forme que predictions.

    Returns
    -------
    Dict[str, float]
        {"temporal_var_rmse": float, "temporal_var_corr": float}
        Si N < 2, les valeurs sont float("nan").
    """
    nan_result = {"temporal_var_rmse": float("nan"), "temporal_var_corr": float("nan")}
    if len(predictions) < 2 or len(targets) < 2 or len(predictions) != len(targets):
        return nan_result

    # Stack: ensure [N, C, H, W]
    pred_list = [p.squeeze(0) if p.dim() == 4 else p for p in predictions]
    tgt_list = [t.squeeze(0) if t.dim() == 4 else t for t in targets]
    pred_stack = torch.stack(pred_list, dim=0)
    target_stack = torch.stack(tgt_list, dim=0)

    # Variance along dim=0 -> [C, H, W]
    var_pred = pred_stack.var(dim=0)
    var_target = target_stack.var(dim=0)

    # Flatten for scalar metrics
    vp = var_pred.flatten()
    vt = var_target.flatten()

    # RMSE between variance maps
    rmse = torch.sqrt(torch.mean((vp - vt) ** 2)).item()

    # Pearson correlation
    vp_c = vp - vp.mean()
    vt_c = vt - vt.mean()
    eps = 1e-8
    num = (vp_c * vt_c).sum()
    denom = torch.sqrt((vp_c ** 2).sum()) * torch.sqrt((vt_c ** 2).sum()) + eps
    corr = (num / denom).item() if denom.item() > eps else 0.0

    return {"temporal_var_rmse": rmse, "temporal_var_corr": corr}


@dataclass
class MetricReport:
    mse: float
    mae: float
    hist_distance: float
    crps: float
    spectrum_distance: float
    baseline_mse: Optional[float] = None
    baseline_mae: Optional[float] = None
    # Phase 4.1: Advanced metrics
    fss: Optional[float] = None
    wasserstein_distance: Optional[float] = None
    energy_score: Optional[float] = None
    # Phase C4: F1 scores for extreme events
    f1_extremes: Optional[Dict[str, float]] = None
    # Phase 1 / 8 — métriques probabilistes (membre unique + CRPS spatial)
    mse_single_member: Optional[float] = None
    mae_single_member: Optional[float] = None
    corr_single_member: Optional[float] = None
    crps_spatial_mean: Optional[float] = None
    spectral_distance_rapsd: Optional[float] = None
    spread_skill_ratio: Optional[float] = None
    crps_p99: Optional[float] = None
    primary_kpi: str = "crps"


def evaluate_metrics(
    samples: Sequence[DiffusionOutput],
    target: Tensor,
    baseline: Optional[Tensor] = None,
    *,
    compute_advanced: bool = True,
    fss_threshold: Optional[float] = None,
    fss_window_size: int = 9,
    include_f1_extremes: bool = True,
    f1_percentiles: Sequence[float] = [95.0, 99.0],
    use_mean_aggregation: bool = False,
    valid_mask: Optional[Tensor] = None,
    crps_max_ensemble_members: Optional[int] = None,
) -> MetricReport:
    """
    Métriques à partir des échantillons. Par défaut : **membre unique** (Phase 1) pour MSE/MAE/spectre ;
    CRPS et métriques d'ensemble utilisent tout l'ensemble. Option ``use_mean_aggregation=True`` : ancien comportement (moyenne).
    ``valid_mask`` [H,W] bool True = pixel valide (Phase 5).
    ``crps_max_ensemble_members`` : borne la taille d'ensemble pour CRPS / carte CRPS (coût O(n²)) ; None = pas de borne.
    """
    if len(samples) == 0:
        raise ValueError("La liste d'échantillons ne doit pas être vide.")
    stacked_means = torch.stack([sample.t_mean for sample in samples], dim=0)
    pred_ensemble_mean = stacked_means.mean(dim=0)
    pred_primary = pred_ensemble_mean if use_mean_aggregation else stacked_means[0]

    mask_t = valid_mask
    if mask_t is not None and mask_t.dim() == 3:
        mask_t = mask_t.squeeze(0)
    if mask_t is not None and mask_t.dim() == 3:
        mask_t = mask_t[0]

    mse = compute_mse(pred_primary, target, mask=None)
    mae = compute_mae(pred_primary, target, mask=None)
    hist_distance = compute_histogram_distance(pred_primary, target)
    if (
        crps_max_ensemble_members is not None
        and crps_max_ensemble_members > 0
        and len(samples) > crps_max_ensemble_members
    ):
        samples_crps = list(samples)[: crps_max_ensemble_members]
    else:
        samples_crps = list(samples)
    crps = compute_crps([sample.t_mean for sample in samples_crps], target)
    spectrum = compute_spectrum_distance(pred_primary, target)

    # --- Phase 1 / 8 : membre unique + CRPS spatial + RAPSD ---
    pred_single = stacked_means[0]
    mse_single = mae_single = corr_single = None
    crps_spatial_mean = spectral_dist_rapsd = spread_skill = crps_p99 = None
    try:
        ps_full = stacked_means.detach().cpu()
        ps = torch.stack([sample.t_mean for sample in samples_crps], dim=0).detach().cpu()
        tg = target.detach().cpu()
        if tg.dim() == 3:
            tg = tg.unsqueeze(0)
        crps_map = compute_crps_pixel_map(ps, tg)
        crps_spatial_mean = float(crps_map.mean().item())
        crps_p99 = float(torch.quantile(crps_map.flatten(), 0.99).item())

        p0 = pred_single.detach().cpu().numpy()
        t0 = tg.squeeze(0).numpy()
        if p0.ndim == 3:
            p0 = p0[0]
        if t0.ndim == 3:
            t0 = t0[0]
        rp = compute_rapsd_numpy(p0)
        rt = compute_rapsd_numpy(t0)
        nbin = min(len(rp), len(rt))
        spectral_dist_rapsd = float(np.mean((rp[:nbin] - rt[:nbin]) ** 2))

        mse_single = float(compute_mse(pred_single, target, mask=None))
        mae_single = float(compute_mae(pred_single, target, mask=None))
        pf = p0.flatten()
        tf = t0.flatten()
        valid = np.isfinite(pf) & np.isfinite(tf)
        if np.sum(valid) > 2:
            corr_single = float(np.corrcoef(pf[valid], tf[valid])[0, 1])

        if mask_t is not None:
            m_np = mask_t.detach().cpu().numpy().astype(bool)
            if m_np.ndim == 3:
                m_np = m_np[0]
            err_abs = np.abs(p0 - t0)
            std_map = ps_full[:, 0].numpy().std(axis=0) if ps_full.shape[1] >= 1 else np.zeros_like(p0)
            spread_skill = compute_spread_skill_ratio(std_map, err_abs, m_np)
            mse = float(np.mean(((p0 - t0) ** 2)[m_np]))
            mae = float(np.mean(err_abs[m_np]))
    except Exception as ex:
        warnings.warn(f"Métriques probabilistes étendues: {ex}")

    baseline_mse = baseline_mae = None
    baseline_tensor = baseline
    if baseline_tensor is None and samples[0].baseline is not None:
        baseline_tensor = samples[0].baseline
    if baseline_tensor is not None:
        baseline_mse = compute_mse(baseline_tensor, target)
        baseline_mae = compute_mae(baseline_tensor, target)

    fss_val = None
    wasserstein_val = None
    energy_score_val = None

    if compute_advanced:
        try:
            if fss_threshold is not None:
                fss_val = compute_fss(pred_primary, target, threshold=fss_threshold, window_size=fss_window_size)
            wasserstein_val = compute_wasserstein_distance([sample.t_mean for sample in samples], target)
            energy_score_val = compute_energy_score([sample.t_mean for sample in samples], target)
        except Exception as e:
            warnings.warn(f"Failed to compute advanced metrics: {e}")

    f1_extremes_val = None
    if include_f1_extremes:
        try:
            f1_extremes_val = compute_f1_extremes(pred_primary, target, threshold_percentiles=f1_percentiles)
        except Exception as e:
            warnings.warn(f"Failed to compute F1 extremes: {e}")

    return MetricReport(
        mse=mse,
        mae=mae,
        hist_distance=hist_distance,
        crps=crps,
        spectrum_distance=spectrum,
        baseline_mse=baseline_mse,
        baseline_mae=baseline_mae,
        fss=fss_val,
        wasserstein_distance=wasserstein_val,
        energy_score=energy_score_val,
        f1_extremes=f1_extremes_val,
        mse_single_member=mse_single,
        mae_single_member=mae_single,
        corr_single_member=corr_single,
        crps_spatial_mean=crps_spatial_mean,
        spectral_distance_rapsd=spectral_dist_rapsd,
        spread_skill_ratio=spread_skill,
        crps_p99=crps_p99,
        primary_kpi="crps",
    )


def plot_probabilistic_dashboard_3x3(
    tgt_display: np.ndarray,
    pred_single: np.ndarray,
    err_display: np.ndarray,
    pred_std: np.ndarray,
    crps_map: np.ndarray,
    mask_display: np.ndarray,
    rapsd_pred: np.ndarray,
    rapsd_tgt: np.ndarray,
    spread_skill_ratio_val: float,
    spearman_rho: float,
    *,
    title: str = "Dashboard probabiliste ST-CDGM",
) -> plt.Figure:
    """
    Grille 3×3 : cible, prédiction, erreur masquée, écart-type, CRPS, masque, spread-skill, RAPSD, calibration.
    """
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    ax = axes.flatten()
    im0 = ax[0].imshow(tgt_display, origin="lower", cmap="Blues")
    ax[0].set_title("Cible HR (blanc = manquant)")
    plt.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(pred_single, origin="lower", cmap="Blues")
    ax[1].set_title("Prédiction (1 membre)")
    plt.colorbar(im1, ax=ax[1], fraction=0.046)
    im2 = ax[2].imshow(err_display, origin="lower", cmap="hot")
    ax[2].set_title("|Erreur| masquée")
    plt.colorbar(im2, ax=ax[2], fraction=0.046)
    im3 = ax[3].imshow(pred_std, origin="lower", cmap="plasma")
    ax[3].set_title("Écart-type intra-ensemble")
    plt.colorbar(im3, ax=ax[3], fraction=0.046)
    im4 = ax[4].imshow(crps_map, origin="lower", cmap="YlOrRd")
    ax[4].set_title("CRPS pixel")
    plt.colorbar(im4, ax=ax[4], fraction=0.046)
    im5 = ax[5].imshow(mask_display.astype(float), origin="lower", cmap="gray_r", vmin=0, vmax=1)
    ax[5].set_title("Masque valide")
    plt.colorbar(im5, ax=ax[5], fraction=0.046)
    ax[6].scatter(pred_std.flatten()[:: max(1, pred_std.size // 5000)], err_display.flatten()[:: max(1, err_display.size // 5000)], alpha=0.15, s=1, c="steelblue")
    mx = float(np.nanmax([np.nanmax(pred_std), np.nanmax(err_display)]))
    ax[6].plot([0, mx], [0, mx], "r--", label="1:1")
    ax[6].set_xlabel("Spread")
    ax[6].set_ylabel("|Erreur|")
    ax[6].set_title("Spread–Skill")
    ax[6].legend()
    rad = np.arange(len(rapsd_pred))
    ax[7].loglog(rad[1:], np.maximum(rapsd_pred[1:], 1e-20), label="Préd")
    ax[7].loglog(rad[1:], np.maximum(rapsd_tgt[1:], 1e-20), label="Cible")
    ax[7].set_title("RAPSD")
    ax[7].legend()
    ax[8].axis("off")
    ax[8].text(
        0.1,
        0.7,
        f"Spread/Skill ≈ {spread_skill_ratio_val:.3f}\nSpearman ρ (dispersion vs |y|) ≈ {spearman_rho:.3f}",
        fontsize=12,
        family="monospace",
        transform=ax[8].transAxes,
    )
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig


def spearman_dispersion_intensity(
    pred_std: np.ndarray,
    target_abs: np.ndarray,
    valid_mask: np.ndarray,
) -> float:
    """Corrélation rang dispersion vs intensité observée (Phase 7.3), sans scipy."""
    ps = pred_std[valid_mask].flatten()
    ta = target_abs[valid_mask].flatten()
    if ps.size < 10:
        return float("nan")

    def _rank(a: np.ndarray) -> np.ndarray:
        return np.argsort(np.argsort(a))

    rp = _rank(ps)
    rt = _rank(ta)
    return float(np.corrcoef(rp, rt)[0, 1])


# ---------------------------------------------------------------------------
# Visualisation et export du DAG
# ---------------------------------------------------------------------------

def plot_dag_heatmap(A_matrix: Tensor, var_names: Sequence[str], *, output_path: Optional[Path] = None) -> None:
    """
    Trace une heatmap du DAG appris.
    """
    A_np = A_matrix.detach().cpu().numpy()
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        A_np,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        xticklabels=var_names,
        yticklabels=var_names,
        center=0.0,
    )
    plt.title("Matrice DAG Apprise")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path)
    else:
        plt.show()
    plt.close()


def export_dag_to_csv(A_matrix: Tensor, var_names: Sequence[str], output_path: Path) -> None:
    """
    Exporte la matrice DAG en CSV.
    """
    A_np = A_matrix.detach().cpu().numpy()
    df = pd.DataFrame(A_np, index=var_names, columns=var_names)
    df.to_csv(output_path, index=True)


def export_dag_to_json(A_matrix: Tensor, var_names: Sequence[str], output_path: Path) -> None:
    """
    Exporte la matrice DAG en JSON (liste d'arêtes).
    """
    A_np = A_matrix.detach().cpu().numpy()
    edges = []
    for i, src in enumerate(var_names):
        for j, dst in enumerate(var_names):
            weight = float(A_np[i, j])
            if weight != 0.0:
                edges.append({"source": src, "target": dst, "weight": weight})
    
    with open(output_path, "w") as f:
        json.dump({"edges": edges, "variables": list(var_names)}, f, indent=2)


def compute_structural_hamming_distance(A_pred: Tensor, A_true: Tensor, threshold: float = 0.0) -> int:
    """
    Calcule la Structural Hamming Distance (SHD) entre deux DAGs.
    
    Phase 4.2: SHD measures the number of edge additions, deletions, or reversals
    needed to transform the predicted DAG into the true DAG.
    
    Parameters
    ----------
    A_pred : Tensor
        Matrice DAG prédite [q, q]
    A_true : Tensor
        Matrice DAG de référence [q, q]
    threshold : float
        Seuil pour binariser les matrices (0.0 = strict, >0 pour seuiller)
    
    Returns
    -------
    int
        Structural Hamming Distance (nombre d'erreurs d'arêtes)
    
    References
    ----------
    - Tsamardinos et al. (2006): "The max-min hill-climbing Bayesian network structure
      learning algorithm"
    """
    # Convert to numpy and binarize
    A_pred_np = A_pred.detach().cpu().numpy().copy()
    A_true_np = A_true.detach().cpu().numpy().copy()
    
    # Apply threshold to binarize (optional)
    if threshold > 0:
        A_pred_binary = (np.abs(A_pred_np) > threshold).astype(int)
        A_true_binary = (np.abs(A_true_np) > threshold).astype(int)
    else:
        # Use non-zero as threshold
        A_pred_binary = (A_pred_np != 0).astype(int)
        A_true_binary = (A_true_np != 0).astype(int)
    
    # Ensure diagonal is zero (no self-loops in DAGs)
    np.fill_diagonal(A_pred_binary, 0)
    np.fill_diagonal(A_true_binary, 0)
    
    # Count differences
    # SHD = number of edges in pred but not in true +
    #       number of edges in true but not in pred +
    #       number of reversed edges
    
    # Find edges present in each DAG
    pred_edges = set()
    true_edges = set()
    
    q = A_pred_binary.shape[0]
    for i in range(q):
        for j in range(q):
            if A_pred_binary[i, j] != 0:
                pred_edges.add((i, j))
            if A_true_binary[i, j] != 0:
                true_edges.add((i, j))
    
    # Count edge additions (in pred but not in true)
    additions = pred_edges - true_edges
    
    # Count edge deletions (in true but not in pred)
    deletions = true_edges - pred_edges
    
    # Count reversals: edge (i,j) in pred and (j,i) in true (or vice versa)
    reversals = 0
    for edge in additions:
        if (edge[1], edge[0]) in deletions:
            # This edge is reversed
            reversals += 1
            additions.discard(edge)
            deletions.discard((edge[1], edge[0]))
    
    # SHD = additions + deletions + reversals
    shd = len(additions) + len(deletions) + reversals
    
    return int(shd)
```

---

### `src/st_cdgm/utils/__init__.py`

```python
"""Small utilities for ST-CDGM."""

from .checkpoint import strip_torch_compile_prefix

__all__ = ["strip_torch_compile_prefix"]
```

---

### `src/st_cdgm/utils/checkpoint.py`

```python
"""Helpers for loading PyTorch checkpoints saved under different wrappers."""

from __future__ import annotations

from typing import Any, Dict


def strip_torch_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove ``_orig_mod.`` key prefix produced by ``torch.compile`` when loading
    into a non-compiled (eager) module.
    """
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if not any(str(k).startswith("_orig_mod.") for k in keys):
        return state_dict
    return {
        (k[len("_orig_mod.") :] if str(k).startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
```

---

### `ops/train_st_cdgm.py`

```python
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
    n_threads = max(1, (os.cpu_count() or 1) // 2)
    torch.set_num_threads(n_threads)
    print(f"[PERF] torch.set_num_threads({n_threads})")

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

    spatial_projector = SpatialConditioningProjector(
        num_vars=len(encoder.configs),
        hidden_dim=cfg.rcn.hidden_dim,
        conditioning_dim=cfg.diffusion.conditioning_dim,
        lr_shape=tuple(cfg.graph.lr_shape),
        target_shape=tuple(cfg.diffusion.get("spatial_target_shape", [6, 7])),
    ).to(device)

    params = (
        list(encoder.parameters()) + list(rcn_cell.parameters())
        + list(diffusion.parameters()) + list(spatial_projector.parameters())
    )
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
```

---

### `ops/preprocess_to_zarr.py`

```python
"""
Script de pré-traitement pour convertir des données NetCDF en format Zarr optimisé.

Ce script applique toutes les transformations nécessaires (normalisation, baseline,
transformations) et écrit les données en format Zarr avec chunks optimisés pour
l'entraînement ST-CDGM.

Usage:
    python ops/preprocess_to_zarr.py \
        --lr_path data/raw/lr.nc \
        --hr_path data/raw/hr.nc \
        --output_dir data/zarr/ \
        --seq_len 10 \
        --baseline_strategy hr_smoothing \
        --normalize
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional, Sequence

# Ensure st_cdgm is importable when run as subprocess (e.g. from notebook)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_src_path = _project_root / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

import numpy as np
import xarray as xr
import zarr

from st_cdgm import NetCDFDataPipeline


def convert_netcdf_to_zarr(
    lr_path: Path,
    hr_path: Path,
    output_dir: Path,
    *,
    static_path: Optional[Path] = None,
    seq_len: int = 10,
    baseline_strategy: str = "hr_smoothing",
    baseline_factor: int = 4,
    normalize: bool = False,
    target_transform: Optional[str] = None,
    lr_variables: Optional[Sequence[str]] = None,
    hr_variables: Optional[Sequence[str]] = None,
    static_variables: Optional[Sequence[str]] = None,
    means_path: Optional[Path] = None,
    stds_path: Optional[Path] = None,
    chunk_size_time: Optional[int] = None,
    chunk_size_lat: Optional[int] = None,
    chunk_size_lon: Optional[int] = None,
    compressor: Optional[zarr.codec.Codec] = None,
) -> None:
    """
    Convertit des données NetCDF en format Zarr optimisé.

    Parameters
    ----------
    lr_path, hr_path, static_path :
        Chemins vers les fichiers NetCDF d'entrée.
    output_dir :
        Répertoire de sortie pour les magasins Zarr.
    seq_len :
        Longueur de séquence pour l'entraînement (utilisée pour optimiser les chunks).
    baseline_strategy, baseline_factor :
        Stratégie de calcul du baseline.
    normalize :
        Activer la normalisation LR.
    target_transform :
        Transformation à appliquer (None, "log", "log1p").
    lr_variables, hr_variables, static_variables :
        Variables à sélectionner.
    means_path, stds_path :
        Chemins vers les statistiques de normalisation pré-calculées.
    chunk_size_time, chunk_size_lat, chunk_size_lon :
        Tailles de chunks personnalisées. Si None, calculées automatiquement.
    compressor :
        Compresseur Zarr (par défaut: Blosc avec compression LZ4).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compresseur par défaut (compression rapide, bon ratio)
    if compressor is None:
        compressor = zarr.Blosc(
            cname="lz4",  # Compression rapide
            clevel=3,  # Niveau de compression modéré
            shuffle=zarr.Blosc.BITSHUFFLE,  # Bon pour données numériques
        )

    print("=" * 80)
    print("🔄 CONVERSION NETCDF → ZARR")
    print("=" * 80)
    print(f"📂 Répertoire de sortie: {output_dir}")
    print(f"📊 Longueur de séquence: {seq_len}")
    print(f"⚙️  Stratégie baseline: {baseline_strategy}")
    print(f"📈 Normalisation: {normalize}")
    print()

    # Étape 1 : Créer le pipeline NetCDF pour appliquer toutes les transformations
    print("📥 Chargement et préparation des données NetCDF...")
    pipeline = NetCDFDataPipeline(
        lr_path=lr_path,
        hr_path=hr_path,
        static_path=static_path,
        seq_len=seq_len,
        baseline_strategy=baseline_strategy,
        baseline_factor=baseline_factor,
        target_transform=target_transform,
        normalize=normalize,
        lr_variables=lr_variables,
        hr_variables=hr_variables,
        static_variables=static_variables,
        means_path=means_path,
        stds_path=stds_path,
    )

    # Étape 2 : Récupérer les datasets préparés
    lr_dataset = pipeline.lr_dataset
    hr_dataset = pipeline.hr_dataset
    baseline_dataset = pipeline.baseline_prepared
    residual_dataset = pipeline.residual_dataset
    static_dataset = pipeline.static_dataset

    dims = pipeline.dims

    # Étape 3 : Déterminer les tailles de chunks optimales
    print("\n🔧 Configuration des chunks Zarr...")
    
    # Chunks temporels : multiple de seq_len pour optimiser l'accès
    time_dim_size = lr_dataset.dims[dims.time]
    if chunk_size_time is None:
        # Choisir un multiple de seq_len qui donne des chunks raisonnables
        # Objectif: chunks de ~100-500 pas de temps
        chunk_size_time = min(max(seq_len * 10, 100), time_dim_size // 4, 500)
        # S'assurer que c'est un multiple de seq_len
        chunk_size_time = (chunk_size_time // seq_len) * seq_len
    
    # Chunks spatiaux : taille raisonnable (64-128 pixels typiquement)
    lr_lat_size = lr_dataset.dims[dims.lr_lat]
    lr_lon_size = lr_dataset.dims[dims.lr_lon]
    hr_lat_size = hr_dataset.dims[dims.hr_lat]
    hr_lon_size = hr_dataset.dims[dims.hr_lon]
    
    if chunk_size_lat is None:
        chunk_size_lat_lr = min(64, lr_lat_size)
        chunk_size_lat_hr = min(64, hr_lat_size)
    else:
        chunk_size_lat_lr = min(chunk_size_lat, lr_lat_size)
        chunk_size_lat_hr = min(chunk_size_lat, hr_lat_size)
    
    if chunk_size_lon is None:
        chunk_size_lon_lr = min(64, lr_lon_size)
        chunk_size_lon_hr = min(64, hr_lon_size)
    else:
        chunk_size_lon_lr = min(chunk_size_lon, lr_lon_size)
        chunk_size_lon_hr = min(chunk_size_lon, hr_lon_size)

    print(f"   LR chunks: ({chunk_size_time}, {chunk_size_lat_lr}, {chunk_size_lon_lr})")
    print(f"   HR chunks: ({chunk_size_time}, {chunk_size_lat_hr}, {chunk_size_lon_hr})")

    # Étape 4 : Convertir et sauvegarder en Zarr
    print("\n💾 Écriture en format Zarr...")

    # LR dataset
    lr_zarr_path = output_dir / "lr.zarr"
    print(f"   LR dataset → {lr_zarr_path}")
    lr_dataset.to_zarr(
        lr_zarr_path,
        mode="w",
        encoding={
            var: {
                "chunks": (chunk_size_time, lr_lat_size, lr_lon_size),
                "compressor": compressor,
            }
            for var in lr_dataset.data_vars
        },
    )

    # HR dataset
    hr_zarr_path = output_dir / "hr.zarr"
    print(f"   HR dataset → {hr_zarr_path}")
    hr_dataset.to_zarr(
        hr_zarr_path,
        mode="w",
        encoding={
            var: {
                "chunks": (chunk_size_time, hr_lat_size, hr_lon_size),
                "compressor": compressor,
            }
            for var in hr_dataset.data_vars
        },
    )

    # Baseline dataset
    baseline_zarr_path = output_dir / "baseline.zarr"
    print(f"   Baseline dataset → {baseline_zarr_path}")
    baseline_dataset.to_zarr(
        baseline_zarr_path,
        mode="w",
        encoding={
            var: {
                "chunks": (chunk_size_time, hr_lat_size, hr_lon_size),
                "compressor": compressor,
            }
            for var in baseline_dataset.data_vars
        },
    )

    # Residual dataset
    residual_zarr_path = output_dir / "residual.zarr"
    print(f"   Residual dataset → {residual_zarr_path}")
    residual_dataset.to_zarr(
        residual_zarr_path,
        mode="w",
        encoding={
            var: {
                "chunks": (chunk_size_time, hr_lat_size, hr_lon_size),
                "compressor": compressor,
            }
            for var in residual_dataset.data_vars
        },
    )

    # Static dataset (si présent)
    if static_dataset is not None:
        static_zarr_path = output_dir / "static.zarr"
        print(f"   Static dataset → {static_zarr_path}")
        static_dataset.to_zarr(
            static_zarr_path,
            mode="w",
            encoding={
                var: {
                    "chunks": (hr_lat_size, hr_lon_size),
                    "compressor": compressor,
                }
                for var in static_dataset.data_vars
            },
        )

    # Étape 5 : Sauvegarder les statistiques de normalisation (si disponibles)
    if normalize and pipeline.lr_stats:
        stats_dir = output_dir / "stats"
        stats_dir.mkdir(exist_ok=True)
        
        stats = pipeline.lr_stats
        if "mean" in stats or len(stats) > 0:
            # Save means if available
            mean_ds = stats.get("mean")
            if mean_ds is None and len(stats) > 0:
                # If stats dict has datasets, save the first one as mean
                mean_ds = list(stats.values())[0]
            if mean_ds is not None:
                mean_path = stats_dir / "mean.zarr"
                mean_ds.to_zarr(mean_path, mode="w")
                print(f"   LR mean stats → {mean_path}")
        
        if "std" in stats or len(stats) > 1:
            # Save stds if available
            std_ds = stats.get("std")
            if std_ds is None and len(stats) > 1:
                # If stats dict has multiple datasets, save the second one as std
                std_ds = list(stats.values())[1]
            if std_ds is not None:
                std_path = stats_dir / "stds.zarr"
                std_ds.to_zarr(std_path, mode="w")
                print(f"   LR std stats → {std_path}")

    # Étape 6 : Sauvegarder les métadonnées
    metadata = {
        "seq_len": seq_len,
        "baseline_strategy": baseline_strategy,
        "baseline_factor": baseline_factor,
        "normalize": normalize,
        "target_transform": target_transform,
        "dims": {
            "time": dims.time,
            "lr_lat": dims.lr_lat,
            "lr_lon": dims.lr_lon,
            "hr_lat": dims.hr_lat,
            "hr_lon": dims.hr_lon,
        },
        "chunk_sizes": {
            "time": chunk_size_time,
            "lr_lat": chunk_size_lat_lr,
            "lr_lon": chunk_size_lon_lr,
            "hr_lat": chunk_size_lat_hr,
            "hr_lon": chunk_size_lon_hr,
        },
    }

    import json

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n📋 Métadonnées → {metadata_path}")

    print("\n" + "=" * 80)
    print("✅ CONVERSION TERMINÉE")
    print("=" * 80)
    print(f"📁 Données Zarr disponibles dans: {output_dir}")
    print("\n💡 Pour utiliser ces données, utilisez ZarrDataPipeline au lieu de NetCDFDataPipeline")


def main():
    parser = argparse.ArgumentParser(
        description="Convertir des données NetCDF en format Zarr optimisé pour ST-CDGM"
    )
    parser.add_argument("--lr_path", type=Path, required=True, help="Chemin vers le dataset LR NetCDF")
    parser.add_argument("--hr_path", type=Path, required=True, help="Chemin vers le dataset HR NetCDF")
    parser.add_argument("--output_dir", type=Path, required=True, help="Répertoire de sortie pour les données Zarr")
    parser.add_argument("--static_path", type=Path, default=None, help="Chemin vers le dataset statique NetCDF (optionnel)")
    parser.add_argument("--seq_len", type=int, default=10, help="Longueur de séquence (pour optimiser les chunks)")
    parser.add_argument("--baseline_strategy", type=str, default="hr_smoothing", choices=["hr_smoothing", "lr_interp"], help="Stratégie de baseline")
    parser.add_argument("--baseline_factor", type=int, default=4, help="Facteur de coarsening pour hr_smoothing")
    parser.add_argument("--normalize", action="store_true", help="Activer la normalisation LR")
    parser.add_argument("--target_transform", type=str, default=None, choices=[None, "log", "log1p"], help="Transformation à appliquer")
    parser.add_argument("--lr_variables", type=str, nargs="+", default=None, help="Variables LR à inclure")
    parser.add_argument("--hr_variables", type=str, nargs="+", default=None, help="Variables HR à inclure")
    parser.add_argument("--static_variables", type=str, nargs="+", default=None, help="Variables statiques à inclure")
    parser.add_argument("--means_path", type=Path, default=None, help="Chemin vers les moyennes pré-calculées")
    parser.add_argument("--stds_path", type=Path, default=None, help="Chemin vers les écarts-types pré-calculés")
    parser.add_argument("--chunk_size_time", type=int, default=None, help="Taille de chunk temporelle (auto si None)")
    parser.add_argument("--chunk_size_lat", type=int, default=None, help="Taille de chunk latitude (auto si None)")
    parser.add_argument("--chunk_size_lon", type=int, default=None, help="Taille de chunk longitude (auto si None)")

    args = parser.parse_args()

    convert_netcdf_to_zarr(
        lr_path=args.lr_path,
        hr_path=args.hr_path,
        output_dir=args.output_dir,
        static_path=args.static_path,
        seq_len=args.seq_len,
        baseline_strategy=args.baseline_strategy,
        baseline_factor=args.baseline_factor,
        normalize=args.normalize,
        target_transform=args.target_transform,
        lr_variables=args.lr_variables,
        hr_variables=args.hr_variables,
        static_variables=args.static_variables,
        means_path=args.means_path,
        stds_path=args.stds_path,
        chunk_size_time=args.chunk_size_time,
        chunk_size_lat=args.chunk_size_lat,
        chunk_size_lon=args.chunk_size_lon,
    )


if __name__ == "__main__":
    main()
```

---

### `ops/preprocess_to_shards.py`

```python
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
import sys
import tarfile
from pathlib import Path
from typing import Dict, Optional

# Ensure st_cdgm is importable when run as subprocess (e.g. from notebook)
_script_dir = Path(__file__).resolve().parent
_project_root = _script_dir.parent
_src_path = _project_root / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

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
```

---

### `scripts/cleanup_repeated_lines.py`

```python
"""
Script pour nettoyer les lignes répétées dans un notebook Jupyter.
"""
import json
import sys
from pathlib import Path

def clean_repeated_lines(notebook_path: Path, max_repeats: int = 3):
    """
    Nettoie les lignes répétées consécutives dans un notebook.
    
    Parameters
    ----------
    notebook_path : Path
        Chemin vers le notebook à nettoyer
    max_repeats : int
        Nombre maximum de répétitions consécutives autorisées
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"Fichier non trouve: {notebook_path}")
        return
    
    print(f"Lecture du notebook: {notebook_path}")
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    total_removed = 0
    cells_modified = 0
    
    for cell_idx, cell in enumerate(nb['cells']):
        if 'source' not in cell:
            continue
        
        source = cell['source']
        if isinstance(source, str):
            source = source.split('\n')
            was_string = True
        else:
            was_string = False
        
        # Supprimer les lignes répétées consécutives
        cleaned_source = []
        prev_line = None
        repeat_count = 0
        
        for line in source:
            # Ignorer les lignes vides dans le comptage de répétitions
            line_stripped = line.strip()
            
            if line_stripped == prev_line and line_stripped:  # Ignorer les lignes vides
                repeat_count += 1
                if repeat_count <= max_repeats:
                    cleaned_source.append(line)
                else:
                    total_removed += 1
            else:
                repeat_count = 1
                cleaned_source.append(line)
                if line_stripped:
                    prev_line = line_stripped
        
        # Restaurer le format original
        if was_string:
            cleaned_source = '\n'.join(cleaned_source)
        else:
            # Pour les listes, garder les nouvelles lignes dans les chaînes
            cleaned_source = [line if line.endswith('\n') or i == len(cleaned_source) - 1 
                             else line + '\n' if not line.endswith('\n') else line
                             for i, line in enumerate(cleaned_source)]
        
        if cleaned_source != source:
            cell['source'] = cleaned_source
            cells_modified += 1
            print(f"  Cellule {cell_idx}: {len(source) - len(cleaned_source)} lignes repetees supprimees")
    
    if cells_modified > 0:
        # Créer une sauvegarde
        backup_path = notebook_path.with_suffix('.ipynb.bak')
        print(f"Sauvegarde creee: {backup_path}")
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        # Sauvegarder le fichier nettoyé
        print(f"Sauvegarde du fichier nettoye: {notebook_path}")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        
        print(f"\nNettoyage termine!")
        print(f"   - {cells_modified} cellule(s) modifiee(s)")
        print(f"   - {total_removed} ligne(s) repetee(s) supprimee(s)")
    else:
        print("\nAucune ligne repetee trouvee.")

if __name__ == "__main__":
    notebook_path = Path("../st_cdgm_training_evaluation.ipynb")
    if len(sys.argv) > 1:
        notebook_path = Path(sys.argv[1])
    
    clean_repeated_lines(notebook_path, max_repeats=1)  # Garder seulement 1 occurrence
```

---

### `scripts/load_model.py`

```python
"""
Utility script to load a model checkpoint.

This module provides functions to load ST-CDGM model checkpoints and restore
the full training state (models, optimizer, config, etc.).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import torch

from st_cdgm import (
    IntelligibleVariableEncoder,
    IntelligibleVariableConfig,
    RCNCell,
    RCNSequenceRunner,
    CausalDiffusionDecoder,
)


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device = torch.device("cpu"),
    return_full_state: bool = False,
) -> Dict[str, Any]:
    """
    Load a model checkpoint.
    
    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file (.pt)
    device : torch.device
        Device to load models onto
    return_full_state : bool
        If True, return optimizer state and full config
    
    Returns
    -------
    Dict containing:
        - encoder: IntelligibleVariableEncoder
        - rcn_cell: RCNCell
        - rcn_runner: RCNSequenceRunner
        - diffusion_decoder: CausalDiffusionDecoder
        - config: DictConfig (if available)
        - metrics: dict (if available)
        - optimizer_state_dict: dict (if return_full_state=True)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract components
    encoder_state = checkpoint["encoder_state_dict"]
    rcn_state = checkpoint.get("rcn_state_dict") or checkpoint.get("rcn_cell_state_dict")
    if rcn_state is None:
        raise KeyError("Checkpoint missing rcn_state_dict or rcn_cell_state_dict")
    diffusion_state = checkpoint["diffusion_state_dict"]
    config = checkpoint.get("config", {})
    metrics = checkpoint.get("metrics", {})
    
    # Reconstruct models from config
    # This assumes config contains the necessary info to rebuild models
    if not config:
        raise ValueError("Checkpoint must contain config to reconstruct models")
    
    # Build encoder
    encoder_cfg = config.get("encoder", {})
    meta_configs = [
        IntelligibleVariableConfig(
            name=mp.get("name", ""),
            meta_path=(mp["src"], mp["relation"], mp["target"]),
            pool=mp.get("pool", "mean"),
        )
        for mp in encoder_cfg.get("metapaths", [])
    ]
    encoder = IntelligibleVariableEncoder(
        configs=meta_configs,
        hidden_dim=encoder_cfg.get("hidden_dim", 128),
        conditioning_dim=encoder_cfg.get("conditioning_dim", 128),
    )
    encoder.load_state_dict(encoder_state)
    encoder.to(device)
    encoder.eval()
    
    # Build RCN
    rcn_cfg = config.get("rcn", {})
    rcn_cell = RCNCell(
        num_vars=len(encoder_cfg.get("metapaths", [])),
        hidden_dim=rcn_cfg.get("hidden_dim", 128),
        driver_dim=rcn_cfg.get("driver_dim", 8),
        reconstruction_dim=rcn_cfg.get("reconstruction_dim", 8),
        dropout=rcn_cfg.get("dropout", 0.0),
    )
    rcn_cell.load_state_dict(rcn_state)
    rcn_cell.to(device)
    rcn_cell.eval()
    rcn_runner = RCNSequenceRunner(
        rcn_cell,
        detach_interval=rcn_cfg.get("detach_interval", None),
    )
    
    # Build diffusion decoder
    diffusion_cfg = config.get("diffusion", {})
    unet_kwargs = dict(diffusion_cfg.get("unet_kwargs", {})) or dict(
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
    diffusion = CausalDiffusionDecoder(
        in_channels=diffusion_cfg.get("in_channels", 1),
        conditioning_dim=diffusion_cfg.get("conditioning_dim", 128),
        height=diffusion_cfg.get("height", 172),
        width=diffusion_cfg.get("width", 179),
        num_diffusion_steps=diffusion_cfg.get("steps", 1000),
        unet_kwargs=unet_kwargs,
        use_gradient_checkpointing=diffusion_cfg.get("use_gradient_checkpointing", False),
        scheduler_type=diffusion_cfg.get("scheduler_type", "ddpm"),
        conv_padding_mode=diffusion_cfg.get("conv_padding_mode", "zeros"),
        anti_checkerboard=diffusion_cfg.get("anti_checkerboard", False),
    )
    diffusion.load_state_dict(diffusion_state)
    diffusion.to(device)
    diffusion.eval()
    
    result = {
        "encoder": encoder,
        "rcn_cell": rcn_cell,
        "rcn_runner": rcn_runner,
        "diffusion_decoder": diffusion,
        "config": config,
        "metrics": metrics,
    }
    
    if return_full_state and "optimizer_state_dict" in checkpoint:
        result["optimizer_state_dict"] = checkpoint["optimizer_state_dict"]
    
    print("✓ Checkpoint loaded successfully")
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and verify model checkpoint")
    parser.add_argument("checkpoint", type=Path, help="Path to checkpoint file")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device=device)
    
    print("\nCheckpoint Summary:")
    print(f"  Config: {checkpoint['config'] is not None}")
    print(f"  Metrics: {list(checkpoint['metrics'].keys())}")
    print(f"  Models loaded on: {device}")
```

---

### `scripts/run_evaluation.py`

```python
"""
Script d'évaluation pour tester un modèle entraîné ST-CDGM.

Ce script charge un modèle sauvegardé, exécute l'inférence et calcule les métriques
d'évaluation (MSE, MAE, CRPS, FSS, F1 extremes, etc.).

Usage:
    # Avec Docker:
    docker-compose exec st-cdgm-training python scripts/run_evaluation.py \
        --checkpoint models/best_model.pt \
        --data_dir data/processed \
        --output_dir results/evaluation

    # Directement:
    python scripts/run_evaluation.py \
        --checkpoint models/best_model.pt \
        --lr_path data/raw/lr.nc \
        --hr_path data/raw/hr.nc
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add workspace to path for imports
workspace_path = Path(__file__).parent.parent
if str(workspace_path) not in sys.path:
    sys.path.insert(0, str(workspace_path))
if str(workspace_path / "src") not in sys.path:
    sys.path.insert(0, str(workspace_path / "src"))

from st_cdgm import (
    NetCDFDataPipeline,
    HeteroGraphBuilder,
    IntelligibleVariableConfig,
    IntelligibleVariableEncoder,
    RCNCell,
    RCNSequenceRunner,
    CausalDiffusionDecoder,
)
from st_cdgm.evaluation import (
    evaluate_metrics,
    MetricReport,
    run_st_cdgm_inference,
)
# Note: load_model will be imported if needed


def run_evaluation(
    checkpoint_path: Path,
    lr_path: Path,
    hr_path: Path,
    output_dir: Path,
    *,
    static_path: Optional[Path] = None,
    num_samples: int = 10,
    num_inference_steps: int = 25,
    scheduler_type: str = "edm",
    seq_len: int = 6,
    device: str = "cuda",
    compute_f1_extremes: bool = True,
    f1_percentiles: Sequence[float] = [95.0, 99.0],
    save_visualizations: bool = True,
) -> None:
    """Run evaluation on a trained model."""
    
    print("=" * 80)
    print("ST-CDGM Model Evaluation")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"LR Path: {lr_path}")
    print(f"HR Path: {hr_path}")
    print(f"Output Dir: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    device_obj = torch.device(device)
    
    # Load model
    print("\nLoading model...")
    from scripts.load_model import load_checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path, device=device_obj)
    
    # Setup data pipeline
    print("\nSetting up data pipeline...")
    pipeline = NetCDFDataPipeline(
        lr_path=str(lr_path),
        hr_path=str(hr_path),
        static_path=str(static_path) if static_path else None,
        seq_len=seq_len,
        normalize=True,
    )
    
    # Get test data (last sequence)
    dataset = pipeline.build_sequence_dataset(seq_len=seq_len, stride=1, as_torch=True)
    # Get a test sample (could be extended to use validation set)
    test_sample = next(iter(dataset))
    
    encoder = checkpoint_data["encoder"]
    rcn_runner = checkpoint_data["rcn_runner"]
    diffusion_decoder = checkpoint_data["diffusion_decoder"]
    config = checkpoint_data["config"]
    
    # Setup graph builder
    graph_cfg = config.get("graph", {})
    builder = HeteroGraphBuilder(
        lr_shape=tuple(graph_cfg.get("lr_shape", [23, 26])),
        hr_shape=tuple(graph_cfg.get("hr_shape", [172, 179])),
        static_dataset=pipeline.get_static_dataset(),
        static_variables=graph_cfg.get("static_variables", []),
        include_mid_layer=graph_cfg.get("include_mid_layer", True),
    )
    
    # Run inference
    print(f"\nRunning inference ({num_samples} samples, {num_inference_steps} steps, {scheduler_type} scheduler)...")
    
    # Prepare input data
    lr_seq = test_sample["lr"]  # [seq_len, channels, lat, lon]
    target = test_sample["hr"][-1]  # Last timestep [channels, H, W]
    baseline = test_sample.get("baseline", None)
    
    diff_cfg = config.get("diffusion", {}) if isinstance(config, dict) else {}
    cfg_scale = float(diff_cfg.get("cfg_scale", 0.0))
    eval_cfg = config.get("evaluation", {}) if isinstance(config, dict) else {}
    crps_cap = eval_cfg.get("crps_max_ensemble_members")

    samples_out, target_batch, baseline_batch, dag_last, mask_batch = run_st_cdgm_inference(
        test_sample,
        builder=builder,
        encoder=encoder,
        rcn_runner=rcn_runner,
        diffusion=diffusion_decoder,
        device=device_obj,
        num_samples=num_samples,
        num_steps=num_inference_steps,
        scheduler_type=scheduler_type,
        cfg_scale=cfg_scale,
    )

    print("\nComputing metrics...")
    target_tensor = target_batch.to(device_obj)
    baseline_tensor = baseline_batch.to(device_obj) if baseline_batch is not None else None
    vm = mask_batch.to(device_obj)
    while vm.dim() > 2:
        vm = vm[0]
    if vm.dim() == 3:
        vm = vm[0]

    metrics = evaluate_metrics(
        samples=samples_out,
        target=target_tensor,
        baseline=baseline_tensor,
        compute_advanced=True,
        include_f1_extremes=compute_f1_extremes,
        f1_percentiles=list(f1_percentiles),
        valid_mask=vm,
        crps_max_ensemble_members=crps_cap,
    )
    
    # Save results
    results_path = output_dir / "evaluation_results.json"
    results_dict = {
        "mse": metrics.mse,
        "mae": metrics.mae,
        "hist_distance": metrics.hist_distance,
        "crps": metrics.crps,
        "spectrum_distance": metrics.spectrum_distance,
        "fss": metrics.fss,
        "wasserstein_distance": metrics.wasserstein_distance,
        "energy_score": metrics.energy_score,
        "f1_extremes": metrics.f1_extremes,
        "baseline_mse": metrics.baseline_mse,
        "baseline_mae": metrics.baseline_mae,
        "crps_spatial_mean": metrics.crps_spatial_mean,
        "spectral_distance_rapsd": metrics.spectral_distance_rapsd,
        "spread_skill_ratio": metrics.spread_skill_ratio,
        "crps_p99": metrics.crps_p99,
        "primary_kpi": metrics.primary_kpi,
    }
    
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Results saved: {results_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("Evaluation Results Summary")
    print("=" * 80)
    print(f"MSE: {metrics.mse:.6f}")
    print(f"MAE: {metrics.mae:.6f}")
    print(f"CRPS: {metrics.crps:.6f}")
    if metrics.fss is not None:
        print(f"FSS: {metrics.fss:.6f}")
    if metrics.f1_extremes is not None:
        print(f"F1 Extremes:")
        for threshold, f1_score in metrics.f1_extremes.items():
            print(f"  {threshold}: {f1_score:.4f}")
    print("=" * 80)
    
    # Save visualizations if requested
    if save_visualizations:
        print("\nGenerating visualizations...")
        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot sample predictions
        pred_mean = samples_out[0].t_mean.cpu().numpy()
        
        # Simple visualization (can be extended)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(target[0], cmap='viridis')
        axes[0].set_title("Target")
        axes[1].imshow(pred_mean[0], cmap='viridis')
        axes[1].set_title("Prediction (Mean)")
        axes[2].imshow(np.abs(target[0] - pred_mean[0]), cmap='hot')
        axes[2].set_title("Absolute Error")
        
        vis_path = vis_dir / "prediction_comparison.png"
        plt.savefig(vis_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Visualizations saved: {vis_dir}")
    
    print("\n" + "=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ST-CDGM model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint",
    )
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
        help="Path to static NetCDF file (optional)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="results/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of samples for evaluation",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of diffusion steps for inference",
    )
    parser.add_argument(
        "--scheduler_type",
        type=str,
        choices=["ddpm", "edm", "dpm_solver++"],
        default="edm",
        help="Diffusion scheduler type",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=6,
        help="Sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--no_f1_extremes",
        action="store_true",
        help="Disable F1 extremes computation",
    )
    parser.add_argument(
        "--no_visualizations",
        action="store_true",
        help="Disable visualization generation",
    )
    
    args = parser.parse_args()
    
    run_evaluation(
        checkpoint_path=args.checkpoint,
        lr_path=args.lr_path,
        hr_path=args.hr_path,
        output_dir=args.output_dir,
        static_path=args.static_path,
        num_samples=args.num_samples,
        num_inference_steps=args.num_inference_steps,
        scheduler_type=args.scheduler_type,
        seq_len=args.seq_len,
        device=args.device,
        compute_f1_extremes=not args.no_f1_extremes,
        save_visualizations=not args.no_visualizations,
    )


if __name__ == "__main__":
    main()
```

---

### `scripts/run_full_pipeline.py`

```python
"""
Script orchestrateur pour exécuter le pipeline complet ST-CDGM.

Ce script exécute dans l'ordre:
1. Preprocessing (NetCDF → Zarr/WebDataset)
2. Training (avec checkpointing)
3. Evaluation (sur le modèle entraîné)

Usage:
    python scripts/run_full_pipeline.py \
        --lr_path data/raw/lr.nc \
        --hr_path data/raw/hr.nc \
        --config config/training_config.yaml \
        --format zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add workspace to path
workspace_path = Path(__file__).parent.parent
if str(workspace_path) not in sys.path:
    sys.path.insert(0, str(workspace_path))
if str(workspace_path / "src") not in sys.path:
    sys.path.insert(0, str(workspace_path / "src"))

from scripts.run_preprocessing import main as preprocess_main
from scripts.run_training import run_training_with_checkpoints
from scripts.run_evaluation import run_evaluation
from omegaconf import OmegaConf


def run_full_pipeline(
    lr_path: Path,
    hr_path: Path,
    config_path: Path,
    *,
    static_path: Path | None = None,
    format: str = "zarr",
    checkpoint_dir: Path = Path("models"),
    results_dir: Path = Path("results"),
    skip_preprocessing: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False,
) -> None:
    """Run the complete ST-CDGM pipeline."""
    
    print("=" * 80)
    print("ST-CDGM Full Pipeline")
    print("=" * 80)
    print(f"LR Path: {lr_path}")
    print(f"HR Path: {hr_path}")
    print(f"Config: {config_path}")
    print(f"Format: {format}")
    print("=" * 80)
    
    # Load config
    cfg = OmegaConf.load(config_path)
    
    # Step 1: Preprocessing
    processed_dir = Path("data/processed")
    if not skip_preprocessing:
        print("\n" + "=" * 80)
        print("STEP 1: Preprocessing")
        print("=" * 80)
        
        # Prepare preprocessing arguments
        preprocess_args = [
            "--lr_path", str(lr_path),
            "--hr_path", str(hr_path),
            "--format", format,
            "--output_dir", str(processed_dir),
            "--seq_len", str(cfg.data.get("seq_len", 10)),
            "--baseline_strategy", cfg.data.get("baseline_strategy", "hr_smoothing"),
        ]
        
        if static_path:
            preprocess_args.extend(["--static_path", str(static_path)])
        if cfg.data.get("normalize", False):
            preprocess_args.append("--normalize")
        
        # Run preprocessing (we'll need to modify to accept args)
        # For now, call the main function directly
        import subprocess
        result = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "run_preprocessing.py")] + preprocess_args,
            check=True,
        )
        print("✓ Preprocessing completed")
    else:
        print("\n⏭ Skipping preprocessing (using existing data)")
    
    # Step 2: Training
    if not skip_training:
        print("\n" + "=" * 80)
        print("STEP 2: Training")
        print("=" * 80)
        
        run_training_with_checkpoints(
            cfg,
            checkpoint_dir=checkpoint_dir,
            save_every=cfg.checkpoint.get("save_every", 5),
            max_checkpoints=cfg.checkpoint.get("max_checkpoints", 5),
        )
        print("✓ Training completed")
    else:
        print("\n⏭ Skipping training")
    
    # Step 3: Evaluation
    if not skip_evaluation:
        print("\n" + "=" * 80)
        print("STEP 3: Evaluation")
        print("=" * 80)
        
        # Find best model checkpoint
        best_checkpoint = checkpoint_dir / "best_model.pt"
        if not best_checkpoint.exists():
            # Find latest checkpoint
            checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
            if checkpoints:
                best_checkpoint = checkpoints[-1]
            else:
                print("⚠ No checkpoint found, skipping evaluation")
                return
        
        run_evaluation(
            checkpoint_path=best_checkpoint,
            lr_path=lr_path,
            hr_path=hr_path,
            output_dir=results_dir / "evaluation",
            static_path=static_path,
            num_samples=cfg.evaluation.get("num_samples", 10),
            num_inference_steps=25,
            scheduler_type=cfg.diffusion.get("scheduler_type", "edm"),
            seq_len=cfg.data.get("seq_len", 6),
            device=cfg.training.get("device", "cuda"),
            compute_f1_extremes=cfg.evaluation.get("compute_f1_extremes", True),
            save_visualizations=cfg.evaluation.get("save_visualizations", True),
        )
        print("✓ Evaluation completed")
    else:
        print("\n⏭ Skipping evaluation")
    
    print("\n" + "=" * 80)
    print("✓ Full pipeline completed successfully!")
    print("=" * 80)
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Results: {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete ST-CDGM pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--lr_path", type=Path, required=True)
    parser.add_argument("--hr_path", type=Path, required=True)
    parser.add_argument("--static_path", type=Path, default=None)
    parser.add_argument("--config", type=Path, default="config/training_config.yaml")
    parser.add_argument("--format", type=str, choices=["zarr", "webdataset"], default="zarr")
    parser.add_argument("--checkpoint_dir", type=Path, default="models")
    parser.add_argument("--results_dir", type=Path, default="results")
    parser.add_argument("--skip_preprocessing", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    
    args = parser.parse_args()
    
    run_full_pipeline(
        lr_path=args.lr_path,
        hr_path=args.hr_path,
        config_path=args.config,
        static_path=args.static_path,
        format=args.format,
        checkpoint_dir=args.checkpoint_dir,
        results_dir=args.results_dir,
        skip_preprocessing=args.skip_preprocessing,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation,
    )


if __name__ == "__main__":
    main()
```

---

### `scripts/run_preprocessing.py`

```python
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
```

---

### `scripts/run_training.py`

```python
"""
Script d'exécution pour l'entraînement ST-CDGM avec checkpointing et callbacks.

Ce script lance l'entraînement complet avec support pour:
- Checkpointing automatique
- Early Stopping
- LR Scheduling
- Sauvegarde des modèles
- Logging structuré

Usage:
    # Avec Docker:
    docker-compose exec st-cdgm-training python scripts/run_training.py \
        --config config/training_config.yaml

    # Directement:
    python scripts/run_training.py \
        --config config/training_config.yaml \
        --checkpoint_dir models \
        --save_every 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Add workspace to path for imports
workspace_path = Path(__file__).parent.parent
if str(workspace_path) not in sys.path:
    sys.path.insert(0, str(workspace_path))
if str(workspace_path / "src") not in sys.path:
    sys.path.insert(0, str(workspace_path / "src"))

from ops.train_st_cdgm import main as train_main
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
)
from st_cdgm.training import EarlyStopping

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


def save_checkpoint(
    encoder: IntelligibleVariableEncoder,
    rcn_cell: RCNCell,
    diffusion_decoder: CausalDiffusionDecoder,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    checkpoint_dir: Path,
    is_best: bool = False,
) -> Path:
    """Save model checkpoint with metadata."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_best:
        checkpoint_path = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}_{timestamp}.pt"
    
    checkpoint = {
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "rcn_state_dict": rcn_cell.state_dict(),
        "diffusion_state_dict": diffusion_decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "timestamp": timestamp,
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Save metadata as JSON
    metadata_path = checkpoint_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump({
            "epoch": epoch,
            "metrics": {k: float(v) for k, v in metrics.items()},
            "timestamp": timestamp,
            "is_best": is_best,
        }, f, indent=2)
    
    print(f"✓ Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    encoder: IntelligibleVariableEncoder,
    rcn_cell: RCNCell,
    diffusion_decoder: CausalDiffusionDecoder,
    optimizer: Optional[optim.Optimizer] = None,
) -> tuple[int, dict]:
    """Load model checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    rcn_cell.load_state_dict(checkpoint["rcn_state_dict"])
    diffusion_decoder.load_state_dict(checkpoint["diffusion_state_dict"])
    
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    epoch = checkpoint.get("epoch", 0)
    metrics = checkpoint.get("metrics", {})
    
    print(f"✓ Checkpoint loaded: {checkpoint_path} (epoch {epoch})")
    return epoch, metrics


def run_training_with_checkpoints(
    cfg: DictConfig,
    checkpoint_dir: Path,
    save_every: int = 5,
    max_checkpoints: int = 5,
    resume_from: Optional[Path] = None,
) -> None:
    """Run training with checkpointing and callbacks."""
    
    print("=" * 80)
    print("ST-CDGM Training with Checkpointing")
    print("=" * 80)
    print(f"Checkpoint Dir: {checkpoint_dir}")
    print(f"Save Every: {save_every} epochs")
    print(f"Max Checkpoints: {max_checkpoints}")
    print("=" * 80)
    
    # Import training setup from train_st_cdgm
    # We'll need to refactor train_st_cdgm to expose setup functions
    # For now, we'll call the main function but with checkpointing logic
    
    # This is a simplified version - in production, we'd refactor train_st_cdgm
    # to separate setup from training loop
    
    n_threads = max(1, (os.cpu_count() or 1) // 2)
    torch.set_num_threads(n_threads)
    print(f"[PERF] torch.set_num_threads({n_threads})")

    device = torch.device(cfg.training.device)
    
    # Setup models (same as train_st_cdgm)
    from ops.train_st_cdgm import build_encoder_for_graph, _iterate_batches
    
    pipeline = NetCDFDataPipeline(
        lr_path=cfg.data.lr_path,
        hr_path=cfg.data.hr_path,
        static_path=cfg.data.static_path,
        seq_len=cfg.data.seq_len,
        baseline_strategy=cfg.data.baseline_strategy,
        baseline_factor=cfg.data.baseline_factor,
        normalize=cfg.data.normalize,
    )
    
    from torch.utils.data import DataLoader
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
    num_workers = int(cfg.training.get("num_workers", 0))
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
    
    # Compile models if enabled
    if cfg.training.compile.get("enabled", False):
        if hasattr(torch, 'compile'):
            compile_mode_rcn = cfg.training.compile.get("rcn_mode", "reduce-overhead")
            compile_mode_diffusion = cfg.training.compile.get("diffusion_mode", "max-autotune")
            compile_mode_encoder = cfg.training.compile.get("encoder_mode", "reduce-overhead")
            
            try:
                rcn_cell = torch.compile(rcn_cell, mode=compile_mode_rcn)
                rcn_runner = RCNSequenceRunner(rcn_cell, detach_interval=cfg.rcn.detach_interval)
                print(f"✓ RCN cell compiled with torch.compile (mode: {compile_mode_rcn})")
            except Exception as e:
                print(f"⚠ torch.compile for RCN cell failed: {e}")
            
            try:
                diffusion = torch.compile(diffusion, mode=compile_mode_diffusion)
                print(f"✓ Diffusion decoder compiled with torch.compile (mode: {compile_mode_diffusion})")
            except Exception as e:
                print(f"⚠ torch.compile for diffusion decoder failed: {e}")
            
            try:
                encoder = torch.compile(encoder, mode=compile_mode_encoder, fullgraph=False)
                print(f"✓ Encoder compiled with torch.compile (mode: {compile_mode_encoder})")
            except Exception as e:
                print(f"⚠ torch.compile for encoder failed: {e}")
    
    spatial_projector = SpatialConditioningProjector(
        num_vars=len(encoder.configs),
        hidden_dim=cfg.rcn.hidden_dim,
        conditioning_dim=cfg.diffusion.conditioning_dim,
        lr_shape=tuple(cfg.graph.lr_shape),
        target_shape=tuple(cfg.diffusion.get("spatial_target_shape", [6, 7])),
    ).to(device)

    params = (
        list(encoder.parameters()) + list(rcn_cell.parameters())
        + list(diffusion.parameters()) + list(spatial_projector.parameters())
    )
    optimizer = optim.Adam(params, lr=cfg.training.lr)
    
    # Setup LR scheduler
    scheduler = None
    if cfg.training.lr_scheduler.get("enabled", False):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=cfg.training.lr_scheduler.get("mode", "min"),
            factor=cfg.training.lr_scheduler.get("factor", 0.5),
            patience=cfg.training.lr_scheduler.get("patience", 3),
            min_lr=cfg.training.lr_scheduler.get("min_lr", 1e-7),
        )
        print("✓ LR Scheduler enabled")
    
    # Setup Early Stopping
    early_stopping = None
    if cfg.training.early_stopping.get("enabled", False):
        early_stopping = EarlyStopping(
            patience=cfg.training.early_stopping.get("patience", 7),
            min_delta=cfg.training.early_stopping.get("min_delta", 0.0),
            restore_best=cfg.training.early_stopping.get("restore_best", True),
            verbose=True,
        )
        print("✓ Early Stopping enabled")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    if resume_from is not None:
        start_epoch, metrics = load_checkpoint(
            resume_from, encoder, rcn_cell, diffusion, optimizer
        )
        best_loss = metrics.get("loss", float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop with checkpointing
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(start_epoch, cfg.training.epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch + 1}/{cfg.training.epochs}")
        print(f"{'='*80}")
        
        batch_iter = _iterate_batches(dataloader, builder, device)
        
        # Get training configuration for train_epoch
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
            dag_method=cfg.loss.get("dag_method", "dagma"),
            dagma_s=cfg.loss.get("dagma_s", 1.0),
            lambda_phy=cfg.loss.get("lambda_phy", 0.0),
            use_predicted_output=cfg.training.physical_loss.get("use_predicted_output", False),
            physical_sample_interval=cfg.training.physical_loss.get("physical_sample_interval", 10),
            physical_num_steps=cfg.training.physical_loss.get("physical_num_steps", 15),
            use_amp=cfg.training.get("use_amp", True),
            scheduler=scheduler,
            use_focal_loss=cfg.loss.get("use_focal_loss", False),
            focal_alpha=cfg.loss.get("focal_alpha", 1.0),
            focal_gamma=cfg.loss.get("focal_gamma", 2.0),
            extreme_weight_factor=cfg.loss.get("extreme_weight_factor", 0.0),
            extreme_percentiles=cfg.loss.get("extreme_percentiles", [95.0, 99.0]),
            reconstruction_loss_type=cfg.loss.get("reconstruction_loss_type", "mse"),
            use_spectral_loss=cfg.loss.get("use_spectral_loss", False),
            lambda_spectral=cfg.loss.get("lambda_spectral", 0.0),
            conditioning_dropout_prob=cfg.diffusion.get("conditioning_dropout_prob", 0.0),
            lambda_dag_prior=cfg.loss.get("lambda_dag_prior", 0.0),
            dag_prior=torch.tensor(cfg.loss.dag_prior, dtype=torch.float32) if cfg.loss.get("dag_prior") else None,
            spatial_projector=spatial_projector,
        )

        if cfg.loss.get("log_spectral_metric_each_epoch", False):
            from st_cdgm.training.training_loop import (
                compute_rapsd_metric_from_batch,
                resolve_train_amp_mode,
            )

            amp_m = resolve_train_amp_mode(device, cfg.training.get("use_amp", True))
            try:
                metric_iter = _iterate_batches(dataloader, builder, device)
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
            except Exception as e:
                print(f"[WARN] RAPSD epoch metric failed: {e}")
        
        current_loss = metrics["loss"]
        
        # Update LR scheduler
        if scheduler is not None:
            scheduler.step(current_loss)
        
        # Early stopping check
        if early_stopping is not None:
            # For early stopping, we'd need validation loss
            # For now, use training loss (not ideal, but functional)
            if early_stopping(current_loss, rcn_cell):  # Use rcn_cell as model proxy
                print("Early stopping triggered!")
                break
        
        # Save checkpoint
        is_best = current_loss < best_loss
        if is_best:
            best_loss = current_loss
        
        if (epoch + 1) % save_every == 0 or is_best:
            checkpoint_path = save_checkpoint(
                encoder, rcn_cell, diffusion, optimizer,
                epoch + 1, metrics, checkpoint_dir, is_best=is_best
            )
            
            # Clean up old checkpoints (keep only last N)
            if not is_best:  # Don't delete best model
                checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
                if len(checkpoints) > max_checkpoints:
                    for old_checkpoint in checkpoints[:-max_checkpoints]:
                        old_checkpoint.unlink()
                        old_checkpoint.with_suffix(".json").unlink(missing_ok=True)
                        print(f"  Deleted old checkpoint: {old_checkpoint.name}")
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best loss: {best_loss:.6f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Train ST-CDGM model with checkpointing and callbacks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default="config/training_config.yaml",
        help="Path to Hydra config file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=Path,
        default="models",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument(
        "--max_checkpoints",
        type=int,
        default=5,
        help="Maximum number of checkpoints to keep",
    )
    parser.add_argument(
        "--resume_from",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    args = parser.parse_args()
    
    # Load Hydra config
    if not args.config.exists():
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    # Use Hydra to load config
    # Since we're calling from outside Hydra, we need to use OmegaConf directly
    cfg = OmegaConf.load(args.config)
    
    # Convert to DictConfig for compatibility
    cfg = OmegaConf.structured(cfg) if OmegaConf.is_config(cfg) else cfg
    
    run_training_with_checkpoints(
        cfg,
        checkpoint_dir=args.checkpoint_dir,
        save_every=args.save_every,
        max_checkpoints=args.max_checkpoints,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
```

---

### `scripts/save_model.py`

```python
"""
Utility script to save a model checkpoint with metadata.

Usage:
    python scripts/save_model.py \
        --encoder model_encoder.pt \
        --rcn model_rcn.pt \
        --diffusion model_diffusion.pt \
        --output models/checkpoint.pt \
        --config config/training_config.yaml
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf


def save_model_checkpoint(
    encoder_path: Path,
    rcn_path: Path,
    diffusion_path: Path,
    output_path: Path,
    config_path: Optional[Path] = None,
    metrics: Optional[dict] = None,
) -> None:
    """Save a combined model checkpoint with metadata."""
    
    # Load model states
    encoder_state = torch.load(encoder_path, map_location="cpu")
    rcn_state = torch.load(rcn_path, map_location="cpu")
    diffusion_state = torch.load(diffusion_path, map_location="cpu")
    
    # Load config if provided
    config = None
    if config_path and config_path.exists():
        config = OmegaConf.load(config_path)
        config = OmegaConf.to_container(config, resolve=True)
    
    # Create checkpoint
    checkpoint = {
        "encoder_state_dict": encoder_state,
        "rcn_state_dict": rcn_state,
        "diffusion_state_dict": diffusion_state,
        "config": config,
        "metrics": metrics or {},
        "timestamp": datetime.now().isoformat(),
    }
    
    # Save checkpoint
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    
    # Save metadata as JSON
    metadata_path = output_path.with_suffix(".json")
    metadata = {
        "encoder_path": str(encoder_path),
        "rcn_path": str(rcn_path),
        "diffusion_path": str(diffusion_path),
        "config_path": str(config_path) if config_path else None,
        "metrics": metrics,
        "timestamp": checkpoint["timestamp"],
    }
    
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✓ Checkpoint saved: {output_path}")
    print(f"✓ Metadata saved: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save model checkpoint")
    parser.add_argument("--encoder", type=Path, required=True)
    parser.add_argument("--rcn", type=Path, required=True)
    parser.add_argument("--diffusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None, help="JSON file with metrics")
    
    args = parser.parse_args()
    
    metrics = None
    if args.metrics and args.metrics.exists():
        with open(args.metrics) as f:
            metrics = json.load(f)
    
    save_model_checkpoint(
        args.encoder, args.rcn, args.diffusion,
        args.output, args.config, metrics
    )
```

---

### `scripts/sync_datastore.py`

```python
"""
Script for synchronizing data between local disk and CyVerse Data Store.

This script provides utilities to:
- Copy data from Data Store to local disk (for better I/O performance)
- Save results from local disk to Data Store (for persistence)
- List files in Data Store
- Check disk space

Usage:
    # Copy from Data Store to local
    python scripts/sync_datastore.py --copy-from-datastore \
        ~/data-store/home/username/data/raw/*.nc \
        ~/climate_data/data/raw/
    
    # Save to Data Store from local
    python scripts/sync_datastore.py --save-to-datastore \
        ~/climate_data/models/*.pt \
        ~/data-store/home/username/st-cdgm/models/
    
    # List files in Data Store
    python scripts/sync_datastore.py --list-datastore \
        ~/data-store/home/username/data/
    
    # Dry run (simulate without copying)
    python scripts/sync_datastore.py --copy-from-datastore \
        --dry-run ~/data-store/home/username/data/raw/*.nc ~/climate_data/data/raw/
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# Add project to path
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.vice_utils import (
    get_datastore_path,
    is_vice_environment,
    recommend_local_copy,
)


def format_size(size_bytes: int) -> str:
    """Format file size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def get_disk_usage(path: Path) -> dict:
    """Get disk usage statistics for a path."""
    try:
        stat = shutil.disk_usage(path)
        return {
            "total": stat.total,
            "used": stat.used,
            "free": stat.free,
            "percent_used": (stat.used / stat.total) * 100,
        }
    except OSError:
        return None


def copy_files(
    source: Path | str,
    destination: Path | str,
    dry_run: bool = False,
    verbose: bool = True,
) -> int:
    """
    Copy files from source to destination.
    
    Args:
        source: Source path (file or directory).
        destination: Destination path (file or directory).
        dry_run: If True, simulate without copying.
        verbose: If True, print progress.
    
    Returns:
        Number of files copied.
    """
    source = Path(source).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    
    if not source.exists():
        print(f"Error: Source path does not exist: {source}")
        return 0
    
    # Ensure destination directory exists
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
    else:
        destination.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    
    if source.is_file():
        # Copy single file
        if dry_run:
            if verbose:
                size = source.stat().st_size
                print(f"[DRY RUN] Would copy: {source} -> {destination} ({format_size(size)})")
            copied_count = 1
        else:
            if verbose:
                size = source.stat().st_size
                print(f"Copying: {source.name} ({format_size(size)})...")
            shutil.copy2(source, destination)
            copied_count = 1
            if verbose:
                print(f"  ✓ Copied to: {destination}")
    
    elif source.is_dir():
        # Copy directory tree
        if dry_run:
            if verbose:
                print(f"[DRY RUN] Would copy directory: {source} -> {destination}")
            # Count files recursively
            for file_path in source.rglob("*"):
                if file_path.is_file():
                    copied_count += 1
                    if verbose:
                        size = file_path.stat().st_size
                        rel_path = file_path.relative_to(source)
                        print(f"  Would copy: {rel_path} ({format_size(size)})")
        else:
            if verbose:
                print(f"Copying directory: {source} -> {destination}")
            
            for file_path in source.rglob("*"):
                if file_path.is_file():
                    rel_path = file_path.relative_to(source)
                    dest_path = destination / rel_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if verbose:
                        size = file_path.stat().st_size
                        print(f"  Copying: {rel_path} ({format_size(size)})...")
                    
                    shutil.copy2(file_path, dest_path)
                    copied_count += 1
            
            if verbose:
                print(f"  ✓ Copied {copied_count} files")
    
    return copied_count


def list_datastore(path: Path | str, recursive: bool = False) -> None:
    """List files in Data Store directory."""
    path = Path(path).expanduser().resolve()
    
    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        return
    
    if not path.is_dir():
        print(f"Error: Path is not a directory: {path}")
        return
    
    print(f"Listing files in: {path}")
    print("=" * 80)
    
    if recursive:
        files = list(path.rglob("*"))
    else:
        files = list(path.iterdir())
    
    # Separate files and directories
    dirs = [f for f in files if f.is_dir()]
    files_only = [f for f in files if f.is_file()]
    
    # Print directories first
    if dirs:
        print("\nDirectories:")
        for d in sorted(dirs):
            rel_path = d.relative_to(path)
            print(f"  📁 {rel_path}/")
    
    # Print files
    if files_only:
        print("\nFiles:")
        total_size = 0
        for f in sorted(files_only):
            rel_path = f.relative_to(path)
            size = f.stat().st_size
            total_size += size
            print(f"  📄 {rel_path} ({format_size(size)})")
        
        print(f"\nTotal: {len(files_only)} files, {format_size(total_size)}")


def copy_from_datastore(
    source_pattern: str,
    destination: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Copy files from Data Store to local disk."""
    if not is_vice_environment():
        print("Warning: Not running in VICE environment. Data Store may not be available.")
    
    # Expand user paths
    source = Path(source_pattern).expanduser()
    destination = Path(destination).expanduser()
    
    # Check if source exists
    if not source.exists():
        print(f"Error: Source does not exist: {source}")
        print("\nTip: Check that your Data Store path is correct.")
        print(f"     Example: ~/data-store/home/<username>/data/raw/")
        return
    
    # Check disk space
    dest_usage = get_disk_usage(destination.parent if destination.is_file() else destination)
    if dest_usage and verbose:
        print(f"Destination disk usage: {dest_usage['percent_used']:.1f}% used")
        print(f"  Free space: {format_size(dest_usage['free'])}")
    
    # Recommend local copy
    if verbose and recommend_local_copy():
        print("\nℹ️  Recommendation: Copying to local disk for better I/O performance.")
        print("   Data Store access is slower for large files.\n")
    
    # Copy files
    count = copy_files(source, destination, dry_run=dry_run, verbose=verbose)
    
    if verbose:
        if dry_run:
            print(f"\n[DRY RUN] Would copy {count} file(s)")
        else:
            print(f"\n✓ Successfully copied {count} file(s)")


def save_to_datastore(
    source: str,
    destination_pattern: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> None:
    """Save files from local disk to Data Store."""
    if not is_vice_environment():
        print("Warning: Not running in VICE environment. Data Store may not be available.")
    
    # Expand user paths
    source = Path(source).expanduser()
    destination = Path(destination_pattern).expanduser()
    
    # Check if source exists
    if not source.exists():
        print(f"Error: Source does not exist: {source}")
        return
    
    # Check disk space in Data Store
    dest_usage = get_disk_usage(destination.parent if destination.is_file() else destination)
    if dest_usage and verbose:
        print(f"Data Store disk usage: {dest_usage['percent_used']:.1f}% used")
        print(f"  Free space: {format_size(dest_usage['free'])}")
    
    if verbose:
        print("\nℹ️  Saving to Data Store for persistence.")
        print("   Files in Data Store persist after VICE session ends.\n")
    
    # Copy files
    count = copy_files(source, destination, dry_run=dry_run, verbose=verbose)
    
    if verbose:
        if dry_run:
            print(f"\n[DRY RUN] Would save {count} file(s) to Data Store")
        else:
            print(f"\n✓ Successfully saved {count} file(s) to Data Store")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Synchronize data between local disk and CyVerse Data Store",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy data from Data Store to local
  python scripts/sync_datastore.py --copy-from-datastore \\
      ~/data-store/home/username/data/raw/*.nc \\
      ~/climate_data/data/raw/
  
  # Save results to Data Store
  python scripts/sync_datastore.py --save-to-datastore \\
      ~/climate_data/models/*.pt \\
      ~/data-store/home/username/st-cdgm/models/
  
  # List files in Data Store
  python scripts/sync_datastore.py --list-datastore \\
      ~/data-store/home/username/data/
  
  # Dry run (simulate)
  python scripts/sync_datastore.py --copy-from-datastore --dry-run \\
      ~/data-store/home/username/data/raw/*.nc \\
      ~/climate_data/data/raw/
        """,
    )
    
    parser.add_argument(
        "--copy-from-datastore",
        nargs=2,
        metavar=("SOURCE", "DEST"),
        help="Copy files from Data Store to local disk",
    )
    parser.add_argument(
        "--save-to-datastore",
        nargs=2,
        metavar=("SOURCE", "DEST"),
        help="Save files from local disk to Data Store",
    )
    parser.add_argument(
        "--list-datastore",
        nargs=1,
        metavar="PATH",
        help="List files in Data Store directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Simulate operations without copying files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="List files recursively (with --list-datastore)",
    )
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    # Check if any action is specified
    if not any([args.copy_from_datastore, args.save_to_datastore, args.list_datastore]):
        parser.print_help()
        return 1
    
    # Handle list datastore
    if args.list_datastore:
        list_datastore(args.list_datastore[0], recursive=args.recursive)
        return 0
    
    # Handle copy from datastore
    if args.copy_from_datastore:
        copy_from_datastore(
            args.copy_from_datastore[0],
            args.copy_from_datastore[1],
            dry_run=args.dry_run,
            verbose=verbose,
        )
        return 0
    
    # Handle save to datastore
    if args.save_to_datastore:
        save_to_datastore(
            args.save_to_datastore[0],
            args.save_to_datastore[1],
            dry_run=args.dry_run,
            verbose=verbose,
        )
        return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

### `scripts/test_installation.py`

```python
"""
Test script to verify ST-CDGM installation and dependencies.

This script checks:
- Python version
- PyTorch installation and CUDA availability
- Required dependencies
- Module imports
- GPU availability and basic operations
- VICE environment detection (if in CyVerse)
- Data Store access (if in VICE)

Usage:
    python scripts/test_installation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add workspace to path
workspace_path = Path(__file__).parent.parent
if str(workspace_path / "src") not in sys.path:
    sys.path.insert(0, str(workspace_path / "src"))
if str(workspace_path) not in sys.path:
    sys.path.insert(0, str(workspace_path))

# Import VICE utilities if available
try:
    from scripts.vice_utils import (
        get_datastore_path,
        is_vice_environment,
        recommend_local_copy,
    )
    HAS_VICE_UTILS = True
except ImportError:
    HAS_VICE_UTILS = False


def test_python_version():
    """Test Python version."""
    print("Testing Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"  [FAIL] Python {version.major}.{version.minor} (requires >= 3.8)")
        return False
    print(f"  [OK] Python {version.major}.{version.minor}.{version.micro}")
    return True


def test_pytorch():
    """Test PyTorch installation and CUDA."""
    print("\nTesting PyTorch...")
    try:
        import torch
        print(f"  [OK] PyTorch {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  [OK] CUDA available: {torch.version.cuda}")
            print(f"  [OK] GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    - Device {i}: {torch.cuda.get_device_name(i)}")
            
            # Test basic GPU operation
            x = torch.randn(10, 10).cuda()
            y = torch.matmul(x, x)
            print(f"  [OK] GPU computation test passed")
        else:
            print(f"  [WARN] CUDA not available (CPU mode only)")
        
        return True
    except ImportError as e:
        print(f"  [FAIL] PyTorch not installed: {e}")
        return False


def test_dependencies():
    """Test required dependencies."""
    print("\nTesting dependencies...")
    dependencies = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("xarray", "xarray"),
        ("torch_geometric", "torch_geometric"),
        ("diffusers", "diffusers"),
        ("hydra", "hydra.core"),
        ("omegaconf", "omegaconf"),
        ("zarr", "zarr"),
        ("matplotlib", "matplotlib"),
    ]
    
    all_ok = True
    for name, module in dependencies:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} not installed")
            all_ok = False
    
    return all_ok


def test_st_cdgm_imports():
    """Test ST-CDGM module imports."""
    print("\nTesting ST-CDGM module imports...")
    try:
        from st_cdgm import (
            NetCDFDataPipeline,
            HeteroGraphBuilder,
            IntelligibleVariableEncoder,
            RCNCell,
            RCNSequenceRunner,
            CausalDiffusionDecoder,
        )
        print("  [OK] All core modules imported successfully")
        return True
    except ImportError as e:
        print(f"  [FAIL] Import error: {e}")
        return False


def test_basic_operations():
    """Test basic PyTorch operations."""
    print("\nTesting basic operations...")
    try:
        import torch
        from st_cdgm import RCNCell
        
        # Test RCNCell creation
        rcn = RCNCell(
            num_vars=3,
            hidden_dim=64,
            driver_dim=8,
            reconstruction_dim=8,
        )
        print("  [OK] RCNCell creation successful")
        
        # Test forward pass (if GPU available)
        if torch.cuda.is_available():
            rcn = rcn.cuda()
            H = torch.randn(3, 10, 64).cuda()
            driver = torch.randn(10, 8).cuda()
            H_next, _, _ = rcn(H, driver)
            print(f"  [OK] GPU forward pass successful (output shape: {H_next.shape})")
        else:
            H = torch.randn(3, 10, 64)
            driver = torch.randn(10, 8)
            H_next, _, _ = rcn(H, driver)
            print(f"  [OK] CPU forward pass successful (output shape: {H_next.shape})")
        
        return True
    except Exception as e:
        print(f"  [FAIL] Operation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vice_environment():
    """Test VICE environment detection and Data Store access."""
    print("\nTesting VICE environment...")
    
    if not HAS_VICE_UTILS:
        print("  [SKIP] VICE utilities not available (scripts/vice_utils.py not found)")
        return True  # Not a failure, just not available
    
    try:
        # Check if running in VICE
        is_vice = is_vice_environment()
        
        if is_vice:
            print("  [INFO] Running in CyVerse VICE environment")
            
            # Check Data Store access
            datastore_path = get_datastore_path()
            if datastore_path:
                print(f"  [OK] Data Store accessible: {datastore_path}")
                
                # Check if Data Store is writable
                try:
                    test_file = datastore_path / ".test_write"
                    test_file.touch()
                    test_file.unlink()
                    print("  [OK] Data Store is writable")
                except (OSError, PermissionError):
                    print("  [WARN] Data Store may not be writable (read-only access)")
            else:
                print("  [WARN] Data Store path not found")
            
            # Recommendation for local copy
            if recommend_local_copy():
                print("  [INFO] Recommendation: Copy data to local disk (~/) for better I/O performance")
                print("         Use: python scripts/sync_datastore.py --copy-from-datastore ...")
            
        else:
            print("  [INFO] Not running in VICE environment (local execution)")
        
        return True  # Always pass, this is informational
        
    except Exception as e:
        print(f"  [WARN] VICE detection failed: {e}")
        return True  # Not a critical failure


def main():
    """Run all tests."""
    print("=" * 80)
    print("ST-CDGM Installation Test")
    print("=" * 80)
    
    tests = [
        ("Python Version", test_python_version),
        ("PyTorch", test_pytorch),
        ("Dependencies", test_dependencies),
        ("ST-CDGM Imports", test_st_cdgm_imports),
        ("Basic Operations", test_basic_operations),
        ("VICE Environment", test_vice_environment),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  [FAIL] {name} test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    all_passed = True
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 80)
    if all_passed:
        print("[SUCCESS] All tests passed!")
        return 0
    else:
        print("[FAIL] Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

### `scripts/test_pipeline.py`

```python
"""
End-to-end pipeline test with synthetic data.

This script creates synthetic data and tests the complete pipeline:
- Preprocessing
- Training (short run)
- Evaluation

Usage:
    python scripts/test_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import xarray as xr

# Add workspace to path
workspace_path = Path(__file__).parent.parent
if str(workspace_path / "src") not in sys.path:
    sys.path.insert(0, str(workspace_path / "src"))


def create_synthetic_data(
    output_dir: Path,
    num_timesteps: int = 100,
    lr_shape: tuple[int, int] = (23, 26),
    hr_shape: tuple[int, int] = (172, 179),
) -> tuple[Path, Path]:
    """Create synthetic NetCDF files for testing."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create time coordinates
    time = np.arange(num_timesteps)
    
    # Create LR data
    lr_lat = np.linspace(-59, -26, lr_shape[0])
    lr_lon = np.linspace(150, 188, lr_shape[1])
    lr_data = np.random.randn(num_timesteps, 8, lr_shape[0], lr_shape[1])
    
    lr_ds = xr.Dataset(
        {
            "temperature": (["time", "level", "lat", "lon"], lr_data),
        },
        coords={
            "time": time,
            "level": np.arange(8),
            "lat": lr_lat,
            "lon": lr_lon,
        },
    )
    
    lr_path = output_dir / "synthetic_lr.nc"
    lr_ds.to_netcdf(lr_path)
    print(f"✓ Created synthetic LR data: {lr_path}")
    
    # Create HR data
    hr_lat = np.linspace(-59, -26, hr_shape[0])
    hr_lon = np.linspace(150, 188, hr_shape[1])
    hr_data = np.random.randn(num_timesteps, 3, hr_shape[0], hr_shape[1])
    
    hr_ds = xr.Dataset(
        {
            "temperature": (["time", "channel", "lat", "lon"], hr_data),
        },
        coords={
            "time": time,
            "channel": np.arange(3),
            "lat": hr_lat,
            "lon": hr_lon,
        },
    )
    
    hr_path = output_dir / "synthetic_hr.nc"
    hr_ds.to_netcdf(hr_path)
    print(f"✓ Created synthetic HR data: {hr_path}")
    
    return lr_path, hr_path


def test_pipeline():
    """Run end-to-end pipeline test."""
    
    print("=" * 80)
    print("ST-CDGM Pipeline Test (Synthetic Data)")
    print("=" * 80)
    
    # Create test data
    test_data_dir = Path("data/test")
    print("\n1. Creating synthetic test data...")
    lr_path, hr_path = create_synthetic_data(test_data_dir, num_timesteps=50)
    
    # Test preprocessing
    print("\n2. Testing preprocessing...")
    try:
        from scripts.run_preprocessing import main as preprocess_main
        # This would need to be adapted to call the function directly
        print("  ⏭ Skipping preprocessing test (requires refactoring)")
    except Exception as e:
        print(f"  ⚠ Preprocessing test skipped: {e}")
    
    # Test training (minimal)
    print("\n3. Testing model creation...")
    try:
        import torch
        from st_cdgm import RCNCell, IntelligibleVariableEncoder, CausalDiffusionDecoder
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create minimal models
        encoder = IntelligibleVariableEncoder(
            configs=[],
            hidden_dim=64,
            conditioning_dim=64,
        ).to(device)
        
        rcn = RCNCell(
            num_vars=3,
            hidden_dim=64,
            driver_dim=8,
        ).to(device)
        
        diffusion = CausalDiffusionDecoder(
            in_channels=3,
            conditioning_dim=64,
            height=32,
            width=32,
            num_diffusion_steps=100,
        ).to(device)
        
        print("  ✓ Models created successfully")
        
        # Test forward pass
        H = torch.randn(3, 10, 64).to(device)
        driver = torch.randn(10, 8).to(device)
        H_next, _, _ = rcn(H, driver)
        print(f"  ✓ Forward pass successful (RCN output: {H_next.shape})")
        
    except Exception as e:
        print(f"  ❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ Pipeline test completed successfully!")
    print("=" * 80)
    print(f"Test data: {test_data_dir}")
    print("\nNote: Full training/evaluation tests require real data and take longer.")
    
    return True


if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
```

---

### `scripts/validate_setup.py`

```python
"""
Script de validation complète pour vérifier que tout est prêt avant déploiement.

Ce script vérifie:
- Syntaxe Python de tous les fichiers
- Structure des fichiers et répertoires
- Configuration YAML valide
- Imports (sans exécuter le code nécessitant torch)
- Présence de tous les fichiers nécessaires
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Tuple

# Couleurs pour Windows (compatible)
GREEN = "[OK]"
RED = "[FAIL]"
YELLOW = "[WARN]"


def check_syntax(file_path: Path) -> Tuple[bool, str]:
    """Vérifie la syntaxe Python d'un fichier."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ast.parse(f.read(), filename=str(file_path))
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def check_imports_safe(file_path: Path) -> Tuple[bool, str]:
    """Vérifie les imports sans exécuter le code."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=str(file_path))
        
        # Vérifier les imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Vérifier que les imports locaux existent
                    if not alias.name.startswith('.'):
                        continue
                    # Imports relatifs - on vérifie juste la syntaxe
                    pass
            elif isinstance(node, ast.ImportFrom):
                if node.module and not node.module.startswith('.'):
                    # Import externe - OK
                    pass
        
        return True, ""
    except Exception as e:
        return False, str(e)


def check_yaml_config(file_path: Path) -> Tuple[bool, str]:
    """Vérifie qu'un fichier YAML est valide."""
    try:
        import yaml
        with open(file_path, 'r', encoding='utf-8') as f:
            yaml.safe_load(f)
        return True, ""
    except ImportError:
        return False, "yaml module not available"
    except Exception as e:
        return False, str(e)


def validate_project_structure() -> List[Tuple[str, bool, str]]:
    """Valide la structure complète du projet."""
    results = []
    
    # Fichiers essentiels
    essential_files = [
        "docker-compose.yml",
        "Dockerfile",
        ".dockerignore",
        "setup.py",
        "requirements.txt",
        "config/docker.env",
        "config/training_config.yaml",
    ]
    
    print("\n" + "=" * 80)
    print("Validation de la Structure du Projet")
    print("=" * 80)
    
    for file_path in essential_files:
        path = Path(file_path)
        exists = path.exists()
        results.append((f"Fichier: {file_path}", exists, "" if exists else "Fichier manquant"))
        status = GREEN if exists else RED
        print(f"  {status} {file_path}")
    
    # Répertoires essentiels
    essential_dirs = [
        "src/st_cdgm",
        "src/st_cdgm/models",
        "src/st_cdgm/data",
        "src/st_cdgm/training",
        "src/st_cdgm/evaluation",
        "scripts",
        "ops",
        "config",
        "docs",
    ]
    
    print("\nRépertoires:")
    for dir_path in essential_dirs:
        path = Path(dir_path)
        exists = path.exists() and path.is_dir()
        results.append((f"Répertoire: {dir_path}", exists, "" if exists else "Répertoire manquant"))
        status = GREEN if exists else RED
        print(f"  {status} {dir_path}")
    
    return results


def validate_python_files() -> List[Tuple[str, bool, str]]:
    """Valide tous les fichiers Python."""
    results = []
    
    print("\n" + "=" * 80)
    print("Validation des Fichiers Python")
    print("=" * 80)
    
    # Fichiers à vérifier
    python_files = []
    
    # Scripts
    scripts_dir = Path("scripts")
    if scripts_dir.exists():
        python_files.extend(scripts_dir.glob("*.py"))
    
    # Ops
    ops_dir = Path("ops")
    if ops_dir.exists():
        python_files.extend(ops_dir.glob("*.py"))
    
    # Source
    src_dir = Path("src/st_cdgm")
    if src_dir.exists():
        python_files.extend(src_dir.rglob("*.py"))
    
    for py_file in python_files:
        # Ignorer __pycache__
        if "__pycache__" in str(py_file):
            continue
        
        # Vérifier syntaxe
        syntax_ok, syntax_msg = check_syntax(py_file)
        results.append((f"Syntaxe: {py_file}", syntax_ok, syntax_msg))
        
        status = GREEN if syntax_ok else RED
        if syntax_ok:
            print(f"  {status} {py_file.name}")
        else:
            print(f"  {status} {py_file.name}: {syntax_msg}")
    
    return results


def validate_configs() -> List[Tuple[str, bool, str]]:
    """Valide les fichiers de configuration."""
    results = []
    
    print("\n" + "=" * 80)
    print("Validation des Configurations")
    print("=" * 80)
    
    config_files = [
        "config/training_config.yaml",
    ]
    
    for config_file in config_files:
        path = Path(config_file)
        if not path.exists():
            results.append((f"Config: {config_file}", False, "Fichier manquant"))
            print(f"  {RED} {config_file}: Fichier manquant")
            continue
        
        yaml_ok, yaml_msg = check_yaml_config(path)
        results.append((f"Config: {config_file}", yaml_ok, yaml_msg))
        status = GREEN if yaml_ok else RED
        print(f"  {status} {config_file}")
        if not yaml_ok:
            print(f"      Erreur: {yaml_msg}")
    
    return results


def validate_docker_files() -> List[Tuple[str, bool, str]]:
    """Valide les fichiers Docker."""
    results = []
    
    print("\n" + "=" * 80)
    print("Validation des Fichiers Docker")
    print("=" * 80)
    
    docker_files = [
        "docker-compose.yml",
        "Dockerfile",
        ".dockerignore",
    ]
    
    for docker_file in docker_files:
        path = Path(docker_file)
        exists = path.exists()
        results.append((f"Docker: {docker_file}", exists, "" if exists else "Fichier manquant"))
        status = GREEN if exists else RED
        print(f"  {status} {docker_file}")
        
        if exists and docker_file.endswith('.yml'):
            # Vérifier que docker-compose.yml est valide (basique)
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    # Vérifications basiques
                    if 'services:' in content or 'version:' in content:
                        print(f"      Structure docker-compose valide")
            except Exception as e:
                results.append((f"Docker: {docker_file} (structure)", False, str(e)))
                print(f"      {RED} Erreur structure: {e}")
    
    return results


def validate_imports_structure() -> List[Tuple[str, bool, str]]:
    """Vérifie la structure des imports sans exécuter."""
    results = []
    
    print("\n" + "=" * 80)
    print("Validation de la Structure des Imports")
    print("=" * 80)
    
    # Vérifier que les modules __init__.py existent
    init_files = [
        "src/st_cdgm/__init__.py",
        "src/st_cdgm/models/__init__.py",
        "src/st_cdgm/data/__init__.py",
        "src/st_cdgm/training/__init__.py",
        "src/st_cdgm/evaluation/__init__.py",
    ]
    
    for init_file in init_files:
        path = Path(init_file)
        exists = path.exists()
        results.append((f"__init__: {init_file}", exists, "" if exists else "Fichier manquant"))
        status = GREEN if exists else RED
        print(f"  {status} {init_file}")
    
    return results


def main():
    """Exécute toutes les validations."""
    print("=" * 80)
    print("Validation Complète du Projet ST-CDGM")
    print("=" * 80)
    
    all_results = []
    
    # Structure du projet
    all_results.extend(validate_project_structure())
    
    # Fichiers Python
    all_results.extend(validate_python_files())
    
    # Configurations
    all_results.extend(validate_configs())
    
    # Docker
    all_results.extend(validate_docker_files())
    
    # Imports
    all_results.extend(validate_imports_structure())
    
    # Résumé
    print("\n" + "=" * 80)
    print("Résumé de la Validation")
    print("=" * 80)
    
    passed = sum(1 for _, ok, _ in all_results if ok)
    total = len(all_results)
    failed = total - passed
    
    for name, ok, msg in all_results:
        if not ok:
            status = RED
            print(f"  {status} {name}")
            if msg:
                print(f"      {msg}")
    
    print(f"\nTotal: {total} | Réussis: {passed} | Échoués: {failed}")
    
    if failed == 0:
        print(f"\n{GREEN} Toutes les validations ont réussi!")
        return 0
    else:
        print(f"\n{RED} {failed} validation(s) ont échoué. Veuillez corriger les erreurs ci-dessus.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

---

### `scripts/validate_antismoothing.py`

```python
#!/usr/bin/env python3
"""
Validation rapide des garde-fous anti-lissage (prompt v6, Phase 4).
Exécute les tests unitaires dédiés sans lancer un entraînement complet.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    tests = ROOT / "tests" / "test_corrections_antilissage.py"
    cmd = [sys.executable, "-m", "pytest", str(tests), "-q", "--tb=short"]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
```

---

### `scripts/vice_utils.py`

```python
"""
Utility functions for CyVerse Discovery Environment (VICE) support.

This module provides functions to detect if code is running in a VICE environment
and manage paths to the CyVerse Data Store.

Usage:
    from scripts.vice_utils import is_vice_environment, get_datastore_path, resolve_data_path
    
    if is_vice_environment():
        datastore_path = get_datastore_path()
        data_path = resolve_data_path("data/raw/lr.nc")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def is_vice_environment() -> bool:
    """
    Detect if code is running in a CyVerse VICE environment.
    
    VICE environments typically have:
    - A ~/data-store/ directory mounted via CSI Driver
    - Environment variables indicating VICE/DE
    - Jupyter Lab or other VICE apps running
    
    Returns:
        True if running in VICE, False otherwise.
    """
    # Check for data-store directory (primary indicator)
    home = Path.home()
    data_store_dir = home / "data-store"
    
    if data_store_dir.exists() and data_store_dir.is_dir():
        # Additional check: verify structure
        home_subdir = data_store_dir / "home"
        if home_subdir.exists():
            return True
    
    # Check for VICE-related environment variables
    vice_env_vars = [
        "VICE_APP_NAME",
        "CYVERSE_USERNAME",
        "DE_APP_NAME",
        "VICE_USER",
    ]
    
    if any(os.environ.get(var) for var in vice_env_vars):
        return True
    
    # Check for Jupyter Lab in typical VICE location
    jupyter_config = home / ".jupyter" / "jupyter_lab_config.py"
    if jupyter_config.exists():
        # Additional check: look for VICE-specific paths in env
        python_path = os.environ.get("PYTHONPATH", "")
        if "vice" in python_path.lower() or "cyverse" in python_path.lower():
            return True
    
    return False


def get_datastore_path(username: Optional[str] = None) -> Optional[Path]:
    """
    Get the path to the CyVerse Data Store for the current user.
    
    Args:
        username: Optional username. If not provided, attempts to detect from
                  environment variables or uses 'USER' or 'USERNAME'.
    
    Returns:
        Path to ~/data-store/home/<username>/ if in VICE, None otherwise.
    """
    if not is_vice_environment():
        return None
    
    # Get username
    if username is None:
        # Try environment variables first
        username = (
            os.environ.get("CYVERSE_USERNAME") or
            os.environ.get("VICE_USER") or
            os.environ.get("USER") or
            os.environ.get("USERNAME") or
            "default"
        )
    
    home = Path.home()
    datastore_path = home / "data-store" / "home" / username
    
    return datastore_path if datastore_path.exists() else None


def resolve_data_path(
    relative_path: str | Path,
    prefer_local: bool = True,
    username: Optional[str] = None,
) -> Path:
    """
    Resolve a relative data path according to the environment.
    
    In VICE:
    - If prefer_local=True: Resolves to ~/climate_data/<relative_path> (fast local disk)
    - If prefer_local=False: Resolves to ~/data-store/home/<username>/<relative_path> (persistent but slow)
    
    Outside VICE:
    - Always resolves relative to current working directory or project root.
    
    Args:
        relative_path: Relative path to data file/directory.
        prefer_local: If True (default), prefer local disk in VICE for performance.
                     If False, use Data Store (slower but persistent).
        username: Optional username for Data Store. Auto-detected if not provided.
    
    Returns:
        Resolved absolute Path.
    """
    relative_path = Path(relative_path)
    
    # If already absolute, return as-is
    if relative_path.is_absolute():
        return relative_path
    
    if is_vice_environment() and prefer_local:
        # In VICE, prefer local disk (~/) for better I/O performance
        home = Path.home()
        # Try to detect project root (look for climate_data or similar)
        project_root = home / "climate_data"
        if not project_root.exists():
            # Fallback to current directory
            project_root = Path.cwd()
        
        resolved = project_root / relative_path
        return resolved
    
    elif is_vice_environment() and not prefer_local:
        # In VICE, use Data Store (persistent but slower)
        datastore_path = get_datastore_path(username)
        if datastore_path is None:
            # Fallback to local if Data Store not available
            home = Path.home()
            project_root = home / "climate_data"
            if not project_root.exists():
                project_root = Path.cwd()
            return project_root / relative_path
        
        # Resolve relative to Data Store
        resolved = datastore_path / relative_path
        return resolved
    
    else:
        # Outside VICE, resolve relative to current working directory
        # Try to find project root (look for src/ or config/ directories)
        current = Path.cwd()
        
        # Check if we're in project root or subdirectory
        if (current / "src").exists() or (current / "config").exists():
            return current / relative_path
        
        # Walk up to find project root
        for parent in current.parents:
            if (parent / "src").exists() or (parent / "config").exists():
                return parent / relative_path
        
        # Fallback to current directory
        return current / relative_path


def recommend_local_copy(file_size_mb: Optional[float] = None) -> bool:
    """
    Recommend whether to copy data locally in VICE based on file size.
    
    In VICE, files accessed via CSI Driver (~/data-store/) are slow for large files.
    This function recommends copying to local disk (~/) if beneficial.
    
    Args:
        file_size_mb: Size of file in MB. If None, always recommends True in VICE.
    
    Returns:
        True if local copy is recommended, False otherwise.
    """
    if not is_vice_environment():
        return False  # Not applicable outside VICE
    
    # For large files (>100MB), local copy is strongly recommended
    if file_size_mb is not None and file_size_mb > 100:
        return True
    
    # For smaller files or unknown size, still recommend local copy
    # as it's generally faster and doesn't hurt
    return True


def get_project_root() -> Path:
    """
    Get the project root directory.
    
    Returns:
        Path to project root (directory containing src/ or config/).
    """
    current = Path.cwd()
    
    # Check if we're in project root
    if (current / "src").exists() or (current / "config").exists():
        return current
    
    # Walk up to find project root
    for parent in current.parents:
        if (parent / "src").exists() or (parent / "config").exists():
            return parent
    
    # Fallback to current directory
    return current


def ensure_directory(path: Path, create_if_missing: bool = True) -> Path:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        path: Path to directory.
        create_if_missing: If True, create directory if it doesn't exist.
    
    Returns:
        Path object (same as input).
    
    Raises:
        OSError: If directory creation fails.
    """
    if create_if_missing and not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path
```

---

### `tests/__init__.py`

```python
"""
Tests unitaires pour ST-CDGM.
"""
```

---

### `tests/test_installation.py`

```python
#!/usr/bin/env python
"""
Script de vérification de l'installation ST-CDGM.
Teste que toutes les dépendances critiques sont installées correctement.
"""

import sys
from typing import List, Tuple

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """
    Vérifie si un package est installé et retourne sa version.
    
    Args:
        package_name: Nom du package à afficher
        import_name: Nom pour l'import (si différent)
    
    Returns:
        (success, version_string)
    """
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError as e:
        return False, str(e)

def main():
    print("=" * 80)
    print("🔍 ST-CDGM Installation Verification")
    print("=" * 80)
    print()
    
    # Python version
    print(f"🐍 Python: {sys.version}")
    print(f"   Executable: {sys.executable}")
    print()
    
    # Liste des packages à vérifier
    packages = [
        ("numpy", None),
        ("pandas", None),
        ("xarray", None),
        ("torch", None),
        ("torchvision", None),
        ("torch_geometric", None),
        ("torch_scatter", None),
        ("torch_sparse", None),
        ("diffusers", None),
        ("accelerate", None),
        ("netCDF4", "netCDF4"),
        ("h5netcdf", None),
        ("xbatcher", None),
        ("dask", None),
        ("matplotlib", None),
        ("seaborn", None),
        ("hydra", None),
        ("omegaconf", None),
        ("pytest", None),
    ]
    
    results = []
    max_name_len = max(len(p[0]) for p in packages)
    
    print("📦 Checking Packages:")
    print("-" * 80)
    
    for package_name, import_name in packages:
        success, info = check_package(package_name, import_name)
        results.append((package_name, success, info))
        
        status = "✅" if success else "❌"
        padding = " " * (max_name_len - len(package_name))
        
        if success:
            print(f"{status} {package_name}{padding} : {info}")
        else:
            print(f"{status} {package_name}{padding} : NOT INSTALLED")
    
    print()
    
    # PyTorch specific checks
    print("🔥 PyTorch Configuration:")
    print("-" * 80)
    try:
        import torch
        print(f"   PyTorch Version: {torch.__version__}")
        print(f"   CUDA Available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   CUDA Version: {torch.version.cuda}")
            print(f"   cuDNN Version: {torch.backends.cudnn.version()}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"           Memory: {props.total_memory / 1024**3:.1f} GB")
        else:
            print("   ⚠️  Running on CPU only")
    except Exception as e:
        print(f"   ❌ Error checking PyTorch: {e}")
    
    print()
    
    # Summary
    print("=" * 80)
    successful = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    if successful == total:
        print(f"✅ SUCCESS: All {total} packages installed correctly!")
        print()
        print("🚀 You're ready to use ST-CDGM!")
        print("   Next steps:")
        print("   1. Open the notebook: jupyter notebook st_cdgm_training_evaluation.ipynb")
        print("   2. Or run the training script: python ops/train_st_cdgm.py")
        return 0
    else:
        print(f"⚠️  WARNING: {successful}/{total} packages installed")
        print()
        print("Missing packages:")
        for name, success, info in results:
            if not success:
                print(f"   ❌ {name}")
        print()
        print("📝 Installation instructions:")
        print("   pip install -r requirements.txt")
        print()
        print("   Or see INSTALLATION.md for detailed instructions")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

---

### `tests/test_corrections_antilissage.py`

```python
"""
Tests de non-régression — corrections anti-lissage / prompt v6 (Phases 4 et 6.2).
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


def _rapsd_loss_minimal(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Copie minimale pour test gradient (évite import diffusers)."""
    B, C, H, W = pred.shape
    losses = []
    for b in range(B):
        for c in range(C):
            p = pred[b, c]
            t = target[b, c]
            fp = torch.fft.fftshift(torch.fft.fft2(p))
            ft = torch.fft.fftshift(torch.fft.fft2(t))
            psd_p = torch.abs(fp) ** 2
            psd_t = torch.abs(ft) ** 2
            cy, cx = H // 2, W // 2
            y_idx = torch.arange(H, device=pred.device, dtype=torch.float32) - cy
            x_idx = torch.arange(W, device=pred.device, dtype=torch.float32) - cx
            yy, xx = torch.meshgrid(y_idx, x_idx, indexing="ij")
            r = torch.sqrt(xx ** 2 + yy ** 2).long().clamp(min=0)
            max_r = int(r.max().item()) + 1
            rapsd_p = torch.zeros(max_r, device=pred.device)
            rapsd_t = torch.zeros(max_r, device=pred.device)
            counts = torch.zeros(max_r, device=pred.device)
            rf = r.flatten()
            rapsd_p.scatter_add_(0, rf, psd_p.flatten())
            rapsd_t.scatter_add_(0, rf, psd_t.flatten())
            counts.scatter_add_(0, rf, torch.ones_like(rf, dtype=torch.float32))
            valid = counts > 0
            log_ratio = torch.log(rapsd_p[valid] + 1e-8) - torch.log(rapsd_t[valid] + 1e-8)
            losses.append((log_ratio ** 2).mean())
    return torch.stack(losses).mean()


def test_bicubic_preserves_peak_vs_bilinear():
    field = torch.zeros(1, 1, 50, 50)
    field[0, 0, 25, 25] = 100.0
    field[0, 0, 24, 25] = 80.0
    field[0, 0, 26, 25] = 80.0
    target_size = (172, 179)
    upsampled_bilinear = F.interpolate(field, size=target_size, mode="bilinear", align_corners=False)
    upsampled_bicubic = F.interpolate(field, size=target_size, mode="bicubic", align_corners=False).clamp(
        min=0.0
    )
    assert upsampled_bicubic.max().item() >= upsampled_bilinear.max().item() - 1e-3


def test_ensemble_mean_reduces_variance_vs_single_member():
    N_members = 10
    H, W = 100, 100
    base = torch.ones(H, W) * 0.5
    members = []
    rng = np.random.default_rng(42)
    for _ in range(N_members):
        member = base.clone()
        px = int(rng.integers(10, H - 10))
        py = int(rng.integers(10, W - 10))
        member[px, py] = 5.0
        members.append(member)
    stack = torch.stack(members, dim=0)
    ensemble_mean = stack.mean(dim=0)
    single_member = stack[0]
    assert single_member.var().item() > ensemble_mean.var().item() * 2.0


def test_bicubic_can_overshoot_negative_then_clamp():
    field = torch.zeros(1, 1, 20, 20)
    field[0, 0, 10, 10] = 50.0
    upsampled = F.interpolate(field, size=(172, 179), mode="bicubic", align_corners=False)
    clamped = upsampled.clamp(min=0.0)
    assert (clamped >= 0).all()


def test_rapsd_loss_is_differentiable():
    pred = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randn(2, 1, 32, 32)
    loss = _rapsd_loss_minimal(pred, target)
    loss.backward()
    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()


def _replace_conv_transpose_with_resize_minimal(module: nn.Module) -> None:
    """Même logique que `diffusion_decoder._replace_conv_transpose_with_resize` (sans import diffusers)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose2d):
            st = child.stride[0]
            if st >= 2 and child.stride[0] == child.stride[1]:
                repl = nn.Sequential(
                    nn.Upsample(scale_factor=float(st), mode="nearest"),
                    nn.Conv2d(
                        child.in_channels,
                        child.out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=child.bias is not None,
                    ),
                )
                if child.bias is not None:
                    nn.init.zeros_(repl[1].bias)
                nn.init.kaiming_normal_(repl[1].weight, nonlinearity="relu")
                setattr(module, name, repl)
            else:
                _replace_conv_transpose_with_resize_minimal(child)
        else:
            _replace_conv_transpose_with_resize_minimal(child)


def test_resizeconv_replacement_preserves_shape_and_removes_transpose():
    """Anti-checkerboard: Upsample+Conv remplace ConvTranspose2d avec même taille spatiale de sortie."""

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.up = nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.up(x)

    m = Tiny()
    x = torch.randn(1, 8, 16, 16)
    with torch.no_grad():
        y_ct = m(x)
    assert any(isinstance(c, nn.ConvTranspose2d) for c in m.modules())

    _replace_conv_transpose_with_resize_minimal(m)
    assert not any(isinstance(c, nn.ConvTranspose2d) for c in m.modules())
    assert isinstance(m.up, nn.Sequential)
    assert isinstance(m.up[0], nn.Upsample) and isinstance(m.up[1], nn.Conv2d)

    with torch.no_grad():
        y_rc = m(x)
    assert y_ct.shape == y_rc.shape
```

---

### `tests/test_st_cdgm_smoke.py`

```python
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
            projection_class_embeddings_input_dim=64,
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
            projection_class_embeddings_input_dim=64,
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
```

---

### `train_ddp.py`

```python
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
        include_mid_layer=CONFIG.graph.get("include_mid_layer", True),
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

    spatial_projector = SpatialConditioningProjector(
        num_vars=len(encoder_configs),
        hidden_dim=CONFIG.rcn.hidden_dim,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        lr_shape=tuple(CONFIG.graph.lr_shape),
        target_shape=tuple(CONFIG.diffusion.get("spatial_target_shape", [6, 7])),
    ).to(device)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(rcn_cell.parameters())
        + list(diffusion.parameters()) + list(spatial_projector.parameters()),
        lr=CONFIG.training.lr,
    )

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
            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": enc.state_dict(),
                "rcn_cell_state_dict": rcn.state_dict(),
                "diffusion_state_dict": diff.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "config": OmegaConf.to_container(CONFIG, resolve=True),
            }, ckpt)
            print(f"Checkpoint saved: {ckpt}")

    if world_size > 1:
        cleanup_ddp()


if __name__ == "__main__":
    main()
```

---

## Notebooks

### `st_cdgm_training_evaluation.ipynb`

**Contenu non embarqué** (JSON volumineux). Le notebook compte **50** cellules (code: 31, markdown: 19).

Ouvrir le fichier `.ipynb` dans le dépôt pour le code et les sorties complets.

---
### `st_cdgm_validation_inference.ipynb`

**Contenu non embarqué** (JSON volumineux). Le notebook compte **8** cellules (code: 7, markdown: 1).

Ouvrir le fichier `.ipynb` dans le dépôt pour le code et les sorties complets.

---
### `resume_training_from_checkpoint.ipynb`

**Contenu non embarqué** (JSON volumineux). Le notebook compte **8** cellules (code: 7, markdown: 1).

Ouvrir le fichier `.ipynb` dans le dépôt pour le code et les sorties complets.

---
### `st_cdgm_publication_figures.ipynb`

**Contenu non embarqué** (JSON volumineux). Le notebook compte **8** cellules (code: 5, markdown: 3).

Ouvrir le fichier `.ipynb` dans le dépôt pour le code et les sorties complets.

---
### `st_cdgm_results_presentation.ipynb`

**Contenu non embarqué** (JSON volumineux). Le notebook compte **31** cellules (code: 18, markdown: 13).

Ouvrir le fichier `.ipynb` dans le dépôt pour le code et les sorties complets.

---

---

## Hierarchie du Code

### Architecture modulaire

```
ST-CDGM
|
+-- Data Layer (src/st_cdgm/data/)
|   +-- pipeline.py          -> NetCDFDataPipeline, ZarrDataPipeline
|   +-- netcdf_utils.py      -> utilitaires NetCDF / metadonnees
|
+-- Graph Layer (src/st_cdgm/models/graph_builder.py)
|   +-- HeteroGraphBuilder   -> Construction graphe heterogene
|
+-- Encoding Layer (src/st_cdgm/models/intelligible_encoder.py)
|   +-- IntelligibleVariableEncoder -> Variables latentes H(0)
|
+-- Causal Layer (src/st_cdgm/models/causal_rcn.py)
|   +-- RCNCell              -> Cellule recurrente causale
|   +-- RCNSequenceRunner    -> Deroulement sequentiel
|
+-- Generation Layer (src/st_cdgm/models/diffusion_decoder.py)
|   +-- CausalDiffusionDecoder -> Generation HR par diffusion
|
+-- Training Layer (src/st_cdgm/training/)
|   +-- training_loop.py     -> train_epoch, pertes
|   +-- callbacks.py
|   +-- multi_gpu.py         -> entrainement multi-GPU (DDP)
|
+-- Utils (src/st_cdgm/utils/)
|   +-- checkpoint.py        -> chargement / fusion de checkpoints
|
+-- Evaluation Layer (src/st_cdgm/evaluation/)
    +-- evaluation_xai.py    -> Metriques, inference autoregressive, DAG
```

### Flux de donnees

```
NetCDF / Zarr
    |
NetCDFDataPipeline
    +-- Alignement temporel
    +-- Normalisation
    +-- Baseline computation
    +-- Residual calculation
    |
IterableDataset / batches
    |
HeteroGraphBuilder
    +-- Construction graphe statique
    |
IntelligibleVariableEncoder
    +-- H(0) initial state
    |
RCNSequenceRunner
    +-- RCNCell (sequence)
    +-- H(t) causal states
    |
CausalDiffusionDecoder
    +-- Conditioning from H(t)
    +-- HR generation (diffusion)
    |
Loss Computation
    +-- L_gen (diffusion)
    +-- L_rec (reconstruction)
    +-- L_dag (DAG constraint)
    |
Optimization
    +-- Backpropagation
```

### Dependances entre modules (schema)

```
setup.py
    |
src/st_cdgm/__init__.py
    +-- models/
    |   +-- causal_rcn.py
    |   +-- diffusion_decoder.py
    |   +-- graph_builder.py
    |   +-- intelligible_encoder.py
    +-- data/
    |   +-- pipeline.py
    |   +-- netcdf_utils.py
    +-- training/
    |   +-- training_loop.py
    |   +-- callbacks.py
    |   +-- multi_gpu.py
    +-- evaluation/
    |   +-- evaluation_xai.py
    +-- utils/
        +-- checkpoint.py
```

---

## Resume des fichiers

### Statistiques (approximatif, genere)

- **Fichiers Python (estimation)** : ~40 fichiers, ~12485 lignes (src + ops + scripts + tests + racine).
- **Configuration** : `config/*.yaml`, `docker.env`, `requirements.txt`, `environment.yml`.
- **Documentation** : `docs/*.md`, `README.md`, `stats.md`.
- **Notebooks** : entrainement/evaluation, validation/inference, figures publication.

### Lignes de code (modules cles)

- `src/st_cdgm/data/pipeline.py`: ~1328 lignes
- `src/st_cdgm/data/netcdf_utils.py`: ~1086 lignes
- `src/st_cdgm/models/causal_rcn.py`: ~397 lignes
- `src/st_cdgm/models/diffusion_decoder.py`: ~721 lignes
- `src/st_cdgm/models/graph_builder.py`: ~480 lignes
- `src/st_cdgm/models/intelligible_encoder.py`: ~338 lignes
- `src/st_cdgm/training/training_loop.py`: ~1111 lignes
- `src/st_cdgm/evaluation/evaluation_xai.py`: ~1054 lignes

---

## Points cles de l'architecture

1. **Modularite** : composants independants et reutilisables.
2. **Extensibilite** : nouveaux modules (p.ex. planificateurs de diffusion).
3. **Configuration** : Hydra / YAML.
4. **Performance** : optimisations documentees dans `docs/OPTIMISATION.md`.
5. **Robustesse** : tests, validation, checkpoints (`utils/checkpoint.py`).

---

**Fin de la documentation complete du projet ST-CDGM** *(genere)*
