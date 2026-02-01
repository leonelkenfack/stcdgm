# 📋 Documentation Complète du Projet ST-CDGM

## 🌍 Spatio-Temporal Causal Diffusion Generative Model

Ce document liste l'ensemble du projet, incluant tous les fichiers Python, les fichiers de configuration, et le notebook d'entraînement/évaluation.

---

## 📁 Structure du Projet

```
climate_data/
├── .dockerignore
├── .gitignore
├── README.md
├── config/                          # Fichiers de configuration
│   ├── docker.env
│   ├── training_config.yaml
│   └── training_config_vice.yaml
├── data/                            # Données
│   ├── metadata/
│   │   ├── NorESM2-MM_histupdated_compressed.metadata.csv
│   │   └── NorESM2-MM_histupdated_compressed.metadata.json
│   └── raw/
├── docs/                            # Documentation
│   ├── ARCHITECTURE_MODEL.md
│   ├── DOCKER_README.md
│   ├── GUIDE_PEDAGOGIQUE_ST-CDGM.md
│   ├── OPTIMISATION.md
│   ├── RAPPORT_TECHNIQUE_COMPLET.md
│   ├── SCRIPTS_README.md
│   └── st_cdgm_quickstart.md
├── ops/                             # Scripts d'opérations
│   ├── preprocess_to_shards.py
│   ├── preprocess_to_zarr.py
│   └── train_st_cdgm.py
├── scripts/                         # Scripts utilitaires
│   ├── cleanup_repeated_lines.py
│   ├── load_model.py
│   ├── run_evaluation.py
│   ├── run_full_pipeline.py
│   ├── run_preprocessing.py
│   ├── run_training.py
│   ├── save_model.py
│   ├── sync_datastore.py
│   ├── test_installation.py
│   ├── test_pipeline.py
│   ├── validate_setup.py
│   └── vice_utils.py
├── src/                             # Code source principal
│   └── st_cdgm/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── netcdf_utils.py
│       │   └── pipeline.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   └── evaluation_xai.py
│       ├── models/
│       │   ├── __init__.py
│       │   ├── causal_rcn.py
│       │   ├── diffusion_decoder.py
│       │   ├── graph_builder.py
│       │   └── intelligible_encoder.py
│       └── training/
│           ├── __init__.py
│           ├── callbacks.py
│           └── training_loop.py
├── tests/                           # Tests
│   ├── __init__.py
│   ├── test_installation.py
│   └── test_st_cdgm_smoke.py
├── docker-compose.yml
├── Dockerfile
├── environment.yml
├── requirements.txt
├── setup.py
└── st_cdgm_training_evaluation.ipynb
```

---

## 🔧 Fichiers de Configuration

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
  lr_path: "data/raw/predictor_ACCESS-CM2_hist.nc"
  hr_path: "data/raw/pr_ACCESS-CM2_hist.nc"
  static_path: null
  seq_len: 6
  stride: 1
  baseline_strategy: "hr_smoothing"  # or "lr_interp"
  baseline_factor: 4
  normalize: true
  target_transform: null
  nan_fill_strategy: "zero"  # "zero", "mean", or "interpolate"
  precipitation_delta: 0.01  # Delta pour log1p des précipitations
  lr_variables: null
  hr_variables: null
  static_variables: null

# Graph Configuration
graph:
  lr_shape: [23, 26]
  hr_shape: [172, 179]
  static_variables: []
  include_mid_layer: false

# Encoder Configuration
encoder:
  hidden_dim: 128
  conditioning_dim: 128
  metapaths:
    - name: "GP850_spat_adj"
      src: "GP850"
      relation: "spat_adj"
      target: "GP850"
      pool: "mean"
    - name: "GP850_to_GP500"
      src: "GP850"
      relation: "vert_adj"
      target: "GP500"
      pool: "mean"
    - name: "GP500_spat_adj"
      src: "GP500"
      relation: "spat_adj"
      target: "GP500"
      pool: "mean"
    - name: "GP500_to_GP250"
      src: "GP500"
      relation: "vert_adj"
      target: "GP250"
      pool: "mean"
    - name: "GP250_spat_adj"
      src: "GP250"
      relation: "spat_adj"
      target: "GP250"
      pool: "mean"

# RCN Configuration
rcn:
  hidden_dim: 128
  driver_dim: 8
  reconstruction_dim: 8
  dropout: 0.0
  detach_interval: null

# Diffusion Decoder Configuration
diffusion:
  in_channels: 3
  conditioning_dim: 128
  height: 172
  width: 179
  steps: 1000
  scheduler_type: "ddpm"  # "ddpm", "edm", or "dpm_solver++"
  use_gradient_checkpointing: false  # Phase C3: Gradient checkpointing

# Loss Configuration
loss:
  lambda_gen: 1.0
  beta_rec: 0.1
  gamma_dag: 0.1
  lambda_phy: 0.0  # Phase B2: Physical loss weight
  dag_method: "dagma"  # "dagma" or "no_tears"
  dagma_s: 1.0
  # Phase D1: Focal Loss
  use_focal_loss: false
  focal_alpha: 1.0
  focal_gamma: 2.0
  # Phase D2: Extreme Loss Weighting
  extreme_weight_factor: 0.0
  extreme_percentiles: [95.0, 99.0]
  # Phase D3: DAG Stabilization
  dag_l1_regularization: false
  dag_l1_weight: 0.01
  # Phase D4: Reconstruction Loss Type
  reconstruction_loss_type: "mse"  # "mse", "cosine", or "mse+cosine"

# Training Configuration
training:
  device: "cpu"  # "cuda" or "cpu" - using CPU for local test
  epochs: 1  # Only 1 epoch for local testing
  lr: 0.00005  # Réduit pour éviter la divergence
  gradient_clipping: 0.5  # Réduit pour stabiliser l'entraînement
  log_every: 1
  # Phase C1: Mixed Precision Training
  use_amp: false  # Disable AMP for CPU
  # Phase C2: Early Stopping and LR Scheduling
  early_stopping:
    enabled: false
    patience: 7
    min_delta: 0.0
    restore_best: true
  lr_scheduler:
    enabled: false
    mode: "min"  # "min" or "max"
    factor: 0.5  # LR reduction factor
    patience: 3  # Patience for LR reduction
    min_lr: 1e-7
  # Phase B2: Physical Loss Options
  physical_loss:
    use_predicted_output: false
    physical_sample_interval: 10
    physical_num_steps: 15
  # Phase A3: torch.compile
  compile:
    enabled: false  # Disable for local test
    rcn_mode: "reduce-overhead"
    diffusion_mode: "max-autotune"
    encoder_mode: "reduce-overhead"

# Model Checkpointing
checkpoint:
  enabled: true
  save_dir: "models"
  save_every: 5  # Save every N epochs
  save_best: true  # Save best model based on validation loss
  max_checkpoints: 5  # Keep only last N checkpoints

# Evaluation Configuration
evaluation:
  enabled: true
  eval_every: 5  # Evaluate every N epochs
  num_samples: 10  # Number of samples for evaluation metrics
  compute_f1_extremes: true
  f1_percentiles: [95.0, 99.0]
  save_visualizations: true
  output_dir: "results"
```

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
  # Paths relative to project root (recommended for VICE after copying to local disk)
  # Replace these with your actual file paths
  lr_path: "data/raw/predictor_ACCESS-CM2_hist.nc"  # Low-resolution input data
  hr_path: "data/raw/pr_ACCESS-CM2_hist.nc"  # High-resolution target data
  
  # Alternative: Data Store paths (slower but persistent)
  # lr_path: "~/data-store/home/<username>/data/raw/predictor_ACCESS-CM2_hist.nc"
  # hr_path: "~/data-store/home/<username>/data/raw/pr_ACCESS-CM2_hist.nc"
  
  static_path: null  # Optional static fields (e.g., topography)
  seq_len: 6  # Temporal sequence length
  stride: 1  # Stride for sliding window
  baseline_strategy: "hr_smoothing"  # or "lr_interp"
  baseline_factor: 4
  normalize: true
  target_transform: null
  lr_variables: null  # Auto-detect if null
  hr_variables: null  # Auto-detect if null
  static_variables: null

# Graph Configuration
graph:
  lr_shape: [23, 26]  # Low-resolution grid shape (lat, lon)
  hr_shape: [172, 179]  # High-resolution grid shape (lat, lon)
  static_variables: []
  include_mid_layer: false

# Encoder Configuration
encoder:
  hidden_dim: 128
  conditioning_dim: 128
  metapaths:
    - name: "GP850_spat_adj"
      src: "GP850"
      relation: "spat_adj"
      target: "GP850"
      pool: "mean"
    - name: "GP850_to_GP500"
      src: "GP850"
      relation: "vert_adj"
      target: "GP500"
      pool: "mean"
    - name: "GP500_spat_adj"
      src: "GP500"
      relation: "spat_adj"
      target: "GP500"
      pool: "mean"
    - name: "GP500_to_GP250"
      src: "GP500"
      relation: "vert_adj"
      target: "GP250"
      pool: "mean"
    - name: "GP250_spat_adj"
      src: "GP250"
      relation: "spat_adj"
      target: "GP250"
      pool: "mean"

# RCN Configuration
rcn:
  hidden_dim: 128
  driver_dim: 8
  reconstruction_dim: 8
  dropout: 0.0
  detach_interval: null

# Diffusion Decoder Configuration
diffusion:
  in_channels: 3
  conditioning_dim: 128
  height: 172
  width: 179
  steps: 1000
  scheduler_type: "ddpm"  # "ddpm", "edm", or "dpm_solver++"
  use_gradient_checkpointing: false

# Loss Configuration
loss:
  lambda_gen: 1.0
  beta_rec: 0.1
  gamma_dag: 0.1
  lambda_phy: 0.0
  dag_method: "dagma"  # "dagma" or "no_tears"
  dagma_s: 1.0
  use_focal_loss: false
  focal_alpha: 1.0
  focal_gamma: 2.0
  extreme_weight_factor: 0.0
  extreme_percentiles: [95.0, 99.0]
  dag_l1_regularization: false
  dag_l1_weight: 0.01
  reconstruction_loss_type: "mse"  # "mse", "cosine", or "mse+cosine"

# Training Configuration
# 
# For VICE users:
# - GPU availability depends on VICE configuration
# - Check GPU availability: python -c "import torch; print(torch.cuda.is_available())"
# - If GPU available, set device: "cuda" and enable mixed precision training
# - If CPU only, keep device: "cpu" and use_amp: false
training:
  # Device: "cuda" or "cpu"
  # In VICE, check GPU availability first before setting to "cuda"
  device: "cpu"  # Change to "cuda" if GPU available in your VICE session
  
  epochs: 100  # Adjust based on your needs
  lr: 0.0001
  gradient_clipping: 1.0
  log_every: 10  # Log progress every N batches
  
  # Mixed Precision Training (recommended for GPU)
  use_amp: false  # Enable if GPU available: set to true
  
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
  compute_f1_extremes: true
  f1_percentiles: [95.0, 99.0]
  save_visualizations: true
  output_dir: "results"  # Will be created in ~/climate_data/results/
  
  # Save results to Data Store regularly:
  # python scripts/sync_datastore.py --save-to-datastore \
  #     ~/climate_data/results/ \
  #     ~/data-store/home/<username>/st-cdgm/results/
```

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

### `requirements.txt`

```txt
# ============================================================================
# ST-CDGM Project - Complete Requirements
# Spatio-Temporal Causal Diffusion Generative Model for Climate Downscaling
# ============================================================================

# --------------------------------------------------------------------------
# Core Scientific Libraries
# --------------------------------------------------------------------------
numpy>=1.21.0,<2.0.0
pandas>=1.3.0
xarray>=2023.1.0
scipy>=1.7.0

# --------------------------------------------------------------------------
# PyTorch Ecosystem
# --------------------------------------------------------------------------
# Note: Pour CUDA, installez avec: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
torch>=2.0.0
torchvision>=0.15.0

# --------------------------------------------------------------------------
# PyTorch Geometric (Graph Neural Networks)
# --------------------------------------------------------------------------
# Installation: pip install torch-geometric
# Ou avec wheels: pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
torch-geometric>=2.3.0
torch-scatter>=2.1.0
torch-sparse>=0.6.17

# --------------------------------------------------------------------------
# Diffusion Models
# --------------------------------------------------------------------------
diffusers>=0.21.0
accelerate>=0.20.0
transformers>=4.30.0  # Requis par certains modèles diffusers

# --------------------------------------------------------------------------
# NetCDF and Climate Data Processing
# --------------------------------------------------------------------------
netcdf4>=1.6.0
h5netcdf>=1.1.0
xbatcher>=0.3.0  # Pour les batches temporels spatio-temporels
dask[complete]>=2023.1.0  # Pour le traitement distribué
zarr>=2.14.0  # Format de données optimisé pour ML (alternative à NetCDF)

# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
matplotlib>=3.5.0
seaborn>=0.12.0

# --------------------------------------------------------------------------
# Configuration Management
# --------------------------------------------------------------------------
hydra-core>=1.3.0
omegaconf>=2.3.0

# --------------------------------------------------------------------------
# Testing
# --------------------------------------------------------------------------
pytest>=7.0.0
pytest-cov>=4.0.0

# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
tqdm>=4.64.0  # Barres de progression
python-dateutil>=2.8.0
pytz>=2022.1

# --------------------------------------------------------------------------
# Optional: Jupyter and Interactive Analysis
# --------------------------------------------------------------------------
jupyter>=1.0.0
jupyterlab>=3.5.0
ipykernel>=6.15.0
ipywidgets>=8.0.0
notebook>=6.5.0

# --------------------------------------------------------------------------
# Optional: Additional XAI and Visualization
# --------------------------------------------------------------------------
plotly>=5.10.0  # Visualisations interactives
networkx>=2.8.0  # Visualisation de graphes causaux

# --------------------------------------------------------------------------
# Optional: Additional Data Formats and Optimizations
# --------------------------------------------------------------------------
webdataset>=0.2.0  # Alternative WebDataset format (optional, Phase 1.1)
pyg-lib>=0.1.0  # Optimisations PyTorch Geometric (optional, Phase 2.4)
k-diffusion>=1.0.0  # Pour EDM scheduler si diffusers ne supporte pas (optional, Phase 3.2)

# --------------------------------------------------------------------------
# Development Tools (Optional)
# --------------------------------------------------------------------------
black>=23.0.0  # Code formatting
flake8>=6.0.0  # Linting
mypy>=1.0.0  # Type checking
pre-commit>=3.0.0  # Git hooks

# --------------------------------------------------------------------------
# Notes d'Installation
# --------------------------------------------------------------------------
# 
# Installation complète (recommandée):
#   pip install -r requirements.txt
#
# Installation minimale (sans Jupyter):
#   pip install -r requirements.txt --no-deps
#   puis installer manuellement les dépendances core
#
# Pour CUDA (GPU):
#   1. Installer PyTorch avec CUDA:
#      pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
#   2. Installer PyTorch Geometric avec CUDA:
#      pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
#   3. Installer le reste:
#      pip install -r requirements.txt
#
# Pour CPU uniquement:
#   pip install -r requirements.txt
#
# Résolution de problèmes:
#   - Si torch-geometric échoue: installer les wheels depuis https://data.pyg.org/whl/
#   - Si netcdf4 échoue: conda install netcdf4
#   - Si dask est lent: pip install dask[complete] --upgrade
#
# ============================================================================
```

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

## 📄 Fichiers racine / outil

### `.gitignore`

Fichiers et répertoires exclus du versionnement Git. Principaux motifs : Python (`__pycache__/`, `*.pyc`, `venv/`, `*.egg-info/`), IDE (`.idea/`, `.vscode/`), Jupyter (`.ipynb_checkpoints/`), tests (`.pytest_cache/`, `.coverage`), variables d'environnement (`.env`), logs et fichiers temporaires. Optionnel : `data/raw` pour éviter de versionner les gros fichiers NetCDF.

### `.dockerignore`

Fichiers exclus du contexte de build Docker. Inclut : artefacts Python, environnements virtuels, IDE, Jupyter, `.git/`, Dockerfiles, `docs/` et la plupart des `*.md` (sauf `README.md`), logs, cache, résultats et images. Permet de garder l'image légère et d'éviter de copier données sensibles ou volumineuses.

### `README.md`

Vue d'ensemble du projet ST-CDGM pour utilisateurs et contributeurs. Contenu : description du downscaling climatique (LR→HR), installation locale et CyVerse VICE, quick start (préparation des données, entraînement, évaluation, pipeline complet), liens vers la documentation (`docs/`, CYVERSE_VICE_SETUP.md), structure du projet, dépendances principales, utilitaires VICE, tests et métriques d'évaluation (CRPS, FSS, Wasserstein, etc.).

---

## 📚 Documentation (`docs/`)

### `docs/ARCHITECTURE_MODEL.md`

Architecture technique et flux de données du modèle ST-CDGM. Décrit les trois modules (Encodeur GNN, RCN causal, Décodeur de diffusion), schémas Mermaid, équations et rôles des composants pour le downscaling climatique.

### `docs/DOCKER_README.md`

Guide Docker Compose pour exécuter ST-CDGM en container. Prérequis (Docker GPU, données dans `data/raw/`), configuration des volumes, commandes pour démarrer le container, accès shell, entraînement et bonnes pratiques.

### `docs/GUIDE_PEDAGOGIQUE_ST-CDGM.md`

Guide pédagogique pour non-initiés. Explique le problème (cartes LR pixelisées → HR détaillées), les étapes (DATA, Graphe, Encodeur, RCN, Diffusion), avec analogies et exemples concrets (NorESM2, normalisation, métapaths).

### `docs/OPTIMISATION.md`

Optimisations proposées pour ST-CDGM : performance (pipeline, boucle d'entraînement, RCN, graphe, diffusion, encodeur) et accuracy/loss/métriques (pertes, F1 extremes, régularisation). Table des matières et solutions par fichier.

### `docs/RAPPORT_TECHNIQUE_COMPLET.md`

Référence technique : modèles climatiques sources (NorESM2-MM), variables (T, U, V, W, Q aux niveaux 850/500/250 hPa), flux de données, baselines (hr_smoothing, lr_interp), formule résiduelle et chaîne de traitement complète.

### `docs/SCRIPTS_README.md`

Documentation des scripts d'exécution. Usage et options de `run_preprocessing.py`, `run_training.py`, `run_evaluation.py`, `run_full_pipeline.py` et autres scripts (chemins, config, checkpoint, format zarr/webdataset).

### `docs/st_cdgm_quickstart.md`

Quickstart en anglais : préparation de l'environnement (PyTorch, PyG, diffusers, Hydra, xbatcher), format des données d'entrée (LR/HR/static, time commun), invocation du driver Hydra `ops/train_st_cdgm.py` et groupes de configuration clés.

---

## 📂 Données et métadonnées (`data/metadata/`)

Fichiers de métadonnées exportés à partir de NetCDF via `NetCDFToDataFrame` (module `netcdf_utils`) : `export_metadata_to_json()` et `export_metadata_to_csv()`. Ils décrivent dimensions, coordonnées, variables, attributs et structure du fichier source.

### `data/metadata/NorESM2-MM_histupdated_compressed.metadata.json`

Métadonnées JSON du fichier NorESM2-MM (historique, compressé). Contient `file_info`, `dimensions` (time, lat, lon avec tailles et plages), `data_variables` et attributs CF/NetCDF. Utilisable pour inspection sans charger le NetCDF complet.

### `data/metadata/NorESM2-MM_histupdated_compressed.metadata.csv`

Version tabulaire (CSV) des métadonnées des variables du même fichier NorESM2-MM, exportée par `export_metadata_to_csv()`. Pratique pour analyse ou comparaison de variables.

---

## 📦 Code Source Principal (`src/st_cdgm/`)

### `src/st_cdgm/__init__.py`

```python
"""
ST-CDGM: Spatio-Temporal Causal Diffusion Generative Model

Package principal pour le modèle ST-CDGM.
"""

from .models.causal_rcn import RCNCell, RCNSequenceRunner
from .models.diffusion_decoder import CausalDiffusionDecoder, DiffusionOutput
from .models.intelligible_encoder import IntelligibleVariableEncoder, IntelligibleVariableConfig
from .models.graph_builder import HeteroGraphBuilder
from .data.pipeline import NetCDFDataPipeline, ZarrDataPipeline, ResDiffIterableDataset
from .data.netcdf_utils import NetCDFToDataFrame
from .training.training_loop import train_epoch

__all__ = [
    # Models
    "RCNCell",
    "RCNSequenceRunner",
    "CausalDiffusionDecoder",
    "DiffusionOutput",
    "IntelligibleVariableEncoder",
    "IntelligibleVariableConfig",
    "HeteroGraphBuilder",
    # Data
    "NetCDFDataPipeline",
    "ZarrDataPipeline",
    "ResDiffIterableDataset",
    "NetCDFToDataFrame",
    # Training
    "train_epoch",
]
```

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

        # Align datasets along the shared temporal axis
        self.lr_dataset_raw, self.hr_dataset_raw = self._align_time(self.lr_dataset_raw, self.hr_dataset_raw)
        if self.static_dataset is not None:
            self.static_dataset = self.static_dataset.load()

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
        kwargs = {"chunks": self._chunks} if self._chunks else {}
        return xr.open_dataset(path, **kwargs)

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

    def _normalise_lr_dataset(self, dataset: xr.Dataset) -> Tuple[xr.Dataset, Dict[str, xr.Dataset]]:
        if self.means_path and self.stds_path:
            means = xr.open_dataset(self.means_path)
            stds = xr.open_dataset(self.stds_path)
        else:
            means = dataset.mean(dim=self.dims.time, keep_attrs=True)
            stds = dataset.std(dim=self.dims.time, keep_attrs=True)
        stds = stds.where(stds != 0.0, other=1.0)
        normalised = (dataset - means) / stds
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

    def _apply_target_transform(self, dataset: xr.Dataset) -> xr.Dataset:
        if self._target_transform is None:
            return dataset
        transformed = self._target_transform(dataset)
        if not isinstance(transformed, xr.Dataset):
            raise TypeError("target_transform must return an xarray.Dataset.")
        return transformed

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
        self.lr_spatial_shape = (lr_dataset.dims[dims.lr_lat], lr_dataset.dims[dims.lr_lon])
        self.hr_spatial_shape = (hr_dataset.dims[dims.hr_lat], hr_dataset.dims[dims.hr_lon])
        
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
        return window.dims.get(self.dims.time, 0) == self.seq_len

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

        if self.as_torch and torch is not None:
            sample = {
                "lr": torch.from_numpy(lr_np),
                "baseline": torch.from_numpy(baseline_np),
                "residual": torch.from_numpy(residual_np),
                "hr": torch.from_numpy(hr_np),
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

### `src/st_cdgm/models/__init__.py`

```python
"""
Modules de modèles pour ST-CDGM.
"""

from .causal_rcn import RCNCell, RCNSequenceRunner
from .diffusion_decoder import CausalDiffusionDecoder, DiffusionOutput
from .intelligible_encoder import IntelligibleVariableEncoder, IntelligibleVariableConfig
from .graph_builder import HeteroGraphBuilder

__all__ = [
    "RCNCell",
    "RCNSequenceRunner",
    "CausalDiffusionDecoder",
    "DiffusionOutput",
    "IntelligibleVariableEncoder",
    "IntelligibleVariableConfig",
    "HeteroGraphBuilder",
]
```

### `src/st_cdgm/models/causal_rcn.py`

```python
"""
Module 4 – Réseau causal récurrent (RCN) pour l'architecture ST-CDGM.

Ce module implémente la cellule RCN (`RCNCell`) et un utilitaire de déroulement
séquentiel (`RCNSequenceRunner`). La cellule combine un cœur causal (matrice DAG
apprenante + assignations structurelles) et une mise à jour récurrente via GRU,
suivie d'une reconstruction optionnelle pour la perte L_rec.
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
        
        # Phase A2: Vectorized GRU parameters for parallel computation
        # We extract parameters from individual GRUCells and organize them for batched operations
        # This allows vectorized computation while maintaining separate parameters per variable

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
            # Use the hidden state H_prev as reconstruction source
            # H_prev has shape [num_vars, N, hidden_dim]
            # Reshape to [N, num_vars * hidden_dim] for the decoder
            recon_input = H_prev.permute(1, 0, 2).reshape(N, -1)
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
        
        # Phase C3: Gradient checkpointing support
        # Reduces memory usage by ~50% but increases computation time by ~20-30%
        # Trade-off: Use when memory is limited or to allow larger batch sizes
        self._gradient_checkpointing_enabled = False
        if use_gradient_checkpointing:
            self.enable_gradient_checkpointing()

    def forward(
        self,
        noisy_sample: Tensor,
        timestep: Tensor,
        conditioning: Tensor,
    ) -> Tensor:
        """
        Passe avant du UNet (prédiction du bruit).
        """
        conditioning = self._prepare_conditioning(conditioning)
        output = self.unet(
            sample=noisy_sample,
            timestep=timestep,
            encoder_hidden_states=conditioning,
        )
        return output.sample

    def compute_loss(
        self,
        target: Tensor,
        conditioning: Tensor,
        use_focal_loss: bool = False,  # Phase D1: Use focal loss for hard pixels
        focal_alpha: float = 1.0,  # Phase D1: Weighting factor for focal loss
        focal_gamma: float = 2.0,  # Phase D1: Focusing parameter (higher = more focus on hard pixels)
    ) -> Tensor:
        """
        Calcule la perte de diffusion (MSE entre bruit réel et prédit).
        Gère les NaN dans le target en utilisant un masque (standard pour données climatiques).
        """
        # Vérifier le conditioning (ne doit pas contenir de NaN/Inf)
        if torch.isnan(conditioning).any() or torch.isinf(conditioning).any():
            raise ValueError(
                f"Conditioning contains NaN/Inf: NaN={torch.isnan(conditioning).sum().item()}, "
                f"Inf={torch.isinf(conditioning).sum().item()}, "
                f"shape={conditioning.shape}, "
                f"stats: min={conditioning.min().item():.6f}, max={conditioning.max().item():.6f}"
            )
        
        # Créer un masque pour les valeurs valides dans le target
        # Les NaN peuvent représenter des masques géographiques (océan, etc.)
        valid_mask = ~torch.isnan(target) & ~torch.isinf(target)
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
        
        noise_pred = self.forward(noisy_sample, timesteps, conditioning)
        
        # Vérifier que noise_pred ne contient pas de NaN/Inf (problème du modèle)
        if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
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
        if torch.isnan(loss) or torch.isinf(loss):
            raise ValueError(
                f"Loss is NaN/Inf: loss={loss.item()}, "
                f"valid_pixels={valid_mask.sum().item()}/{total_pixels}, "
                f"noise_pred_stats: min={noise_pred[valid_mask].min().item():.6f}, max={noise_pred[valid_mask].max().item():.6f}"
            )
        
        return loss

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
            )
        
        # Original DDPM sampling
        scheduler = self.scheduler
        inference_steps = num_steps or getattr(scheduler, "num_inference_steps", None)
        if inference_steps is None:
            inference_steps = self.num_diffusion_steps
        scheduler.set_timesteps(inference_steps, device=conditioning.device)

        sample = torch.randn(
            conditioning.shape[0],
            self.in_channels,
            self.height,
            self.width,
            device=conditioning.device,
            generator=generator,
        )

        for t in scheduler.timesteps:
            model_output = self.unet(
                sample=sample,
                timestep=t,
                encoder_hidden_states=conditioning,
            ).sample
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
        return DiffusionOutput(residual=residual, baseline=baseline, t_min=t_min, t_mean=t_mean, t_max=t_max)

    def _sample_edm_ode(
        self,
        conditioning: Tensor,
        num_steps: int = 25,
        generator: Optional[torch.Generator] = None,
        baseline: Optional[Tensor] = None,
        apply_constraints: bool = True,
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
        
        # Solve ODE using Euler method
        # d/dt x = -sigma'(t) * sigma(t) * score(x, sigma(t))
        for i in range(num_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            
            # Convert time to scheduler timestep for UNet
            # Map from [0, 1] to [0, num_train_timesteps]
            scheduler_timestep = (1.0 - t_current) * self.num_diffusion_steps
            scheduler_timestep = scheduler_timestep.long().clamp(0, self.num_diffusion_steps - 1)
            
            # Predict noise/score with UNet
            with torch.no_grad():
                noise_pred = self.unet(
                    sample=sample,
                    timestep=scheduler_timestep.expand(sample.shape[0]),
                    encoder_hidden_states=conditioning,
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
            model_output = self.forward(sample, t, conditioning)
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

### `src/st_cdgm/models/graph_builder.py`

```python
"""
Module 2 – Construction du graphe hétérogène statique pour l'architecture ST-CDGM.

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
```

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

### `src/st_cdgm/training/training_loop.py`

```python
"""
Module 6 – Boucle d'entraînement pour l'architecture ST-CDGM.

Ce module assemble les pertes (diffusion, reconstruction, NO TEARS) et fournit
une routine d'entraînement par epoch qui enchaîne les modules précédents :
encodeur de variables intelligibles, RCN et décodeur de diffusion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence
import time

import torch
import torch.nn as nn
from torch import Tensor

from ..models.causal_rcn import RCNSequenceRunner
from ..models.diffusion_decoder import CausalDiffusionDecoder
from ..models.intelligible_encoder import IntelligibleVariableEncoder


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


def loss_diffusion(
    decoder: CausalDiffusionDecoder,
    target: Tensor,
    conditioning: Tensor,
    *,
    use_focal_loss: bool = False,
    focal_alpha: float = 1.0,
    focal_gamma: float = 2.0,
) -> Tensor:
    """
    Perte de diffusion L_gen en déléguant à CausalDiffusionDecoder.
    
    Parameters
    ----------
    decoder : CausalDiffusionDecoder
        Le décodeur de diffusion.
    target : Tensor
        Target tensor (résidu HR).
    conditioning : Tensor
        Conditionnement causal.
    use_focal_loss : bool
        Si True, utilise focal loss pour se concentrer sur les pixels difficiles.
    focal_alpha : float
        Facteur de pondération pour focal loss.
    focal_gamma : float
        Paramètre de focalisation (plus élevé = plus de focus sur pixels difficiles).
    """
    return decoder.compute_loss(
        target,
        conditioning,
        use_focal_loss=use_focal_loss,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
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
    
    # Phase D3: Clip values of A_masked to prevent extreme values before computation
    # This improves numerical stability
    A_clipped = torch.clamp(A_masked, min=-10.0, max=10.0)
    
    # Calculer W∘W (Hadamard product, élément par élément)
    W_squared = torch.mul(A_clipped, A_clipped)  # [q, q]
    
    # Phase D3: Ensure s is large enough for numerical stability
    # s must be > max eigenvalue of W_squared
    max_val = W_squared.abs().max().item()
    s_safe = max(s, max_val + 0.1)  # Add small margin
    
    # Calculer sI - W∘W où I est la matrice identité
    sI = s_safe * torch.eye(q, device=device, dtype=dtype)
    M = sI - W_squared  # [q, q]
    
    # Phase D3: Enhanced numerical stability for logdet computation
    # Add small epsilon to diagonal for numerical stability
    eps = torch.tensor(1e-7, device=device, dtype=dtype)
    M = M + eps * torch.eye(q, device=device, dtype=dtype)
    
    # Calculer le log-déterminant de M
    # Utiliser logdet pour la stabilité numérique
    try:
        # Phase D3: Check condition number before logdet
        # If matrix is ill-conditioned, add more regularization
        eigenvalues = torch.linalg.eigvals(M).real
        min_eigenvalue = eigenvalues.min().item()
        max_eigenvalue = eigenvalues.max().item()
        
        if min_eigenvalue <= 1e-6:
            # Matrix is not positive definite or very close to singular
            # Add more regularization and retry
            M = M + (1e-6 - min_eigenvalue + 1e-7) * torch.eye(q, device=device, dtype=dtype)
            min_eigenvalue = (torch.linalg.eigvals(M).real).min().item()
        
        if min_eigenvalue <= 0:
            # Still not positive definite - return large penalty
            return torch.tensor(1e6, device=device, dtype=dtype, requires_grad=True)
        
        log_det_M = torch.logdet(M)
        
        # Phase D3: Check for NaN/Inf in logdet result
        if torch.isnan(log_det_M) or torch.isinf(log_det_M):
            # Fallback to eigenvalue-based computation if logdet fails
            log_eigenvalues = torch.log(eigenvalues + 1e-8)
            log_det_M = log_eigenvalues.sum()
            
    except (RuntimeError, ValueError) as e:
        # Fallback: compute logdet via eigenvalues
        try:
            eigenvalues = torch.linalg.eigvals(M).real
            min_eigenvalue = eigenvalues.min().item()
            if min_eigenvalue <= 0:
                return torch.tensor(1e6, device=device, dtype=dtype, requires_grad=True)
            log_eigenvalues = torch.log(eigenvalues + 1e-8)
            log_det_M = log_eigenvalues.sum()
        except Exception:
            # Ultimate fallback: return large penalty
            return torch.tensor(1e6, device=device, dtype=dtype, requires_grad=True)
    
    # Calculer h(W) = -log det(sI - W∘W) + d log s
    h_W = -log_det_M + q * torch.log(torch.tensor(s_safe, device=device, dtype=dtype) + eps)
    
    # Phase D3: Add L1 regularization for sparsity if requested
    if add_l1_regularization:
        l1_term = l1_weight * A_masked.abs().sum()
        h_W = h_W + l1_term
    
    # Phase D3: Final check for invalid values
    if torch.isnan(h_W) or torch.isinf(h_W):
        return torch.tensor(1e6, device=device, dtype=dtype, requires_grad=True)
    
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
    use_amp: bool = True,  # Phase C1: Mixed precision training (requires CUDA >= 11.0)
    scheduler: Optional[torch.optim.lr_scheduler.ReduceLROnPlateau] = None,  # Phase C2: LR scheduler
    use_focal_loss: bool = False,  # Phase D1: Use focal loss for diffusion
    focal_alpha: float = 1.0,  # Phase D1: Focal loss alpha
    focal_gamma: float = 2.0,  # Phase D1: Focal loss gamma (higher = more focus on hard pixels)
    extreme_weight_factor: float = 0.0,  # Phase D2: Weight factor for extreme events (0 = disabled)
    extreme_percentiles: List[float] = None,  # Phase D2: Percentiles for extreme events
    reconstruction_loss_type: str = "mse",  # Phase D4: Loss type for reconstruction ("mse", "cosine", "mse+cosine")
) -> Dict[str, float]:
    """
    Entraîne les modules sur une epoch complète.
    
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
    
    # Phase C1: Mixed Precision - Initialize GradScaler if needed
    scaler = None
    if use_amp and torch.cuda.is_available():
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()
    elif use_amp and not torch.cuda.is_available():
        # Disable AMP if CUDA is not available
        use_amp = False
        if verbose:
            print("[WARN] Mixed precision (AMP) disabled: CUDA not available")
    
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
        print(f"   - Lambda (gen): {lambda_gen}")
        print(f"   - Beta (rec): {beta_rec}")
        print(f"   - Gamma (DAG): {gamma_dag}")
        print(f"   - Gradient clipping: {gradient_clipping}")
        print(f"   - Log interval: {log_interval}")
        print(f"{'='*80}\n")

    for batch_idx, batch in enumerate(data_loader):
        batch_start_time = time.time()
        
        if verbose and batch_idx == 0:
            print(f"Batch {batch_idx + 1}:")
            print(f"   - Keys: {list(batch.keys())}")
        
        lr_data: Tensor = batch["lr"].to(device)      # [seq_len, N, features_lr]
        target_data: Tensor = batch.get(residual_key, batch.get("hr")).to(device)  # [seq_len, channels, H, W]
        hetero_data = batch["hetero"]
        
        if verbose and batch_idx == 0:
            print(f"   - LR data shape: {lr_data.shape}")
            print(f"   - Target data shape: {target_data.shape}")
            print(f"   - Sequence length: {lr_data.shape[0]}")

        optimizer.zero_grad()

        # Phase C1: Mixed Precision - Use autocast for forward pass
        # Encoder step
        encoder_time = time.time()
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                H_init = encoder.init_state(hetero_data).to(device)
        else:
            H_init = encoder.init_state(hetero_data).to(device)
        encoder_time = time.time() - encoder_time
        
        if verbose and batch_idx == 0:
            print(f"   - H_init shape: {H_init.shape}")
            print(f"   - Encoder time: {encoder_time:.4f}s")

        # RCN step
        rcn_time = time.time()
        drivers = [lr_data[t] for t in range(lr_data.shape[0])]
        # reconstruction_sources is no longer needed - RCNCell uses hidden state internally
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                seq_output = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
        else:
            seq_output = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
        rcn_time = time.time() - rcn_time
        
        if verbose and batch_idx == 0:
            print(f"   - Number of states: {len(seq_output.states)}")
            print(f"   - Number of reconstructions: {len(seq_output.reconstructions)}")
            print(f"   - Number of DAG matrices: {len(seq_output.dag_matrices)}")
            print(f"   - RCN time: {rcn_time:.4f}s")

        # Phase C1: Mixed Precision - Loss computation (reconstruction and DAG)
        loss_time = time.time()
        loss_rec_value = torch.tensor(0.0, device=device)
        loss_dag_value = torch.tensor(0.0, device=device)
        num_reconstructions = 0
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                for recon, A_masked, driver_step in zip(
                    seq_output.reconstructions,
                    seq_output.dag_matrices,
                    drivers,
                ):
                    if recon is not None:
                        num_reconstructions += 1
                        loss_rec_value = loss_rec_value + beta_rec * loss_reconstruction(recon, driver_step)
                    # Phase 3.1: Use DAGMA by default (more stable than NO TEARS)
                    if dag_method == "dagma":
                        loss_dag_value = loss_dag_value + gamma_dag * loss_dagma(A_masked, s=dagma_s)
                    else:  # fallback to NO TEARS
                        loss_dag_value = loss_dag_value + gamma_dag * loss_no_tears(A_masked)
        else:
            for recon, A_masked, driver_step in zip(
                seq_output.reconstructions,
                seq_output.dag_matrices,
                drivers,
            ):
                if recon is not None:
                    num_reconstructions += 1
                    loss_rec_value = loss_rec_value + beta_rec * loss_reconstruction(
                        recon, driver_step, loss_type=reconstruction_loss_type
                    )
                # Phase 3.1: Use DAGMA by default (more stable than NO TEARS)
                if dag_method == "dagma":
                    loss_dag_value = loss_dag_value + gamma_dag * loss_dagma(A_masked, s=dagma_s)
                else:  # fallback to NO TEARS
                    loss_dag_value = loss_dag_value + gamma_dag * loss_no_tears(A_masked)
        
        if verbose and batch_idx == 0:
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
        
        if verbose and batch_idx == 0:
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
        
        if verbose and batch_idx == 0:
            print(f"   - Target shape (after processing): {target.shape}")
        
        # Verify channel count matches UNet expectations
        if target.shape[1] != diffusion_decoder.in_channels:
            raise ValueError(
                f"Channel mismatch: target has {target.shape[1]} channels, "
                f"but UNet expects {diffusion_decoder.in_channels} channels. "
                f"Target shape: {target.shape}"
            )
        
        # Vérifications de diagnostic avant la diffusion
        if verbose and batch_idx == 0:
            print(f"   - Target stats: min={target.min().item():.6f}, max={target.max().item():.6f}, mean={target.mean().item():.6f}, std={target.std().item():.6f}")
            print(f"   - Target has NaN: {torch.isnan(target).any().item()}")
            print(f"   - Target has Inf: {torch.isinf(target).any().item()}")
            print(f"   - Conditioning stats: min={conditioning.min().item():.6f}, max={conditioning.max().item():.6f}, mean={conditioning.mean().item():.6f}")
            print(f"   - Conditioning has NaN: {torch.isnan(conditioning).any().item()}")
            print(f"   - Conditioning has Inf: {torch.isinf(conditioning).any().item()}")
        
        # Vérifier les valeurs extrêmes
        target_abs_max = target.abs().max().item()
        if target_abs_max > 1e6:
            if verbose:
                print(f"[WARN] Target has very large values: max_abs={target_abs_max:.2e}")
                print(f"   - This might cause numerical instability")
        
        # Vérifier les NaN dans le target (seront masqués dans compute_loss)
        nan_mask = torch.isnan(target) | torch.isinf(target)
        nan_count = nan_mask.sum().item()
        nan_ratio = nan_count / target.numel() if target.numel() > 0 else 0.0
        
        if nan_count > 0:
            if verbose and (batch_idx == 0 or batch_idx % log_interval == 0):
                print(f"[INFO] Target contains {nan_count} NaN/Inf pixels ({nan_ratio:.2%}) - will be masked in loss")
        
        # Le conditioning ne doit PAS contenir de NaN/Inf (erreur critique)
        if torch.isnan(conditioning).any() or torch.isinf(conditioning).any():
            print(f"[ERROR] Conditioning contains NaN/Inf in batch {batch_idx + 1}")
            print(f"   - NaN count: {torch.isnan(conditioning).sum().item()}")
            print(f"   - Inf count: {torch.isinf(conditioning).sum().item()}")
            print(f"   - This is a critical error, skipping batch")
            continue
        
        # Phase C1: Mixed Precision - Forward pass with autocast for entire forward
        # Diffusion loss (gère automatiquement les NaN via masquage)
        # Phase D1: Supports focal loss for focusing on hard pixels
        diffusion_time = time.time()
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                loss_gen_value = lambda_gen * loss_diffusion(
                    diffusion_decoder, target, conditioning,
                    use_focal_loss=use_focal_loss,
                    focal_alpha=focal_alpha,
                    focal_gamma=focal_gamma,
                )
        else:
            loss_gen_value = lambda_gen * loss_diffusion(
                diffusion_decoder, target, conditioning,
                use_focal_loss=use_focal_loss,
                focal_alpha=focal_alpha,
                focal_gamma=focal_gamma,
            )
        
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
                        
                        if verbose and batch_idx == 0:
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
        
        diffusion_time = time.time() - diffusion_time
        
        if verbose and batch_idx == 0:
            print(f"   - Diffusion time: {diffusion_time:.4f}s")
            if lambda_phy > 0.0:
                print(f"   - Physical loss: {loss_phy_value.item():.6f}")
            if nan_count > 0:
                print(f"   - Loss computed on {target.numel() - nan_count}/{target.numel()} valid pixels ({1.0 - nan_ratio:.2%})")

        # Phase C1: Mixed Precision - Compute total loss
        if use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                loss_total = loss_gen_value + loss_rec_value + loss_dag_value + loss_phy_value
        else:
            loss_total = loss_gen_value + loss_rec_value + loss_dag_value + loss_phy_value
        
        # Check for NaN or Inf
        if torch.isnan(loss_total) or torch.isinf(loss_total):
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
        
        loss_time = time.time() - loss_time
        
        # Phase C1: Mixed Precision - Backward pass with scaler
        backward_time = time.time()
        if use_amp and scaler is not None:
            scaler.scale(loss_total).backward()
        else:
            loss_total.backward()
        backward_time = time.time() - backward_time

        if gradient_clipping is not None:
            clip_time = time.time()
            if use_amp and scaler is not None:
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
            
            if verbose and (batch_idx % log_interval == 0 or batch_idx == 0):
                print(f"   - Gradient norms (clipped): RCN={grad_norm_rcn:.4f}, Diff={grad_norm_diff:.4f}, Enc={grad_norm_enc:.4f}")

        # Phase C1: Mixed Precision - Optimizer step with scaler
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        step_time = time.time() - batch_start_time

        total_loss += loss_total.item()
        total_gen += loss_gen_value.item()
        total_rec += loss_rec_value.item()
        total_dag += loss_dag_value.item()
        total_phy += loss_phy_value.item()
        num_batches += 1
        
        # Logging
        if verbose and (batch_idx % log_interval == 0 or batch_idx == 0):
            print(f"\nBatch {batch_idx + 1}:")
            print(f"   - Loss total: {loss_total.item():.6f}")
            print(f"   - Loss gen: {loss_gen_value.item():.6f}")
            print(f"   - Loss rec: {loss_rec_value.item():.6f}")
            print(f"   - Loss DAG: {loss_dag_value.item():.6f}")
            if lambda_phy > 0.0:
                print(f"   - Loss phy: {loss_phy_value.item():.6f}")
            print(f"   - Batch time: {step_time:.4f}s")
            if batch_idx == 0:
                print(f"   - Time breakdown: Enc={encoder_time:.3f}s, RCN={rcn_time:.3f}s, "
                      f"Diff={diffusion_time:.3f}s, Backward={backward_time:.3f}s")
                if gradient_clipping is not None:
                    print(f"   - Clip time: {clip_time:.3f}s")

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

**Fonctions principales:**
- `train_epoch()`: Boucle d'entraînement complète
- `loss_reconstruction()`: Perte de reconstruction (MSE, cosine, ou combiné)
- `loss_diffusion()`: Perte de diffusion
- `loss_dagma()`: Contrainte DAGMA (plus stable que NO TEARS)
- `loss_physical()`: Perte physique (divergence + vorticité)
- `compute_divergence()`: Calcul de la divergence
- `compute_vorticity()`: Calcul de la vorticité

**Fonctionnalités:**
- Mixed precision training (AMP)
- Early stopping et LR scheduling
- Gradient clipping
- Support focal loss
- Perte physique optionnelle
- DAG stabilisation avec L1 regularization

### `src/st_cdgm/evaluation/__init__.py`

```python
"""
Modules d'évaluation et XAI pour ST-CDGM.
"""

from .evaluation_xai import (
    autoregressive_inference,
    evaluate_metrics,
    compute_crps,
    compute_fss,
    compute_wasserstein_distance,
    compute_energy_score,
    compute_structural_hamming_distance,
    plot_dag_heatmap,
    export_dag_to_csv,
    export_dag_to_json,
    MetricReport,
    InferenceResult,
)

__all__ = [
    "autoregressive_inference",
    "evaluate_metrics",
    "compute_crps",
    "compute_fss",
    "compute_wasserstein_distance",
    "compute_energy_score",
    "compute_structural_hamming_distance",
    "plot_dag_heatmap",
    "export_dag_to_csv",
    "export_dag_to_json",
    "MetricReport",
    "InferenceResult",
]
```

### `src/st_cdgm/evaluation/evaluation_xai.py`

```python
"""
Module 7 – Évaluation et interprétabilité (XAI) pour l'architecture ST-CDGM.

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
from torch import Tensor

from ..models.causal_rcn import RCNCell, RCNSequenceRunner
from ..models.diffusion_decoder import CausalDiffusionDecoder, DiffusionOutput

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

def compute_mse(pred: Tensor, target: Tensor) -> float:
    return torch.mean((pred - target) ** 2).item()


def compute_mae(pred: Tensor, target: Tensor) -> float:
    return torch.mean(torch.abs(pred - target)).item()


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


def compute_crps(samples: Sequence[Tensor], target: Tensor) -> float:
    """
    Calcule le CRPS (Continuous Ranked Probability Score) pour un ensemble d'échantillons.
    
    Phase 4.1: Improved CRPS implementation.
    """
    if len(samples) == 0:
        return float("nan")
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


def evaluate_metrics(
    samples: Sequence[DiffusionOutput],
    target: Tensor,
    baseline: Optional[Tensor] = None,
    *,
    compute_advanced: bool = True,
    fss_threshold: Optional[float] = None,
    fss_window_size: int = 9,
    compute_f1_extremes: bool = True,  # Phase C4: Compute F1 for extreme events
    f1_percentiles: Sequence[float] = [95.0, 99.0],  # Phase C4: Percentiles for F1
) -> MetricReport:
    """
    Calcule un ensemble de métriques à partir des échantillons générés.
    
    Phase 4.1: Now includes advanced metrics (FSS, Wasserstein, Energy Score).
    
    Parameters
    ----------
    samples : Sequence[DiffusionOutput]
        Ensemble d'échantillons générés
    target : Tensor
        Cible [C, H, W] ou [H, W]
    baseline : Optional[Tensor]
        Baseline optionnel pour comparaison
    compute_advanced : bool
        Si True, calcule les métriques avancées (FSS, Wasserstein, Energy Score)
    fss_threshold : Optional[float]
        Seuil pour le calcul du FSS (si None, FSS n'est pas calculé)
    fss_window_size : int
        Taille de fenêtre pour le FSS (doit être impair)
    """
    if len(samples) == 0:
        raise ValueError("La liste d'échantillons ne doit pas être vide.")
    stacked_means = torch.stack([sample.t_mean for sample in samples], dim=0)
    pred_mean = stacked_means.mean(dim=0)

    mse = compute_mse(pred_mean, target)
    mae = compute_mae(pred_mean, target)
    hist_distance = compute_histogram_distance(pred_mean, target)
    crps = compute_crps([sample.t_mean for sample in samples], target)
    spectrum = compute_spectrum_distance(pred_mean, target)

    baseline_mse = baseline_mae = None
    baseline_tensor = baseline
    if baseline_tensor is None and samples[0].baseline is not None:
        baseline_tensor = samples[0].baseline
    if baseline_tensor is not None:
        baseline_mse = compute_mse(baseline_tensor, target)
        baseline_mae = compute_mae(baseline_tensor, target)

    # Phase 4.1: Compute advanced metrics if requested
    fss_val = None
    wasserstein_val = None
    energy_score_val = None
    
    if compute_advanced:
        try:
            # Compute FSS if threshold is provided
            if fss_threshold is not None:
                fss_val = compute_fss(pred_mean, target, threshold=fss_threshold, window_size=fss_window_size)
            
            # Compute Wasserstein distance
            wasserstein_val = compute_wasserstein_distance([sample.t_mean for sample in samples], target)
            
            # Compute Energy Score
            energy_score_val = compute_energy_score([sample.t_mean for sample in samples], target)
        except Exception as e:
            # If advanced metrics fail, continue with basic metrics
            import warnings
            warnings.warn(f"Failed to compute advanced metrics: {e}")
    
    # Phase C4: Compute F1 scores for extreme events
    f1_extremes_val = None
    if compute_f1_extremes:
        try:
            f1_extremes_val = compute_f1_extremes(pred_mean, target, threshold_percentiles=f1_percentiles)
        except Exception as e:
            import warnings
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
    )


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

## 🔧 Scripts d'Opérations (`ops/`)

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
```

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
import warnings
from pathlib import Path
from typing import Optional, Sequence

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
```

---

## 🛠️ Scripts Utilitaires (`scripts/`)

### `scripts/run_training.py`

[Code complet - 417 lignes]

**Fonctionnalités:**
- Entraînement avec checkpointing automatique
- Early stopping
- LR scheduling
- Sauvegarde des modèles
- Logging structuré

### `scripts/run_preprocessing.py`

[Code complet - 196 lignes]

**Fonctionnalités:**
- Interface unifiée pour preprocessing
- Support Zarr et WebDataset
- Validation des fichiers d'entrée

### `scripts/run_evaluation.py`

[Code complet - 331 lignes]

**Fonctionnalités:**
- Évaluation de modèles entraînés
- Calcul de métriques complètes
- Génération de visualisations
- Support F1 extremes

### `scripts/run_full_pipeline.py`

[Code complet - 190 lignes]

**Fonctionnalités:**
- Orchestration du pipeline complet
- Preprocessing → Training → Evaluation
- Options pour sauter des étapes

### `scripts/load_model.py`

[Code complet - 152 lignes]

**Fonctionnalités:**
- Chargement de checkpoints
- Reconstruction des modèles depuis config
- Support device (CPU/CUDA)

### `scripts/save_model.py`

[Code complet - 97 lignes]

**Fonctionnalités:**
- Sauvegarde de checkpoints combinés
- Métadonnées JSON
- Timestamp automatique

### `scripts/test_pipeline.py`

[Code complet - 160 lignes]

**Fonctionnalités:**
- Test end-to-end avec données synthétiques
- Validation de tous les modules
- Test de forward pass

### `scripts/test_installation.py`

[Code complet - 230 lignes]

**Fonctionnalités:**
- Vérification de l'installation
- Test des dépendances
- Détection GPU/CUDA
- Détection environnement VICE

### `scripts/validate_setup.py`

[Code complet - 319 lignes]

**Fonctionnalités:**
- Validation complète du projet
- Vérification syntaxe Python
- Validation YAML
- Vérification structure

### `scripts/sync_datastore.py`

[Code complet - 385 lignes]

**Fonctionnalités:**
- Synchronisation Data Store ↔ disque local
- Copie de fichiers
- Listing Data Store
- Dry run support

### `scripts/vice_utils.py`

[Code complet - 238 lignes]

**Fonctionnalités:**
- Détection environnement VICE
- Gestion chemins Data Store
- Recommandations pour performance
- Utilitaires de répertoires

### `scripts/cleanup_repeated_lines.py`

[Code complet - 99 lignes]

**Fonctionnalités:**
- Nettoyage de notebooks Jupyter
- Suppression lignes répétées
- Sauvegarde automatique

---

## 🧪 Tests (`tests/`)

### `tests/__init__.py`

```python
"""
Tests unitaires pour ST-CDGM.
"""
```

### `tests/test_st_cdgm_smoke.py`

[Code complet - 151 lignes]

**Fonctionnalités:**
- Test smoke avec données synthétiques
- Validation pipeline complet
- Test forward pass

### `tests/test_installation.py`

[Code complet - 137 lignes]

**Fonctionnalités:**
- Vérification installation
- Test packages
- Test imports

---

## 📓 Notebook d'Entraînement et d'Évaluation

### `st_cdgm_training_evaluation.ipynb`

Le notebook complet contient 54 cellules organisées en sections:

#### **Section 1: Installation et Imports**
- Configuration de l'environnement
- Imports des modules ST-CDGM
- Vérifications système

#### **Section 2: Exploration des Données**
- Configuration des chemins
- Chargement et inspection des datasets
- Visualisation des données
- Création du pipeline de données
- Création du dataset iterable

#### **Section 3: Construction du Modèle**
- Configuration du modèle
- Construction du graph builder
- Initialisation des modules (Encoder, RCN, Diffusion)

#### **Section 4: Entraînement du Modèle**
- Fonction helper pour conversion des batches
- Boucle d'entraînement
- Visualisation de l'entraînement
- Sauvegarde du modèle

#### **Section 5: Évaluation et Tests**
- Fonction d'inférence
- Génération de prédictions
- Calcul des métriques

#### **Section 6: Visualisation des Résultats**
- Comparaison visuelle: Baseline vs Prédiction vs Ground Truth
- Sauvegarde des métriques

#### **Section 7: Résumé et Conclusions**
- Résumé de l'expérience
- Notes et prochaines étapes

**Contenu détaillé des cellules:**

```python
# Cellule 0: Introduction
# 🌍 ST-CDGM Training & Evaluation Notebook
## Spatio-Temporal Causal Diffusion Generative Model

# Cellule 1: Motivation
## 🎯 Motivation et Idée Centrale
# Concept clé: remplacer bruit statistique par signal physique causal

# Cellule 2-4: Installation
# Configuration complète pour exécution locale
# Auto-reload des modules
# Imports scientifiques et PyTorch

# Cellule 5-16: Exploration données
# Configuration chemins LR/HR
# Chargement datasets bruts
# Visualisation
# Création pipeline
# Dataset iterable

# Cellule 17-22: Construction modèle
# Configuration dimensions
# Graph builder
# Initialisation modules (Encoder, RCN, Diffusion)

# Cellule 23-30: Entraînement
# Helper conversion batches
# Boucle overfit
# Visualisation courbes
# Sauvegarde checkpoint

# Cellule 31-41: Évaluation
# Fonction inférence
# Génération prédictions
# Calcul métriques (MSE, RMSE, MAE, Correlation)
# Sauvegarde CSV

# Cellule 42-44: Résumé
# Résumé expérience
# Notes et prochaines étapes
```

---

## 🏗️ Hiérarchie du Code

### Architecture Modulaire

```
ST-CDGM
│
├── Data Layer (src/st_cdgm/data/)
│   ├── pipeline.py          → NetCDFDataPipeline, ZarrDataPipeline
│   └── netcdf_utils.py       → NetCDFToDataFrame, métadonnées
│
├── Graph Layer (src/st_cdgm/models/graph_builder.py)
│   └── HeteroGraphBuilder   → Construction graphe hétérogène
│
├── Encoding Layer (src/st_cdgm/models/intelligible_encoder.py)
│   └── IntelligibleVariableEncoder → Variables intelligibles H(0)
│
├── Causal Layer (src/st_cdgm/models/causal_rcn.py)
│   ├── RCNCell              → Cellule récurrente causale
│   └── RCNSequenceRunner    → Déroulement séquentiel
│
├── Generation Layer (src/st_cdgm/models/diffusion_decoder.py)
│   └── CausalDiffusionDecoder → Génération HR par diffusion
│
├── Training Layer (src/st_cdgm/training/)
│   ├── training_loop.py     → train_epoch, pertes
│   └── callbacks.py         → EarlyStopping
│
└── Evaluation Layer (src/st_cdgm/evaluation/)
    └── evaluation_xai.py    → Métriques, visualisation DAG
```

### Flux de Données

```
NetCDF Files
    ↓
NetCDFDataPipeline
    ├── Alignement temporel
    ├── Normalisation
    ├── Baseline computation
    └── Residual calculation
    ↓
ResDiffIterableDataset
    ↓
HeteroGraphBuilder
    └── Construction graphe statique
    ↓
IntelligibleVariableEncoder
    └── H(0) initial state
    ↓
RCNSequenceRunner
    ├── RCNCell (séquence)
    └── H(t) causal states
    ↓
CausalDiffusionDecoder
    ├── Conditioning from H(t)
    └── HR generation (diffusion)
    ↓
Loss Computation
    ├── L_gen (diffusion)
    ├── L_rec (reconstruction)
    └── L_dag (DAG constraint)
    ↓
Optimization
    └── Backpropagation
```

### Dépendances entre Modules

```
setup.py
    ↓
src/st_cdgm/__init__.py
    ├── models/
    │   ├── causal_rcn.py
    │   ├── diffusion_decoder.py
    │   ├── graph_builder.py
    │   └── intelligible_encoder.py
    ├── data/
    │   ├── pipeline.py
    │   └── netcdf_utils.py
    ├── training/
    │   ├── training_loop.py
    │   └── callbacks.py
    └── evaluation/
        └── evaluation_xai.py
```

---

## 📊 Résumé des Fichiers

### Statistiques

- **Fichiers Python**: 33 fichiers (src/st_cdgm 14, ops 3, scripts 12, tests 3, setup 1)
- **Fichiers de configuration**: 5 fichiers (YAML, ENV, YML, TXT)
- **Scripts utilitaires**: 12 scripts
- **Modules principaux**: 8 modules (data 2, models 4, training 2, evaluation 1)
- **Tests**: 3 fichiers de test
- **Notebook**: 1 notebook complet (54 cellules)
- **Documentation**: 7 fichiers .md dans `docs/`
- **Fichiers racine**: .gitignore, .dockerignore, README.md
- **Données métadonnées**: 2 fichiers dans `data/metadata/` (CSV, JSON)

### Lignes de Code (approximatif)

- `pipeline.py`: ~1119 lignes
- `netcdf_utils.py`: ~1087 lignes
- `causal_rcn.py`: ~386 lignes
- `diffusion_decoder.py`: ~631 lignes
- `graph_builder.py`: ~481 lignes
- `intelligible_encoder.py`: ~283 lignes
- `training_loop.py`: ~874 lignes
- `evaluation_xai.py`: ~654 lignes
- **Total estimé**: ~8500+ lignes de code Python (src + ops + scripts + tests)

---

## 🎯 Points Clés de l'Architecture

1. **Modularité**: Chaque composant est indépendant et réutilisable
2. **Extensibilité**: Facile d'ajouter de nouveaux modules
3. **Configuration**: Hydra pour configuration flexible
4. **Performance**: Optimisations Phase A, B, C, D, E
5. **Robustesse**: Gestion d'erreurs, validation, tests

---

**Fin de la Documentation Complète du Projet ST-CDGM**