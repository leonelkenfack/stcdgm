# === Cell 1 : Bootstrap Colab ===
import subprocess, shlex, os, sys
from pathlib import Path

REPO_DIR   = Path('/content/climate_data')
GIT_URL    = 'https://github.com/leonelkenfack/stcdgm.git'
GIT_BRANCH = 'four-node-causal'

if not (REPO_DIR / '.git').exists():
    subprocess.run(shlex.split(f'git clone --depth 200 -b {GIT_BRANCH} {GIT_URL} {REPO_DIR}'), check=True)
else:
    subprocess.run(shlex.split(f'git -C {REPO_DIR} fetch --depth=200 origin {GIT_BRANCH}'), check=True)
    subprocess.run(shlex.split(f'git -C {REPO_DIR} reset --hard origin/{GIT_BRANCH}'), check=True)

os.chdir(str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / 'src'))

try:
    import torch_geometric, cftime, h5netcdf, xbatcher, diffusers
    from omegaconf import OmegaConf
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q',
        'torch_geometric', 'omegaconf==2.3.0', 'hydra-core==1.3.2',
        'diffusers==0.36.0', 'einops', 'scipy', 'h5py', 'netCDF4',
        'xarray', 'dask', 'zarr', 'safetensors==0.7.0',
        'xbatcher', 'webdataset', 'cftime', 'h5netcdf',
    ], check=True)

try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=False)
except ModuleNotFoundError:
    print('[bootstrap] not on Colab')

import torch, numpy as np, json, time
import xarray as xr
from omegaconf import OmegaConf

# Reproducibility (Expert IA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Pre-registration (Expert Recherche)
_git_sha = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()
print(f'[bootstrap] git SHA = {_git_sha}  branch = {GIT_BRANCH}')
print(f'[bootstrap] cwd={os.getcwd()}  torch={torch.__version__}  cuda={torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'[bootstrap] GPU = {torch.cuda.get_device_name(0)}  VRAM = {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')