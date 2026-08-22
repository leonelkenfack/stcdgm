# >>> Cell 1 : bootstrap GPU
import os, sys, json, math, time, warnings, subprocess
from pathlib import Path
warnings.filterwarnings("ignore")

IN_COLAB = "google.colab" in sys.modules
REPO = Path("/content/climate_data") if IN_COLAB else Path.cwd()

if IN_COLAB and not REPO.exists():
    # SSD local, jamais Drive : xarray sur Drive est environ 20x plus lent.
    subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/leonelkenfack/climate_data.git", str(REPO)],
                   check=True)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "xbatcher", "omegaconf", "diffusers", "torch-geometric"], check=True)

os.chdir(REPO)
sys.path.insert(0, str(REPO / "src"))

import numpy as np, torch
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"torch {torch.__version__} | device = {DEVICE}")
if DEVICE.type == "cuda":
    _p = torch.cuda.get_device_properties(0)
    print(f"  {_p.name} | {_p.total_memory / 2**30:.1f} GiB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("  ATTENTION : ce notebook est prevu pour GPU, l'etage 2 sera tres lent sur CPU.")
