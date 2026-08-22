# >>> Cell 1 : bootstrap Colab + montage Drive
import os, sys, json, math, time, warnings, subprocess
from pathlib import Path
warnings.filterwarnings("ignore")

GIT_URL    = "https://github.com/leonelkenfack/stcdgm.git"
GIT_BRANCH = "four-node-causal"
REPO_DIR   = "/content/climate_data"
DRIVE_ROOT = "/content/drive/MyDrive/climate_data"

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    # Les .nc d'entrainement (3 Go) sont gitignores : ils vivent sur Drive,
    # pas dans le depot. Sans ce montage, la Cell 2 s'arrete faute de donnees.
    if not Path("/content/drive").exists():
        from google.colab import drive
        drive.mount("/content/drive")

    if not Path(REPO_DIR).exists():
        # Clone sur le SSD local, jamais sur Drive : xarray y est ~20x plus lent.
        subprocess.check_call(["git", "clone", "--depth=200", "-b", GIT_BRANCH,
                               GIT_URL, REPO_DIR])
    else:
        subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "origin"])
        subprocess.check_call(["git", "-C", REPO_DIR, "checkout", GIT_BRANCH])
        subprocess.check_call(["git", "-C", REPO_DIR, "pull", "origin", GIT_BRANCH])
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install",
                           "xbatcher", "omegaconf", "diffusers", "torch-geometric"])
    ROOT = Path(REPO_DIR)
else:
    ROOT = Path.cwd()

# chdir a la racine : les chemins relatifs (config/*.yaml, data/, checkpoints/)
# echouent sinon depuis /content.
os.chdir(ROOT)
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np, torch
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"racine  : {ROOT}")
print(f"torch {torch.__version__} | device = {DEVICE}")
if DEVICE.type == "cuda":
    _p = torch.cuda.get_device_properties(0)
    print(f"  {_p.name} | {_p.total_memory / 2**30:.1f} GiB")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
else:
    print("  ATTENTION : prevu pour GPU. L'etage 2 sera tres lent sur CPU.")
