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
    # netcdf4 ET h5netcdf sont OBLIGATOIRES : nos .nc sont au format NetCDF-4
    # (HDF5). Sans eux xarray se rabat sur scipy, qui ne lit que le NetCDF-3, et
    # echoue avec "is not a valid NetCDF 3 file" — message trompeur, le fichier
    # est bon, c'est le moteur qui manque.
    subprocess.check_call([sys.executable, "-m", "pip", "-q", "install",
                           "netcdf4", "h5netcdf",
                           "xbatcher", "omegaconf", "diffusers", "torch-geometric"])
    ROOT = Path(REPO_DIR)

    # Verification que le code V8 est REELLEMENT arrive. Si la branche n'a pas
    # ete poussee, le clone rend une version anterieure et le notebook echoue
    # 4 cellules plus loin sur un ImportError incomprehensible.
    _requis = ["src/st_cdgm/data/derived.py", "src/st_cdgm/priors.py",
               "src/st_cdgm/models/bernoulli_gamma.py",
               "src/st_cdgm/evaluation/jensen.py", "config/dag_prior_v8_c7.yaml"]
    _absents = [f for f in _requis if not (ROOT / f).exists()]
    if _absents:
        _head = subprocess.check_output(
            ["git", "-C", REPO_DIR, "log", "-1", "--oneline"]).decode().strip()
        raise RuntimeError(
            "Le code V8 n'est pas dans la branche clonee. Manquants : "
            + ", ".join(_absents)
            + f" | HEAD = {_head} | branche = {GIT_BRANCH}. "
            "Pousser la branche (git push origin " + GIT_BRANCH + ") "
            "avant de relancer.")
else:
    ROOT = Path.cwd()

# chdir a la racine : les chemins relatifs (config/*.yaml, data/, checkpoints/)
# echouent sinon depuis /content.
os.chdir(ROOT)
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Verifier qu'un moteur NetCDF-4 est reellement disponible AVANT la Cell 3 :
# sinon l'erreur ne surgit qu'a l'ouverture du fichier, avec un message qui
# accuse le fichier au lieu de l'environnement.
import importlib
_moteurs = [m for m in ("netCDF4", "h5netcdf") if importlib.util.find_spec(m)]
if not _moteurs:
    raise ImportError(
        "Aucun moteur NetCDF-4 disponible (netCDF4, h5netcdf). xarray se "
        "rabattrait sur scipy, qui ne lit que le NetCDF-3 et rejetterait nos "
        "fichiers avec un message trompeur. Installer : pip install netcdf4 h5netcdf")
print(f"moteurs NetCDF : {_moteurs}")

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
