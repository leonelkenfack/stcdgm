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
    # Liste EPINGLEE, reprise telle quelle du 9-node et de V6' qui tournaient.
    # Une liste courte et non epinglee produit la cascade "une erreur par run" :
    #   cftime            -> decodage du calendrier 'noleap' de nos predicteurs
    #                        (sinon crash sur TOUT open NetCDF)
    #   netcdf4/h5netcdf  -> moteurs NetCDF-4. Sans eux xarray se rabat sur
    #                        scipy, qui ne lit que le NetCDF-3, et rejette nos
    #                        fichiers avec "is not a valid NetCDF 3 file" —
    #                        message trompeur : le fichier est bon.
    #   xbatcher          -> NetCDFDataPipeline.__init__ leve ImportError sans lui
    #   diffusers==0.36.0 + la pile epinglee -> UNet2DConditionModel stable
    # Le try/except evite de reinstaller a chaque relance de la cellule.
    try:
        import torch_geometric, cftime, h5netcdf, xbatcher, diffusers, omegaconf  # noqa: F401,E401
        print("deps critiques presentes — pip install saute.")
    except Exception as _e:
        print(f"pip install requis : {_e}")
        _DEPS = ["omegaconf==2.3.0", "hydra-core==1.3.2", "diffusers==0.36.0",
                 "transformers==4.57.6", "accelerate==1.12.0",
                 "huggingface-hub==0.36.0", "safetensors==0.7.0",
                 "xbatcher", "webdataset", "cftime", "h5netcdf", "netcdf4",
                 "numcodecs", "scipy", "torch-geometric", "xformers"]
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "--no-warn-script-location", *_DEPS])
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
_cal = importlib.util.find_spec("cftime") is not None
if not _moteurs or not _cal:
    raise ImportError(
        f"Environnement incomplet — moteurs NetCDF-4 : {_moteurs or 'AUCUN'}, "
        f"cftime : {'oui' if _cal else 'NON'}. Sans moteur NetCDF-4 xarray se "
        "rabat sur scipy (NetCDF-3 seulement) et accuse le fichier ; sans "
        "cftime le calendrier 'noleap' de nos predicteurs ne se decode pas. "
        "Installer : pip install netcdf4 h5netcdf cftime, puis relancer.")
print(f"moteurs NetCDF : {_moteurs} | cftime : oui")

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
