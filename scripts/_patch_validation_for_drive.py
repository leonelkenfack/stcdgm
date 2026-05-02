"""
Adapte ``st_cdgm_validation_inference.ipynb`` pour Colab + Drive :

1. Insère un COLAB_BOOTSTRAP en tête : montage Drive, ``git clone`` (ou
   ``git pull``) du repo vers ``/content/climate_data`` (SSD), ``pip
   install --no-deps -e .``, ``chdir`` à la racine projet. Hors Colab :
   no-op + remontée automatique à la racine.

2. Insère DRIVE_OVERRIDE juste après le chargement de CONFIG : redirige
   ``CONFIG.checkpoint.save_dir`` vers ``/content/drive/MyDrive/climate_data/ckpt``
   (où la cellule de persistance d'entraînement écrit ``epoch_last.pth``)
   et expose ``TEST_ROOT_OVERRIDE`` = ``/content/drive/MyDrive/climate_data/data/raw/test``.

3. Étend la recherche de checkpoint pour reconnaître les fichiers produits
   par ``persist_epoch_checkpoint`` (``epoch_last.pth``, ``epoch_best.pth``).

4. Si ``TEST_ROOT_OVERRIDE`` est défini (Colab), l'utilise pour ``TEST_ROOT``.

5. Reconstruit ``diffusion`` via ``CONFIG.diffusion.unet_kwargs`` (au lieu
   du UNet hardcodé qui ne match plus le checkpoint Sprint 4) et instancie
   ``CausalConditioningProjector`` quand ``encoder.causal_conditioning=true``.

Toutes les modifs sont sentinellées (``# >>> VALIDATION_*``) → idempotent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_validation_inference.ipynb"


COLAB_BOOTSTRAP = '''\
# >>> VALIDATION_COLAB_BOOTSTRAP
# Bootstrap Colab pour la validation : clone/pull du repo sur SSD,
# install editable, montage Drive (où vit le checkpoint).
# Hors Colab : no-op + chdir vers la racine projet.
import os, sys, subprocess, time, shlex
from pathlib import Path

GIT_URL = "https://github.com/leonelkenfack/stcdgm.git"
GIT_BRANCH = "main"
LOCAL_PROJECT = "/content/climate_data"  # SSD (PAS Drive — FUSE trop lent)
GIT_PULL_ON_RESUME = True
SKIP_PIP_IF_IMPORTABLE = True

_IS_COLAB = "google.colab" in sys.modules or Path("/content").exists()


def _run(cmd, *, check=True, timeout=None):
    print(f"$ {cmd}")
    t0 = time.time()
    rc = subprocess.call(shlex.split(cmd), timeout=timeout)
    dt = time.time() - t0
    print(f"  -> rc={rc}  ({dt:.1f}s)")
    if check and rc != 0:
        raise RuntimeError(f"Commande echouee : {cmd!r} (rc={rc})")
    return rc


if _IS_COLAB:
    _T0 = time.time()
    print("Colab detecte - bootstrap validation en cours...")

    # 1) Drive (idempotent)
    from google.colab import drive  # type: ignore[import-not-found]
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    else:
        print("   /content/drive deja monte.")

    # 2) Clone/pull vers SSD
    project_path = Path(LOCAL_PROJECT)
    if not (project_path / ".git").exists():
        project_path.parent.mkdir(parents=True, exist_ok=True)
        _run(f"git clone --depth 1 -b {GIT_BRANCH} {GIT_URL} {LOCAL_PROJECT}")
    elif GIT_PULL_ON_RESUME:
        try:
            _run(f"git -C {LOCAL_PROJECT} pull --ff-only", timeout=60, check=False)
        except Exception as e:
            print(f"   git pull a leve : {e}")

    os.chdir(project_path)
    print(f"   chdir -> {os.getcwd()}")

    # 3) Pip install (skip si st_cdgm deja importable)
    src_path = str(project_path / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    _need_pip = True
    if SKIP_PIP_IF_IMPORTABLE:
        try:
            import st_cdgm  # noqa: F401
            from omegaconf import OmegaConf  # noqa: F401
            from diffusers import UNet2DConditionModel  # noqa: F401
            import torch_geometric  # noqa: F401
            _need_pip = False
            print("OK Imports critiques presents - pip install saute.")
        except ImportError as _imp_err:
            print(f"   import a echoue ({_imp_err}) - pip install requis.")

    if _need_pip:
        EXTRA_DEPS = [
            "omegaconf==2.3.0",
            "hydra-core==1.3.2",
            "diffusers==0.36.0",
            "transformers==4.57.6",
            "accelerate==1.12.0",
            "huggingface-hub==0.36.0",
            "safetensors==0.7.0",
            "xbatcher",
            "webdataset",
            "cftime",
            "h5netcdf",
            "numcodecs",
            "torch-geometric",
            "xformers",  # parite avec le notebook training (sm_75 / sm_80 attention)
        ]
        deps_str = " ".join(shlex.quote(p) for p in EXTRA_DEPS)
        _run(
            f"{shlex.quote(sys.executable)} -m pip install --no-warn-script-location {deps_str}",
            timeout=600,
        )
        _run(
            f"{shlex.quote(sys.executable)} -m pip install --no-warn-script-location "
            f"--no-deps -e {LOCAL_PROJECT}",
            timeout=120,
        )

    print(f"Bootstrap validation termine en {time.time() - _T0:.1f}s.")

else:
    _here = Path.cwd()
    for _candidate in [_here, *_here.parents]:
        if (_candidate / "config" / "training_config.yaml").exists() and (_candidate / "setup.py").exists():
            if _candidate != _here:
                os.chdir(_candidate)
                print(f"chdir -> {os.getcwd()} (racine projet detectee)")
            break
    print("Hors Colab - bootstrap saute.")
'''


DRIVE_OVERRIDE = '''\
# >>> VALIDATION_DRIVE_OVERRIDE
# Pendant pour la validation de la cellule DATA_ROOT_DRIVE du notebook
# training. Resout DATA_ROOT (Drive sur Colab, local sinon), redirige
# CONFIG.checkpoint.save_dir + chemins static/norm sous DATA_ROOT,
# telecharge ce que l'inference exige depuis Zenodo (4 fichiers test) +
# Drive public (static + mean + std), et expose TEST_ROOT_OVERRIDE pour
# la cellule "Configuration test" plus bas.
#
# Hors Colab : DATA_ROOT = data/raw (assume deja peuple).
import os, sys, time
import urllib.request, urllib.error
from pathlib import Path

_ON_COLAB = "google.colab" in sys.modules or Path("/content").exists()
DATA_ROOT_LOCAL = Path("data/raw")
DATA_ROOT_DRIVE = Path("/content/drive/MyDrive/climate_data/data")

if _ON_COLAB and DATA_ROOT_DRIVE.parent.parent.exists():
    DATA_ROOT = DATA_ROOT_DRIVE
    print(f"DATA_ROOT = Drive ({DATA_ROOT})")
    # Persistance checkpoint : meme racine que data/, avec /ckpt en frere.
    _drive_ckpt = DATA_ROOT_DRIVE.parent / "ckpt"
    _drive_ckpt.mkdir(parents=True, exist_ok=True)
    CONFIG.checkpoint.save_dir = str(_drive_ckpt)
    print(f"OK CONFIG.checkpoint.save_dir -> {CONFIG.checkpoint.save_dir}")
else:
    DATA_ROOT = DATA_ROOT_LOCAL
    print(f"DATA_ROOT = local ({DATA_ROOT.resolve()})")

DATA_ROOT.mkdir(parents=True, exist_ok=True)


def _relocate(p):
    if not p:
        return p
    s = str(p)
    if s.startswith("data/raw/"):
        return str(DATA_ROOT / s[len("data/raw/"):])
    return s


# Static + norm_coefs viennent de CONFIG.data ; on relocalise sous DATA_ROOT.
if CONFIG.data.get("static_path"):
    CONFIG.data.static_path = _relocate(CONFIG.data.static_path)
STATIC_PATH = str(CONFIG.data.static_path) if CONFIG.data.get("static_path") else None
MEAN_PATH = str(DATA_ROOT / "normalization_coefs" / "mean_1974_2011.nc")
STD_PATH = str(DATA_ROOT / "normalization_coefs" / "std_1974_2011.nc")
TEST_ROOT_OVERRIDE = DATA_ROOT / "test"
TEST_ROOT_OVERRIDE.mkdir(parents=True, exist_ok=True)


def stream_download(url, dest, *, retries=5, chunk_size=1024 * 1024,
                    connect_timeout=30):
    """Streaming + atomique + reprise (Range header). Cf. notebook training."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        already = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if already > 0:
            req.add_header("Range", f"bytes={already}-")
            print(f"  reprise a {already / 1e6:.1f} MB")
        try:
            with urllib.request.urlopen(req, timeout=connect_timeout) as resp:
                total = resp.length or (
                    int(resp.headers["Content-Length"])
                    if resp.headers.get("Content-Length") else None
                )
                grand = (total + already) if total else None
                mode = "ab" if already > 0 else "wb"
                with open(part, mode) as f:
                    downloaded = already
                    last_log, last_bytes = time.time(), downloaded
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_log >= 5.0:
                            speed = (downloaded - last_bytes) / (now - last_log) / 1e6
                            if grand:
                                pct = 100.0 * downloaded / grand
                                print(f"    {downloaded/1e6:7.1f}/{grand/1e6:7.1f} MB ({pct:5.1f}%) {speed:5.1f} MB/s")
                            else:
                                print(f"    {downloaded/1e6:7.1f} MB {speed:5.1f} MB/s")
                            last_log, last_bytes = now, downloaded
            os.replace(part, dest)
            print(f"  OK {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return True
        except urllib.error.HTTPError as e:
            if e.code in (503, 504, 429):
                wait = min(60, 2 ** attempt)
                print(f"  HTTP {e.code} - retry {wait}s")
                time.sleep(wait)
            elif e.code == 416:
                os.replace(part, dest)
                return True
            else:
                print(f"  HTTP {e.code}: {e.reason}")
                return False
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = min(60, 2 ** attempt)
            print(f"  reseau ({type(e).__name__}: {e}) - retry {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"  {type(e).__name__}: {e}")
            return False
    return False


# 4 fichiers test Zenodo
URLS_TEST = [
    ("EC-Earth3_histupdated_compressed.nc",
     "https://zenodo.org/records/10889046/files/EC-Earth3_histupdated_compressed.nc?download=1"),
    ("EC-Earth3_historical_precip_compressed.nc",
     "https://zenodo.org/records/10889046/files/EC-Earth3_historical_precip_compressed.nc?download=1"),
    ("NorESM2-MM_histupdated_compressed.nc",
     "https://zenodo.org/records/10889046/files/NorESM2-MM_histupdated_compressed.nc?download=1"),
    ("NorESM2-MM_historical_precip_compressed.nc",
     "https://zenodo.org/records/10889046/files/NorESM2-MM_historical_precip_compressed.nc?download=1"),
]

print(f"\\nTEST_ROOT: {TEST_ROOT_OVERRIDE}")
for _filename, _url in URLS_TEST:
    _filepath = TEST_ROOT_OVERRIDE / _filename
    if _filepath.exists() and _filepath.stat().st_size > 0:
        print(f"  OK {_filename} ({_filepath.stat().st_size / 1e6:.1f} MB)")
        continue
    print(f"  DL {_filename}")
    if not stream_download(_url, str(_filepath)):
        raise RuntimeError(f"Echec telechargement {_filename}")


# Static + norm_coefs : Drive public via gdown (fallback)
def _exists_with_drive_sync(p):
    if not p:
        return False
    pth = Path(p)
    if pth.exists():
        return True
    parent = pth.parent
    if not parent.exists():
        return False
    try:
        os.listdir(parent)  # force FUSE enumeration
    except OSError:
        return False
    return pth.exists()


_PUBLIC_DRIVE_FALLBACKS = {
    "static_predictors/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc":
        "1KY6IS1W5Wt-l_xyV7Qw8caA49zPzuSEx",
    "normalization_coefs/mean_1974_2011.nc":
        "14wVaJTUDgLwLlFcqRFA6pzJg9tZtAVQ0",
    "normalization_coefs/std_1974_2011.nc":
        "1ycqq9DqpfdOOiyQqgKs797OzRdHND3ZL",
}


def _gdown_install_if_needed():
    try:
        import gdown  # noqa: F401
        return True
    except ImportError:
        import subprocess as _sp
        try:
            _sp.check_call([sys.executable, "-m", "pip", "install", "-q", "gdown"], timeout=120)
            import gdown  # noqa: F401
            return True
        except Exception as e:
            print(f"  pip install gdown a echoue: {e}")
            return False


def _try_public_drive_download(path):
    if not path:
        return False
    pth = Path(path)
    rel_key = None
    for _key in _PUBLIC_DRIVE_FALLBACKS:
        if str(pth).endswith(_key.replace("/", os.sep)) or str(pth).endswith(_key):
            rel_key = _key
            break
    if rel_key is None:
        return False
    file_id = _PUBLIC_DRIVE_FALLBACKS[rel_key]
    pth.parent.mkdir(parents=True, exist_ok=True)
    if not _gdown_install_if_needed():
        return False
    import gdown
    try:
        print(f"  gdown.download(id={file_id}) -> {pth}")
        gdown.download(id=file_id, output=str(pth), quiet=False)
        return pth.exists() and pth.stat().st_size > 0
    except Exception as e:
        print(f"  gdown a echoue: {e}")
        return False


for _var, _name in [("STATIC_PATH", "Static"), ("MEAN_PATH", "Mean"), ("STD_PATH", "Std")]:
    _p = globals()[_var]
    if _exists_with_drive_sync(_p):
        print(f"OK {_name}: {_p}")
        continue
    print(f"  {_name} absent: {_p} - tentative Drive public...")
    if _try_public_drive_download(_p):
        print(f"  OK {_name} telecharge.")
    else:
        globals()[_var] = None
        print(f"  {_name} -> None")
'''


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


def _make_code_cell(src, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def patch_notebook() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    # 1) Bootstrap : insere apres le markdown intro (cell 0), avant cell 1.
    boot_idx = _find_cell(cells, lambda s: "VALIDATION_COLAB_BOOTSTRAP" in s)
    if boot_idx is None:
        cells.insert(1, _make_code_cell(COLAB_BOOTSTRAP, "valbootstrap"))
        n_changed += 1
        print("  + COLAB_BOOTSTRAP inseree en position 1")
    else:
        if "".join(cells[boot_idx]["source"]) != COLAB_BOOTSTRAP:
            cells[boot_idx]["source"] = COLAB_BOOTSTRAP.splitlines(keepends=True)
            cells[boot_idx]["outputs"] = []
            cells[boot_idx]["execution_count"] = None
            n_changed += 1
            print(f"  ~ COLAB_BOOTSTRAP mise a jour (cell {boot_idx})")
        else:
            print(f"  = COLAB_BOOTSTRAP deja a jour (cell {boot_idx})")

    # 2) Drive override : insere juste apres la cellule qui charge CONFIG.
    cfg_idx = _find_cell(cells, lambda s: "OmegaConf.load(\"config/training_config.yaml\")" in s)
    if cfg_idx is None:
        raise RuntimeError("Cellule de chargement CONFIG introuvable")
    over_idx = _find_cell(cells, lambda s: "VALIDATION_DRIVE_OVERRIDE" in s)
    if over_idx is None:
        cells.insert(cfg_idx + 1, _make_code_cell(DRIVE_OVERRIDE, "valdriveoverride"))
        n_changed += 1
        print(f"  + DRIVE_OVERRIDE inseree en position {cfg_idx + 1}")
    else:
        if "".join(cells[over_idx]["source"]) != DRIVE_OVERRIDE:
            cells[over_idx]["source"] = DRIVE_OVERRIDE.splitlines(keepends=True)
            cells[over_idx]["outputs"] = []
            cells[over_idx]["execution_count"] = None
            n_changed += 1
            print(f"  ~ DRIVE_OVERRIDE mise a jour (cell {over_idx})")
        else:
            print(f"  = DRIVE_OVERRIDE deja a jour (cell {over_idx})")

    # 3) Checkpoint search : forcer epoch_last (poids actuels) en tete.
    ckpt_idx = _find_cell(cells, lambda s: "CHECKPOINT_PATH" in s and 'os.environ.get("ST_CDGM_CHECKPOINT"' in s)
    if ckpt_idx is None:
        print("  ! Cellule checkpoint introuvable - skip")
    else:
        src = "".join(cells[ckpt_idx]["source"])
        # Trois formes a reconnaitre : original (pre-patch), patch v1 (epoch_best en tete),
        # patch v2 (epoch_last en tete = forme finale).
        FORM_ORIG = (
            'for name in (\n'
            '    "st_cdgm_checkpoint_best.pth",\n'
            '    "st_cdgm_checkpoint.pth",\n'
            '    "st_cdgm_checkpoint_last.pth",\n'
            '):'
        )
        FORM_V1 = (
            '# >>> VALIDATION_CKPT_NAMES\n'
            '# Inclut les fichiers produits par persist_epoch_checkpoint (Sprint 4)\n'
            '# en plus des noms historiques.\n'
            'for name in (\n'
            '    "epoch_best.pth",\n'
            '    "epoch_last.pth",\n'
            '    "st_cdgm_checkpoint_best.pth",\n'
            '    "st_cdgm_checkpoint.pth",\n'
            '    "st_cdgm_checkpoint_last.pth",\n'
            '):'
        )
        FORM_V2 = (
            '# >>> VALIDATION_CKPT_NAMES\n'
            "# Priorite epoch_last (poids actuels = derniere epoque entrainee)\n"
            "# avant epoch_best (utilisable via env ST_CDGM_CHECKPOINT si vraiment\n"
            "# voulu).\n"
            'for name in (\n'
            '    "epoch_last.pth",\n'
            '    "epoch_best.pth",\n'
            '    "st_cdgm_checkpoint_last.pth",\n'
            '    "st_cdgm_checkpoint.pth",\n'
            '    "st_cdgm_checkpoint_best.pth",\n'
            '):'
        )
        if FORM_V2 in src:
            print(f"  = checkpoint candidates deja en mode epoch_last (cell {ckpt_idx})")
        else:
            replaced = False
            for old in (FORM_V1, FORM_ORIG):
                if old in src:
                    src = src.replace(old, FORM_V2, 1)
                    replaced = True
                    break
            if replaced:
                cells[ckpt_idx]["source"] = src.splitlines(keepends=True)
                cells[ckpt_idx]["outputs"] = []
                cells[ckpt_idx]["execution_count"] = None
                n_changed += 1
                print(f"  ~ checkpoint candidates -> epoch_last en priorite (cell {ckpt_idx})")
            else:
                print("  ! Bloc 'for name in (...)' non reconnu - skip")

    # 4) Test data : utiliser TEST_ROOT_OVERRIDE si defini.
    test_idx = _find_cell(cells, lambda s: "TEST_GCMS = discover_test_gcms()" in s)
    if test_idx is None:
        print("  ! Cellule TEST_ROOT introuvable - skip")
    else:
        src = "".join(cells[test_idx]["source"])
        OLD = '# Configuration test (dossiers et GCM)\nTEST_ROOT = Path("data/raw/test")'
        NEW = (
            '# Configuration test (dossiers et GCM)\n'
            '# >>> VALIDATION_TEST_ROOT\n'
            '# Sur Colab, DRIVE_OVERRIDE expose TEST_ROOT_OVERRIDE pointant vers\n'
            '# Drive ; sinon on garde le chemin local.\n'
            '_override = globals().get("TEST_ROOT_OVERRIDE", None)\n'
            'TEST_ROOT = Path(_override) if _override is not None else Path("data/raw/test")'
        )
        if "VALIDATION_TEST_ROOT" in src:
            print(f"  = TEST_ROOT deja override-aware (cell {test_idx})")
        elif OLD in src:
            cells[test_idx]["source"] = src.replace(OLD, NEW, 1).splitlines(keepends=True)
            cells[test_idx]["outputs"] = []
            cells[test_idx]["execution_count"] = None
            n_changed += 1
            print(f"  ~ TEST_ROOT branche sur override (cell {test_idx})")
        else:
            print("  ! Bloc TEST_ROOT non trouve - skip")

    # 5) Diffusion rebuild via CONFIG.diffusion.unet_kwargs + Causal projector.
    diff_idx = _find_cell(cells, lambda s: "CausalDiffusionDecoder(" in s and "block_out_channels=(32, 64)" in s)
    if diff_idx is None:
        # Peut-etre deja patche
        if _find_cell(cells, lambda s: "VALIDATION_DIFF_REBUILD" in s) is not None:
            print("  = diffusion rebuild deja a jour")
        else:
            print("  ! Cellule diffusion rebuild introuvable - skip")
    else:
        src = "".join(cells[diff_idx]["source"])
        OLD_DIFF = '''diffusion = CausalDiffusionDecoder(
    in_channels=hr_channels,
    conditioning_dim=CONFIG.diffusion.conditioning_dim,
    height=CONFIG.diffusion.height,
    width=CONFIG.diffusion.width,
    num_diffusion_steps=CONFIG.diffusion.steps,
    unet_kwargs=dict(
        layers_per_block=1,
        block_out_channels=(32, 64),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        mid_block_type="UNetMidBlock2D",
        norm_num_groups=8,
        class_embed_type="projection",
        projection_class_embeddings_input_dim=len(encoder_configs) * CONFIG.diffusion.conditioning_dim,
        resnet_time_scale_shift="scale_shift",
        attention_head_dim=32,
        only_cross_attention=[False, True],
    ),
).to(DEVICE)'''

        NEW_DIFF = '''# >>> VALIDATION_DIFF_REBUILD
# Sprint 4 : reconstruire le UNet a partir de CONFIG.diffusion.unet_kwargs
# (et non d'un dict hardcode obsolete) sinon les state_dict du checkpoint
# ne matchent plus (block_out_channels, mid_block_type, num_dag_tokens...).
from omegaconf import OmegaConf as _OC
UNET_KWARGS = _OC.to_container(CONFIG.diffusion.unet_kwargs, resolve=True)
for _k in ("down_block_types", "up_block_types"):
    if _k in UNET_KWARGS and isinstance(UNET_KWARGS[_k], list):
        UNET_KWARGS[_k] = tuple(UNET_KWARGS[_k])
UNET_KWARGS["projection_class_embeddings_input_dim"] = (
    len(encoder_configs) * CONFIG.diffusion.conditioning_dim
)

diffusion = CausalDiffusionDecoder(
    in_channels=hr_channels,
    conditioning_dim=CONFIG.diffusion.conditioning_dim,
    height=CONFIG.diffusion.height,
    width=CONFIG.diffusion.width,
    num_diffusion_steps=CONFIG.diffusion.steps,
    unet_kwargs=UNET_KWARGS,
    use_gradient_checkpointing=bool(
        CONFIG.diffusion.get("use_gradient_checkpointing", False)
    ),
    scheduler_type=CONFIG.diffusion.get("scheduler_type", "ddpm"),
    conv_padding_mode=CONFIG.diffusion.get("conv_padding_mode", "zeros"),
    anti_checkerboard=bool(CONFIG.diffusion.get("anti_checkerboard", False)),
).to(DEVICE)'''

        if OLD_DIFF in src:
            src = src.replace(OLD_DIFF, NEW_DIFF, 1)

            # Aussi : remplacer le bloc spatial_projector pour supporter
            # CausalConditioningProjector (Sprint 2/4).
            OLD_SP = '''spatial_projector = None
_sp_state = (
    checkpoint.get("spatial_projector_state_dict")
    or checkpoint.get("spatial_projector_state")
)
if _sp_state is not None:
    from st_cdgm.models.intelligible_encoder import SpatialConditioningProjector
    _spatial_target_shape = tuple(CONFIG.diffusion.get("spatial_target_shape", [6, 7]))
    spatial_projector = SpatialConditioningProjector(
        num_vars=len(encoder_configs),
        hidden_dim=CONFIG.rcn.hidden_dim,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        lr_shape=lr_shape,
        target_shape=_spatial_target_shape,
    ).to(DEVICE)
    spatial_projector.load_state_dict(strip_torch_compile_prefix(_sp_state))
    spatial_projector.eval()
    print(f"SpatialConditioningProjector reconstruit (target_shape={_spatial_target_shape}).")
else:
    print("[INFO] Pas de spatial_projector dans le checkpoint — fallback conditioning global.")'''

            NEW_SP = '''# >>> VALIDATION_PROJECTOR_REBUILD
# Sprint 2/4 : reconstruire CausalConditioningProjector si la config
# encoder.causal_conditioning=true (sinon SpatialConditioningProjector).
spatial_projector = None
_sp_state = (
    checkpoint.get("spatial_projector_state_dict")
    or checkpoint.get("spatial_projector_state")
)
if _sp_state is not None:
    from st_cdgm.models.intelligible_encoder import (
        SpatialConditioningProjector,
        CausalConditioningProjector,
    )
    _spatial_target_shape = tuple(CONFIG.diffusion.get("spatial_target_shape", [6, 7]))
    _use_causal = bool(CONFIG.encoder.get("causal_conditioning", False))
    if _use_causal:
        spatial_projector = CausalConditioningProjector(
            num_vars=len(encoder_configs),
            hidden_dim=CONFIG.rcn.hidden_dim,
            conditioning_dim=CONFIG.diffusion.conditioning_dim,
            lr_shape=lr_shape,
            target_shape=_spatial_target_shape,
            num_dag_tokens=int(CONFIG.encoder.get("num_dag_tokens", 1)),
        ).to(DEVICE)
        print(f"CausalConditioningProjector reconstruit (num_dag_tokens={CONFIG.encoder.get('num_dag_tokens', 1)}).")
    else:
        spatial_projector = SpatialConditioningProjector(
            num_vars=len(encoder_configs),
            hidden_dim=CONFIG.rcn.hidden_dim,
            conditioning_dim=CONFIG.diffusion.conditioning_dim,
            lr_shape=lr_shape,
            target_shape=_spatial_target_shape,
        ).to(DEVICE)
        print(f"SpatialConditioningProjector reconstruit (target_shape={_spatial_target_shape}).")
    spatial_projector.load_state_dict(strip_torch_compile_prefix(_sp_state))
    spatial_projector.eval()
else:
    print("[INFO] Pas de spatial_projector dans le checkpoint — fallback conditioning global.")'''

            if OLD_SP in src:
                src = src.replace(OLD_SP, NEW_SP, 1)
            else:
                print("  ! bloc spatial_projector non trouve - laisse en l'etat")

            cells[diff_idx]["source"] = src.splitlines(keepends=True)
            cells[diff_idx]["outputs"] = []
            cells[diff_idx]["execution_count"] = None
            n_changed += 1
            print(f"  ~ diffusion + projector rebuild via CONFIG.diffusion.unet_kwargs (cell {diff_idx})")
        else:
            print("  ! Bloc diffusion hardcode non trouve - skip")

    # 6) Fix appel CausalConditioningProjector dans generate_prediction_stable.
    # Le projector causal exige A_dag en 2eme arg (sinon TypeError).
    inf_idx = _find_cell(cells, lambda s: "def generate_prediction_stable" in s)
    if inf_idx is None:
        print("  ! generate_prediction_stable introuvable - skip")
    else:
        src = "".join(cells[inf_idx]["source"])
        OLD_CALL = (
            '    conditioning_spatial = None\n'
            '    _sp = globals().get("spatial_projector", None)\n'
            '    if _sp is not None:\n'
            '        conditioning_spatial = _sp(H_last).to(DEVICE)'
        )
        NEW_CALL = (
            '    # >>> VALIDATION_PROJECTOR_CALL\n'
            '    # Le CausalConditioningProjector exige A_dag en 2eme arg ;\n'
            '    # le SpatialConditioningProjector n\'en prend pas. On detecte\n'
            '    # par presence de l\'attribut dag_mlp (Sprint 2 / Sprint 4).\n'
            '    conditioning_spatial = None\n'
            '    _sp = globals().get("spatial_projector", None)\n'
            '    if _sp is not None:\n'
            '        _sp_base = _sp.module if hasattr(_sp, "module") else _sp\n'
            '        if hasattr(_sp_base, "dag_mlp"):\n'
            '            _rcn_base = rcn_runner.cell\n'
            '            _rcn_base = _rcn_base.module if hasattr(_rcn_base, "module") else _rcn_base\n'
            '            _A = _rcn_base.A_dag\n'
            '            _A_masked = _A - torch.diag(torch.diagonal(_A))\n'
            '            conditioning_spatial = _sp(H_last, _A_masked).to(DEVICE)\n'
            '        else:\n'
            '            conditioning_spatial = _sp(H_last).to(DEVICE)'
        )
        if "VALIDATION_PROJECTOR_CALL" in src:
            print(f"  = projector call deja patche (cell {inf_idx})")
        elif OLD_CALL in src:
            cells[inf_idx]["source"] = src.replace(OLD_CALL, NEW_CALL, 1).splitlines(keepends=True)
            cells[inf_idx]["outputs"] = []
            cells[inf_idx]["execution_count"] = None
            n_changed += 1
            print(f"  ~ projector call branche sur A_dag pour CausalConditioningProjector (cell {inf_idx})")
        else:
            print("  ! Bloc projector call original non trouve - skip")

    # 7) Plafond N_TEST_SAMPLES : 10 -> 100.
    eval_idx = _find_cell(cells, lambda s: "N_TEST_SAMPLES = min(N_TEST_SAMPLES, 10)" in s
                          or "VALIDATION_N_SAMPLES" in s)
    if eval_idx is None:
        print("  ! cellule N_TEST_SAMPLES introuvable - skip")
    else:
        src = "".join(cells[eval_idx]["source"])
        OLD_BLOCK = (
            '# Plafond dur pour limiter le temps tout en gardant une eval fiable\n'
            'if _max_test_samples is None:\n'
            '    N_TEST_SAMPLES = 10\n'
            'else:\n'
            '    try:\n'
            '        N_TEST_SAMPLES = int(_max_test_samples)\n'
            '    except Exception:\n'
            '        N_TEST_SAMPLES = 10\n'
            '    if N_TEST_SAMPLES <= 0:\n'
            '        N_TEST_SAMPLES = 10\n'
            '    N_TEST_SAMPLES = min(N_TEST_SAMPLES, 10)'
        )
        NEW_BLOCK = (
            '# >>> VALIDATION_N_SAMPLES\n'
            '# Plafond eleve a 100 (suffisant pour stabiliser CRPS/MSE/F1 ; couts ~50s/echantillon -> ~80 min/GCM)\n'
            'if _max_test_samples is None:\n'
            '    N_TEST_SAMPLES = 100\n'
            'else:\n'
            '    try:\n'
            '        N_TEST_SAMPLES = int(_max_test_samples)\n'
            '    except Exception:\n'
            '        N_TEST_SAMPLES = 100\n'
            '    if N_TEST_SAMPLES <= 0:\n'
            '        N_TEST_SAMPLES = 100\n'
            '    N_TEST_SAMPLES = min(N_TEST_SAMPLES, 100)'
        )
        if "VALIDATION_N_SAMPLES" in src:
            print(f"  = N_TEST_SAMPLES deja a 100 (cell {eval_idx})")
        elif OLD_BLOCK in src:
            cells[eval_idx]["source"] = src.replace(OLD_BLOCK, NEW_BLOCK, 1).splitlines(keepends=True)
            cells[eval_idx]["outputs"] = []
            cells[eval_idx]["execution_count"] = None
            n_changed += 1
            print(f"  ~ N_TEST_SAMPLES 10 -> 100 (cell {eval_idx})")
        else:
            print("  ! Bloc N_TEST_SAMPLES original non trouve - skip")

    # Sauvegarde
    if n_changed > 0:
        NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n{n_changed} modification(s) appliquee(s) a {NB.name}")
    return n_changed


def main() -> int:
    n = patch_notebook()
    return 0 if n >= 0 else 1


if __name__ == "__main__":
    sys.exit(main())
