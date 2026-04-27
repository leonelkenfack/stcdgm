"""
Unifie tous les téléchargements du notebook autour d'un ``DATA_ROOT`` unique :

- Sur Colab (avec Drive monté) : ``/content/drive/MyDrive/climate_data_data``
  → les données survivent à un kill runtime et ne sont téléchargées qu'une fois.
- Hors Colab : ``data/raw`` relatif au projet (comportement YAML par défaut).

Cellules touchées :
- Cell 15 : ajoute la détection DATA_ROOT en tête, override
  ``CONFIG.data.{lr_path, hr_path, static_path}`` + ``MEAN_PATH``/``STD_PATH``
  pour pointer sous DATA_ROOT, garde la fonction ``stream_download`` et les
  téléchargements HR/LR.
- Cell 16 : passe par ``stream_download`` au lieu d'``urlretrieve``, écrit
  dans ``DATA_ROOT / "test"``.

Idempotent : sentinelle ``# >>> DATA_ROOT_DRIVE``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL = "# >>> DATA_ROOT_DRIVE"

CELL_15_NEW = '''\
# >>> DATA_ROOT_DRIVE
# Téléchargement Zenodo + relocalisation Drive.
#
# DATA_ROOT survit à un kill runtime sur Colab : Drive est persistent. Hors
# Colab on garde ``data/raw`` (relatif au projet sur SSD ou local). Le YAML
# reste la source de vérité pour la structure ``train/``, ``test/``,
# ``static_predictors/``, ``normalization_coefs/`` — on remplace juste le
# préfixe ``data/raw``.
import os
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path


_ON_COLAB = "google.colab" in sys.modules or Path("/content").exists()
DATA_ROOT_LOCAL = Path("data/raw")
# Racine Drive unifiée — même nom que le dossier local du projet
# (``climate_data``), donc cohérente avec :
#   - le clone SSD du bootstrap : /content/climate_data
#   - le dossier local Desktop/climate_data
# Layout sur Drive :
#   /content/drive/MyDrive/climate_data/
#   ├── data/   ← ici (DATA_ROOT, ce que cette cellule télécharge)
#   └── ckpt/   ← cellule 44 (CONFIG.checkpoint.save_dir)
DATA_ROOT_DRIVE = Path("/content/drive/MyDrive/climate_data/data")

if _ON_COLAB and DATA_ROOT_DRIVE.parent.parent.exists():  # /content/drive/MyDrive/ existe
    DATA_ROOT = DATA_ROOT_DRIVE
    print(f"📁 DATA_ROOT = Drive ({DATA_ROOT})")
else:
    DATA_ROOT = DATA_ROOT_LOCAL
    print(f"📁 DATA_ROOT = local ({DATA_ROOT.resolve()})")

DATA_ROOT.mkdir(parents=True, exist_ok=True)

# Réécrit les chemins CONFIG pour pointer sous DATA_ROOT.
def _relocate(p):
    if not p:
        return p
    s = str(p)
    if s.startswith("data/raw/"):
        return str(DATA_ROOT / s[len("data/raw/"):])
    return s

for _key in ("lr_path", "hr_path", "static_path"):
    if CONFIG.data.get(_key):
        CONFIG.data[_key] = _relocate(CONFIG.data[_key])

LR_PATH = str(CONFIG.data.lr_path)
HR_PATH = str(CONFIG.data.hr_path)
STATIC_PATH = str(CONFIG.data.static_path) if CONFIG.data.get("static_path") else None
MEAN_PATH = str(DATA_ROOT / "normalization_coefs" / "mean_1974_2011.nc")
STD_PATH = str(DATA_ROOT / "normalization_coefs" / "std_1974_2011.nc")

URL_ZENODO_HR = "https://zenodo.org/records/10889046/files/pr_ACCESS-CM2_hist.nc?download=1"
URL_ZENODO_LR = "https://zenodo.org/records/10889046/files/predictor_ACCESS-CM2_hist.nc?download=1"


def stream_download(url: str, dest: str, *, retries: int = 5,
                    chunk_size: int = 1024 * 1024,
                    connect_timeout: int = 30,
                    read_timeout: int = 120) -> bool:
    """
    Streaming + atomique + reprise.

    Écrit dans ``dest + ".part"`` puis ``os.replace`` à la fin. Si un
    ``.part`` existe déjà, reprend via header HTTP Range. Backoff
    exponentiel sur 503/504/429/timeout. ``urlretrieve`` n'accepte PAS de
    timeout — c'est pour ça que l'ancienne version crashait silencieusement.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    for attempt in range(1, retries + 1):
        already = part.stat().st_size if part.exists() else 0
        req = urllib.request.Request(url)
        if already > 0:
            req.add_header("Range", f"bytes={already}-")
            print(f"  ↻ reprise à {already / 1e6:.1f} MB")

        try:
            with urllib.request.urlopen(req, timeout=connect_timeout) as resp:
                total = resp.length
                if total is None and resp.headers.get("Content-Length"):
                    total = int(resp.headers["Content-Length"])
                grand_total = (total + already) if total else None
                mode = "ab" if already > 0 else "wb"
                with open(part, mode) as f:
                    downloaded = already
                    last_log = time.time()
                    last_log_bytes = downloaded
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        now = time.time()
                        if now - last_log >= 5.0:
                            speed = (downloaded - last_log_bytes) / (now - last_log) / 1e6
                            if grand_total:
                                pct = 100.0 * downloaded / grand_total
                                eta = (grand_total - downloaded) / max(speed * 1e6, 1) if speed > 0 else float("inf")
                                print(f"    {downloaded / 1e6:7.1f} / {grand_total / 1e6:7.1f} MB "
                                      f"({pct:5.1f}%) — {speed:5.1f} MB/s — ETA {eta:5.0f}s")
                            else:
                                print(f"    {downloaded / 1e6:7.1f} MB — {speed:5.1f} MB/s")
                            last_log = now
                            last_log_bytes = downloaded
            os.replace(part, dest)
            print(f"  ✅ {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")
            return True

        except urllib.error.HTTPError as e:
            if e.code in (503, 504, 429):
                wait = min(60, 2 ** attempt)
                print(f"  ⚠️  HTTP {e.code} — retry dans {wait}s")
                time.sleep(wait)
            elif e.code == 416:
                print("  ↻ HTTP 416 — .part déjà complet, finalisation.")
                os.replace(part, dest)
                return True
            else:
                print(f"  ❌ HTTP {e.code}: {e.reason}")
                return False
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = min(60, 2 ** attempt)
            print(f"  ⚠️  réseau ({type(e).__name__}: {e}) — retry dans {wait}s")
            time.sleep(wait)
        except Exception as e:
            print(f"  ❌ {type(e).__name__}: {e}")
            return False

    print(f"  ❌ Échec après {retries} tentatives.")
    return False


# Téléchargements HR/LR train
if not Path(HR_PATH).exists():
    print(f"⏳ HR: {HR_PATH}")
    if not stream_download(URL_ZENODO_HR, HR_PATH):
        raise RuntimeError(f"Échec téléchargement HR depuis {URL_ZENODO_HR}")

if not Path(LR_PATH).exists():
    print(f"⏳ LR: {LR_PATH}")
    if not stream_download(URL_ZENODO_LR, LR_PATH):
        raise RuntimeError(f"Échec téléchargement LR depuis {URL_ZENODO_LR}")

for _path, _name in [(LR_PATH, "LR"), (HR_PATH, "HR")]:
    if _path and Path(_path).exists():
        print(f"✅ {_name}: {_path} ({Path(_path).stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"❌ {_name} manquant: {_path}")

def _exists_with_drive_sync(p: str) -> bool:
    """
    ``Path.exists()`` peut retourner False sur Drive (FUSE) si le dossier
    parent n'a pas encore été énuméré ou si le cache de listing est stale.
    On force une énumération via ``os.listdir`` puis on re-stat.
    """
    if not p:
        return False
    pth = Path(p)
    if pth.exists():
        return True
    parent = pth.parent
    if not parent.exists():
        return False
    try:
        os.listdir(parent)  # force FUSE Drive à énumérer
    except OSError:
        return False
    return pth.exists()

for _var, _name in [("STATIC_PATH", "Static"), ("MEAN_PATH", "Mean"), ("STD_PATH", "Std")]:
    _p = globals()[_var]
    if _exists_with_drive_sync(_p):
        print(f"✅ {_name}: {_p}")
    else:
        # Diagnostic : lister ce qui EST réellement présent dans le parent.
        _parent = Path(_p).parent if _p else None
        if _parent and _parent.exists():
            try:
                _seen = sorted(os.listdir(_parent))[:8]
            except OSError as _ose:
                _seen = f"(listing failed: {_ose})"
            print(f"⚠️  {_name} absent: {_p}")
            print(f"     parent {_parent} contient: {_seen}")
        else:
            print(f"⚠️  {_name} absent: {_p} (parent inexistant)")
        print(f"     → mis à None")
        globals()[_var] = None

if DATA_ROOT.exists():
    print(f"\\n📁 {DATA_ROOT}/ :")
    for _sub in sorted(DATA_ROOT.iterdir()):
        if _sub.is_dir():
            _files = list(_sub.glob("*.nc")) + list(_sub.glob("*.json"))
            print(f"   {_sub.name}/: {[f.name for f in _files[:5]]}{'...' if len(_files) > 5 else ''}")
        else:
            print(f"   {_sub.name}")
'''


CELL_16_NEW = '''\
# >>> DATA_ROOT_DRIVE
# Téléchargement des datasets de test (EC-Earth3, NorESM2-MM) via le
# ``stream_download`` défini en cellule 15 (atomique + reprise + timeout).
# Écrit sous ``DATA_ROOT / "test"`` — sur Drive en Colab, sinon local.
from pathlib import Path

TEST_ROOT = DATA_ROOT / "test"
TEST_ROOT.mkdir(parents=True, exist_ok=True)

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

print(f"📁 TEST_ROOT: {TEST_ROOT}")
for _filename, _url in URLS_TEST:
    _filepath = TEST_ROOT / _filename
    if _filepath.exists():
        print(f"✓ {_filename} déjà présent ({_filepath.stat().st_size / 1e6:.1f} MB)")
        continue
    print(f"⏳ {_filename}")
    if not stream_download(_url, str(_filepath)):
        raise RuntimeError(f"Échec téléchargement {_filename} depuis {_url}")
'''


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    target15 = target16 = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if SENTINEL in src and target15 is None and "URL_ZENODO_HR" in src:
            target15 = i
        elif SENTINEL in src and target16 is None and "URLS_TEST" in src:
            target16 = i
        elif "URL_ZENODO_HR" in src and target15 is None:
            target15 = i
        elif "URLS_TEST" in src and target16 is None:
            target16 = i

    if target15 is None or target16 is None:
        print(f"[ERROR] Cellules introuvables (15={target15}, 16={target16}).")
        return 2

    cells[target15]["source"] = CELL_15_NEW.splitlines(keepends=True)
    cells[target15]["outputs"] = []
    cells[target15]["execution_count"] = None
    cells[target16]["source"] = CELL_16_NEW.splitlines(keepends=True)
    cells[target16]["outputs"] = []
    cells[target16]["execution_count"] = None
    print(f"  ↪ Cell {target15} (HR/LR + DATA_ROOT) réécrite")
    print(f"  ↪ Cell {target16} (test data via stream_download) réécrite")

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Saved {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
