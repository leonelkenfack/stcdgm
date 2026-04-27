"""
Replace the broken Zenodo download cell.

Bug: ``urllib.request.urlretrieve(url, dest, timeout=120)`` raises
``TypeError: urlretrieve() got an unexpected keyword argument 'timeout'``.
The ``except Exception`` block swallows it as a generic "unexpected error",
retries 5× on the same TypeError, then raises ``RuntimeError`` — looking
"silent" but crashing.

Fix:
- Streaming download via ``urllib.request.urlopen(req, timeout=...)`` (no
  third-party deps needed; ``requests`` may not be in Colab's pre-install).
- Atomic write : ``.part`` file + ``os.replace`` so kill mid-download leaves
  no corrupt .nc on Drive/disk.
- Resumable: HTTP ``Range`` header continues a partial ``.part`` if present.
- Live progress (MB / %) + speed every ~5 s.
- Exponential backoff on transient HTTP errors.

Idempotent: sentinel ``# >>> ZENODO_DL_FIX``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL = "# >>> ZENODO_DL_FIX"

NEW_CELL = '''\
# >>> ZENODO_DL_FIX
# Téléchargement Zenodo robuste : streaming, atomique, reprise après coupure,
# progress live. Remplace l'ancien ``urlretrieve(..., timeout=120)`` (crash :
# ``urlretrieve()`` n'accepte pas de ``timeout``).
import os
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── Chemins (lus depuis CONFIG) ─────────────────────────────────────────
DATA_ROOT = Path("data/raw")
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
    Téléchargement streaming + atomique + reprise.

    Écrit dans ``dest + ".part"`` puis ``os.replace`` à la fin → garantit
    qu'un kill au milieu ne laisse PAS un .nc partiel piégeux à côté.
    Si un ``.part`` existe déjà, reprend via header HTTP Range. Backoff
    exponentiel sur 503/504/timeout.
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
            # ``urlopen`` accepte timeout — c'est ``urlretrieve`` qui ne l'accepte pas.
            with urllib.request.urlopen(req, timeout=connect_timeout) as resp:
                total = resp.length  # bytes restants à partir de la position
                if total is None and resp.headers.get("Content-Length"):
                    total = int(resp.headers["Content-Length"])
                grand_total = (total + already) if total else None

                mode = "ab" if already > 0 else "wb"
                with open(part, mode) as f:
                    downloaded = already
                    last_log = time.time()
                    last_log_bytes = downloaded
                    while True:
                        # ``read`` accepte une taille mais pas de timeout par appel ;
                        # le ``timeout=connect_timeout`` ci-dessus s'applique au socket.
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
                # Range Not Satisfiable → .part probablement complet, on tente le replace.
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
            # Vraie erreur inattendue : on l'affiche et on sort (pas de retry à l'aveugle).
            print(f"  ❌ {type(e).__name__}: {e}")
            return False

    print(f"  ❌ Échec après {retries} tentatives.")
    return False


# ── Téléchargement ────────────────────────────────────────────────────
if not Path(HR_PATH).exists():
    print(f"⏳ HR: {HR_PATH}")
    if not stream_download(URL_ZENODO_HR, HR_PATH):
        raise RuntimeError(f"Échec téléchargement HR depuis {URL_ZENODO_HR}")

if not Path(LR_PATH).exists():
    print(f"⏳ LR: {LR_PATH}")
    if not stream_download(URL_ZENODO_LR, LR_PATH):
        raise RuntimeError(f"Échec téléchargement LR depuis {URL_ZENODO_LR}")

# ── Vérifications ─────────────────────────────────────────────────────
for path, name in [(LR_PATH, "LR"), (HR_PATH, "HR")]:
    if path and Path(path).exists():
        print(f"✅ {name}: {path} ({Path(path).stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"❌ {name} manquant: {path}")

for path_var, name in [("STATIC_PATH", "Static"), ("MEAN_PATH", "Mean"), ("STD_PATH", "Std")]:
    p = globals()[path_var]
    if p and Path(p).exists():
        print(f"✅ {name}: {p}")
    else:
        print(f"⚠️  {name} absent: {p} → mis à None")
        globals()[path_var] = None

# Structure data/raw/
data_dir = Path("data/raw")
if data_dir.exists():
    print("\\n📁 data/raw/ :")
    for sub in sorted(data_dir.iterdir()):
        if sub.is_dir():
            files = list(sub.glob("*.nc")) + list(sub.glob("*.json"))
            print(f"   {sub.name}/: {[f.name for f in files[:5]]}{'...' if len(files) > 5 else ''}")
        else:
            print(f"   {sub.name}")
else:
    print("\\n⚠️  data/raw/ inexistant — les téléchargements ont peut-être échoué.")
'''


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    target = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if SENTINEL in src:
            target = i
            print(f"✓ Cellule {i} déjà patchée — réécriture pour synchro.")
            break
        if "URL_ZENODO_HR" in src and ("urlretrieve" in src or "download_with_retries" in src):
            target = i
            print(f"  ↪ Remplace cellule download Zenodo {i}.")
            break

    if target is None:
        print("[ERROR] Cellule download Zenodo introuvable.")
        return 2

    nb["cells"][target]["source"] = NEW_CELL.splitlines(keepends=True)
    nb["cells"][target]["outputs"] = []
    nb["cells"][target]["execution_count"] = None

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Saved {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
