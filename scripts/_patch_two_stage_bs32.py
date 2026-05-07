"""
BS32 — Drive → local SSD copy for NetCDF training files.

Stage 2 was running at 20 sec/batch on A100 80GB. Bench analysis
(``stage2_bottleneck_analysis.md``) identified the dominant cost as
NetCDF mmap latency from Google Drive (~8-15 sec/batch). Copying the
training .nc files to ``/content/data_local`` and overriding
``CONFIG.data.*_path`` cuts this to <1 sec/batch — typically 2-3×
speedup overall.

Patch site: cell 16 (DATA_ROOT_DRIVE) — extends the existing logic.
After ``DATA_ROOT`` is decided (Drive vs local), if it's Drive and
we're on Colab, copy the *training* .nc files (LR + HR + static +
normalization coefs) to ``/content/data_local`` and switch
``DATA_ROOT`` to that path before the CONFIG path relocation. Test
files (used only at evaluation) are NOT copied — they stay on Drive
to save SSD space.

Idempotent: the copy is skipped if the destination file already
exists. Sentinel ``BS32_DRIVE_TO_SSD``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


# Anchor: the lines that decide DATA_ROOT. We insert the copy block
# right after.
OLD_ANCHOR = '''if _ON_COLAB and DATA_ROOT_DRIVE.parent.parent.exists():  # /content/drive/MyDrive/ existe
    DATA_ROOT = DATA_ROOT_DRIVE
    print(f"📁 DATA_ROOT = Drive ({DATA_ROOT})")
else:
    DATA_ROOT = DATA_ROOT_LOCAL
    print(f"📁 DATA_ROOT = local ({DATA_ROOT.resolve()})")

DATA_ROOT.mkdir(parents=True, exist_ok=True)'''


NEW_ANCHOR = '''if _ON_COLAB and DATA_ROOT_DRIVE.parent.parent.exists():  # /content/drive/MyDrive/ existe
    DATA_ROOT = DATA_ROOT_DRIVE
    print(f"📁 DATA_ROOT = Drive ({DATA_ROOT})")
else:
    DATA_ROOT = DATA_ROOT_LOCAL
    print(f"📁 DATA_ROOT = local ({DATA_ROOT.resolve()})")

DATA_ROOT.mkdir(parents=True, exist_ok=True)

# >>> BS32_DRIVE_TO_SSD
# On Colab, Drive NetCDF mmap latency dominates training time
# (~8-15 sec/batch on A100). Copy the training .nc files to local
# SSD once per session and switch DATA_ROOT to point there. Test
# files stay on Drive (used only at evaluation, infrequent reads).
# See ``stage2_bottleneck_analysis.md`` for the bench-grounded
# rationale and expected speedup.
import shutil as _shutil

_DATA_ROOT_LOCAL_SSD = Path("/content/data_local")
_BS32_ENABLED = bool(globals().get("DATA_LOCAL_SSD", True))   # opt-out via DATA_LOCAL_SSD = False

if (_ON_COLAB and _BS32_ENABLED and DATA_ROOT == DATA_ROOT_DRIVE):
    # Files needed for training. Test files are deliberately excluded.
    _files_to_copy = [
        # train/
        ("train/predictor_ACCESS-CM2_hist.nc",   "predictor_ACCESS-CM2_hist.nc"),
        ("train/pr_ACCESS-CM2_hist.nc",          "pr_ACCESS-CM2_hist.nc"),
        # static_predictors/
        ("static_predictors/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc",
         "ERA5_eval_ccam_12km.198110_NZ_Invariant.nc"),
        # normalization_coefs/
        ("normalization_coefs/mean_1974_2011.nc", "mean_1974_2011.nc"),
        ("normalization_coefs/std_1974_2011.nc",  "std_1974_2011.nc"),
    ]
    _ssd_train = _DATA_ROOT_LOCAL_SSD / "train"
    _ssd_static = _DATA_ROOT_LOCAL_SSD / "static_predictors"
    _ssd_norm = _DATA_ROOT_LOCAL_SSD / "normalization_coefs"
    for _d in (_ssd_train, _ssd_static, _ssd_norm):
        _d.mkdir(parents=True, exist_ok=True)

    import time as _time
    _t_total = _time.time()
    _bytes_copied = 0
    for _rel, _name in _files_to_copy:
        _src = DATA_ROOT_DRIVE / _rel
        if "train/" in _rel:
            _dst = _ssd_train / _name
        elif "static_predictors/" in _rel:
            _dst = _ssd_static / _name
        else:
            _dst = _ssd_norm / _name

        if not _src.exists():
            print(f"   ⚠️  source absente, skip: {_src}")
            continue
        if _dst.exists() and _dst.stat().st_size == _src.stat().st_size:
            print(f"   ✓ déjà sur SSD: {_dst.name} ({_dst.stat().st_size/1e6:.0f} MB)")
            continue
        _t0 = _time.time()
        print(f"   ⏳ copie {_src.name}...", flush=True)
        _shutil.copy2(_src, _dst)
        _dt = _time.time() - _t0
        _bytes_copied += _dst.stat().st_size
        print(f"   ✓ {_dst.name} ({_dst.stat().st_size/1e6:.0f} MB en {_dt:.1f} s, "
              f"{_dst.stat().st_size/1e6/_dt:.0f} MB/s)")

    _t_total = _time.time() - _t_total
    if _bytes_copied > 0:
        print(f"📦 BS32 SSD copy: {_bytes_copied/1e9:.2f} GB en {_t_total:.1f} s")
    print(f"📁 DATA_ROOT redirigé vers SSD local : {_DATA_ROOT_LOCAL_SSD}")
    DATA_ROOT = _DATA_ROOT_LOCAL_SSD
elif _ON_COLAB and not _BS32_ENABLED:
    print("⚠️  BS32 désactivé (DATA_LOCAL_SSD=False) — lecture directe Drive (lent)")'''


def patch_cell_16() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "DATA_ROOT_DRIVE = Path" in s
                     and "DATA_ROOT.mkdir" in s)
    if idx is None:
        print("  ! cell 16 (DATA_ROOT_DRIVE) not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS32_DRIVE_TO_SSD" in src:
        print(f"  = cell {idx} already patched (BS32)")
        return 0
    if OLD_ANCHOR not in src:
        print(f"  ! cell {idx}: anchor pattern not found")
        return 0
    src = src.replace(OLD_ANCHOR, NEW_ANCHOR, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {idx}: BS32 Drive → SSD copy block inserted")
    return 1


def main() -> int:
    print("=== BS32 : Drive → local SSD copy for NetCDF training files ===")
    n = patch_cell_16()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
