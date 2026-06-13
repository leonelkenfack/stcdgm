# >>> BS40_EXPORT_V4_ARTIFACTS — copier metriques vers results/ (post-run Colab)
import json as _json_bs40
import shutil as _shutil_bs40
from pathlib import Path as _Path_bs40

_results_dir = _Path_bs40("results")
_results_dir.mkdir(parents=True, exist_ok=True)
_ckpt_dir = _Path_bs40(str(CKPT_SAVE_DIR))
_fv = _ckpt_dir / "final_validation_metrics.json"
if _fv.exists():
    _shutil_bs40.copy2(_fv, _results_dir / "v4_metrics.json")
    print(f"Copied {_fv} -> results/v4_metrics.json")
else:
    print(f"No metrics at {_fv} — run FINAL_VALIDATION first")
_abl = _Path_bs40(f"ablation_suite_{RUN_VARIANT}.json")
if _abl.exists():
    _shutil_bs40.copy2(_abl, _results_dir / "v4_bs35_ablation.json")
    print(f"Copied {_abl} -> results/v4_bs35_ablation.json")
