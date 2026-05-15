"""Patch st_cdgm_training_evaluation.ipynb for V4 default runbook."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "st_cdgm_training_evaluation.ipynb"


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

    config_old = (
        '# BS37 Sprint switcher — pour basculer entre V2 / V3 / V4 sans toucher\n'
        '# au notebook, definir avant ce cell :\n'
        '#   import os ; os.environ[\'CONFIG_FILE\'] = \'training_config_v3.yaml\'\n'
        '# (ou \'training_config_v4_corrdiff_large.yaml\'). Defaut = V2 corrdiff_normal.\n'
        '_default_override = "training_config_corrdiff_normal.yaml"  # V2-eval+ (Sprint A)\n'
    )
    config_new = (
        '# BS40 V4 RUNBOOK — version finale (defaut repo = V4 CorrDiff Large).\n'
        '# Colab A100 : avant two-stage, definir aussi :\n'
        '#   FORCE_S2_RESTART = False\n'
        '#   BS32B_CACHE_PATH = \'/content/stage1_cache_v4.pt\'\n'
        '# Variantes : training_config_corrdiff_normal.yaml (V2), training_config_v3.yaml (V3).\n'
        '# Override : os.environ[\'CONFIG_FILE\'] = \'training_config_v3.yaml\'\n'
        '_default_override = "training_config_v4_corrdiff_large.yaml"\n'
    )

    bs32b_old = (
        '#   BS32B_CACHE_PATH = \'/content/stage1_cache_v3.pt\'   # Sprint B (V3)\n'
        '#   BS32B_CACHE_PATH = \'/content/stage1_cache_v4.pt\'   # Sprint C (V4)\n'
        '# (definir cette variable globale AVANT d\'executer cette cellule)\n'
        '_BS32B_CACHE_PATH = Path(globals().get("BS32B_CACHE_PATH", "/content/stage1_cache_v2.pt"))\n'
    )
    bs32b_new = (
        '#   BS32B_CACHE_PATH = \'/content/stage1_cache_v3.pt\'   # V3\n'
        '#   BS32B_CACHE_PATH = \'/content/stage1_cache_v4.pt\'   # V4\n'
        '# (definir cette variable globale AVANT d\'executer cette cellule)\n'
        '_bs32b_default = "/content/stage1_cache_v4.pt"\n'
        'try:\n'
        '    _ov = str(getattr(CONFIG, "checkpoint", {}).get("save_dir", ""))\n'
        '    if "v3" in _ov:\n'
        '        _bs32b_default = "/content/stage1_cache_v3.pt"\n'
        '    elif "v2" in _ov or "corrdiff_normal" in _ov:\n'
        '        _bs32b_default = "/content/stage1_cache_v2.pt"\n'
        'except Exception:\n'
        '    pass\n'
        '_BS32B_CACHE_PATH = Path(globals().get("BS32B_CACHE_PATH", _bs32b_default))\n'
    )

    export_cell = (
        '# >>> BS40_EXPORT_V4_ARTIFACTS — copier metriques vers results/ (post-run Colab)\n'
        'import json as _json_bs40\n'
        'import shutil as _shutil_bs40\n'
        'from pathlib import Path as _Path_bs40\n'
        '\n'
        '_results_dir = _Path_bs40("results")\n'
        '_results_dir.mkdir(parents=True, exist_ok=True)\n'
        '_ckpt_dir = _Path_bs40(str(CKPT_SAVE_DIR))\n'
        '_fv = _ckpt_dir / "final_validation_metrics.json"\n'
        'if _fv.exists():\n'
        '    _shutil_bs40.copy2(_fv, _results_dir / "v4_metrics.json")\n'
        '    print(f"Copied {_fv} -> results/v4_metrics.json")\n'
        'else:\n'
        '    print(f"No metrics at {_fv} — run FINAL_VALIDATION first")\n'
        '_abl = _Path_bs40(f"ablation_suite_{RUN_VARIANT}.json")\n'
        'if _abl.exists():\n'
        '    _shutil_bs40.copy2(_abl, _results_dir / "v4_bs35_ablation.json")\n'
        '    print(f"Copied {_abl} -> results/v4_bs35_ablation.json")\n'
    )

    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", []))
        if config_old in src:
            src = src.replace(config_old, config_new)
            cell["source"] = _to_lines(src)
            print(f"patched config cell {i}")
        if bs32b_old in src:
            src = src.replace(bs32b_old, bs32b_new)
            cell["source"] = _to_lines(src)
            print(f"patched BS32b cell {i}")

    # Append export cell if missing
    if not any("BS40_EXPORT_V4_ARTIFACTS" in "".join(c.get("source", [])) for c in nb["cells"]):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": _to_lines(export_cell),
        })
        print("added BS40 export cell")

    NB_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", NB_PATH)


def _to_lines(src: str) -> list[str]:
    lines = src.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


if __name__ == "__main__":
    main()
