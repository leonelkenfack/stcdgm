"""
BS31b + BS31c — config selector + Stage 2 force-restart flag.

BS31b — Cell 13 (config loader):
  Replace single-file load with base + optional override merge. Picks
  the YAML to use from the ``CONFIG_FILE`` env variable, defaulting to
  ``training_config_corrdiff_mini.yaml`` when present (the new
  CorrDiff-Mini override). Falls back gracefully to base
  ``training_config.yaml`` if the override is missing. Useful for both
  this run and any future ablation where we swap configs without
  copy-pasting the whole base.

BS31c — Cell 50 (BS18 RESUME block):
  Add an opt-in ``FORCE_S2_RESTART`` global flag. When set to True at
  cell-scope before running cell 50, BS18 RESUME ignores
  ``two_stage_state.stage2_epoch_done`` and forces the Stage 2 loop to
  start from epoch 0. Stage 1 resume + weight loading via BS20 are
  unaffected — encoder / rcn / regression_head still load from the
  checkpoint. Diffusion weights load via BS23 shape-mismatch tolerance
  (most tensors will be skipped with the new [64, 128, 128] UNet → de
  facto xavier-init for diffusion). The combination preserves Stage 1
  while restarting Stage 2 cleanly.

Idempotent — sentinels ``BS31b_CONFIG_SELECT`` and ``BS31c_FORCE_S2_RESTART``.
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


# =====================================================================
# Cell 13 — config selector (BS31b)
# =====================================================================

NEW_CELL_13 = '''# >>> BS31b_CONFIG_SELECT
# Configuration loader with optional override-merge.
#
# Pick which YAML governs the run:
#   1. CONFIG_FILE env var if set → that file under ``config/``
#   2. else, ``training_config_corrdiff_mini.yaml`` if present
#      (BS31b CorrDiff-Mini override, ~12-15M params Stage 2 UNet)
#   3. fallback ``training_config.yaml`` (base, ~1-2M Stage 2 UNet)
#
# When an override is selected, it is *merged on top of* the base, so
# only the diffs need to be in the override file. See
# ``architecture_journey.md §6`` for context.
import os
from pathlib import Path
from omegaconf import OmegaConf

_base_path = Path("config/training_config.yaml")
_default_override = "training_config_corrdiff_mini.yaml"
_override_name = os.environ.get("CONFIG_FILE", _default_override)
_override_path = Path("config") / _override_name

CONFIG = OmegaConf.load(_base_path)

if _override_path.exists() and _override_path.resolve() != _base_path.resolve():
    _override = OmegaConf.load(_override_path)
    CONFIG = OmegaConf.merge(CONFIG, _override)
    print(f"📂 Config base + override : training_config.yaml + {_override_path.name}")
elif _override_path.exists():
    print(f"📂 Config (single file)    : {_override_path.name}")
else:
    print(f"📂 Config (no override)    : training_config.yaml")
    print(f"   (override {_override_path.name} introuvable — utilisé si placé dans config/)")

print(f"  - Device: {CONFIG.training.device}")
print(f"  - Epochs: {CONFIG.training.epochs}")
print(f"  - Lambda gen: {CONFIG.loss.lambda_gen}, Beta rec: {CONFIG.loss.beta_rec}, Gamma DAG: {CONFIG.loss.gamma_dag}")
print(f"  - Stage 2 UNet block_out_channels: {list(CONFIG.diffusion.unet_kwargs.block_out_channels)}")
'''


def patch_cell_13() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: 'OmegaConf.load("config/training_config.yaml")' in s
                     and "Config chargée depuis" in s)
    if idx is None:
        # Maybe already patched
        idx2 = _find_cell(cells, lambda s: "BS31b_CONFIG_SELECT" in s)
        if idx2 is not None:
            print(f"  = cell {idx2} already patched (BS31b)")
            return 0
        print("  ! cell 13 (config loader) not found")
        return 0
    cells[idx]["source"] = NEW_CELL_13.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {idx}: config loader → base + override merge")
    return 1


# =====================================================================
# Cell 50 — FORCE_S2_RESTART flag (BS31c)
# =====================================================================

OLD_S2_RESUME = '''        _s2_resume_from = int(_ts_state.get("stage2_epoch_done", 0))'''

NEW_S2_RESUME = '''        _s2_resume_from = int(_ts_state.get("stage2_epoch_done", 0))
        # >>> BS31c_FORCE_S2_RESTART
        # When the user has changed the diffusion architecture (e.g. via
        # CorrDiff-Mini override BS31b), Stage 2 weights are no longer
        # compatible — BS23 will skip them as shape mismatches → de-facto
        # xavier init. In that case the Stage 2 epoch counter from the
        # old run is meaningless; force it back to 0 so we retrain from
        # scratch. Set ``FORCE_S2_RESTART = True`` at cell scope before
        # running this cell to activate.
        if globals().get("FORCE_S2_RESTART", False):
            if _s2_resume_from > 0:
                print(f"   ⚡ FORCE_S2_RESTART=True — skipping Stage 2 resume "
                      f"(was at epoch {_s2_resume_from}/{int(_ts_state.get('stage2_epoch_done', 0))}).")
            _s2_resume_from = 0
            _skip_calibration = False  # recalibrate σ_data on the new residual scale
            _saved_sigma_data = None
            _saved_sigma_min = None
            _skip_ablation = False     # re-run O3 ablation with restored Stage 1 weights'''


def patch_cell_50() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "TWO_STAGE_TRAINING_LOOP" in s and "for s2_epoch" in s)
    if idx is None:
        print("  ! cell 50 not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS31c_FORCE_S2_RESTART" in src:
        print(f"  = cell {idx} already patched (BS31c)")
        return 0
    if OLD_S2_RESUME not in src:
        print(f"  ! cell {idx}: BS18 _s2_resume_from pattern not found")
        return 0
    src = src.replace(OLD_S2_RESUME, NEW_S2_RESUME, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {idx}: BS31c FORCE_S2_RESTART flag inserted in BS18 RESUME")
    return 1


def main() -> int:
    print("=== BS31b : config selector (cell 13) ===")
    n1 = patch_cell_13()
    print("\n=== BS31c : FORCE_S2_RESTART flag (cell 50) ===")
    n2 = patch_cell_50()
    print(f"\n{n1 + n2} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
