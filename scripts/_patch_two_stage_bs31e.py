"""
BS31e — add Pearson correlation to FINAL_VALIDATION metrics.

The paper target ``Corr ≈ 0.896`` (Bomgni et al. 2026 §sec:metrics)
isn't measured anywhere in our pipeline. RMSE / F1 / RAPSD give a
partial picture but the per-sample prediction-target correlation is
the canonical skill score for downscaling.

Two flavours computed:
- ``corr_global`` : Pearson over *all* valid pixels concatenated
  (single scalar). Sensitive to the global pixel distribution.
- ``corr_per_sample`` : Pearson per sample then averaged. Closer
  to the paper's ``Corr`` (which is reported per sample).

Both saved to JSON. Idempotent — sentinel ``BS31e_PEARSON_CORR``.
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


# Insert correlation computation after the shortcut diagnostic block,
# before the F1 try/except.
OLD_PRE_F1 = '''_f1 = {}
try:
    _f1 = compute_f1_extremes(_pred_clean, _targ_clean, threshold_percentiles=[95.0, 99.0])'''


NEW_PRE_F1 = '''# >>> BS31e_PEARSON_CORR — paper metric (Bomgni et al. 2026 target ~0.896)
def _pearson(a, b, eps=1e-8):
    """Pearson correlation between two flat tensors. Assumes inputs already finite."""
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    den = torch.sqrt((a_c * a_c).sum() * (b_c * b_c).sum() + eps)
    return float((num / den).item())

# Global Pearson over all valid pixels concatenated.
_corr_global = float("nan")
_corr_per_sample = float("nan")
_corr_per_sample_list = []
try:
    _p_flat = _pred_mean[_valid]
    _t_flat = _targets[_valid]
    if _p_flat.numel() > 1 and _t_flat.numel() > 1:
        _corr_global = _pearson(_p_flat, _t_flat)
    # Per-sample correlation (matches paper convention).
    for _i in range(_pred_mean.shape[0]):
        _vi = _valid[_i]
        if _vi.sum() < 2:
            continue
        _pi = _pred_mean[_i][_vi]
        _ti = _targets[_i][_vi]
        _c = _pearson(_pi, _ti)
        if _c == _c:  # filter NaN
            _corr_per_sample_list.append(_c)
    if _corr_per_sample_list:
        _corr_per_sample = float(np.mean(_corr_per_sample_list))
except Exception as _e:
    print(f"⚠️  Pearson correlation a échoué : {_e}")

_f1 = {}
try:
    _f1 = compute_f1_extremes(_pred_clean, _targ_clean, threshold_percentiles=[95.0, 99.0])'''


# Print block — append correlation just before "F1 extremes" loop.
OLD_PRINT_F1 = '''for _k, _v in _f1.items():
    print(f"  {_k:<22}: {_v:.4f}")'''


NEW_PRINT_F1 = '''# BS31e_PEARSON_CORR print
print(f"  Pearson Corr (global) : {_corr_global:.4f}")
print(f"  Pearson Corr (per-sample avg) : {_corr_per_sample:.4f} "
      f"(over {len(_corr_per_sample_list)} samples)  ← paper target ~0.896")
for _k, _v in _f1.items():
    print(f"  {_k:<22}: {_v:.4f}")'''


# JSON payload — inject correlation field.
OLD_JSON = '''    "f1_extremes": _f1,
    "rapsd_distance": _rapsd_d,'''


NEW_JSON = '''    "f1_extremes": _f1,
    "pearson_corr": {
        "global": _corr_global,
        "per_sample_avg": _corr_per_sample,
        "per_sample_n": len(_corr_per_sample_list),
        "per_sample_list": _corr_per_sample_list[:64],  # cap for JSON size
    },
    "rapsd_distance": _rapsd_d,'''


def patch_cell_60() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "FINAL_VALIDATION (BS30" in s)
    if idx is None:
        print("  ! FINAL_VALIDATION cell not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS31e_PEARSON_CORR" in src:
        print(f"  = cell {idx} already patched (BS31e)")
        return 0
    n = 0
    for old, new, label in (
        (OLD_PRE_F1, NEW_PRE_F1, "Pearson computation block"),
        (OLD_PRINT_F1, NEW_PRINT_F1, "Pearson print"),
        (OLD_JSON, NEW_JSON, "Pearson JSON field"),
    ):
        if old in src:
            src = src.replace(old, new, 1)
            n += 1
            print(f"  ~ cell {idx}: {label}")
        else:
            print(f"  ! cell {idx}: pattern '{label}' not found")
    if n > 0:
        cells[idx]["source"] = src.splitlines(keepends=True)
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def main() -> int:
    print("=== BS31e : Pearson correlation in FINAL_VALIDATION ===")
    n = patch_cell_60()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
