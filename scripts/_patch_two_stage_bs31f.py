"""
BS31f — fix metric scope in FINAL_VALIDATION (compare full prediction
``μ_HR + δ̂`` vs target ``HR_residual``, not raw residual ``δ̂`` alone).

The bug
-------
``cell 60 (BS30)`` was computing all pixel-level metrics (RMSE, MAE,
F1 extremes, Pearson Corr, RAPSD) between ``_pred_mean`` and
``_targets`` where:

* ``_pred_mean`` = output of ``diffusion.sample()`` = predicted residual
  ``δ̂`` (in causal_concat mode the diffusion learns to predict
  ``δ_target = HR_residual - μ_HR``, NOT HR_residual directly).
* ``_targets`` = ``batch["residual"][-1]`` = HR_residual = HR - baseline.

So we were comparing ``δ̂`` to ``HR_residual`` while the actual
reconstruction is ``μ_HR + δ̂ ≈ HR_residual``. Almost all the signal
lives in ``μ_HR`` (Stage 1 carries the deterministic mean); ``δ̂``
captures only the high-frequency / extreme residual. Comparing them
directly gives near-zero Pearson by construction.

Symptom: Pearson Corr 0.036 with paper target ~0.896 — looked
catastrophic but the metric was wrong.

The fix
-------
For all pixel/spectral/correlation metrics in ``cell 60``:

    _pred_full = _pred_mean + _mu_HR_concat   # full HR_residual prediction

then RMSE, MAE, F1, Corr, RAPSD use ``_pred_full`` against ``_targets``.

The shortcut diagnostic (BS31a) keeps using ``_pred_mean`` raw because
it specifically asks "is δ̂ collapsing to identity-on-μ_HR?" — that
question requires inspecting the raw residual, not the full
reconstruction.

Also save a JSON field ``metrics_scope`` distinguishing the two so
future eval runs are unambiguous.

Idempotent — sentinel ``BS31f_FULL_PREDICTION``.
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


# Anchor: the ``_pred_clean`` / ``_targ_clean`` lines that prepare
# tensors for downstream metrics. We insert ``_pred_full`` right after.
OLD_PRED_CLEAN = '''_pred_mean = torch.cat(_all_means, dim=0).cpu()
_pred_std = torch.cat(_all_stds, dim=0).cpu()
_targets = torch.cat(_all_targets, dim=0).cpu()
_valid = torch.isfinite(_targets)
_pred_clean = torch.where(_valid, _pred_mean, torch.zeros_like(_pred_mean))
_targ_clean = torch.where(_valid, _targets, torch.zeros_like(_targets))'''


NEW_PRED_CLEAN = '''_pred_mean = torch.cat(_all_means, dim=0).cpu()
_pred_std = torch.cat(_all_stds, dim=0).cpu()
_targets = torch.cat(_all_targets, dim=0).cpu()
_valid = torch.isfinite(_targets)

# >>> BS31f_FULL_PREDICTION
# In causal_concat mode the diffusion outputs δ̂ (the residual on top of
# μ_HR), but ``_targets`` is HR_residual = HR - baseline. To measure the
# end-to-end reconstruction skill, build ``_pred_full = μ_HR + δ̂`` and
# use it for all pixel-level metrics. Keep ``_pred_mean`` raw for the
# shortcut diagnostic (which inspects δ̂ specifically).
if _all_mu_HR:
    _mu_concat_eval = torch.cat(_all_mu_HR, dim=0).cpu()
    if _mu_concat_eval.shape == _pred_mean.shape:
        _pred_full = _pred_mean + _mu_concat_eval
        _metrics_scope = "full_prediction (μ_HR + δ̂)"
    else:
        print(f"   ⚠️  BS31f : μ_HR shape mismatch — fallback raw δ̂ for metrics")
        _pred_full = _pred_mean
        _metrics_scope = "delta_only (BS31f fallback)"
else:
    _pred_full = _pred_mean
    _metrics_scope = "delta_only (no μ_HR cached)"
print(f"   📐 metrics scope : {_metrics_scope}")

_pred_clean = torch.where(_valid, _pred_full, torch.zeros_like(_pred_full))
_targ_clean = torch.where(_valid, _targets, torch.zeros_like(_targets))'''


# ---------------------------------------------------------------------
# Patch 2 — Pearson uses _pred_full, not _pred_mean
# ---------------------------------------------------------------------

OLD_CORR_PER_SAMPLE = '''try:
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
        _corr_per_sample = float(np.mean(_corr_per_sample_list))'''


NEW_CORR_PER_SAMPLE = '''try:
    # BS31f : Pearson computed on the FULL prediction (μ_HR + δ̂),
    # not on the raw residual δ̂. See sentinel BS31f_FULL_PREDICTION.
    _p_flat = _pred_full[_valid]
    _t_flat = _targets[_valid]
    if _p_flat.numel() > 1 and _t_flat.numel() > 1:
        _corr_global = _pearson(_p_flat, _t_flat)
    # Per-sample correlation (matches paper convention).
    for _i in range(_pred_full.shape[0]):
        _vi = _valid[_i]
        if _vi.sum() < 2:
            continue
        _pi = _pred_full[_i][_vi]
        _ti = _targets[_i][_vi]
        _c = _pearson(_pi, _ti)
        if _c == _c:  # filter NaN
            _corr_per_sample_list.append(_c)
    if _corr_per_sample_list:
        _corr_per_sample = float(np.mean(_corr_per_sample_list))'''


# ---------------------------------------------------------------------
# Patch 3 — JSON includes the scope tag
# ---------------------------------------------------------------------

OLD_JSON_EVAL = '''    "n_test_batches": len(_all_targets),
    "k_samples": K_SAMPLES,'''

NEW_JSON_EVAL = '''    "n_test_batches": len(_all_targets),
    "k_samples": K_SAMPLES,
    "metrics_scope": _metrics_scope,'''


def patch_cell_60() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "FINAL_VALIDATION (BS30" in s)
    if idx is None:
        print("  ! FINAL_VALIDATION cell not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS31f_FULL_PREDICTION" in src:
        print(f"  = cell {idx} already patched (BS31f)")
        return 0

    n = 0
    for old, new, label in (
        (OLD_PRED_CLEAN, NEW_PRED_CLEAN, "pred_full computation block"),
        (OLD_CORR_PER_SAMPLE, NEW_CORR_PER_SAMPLE, "Pearson uses _pred_full"),
        (OLD_JSON_EVAL, NEW_JSON_EVAL, "JSON scope tag"),
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
    print("=== BS31f : metrics on full prediction (μ_HR + δ̂), not raw δ̂ ===")
    n = patch_cell_60()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
