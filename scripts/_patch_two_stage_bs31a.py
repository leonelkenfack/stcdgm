"""
BS31a — diagnostic shortcut learning in FINAL_VALIDATION (cell 60).

Adds explicit measurement of the collapse-to-identity-on-μ_HR
hypothesis. After Stage 2 with [32, 64] UNet, the FINAL_VALIDATION
metrics show μ_HR ablation Δ/signal=96% but F1 extremes catastrophic
and RAPSD distance huge — consistent with the UNet learning
``δ̂ ≈ 0`` (output recopies μ_HR instead of refining it).

This patch :
- Accumulates ``mu_HR`` across the eval loop (alongside outputs/targets)
- After the loop, computes :
    ‖output‖             — magnitude de la sortie diffusion
    ‖μ_HR‖              — magnitude de la prédiction Stage 1
    ‖target‖            — magnitude de la cible HR (résiduel)
    ‖output − μ_HR‖     — différence diffusion vs Stage 1
    ‖target − μ_HR‖     — vrai résiduel à apprendre
    shortcut_ratio = ‖output − μ_HR‖ / ‖output‖
- Prints + saves to JSON under ``shortcut_diagnostic``.

Decision criterion (per chainlogic STEP 2) :
- shortcut_ratio < 0.10 → SHORTCUT CONFIRMED → proceed to STEP 3 (resize UNet)
- shortcut_ratio > 0.30 → hypothesis refuted → REPLAN (other root cause)
- 0.10 ≤ ratio < 0.30 → ambiguous, look at other signals

Idempotent — sentinel ``BS31_DIAG_SHORTCUT``.
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


# ──────────────────────────────────────────────────────────────────────
# Patch 1: accumulate mu_HR + target during the eval loop
# ──────────────────────────────────────────────────────────────────────

OLD_INIT = '''_all_means = []
_all_stds = []
_all_targets = []
_intervention = []'''

NEW_INIT = '''_all_means = []
_all_stds = []
_all_targets = []
_intervention = []
# >>> BS31_DIAG_SHORTCUT
_all_mu_HR = []  # accumulate Stage-1 μ_HR per batch for shortcut probe'''


OLD_APPEND = '''            _all_means.append(_samples_k.mean(dim=0))
            _all_stds.append(_samples_k.std(dim=0))
            _all_targets.append(_target)'''

NEW_APPEND = '''            _all_means.append(_samples_k.mean(dim=0))
            _all_stds.append(_samples_k.std(dim=0))
            _all_targets.append(_target)
            # BS31_DIAG_SHORTCUT
            if _causal_concat and _mu_HR is not None:
                _all_mu_HR.append(_mu_HR.detach())'''


# ──────────────────────────────────────────────────────────────────────
# Patch 2: shortcut diagnostic computation + print + JSON field
# Insert AFTER ``_pred_clean / _targ_clean`` are defined and BEFORE the
# print block.
# ──────────────────────────────────────────────────────────────────────

OLD_BEFORE_F1 = '''_f1 = {}
try:
    _f1 = compute_f1_extremes(_pred_clean, _targ_clean, threshold_percentiles=[95.0, 99.0])'''

NEW_BEFORE_F1 = '''# >>> BS31_DIAG_SHORTCUT — empirical residual-collapse probe
_shortcut = {
    "norm_output": None,
    "norm_mu_HR": None,
    "norm_target": None,
    "norm_output_minus_mu_HR": None,
    "norm_target_minus_mu_HR": None,
    "shortcut_ratio": None,
    "verdict": "N/A",
}
if _all_mu_HR:
    _mu_concat = torch.cat(_all_mu_HR, dim=0).cpu()
    # Align shapes: _pred_mean and _targets are already CPU
    if _mu_concat.shape == _pred_mean.shape:
        _valid_mu = _valid & torch.isfinite(_mu_concat)
        _out = _pred_mean[_valid_mu]
        _mu = _mu_concat[_valid_mu]
        _tg = _targets[_valid_mu]
        _shortcut["norm_output"] = float(_out.abs().mean().item())
        _shortcut["norm_mu_HR"] = float(_mu.abs().mean().item())
        _shortcut["norm_target"] = float(_tg.abs().mean().item())
        _shortcut["norm_output_minus_mu_HR"] = float((_out - _mu).abs().mean().item())
        _shortcut["norm_target_minus_mu_HR"] = float((_tg - _mu).abs().mean().item())
        _ratio = (_shortcut["norm_output_minus_mu_HR"]
                  / max(_shortcut["norm_output"], 1e-12))
        _shortcut["shortcut_ratio"] = float(_ratio)
        if _ratio < 0.10:
            _shortcut["verdict"] = "SHORTCUT_CONFIRMED"
        elif _ratio < 0.30:
            _shortcut["verdict"] = "AMBIGUOUS"
        else:
            _shortcut["verdict"] = "REFINEMENT_OK"
    else:
        print(f"   ⚠️  shortcut diag : shape mismatch μ_HR{tuple(_mu_concat.shape)} vs pred{tuple(_pred_mean.shape)}")

_f1 = {}
try:
    _f1 = compute_f1_extremes(_pred_clean, _targ_clean, threshold_percentiles=[95.0, 99.0])'''


# Patch 3: print the diagnostic right before the μ_HR ablation block.
OLD_PRINT_AFTER = '''if _dag_avg is not None:
    _pct = _dag_avg * 100.0
    print(f"  🧠 μ_HR ABLATION (Stage-1 → Stage-2 conditioning):")'''

NEW_PRINT_AFTER = '''# >>> BS31_DIAG_SHORTCUT print block
if _shortcut["shortcut_ratio"] is not None:
    print(f"  🔬 SHORTCUT DIAGNOSTIC (residual collapse probe):")
    print(f"     ‖output‖              = {_shortcut['norm_output']:.5f}")
    print(f"     ‖μ_HR‖                = {_shortcut['norm_mu_HR']:.5f}")
    print(f"     ‖target‖              = {_shortcut['norm_target']:.5f}")
    print(f"     ‖output − μ_HR‖       = {_shortcut['norm_output_minus_mu_HR']:.5f}")
    print(f"     ‖target − μ_HR‖ (true)= {_shortcut['norm_target_minus_mu_HR']:.5f}")
    _r = _shortcut['shortcut_ratio']
    print(f"     shortcut_ratio        = {_r:.4f}  →  verdict: {_shortcut['verdict']}")
    if _shortcut['verdict'] == "SHORTCUT_CONFIRMED":
        print(f"     ⚠️  Diffusion ≈ identité sur μ_HR. UNet sous-capacité probable.")
    elif _shortcut['verdict'] == "AMBIGUOUS":
        print(f"     〰️  Zone grise — examiner aussi F1, RAPSD, ablation μ_HR.")
    else:
        print(f"     ✅ Diffusion produit un raffinement réel sur μ_HR.")
    print()

if _dag_avg is not None:
    _pct = _dag_avg * 100.0
    print(f"  🧠 μ_HR ABLATION (Stage-1 → Stage-2 conditioning):")'''


# Patch 4: include shortcut diag in the JSON payload.
OLD_JSON = '''    "mu_HR_ablation": {
        "delta_signal_ratio_avg": _dag_avg,
        "per_batch": _intervention,
        "verdict": verdict,
    },'''

NEW_JSON = '''    "mu_HR_ablation": {
        "delta_signal_ratio_avg": _dag_avg,
        "per_batch": _intervention,
        "verdict": verdict,
    },
    "shortcut_diagnostic": _shortcut,'''


def patch_cell_60() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "FINAL_VALIDATION (BS30" in s)
    if idx is None:
        print("  ! BS30 FINAL_VALIDATION cell not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS31_DIAG_SHORTCUT" in src:
        print(f"  = cell {idx} already patched (BS31a)")
        return 0

    n = 0
    for old, new, label in (
        (OLD_INIT, NEW_INIT, "init _all_mu_HR"),
        (OLD_APPEND, NEW_APPEND, "append μ_HR per batch"),
        (OLD_BEFORE_F1, NEW_BEFORE_F1, "shortcut diag computation"),
        (OLD_PRINT_AFTER, NEW_PRINT_AFTER, "shortcut print block"),
        (OLD_JSON, NEW_JSON, "shortcut field in JSON payload"),
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
    print("=== BS31a : shortcut learning diagnostic in FINAL_VALIDATION ===")
    n = patch_cell_60()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
