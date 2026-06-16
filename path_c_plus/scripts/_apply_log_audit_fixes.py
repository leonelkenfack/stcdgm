"""Apply council audit fixes to the detailed-log additions.

AI Eng (GO-WITH-CHANGE, 2 minor):
- A1 : use math.isfinite() instead of `== float("inf")` (also catches -inf and NaN)
- A2 : move Cell 12 sanity-range verdict OUTSIDE the `if not fv_v2_path.exists():`
       block so it prints on cached-load reruns too.

Climate ML (GO-WITH-CHANGE, 5 calibration fixes):
- C1 : dag_sensitivity range "< 1.0" -> "0.05 <= ds <= 0.6"
       (smoke #4 memory: ds~0.16 is real learning; >0.6 enters copy-mode artefact regime)
- C2 : _LOSS_EXPLODE_THR 50.0 -> 5.0 (diffusion loss > 5 with sigma_data=0.5 is pathological)
- C3 : RMSE strict-beat -> non-inferiority within +5% (seed std ~0.005-0.008)
- C4 : Pearson_PS strict-beat -> non-inferiority within -1.2% (seed std ~0.008)
- C5 : F1@p99 strict-beat -> non-inferiority within -6% (heavy-tail noisy, seed std 0.03-0.04)
- C6 : SSR band [0.6, 1.4] -> [0.4, 1.6] (noncausal v4 ref=0.4116 was OUTSIDE old band -> inconsistent)
- C7 : loss_diff range "1.0 -> 0.1-0.3" -> "0.5-1.5 -> 0.05-0.20" (calibrated for sigma_data=0.5)
- C8 : epoch time "7-9 min" -> "5-12 min" (absorb Drive I/O stalls)
- C9 : contrastive_dag range "0.01-0.1" -> "0.005-0.15"
"""
from __future__ import annotations
import json
from pathlib import Path

NB = Path(r"c:/Users/reall/Desktop/climate_data/path_c_plus/scripts/st_cdgm_seed42_eval.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))


def cell(cell_id):
    for c in nb["cells"]:
        if c.get("id") == cell_id: return c
    raise KeyError(cell_id)


def patch(cell_id, old, new, *, count=1):
    src = cell(cell_id)["source"]
    hits = 0
    for i, ln in enumerate(src):
        if old in ln:
            src[i] = ln.replace(old, new)
            hits += 1
    if hits != count:
        raise RuntimeError(f"[{cell_id}] expected {count} hit(s) for {old!r}, got {hits}")


# ==================================================== A1 (NaN/Inf abort fix)
# Add `import math as _math_v11` near the top of cell 11
src11 = cell("cell-11-fix2-train")["source"]
for i, ln in enumerate(src11):
    if "import tempfile as _tempfile_v11" in ln:
        src11.insert(i, "import math as _math_v11\n")
        break

patch(
    "cell-11-fix2-train",
    'if _ld != _ld or _ld == float("inf"):',
    'if not _math_v11.isfinite(_ld):',
)
print("A1 : NaN/Inf abort now uses math.isfinite (catches -inf too)")


# ==================================================== C2 (loss explode threshold)
patch(
    "cell-11-fix2-train",
    "_LOSS_EXPLODE_THR = 50.0",
    "_LOSS_EXPLODE_THR = 5.0",
)
print("C2 : _LOSS_EXPLODE_THR 50.0 -> 5.0")


# ==================================================== C1, C7, C8, C9 (pre-training expected ranges)
patch(
    "cell-11-fix2-train",
    '  loss_diff      : starts ~1.0, descends to ~0.1-0.3 by epoch 200',
    '  loss_diff      : starts ~0.5-1.5, descends to ~0.05-0.20 by epoch 200',
)
patch(
    "cell-11-fix2-train",
    '  contrastive_dag: ~0.01-0.1 (small contribution)',
    '  contrastive_dag: ~0.005-0.15 (small contribution, lambda=0.5 applied)',
)
patch(
    "cell-11-fix2-train",
    '  dag_sensitivity: > 0 (DAG used) and ideally < 1.0 (avoid copy-mode)',
    '  dag_sensitivity: target 0.05 <= ds <= 0.6 (smoke#4 calibration)',
)
patch(
    "cell-11-fix2-train",
    '  epoch time     : ~7-9 min on A100',
    '  epoch time     : ~5-12 min on A100 (absorbs Drive I/O stalls)',
)
print("C1+C7+C8+C9 : pre-training expected ranges recalibrated")


# ==================================================== C3-C6 (Cell 12 sanity range table)
src12 = cell("cell-12-fix2-eval")["source"]

# Recalibrate noncausal-reference thresholds
patch(
    "cell-12-fix2-eval",
    '_verdicts.append(("RMSE",       _rmse_v2,      f"want < {_NC_RMSE:.4f} (noncausal)", _rmse_v2 < _NC_RMSE))',
    '_NI_RMSE = _NC_RMSE * 1.05  # non-inferiority +5%\n'
    '    _verdicts.append(("RMSE",       _rmse_v2,      f"want < {_NI_RMSE:.4f} (NI vs noncausal {_NC_RMSE:.4f})", _rmse_v2 < _NI_RMSE))',
)
patch(
    "cell-12-fix2-eval",
    '_verdicts.append(("Pearson_PS", _corr_per_sample_v2, f"want > {_NC_PEAR:.4f}",       _corr_per_sample_v2 > _NC_PEAR))',
    '_NI_PEAR = _NC_PEAR * 0.988  # non-inferiority -1.2%\n'
    '    _verdicts.append(("Pearson_PS", _corr_per_sample_v2, f"want > {_NI_PEAR:.4f} (NI vs noncausal {_NC_PEAR:.4f})", _corr_per_sample_v2 > _NI_PEAR))',
)
patch(
    "cell-12-fix2-eval",
    '_verdicts.append(("F1@p99",     _f1_v2.get("p99", float("nan")), f"want > {_NC_F1P99:.4f}", _f1_v2.get("p99", 0) > _NC_F1P99))',
    '_NI_F1P99 = _NC_F1P99 * 0.94  # non-inferiority -6% (heavy-tail noise)\n'
    '    _verdicts.append(("F1@p99",     _f1_v2.get("p99", float("nan")), f"want > {_NI_F1P99:.4f} (NI vs noncausal {_NC_F1P99:.4f})", _f1_v2.get("p99", 0) > _NI_F1P99))',
)
patch(
    "cell-12-fix2-eval",
    '_verdicts.append(("SSR",        _ssr_v2,       "want in [0.6, 1.4] (calibration)",  0.6 <= _ssr_v2 <= 1.4))',
    '_verdicts.append(("SSR",        _ssr_v2,       "want in [0.4, 1.6] (calibration, ref noncausal=0.4116)",  0.4 <= _ssr_v2 <= 1.6))',
)
print("C3+C4+C5+C6 : v2 vs noncausal verdict thresholds recalibrated (non-inferiority)")


# ==================================================== A2 (move sanity-range verdict outside if/else)
# Strategy : find the verdict block (between "# === detailed log : sanity-range verdicts ===" and
# the print "Overall : ...") inside the `if not fv_v2_path.exists():` branch, then move it
# AFTER the `else:` clause so it runs in both branches.

# First locate the verdict block within src12
v_start, v_end = None, None
for i, ln in enumerate(src12):
    if '# === detailed log : sanity-range verdicts ===' in ln and v_start is None:
        v_start = i
    if v_start is not None and v_end is None and 'Overall : {_n_ok}/{len(_verdicts)} metrics' in ln:
        v_end = i + 1
        break
if v_start is None or v_end is None:
    raise RuntimeError(f"could not locate verdict block (v_start={v_start} v_end={v_end})")

verdict_lines = src12[v_start:v_end]
# Re-indent from 4 spaces (inside if) to 0 spaces (top-level).
# Source-list entries may contain embedded \n (because patch() above inserted
# multi-line replacements). Handle each physical line within an entry.
def _strip4(_ln):
    parts = _ln.split("\n")
    out = []
    for p in parts:
        out.append(p[4:] if p.startswith("    ") else p)
    return "\n".join(out)

reindented = [_strip4(ln) for ln in verdict_lines]
# Replace `_rmse_v2`, `_corr_per_sample_v2`, `_f1_v2`, `_ssr_v2`, `_rapsd_v2` (locals inside if-branch)
# with reads from `fv_v2` dict (which exists in both branches).
remap = {
    "_rmse_v2": 'fv_v2.get("rmse", float("nan"))',
    "_corr_per_sample_v2": 'fv_v2.get("pearson_corr", {}).get("per_sample_avg", float("nan"))',
    '_f1_v2.get("p99", float("nan"))': 'fv_v2.get("f1_extremes", {}).get("p99", float("nan"))',
    '_f1_v2.get("p99", 0)': 'fv_v2.get("f1_extremes", {}).get("p99", 0)',
    "_ssr_v2": 'fv_v2.get("spread_skill_ratio", float("nan"))',
    "_rapsd_v2": 'fv_v2.get("rapsd_distance", float("nan"))',
}
final = []
for ln in reindented:
    for k, v in remap.items():
        ln = ln.replace(k, v)
    final.append(ln)

# Remove the original (still inside if-branch) verdict
del src12[v_start:v_end]

# Insert recreated block at TRUE top-level, AFTER the `else: ... fv_v2 = json.loads(...)` branch.
# Anchor : the line `print(f"\nStage 2 v2 eval done. Use Cell 13 for final comparison.")`
for i, ln in enumerate(src12):
    if 'Stage 2 v2 eval done. Use Cell 13' in ln:
        # Insert verdict block BEFORE this final print
        src12[i:i] = final + ["\n"]
        break
print("A2 : sanity-range verdict moved outside if/else (prints on cached-load reruns too)")


# ==================================================== write back
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

# Re-validate syntax
for cid in ("cell-10-fix2-prep", "cell-11-fix2-train", "cell-12-fix2-eval"):
    src = "".join(cell(cid)["source"])
    try:
        compile(src, cid, "exec")
        print(f"{cid}: OK")
    except SyntaxError as e:
        print(f"{cid}: SYNTAX ERROR at line {e.lineno}: {e.msg}")
        for ln_no in range(max(0, e.lineno - 3), min(len(src.split(chr(10))), e.lineno + 3)):
            print(f"  {ln_no+1:4d} | {src.split(chr(10))[ln_no]}")
        raise

print("\nAll audit fixes applied and notebook validates.")
