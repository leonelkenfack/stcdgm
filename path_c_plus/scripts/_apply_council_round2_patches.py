"""Apply round-2 council audit patches to st_cdgm_seed42_eval.ipynb.

5 bugs identified by 4 parallel reviewer agents (AI Eng, Math Prof, Climate ML, Lit Review) :

B1 (BLOCKING)  : Cell 11 references undefined `_persist_state_dict` — would crash on first
                 epoch save and burn 25-30h with zero checkpoints.

B2 (BLOCKING)  : Cell 12 loads `ema_{EMA_CHOICE}_state_dict` but Cell 11 saves under
                 `ema_state_dict`. EMA_CHOICE="0.9995" doesn't even match the trained
                 decay 0.999. Result: eval silently runs on non-EMA weights.

B3 (BLOCKING)  : Cell 10 sets `mardani_fix_zero_mu_HR_in_cache=True`. Cell 12 reads
                 `mardani_fix_zero_mu_HR_in_conditioning` → defaults False → eval feeds
                 REAL mu_HR while training fed zeros. Distribution shift invalidates v2
                 metrics. Fix: add the conditioning key to V2_CONFIG.

B4 (BLOCKING)  : `conditioning_dropout_prob=0.15` becomes no-op when mu_HR is already
                 zero in cache (dropped branch == kept branch). CFG=1.15 then has no
                 valid unconditional branch. Fix: cond_dropout=0.0 and cfg_scale=1.0.

B5 (NON-BLOCK) : compute_f1_extremes returns keys "p95"/"p99", not "f1_p95.0"/"f1_p99.0".
                 Multiple lookups across cells 6 + 7 + 12 + 13 silently return None.
                 Fix: replace the wrong keys throughout the printing/comparison code.
"""
from __future__ import annotations
import json
from pathlib import Path

NB_PATH = Path(r"c:/Users/reall/Desktop/climate_data/path_c_plus/scripts/st_cdgm_seed42_eval.ipynb")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))


def patch_source_line(cell_id: str, old: str, new: str, *, expect: int = 1) -> int:
    for c in nb["cells"]:
        if c.get("id") != cell_id:
            continue
        src = c.get("source") or []
        hits = 0
        for i, ln in enumerate(src):
            if old in ln:
                src[i] = ln.replace(old, new)
                hits += 1
        if hits != expect:
            raise RuntimeError(
                f"[{cell_id}] expected {expect} hit(s) for {old!r}, got {hits}"
            )
        return hits
    raise RuntimeError(f"cell id {cell_id!r} not found")


def patch_anywhere(old: str, new: str, *, min_hits: int = 1) -> int:
    total = 0
    for c in nb["cells"]:
        src = c.get("source") or []
        for i, ln in enumerate(src):
            if old in ln:
                src[i] = ln.replace(old, new)
                total += 1
    if total < min_hits:
        raise RuntimeError(f"expected ≥{min_hits} hit(s) for {old!r}, got {total}")
    return total


# ---------------------------------------------------------------- B1
# Cell 11 : replace _persist_state_dict(...) with inline state_dict()
b1a = patch_source_line(
    "cell-11-fix2-train",
    '"diffusion_state_dict": _persist_state_dict(diffusion_v2)',
    '"diffusion_state_dict": (diffusion_v2.module if hasattr(diffusion_v2, "module") else diffusion_v2).state_dict()',
)
b1b = patch_source_line(
    "cell-11-fix2-train",
    '"ema_state_dict": _persist_state_dict(ema_diffusion_v2)',
    '"ema_state_dict": (ema_diffusion_v2.module if hasattr(ema_diffusion_v2, "module") else ema_diffusion_v2).state_dict()',
)
print(f"B1 : _persist_state_dict patched in Cell 11 ({b1a + b1b} replacements)")


# ---------------------------------------------------------------- B2
# Cell 12 : EMA_CHOICE + EMA load key
b2a = patch_source_line(
    "cell-12-fix2-eval",
    'EMA_CHOICE = "0.9995"',
    'EMA_CHOICE = str(V2_CONFIG.get("ema_decay", 0.999))  # council fix : actual trained decay',
)
b2b = patch_source_line(
    "cell-12-fix2-eval",
    '_load_sd(diffusion_v2, ck_v2.get(f"ema_{EMA_CHOICE}_state_dict"))',
    '_load_sd(diffusion_v2, ck_v2.get("ema_state_dict"))  # council fix : single trained EMA key',
)
print(f"B2 : EMA load key + EMA_CHOICE patched in Cell 12 ({b2a + b2b} replacements)")


# ---------------------------------------------------------------- B3 + B4
# Cell 10 : V2_CONFIG already rewritten by separate Cell 10 NotebookEdit -- here we just
# patch in place because the file is too large for the Read tool.
# B3 : add mardani_fix_zero_mu_HR_in_conditioning right after the cache key
src_cell10 = next(c for c in nb["cells"] if c.get("id") == "cell-10-fix2-prep")["source"]

# Find the mardani cache line and insert the conditioning key right after it
inserted_b3 = False
for i, ln in enumerate(src_cell10):
    if '"mardani_fix_zero_mu_HR_in_cache": True,' in ln and not inserted_b3:
        indent = ln[: len(ln) - len(ln.lstrip())]
        new_line = (
            f'{indent}# Council audit fix : eval must also zero mu_HR to match training distribution\n'
        )
        new_key = f'{indent}"mardani_fix_zero_mu_HR_in_conditioning": True,\n'
        src_cell10.insert(i + 1, new_line)
        src_cell10.insert(i + 2, new_key)
        inserted_b3 = True
        break
if not inserted_b3:
    raise RuntimeError("B3 : could not locate cache key line in Cell 10")
print("B3 : mardani_fix_zero_mu_HR_in_conditioning=True inserted into V2_CONFIG")

# B4a : cfg_scale 1.15 -> 1.0
b4a = patch_source_line(
    "cell-10-fix2-prep",
    '"sampler_cfg_scale": 1.15,',
    '"sampler_cfg_scale": 1.0,  # council fix : no uncond branch (mu_HR=0 in cache)',
)
# B4b : conditioning_dropout_prob 0.15 -> 0.0
b4b = patch_source_line(
    "cell-10-fix2-prep",
    '"conditioning_dropout_prob": 0.15,',
    '"conditioning_dropout_prob": 0.0,  # council fix : no-op when mu_HR already 0 in cache',
)
print(f"B4 : cfg_scale and cond_dropout patched in Cell 10 ({b4a + b4b} replacements)")


# ---------------------------------------------------------------- B5
# Replace f1_p95.0 -> p95 and f1_p99.0 -> p99 across the notebook.
# These are the (wrong) keys used by .get() lookups; the function returns p95/p99.
b5a = patch_anywhere("'f1_p95.0'", "'p95'", min_hits=4)
b5b = patch_anywhere("'f1_p99.0'", "'p99'", min_hits=4)
print(f"B5 : f1_p95.0/f1_p99.0 -> p95/p99 patched ({b5a + b5b} replacements)")


# ---------------------------------------------------------------- write back
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"\nPatched notebook written to {NB_PATH}")
