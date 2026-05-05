"""
BS23 — graceful shape-mismatch handling in _persist_load_state_dict.

Observed: a checkpoint saved with an older UNet (block_out_channels
[64,128,256,256], 1-channel conv_in) is being loaded into the current
UNet (block_out_channels [32,64,64,64], 3-channel conv_in for the
causal_concat conditioning [δ_noisy, μ_HR, baseline_log]). BS22 already
fixed key normalization, but ``load_state_dict(strict=False)`` still
*raises* on shape mismatch — so the entire diffusion stays at xavier
init even though most encoder/rcn keys would have matched.

Fix: per-tensor shape check before assigning. Tensors with mismatched
shape are dropped + reported. Final ``load_state_dict`` only sees
shape-compatible entries.

For the user's specific case:
- encoder, rcn_cell, regression_head have unchanged architecture
  → still load 100%.
- diffusion architecture changed → almost nothing matches → essentially
  starts from xavier (which is fine because Stage 1 doesn't use the
  diffusion anyway, and Stage 2 will train it fresh).

Idempotent — sentinel ``BS23 — shape-mismatch tolerance``.
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


# Pattern matching the BS22 implementation we just installed.
OLD_BODY = '''def _persist_load_state_dict(m, sd):
    """Charge un state_dict en respectant les wrappers DDP / torch.compile.

    BS22 — bidirectional ``_orig_mod`` normalization:
    ``torch.compile`` can be applied to *nested* submodules (e.g.
    ``diffusion.unet``), so the prefix may appear at any depth. We strip
    every ``_orig_mod.`` token in saved keys then re-map onto the live
    module's expected keys (which themselves may carry ``_orig_mod`` at
    arbitrary positions). Returns silently on success, prints a single
    summary line if any key was unmatched.
    """
    if m is None or sd is None:
        return
    base = m.module if hasattr(m, "module") and not hasattr(m, "_orig_mod") else m
    base = getattr(base, "_orig_mod", base)

    # Strip _orig_mod tokens at any depth from saved keys.
    stripped_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    target_keys = list(base.state_dict().keys())
    matched = {}
    for tk in target_keys:
        norm_tk = tk.replace("_orig_mod.", "")
        if norm_tk in stripped_sd:
            matched[tk] = stripped_sd[norm_tk]
    n_target = len(target_keys)
    n_matched = len(matched)
    if n_matched < n_target:
        # Useful when a checkpoint was saved before architecture changes.
        print(f"   ↳ _persist_load: matched {n_matched}/{n_target} weights (some keys absent in checkpoint)")
    base.load_state_dict(matched, strict=False)'''

NEW_BODY = '''def _persist_load_state_dict(m, sd):
    """Charge un state_dict en respectant les wrappers DDP / torch.compile.

    BS22 — bidirectional ``_orig_mod`` normalization (handles nested compile).
    BS23 — shape-mismatch tolerance: if architecture changed since the
    checkpoint was written (e.g. UNet block_out_channels reduced, or
    causal_concat extended conv_in from 1→3 channels), drop the
    incompatible tensors with a clear warning instead of crashing the
    whole load.
    """
    if m is None or sd is None:
        return
    base = m.module if hasattr(m, "module") and not hasattr(m, "_orig_mod") else m
    base = getattr(base, "_orig_mod", base)

    # BS22: strip _orig_mod tokens at any depth from saved keys.
    stripped_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    target_sd = base.state_dict()
    target_keys = list(target_sd.keys())

    matched = {}
    skipped_shape = []
    for tk in target_keys:
        norm_tk = tk.replace("_orig_mod.", "")
        if norm_tk not in stripped_sd:
            continue
        v = stripped_sd[norm_tk]
        # BS23: per-tensor shape check.
        live_shape = tuple(target_sd[tk].shape) if hasattr(target_sd[tk], "shape") else None
        ckpt_shape = tuple(v.shape) if hasattr(v, "shape") else None
        if live_shape is not None and ckpt_shape is not None and live_shape != ckpt_shape:
            skipped_shape.append((tk, ckpt_shape, live_shape))
            continue
        matched[tk] = v

    n_target = len(target_keys)
    n_matched = len(matched)
    n_skipped = len(skipped_shape)

    if n_matched < n_target or n_skipped > 0:
        msg = f"   ↳ _persist_load: matched {n_matched}/{n_target} weights"
        if n_skipped:
            msg += f", skipped {n_skipped} shape mismatches (architecture drift)"
        print(msg)
        # Show up to 3 examples so the user can spot the dimension change.
        for tk, sshape, tshape in skipped_shape[:3]:
            print(f"      • {tk}: ckpt{sshape} ≠ live{tshape}")
        if n_skipped > 3:
            print(f"      … (+{n_skipped - 3} more shape mismatches)")

    base.load_state_dict(matched, strict=False)'''


def patch_cell_47() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "def _persist_load_state_dict" in s
                     and "PERSIST_HELPERS" in s)
    if idx is None:
        print("  ! cell 47 (PERSIST_HELPERS) not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS23 — shape-mismatch tolerance" in src:
        print("  = cell 47 already patched (BS23)")
        return 0
    if OLD_BODY not in src:
        print("  ! cell 47: BS22 baseline pattern not found — apply BS22 first")
        return 0
    src = src.replace(OLD_BODY, NEW_BODY, 1)
    cells[idx]["source"] = src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("  ~ cell 47: _persist_load_state_dict extended with BS23 shape filter")
    return 1


def main() -> int:
    print("=== BS23 : shape-mismatch tolerance in _persist_load_state_dict ===")
    n = patch_cell_47()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
