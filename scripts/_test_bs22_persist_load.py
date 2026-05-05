"""
Smoke test for BS22 _persist_load_state_dict bidirectional normalization.

Simulates the failure case observed in Colab: checkpoint saved with
``unet.conv_in.weight`` but live module exposes ``unet._orig_mod.conv_in.weight``
because ``torch.compile`` was applied to the nested ``unet`` submodule.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class FakeOrigMod(nn.Module):
    """Wrapper that mimics torch.compile's _orig_mod attribute."""

    def __init__(self, inner):
        super().__init__()
        self._orig_mod = inner

    def forward(self, x):
        return self._orig_mod(x)

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        return sd


class FakeDecoder(nn.Module):
    """Mimics CausalDiffusionDecoder layout: top-level module wrapping a nested compiled unet."""

    def __init__(self, compiled: bool):
        super().__init__()
        inner_unet = nn.Sequential(
            nn.Linear(4, 4),
            nn.Linear(4, 4),
        )
        if compiled:
            self.unet = FakeOrigMod(inner_unet)
        else:
            self.unet = inner_unet
        self.head = nn.Linear(4, 1)


def _persist_load_state_dict(m, sd):
    """BS22 implementation copied from notebook cell 47."""
    if m is None or sd is None:
        return
    base = m.module if hasattr(m, "module") and not hasattr(m, "_orig_mod") else m
    base = getattr(base, "_orig_mod", base)

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
        print(f"   ↳ matched {n_matched}/{n_target} weights")
    base.load_state_dict(matched, strict=False)


def assert_weights_equal(a: nn.Module, b: nn.Module, label: str):
    sa = {k.replace("_orig_mod.", ""): v for k, v in a.state_dict().items()}
    sb = {k.replace("_orig_mod.", ""): v for k, v in b.state_dict().items()}
    assert set(sa.keys()) == set(sb.keys()), (
        f"[{label}] key sets differ\n  A: {sorted(sa)}\n  B: {sorted(sb)}"
    )
    for k in sa:
        if not torch.allclose(sa[k], sb[k]):
            raise AssertionError(f"[{label}] tensor {k} differs")
    print(f"  ✓ {label}: all {len(sa)} tensors equal after load")


def case_compiled_to_compiled():
    """Save from compiled module, load into compiled module."""
    src = FakeDecoder(compiled=True)
    dst = FakeDecoder(compiled=True)  # different random init
    sd = src.state_dict()
    print(f"  source keys: {list(sd.keys())[:3]}...")
    _persist_load_state_dict(dst, sd)
    assert_weights_equal(src, dst, "compiled→compiled")


def case_uncompiled_save_compiled_load():
    """The actual bug case: saved keys lack _orig_mod, target has nested _orig_mod."""
    src = FakeDecoder(compiled=False)  # saved from this layout
    dst = FakeDecoder(compiled=True)   # live module: unet._orig_mod.*
    sd = src.state_dict()
    print(f"  source keys: {list(sd.keys())[:3]}...")
    print(f"  target keys: {list(dst.state_dict().keys())[:3]}...")
    _persist_load_state_dict(dst, sd)
    assert_weights_equal(src, dst, "uncompiled→compiled")


def case_compiled_save_uncompiled_load():
    """Reverse: saved keys have _orig_mod, target doesn't."""
    src = FakeDecoder(compiled=True)
    dst = FakeDecoder(compiled=False)
    sd = src.state_dict()
    print(f"  source keys: {list(sd.keys())[:3]}...")
    print(f"  target keys: {list(dst.state_dict().keys())[:3]}...")
    _persist_load_state_dict(dst, sd)
    assert_weights_equal(src, dst, "compiled→uncompiled")


def case_partial_match():
    """Checkpoint missing some keys (architecture changed)."""

    class FakeDecoderBig(FakeDecoder):
        def __init__(self):
            super().__init__(compiled=True)
            self.extra = nn.Linear(4, 4)

    src = FakeDecoder(compiled=False)
    dst = FakeDecoderBig()
    sd = src.state_dict()
    _persist_load_state_dict(dst, sd)
    print("  ✓ partial: completed without error")


if __name__ == "__main__":
    print("=== BS22 _persist_load_state_dict smoke tests ===\n")
    print("Case 1: compiled → compiled")
    case_compiled_to_compiled()
    print("\nCase 2 (the bug): uncompiled save → compiled load")
    case_uncompiled_save_compiled_load()
    print("\nCase 3: compiled save → uncompiled load")
    case_compiled_save_uncompiled_load()
    print("\nCase 4: partial match (extra layer in target)")
    case_partial_match()
    print("\n✅ All BS22 smoke tests passed.")
