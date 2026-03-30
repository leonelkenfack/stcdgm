"""Helpers for loading PyTorch checkpoints saved under different wrappers."""

from __future__ import annotations

from typing import Any, Dict


def strip_torch_compile_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove ``_orig_mod.`` key prefix produced by ``torch.compile`` when loading
    into a non-compiled (eager) module.
    """
    if not state_dict:
        return state_dict
    keys = list(state_dict.keys())
    if not any(str(k).startswith("_orig_mod.") for k in keys):
        return state_dict
    return {
        (k[len("_orig_mod.") :] if str(k).startswith("_orig_mod.") else k): v
        for k, v in state_dict.items()
    }
