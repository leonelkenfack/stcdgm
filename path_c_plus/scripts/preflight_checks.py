"""Pre-flight checks for Path C+ — call at notebook bootstrap before training.

Surfaces the K8/K9/K5 footgun: temporal split fields exist in config but
are not yet enforced in pipeline.py until commits 21 (K9) and 22 (K5).

Usage in notebook:
    from path_c_plus.scripts.preflight_checks import check_temporal_split_enforcement
    check_temporal_split_enforcement(CONFIG)
"""
from __future__ import annotations

import warnings
from typing import Any

K8_FIELDS = [
    "train_start_date", "train_end_date",
    "val_start_date", "val_end_date",
    "test_start_date", "test_end_date",
    "temporal_holdout_start_date", "temporal_holdout_end_date",
]


def check_temporal_split_enforcement(config: Any) -> bool:
    """DS revise check: if K8 fields are present in config but pipeline.py
    does not yet enforce them, emit a loud warning.

    Returns True if K8 fields are detected (regardless of enforcement).
    """
    if not hasattr(config, "data"):
        return False

    data_cfg = config.data
    declared = []
    for field in K8_FIELDS:
        if hasattr(data_cfg, field) and getattr(data_cfg, field) is not None:
            declared.append(field)

    if not declared:
        return False

    # Check if pipeline.py has been updated to enforce these (K9 + K5 commits)
    # Heuristic: check if NetCDFDataPipeline.__init__ accepts train_start_date kwarg
    try:
        from st_cdgm.data.pipeline import NetCDFDataPipeline
        import inspect
        sig = inspect.signature(NetCDFDataPipeline.__init__)
        enforced = "train_start_date" in sig.parameters
    except Exception:
        enforced = False

    if not enforced:
        msg = (
            "\n" + "!" * 78 + "\n"
            "[K8 PRE-FLIGHT WARNING] Temporal split fields declared but NOT enforced\n"
            "\n"
            f"  Declared in config.data: {', '.join(declared)}\n"
            "\n"
            "  However, NetCDFDataPipeline.__init__ does not accept these parameters.\n"
            "  The fields are descriptive only until commits 21 (K9 temporal split)\n"
            "  and 22 (K5 train-only normalization) land.\n"
            "\n"
            "  CONSEQUENCE: training is still subject to:\n"
            "    - K5 data leakage: normalization stats computed on full dataset\n"
            "    - K9 random split: validation samples drawn from train period\n"
            "    - DS audit baseline numbers may still be invalid\n"
            "\n"
            "  This is OK for code development but DO NOT trust the metrics\n"
            "  from training runs initiated before commits 21+22 are applied.\n"
            + "!" * 78 + "\n"
        )
        print(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        return True

    return True


def run_all_preflight_checks(config: Any) -> dict:
    """Run all Path C+ pre-flight checks and return a report."""
    report = {
        "k8_temporal_split": check_temporal_split_enforcement(config),
        # Add more checks here as P0 fixes land
    }
    return report


if __name__ == "__main__":
    # Smoke test (no real config, just verify imports)
    from types import SimpleNamespace
    fake_config = SimpleNamespace(
        data=SimpleNamespace(
            train_start_date="1980-01-01",
            train_end_date="2009-12-31",
            val_start_date="2010-01-01",
        )
    )
    check_temporal_split_enforcement(fake_config)
