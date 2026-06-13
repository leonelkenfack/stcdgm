"""Tombstone migration: mark pre-K1 result JSONs as invalid for H1 analysis.

DS Round-2 Condition A + Round-9 Clause PC4: any results JSON produced BEFORE
the K1 fix (commit 685e994) had causal_concat=True on both Oracle AND CorrDiff
sampling paths -> apples-to-mangoes comparison, methodologically invalid.

This script adds two fields to every legacy JSON:
  "schema_version": "legacy-pre-K1"
  "valid_for_analysis": false
  "tombstone_reason": "K1 audit: causal_concat=True applied to both Oracle and CorrDiff sampling. K30 audit: V5-mini ran only 10/200 epochs. J29 audit: cfg_scale silently ignored. See path_c_plus/HYPERPLAN.md section 'ALERTE METHODOLOGIQUE MAJEURE'."

Usage:
  python path_c_plus/scripts/_tombstone_legacy_jsons.py [--dry-run]

By default, runs in dry-run mode (no writes). Pass --apply to actually modify.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

# Paths to scan for legacy JSONs
LEGACY_DIRS = [
    "results/v5_evaluation",
    "results/v5_baseline_corrected_eval",
    # Add SMOKE_DIR results if user has them locally
]

TOMBSTONE_FIELDS = {
    "schema_version": "legacy-pre-K1",
    "valid_for_analysis": False,
    "tombstone_reason": (
        "K1 audit: causal_concat=True applied to both Oracle and CorrDiff "
        "sampling. K30 audit: V5-mini ran only 10/200 epochs. J29 audit: "
        "cfg_scale silently ignored. See path_c_plus/HYPERPLAN.md section "
        "'ALERTE METHODOLOGIQUE MAJEURE'."
    ),
    "tombstone_applied_by": "path_c_plus/scripts/_tombstone_legacy_jsons.py",
}

# Schema versions used to distinguish Path-C+ eras (DS Batch-D follow-up).
# Phase A0'' eligibility = schema_version == "path-c-plus-batch-D-v1".
SCHEMA_VERSION_BATCH_D = "path-c-plus-batch-D-v1"
SCHEMA_VERSION_PRE_BATCH_D = "path-c-plus-pre-batch-D"  # smoke #1/#2 era

BATCH_D_FIXES_APPLIED = ["J29", "K2", "K3", "K9", "K5"]


def stamp_batch_d_json(result_dict: dict, *,
                       k9_train: list = None,
                       k9_val: list = None,
                       k9_test: list = None,
                       k5_train_window: list = None,
                       j29_scheduler_type: str = None,
                       j29_cfg_scale: float = None,
                       pre_registration_commit: str = None) -> dict:
    """Stamp a fresh result dict as Phase A0'' eligible (Batch-D era).

    Usage:
        result = {...your computed metrics...}
        result = stamp_batch_d_json(result, k9_train=["1980-01-01","2009-12-31"], ...)
        json.dump(result, open("results/phase_a0pp/seed0.json","w"))

    Audit gate PC4 will then accept these JSONs as valid_for_analysis=True.
    """
    result_dict["schema_version"] = SCHEMA_VERSION_BATCH_D
    result_dict["path_c_plus_batch"] = "D"
    result_dict["fixes_applied"] = list(BATCH_D_FIXES_APPLIED)
    result_dict["valid_for_analysis"] = True
    if k9_train is not None:
        result_dict["k9_temporal_split"] = {
            "train": k9_train, "val": k9_val, "test": k9_test,
        }
    if k5_train_window is not None:
        result_dict["k5_train_window"] = k5_train_window
    if j29_scheduler_type is not None:
        result_dict["j29_scheduler_type"] = j29_scheduler_type
        result_dict["j29_cfg_scale"] = j29_cfg_scale
    if pre_registration_commit is not None:
        result_dict["pre_registration_commit"] = pre_registration_commit
    return result_dict


def find_legacy_jsons(base_paths: List[str]) -> List[Path]:
    """Find all JSON files under the legacy directories."""
    found = []
    for base in base_paths:
        bp = Path(base)
        if not bp.exists():
            print(f"[INFO] {base}: directory does not exist locally, skipping")
            continue
        for p in bp.rglob("*.json"):
            found.append(p)
    return found


def is_already_tombstoned(data: dict) -> bool:
    """Check if a JSON has already been tombstoned."""
    return (
        data.get("schema_version") == "legacy-pre-K1"
        and data.get("valid_for_analysis") is False
    )


def tombstone_json(path: Path, dry_run: bool = True) -> dict:
    """Apply tombstone fields to a single JSON. Returns status dict."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        return {"path": str(path), "status": "ERROR", "error": str(e)}

    if is_already_tombstoned(data):
        return {"path": str(path), "status": "ALREADY_TOMBSTONED"}

    # Add tombstone fields at the top level
    for k, v in TOMBSTONE_FIELDS.items():
        data[k] = v

    if dry_run:
        return {"path": str(path), "status": "WOULD_TOMBSTONE"}

    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return {"path": str(path), "status": "TOMBSTONED"}
    except OSError as e:
        return {"path": str(path), "status": "WRITE_ERROR", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true",
                        help="Actually modify files (default: dry-run only)")
    args = parser.parse_args()

    dry_run = not args.apply
    mode = "DRY-RUN" if dry_run else "APPLY"

    print(f"=== Tombstone Legacy JSONs ({mode}) ===")
    print()
    print(f"Scanning {len(LEGACY_DIRS)} directories...")

    legacy_files = find_legacy_jsons(LEGACY_DIRS)
    print(f"Found {len(legacy_files)} JSON files")
    print()

    results = []
    for p in legacy_files:
        result = tombstone_json(p, dry_run=dry_run)
        results.append(result)
        print(f"  [{result['status']:20s}] {result['path']}")
        if "error" in result:
            print(f"                       ERROR: {result['error']}")

    print()
    by_status: dict = {}
    for r in results:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1
    print("Summary:")
    for status, count in sorted(by_status.items()):
        print(f"  {status:20s} {count}")

    if dry_run:
        print()
        print("DRY-RUN complete. Run with --apply to actually modify files.")


if __name__ == "__main__":
    main()
