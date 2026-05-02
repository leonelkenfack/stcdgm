"""
BS7 — Build train_dataloader / val_dataloader inside TWO_STAGE_TRAINING_LOOP.

The dataloader construction (split + DataLoader objects) lives inside the
LEGACY training cell, which is now wrapped by TWO_STAGE_BS4_LEGACY_GUARD's
``if CONFIG ... else:`` branch. When ``two_stage.enabled=True`` the legacy
cell early-returns, so ``train_dataloader`` / ``val_dataloader`` are never
defined, and the two-stage cell crashes with ``NameError``.

Fix : inject a self-contained data-prep block at the top of cell 50 (right
after the SKIP_STAGE_1 detection, before BS6 lazy-param materialisation),
guarded by a runtime check so it does nothing if the loaders already exist.

Idempotent — sentinel-guarded by ``TWO_STAGE_BS7_BUILD_DATALOADERS``.
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


OLD_BLOCK = '''ts_cfg = CONFIG.two_stage
S1_EPOCHS = int(ts_cfg.stage1.epochs_max)
S2_EPOCHS = int(ts_cfg.stage2.epochs_max)

# >>> TWO_STAGE_BS6_MATERIALIZE_LAZY'''


NEW_BLOCK = '''ts_cfg = CONFIG.two_stage
S1_EPOCHS = int(ts_cfg.stage1.epochs_max)
S2_EPOCHS = int(ts_cfg.stage2.epochs_max)

# >>> TWO_STAGE_BS7_BUILD_DATALOADERS
# train_dataloader / val_dataloader normally come from the LEGACY training
# cell, but TWO_STAGE_BS4_LEGACY_GUARD makes that cell early-return when
# two_stage.enabled is True. Build the loaders here so the two-stage loop
# is self-contained. No-op if they already exist.
from torch.utils.data import IterableDataset, random_split, DataLoader as _DataLoader

try:
    train_dataloader  # noqa: F821 — probe for prior definition
    val_dataloader    # noqa: F821
    print("ℹ️  train/val_dataloader déjà définis — réutilisation.")
except NameError:
    print("📦 Two-Stage : construction des DataLoaders (legacy cell skipped)")
    _seed = int(CONFIG.training.get("seed", 42))
    _val_fraction = float(CONFIG.training.get("val_fraction", 0.1))
    _val_fraction = min(max(_val_fraction, 0.05), 0.5)

    try:
        _total_samples = len(dataset)
    except TypeError:
        _total_samples = None

    class _IndexFilteredIterable(IterableDataset):
        """Split deterministe d'un IterableDataset par modulo d'index."""
        def __init__(self, base, keep_fn):
            self.base = base
            self.keep_fn = keep_fn
        def __iter__(self):
            for i, sample in enumerate(self.base):
                if self.keep_fn(i):
                    yield sample

    if _total_samples is None:
        _val_every = max(2, int(round(1.0 / max(_val_fraction, 0.05))))
        train_dataset = _IndexFilteredIterable(dataset, lambda i, k=_val_every: i % k != 0)
        val_dataset = _IndexFilteredIterable(dataset, lambda i, k=_val_every: i % k == 0)
        print(f"   📚 IterableDataset split (1 sample / {_val_every} en validation)")
    else:
        if _total_samples < 2:
            raise RuntimeError("Dataset trop petit pour un split train/val (len < 2).")
        _val_size = max(1, int(_total_samples * _val_fraction))
        if _val_size >= _total_samples:
            _val_size = _total_samples - 1
        _train_size = _total_samples - _val_size
        _split_gen = torch.Generator().manual_seed(_seed)
        train_dataset, val_dataset = random_split(
            dataset, [_train_size, _val_size], generator=_split_gen
        )
        print(f"   📚 Dataset split: train={_train_size}, val={_val_size}, total={_total_samples}")

    BATCH_SIZE = int(CONFIG.training.get("batch_size", 1))
    NUM_WORKERS = int(CONFIG.training.get("num_workers", 0))
    if NUM_WORKERS < 0:
        NUM_WORKERS = 0
    PIN_MEMORY = bool(DEVICE.type == "cuda")
    if NUM_WORKERS == 0:
        PREFETCH_FACTOR = None
        PERSISTENT_WORKERS = False
    else:
        PREFETCH_FACTOR = int(CONFIG.training.get("prefetch_factor", 2))
        PERSISTENT_WORKERS = bool(CONFIG.training.get("persistent_workers", True))

    def build_loader(ds, shuffle):
        kwargs = {
            "dataset": ds,
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "persistent_workers": PERSISTENT_WORKERS,
            "collate_fn": lambda x: x,
        }
        if not isinstance(ds, IterableDataset):
            kwargs["shuffle"] = shuffle
        if PREFETCH_FACTOR is not None:
            kwargs["prefetch_factor"] = PREFETCH_FACTOR
        return _DataLoader(**kwargs)

    train_dataloader = build_loader(train_dataset, shuffle=True)
    val_dataloader = build_loader(val_dataset, shuffle=False) if val_dataset is not None else None
    print(f"   ✓ train/val_dataloader prêts "
          f"(BS={BATCH_SIZE}, NW={NUM_WORKERS}, pin={PIN_MEMORY})")

# >>> TWO_STAGE_BS6_MATERIALIZE_LAZY'''


def patch_bs7_build_dataloaders() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    idx = _find_cell(
        cells,
        lambda s: "TWO_STAGE_TRAINING_LOOP" in s
        and "TWO_STAGE_BS6_MATERIALIZE_LAZY" in s,
    )
    if idx is None:
        print("  ! TWO_STAGE_TRAINING_LOOP cell with BS6 marker not found")
        return 0

    src = "".join(cells[idx]["source"])
    if "TWO_STAGE_BS7_BUILD_DATALOADERS" in src:
        print("  = BS7 already patched")
        return 0

    if OLD_BLOCK not in src:
        print("  ! BS7 OLD_BLOCK pattern not found — manual review needed")
        return 0

    new_src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    cells[idx]["source"] = new_src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ BS7 patched (cell {idx}): self-contained dataloader build inserted")
    return 1


def main() -> int:
    print("=== BS7 : build train/val_dataloader inside two-stage cell ===")
    n = patch_bs7_build_dataloaders()
    print(f"\n{n} modification(s) appliquée(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
