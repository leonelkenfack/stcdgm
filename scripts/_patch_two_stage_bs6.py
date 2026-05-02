"""
BS6 — Materialise lazy params (PyG SAGEConv with in_channels=(-1,-1)) before
counting them for the Stage 1 optimizer.

The encoder declares SAGEConv layers with ``in_channels=(-1, -1)`` so PyG
infers the input feature dim from the first forward pass. Until that pass
runs, ``encoder.hetero_conv.convs[...].lin_l.weight`` is an
``UninitializedParameter``, and PyTorch raises::

    ValueError: Attempted to use an uninitialized parameter in <numel>.
    This error happens when you are using a `LazyModule` ...

Fix : run a single dummy forward through encoder → rcn → regression_head
just BEFORE building ``stage1_params``, so all lazy params are materialised
and the optimizer + ``p.numel()`` calls succeed.

Idempotent — sentinel-guarded by ``TWO_STAGE_BS6_MATERIALIZE_LAZY``.
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

# -------- Optimizer Stage 1 (encoder + RCN + regression_head) --------
stage1_params = (
    list(encoder.parameters())
    + list(rcn_cell.parameters())
    + list(regression_head.parameters())
)'''


NEW_BLOCK = '''ts_cfg = CONFIG.two_stage
S1_EPOCHS = int(ts_cfg.stage1.epochs_max)
S2_EPOCHS = int(ts_cfg.stage2.epochs_max)

# >>> TWO_STAGE_BS6_MATERIALIZE_LAZY
# The encoder uses SAGEConv(in_channels=(-1, -1)) so PyG infers the input
# feature dim lazily on the first forward pass. Until that pass runs the
# weights are torch.nn.parameter.UninitializedParameter and any call to
# `p.numel()` raises ValueError. Run a single dummy forward through
# encoder -> rcn -> regression_head to materialise the lazy params BEFORE
# we build the Stage 1 optimizer.
print("🔧 Materialisation des params lazy (SAGEConv in_channels=-1)...")
encoder.train(); rcn_cell.train(); regression_head.train()
_warmup_done = False
for _converted_batches in iterate_batches(train_dataloader, builder, DEVICE):
    for _batch in _converted_batches:
        with torch.no_grad():
            _H_init = encoder.init_state(_batch["hetero"]).to(DEVICE)
            _lr = _batch["lr"].to(DEVICE)
            _drivers = [_lr[t] for t in range(_lr.shape[0])]
            _seq = rcn_runner.run(_H_init, _drivers, reconstruction_sources=None)
            _ = regression_head(_seq.states[-1])
        _warmup_done = True
        break
    if _warmup_done:
        break
if not _warmup_done:
    raise RuntimeError(
        "TWO_STAGE_BS6_MATERIALIZE_LAZY: dataloader returned no batch — "
        "cannot warm up encoder."
    )
print(f"   ✓ Encoder params materialised "
      f"({sum(p.numel() for p in encoder.parameters()):,} total)")
del _converted_batches, _batch, _H_init, _lr, _drivers, _seq, _warmup_done

# -------- Optimizer Stage 1 (encoder + RCN + regression_head) --------
stage1_params = (
    list(encoder.parameters())
    + list(rcn_cell.parameters())
    + list(regression_head.parameters())
)'''


def patch_bs6_materialize_lazy() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    idx = _find_cell(cells, lambda s: "TWO_STAGE_TRAINING_LOOP" in s and "stage1_params = (" in s)
    if idx is None:
        print("  ! TWO_STAGE_TRAINING_LOOP cell not found")
        return 0

    src = "".join(cells[idx]["source"])
    if "TWO_STAGE_BS6_MATERIALIZE_LAZY" in src:
        print("  = BS6 already patched")
        return 0

    if OLD_BLOCK not in src:
        print("  ! BS6 OLD_BLOCK pattern not found — manual review needed")
        return 0

    new_src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    cells[idx]["source"] = new_src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ BS6 patched (cell {idx}): lazy-param materialisation block inserted")
    return 1


def main() -> int:
    print("=== BS6 : materialise SAGEConv lazy params before optimizer ===")
    n = patch_bs6_materialize_lazy()
    print(f"\n{n} modification(s) appliquée(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
