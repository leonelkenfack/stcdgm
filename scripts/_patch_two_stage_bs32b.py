"""
BS32b — pre-cache Stage 1 outputs and run Stage 2 with thin batched loop.

Rationale
---------
Stage 1 (encoder + 16-step RCN + regression_head) is *frozen* during
Stage 2 (per Mardani et al. 2024 §5.3.2). With the production training
loop running 64 micros at bs=1 per logical batch, Stage 1 is forwarded
~64× per batch every epoch, even though every output is deterministic
given the frozen weights and fixed input.

Pre-caching collapses ~14000 × N_epochs forwards to 14000 × 1, then
Stage 2 trains on a TensorDataset of pre-computed
(μ_HR, baseline_log, δ_target) and forwards the diffusion UNet at the
*full* batch size (bs=64) instead of bs=1 in a Python loop — the
diffusion forward+backward becomes 1 GPU call per logical batch
instead of 64.

Implementation
--------------
1. Library functions added in ``src/st_cdgm/training/two_stage.py``:
    - ``precompute_stage1_outputs(...)`` — iterates the train dataset,
      runs Stage 1 forward per sample, returns a dict of stacked
      tensors (CPU).
    - ``train_epoch_stage2_cached(...)`` — thin loop that reads from
      the cached dataloader and runs diffusion forward+backward at
      full batch size.

2. Cell 50 patches (this script):
    - Insert a "BS32B_PRECOMPUTE" block right after the Stage 2 setup
      (post-freeze, post-optimizer-build, pre-loop) that:
        * checks for an on-disk cache (default: ``/content/stage1_cache.pt``)
        * if missing, runs ``precompute_stage1_outputs`` and saves
        * builds a ``TensorDataset`` + standard ``DataLoader`` over it
    - Replace the ``train_epoch_stage2(encoder=..., rcn_runner=..., ...)``
      call inside the for-loop with ``train_epoch_stage2_cached(
      diffusion_decoder=diffusion, optimizer=optimizer_s2,
      cached_dataloader=cached_dataloader, ...)``.
    - Opt-out via ``BS32B_USE_CACHE = False`` global before running the
      cell.

Idempotent — sentinel ``BS32b_PRECOMPUTE``.
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


# ---------------------------------------------------------------------
# Patch 1 — insert precompute + cached dataloader build BEFORE the
# Stage 2 training header. Anchor: ``# -------- Stage 2 training --------``
# ---------------------------------------------------------------------

OLD_S2_HEADER = '''# -------- Stage 2 training --------
print("\\n" + "=" * 80)
print("🚀 STAGE 2 — EDM Diffusion on small residual (concat conditioning)")
print("=" * 80)'''


NEW_S2_HEADER = '''# >>> BS32b_PRECOMPUTE
# Pre-cache Stage 1 outputs (μ_HR, baseline_log, δ_target, valid_mask)
# once and switch Stage 2 onto a thin training loop with full-batch
# diffusion forward (bs=BATCH_SIZE instead of 64×bs=1). See
# stage2_bottleneck_analysis.md §6 and architecture_journey.md §6.
import torch as _torch_bs32b
from torch.utils.data import TensorDataset as _TD_bs32b, DataLoader as _DL_bs32b
from st_cdgm.training.two_stage import (
    precompute_stage1_outputs as _bs32b_precompute,
    train_epoch_stage2_cached as _bs32b_cached_step,
)

_BS32B_USE_CACHE = bool(globals().get("BS32B_USE_CACHE", True))
_BS32B_CACHE_PATH = Path(globals().get("BS32B_CACHE_PATH", "/content/stage1_cache.pt"))

cached_dataloader = None
if _BS32B_USE_CACHE:
    if _BS32B_CACHE_PATH.exists():
        print(f"📦 BS32b — cache trouvé : {_BS32B_CACHE_PATH} "
              f"({_BS32B_CACHE_PATH.stat().st_size/1e9:.2f} GB)")
        _bs32b_cache = _torch_bs32b.load(
            _BS32B_CACHE_PATH, map_location="cpu", weights_only=False
        )
    else:
        print(f"📦 BS32b — cache absent, calcul Stage 1 sur train_dataset…")
        _bs32b_cache = _bs32b_precompute(
            encoder=encoder,
            rcn_runner=rcn_runner,
            regression_head=regression_head,
            train_dataset=train_dataset,
            iterate_batches_fn=lambda s: convert_sample_to_batch(s, builder, DEVICE),
            device=DEVICE,
        )
        try:
            _BS32B_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _torch_bs32b.save(_bs32b_cache, _BS32B_CACHE_PATH)
            print(f"✓ BS32b cache écrit : {_BS32B_CACHE_PATH} "
                  f"({_BS32B_CACHE_PATH.stat().st_size/1e9:.2f} GB)")
        except Exception as _e:
            print(f"⚠️  BS32b — sauvegarde cache échouée : {_e}")

    _N_cache = _bs32b_cache["mu_HR"].shape[0]
    print(f"   cache size : {_N_cache} samples, "
          f"shape mu_HR={tuple(_bs32b_cache['mu_HR'].shape)}")

    class _BS32bCachedDataset(_torch_bs32b.utils.data.Dataset):
        def __init__(self, cache):
            self.mu = cache["mu_HR"]
            self.base = cache["baseline_log"]
            self.delta = cache["delta_target"]
            self.mask = cache["valid_mask"]

        def __len__(self):
            return self.mu.shape[0]

        def __getitem__(self, idx):
            return {
                "mu_HR": self.mu[idx],
                "baseline_log": self.base[idx],
                "delta_target": self.delta[idx],
                "valid_mask": self.mask[idx],
            }

    _bs32b_dataset = _BS32bCachedDataset(_bs32b_cache)
    cached_dataloader = _DL_bs32b(
        _bs32b_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,         # cache is in RAM, no IO benefit
        pin_memory=PIN_MEMORY,
        drop_last=False,
    )
    print(f"   cached_dataloader prêt : batch_size={BATCH_SIZE}")

# -------- Stage 2 training --------
print("\\n" + "=" * 80)
print("🚀 STAGE 2 — EDM Diffusion on small residual (concat conditioning)")
print("=" * 80)'''


# ---------------------------------------------------------------------
# Patch 2 — swap the training call inside the Stage 2 loop. If the
# cached dataloader is available, use it; otherwise fall back to the
# legacy per-micro path.
# ---------------------------------------------------------------------

OLD_S2_CALL = '''    s2_metrics = train_epoch_stage2(
        encoder=encoder,
        rcn_runner=rcn_runner,
        regression_head=regression_head,
        diffusion_decoder=diffusion,
        optimizer=optimizer_s2,
        data_loader=iterate_batches(train_dataloader, builder, DEVICE),
        device=DEVICE,
        gradient_clipping=CONFIG.training.gradient_clipping,
        log_interval=CONFIG.training.log_every,
        use_amp=CONFIG.training.get("use_amp", True),
    )'''


NEW_S2_CALL = '''    if cached_dataloader is not None:
        s2_metrics = _bs32b_cached_step(
            diffusion_decoder=diffusion,
            optimizer=optimizer_s2,
            cached_dataloader=cached_dataloader,
            device=DEVICE,
            use_amp=CONFIG.training.get("use_amp", True),
            gradient_clipping=CONFIG.training.gradient_clipping,
            log_every=int(CONFIG.training.get("log_every", 20)),
        )
    else:
        s2_metrics = train_epoch_stage2(
            encoder=encoder,
            rcn_runner=rcn_runner,
            regression_head=regression_head,
            diffusion_decoder=diffusion,
            optimizer=optimizer_s2,
            data_loader=iterate_batches(train_dataloader, builder, DEVICE),
            device=DEVICE,
            gradient_clipping=CONFIG.training.gradient_clipping,
            log_interval=CONFIG.training.log_every,
            use_amp=CONFIG.training.get("use_amp", True),
        )'''


def patch_cell_50() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "TWO_STAGE_TRAINING_LOOP" in s and "for s2_epoch" in s)
    if idx is None:
        print("  ! cell 50 not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "BS32b_PRECOMPUTE" in src:
        print(f"  = cell {idx} already patched (BS32b)")
        return 0

    n = 0
    if OLD_S2_HEADER in src:
        src = src.replace(OLD_S2_HEADER, NEW_S2_HEADER, 1)
        n += 1
        print(f"  ~ cell {idx}: precompute block inserted before Stage 2 header")
    else:
        print(f"  ! cell {idx}: Stage 2 header anchor not found")

    if OLD_S2_CALL in src:
        src = src.replace(OLD_S2_CALL, NEW_S2_CALL, 1)
        n += 1
        print(f"  ~ cell {idx}: Stage 2 training call swapped to cached path")
    else:
        print(f"  ! cell {idx}: Stage 2 training call anchor not found")

    if n > 0:
        cells[idx]["source"] = src.splitlines(keepends=True)
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def main() -> int:
    print("=== BS32b : pre-cache Stage 1 + thin batched Stage 2 (cell 50) ===")
    n = patch_cell_50()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
