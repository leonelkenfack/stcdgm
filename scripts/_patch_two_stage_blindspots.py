"""
Fix the four blindspots identified in the XRAY analysis post-Phase A.

BS1 — torch.compile applied to ``diffusion_base.forward()`` only, bypassing
      ``forward_edm()`` which is what two-stage uses. Fix: compile
      ``diffusion.unet`` directly so both paths benefit.

BS2 — ``persist_epoch_checkpoint`` reads the global ``optimizer`` (legacy)
      instead of ``optimizer_s2``. Fix: alias ``optimizer = optimizer_s2``
      at the start of Stage 2.

BS3 — ``TWO_STAGE_TRAINING_LOOP`` always restarts at Stage 1 epoch 0 on
      re-run. Fix: add ``SKIP_STAGE_1`` flag with manual override + auto-
      detection from checkpoint epoch counter.

BS4 — Legacy single-stage training cell still in notebook can corrupt the
      two-stage checkpoint if user runs it after TWO_STAGE_TRAINING_LOOP.
      Fix: add a guard at the top that skips the cell when
      ``two_stage.enabled=true``.

Idempotent — sentinel-guarded.
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


# =====================================================================
# BS1 — compile self.unet directly
# =====================================================================


def patch_bs1_compile_unet() -> int:
    """The compile cell currently wraps ``diffusion_base`` (the whole
    decoder). Two-stage uses ``forward_edm`` which bypasses ``forward()``,
    so the compile gives no speedup.

    Fix: compile ``diffusion_base.unet`` instead, so the inner UNet's
    forward (called by both ``forward`` and ``forward_edm``) benefits.
    """
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    idx = _find_cell(
        cells,
        lambda s: "diffusion_compiled = torch.compile(diffusion_base"
        in s,
    )
    if idx is None:
        print("  ! BS1 compile cell not found")
        return 0

    src = "".join(cells[idx]["source"])
    if "TWO_STAGE_BS1_COMPILE_UNET" in src:
        print("  = BS1 already patched")
        return 0

    OLD_BLOCK = '''    try:
        diffusion_compiled = torch.compile(diffusion_base, mode=diff_mode, fullgraph=False)
        if hasattr(diffusion, "module"):
            gpus = CONFIG.training.multi_gpu.get("gpus", [0])
            diffusion = torch.nn.DataParallel(diffusion_compiled, device_ids=gpus)
        else:
            diffusion = diffusion_compiled
        print(f"   ✓ diffusion compiled (mode={diff_mode})")
    except Exception as _e:
        print(f"⚠️  torch.compile(diffusion) a échoué : {_e} — fallback eager.")'''

    NEW_BLOCK = '''    try:
        # >>> TWO_STAGE_BS1_COMPILE_UNET
        # Compile the inner UNet directly so BOTH `forward()` (legacy DDPM)
        # AND `forward_edm()` (two-stage EDM) benefit. Wrapping the whole
        # CausalDiffusionDecoder only intercepts `__call__` -> `forward`,
        # which leaves `forward_edm` running in eager mode (5-10× slower
        # on A100 for 30k+ pixel UNets).
        diffusion_base.unet = torch.compile(
            diffusion_base.unet, mode=diff_mode, fullgraph=False
        )
        # ``diffusion`` itself is left untouched — its python methods stay
        # eager but every call to ``self.unet(...)`` (the heavy compute)
        # routes through the compiled version. State_dict load/save and
        # checkpoint resumption work normally.
        print(f"   ✓ diffusion.unet compiled (mode={diff_mode}) — applies to forward() AND forward_edm()")
    except Exception as _e:
        print(f"⚠️  torch.compile(diffusion.unet) a échoué : {_e} — fallback eager.")'''

    if OLD_BLOCK not in src:
        print("  ! BS1 OLD_BLOCK pattern not found — manual review needed")
        return 0

    new_src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    cells[idx]["source"] = new_src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ BS1 patched (cell {idx}): compile self.unet directly")
    return 1


# =====================================================================
# BS2 + BS3 — TWO_STAGE_TRAINING_LOOP : alias optimizer + SKIP_STAGE_1
# =====================================================================


def patch_bs2_bs3_two_stage_loop() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    idx = _find_cell(cells, lambda s: "TWO_STAGE_TRAINING_LOOP" in s)
    if idx is None:
        print("  ! TWO_STAGE_TRAINING_LOOP cell not found")
        return 0

    src = "".join(cells[idx]["source"])
    if "TWO_STAGE_BS2_OPTIMIZER_ALIAS" in src and "TWO_STAGE_BS3_SKIP_STAGE_1" in src:
        print("  = BS2 + BS3 already patched")
        return 0

    n = 0

    # BS3 : add SKIP_STAGE_1 flag at top, just after the assert
    if "TWO_STAGE_BS3_SKIP_STAGE_1" not in src:
        OLD_TOP = '''assert CONFIG.get("two_stage", {}).get("enabled", False), (
    "TWO_STAGE_TRAINING_LOOP cell requires two_stage.enabled=true"
)

ts_cfg = CONFIG.two_stage'''
        NEW_TOP = '''assert CONFIG.get("two_stage", {}).get("enabled", False), (
    "TWO_STAGE_TRAINING_LOOP cell requires two_stage.enabled=true"
)

# >>> TWO_STAGE_BS3_SKIP_STAGE_1
# Resume control. Set to True manually if you have already trained Stage 1
# and want to skip directly to recalibration + Stage 2 (e.g. after a Drive
# checkpoint reload). Auto-detected from checkpoint epoch counter when
# possible — if epoch_last.pth contains an epoch >= S1_EPOCHS the flag
# is forced to True.
SKIP_STAGE_1 = False

try:
    from pathlib import Path as _Path
    _ck_path = _Path(str(CKPT_SAVE_DIR)) / "epoch_last.pth"
    if _ck_path.exists():
        import torch as _torch_for_resume
        _ck = _torch_for_resume.load(_ck_path, map_location="cpu", weights_only=False)
        _ck_epoch = int(_ck.get("epoch", 0))
        _S1_CAP = int(CONFIG.two_stage.stage1.epochs_max)
        if _ck_epoch >= _S1_CAP:
            print(f"🔁 Auto-detected resume: checkpoint epoch={_ck_epoch} >= "
                  f"stage1.epochs_max={_S1_CAP} → SKIP_STAGE_1 = True")
            SKIP_STAGE_1 = True
        del _ck  # free memory
except Exception as _e:
    print(f"   (resume detection failed: {_e})")

ts_cfg = CONFIG.two_stage'''

        if OLD_TOP in src:
            src = src.replace(OLD_TOP, NEW_TOP, 1)
            n += 1
            print("  ~ BS3 SKIP_STAGE_1 flag inserted")
        else:
            print("  ! BS3 OLD_TOP not found")

    # BS3 : wrap Stage 1 loop in `if not SKIP_STAGE_1:`
    if "if not SKIP_STAGE_1:" not in src:
        OLD_STAGE1_HEADER = '''# -------- Stage 1 training --------
print("\\n" + "=" * 80)
print("🚀 STAGE 1 — Deterministic Causal Mean Prediction")
print("=" * 80)
val_mse_history = []
best_s1_val = math.inf
best_s1_epoch = 0
patience = int(ts_cfg.stage1.early_stop_patience)
no_improve_s1 = 0

for s1_epoch in range(S1_EPOCHS):'''

        NEW_STAGE1_HEADER = '''# -------- Stage 1 training --------
val_mse_history = []
best_s1_val = math.inf
best_s1_epoch = 0
patience = int(ts_cfg.stage1.early_stop_patience)
no_improve_s1 = 0

if SKIP_STAGE_1:
    print("\\n⏭️  SKIP_STAGE_1=True — Stage 1 sauté (resume mode).")
else:
    print("\\n" + "=" * 80)
    print("🚀 STAGE 1 — Deterministic Causal Mean Prediction")
    print("=" * 80)

for s1_epoch in (range(0) if SKIP_STAGE_1 else range(S1_EPOCHS)):'''

        if OLD_STAGE1_HEADER in src:
            src = src.replace(OLD_STAGE1_HEADER, NEW_STAGE1_HEADER, 1)
            n += 1
            print("  ~ BS3 Stage 1 loop wrapped with skip guard")
        else:
            print("  ! BS3 OLD_STAGE1_HEADER not found")

    # BS2 : alias optimizer = optimizer_s2 before Stage 2 loop
    if "TWO_STAGE_BS2_OPTIMIZER_ALIAS" not in src:
        OLD_S2_OPT = '''optimizer_s2 = torch.optim.AdamW(
    diffusion.parameters(),
    lr=float(ts_cfg.stage2.lr),
    weight_decay=1e-4,
)
print(f"🎓 Stage 2 optimizer : AdamW lr={ts_cfg.stage2.lr}, "
      f"params={sum(p.numel() for p in diffusion.parameters()):,}")'''

        NEW_S2_OPT = '''optimizer_s2 = torch.optim.AdamW(
    diffusion.parameters(),
    lr=float(ts_cfg.stage2.lr),
    weight_decay=1e-4,
)
# >>> TWO_STAGE_BS2_OPTIMIZER_ALIAS
# persist_epoch_checkpoint reads the global ``optimizer`` for state_dict
# saving. Alias so the Stage 2 optimizer (which is what is actually
# being trained) gets persisted instead of the stale legacy optimizer.
optimizer = optimizer_s2

print(f"🎓 Stage 2 optimizer : AdamW lr={ts_cfg.stage2.lr}, "
      f"params={sum(p.numel() for p in diffusion.parameters()):,}")'''

        if OLD_S2_OPT in src:
            src = src.replace(OLD_S2_OPT, NEW_S2_OPT, 1)
            n += 1
            print("  ~ BS2 optimizer = optimizer_s2 alias inserted")
        else:
            print("  ! BS2 OLD_S2_OPT not found")

    if n > 0:
        cells[idx]["source"] = src.splitlines(keepends=True)
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


# =====================================================================
# BS4 — Guard legacy training cell against two_stage active
# =====================================================================


def patch_bs4_legacy_guard() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Find the legacy training cell: it has PERSIST_RESUME + train_epoch(
    idx = _find_cell(
        cells,
        lambda s: "PERSIST_RESUME" in s
        and "train_epoch(" in s
        and "TWO_STAGE_TRAINING_LOOP" not in s,
    )
    if idx is None:
        print("  ! BS4 legacy training cell not found")
        return 0

    src = "".join(cells[idx]["source"])
    if "TWO_STAGE_BS4_LEGACY_GUARD" in src:
        print("  = BS4 already patched")
        return 0

    GUARD_BLOCK = '''# >>> TWO_STAGE_BS4_LEGACY_GUARD
# This is the LEGACY single-stage training cell (DDPM/EDM with cross-attn
# conditioning). When two_stage.enabled=true, the dedicated
# TWO_STAGE_TRAINING_LOOP cell is the one that should be run; executing
# this legacy cell on top of it would corrupt the two-stage checkpoint.
if CONFIG.get("two_stage", {}).get("enabled", False):
    print(
        "⏭️  two_stage.enabled=True → cellule LEGACY skipped.\\n"
        "    Le training se fait dans TWO_STAGE_TRAINING_LOOP (cellule au-dessus).\\n"
        "    Mettre two_stage.enabled=false dans le YAML pour repasser en single-stage."
    )
else:
'''

    # Indent the entire current cell content by 4 spaces and prepend the guard.
    indented_lines = []
    for line in src.split("\n"):
        # Only indent non-empty lines; preserve blanks
        if line:
            indented_lines.append("    " + line)
        else:
            indented_lines.append("")
    new_src = GUARD_BLOCK + "\n".join(indented_lines)

    cells[idx]["source"] = new_src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ BS4 legacy training cell guarded (cell {idx})")
    return 1


def main() -> int:
    print("=== BS1 : compile self.unet ===")
    n1 = patch_bs1_compile_unet()
    print("\n=== BS2 + BS3 : optimizer alias + SKIP_STAGE_1 ===")
    n2 = patch_bs2_bs3_two_stage_loop()
    print("\n=== BS4 : legacy training cell guard ===")
    n3 = patch_bs4_legacy_guard()
    print(f"\n{n1 + n2 + n3} modification(s) appliquée(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
