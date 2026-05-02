"""
BS12 — wire anti-collapse args (lambda_dag_prior, dag_prior, dag_grad_gate_value,
abort_on_collapse, collapse_threshold) into TWO_STAGE_TRAINING_LOOP cell.

Adds the dag_prior tensor build right after ts_cfg lookup, and extends the
``train_epoch_stage1(...)`` call with the four new kwargs. Defaults are read
from the YAML config (config/training_config.yaml — fields ``loss.dag_prior``
and ``two_stage.stage1.{lambda_dag_prior,dag_grad_gate_value,
abort_on_collapse,collapse_threshold}``).

Idempotent — sentinel TWO_STAGE_BS12_DAG_HEALTH.
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


# Build a torch tensor of CONFIG.loss.dag_prior right after the legacy
# ``S2_EPOCHS = ...`` line so it is available for both Stage 1 epochs.
DAG_PRIOR_BUILD = '''ts_cfg = CONFIG.two_stage
S1_EPOCHS = int(ts_cfg.stage1.epochs_max)
S2_EPOCHS = int(ts_cfg.stage2.epochs_max)

# >>> TWO_STAGE_BS12_DAG_HEALTH
# Anti-collapse: pass a physical prior matrix + dag_grad_gate=1 + soft L1 to
# train_epoch_stage1 so A_dag receives positive supervision and is no longer
# detached from the SCM gradient path. See hyperplan §15 (DAG health) and
# the Colab logs of 2026-05-02 (epoch 1: ||A||_F 0.34 → 0.0009).
import torch as _torch_bs12
_dag_prior_tensor = None
try:
    _prior_list = CONFIG.get("loss", {}).get("dag_prior")
    if _prior_list is not None:
        _dag_prior_tensor = _torch_bs12.tensor(_prior_list, dtype=_torch_bs12.float32)
        print(f"📐 dag_prior chargé : shape {tuple(_dag_prior_tensor.shape)}")
except Exception as _e_bs12:
    print(f"⚠️  dag_prior non chargé : {_e_bs12}")

_lambda_dag_prior = float(ts_cfg.stage1.get("lambda_dag_prior", 0.0))
_dag_grad_gate_value = float(ts_cfg.stage1.get("dag_grad_gate_value", 1.0))
_abort_on_collapse = bool(ts_cfg.stage1.get("abort_on_collapse", True))
_collapse_threshold = float(ts_cfg.stage1.get("collapse_threshold", 0.05))
print(
    f"🩺 anti-collapse: λ_prior={_lambda_dag_prior}, "
    f"gate={_dag_grad_gate_value}, abort={_abort_on_collapse}, "
    f"threshold={_collapse_threshold}"
)
'''


# The old call signature.
OLD_CALL = '''    s1_metrics = train_epoch_stage1(
        encoder=encoder,
        rcn_runner=rcn_runner,
        regression_head=regression_head,
        optimizer=optimizer_s1,
        data_loader=iterate_batches(train_dataloader, builder, DEVICE),
        device=DEVICE,
        epoch_idx=s1_epoch,
        lambda_reg=float(ts_cfg.stage1.lambda_reg),
        beta_rec=float(ts_cfg.stage1.beta_rec),
        gamma_dag_max=float(ts_cfg.stage1.gamma_dag_max),
        gamma_dag_warmup_epochs=int(ts_cfg.stage1.gamma_dag_warmup_epochs),
        lambda_l1=float(ts_cfg.stage1.lambda_l1),
        gradient_clipping=CONFIG.training.gradient_clipping,
        log_interval=CONFIG.training.log_every,
        use_amp=CONFIG.training.get("use_amp", True),
    )'''


NEW_CALL = '''    s1_metrics = train_epoch_stage1(
        encoder=encoder,
        rcn_runner=rcn_runner,
        regression_head=regression_head,
        optimizer=optimizer_s1,
        data_loader=iterate_batches(train_dataloader, builder, DEVICE),
        device=DEVICE,
        epoch_idx=s1_epoch,
        lambda_reg=float(ts_cfg.stage1.lambda_reg),
        beta_rec=float(ts_cfg.stage1.beta_rec),
        gamma_dag_max=float(ts_cfg.stage1.gamma_dag_max),
        gamma_dag_warmup_epochs=int(ts_cfg.stage1.gamma_dag_warmup_epochs),
        lambda_l1=float(ts_cfg.stage1.lambda_l1),
        # BS12 anti-collapse:
        lambda_dag_prior=_lambda_dag_prior,
        dag_prior=_dag_prior_tensor,
        dag_grad_gate_value=_dag_grad_gate_value,
        abort_on_collapse=_abort_on_collapse,
        collapse_threshold=_collapse_threshold,
        gradient_clipping=CONFIG.training.gradient_clipping,
        log_interval=CONFIG.training.log_every,
        use_amp=CONFIG.training.get("use_amp", True),
    )'''


def patch_bs12() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(
        cells,
        lambda s: "TWO_STAGE_TRAINING_LOOP" in s and "train_epoch_stage1(" in s,
    )
    if idx is None:
        print("  ! TWO_STAGE_TRAINING_LOOP cell not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "TWO_STAGE_BS12_DAG_HEALTH" in src:
        print("  = BS12 already patched")
        return 0

    n = 0
    if "TWO_STAGE_BS12_DAG_HEALTH" not in src:
        old_top = (
            "ts_cfg = CONFIG.two_stage\n"
            "S1_EPOCHS = int(ts_cfg.stage1.epochs_max)\n"
            "S2_EPOCHS = int(ts_cfg.stage2.epochs_max)\n"
        )
        if old_top in src:
            src = src.replace(old_top, DAG_PRIOR_BUILD, 1)
            n += 1
            print("  ~ BS12 dag_prior + anti-collapse args inserted")
        else:
            print("  ! BS12 OLD top block not found")

    if OLD_CALL in src:
        src = src.replace(OLD_CALL, NEW_CALL, 1)
        n += 1
        print("  ~ BS12 train_epoch_stage1 call extended")
    else:
        print("  ! BS12 OLD_CALL pattern not found")

    if n > 0:
        cells[idx]["source"] = src.splitlines(keepends=True)
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def main() -> int:
    print("=== BS12 : DAG anti-collapse args wired into Stage 1 cell ===")
    n = patch_bs12()
    print(f"\n{n} modification(s) appliquée(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
