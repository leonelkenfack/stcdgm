"""
BS18 — per-epoch checkpoint with two-stage resume.

Extends the persistence layer so a Colab crash mid-Stage-1 or mid-Stage-2
no longer wastes the previously completed epochs.

Cell 47 (PERSIST_HELPERS)
-------------------------
- ``_build_epoch_payload`` gains optional ``two_stage_state: dict`` kwarg.
  Stored under ``payload["two_stage"]`` when provided.
- ``persist_epoch_checkpoint`` accepts the same kwarg and forwards it.

Cell 50 (TWO_STAGE_TRAINING_LOOP)
---------------------------------
- New ``TWO_STAGE_BS18_RESUME`` block reads ``two_stage`` from the
  ``epoch_last.pth`` payload and sets:
  - ``_s1_resume_from``  (Stage 1 epochs already done)
  - ``_s2_resume_from``  (Stage 2 epochs already done)
  - ``_skip_calibration``, ``_saved_sigma_data``
  - ``_skip_ablation``, ``_saved_ablation_passed``
- Stage 1 loop ranged from ``_s1_resume_from``.
- Calibration block wrapped in ``if not _skip_calibration``.
- Ablation block wrapped in ``if not _skip_ablation``.
- Stage 2 loop ranged from ``_s2_resume_from``.
- ``persist_epoch_checkpoint(... two_stage_state=...)`` after every epoch.

Idempotent — sentinel TWO_STAGE_BS18_RESUME.
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
# Cell 47 patch — extend persistence helpers
# =====================================================================

OLD_BUILD_SIG = "def _build_epoch_payload(*, epoch_done: int, train_metrics: dict, val_loss: float,\n                         history: dict, best_val_loss: float, best_epoch: int,\n                         no_improve_epochs: int, BEST_MODEL_STATES) -> dict:"
NEW_BUILD_SIG = "def _build_epoch_payload(*, epoch_done: int, train_metrics: dict, val_loss: float,\n                         history: dict, best_val_loss: float, best_epoch: int,\n                         no_improve_epochs: int, BEST_MODEL_STATES,\n                         two_stage_state: 'Optional[dict]' = None) -> dict:"

OLD_BUILD_TAIL = '''    if "regression_head" in globals() and regression_head is not None:
        payload["regression_head_state_dict"] = _persist_state_dict(regression_head)
    return payload'''

NEW_BUILD_TAIL = '''    if "regression_head" in globals() and regression_head is not None:
        payload["regression_head_state_dict"] = _persist_state_dict(regression_head)
    # >>> TWO_STAGE_BS18_PAYLOAD
    # Two-stage state for crash-safe resume (BS18). Tracks which epochs of
    # which stage are done + whether σ_data calibration / O3 ablation have
    # already been performed and accepted.
    if two_stage_state is not None:
        payload["two_stage"] = dict(two_stage_state)
    return payload'''

OLD_PERSIST_SIG = "def persist_epoch_checkpoint(*, epoch_idx: int, train_metrics: dict, val_loss: float,\n                             history: dict, best_val_loss: float, best_epoch: int,\n                             no_improve_epochs: int, BEST_MODEL_STATES,\n                             improved: bool) -> None:"
NEW_PERSIST_SIG = "def persist_epoch_checkpoint(*, epoch_idx: int, train_metrics: dict, val_loss: float,\n                             history: dict, best_val_loss: float, best_epoch: int,\n                             no_improve_epochs: int, BEST_MODEL_STATES,\n                             improved: bool,\n                             two_stage_state: 'Optional[dict]' = None) -> None:"

OLD_PERSIST_BUILD = '''    payload = _build_epoch_payload(
        epoch_done=epoch_idx + 1,
        train_metrics=train_metrics, val_loss=val_loss, history=history,
        best_val_loss=best_val_loss, best_epoch=best_epoch,
        no_improve_epochs=no_improve_epochs,
        BEST_MODEL_STATES=BEST_MODEL_STATES,
    )'''

NEW_PERSIST_BUILD = '''    payload = _build_epoch_payload(
        epoch_done=epoch_idx + 1,
        train_metrics=train_metrics, val_loss=val_loss, history=history,
        best_val_loss=best_val_loss, best_epoch=best_epoch,
        no_improve_epochs=no_improve_epochs,
        BEST_MODEL_STATES=BEST_MODEL_STATES,
        two_stage_state=two_stage_state,
    )'''

OLD_PERSIST_GUARD = '''    if (epoch_idx + 1) % max(1, PERSIST_EVERY) != 0:
        return'''


NEW_PERSIST_GUARD = '''    # BS18: when two_stage_state is provided we always persist (no PERSIST_EVERY
    # gating) so resume can pick up after any single epoch — A100 hours are
    # too expensive to lose. Single-stage path keeps the original throttle.
    if two_stage_state is None and (epoch_idx + 1) % max(1, PERSIST_EVERY) != 0:
        return'''


def patch_cell_47() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "def persist_epoch_checkpoint" in s)
    if idx is None:
        print("  ! persist cell not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "TWO_STAGE_BS18_PAYLOAD" in src:
        print("  = cell 47 already patched")
        return 0
    n = 0
    for old, new, label in (
        (OLD_BUILD_SIG, NEW_BUILD_SIG, "build sig"),
        (OLD_BUILD_TAIL, NEW_BUILD_TAIL, "build tail"),
        (OLD_PERSIST_SIG, NEW_PERSIST_SIG, "persist sig"),
        (OLD_PERSIST_BUILD, NEW_PERSIST_BUILD, "persist build"),
        (OLD_PERSIST_GUARD, NEW_PERSIST_GUARD, "persist guard"),
    ):
        if old in src:
            src = src.replace(old, new, 1)
            n += 1
            print(f"  ~ cell 47: {label}")
        else:
            print(f"  ! cell 47: {label} pattern not found")
    if n > 0:
        cells[idx]["source"] = src.splitlines(keepends=True)
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


# =====================================================================
# Cell 50 patch — resume logic + two_stage_state on persist calls
# =====================================================================

# Insert a resume block right after the BS3 SKIP_STAGE_1 detection.
RESUME_BLOCK = '''
# >>> TWO_STAGE_BS18_RESUME
# Per-epoch resume. Reads ``two_stage`` from the previous run's
# ``epoch_last.pth`` and sets the start indices for Stage 1 / Stage 2,
# plus flags telling whether σ_data calibration and the O3 ablation
# gate have already been performed and accepted (so we don't redo them
# after a Stage 2 mid-run crash).
_s1_resume_from = 0
_s2_resume_from = 0
_skip_calibration = False
_saved_sigma_data = None
_saved_sigma_min = None
_skip_ablation = False
_saved_ablation_ratio = None
try:
    from pathlib import Path as _Path_bs18
    _ck_path_bs18 = _Path_bs18(str(CKPT_SAVE_DIR)) / "epoch_last.pth"
    if _ck_path_bs18.exists():
        import torch as _torch_bs18
        _ck_bs18 = _torch_bs18.load(_ck_path_bs18, map_location="cpu", weights_only=False)
        _ts_state = _ck_bs18.get("two_stage", {}) or {}
        _s1_resume_from = int(_ts_state.get("stage1_epoch_done", 0))
        _s2_resume_from = int(_ts_state.get("stage2_epoch_done", 0))
        _saved_sigma_data = _ts_state.get("sigma_data")
        _saved_sigma_min = _ts_state.get("sigma_min")
        if _saved_sigma_data is not None:
            _skip_calibration = True
        _saved_ablation_ratio = _ts_state.get("ablation_ratio")
        if _ts_state.get("ablation_passed") is True:
            _skip_ablation = True
        del _ck_bs18
except Exception as _e_bs18:
    print(f"   (BS18 resume read failed: {_e_bs18})")

if _s1_resume_from > 0 or _s2_resume_from > 0:
    print(
        f"🔁 BS18 resume: Stage1 done={_s1_resume_from}/{int(ts_cfg.stage1.epochs_max)}, "
        f"Stage2 done={_s2_resume_from}/{int(ts_cfg.stage2.epochs_max)}, "
        f"calib={'skip' if _skip_calibration else 'redo'}, "
        f"ablation={'skip' if _skip_ablation else 'redo'}"
    )
'''


def patch_cell_50() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    idx = _find_cell(cells, lambda s: "TWO_STAGE_TRAINING_LOOP" in s and "for s2_epoch" in s)
    if idx is None:
        print("  ! cell 50 not found")
        return 0
    src = "".join(cells[idx]["source"])
    if "TWO_STAGE_BS18_RESUME" in src:
        print("  = cell 50 already patched")
        return 0
    n = 0

    # 1. Insert RESUME_BLOCK right after the BS3 SKIP_STAGE_1 try/except.
    anchor = '''ts_cfg = CONFIG.two_stage'''
    if anchor in src:
        src = src.replace(anchor, RESUME_BLOCK + "\n" + anchor, 1)
        n += 1
        print("  ~ cell 50: BS18 resume block inserted")

    # 2. Stage 1 loop range: range(0) if SKIP_STAGE_1 else range(_s1_resume_from, S1_EPOCHS)
    OLD_S1_RANGE = "for s1_epoch in (range(0) if SKIP_STAGE_1 else range(S1_EPOCHS)):"
    NEW_S1_RANGE = "for s1_epoch in (range(0) if SKIP_STAGE_1 else range(_s1_resume_from, S1_EPOCHS)):"
    if OLD_S1_RANGE in src:
        src = src.replace(OLD_S1_RANGE, NEW_S1_RANGE, 1)
        n += 1
        print("  ~ cell 50: Stage 1 loop ranged from _s1_resume_from")

    # 3. Persist after Stage 1 epoch (currently no persist call exists on
    #    Stage 1 path — add one right after val_mse_history.append).
    OLD_S1_TAIL = '''val_mse_history.append(val_mse)
    print(f"  Stage1 epoch {s1_epoch + 1} | TrainLoss={s1_metrics['loss']:.5f} | "
          f"ValMSE={val_mse:.5f}")'''
    NEW_S1_TAIL = '''val_mse_history.append(val_mse)
    print(f"  Stage1 epoch {s1_epoch + 1} | TrainLoss={s1_metrics['loss']:.5f} | "
          f"ValMSE={val_mse:.5f}")
    # BS18: persist Stage 1 state every epoch.
    try:
        persist_epoch_checkpoint(
            epoch_idx=s1_epoch,
            train_metrics={"loss": s1_metrics['loss'], "stage": 1},
            val_loss=val_mse,
            history={"val_mse_stage1": val_mse_history},
            best_val_loss=best_s1_val,
            best_epoch=best_s1_epoch,
            no_improve_epochs=no_improve_s1,
            BEST_MODEL_STATES=None,
            improved=(val_mse < best_s1_val),
            two_stage_state={
                "stage": 1,
                "stage1_epoch_done": s1_epoch + 1,
                "stage2_epoch_done": 0,
                "sigma_data": None,
                "ablation_passed": False,
            },
        )
    except Exception as _e_s1p:
        print(f"  ⚠️  Stage 1 persist failed: {_e_s1p}")'''
    if OLD_S1_TAIL in src:
        src = src.replace(OLD_S1_TAIL, NEW_S1_TAIL, 1)
        n += 1
        print("  ~ cell 50: Stage 1 persist call added")

    # 4. Calibration: wrap in `if not _skip_calibration` else restore.
    OLD_CALIB_HEAD = '''calib = calibrate_sigma_data_two_stage('''
    NEW_CALIB_HEAD = '''if _skip_calibration and _saved_sigma_data is not None:
    print(f"  🔁 BS18 resume: σ_data calibration skipped (loaded {float(_saved_sigma_data):.5f})")
    new_sigma_data = float(_saved_sigma_data)
    new_sigma_min = float(_saved_sigma_min if _saved_sigma_min is not None
                          else max(1e-4, new_sigma_data * float(ts_cfg.stage2.sigma_min_scale_factor)))
    old_sigma_data = float(CONFIG.diffusion.edm.sigma_data)
    CONFIG.diffusion.edm.sigma_data = new_sigma_data
    CONFIG.diffusion.edm.sigma_min = new_sigma_min
    from st_cdgm.models.edm_preconditioner import EDMConfig as _EDMConfig
    _edm_config = _EDMConfig(
        sigma_data=new_sigma_data, sigma_min=new_sigma_min,
        sigma_max=float(CONFIG.diffusion.edm.sigma_max), rho=float(CONFIG.diffusion.edm.rho),
        P_mean=float(CONFIG.diffusion.edm.P_mean), P_std=float(CONFIG.diffusion.edm.P_std),
    )
    diffusion.edm_config = _edm_config
    calib = {"sigma_data": new_sigma_data}
else:
    calib = calibrate_sigma_data_two_stage('''
    if OLD_CALIB_HEAD in src:
        src = src.replace(OLD_CALIB_HEAD, NEW_CALIB_HEAD, 1)
        n += 1
        print("  ~ cell 50: calibration wrapped with skip-on-resume")

    # 5. Ablation: wrap in `if not _skip_ablation`.
    OLD_ABL_HEAD = '''ablation_report = causal_ablation_check('''
    NEW_ABL_HEAD = '''if _skip_ablation:
    print(f"  🔁 BS18 resume: O3 ablation skipped (passed previously, ratio={_saved_ablation_ratio})")
    ablation_report = {"passes": True, "ratio": float(_saved_ablation_ratio or 0.0),
                        "threshold": float(abl_cfg.threshold)}
else:
    ablation_report = causal_ablation_check('''
    if OLD_ABL_HEAD in src:
        src = src.replace(OLD_ABL_HEAD, NEW_ABL_HEAD, 1)
        n += 1
        print("  ~ cell 50: ablation wrapped with skip-on-resume")

    # 6. Stage 2 loop ranged from _s2_resume_from.
    OLD_S2_RANGE = "for s2_epoch in range(S2_EPOCHS):"
    NEW_S2_RANGE = "for s2_epoch in range(_s2_resume_from, S2_EPOCHS):"
    if OLD_S2_RANGE in src:
        src = src.replace(OLD_S2_RANGE, NEW_S2_RANGE, 1)
        n += 1
        print("  ~ cell 50: Stage 2 loop ranged from _s2_resume_from")

    # 7. Stage 2 persist call: include two_stage_state.
    OLD_S2_PERSIST = '''persist_epoch_checkpoint(
            epoch_idx=s2_epoch,
            train_metrics={"loss": s2_metrics["loss_diff"]},
            val_loss=float("nan"),
            history=history,
            best_val_loss=math.inf,
            best_epoch=s2_epoch + 1,
            no_improve_epochs=0,
            BEST_MODEL_STATES=None,
            improved=False,
        )'''
    NEW_S2_PERSIST = '''persist_epoch_checkpoint(
            epoch_idx=s2_epoch,
            train_metrics={"loss": s2_metrics["loss_diff"], "stage": 2},
            val_loss=float("nan"),
            history=history,
            best_val_loss=math.inf,
            best_epoch=s2_epoch + 1,
            no_improve_epochs=0,
            BEST_MODEL_STATES=None,
            improved=False,
            two_stage_state={
                "stage": 2,
                "stage1_epoch_done": int(ts_cfg.stage1.epochs_max),
                "stage2_epoch_done": s2_epoch + 1,
                "sigma_data": float(new_sigma_data),
                "sigma_min": float(new_sigma_min),
                "ablation_passed": bool(ablation_report.get("passes", False)),
                "ablation_ratio": float(ablation_report.get("ratio", 0.0)),
            },
        )'''
    if OLD_S2_PERSIST in src:
        src = src.replace(OLD_S2_PERSIST, NEW_S2_PERSIST, 1)
        n += 1
        print("  ~ cell 50: Stage 2 persist call extended with two_stage_state")

    if n > 0:
        cells[idx]["source"] = src.splitlines(keepends=True)
        cells[idx]["outputs"] = []
        cells[idx]["execution_count"] = None
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n


def main() -> int:
    print("=== BS18a : extend persist helpers (cell 47) ===")
    n1 = patch_cell_47()
    print("\n=== BS18b/c/d : resume logic in cell 50 ===")
    n2 = patch_cell_50()
    print(f"\n{n1 + n2} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
