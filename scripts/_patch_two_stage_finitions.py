"""
Finitions Two-Stage : couvre les 2 gaps critiques + 1 mineur identifiés
post-Phase A.

Gap #1 : PERSIST_HELPERS doit aussi sauvegarder/charger
``regression_head_state_dict`` — sinon checkpoint Stage 1 perdu au reload.

Gap #2 : Validation notebook doit
  (a) construire ``regression_head`` dans la cellule diffusion rebuild
  (b) charger ses poids depuis le checkpoint
  (c) calculer ``μ_HR`` et passer ``mu_HR=`` + ``baseline_log=`` à
      ``diffusion.sample(...)`` dans ``generate_prediction_stable``.

Toutes les éditions sentinellées → idempotent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"
VAL_NB = ROOT / "st_cdgm_validation_inference.ipynb"


# =====================================================================
# Helper functions
# =====================================================================


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


def _make_code_cell(src, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


# =====================================================================
# Gap #1 — Persistence : étend PERSIST_HELPERS pour regression_head
# =====================================================================


def patch_persist_helpers_regression_head() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    idx = _find_cell(cells, lambda s: "# >>> PERSIST_HELPERS" in s)
    if idx is None:
        print("  ! PERSIST_HELPERS introuvable")
        return 0

    src = "".join(cells[idx]["source"])
    if "regression_head_state_dict" in src:
        print("  = PERSIST_HELPERS already saves regression_head")
        return 0

    # ---- save side : add regression_head save ----
    SAVE_OLD = (
        '    if "spatial_projector" in globals() and spatial_projector is not None:\n'
        '        payload["spatial_projector_state_dict"] = _persist_state_dict(spatial_projector)\n'
        '    if "hr_ident_head" in globals() and hr_ident_head is not None:\n'
        '        payload["hr_ident_head_state_dict"] = _persist_state_dict(hr_ident_head)\n'
    )
    SAVE_NEW = (
        '    if "spatial_projector" in globals() and spatial_projector is not None:\n'
        '        payload["spatial_projector_state_dict"] = _persist_state_dict(spatial_projector)\n'
        '    if "hr_ident_head" in globals() and hr_ident_head is not None:\n'
        '        payload["hr_ident_head_state_dict"] = _persist_state_dict(hr_ident_head)\n'
        '    # >>> TWO_STAGE_PERSIST_REGRESSION_HEAD\n'
        '    # Stage 1 weights : indispensable au reload (checkpoint sans ça\n'
        '    # = causalité perdue, regression_head ré-init aléatoire).\n'
        '    if "regression_head" in globals() and regression_head is not None:\n'
        '        payload["regression_head_state_dict"] = _persist_state_dict(regression_head)\n'
    )

    # ---- load side : add regression_head load ----
    LOAD_OLD = (
        '    if "spatial_projector" in globals() and spatial_projector is not None:\n'
        '        _persist_load_state_dict(spatial_projector, ckpt.get("spatial_projector_state_dict"))\n'
        '    if "hr_ident_head" in globals() and hr_ident_head is not None:\n'
        '        _persist_load_state_dict(hr_ident_head, ckpt.get("hr_ident_head_state_dict"))\n'
    )
    LOAD_NEW = (
        '    if "spatial_projector" in globals() and spatial_projector is not None:\n'
        '        _persist_load_state_dict(spatial_projector, ckpt.get("spatial_projector_state_dict"))\n'
        '    if "hr_ident_head" in globals() and hr_ident_head is not None:\n'
        '        _persist_load_state_dict(hr_ident_head, ckpt.get("hr_ident_head_state_dict"))\n'
        '    if "regression_head" in globals() and regression_head is not None:\n'
        '        _persist_load_state_dict(regression_head, ckpt.get("regression_head_state_dict"))\n'
    )

    new_src = src
    if SAVE_OLD in new_src:
        new_src = new_src.replace(SAVE_OLD, SAVE_NEW, 1)
    else:
        print("  ! save block pattern not found — manual review needed")
        return 0

    if LOAD_OLD in new_src:
        new_src = new_src.replace(LOAD_OLD, LOAD_NEW, 1)
    else:
        print("  ! load block pattern not found — manual review needed")
        return 0

    cells[idx]["source"] = new_src.splitlines(keepends=True)
    cells[idx]["outputs"] = []
    cells[idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ PERSIST_HELPERS extended for regression_head (cell {idx})")
    return 1


# =====================================================================
# Gap #2 — Validation notebook two-stage support
# =====================================================================


VAL_REGRESSION_HEAD_BUILD = '''# >>> TWO_STAGE_VAL_REGRESSION_HEAD
# Build the GraphToGridDecoder for two-stage validation.
# Loads weights from the same checkpoint that holds encoder + RCN +
# diffusion. Falls back to None if checkpoint predates two-stage.
from st_cdgm.models.regression_head import GraphToGridDecoder

regression_head = None
if CONFIG.get("two_stage", {}).get("enabled", False):
    rh_cfg = CONFIG.two_stage.regression_head
    regression_head = GraphToGridDecoder(
        d_model=int(rh_cfg.d_model),
        hr_h=int(CONFIG.diffusion.height),
        hr_w=int(CONFIG.diffusion.width),
        intermediate_h=int(rh_cfg.intermediate_h),
        intermediate_w=int(rh_cfg.intermediate_w),
        n_heads=int(rh_cfg.n_heads),
        refine_channels=int(rh_cfg.refine_channels),
        output_channels=1,
    ).to(DEVICE)

    rh_state = checkpoint.get("regression_head_state_dict")
    if rh_state is not None:
        regression_head.load_state_dict(strip_torch_compile_prefix(rh_state))
        regression_head.eval()
        print(f"✓ regression_head reconstruit + chargé "
              f"({regression_head.num_params():,} params)")
    else:
        print("⚠️  Checkpoint sans regression_head_state_dict — "
              "validation pre-two-stage. μ_HR sera ré-init aléatoire.")
else:
    print("ℹ️  two_stage.enabled=False — regression_head non instancié.")
'''


VAL_GENERATE_PREDICTION_PATCH_OLD = '''    H_last = seq_output.states[-1]
    conditioning = encoder.project_state_tensor(H_last).to(DEVICE)
    # Sprint 1 / B4 fix: use SpatialConditioningProjector at inference if the
    # checkpoint provides one; falls back to mean-pooled tokens otherwise.
    # >>> VALIDATION_PROJECTOR_CALL'''


VAL_GENERATE_PREDICTION_PATCH_NEW = '''    H_last = seq_output.states[-1]
    conditioning = encoder.project_state_tensor(H_last).to(DEVICE)

    # >>> TWO_STAGE_VAL_GENERATE
    # In two-stage mode, compute mu_HR via the regression head and pass it +
    # baseline_log (= the log1p baseline already provided by the pipeline) to
    # the diffusion sampler via channel concat. Falls back to legacy
    # cross-attn path when two_stage is disabled or regression_head is None.
    _two_stage_active = (
        CONFIG.get("two_stage", {}).get("enabled", False)
        and globals().get("regression_head", None) is not None
        and getattr(diffusion, "causal_concat", False)
    )
    mu_HR_for_sample = None
    baseline_log_for_sample = None
    if _two_stage_active:
        mu_HR_for_sample = regression_head(H_last)
        if mu_HR_for_sample.shape != baseline_batch.shape:
            mu_HR_for_sample = torch.nn.functional.interpolate(
                mu_HR_for_sample, size=baseline_batch.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        baseline_log_for_sample = baseline_batch  # already log1p in pipeline

    # Sprint 1 / B4 fix: use SpatialConditioningProjector at inference if the
    # checkpoint provides one; falls back to mean-pooled tokens otherwise.
    # >>> VALIDATION_PROJECTOR_CALL'''


VAL_DIFFUSION_SAMPLE_PATCH_OLD = '''        generated = diffusion.sample(
            conditioning,
            num_steps=EVAL_NUM_STEPS,
            scheduler_type=EVAL_SCHEDULER,
            apply_constraints=False,
            baseline=baseline_batch,
            conditioning_spatial=conditioning_spatial,
        )'''


VAL_DIFFUSION_SAMPLE_PATCH_NEW = '''        # >>> TWO_STAGE_VAL_DIFFUSION_SAMPLE
        if _two_stage_active:
            generated = diffusion.sample(
                conditioning=None,
                num_steps=EVAL_NUM_STEPS,
                scheduler_type=CONFIG.diffusion.get("scheduler_type", "edm_karras"),
                apply_constraints=False,
                baseline=baseline_batch,
                mu_HR=mu_HR_for_sample,
                baseline_log=baseline_log_for_sample,
            )
        else:
            generated = diffusion.sample(
                conditioning,
                num_steps=EVAL_NUM_STEPS,
                scheduler_type=EVAL_SCHEDULER,
                apply_constraints=False,
                baseline=baseline_batch,
                conditioning_spatial=conditioning_spatial,
            )'''


def patch_validation_two_stage() -> int:
    """Patch validation notebook for two-stage support.

    Inserts:
      - regression_head build cell after the diffusion rebuild
      - patches generate_prediction_stable + diffusion.sample call
    """
    nb = json.loads(VAL_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    # 1. Insert regression_head build cell AFTER VALIDATION_DIFF_REBUILD
    if _find_cell(cells, lambda s: "TWO_STAGE_VAL_REGRESSION_HEAD" in s) is None:
        diff_idx = _find_cell(
            cells,
            lambda s: "VALIDATION_DIFF_REBUILD" in s
            and "diffusion = CausalDiffusionDecoder(" in s,
        )
        if diff_idx is not None:
            # Insert AFTER the diffusion build cell (before the helpers cell)
            cells.insert(
                diff_idx + 1,
                _make_code_cell(VAL_REGRESSION_HEAD_BUILD, "two_stage_val_reg_head"),
            )
            n_changed += 1
            print(f"  + val regression_head build inséré en cell {diff_idx + 1}")

    # 2. Patch generate_prediction_stable + diffusion.sample call
    helpers_idx = _find_cell(
        cells, lambda s: "def generate_prediction_stable" in s
    )
    if helpers_idx is None:
        print("  ! generate_prediction_stable introuvable")
        return n_changed

    src = "".join(cells[helpers_idx]["source"])
    needs_save = False

    if "TWO_STAGE_VAL_GENERATE" not in src:
        if VAL_GENERATE_PREDICTION_PATCH_OLD in src:
            src = src.replace(
                VAL_GENERATE_PREDICTION_PATCH_OLD,
                VAL_GENERATE_PREDICTION_PATCH_NEW,
                1,
            )
            print(f"  ~ generate_prediction_stable patched for two-stage")
            needs_save = True
        else:
            print("  ! generate_prediction_stable old pattern not found")
    else:
        print("  = generate_prediction_stable already two-stage-aware")

    if "TWO_STAGE_VAL_DIFFUSION_SAMPLE" not in src:
        if VAL_DIFFUSION_SAMPLE_PATCH_OLD in src:
            src = src.replace(
                VAL_DIFFUSION_SAMPLE_PATCH_OLD,
                VAL_DIFFUSION_SAMPLE_PATCH_NEW,
                1,
            )
            print(f"  ~ diffusion.sample call patched for two-stage")
            needs_save = True
        else:
            print("  ! diffusion.sample old pattern not found")
    else:
        print("  = diffusion.sample call already two-stage-aware")

    if needs_save:
        cells[helpers_idx]["source"] = src.splitlines(keepends=True)
        cells[helpers_idx]["outputs"] = []
        cells[helpers_idx]["execution_count"] = None
        n_changed += 1

    if n_changed > 0:
        VAL_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n_changed


def main() -> int:
    print("=== Gap #1 : PERSIST_HELPERS regression_head ===")
    n1 = patch_persist_helpers_regression_head()

    print("\n=== Gap #2 : Validation notebook two-stage ===")
    n2 = patch_validation_two_stage()

    print(f"\n{n1 + n2} modification(s) appliquée(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
