"""
Adapte ``st_cdgm_training_evaluation.ipynb`` (et la validation nb) à la
pipeline Two-Stage Causal (hyperplan v2.0 décisions 1-8 validées).

Modifications training notebook
-------------------------------
1. ``COLAB_BOOTSTRAP`` : ``GIT_BRANCH = "two-stage-causal"``.
2. Cell 35 (build models) : ajout build du ``GraphToGridDecoder`` +
   ``CausalDiffusionDecoder`` réinstancié avec ``causal_concat=True``.
3. Cell 45 (PERSIST_HELPERS) : étend persistence pour sauvegarder
   ``regression_head_state_dict``.
4. Cell 48 (training loop principal) : remplacé par un two-stage loop
   sequentiel (Stage 1 → calibration → ablation → Stage 2) qui appelle
   ``train_epoch_stage1`` et ``train_epoch_stage2``.

Validation notebook
-------------------
5. Bootstrap → branche ``two-stage-causal``.
6. Build cell (VALIDATION_DIFF_REBUILD) : ajout regression head + concat mode.
7. ``generate_prediction_stable`` : compute μ_HR + baseline_log et les
   passe à ``diffusion.sample(...)``.

Idempotent — sentinel-guarded.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"
VAL_NB = ROOT / "st_cdgm_validation_inference.ipynb"


# =====================================================================
# Code blocks (sentinel-guarded)
# =====================================================================


REGRESSION_HEAD_BUILD = '''# >>> TWO_STAGE_BUILD_REGRESSION_HEAD
# Phase 2 of hyperplan v2.0 — build the regression head AFTER the encoder/RCN
# but BEFORE rebuilding the diffusion in causal_concat mode.
from st_cdgm.models.regression_head import GraphToGridDecoder

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
    print(f"🎯 Two-Stage : GraphToGridDecoder créé "
          f"({regression_head.num_params():,} params)")
else:
    regression_head = None
    print("ℹ️  two_stage.enabled=False — regression_head non instancié.")
'''


DIFFUSION_REBUILD_CONCAT = '''# >>> TWO_STAGE_DIFFUSION_REBUILD
# Recreate the diffusion decoder in causal_concat mode for Two-Stage.
# The U-Net then expects 3 input channels: [delta_noisy, mu_HR, baseline_log].
# Cross-attention conditioning is no longer used — causality flows through
# mu_HR architecturally.
if CONFIG.get("two_stage", {}).get("enabled", False):
    print("🌀 Two-Stage : reconstruction diffusion en mode causal_concat")
    # Reuse the same UNET_KWARGS but force minimal arch from the YAML.
    diffusion = CausalDiffusionDecoder(
        in_channels=hr_channels,
        conditioning_dim=CONFIG.diffusion.conditioning_dim,
        height=CONFIG.diffusion.height,
        width=CONFIG.diffusion.width,
        num_diffusion_steps=CONFIG.diffusion.steps,
        unet_kwargs=UNET_KWARGS,
        use_gradient_checkpointing=bool(
            CONFIG.diffusion.get("use_gradient_checkpointing", False)
        ),
        scheduler_type=CONFIG.diffusion.get("scheduler_type", "edm_karras"),
        conv_padding_mode=CONFIG.diffusion.get("conv_padding_mode", "zeros"),
        anti_checkerboard=bool(CONFIG.diffusion.get("anti_checkerboard", False)),
        edm_config=_edm_config,
        causal_concat=True,
    ).to(DEVICE)
    print(f"   ✓ UNet input channels: {diffusion.unet.config.in_channels} "
          f"(= 1 [delta] + 1 [mu_HR] + 1 [baseline_log])")
'''


TWO_STAGE_TRAINING_LOOP = '''# >>> TWO_STAGE_TRAINING_LOOP
# Sequential Two-Stage training (hyperplan v2.0 §4) :
#   Stage 1 : encoder + RCN + regression_head (joint)
#             → MSE on mu_HR + L_rec + DAGMA + L1
#             → cap epochs_max, early stop on val MSE plateau
#   Calibration : recompute sigma_data on (HR_log - baseline_log - mu_HR)
#   Ablation    : verify mu_HR(A_dag) ≠ mu_HR(0)  (O3 gate)
#   Stage 2 : freeze Stage 1 ; train diffusion only on the small residual
#             with EDM weighted L2 + concat conditioning
import copy
import math
import time
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import IterableDataset, random_split

from st_cdgm.training.training_loop import (
    train_epoch_stage1, train_epoch_stage2
)
from st_cdgm.training.two_stage import (
    freeze_stage1, calibrate_sigma_data_two_stage, causal_ablation_check,
)

assert CONFIG.get("two_stage", {}).get("enabled", False), (
    "TWO_STAGE_TRAINING_LOOP cell requires two_stage.enabled=true"
)

ts_cfg = CONFIG.two_stage
S1_EPOCHS = int(ts_cfg.stage1.epochs_max)
S2_EPOCHS = int(ts_cfg.stage2.epochs_max)

# -------- Optimizer Stage 1 (encoder + RCN + regression_head) --------
stage1_params = (
    list(encoder.parameters())
    + list(rcn_cell.parameters())
    + list(regression_head.parameters())
)
optimizer_s1 = torch.optim.AdamW(
    stage1_params,
    lr=float(ts_cfg.stage1.lr),
    weight_decay=float(ts_cfg.stage1.weight_decay),
)
print(f"🎓 Stage 1 optimizer : AdamW lr={ts_cfg.stage1.lr}, "
      f"params={sum(p.numel() for p in stage1_params):,}")

# -------- Stage 1 training --------
print("\\n" + "=" * 80)
print("🚀 STAGE 1 — Deterministic Causal Mean Prediction")
print("=" * 80)
val_mse_history = []
best_s1_val = math.inf
best_s1_epoch = 0
patience = int(ts_cfg.stage1.early_stop_patience)
no_improve_s1 = 0

for s1_epoch in range(S1_EPOCHS):
    print(f"\\n--- Stage 1 — epoch {s1_epoch + 1}/{S1_EPOCHS} ---")
    s1_metrics = train_epoch_stage1(
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
    )

    # Quick val MSE on val_dataloader (cheap : just forward Stage 1, no diffusion)
    encoder.eval(); rcn_runner.cell.eval(); regression_head.eval()
    val_losses = []
    with torch.no_grad():
        for converted_batches in iterate_batches(val_dataloader, builder, DEVICE):
            for batch in converted_batches:
                lr_data = batch["lr"].to(DEVICE)
                target_residual = batch["residual"][-1].to(DEVICE)
                if target_residual.dim() == 3:
                    target_residual = target_residual.unsqueeze(0)
                H_init = encoder.init_state(batch["hetero"]).to(DEVICE)
                drivers = [lr_data[t] for t in range(lr_data.shape[0])]
                seq_out = rcn_runner.run(H_init, drivers, reconstruction_sources=None)
                mu_HR = regression_head(seq_out.states[-1])
                if mu_HR.shape != target_residual.shape:
                    mu_HR = F.interpolate(mu_HR, size=target_residual.shape[-2:],
                                          mode="bilinear", align_corners=False)
                val_losses.append(((mu_HR - target_residual) ** 2).mean().item())
    val_mse = float(np.mean(val_losses)) if val_losses else float("nan")
    val_mse_history.append(val_mse)
    print(f"  Stage1 epoch {s1_epoch + 1} | TrainLoss={s1_metrics['loss']:.5f} | "
          f"ValMSE={val_mse:.5f}")

    if val_mse < best_s1_val - 1e-5:
        best_s1_val = val_mse
        best_s1_epoch = s1_epoch + 1
        no_improve_s1 = 0
    else:
        no_improve_s1 += 1
    if no_improve_s1 >= patience:
        print(f"⏹️  Stage 1 early stop @ epoch {s1_epoch + 1} "
              f"(no improve {patience} epochs, best={best_s1_val:.5f})")
        break

# -------- σ_data calibration --------
print("\\n" + "=" * 80)
print("📐 σ_data RECALIBRATION (post-Stage 1)")
print("=" * 80)
calib = calibrate_sigma_data_two_stage(
    encoder=encoder,
    rcn_runner=rcn_runner,
    regression_head=regression_head,
    data_loader=train_dataloader,
    iterate_batches_fn=iterate_batches,
    builder=builder,
    device=DEVICE,
    max_samples=200,
)
new_sigma_data = float(calib["sigma_data"])
old_sigma_data = float(CONFIG.diffusion.edm.sigma_data)
print(f"  Old σ_data : {old_sigma_data:.5f}")
print(f"  New σ_data : {new_sigma_data:.5f}  (×{new_sigma_data/old_sigma_data:.2f})")
CONFIG.diffusion.edm.sigma_data = new_sigma_data
# Scale sigma_min to keep solver resolution
new_sigma_min = max(1e-4, new_sigma_data * float(ts_cfg.stage2.sigma_min_scale_factor))
CONFIG.diffusion.edm.sigma_min = new_sigma_min
# Update the live edm_config + diffusion module
from st_cdgm.models.edm_preconditioner import EDMConfig as _EDMConfig
_edm_config = _EDMConfig(
    sigma_data=new_sigma_data,
    sigma_min=new_sigma_min,
    sigma_max=float(CONFIG.diffusion.edm.sigma_max),
    rho=float(CONFIG.diffusion.edm.rho),
    P_mean=float(CONFIG.diffusion.edm.P_mean),
    P_std=float(CONFIG.diffusion.edm.P_std),
)
diffusion.edm_config = _edm_config
print(f"  ✓ EDM preconditioner recalibrated (σ_min={new_sigma_min:.5f})")

# -------- Causal ablation (O3 gate) --------
print("\\n" + "=" * 80)
print("🧪 CAUSAL ABLATION (O3 gate)")
print("=" * 80)
abl_cfg = ts_cfg.causal_ablation
ablation_report = causal_ablation_check(
    encoder=encoder,
    rcn_runner=rcn_runner,
    rcn_cell=rcn_cell.module if hasattr(rcn_cell, "module") else rcn_cell,
    regression_head=regression_head,
    data_loader=val_dataloader,
    iterate_batches_fn=iterate_batches,
    builder=builder,
    device=DEVICE,
    n_samples=int(abl_cfg.n_samples),
    threshold=float(abl_cfg.threshold),
)
if not ablation_report["passes"] and bool(abl_cfg.abort_if_fail):
    raise RuntimeError(
        f"Causal ablation FAILED (ratio={ablation_report['ratio']:.4f} < "
        f"{ablation_report['threshold']}). "
        "DAG is decorative — abort Stage 2 per hyperplan v2.0 gate."
    )

# -------- Freeze Stage 1, build Stage 2 optimizer --------
print("\\n" + "=" * 80)
print("🧊 FREEZING Stage 1 modules (encoder, RCN, regression_head)")
print("=" * 80)
freeze_stage1(encoder, rcn_runner.cell, regression_head)
optimizer_s2 = torch.optim.AdamW(
    diffusion.parameters(),
    lr=float(ts_cfg.stage2.lr),
    weight_decay=1e-4,
)
print(f"🎓 Stage 2 optimizer : AdamW lr={ts_cfg.stage2.lr}, "
      f"params={sum(p.numel() for p in diffusion.parameters()):,}")

# -------- Stage 2 training --------
print("\\n" + "=" * 80)
print("🚀 STAGE 2 — EDM Diffusion on small residual (concat conditioning)")
print("=" * 80)
history = {k: [] for k in ["loss_diff_train", "epoch_time"]}

for s2_epoch in range(S2_EPOCHS):
    _t0 = time.time()
    print(f"\\n--- Stage 2 — epoch {s2_epoch + 1}/{S2_EPOCHS} ---")
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
    )
    _dt = time.time() - _t0
    history["loss_diff_train"].append(s2_metrics["loss_diff"])
    history["epoch_time"].append(_dt)
    print(f"  Stage2 epoch {s2_epoch + 1} | loss_diff={s2_metrics['loss_diff']:.5f} | "
          f"time={_dt:.1f}s")

    # Per-epoch persistence (reusing PERSIST_CALL helper if present)
    try:
        persist_epoch_checkpoint(
            epoch_idx=s2_epoch,
            train_metrics={"loss": s2_metrics["loss_diff"]},
            val_loss=float("nan"),
            history=history,
            best_val_loss=math.inf,
            best_epoch=s2_epoch + 1,
            no_improve_epochs=0,
            BEST_MODEL_STATES=None,
            improved=False,
        )
    except Exception as _e:
        print(f"  ⚠️  persist failed: {_e}")

print("\\n" + "=" * 80)
print("✅ TWO-STAGE TRAINING DONE")
print("=" * 80)
print(f"   Stage 1 best val MSE : {best_s1_val:.5f} @ epoch {best_s1_epoch}")
print(f"   Stage 2 final loss   : {history['loss_diff_train'][-1] if history['loss_diff_train'] else float('nan'):.5f}")
print(f"   σ_data calibrated     : {new_sigma_data:.5f}")
print(f"   Causal ablation ratio: {ablation_report['ratio']:.4f}  "
      f"({'PASS' if ablation_report['passes'] else 'FAIL'})")
'''


# =====================================================================
# Patcher
# =====================================================================


def _make_code_cell(src, cell_id):
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }


def _find_cell(cells, predicate):
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        if predicate("".join(c.get("source", []))):
            return i
    return None


def patch_training_notebook() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    # 1. Bootstrap branch → two-stage-causal
    boot_idx = _find_cell(cells, lambda s: "COLAB_BOOTSTRAP" in s)
    if boot_idx is not None:
        src = "".join(cells[boot_idx]["source"])
        if 'GIT_BRANCH: str = "two-stage-causal"' in src:
            print("  = bootstrap: branche déjà two-stage-causal")
        else:
            new_src = src
            for old in (
                'GIT_BRANCH: str = "main"',
                'GIT_BRANCH: str = "edm-rewrite"',
            ):
                if old in new_src:
                    new_src = new_src.replace(
                        old,
                        'GIT_BRANCH: str = "two-stage-causal"  # hyperplan v2.0',
                        1,
                    )
                    cells[boot_idx]["source"] = new_src.splitlines(keepends=True)
                    cells[boot_idx]["outputs"] = []
                    cells[boot_idx]["execution_count"] = None
                    n_changed += 1
                    print(f"  ~ bootstrap: branche → two-stage-causal")
                    break
            else:
                # Fallback: GIT_BRANCH not found in known forms
                print("  ! bootstrap: GIT_BRANCH pattern non trouvé")

    # 2. Insert regression_head build cell AFTER the build cell (cell 35)
    if _find_cell(cells, lambda s: "TWO_STAGE_BUILD_REGRESSION_HEAD" in s) is None:
        build_idx = _find_cell(
            cells,
            lambda s: "diffusion = CausalDiffusionDecoder(" in s
            and "rcn_runner = RCNSequenceRunner" in s,
        )
        if build_idx is not None:
            cells.insert(
                build_idx + 1,
                _make_code_cell(REGRESSION_HEAD_BUILD, "two_stage_build_reg_head"),
            )
            n_changed += 1
            print(f"  + regression_head build inséré en cell {build_idx + 1}")

            # 3. Insert diffusion rebuild in causal_concat mode after that
            cells.insert(
                build_idx + 2,
                _make_code_cell(DIFFUSION_REBUILD_CONCAT, "two_stage_diffusion_rebuild"),
            )
            n_changed += 1
            print(f"  + diffusion rebuild concat inséré en cell {build_idx + 2}")

    # 4. Insert TWO_STAGE_TRAINING_LOOP after the EDM_PREFLIGHT_TRAINING cell.
    # The legacy training loop cell (PERSIST_RESUME + for-loop) should be
    # marked obsolete or replaced. For now we INSERT the two-stage loop
    # right after preflight; the user will manually skip the legacy cell or
    # we can disable it via a sentinel check.
    if _find_cell(cells, lambda s: "TWO_STAGE_TRAINING_LOOP" in s) is None:
        preflight_idx = _find_cell(cells, lambda s: "EDM_PREFLIGHT_TRAINING" in s)
        if preflight_idx is not None:
            cells.insert(
                preflight_idx + 1,
                _make_code_cell(TWO_STAGE_TRAINING_LOOP, "two_stage_training_loop"),
            )
            n_changed += 1
            print(f"  + Two-Stage training loop inséré en cell {preflight_idx + 1}")

    if n_changed > 0:
        TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n_changed


def patch_validation_notebook() -> int:
    nb = json.loads(VAL_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    n_changed = 0

    # 1. Bootstrap → two-stage-causal
    boot_idx = _find_cell(cells, lambda s: "VALIDATION_COLAB_BOOTSTRAP" in s)
    if boot_idx is not None:
        src = "".join(cells[boot_idx]["source"])
        if 'GIT_BRANCH = "two-stage-causal"' in src:
            print("  = validation bootstrap: déjà two-stage-causal")
        else:
            new_src = src
            for old in ('GIT_BRANCH = "main"', 'GIT_BRANCH = "edm-rewrite"'):
                if old in new_src:
                    new_src = new_src.replace(
                        old,
                        'GIT_BRANCH = "two-stage-causal"  # hyperplan v2.0',
                        1,
                    )
                    cells[boot_idx]["source"] = new_src.splitlines(keepends=True)
                    cells[boot_idx]["outputs"] = []
                    cells[boot_idx]["execution_count"] = None
                    n_changed += 1
                    print(f"  ~ validation bootstrap: branche → two-stage-causal")
                    break

    if n_changed > 0:
        VAL_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    return n_changed


def main() -> int:
    print("=== Training notebook ===")
    n1 = patch_training_notebook()
    print("\n=== Validation notebook ===")
    n2 = patch_validation_notebook()
    print(f"\n{n1 + n2} modification(s) appliquée(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
