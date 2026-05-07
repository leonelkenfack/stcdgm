"""
BS25 — adapt cell 53 (loss curves) to the two-stage history schema.

The legacy single-stage training populated::

    history["loss"], history["loss_gen"], history["loss_rec"], history["loss_dag"]

Two-stage training populates instead::

    val_mse_history          # list, Stage 1 ValMSE per epoch
    history["loss_diff_train"]  # list, Stage 2 diffusion loss per epoch
    history["epoch_time"]       # list, wall-clock per epoch

Cell 53 was a hard-coded 2x2 plot of the legacy keys → ``KeyError: 'loss'``.

This patch rewrites cell 53 to:
- detect which schema is present (legacy vs two-stage),
- render Stage 1 ValMSE + Stage 2 diffusion loss on the two-stage path,
- fall back to the legacy 4-panel plot if those keys still exist,
- emit a clear message when neither schema is populated (e.g. cell run
  before training completed or after a fresh kernel).

Idempotent — sentinel ``BS25_PLOTS_TWO_STAGE``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NB = ROOT / "st_cdgm_training_evaluation.ipynb"

NEW_SOURCE = '''# >>> BS25_PLOTS_TWO_STAGE
# Loss curves — two-stage aware. Detects the available history schema
# and plots accordingly. Falls back to the legacy single-stage layout
# when those keys are present.
import matplotlib.pyplot as plt

_have_legacy = isinstance(globals().get("history"), dict) and "loss" in history
_have_s1 = isinstance(globals().get("val_mse_history"), list) and len(val_mse_history) > 0
_have_s2 = (
    isinstance(globals().get("history"), dict)
    and "loss_diff_train" in history
    and len(history["loss_diff_train"]) > 0
)

if _have_legacy:
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    epochs = range(1, len(history["loss"]) + 1)
    axes[0, 0].plot(epochs, history["loss"], linewidth=2, color="navy")
    axes[0, 0].set_title("Total Loss", fontsize=12, fontweight="bold")
    axes[0, 1].plot(epochs, history["loss_gen"], linewidth=2, color="crimson")
    axes[0, 1].set_title("Generation Loss (Diffusion)", fontsize=12, fontweight="bold")
    axes[1, 0].plot(epochs, history["loss_rec"], linewidth=2, color="green")
    axes[1, 0].set_title("Reconstruction Loss", fontsize=12, fontweight="bold")
    axes[1, 1].plot(epochs, history["loss_dag"], linewidth=2, color="orange")
    axes[1, 1].axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Target")
    axes[1, 1].set_title("DAG Constraint (NO TEARS)", fontsize=12, fontweight="bold")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Courbes legacy sauvegardées dans 'training_curves.png'")

elif _have_s1 or _have_s2:
    n_panels = (1 if _have_s1 else 0) + (1 if _have_s2 else 0)
    fig, axes = plt.subplots(1, max(2, n_panels), figsize=(8 * max(2, n_panels), 5))
    if max(2, n_panels) == 1:
        axes = [axes]

    panel = 0
    if _have_s1:
        ax = axes[panel]
        epochs_s1 = range(1, len(val_mse_history) + 1)
        ax.plot(epochs_s1, val_mse_history, linewidth=2, color="navy", marker="o")
        ax.set_title("Stage 1 — ValMSE (μ_HR prediction)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Validation MSE")
        ax.grid(True, alpha=0.3)
        panel += 1

    if _have_s2:
        ax = axes[panel]
        epochs_s2 = range(1, len(history["loss_diff_train"]) + 1)
        ax.plot(epochs_s2, history["loss_diff_train"], linewidth=2,
                color="crimson", marker="s")
        ax.set_title("Stage 2 — Diffusion Train Loss", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("EDM denoising loss")
        ax.grid(True, alpha=0.3)
        panel += 1

    # Hide any unused panel.
    for j in range(panel, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ Courbes two-stage sauvegardées dans 'training_curves.png'")
    if _have_s1:
        print(f"   Stage 1: {len(val_mse_history)} epochs | "
              f"final ValMSE = {val_mse_history[-1]:.5f} | "
              f"best = {min(val_mse_history):.5f}")
    if _have_s2:
        print(f"   Stage 2: {len(history['loss_diff_train'])} epochs | "
              f"final loss_diff = {history['loss_diff_train'][-1]:.5f}")

else:
    print("⚠️  Aucune métrique disponible — relance les cellules d'entraînement.")
    print("    Cherchait : history['loss'] (legacy) | val_mse_history (Stage 1) | history['loss_diff_train'] (Stage 2)")
'''


def patch_cell_53() -> int:
    nb = json.loads(TRAIN_NB.read_text(encoding="utf-8"))
    cells = nb["cells"]
    # Cell 53 = the "Courbes de loss" plot block.
    target_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        s = "".join(c.get("source", []))
        if 'history["loss"]' in s and 'training_curves.png' in s:
            target_idx = i
            break
    if target_idx is None:
        print("  ! cell 53 (loss curves) not found")
        return 0
    s = "".join(cells[target_idx]["source"])
    if "BS25_PLOTS_TWO_STAGE" in s:
        print("  = cell 53 already patched (BS25)")
        return 0
    cells[target_idx]["source"] = NEW_SOURCE.splitlines(keepends=True)
    cells[target_idx]["outputs"] = []
    cells[target_idx]["execution_count"] = None
    TRAIN_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ~ cell {target_idx}: replaced with two-stage-aware loss curves")
    return 1


def main() -> int:
    print("=== BS25 : two-stage-aware loss curves (cell 53) ===")
    n = patch_cell_53()
    print(f"\n{n} edit(s) applied")
    return 0


if __name__ == "__main__":
    sys.exit(main())
