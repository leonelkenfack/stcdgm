"""
Cell 34 oublie de transmettre ``use_gradient_checkpointing`` (et plusieurs
autres options) du YAML au constructeur de ``CausalDiffusionDecoder`` →
gradient checkpointing inactif → toutes les activations forward du UNet
sont gardées pour le backward → ~8 GB pré-OOM.

Fix : ajouter ``use_gradient_checkpointing``, ``scheduler_type``,
``conv_padding_mode``, ``anti_checkerboard`` au constructeur, comme dans
``ops/train_st_cdgm.py``.

Idempotent via la présence de ``use_gradient_checkpointing=`` dans le
constructeur.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"

OLD_BLOCK = """diffusion = CausalDiffusionDecoder(
    in_channels=hr_channels,
    conditioning_dim=CONFIG.diffusion.conditioning_dim,
    height=CONFIG.diffusion.height,
    width=CONFIG.diffusion.width,
    num_diffusion_steps=CONFIG.diffusion.steps,
    unet_kwargs=UNET_KWARGS,
).to(DEVICE)"""

NEW_BLOCK = """diffusion = CausalDiffusionDecoder(
    in_channels=hr_channels,
    conditioning_dim=CONFIG.diffusion.conditioning_dim,
    height=CONFIG.diffusion.height,
    width=CONFIG.diffusion.width,
    num_diffusion_steps=CONFIG.diffusion.steps,
    unet_kwargs=UNET_KWARGS,
    # OOM fix : sans cette ligne, gradient_checkpointing était False (défaut)
    # malgré le YAML, donc toutes les activations forward du UNet étaient
    # gardées pour le backward — ~8 GB pré-OOM sur T4.
    use_gradient_checkpointing=bool(
        CONFIG.diffusion.get("use_gradient_checkpointing", False)
    ),
    scheduler_type=CONFIG.diffusion.get("scheduler_type", "ddpm"),
    conv_padding_mode=CONFIG.diffusion.get("conv_padding_mode", "zeros"),
    anti_checkerboard=bool(CONFIG.diffusion.get("anti_checkerboard", False)),
).to(DEVICE)"""


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    target = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if "CausalDiffusionDecoder(" in src and "unet_kwargs=UNET_KWARGS" in src:
            target = i
            break

    if target is None:
        print("[ERROR] Cellule construction CausalDiffusionDecoder introuvable.")
        return 2

    src = "".join(cells[target].get("source", []))

    if "use_gradient_checkpointing=bool(" in src:
        print(f"✓ Cell {target} déjà patchée (use_gradient_checkpointing transmis).")
        return 0

    if OLD_BLOCK not in src:
        print(f"[ERROR] Bloc constructeur exact non trouvé dans cell {target}.")
        print("        Vérifier que cell 34 contient bien le bloc original.")
        return 3

    new_src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    cells[target]["source"] = new_src.splitlines(keepends=True)
    cells[target]["outputs"] = []
    cells[target]["execution_count"] = None

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"  ↪ Cell {target} : ajout use_gradient_checkpointing + scheduler_type + conv_padding_mode + anti_checkerboard")
    print(f"💾 Saved {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
