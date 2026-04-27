"""
Inject Google Drive mount + ``CONFIG.checkpoint.save_dir`` override at the top
of the persistence-helpers cell (the cell starting with ``# >>> PERSIST_HELPERS``)
in ``st_cdgm_training_evaluation.ipynb``.

Why prepend rather than a separate cell:
- Helpers cell already reads ``CONFIG.checkpoint.save_dir`` on import (line:
  ``CKPT_SAVE_DIR = Path(CONFIG.checkpoint.get("save_dir", ...))``).
- If the override lived in a separate cell, the user could forget to run it
  before the helpers cell and end up writing checkpoints into the ephemeral
  ``models/`` of the Colab VM. Putting both in the same cell makes the order
  enforced by execution.

Guards:
- ``try / except ImportError`` around ``from google.colab import drive`` so
  the cell stays runnable on local CPU dev hosts.
- ``os.path.ismount('/content/drive')`` skip-if-already-mounted.

Idempotent: detects sentinel ``# >>> COLAB_DRIVE_MOUNT``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL_HELPERS = "# >>> PERSIST_HELPERS"
SENTINEL_DRIVE = "# >>> COLAB_DRIVE_MOUNT"

DRIVE_BLOCK = '''\
# >>> COLAB_DRIVE_MOUNT
# Monter Google Drive et rediriger les checkpoints vers Drive.
# Sur un host non-Colab (CPU local, CI, …) la garde ImportError saute le
# bloc proprement et ``CONFIG.checkpoint.save_dir`` reste celui du YAML.
try:
    from google.colab import drive  # type: ignore[import-not-found]
    import os as _os
    if not _os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    else:
        print("ℹ️  /content/drive déjà monté — skip drive.mount().")
    # Chemin Drive persistant — même nom que le dossier local (climate_data).
    CONFIG.checkpoint.save_dir = "/content/drive/MyDrive/climate_data/ckpt"
    print(f"📂 Drive monté — checkpoints → {CONFIG.checkpoint.save_dir}")
except ImportError:
    # Hors Colab : on garde le save_dir du YAML (par défaut "models/").
    print(f"ℹ️  Hors Colab — checkpoints → {CONFIG.checkpoint.get('save_dir', 'models')}")

'''


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))

    # Find the persistence-helpers cell.
    target_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if SENTINEL_HELPERS in src:
            target_idx = i
            break

    if target_idx is None:
        print(
            "[ERROR] Cellule helpers non trouvée. Lancez d'abord "
            "scripts/_patch_persist_per_epoch.py."
        )
        return 2

    src = "".join(nb["cells"][target_idx].get("source", []))
    if SENTINEL_DRIVE in src:
        # Bloc déjà présent — on le remplace pour propager les changements
        # (chemin save_dir mis à jour, etc.).
        # On retire le bloc existant entre SENTINEL_DRIVE et la prochaine
        # ligne vide, puis on ré-injecte la version courante.
        end_idx = src.find(SENTINEL_HELPERS)
        if end_idx == -1:
            print(f"[ERROR] Marqueur PERSIST_HELPERS introuvable dans cellule {target_idx}.")
            return 3
        new_src = DRIVE_BLOCK + src[end_idx:]
        nb["cells"][target_idx]["source"] = new_src.splitlines(keepends=True)
        NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"✓ Drive-mount mis à jour dans la cellule {target_idx} (path = climate_data).")
        return 0

    new_src = DRIVE_BLOCK + src
    nb["cells"][target_idx]["source"] = new_src.splitlines(keepends=True)

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"✓ Drive-mount injecté en tête de la cellule {target_idx}.")
    print(f"\U0001f4be Saved {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
