"""
One-shot patch: add per-epoch atomic persistence to st_cdgm_training_evaluation.ipynb.

Inserts a markdown cell + a code cell (helpers) BEFORE the training cell
(`for epoch in range(...)`), and rewrites the training cell to:
  1. call ``maybe_resume_from_disk()`` before the for-loop
  2. start ``range`` at ``_epoch_start`` (resume offset)
  3. call ``persist_epoch_checkpoint(...)`` at the end of each epoch body
     (after history + BEST_MODEL_STATES update, before the early-stop break).

Idempotent: running it twice is a no-op (detects sentinel ``# >>> PERSIST_HELPERS``).

Usage:
    python scripts/_patch_persist_per_epoch.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL_HELPERS = "# >>> PERSIST_HELPERS"
SENTINEL_RESUME  = "# >>> PERSIST_RESUME"
SENTINEL_PERSIST = "# >>> PERSIST_CALL"

MARKDOWN_CELL = """\
## 💾 Persistance atomique par époque (Colab-friendly)

Sur Colab Pro, le runtime peut être déconnecté à tout moment (idle timeout,
limites GPU, etc.). Pour éviter de perdre des heures d'entraînement, on
persiste **après chaque époque** sur disque, idéalement Google Drive :

```python
from google.colab import drive
drive.mount('/content/drive')
CONFIG.checkpoint.save_dir = '/content/drive/MyDrive/st_cdgm/ckpt'
```

La cellule suivante définit :

- `persist_epoch_checkpoint(...)` — appelé après chaque époque, écrit
  atomiquement `epoch_last.pth` (et `epoch_best.pth` si val improved).
- `maybe_resume_from_disk()` — à appeler avant le `for epoch in range(...)`.
  Si `epoch_last.pth` existe, recharge tous les state_dicts (encoder, RCN,
  diffusion, optimizer, scheduler, RNG, history, BEST_MODEL_STATES) et
  renvoie l'offset `epoch_start` à utiliser dans `range`.

Réglages dans `CONFIG.checkpoint` : `persist_every` (1 = à chaque époque),
`keep_last_k` (0 = pas d'historique, sinon garde les K derniers `epoch_NNN.pth`).
"""

HELPERS_CELL = '''\
# >>> PERSIST_HELPERS
# Persistance atomique en fin de chaque époque + reprise depuis disque.
# Conçu pour Colab : kill runtime → reprise sans perte (sauf l'époque en cours).

import os
import tempfile
from pathlib import Path
import math
import torch
import numpy as np
from omegaconf import OmegaConf

CKPT_SAVE_DIR = Path(CONFIG.checkpoint.get("save_dir", "models"))
CKPT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
PERSIST_EVERY = int(CONFIG.checkpoint.get("persist_every", 1))
KEEP_LAST_K = int(CONFIG.checkpoint.get("keep_last_k", 0))


def _atomic_save(payload: dict, path: Path) -> None:
    """Écriture atomique : tmp dans le même dossier puis ``os.replace``.
    Garantit qu'un kill mid-write ne laisse pas un .pth corrompu."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        os.close(fd)
        torch.save(payload, tmp)
        os.replace(tmp, path)  # atomique sur même filesystem (POSIX + Windows)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _persist_state_dict(m):
    """state_dict du module sous-jacent (strip DDP + torch.compile)."""
    if m is None:
        return None
    base = m.module if hasattr(m, "module") and not hasattr(m, "_orig_mod") else m
    base = getattr(base, "_orig_mod", base)
    return base.state_dict()


def _persist_load_state_dict(m, sd):
    """Charge un state_dict en respectant les wrappers DDP / torch.compile."""
    if m is None or sd is None:
        return
    try:
        from st_cdgm.utils.checkpoint import strip_torch_compile_prefix
        sd = strip_torch_compile_prefix(sd)
    except Exception:
        pass
    base = m.module if hasattr(m, "module") and not hasattr(m, "_orig_mod") else m
    base = getattr(base, "_orig_mod", base)
    base.load_state_dict(sd)


def _build_epoch_payload(*, epoch_done: int, train_metrics: dict, val_loss: float,
                         history: dict, best_val_loss: float, best_epoch: int,
                         no_improve_epochs: int, BEST_MODEL_STATES) -> dict:
    payload = {
        "schema_version": 1,
        "epoch": int(epoch_done),  # nombre d'époques *complétées*
        "epochs_total": int(CONFIG.training.epochs),
        "config": OmegaConf.to_container(CONFIG, resolve=True),
        "train_metrics": train_metrics,
        "val_loss": float(val_loss) if val_loss is not None and np.isfinite(val_loss) else None,
        "history": history,
        "best_val_loss": (float(best_val_loss) if np.isfinite(best_val_loss) else None),
        "best_epoch": int(best_epoch),
        "no_improve_epochs": int(no_improve_epochs),
        "BEST_MODEL_STATES": BEST_MODEL_STATES,
        # Modules vivants : suffisant pour reprendre exactement d'où l'on est
        "encoder_state_dict": _persist_state_dict(encoder),
        "rcn_cell_state_dict": _persist_state_dict(rcn_cell),
        "diffusion_state_dict": _persist_state_dict(diffusion),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (scheduler.state_dict() if scheduler is not None else None),
        # RNG : la suite de l'entraînement reste reproductible après resume
        "rng_torch_cpu": torch.get_rng_state(),
        "rng_torch_cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        "rng_numpy": np.random.get_state(),
    }
    if "spatial_projector" in globals() and spatial_projector is not None:
        payload["spatial_projector_state_dict"] = _persist_state_dict(spatial_projector)
    if "hr_ident_head" in globals() and hr_ident_head is not None:
        payload["hr_ident_head_state_dict"] = _persist_state_dict(hr_ident_head)
    return payload


def persist_epoch_checkpoint(*, epoch_idx: int, train_metrics: dict, val_loss: float,
                             history: dict, best_val_loss: float, best_epoch: int,
                             no_improve_epochs: int, BEST_MODEL_STATES,
                             improved: bool) -> None:
    """À appeler une fois par époque, après history.append + best update.

    ``epoch_idx`` est le compteur 0-indexé de la boucle ``for epoch in range(...)``.
    Le payload, lui, stocke ``epoch = epoch_idx + 1`` = nb d'époques complétées,
    qui est aussi le ``_epoch_start`` à utiliser au prochain run."""
    if (epoch_idx + 1) % max(1, PERSIST_EVERY) != 0:
        return
    payload = _build_epoch_payload(
        epoch_done=epoch_idx + 1,
        train_metrics=train_metrics, val_loss=val_loss, history=history,
        best_val_loss=best_val_loss, best_epoch=best_epoch,
        no_improve_epochs=no_improve_epochs,
        BEST_MODEL_STATES=BEST_MODEL_STATES,
    )
    last_path = CKPT_SAVE_DIR / "epoch_last.pth"
    _atomic_save(payload, last_path)
    if improved:
        _atomic_save(payload, CKPT_SAVE_DIR / "epoch_best.pth")
    if KEEP_LAST_K > 0:
        snap = CKPT_SAVE_DIR / f"epoch_{epoch_idx + 1:04d}.pth"
        _atomic_save(payload, snap)
        snaps = sorted(CKPT_SAVE_DIR.glob("epoch_[0-9]*.pth"))
        for old in snaps[:-KEEP_LAST_K]:
            try:
                old.unlink()
            except OSError:
                pass
    _size_mb = last_path.stat().st_size / (1024 * 1024)
    print(f"💾 [epoch {epoch_idx + 1}] persisted {last_path} ({_size_mb:.1f} MB)"
          + (" + best" if improved else ""))


def maybe_resume_from_disk():
    """À appeler AVANT le for-loop. Renvoie un dict avec les compteurs à
    restaurer si ``epoch_last.pth`` existe, sinon ``None`` (entraînement frais).

    Recharge en place : encoder, rcn_cell, diffusion, spatial_projector,
    hr_ident_head, optimizer, scheduler, RNG états (torch + numpy + cuda).
    """
    last_path = CKPT_SAVE_DIR / "epoch_last.pth"
    if not last_path.exists():
        print(f"🆕 Pas de checkpoint dans {CKPT_SAVE_DIR} — entraînement frais.")
        return None
    print(f"🔁 Reprise depuis {last_path}")
    ckpt = torch.load(last_path, map_location=DEVICE, weights_only=False)
    _persist_load_state_dict(encoder, ckpt.get("encoder_state_dict"))
    _persist_load_state_dict(rcn_cell, ckpt.get("rcn_cell_state_dict"))
    _persist_load_state_dict(diffusion, ckpt.get("diffusion_state_dict"))
    if "spatial_projector" in globals() and spatial_projector is not None:
        _persist_load_state_dict(spatial_projector, ckpt.get("spatial_projector_state_dict"))
    if "hr_ident_head" in globals() and hr_ident_head is not None:
        _persist_load_state_dict(hr_ident_head, ckpt.get("hr_ident_head_state_dict"))
    if ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if ckpt.get("rng_torch_cpu") is not None:
        try:
            torch.set_rng_state(ckpt["rng_torch_cpu"])
        except TypeError:
            torch.set_rng_state(torch.as_tensor(ckpt["rng_torch_cpu"], dtype=torch.uint8).cpu())
    if torch.cuda.is_available() and ckpt.get("rng_torch_cuda") is not None:
        try:
            torch.cuda.set_rng_state_all(ckpt["rng_torch_cuda"])
        except (TypeError, RuntimeError):
            pass
    if ckpt.get("rng_numpy") is not None:
        np.random.set_state(ckpt["rng_numpy"])
    print(f"   epoch précédente: {ckpt['epoch']} / {ckpt['epochs_total']}, "
          f"val_loss={ckpt.get('val_loss')}, best_val_loss={ckpt.get('best_val_loss')}")
    return {
        "epoch_start": int(ckpt["epoch"]),
        "best_val_loss": (ckpt["best_val_loss"] if ckpt.get("best_val_loss") is not None else math.inf),
        "best_epoch": int(ckpt.get("best_epoch", 0)),
        "no_improve_epochs": int(ckpt.get("no_improve_epochs", 0)),
        "history": ckpt.get("history"),
        "BEST_MODEL_STATES": ckpt.get("BEST_MODEL_STATES"),
    }


print(f"📁 Save dir: {CKPT_SAVE_DIR.resolve()}")
print(f"   persist_every={PERSIST_EVERY}, keep_last_k={KEEP_LAST_K}")
'''

RESUME_BLOCK = '''\
# >>> PERSIST_RESUME
# Reprise depuis disque si epoch_last.pth existe (Colab après déconnexion).
_resume = maybe_resume_from_disk()
_epoch_start = 0
if _resume is not None:
    _epoch_start = _resume["epoch_start"]
    best_val_loss = _resume["best_val_loss"]
    best_epoch = _resume["best_epoch"]
    no_improve_epochs = _resume["no_improve_epochs"]
    if _resume.get("history") is not None:
        history = _resume["history"]
    if _resume.get("BEST_MODEL_STATES") is not None:
        BEST_MODEL_STATES = _resume["BEST_MODEL_STATES"]
    print(f"▶️  Resume: démarrage à l'epoch {_epoch_start + 1}/{int(CONFIG.training.epochs)}")

'''

PERSIST_CALL = '''\
    # >>> PERSIST_CALL
    # Persistance atomique : after history + best update, before early-stop check.
    persist_epoch_checkpoint(
        epoch_idx=epoch,
        train_metrics=train_metrics, val_loss=val_loss, history=history,
        best_val_loss=best_val_loss, best_epoch=best_epoch,
        no_improve_epochs=no_improve_epochs,
        BEST_MODEL_STATES=BEST_MODEL_STATES,
        improved=improved,
    )

'''


def _make_md_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _make_code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # 1) Find the training cell — accept both the original and the patched form.
    train_cell_idx = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if (
            "for epoch in range(int(CONFIG.training.epochs))" in src
            or "for epoch in range(_epoch_start, int(CONFIG.training.epochs))" in src
        ):
            train_cell_idx = i
            break
    if train_cell_idx is None:
        print("[ERROR] Could not locate training cell with `for epoch in range(...)`.")
        return 2

    # 2) Idempotency check via sentinels.
    helpers_present = any(
        c["cell_type"] == "code" and SENTINEL_HELPERS in "".join(c.get("source", []))
        for c in cells
    )
    train_src = "".join(cells[train_cell_idx].get("source", []))
    resume_present = SENTINEL_RESUME in train_src
    persist_present = SENTINEL_PERSIST in train_src

    if helpers_present and resume_present and persist_present:
        print("✓ Notebook déjà patché (helpers + resume + persist call détectés). Mise à jour du block helper...")
        # We don't return 0 here, so we can update the cell below and save.

    # 3) Insert markdown + helpers cells BEFORE the training cell.
    if not helpers_present:
        cells.insert(train_cell_idx, _make_code_cell(HELPERS_CELL))
        cells.insert(train_cell_idx, _make_md_cell(MARKDOWN_CELL))
        train_cell_idx += 2  # we shifted the training cell down by 2
        print(f"  ✓ Inserted persistence cells at index {train_cell_idx - 2}, {train_cell_idx - 1}")
    else:
        for c in cells:
            if c["cell_type"] == "code" and SENTINEL_HELPERS in "".join(c.get("source", [])):
                c["source"] = HELPERS_CELL.splitlines(keepends=True)
        print("  ✓ Updated existing persistence helpers cell.")

    # 4) Patch training cell.
    train_src = "".join(cells[train_cell_idx].get("source", []))

    if not resume_present:
        # Insert RESUME_BLOCK right before the for-loop line.
        anchor = "for epoch in range(int(CONFIG.training.epochs)):"
        if anchor not in train_src:
            print(f"[ERROR] Anchor '{anchor}' not found in training cell.")
            return 3
        train_src = train_src.replace(anchor, RESUME_BLOCK + anchor, 1)
        # Also rewrite the range to start at _epoch_start.
        train_src = train_src.replace(
            anchor,
            "for epoch in range(_epoch_start, int(CONFIG.training.epochs)):",
            1,
        )

    if not persist_present:
        # Insert PERSIST_CALL right after the if/else block that updates
        # BEST_MODEL_STATES / no_improve_epochs, BEFORE the val_loss_str print.
        anchor = '    val_loss_str = "n/a"'
        if anchor not in train_src:
            print(f"[ERROR] Anchor '{anchor[:40]}…' not found in training cell.")
            return 4
        train_src = train_src.replace(anchor, PERSIST_CALL + anchor, 1)

    cells[train_cell_idx]["source"] = train_src.splitlines(keepends=True)
    print(f"  ✓ Patched training cell (idx={train_cell_idx})")

    # 5) Save back.
    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Saved {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
