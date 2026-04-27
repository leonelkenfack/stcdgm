"""
Replace cell 5 of ``st_cdgm_training_evaluation.ipynb`` with a Colab-aware
bootstrap that handles the "multi-file project on Colab" pain point:

- mounts Google Drive
- clones the project into Drive on first run (or pulls latest if it's a git repo)
- ``os.chdir`` into the project root so relative paths (``config/...``,
  ``src/st_cdgm/...``) resolve normally
- ``pip install -e .`` so ``import st_cdgm`` works without sys.path tricks
- adds ``src/`` to sys.path defensively (in case pip install hasn't finished
  importing into the kernel yet)

Hors Colab : no-op (les chemins existants fonctionnent déjà depuis la racine
projet en local).

Idempotent : sentinelle ``# >>> COLAB_BOOTSTRAP`` détectée → skip.

Usage:
    python scripts/_patch_colab_bootstrap.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL = "# >>> COLAB_BOOTSTRAP"

BOOTSTRAP_CELL = '''\
# >>> COLAB_BOOTSTRAP
# Bootstrap Colab optimisé — premier run ~3 min, re-runs ~30 s.
# Stratégie :
#   • Code sur SSD local (/content/) — git clone 5-10× plus rapide que vers Drive.
#   • Drive UNIQUEMENT pour les checkpoints (cf. cellule helpers plus bas).
#   • Pas de ``pip install -r requirements.txt`` brut (déclenche la compilation
#     CUDA de torch-scatter/torch-sparse → 20-30 min). À la place : install
#     pinned des seules deps non pré-installées par Colab.
#   • Wheels PyG pré-construits via le bon index (sinon torch-scatter/torch-sparse
#     compilent depuis les sources — interdit ici).
#
# Hors Colab : no-op.
import os, sys, subprocess, time, shlex
from pathlib import Path

# ── Configuration utilisateur ──────────────────────────────────────────
GIT_URL: str | None = "https://github.com/leonelkenfack/stcdgm.git"
GIT_BRANCH: str = "main"
LOCAL_PROJECT = "/content/climate_data"  # SSD — toujours rapide
GIT_PULL_ON_RESUME = True
SKIP_PIP_IF_IMPORTABLE = True  # si ``import st_cdgm`` réussit déjà → skip pip

_IS_COLAB = "google.colab" in sys.modules or Path("/content").exists()


def _run(cmd: str, *, check: bool = True, timeout: int | None = None) -> int:
    """Exécute une commande shell avec timing visible."""
    print(f"$ {cmd}")
    t0 = time.time()
    rc = subprocess.call(shlex.split(cmd), timeout=timeout)
    dt = time.time() - t0
    print(f"  ↳ rc={rc}  ({dt:.1f}s)")
    if check and rc != 0:
        raise RuntimeError(f"Commande échouée : {cmd!r} (rc={rc})")
    return rc


if _IS_COLAB:
    _T0 = time.time()
    print("🛰️  Colab détecté — bootstrap en cours…\\n")

    # 1) Monter Drive (idempotent — pour la cellule de persistance plus loin)
    from google.colab import drive  # type: ignore[import-not-found]
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")
    else:
        print("   /content/drive déjà monté.")

    # 2) Clone vers SSD (PAS vers Drive — FUSE est lent pour des milliers de petits fichiers)
    project_path = Path(LOCAL_PROJECT)
    if not (project_path / ".git").exists():
        if GIT_URL is None:
            raise RuntimeError(
                "GIT_URL=None et projet absent du SSD. Renseignez GIT_URL ci-dessus, "
                "ou pré-uploadez le projet à " + LOCAL_PROJECT + " avant ce run."
            )
        project_path.parent.mkdir(parents=True, exist_ok=True)
        _run(f"git clone --depth 1 -b {GIT_BRANCH} {GIT_URL} {LOCAL_PROJECT}")
    elif GIT_PULL_ON_RESUME:
        try:
            _run(f"git -C {LOCAL_PROJECT} pull --ff-only", timeout=60, check=False)
        except Exception as e:
            print(f"   ⚠️  git pull a levé : {e}")

    # 3) cd dans la racine — config/, src/, etc. en chemins relatifs
    os.chdir(project_path)
    print(f"   chdir → {os.getcwd()}\\n")

    # 4) Test d'import — si st_cdgm marche déjà, on saute pip (énorme gain au re-run)
    src_path = str(project_path / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    _need_pip = True
    if SKIP_PIP_IF_IMPORTABLE:
        try:
            import st_cdgm  # noqa: F401
            from omegaconf import OmegaConf  # noqa: F401
            from diffusers import UNet2DConditionModel  # noqa: F401
            import torch_geometric  # noqa: F401
            _need_pip = False
            print("✓ Imports critiques OK — pip install sauté.")
        except ImportError as _imp_err:
            print(f"   import st_cdgm a échoué ({_imp_err}) — pip install requis.")

    if _need_pip:
        # 5) Versions PyTorch / CUDA déjà installées par Colab
        import torch
        TORCH_VER = torch.__version__.split("+")[0]  # ex. "2.5.1"
        TORCH_TAG = f"torch-{TORCH_VER}"             # ex. "torch-2.5.1"
        CUDA_TAG = "cu" + (torch.version.cuda or "121").replace(".", "") if torch.cuda.is_available() else "cpu"
        print(f"   torch={TORCH_VER}, cuda={CUDA_TAG}\\n")

        # 6) Install des deps NON pré-installées par Colab
        # (numpy/pandas/scipy/sklearn/matplotlib/torch/torchvision/torchaudio/
        #  zarr/dask/xarray/h5netcdf/cartopy/tqdm/requests/ipykernel sont déjà là)
        EXTRA_DEPS = [
            "omegaconf==2.3.0",
            "hydra-core==1.3.2",
            "diffusers==0.36.0",
            "transformers==4.57.6",
            "accelerate==1.12.0",
            "huggingface-hub==0.36.0",
            "safetensors==0.7.0",
            "xbatcher",
            "webdataset",
            "cftime",
            "h5netcdf",
            "numcodecs",
            "torch-geometric",  # v2.3+ ne nécessite plus torch-scatter/torch-sparse
        ]
        deps_str = " ".join(shlex.quote(p) for p in EXTRA_DEPS)
        _run(
            f"{shlex.quote(sys.executable)} -m pip install --no-warn-script-location {deps_str}",
            timeout=600,
        )

        # 7) torch-scatter / torch-sparse — *optionnels* avec PyG ≥ 2.3 (fallback
        # pure-PyTorch). Décommentez si vous tombez sur un module qui les exige.
        # _PYG_INDEX = f"https://data.pyg.org/whl/{TORCH_TAG}+{CUDA_TAG}.html"
        # _run(
        #     f"{shlex.quote(sys.executable)} -m pip install --no-warn-script-location "
        #     f"torch-scatter torch-sparse -f {_PYG_INDEX}",
        #     timeout=600, check=False,
        # )

        # 8) Editable install du package — --no-deps pour ne PAS retomber sur
        # requirements.txt (qui réinstallerait torch et compilerait torch-scatter).
        _run(
            f"{shlex.quote(sys.executable)} -m pip install --no-warn-script-location "
            f"--no-deps -e {LOCAL_PROJECT}",
            timeout=120,
        )

        # 9) Re-test des imports critiques
        try:
            import st_cdgm  # noqa: F401
            from omegaconf import OmegaConf  # noqa: F401
            print("✓ st_cdgm importable.")
        except ImportError as e:
            print(f"⚠️  st_cdgm pas encore importable depuis ce kernel : {e}")
            print("   → Probablement un cache d'import — Runtime → Restart runtime puis re-run.")

    print(f"\\n✅ Bootstrap Colab terminé en {time.time() - _T0:.1f}s.")

else:
    # Hors Colab : remonte automatiquement à la racine projet.
    _here = Path.cwd()
    for _candidate in [_here, *_here.parents]:
        if (_candidate / "config" / "training_config.yaml").exists() and (_candidate / "setup.py").exists():
            if _candidate != _here:
                os.chdir(_candidate)
                print(f"📂 chdir → {os.getcwd()} (racine projet détectée)")
            break
    print("ℹ️  Hors Colab — bootstrap sauté (assume install déjà faite).")
'''


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Find existing cell 5 (the placeholder ``# Installation des dépendances…``)
    # OR an already-bootstrapped cell with the sentinel.
    target_idx = None
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if SENTINEL in src:
            target_idx = i
            print(f"✓ Bootstrap déjà présent (cell {i}) — réécriture pour synchro.")
            break
        if "Installation des dépendances et du package st_cdgm" in src:
            target_idx = i
            print(f"  ↪ Remplace le placeholder install cell {i}.")
            break

    if target_idx is None:
        # Insert as a new cell right after the threading-config cell (cell 4)
        # and before the first imports. Locate cell 4 by content.
        for i, c in enumerate(cells):
            if c["cell_type"] != "code":
                continue
            src = "".join(c.get("source", []))
            if "ST_CDGM_CPU_THREADS" in src:
                target_idx = i + 1
                cells.insert(target_idx, {
                    "cell_type": "code", "execution_count": None,
                    "metadata": {}, "outputs": [], "source": [],
                })
                print(f"  ↪ Insertion d'une nouvelle cellule bootstrap à l'index {target_idx}.")
                break

    if target_idx is None:
        print("[ERROR] Impossible de placer le bootstrap (placeholder ni cell threading trouvés).")
        return 2

    cells[target_idx]["source"] = BOOTSTRAP_CELL.splitlines(keepends=True)
    cells[target_idx]["outputs"] = []
    cells[target_idx]["execution_count"] = None

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Saved {NB}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
