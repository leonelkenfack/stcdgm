# Generator: writes PROJECT_COMPLET.md at repository root.
# Run: python scripts/gen_project_complet_st_cdgm.py
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_PATHS = (
    "st_cdgm_training_evaluation.ipynb",
    "st_cdgm_validation_inference.ipynb",
    "resume_training_from_checkpoint.ipynb",
    "st_cdgm_publication_figures.ipynb",
    "st_cdgm_results_presentation.ipynb",
)

EMBEDDED_PATHS: tuple[str, ...] = (
    "config/docker.env",
    "config/training_config.yaml",
    "config/training_config_vice.yaml",
    "docker-compose.yml",
    "Dockerfile",
    "environment.yml",
    "requirements.txt",
    "setup.py",
    ".gitignore",
    ".dockerignore",
    "data/metadata/NorESM2-MM_histupdated_compressed.metadata.json",
    "data/metadata/NorESM2-MM_histupdated_compressed.metadata.csv",
    "src/st_cdgm/__init__.py",
    "src/st_cdgm/data/__init__.py",
    "src/st_cdgm/data/pipeline.py",
    "src/st_cdgm/data/netcdf_utils.py",
    "src/st_cdgm/models/__init__.py",
    "src/st_cdgm/models/causal_rcn.py",
    "src/st_cdgm/models/diffusion_decoder.py",
    "src/st_cdgm/models/graph_builder.py",
    "src/st_cdgm/models/intelligible_encoder.py",
    "src/st_cdgm/training/__init__.py",
    "src/st_cdgm/training/callbacks.py",
    "src/st_cdgm/training/training_loop.py",
    "src/st_cdgm/training/multi_gpu.py",
    "src/st_cdgm/evaluation/__init__.py",
    "src/st_cdgm/evaluation/evaluation_xai.py",
    "src/st_cdgm/utils/__init__.py",
    "src/st_cdgm/utils/checkpoint.py",
    "ops/train_st_cdgm.py",
    "ops/preprocess_to_zarr.py",
    "ops/preprocess_to_shards.py",
    "scripts/cleanup_repeated_lines.py",
    "scripts/load_model.py",
    "scripts/run_evaluation.py",
    "scripts/run_full_pipeline.py",
    "scripts/run_preprocessing.py",
    "scripts/run_training.py",
    "scripts/save_model.py",
    "scripts/sync_datastore.py",
    "scripts/test_installation.py",
    "scripts/test_pipeline.py",
    "scripts/validate_setup.py",
    "scripts/validate_antismoothing.py",
    "scripts/vice_utils.py",
    "tests/__init__.py",
    "tests/test_installation.py",
    "tests/test_corrections_antilissage.py",
    "tests/test_st_cdgm_smoke.py",
    "train_ddp.py",
)

SKIP_TREE_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".venv",
    "venv",
    ".cursor",
    "node_modules",
    "mcps",
    ".idea",
    "downscaling",
    "site-packages",
}
SKIP_TREE_SUFFIXES = (".pth", ".pyc")
ALLOWED_TOP_DIRS = frozenset({"config", "data", "docs", "ops", "scripts", "src", "tests", "models"})
SKIP_ROOT_FILES = frozenset({"PROJECT_COMPLET.md", "_cell40.py"})


def _should_skip(path: Path) -> bool:
    if path.name in SKIP_TREE_NAMES:
        return True
    if path.name.endswith(SKIP_TREE_SUFFIXES):
        return True
    if "site-packages" in path.parts:
        return True
    return False


def _tree_append(lines: list[str], path: Path, prefix: str, is_last: bool) -> None:
    conn = "└── " if is_last else "├── "
    name = path.name + ("/" if path.is_dir() else "")
    lines.append(f"{prefix}{conn}{name}")


def _walk_dir(lines: list[str], directory: Path, prefix: str, root: Path) -> None:
    try:
        entries = [p for p in directory.iterdir() if not _should_skip(p)]
    except OSError:
        return
    entries.sort(key=lambda p: (p.is_file(), p.name.lower()))
    for i, p in enumerate(entries):
        last = i == len(entries) - 1
        _tree_append(lines, p, prefix, last)
        if p.is_dir():
            ext = "    " if last else "│   "
            if p.name == "data" and (p / "raw").is_dir():
                lines.append(f"{ext}├── metadata/")
                metas = sorted((p / "metadata").glob("*"), key=lambda x: x.name)
                for j, meta in enumerate(metas):
                    mlast = j == len(metas) - 1
                    mp = f"{ext}│   └── " if mlast else f"{ext}│   ├── "
                    lines.append(f"{mp}{meta.name}")
                lines.append(f"{ext}└── raw/")
            elif p.name == "models" and p.parent == root:
                lines.append(f"{ext}└── *.pth (checkpoints)")
            else:
                _walk_dir(lines, p, prefix + ext, root)


def _build_tree_string(root: Path) -> str:
    lines: list[str] = ["climate_data/"]
    try:
        top_files = sorted(
            [
                p
                for p in root.iterdir()
                if p.is_file() and not _should_skip(p) and p.name not in SKIP_ROOT_FILES
            ],
            key=lambda x: x.name.lower(),
        )
        top_dirs = sorted(
            [p for p in root.iterdir() if p.is_dir() and p.name in ALLOWED_TOP_DIRS and not _should_skip(p)],
            key=lambda x: x.name.lower(),
        )
        top = top_files + top_dirs
    except OSError:
        return "\n".join(lines)
    for i, p in enumerate(top):
        last = i == len(top) - 1
        _tree_append(lines, p, "", last)
        if p.is_dir():
            ext = "    " if last else "│   "
            if p.name == "data" and (p / "raw").is_dir():
                lines.append(f"{ext}├── metadata/")
                metas = sorted((p / "metadata").glob("*"), key=lambda x: x.name)
                for j, meta in enumerate(metas):
                    mlast = j == len(metas) - 1
                    mp = f"{ext}│   └── " if mlast else f"{ext}│   ├── "
                    lines.append(f"{mp}{meta.name}")
                lines.append(f"{ext}└── raw/")
            elif p.name == "models" and p.parent == root:
                lines.append(f"{ext}└── *.pth (checkpoints)")
            else:
                _walk_dir(lines, p, ext, root)
    return "\n".join(lines)


def _lang_for_suffix(path: Path) -> str:
    s = path.suffix.lower()
    return {
        ".py": "python",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".json": "json",
        ".csv": "csv",
        ".env": "env",
        ".txt": "text",
        ".gitignore": "gitignore",
        ".dockerignore": "gitignore",
    }.get(s, "text")


def _fence_for_content(content: str) -> tuple[str, str]:
    if "```" in content:
        return "````", "````"
    return "```", "```"


def _section_file(root: Path, rel: str) -> str:
    path = root / rel
    if not path.is_file():
        return f"### `{rel}`\n\n*[Fichier absent : {rel}]*\n\n---\n\n"
    content = path.read_text(encoding="utf-8", errors="replace")
    lang = _lang_for_suffix(path)
    if path.name == "Dockerfile":
        lang = "dockerfile"
    open_f, close_f = _fence_for_content(content)
    body = f"{open_f}{lang}\n{content.rstrip()}\n{close_f}\n"
    return f"### `{rel}`\n\n{body}\n---\n\n"


def _notebook_summary(root: Path, rel: str) -> str:
    p = root / rel
    if not p.is_file():
        return f"### `{rel}`\n\n*[Notebook absent : {rel}]*\n\n---\n\n"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        return f"### `{rel}`\n\n*[JSON invalide : {e}]*\n\n---\n\n"
    cells = data.get("cells", [])
    n = len(cells)
    kinds: dict[str, int] = {}
    for c in cells:
        k = c.get("cell_type", "?")
        kinds[k] = kinds.get(k, 0) + 1
    kinds_s = ", ".join(f"{k}: {v}" for k, v in sorted(kinds.items()))
    lines = [
        f"### `{rel}`",
        "",
        f"**Contenu non embarqué** (JSON volumineux). Le notebook compte **{n}** cellules ({kinds_s}).",
        "",
        "Ouvrir le fichier `.ipynb` dans le dépôt pour le code et les sorties complets.",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _line_counts(root: Path, patterns: Iterable[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for rel in patterns:
        p = root / rel
        if p.is_file():
            out[rel] = sum(1 for _ in open(p, "rb"))
    return out


def _count_py_files(root: Path) -> tuple[int, int]:
    n_files = 0
    n_lines = 0
    for dirpath, dirnames, filenames in os.walk(root / "src"):
        dirnames[:] = [d for d in dirnames if d not in SKIP_TREE_NAMES]
        for fn in filenames:
            if fn.endswith(".py"):
                n_files += 1
                n_lines += sum(1 for _ in open(Path(dirpath) / fn, "rb"))
    for sub in ("ops", "scripts", "tests"):
        p = root / sub
        if not p.is_dir():
            continue
        for fp in p.rglob("*.py"):
            if "__pycache__" in str(fp):
                continue
            n_files += 1
            n_lines += sum(1 for _ in open(fp, "rb"))
    for name in ("setup.py", "train_ddp.py"):
        fp = root / name
        if fp.is_file():
            n_files += 1
            n_lines += sum(1 for _ in open(fp, "rb"))
    return n_files, n_lines


def generate(root: Path) -> str:
    intro = f"""# Documentation complete du projet ST-CDGM

## Spatio-Temporal Causal Diffusion Generative Model

Ce document est genere automatiquement par `scripts/gen_project_complet_st_cdgm.py`.
Derniere generation : **{datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")}**.

Les **notebooks** (.ipynb) ne sont pas embarques en integralite : seul un resume (nombre de cellules) est fourni.

---

## Structure du Projet

```
{_build_tree_string(root)}
```

---

## Fichiers de configuration et code source

"""

    parts: list[str] = [intro]

    for rel in EMBEDDED_PATHS:
        parts.append(_section_file(root, rel))

    parts.append("## Notebooks\n\n")
    for nb in NOTEBOOK_PATHS:
        parts.append(_notebook_summary(root, nb))

    parts.append(
        """
---

## Hierarchie du Code

### Architecture modulaire

```
ST-CDGM
|
+-- Data Layer (src/st_cdgm/data/)
|   +-- pipeline.py          -> NetCDFDataPipeline, ZarrDataPipeline
|   +-- netcdf_utils.py      -> utilitaires NetCDF / metadonnees
|
+-- Graph Layer (src/st_cdgm/models/graph_builder.py)
|   +-- HeteroGraphBuilder   -> Construction graphe heterogene
|
+-- Encoding Layer (src/st_cdgm/models/intelligible_encoder.py)
|   +-- IntelligibleVariableEncoder -> Variables latentes H(0)
|
+-- Causal Layer (src/st_cdgm/models/causal_rcn.py)
|   +-- RCNCell              -> Cellule recurrente causale
|   +-- RCNSequenceRunner    -> Deroulement sequentiel
|
+-- Generation Layer (src/st_cdgm/models/diffusion_decoder.py)
|   +-- CausalDiffusionDecoder -> Generation HR par diffusion
|
+-- Training Layer (src/st_cdgm/training/)
|   +-- training_loop.py     -> train_epoch, pertes
|   +-- callbacks.py
|   +-- multi_gpu.py         -> entrainement multi-GPU (DDP)
|
+-- Utils (src/st_cdgm/utils/)
|   +-- checkpoint.py        -> chargement / fusion de checkpoints
|
+-- Evaluation Layer (src/st_cdgm/evaluation/)
    +-- evaluation_xai.py    -> Metriques, inference autoregressive, DAG
```

### Flux de donnees

```
NetCDF / Zarr
    |
NetCDFDataPipeline
    +-- Alignement temporel
    +-- Normalisation
    +-- Baseline computation
    +-- Residual calculation
    |
IterableDataset / batches
    |
HeteroGraphBuilder
    +-- Construction graphe statique
    |
IntelligibleVariableEncoder
    +-- H(0) initial state
    |
RCNSequenceRunner
    +-- RCNCell (sequence)
    +-- H(t) causal states
    |
CausalDiffusionDecoder
    +-- Conditioning from H(t)
    +-- HR generation (diffusion)
    |
Loss Computation
    +-- L_gen (diffusion)
    +-- L_rec (reconstruction)
    +-- L_dag (DAG constraint)
    |
Optimization
    +-- Backpropagation
```

### Dependances entre modules (schema)

```
setup.py
    |
src/st_cdgm/__init__.py
    +-- models/
    |   +-- causal_rcn.py
    |   +-- diffusion_decoder.py
    |   +-- graph_builder.py
    |   +-- intelligible_encoder.py
    +-- data/
    |   +-- pipeline.py
    |   +-- netcdf_utils.py
    +-- training/
    |   +-- training_loop.py
    |   +-- callbacks.py
    |   +-- multi_gpu.py
    +-- evaluation/
    |   +-- evaluation_xai.py
    +-- utils/
        +-- checkpoint.py
```

---

## Resume des fichiers

### Statistiques (approximatif, genere)

"""
    )

    py_files, py_lines = _count_py_files(root)
    key_modules = [
        "src/st_cdgm/data/pipeline.py",
        "src/st_cdgm/data/netcdf_utils.py",
        "src/st_cdgm/models/causal_rcn.py",
        "src/st_cdgm/models/diffusion_decoder.py",
        "src/st_cdgm/models/graph_builder.py",
        "src/st_cdgm/models/intelligible_encoder.py",
        "src/st_cdgm/training/training_loop.py",
        "src/st_cdgm/evaluation/evaluation_xai.py",
    ]
    lc = _line_counts(root, key_modules)

    parts.append(f"- **Fichiers Python (estimation)** : ~{py_files} fichiers, ~{py_lines} lignes (src + ops + scripts + tests + racine).\n")
    parts.append("- **Configuration** : `config/*.yaml`, `docker.env`, `requirements.txt`, `environment.yml`.\n")
    parts.append("- **Documentation** : `docs/*.md`, `README.md`, `stats.md`.\n")
    parts.append("- **Notebooks** : entrainement/evaluation, validation/inference, figures publication.\n\n")
    parts.append("### Lignes de code (modules cles)\n\n")
    for rel in key_modules:
        n = lc.get(rel, 0)
        parts.append(f"- `{rel}`: ~{n} lignes\n")
    parts.append("\n---\n\n## Points cles de l'architecture\n\n")
    parts.append(
        "1. **Modularite** : composants independants et reutilisables.\n"
        "2. **Extensibilite** : nouveaux modules (p.ex. planificateurs de diffusion).\n"
        "3. **Configuration** : Hydra / YAML.\n"
        "4. **Performance** : optimisations documentees dans `docs/OPTIMISATION.md`.\n"
        "5. **Robustesse** : tests, validation, checkpoints (`utils/checkpoint.py`).\n\n"
        "---\n\n"
        "**Fin de la documentation complete du projet ST-CDGM** *(genere)*\n"
    )

    return "".join(parts)


def main() -> None:
    root = REPO_ROOT
    out = root / "PROJECT_COMPLET.md"
    text = generate(root)
    out.write_text(text, encoding="utf-8")
    print(f"OK: {out} ({len(text)} chars)")


if __name__ == "__main__":
    main()
