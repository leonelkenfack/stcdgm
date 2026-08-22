"""Assemble `st_cdgm_v8.ipynb` a partir des cellules de `v8_cells/`.

Les cellules vivent comme de VRAIS fichiers `.py` / `.md` plutot que comme des
chaines dans ce script : elles restent editables, verifiables par un linter, et
`git diff` reste lisible. L'assembleur ne fait que les concatener dans l'ordre
de leur prefixe numerique.

    python path_c_plus/scripts/_build_v8_notebook.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = HERE / "v8_cells"
OUT = HERE / "st_cdgm_v8.ipynb"


def cell(path: Path) -> dict:
    text = path.read_text(encoding="utf-8").rstrip("\n")
    src = text.splitlines(keepends=True)
    if path.suffix == ".md":
        return {"cell_type": "markdown", "metadata": {}, "source": src}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": src}


def main() -> None:
    paths = sorted(CELLS.glob("*.*"))
    if not paths:
        raise SystemExit(f"aucune cellule dans {CELLS}")
    nb = {
        "cells": [cell(p) for p in paths],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"ecrit : {OUT}")
    for p in paths:
        print(f"  {p.name:24s} {len(p.read_text(encoding='utf-8').splitlines()):4d} lignes")


if __name__ == "__main__":
    main()
