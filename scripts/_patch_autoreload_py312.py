"""
Fix the ``%load_ext autoreload`` failure on Colab (Python 3.12).

Colab now ships Python 3.12, which removed the stdlib ``imp`` module. The
``autoreload`` extension shipped with Colab's IPython still does
``from imp import reload`` and crashes with ``ModuleNotFoundError: No module
named 'imp'``.

Fix: prepend a tiny ``imp`` polyfill before the magic, and wrap the magic in
try/except so it degrades gracefully on any other future breakage.

Patches every code cell that contains ``%load_ext autoreload`` (the notebook
has three duplicate "Configuration complète pour exécution locale" cells).
Idempotent via sentinel ``# >>> AUTORELOAD_PY312_FIX``.

Usage:
    python scripts/_patch_autoreload_py312.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "st_cdgm_training_evaluation.ipynb"
SENTINEL = "# >>> AUTORELOAD_PY312_FIX"
OLD_MAGIC = "%load_ext autoreload\n%autoreload 2"
OLD_MAGIC_ALT = "%load_ext autoreload\n%autoreload 2\n"

REPLACEMENT = '''\
# >>> AUTORELOAD_PY312_FIX
# Python 3.12 (Colab) a retiré le module stdlib ``imp``, mais l'extension
# IPython ``autoreload`` livrée par Colab importe encore ``from imp import
# reload``. On polyfill ``imp`` à la volée avant de charger l'extension,
# puis on wrappe le ``%load_ext`` dans un try/except pour rester safe.
import sys as _sys
if "imp" not in _sys.modules:
    try:
        import imp as _imp  # type: ignore[deprecated]  # noqa: F401
    except ModuleNotFoundError:
        import importlib as _il, types as _types
        _imp_stub = _types.ModuleType("imp")
        _imp_stub.reload = _il.reload  # type: ignore[attr-defined]
        _sys.modules["imp"] = _imp_stub

try:
    get_ipython().run_line_magic("load_ext", "autoreload")  # type: ignore[name-defined]
    get_ipython().run_line_magic("autoreload", "2")  # type: ignore[name-defined]
except Exception as _ar_e:
    print(f"⚠️  autoreload non chargé : {_ar_e}")'''


def _patch_source(src: str) -> str | None:
    """Returns the patched source, or None if no change needed."""
    if SENTINEL in src:
        return None  # already patched
    # Match the magic with or without trailing newline.
    if OLD_MAGIC_ALT in src:
        return src.replace(OLD_MAGIC_ALT, REPLACEMENT + "\n", 1)
    if OLD_MAGIC in src:
        return src.replace(OLD_MAGIC, REPLACEMENT, 1)
    return None


def main() -> int:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    n_patched = 0
    n_already = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] != "code":
            continue
        src = "".join(c.get("source", []))
        if SENTINEL in src:
            n_already += 1
            continue
        new_src = _patch_source(src)
        if new_src is None:
            continue
        c["source"] = new_src.splitlines(keepends=True)
        c["outputs"] = []
        c["execution_count"] = None
        n_patched += 1
        print(f"  ↪ Patched cell {i} (autoreload→polyfill)")

    if n_already and not n_patched:
        print(f"✓ Notebook déjà patché ({n_already} cellule(s) avec sentinelle).")
        return 0
    if not n_patched:
        print("[ERROR] Aucune cellule avec '%load_ext autoreload' trouvée.")
        return 2

    NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"💾 Saved {NB} ({n_patched} cell(s) patched, {n_already} already OK)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
