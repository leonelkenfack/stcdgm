"""
Pytest conftest: ensure the local ``src/`` layout is on ``sys.path``.

The project uses a ``src/`` layout for the ``st_cdgm`` package but the repo
is not installed in editable mode in every dev environment (notebooks work
because they ``sys.path.insert`` at the top). This conftest does the same
thing for pytest so tests can ``from st_cdgm import ...`` without a prior
``pip install -e .``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
