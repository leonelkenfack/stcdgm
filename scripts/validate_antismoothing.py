#!/usr/bin/env python3
"""
Validation rapide des garde-fous anti-lissage (prompt v6, Phase 4).
Exécute les tests unitaires dédiés sans lancer un entraînement complet.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    tests = ROOT / "tests" / "test_corrections_antilissage.py"
    cmd = [sys.executable, "-m", "pytest", str(tests), "-q", "--tb=short"]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
