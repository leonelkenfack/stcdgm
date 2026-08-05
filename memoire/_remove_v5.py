"""Replace V5 markers with Oracle in main text.
Preserves historical trajectory mentions (V4, V5-mini, V5 complète) which are needed for narrative.
"""
import re
from pathlib import Path

BASE = Path(r"c:\Users\reall\Desktop\climate_data\memoire")

def _r(s): return s  # marker for replacement (no escape processing)

REPLACEMENTS = [
    (r"\\Oracle\{\}\s+V5\b", r"\\\\Oracle{}"),
    (r"\\Oracle\{\}~V5\b", r"\\\\Oracle{}"),
    (r"\bORACLE\s+V5\b", "ORACLE"),
    (r"\bOracle\s+V5\b", "Oracle"),
    (r"\bla\s+variante\s+V5\s+finale\b", "la variante finale"),
    (r"\bvariante\s+V5\s+(?:finale|mini|complète|retenue)\b", "variante finale"),
    (r"\bconfiguration\s+V5\s+(?:finale|retenue)\b", "configuration finale"),
    (r"&\s*V5\s+sous-prédit\b", r"& \\\\Oracle{} sous-prédit"),
    (r"mesurée\s+en\s+V5\b", r"mesurée pour \\\\Oracle{}"),
    (r"dans\s+sa\s+configuration\s+V5\s+finale", "dans sa configuration finale"),
    (r"La valeur mesurée en V5 est", "La valeur mesurée est"),
    (r"Topologie effective d'\\Oracle\{\}\s+V5", r"Topologie effective d'\\\\Oracle{}"),
]

# Files to process
FILES = ["chapitre_4.tex", "introduction_generale.tex", "conclusion_generale.tex", "pages_preliminaires.tex"]

for fname in FILES:
    fp = BASE / fname
    if not fp.exists():
        print(f"{fname}: not found")
        continue
    text = fp.read_text(encoding="utf-8")
    orig = text
    for pattern, replacement in REPLACEMENTS:
        text = re.sub(pattern, replacement, text)
    if text != orig:
        fp.write_text(text, encoding="utf-8")
        # count V5 still present
        remaining = len(re.findall(r"\bV5\b", text))
        print(f"{fname}: modified, {remaining} V5 remaining (likely in trajectory context)")
    else:
        print(f"{fname}: unchanged")
