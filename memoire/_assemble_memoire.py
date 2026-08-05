"""
Assemble memoire_complet.tex from standalone .tex files.

Strategy:
- Use the preamble of the existing memoire.tex (lines 1 to \begin{document}).
- Extract body of each standalone file (between \begin{document} and \end{document}).
- Concatenate in correct order: pages_prelim, intro_generale, chap1, chap2, chap3, chap4, conclusion.
- Reuse the existing bibliography block (\begin{thebibliography} ... \end{thebibliography}) from memoire.tex.
- Close with \end{document}.

Each standalone file's preamble/setcounter is stripped; only the document body is kept.
"""
import re
import os
from pathlib import Path

BASE = Path(r"c:\Users\reall\Desktop\climate_data\memoire")
OUTPUT = BASE / "memoire_complet.tex"
EXISTING_MASTER = BASE / "memoire.tex"


def extract_body(path: Path, strip_chapter_setcounter: bool = True) -> str:
    """Extract content between \\begin{document} and \\end{document}.

    Also strips:
    - inline \\begin{thebibliography}...\\end{thebibliography} blocks (chapters
      that were standalone-compilable have their own local bibliography; we
      consolidate to a single bibliography at the end of the document).
    - stray \\setcounter / \\pagenumbering that were only meant for standalone.
    """
    text = path.read_text(encoding="utf-8")
    m_start = re.search(r"\\begin\{document\}", text)
    m_end = re.search(r"\\end\{document\}", text)
    if not (m_start and m_end):
        raise ValueError(f"Cannot find document markers in {path.name}")
    body = text[m_start.end(): m_end.start()].strip()

    # Strip any embedded bibliography block (we keep only the global one at the end)
    body = re.sub(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
        "",
        body,
        flags=re.DOTALL,
    )
    # Also strip the bibliography section header if it stands alone before the block
    body = re.sub(
        r"%\s*=+\s*\n%\s*Bibliographie[^\n]*\n%\s*=+\s*\n",
        "",
        body,
        flags=re.IGNORECASE,
    )

    if strip_chapter_setcounter:
        body = re.sub(r"^\\setcounter\{(chapter|page)\}\{\d+\}\s*$", "", body, flags=re.MULTILINE)
        body = re.sub(r"^\\pagenumbering\{(roman|arabic)\}\s*$", "", body, flags=re.MULTILINE)
    return body.strip()


def extract_master_preamble_and_biblio() -> tuple[str, str]:
    """From existing memoire.tex, get preamble (up to \\begin{document}) and bibliography block."""
    text = EXISTING_MASTER.read_text(encoding="utf-8")
    # Preamble: everything before \begin{document}
    m_begin = re.search(r"\\begin\{document\}", text)
    preamble = text[: m_begin.end()]
    # Bibliography: from \begin{thebibliography} to \end{thebibliography} (inclusive)
    m_bib = re.search(
        r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}",
        text,
        re.DOTALL,
    )
    biblio = m_bib.group(0) if m_bib else ""
    return preamble, biblio


def main() -> None:
    preamble, biblio = extract_master_preamble_and_biblio()

    parts = []
    parts.append(preamble)

    parts.append("\n% Pages preliminaires en chiffres romains\n\\pagenumbering{roman}\n")
    parts.append(extract_body(BASE / "pages_preliminaires.tex"))

    parts.append("\n\\cleardoublepage\n\\tableofcontents\n\\cleardoublepage\n")
    parts.append("\n% Corps du memoire en chiffres arabes\n\\pagenumbering{arabic}\n\\setcounter{page}{1}\n")

    parts.append(extract_body(BASE / "introduction_generale.tex"))
    parts.append("\n\\cleardoublepage\n")

    parts.append("\n% ============= CHAPITRE I =============\n")
    parts.append(extract_body(BASE / "chapitre_1.tex"))
    parts.append("\n\\cleardoublepage\n")

    parts.append("\n% ============= CHAPITRE II =============\n")
    parts.append(extract_body(BASE / "chapitre_2.tex"))
    parts.append("\n\\cleardoublepage\n")

    parts.append("\n% ============= CHAPITRE III =============\n")
    parts.append(extract_body(BASE / "chapitre_3.tex"))
    parts.append("\n\\cleardoublepage\n")

    parts.append("\n% ============= CHAPITRE IV =============\n")
    parts.append(extract_body(BASE / "chapitre_4.tex"))
    parts.append("\n\\cleardoublepage\n")

    parts.append("\n% ============= CONCLUSION =============\n")
    parts.append(extract_body(BASE / "conclusion_generale.tex"))
    parts.append("\n\\cleardoublepage\n")

    parts.append("\n% ============= BIBLIOGRAPHIE =============\n")
    parts.append(biblio)
    parts.append("\n\n\\end{document}\n")

    OUTPUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Wrote {OUTPUT} ({sum(len(p.splitlines()) for p in parts)} lines)")


if __name__ == "__main__":
    main()
