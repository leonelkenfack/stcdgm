"""Build the these/-style modular structure for the memoire.
Extract body (between \\begin{document} and \\end{document}) of each standalone .tex
and place it as a fragment in My-Thesis/<section>/<name>.tex.
Then create:
- defs-and-imports.tex (preamble shared from memoire.tex master)
- My-Thesis.tex (point d'entree using \\input{} on each fragment)
"""
import re
from pathlib import Path

BASE = Path(r"c:\Users\reall\Desktop\climate_data\memoire")
THESIS = BASE / "My-Thesis"


def extract_body(path: Path) -> str:
    """Return content between \\begin{document} and \\end{document}, stripped of
    inline \\begin{thebibliography} blocks and stray \\setcounter / \\pagenumbering."""
    text = path.read_text(encoding="utf-8")
    m_start = re.search(r"\\begin\{document\}", text)
    m_end = re.search(r"\\end\{document\}", text)
    if not (m_start and m_end):
        raise ValueError(f"cannot find document markers in {path.name}")
    body = text[m_start.end(): m_end.start()].strip()
    body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", body, flags=re.DOTALL)
    body = re.sub(r"^\\setcounter\{(chapter|page)\}\{\d+\}\s*$", "", body, flags=re.MULTILINE)
    body = re.sub(r"^\\pagenumbering\{(roman|arabic)\}\s*$", "", body, flags=re.MULTILINE)
    return body.strip()


def extract_preamble(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"\\begin\{document\}", text)
    return text[: m.start()]


def extract_biblio(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", text, re.DOTALL)
    return m.group(0) if m else ""


# 1. defs-and-imports.tex : maintenant maintenu manuellement (numerotation par
#    chapitre, etc.). NE PAS overwriter depuis memoire.tex.
# preamble = extract_preamble(BASE / "memoire.tex")
# (BASE / "defs-and-imports.tex").write_text(preamble, encoding="utf-8")
print("defs-and-imports.tex: PRESERVED (maintenu manuellement)")

# 2. Build each fragment in My-Thesis/<dir>/<name>.tex
MAPPING = {
    # NOTE: pages_preliminaires.tex DESACTIVE — pages preliminaires sont
    # maintenant gerees individuellement (titlepage, Dedication, Acknowledgements,
    # Resume, Abstract, list-of-acronyms) directement dans My-Thesis/.
    # NOTE: introduction_generale.tex DESACTIVE — Introduction.tex est
    # maintenue manuellement dans My-Thesis/Introduction/ (style FEUZING).
    # "introduction_generale.tex": ("Introduction", "Introduction.tex"),
    "chapitre_1.tex": ("Chap1", "Chap1.tex"),
    "chapitre_2.tex": ("Chap2", "Chap2.tex"),
    "chapitre_3.tex": ("Chap3", "Chap3.tex"),
    "chapitre_4.tex": ("Chap4", "Chap4.tex"),
    "conclusion_generale.tex": ("Conclusion", "Conclusion.tex"),
}

for src, (subdir, dst) in MAPPING.items():
    src_path = BASE / src
    if not src_path.exists():
        print(f"SKIP {src}: not found")
        continue
    body = extract_body(src_path)
    out = THESIS / subdir / dst
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body, encoding="utf-8")
    print(f"  wrote {out.relative_to(BASE)} ({len(body.splitlines())} lines)")

# 3. Extract biblio
# DESACTIVE: bibliography_block.tex est maintenant maintenu manuellement
# (apres reduction de moitie via _reduce_bibliography.py). Ne pas overwrite.
# biblio = extract_biblio(BASE / "memoire.tex")
# (THESIS / "Appendices").mkdir(parents=True, exist_ok=True)
# (THESIS / "Appendices" / "bibliography_block.tex").write_text(biblio, encoding="utf-8")
print("bibliography block: PRESERVED (maintenu manuellement)")

# 4. Write My-Thesis.tex master file
master = r"""% =============================================================
% My-Thesis.tex --- Memoire ORACLE, point d'entree modulaire
% Structure inspiree de la these URIFIA (these/My-Thesis.tex)
% Compile avec: pdflatex -interaction=nonstopmode My-Thesis.tex
% =============================================================

% Import du preambule partage (style URIFIA)
\input{defs-and-imports.tex}

\begin{document}

% --- Pages preliminaires ---
% Pages preliminaires en chiffres romains
\pagenumbering{roman}

% Page de garde
\input{My-Thesis/other-pages/titlepage.tex}

% Dedicace
\input{My-Thesis/Dedication/Dedication.tex}

% Remerciements
\input{My-Thesis/Acknowledgements/Acknowledgements.tex}

% Table des matieres
\cleardoublepage
\tableofcontents

% Liste des figures
\cleardoublepage
\listoffigures
\addcontentsline{toc}{chapter}{Liste des figures}

% Liste des tableaux
\cleardoublepage
\listoftables
\addcontentsline{toc}{chapter}{Liste des tableaux}

% Liste des abreviations et sigles
\input{My-Thesis/other-pages/list-of-acronyms.tex}

% Resume FR
\input{My-Thesis/Resume/Resume.tex}

% Abstract EN
\input{My-Thesis/Abstract/Abstract.tex}

\cleardoublepage

% Corps en chiffres arabes
\pagenumbering{arabic}
\setcounter{page}{1}

% --- Introduction generale ---
\input{My-Thesis/Introduction/Introduction.tex}
\cleardoublepage

% --- Chapitres ---
\input{My-Thesis/Chap1/Chap1.tex}
\cleardoublepage
\input{My-Thesis/Chap2/Chap2.tex}
\cleardoublepage
\input{My-Thesis/Chap3/Chap3.tex}
\cleardoublepage
\input{My-Thesis/Chap4/Chap4.tex}
\cleardoublepage

% --- Conclusion + Perspectives ---
\input{My-Thesis/Conclusion/Conclusion.tex}
\cleardoublepage

% --- Annexes (deplacees depuis le Chapitre IV) ---
\input{My-Thesis/Appendices/Appendix_Chap4.tex}
\cleardoublepage

% --- Bibliographie ---
\input{My-Thesis/Appendices/bibliography_block.tex}

\end{document}
"""
(BASE / "My-Thesis.tex").write_text(master, encoding="utf-8")
print("My-Thesis.tex written")
print("Done.")
