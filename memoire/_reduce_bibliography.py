"""Reduce bibliography: cut ~half of inproceedings + half of articles.
Removes bibitems from bibliography_block.tex AND removes/edits \\cite{} calls in all .tex.
"""
import re
from pathlib import Path

BASE = Path(r"c:\Users\reall\Desktop\climate_data\memoire")
THESIS = BASE / "My-Thesis"

# Keys to CUT (27 refs). Keep ~30 essentials.
CUT_KEYS = {
    # Super-resolution secondary
    "dong_srcnn_2014",
    "lim_edsr_2017",
    "liang_swinir_2021",
    "pan_pan_2019",
    "saharia_sr3_2022",
    "yue_resshift_2023",
    "rombach_latent_2022",
    "song_consistency_2023",
    # Older/specialized downscaling
    "groenke_flow_2020",
    "lange_vae_2022",
    "li_fno_2021",
    "bonev_sfno_2023",
    "price_rasp_2022",
    "lopezgomez_precipformer_2023",
    "wilby_wigley_1997",
    "watson_climatebench_2022",
    "rasp_weatherbench2_2023",
    "almazroui_2020",
    "lin_hgscm_2023",
    # Technical helpers
    "fey_pyg_2019",
    "hoyer_xarray_2017",
    "yadan_hydra_2019",
    "lu_dpmpp_2022",
    # Reports + uncited
    "wmo_gcos_2022",
    "sundararajan_ig_2017",  # not cited
    "nathaniel_chaosbench_2024",
    "spirtes_2000",  # cited but pearl_causality_2009 covers
}


def strip_cites(text: str, cut: set) -> str:
    """Remove \\cite{KEY} entirely if all keys are cut, else strip cut keys."""
    def repl(m):
        full = m.group(0)
        keys_str = m.group(1)
        keys = [k.strip() for k in keys_str.split(",")]
        remaining = [k for k in keys if k not in cut]
        if not remaining:
            return ""  # remove the entire \cite call
        return f"~\\cite{{{','.join(remaining)}}}" if full.startswith("~\\cite") else f"\\cite{{{','.join(remaining)}}}"
    # Handle ~\cite and \cite (with optional tilde)
    text = re.sub(r"~?\\cite\{([^}]+)\}", repl, text)
    return text


# 1. Strip \cite{} calls from all source files
sources = [
    BASE / "chapitre_1.tex",
    BASE / "chapitre_2.tex",
    BASE / "chapitre_3.tex",
    BASE / "chapitre_4.tex",
    BASE / "introduction_generale.tex",
    BASE / "conclusion_generale.tex",
    BASE / "pages_preliminaires.tex",
]

for src in sources:
    if not src.exists():
        continue
    original = src.read_text(encoding="utf-8")
    modified = strip_cites(original, CUT_KEYS)
    if modified != original:
        src.write_text(modified, encoding="utf-8")
        # Count removals
        n_orig = len(re.findall(r"\\cite\{", original))
        n_new = len(re.findall(r"\\cite\{", modified))
        print(f"  {src.name}: {n_orig} -> {n_new} \\cite calls")

# 2. Remove bibitems from bibliography_block.tex
bib_file = THESIS / "Appendices" / "bibliography_block.tex"
text = bib_file.read_text(encoding="utf-8")
n_before = len(re.findall(r"\\bibitem\{", text))

# Remove each bibitem block (from \bibitem{KEY} to the next \bibitem or \end{thebibliography})
for key in CUT_KEYS:
    pattern = re.compile(
        r"\\bibitem\{" + re.escape(key) + r"\}.*?(?=\n\n\\bibitem|\n\\end\{thebibliography\})",
        re.DOTALL,
    )
    text = pattern.sub("", text)

# Cleanup double blank lines
text = re.sub(r"\n{3,}", "\n\n", text)

bib_file.write_text(text, encoding="utf-8")
n_after = len(re.findall(r"\\bibitem\{", text))
print(f"\nbibitems: {n_before} -> {n_after} (cut {n_before - n_after})")
print("Done.")
