"""Remove misleading pre-recompute, restore official V5-mini baseline as comparison
reference, and align post-FT recompute params with V5-mini training (K=64, n_steps=32).

Why : the pre-recompute step I added in round 3 was producing metrics with
inflated/inflated-down values (e.g. F1-p99 jumped from 0.512 to 0.808 on the
SAME baseline model) because it used K=12, n_steps=18 and 16 random samples
— all DIFFERENT from V5-mini training (K=64, n_steps=32, batched val).
This created an illusion of fairness while masking the protocol shift.

The honest fix : remove the pre-recompute entirely. Use the official
V5-mini final_validation_metrics.json as the baseline reference. For
post-FT, run recompute with EXACTLY the same params V5-mini training used.

Changes :
1. Cell 6 : remove the pre-recompute block (_v5_aligned / _nc_aligned writes)
2. Cell 8 : revert load_metrics to read only the official final_validation_metrics.json
3. Cell 6 : post-FT recompute call now uses K_samples=64, n_steps=32
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NB = Path("st_cdgm_v5_evaluation.ipynb")
nb = json.load(NB.open(encoding="utf-8"))


# ============================================================
# Cell 6 : remove pre-recompute block
# ============================================================
src6 = "".join(nb["cells"][6]["source"])

PRE_RECOMPUTE_BLOCK_START = "# === Pre-Phase F : recompute baselines avec formules alignees ==="
PRE_RECOMPUTE_BLOCK_END = "import warnings  # safety\n"

if PRE_RECOMPUTE_BLOCK_START in src6 and PRE_RECOMPUTE_BLOCK_END in src6:
    start = src6.find(PRE_RECOMPUTE_BLOCK_START)
    end = src6.find(PRE_RECOMPUTE_BLOCK_END) + len(PRE_RECOMPUTE_BLOCK_END)
    src6 = src6[:start] + src6[end:]
    print("[OK] Cell 6 : bloc pre-recompute supprime")

# Update the post-FT recompute call to use V5-mini training params
OLD_RECOMPUTE_CALL = """        K_samples=12, n_steps=18, n_batches=16,
        epoch=EPOCHS, causal_concat=True,
    )"""

NEW_RECOMPUTE_CALL = """        # K=64, n_steps=32 = exactement les params V5-mini training pour
        # que les nombres soient directement comparables au baseline publie.
        # n_batches=16 = meme nombre de batches que V5-mini training.
        K_samples=64, n_steps=32, n_batches=16,
        epoch=EPOCHS, causal_concat=True,
    )"""

if OLD_RECOMPUTE_CALL in src6:
    src6 = src6.replace(OLD_RECOMPUTE_CALL, NEW_RECOMPUTE_CALL)
    print("[OK] Cell 6 : post-FT recompute -> K=64, n_steps=32 (= V5-mini training)")

nb["cells"][6]["source"] = [l + "\n" for l in src6.split("\n")[:-1]] + [src6.split("\n")[-1]]


# ============================================================
# Cell 8 : revert load_metrics (remove aligned preference)
# ============================================================
src8 = "".join(nb["cells"][8]["source"])

OLD_LOAD = '''def load_metrics(d):
    """Load aligned baseline if present (post-round3 fix for apples-to-apples).

    Order de priorite :
    1. final_validation_metrics_aligned.json (formules nouvelles, baseline pre-recompute)
    2. final_validation_metrics.json (legacy, baseline V5-mini ou CorrDiff training-time)
    """
    p_aligned = Path(d) / "final_validation_metrics_aligned.json"
    if p_aligned.exists():
        return json.loads(p_aligned.read_text(encoding="utf-8"))
    p = Path(d) / "final_validation_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))'''

NEW_LOAD = '''def load_metrics(d):
    """Lit le JSON officiel produit par le training original (V5-mini ou CorrDiff).

    Ne lit PAS les _aligned.json (pre-recompute supprime car produisait des
    chiffres incompatibles avec le baseline publie a cause de params
    d'inference differents).
    """
    p = Path(d) / "final_validation_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))'''

if OLD_LOAD in src8:
    src8 = src8.replace(OLD_LOAD, NEW_LOAD)
    print("[OK] Cell 8 : load_metrics revertee, lit uniquement le JSON officiel")

nb["cells"][8]["source"] = [l + "\n" for l in src8.split("\n")[:-1]] + [src8.split("\n")[-1]]


# ============================================================
# Save
# ============================================================
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print("=" * 60)
print("[OK] Patch applique")
print("=" * 60)
print()
print("Resume des changements :")
print("  1. Cell 6 : suppression du pre-recompute baselines (etait trompeur)")
print("  2. Cell 8 : retour a la lecture du JSON officiel V5-mini comme baseline")
print("  3. Cell 6 : post-FT recompute utilise K=64, n_steps=32 (= V5-mini training)")
print()
print("Impact :")
print("  - Phase 6 comparera : V5-mini publie (officiel) vs post-FT (mêmes params)")
print("  - Les nombres post-FT seront DIRECTEMENT comparables au baseline publie")
print("  - Cout : recompute post-FT prendra plus longtemps (~30 min vs 10 min)")
print()
print("Action utilisateur :")
print("  - Stopper la cellule Cell 6 si pas encore au stade training")
print("  - Si tu veux nettoyer les aligned JSONs deja crees :")
print("    !rm /content/drive/MyDrive/climate_data/ckpt_v2_corrdiff_normal/final_validation_metrics_aligned.json")
print("    !rm /content/drive/MyDrive/climate_data/ckpt_noncausal/final_validation_metrics_aligned.json")
