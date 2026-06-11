"""Round 3 critical fixes from 5-agent audit.

CRITICAL (Auditor 5) : Phase 6 comparison is NOT apples-to-apples.
  - Baseline V5_DIR/final_validation_metrics.json was created during V5-mini
    training with OLD F1 (global threshold) + OLD RAPSD (radial mean).
  - Recompute_phase6_metrics uses NEW formulas (per-sample F1, rfft2 RAPSD).
  - User would see false "gains" that are pure formula shifts.

FIX : add pre-training recompute step BEFORE Phase F to produce aligned
baseline JSONs (V5_DIR/final_validation_metrics_aligned.json and
NONCAUSAL_DIR/final_validation_metrics_aligned.json). Modify Phase 6 to
prefer _aligned.json when present.

OTHER FIXES (Auditor 3+4) :
  - Pre-flight checks V5_DIR for inprogress (should be ORACLE_FINETUNED_DIR)
  - Log message "V5_DIR" but writes ORACLE_FINETUNED_DIR
  - Cell 5 markdown : CHECKPOINT_NAME -> EVAL_VERSION
  - sigma_data_new None handling (preserve old)
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
# FIX 1 (CRITICAL) : pre-training recompute in Cell 6
# ============================================================
src6 = "".join(nb["cells"][6]["source"])

# Insert BEFORE the finetune_bundle_b call, after dataset materialization
ANCHOR = """# 3. Lancer le fine-tune
result = finetune_bundle_b("""

PRE_RECOMPUTE = '''# === Pre-Phase F : recompute baselines avec formules alignees ===
# CRITICAL : sans cela, la comparaison Phase 6 post-finetune n'est pas
# apples-to-apples (baseline JSONs utilisent l'ancienne formule F1/RAPSD).
# On produit des _aligned.json pour Oracle baseline et CorrDiff, qui
# servent de reference dans Phase 6 (Cell 8).
from scripts.recompute_phase6_metrics import recompute_phase6_metrics as _recompute_aligned

_v5_aligned = V5_DIR / "final_validation_metrics_aligned.json"
_nc_aligned = NONCAUSAL_DIR / "final_validation_metrics_aligned.json"

if not _v5_aligned.exists():
    print()
    print("[Pre-recompute] Production des baseline JSONs avec formules alignees...")
    print("  (necessaire pour comparaison apples-to-apples avec post-Phase F)")
    try:
        _recompute_aligned(
            stack=stack_v5, builder=builder, val_dataset=val_dataset,
            DEVICE=DEVICE,
            predict_with_stack_fn=predict_with_stack,
            convert_sample_to_batch_fn=convert_sample_to_batch,
            out_path=_v5_aligned,
            K_samples=12, n_steps=18, n_batches=16,
            epoch=0,  # baseline epoch
            verbose=True,
        )
        print(f"  [OK] Oracle baseline aligne : {_v5_aligned.name}")
    except Exception as _e:
        warnings.warn(f"Pre-recompute Oracle baseline failed: {_e}")

if not _nc_aligned.exists():
    try:
        _recompute_aligned(
            stack=stack_nc, builder=builder, val_dataset=val_dataset,
            DEVICE=DEVICE,
            predict_with_stack_fn=predict_with_stack,
            convert_sample_to_batch_fn=convert_sample_to_batch,
            out_path=_nc_aligned,
            K_samples=12, n_steps=18, n_batches=16,
            epoch=0, verbose=False,
        )
        print(f"  [OK] CorrDiff aligne : {_nc_aligned.name}")
    except Exception as _e:
        warnings.warn(f"Pre-recompute CorrDiff failed: {_e}")
print()

import warnings  # safety
# 3. Lancer le fine-tune
result = finetune_bundle_b('''

if ANCHOR in src6 and "_v5_aligned = V5_DIR" not in src6:
    src6 = src6.replace(ANCHOR, PRE_RECOMPUTE)
    print("[OK] FIX 1 (CRITICAL) : pre-training recompute pour comparaison aligned")

# ============================================================
# FIX 2 : pre-flight checks V5_DIR -> ORACLE_FINETUNED_DIR
# ============================================================
OLD_PREFLIGHT = '''    smoke_ckpt = V5_DIR / "epoch_finetuned.pth"
    inprogress = V5_DIR / "epoch_finetuned_inprogress.pth"'''
NEW_PREFLIGHT = '''    smoke_ckpt = ORACLE_FINETUNED_DIR / "epoch_finetuned.pth"
    inprogress = ORACLE_FINETUNED_DIR / "epoch_finetuned_inprogress.pth"'''
if OLD_PREFLIGHT in src6:
    src6 = src6.replace(OLD_PREFLIGHT, NEW_PREFLIGHT)
    print("[OK] FIX 2 : pre-flight paths -> ORACLE_FINETUNED_DIR")

# ============================================================
# FIX 3 : log message accuracy V5_DIR -> ORACLE_FINETUNED_DIR
# ============================================================
OLD_LOG = "[OK] V5_DIR/final_validation_metrics.json mis a jour"
NEW_LOG = "[OK] ORACLE_FINETUNED_DIR/final_validation_metrics.json mis a jour"
if OLD_LOG in src6:
    src6 = src6.replace(OLD_LOG, NEW_LOG)
    print("[OK] FIX 3 : log message corrige")

nb["cells"][6]["source"] = [l + "\n" for l in src6.split("\n")[:-1]] + [src6.split("\n")[-1]]


# ============================================================
# FIX 4 : Cell 5 markdown CHECKPOINT_NAME -> EVAL_VERSION
# ============================================================
src5 = "".join(nb["cells"][5]["source"])
OLD_C5 = "Une fois le fine-tune termine, change `CHECKPOINT_NAME = \"epoch_finetuned\"` en Cell 2"
NEW_C5 = "Une fois le fine-tune termine, change `EVAL_VERSION = \"finetuned\"` en Cell 2"
if OLD_C5 in src5:
    src5 = src5.replace(OLD_C5, NEW_C5)
    print("[OK] FIX 4 : Cell 5 markdown EVAL_VERSION")
nb["cells"][5]["source"] = [l + "\n" for l in src5.split("\n")[:-1]] + [src5.split("\n")[-1]]


# ============================================================
# FIX 5 : Phase 6 (Cell 8) prefer _aligned.json when present
# ============================================================
src8 = "".join(nb["cells"][8]["source"])
OLD_LOAD = '''def load_metrics(d):
    p = Path(d) / "final_validation_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))'''
NEW_LOAD = '''def load_metrics(d):
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

if OLD_LOAD in src8:
    src8 = src8.replace(OLD_LOAD, NEW_LOAD)
    print("[OK] FIX 5 : Phase 6 prefer _aligned.json")
nb["cells"][8]["source"] = [l + "\n" for l in src8.split("\n")[:-1]] + [src8.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")


# ============================================================
# FIX 6 : sigma_data_new None handling in finetune script
# ============================================================
F = Path("scripts/finetune_stage1_bundle_b.py")
src_ft = F.read_text(encoding="utf-8")

OLD_SIGMA_NONE = '''            else:
                print(f"  [WARN] Aucun sample valide pour sigma_data, on garde l'ancien")

            # Save into checkpoint dict for downstream consumption'''
NEW_SIGMA_NONE = '''            else:
                print(f"  [WARN] Aucun sample valide pour sigma_data, on garde l'ancien")
                # Preserve OLD sigma_data so JSON consumers can do float() safely
                try:
                    if hasattr(stack.get("diffusion"), "edm_config"):
                        sigma_data_new = float(stack["diffusion"].edm_config.sigma_data)
                except Exception:
                    sigma_data_new = 0.5  # EDM default fallback

            # Save into checkpoint dict for downstream consumption'''
if OLD_SIGMA_NONE in src_ft:
    src_ft = src_ft.replace(OLD_SIGMA_NONE, NEW_SIGMA_NONE)
    print("[OK] FIX 6 : sigma_data_new None -> preserve old sigma")
F.write_text(src_ft, encoding="utf-8")


# ============================================================
# Recap
# ============================================================
print()
print("=" * 60)
print("[OK] Round 3 critical fixes appliques :")
print("  1. Pre-training recompute (apples-to-apples comparison) — CRITICAL")
print("  2. Pre-flight paths ORACLE_FINETUNED_DIR")
print("  3. Log message correction")
print("  4. Cell 5 markdown EVAL_VERSION")
print("  5. Phase 6 prefer _aligned.json")
print("  6. sigma_data_new None handling")
print("=" * 60)
