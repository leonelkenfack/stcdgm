"""Force always-finetuned evaluation in st_cdgm_v5_evaluation.ipynb.

Goal : remove the manual EVAL_VERSION / CHECKPOINT_NAME switch. After this
patch, Cells 4, 8, 10, 12, 14, 16, 18, 20 unconditionally evaluate the
fine-tuned Oracle (ORACLE_FINETUNED_DIR/epoch_finetuned.pth).

Changes :

  Cell 2 : remove EVAL_VERSION selector, hardcode ORACLE_DIR = ORACLE_FINETUNED_DIR
           and CHECKPOINT_NAME = "epoch_finetuned". Add a one-time bootstrap
           copy V5_DIR/epoch_last.pth -> ORACLE_FINETUNED_DIR/epoch_finetuned.pth
           so Cell 4 can boot even before Phase F has ever run.

  Cell 6 : after Phase F finishes, force re-load stack_v5 from disk so the
           subsequent phases see the freshly saved FT weights even if the user
           later restarts the kernel and re-runs from Cell 4. Update terminal
           hint to drop the manual EVAL_VERSION switch instruction.

The baseline directory V5_DIR/ is read ONLY by Phase 6 (Cell 8) for the
post-FT vs baseline-Oracle vs CorrDiff comparison. It is never written to.
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
# Cell 2 : drop EVAL_VERSION switch
# ============================================================
src2 = "".join(nb["cells"][2]["source"])

OLD_SELECTOR = '''# Phase F (2026-06-11) : nouveau dir checkpoint pour preserver Oracle baseline
ORACLE_FINETUNED_DIR = Path("/content/drive/MyDrive/climate_data/oracle_finetuned")
# Selector : baseline = lit V5_DIR (intact) ; finetuned = lit ORACLE_FINETUNED_DIR
EVAL_VERSION = "baseline"  # change a "finetuned" apres Phase F pour evaluer le nouveau modele
ORACLE_DIR = ORACLE_FINETUNED_DIR if EVAL_VERSION == "finetuned" else V5_DIR
print(f"[Phase F] EVAL_VERSION = {EVAL_VERSION} -> ORACLE_DIR = {ORACLE_DIR.name}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === Phase G (post-Oracle baseline, 2026-06-09) — selecteur de checkpoint ===
# "epoch_last" = Oracle baseline (gel §10.8 du memoire)
# "epoch_finetuned" = checkpoint post-Phase F (Bundle B + CASTLE + G_phys)
CHECKPOINT_NAME = "epoch_last"   # change a "epoch_finetuned" pour evaluer post-train
print(f"[Phase G] CHECKPOINT_NAME = {CHECKPOINT_NAME}")'''

NEW_SELECTOR = '''# Phase F (2026-06-11) : evaluation TOUJOURS sur la version fine-tunee.
# Plus de selecteur conditionnel : ORACLE_DIR pointe toujours vers
# ORACLE_FINETUNED_DIR. Le baseline V5_DIR/ reste intact, lu UNIQUEMENT par
# Phase 6 (Cell 8) pour la comparaison FT vs baseline Oracle vs CorrDiff.
ORACLE_FINETUNED_DIR = Path("/content/drive/MyDrive/climate_data/oracle_finetuned")
ORACLE_DIR = ORACLE_FINETUNED_DIR
CHECKPOINT_NAME = "epoch_finetuned"

# Bootstrap (one-time) : si premier run et Phase F pas encore lancee, on
# copie le baseline Oracle pour que Cell 4 puisse booter le stack. Phase F
# (Cell 6) ecrasera ensuite ce fichier avec les vrais poids fine-tunes.
ORACLE_FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
_ft_ckpt = ORACLE_FINETUNED_DIR / f"{CHECKPOINT_NAME}.pth"
if not _ft_ckpt.exists():
    _baseline_ckpt_src = V5_DIR / "epoch_last.pth"
    if _baseline_ckpt_src.exists():
        import shutil as _sh
        print(f"[Bootstrap] {_ft_ckpt.name} absent — copie depuis baseline pour booter")
        print(f"            (Phase F ecrasera apres entrainement)")
        _sh.copy(_baseline_ckpt_src, _ft_ckpt)
    else:
        print(f"[ERREUR] Ni {_ft_ckpt} ni {_baseline_ckpt_src} n'existent")
print(f"[Eval] ORACLE_DIR = {ORACLE_DIR.name}  CHECKPOINT_NAME = {CHECKPOINT_NAME}")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)'''

if OLD_SELECTOR in src2:
    src2 = src2.replace(OLD_SELECTOR, NEW_SELECTOR)
    print("[OK] Cell 2 : EVAL_VERSION switch supprime, force toujours-FT")
else:
    print("[SKIP] Cell 2 : selector deja patche ou format inattendu")

nb["cells"][2]["source"] = [l + "\n" for l in src2.split("\n")[:-1]] + [src2.split("\n")[-1]]


# ============================================================
# Cell 6 : reload stack_v5 from disk after Phase F + update hint
# ============================================================
src6 = "".join(nb["cells"][6]["source"])

# 1. Drop the misleading instruction about EVAL_VERSION switch
OLD_HINT_BLOCK = '''print("PROCHAINE ETAPE :")
print("  1. Si SMOKE_TEST = True : passe a SMOKE_TEST = False et relance cette cellule")
print("  2. Une fois fine-tune complet : change EVAL_VERSION = \\"finetuned\\" en Cell 2")
print("  3. Re-execute Cell 4 (recharge stack depuis ORACLE_FINETUNED_DIR), puis Cells 8, 10, 12, 14, 16, 18, 20, 22 (Phases 6-12)")
print("  4. Compare via : !python -m scripts.compare_eval_results ...")'''

NEW_HINT_BLOCK = '''print("PROCHAINE ETAPE :")
print("  1. Si SMOKE_TEST = True : passe a SMOKE_TEST = False et relance cette cellule")
print("  2. Lance Cells 8, 10, 12, 14, 16, 18, 20, 22 (Phases 6-12)")
print("     -> ORACLE_DIR pointe deja vers ORACLE_FINETUNED_DIR (cell 2 hardcoded)")
print("     -> stack_v5 contient deja les poids FT en memoire (modifies in-place")
print("        par finetune_bundle_b) ET a ete rechargees depuis le disque ci-dessus.")
print("  3. Compare via : !python -m scripts.compare_eval_results ...")'''

if OLD_HINT_BLOCK in src6:
    src6 = src6.replace(OLD_HINT_BLOCK, NEW_HINT_BLOCK)
    print("[OK] Cell 6 : hint post-train mis a jour (plus de switch manuel)")

# 2. After finetune + recompute, reload stack_v5 from disk to guarantee
#    stale-kernel safety (kernel restart between Cell 6 and Cells 8-22 will
#    re-run Cell 4 which loads ORACLE_FINETUNED_DIR/epoch_finetuned.pth).
ANCHOR_RELOAD = '''# Memory cleanup : free 9-10 GB avant Phases 7-12
print()
print("[Cleanup] Liberation memoire post-training...")'''

RELOAD_BLOCK = '''# === Force reload de stack_v5 depuis disque ===
# Garantit que les Phases 6-12 voient les poids FT meme si le kernel est
# restart entre temps. Cell 4 fera la meme chose au prochain boot.
print()
print("[Phase F] Reload stack_v5 depuis ORACLE_FINETUNED_DIR/epoch_finetuned.pth...")
try:
    _ft_path = ORACLE_FINETUNED_DIR / "epoch_finetuned.pth"
    if _ft_path.exists():
        stack_v5 = build_stack(_ft_path, "Oracle-FT")
        print(f"  [OK] stack_v5 recharge depuis {_ft_path.name}")
    else:
        print(f"  [WARN] {_ft_path} absent — stack_v5 garde les poids in-memory du training")
except Exception as _e:
    print(f"  [WARN] Reload echec : {type(_e).__name__}: {_e}")
    print(f"         stack_v5 garde les poids in-memory (toujours valides)")

# Memory cleanup : free 9-10 GB avant Phases 7-12
print()
print("[Cleanup] Liberation memoire post-training...")'''

if ANCHOR_RELOAD in src6 and "stack_v5 = build_stack(_ft_path" not in src6:
    src6 = src6.replace(ANCHOR_RELOAD, RELOAD_BLOCK)
    print("[OK] Cell 6 : reload stack_v5 post-FT ajoute")

nb["cells"][6]["source"] = [l + "\n" for l in src6.split("\n")[:-1]] + [src6.split("\n")[-1]]


# ============================================================
# Save
# ============================================================
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print("=" * 60)
print("[OK] Notebook patche : toujours-FT")
print("=" * 60)
print()
print("Comportement attendu :")
print("  - Cell 2 : ORACLE_DIR=ORACLE_FINETUNED_DIR, CHECKPOINT_NAME=epoch_finetuned")
print("  - Cell 4 : charge automatiquement ORACLE_FINETUNED_DIR/epoch_finetuned.pth")
print("  - Cell 6 : entraine + recompute Phase 6 + reload stack_v5 from disk")
print("  - Cells 8-22 : utilisent stack_v5 (FT), ORACLE_DIR=FT")
print("  - V5_DIR/ (baseline) : LU uniquement par recompute_phase6 si necessaire")
print("                         pour comparaison. JAMAIS ecrit.")
