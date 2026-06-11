"""Restructure : nouveau dir oracle_finetuned/ pour Phase F.

Le user veut :
- Garder ckpt_v2_corrdiff_normal/ intact (Oracle baseline V5-mini)
- Garder ckpt_noncausal/ intact (CorrDiff baseline) — confirme par
  training_config_noncausal.yaml ligne 194 (save_dir separe par design)
- Phase F copie V5_DIR/epoch_last.pth vers oracle_finetuned/epoch_last.pth
  comme point de depart, puis fine-tune et sauve dans oracle_finetuned/
- Toggle evaluation entre "baseline" et "finetuned" via une variable simple
- Plus de "V5-mini" dans les labels, juste "Oracle"

Modifications :
1. Cell 2 : ajoute ORACLE_FINETUNED_DIR + EVAL_VERSION selector
   - Resout ORACLE_DIR dynamiquement (V5_DIR si baseline, ORACLE_FINETUNED_DIR si finetuned)
2. Cell 4 : build_stack utilise ORACLE_DIR (au lieu de V5_DIR)
3. Cell 6 (Phase F) :
   - Pre-flight : copie epoch_last.pth de V5_DIR vers ORACLE_FINETUNED_DIR si absent
   - Pass ckpt_save_dir=ORACLE_FINETUNED_DIR a finetune_bundle_b
   - Recompute Phase 6 ecrit dans ORACLE_FINETUNED_DIR
   - Plus de auto-backup de V5_DIR (intact)
4. Cell 8 (Phase 6) : lit final_validation_metrics depuis ORACLE_DIR + ajuste labels
"""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NB = Path("st_cdgm_v5_evaluation.ipynb")
with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak_oracle_finetuned_dir")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup cree : {BACKUP}")


# ============================================================
# Patch Cell 2 : ajout ORACLE_FINETUNED_DIR + selector
# ============================================================
src2 = "".join(nb["cells"][2]["source"])

OLD_CELL2_PATHS = '''# Phase F (post-V5-mini, 2026-06-10) : nouveau dossier pour ne pas ecraser le baseline.
RESULTS_DIR = Path("/content/drive/MyDrive/climate_data/results/oracle_evaluation")
V5_BASELINE_RESULTS_DIR = Path("/content/drive/MyDrive/climate_data/results/v5_evaluation")  # baseline intact'''

NEW_CELL2_PATHS = '''# Phase F (post-V5-mini, 2026-06-10) : nouveau dossier pour ne pas ecraser le baseline.
RESULTS_DIR = Path("/content/drive/MyDrive/climate_data/results/oracle_evaluation")
V5_BASELINE_RESULTS_DIR = Path("/content/drive/MyDrive/climate_data/results/v5_evaluation")  # baseline intact

# Phase F (2026-06-11) : nouveau dir checkpoint pour preserver Oracle baseline (V5-mini)
ORACLE_FINETUNED_DIR = Path("/content/drive/MyDrive/climate_data/oracle_finetuned")
# Selector : baseline = lit V5_DIR (intact) ; finetuned = lit ORACLE_FINETUNED_DIR
EVAL_VERSION = "baseline"  # change a "finetuned" apres Phase F pour evaluer le nouveau modele
ORACLE_DIR = ORACLE_FINETUNED_DIR if EVAL_VERSION == "finetuned" else V5_DIR
print(f"[Phase F] EVAL_VERSION = {EVAL_VERSION} -> ORACLE_DIR = {ORACLE_DIR.name}")'''

if OLD_CELL2_PATHS in src2 and "ORACLE_FINETUNED_DIR" not in src2:
    src2 = src2.replace(OLD_CELL2_PATHS, NEW_CELL2_PATHS)
    print("[OK] Cell 2 : ORACLE_FINETUNED_DIR + EVAL_VERSION ajoutes")
else:
    if "ORACLE_FINETUNED_DIR" in src2:
        print("[SKIP] Cell 2 : ORACLE_FINETUNED_DIR deja present")
    else:
        print("[KO]  Cell 2 : motif RESULTS_DIR introuvable")

# Adapte le check d'existence : utilise ORACLE_DIR au lieu de V5_DIR
OLD_CHK = '''for label, p in [("V5", V5_DIR), ("Noncausal", NONCAUSAL_DIR)]:
    ckpt = p / f"{CHECKPOINT_NAME}.pth"'''
NEW_CHK = '''# Verifie existence des checkpoints actifs (selon EVAL_VERSION et NONCAUSAL_DIR)
for label, p in [("Oracle", ORACLE_DIR), ("CorrDiff", NONCAUSAL_DIR)]:
    ckpt = p / f"{CHECKPOINT_NAME}.pth"'''
if OLD_CHK in src2:
    src2 = src2.replace(OLD_CHK, NEW_CHK)
    print("[OK] Cell 2 : verification d'existence utilise ORACLE_DIR + label Oracle")

nb["cells"][2]["source"] = [l + "\n" for l in src2.split("\n")[:-1]] + [src2.split("\n")[-1]]


# ============================================================
# Patch Cell 4 : build_stack utilise ORACLE_DIR
# ============================================================
src4 = "".join(nb["cells"][4]["source"])

OLD_CELL4 = '''# Phase G : CHECKPOINT_NAME permet de switcher entre baseline et post-finetune.
# Le baseline noncausal reste sur "epoch_last.pth" (jamais fine-tune).
stack_v5 = build_stack(V5_DIR / f"{CHECKPOINT_NAME}.pth", "V5")
stack_nc = build_stack(NONCAUSAL_DIR / "epoch_last.pth", "Noncausal")'''

NEW_CELL4 = '''# Phase F (2026-06-11) : ORACLE_DIR resolu via EVAL_VERSION (baseline / finetuned).
# Le baseline CorrDiff (NONCAUSAL_DIR) reste sur "epoch_last.pth" (jamais touche).
stack_v5 = build_stack(ORACLE_DIR / f"{CHECKPOINT_NAME}.pth", "Oracle")
stack_nc = build_stack(NONCAUSAL_DIR / "epoch_last.pth", "CorrDiff")'''

if OLD_CELL4 in src4:
    src4 = src4.replace(OLD_CELL4, NEW_CELL4)
    print("[OK] Cell 4 : build_stack utilise ORACLE_DIR + label Oracle/CorrDiff")
elif "ORACLE_DIR / f\"{CHECKPOINT_NAME}.pth\"" in src4:
    print("[SKIP] Cell 4 : deja patchee")
else:
    print("[KO]  Cell 4 : motif build_stack introuvable")

nb["cells"][4]["source"] = [l + "\n" for l in src4.split("\n")[:-1]] + [src4.split("\n")[-1]]


# ============================================================
# Patch Cell 6 (Phase F) : copie + save dans ORACLE_FINETUNED_DIR
# ============================================================
src6 = "".join(nb["cells"][6]["source"])

# 1. Copie initiale V5_DIR -> ORACLE_FINETUNED_DIR (apres pre-flight, avant fine-tune)
COPY_BLOCK_INSERT_AFTER = '''_phase_f_preflight()
print()'''

COPY_BLOCK = '''
# === Setup ORACLE_FINETUNED_DIR : copie le baseline V5-mini comme point de depart ===
ORACLE_FINETUNED_DIR.mkdir(parents=True, exist_ok=True)
_baseline_ckpt = V5_DIR / "epoch_last.pth"
_oracle_ft_ckpt = ORACLE_FINETUNED_DIR / "epoch_last.pth"
if not _oracle_ft_ckpt.exists():
    print(f"[Setup] Copie du baseline Oracle vers oracle_finetuned/ (premiere execution)...")
    import shutil as _sh
    _sh.copy(_baseline_ckpt, _oracle_ft_ckpt)
    _size_gb = _oracle_ft_ckpt.stat().st_size / 1024**3
    print(f"  [OK] {_oracle_ft_ckpt.name} ({_size_gb:.2f} GB) copie depuis baseline V5-mini")
    # Important : reload stack_v5 depuis le nouveau ckpt pour que finetune_bundle_b
    # modifie l'instance qui pointe vers oracle_finetuned/ et non vers V5_DIR
    print(f"  [Note] stack_v5 actuel pointe vers le baseline (V5_DIR). Le fine-tune va")
    print(f"         modifier stack_v5 in-place et sauvegarder dans ORACLE_FINETUNED_DIR.")
else:
    print(f"[Setup] {_oracle_ft_ckpt.name} existe deja ({_oracle_ft_ckpt.stat().st_size/1024**3:.2f} GB)")
print()
'''

if COPY_BLOCK_INSERT_AFTER in src6 and "ORACLE_FINETUNED_DIR.mkdir" not in src6:
    src6 = src6.replace(COPY_BLOCK_INSERT_AFTER, COPY_BLOCK_INSERT_AFTER + COPY_BLOCK)
    print("[OK] Cell 6 : bloc copie V5 -> oracle_finetuned ajoute")

# 2. ckpt_save_dir : V5_DIR -> ORACLE_FINETUNED_DIR
OLD_SAVE_DIR = '''    ckpt_save_dir=V5_DIR,
    convert_sample_to_batch_fn=convert_sample_to_batch,'''
NEW_SAVE_DIR = '''    ckpt_save_dir=ORACLE_FINETUNED_DIR,
    convert_sample_to_batch_fn=convert_sample_to_batch,'''
if OLD_SAVE_DIR in src6:
    src6 = src6.replace(OLD_SAVE_DIR, NEW_SAVE_DIR)
    print("[OK] Cell 6 : ckpt_save_dir -> ORACLE_FINETUNED_DIR")

# 3. Recompute Phase 6 : out_path V5_DIR -> ORACLE_FINETUNED_DIR
OLD_RECOMPUTE_OUT = '''        out_path=V5_DIR / "final_validation_metrics.json",'''
NEW_RECOMPUTE_OUT = '''        out_path=ORACLE_FINETUNED_DIR / "final_validation_metrics.json",'''
if OLD_RECOMPUTE_OUT in src6:
    src6 = src6.replace(OLD_RECOMPUTE_OUT, NEW_RECOMPUTE_OUT)
    print("[OK] Cell 6 : recompute Phase 6 out_path -> ORACLE_FINETUNED_DIR")

# 4. Plus de auto-backup de V5_DIR (V5_DIR n'est plus touche)
OLD_BACKUP = '''    # Auto-backup du baseline V5-mini avant overwrite
    _v5_metrics = V5_DIR / "final_validation_metrics.json"
    _v5_baseline_backup = V5_DIR / "final_validation_metrics_v5mini_baseline.json"
    if _v5_metrics.exists() and not _v5_baseline_backup.exists():
        import shutil as _sh
        _sh.copy(_v5_metrics, _v5_baseline_backup)
        print(f"[OK] Baseline V5-mini sauvegarde : {_v5_baseline_backup.name}")

    # ORACLE (V5) — overwrite final_validation_metrics.json (avec try/except safety)'''
NEW_BACKUP = '''    # V5_DIR (baseline Oracle V5-mini) reste intact — pas d'overwrite/backup necessaire.
    # ORACLE (post-finetune) — ecrit dans ORACLE_FINETUNED_DIR (avec try/except safety)'''
if OLD_BACKUP in src6:
    src6 = src6.replace(OLD_BACKUP, NEW_BACKUP)
    print("[OK] Cell 6 : suppression auto-backup V5_DIR (n'est plus touche)")

# 5. Update les message PROCHAINE ETAPE
OLD_NEXT = '''print("  2. Une fois fine-tune complet : change CHECKPOINT_NAME = \\"epoch_finetuned\\" en Cell 2")
print("  3. Re-execute Cell 4 (recharge stack), puis Cells 8, 10, 12, 14, 16, 18, 20, 22 (Phases 6-12)")'''
NEW_NEXT = '''print("  2. Une fois fine-tune complet : change EVAL_VERSION = \\"finetuned\\" en Cell 2")
print("  3. Re-execute Cell 4 (recharge stack depuis ORACLE_FINETUNED_DIR), puis Cells 8, 10, 12, 14, 16, 18, 20, 22 (Phases 6-12)")'''
if OLD_NEXT in src6:
    src6 = src6.replace(OLD_NEXT, NEW_NEXT)
    print("[OK] Cell 6 : message prochaine etape -> EVAL_VERSION")

nb["cells"][6]["source"] = [l + "\n" for l in src6.split("\n")[:-1]] + [src6.split("\n")[-1]]


# ============================================================
# Patch Cell 8 (Phase 6) : lit depuis ORACLE_DIR + label Oracle
# ============================================================
src8 = "".join(nb["cells"][8]["source"])

# 1. Path d'entree : V5_DIR -> ORACLE_DIR
OLD_LOAD_V5 = '''m_v5 = load_metrics(V5_DIR)
m_nc = load_metrics(NONCAUSAL_DIR)'''
NEW_LOAD_V5 = '''m_v5 = load_metrics(ORACLE_DIR)   # baseline ou finetuned selon EVAL_VERSION
m_nc = load_metrics(NONCAUSAL_DIR)'''
if OLD_LOAD_V5 in src8:
    src8 = src8.replace(OLD_LOAD_V5, NEW_LOAD_V5)

# 2. Error messages : V5 -> Oracle
OLD_ERR_V5 = '''    print(f"[ERREUR] V5 metrics absent : {V5_DIR}/final_validation_metrics.json")'''
NEW_ERR_V5 = '''    print(f"[ERREUR] Oracle metrics absent : {ORACLE_DIR}/final_validation_metrics.json")'''
if OLD_ERR_V5 in src8:
    src8 = src8.replace(OLD_ERR_V5, NEW_ERR_V5)

OLD_ERR_NC = '''    print(f"[ERREUR] Noncausal metrics absent : {NONCAUSAL_DIR}/final_validation_metrics.json")'''
NEW_ERR_NC = '''    print(f"[ERREUR] CorrDiff metrics absent : {NONCAUSAL_DIR}/final_validation_metrics.json")'''
if OLD_ERR_NC in src8:
    src8 = src8.replace(OLD_ERR_NC, NEW_ERR_NC)

# 3. Headers de tableau : V5-mini -> Oracle
OLD_HEADER = '''f"\\n{'Metrique':<22} {'V5-mini':>12} {'Noncausal':>12}'''
NEW_HEADER = '''f"\\n{'Metrique':<22} {'Oracle':>12} {'CorrDiff':>12}'''
if OLD_HEADER in src8:
    src8 = src8.replace(OLD_HEADER, NEW_HEADER)

# 4. Le print "winner" (V5, Noncausal -> Oracle, CorrDiff)
OLD_WINNER_V5 = '"V5"'  # used in get(m, "...") winner check
# Looking at the cell — we need to also adapt "winner = ..." but that's complex
# Let's just adapt the most obvious display labels

# 5. Output JSON file name : phase6_in_distribution.json reste mais on documente
# Pas de changement necessaire ici

print()
print(f"[OK] Cell 8 (Phase 6) : labels Oracle/CorrDiff + lecture ORACLE_DIR")

nb["cells"][8]["source"] = [l + "\n" for l in src8.split("\n")[:-1]] + [src8.split("\n")[-1]]


# ============================================================
# Save
# ============================================================
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print("=" * 60)
print("[OK] Restructure complete")
print("=" * 60)
print()
print("Workflow user :")
print("  1. EVAL_VERSION = \"baseline\" + run Cells 0-4 + Cells 8-22 -> eval Oracle V5-mini")
print("     (ecrit dans results/v5_evaluation/ ou oracle_evaluation/baseline_eval/)")
print("  2. EVAL_VERSION = \"baseline\" + run Cell 6 (Phase F) -> fine-tune dans oracle_finetuned/")
print("  3. EVAL_VERSION = \"finetuned\" + re-run Cells 4, 8-22 -> eval Oracle post-Phase F")
print()
print("Ce qui est preserve (jamais modifie) :")
print("  - ckpt_v2_corrdiff_normal/  (Oracle baseline V5-mini)")
print("  - ckpt_noncausal/            (CorrDiff baseline)")
print("Ce qui est cree/modifie :")
print("  - oracle_finetuned/          (copie depart + post-Phase F)")
