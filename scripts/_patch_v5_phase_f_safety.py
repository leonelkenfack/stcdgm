"""Patch consolide de securite pour Cell 6 (Phase F) :

1. Pre-flight check : verifie RAM dispo + disque Drive dispo avant lancer.
2. Auto-backup de final_validation_metrics.json baseline avant recompute.
3. Cleanup auto des ancien epoch_finetuned.pth (smoke test) si demande.
4. Try/except autour du recompute Phase 6 (training pas perdu si crash).
5. Memory cleanup a la fin du cell : del materialized samples + gc.collect().
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak_phaseF_safety")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup cree : {BACKUP}")

# Trouve la cell Phase F
phase_f_idx = None
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] == "code" and "Phase F : fine-tune Bundle B + CASTLE + G_phys" in src:
        phase_f_idx = i
        break
if phase_f_idx is None:
    print("[KO] Cell Phase F introuvable")
    sys.exit(1)
print(f"[OK] Cell Phase F a l'index {phase_f_idx}")

src = "".join(nb["cells"][phase_f_idx]["source"])


# === Patch 1 : Pre-flight check au tout debut ===
PREFLIGHT = '''# === Pre-flight checks ===
import os, shutil, psutil, gc

def _gb(b):
    return b / (1024**3)

def _phase_f_preflight():
    issues = []
    # RAM
    ram = psutil.virtual_memory()
    print(f"  RAM dispo : {_gb(ram.available):.1f} GB / {_gb(ram.total):.1f} GB total")
    if _gb(ram.available) < 12.0:
        issues.append(f"RAM dispo < 12 GB ({_gb(ram.available):.1f} GB) — risque OOM")
    # Disque Drive
    try:
        du = shutil.disk_usage("/content/drive/MyDrive")
        print(f"  Disque Drive : {_gb(du.free):.1f} GB libre / {_gb(du.total):.1f} GB total")
        if _gb(du.free) < 5.0:
            issues.append(f"Drive < 5 GB libre — risque saturation checkpoints")
    except Exception as e:
        print(f"  [WARN] disk check failed: {e}")
    # Detect smoke-test artifact
    smoke_ckpt = V5_DIR / "epoch_finetuned.pth"
    inprogress = V5_DIR / "epoch_finetuned_inprogress.pth"
    if smoke_ckpt.exists() and not inprogress.exists():
        # Detect smoke test (file present but no inprogress = a previous run finished)
        try:
            import torch as _t
            _ck = _t.load(smoke_ckpt, map_location="cpu", weights_only=False)
            saved_ep = _ck.get("epoch", "?")
            if isinstance(saved_ep, int) and saved_ep < EPOCHS:
                print(f"  [INFO] Smoke test ancien detecte : {smoke_ckpt.name} (epoch={saved_ep})")
                print(f"         Sera ecrase au final save (epoch_finetuned.pth)")
        except Exception:
            pass
    if issues:
        print()
        print("  [WARNING] Issues detectees :")
        for it in issues:
            print(f"    - {it}")
        print()
    return issues

print()
print("[Phase F] Pre-flight check...")
_phase_f_preflight()
print()

'''

# Inserer juste apres le bloc EPOCHS / SANITY_EVERY
INSERT_AFTER = "SANITY_EVERY = 1 if SMOKE_TEST else 5"
if INSERT_AFTER in src and "_phase_f_preflight" not in src:
    src = src.replace(INSERT_AFTER, INSERT_AFTER + "\n" + PREFLIGHT)
    print("[OK] Patch 1 : pre-flight check ajoute")
else:
    print("[SKIP] Patch 1 : deja present ou anchor introuvable")


# === Patch 2 : Auto-backup baseline JSON avant le recompute ===
OLD_RECOMPUTE = '''    # ORACLE (V5) — overwrite final_validation_metrics.json
    recompute_phase6_metrics('''

NEW_RECOMPUTE = '''    # Auto-backup du baseline V5-mini avant overwrite
    _v5_metrics = V5_DIR / "final_validation_metrics.json"
    _v5_baseline_backup = V5_DIR / "final_validation_metrics_v5mini_baseline.json"
    if _v5_metrics.exists() and not _v5_baseline_backup.exists():
        import shutil as _sh
        _sh.copy(_v5_metrics, _v5_baseline_backup)
        print(f"[OK] Baseline V5-mini sauvegarde : {_v5_baseline_backup.name}")

    # ORACLE (V5) — overwrite final_validation_metrics.json (avec try/except safety)
    try:
        recompute_phase6_metrics('''

if OLD_RECOMPUTE in src and "_v5_baseline_backup" not in src:
    src = src.replace(OLD_RECOMPUTE, NEW_RECOMPUTE)
    # On doit aussi fermer le try/except : ajouter except clause apres l'appel
    # On cherche la ligne juste apres l'appel recompute pour fermer
    OLD_END = '''    print()
    print("[OK] V5_DIR/final_validation_metrics.json mis a jour")
    print()'''
    NEW_END = '''        print()
        print("[OK] V5_DIR/final_validation_metrics.json mis a jour")
        print()
    except Exception as _e:
        print(f"[WARN] Recompute Phase 6 a echoue : {type(_e).__name__}: {_e}")
        print(f"       Le checkpoint training reste valide (epoch_finetuned.pth)")
        print(f"       Tu peux relancer le recompute manuellement plus tard")
        print()'''
    if OLD_END in src:
        src = src.replace(OLD_END, NEW_END)
        print("[OK] Patch 2 : auto-backup + try/except recompute")
    else:
        print("[WARN] Patch 2 : try/except wrap partiel — verifier manuellement")
else:
    print("[SKIP] Patch 2 : deja present ou anchor introuvable")


# === Patch 3 : Memory cleanup a la fin du cell ===
OLD_END_CELL = '''print("PROCHAINE ETAPE :")'''
NEW_END_CELL = '''# Memory cleanup : free 9-10 GB avant Phases 7-12
print()
print("[Cleanup] Liberation memoire post-training...")
try:
    del _samples
except NameError:
    pass
try:
    del _val_samples
except NameError:
    pass
try:
    del train_iter, train_dataset, val_dataset
except NameError:
    pass
import gc as _gc
_gc.collect()
if 'torch' in dir() and torch.cuda.is_available():
    torch.cuda.empty_cache()
import psutil as _psutil
print(f"  [OK] RAM apres cleanup : {_psutil.virtual_memory().available / 1024**3:.1f} GB dispo")
print()

print("PROCHAINE ETAPE :")'''

# Trouver une instance unique : la 1ere occurrence apres le bloc recompute
idx_end = src.find(NEW_END_CELL[-50:])  # marker unique
if "_psutil.virtual_memory" not in src and OLD_END_CELL in src:
    # Replace the first occurrence (the one outside the recompute block)
    # We need to be precise — the print is in a unique context
    src = src.replace(OLD_END_CELL, NEW_END_CELL, 1)
    print("[OK] Patch 3 : memory cleanup ajoute")
else:
    print("[SKIP] Patch 3 : deja present")


# === Sauvegarde ===
nb["cells"][phase_f_idx]["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print("=" * 60)
print("[OK] Patches safety appliques. Pull dans Colab pour les utiliser.")
print("=" * 60)
