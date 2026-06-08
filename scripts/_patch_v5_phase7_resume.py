"""Patch Phase 7 (Cell 8) : persistance par run sur Drive + resume au demarrage.

Strategie :
- Apres chaque run : sauvegarde npz (pred_mean float32, truth float32, ens float16, times)
  dans RESULTS_DIR / "phase7_runs" / "<variant>_<gcm>.npz" (sur Drive)
- Au demarrage : scanne le dossier, restaure phase7_arrays/all_results/all_prob
  depuis npz + JSON sidecars. Skip recompute pour les runs deja persistes.
- Robustesse : un run partiellement persiste (ex: npz OK mais JSON manquant) est recompute.

float16 pour ens : suffisant pour CRPS/spread (precision ~10e-4 dans range log1p [0,5]),
divise la taille de l ensemble par 2 (~270 MB / run au lieu de 540 MB).
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak12")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #12 cree : {BACKUP}")

cell8_src = "".join(nb["cells"][8]["source"])

# --- 1. Ajoute ARRAYS_DIR + helper _resume_from_disk apres `phase7_arrays = {}` ---

OLD_INIT = """phase7_arrays = {}    # arrays in memory pour Phases 9-12
t_global = time.time()"""

NEW_INIT = """phase7_arrays = {}    # arrays in memory pour Phases 9-12

# Persistance : un dossier dedie sur Drive pour les arrays
ARRAYS_DIR = RESULTS_DIR / "phase7_runs"
ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
print(f"[INFO] Resume directory : {ARRAYS_DIR}")


def _resume_from_disk(stack_name, gcm_tag, ckpt_dir):
    \"\"\"Restaure un run depuis disque si npz + 2 JSONs existent et sont valides.

    Returns
    -------
    tuple (preds, truths, times, ens_full, aligned_data, prob_data) ou None.
    \"\"\"
    arrays_path = ARRAYS_DIR / f"{stack_name}_{gcm_tag}.npz"
    aligned_path = ckpt_dir / f"aligned_metrics_{gcm_tag}_{stack_name.lower()}.json"
    prob_path = ckpt_dir / f"probabilistic_metrics_{gcm_tag}_{stack_name.lower()}.json"
    if not (arrays_path.exists() and aligned_path.exists() and prob_path.exists()):
        return None
    try:
        with np.load(arrays_path, allow_pickle=True) as z:
            preds = z["pred_mean"].astype(np.float32)
            truths = z["truth"].astype(np.float32)
            ens_full = z["ens"].astype(np.float32)      # cast back depuis float16
            times = z["times"]
        aligned_data = json.loads(aligned_path.read_text(encoding="utf-8"))
        prob_data = json.loads(prob_path.read_text(encoding="utf-8"))
        return preds, truths, times, ens_full, aligned_data, prob_data
    except Exception as e:
        print(f"  [WARN] {stack_name}_{gcm_tag} restore failed ({type(e).__name__}: {e}) - recompute")
        return None


def _persist_to_disk(stack_name, gcm_tag, preds, truths, ens_full, times):
    \"\"\"Sauvegarde arrays npz sur Drive. ens en float16 pour economiser l espace.\"\"\"
    arrays_path = ARRAYS_DIR / f"{stack_name}_{gcm_tag}.npz"
    try:
        np.savez_compressed(
            str(arrays_path),
            pred_mean=preds.astype(np.float32),
            truth=truths.astype(np.float32),
            ens=ens_full.astype(np.float16),    # half precision pour ens (gain ~50%)
            times=np.array(times, dtype="datetime64[D]"),
        )
        size_mb = arrays_path.stat().st_size / 1e6
        return True, size_mb
    except Exception as e:
        print(f"       [WARN] persist failed : {type(e).__name__}: {e}")
        return False, 0.0


t_global = time.time()"""

if OLD_INIT not in cell8_src:
    print("[ERREUR] init block introuvable")
    sys.exit(1)
cell8_src = cell8_src.replace(OLD_INIT, NEW_INIT)

# --- 2. Ajoute le check resume + skip dans la boucle ---

OLD_PRE_COMPUTE = """        if not GCM_REGISTRY[gcm_tag][0].exists():
            print(f"  [SKIP] {GCM_REGISTRY[gcm_tag][0]} absent")
            continue

        try:
            preds, truths, times, ens_full = collect_predictions_for_gcm(stack, gcm_tag)"""

NEW_PRE_COMPUTE = """        if not GCM_REGISTRY[gcm_tag][0].exists():
            print(f"  [SKIP] {GCM_REGISTRY[gcm_tag][0]} absent")
            continue

        # === RESUME : check si run deja persiste ===
        resumed = _resume_from_disk(stack_name, gcm_tag, ckpt_dir)
        if resumed is not None:
            preds, truths, times, ens_full, aligned_data, prob_data = resumed
            all_results[run_label] = aligned_data
            all_prob[run_label] = prob_data
            phase7_arrays[run_label] = {
                "pred_mean": preds, "truth": truths,
                "ens": ens_full, "times": times,
            }
            crps = prob_data.get('crps_model_global_mm', 0.0)
            crps_ss = prob_data.get('crps_skill_score', 0.0)
            rmse = prob_data.get('rmse_global_mm', 0.0)
            spread = prob_data.get('spread_skill_ratio', 0.0)
            print(f"  [RESUME] charge depuis disque (skip compute, gain ~90 min)")
            print(f"           CRPS={crps:.3f}mm  CRPS-SS={crps_ss:+.3f}  "
                  f"RMSE={rmse:.3f}mm  spread/skill={spread:.3f}")
            continue

        try:
            preds, truths, times, ens_full = collect_predictions_for_gcm(stack, gcm_tag)"""

if OLD_PRE_COMPUTE not in cell8_src:
    print("[ERREUR] pre-compute block introuvable")
    sys.exit(1)
cell8_src = cell8_src.replace(OLD_PRE_COMPUTE, NEW_PRE_COMPUTE)

# --- 3. Ajoute la persistance npz juste apres population de phase7_arrays ---

OLD_PERSIST_POINT = """            phase7_arrays[run_label] = {
                "pred_mean": preds.astype("float32"),
                "truth": truths.astype("float32"),
                "ens": ens_full.astype("float32"),
                "times": times,
            }
            print(f"  [OK] aligned + probabilistic ({(time.time()-t0):.1f}s)")"""

NEW_PERSIST_POINT = """            phase7_arrays[run_label] = {
                "pred_mean": preds.astype("float32"),
                "truth": truths.astype("float32"),
                "ens": ens_full.astype("float32"),
                "times": times,
            }
            # Persiste sur Drive pour resume futur
            ok, size_mb = _persist_to_disk(stack_name, gcm_tag, preds, truths, ens_full, times)
            if ok:
                print(f"       [PERSIST] {stack_name}_{gcm_tag}.npz sauve ({size_mb:.0f} MB sur Drive)")
            print(f"  [OK] aligned + probabilistic ({(time.time()-t0):.1f}s)")"""

if OLD_PERSIST_POINT not in cell8_src:
    print("[ERREUR] persist point introuvable")
    sys.exit(1)
cell8_src = cell8_src.replace(OLD_PERSIST_POINT, NEW_PERSIST_POINT)

# --- Save back ---
nb["cells"][8]["source"] = [l + "\n" for l in cell8_src.split("\n")[:-1]] + [cell8_src.split("\n")[-1]]

NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("[OK] Cell 8 patche avec resume + persistance par run")
print("     Strategie : npz sur Drive (ens en float16, ~270 MB/run)")
print("     Au prochain lancement : skip auto des runs deja calcules")
