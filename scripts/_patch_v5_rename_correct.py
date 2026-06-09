"""Corrige le rename precedent qui etait inverse :
- V5 (modele causal, le main du memoire) -> ORACLE
- Noncausal (baseline) -> CorrDiff

L'ancien patch avait fait Noncausal -> ORACLE, ce qui etait l'inverse. On revient
en arriere puis on applique le bon mapping. Les cles Python, chemins disque, et
noms de variables (stack_v5, stack_nc) sont preserves.
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

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak15")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #15 cree : {BACKUP}")


def _patch_cell(idx, replacements, label):
    src = "".join(nb["cells"][idx]["source"])
    n_done = 0
    for old, new in replacements:
        if old in src:
            src = src.replace(old, new)
            n_done += 1
    nb["cells"][idx]["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
    print(f"[OK] {label} (cell {idx}) : {n_done}/{len(replacements)} replacements")


# === Cell 8 (Phase 7 recap) =================================================
# Revert: _disp = "ORACLE" if Noncausal -> "CorrDiff" if Noncausal else "ORACLE" if V5
_patch_cell(8, [
    ('_disp = "ORACLE" if variant == "Noncausal" else variant',
     '_disp = "CorrDiff" if variant == "Noncausal" else "ORACLE"'),
], "Phase 7 recap")


# === Cell 10 (Phase 8) ======================================================
# Revert previous changes (Noncausal->ORACLE) AND add V5->ORACLE
_patch_cell(10, [
    # === REVERT : "ORACLE" qui etait Noncausal -> "CorrDiff" ===
    ('label="ORACLE"', 'label="CorrDiff"'),
    ('axes[1,0].set_title("ORACLE prediction")', 'axes[1,0].set_title("CorrDiff prediction")'),
    ('axes[0].set_title("Sensibilite de mu_HR par variable LR (V5 vs ORACLE)")',
     'axes[0].set_title("Sensibilite de mu_HR par variable LR (ORACLE vs CorrDiff)")'),
    ('axes[row_idx, 3].set_title(f"Delta ORACLE (sample 0)", fontsize=10)',
     'axes[row_idx, 3].set_title(f"Delta CorrDiff (sample 0)", fontsize=10)'),
    ('axes[1].set_title("Ratio des sensibilites V5 / ORACLE — > 1 = V5 utilise plus cette variable")',
     'axes[1].set_title("Ratio des sensibilites ORACLE / CorrDiff — > 1 = ORACLE utilise plus cette variable")'),
    ('plt.suptitle("Comparaison spatiale V5 vs ORACLE (meme input)", fontsize=12)',
     'plt.suptitle("Comparaison spatiale ORACLE vs CorrDiff (meme input)", fontsize=12)'),
    ('axes[1,1].set_title(f"V5 - ORACLE\\n(mean abs = {np.abs(diff).mean():.4f})")',
     'axes[1,1].set_title(f"ORACLE - CorrDiff\\n(mean abs = {np.abs(diff).mean():.4f})")'),
    ('print(f"Q_int ORACLE     : {q_nc:.3f}', 'print(f"Q_int CorrDiff   : {q_nc:.3f}'),
    ('print(f"Q_int ORACLE     : {results_full[\'Q_int\'][\'Noncausal\']}")',
     'print(f"Q_int CorrDiff   : {results_full[\'Q_int\'][\'Noncausal\']}")'),
    # === REVERT _disp mapping ===
    ('_disp = "ORACLE" if variant == "Noncausal" else variant',
     '_disp = "CorrDiff" if variant == "Noncausal" else "ORACLE"'),
    # === REVERT histogramme + RAPSD ===
    ('label="ORACLE", color="orange", density=True', 'label="CorrDiff", color="orange", density=True'),
    ('label="ORACLE", color="orange", linewidth=1.5', 'label="CorrDiff", color="orange", linewidth=1.5'),

    # === NEW : V5 (display) -> ORACLE ===
    # Plots V5 normal / + intervention / Delta V5
    ('axes[row_idx, 0].set_title(f"V5 normal", fontsize=10)',
     'axes[row_idx, 0].set_title(f"ORACLE normal", fontsize=10)'),
    ('axes[row_idx, 1].set_title(f"V5 + {spec[\'name\']}", fontsize=10)',
     'axes[row_idx, 1].set_title(f"ORACLE + {spec[\'name\']}", fontsize=10)'),
    ('axes[row_idx, 2].set_title(f"Delta V5 (sample 0)", fontsize=10)',
     'axes[row_idx, 2].set_title(f"Delta ORACLE (sample 0)", fontsize=10)'),
    # Figure 6 V5 vs noncausal
    ('axes[0,1].set_title("V5 prediction")', 'axes[0,1].set_title("ORACLE prediction")'),
    # Bar labels "V5" -> "ORACLE"
    ('axes[0].bar(x - w/2, v5_vals, w, label="V5", color="steelblue")',
     'axes[0].bar(x - w/2, v5_vals, w, label="ORACLE", color="steelblue")'),
    # Histogramme + RAPSD V5
    ('ax.hist(pred_v5.flatten(), bins=bins, alpha=0.5, label="V5", color="steelblue", density=True)',
     'ax.hist(pred_v5.flatten(), bins=bins, alpha=0.5, label="ORACLE", color="steelblue", density=True)'),
    ('ax.loglog(k, psd_v5[:len(k)], label="V5", color="steelblue", linewidth=1.5)',
     'ax.loglog(k, psd_v5[:len(k)], label="ORACLE", color="steelblue", linewidth=1.5)'),
    # Print Q_int V5
    ('print(f"Q_int V5         : {results_full[\'Q_int\'][\'V5\']}")',
     'print(f"Q_int ORACLE     : {results_full[\'Q_int\'][\'V5\']}")'),
    # Title DAG V5
    ('ax.set_title("DAG appris (V5) — matrice d\'adjacence A_dag\\n(diagonale masquee)", fontsize=11)',
     'ax.set_title("DAG appris (ORACLE) — matrice d\'adjacence A_dag\\n(diagonale masquee)", fontsize=11)'),
    # Plot title alpha
    ('print(f"[OK] Figure 2 : alpha mean={alphas.mean():.3f}, prop>0.6={((alphas>0.6).mean()*100):.1f}%")',
     'print(f"[OK] Figure 2 : alpha mean={alphas.mean():.3f}, prop>0.6={((alphas>0.6).mean()*100):.1f}%")'),  # no change just keep
], "Phase 8")


# === Cell 12 (Phase 9) ======================================================
# Update DISPLAY dict
_patch_cell(12, [
    ('DISPLAY = {"V5": "V5", "Noncausal": "ORACLE"}',
     'DISPLAY = {"V5": "ORACLE", "Noncausal": "CorrDiff"}'),
], "Phase 9")


# === Cell 14 (Phase 10) =====================================================
_patch_cell(14, [
    ('DISPLAY = {"V5": "V5", "Noncausal": "ORACLE"}',
     'DISPLAY = {"V5": "ORACLE", "Noncausal": "CorrDiff"}'),
], "Phase 10")


# === Cell 16 (Phase 11) =====================================================
# DAG specific to V5 only
_patch_cell(16, [
    ('axes[0].set_title("DAG appris (V5)")', 'axes[0].set_title("DAG appris (ORACLE)")'),
], "Phase 11")


# === Cell 18 (Phase 12) =====================================================
_patch_cell(18, [
    # REVERT previous "ORACLE" labels (which were Noncausal) -> CorrDiff
    ('axes[0].bar(x + w/2, nc_vals, w, label="ORACLE", color="orange")',
     'axes[0].bar(x + w/2, nc_vals, w, label="CorrDiff", color="orange")'),
    ('axes[0].set_title("Integrated Gradients par variable LR (V5 vs ORACLE)\\n(baseline = climato moyenne)")',
     'axes[0].set_title("Integrated Gradients par variable LR (ORACLE vs CorrDiff)\\n(baseline = climato moyenne)")'),
    ('axes[1].set_title("Ratio IG V5 / ORACLE — > 1 = V5 exploite davantage cette variable")',
     'axes[1].set_title("Ratio IG ORACLE / CorrDiff — > 1 = ORACLE exploite davantage cette variable")'),
    ('axes[1].set_ylabel("Ratio IG V5 / ORACLE")', 'axes[1].set_ylabel("Ratio IG ORACLE / CorrDiff")'),
    ('# === IG pour V5 vs ORACLE sur le meme sample',
     '# === IG pour ORACLE vs CorrDiff sur le meme sample'),

    # NEW : V5 bar label -> ORACLE
    ('axes[0].bar(x - w/2, v5_vals, w, label="V5", color="steelblue")',
     'axes[0].bar(x - w/2, v5_vals, w, label="ORACLE", color="steelblue")'),
], "Phase 12")


# --- Save ---
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print("[OK] Rename corrige :")
print("     V5 (display)         -> ORACLE")
print("     Noncausal (display)  -> CorrDiff")
print("     Cles Python, chemins, et stacks (stack_v5/stack_nc) inchanges.")
