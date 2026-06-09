"""Patch : renomme "Noncausal" -> "ORACLE" dans les LABELS et NOMS DE FICHIERS
de toutes les figures (Phases 8 a 12).

Strategie :
- Introduit un dict DISPLAY = {"V5": "V5", "Noncausal": "ORACLE"} en tete des cellules code
- Dans les figures (titles, legend labels, file paths) : remplace toutes les
  occurrences exactes de "Noncausal" et "NC" par "ORACLE" ou utilise DISPLAY[variant]
- Garde les cles dict, paths checkpoints, et nom de stack_name "Noncausal" pour la
  compatibilite avec le resume Phase 7 (npz et JSON sur Drive nommes ..._noncausal).
"""
import json
import sys
import re
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

NB = Path("st_cdgm_v5_evaluation.ipynb")
with NB.open(encoding="utf-8") as f:
    nb = json.load(f)

BACKUP = Path("st_cdgm_v5_evaluation.ipynb.bak14")
if not BACKUP.exists():
    BACKUP.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[OK] Backup #14 cree : {BACKUP}")


# --- Cell 10 (Phase 8) ---
src = "".join(nb["cells"][10]["source"])
# Replacements dans des strings d'affichage matplotlib + prints utilisateur
replacements_p8 = [
    # Bar labels et legend
    ('label="Noncausal"', 'label="ORACLE"'),
    # Set titles dans Figure 6 (comparaison spatiale)
    ('axes[1,0].set_title("Noncausal prediction")', 'axes[1,0].set_title("ORACLE prediction")'),
    # Title bar chart sensibilite
    ('axes[0].set_title("Sensibilite de mu_HR par variable LR")',
     'axes[0].set_title("Sensibilite de mu_HR par variable LR (V5 vs ORACLE)")'),
    # "Delta NC" dans Figure 3 (interventions)
    ('axes[row_idx, 3].set_title(f"Delta NC (sample 0)", fontsize=10)',
     'axes[row_idx, 3].set_title(f"Delta ORACLE (sample 0)", fontsize=10)'),
    # Ratio title
    ('axes[1].set_title("Ratio des sensibilites — > 1 = V5 utilise plus cette variable")',
     'axes[1].set_title("Ratio des sensibilites V5 / ORACLE — > 1 = V5 utilise plus cette variable")'),
    # Title comparison spatiale
    ('plt.suptitle("Comparaison spatiale V5 vs Noncausal (meme input)", fontsize=12)',
     'plt.suptitle("Comparaison spatiale V5 vs ORACLE (meme input)", fontsize=12)'),
    # Title diff in Figure 6
    ('axes[1,1].set_title(f"V5 - Noncausal\\n(mean abs = {np.abs(diff).mean():.4f})")',
     'axes[1,1].set_title(f"V5 - ORACLE\\n(mean abs = {np.abs(diff).mean():.4f})")'),
    # Histogramme Noncausal label
    ('ax.hist(pred_nc.flatten(), bins=bins, alpha=0.5, label="Noncausal", color="orange", density=True)',
     'ax.hist(pred_nc.flatten(), bins=bins, alpha=0.5, label="ORACLE", color="orange", density=True)'),
    # RAPSD label
    ('ax.loglog(k, psd_nc[:len(k)], label="Noncausal", color="orange", linewidth=1.5)',
     'ax.loglog(k, psd_nc[:len(k)], label="ORACLE", color="orange", linewidth=1.5)'),
    # Prints Q_int
    ('print(f"Q_int Noncausal  : {q_nc:.3f}',
     'print(f"Q_int ORACLE     : {q_nc:.3f}'),
]
n_p8 = 0
for old, new in replacements_p8:
    if old in src:
        src = src.replace(old, new)
        n_p8 += 1
nb["cells"][10]["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
print(f"[OK] Cell 10 Phase 8 : {n_p8}/{len(replacements_p8)} remplacements")


# --- Cell 12 (Phase 9) ---
src = "".join(nb["cells"][12]["source"])
# Variant dans Q-Q : `f"{variant} / {gcm}"` -> utiliser DISPLAY
# Legend `label=variant` aussi
# Inject DISPLAY dict at top after imports
INJECT_AFTER = 'PHASE9_DIR.mkdir(parents=True, exist_ok=True)'
INJECT_DISPLAY = '\n\n# Display names pour figures (data structures gardent "Noncausal")\nDISPLAY = {"V5": "V5", "Noncausal": "ORACLE"}'

if INJECT_AFTER in src and 'DISPLAY = {"V5"' not in src:
    src = src.replace(INJECT_AFTER, INJECT_AFTER + INJECT_DISPLAY)

# Replacements
replacements_p9 = [
    # Q-Q title `f"{variant} / {gcm}"` -> `f"{DISPLAY[variant]} / {gcm}"`
    ('ax.set_title(f"{variant} / {gcm}"); ax.axis("off"); continue',
     'ax.set_title(f"{DISPLAY[variant]} / {gcm}"); ax.axis("off"); continue'),
    ('ax.set_title(f"{variant} / {gcm} ({id_tag})", fontsize=10)',
     'ax.set_title(f"{DISPLAY[variant]} / {gcm} ({id_tag})", fontsize=10)'),
    # Return-period legend
    ('label=f"{variant} pred", alpha=0.8, markersize=3',
     'label=f"{DISPLAY[variant]} pred", alpha=0.8, markersize=3'),
    # Reliability legend
    ('ax.plot(mp_, fp_, marker="o", color=color, label=variant, linewidth=2)',
     'ax.plot(mp_, fp_, marker="o", color=color, label=DISPLAY[variant], linewidth=2)'),
    # FSS legend
    ('ax.plot(scales, fss_by_scale, marker="o", color=color, label=variant)',
     'ax.plot(scales, fss_by_scale, marker="o", color=color, label=DISPLAY[variant])'),
]
n_p9 = 0
for old, new in replacements_p9:
    if old in src:
        src = src.replace(old, new)
        n_p9 += 1
nb["cells"][12]["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
print(f"[OK] Cell 12 Phase 9 : {n_p9}/{len(replacements_p9)} remplacements + DISPLAY inject")


# --- Cell 14 (Phase 10) ---
src = "".join(nb["cells"][14]["source"])
INJECT_AFTER_P10 = 'PHASE10_DIR.mkdir(parents=True, exist_ok=True)'
if INJECT_AFTER_P10 in src and 'DISPLAY = {"V5"' not in src:
    src = src.replace(INJECT_AFTER_P10, INJECT_AFTER_P10 + INJECT_DISPLAY)

replacements_p10 = [
    # Suptitle
    ('plt.suptitle(f"Cartes de biais des indices d\'extremes — {variant}",',
     'plt.suptitle(f"Cartes de biais des indices d\'extremes — {DISPLAY[variant]}",'),
    # File path
    ('plt.savefig(PHASE10_DIR / f"extreme_bias_{variant}.png", dpi=120,',
     'plt.savefig(PHASE10_DIR / f"extreme_bias_{DISPLAY[variant]}.png", dpi=120,'),
    # Print
    ('print(f"[OK] Figure {variant} : 4 indices x 3 GCMs")',
     'print(f"[OK] Figure {DISPLAY[variant]} : 4 indices x 3 GCMs")'),
]
n_p10 = 0
for old, new in replacements_p10:
    if old in src:
        src = src.replace(old, new)
        n_p10 += 1
nb["cells"][14]["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
print(f"[OK] Cell 14 Phase 10 : {n_p10}/{len(replacements_p10)} remplacements + DISPLAY inject")


# --- Cell 18 (Phase 12) ---
src = "".join(nb["cells"][18]["source"])
replacements_p12 = [
    # Legend bars
    ('axes[0].bar(x + w/2, nc_vals, w, label="Noncausal", color="orange")',
     'axes[0].bar(x + w/2, nc_vals, w, label="ORACLE", color="orange")'),
    # Title bar chart
    ('axes[0].set_title("Integrated Gradients par variable LR\\n(baseline = climato moyenne)")',
     'axes[0].set_title("Integrated Gradients par variable LR (V5 vs ORACLE)\\n(baseline = climato moyenne)")'),
    # Ratio title
    ('axes[1].set_title("Ratio IG — > 1 = V5 exploite davantage cette variable")',
     'axes[1].set_title("Ratio IG V5 / ORACLE — > 1 = V5 exploite davantage cette variable")'),
]
n_p12 = 0
for old, new in replacements_p12:
    if old in src:
        src = src.replace(old, new)
        n_p12 += 1
nb["cells"][18]["source"] = [l + "\n" for l in src.split("\n")[:-1]] + [src.split("\n")[-1]]
print(f"[OK] Cell 18 Phase 12 : {n_p12}/{len(replacements_p12)} remplacements")


# Save
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"\n[OK] Notebook sauvegarde. Tout 'Noncausal' visible dans figures/prints -> 'ORACLE'.")
print(f"     Cles Python, paths checkpoints, et noms de fichiers Phase 7 (npz/json) preserves.")
