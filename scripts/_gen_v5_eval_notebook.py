"""Génère st_cdgm_v5_evaluation.ipynb — Phase 6 + 7 + 8."""
import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in text.split("\n")[:-1]] + [text.split("\n")[-1]],
    }


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [ln + "\n" for ln in text.split("\n")[:-1]] + [text.split("\n")[-1]],
    }


cells = []

# Cell 0
cells.append(md_cell("""# Évaluation V5-mini — Phases 6 / 7 / 8

**Notebook unique** pour évaluer la version Oracle V5-mini contre le baseline CorrDiff noncausal.

**Trois phases successives** :

1. **Phase 6 — Comparaison in-distribution standardisée**
   Tableau métrique × variant sur le jeu de test historique. Charge les JSON `final_validation_metrics.json` produits par le notebook de training.

2. **Phase 7 — Évaluation hors-distribution (OOD)**
   Si un dataset SSP5-8.5 ou EC-Earth3 est dispo : compare Δ_OOD. Sinon : diagnostics de cohérence physique (RAPSD, distribution centiles).

3. **Phase 8 — Protocole d'intervention `do(·)`**
   Charge les modèles et applique 3 interventions standardisées sur l'humidité, la température, le vent. Mesure Q_int.

**Pré-requis** :
- Checkpoints sur Drive :
  - V5-mini : `/content/drive/MyDrive/climate_data/ckpt_v2_corrdiff_normal/`
  - Noncausal : `/content/drive/MyDrive/climate_data/ckpt_noncausal/`
- Phase 6 ne nécessite pas le rechargement des modèles (post-hoc).
- Phase 8 nécessite le rechargement complet.

**Sortie** : `/content/drive/MyDrive/climate_data/results/v5_evaluation/`"""))

# Cell 1 — Setup
cells.append(code_cell("""# Setup Colab : mount Drive + clone repo si nécessaire
import os
import sys
from pathlib import Path

try:
    import google.colab  # noqa
    ON_COLAB = True
except ImportError:
    ON_COLAB = False
print(f"On Colab : {ON_COLAB}")

if ON_COLAB:
    from google.colab import drive
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")

    REPO_DIR = Path("/content/climate_data")
    if not REPO_DIR.exists():
        os.system("git clone https://github.com/leonelkenfack/climate_data.git /content/climate_data")
    os.chdir(REPO_DIR)

REPO_ROOT = Path(os.getcwd())
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

print(f"REPO_ROOT : {REPO_ROOT}")"""))

# Cell 2 — Chemins
cells.append(code_cell("""# === CONFIGURATION DES CHEMINS ===
from pathlib import Path

# V5-mini (directory historiquement nommé ckpt_v2_corrdiff_normal)
V5_DIR = Path("/content/drive/MyDrive/climate_data/ckpt_v2_corrdiff_normal")

# Noncausal = baseline CorrDiff générique
NONCAUSAL_DIR = Path("/content/drive/MyDrive/climate_data/ckpt_noncausal")

# Sorties
RESULTS_DIR = Path("/content/drive/MyDrive/climate_data/results/v5_evaluation")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Vérification existence
for label, p in [("V5", V5_DIR), ("Noncausal", NONCAUSAL_DIR)]:
    ckpt = p / "epoch_last.pth"
    fv = p / "final_validation_metrics.json"
    status_ckpt = "OK" if ckpt.exists() else "ABSENT"
    status_fv = "OK" if fv.exists() else "ABSENT"
    size_gb = ckpt.stat().st_size / 1e9 if ckpt.exists() else 0
    print(f"  {label:10s} : ckpt {status_ckpt} ({size_gb:.2f} GB)  metrics.json {status_fv}")
    print(f"             {p}")

print(f"\\nRésultats -> {RESULTS_DIR}")"""))

# Cell 3 — Phase 6 intro
cells.append(md_cell("""---

## Phase 6 — Comparaison in-distribution standardisée

Lit `final_validation_metrics.json` de chaque variante et produit un tableau métrique × variant. **Pas de rechargement des modèles**, post-hoc.

Métriques comparées :
- Pearson global + per-sample
- RMSE / MAE / Spread / Spread-RMSE ratio
- F1-p95 / F1-p99 (Pearson restreint aux centiles)
- RAPSD distance
- μ_HR ablation ratio (Δ_O3 architectural)"""))

# Cell 4 — Phase 6 code
cells.append(code_cell("""# Phase 6 : comparaison in-distribution post-hoc
import json
import numpy as np

def load_metrics(d):
    p = Path(d) / "final_validation_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))

m_v5 = load_metrics(V5_DIR)
m_nc = load_metrics(NONCAUSAL_DIR)

if m_v5 is None:
    print(f"[ERREUR] V5 metrics absent : {V5_DIR}/final_validation_metrics.json")
if m_nc is None:
    print(f"[ERREUR] Noncausal metrics absent : {NONCAUSAL_DIR}/final_validation_metrics.json")

if m_v5 and m_nc:
    def get(m, *path, default=None):
        v = m
        for k in path:
            if not isinstance(v, dict) or k not in v:
                return default
            v = v[k]
        return v

    rows = [
        ("Pearson global",     get(m_v5, "pearson_corr", "global"),         get(m_nc, "pearson_corr", "global"),         "haut"),
        ("Pearson per-sample", get(m_v5, "pearson_corr", "per_sample_avg"), get(m_nc, "pearson_corr", "per_sample_avg"), "haut"),
        ("RMSE",               get(m_v5, "rmse"),                            get(m_nc, "rmse"),                            "bas"),
        ("MAE",                get(m_v5, "mae"),                             get(m_nc, "mae"),                             "bas"),
        ("Spread (ens std)",   get(m_v5, "spread_mean"),                     get(m_nc, "spread_mean"),                     "calib"),
        ("F1-p95",             get(m_v5, "f1_extremes", "p95"),              get(m_nc, "f1_extremes", "p95"),              "haut"),
        ("F1-p99",             get(m_v5, "f1_extremes", "p99"),              get(m_nc, "f1_extremes", "p99"),              "haut"),
        ("RAPSD distance",     get(m_v5, "rapsd_distance"),                  get(m_nc, "rapsd_distance"),                  "bas"),
        ("mu_HR ablation",     get(m_v5, "mu_HR_ablation", "delta_signal_ratio_avg"), get(m_nc, "mu_HR_ablation", "delta_signal_ratio_avg"), "haut"),
    ]

    sr_v5 = (rows[4][1] or 0) / (rows[2][1] or 1)
    sr_nc = (rows[4][2] or 0) / (rows[2][2] or 1)
    rows.insert(5, ("Spread/RMSE", sr_v5, sr_nc, "vers 1"))

    print(f"\\n{'Metrique':<22} {'V5-mini':>12} {'Noncausal':>12} {'D abs':>10} {'D rel %':>10} {'sens':>8} {'gagnant':>10}")
    print("-" * 100)
    summary = {}
    for name, v5, nc, sens in rows:
        if v5 is None or nc is None:
            print(f"{name:<22} {'n/a':>12} {'n/a':>12}")
            continue
        d_abs = v5 - nc
        d_rel = 100 * d_abs / nc if abs(nc) > 1e-12 else float('nan')
        if abs(d_rel) < 1.0:
            winner = "egal"
        elif sens == "haut" and d_abs > 0:
            winner = "V5"
        elif sens == "bas" and d_abs < 0:
            winner = "V5"
        elif sens == "calib":
            winner = "V5" if d_abs > 0 else "Noncausal"
        elif sens == "vers 1":
            winner = "V5" if abs(v5 - 1) < abs(nc - 1) else "Noncausal"
        else:
            winner = "Noncausal"
        summary[name] = {"v5": v5, "noncausal": nc, "delta_abs": d_abs, "delta_rel_pct": d_rel, "winner": winner}
        print(f"{name:<22} {v5:>12.4f} {nc:>12.4f} {d_abs:>+10.4f} {d_rel:>+10.2f} {sens:>8} {winner:>10}")

    out = RESULTS_DIR / "phase6_in_distribution.json"
    out.write_text(json.dumps({
        "v5_dir": str(V5_DIR),
        "noncausal_dir": str(NONCAUSAL_DIR),
        "comparison": summary,
        "raw_v5_metrics": m_v5,
        "raw_noncausal_metrics": m_nc,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\\n[OK] Phase 6 sauvegardee : {out}")"""))

# Cell 5 — Phase 7 intro
cells.append(md_cell("""---

## Phase 7 — Évaluation OOD (changement de régime)

Deux modes :

### Mode A — Dataset OOD disponible
Définis `OOD_DATA_PATH` ci-dessous (ex. `EC-Earth3_histupdated_compressed.nc`) et lance.

### Mode B — Fallback (pas de dataset OOD)
Si pas de dataset, on calcule des diagnostics de cohérence physique à partir des `final_validation_metrics.json` :
- Spread/RMSE (calibration probabiliste)
- RAPSD distance (structure spectrale)
- μ_HR ablation (dépendance causale architecturale)
- Shortcut ratio (qualité diffusion)

Le mode B ne mesure pas Δ_OOD strict mais donne un argument qualitatif sur la robustesse structurelle."""))

# Cell 6 — Phase 7 code
cells.append(code_cell("""# Phase 7 : OOD test (mode A si data dispo, mode B fallback sinon)
import json

# === Configure ici si tu as un dataset OOD ===
OOD_DATA_PATH = None  # ex: Path("/content/data_local/test/EC-Earth3_histupdated_compressed.nc")
# =============================================

if OOD_DATA_PATH is not None and Path(OOD_DATA_PATH).exists():
    print(f"Mode A : OOD dataset detecte ({OOD_DATA_PATH})")
    print("Pour eval OOD complet :")
    print("  - Surcharger CONFIG.data.hr_path = OOD_DATA_PATH")
    print("  - Relancer st_cdgm_validation_inference.ipynb avec CKPT_SAVE_DIR pointant V5_DIR")
    print("  - Sauvegarder les metrics OOD dans V5_DIR/final_validation_metrics_OOD.json")
    print("  - Idem pour noncausal")

else:
    print("Mode B : diagnostics de coherence physique (post-hoc)")
    print()

    m_v5 = json.loads((V5_DIR / "final_validation_metrics.json").read_text(encoding="utf-8"))
    m_nc = json.loads((NONCAUSAL_DIR / "final_validation_metrics.json").read_text(encoding="utf-8"))

    print(f"{'Diagnostic':<32} {'V5-mini':>12} {'Noncausal':>12} {'Interpretation':<35}")
    print("-" * 100)

    sr_v5 = m_v5["spread_mean"] / m_v5["rmse"]
    sr_nc = m_nc["spread_mean"] / m_nc["rmse"]
    print(f"{'Spread/RMSE (calibration)':<32} {sr_v5:>12.4f} {sr_nc:>12.4f} {'ideal=1, sub-disp si <1':<35}")
    print(f"{'RAPSD distance':<32} {m_v5['rapsd_distance']:>12.4f} {m_nc['rapsd_distance']:>12.4f} {'plus bas = mieux':<35}")

    ab_v5 = m_v5["mu_HR_ablation"]["delta_signal_ratio_avg"]
    ab_nc = m_nc["mu_HR_ablation"]["delta_signal_ratio_avg"]
    print(f"{'mu_HR ablation D/signal':<32} {ab_v5:>12.4f} {ab_nc:>12.4f} {'plus haut = plus causal':<35}")

    sc_v5 = m_v5["shortcut_diagnostic"]["shortcut_ratio"]
    sc_nc = m_nc["shortcut_diagnostic"]["shortcut_ratio"]
    print(f"{'Shortcut ratio':<32} {sc_v5:>12.4f} {sc_nc:>12.4f} {'>1 = vraie diffusion':<35}")

    print()
    print("Verdict qualitatif :")
    if ab_v5 > ab_nc:
        print(f"  [+] V5-mini : dependance mu_HR plus forte (+{(ab_v5-ab_nc)*100:.1f}%) -> causalite structurelle")
    if sr_v5 > sr_nc:
        print(f"  [+] V5-mini : meilleure calibration probabiliste (+{(sr_v5-sr_nc)*100:.1f}%)")
    if m_v5["rapsd_distance"] < m_nc["rapsd_distance"]:
        delta_rapsd = (m_nc['rapsd_distance']-m_v5['rapsd_distance'])/m_nc['rapsd_distance']*100
        print(f"  [+] V5-mini : meilleure fidelite spectrale (-{delta_rapsd:.1f}%)")
    print()
    print("[INFO] D_OOD strict non mesure (pas de dataset OOD pointe)")

    out = RESULTS_DIR / "phase7_ood_physical_diagnostics.json"
    out.write_text(json.dumps({
        "mode": "physical_diagnostics_fallback",
        "v5": {
            "spread_rmse_ratio": sr_v5,
            "rapsd_distance": m_v5["rapsd_distance"],
            "mu_HR_ablation": ab_v5,
            "shortcut_ratio": sc_v5,
        },
        "noncausal": {
            "spread_rmse_ratio": sr_nc,
            "rapsd_distance": m_nc["rapsd_distance"],
            "mu_HR_ablation": ab_nc,
            "shortcut_ratio": sc_nc,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\\n[OK] Phase 7 (fallback) sauvegardee : {out}")"""))

# Cell 7 — Phase 8 intro
cells.append(md_cell("""---

## Phase 8 — Protocole d'intervention `do(·)`

Cette phase **charge les modèles** et applique 3 interventions standardisées :

1. `do(q_850 × 1.20)` — humidité ↑20 % → précipitation devrait ↑
2. `do(t_850 += 3 K)` — réchauffement → précipitation devrait ↑
3. `do(u_850 × 1.10)` — vent ↑10 % → effet attendu marqué (advection humide)

Pour chaque intervention : `Δ_pred = predict(intervened) - predict(normal)` confronté au signe physiquement attendu.

**Q_int** = fraction des interventions où le signe correspond à l'attendu."""))

# Cell 8 — Phase 8 code (skeleton + helper)
cells.append(code_cell("""# Phase 8 : Intervention test
# Cette cellule charge les helpers et liste les interventions standardisées.
# L'exécution complète nécessite de charger les deux modèles.

from scripts.intervention_test import (
    INTERVENTIONS,
    resolve_variable_indices,
    apply_intervention,
    evaluate_intervention,
    compute_q_int,
    save_intervention_results,
)

print("Interventions standardisees :")
for spec in INTERVENTIONS:
    print(f"  - {spec['name']:<28s}  {spec['variable_name']}: "
          f"{spec['delta_type']} {spec['delta_value']:>6}  attendu {spec['expected_sign']:+d}")
print()

# Résolution des indices de variables LR depuis la config
# (Nécessite que CONFIG soit chargé — typiquement fait dans le notebook training_evaluation)
print("Pour executer Phase 8, dans une cellule suivante :")
print()
print("  1. Charger CONFIG (cf. notebook training_evaluation cell 14-20)")
print("  2. Recharger encoder + rcn_runner + regression_head + diffusion depuis V5_DIR")
print("     (puis a nouveau depuis NONCAUSAL_DIR)")
print("  3. Definir une fonction predict(batch) qui retourne l'ensemble [K, B, 1, H, W]")
print("  4. Appeler les helpers de scripts.intervention_test :")
print()
print("Exemple de boucle (a adapter) :")
print('-' * 60)"""))

# Cell 9 — Phase 8 execution template
cells.append(code_cell("""# Phase 8 : execution complete (a executer apres chargement des modeles)
#
# Pre-requis : CONFIG, V5_models (encoder/rcn/rh/diffusion), NC_models, builder, test_dataset
#
# Decommente et adapte selon ton setup :

# import torch
# from st_cdgm.evaluation.two_stage_inference import build_two_stage_inputs, sample_once_edm
# # convert_sample_to_batch est defini dans le notebook training_evaluation

# # Resolution des indices de variables
# lr_vars = list(CONFIG.data.lr_variables)
# resolved = resolve_variable_indices(lr_vars)
# print(f"Resolved interventions: "
#       f"{sum(1 for s in resolved if s['variable_idx'] is not None)}/{len(resolved)} variables found")

# # Sample batch normal
# sample = next(iter(test_dataset))
# batch_normal = convert_sample_to_batch(sample, builder, DEVICE)

# # Fonctions predict (a ecrire selon ton setup ; ressemblent a generate_prediction du training notebook)
# @torch.no_grad()
# def predict_with_v5(batch):
#     # 1. encoder + RCN -> H_T -> mu_HR via V5 stack
#     # 2. sample_once_edm avec mu_HR et baseline_log
#     # 3. Repeter K fois pour ensemble
#     ...
#     return ensemble  # [K, B, 1, H, W]

# @torch.no_grad()
# def predict_with_noncausal(batch):
#     ...
#     return ensemble

# # Eval des interventions
# results_v5 = [evaluate_intervention(predict_with_v5, batch_normal, spec) for spec in resolved]
# results_nc = [evaluate_intervention(predict_with_noncausal, batch_normal, spec) for spec in resolved]

# save_intervention_results(results_v5, results_nc, RESULTS_DIR / 'phase8_intervention.json')
# print(f"Q_int V5: {compute_q_int(results_v5):.3f}")
# print(f"Q_int NC: {compute_q_int(results_nc):.3f}")

print("Voir cellule precedente pour le template d'execution.")"""))

# Cell 10 — Synthese
cells.append(md_cell("""---

## Synthèse — résultats consolidés

Après exécution des Phases 6, 7, 8, les résultats sont dans :
```
RESULTS_DIR/
├── phase6_in_distribution.json
├── phase7_ood_physical_diagnostics.json (ou phase7_ood.json si Mode A)
└── phase8_intervention.json
```

### Construction du verdict pour la soutenance

**Si V5 gagne in-distribution + OOD + intervention** :
> *« Oracle V5-mini bat le baseline CorrDiff noncausal sur la majorité des métriques in-distribution, dégrade moins sous changement de climat, et satisfait Q_int ≥ 0.9 sur le protocole d'intervention. »*

**Si V5 gagne seulement sur calibration + structure + intervention** (cas trilemme) :
> *« V5-mini affiche un compromis assumé : performance in-distribution proche du noncausal, calibration probabiliste significativement améliorée (+30 %), fidélité spectrale +10 %, et capacité d'intervention démontrée. »*

**Si V5 ne gagne nulle part** :
> *Diagnostic montrant que les pertes V5 ont peut-être été mal calibrées. Plan B : ablation study.*"""))

# Cell 11 — Synthese code
cells.append(code_cell("""# Synthese des resultats sauvegardes
import json

print("=" * 70)
print("SYNTHESE V5 - fichiers generes")
print("=" * 70)
for fname in ["phase6_in_distribution", "phase7_ood_physical_diagnostics", "phase7_ood", "phase8_intervention"]:
    p = RESULTS_DIR / f"{fname}.json"
    if p.exists():
        size_kb = p.stat().st_size / 1024
        print(f"  [OK] {p.name:50s}  {size_kb:>8.1f} KB")
    else:
        print(f"  [--] {p.name:50s}  (non genere)")

print()
print("Pour le memoire, voir :")
print("  - Tableau metrique x variant      -> phase6_in_distribution.json (cle 'comparison')")
print("  - D_OOD ou diagnostics physiques  -> phase7_*.json")
print("  - Q_int et signe interventions    -> phase8_intervention.json")"""))


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("st_cdgm_v5_evaluation.ipynb")
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"[OK] Notebook cree : {out} ({len(cells)} cellules)")
