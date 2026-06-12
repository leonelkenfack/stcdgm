# Path C+ — 4-node Causal SCM Refactor

**Branch** : `four-node-causal` (depuis `two-stage-causal`)
**Target hardware** : Colab Pro+ A100 40-80GB
**Status** : Phase 0.1 setup en cours
**Start date** : 2026-06-12

## Contenu de ce dossier

| Fichier | Rôle |
|---|---|
| `HYPERPLAN.md` | Plan complet 14-15 semaines, 95 issues identifiés, ablation matrix |
| `LAUNCH_COLAB_A100.md` | Guide concret de lancement sur Colab A100 |
| `P0_CHECKLIST.md` | 22 P0 fixes à appliquer en 22 commits |
| `PRE_REGISTRATION.md` | 5 hypothèses pre-registered (H1-H5) avant tout retrain |
| `audits/` | Audits complets des 3 experts |
| `scripts/` | Scripts spécifiques au Path C+ (setup A100, pre-flight) |
| `tests/` | Unit tests pour les 22 P0 fixes |

## Plan d'exécution résumé

```
Phase 0   : Setup + 22 P0 fixes (semaines 1-2)         — 0 compute, dev seul
Phase A0' : RE-EVAL V5-mini avec fixes (semaine 3)     — 10h Colab A100
Phase A0'': RE-TRAIN V5-mini propre 3 seeds (sem 3-5)  — 60-75h T4 OR 5-8h A100
Phase A1  : 6-node minimal fix 3 seeds (semaine 6)     — 30-40h T4 OR 2.5-5h A100
DECISION  : Q_phys ≥ 0.65 ? → STOP, sinon Path C+ full
Phase C+  : 4-node refactor + Stage1+2 (semaines 8-15) — 90-144h T4 OR 6-12h A100
Phase Z   : Documentation + stat analysis (semaine 15) — 0 compute
```

## Budget Colab A100

| Scénario | Compute units A100 | Coût Pro+ équiv |
|---|---|---|
| Path A1 success | ~8-20h × 13 units/h = 100-260 units | $10-30 (pay-as-you-go) |
| Path C+ complet | ~19-60h × 13 units/h = 250-780 units | $30-100 (pay-as-you-go) |

Colab Pro+ : $50/mois donne ~500 compute units, suffisant pour Path A1 en 1 mois ou Path C+ en 2-3 mois.

Stratégie recommandée : Pro+ pour le 1er mois (dev + A1), pay-as-you-go pour Path C+ si nécessaire.

## Quick start

```bash
# 1. Verify branch
git branch --show-current  # → four-node-causal

# 2. Read the plan
less path_c_plus/HYPERPLAN.md
less path_c_plus/LAUNCH_COLAB_A100.md
less path_c_plus/P0_CHECKLIST.md

# 3. Run pre-flight checks (after P0 fixes applied)
pytest path_c_plus/tests/

# 4. Launch on Colab A100
# Upload path_c_plus/scripts/colab_a100_bootstrap.ipynb to Colab
# Connect to A100 runtime
# Run all cells
```

## Issues consensus (95 total, 22 P0)

Voir `HYPERPLAN.md` §1 pour la liste complète. Catégorisation :

- **22 P0** (showstoppers) : invalident résultats, MUST fix
- **47 P1** (degrade results) : fix avant publication
- **21 P2** (perf/UX) : fix avant final run
- **5 P3** (cosmetic) : optional

## Décisions importantes

1. **Provider GPU** : Colab Pro+ A100 (40GB ou 80GB selon dispo)
2. **Stratégie fixes** : 22 commits individuels (1 P0 = 1 commit, traçabilité maximale)
3. **Path** : Full Path C+ (refactor 4-node + PCMCI + Stage 1+2 retrain) avec ablation A1 préalable
4. **Multi-seed** : 3 seeds [42, 7, 123] pour toute claim statistique
5. **Pre-registration** : H1-H5 commit avant tout training
