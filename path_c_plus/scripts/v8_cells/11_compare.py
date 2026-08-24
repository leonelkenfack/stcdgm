# >>> Cell 11 : comparaison aux modeles deja evalues + verdict pre-enregistre
# Les autres modeles ne sont PAS reevalues : on relit la table produite par le
# 3-way V6' (metrique Conv A corrigee, meme split, meme K, meme composition mm).
# Cette cellule ne vaut que si le protocole coincide — elle le verifie et le dit
# plutot que d'aligner des nombres incomparables.
TABLE = ["conv_A_F1p99", "conv_A_F1p95", "conv_B_F1p99", "conv_B_F1p95",
         "pearson_global", "rmse", "mae", "crps", "ssr", "fss_p99",
         "rapsd_distance", "bias_mm", "rx1day_bias", "qbias_p99_pct",
         "qbias_p999_pct"]
# V5 = ORACLE (le modele causal), noncausal = CorrDiff (la baseline).
LABELS = {"V6_prime": "V6'", "V5_causal": "ORACLE", "noncausal_v4": "CorrDiff"}

_BL_LOCAL = RESULTS_DIR / "baselines_3way.json"
_cands = [_BL_LOCAL]
if IN_COLAB:
    _d = Path(DRIVE_ROOT) / "oracle_v6_prime"
    _cands = sorted(_d.glob("seed_*/v6_prime_3way_final_*.json")) + _cands
_bl_path = next((p for p in _cands if p.exists()), None)

baselines, refs_status, ref_protocol = {}, "ABSENTES", None
if _bl_path is not None:
    _bl = json.load(open(_bl_path))
    baselines = {k: v for k, v in _bl.get("three_way", {}).items()
                 if k in LABELS}
    refs_status = "RECOMPUTED_IN_PROTOCOL"
    if not baselines:
        print(f"!! {_bl_path} ne contient pas de bloc `three_way` exploitable "
              f"(clefs vues : {sorted(_bl.get('three_way', {}))}).")
    # Copie locale : le prochain run n'aura plus besoin de Drive.
    if baselines and _bl_path != _BL_LOCAL:
        json.dump({"three_way": baselines, "source": str(_bl_path)},
                  open(_BL_LOCAL, "w"), indent=2, default=float)
    print(f"references relues : {_bl_path}")
    _rp = Path(DRIVE_ROOT) / "oracle_v6_prime/recomputed_pergrid_references.json"
    if IN_COLAB and _rp.exists():
        ref_protocol = json.load(open(_rp)).get("protocol")

if not baselines:
    # Repli : les seuils pre-enregistres du depot. Le prereg V6' les marque
    # lui-meme STALE — ils ont ete calcules avec une Conv A BUGGEE (quantile
    # scalaire au lieu du broadcast per-pixel). On les affiche pour ne pas
    # laisser la table vide, jamais pour en tirer un verdict.
    _pr = json.load(open("path_c_plus/audit/V6_PRIME_seuils_preregistered.json"))
    _t = _pr["targets_to_beat"]
    baselines = {
        "V5_causal":    {"conv_A_F1p99": float(_t["co_primary_1_per_gridpoint"]["v5_causal_seed42"]),
                         "conv_B_F1p99": float(_t["co_primary_2_pooled"]["v5_causal_seed42"])},
        "noncausal_v4": {"conv_A_F1p99": float(_t["co_primary_1_per_gridpoint"]["noncausal_v4"]),
                         "conv_B_F1p99": float(_t["co_primary_2_pooled"]["noncausal_v4"])},
    }
    refs_status = "STALE_ANCIENNE_METRIQUE"
    print("!! Table 3-way introuvable. Repli sur les seuils pre-enregistres,")
    print("!! calcules avec la Conv A BUGGEE (prereg V6', REFS_STALE_WARNING).")
    print("!! Aucun verdict co-primaire n'est recevable dans cet etat : relancer")
    print("!! les Cells 13-15 du notebook V6' pour regenerer la table in-protocol.")

# --- parite de protocole ---------------------------------------------------
# Comparer sans reevaluer n'est licite que si les reglages coincident. Un K ou
# un nombre de pas different change les metriques a modele EGAL.
_mine = {"K": K_VERDICT, "num_steps": NUM_STEPS, "n_test": N_TEST}
_ecarts = []
if ref_protocol:
    for _k in ("K", "num_steps", "n_test"):
        if _k in ref_protocol and int(ref_protocol[_k]) != int(_mine[_k]):
            _ecarts.append(f"{_k}: reference={ref_protocol[_k]} vs V8={_mine[_k]}")
    print(f"protocole de reference : {ref_protocol}")
else:
    print("protocole de reference non retrouve (recomputed_pergrid_references.json) :")
    print("  parite NON verifiee ; les valeurs attendues sont K=32, 24 pas, cfg 0.0.")
comparable = (refs_status == "RECOMPUTED_IN_PROTOCOL") and not _ecarts
if _ecarts:
    print("!! ECART DE PROTOCOLE - la comparaison n'est pas apples-to-apples :")
    for _e in _ecarts:
        print(f"!!   {_e}")

# --- table -----------------------------------------------------------------
rows = {"V8": res}
rows.update({LABELS[k]: v for k, v in baselines.items()})
_lower = {"rmse", "mae", "rapsd_distance", "crps"}          # plus bas = mieux
_abs0 = {"bias_mm", "rx1day_bias", "qbias_p99_pct", "qbias_p999_pct"}  # proche de 0
print()
print("=" * 78)
print(f"COMPARAISON ({refs_status}, Conv A per-pixel, composition mm)")
print("=" * 78)
print(f"{'Metrique':22s} " + " ".join(f"{n:>12s}" for n in rows))
for met in TABLE:
    vals = {n: rows[n].get(met, float("nan")) for n in rows}
    _fin = [n for n in vals if vals[n] == vals[n]]
    if not _fin:
        continue
    if len(_fin) < 2:
        best = None          # une seule valeur : rien a comparer, pas de "best"
    elif met in _abs0:
        best = min(_fin, key=lambda n: abs(vals[n]))
    elif met == "ssr":                       # calibration : ~1 est le mieux
        best = min(_fin, key=lambda n: abs(vals[n] - 1.0))
    elif met in _lower:
        best = min(_fin, key=lambda n: vals[n])
    else:
        best = max(_fin, key=lambda n: vals[n])
    _cells = " ".join(f"{vals[n]:12.4f}" if vals[n] == vals[n] else f"{'-':>12s}"
                      for n in rows)
    print(f"{met:22s} {_cells}" + (f"   <- {best}" if best else ""))

# --- verdict ---------------------------------------------------------------
# Bande pre-enregistree (critique froide 2026-07-05) : le DAG gele est une
# feature OOD ASSUMEE, une regression in-distribution de ~2-8 % est PREVUE et
# acceptee. La cible V8 est la PARITE ID, pas un gain ID.
BANDE_ID_PCT = 8.0


def _delta(met, ref_key):
    r = baselines.get(ref_key, {}).get(met, float("nan"))
    v = res.get(met, float("nan"))
    if r != r or v != v or abs(r) < 1e-9:
        return float("nan"), float("nan")
    return v - r, 100.0 * (v - r) / abs(r)


verdict = {"refs_status": refs_status, "comparable": bool(comparable),
           "ecarts_protocole": _ecarts, "metrics_v8": res,
           "baselines": baselines, "protocole_v8": _mine,
           "bande_id_pct": BANDE_ID_PCT, "deltas": {}}
print()
print("=== ECARTS vs ORACLE (V5) et CorrDiff ===")
for met in ("conv_A_F1p99", "conv_B_F1p99", "rmse"):
    for ref_key, nom in (("V5_causal", "ORACLE"), ("noncausal_v4", "CorrDiff")):
        d, pct = _delta(met, ref_key)
        if pct != pct:
            continue
        verdict["deltas"][f"{met}_vs_{nom}"] = {"abs": d, "pct": pct}
        _st = ("PARITE" if abs(pct) <= BANDE_ID_PCT
               else ("GAIN" if (pct > 0) != (met == "rmse") else "HORS_BANDE"))
        print(f"  {met:14s} vs {nom:9s} : {d:+.4f} ({pct:+.1f} %)  {_st}")

verdict["lecture"] = (
    "ATTRIBUTION : plusieurs interrupteurs V8 sont actifs simultanement, ce run "
    "ne dit RIEN sur la contribution de chacun — il faut les runs P1 a un seul "
    "changement. "
    "CIBLE : parite in-distribution + gain OOD, pas un gain ID. Une regression "
    "ID de quelques pour cent est PREVUE et acceptee : le DAG gele est une "
    "feature OOD assumee (critique froide 2026-07-05). "
    "STATUT : ce verdict n'est PAS final. Il le devient sur EC-Earth3, puis UNE "
    "SEULE FOIS sur le holdout NorESM2-MM. "
    "INCERTITUDE : une seule graine, pas d'intervalle de confiance par blocs — "
    "un ecart de 1-2 points n'est pas interpretable.")
if not comparable:
    verdict["lecture"] = ("COMPARAISON NON VALIDE (" + refs_status + "). "
                          + verdict["lecture"])
json.dump(verdict, open(RESULTS_DIR / "v8_verdict.json", "w"),
          indent=2, default=float)
print()
print(f"=== ECRIT {RESULTS_DIR / 'v8_verdict.json'} ===")
print(verdict["lecture"])
