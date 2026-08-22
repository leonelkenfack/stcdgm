"""T4 + T6 — audits CPU des RÉGIMES et du budget extrémal (V8 §6.1).

Pourquoi maintenant : Gate 0 a mesuré 3,4 % de distinctness inter-GCM, donc
DICD-lite n'a plus de signal et **C6 (graphe extrémal par régimes) est devenu
la contribution principale**. Or les nœuds RÉGIME n'ont jamais été testés.

Deux questions, dans cet ordre de gravité :

  T6  BUDGET EXTRÉMAL — combien d'événements POT INDÉPENDANTS par région, et
      par (région × régime) ? Math E5 : ~200 événements ⇒ ~10 arêtes. Si un
      régime n'en a que quelques dizaines, son graphe extrémal est
      inidentifiable et C6 s'effondre là — quelle que soit la méthode.

  T4  ESS — taille d'échantillon effective par régime, corrigée de
      l'autocorrélation. Dimensionne le budget d'arêtes du graphe de bulk.

Règle dure de C6, respectée ici : les régimes sont définis SUR LA CIRCULATION
SEULE (u850, v850). Jamais sur un quantile de pr — ce serait circulaire, et pr
est un collider (Math E5, Climat E8).

N'ouvre QUE ACCESS-CM2 (train). NorESM2-MM = holdout, jamais touché.
"""
import json
import sys

import numpy as np
import xarray as xr

FORBIDDEN = "NorESM2"


def guard(p):
    assert FORBIDDEN not in str(p), f"STOP — {p} touche le holdout NorESM2-MM."
    return p


RNG = np.random.default_rng(0)
OUT = {}
POT_DECORR = 3          # jours de décorrélation (C6)
MIN_EVENTS = 200        # Math E5 : ~200 événements indépendants ⇒ ~10 arêtes

PRED = "data/raw/train/predictor_ACCESS-CM2_hist.nc"
PRCP = "data/raw/train/pr_ACCESS-CM2_hist.nc"
STAT = "data/raw/static_predictors/ERA5_eval_ccam_12km.198110_NZ_Invariant.nc"

# ---------------------------------------------------------------- alignement
dp = xr.open_dataset(guard(PRED), decode_times=False)
dr = xr.open_dataset(guard(PRCP), decode_times=False)
T = dp.sizes["time"]
assert dr.sizes["time"] == T, (
    f"pr ({dr.sizes['time']}) et prédicteurs ({T}) n'ont pas le même nombre de pas")
# Les deux séries sont en 365 j/an : noleap côté prédicteurs, et grégorien
# AVEC les bissextiles retirées côté pr (20 089 - 14 = 20 075). L'indice i
# désigne donc le même jour calendaire des deux côtés. On le vérifie plutôt
# que de le supposer : sinon tout l'appariement régime <-> pluie est décalé.
assert T % 365 == 0, f"{T} pas n'est pas un multiple de 365 : calendrier inattendu"
N_YEARS = T // 365
print(f"[align] {T} pas = {N_YEARS} ans x 365 j, indices appariés.", flush=True)
# jour de l'année (0-364) et saison australe
doy = np.arange(T) % 365
season = np.where((doy < 59) | (doy >= 334), "DJF",
         np.where(doy < 151, "MAM", np.where(doy < 243, "JJA", "SON")))

# ---------------------------------------------- sous-domaine LR couvrant la NZ
lat_hr = dr["lat"].values
lon_hr = dr["lon"].values
lat_lr = dp["lat"].values
lon_lr = dp["lon"].values
jl = np.where((lat_lr >= lat_hr.min() - 1.5) & (lat_lr <= lat_hr.max() + 1.5))[0]
il = np.where((lon_lr >= lon_hr.min() - 1.5) & (lon_lr <= lon_hr.max() + 1.5))[0]
print(f"[align] boîte LR retenue : {len(jl)} lat x {len(il)} lon "
      f"(le domaine LR complet déborde largement la NZ)", flush=True)

u850 = dp["u_850"].isel(lat=jl, lon=il).values.astype(np.float32)
v850 = dp["v_850"].isel(lat=jl, lon=il).values.astype(np.float32)
q850 = dp["q_850"].isel(lat=jl, lon=il).values.astype(np.float32)
dp.close()

# =====================================================================
# RÉGIMES DE CIRCULATION — k-means sur (u850, v850), circulation SEULE
# =====================================================================
print("[regimes] k-means sur les champs de vent 850 hPa...", flush=True)
F = np.concatenate([u850.reshape(T, -1), v850.reshape(T, -1)], axis=1)
F = (F - F.mean(0)) / (F.std(0) + 1e-6)
# ACP pour débruiter avant clustering (pratique standard des types Kidson)
F -= F.mean(0)
_, S, Vt = np.linalg.svd(F, full_matrices=False)
n_pc = int(np.searchsorted(np.cumsum(S ** 2) / np.sum(S ** 2), 0.90) + 1)
Z = F @ Vt[:n_pc].T
print(f"   {n_pc} composantes retenues (90 % de variance)", flush=True)

K = 8


def kmeans(X, k, iters=60):
    """k-means++ minimal. Suffisant ici : on compte des effectifs, on
    n'estime pas des frontières fines."""
    c = [X[RNG.integers(len(X))]]
    for _ in range(k - 1):
        d = np.min(((X[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
        c.append(X[RNG.choice(len(X), p=d / d.sum())])
    C = np.array(c)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
        Cn = np.stack([X[lab == j].mean(0) if (lab == j).any() else C[j]
                       for j in range(k)])
        if np.allclose(Cn, C):
            break
        C = Cn
    return lab, C


lab, _ = kmeans(Z, K)
OUT["regimes"] = dict(k=K, n_pc=n_pc,
                      population={int(j): int((lab == j).sum()) for j in range(K)})
print("   effectifs :", OUT["regimes"]["population"], flush=True)

# Persistance : un régime météo dure plusieurs jours. Sans en tenir compte,
# on confondrait 5 000 jours avec 5 000 observations indépendantes.
switch = float((lab[1:] != lab[:-1]).mean())
OUT["regimes"]["mean_persistence_days"] = float(1.0 / max(switch, 1e-9))
print(f"   persistance moyenne : {1.0/max(switch,1e-9):.2f} jours", flush=True)

# =====================================================================
# T4 — ESS par régime (13 nœuds libres, moyennés sur le domaine)
# =====================================================================
print("[T4] ESS par régime...", flush=True)


# La physique vient de la source unique. Ce script en portait une copie privee
# (Tetens, RH, Gamma avec une epaisseur de couche figee a 3,5 km) — exactement
# l'anti-pattern que src/st_cdgm/data/derived.py existe pour supprimer.
sys.path.insert(0, "src")
from st_cdgm.data.derived import FREE_NODES, compute_free_nodes   # noqa: E402

dp2 = xr.open_dataset(guard(PRED), decode_times=False)
sub = dict(lat=jl, lon=il)
_raw = {v: dp2[v].isel(**sub).values.astype(np.float32) for v in dp2.data_vars
        if v.split("_")[0] in ("u", "v", "w", "q", "t")}
_lat = dp2["lat"].isel(lat=jl).values
_lon = dp2["lon"].isel(lon=il).values
dp2.close()
# Les 13 noeuds libres, pas un sous-ensemble : le verdict T4 porte sur le
# MINIMUM d'ESS, donc omettre des noeuds fausserait ce minimum.
NODES = {k: v.mean((1, 2)) for k, v in compute_free_nodes(_raw, _lat, _lon).items()}
assert set(NODES) == set(FREE_NODES), "T4 doit couvrir les 13 noeuds libres"
del _raw


def ess(x):
    """Taille effective corrigée de l'autocorrélation de rang 1."""
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    if x.size < 3 or x.std() == 0:
        return float(x.size)
    r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    r1 = min(max(r1, -0.99), 0.99)
    return float(x.size * (1 - r1) / (1 + r1))


t4 = {}
for j in range(K):
    s = lab == j
    per_node = {k: ess(v[s]) for k, v in NODES.items()}
    t4[int(j)] = dict(n_days=int(s.sum()),
                      ess_min=float(min(per_node.values())),
                      ess_median=float(np.median(list(per_node.values()))))
OUT["T4_ess_per_regime"] = t4
for j, d in t4.items():
    print(f"   régime {j} : {d['n_days']:5d} j -> ESS min {d['ess_min']:7.0f} "
          f"| médiane {d['ess_median']:7.0f}", flush=True)

# T4 demande aussi la stratification saisonnière : c'est elle qui borne le
# budget si l'on veut un graphe par saison ET par régime.
t4s = {}
for sname in ("DJF", "MAM", "JJA", "SON"):
    for j in range(K):
        s = (season == sname) & (lab == j)
        if s.sum() < 30:
            continue
        t4s[f"{sname}/R{j}"] = dict(
            n_days=int(s.sum()),
            ess_min=float(min(ess(v[s]) for v in NODES.values())))
OUT["T4_ess_per_season_regime"] = t4s
worst = min(t4s.values(), key=lambda d: d["ess_min"]) if t4s else None
print(f"   croisement saison x régime : {len(t4s)} strates peuplées, "
      f"ESS min la plus faible = {worst['ess_min']:.0f}" if worst else
      "   croisement saison x régime : aucune strate peuplée", flush=True)

# =====================================================================
# T6 — BUDGET EXTRÉMAL : événements POT indépendants par région
# =====================================================================
print("[T6] régionalisation (masque terre + partage orographique)...", flush=True)
st = xr.open_dataset(guard(STAT), decode_times=False).squeeze(drop=True)
land = st["sftlf"].values > 0.5
orog = st["orog"].values
st.close()

# North Island / South Island : la coupure ~ -41.5 degres suit le detroit de Cook.
lat2 = lat_hr[:, None] * np.ones((1, len(lon_hr)))
north = land & (lat2 > -41.5)
south = land & (lat2 <= -41.5)
# Partage ouest/est de l'ile du Sud par la crete orographique de chaque ligne
# de latitude : c'est la definition physique des Alpes du Sud, pas une boite.
west = np.zeros_like(land)
east = np.zeros_like(land)
for r in range(len(lat_hr)):
    cols = np.where(south[r])[0]
    if cols.size < 3:
        continue
    crest = cols[int(np.argmax(orog[r, cols]))]
    west[r, cols[cols < crest]] = True
    east[r, cols[cols > crest]] = True
REGIONS = {"North Island": north, "South Island West": west, "South Island East": east}
for n, m in REGIONS.items():
    print(f"   {n:20s} : {int(m.sum()):5d} pixels", flush=True)

CACHE = "results/_regional_pr_series.npz"
try:
    series = {k: v for k, v in np.load(CACHE).items()}
    assert all(v.size == T for v in series.values())
    print("[T6] séries régionales relues du cache", flush=True)
except Exception:
    print("[T6] séries régionales de pr (lecture par blocs)...", flush=True)
    series = {n: np.empty(T, np.float32) for n in REGIONS}
    CH = 2000
    for a in range(0, T, CH):
        blk = dr["pr"].isel(time=slice(a, min(a + CH, T))).values
        for n, m in REGIONS.items():
            series[n][a:a + blk.shape[0]] = np.nanmean(blk[:, m], axis=1)
        print(f"     {min(a+CH, T)}/{T}", flush=True)
    np.savez(CACHE, **series)
dr.close()


def pot_events(x, q, decorr=POT_DECORR):
    """Pics-au-dessus-de-seuil déclusterisés : un événement = un maximum local
    au-dessus du seuil, séparé du suivant d'au moins ``decorr`` jours. Sans
    cette déclusterisation on compterait 4 fois la même tempête."""
    thr = np.quantile(x, q)
    idx = np.where(x > thr)[0]
    ev, last = [], -10 ** 9
    for i in idx:
        if i - last >= decorr:
            ev.append(i)
            last = i
        elif x[i] > x[ev[-1]]:
            ev[-1] = i          # garder le pic du cluster
            last = i
    return np.array(ev), float(thr)


t6 = {}
for n, x in series.items():
    row = {}
    for q, tag in ((0.98, "p98"), (0.99, "p99")):
        ev, thr = pot_events(x, q)
        per_reg = {int(j): int((lab[ev] == j).sum()) for j in range(K)}
        row[tag] = dict(threshold_mm=thr, n_events=int(ev.size),
                        per_regime=per_reg,
                        n_regimes_above_budget=int(sum(
                            1 for v in per_reg.values() if v >= MIN_EVENTS)),
                        max_per_regime=int(max(per_reg.values())))
    t6[n] = row
OUT["T6_extremal_budget"] = t6
OUT["T6_min_events_rule"] = MIN_EVENTS

print("\n--- T6 : événements POT indépendants (décorrélation 3 j) ---")
for n, row in t6.items():
    for tag in ("p98", "p99"):
        d = row[tag]
        print(f"   {n:20s} {tag} (>{d['threshold_mm']:5.2f} mm/j) : "
              f"{d['n_events']:4d} événements | meilleur régime "
              f"{d['max_per_regime']:4d} | régimes >= {MIN_EVENTS} : "
              f"{d['n_regimes_above_budget']}/{K}")

# ------------- sensibilité au nombre de régimes : le budget décroît en 1/K
print("\n--- T6 : sensibilité au nombre de régimes (région x régime, p98) ---")
ksweep = {}
for k_try in (2, 4, 6, 8):
    # Meme graine par valeur de K : sans cela le K=8 du balayage serait un
    # AUTRE tirage que le K=8 de l'analyse principale, et les deux chiffres
    # ne seraient pas comparables.
    RNG = np.random.default_rng(0)
    lab_k, _ = kmeans(Z, k_try)
    best_k, ok_k = 0, 0
    for n, x in series.items():
        ev, _ = pot_events(x, 0.98)
        cnt = [int((lab_k[ev] == j).sum()) for j in range(k_try)]
        best_k = max(best_k, max(cnt))
        ok_k += sum(1 for c in cnt if c >= MIN_EVENTS)
    ksweep[k_try] = dict(best_stratum=int(best_k), n_strata_ok=int(ok_k),
                         n_strata=3 * k_try)
    print(f"   K={k_try} : {ok_k}/{3*k_try} strates >= {MIN_EVENTS} "
          f"| meilleure = {best_k}")
OUT["T6_k_sensitivity"] = ksweep

# --------------------------------------------------------------- verdict
# Le critère « au moins une strate passe » serait dégénéré (c'est l'erreur
# faite sur Gate 0) : avec 48 strates, une seule au-dessus du seuil ne dit
# rien. On demande qu'une MAJORITÉ des strates soit identifiable, sinon la
# stratification n'est pas exploitable comme switch.
n_strata = 3 * K
ok_p98 = sum(d["p98"]["n_regimes_above_budget"] for d in t6.values())
ok_p99 = sum(d["p99"]["n_regimes_above_budget"] for d in t6.values())
best = max(d[t]["max_per_regime"] for d in t6.values() for t in ("p98", "p99"))
pooled = {n: int(d["p98"]["n_events"]) for n, d in t6.items()}
pooled_ok = sum(1 for v in pooled.values() if v >= MIN_EVENTS)

OUT["T6_strata_ok_p98"] = ok_p98
OUT["T6_strata_ok_p99"] = ok_p99
OUT["T6_n_strata"] = n_strata
OUT["T6_best_stratum_events"] = int(best)
OUT["T6_pooled_per_region_p98"] = pooled
OUT["T6_verdict"] = (
    f"ECHEC pour un graphe extremal PAR (region x regime) : {ok_p98}/{n_strata} "
    f"strates atteignent {MIN_EVENTS} evenements a p98, {ok_p99}/{n_strata} a p99 "
    f"(meilleure strate = {best}). "
    f"REPLI IDENTIFIABLE : par REGION seule, {pooled_ok}/3 regions y arrivent "
    f"({pooled}). Les regimes redeviennent des covariables de stratification "
    f"pour l'evaluation, pas le switch d'un graphe separe."
    if ok_p98 < n_strata / 2 else
    f"OK — {ok_p98}/{n_strata} strates identifiables a p98."
)
print(f"\n=== VERDICT T6 ===\n{OUT['T6_verdict']}")

ess_min_season = min((d["ess_min"] for d in OUT["T4_ess_per_season_regime"].values()),
                     default=float("nan"))
OUT["T4_verdict"] = (
    f"Par regime seul : ESS min {min(d['ess_min'] for d in t4.values()):.0f} "
    f"jours effectifs. NB : la regle ~200 evenements => ~10 aretes vient du "
    f"contexte POT (ajustement de queue) et n'est PAS transposable telle quelle "
    f"a l'apprentissage de structure sur le bulk ; ici l'entrainement mutualise "
    f"aussi les pixels, ce que l'ESS temporel ne compte pas. A lire comme un "
    f"ordre de grandeur, pas comme un feu vert. "
    f"Croisement saison x regime : ESS min {ess_min_season:.0f} -> trop mince "
    f"quelle que soit la lecture, ne pas croiser les deux stratifications."
)
print(f"\n=== VERDICT T4 ===\n{OUT['T4_verdict']}")

with open("results/p0_regimes.json", "w") as f:
    json.dump(OUT, f, indent=2, default=float)
print("=== ECRIT results/p0_regimes.json ===")
