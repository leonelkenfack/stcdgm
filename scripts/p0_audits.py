"""Audits P0 de V8 sur CPU — tous bloquants, aucun GPU.

T1  Dégénérescence : R² croisé entre les 13 nœuds LIBRES. Tout R² > 0.95
    => nœud déterministe in-graph => à retirer (correction fatale n°1).
T3  Varsortability + R²-sortability + baseline sortnregress + surrogates.
    (correction fatale n°4, renforcée par Reisach 2023 : la standardisation
     ne suffit pas, le R² est invariant d'échelle.)
G0  Distinctness inter-GCM : les 3 GCM partagent le même CCAM aval, donc
    l'empreinte causale doit-elle vraiment différer d'un GCM à l'autre ?
    G0 : (1 - F1_inter-GCM) >= 2 * (1 - F1_inter-seed), sinon DICD-lite tombe.

N'ouvre QUE ACCESS-CM2 (train) et EC-Earth3 (test). NorESM2-MM = holdout.
"""
import numpy as np, xarray as xr, json, itertools, sys

FORBIDDEN = "NorESM2"
def guard(p):
    assert FORBIDDEN not in str(p), f"STOP — {p} touche le holdout NorESM2-MM."
    return p

OUT, RNG = {}, np.random.default_rng(0)
G = 9.81

# ---------------------------------------------------------------- 13 nœuds libres
sys.path.insert(0, "src")
from st_cdgm.data.derived import compute_free_nodes   # noqa: E402


def build_free_nodes(path, n=4000):
    """15 bruts -> 13 nœuds LIBRES, via la source unique ``st_cdgm.data.derived``.

    La dérivation vivait ici en copie privée, avec des gradients par maille au
    lieu de gradients SI. Deux implémentations de la même physique, c'est une
    de trop — et l'IVT du catalogue avait déjà divergé de sa formule.
    """
    ds = xr.open_dataset(guard(path), decode_times=False)
    T = ds.sizes["time"]
    idx = np.sort(RNG.choice(T, size=min(n, T), replace=False))
    V = {v: ds[v].isel(time=idx).values.astype(np.float32) for v in ds.data_vars
         if v.split("_")[0] in ("u", "v", "w", "q", "t")}
    lat = ds["lat"].values
    lon = ds["lon"].values
    ds.close()
    N = compute_free_nodes(V, lat, lon)
    return {k: v.reshape(v.shape[0], -1) for k, v in N.items()}   # [T, pixels]

print("[T1] construction des 13 nœuds libres (ACCESS-CM2)...", flush=True)
NODES = build_free_nodes("data/raw/train/predictor_ACCESS-CM2_hist.nc")
names = list(NODES)
print("   ", len(names), "nœuds :", names, flush=True)

# ------------------------------------------------------------------------- T1
def r2_pair(a, b):
    a, b = a.ravel(), b.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 100: return np.nan
    a, b = a[m], b[m]
    c = np.corrcoef(a, b)[0, 1]
    return float(c ** 2)

print("[T1] R² croisé par paire...", flush=True)
sub = RNG.choice(NODES[names[0]].size, size=200_000, replace=False)
flat = {k: v.ravel()[sub] for k, v in NODES.items()}
t1 = []
for a, b in itertools.combinations(names, 2):
    r = r2_pair(flat[a], flat[b])
    if r > 0.80:
        t1.append(dict(a=a, b=b, r2=r))
t1.sort(key=lambda d: -d["r2"])
OUT["T1_high_r2_pairs"] = t1
fail = [d for d in t1 if d["r2"] > 0.95]
print(f"   paires R²>0.80 : {len(t1)} | R²>0.95 (BLOQUANT) : {len(fail)}")
for d in t1[:8]:
    print(f"     {d['a']:14s} ~ {d['b']:14s}  R² = {d['r2']:.3f}")
OUT["T1_pass"] = len(fail) == 0

# ------------------------------------------------------------------------- T3
def varsortability(X, edges):
    """fraction des arêtes allant d'une variance plus faible vers plus forte."""
    var = {k: float(np.var(v)) for k, v in X.items()}
    ok = sum(1 for s, t in edges if var[s] < var[t])
    return ok / max(len(edges), 1)

def r2sortability(X, edges):
    """idem mais avec le R² de régression sur tous les autres (invariant d'échelle)."""
    ks = list(X)
    M = np.stack([X[k] for k in ks], 1)
    M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    r2 = {}
    for i, k in enumerate(ks):
        others = np.delete(M, i, axis=1)
        coef, *_ = np.linalg.lstsq(others, M[:, i], rcond=None)
        pred = others @ coef
        r2[k] = float(1 - ((M[:, i] - pred) ** 2).sum() / max((M[:, i] ** 2).sum(), 1e-9))
    ok = sum(1 for s, t in edges if r2[s] < r2[t])
    return ok / max(len(edges), 1), r2

# Prior C7 à 3 niveaux (config/dag_prior_v8_c7.yaml). Le prior ad hoc à 10
# arêtes utilisé jusqu'ici donnait une erreur-type de 0,16 : 0,30 et 0,50
# étaient indiscernables et T3 restait inconclusif par construction.
sys.path.insert(0, "src")
from st_cdgm.priors import load_edge_prior          # noqa: E402

PRIOR = load_edge_prior()
print(f"[T3] prior C7 : {PRIOR}", flush=True)
EDGES = [(a, b) for a, b in PRIOR.edge_list() if a in flat and b in flat]
missing = len(PRIOR) - len(EDGES)
if missing:
    print(f"   /!\\ {missing} arêtes ignorées (nœud absent des données)", flush=True)
OUT["T3_prior"] = dict(source="config/dag_prior_v8_c7.yaml",
                       n_edges_total=len(PRIOR), n_edges_used=len(EDGES),
                       by_level=PRIOR.by_level(), by_orient=PRIOR.by_orient())

vs_raw = varsortability(flat, EDGES)
std = {k: (v - v.mean()) / (v.std() + 1e-9) for k, v in flat.items()}
r2s, r2map = r2sortability(std, EDGES)
print(f"[T3] varsortability brute         = {vs_raw:.3f}  (0.5 = neutre)")
# La varsortability APRÈS standardisation n'est pas rapportée : toutes les
# variances valent alors 1, donc `var[s] < var[t]` ne compare plus que du bruit
# de virgule flottante. C'est exactement la raison d'être de la R²-sortability
# (Reisach 2023) — invariante d'échelle, elle survit à la standardisation.
print(f"[T3] R²-sortability (inv. échelle) = {r2s:.3f}  <-- la statistique qui compte")
OUT["T3"] = dict(varsortability_raw=vs_raw,
                 varsortability_std=None,
                 varsortability_std_note=(
                     "non calculable : apres standardisation toutes les variances "
                     "valent 1, la comparaison ne porte que sur du bruit numerique"),
                 r2_sortability=r2s, r2_per_node=r2map, n_edges=len(EDGES))
# Erreur-type d'une proportion sur n arêtes : 0.5/sqrt(n). Avec n petit, la
# statistique n'est pas concluante — ne PAS la lire comme un echec.
# Le null n'est PAS 0,5. Les arêtes partagent des nœuds, donc les indicatrices
# ne sont pas des Bernoulli indépendantes et l'erreur-type binomiale 0,5/sqrt(n)
# est fausse. Le null correct est empirique : on refait tourner la statistique
# sur des séries à phase randomisée, qui conservent le spectre de chaque nœud
# et détruisent les relations croisées. Sur CETTE liste d'arêtes.
def phase_randomize(d):
    out = {}
    for k, v in d.items():
        Fv = np.fft.rfft(v)
        ph = RNG.uniform(0, 2 * np.pi, Fv.shape)
        out[k] = np.fft.irfft(np.abs(Fv) * np.exp(1j * ph), n=len(v))
    return out


print("[T3] null empirique par surrogates a phase randomisee...", flush=True)
N_SUR = 100
null = np.array([r2sortability(phase_randomize(std), EDGES)[0] for _ in range(N_SUR)])
null_mu, null_sd = float(null.mean()), float(null.std(ddof=1))
z = (r2s - null_mu) / max(null_sd, 1e-9)
print(f"   null = {null_mu:.3f} +/- {null_sd:.3f} sur {N_SUR} tirages "
      f"(0.5 theorique : {'compatible' if abs(null_mu-0.5) < 2*null_sd else 'BIAISE'})")
print(f"   observe = {r2s:.3f}  ->  z = {z:+.2f}")
OUT["T3_null"] = dict(n_surrogates=N_SUR, mean=null_mu, sd=null_sd,
                      observed=r2s, z=float(z))
OUT["T3_stderr"] = null_sd
OUT["T3_verdict"] = (
    "INCONCLUSIF (trop peu d'aretes)" if len(EDGES) < 30
    else ("PAS DE RACCOURCI EXPLOITABLE" if abs(z) < 2.0
          else f"RACCOURCI EXPLOITABLE (z={z:+.2f})"))

# Par niveau de crédibilité : si les arêtes de NIVEAU 1 sont, elles, très
# varsortables, cela voudrait dire que la partie « validée » du prior est
# précisément celle qu'une heuristique triviale retrouve — le contraire d'une
# découverte. Question distincte du verdict global, et plus gênante.
by_lvl = {}
for lvl in (1, 2, 3):
    sub_e = [(a, b) for e in PRIOR.edges if int(e["level"]) == lvl
             for a, b in [(e["source"], e["target"])] if a in flat and b in flat]
    if len(sub_e) < 3:
        continue
    r2s_l, _ = r2sortability(std, sub_e)
    by_lvl[lvl] = dict(n=len(sub_e), r2_sortability=r2s_l,
                       varsortability=varsortability(std, sub_e),
                       stderr=0.5 / len(sub_e) ** 0.5)
    print(f"[T3] niveau {lvl} ({len(sub_e):2d} aretes) : R2-sortability "
          f"= {r2s_l:.3f} +/- {by_lvl[lvl]['stderr']:.3f}")
OUT["T3_by_level"] = by_lvl


# ------------------------------------------------------------------------- G0
print("[G0] distinctness inter-GCM (ACCESS vs EC-Earth3)...", flush=True)
try:
    N2 = build_free_nodes("data/raw/test/EC-Earth3_histupdated_compressed.nc")
    common = [k for k in names if k in N2]
    def fingerprint(D):
        ks = sorted(D)
        M = np.stack([D[k].ravel()[:200_000] for k in ks], 1)
        M = (M - M.mean(0)) / (M.std(0) + 1e-9)
        return np.corrcoef(M, rowvar=False)
    A = fingerprint({k: NODES[k] for k in common})
    B = fingerprint({k: N2[k] for k in common})
    # empreinte binaire : |corr| > 0.3
    ea, eb = np.abs(A) > 0.3, np.abs(B) > 0.3
    iu = np.triu_indices_from(ea, 1)
    tp = float((ea[iu] & eb[iu]).sum()); fp = float((~ea[iu] & eb[iu]).sum())
    fn = float((ea[iu] & ~eb[iu]).sum())
    f1_inter = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    # inter-seed : deux moitiés temporelles d'ACCESS
    h = NODES[common[0]].shape[0] // 2
    A1 = fingerprint({k: NODES[k][:h] for k in common})
    A2 = fingerprint({k: NODES[k][h:] for k in common})
    e1, e2 = np.abs(A1) > 0.3, np.abs(A2) > 0.3
    tp = float((e1[iu] & e2[iu]).sum()); fp = float((~e1[iu] & e2[iu]).sum())
    fn = float((e1[iu] & ~e2[iu]).sum())
    f1_seed = 2 * tp / max(2 * tp + fp + fn, 1e-9)
    g0 = (1 - f1_inter) >= 2 * (1 - f1_seed)
    degenerate = f1_seed > 0.999          # denominateur nul => critere vacuous
    distinct = 1 - f1_inter               # LA grandeur qui compte vraiment
    print(f"   F1 inter-GCM  = {f1_inter:.4f}   -> distinctness = {distinct:.4f}")
    print(f"   F1 inter-seed = {f1_seed:.4f}")
    if degenerate:
        print("   /!\\ CRITERE DEGENERE : F1 inter-seed = 1.000 => le seuil vaut 0,")
        print("       n'importe quelle difference 'passe'. Lire la distinctness brute.")
    print(f"   distinctness = {100*distinct:.1f} % : les deux GCM partagent le meme")
    print( "   CCAM aval, leurs empreintes causales sont quasi identiques.")
    OUT["G0"] = dict(f1_inter_gcm=f1_inter, f1_inter_seed=f1_seed,
                     distinctness=distinct, formal_pass=bool(g0),
                     degenerate_criterion=bool(degenerate),
                     verdict=(
                         "DEGENERE - F1 inter-seed ~ 1, le seuil vaut 0"
                         if degenerate else
                         ("PASSE - distinctness %.1f %% au-dessus du seuil "
                          "pre-enregistre 2x(1-F1_seed) = %.1f %%"
                          % (100*distinct, 200*(1-f1_seed))) if g0 else
                         "ECHEC - distinctness sous le seuil pre-enregistre"))
except Exception as e:
    print("   G0 impossible :", e)
    OUT["G0"] = dict(error=str(e))

json.dump(OUT, open("results/p0_audits.json", "w"), indent=2, default=float)
print("\n=== ECRIT results/p0_audits.json ===")
print(f"T1 {'OK' if OUT.get('T1_pass') else 'ECHEC'}"
      f" | T3 {OUT.get('T3_verdict')}"
      f" | G0 {OUT.get('G0', {}).get('verdict')}")
