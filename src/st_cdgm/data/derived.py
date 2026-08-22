"""Les 13 nœuds LIBRES de V8, dérivés des 15 canaux LR bruts — source unique.

Pourquoi ce module existe
-------------------------
Trois raisons, dans l'ordre de gravité.

1. **Les nœuds dits « physiques » de la pile actuelle n'existent pas.**
   ``HeteroGraphBuilder`` déclare les types ``Q850``, ``W500``, ``IVT``,
   ``U850``, ``V850``, mais ``lr_grid_to_nodes`` est un simple *reshape* : tous
   reçoivent les MÊMES 15 canaux bruts, et tous portent le MÊME jeu d'arêtes
   (``spat_adj`` auto-boucle). Ils ne diffèrent que par des poids appris, donc
   ils sont interchangeables à une permutation de paramètres près. « IVT » est
   une étiquette, pas une quantité — aucune ligne ne la calcule.

2. **La formule était déjà fausse une fois.** L'IVT du catalogue était écrite
   ``somme q·||V||·dp`` — l'intégrale des normes — alors que l'IVT est la NORME
   de l'intégrale vectorielle. Une formule qui vit dans de la prose finit par
   diverger de son implémentation.

3. **Il en existait une copie privée** dans ``scripts/p0_audits.py``. Deux
   implémentations de la même physique, c'est une de trop.

Conventions
-----------
Entrée : dict ``{nom: champ [T, H, W]}`` pour les 15 bruts (``u_850`` … ``t_250``),
plus ``lat``/``lon`` en degrés pour les dérivées spatiales. Sortie : dict des
13 nœuds libres, mêmes formes, **en unités SI** — contrairement à la version
d'audit qui utilisait des gradients par maille (facteur d'échelle constant,
inoffensif pour des corrélations, faux pour de la physique).

Les 15 bruts restent disponibles : ils gardent leur rôle de DRIVER (§2.1). Ce
module ne produit que les nœuds du graphe de découverte.
"""
from __future__ import annotations

import numpy as np

__all__ = ["FREE_NODES", "RAW_VARS", "compute_free_nodes", "grid_spacing_m",
           "saturation_vapour_pressure", "relative_humidity",
           "free_nodes_dataset"]

#: Ordre canonique des 13 nœuds libres (§2.7). C'est CET ordre qui doit être
#: passé en ``prior_node_order`` à ``train_epoch_stage1``.
FREE_NODES = (
    "u850", "v850", "t850", "w500", "RH850", "RH500", "Gamma850_500",
    "zeta500", "normV250", "shear850_250", "IVT_u", "IVT_v", "MFC",
)

#: Les 15 canaux bruts attendus en entrée.
RAW_VARS = tuple(f"{v}_{lev}" for v in ("u", "v", "w", "q", "t")
                 for lev in (850, 500, 250))

G = 9.80665                 # m/s²
R_EARTH = 6_371_000.0       # m
RD_OVER_RV = 0.622          # rapport des masses molaires vapeur/air sec
R_DRY = 287.0               # J/(kg.K), constante des gaz pour l'air sec
P_850, P_500 = 85_000.0, 50_000.0   # Pa
DP_LAYER = 30_000.0         # Pa, épaisseur de la couche utilisée pour l'IVT


def saturation_vapour_pressure(t_kelvin: np.ndarray) -> np.ndarray:
    """Pression de vapeur saturante (Pa), formule de Tetens sur l'eau liquide."""
    tc = t_kelvin - 273.15
    return 610.94 * np.exp(17.625 * tc / (tc + 243.04))


def relative_humidity(q: np.ndarray, t: np.ndarray, p_pa: float) -> np.ndarray:
    """Humidité relative (fraction) depuis l'humidité spécifique.

    ``RH = q·p / (0.622·e_s(t))``. Bornée à 2 : au-delà, c'est du bruit
    numérique ou de la sursaturation non physique, pas un signal.
    """
    return np.clip(q * p_pa / (RD_OVER_RV * saturation_vapour_pressure(t)), 0.0, 2.0)


def grid_spacing_m(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, float]:
    """Pas de grille en mètres. Retourne ``(dx[lat], dy)``.

    ``dx`` dépend de la latitude (convergence des méridiens) : l'ignorer
    biaiserait systématiquement la vorticité et la convergence vers les pôles,
    et notre domaine couvre 25 degrés de latitude.
    """
    dlat = float(np.abs(np.diff(lat)).mean())
    dlon = float(np.abs(np.diff(lon)).mean())
    dy = dlat * np.pi / 180.0 * R_EARTH
    dx = dlon * np.pi / 180.0 * R_EARTH * np.cos(np.deg2rad(lat))
    return dx.astype(np.float64), float(dy)


def _ddx(field: np.ndarray, dx: np.ndarray) -> np.ndarray:
    """d/dx sur l'axe lon (dernier), avec le pas variable en latitude."""
    return np.gradient(field, axis=-1) / dx[None, :, None]


def _ddy(field: np.ndarray, dy: float) -> np.ndarray:
    """d/dy sur l'axe lat (avant-dernier)."""
    return np.gradient(field, axis=-2) / dy


def compute_free_nodes(raw: dict, lat: np.ndarray, lon: np.ndarray) -> dict:
    """15 canaux bruts -> les 13 nœuds libres, en SI.

    Parameters
    ----------
    raw :
        ``{nom: [T, H, W]}`` contenant au moins ``RAW_VARS``.
    lat, lon :
        coordonnées en degrés, longueurs H et W.

    Returns
    -------
    dict ``{nom: [T, H, W]}`` dans l'ordre de :data:`FREE_NODES`.
    """
    missing = [v for v in RAW_VARS if v not in raw]
    if missing:
        raise ValueError(f"Canaux bruts manquants : {missing}")
    h, w = len(lat), len(lon)
    for v in RAW_VARS:
        if raw[v].shape[-2:] != (h, w):
            raise ValueError(
                f"{v} a la forme {raw[v].shape[-2:]}, attendu ({h}, {w}) "
                f"d'après lat/lon.")

    dx, dy = grid_spacing_m(np.asarray(lat, float), np.asarray(lon, float))
    f = {k: np.asarray(raw[k], np.float64) for k in RAW_VARS}
    out: dict = {}

    # --- repris tels quels (bruts ET libres, cf. §2.1) ---------------------
    out["u850"] = f["u_850"]
    out["v850"] = f["v_850"]
    out["t850"] = f["t_850"]
    out["w500"] = f["w_500"]

    # --- thermodynamique ---------------------------------------------------
    # Coordonnées (RH, t, Gamma) et non (q, t, q_sat) : q_sat est une fonction
    # EXACTE de t, donc le triplet brut serait déterministe dans le graphe et
    # violerait l'audit T1 (correction fatale n°1 du conseil).
    out["RH850"] = relative_humidity(f["q_850"], f["t_850"], 85_000.0)
    out["RH500"] = relative_humidity(f["q_500"], f["t_500"], 50_000.0)
    # Gradient thermique vertical, K/km. L'épaisseur de la couche 850-500 hPa
    # vient de l'équation hypsométrique avec la température moyenne RÉELLE de
    # la couche, pas d'une constante : dz = (Rd.T_moy/g).ln(p1/p2). Une valeur
    # figée de 3,5 km — l'erreur precedente — surestimait Gamma de ~18 % (elle
    # donnait 6,73 K/km la ou l'atmosphere standard donne 4,12 km d'epaisseur
    # et donc 5,7 K/km, valeur usuelle pour cette couche aux moyennes latitudes).
    t_mean = 0.5 * (f["t_850"] + f["t_500"])
    dz_km = (R_DRY * t_mean / G) * np.log(P_850 / P_500) / 1000.0
    out["Gamma850_500"] = (f["t_850"] - f["t_500"]) / dz_km

    # --- cinématique -------------------------------------------------------
    # Vorticité relative, s^-1. zeta = dv/dx - du/dy.
    out["zeta500"] = _ddx(f["v_500"], dx) - _ddy(f["u_500"], dy)
    out["normV250"] = np.sqrt(f["u_250"] ** 2 + f["v_250"] ** 2)
    out["shear850_250"] = np.sqrt((f["u_250"] - f["u_850"]) ** 2
                                  + (f["v_250"] - f["v_850"]) ** 2)

    # --- flux d'humidité ---------------------------------------------------
    # IVT VECTORIEL. La norme se prend APRÈS l'intégrale, jamais avant :
    # ||somme q·V·dp||  et  somme q·||V||·dp  diffèrent dès que le vent tourne
    # avec l'altitude, ce qui est le cas général. Voir le self-check.
    out["IVT_u"] = (f["q_850"] * f["u_850"] + f["q_500"] * f["u_500"]) * DP_LAYER / G
    out["IVT_v"] = (f["q_850"] * f["v_850"] + f["q_500"] * f["v_500"]) * DP_LAYER / G
    # Convergence de flux d'humidité au niveau 850 : MFC = -div(q·V), en s^-1
    # (par unité d'humidité spécifique).
    qu, qv = f["q_850"] * f["u_850"], f["q_850"] * f["v_850"]
    out["MFC"] = -(_ddx(qu, dx) + _ddy(qv, dy))

    assert set(out) == set(FREE_NODES), "l'ensemble produit ne correspond pas à FREE_NODES"
    return {k: out[k].astype(np.float32) for k in FREE_NODES}


def free_nodes_dataset(ds, lat_name: str = "lat", lon_name: str = "lon"):
    """``xr.Dataset`` des 15 bruts -> ``xr.Dataset`` des 13 nœuds libres.

    Point de couture avec le pipeline : la dérivation est faite **une fois**
    sur le jeu complet, pas par fenêtre, et le résultat remplace simplement
    les variables LR. Tout l'aval — ``_dataset_to_numpy``, ``lr_grid_to_nodes``,
    le ``driver`` du RCN — voit alors 13 canaux dans l'ordre de
    :data:`FREE_NODES`, qui est exactement ce qu'attend le routage diagonal
    V7-M2 avec sa carte identité par défaut.
    """
    import xarray as xr

    raw = {v: ds[v].values for v in RAW_VARS if v in ds.data_vars}
    lat = np.asarray(ds[lat_name].values, float)
    lon = np.asarray(ds[lon_name].values, float)
    nodes = compute_free_nodes(raw, lat, lon)

    template = ds[RAW_VARS[0]]
    out = xr.Dataset(
        {name: (template.dims, field) for name, field in nodes.items()},
        coords=template.coords,
    )
    out.attrs = dict(ds.attrs)
    out.attrs["derived_from"] = "st_cdgm.data.derived.compute_free_nodes"
    for name in FREE_NODES:
        out[name].attrs["long_name"] = name
    return out


if __name__ == "__main__":
    # Self-check sur une atmosphere synthetique dont on connait les reponses.
    rng = np.random.default_rng(0)
    T, H, W = 8, 23, 26
    lat = np.linspace(-59.4, -26.4, H)
    lon = np.linspace(150.6, 188.1, W)

    raw = {}
    for lev, (uu, vv, tt, qq) in {
        850: (8.0, 2.0, 285.0, 6e-3),
        500: (18.0, 1.0, 258.0, 1.2e-3),
        250: (35.0, 0.5, 225.0, 6e-5),
    }.items():
        raw[f"u_{lev}"] = uu + rng.normal(0, 2, (T, H, W))
        raw[f"v_{lev}"] = vv + rng.normal(0, 2, (T, H, W))
        raw[f"t_{lev}"] = tt + rng.normal(0, 1, (T, H, W))
        raw[f"q_{lev}"] = np.abs(qq + rng.normal(0, qq * 0.2, (T, H, W)))
        raw[f"w_{lev}"] = rng.normal(0, 0.05, (T, H, W))

    n = compute_free_nodes(raw, lat, lon)
    assert list(n) == list(FREE_NODES)
    print(f"{len(n)} noeuds produits, formes {n['MFC'].shape}")

    # --- Ordres de grandeur : une formule fausse se voit ici avant tout ----
    checks = {
        "RH850":        (0.05, 1.20, ""),
        "Gamma850_500": (4.5,  7.5,  "K/km"),
        "zeta500":      (1e-6, 1e-3, "s^-1 (|.| median)"),
        "normV250":     (10.0, 80.0, "m/s"),
        "IVT_u":        (10.0, 1500., "kg/m/s (|.| median)"),
    }
    for k, (lo, hi, unit) in checks.items():
        v = np.abs(n[k]) if k in ("zeta500", "IVT_u") else n[k]
        med = float(np.median(v))
        print(f"  {k:14s} mediane = {med:10.4g} {unit}")
        assert lo <= med <= hi, f"{k} hors plage physique [{lo}, {hi}] : {med:g}"

    # --- LE test de non-regression : l'IVT est vectoriel ------------------
    # ||somme q·V·dp||  !=  somme q·||V||·dp  des que le vent tourne avec
    # l'altitude. C'est l'erreur exacte qui figurait dans le catalogue.
    vec = np.hypot(n["IVT_u"], n["IVT_v"])
    scal = (raw["q_850"] * np.hypot(raw["u_850"], raw["v_850"])
            + raw["q_500"] * np.hypot(raw["u_500"], raw["v_500"])) * DP_LAYER / G
    # Tolerance RELATIVE : la sortie est en float32 (~1e-7 relatif) alors que
    # scal est calcule en float64. Sur des magnitudes de ~200 kg/m/s, un seuil
    # absolu de 1e-6 serait sous le bruit d'arrondi.
    assert (vec <= scal * (1 + 1e-5)).all(), "l'inegalite triangulaire est violee"
    gap = float(np.median((scal - vec) / vec))
    print(f"  IVT scalaire surestime le vectoriel de {100*gap:.1f} % (median)")
    assert gap > 1e-3, ("le test ne discrimine pas : choisir un vent qui tourne "
                        "davantage avec l'altitude")

    # --- dx doit varier avec la latitude ----------------------------------
    dx, dy = grid_spacing_m(lat, lon)
    assert dx.max() / dx.min() > 1.3, (
        "dx constant : la convergence des meridiens est ignoree, ce qui biaise "
        "zeta et MFC sur 25 degres de latitude")
    print(f"  dx = {dx.min()/1000:.0f}..{dx.max()/1000:.0f} km  |  dy = {dy/1000:.0f} km")

    # --- Non-degenerescence (T1) : aucun noeud ne doit etre une fonction ---
    # deterministe d'un autre, sinon le graphe de decouverte est mal pose.
    import itertools
    flat = {k: v.ravel() for k, v in n.items()}
    worst = max(
        ((a, b, float(np.corrcoef(flat[a], flat[b])[0, 1] ** 2))
         for a, b in itertools.combinations(FREE_NODES, 2)),
        key=lambda t: t[2])
    print(f"  R2 max entre paires : {worst[2]:.3f} ({worst[0]} ~ {worst[1]})")
    assert worst[2] < 0.95, f"noeud deterministe in-graph : {worst}"
    print("OK — 13 noeuds libres, unites SI, IVT vectoriel, aucune degenerescence.")
