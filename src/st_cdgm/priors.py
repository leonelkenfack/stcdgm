"""Chargement du prior par arête à 3 niveaux (composant C7 de V8).

Le fichier `config/dag_prior_v8_c7.yaml` porte, par arête : niveau de
crédibilité (1/2/3), lag, provenance de l'orientation (table exigée par T7)
et justification physique. Ce module le relit, vérifie ce qui doit l'être, et
en produit les matrices que la loss DAGMA consomme.

Ce qui est vérifié ici plutôt que découvert en cours de run :
* le prior est **acyclique** — un cycle rendrait h_DAGMA(A_prior) > 0 et le
  terme de prior tirerait A vers une cible interdite, en silence ;
* tous les nœuds cités appartiennent bien à la liste des 13 libres ;
* aucune paire (source, cible) en double, qui ferait compter une arête deux
  fois dans la pénalité.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

__all__ = ["EdgePrior", "load_edge_prior", "DEFAULT_PRIOR_PATH",
           "LEVEL_WEIGHT", "DEFAULT_LEVEL_FLOORS", "LAMBDA_PRIOR_START",
           "anneal_lambda_prior"]

DEFAULT_PRIOR_PATH = Path(__file__).resolve().parents[2] / "config" / "dag_prior_v8_c7.yaml"

# Poids par niveau de crédibilité. Rapports 1 / 0,5 / 0,2 : une arête
# spéculative pèse cinq fois moins qu'une arête de cascade QG validée.
# L'annealing V7-M5 multiplie ces poids, il ne les remplace pas.
LEVEL_WEIGHT = {1: 1.0, 2: 0.5, 3: 0.2}

# Annealing V7-M5, appliqué PAR NIVEAU (c'est tout l'objet de C7 : l'appliquer
# uniformément revient à traiter une arête spéculative comme une cascade QG).
# Départ commun 0,40 — le prior guide fortement au début, quand A_dag est
# encore aléatoire et vulnérable au collapse. Puis les planchers divergent :
# le niveau 3 se relâche presque complètement, de sorte que les données
# PUISSENT le rejeter — un rejet systématique est un signal de découverte
# négatif à rapporter (C7), pas un échec à masquer.
LAMBDA_PRIOR_START = 0.40
DEFAULT_LEVEL_FLOORS = {1: 0.08, 2: 0.04, 3: 0.02}


def anneal_lambda_prior(epoch: int, total_epochs: int, floor: float,
                        start: float = LAMBDA_PRIOR_START) -> float:
    """Décroissance cosinus de ``start`` vers ``floor`` (même forme que
    ``lambda_l1_cosine_anneal``, pour ne pas multiplier les profils)."""
    if total_epochs <= 1:
        return float(floor)
    t = min(max(epoch / (total_epochs - 1), 0.0), 1.0)
    return float(floor + 0.5 * (start - floor) * (1.0 + np.cos(np.pi * t)))


class EdgePrior:
    """Prior C7 chargé : noms, arêtes, matrices par niveau."""

    def __init__(self, nodes: list[str], edges: list[dict]) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.index = {n: i for i, n in enumerate(self.nodes)}

    # ------------------------------------------------------------------

    def _order(self, node_order: list[str] | None) -> dict[str, int]:
        """Index des nœuds dans l'ordre demandé, ou celui du fichier.

        L'ordre du RCN vient de ``CONFIG.encoder.metapaths`` et n'a AUCUNE
        raison de coïncider avec l'ordre du YAML. Sans cette réconciliation
        explicite, la matrice serait transposée-permutée et le prior tirerait
        A_dag vers des arêtes arbitraires — sans lever la moindre erreur.
        """
        if node_order is None:
            return self.index
        missing = set(self.nodes) - set(node_order)
        extra = set(node_order) - set(self.nodes)
        if missing or extra:
            raise ValueError(
                "L'ordre de nœuds fourni ne correspond pas au prior. "
                f"Absents de l'ordre : {sorted(missing)}. "
                f"Inconnus du prior : {sorted(extra)}. "
                "Le prior ne porte que sur les nœuds LIBRES du graphe de "
                "découverte ; les câblés et les CTX n'y ont pas leur place.")
        return {n: i for i, n in enumerate(node_order)}

    def matrix(self, weighted: bool = True, lag: int | None = None,
               node_order: list[str] | None = None) -> np.ndarray:
        """Matrice ``[q, q]`` du prior. ``A[i, j] != 0`` <=> i -> j.

        ``weighted`` applique ``LEVEL_WEIGHT`` ; sinon binaire.
        ``lag`` filtre sur le décalage (None = tous).
        ``node_order`` réindexe sur l'ordre des variables du RCN.
        """
        idx = self._order(node_order)
        q = len(idx)
        A = np.zeros((q, q), dtype=np.float32)
        for e in self.edges:
            if lag is not None and int(e["lag"]) != lag:
                continue
            w = LEVEL_WEIGHT[int(e["level"])] if weighted else 1.0
            A[idx[e["source"]], idx[e["target"]]] = w
        return A

    def level_masks(self, node_order: list[str] | None = None
                    ) -> dict[int, np.ndarray]:
        """Masques booléens ``[q, q]``, un par niveau de crédibilité.

        Les entrées HORS prior ne sont dans aucun masque : elles ne sont donc
        tirées vers rien. C'est délibéré — la contrainte n°3 de V8 est la
        **découverte au-delà du prior**, et une MSE vers la matrice complète
        pénaliserait activement toute arête que le prior ignore. La parcimonie
        de ces entrées reste assurée par le terme L1, pas par le prior.
        """
        idx = self._order(node_order)
        q = len(idx)
        out = {lvl: np.zeros((q, q), dtype=bool) for lvl in LEVEL_WEIGHT}
        for e in self.edges:
            out[int(e["level"])][idx[e["source"]], idx[e["target"]]] = True
        return out

    def edge_list(self, lag: int | None = None) -> list[tuple[str, str]]:
        return [(e["source"], e["target"]) for e in self.edges
                if lag is None or int(e["lag"]) == lag]

    def by_level(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for e in self.edges:
            out[int(e["level"])] = out.get(int(e["level"]), 0) + 1
        return dict(sorted(out.items()))

    def by_orient(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.edges:
            out[e["orient"]] = out.get(e["orient"], 0) + 1
        return dict(sorted(out.items()))

    def __len__(self) -> int:
        return len(self.edges)

    def __repr__(self) -> str:
        return (f"EdgePrior({len(self.nodes)} nœuds, {len(self.edges)} arêtes, "
                f"niveaux {self.by_level()})")


def _assert_acyclic(nodes: list[str], edges: list[dict]) -> None:
    """Tri topologique de Kahn. Lève en nommant le cycle résiduel."""
    idx = {n: i for i, n in enumerate(nodes)}
    q = len(nodes)
    adj: list[list[int]] = [[] for _ in range(q)]
    indeg = [0] * q
    for e in edges:
        a, b = idx[e["source"]], idx[e["target"]]
        adj[a].append(b)
        indeg[b] += 1
    stack = [i for i in range(q) if indeg[i] == 0]
    seen = 0
    while stack:
        i = stack.pop()
        seen += 1
        for j in adj[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                stack.append(j)
    if seen != q:
        residual = [nodes[i] for i in range(q) if indeg[i] > 0]
        raise ValueError(
            "Le prior C7 contient un CYCLE. Nœuds impliqués : "
            f"{residual}. Un prior cyclique donnerait h_DAGMA(A_prior) > 0 et "
            "tirerait A vers une cible que la contrainte d'acyclicité interdit."
        )


def load_edge_prior(path: str | Path = DEFAULT_PRIOR_PATH) -> EdgePrior:
    """Relit le YAML du prior, valide, renvoie un :class:`EdgePrior`."""
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)

    nodes: list[str] = list(spec["nodes"])
    edges: list[dict] = list(spec["edges"])
    known = set(nodes)

    seen: set[tuple[str, str]] = set()
    for e in edges:
        for side in ("source", "target"):
            if e[side] not in known:
                raise ValueError(
                    f"Arête {e['source']} -> {e['target']} : nœud {e[side]!r} "
                    f"absent de la liste des libres. Le prior ne doit porter "
                    f"que sur le graphe de découverte.")
        if e["source"] == e["target"]:
            raise ValueError(f"Boucle sur {e['source']} : interdite dans un DAG.")
        if int(e["level"]) not in LEVEL_WEIGHT:
            raise ValueError(f"Niveau {e['level']} inconnu (attendu 1, 2 ou 3).")
        key = (e["source"], e["target"])
        if key in seen:
            raise ValueError(
                f"Arête {key[0]} -> {key[1]} déclarée deux fois : elle serait "
                f"comptée deux fois dans la pénalité.")
        seen.add(key)

    _assert_acyclic(nodes, edges)
    return EdgePrior(nodes, edges)


if __name__ == "__main__":
    p = load_edge_prior()
    print(p)
    print("  orientation :", p.by_orient())
    A = p.matrix()
    print(f"  matrice {A.shape}, {int((A != 0).sum())} arêtes non nulles, "
          f"poids total {A.sum():.2f}")
    print(f"  lag 0 : {len(p.edge_list(lag=0))} | lag 1 : {len(p.edge_list(lag=1))}")

    # Une arête orientée par PRECEDENCE doit avoir un lag > 0 : c'est la
    # definition. L'inverse trahirait une orientation revendiquee a tort.
    for e in p.edges:
        if e["orient"] == "precedence":
            assert int(e["lag"]) > 0, (
                f"{e['source']} -> {e['target']} dit s'orienter par precedence "
                f"mais a lag=0 : l'orientation n'est alors PAS justifiee.")

    # w500 est diagnostique (Climat E6) : il ne peut pas causer une variable
    # dynamique. Seules des consequences thermodynamiques sont admises.
    DYNAMIC = {"u850", "v850", "zeta500", "normV250", "shear850_250"}
    for e in p.edges:
        assert not (e["source"] == "w500" and e["target"] in DYNAMIC), (
            f"w500 -> {e['target']} : w est diagnostique en GCM hydrostatique, "
            f"il ne peut pas forcer une variable dynamique.")

    # L'erreur-type d'une proportion sur n aretes borne ce que T3 peut trancher.
    n = len(p)
    se = 0.5 / n ** 0.5
    print(f"  erreur-type T3 = {se:.3f} sur {n} aretes "
          f"(il fallait < 0.10 pour distinguer 0.5 de 0.7)")
    assert se < 0.10, f"prior trop petit : T3 resterait inconclusif ({se:.3f})"

    # --- Reindexation : le piege silencieux ---------------------------------
    # L'ordre du RCN vient de CONFIG.encoder.metapaths. Une permutation mal
    # gérée donnerait une matrice plausible mais fausse, sans lever d'erreur.
    shuffled = list(reversed(p.nodes))
    A2 = p.matrix(node_order=shuffled)
    for e in p.edges:
        i, j = shuffled.index(e["source"]), shuffled.index(e["target"])
        assert A2[i, j] == LEVEL_WEIGHT[int(e["level"])], (
            f"reindexation cassee sur {e['source']} -> {e['target']}")
    assert int((A2 != 0).sum()) == len(p), "arete perdue a la reindexation"
    try:
        p.matrix(node_order=p.nodes[:-1] + ["inconnu"])
        raise SystemExit("un ordre incoherent aurait du lever ValueError")
    except ValueError:
        pass
    print("OK — reindexation sur l'ordre du RCN verifiee (et ordre invalide rejete)")

    # --- Masques par niveau : disjoints, et SANS les entrees hors prior -----
    masks = p.level_masks()
    tot = sum(int(m.sum()) for m in masks.values())
    assert tot == len(p), f"masques : {tot} entrees pour {len(p)} aretes"
    stacked = sum(m.astype(int) for m in masks.values())
    assert stacked.max() <= 1, "une arete apparait dans deux niveaux"
    q = len(p.nodes)
    assert tot < q * q, "les masques couvrent tout : la decouverte hors prior serait penalisee"
    print(f"OK — masques disjoints, {tot}/{q*q} entrees contraintes "
          f"({100*tot/(q*q):.0f} %) — le reste est libre de decouvrir")

    # --- Annealing par niveau : c'est TOUT l'objet de C7 --------------------
    N = 30
    print("  annealing (epoch 0 -> 29) :")
    for lvl, floor in sorted(DEFAULT_LEVEL_FLOORS.items()):
        a = anneal_lambda_prior(0, N, floor)
        b = anneal_lambda_prior(N - 1, N, floor)
        print(f"    niveau {lvl} : {a:.3f} -> {b:.3f}")
        assert abs(a - LAMBDA_PRIOR_START) < 1e-6, "l'annealing doit partir de 0.40"
        assert abs(b - floor) < 1e-6, "l'annealing doit atteindre son plancher"
    # Le niveau 3 doit finir STRICTEMENT plus bas que le niveau 1 : sinon
    # l'annealing est uniforme et C7 n'apporte rien.
    end3 = anneal_lambda_prior(N - 1, N, DEFAULT_LEVEL_FLOORS[3])
    end1 = anneal_lambda_prior(N - 1, N, DEFAULT_LEVEL_FLOORS[1])
    assert end3 < end1, (
        "le niveau speculatif doit se relacher plus que la cascade validee, "
        "sinon l'annealing est uniforme et C7 est decoratif")
    print(f"    ratio final L1/L3 = {end1/end3:.1f}x")
    print("OK — prior C7 acyclique, oriente, reindexable et anneale par niveau.")
