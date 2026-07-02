"""Physical-prior mask for the learned causal DAG (Phase D, 2026-06-09).

Encode le couplage vertical descendant attendu par la dynamique
quasi-géostrophique (QG) atmosphérique sur les 6 nœuds du RCN :
- forçage upper-level (250 hPa) descend vers mid-troposphère (500 hPa)
- mid-troposphère force la basse couche (850 hPa)
- basse couche structure la pression de surface (SP_HR)
- les méta-paths apprises alimentent leur nœud cible

Cette structure n'est PAS une découverte causale stricte au sens Pearl —
l'identifiabilité MEC l'interdit depuis données obs-only (voir §11.4 de
``architecture_journey.md``). C'est un **prior structurel physiquement
inspiré** qu'on injecte pour briser la zone plate où DAGMA seul collapse
``A_dag`` à des magnitudes uniformes (V5-mini Q_phys = 0.40).

Combiné avec :
- CASTLE-style anchor (`models/causal_rcn.py:CASTLEAnchor`) qui force
  l'ancrage prédictif des arêtes,
- DAGMA acyclicité (toujours en place),
- L1 sparsity annealing (cosine 0.10 → 0.01, Phase B),

ce masque vise à porter Q_phys de 0.40 → 0.65-0.80 sans dégrader les
métriques globales.

Cible pathologie #2 (Q_phys mediocre) selon §11.2 de architecture_journey.md.

Usage typique
-------------
.. code-block:: python

    from st_cdgm.training.physics_prior import build_physical_mask, physical_prior_loss

    G_phys = build_physical_mask(num_vars=6)   # une fois au setup
    # Dans la boucle de training :
    prior_loss = physical_prior_loss(A_dag, G_phys, alpha=0.20)
    # Passer via stage1_compute_loss(dag_prior_loss=prior_loss, lambda_dag_prior=0.05, ...)

References
----------
- Holton & Hakim 2013, *Introduction to Dynamic Meteorology* (5e éd.), Ch. 6
  pour le couplage QG descendant.
- Karniadakis et al. 2021, *Physics-informed machine learning*, Nat Rev Phys 3,
  pour le cadre PINN d'injection de priors physiques durs.
- Pearl 2009, *Causality* — justifie la position "prior structurel" plutôt que
  "découverte causale".
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor


# Ordre canonique des variables encodées par le RCN. **DOIT** matcher
# l'ordre `var_labels` utilisé dans les figures de Phase 8 (cf
# `st_cdgm_v5_evaluation.ipynb` Cell 10) et dans la construction du DAG
# de Phase 11.
VAR_LABELS: List[str] = [
    "GP850_spat",     # géopotentiel 850 hPa (basse couche)
    "GP850->GP500",   # méta-path 850 -> 500
    "GP500_spat",     # géopotentiel 500 hPa (mid-troposphère)
    "GP500->GP250",   # méta-path 500 -> 250
    "GP250_spat",     # géopotentiel 250 hPa (haute troposphère, tropopause)
    "SP_HR",          # pression au niveau de la mer haute résolution
]


# Arêtes physiquement attendues : tuples (source, cible, signe).
# Convention : `G_phys[i, j] = sign(arête i → j)`.
#
# Logique :
# - Descente verticale (QG) : 250 → 500 → 850 → SP_HR (signe +).
# - Méta-paths alimentent leur nœud cible (signe +).
#
# Note : on n'encode PAS les arêtes inverses (e.g. SP_HR → GP850_spat)
# qui correspondraient à un régime convectif dominant. C'est un choix
# délibéré : la branche `two-stage-causal` cible le régime synoptique
# qui domine la précipitation orographique NZ (atmospheric rivers).
EXPECTED_EDGES: List[Tuple[str, str, int]] = [
    ("GP250_spat", "GP500_spat", +1),
    ("GP500_spat", "GP850_spat", +1),
    ("GP850_spat", "SP_HR", +1),
    ("GP850->GP500", "GP500_spat", +1),
    ("GP500->GP250", "GP250_spat", +1),
]


# ---------------------------------------------------------------------------
# Extension 9-node (option C, Phase 0) — OPT-IN, n'altère pas les défauts 6-node.
#
# Ajoute la "chaîne humide" (décomposition Held-Soden) au-dessus de la chaîne
# sèche QG : la dynamique synoptique (GP*) force humidité Q850 et ascendance
# W500, qui alimentent le transport de vapeur IVT, lequel — avec W500 — ferme
# la chaîne vers la précipitation de surface SP_HR.
#
# Convention d'ordre (DOIT matcher config.encoder.metapaths + config.loss.dag_prior) :
#   0 GP850_spat  1 GP850->GP500  2 GP500_spat  3 GP500->GP250  4 GP250_spat
#   5 Q850        6 W500          7 IVT          8 SP_HR (static, toujours en dernier)
#
# Validé apprenable par smoke_9node_extended_dag.ipynb (Q_phys_cont=0.682,
# skeleton F1=1.0, n_extra=0). Voir path_c_plus/audit/PHASE0_SPEC_9NODE.md.
# ---------------------------------------------------------------------------
VAR_LABELS_9NODE: List[str] = [
    "GP850_spat",
    "GP850->GP500",
    "GP500_spat",
    "GP500->GP250",
    "GP250_spat",
    "Q850",            # humidité spécifique 850 hPa (basse couche humide)
    "W500",            # vitesse verticale 500 hPa (ascendance)
    "IVT",             # transport intégré de vapeur (proxy 3 niveaux)
    "SP_HR",           # pression de surface HR (static, repoussé à l'index 8)
]


EXPECTED_EDGES_9NODE: List[Tuple[str, str, int]] = [
    # --- chaîne sèche QG (identique au 6-node) ---
    ("GP250_spat", "GP500_spat", +1),
    ("GP500_spat", "GP850_spat", +1),
    ("GP850_spat", "SP_HR", +1),
    ("GP850->GP500", "GP500_spat", +1),
    ("GP500->GP250", "GP250_spat", +1),
    # --- chaîne humide (Held-Soden) ---
    ("GP850_spat", "Q850", +1),     # basse couche synoptique -> humidité
    ("GP500_spat", "W500", +1),     # mid-troposphère -> ascendance
    ("Q850", "IVT", +1),            # humidité -> transport de vapeur
    ("W500", "SP_HR", +1),          # ascendance -> précip de surface
    ("IVT", "SP_HR", +1),           # transport de vapeur -> précip de surface
]


# ---------------------------------------------------------------------------
# Extension 11-node (V6, audit indépendant 2026-06-30) — OPT-IN.
#
# Ajoute les deux composantes du vent bas-niveau U850, V850 au-dessus de la
# chaîne humide 9-node. Recommandation Climat (verbatim) : "u850/v850 →
# précipitation orographique locale devrait surperformer structurellement sur
# les queues". Les deux nœuds forcent (a) le transport de vapeur IVT (advection
# horizontale de l'humidité) et (b) directement la précipitation orographique
# SP_HR (soulèvement forcé sur les Alpes du Sud).
#
# Convention d'ordre (SP_HR TOUJOURS en dernier, comme 9-node) :
#   0 GP850_spat  1 GP850->GP500  2 GP500_spat  3 GP500->GP250  4 GP250_spat
#   5 Q850        6 W500          7 IVT          8 U850  9 V850  10 SP_HR
#
# G_phys 11×11 = 10 arêtes 9-node + 4 arêtes vent = 14 arêtes physiques.
# La sparsité reste faible (14 / 121 ≈ 11.6 %).
# ---------------------------------------------------------------------------
VAR_LABELS_V6: List[str] = [
    "GP850_spat",
    "GP850->GP500",
    "GP500_spat",
    "GP500->GP250",
    "GP250_spat",
    "Q850",
    "W500",
    "IVT",
    "U850",            # vent zonal 850 hPa (advection zonale)
    "V850",            # vent méridien 850 hPa (advection méridienne, ARs N→S)
    "SP_HR",           # pression de surface HR (static, repoussé à l'index 10)
]


EXPECTED_EDGES_V6: List[Tuple[str, str, int]] = [
    # --- chaîne sèche QG ---
    ("GP250_spat", "GP500_spat", +1),
    ("GP500_spat", "GP850_spat", +1),
    ("GP850_spat", "SP_HR", +1),
    ("GP850->GP500", "GP500_spat", +1),
    ("GP500->GP250", "GP250_spat", +1),
    # --- chaîne humide (Held-Soden) ---
    ("GP850_spat", "Q850", +1),
    ("GP500_spat", "W500", +1),
    ("Q850", "IVT", +1),
    ("W500", "SP_HR", +1),
    ("IVT", "SP_HR", +1),
    # --- chaîne vent bas-niveau (V6, audit Climat — vérif code 2026-06-30) ---
    ("U850", "IVT", +1),        # vent zonal -> transport zonal de vapeur
    ("V850", "IVT", -1),        # ARs NZ = flux NW : humide = northerly = v<0 (HS)
    ("U850", "SP_HR", +1),      # westerlies dominants + West Coast domine le total
    ("V850", "SP_HR", -1),      # signe climatologique : v<0 (northerly) = humide,
                                # v>0 (southerly) = air froid sec. Audit Climat :
                                # "un prior de signe faux est pire que pas de prior".
]


def build_physical_mask(
    num_vars: int = 6,
    var_labels: Sequence[str] = None,
    expected_edges: Sequence[Tuple[str, str, int]] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> Tensor:
    """Construit la matrice G_phys [num_vars, num_vars] signée.

    `G_phys[i, j]` vaut le signe attendu de l'arête `i → j` (i.e. l'influence
    causale de la variable `i` sur la variable `j`), ou 0 si non spécifiée.

    Parameters
    ----------
    num_vars : int
        Dimension de la matrice. Doit matcher le ``num_vars`` du ``RCNCell``.
    var_labels : Sequence[str], optional
        Override des labels par défaut. Doit avoir longueur ``num_vars``.
    expected_edges : Sequence[Tuple[str, str, int]], optional
        Override des arêtes attendues. Format : ``(src_label, tgt_label, sign)``.
        Arêtes avec labels absents de ``var_labels`` sont ignorées silencieusement.
    dtype : torch.dtype
        Type de sortie. Défaut float32.
    device : torch.device, optional
        Device de sortie. Défaut CPU.

    Returns
    -------
    Tensor [num_vars, num_vars], signé {-1, 0, +1}.

    Examples
    --------
    >>> G = build_physical_mask(num_vars=6)
    >>> G.shape, G.sum().item(), (G != 0).sum().item()
    (torch.Size([6, 6]), 5.0, 5)
    """
    labels = list(var_labels) if var_labels is not None else VAR_LABELS
    edges = list(expected_edges) if expected_edges is not None else EXPECTED_EDGES

    if len(labels) != num_vars:
        raise ValueError(
            f"len(var_labels)={len(labels)} ne matche pas num_vars={num_vars}"
        )

    label_to_idx = {label: i for i, label in enumerate(labels)}
    G = torch.zeros(num_vars, num_vars, dtype=dtype, device=device)
    for src, tgt, sign in edges:
        i = label_to_idx.get(src)
        j = label_to_idx.get(tgt)
        if i is None or j is None:
            # Arête avec label inconnu : on skip silencieusement (compat
            # avec des configs où certains nœuds n'existent pas)
            continue
        G[i, j] = float(sign)
    return G


def physical_prior_loss(
    A_dag: Tensor,
    G_phys: Tensor,
    alpha: float = 0.20,
    mask_diagonal: bool = True,
    normalize: bool = False,
) -> Tensor:
    """Pénalise les écarts entre ``A_dag`` et ``α · G_phys``.

    Le facteur `α` représente la **magnitude attendue** d'une arête physique
    dans l'espace continu de `A_dag`. Sans α, le terme pénaliserait `A_dag`
    pour ne pas atteindre exactement {-1, 0, +1}, ce qui serait trop rigide.
    Avec `α = 0.20`, on accepte des magnitudes raisonnables qui pourraient
    être différenciées par CASTLE et la dynamique de training.

    I1 fix (audit math prof): the previous version divided by ``off_diag.sum()``
    (= N(N-1) = 30 for 6-node graph) which silently reduced the effective
    ``lambda_dag_prior`` by 30×. KKT analysis showed that with the
    normalization, ``lambda_phys = 0.05`` (V5-mini default) produced a
    physical-prior gradient ~82× weaker than the L1 sparsity gradient,
    making the prior signal ineffective. This fix changes the default to
    ``normalize=False`` (sum-of-squared-errors) which gives the prior force
    proportional to ``lambda_dag_prior``. Old callers can set
    ``normalize=True`` for backward compat with the original V5-mini hyperparams.

    Recommended Path C+ values with normalize=False:
        4-node: lambda_dag_prior=0.40, alpha=0.25 (KKT: lambda_l1 < lambda_phys/6)
        6-node: lambda_dag_prior=0.40, alpha=0.20 (KKT: lambda_l1 < lambda_phys/15)

    Parameters
    ----------
    A_dag : Tensor [num_vars, num_vars]
        Matrice DAG actuelle (post-masque diagonal). Doit avoir
        ``requires_grad=True`` pour que le prior soit appris.
    G_phys : Tensor [num_vars, num_vars]
        Masque physique signé, typiquement ``build_physical_mask()``.
    alpha : float
        Magnitude attendue des arêtes physiques. Défaut 0.20.
        Avec V5-mini ayant ``||A_dag||`` per-entry ≈ 0.18, α=0.20 est
        cohérent avec l'échelle observée.
    mask_diagonal : bool
        Si True, force la diagonale de G_phys à 0 et ne pénalise pas la
        diagonale d'A_dag (pas de self-loops dans un DAG).

    Returns
    -------
    Tensor scalaire — MSE entre A_dag (off-diagonal) et α · G_phys.

    Notes
    -----
    - À utiliser via le kwarg ``dag_prior_loss`` de ``stage1_compute_loss``
      (déjà supporté par V5-mini, pas besoin de modifier la signature).
    - Compute négligeable : MSE sur matrice [num_vars × num_vars] = 36 ops pour
      num_vars=6.
    - Compatible avec CASTLE (Phase A) et L1 annealing (Phase B) — les trois
      mécanismes agissent sur des aspects différents du DAG :
      * CASTLE : magnitudes par prédictivité
      * L1 anneal : sparsification progressive
      * G_phys prior : alignement directionnel + signe physique
    """
    if A_dag.shape != G_phys.shape:
        raise ValueError(
            f"A_dag shape {tuple(A_dag.shape)} != G_phys shape {tuple(G_phys.shape)}"
        )
    G_phys = G_phys.to(device=A_dag.device, dtype=A_dag.dtype)
    target = alpha * G_phys
    if mask_diagonal:
        eye = torch.eye(A_dag.size(0), device=A_dag.device, dtype=A_dag.dtype)
        off_diag = 1.0 - eye
        sse = ((A_dag - target) * off_diag).pow(2).sum()
        if normalize:
            # Legacy V5-mini behavior (silently weakens prior by N(N-1))
            return sse / off_diag.sum().clamp(min=1.0)
        # I1 fix: unnormalized sum so lambda_dag_prior has its full effect
        return sse
    if normalize:
        return ((A_dag - target) ** 2).mean()
    return ((A_dag - target) ** 2).sum()


__all__ = [
    "VAR_LABELS",
    "EXPECTED_EDGES",
    "VAR_LABELS_9NODE",
    "EXPECTED_EDGES_9NODE",
    "VAR_LABELS_V6",
    "EXPECTED_EDGES_V6",
    "build_physical_mask",
    "physical_prior_loss",
]
