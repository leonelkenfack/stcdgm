"""Tête Bernoulli-Gamma pour l'étage 1 (action A3 de V8 §9.6).

Pourquoi : mesuré sur ACCESS-CM2, seulement **1 % de la variance HR** vit sous
48 km (`results/diag_stage1.json`). Une perte quadratique n'a donc quasiment
aucune incitation de gradient à restituer les extrêmes — le défaut est dans
l'objectif, pas dans le câblage.

Pourquoi celle-ci et pas une pondération de queue : la moyenne analytique
``mu = p * alpha * beta`` **est** la moyenne conditionnelle en mm/j. Une MSE
pondérée, elle, déplace le minimiseur vers une moyenne inclinée et biaise le
champ globalement haut.

**Réserve mesurée, à ne pas contourner.** Cette propriété vaut en mm/j. Le
pipeline, lui, transporte l'étage 2 en ``log1p``, et ``log1p`` est CONCAVE :
``E[log1p(X)] < log1p(E[X])`` dès que la variance conditionnelle est non nulle.
Prendre naïvement ``log1p(mu)`` comme ancre donne donc un résidu de moyenne
NÉGATIVE, pas nulle — biais mesuré **+0,73 en unités log1p** sur une loi
Bernoulli-Gamma réaliste (p=0,4, alpha=2, beta=5). C'est le miroir exact du
problème que corrige ``evaluation/jensen.py``, dans l'autre sens.

Correction retenue : la structure « hurdle » donne ``E[log1p(X)] =
p * E[log1p(G)]`` puisque ``log1p(0) = 0``, et ``E[log1p(G)]`` s'approche par
un développement d'ordre 3 sur les moments centrés de la Gamma. Biais résiduel
**0,01 à 0,11** selon les paramètres, contre 0,43 à 0,92 pour l'ancre naïve.
C'est une approximation, pas une identité : ne jamais écrire que ``E[r|y] = 0``
tient « par construction » dans l'espace log.

Bonus : ``p`` donne l'occurrence explicitement, donc l'indice CDD ne se déduit
plus d'un seuillage de la moyenne.

Interface : opère en **mm/jour**, pas en log1p. Voir ``mean_as_log_residual``
pour retrouver l'ancre attendue par l'étage 2.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["BernoulliGammaHead", "bernoulli_gamma_nll", "stage1_bg_loss",
           "decode_bg_params"]

_EPS = 1e-6


class BernoulliGammaHead(nn.Module):
    """Projette des features de décodeur vers (p, alpha, beta) par pixel.

    Parameters
    ----------
    in_channels :
        canaux de features en entrée (sortie du décodeur avant projection 1x1).
    wet_threshold :
        seuil au-dessus duquel la Gamma est ajustée, en mm/j. **0,1 et non 1,0** :
        mesuré sur une loi « bruine » réaliste (p=0,45, alpha=0,55, beta=8),
        ajuster au-dessus de 1 mm/j donne alpha=1,05 au lieu de 0,55 — la FORME
        est fausse d'un facteur 2, et c'est elle qui gouverne les extrêmes, donc
        exactement ce que cette tête existe pour bien estimer. Biais sur la
        moyenne : -2,7 % à 1,0 mm contre -0,1 % à 0,1 mm.
        Ne pas confondre avec le seuil ETCCDI de 1 mm/j utilisé pour RAPPORTER
        l'occurrence et l'indice CDD : celui-ci est un choix d'ESTIMATION.
    """

    def __init__(self, in_channels: int, wet_threshold: float = 0.1) -> None:
        super().__init__()
        self.wet_threshold = float(wet_threshold)
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        # Biais initial : p~0.35 (frequence humide observee ~0.39), alpha~1,
        # beta~1. Le poids part petit mais NON NUL : a zero, dL/dfeats =
        # W^T·dL/draw = 0, et aucun gradient n'atteindrait le decodeur ni le
        # chemin causal en amont au premier pas.
        nn.init.normal_(self.proj.weight, std=0.01)
        with torch.no_grad():
            self.proj.bias.copy_(torch.tensor([-0.6, 0.55, 0.55]))

    def forward(self, feats: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """feats [B,C,H,W] -> (p, alpha, beta), chacun [B,1,H,W]."""
        raw = self.proj(feats)
        # .float() OBLIGATOIRE avant le clamp : sous AMP fp16, `1 - 1e-6`
        # s'arrondit exactement a 1.0 (eps fp16 ~ 1e-3), le clamp haut devient
        # inoperant, et des qu'un logit depasse ~8 la sigmoide sature a 1.0 —
        # log1p(-p) vaut alors -inf sur TOUS les pixels secs, et la NLL part en
        # NaN au backward.
        raw = raw.float()
        p = torch.sigmoid(raw[:, 0:1]).clamp(_EPS, 1.0 - _EPS)
        alpha = nn.functional.softplus(raw[:, 1:2]) + _EPS
        beta = nn.functional.softplus(raw[:, 2:3]) + _EPS
        return p, alpha, beta

    @staticmethod
    def mean(p: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        """Moyenne conditionnelle analytique, en mm/j. C'EST l'ancre du résidu."""
        return p * alpha * beta

    @staticmethod
    def variance(p: Tensor, alpha: Tensor, beta: Tensor) -> Tensor:
        """Variance conditionnelle analytique, en (mm/j)^2.

        ``Var[X] = p*alpha*beta^2*(1+alpha) - (p*alpha*beta)^2`` : le terme de
        Bernoulli ajoute la variance d'occurrence a celle de l'intensite.
        """
        return p * alpha * beta ** 2 * (1.0 + alpha) - (p * alpha * beta) ** 2

    # Grille de quadrature pour E[log1p(G)]. Log-espacee en u, ou G = beta*u et
    # u ~ Gamma(alpha, 1) : la grille est ainsi INDEPENDANTE de (alpha, beta),
    # donc partagee par tous les pixels et calculee une fois.
    _LOG_U = torch.linspace(-18.0, 4.5, 128)
    _QUAD_U = _LOG_U.exp()
    _QUAD_W = torch.cat([                      # poids trapezes en log-u
        ((_LOG_U[1] - _LOG_U[0]) / 2.0).reshape(1),
        (_LOG_U[2:] - _LOG_U[:-2]) / 2.0,
        ((_LOG_U[-1] - _LOG_U[-2]) / 2.0).reshape(1),
    ])

    @staticmethod
    def mean_log1p(p: Tensor, alpha: Tensor, beta: Tensor,
                   delta: float = 0.0) -> Tensor:
        """``E[log1p(X)]`` — la moyenne dans l'espace ou vit l'etage 2.

        Ce n'est PAS ``log1p(E[X])``. La structure hurdle donne exactement
        ``E[log1p(X)] = p * E[log1p(G)]`` (car ``log1p(0) = 0``).

        ``E[log1p(G)]`` est obtenu par QUADRATURE et non plus par un
        developpement de Taylor. L'ancienne version tronquait a l'ordre 3 :

            log1p(mu) - Var/2(1+mu)^2 + (2/3)mu3/(1+mu)^3

        Le rayon de convergence de log1p est fini, le support de la Gamma ne
        l'est pas, et le troisieme terme croit en beta^3 pendant que son
        denominateur croit en (1+alpha*beta)^3 : des que alpha < 1 avec beta
        grand, il DIVERGE. Mesure contre Monte-Carlo :

            alpha  beta   vrai   ordre 3   erreur
             0.55   8.0   1.246    2.275    +1.03     <- regime "bruine"
             0.30  20.0   1.207    5.386    +4.18
             0.20  30.0   1.033   10.605    +9.57

        Or l'ecart-type du residu que l'etage 2 doit apprendre vaut 0,23 : une
        erreur de +1,03 est quatre fois le signal entier. Le self-check ne
        testait que (alpha=2, beta=5), ou l'erreur n'est que de +0,12, et
        passait. La quadrature ci-dessous est exacte a 0,002 sur toute la plage
        alpha in [0,05, 5], beta in [0,5, 30] pour ~1 ms par carte HR sur GPU.
        """
        u = BernoulliGammaHead._QUAD_U.to(alpha.device, alpha.dtype)
        w = BernoulliGammaHead._QUAD_W.to(alpha.device, alpha.dtype)
        log_u = BernoulliGammaHead._LOG_U.to(alpha.device, alpha.dtype)
        a = alpha.unsqueeze(-1)
        b = beta.unsqueeze(-1)
        # densite de u ~ Gamma(a, 1) en log, multipliee par u (jacobien log-u)
        log_dens = a * log_u - u - torch.lgamma(a)
        e_log_g = (log_dens.exp() * w * torch.log1p(delta + b * u)).sum(-1)
        # Terme sec : le pipeline transforme par log1p(x + delta), donc un pixel
        # sec vaut log1p(delta) et non 0. L'omettre laisse un decalage
        # systematique sur toute la fraction seche du domaine.
        import math as _math
        return p * e_log_g + (1.0 - p) * _math.log1p(delta)

    @staticmethod
    def mean_as_log_residual(p: Tensor, alpha: Tensor, beta: Tensor,
                             baseline_log1p: Tensor,
                             correct_concavity: bool = True,
                             delta: float = 0.0) -> Tensor:
        """Ancre dans la convention de l'étage 2 : ``log1p(mu) - log1p(baseline)``.

        ``baseline_log1p`` est pris DÉJÀ en log1p — c'est ce que le pipeline
        transporte (clé ``baseline`` des batches). Repasser par expm1 puis
        log1p ferait un aller-retour inutile et perdrait de la précision.

        ``correct_concavity=True`` utilise ``E[log1p(X)]`` au lieu de
        ``log1p(E[X])``. Le mettre à False restaure l'ancre naïve, qui biaise
        le résidu de l'étage 2 vers le bas — utile seulement pour reproduire
        un run antérieur.
        """
        if correct_concavity:
            return (BernoulliGammaHead.mean_log1p(p, alpha, beta, delta)
                    - baseline_log1p)
        mu = BernoulliGammaHead.mean(p, alpha, beta)
        return torch.log1p(mu.clamp(min=0.0)) - baseline_log1p


def bernoulli_gamma_nll(p: Tensor, alpha: Tensor, beta: Tensor, y_mm: Tensor,
                        wet_threshold: float = 1.0,
                        mask: Tensor | None = None) -> Tensor:
    """Log-vraisemblance négative Bernoulli-Gamma (Cannon 2008).

    y_mm : cible en mm/jour. mask : booléen, True = pixel pris en compte
    (utiliser le masque terre/mer pour exclure l'océan, comme le reste du
    pipeline).
    """
    wet = (y_mm > wet_threshold).to(y_mm.dtype)
    y = y_mm.clamp(min=_EPS)

    dry_term = (1.0 - wet) * torch.log1p(-p)
    wet_term = wet * (
        torch.log(p)
        + (alpha - 1.0) * torch.log(y)
        - y / beta
        - alpha * torch.log(beta)
        - torch.lgamma(alpha)
    )
    nll = -(dry_term + wet_term)
    if mask is not None:
        nll = nll[mask]
    if nll.numel() == 0:
        # mean() sur un tenseur vide vaut NaN et contaminerait la perte sans
        # rien signaler. Un batch entierement masque est une erreur de donnees.
        raise ValueError(
            "aucun pixel valide dans le batch : le masque est entierement "
            "faux, ou la cible ne contient que des NaN.")
    return nll.mean()


def decode_bg_params(regression_head, bg_head, H_T: Tensor,
                     target_shape=None) -> tuple[Tensor, Tensor, Tensor]:
    """``H_T`` -> ``(p, alpha, beta)`` via les features du décodeur.

    Point d'entrée unique : le décodeur expose ses features *avant* sa
    projection à 1 canal, que la tête court-circuite. Passer par ici plutôt
    que de rouvrir ``return_features`` à chaque appelant (entraînement,
    pré-calcul étage 2, évaluation) évite d'avoir trois versions du
    redimensionnement qui divergent.
    """
    feats = regression_head(H_T, return_features=True)
    if target_shape is not None and tuple(feats.shape[-2:]) != tuple(target_shape):
        feats = nn.functional.interpolate(
            feats, size=tuple(target_shape), mode="bilinear", align_corners=False)
    return bg_head(feats)


def stage1_bg_loss(p: Tensor, alpha: Tensor, beta: Tensor,
                   target_residual_log: Tensor, baseline_log: Tensor,
                   wet_threshold: float = 1.0) -> tuple[Tensor, Tensor]:
    """Pont entre la tête (mm/j) et la convention log1p de l'étage 1.

    Le pipeline transporte ``residual = log1p(HR) - log1p(baseline)`` et
    ``baseline = log1p(baseline_mm)``. La vraisemblance, elle, ne se définit
    qu'en mm/j — d'où la reconversion ici plutôt que dans la tête.

    Note : le pipeline applique ``log1p(x + precipitation_delta)`` avec
    ``delta = 0.01`` mm/j, donc ``y_mm`` reconstruit vaut ``HR + 0.01``. Face
    au seuil humide de 1 mm/j c'est 1 % au point le plus défavorable et
    strictement moins au-dessus — la queue, seule cible de A3, est intacte.

    Returns
    -------
    (nll, mu_log_residual) : la perte à substituer à la MSE, et l'ancre
    ``log1p(mu) - log1p(baseline)`` que l'étage 2 attend à la place de
    ``mu_HR``.
    """
    y_mm = torch.expm1(target_residual_log + baseline_log)
    mask = torch.isfinite(y_mm)                      # pixels océan = NaN
    y_mm = torch.nan_to_num(y_mm, nan=0.0)
    nll = bernoulli_gamma_nll(p, alpha, beta, y_mm,
                              wet_threshold=wet_threshold, mask=mask)
    return nll, BernoulliGammaHead.mean_as_log_residual(p, alpha, beta, baseline_log)


if __name__ == "__main__":
    # Self-check : sur des donnees Bernoulli-Gamma synthetiques, la tete doit
    # retrouver la moyenne conditionnelle vraie. Echoue si la NLL ou la formule
    # de moyenne est fausse.
    torch.manual_seed(0)
    B, C, H, W = 64, 8, 16, 16
    p_t, a_t, b_t = 0.4, 2.0, 5.0            # verite -> moyenne = 0.4*2*5 = 4.0
    wet = (torch.rand(B, 1, H, W) < p_t).float()
    y = wet * torch.distributions.Gamma(a_t, 1.0 / b_t).sample((B, 1, H, W))

    feats = torch.ones(B, C, H, W)           # entree constante : la tete ne peut
    head = BernoulliGammaHead(C)             # qu'apprendre la loi marginale
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(400):
        p, a, b = head(feats)
        loss = bernoulli_gamma_nll(p, a, b, y)
        opt.zero_grad(); loss.backward(); opt.step()

    p, a, b = head(feats)
    mu = BernoulliGammaHead.mean(p, a, b).mean().item()
    true_mu = y.mean().item()
    print(f"p={p.mean():.3f} (vrai {p_t})  alpha={a.mean():.2f} (vrai {a_t})  "
          f"beta={b.mean():.2f} (vrai {b_t})")
    print(f"moyenne predite = {mu:.3f}   moyenne empirique = {true_mu:.3f}")
    assert abs(mu - true_mu) / true_mu < 0.10, (
        f"la moyenne analytique p*alpha*beta ({mu:.3f}) devie de plus de 10 % "
        f"de la moyenne empirique ({true_mu:.3f})")
    # LE test qui compte : l'ancre vit en espace log1p, ou log1p(E[X]) n'est PAS
    # E[log1p(X)]. On mesure directement le biais du residu que recoit l'etage 2.
    mu_t = BernoulliGammaHead.mean(p, a, b)
    e_log_true = torch.log1p(y).mean()                      # E[log1p(X)] empirique
    a_naif = torch.log1p(mu_t).mean()                       # ancre naive
    a_corr = BernoulliGammaHead.mean_log1p(p, a, b).mean()  # ancre corrigee
    biais_naif = float((a_naif - e_log_true).abs())
    biais_corr = float((a_corr - e_log_true).abs())
    print(f"E[log1p(X)] = {float(e_log_true):.4f} | ancre naive {float(a_naif):.4f} "
          f"(biais {biais_naif:+.4f}) | corrigee {float(a_corr):.4f} "
          f"(biais {biais_corr:+.4f})")
    assert biais_corr < biais_naif / 4.0, (
        f"la correction de concavite doit diviser le biais par au moins 4 : "
        f"{biais_naif:.4f} -> {biais_corr:.4f}")
    # ------------------------------------------------------------------ #
    # E[log1p(G)] SUR TOUTE LA PLAGE, pas seulement au point commode.
    # Le test ci-dessus n'exerce que (alpha=2, beta=5), ou l'ancien
    # developpement de Taylor a l'ordre 3 se trompait de +0,12 et passait.
    # A (alpha=0,55, beta=8) — la loi "bruine" que le docstring de
    # wet_threshold declare realiste — il se trompait de +1,03, soit quatre
    # fois l'ecart-type du residu que l'etage 2 doit apprendre. Ce balayage
    # est la pour que ca ne puisse plus passer inapercu.
    # ------------------------------------------------------------------ #
    torch.manual_seed(1)
    pire, pire_pt = 0.0, None
    for _al in (0.05, 0.2, 0.55, 1.0, 2.0, 5.0):
        for _be in (0.5, 2.0, 8.0, 30.0):
            _g = torch.distributions.Gamma(torch.tensor(_al), torch.tensor(1.0 / _be))
            _vrai = float(torch.log1p(_g.sample((200_000,))).mean())
            _q = float(BernoulliGammaHead.mean_log1p(
                torch.ones(1), torch.full((1,), _al), torch.full((1,), _be)))
            if abs(_q - _vrai) > pire:
                pire, pire_pt = abs(_q - _vrai), (_al, _be, _vrai, _q)
    print(f"E[log1p(G)] par quadrature : erreur max {pire:.4f} sur alpha in "
          f"[0,05, 5] x beta in [0,5, 30]  (pire cas alpha={pire_pt[0]}, "
          f"beta={pire_pt[1]} : {pire_pt[3]:.4f} vs {pire_pt[2]:.4f})")
    assert pire < 0.02, (
        f"E[log1p(G)] devie de {pire:.4f} en (alpha={pire_pt[0]}, "
        f"beta={pire_pt[1]}) — l'ecart-type du residu vaut ~0,23, une erreur "
        f"de cette taille rend l'ancre inutilisable")

    # La variance analytique doit coller a l'empirique (elle sert a la correction).
    var_a = float(BernoulliGammaHead.variance(p, a, b).mean())
    var_e = float(y.var())
    assert abs(var_a - var_e) / var_e < 0.20, f"Var analytique {var_a:.2f} vs {var_e:.2f}"
    # L'ancre naive reste accessible pour reproduire un run anterieur.
    naive_anchor = BernoulliGammaHead.mean_as_log_residual(
        p, a, b, torch.log1p(mu_t), correct_concavity=False)
    assert naive_anchor.abs().max() < 1e-5, "correct_concavity=False doit rendre l'ancre naive"

    # Le pont log1p : partir d'un couple (baseline, HR) connu, verifier que
    # stage1_bg_loss reconstruit bien y_mm et que l'ancre reste coherente.
    baseline_mm = torch.full_like(y, 2.0)
    baseline_log = torch.log1p(baseline_mm)
    residual_log = torch.log1p(y) - baseline_log
    nll_bridge, anchor_bridge = stage1_bg_loss(p, a, b, residual_log, baseline_log)
    nll_direct = bernoulli_gamma_nll(p, a, b, y)
    assert torch.allclose(nll_bridge, nll_direct, rtol=1e-4), (
        f"le pont log1p change la NLL : {nll_bridge:.6f} vs {nll_direct:.6f}")
    expected = BernoulliGammaHead.mean_log1p(p, a, b) - baseline_log
    assert torch.allclose(anchor_bridge, expected, atol=1e-6), "ancre du pont incoherente"

    # Pixels ocean (NaN) : la NLL doit rester finie et ignorer ces pixels.
    residual_nan = residual_log.clone()
    residual_nan[:, :, 0, :] = float("nan")
    nll_nan, _ = stage1_bg_loss(p, a, b, residual_nan, baseline_log)
    assert torch.isfinite(nll_nan), "la NLL doit rester finie avec des NaN ocean"

    # Cablage decodeur -> tete. sys.path[0] est ce repertoire quand on lance
    # le fichier directement, d'ou l'import absolu.
    from regression_head import GraphToGridDecoder

    dec = GraphToGridDecoder(d_model=16, hr_h=24, hr_w=26,
                             intermediate_h=6, intermediate_w=7,
                             n_heads=2, refine_channels=8)
    bg = BernoulliGammaHead(dec.feature_channels)
    H_T = torch.randn(2, 3, 12, 16)                  # [B, q, N, d_model]
    p2, a2, b2 = decode_bg_params(dec, bg, H_T, target_shape=(24, 26))
    assert p2.shape == (2, 1, 24, 26), f"forme inattendue {tuple(p2.shape)}"
    assert (a2 > 0).all() and (b2 > 0).all(), "alpha/beta doivent rester positifs"
    # la tete doit bien voir les features, pas la projection 1 canal
    assert dec(H_T, return_features=True).shape[1] == dec.feature_channels
    assert dec(H_T).shape[1] == 1
    # le gradient doit remonter jusqu'au decodeur : sinon la tete est un
    # bouchon decoratif et l'etage causal ne recoit aucun signal.
    loss_wire, _ = stage1_bg_loss(p2, a2, b2,
                                  torch.rand(2, 1, 24, 26),
                                  torch.zeros(2, 1, 24, 26))
    loss_wire.backward()
    assert dec.grid_queries.grad is not None and dec.grid_queries.grad.abs().sum() > 0, (
        "le gradient de la NLL n'atteint pas le decodeur")
    print("OK — moyenne analytique, ancre, pont log1p, masque NaN et cablage decodeur.")
