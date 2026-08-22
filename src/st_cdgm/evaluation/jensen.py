"""Correction du biais de Jensen au retour log1p -> mm (action A1 de V8 §9.6).

Le problème
-----------
L'étage 1 minimise une MSE sur ``log1p``. Son minimiseur est donc
``E[log1p(x) | y]``, pas ``E[x | y]``. Comme ``expm1`` est convexe, l'inégalité
de Jensen donne ``expm1(E[log1p x]) < E[x]`` : la reconversion naïve
**sous-estime systématiquement**, et l'écart croît avec l'intensité.

Mesuré sur ACCESS-CM2 (SPEC-2, 4 000 jours) : biais absolu moyen **9,11 %**,
jusqu'à **18,5 %** sur le bin le plus intense. La correction lognormale
``expm1(mu + s²/2)`` le ramène à **0,46 %**.

Portée
------
S'applique à une **moyenne conditionnelle** reconvertie en mm — la sortie de
l'étage 1 seule. Ne s'applique **pas** :

* aux tirages de l'étage 2 : un échantillon de ``log1p(x)`` repasse en mm par
  ``expm1`` sans biais, c'est une transformation de variable, pas d'espérance ;
* à la moyenne d'ensemble : là, le remède est d'ordonner ``expm1`` AVANT la
  moyenne (``mean_k expm1(·)``), pas d'ajouter ``s²/2`` ;
S'applique EN REVANCHE à la tête Bernoulli-Gamma (V8 A3), contrairement à ce
qui était écrit ici au départ. Son ancre est ``mean_as_log_residual``, qui
renvoie ``E[log1p(X)]`` — une moyenne EN ESPACE LOG. La repasser en mm demande
donc bien ``expm1(m + s²/2)``. C'est ``log1p(E[X])`` qui aurait été exempt, et
c'est précisément l'ancre que la tête n'utilise plus (elle biaisait le résidu
de l'étage 2 de +0,73 en unités log1p). Les deux corrections sont cohérentes :
la tête produit une moyenne log, ce module la ramène en mm.

Le ``s²`` utile
---------------
``s² = Var[log1p(x) | y]``, c'est-à-dire la variance de l'erreur résiduelle de
l'étage 1. Elle est **hétéroscédastique** : elle croît avec l'intensité, et
c'est justement là que le biais fait mal. Un ``s²`` scalaire global
sur-corrigerait les régimes secs et sous-corrigerait la queue — d'où la table
1-D indexée par la prédiction, qui reproduit la construction de SPEC-2.
Un ``s²`` global reste disponible via ``n_bins=1``.
"""
from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["JensenCorrector", "PRECIP_DELTA"]

PRECIP_DELTA = 0.01  # pipeline.py precipitation_delta


class JensenCorrector:
    """Table 1-D de ``s²`` indexée par le champ log prédit.

    Parameters
    ----------
    edges :
        bornes des bins sur ``baseline_log + mu_log``, croissantes, taille B+1.
    sigma2 :
        variance résiduelle par bin, taille B.
    """

    def __init__(self, edges: Tensor, sigma2: Tensor) -> None:
        edges = torch.as_tensor(edges, dtype=torch.float32).flatten()
        sigma2 = torch.as_tensor(sigma2, dtype=torch.float32).flatten()
        if edges.numel() != sigma2.numel() + 1:
            raise ValueError(
                f"edges ({edges.numel()}) doit valoir sigma2 ({sigma2.numel()}) + 1")
        if (sigma2 < 0).any():
            raise ValueError("sigma2 négatif : ce n'est pas une variance")
        self.edges = edges
        self.sigma2 = sigma2

    # ------------------------------------------------------------------

    @classmethod
    def fit(cls, mu_log: Tensor, target_log: Tensor, baseline_log: Tensor,
            n_bins: int = 20, min_count: int = 500) -> "JensenCorrector":
        """Estime ``s²(prédiction)`` sur un jeu où la vérité est connue.

        À appeler **sur le train**, jamais sur le jeu d'évaluation : ``s²``
        est un paramètre de calibration, l'estimer sur le test ferait fuiter
        la cible dans la métrique.

        ``mu_log`` / ``target_log`` sont des RÉSIDUS (convention pipeline),
        ``baseline_log`` est déjà en log1p. Les non-finis sont ignorés.
        """
        mu = torch.as_tensor(mu_log, dtype=torch.float32).flatten()
        tg = torch.as_tensor(target_log, dtype=torch.float32).flatten()
        bl = torch.as_tensor(baseline_log, dtype=torch.float32).flatten()
        if not (mu.numel() == tg.numel() == bl.numel()):
            raise ValueError("mu_log, target_log et baseline_log doivent avoir "
                             f"la même taille ({mu.numel()}, {tg.numel()}, {bl.numel()})")

        pred_full = bl + mu
        err = tg - mu                       # erreur résiduelle de l'étage 1
        ok = torch.isfinite(pred_full) & torch.isfinite(err)
        if ok.sum() < max(min_count, 2):
            raise ValueError(f"trop peu de pixels finis pour calibrer ({int(ok.sum())})")
        pred_full, err = pred_full[ok], err[ok]

        n_bins = max(1, int(n_bins))
        qs = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float32)
        # torch.quantile leve au-dela de 2^24 elements. Les bornes de bins n'ont
        # pas besoin de tout l'echantillon : un sous-echantillon regulier de 4 M
        # points les estime a la precision utile. La VARIANCE par bin, elle,
        # reste calculee sur la totalite.
        _MAX_Q = 4_000_000
        ref = pred_full if pred_full.numel() <= _MAX_Q else pred_full[
            :: max(1, pred_full.numel() // _MAX_Q)]
        edges = torch.quantile(ref, qs)
        edges = torch.unique(edges)                       # bins vides collapsés
        if edges.numel() < 2:                             # champ constant
            edges = torch.tensor([pred_full.min() - 1.0, pred_full.max() + 1.0])

        idx = torch.bucketize(pred_full, edges[1:-1], right=False)
        n_eff = edges.numel() - 1
        global_s2 = err.var(unbiased=True)
        sigma2 = torch.full((n_eff,), float(global_s2))
        for k in range(n_eff):
            sel = idx == k
            # Sous min_count, l'estimation de variance est trop bruitée : on
            # retombe sur la variance globale plutôt que d'injecter du bruit
            # dans la correction.
            if int(sel.sum()) >= min_count:
                sigma2[k] = err[sel].var(unbiased=True)
        return cls(edges, sigma2)

    # ------------------------------------------------------------------

    def sigma2_at(self, pred_full_log: Tensor) -> Tensor:
        """``s²`` interpolé au plus proche bin, forme de l'entrée."""
        x = torch.as_tensor(pred_full_log, dtype=torch.float32)
        idx = torch.bucketize(x.flatten(), self.edges[1:-1].to(x.device), right=False)
        idx = idx.clamp(0, self.sigma2.numel() - 1)
        return self.sigma2.to(x.device)[idx].view(x.shape)

    def to_mm(self, mu_log: Tensor, baseline_log: Tensor,
              delta: float = PRECIP_DELTA, corrected: bool = True) -> Tensor:
        """Moyenne conditionnelle de l'étage 1, en mm/j, biais de Jensen corrigé.

        ``corrected=False`` donne la reconversion naïve — utile pour chiffrer
        l'effet de la correction sur un même jeu.
        """
        mu = torch.as_tensor(mu_log, dtype=torch.float32)
        bl = torch.as_tensor(baseline_log, dtype=torch.float32)
        pred_full = bl + mu
        if corrected:
            pred_full = pred_full + 0.5 * self.sigma2_at(pred_full)
        return (torch.expm1(pred_full) - delta).clamp(min=0.0)

    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {"edges": self.edges, "sigma2": self.sigma2}

    @classmethod
    def load_state_dict(cls, state: dict) -> "JensenCorrector":
        return cls(state["edges"], state["sigma2"])

    def __repr__(self) -> str:
        return (f"JensenCorrector(n_bins={self.sigma2.numel()}, "
                f"s2={float(self.sigma2.min()):.3f}..{float(self.sigma2.max()):.3f})")


if __name__ == "__main__":
    # Self-check : on fabrique une verite lognormale HETEROSCEDASTIQUE (la
    # variance conditionnelle croit avec l'intensite, comme la precipitation
    # reelle), on donne au correcteur le mu PARFAIT, et on verifie qu'il
    # reduit le biais la ou SPEC-2 l'a mesure.
    torch.manual_seed(0)
    N = 400_000
    baseline_log = torch.rand(N) * 3.0                 # log1p de la baseline
    mu_true = baseline_log * 0.9                       # E[log1p x | y], residuel
    sigma_true = 0.2 + 0.5 * (baseline_log / 3.0)      # heteroscedastique
    target = mu_true + sigma_true * torch.randn(N)     # log1p(x) - baseline

    corr = JensenCorrector.fit(mu_true, target, baseline_log, n_bins=20)
    print(corr)

    truth_mm = torch.expm1(baseline_log + target) - PRECIP_DELTA
    naive_mm = corr.to_mm(mu_true, baseline_log, corrected=False)
    corr_mm = corr.to_mm(mu_true, baseline_log, corrected=True)

    # Biais par bin d'intensite : c'est la forme sous laquelle SPEC-2 l'a mesure.
    qs = torch.quantile(baseline_log, torch.linspace(0, 1, 11))
    b_naive, b_corr = [], []
    for k in range(len(qs) - 1):
        s = (baseline_log >= qs[k]) & (baseline_log < qs[k + 1])
        if s.sum() < 500:
            continue
        t = truth_mm[s].mean()
        b_naive.append(float((t - naive_mm[s].mean()).abs() / t) * 100)
        b_corr.append(float((t - corr_mm[s].mean()).abs() / t) * 100)
    mean_naive = sum(b_naive) / len(b_naive)
    mean_corr = sum(b_corr) / len(b_corr)
    print(f"biais absolu moyen : naif = {mean_naive:5.2f} %   corrige = {mean_corr:5.2f} %")
    print(f"pire bin           : naif = {max(b_naive):5.2f} %   corrige = {max(b_corr):5.2f} %")

    assert mean_corr < mean_naive / 3.0, (
        f"la correction doit diviser le biais moyen par >3 "
        f"({mean_naive:.2f} % -> {mean_corr:.2f} %)")
    assert max(b_corr) < max(b_naive), "la correction doit aussi aider le pire bin"

    # La correction ne doit JAMAIS reduire la prediction : s^2 >= 0.
    assert (corr_mm >= naive_mm - 1e-4).all(), "expm1(mu + s^2/2) >= expm1(mu)"
    # s^2 constant (n_bins=1) = variante homoscedastique, doit rester valide
    # mais moins bonne : c'est l'argument de la table 1-D.
    flat = JensenCorrector.fit(mu_true, target, baseline_log, n_bins=1)
    flat_mm = flat.to_mm(mu_true, baseline_log)
    b_flat = []
    for k in range(len(qs) - 1):
        s = (baseline_log >= qs[k]) & (baseline_log < qs[k + 1])
        if s.sum() < 500:
            continue
        t = truth_mm[s].mean()
        b_flat.append(float((t - flat_mm[s].mean()).abs() / t) * 100)
    mean_flat = sum(b_flat) / len(b_flat)
    print(f"s2 global (n_bins=1) : {mean_flat:5.2f} %  <- pourquoi la table 1-D")
    assert mean_corr < mean_flat, "la table 1-D doit battre le s2 scalaire"

    # Aller-retour de serialisation.
    again = JensenCorrector.load_state_dict(corr.state_dict())
    assert torch.allclose(again.to_mm(mu_true, baseline_log), corr_mm)
    print("OK — correction de Jensen validee (heteroscedastique, monotonie, serialisation).")
