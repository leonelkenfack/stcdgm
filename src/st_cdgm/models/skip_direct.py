"""Skip-connection conditionnelle LR → μ_HR pour Oracle V5 (A1).

Module introduit par V5-mini pour récupérer 50-80 % du handicap d'expressivité
d'Oracle face au U-Net générique de CorrDiff, sans rompre la propriété (O3).

Principe
--------
Le module mélange deux voies parallèles :

    μ_HR = α(y_LR) · CRM(y_LR ; A_dag)  +  (1 - α(y_LR)) · direct(y_LR)

où :
- ``CRM(...)`` est la moyenne haute résolution produite par le chemin causal
  d'Oracle (encodeur → RCN → décodeur graphe-à-grille → regression_head).
- ``direct(...)`` est un sur-échantillonnage convolutif léger qui exploite
  directement la basse résolution sans passer par le DAG.
- ``α(y_LR) ∈ [0, 1]`` est un gate scalaire global appris par un MLP qui
  voit l'entrée basse résolution pooled. Il décide pour chaque exemple
  la fraction de blending.

Préservation de (O3)
--------------------
Une régularisation ``L_o3_preserve = λ_o3 · max(0, alpha_floor - α.mean())²``
pénalise les états où α descend sous un plancher (typ. 0.6), forçant la voie
causale à rester dominante. Sans cette pénalité, α convergerait souvent vers
0 (la voie directe est moins contrainte donc plus facile à minimiser).

À l'initialisation
------------------
Le biais du dernier neurone du gate est positionné à ``alpha_init_logit``
(typ. 2.0 → sigmoid ≈ 0.88) pour que le chemin causal domine dès le pas 0.

Coût de calcul
--------------
- ``direct`` : 2 convolutions (kernel 3) + bilinear upsample + 1 conv 1×1.
  Pour la résolution NZ 23×26 → 172×179, c'est ~0.5M FLOPs/forward, négligeable.
- ``alpha_mlp`` : pooling global + 2 Linear, ~1k FLOPs/forward.

Sur A100 / training de 30h, surcoût mesuré ~3h (10 %).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["ConditionalSkipBlock"]


class ConditionalSkipBlock(nn.Module):
    """Skip-connection conditionnelle entre la voie causale et une voie directe.

    Parameters
    ----------
    lr_channels :
        Nombre de canaux d'entrée basse résolution (typ. 15 variables × T pas).
    hr_shape :
        Forme HR cible ``(H, W)`` (typ. ``(172, 179)`` pour NZ).
    alpha_init_logit :
        Biais initial du gate. ``sigmoid(2.0) ≈ 0.88`` → chemin causal dominant.
    alpha_floor :
        Plancher utilisé par ``compute_o3_preserve_loss`` pour pénaliser α bas.
        Doit être en accord avec ``cfg.v5.skip_connection.alpha_floor``.
    direct_channels :
        Largeur du réseau ``direct`` (32 par défaut, suffisant sur NZ).

    Attributes
    ----------
    direct : nn.Sequential
        Voie résiduelle directe LR → HR (sans DAG).
    alpha_mlp : nn.Sequential
        Gate scalaire ``α(y_LR) ∈ [0, 1]``.
    alpha_floor : float
        Plancher pour la régularisation (O3).
    """

    def __init__(
        self,
        lr_channels: int,
        hr_shape: Tuple[int, int],
        *,
        alpha_init_logit: float = 2.0,
        alpha_floor: float = 0.6,
        direct_channels: int = 32,
        # >>> V8 — A4 (docs/architecture_v8_design.md §9.2)
        mode: str = "convex",
        direct_ceiling: float = 0.15,
    ) -> None:
        super().__init__()
        if mode not in ("convex", "residual"):
            raise ValueError(f"mode inconnu : {mode!r}")
        self.hr_shape = tuple(hr_shape)
        self.alpha_floor = float(alpha_floor)
        self.mode = mode
        self.direct_ceiling = float(direct_ceiling)
        # En mode résiduel, α change de sens : ce n'est plus le POIDS du chemin
        # causal (qu'on veut haut) mais l'AMPLITUDE de la correction directe
        # (qu'on veut basse). Le biais initial suit ce retournement.
        if mode == "residual" and alpha_init_logit > 0.0:
            alpha_init_logit = -abs(alpha_init_logit)

        # ─── Voie directe : Conv → GELU → Upsample → Conv → 1×1 ───
        self.direct = nn.Sequential(
            nn.Conv2d(lr_channels, direct_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(size=self.hr_shape, mode="bilinear", align_corners=False),
            nn.Conv2d(direct_channels, direct_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(direct_channels // 2, 1, kernel_size=1),
        )

        # ─── Gate α(y_LR) : Pooling → MLP → Sigmoid ───
        self.alpha_mlp = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(lr_channels, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        # Bias initial du dernier Linear pour démarrer α ≈ sigmoid(alpha_init_logit)
        with torch.no_grad():
            self.alpha_mlp[-1].bias.fill_(float(alpha_init_logit))
            self.alpha_mlp[-1].weight.normal_(mean=0.0, std=0.01)

    # ------------------------------------------------------------------

    def forward(self, lr: Tensor, mu_causal: Tensor,
                residual_scale: float = 1.0) -> Tuple[Tensor, Tensor]:
        """Combine voie causale et voie directe.

        Parameters
        ----------
        lr :
            Tenseur basse résolution ``[B, C_lr, H_lr, W_lr]``.
        mu_causal :
            Sortie du chemin causal Oracle, ``[B, 1, H_hr, W_hr]``.
        residual_scale :
            Mode ``"residual"`` uniquement : facteur d'annealing appliqué à la
            correction directe. À 0, ``μ = μ_causal`` exactement.

        Returns
        -------
        mu : Tensor
            Moyenne haute résolution ``[B, 1, H_hr, W_hr]``.
        alpha : Tensor
            Gate scalaire ``[B, 1]`` ∈ [0, 1]. Mode ``"convex"`` : fraction du
            chemin causal. Mode ``"residual"`` : amplitude de la correction
            directe (sens inversé).
        """
        alpha_logit = self.alpha_mlp(lr)  # [B, 1]
        alpha = torch.sigmoid(alpha_logit)  # [B, 1] ∈ (0, 1)

        direct = self.direct(lr)  # [B, 1, H, W]
        if direct.shape != mu_causal.shape:
            direct = torch.nn.functional.interpolate(
                direct, size=mu_causal.shape[-2:],
                mode="bilinear", align_corners=False,
            )

        alpha_4d = alpha.view(-1, 1, 1, 1)
        if self.mode == "residual":
            # A4 : le mélange convexe laisse (1-α)·direct INTACT sous ablation
            # A_dag := 0 — le chemin causal est donc contournable, et le ratio
            # d'ablation est plafonné par la part directe quoi qu'il arrive.
            # En additif annelé, à residual_scale = 0 on a μ = μ_causal
            # exactement : l'interventionnabilité est restaurée par
            # construction, pas espérée d'une pénalité.
            mu = mu_causal + float(residual_scale) * alpha_4d * direct
        else:
            mu = alpha_4d * mu_causal + (1.0 - alpha_4d) * direct
        return mu, alpha

    # ------------------------------------------------------------------

    def compute_o3_preserve_loss(self, alpha: Tensor) -> Tensor:
        """Régularisation ``L_o3_preserve`` pour préserver la dominance causale.

        Pénalise les valeurs moyennes de ``α`` en-dessous de ``alpha_floor``.
        Sans cette régularisation, le gradient tirerait α vers 0 (la voie
        directe est moins contrainte) et la propriété (O3) serait perdue.

        Parameters
        ----------
        alpha :
            Tenseur ``[B, 1]`` produit par ``forward``.

        En mode ``"residual"``, α est l'amplitude de la correction directe :
        la pénalité s'inverse et retient α **sous** ``direct_ceiling``.

        Returns
        -------
        Tensor
            Scalaire à ajouter à la perte Stage 1 (multiplié par
            ``cfg.v5.skip_connection.lambda_o3_preserve`` côté appelant).
        """
        mean_alpha = alpha.mean()
        if self.mode == "residual":
            gap = torch.relu(mean_alpha - self.direct_ceiling)
        else:
            gap = torch.relu(self.alpha_floor - mean_alpha)
        return gap.pow(2)

    # ------------------------------------------------------------------

    def num_params(self) -> int:
        """Compte des paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Self-check A4. La propriete qui compte n'est pas numerique mais
    # structurelle : sous annealing complet, le chemin direct DISPARAIT, donc
    # une ablation A_dag := 0 ne peut plus etre absorbee par lui.
    torch.manual_seed(0)
    C, HR = 15, (24, 26)
    lr = torch.randn(4, C, 6, 7)
    mu_causal = torch.randn(4, 1, *HR)

    conv = ConditionalSkipBlock(C, HR, mode="convex")
    resi = ConditionalSkipBlock(C, HR, mode="residual")

    # Temoin : le mode convexe est inchange (alpha·mu + (1-alpha)·direct).
    mu_c, a_c = conv(lr, mu_causal)
    a4 = a_c.view(-1, 1, 1, 1)
    expected = a4 * mu_causal + (1 - a4) * conv.direct(lr)
    assert torch.allclose(mu_c, expected, atol=1e-5), "mode convexe altere"
    assert a_c.mean() > 0.8, f"alpha convexe doit demarrer haut, vaut {a_c.mean():.3f}"

    # A4 : a residual_scale = 0, la sortie est EXACTEMENT mu_causal.
    mu_r0, _ = resi(lr, mu_causal, residual_scale=0.0)
    assert torch.equal(mu_r0, mu_causal), (
        "annealing complet : mu doit valoir mu_causal exactement, sinon "
        "l'interventionnabilite reste plafonnee par la voie directe")
    # ... et a scale > 0 elle s'en ecarte (le module n'est pas un no-op).
    mu_r1, a_r = resi(lr, mu_causal, residual_scale=1.0)
    assert not torch.equal(mu_r1, mu_causal), "mode residuel inerte a scale=1"
    assert a_r.mean() < 0.2, (
        f"alpha residuel = amplitude de la correction directe, doit demarrer "
        f"bas, vaut {a_r.mean():.3f}")

    # La penalite change de sens avec le mode.
    hi, lo = torch.full((4, 1), 0.9), torch.full((4, 1), 0.05)
    assert conv.compute_o3_preserve_loss(hi) == 0 and conv.compute_o3_preserve_loss(lo) > 0, \
        "convexe : penaliser alpha BAS"
    assert resi.compute_o3_preserve_loss(lo) == 0 and resi.compute_o3_preserve_loss(hi) > 0, \
        "residuel : penaliser alpha HAUT"

    # Ce qui survit a l'ablation mu_causal := 0, rapporte a |direct| — sinon
    # on compare des sorties de reseaux frais et le tirage aleatoire domine.
    with torch.no_grad():
        # Chaque mode a SA propre voie directe (deux reseaux tires
        # separement) : normaliser par celle de l'autre ne comparerait rien.
        d_resi = resi.direct(lr).abs().mean()
        d_conv = conv.direct(lr).abs().mean()
        for scale in (1.0, 0.5, 0.0):
            z, _ = resi(lr, torch.zeros_like(mu_causal), residual_scale=scale)
            print(f"  residuel scale={scale:.1f} : |survivant| / |direct| = "
                  f"{float(z.abs().mean() / d_resi):.3f}   (= scale x alpha, "
                  f"alpha={float(a_r.mean()):.3f} <= {resi.direct_ceiling})")
        zc, _ = conv(lr, torch.zeros_like(mu_causal))
        print(f"  convexe (alpha={float(a_c.mean()):.2f}) : |survivant| / |direct| = "
              f"{float(zc.abs().mean() / d_conv):.3f}   (= 1 - alpha)")
        print(f"  convexe au plancher alpha={conv.alpha_floor} : "
              f"{1.0 - conv.alpha_floor:.3f}  <- valeur PLANCHER, et la penalite")
        print("     ne contraint que la MOYENNE du batch (cf. §9.7) : rien n'empeche")
        print("     alpha d'etre bas les jours convectifs et haut les jours calmes.")
    print("OK — A4 : annealing a 0 restaure l'interventionnabilite par construction.")
