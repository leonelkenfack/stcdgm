"""
Regression head for ST-CDGM Two-Stage Causal Architecture.

Maps the RCN causal latent state ``H_T ∈ ℝ^{[B, q, N, d]}`` to the
high-resolution residual mean ``μ_HR ∈ ℝ^{[B, 1, H_HR, W_HR]}``.

Design (per Gemini Pro Deep Research validated 2026-05-02, hyperplan v2.0)
---------------------------------------------------------------------------

The decoder is **structurally load-bearing** for the causal DAG: its only
input is ``H_T`` which depends on ``A_dag`` through the RCN dynamics. Setting
``A_dag := 0`` changes ``H_T`` and therefore changes ``μ_HR`` — there is no
alternative input path. This guarantees ``Objective O6 (intervention)`` of
the paper at the architecture level, not merely as an optimisation outcome.

Memory considerations
---------------------

A naive cross-attention from H_T tokens (≈3000 tokens) to a 172×179 query
grid (≈30k queries) at d=128, 4 heads, batch=16 produces ~24 GB of attention
weights — would not fit on A100 80GB.

We therefore cross-attend at an *intermediate* resolution (``intermediate_h``,
``intermediate_w``), and bilinearly upsample + refine via a shallow CNN to
the full HR grid. Default intermediate is 43×45 ≈ 2× LR resolution. Memory
budget: ~1.5 GB for attention scores at batch 16. Total decoder param count
is ~670 k — strictly separate from the diffusion U-Net.

The grid queries are learnable parameters (a positional embedding) so the
decoder is free to discover its own coordinate system that best aligns with
the H_T topology.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


def _sincos_2d(h: int, w: int, d_model: int, scale: float = 100.0) -> Tensor:
    """Encodage positionnel sinusoïdal 2-D, ``[h*w, d_model]``.

    Les positions sont normalisées sur ``[0, scale]`` **indépendamment de la
    taille de la grille** : la ligne 11 d'une grille LR 23×26 et la ligne 21
    d'une grille 43×45 reçoivent le même code. C'est la condition pour que
    requêtes et clés, définies sur deux grilles de résolutions différentes
    mais couvrant le MÊME domaine physique, partagent un système de
    coordonnées — sans quoi l'attention ne peut pas exploiter la proximité
    géométrique.
    """
    if d_model % 4 != 0:
        raise ValueError(f"d_model={d_model} doit être divisible par 4 pour le PE 2-D")
    d4 = d_model // 4
    omega = torch.exp(torch.arange(d4, dtype=torch.float32)
                      * (-math.log(10000.0) / max(d4 - 1, 1)))
    ys = (torch.arange(h, dtype=torch.float32) + 0.5) / h * scale
    xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w * scale
    ay = ys[:, None] * omega[None, :]                       # [h, d4]
    ax = xs[:, None] * omega[None, :]                       # [w, d4]
    ey = torch.cat([ay.sin(), ay.cos()], dim=1)             # [h, 2*d4]
    ex = torch.cat([ax.sin(), ax.cos()], dim=1)             # [w, 2*d4]
    pe = torch.cat([ey[:, None, :].expand(h, w, 2 * d4),
                    ex[None, :, :].expand(h, w, 2 * d4)], dim=2)
    return pe.reshape(h * w, d_model)


class GraphToGridDecoder(nn.Module):
    """Cross-attention decoder mapping causal graph state to HR grid.

    Pipeline:
      1. Flatten ``H_T`` to a sequence of ``q × N`` tokens, each in
         ``ℝ^{d_model}``.
      2. Cross-attention: queries are a learnable
         ``[intermediate_h × intermediate_w, d_model]`` grid; keys and
         values are the H_T tokens.
      3. Reshape attention output to ``[d_model, intermediate_h,
         intermediate_w]`` and refine via a shallow CNN with bilinear
         upsampling to the target HR resolution.

    Parameters
    ----------
    d_model : int
        Hidden dimension of H_T (must match ``encoder.hidden_dim``).
    hr_h, hr_w : int
        Target HR grid dimensions (e.g. 172, 179 for the NZ domain).
    intermediate_h, intermediate_w : int
        Intermediate grid for cross-attention. Memory-quality trade-off.
        Default 43×45 ≈ 2× LR resolution → 1935 queries.
    n_heads : int
        Number of attention heads. ``d_model`` must be divisible by it.
    refine_channels : int
        Number of channels in the first refinement conv. Halves at each
        upsampling level until the final 1-channel projection.
    output_channels : int
        Output channels (1 for precipitation residual mean).
    """

    def __init__(
        self,
        d_model: int = 128,
        hr_h: int = 172,
        hr_w: int = 179,
        intermediate_h: int = 43,
        intermediate_w: int = 45,
        n_heads: int = 4,
        refine_channels: int = 64,
        output_channels: int = 1,
        # >>> V8 — A2a (docs/architecture_v8_design.md §9.3)
        query_mode: str = "learned",
        lr_h: int = 23,
        lr_w: int = 26,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}"
            )
        if query_mode not in ("learned", "spatial"):
            raise ValueError(f"query_mode inconnu : {query_mode!r}")

        self.d_model = d_model
        self.query_mode = query_mode
        self.lr_h = lr_h
        self.lr_w = lr_w
        self.hr_h = hr_h
        self.hr_w = hr_w
        self.intermediate_h = intermediate_h
        self.intermediate_w = intermediate_w
        self.output_channels = output_channels

        # Learnable query grid at intermediate resolution.
        # Initialised small so cross-attention starts close to identity in
        # output statistics — avoids early instability.
        n_queries = intermediate_h * intermediate_w
        self.grid_queries = nn.Parameter(
            torch.randn(1, n_queries, d_model) * 0.02
        )

        # >>> V8 — A2a. Mesuré (§9.3) : le plafond de la grille 43×45 est 0,750
        # et le modèle plafonne à 0,512 — l'écart n'est donc PAS la résolution.
        # En mode "learned", ni les requêtes ni les clés ne portent de position :
        # l'attention doit apprendre 1935 × (q·N) affinités par le seul contenu,
        # alors que H_T est DÉJÀ un champ spatial sur la grille LR. En mode
        # "spatial" les requêtes sont amorcées par l'état lui-même rééchantillonné
        # (donc dépendantes de l'entrée) et les deux côtés reçoivent le même
        # encodage positionnel : l'attention devient un raffinement géométrique
        # local au lieu d'une recherche globale par contenu.
        if query_mode == "spatial":
            self.register_buffer(
                "pe_query",
                _sincos_2d(intermediate_h, intermediate_w, d_model).unsqueeze(0),
                persistent=False,
            )
            self.register_buffer(
                "pe_key",
                _sincos_2d(lr_h, lr_w, d_model).unsqueeze(0),
                persistent=False,
            )

        # Cross-attention: queries attend to H_T tokens.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

        # Choose an interim spatial step ~halfway between intermediate and HR.
        # 43→86, 45→90 ≈ HR/2; bilinear keeps it parameterless.
        interim_h = (hr_h // 2) + (hr_h % 2)
        interim_w = (hr_w // 2) + (hr_w % 2)

        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, refine_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Upsample(size=(interim_h, interim_w), mode="bilinear", align_corners=False),
            nn.Conv2d(
                refine_channels, refine_channels // 2, kernel_size=3, padding=1
            ),
            nn.GELU(),
            nn.Upsample(size=(hr_h, hr_w), mode="bilinear", align_corners=False),
            nn.Conv2d(refine_channels // 2, output_channels, kernel_size=3, padding=1),
        )
        # Largeur des features juste avant la projection finale — c'est ce que
        # consomme une tête probabiliste (BernoulliGammaHead) branchée dessus.
        self.feature_channels = refine_channels // 2

    def forward(self, H_T: Tensor, return_features: bool = False) -> Tensor:
        """Apply cross-attention + CNN refinement.

        Parameters
        ----------
        H_T : Tensor
            Causal latent state. Accepted shapes:
            ``[q, N, d_model]`` (single sample, RCN native output) or
            ``[B, q, N, d_model]`` (batched).
        return_features : bool
            Return the HR features *before* the final projection instead of
            ``μ_HR``, so a probabilistic head can consume them. Shape
            ``[B, feature_channels, hr_h, hr_w]``.

        Returns
        -------
        Tensor
            ``μ_HR`` of shape ``[B, output_channels, hr_h, hr_w]``.
            Always batched, even when input was a single sample.
        """
        if H_T.dim() == 3:
            H_T = H_T.unsqueeze(0)
        if H_T.dim() != 4:
            raise ValueError(
                f"GraphToGridDecoder expects H_T with 3 or 4 dims, "
                f"got {H_T.dim()} dims of shape {tuple(H_T.shape)}"
            )

        B, q, N, d = H_T.shape
        if d != self.d_model:
            raise ValueError(
                f"H_T last dim {d} != d_model {self.d_model}"
            )

        # Flatten q × N into token sequence: [B, q*N, d_model]
        tokens = H_T.reshape(B, q * N, d)

        if self.query_mode == "spatial":
            if N != self.lr_h * self.lr_w:
                raise ValueError(
                    f"query_mode='spatial' suppose des tokens posés sur la grille "
                    f"LR {self.lr_h}×{self.lr_w}={self.lr_h * self.lr_w}, "
                    f"or N={N}. Corriger lr_h/lr_w."
                )
            # Amorce dépendante de l'entrée : l'état moyenné sur les q variables
            # est déjà un champ [B, d, lr_h, lr_w] — on le rééchantillonne sur la
            # grille des requêtes. Le contenu par variable, lui, arrive par
            # l'attention sur les clés.
            spatial = H_T.mean(dim=1).transpose(1, 2).reshape(
                B, d, self.lr_h, self.lr_w
            )
            seed = torch.nn.functional.interpolate(
                spatial, size=(self.intermediate_h, self.intermediate_w),
                mode="bilinear", align_corners=False,
            )
            queries = seed.flatten(2).transpose(1, 2) + self.pe_query
            keys = tokens + self.pe_key.repeat(1, q, 1)
        else:
            queries = self.grid_queries.expand(B, -1, -1)
            keys = tokens

        # Cross-attention; attention scores shape [B, n_heads, n_queries, q*N]
        # are released as soon as cross_attn returns — only attn_out is kept.
        # Les valeurs restent SANS encodage positionnel : le PE sert à décider
        # où regarder, pas à polluer ce qui est transporté.
        attn_out, _ = self.cross_attn(query=queries, key=keys, value=tokens)

        # Residual + LayerNorm
        grid_features = self.norm(queries + attn_out)

        # Reshape to spatial grid: [B, d_model, intermediate_h, intermediate_w]
        grid_features = grid_features.transpose(1, 2).contiguous().view(
            B, self.d_model, self.intermediate_h, self.intermediate_w
        )

        # CNN refine + upsample to HR. Slicing the Sequential shares the same
        # modules, so no state_dict key changes and no duplicated parameters.
        if return_features:
            return self.upsample[:-1](grid_features)
        return self.upsample(grid_features)

    def num_params(self) -> int:
        """Return total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


__all__ = ["GraphToGridDecoder"]


if __name__ == "__main__":
    # Self-check A2a. Ce qui doit tenir : (1) requetes et cles partagent un
    # systeme de coordonnees malgre deux grilles de tailles differentes ;
    # (2) les requetes dependent vraiment de l'entree ; (3) le gradient passe.
    torch.manual_seed(0)
    LRH, LRW, IH, IW, D = 23, 26, 43, 45, 64

    # (1) Le PE doit apparier les MEMES positions physiques. La ligne centrale
    # de la grille LR doit ressembler le plus a la ligne centrale de la grille
    # intermediaire, pas a une ligne quelconque.
    pk = _sincos_2d(LRH, LRW, D).reshape(LRH, LRW, D)
    pq = _sincos_2d(IH, IW, D).reshape(IH, IW, D)
    for lr_row in (0, LRH // 2, LRH - 1):
        ref = pk[lr_row, LRW // 2]
        cand = pq[:, IW // 2]
        sim = torch.nn.functional.cosine_similarity(cand, ref[None, :], dim=1)
        best = int(sim.argmax())
        expected = int((lr_row + 0.5) / LRH * IH)
        assert abs(best - expected) <= 2, (
            f"PE desaligne : ligne LR {lr_row}/{LRH} appariee a {best}/{IH}, "
            f"attendu ~{expected}")
    print("OK — PE partage entre grilles LR et intermediaire")

    H_T = torch.randn(2, 11, LRH * LRW, D)
    common = dict(d_model=D, hr_h=172, hr_w=179, intermediate_h=IH,
                  intermediate_w=IW, n_heads=4, lr_h=LRH, lr_w=LRW)
    dec_l = GraphToGridDecoder(query_mode="learned", **common)
    dec_s = GraphToGridDecoder(query_mode="spatial", **common)
    assert dec_l(H_T).shape == (2, 1, 172, 179)
    assert dec_s(H_T).shape == (2, 1, 172, 179)

    # (2) Dependance a l'entree : en mode "learned" les requetes sont un
    # parametre fige, identiques quelle que soit l'entree ; en mode "spatial"
    # elles doivent bouger avec H_T. C'est TOUT l'objet de A2a.
    def queries_of(dec, x):
        B, q, N, d = x.shape
        if dec.query_mode == "learned":
            return dec.grid_queries.expand(B, -1, -1)
        sp = x.mean(dim=1).transpose(1, 2).reshape(B, d, dec.lr_h, dec.lr_w)
        seed = torch.nn.functional.interpolate(
            sp, size=(dec.intermediate_h, dec.intermediate_w),
            mode="bilinear", align_corners=False)
        return seed.flatten(2).transpose(1, 2) + dec.pe_query

    H_T2 = torch.randn_like(H_T)
    assert torch.allclose(queries_of(dec_l, H_T), queries_of(dec_l, H_T2)), \
        "mode 'learned' : les requetes ne doivent PAS bouger (temoin)"
    dq = (queries_of(dec_s, H_T) - queries_of(dec_s, H_T2)).abs().mean()
    assert dq > 1e-3, f"mode 'spatial' : requetes insensibles a l'entree ({dq:.2e})"
    print(f"OK — requetes dependantes de l'entree (ecart moyen {dq:.3f})")

    # (3) Gradient jusqu'aux poids d'attention, et features exposees a la tete.
    out = dec_s(H_T, return_features=True)
    assert out.shape[1] == dec_s.feature_channels
    dec_s(H_T).sum().backward()
    g = dec_s.cross_attn.in_proj_weight.grad
    assert g is not None and g.abs().sum() > 0, "gradient bloque dans l'attention"

    # Garde-fou : N incoherent avec lr_h/lr_w doit echouer bruyamment, pas
    # produire un reshape silencieusement faux.
    try:
        dec_s(torch.randn(1, 11, 600, D))
        raise SystemExit("un N incoherent aurait du lever ValueError")
    except ValueError:
        pass
    print("OK — gradient, features et garde sur la forme des tokens")
