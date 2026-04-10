"""
Module 4 – Réseau causal récurrent (RCN) pour l’architecture ST-CDGM.

Ce module implémente la cellule RCN (`RCNCell`) et un utilitaire de déroulement
séquentiel (`RCNSequenceRunner`). La cellule combine un cœur causal (matrice DAG
apprenante + assignations structurelles) et une mise à jour récurrente via GRU,
suivie d’une reconstruction optionnelle pour la perte L_rec.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _mask_diagonal(matrix: Tensor) -> Tensor:
    """
    Renvoie ``matrix`` avec la diagonale forcée à zéro, en restant différentiable.

    Remplace l'ancienne ``MaskDiagonal(torch.autograd.Function)`` qui clonait
    + ``fill_diagonal_`` et créait une allocation par step. La forme actuelle
    est in-graph (autograd suit naturellement) et plus rapide sur CPU.
    """
    return matrix - torch.diag(torch.diagonal(matrix))


class RCNCell(nn.Module):
    """
    Cellule du réseau causal récurrent.

    Parameters
    ----------
    num_vars :
        Nombre de variables intelligibles (q).
    hidden_dim :
        Dimension de l'état caché par variable.
    driver_dim :
        Dimension du forçage externe (features LR).
    reconstruction_dim :
        Dimension de la reconstruction (optionnel). Si None, la reconstruction est omise.
    activation :
        Fonction d'activation utilisée dans les MLPs d'assignation structurelle.
    dropout :
        Dropout appliqué sur les sorties GRU pour régularisation.
    """

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        driver_dim: int,
        *,
        reconstruction_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
        detach_dag_in_state: bool = True,
    ) -> None:
        super().__init__()

        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.driver_dim = driver_dim
        self.reconstruction_dim = reconstruction_dim
        self.activation = activation or nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Phase DAG-decouple: the diffusion loss must NOT pull on A_dag via the
        # state/conditioning path. When True (default), the copy of A used to
        # build the SCM prediction is detached, so A_dag is trained *only* by
        # L_dag (and through L_rec, which flows via H_prev). The returned
        # ``A_masked`` stays attached so those losses still propagate.
        self.detach_dag_in_state = detach_dag_in_state

        # Matrice DAG apprenable
        self.A_dag = nn.Parameter(torch.randn(num_vars, num_vars))

        # Phase B-perf: MLPs d'assignation structurelle vectorisés.
        # Anciennement ``nn.ModuleList[nn.Sequential]`` itéré dans une boucle
        # Python — coûteux sur CPU. Stockés ici comme paramètres batchés
        # ``[q, hidden, hidden]`` pour appliquer l'ensemble des q MLPs en deux
        # ``bmm``. Sémantique préservée : 2 couches Linear + activation entre.
        self.struct_W1 = nn.Parameter(torch.empty(num_vars, hidden_dim, hidden_dim))
        self.struct_b1 = nn.Parameter(torch.empty(num_vars, 1, hidden_dim))
        self.struct_W2 = nn.Parameter(torch.empty(num_vars, hidden_dim, hidden_dim))
        self.struct_b2 = nn.Parameter(torch.empty(num_vars, 1, hidden_dim))

        # Encodeur du forçage externe
        self.driver_encoder = nn.Sequential(
            nn.Linear(driver_dim, hidden_dim),
            self.activation,
        )

        # Phase B-perf: GRU vectorisé. Au lieu de stacker à chaque forward les
        # poids de q ``nn.GRUCell``, on stocke directement des paramètres
        # batchés ``[q, 3*hidden, hidden]`` (et biais ``[q, 3*hidden]``).
        # L'init reproduit celle de ``nn.GRUCell`` (uniform[-k, k] avec
        # k = 1/sqrt(hidden_dim)) pour préserver la dynamique d'entraînement.
        self.gru_W_ih = nn.Parameter(torch.empty(num_vars, 3 * hidden_dim, hidden_dim))
        self.gru_W_hh = nn.Parameter(torch.empty(num_vars, 3 * hidden_dim, hidden_dim))
        self.gru_b_ih = nn.Parameter(torch.empty(num_vars, 3 * hidden_dim))
        self.gru_b_hh = nn.Parameter(torch.empty(num_vars, 3 * hidden_dim))

        # Embedding d'identité par variable — spécialise le driver pour chaque GRU
        self.var_embed = nn.Embedding(num_vars, hidden_dim)

        # Décodeur de reconstruction optionnel
        if self.reconstruction_dim is not None:
            self.reconstruction_decoder = nn.Linear(
                num_vars * hidden_dim, self.reconstruction_dim
            )
        else:
            self.reconstruction_decoder = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Réinitialise les paramètres internes.
        """
        nn.init.xavier_uniform_(self.A_dag)

        # Structural MLPs vectorisés : init Xavier uniforme couche par couche.
        for w in (self.struct_W1, self.struct_W2):
            for k in range(self.num_vars):
                nn.init.xavier_uniform_(w[k])
        nn.init.zeros_(self.struct_b1)
        nn.init.zeros_(self.struct_b2)

        # GRU vectorisé : reproduit l'init de ``nn.GRUCell``
        # (uniform[-k, k] avec k = 1/sqrt(hidden_dim)).
        gru_bound = 1.0 / math.sqrt(self.hidden_dim)
        for p in (self.gru_W_ih, self.gru_W_hh, self.gru_b_ih, self.gru_b_hh):
            nn.init.uniform_(p, -gru_bound, gru_bound)

        for layer in self.driver_encoder:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        nn.init.normal_(self.var_embed.weight, std=0.02)
        if self.reconstruction_decoder is not None:
            nn.init.xavier_uniform_(self.reconstruction_decoder.weight)
            nn.init.zeros_(self.reconstruction_decoder.bias)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Backward-compat shim : convertit en place les clés des anciens checkpoints
        ``structural_mlps.k.{0,2}.{weight,bias}`` et ``gru_cells.k.{weight_ih,
        weight_hh, bias_ih, bias_hh}`` vers les nouveaux paramètres batchés
        ``struct_W{1,2}`` / ``struct_b{1,2}`` / ``gru_{W,b}_{ih,hh}``.

        Permet de reprendre un checkpoint pré-vectorisation sans re-train.
        """
        # Détection de l'ancien format
        old_struct_keys = [k for k in state_dict if k.startswith(prefix + "structural_mlps.")]
        old_gru_keys = [k for k in state_dict if k.startswith(prefix + "gru_cells.")]
        if old_struct_keys or old_gru_keys:
            q = self.num_vars
            d = self.hidden_dim
            device = self.A_dag.device

            if old_struct_keys:
                W1 = torch.empty(q, d, d, device=device)
                W2 = torch.empty(q, d, d, device=device)
                b1 = torch.empty(q, 1, d, device=device)
                b2 = torch.empty(q, 1, d, device=device)
                for k in range(q):
                    # nn.Linear stocke weight en [out, in]; notre bmm veut [in, out].
                    W1[k] = state_dict.pop(f"{prefix}structural_mlps.{k}.0.weight").t()
                    b1[k, 0] = state_dict.pop(f"{prefix}structural_mlps.{k}.0.bias")
                    W2[k] = state_dict.pop(f"{prefix}structural_mlps.{k}.2.weight").t()
                    b2[k, 0] = state_dict.pop(f"{prefix}structural_mlps.{k}.2.bias")
                state_dict[f"{prefix}struct_W1"] = W1
                state_dict[f"{prefix}struct_b1"] = b1
                state_dict[f"{prefix}struct_W2"] = W2
                state_dict[f"{prefix}struct_b2"] = b2

            if old_gru_keys:
                W_ih = torch.empty(q, 3 * d, d, device=device)
                W_hh = torch.empty(q, 3 * d, d, device=device)
                b_ih = torch.empty(q, 3 * d, device=device)
                b_hh = torch.empty(q, 3 * d, device=device)
                for k in range(q):
                    W_ih[k] = state_dict.pop(f"{prefix}gru_cells.{k}.weight_ih")
                    W_hh[k] = state_dict.pop(f"{prefix}gru_cells.{k}.weight_hh")
                    b_ih[k] = state_dict.pop(f"{prefix}gru_cells.{k}.bias_ih")
                    b_hh[k] = state_dict.pop(f"{prefix}gru_cells.{k}.bias_hh")
                state_dict[f"{prefix}gru_W_ih"] = W_ih
                state_dict[f"{prefix}gru_W_hh"] = W_hh
                state_dict[f"{prefix}gru_b_ih"] = b_ih
                state_dict[f"{prefix}gru_b_hh"] = b_hh

        return super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        H_prev: Tensor,
        driver: Tensor,
        reconstruction_source: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """
        Applique une mise à jour récurrente.

        Parameters
        ----------
        H_prev :
            État précédent [q, N, hidden_dim].
        driver :
            Forçage externe [N, driver_dim].
        reconstruction_source :
            Tenseur utilisé comme base pour la reconstruction (par défaut ``H_prev``).

        Returns
        -------
        H_next :
            Nouvel état [q, N, hidden_dim].
        reconstruction :
            Reconstruction optionnelle des features [N, reconstruction_dim].
        A_masked :
            Matrice DAG avec diagonale masquée (pour L_dag).
        """
        if H_prev.dim() != 3:
            raise ValueError("H_prev doit avoir la forme [q, N, hidden_dim].")
        q, N, hidden_dim = H_prev.shape
        if q != self.num_vars or hidden_dim != self.hidden_dim:
            raise ValueError("Dimensions de H_prev incompatibles avec la cellule RCN.")
        if driver.shape[0] != N:
            raise ValueError("driver doit partager la dimension N avec H_prev.")
        if driver.shape[1] != self.driver_dim:
            raise ValueError("Dimension du driver incompatible.")

        A_masked = _mask_diagonal(self.A_dag)

        # Phase DAG-decouple: isolate the copy of A that feeds the state.
        # `A_masked` (attached) is returned for L_dag / L_rec; `A_for_state`
        # carries no gradient to `A_dag`, so the diffusion loss cannot pull
        # the DAG toward representations that help generation but break
        # acyclicity / physical interpretation.
        A_for_state = A_masked.detach() if self.detach_dag_in_state else A_masked

        # Étape 1 : Prédiction interne via SCM, entièrement vectorisée.
        # weighted_sum : [q, N, hidden]
        weighted_sum = torch.einsum("ik,inj->knj", A_for_state, H_prev)
        # MLP structurel batché : 2 ``bmm`` au lieu d'une boucle Python sur q.
        H_hat_tensor = torch.bmm(weighted_sum, self.struct_W1) + self.struct_b1  # [q, N, hidden]
        H_hat_tensor = self.activation(H_hat_tensor)
        H_hat_tensor = torch.bmm(H_hat_tensor, self.struct_W2) + self.struct_b2  # [q, N, hidden]

        # Étape 2 : GRU vectorisé. Plus de stack de poids à chaque step :
        # les paramètres sont déjà ``[q, 3*hidden, hidden]``.
        driver_emb = self.driver_encoder(driver)  # [N, hidden_dim]
        driver_batch = driver_emb.unsqueeze(0).expand(self.num_vars, -1, -1)  # [q, N, hidden]
        var_ids = torch.arange(self.num_vars, device=driver_emb.device)
        driver_batch = driver_batch + self.var_embed(var_ids).unsqueeze(1)    # [q, N, hidden]
        hidden_batch = H_hat_tensor

        # gi/gh : [q, N, 3*hidden]
        gi_batch = torch.bmm(driver_batch, self.gru_W_ih.transpose(1, 2)) + self.gru_b_ih.unsqueeze(1)
        gh_batch = torch.bmm(hidden_batch, self.gru_W_hh.transpose(1, 2)) + self.gru_b_hh.unsqueeze(1)
        
        # Split into reset, update, and new gates
        # GRU gates are ordered as: reset, update, new
        i_r, i_u, i_n = gi_batch.chunk(3, dim=2)  # Each: [q, N, hidden_dim]
        h_r, h_u, h_n = gh_batch.chunk(3, dim=2)  # Each: [q, N, hidden_dim]
        
        # GRU computations (all vectorized)
        reset_gate = torch.sigmoid(i_r + h_r)  # [q, N, hidden_dim]
        update_gate = torch.sigmoid(i_u + h_u)  # [q, N, hidden_dim]
        new_gate = torch.tanh(i_n + reset_gate * h_n)  # [q, N, hidden_dim]
        
        # Final hidden state: h_new = (1 - z) * n + z * h
        H_next_tensor = (1 - update_gate) * new_gate + update_gate * hidden_batch  # [q, N, hidden_dim]
        
        # Apply dropout in vectorized manner
        H_next_tensor = self.dropout(H_next_tensor)

        # Reconstruction facultative
        reconstruction = None
        if self.reconstruction_decoder is not None:
            # Use provided reconstruction source or default to H_prev
            source = reconstruction_source if reconstruction_source is not None else H_prev
            
            # Ensure source has shape [N, num_vars * hidden_dim]
            # If source is [q, N, hidden_dim] (like H_prev), permute and reshape
            if source.dim() == 3 and source.shape[0] == self.num_vars and source.shape[2] == self.hidden_dim:
                recon_input = source.permute(1, 0, 2).reshape(N, -1)
            elif source.dim() == 2:
                # Assume already flattened or correct shape [N, features]
                recon_input = source
            else:
                # Fallback reshape trying to preserve N
                recon_input = source.reshape(N, -1)
                
            reconstruction = self.reconstruction_decoder(recon_input)

        return H_next_tensor, reconstruction, A_masked

    def pool_state(
        self,
        H: Tensor,
        *,
        batch: Optional[Tensor] = None,
        pool: str = "mean",
    ) -> Tensor:
        """
        Agrège l'état caché sur la dimension spatiale.

        Parameters
        ----------
        H :
            Tenseur [q, N, hidden_dim].
        batch :
            Vecteur d'indices de graphe pour chaque nœud (longueur N).
        pool :
            ``"mean"`` (défaut) ou ``"max"``.

        Returns
        -------
        Tensor
            Tenseur [num_graphs, q, hidden_dim].
        """
        pool = pool.lower()
        if batch is None:
            if pool == "mean":
                return H.mean(dim=1, keepdim=False).unsqueeze(0)
            if pool == "max":
                return H.amax(dim=1, keepdim=False).unsqueeze(0)
            raise ValueError(f"Pooling '{pool}' non pris en charge sans batch.")

        num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        pooled: List[Tensor] = []
        for g in range(num_graphs):
            mask = batch == g
            if not torch.any(mask):
                pooled.append(torch.zeros(self.num_vars, self.hidden_dim, device=H.device, dtype=H.dtype))
                continue
            if pool == "mean":
                pooled.append(H[:, mask, :].mean(dim=1))
            elif pool == "max":
                pooled.append(H[:, mask, :].amax(dim=1))
            else:
                raise ValueError(f"Pooling '{pool}' non pris en charge.")
        return torch.stack(pooled, dim=0)

    def dag_matrix(self, *, masked: bool = True) -> Tensor:
        """
        Retourne la matrice causale (masquée ou brute).
        """
        return _mask_diagonal(self.A_dag) if masked else self.A_dag

    @torch.no_grad()
    def project_dag_spectral(self, max_radius: float = 0.95) -> float:
        """
        Hard acyclicity projection: scale ``A_dag`` in place so that
        ``||A ∘ A||`` has spectral radius strictly below 1 (default 0.95).

        This is a **hard** constraint applied after each ``optimizer.step()``
        and therefore never contributes to the loss / backward pass. It lets
        us keep ``gamma_dag`` small (so DAGMA does not fight the diffusion
        objective in gradient space) while still guaranteeing the learned
        matrix stays on the acyclic cone — the key property of the DAGMA
        M-matrix characterisation used by ``loss_dagma``.

        Returns the scale factor applied (1.0 if no rescaling was needed),
        mostly for logging.
        """
        A = self.A_dag.data
        # Keep the diagonal strictly zero: small numerical drift during FP
        # updates is harmless but pollutes the spectral-radius bound.
        A.fill_diagonal_(0.0)
        W_sq = A * A
        # Gershgorin bound: rho(W²) <= max row-sum of |W²| = max row-sum of W²
        # (all entries non-negative). This is tight and O(q²), matching the
        # bound used inside ``loss_dagma``.
        row_sum_max = float(W_sq.sum(dim=1).max().item())
        if row_sum_max <= max_radius:
            return 1.0
        # We want rho(W²') <= max_radius where W' = alpha * W, and
        # rho(alpha² * W²) = alpha² * rho(W²) <= max_radius
        #   => alpha = sqrt(max_radius / rho(W²))
        alpha = math.sqrt(max_radius / (row_sum_max + 1e-12))
        A.mul_(alpha)
        return alpha

    def _prepare_reconstruction_features(self, tensor: Tensor, N: int) -> Tensor:
        """
        Mise en forme standard des caractéristiques utilisées pour la reconstruction.
        """
        if tensor.dim() == 2:
            if tensor.size(0) != N:
                raise ValueError(
                    f"reconstruction_source attend {N} nœuds, obtenu {tensor.size(0)}."
                )
            return tensor
        if tensor.dim() == 3:
            if tensor.size(1) != N:
                raise ValueError(
                    f"reconstruction_source attend {N} nœuds sur la dimension 1, obtenu {tensor.size(1)}."
                )
            return tensor.permute(1, 0, 2).reshape(N, -1)
        raise ValueError("reconstruction_source doit être de dimension 2 ou 3.")


@dataclass
class RCNSequenceOutput:
    """
    Résultats du déroulement séquentiel de la cellule RCN.
    """

    states: List[Tensor]
    reconstructions: List[Optional[Tensor]]
    dag_matrices: List[Tensor]


class RCNSequenceRunner:
    """
    Utilitaire pour dérouler la cellule RCN sur des séquences temporelles.
    """

    def __init__(self, cell: RCNCell, *, detach_interval: Optional[int] = None) -> None:
        self.cell = cell
        self.detach_interval = detach_interval

    def run(
        self,
        H_init: Tensor,
        drivers: Sequence[Tensor],
        reconstruction_sources: Optional[Sequence[Optional[Tensor]]] = None,
    ) -> RCNSequenceOutput:
        """
        Déroule la cellule sur la séquence de drivers.

        Parameters
        ----------
        H_init :
            État initial [q, N, hidden_dim].
        drivers :
            Séquence de tenseurs [N, driver_dim] de longueur T.
        reconstruction_sources :
            Séquence optionnelle alignée sur ``drivers`` contenant les tenseurs
            utilisés pour la reconstruction (ex: features LR à reconstruire).
        """
        H_t = H_init
        states: List[Tensor] = []
        reconstructions: List[Optional[Tensor]] = []
        dag_matrices: List[Tensor] = []

        for t, driver in enumerate(drivers):
            recon_source = None
            if reconstruction_sources is not None:
                recon_source = reconstruction_sources[t]
            H_next, recon, A_masked = self.cell(H_t, driver, reconstruction_source=recon_source)
            states.append(H_next)
            reconstructions.append(recon)
            dag_matrices.append(A_masked)

            if self.detach_interval is not None and (t + 1) % self.detach_interval == 0:
                H_t = H_next.detach()
            else:
                H_t = H_next

        return RCNSequenceOutput(states=states, reconstructions=reconstructions, dag_matrices=dag_matrices)


