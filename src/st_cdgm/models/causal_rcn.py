"""
Module 4 – Réseau causal récurrent (RCN) pour l’architecture ST-CDGM.

Ce module implémente la cellule RCN (`RCNCell`) et un utilitaire de déroulement
séquentiel (`RCNSequenceRunner`). La cellule combine un cœur causal (matrice DAG
apprenante + assignations structurelles) et une mise à jour récurrente via GRU,
suivie d’une reconstruction optionnelle pour la perte L_rec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MaskDiagonal(torch.autograd.Function):
    """
    Fonction autograd pour imposer une diagonale nulle sur la matrice DAG.
    """

    @staticmethod
    def forward(ctx, matrix: Tensor) -> Tensor:
        ctx.save_for_backward(matrix)
        out = matrix.clone()
        out.fill_diagonal_(0.0)
        return out

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor]:
        grad_input = grad_output.clone()
        grad_input.fill_diagonal_(0.0)
        return (grad_input,)


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
    ) -> None:
        super().__init__()

        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.driver_dim = driver_dim
        self.reconstruction_dim = reconstruction_dim
        self.activation = activation or nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Matrice DAG apprenable
        self.A_dag = nn.Parameter(torch.randn(num_vars, num_vars))

        # MLPs d'assignation structurelle (une par variable cible)
        self.structural_mlps = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    self.activation,
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_vars)
            ]
        )

        # Encodeur du forçage externe
        self.driver_encoder = nn.Sequential(
            nn.Linear(driver_dim, hidden_dim),
            self.activation,
        )

        # GRUCell par variable
        self.gru_cells = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_vars)]
        )
        
        # Phase A2: Vectorized GRU parameters for parallel computation
        # We extract parameters from individual GRUCells and organize them for batched operations
        # This allows vectorized computation while maintaining separate parameters per variable

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
        for mlp in self.structural_mlps:
            for layer in mlp:
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        for gru in self.gru_cells:
            gru.reset_parameters()
        for layer in self.driver_encoder:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        if self.reconstruction_decoder is not None:
            nn.init.xavier_uniform_(self.reconstruction_decoder.weight)
            nn.init.zeros_(self.reconstruction_decoder.bias)

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

        A_masked = MaskDiagonal.apply(self.A_dag)

        # Étape 1 : Prédiction interne via SCM (vectorisée)
        weighted_sum = torch.einsum("ik,inj->knj", A_masked, H_prev)
        H_hat = []
        for k in range(self.num_vars):
            h_k_hat = self.structural_mlps[k](weighted_sum[k])
            H_hat.append(h_k_hat)
        H_hat_tensor = torch.stack(H_hat, dim=0)

        # Étape 2 : Mise à jour par forçage externe (GRU)
        # Phase A2: Fully vectorized GRU computation - eliminates Python loop
        # Pre-encode driver once, reuse for all GRU cells
        driver_emb = self.driver_encoder(driver)  # [N, hidden_dim]
        
        # Phase A2: Vectorized GRU computation with separate parameters per variable
        # Extract all GRU parameters and process in a single batched operation
        # This eliminates the Python loop overhead and allows better GPU utilization
        
        # Prepare batched inputs: [q, N, hidden_dim]
        # driver_emb: [N, hidden_dim] -> expand to [q, N, hidden_dim]
        driver_batch = driver_emb.unsqueeze(0).expand(self.num_vars, -1, -1)  # [q, N, hidden_dim]
        hidden_batch = H_hat_tensor  # [q, N, hidden_dim]
        
        # Vectorized GRU computation for all variables in parallel
        # Each GRU cell has separate parameters, so we batch the computation manually
        # by extracting weights and doing batched matrix operations
        
        # Extract weights from all GRU cells and stack them: [q, ...]
        weight_ih_list = []
        weight_hh_list = []
        bias_ih_list = []
        bias_hh_list = []
        
        for k in range(self.num_vars):
            gru_cell = self.gru_cells[k]
            # GRUCell weights: [3*hidden_dim, hidden_dim] (for input->reset/update/new)
            weight_ih_list.append(gru_cell.weight_ih)  # [3*hidden_dim, hidden_dim]
            weight_hh_list.append(gru_cell.weight_hh)  # [3*hidden_dim, hidden_dim]
            bias_ih_list.append(gru_cell.bias_ih if gru_cell.bias_ih is not None else torch.zeros(3*hidden_dim, device=driver_emb.device))
            bias_hh_list.append(gru_cell.bias_hh if gru_cell.bias_hh is not None else torch.zeros(3*hidden_dim, device=driver_emb.device))
        
        # Stack all parameters: [q, 3*hidden_dim, hidden_dim]
        W_ih_batch = torch.stack(weight_ih_list, dim=0)  # [q, 3*hidden_dim, hidden_dim]
        W_hh_batch = torch.stack(weight_hh_list, dim=0)  # [q, 3*hidden_dim, hidden_dim]
        b_ih_batch = torch.stack(bias_ih_list, dim=0)  # [q, 3*hidden_dim]
        b_hh_batch = torch.stack(bias_hh_list, dim=0)  # [q, 3*hidden_dim]
        
        # Batched matrix operations
        # Input transformation: [q, N, hidden_dim] @ [q, hidden_dim, 3*hidden_dim] -> [q, N, 3*hidden_dim]
        gi_batch = torch.bmm(driver_batch, W_ih_batch.transpose(1, 2)) + b_ih_batch.unsqueeze(1)  # [q, N, 3*hidden_dim]
        gh_batch = torch.bmm(hidden_batch, W_hh_batch.transpose(1, 2)) + b_hh_batch.unsqueeze(1)  # [q, N, 3*hidden_dim]
        
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
            # Use the hidden state H_prev as reconstruction source
            # H_prev has shape [num_vars, N, hidden_dim]
            # Reshape to [N, num_vars * hidden_dim] for the decoder
            recon_input = H_prev.permute(1, 0, 2).reshape(N, -1)
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
        return MaskDiagonal.apply(self.A_dag) if masked else self.A_dag

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


