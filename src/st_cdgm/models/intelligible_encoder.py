"""
Module 3 – Encodeur de variables intelligibles via HeteroConv.

Ce module fournit une classe `IntelligibleVariableEncoder` qui agrège les
informations d'un `HeteroData` en suivant différents méta-chemins (advection,
convection, influence statique, etc.) afin de produire un état caché initial
`H(0)` pour le réseau causal récurrent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import (
    HeteroConv,
    MessagePassing,
    SAGEConv,
    global_max_pool,
    global_mean_pool,
)


MetaPath = Tuple[str, str, str]


@dataclass
class IntelligibleVariableConfig:
    """
    Configuration d'une variable intelligible.

    Attributes
    ----------
    name :
        Nom de la variable intelligible (ex: 'h_advection').
    meta_path :
        Méta-chemin torch_geometric (source_type, relation_type, target_type).
    conv_class :
        Classe de convolution à utiliser; défaut `SAGEConv`.
    conv_kwargs :
        Paramètres additionnels passés au constructeur de la convolution.
    pool :
        Mode de pooling à appliquer en sortie ("mean", "max", None) lors du
        calcul des états agrégés pour le conditionnement.
    """

    name: str
    meta_path: MetaPath
    conv_class: type[MessagePassing] = SAGEConv
    conv_kwargs: Optional[Dict] = None
    pool: Optional[str] = None


class IntelligibleVariableEncoder(nn.Module):
    """
    Encodeur HeteroConv produisant les variables intelligibles pour H(0).
    """

    def __init__(
        self,
        configs: Iterable[IntelligibleVariableConfig],
        hidden_dim: int,
        *,
        activation: Optional[nn.Module] = None,
        use_layer_norm: bool = True,
        default_pool: str = "mean",
        conditioning_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation or nn.ReLU()
        self.use_layer_norm = use_layer_norm
        self.default_pool = default_pool

        self.configs: List[IntelligibleVariableConfig] = list(configs)
        if not self.configs:
            raise ValueError("Au moins une configuration de variable intelligible est requise.")

        convs_dict = {}
        for cfg in self.configs:
            kwargs = {"out_channels": hidden_dim}
            if cfg.conv_class is SAGEConv:
                kwargs["in_channels"] = (-1, -1)  # auto-infer
            if cfg.conv_kwargs:
                kwargs.update(cfg.conv_kwargs)
            convs_dict[cfg.meta_path] = cfg.conv_class(**kwargs)

        self.hetero_conv = HeteroConv(convs_dict, aggr="sum")
        
        # Phase B1: Check if pyg-lib is available for Grouped GEMM optimizations
        self._check_pyg_lib_availability()

        if self.use_layer_norm:
            self.layer_norm = nn.LayerNorm(hidden_dim)
        else:
            self.layer_norm = nn.Identity()

        if conditioning_dim is None or conditioning_dim == hidden_dim:
            self.conditioning_dim = hidden_dim
            self.conditioning_projection = nn.Identity()
        else:
            self.conditioning_dim = conditioning_dim
            self.conditioning_projection = nn.Linear(hidden_dim, conditioning_dim)

    def forward(self, data: HeteroData, *, pooled: bool = False) -> Dict[str, Tensor]:
        """
        Applique l'encodeur et retourne un dict {variable_name: embeddings}.
        Si ``pooled=True``, les embeddings sont agrégés par graphe (global pooling).
        """
        x_dict = {node_type: data[node_type].x for node_type in data.node_types}
        embeddings = self.hetero_conv(x_dict, data.edge_index_dict)

        outputs: Dict[str, Tensor] = {}
        for cfg in self.configs:
            tensor = embeddings[cfg.meta_path[-1]]
            tensor = self.layer_norm(tensor)
            tensor = self.activation(tensor)

            if pooled:
                pool_type = cfg.pool or self.default_pool
                batch_attr = getattr(data[cfg.meta_path[-1]], "batch", None)
                if batch_attr is None:
                    batch_attr = torch.zeros(tensor.size(0), dtype=torch.long, device=tensor.device)
                tensor = self._apply_pooling(tensor, batch_attr, pool_type)

            outputs[cfg.name] = tensor

        return outputs

    def init_state(self, data: HeteroData) -> Tensor:
        """
        Génère l'état initial H(0) comme tenseur [q, num_nodes, hidden_dim].
        """
        embeddings = self.forward(data, pooled=False)
        tensors = [embeddings[cfg.name] for cfg in self.configs]
        aligned = []

        for tensor, cfg in zip(tensors, self.configs):
            if tensor.dim() == 2:
                aligned.append(tensor)
            elif tensor.dim() == 3:
                aligned.append(tensor.squeeze(0))
            else:
                raise ValueError(
                    f"Embedding pour {cfg.name} a une dimension inattendue: {tensor.shape}"
                )

        stacked = torch.stack(aligned, dim=0)
        return stacked

    def pooled_state(self, data: HeteroData) -> Tensor:
        """
        Retourne un tenseur [batch, q, hidden_dim] agrégé par graphe.
        """
        pooled_embeddings = self.forward(data, pooled=True)
        tensors = []
        for cfg in self.configs:
            tensor = pooled_embeddings[cfg.name]
            if tensor.dim() == 1:
                tensor = tensor.unsqueeze(0)
            tensors.append(tensor)
        stacked = torch.stack(tensors, dim=1)
        return stacked

    def project_conditioning(self, data: HeteroData) -> Tensor:
        """
        Projette l'état causal agrégé dans l'espace de conditionnement diffusion.

        Returns
        -------
        Tensor
            Tensor de forme [batch, q, conditioning_dim]
        """
        pooled = self.pooled_state(data)
        batch, q, hidden = pooled.shape
        projected = self.conditioning_projection(pooled.view(batch * q, hidden))
        projected = projected.view(batch, q, -1)
        return projected

    def project_state_tensor(
        self,
        state: Tensor,
        *,
        batch_index: Optional[Tensor] = None,
        pool: Optional[str] = None,
    ) -> Tensor:
        """
        Convertit un tenseur d'état causal H(t) en représentation cross-attention.

        Parameters
        ----------
        state :
            Tenseur de forme [q, N, hidden] (un graphe) ou [batch, q, N, hidden].
        batch_index :
            Assignation des nœuds à chaque graphe (longueur N). Optionnel si ``state``
            contient déjà une dimension batch explicite.
        pool :
            Mode de pooling à appliquer ("mean" par défaut, "max" accepté).

        Returns
        -------
        Tensor
            Tenseur [batch, q, conditioning_dim] prêt pour cross-attention.
        """
        pool_type = (pool or self.default_pool).lower()

        if state.dim() == 4:
            # state: [batch, q, N, hidden]
            if pool_type == "max":
                pooled = state.max(dim=2).values
            else:
                pooled = state.mean(dim=2)
        elif state.dim() == 3:
            # state: [q, N, hidden]
            if batch_index is None:
                if pool_type == "max":
                    pooled = state.max(dim=1).values.unsqueeze(0)
                else:
                    pooled = state.mean(dim=1).unsqueeze(0)
            else:
                num_batches = int(batch_index.max().item()) + 1
                batch_chunks = []
                for b in range(num_batches):
                    mask = batch_index == b
                    if not torch.any(mask):
                        if pool_type == "max":
                            aggregated = state.max(dim=1).values
                        else:
                            aggregated = state.mean(dim=1)
                    else:
                        selected = state[:, mask, :]
                        if pool_type == "max":
                            aggregated = selected.max(dim=1).values
                        else:
                            aggregated = selected.mean(dim=1)
                    batch_chunks.append(aggregated)
                pooled = torch.stack(batch_chunks, dim=0)
        else:
            raise ValueError(f"Tenseur d'état inattendu de forme {tuple(state.shape)}.")

        batch, q, hidden = pooled.shape
        projected = self.conditioning_projection(pooled.view(batch * q, hidden))
        projected = projected.view(batch, q, -1)
        return projected

    def reset_parameters(self) -> None:
        """
        Réinitialise les poids des convolutions.
        """
        for module in self.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def _check_pyg_lib_availability(self) -> None:
        """
        Vérifie si pyg-lib est disponible pour les optimisations Grouped GEMM.
        Log un message informatif si disponible.
        """
        try:
            import pyg_lib
            # pyg-lib est disponible, HeteroConv bénéficiera automatiquement des optimisations
            # Pas besoin de configuration supplémentaire
            pass
        except ImportError:
            # pyg-lib n'est pas disponible, ce n'est pas critique
            # HeteroConv fonctionnera normalement sans les optimisations Grouped GEMM
            pass

    def _apply_pooling(self, tensor: Tensor, batch: Tensor, pool_type: Optional[str]) -> Tensor:
        pool_type = (pool_type or "").lower()
        if pool_type in {"mean", "avg"}:
            return global_mean_pool(tensor, batch)
        if pool_type == "max":
            return global_max_pool(tensor, batch)
        if pool_type in {"", "none", None}:
            return tensor
        raise ValueError(f"Pooling '{pool_type}' non pris en charge.")


