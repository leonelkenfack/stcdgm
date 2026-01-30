"""
Module 2 – Construction du graphe hétérogène statique pour l’architecture ST-CDGM.

Ce module fournit une classe utilitaire `HeteroGraphBuilder` qui prépare un
objet `torch_geometric.data.HeteroData` en codant les relations physiques
principales : advection (spatiale), convection (verticale) et influence
statique (topographie HR vers dynamique LR).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
    from torch_geometric.data import HeteroData
except ImportError as exc:  # pragma: no cover - dépendance optionnelle
    raise ImportError(
        "torch et torch_geometric sont requis pour utiliser HeteroGraphBuilder."
    ) from exc

import xarray as xr


GridShape = Tuple[int, int]


@dataclass
class GraphBuildReport:
    """
    Informations de diagnostic sur la construction du graphe.
    """

    num_nodes_lr: int
    num_nodes_hr: int
    edges_spatial: Dict[str, int]
    edges_vertical: Dict[str, int]
    edges_static: Dict[str, int]
    hr_to_lr_parent: Sequence[int]


class HeteroGraphBuilder:
    """
    Construit un graphe hétérogène statique basé sur des grilles LR/HR.

    Parameters
    ----------
    lr_shape :
        Shape (lat, lon) de la grille basse résolution.
    hr_shape :
        Shape (lat, lon) de la grille haute résolution.
    static_dataset :
        Dataset xarray contenant les variables statiques HR (topographie, etc.).
    static_variables :
        Liste de variables statiques à intégrer. Toutes si None.
    include_mid_layer :
        Contrôle la présence de la couche intermédiaire GP500/GP250.
    """

    def __init__(
        self,
        lr_shape: GridShape,
        hr_shape: GridShape,
        *,
        static_dataset: Optional[xr.Dataset] = None,
        static_variables: Optional[Sequence[str]] = None,
        include_mid_layer: bool = True,
    ) -> None:
        self.lr_shape = lr_shape
        self.hr_shape = hr_shape
        self.static_dataset = static_dataset
        self.static_variables = static_variables
        self.include_mid_layer = include_mid_layer

        self._validate_shapes()

        self.num_nodes_lr = self.lr_shape[0] * self.lr_shape[1]
        self.num_nodes_hr = self.hr_shape[0] * self.hr_shape[1]

        self.dynamic_node_types = ["GP850"]
        if self.include_mid_layer:
            self.dynamic_node_types.extend(["GP500", "GP250"])
        
        # Static node types (always includes SP_HR if static dataset is provided)
        self.static_node_types = ["SP_HR"] if self.static_dataset is not None else []

        self._spatial_edge_index = self._build_spatial_adjacency(self.lr_shape)
        self._vertical_edge_index = self._build_vertical_edges(self.num_nodes_lr)
        self._static_edge_index = self._build_static_influence_mapping(self.lr_shape, self.hr_shape)
        self._hr_parent_index = self._static_edge_index[1].clone()

        if self.static_dataset is not None:
            self._static_features = self._extract_static_features(self.static_dataset)
        else:
            self._static_features = torch.zeros((self.num_nodes_hr, 0), dtype=torch.float32)

        self._template_cache: Optional[HeteroData] = None
        self._report_cache: Optional[GraphBuildReport] = None

    def _validate_shapes(self) -> None:
        if len(self.lr_shape) != 2 or len(self.hr_shape) != 2:
            raise ValueError("Les shapes LR et HR doivent être de longueur 2 (lat, lon).")
        if any(dim <= 0 for dim in self.lr_shape + self.hr_shape):
            raise ValueError("Toutes les dimensions de grille doivent être positives.")
        # Note: We now support non-integer ratios by using interpolation/rounding
        # The mapping will use the nearest LR node for each HR node

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------
    def build(self) -> Tuple[HeteroData, GraphBuildReport]:
        """
        Construit et retourne l'objet HeteroData ainsi qu'un rapport de diagnostic.
        """
        data = HeteroData()

        for node_type in self.dynamic_node_types:
            data[node_type].num_nodes = self.num_nodes_lr

        data["SP_HR"].num_nodes = self.num_nodes_hr
        data["SP_HR"].x = self._static_features.clone()

        edges_spatial: Dict[str, int] = {}
        spatial_index = self._spatial_edge_index.clone()
        data["GP850", "spat_adj", "GP850"].edge_index = spatial_index
        edges_spatial["GP850"] = spatial_index.size(1)
        if self.include_mid_layer:
            data["GP500", "spat_adj", "GP500"].edge_index = spatial_index.clone()
            data["GP250", "spat_adj", "GP250"].edge_index = spatial_index.clone()
            edges_spatial["GP500"] = spatial_index.size(1)
            edges_spatial["GP250"] = spatial_index.size(1)

        edges_vertical: Dict[str, int] = {}
        if self.include_mid_layer:
            vert_edge = self._vertical_edge_index.clone()
            data["GP850", "vert_adj", "GP500"].edge_index = vert_edge
            data["GP500", "vert_adj", "GP850"].edge_index = vert_edge[[1, 0], :]

            data["GP500", "vert_adj", "GP250"].edge_index = vert_edge.clone()
            data["GP250", "vert_adj", "GP500"].edge_index = vert_edge[[1, 0], :]

            edges_vertical["GP850↔GP500"] = vert_edge.size(1) * 2
            edges_vertical["GP500↔GP250"] = vert_edge.size(1) * 2

        edges_static: Dict[str, int] = {}
        static_edge_index = self._static_edge_index.clone()
        data["SP_HR", "causes", "GP850"].edge_index = static_edge_index
        edges_static["SP_HR→GP850"] = static_edge_index.size(1)

        if self.include_mid_layer:
            data["SP_HR", "causes", "GP500"].edge_index = static_edge_index.clone()
            data["SP_HR", "causes", "GP250"].edge_index = static_edge_index.clone()
            edges_static["SP_HR→GP500"] = static_edge_index.size(1)
            edges_static["SP_HR→GP250"] = static_edge_index.size(1)

        self._assign_default_batch(data)

        report = GraphBuildReport(
            num_nodes_lr=self.num_nodes_lr,
            num_nodes_hr=self.num_nodes_hr,
            edges_spatial=edges_spatial,
            edges_vertical=edges_vertical,
            edges_static=edges_static,
            hr_to_lr_parent=self._hr_parent_index.tolist(),
        )
        self._validate_edge_ranges(data)

        self._template_cache = data.clone()
        self._report_cache = report

        return data, report

    def build_template(self) -> HeteroData:
        """Return a fresh clone of the cached template graph."""
        if self._template_cache is None:
            template, _ = self.build()
            self._template_cache = template.clone()
        return self._template_cache.clone()

    def get_report(self) -> GraphBuildReport:
        if self._report_cache is None:
            _, report = self.build()
            self._report_cache = report
        return self._report_cache

    def prepare_step_data(
        self,
        features: Dict[str, torch.Tensor],
        *,
        clone_template: bool = False,
    ) -> HeteroData:
        """
        Retourne un HeteroData prêt à l'emploi avec les features dynamiques injectées.
        
        Phase 2.1: Optimized to avoid full graph cloning by default.
        The template graph (edge_index, static features) is reused, only dynamic
        node features are updated in-place.
        
        Parameters
        ----------
        features : Dict[str, torch.Tensor]
            Dynamic node features to inject (e.g., atmospheric variables per timestep)
        clone_template : bool
            If True, creates a full deep copy of the template (slower but safe if you
            need to modify the graph structure). If False (default), reuses the template
            and only updates dynamic features in-place (faster, recommended for training).
        
        Returns
        -------
        HeteroData
            Graph with dynamic features injected
        """
        if self._template_cache is None:
            self._template_cache = self.build_template()
        
        # Phase 2.1: Only clone if explicitly requested (for backward compatibility)
        # Otherwise, reuse template and inject features in-place (much faster)
        if clone_template:
            data = self._template_cache.clone()
        else:
            data = self._template_cache
        
        self.inject_dynamic_features(data, features)
        return data

    def inject_dynamic_features(self, data: HeteroData, features: Dict[str, torch.Tensor]) -> None:
        """
        Injecte des features nodales dynamiques (par pas de temps).
        
        Phase 2.1: Modifies node features in-place. This is safe when called on a
        cloned template, or when the input tensor is a new tensor (not a view of
        previously injected data).
        
        Note: If you're reusing the same HeteroData instance (clone_template=False),
        ensure that the input tensors are new tensors, not views/slices of previously
        injected tensors, to avoid unintended side effects.
        """
        for node_type, tensor in features.items():
            if node_type not in data.node_types:
                raise KeyError(f"Node type '{node_type}' absent du graphe.")
            expected_nodes = data[node_type].num_nodes
            if tensor.shape[0] != expected_nodes:
                raise ValueError(
                    f"Features pour '{node_type}' incompatibles: "
                    f"{tensor.shape[0]} nœuds fournis, {expected_nodes} attendus."
                )
            # Phase 2.1: In-place modification - safe as long as input tensor is new
            # or we're working on a cloned template
            data[node_type].x = tensor

    def get_hr_to_lr_parent_index(self) -> torch.Tensor:
        """Retourne le mapping hr->lr (1D tensor de taille num_nodes_hr)."""
        return self._hr_parent_index.clone()

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------
    def lr_grid_to_nodes(
        self,
        grid: torch.Tensor,
        *,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """
        Convertit un champ LR (C,H,W) ou (H,W,C) en matrice nodale [N_lr, C].
        """
        if channel_last:
            grid = grid.permute(2, 0, 1)
        if grid.dim() != 3:
            raise ValueError("Le tenseur LR doit être de dimension 3 (C,H,W).")
        c, h, w = grid.shape
        if (h, w) != self.lr_shape:
            raise ValueError(f"Shape LR attendu {self.lr_shape}, obtenu {(h, w)}")
        nodes = grid.reshape(c, -1).transpose(0, 1).contiguous()
        return nodes

    def hr_grid_to_nodes(
        self,
        grid: torch.Tensor,
        *,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """
        Convertit un champ HR (C,H,W) ou (H,W,C) en matrice nodale [N_hr, C].
        """
        if channel_last:
            grid = grid.permute(2, 0, 1)
        if grid.dim() != 3:
            raise ValueError("Le tenseur HR doit être de dimension 3 (C,H,W).")
        c, h, w = grid.shape
        if (h, w) != self.hr_shape:
            raise ValueError(f"Shape HR attendu {self.hr_shape}, obtenu {(h, w)}")
        nodes = grid.reshape(c, -1).transpose(0, 1).contiguous()
        return nodes

    def hr_nodes_to_grid(
        self,
        nodes: torch.Tensor,
        *,
        channel_last: bool = False,
    ) -> torch.Tensor:
        """
        Convertit un tenseur nodal HR [N_hr, C] en champ grille (C,H,W) ou (H,W,C).
        """
        if nodes.dim() != 2 or nodes.size(0) != self.num_nodes_hr:
            raise ValueError(
                f"Tenseur nodal HR incompatible, attendu ({self.num_nodes_hr}, C), obtenu {tuple(nodes.shape)}"
            )
        c = nodes.size(1)
        grid = nodes.transpose(0, 1).reshape(c, self.hr_shape[0], self.hr_shape[1]).contiguous()
        if channel_last:
            grid = grid.permute(1, 2, 0)
        return grid

    def expand_lr_nodes_to_hr(self, lr_nodes: torch.Tensor) -> torch.Tensor:
        """
        Broadcast des features LR [N_lr, C] sur la grille HR via le mapping parent.
        """
        if lr_nodes.dim() != 2 or lr_nodes.size(0) != self.num_nodes_lr:
            raise ValueError(
                f"Tenseur nodal LR incompatible, attendu ({self.num_nodes_lr}, C), obtenu {tuple(lr_nodes.shape)}"
            )
        expanded = lr_nodes[self._hr_parent_index]
        return expanded.contiguous()

    def _assign_default_batch(self, data: HeteroData) -> None:
        """Initialise le vecteur batch (nécessaire pour le pooling global)."""
        for node_type in data.node_types:
            num_nodes = data[node_type].num_nodes
            data[node_type].batch = torch.zeros(num_nodes, dtype=torch.long)

    # ------------------------------------------------------------------
    # Construction des arêtes
    # ------------------------------------------------------------------
    @staticmethod
    def _build_spatial_adjacency(shape: GridShape) -> torch.Tensor:
        """
        Retourne l'edge_index (2, E) pour la connectivité 8-voisins.
        """
        lat, lon = shape
        indices = []
        offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

        for i in range(lat):
            for j in range(lon):
                src = i * lon + j
                for di, dj in offsets:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < lat and 0 <= nj < lon:
                        dst = ni * lon + nj
                        indices.append((src, dst))

        edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
        return edge_index

    @staticmethod
    def _build_vertical_edges(num_nodes: int) -> torch.Tensor:
        """
        Crée des arêtes verticales (mapping identique entre deux couches).
        """
        indices = torch.arange(num_nodes, dtype=torch.long)
        edge_index = torch.stack([indices, indices], dim=0)
        return edge_index

    @staticmethod
    def _build_static_influence_mapping(
        lr_shape: GridShape,
        hr_shape: GridShape,
    ) -> torch.Tensor:
        """
        Mappe chaque nœud HR vers son nœud LR parent le plus proche.
        Supporte les ratios non-entiers en utilisant une interpolation par arrondi.
        """
        # Calculate actual ratios (may be non-integer)
        ratio_lat = hr_shape[0] / lr_shape[0]
        ratio_lon = hr_shape[1] / lr_shape[1]
        
        indices = []

        for i_hr in range(hr_shape[0]):
            for j_hr in range(hr_shape[1]):
                # Map HR coordinates to LR coordinates using the ratio
                # Use rounding to find the nearest LR node
                i_lr = int(round(i_hr / ratio_lat))
                j_lr = int(round(j_hr / ratio_lon))
                
                # Clamp to valid LR indices
                i_lr = max(0, min(i_lr, lr_shape[0] - 1))
                j_lr = max(0, min(j_lr, lr_shape[1] - 1))
                
                hr_idx = i_hr * hr_shape[1] + j_hr
                lr_idx = i_lr * lr_shape[1] + j_lr
                indices.append((hr_idx, lr_idx))

        edge_index = torch.tensor(indices, dtype=torch.long).t().contiguous()
        return edge_index

    # ------------------------------------------------------------------
    # Gestion des features statiques
    # ------------------------------------------------------------------
    def _extract_static_features(self, dataset: xr.Dataset) -> torch.Tensor:
        """
        Transforme les variables statiques en un tenseur torch [num_nodes_hr, num_features].
        """
        vars_to_use: Iterable[str]
        if self.static_variables is None:
            # Filter out bounds variables (common in climate data)
            vars_to_use = [
                var for var in dataset.data_vars.keys()
                if "bnds" not in str(var).lower() and "bounds" not in str(var).lower()
            ]
        else:
            vars_to_use = self.static_variables

        features = []
        for var in vars_to_use:
            if var not in dataset:
                raise KeyError(f"Variable statique '{var}' absente du dataset.")
            arr = dataset[var].values
            
            # Skip variables that don't have the expected spatial shape
            # (e.g., bounds variables, or variables with extra dimensions)
            if arr.ndim < 2:
                continue  # Skip 1D variables
            
            # Check if the last two dimensions match hr_shape
            if arr.shape[-2:] != self.hr_shape:
                # Try to squeeze singleton dimensions
                arr_squeezed = arr.squeeze()
                if arr_squeezed.ndim >= 2 and arr_squeezed.shape[-2:] == self.hr_shape:
                    arr = arr_squeezed
                else:
                    # Skip variables that don't match the expected shape
                    continue
            
            features.append(arr.reshape(-1))

        if not features:
            return torch.zeros(
                (self.hr_shape[0] * self.hr_shape[1], 0), dtype=torch.float32
            )

        stacked = np.stack(features, axis=-1)
        tensor = torch.from_numpy(stacked.astype(np.float32))
        return tensor

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_edge_ranges(data: HeteroData) -> None:
        """
        Vérifie que les indices d'arêtes sont dans les bornes pour chaque type.
        """
        for key, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = key
            src_nodes = data[src_type].num_nodes
            dst_nodes = data[dst_type].num_nodes
            if edge_index.numel() == 0:
                continue
            if edge_index.min().item() < 0:
                raise ValueError(f"Indices négatifs détectés pour {key}.")
            if edge_index[0].max().item() >= src_nodes:
                raise ValueError(f"Indice source hors bornes pour {key}.")
            if edge_index[1].max().item() >= dst_nodes:
                raise ValueError(f"Indice destination hors bornes pour {key}.")


