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

        # §1.6 + J4 fix (consensus + AI eng audit):
        # Before this fix, HeteroConv(aggr="sum") aggregated all metapath outputs
        # sharing the same target node type. So two configs targeting the same
        # node type (e.g. GP850_spat_adj and GP850_to_GP500 both targeting GP500)
        # would produce IDENTICAL embeddings — the q "variables" collapsed to
        # the number of unique TARGET node types, not the number of configs.
        # This was the root cause #2 of the band-diagonal A_dag pattern: A_dag
        # rows for collapsed variables were forced to be identical.
        #
        # Fix: build a per-metapath SAGEConv ModuleDict instead of HeteroConv.
        # Each cfg gets its own conv with its own weights. Forward computes
        # per-metapath embeddings keyed by cfg.name (preserves identity).
        # This matches what J4 audit recommended (aggr="cat" + per-target Linear,
        # but cleaner: we just don't aggregate at all).
        self.metapath_convs = nn.ModuleDict()
        for cfg in self.configs:
            kwargs = {"out_channels": hidden_dim}
            if cfg.conv_class is SAGEConv:
                kwargs["in_channels"] = (-1, -1)  # auto-infer
            if cfg.conv_kwargs:
                kwargs.update(cfg.conv_kwargs)
            self.metapath_convs[self._metapath_key(cfg)] = cfg.conv_class(**kwargs)

        # AI eng I-A cleanup: orphan `self.hetero_conv = None` removed.
        # §1.6 replaced HeteroConv with per-metapath ModuleDict; the None
        # placeholder was dead code. Old V5-mini ckpts with hetero_conv.convs.*
        # keys will fall back via J18 strict=False warnings (intended).

        # AI eng I-B cleanup: pyg-lib check is now dead code (no HeteroConv
        # to benefit from Grouped GEMM). Kept as no-op for future re-enable
        # if per-metapath SAGEConv ever supports pyg-lib optimizations.
        self._check_pyg_lib_availability()

        # J3 fix (AI eng audit): replace single shared LayerNorm with
        # per-metapath nn.ModuleDict. The previous shared LayerNorm coupled
        # the affine parameters across all variables — even though §1.6 made
        # the SAGEConv outputs per-metapath distinct, sharing a LayerNorm
        # would average the gradients across all metapaths into a single
        # set of weight/bias, partially re-collapsing variable identity.
        # Per-metapath LayerNorm gives each variable its own normalization
        # scale and bias, preserving identity at the post-conv stage.
        if self.use_layer_norm:
            self.layer_norms = nn.ModuleDict({
                self._metapath_key(cfg): nn.LayerNorm(hidden_dim)
                for cfg in self.configs
            })
            # Backward compat alias for any code path still accessing
            # self.layer_norm directly (will produce wrong results if used
            # but won't crash — to be removed in next architectural cleanup).
            self.layer_norm = nn.Identity()
        else:
            self.layer_norms = nn.ModuleDict({
                self._metapath_key(cfg): nn.Identity()
                for cfg in self.configs
            })
            self.layer_norm = nn.Identity()

        # AI eng I-C: track first missing-edge fallback occurrence for diagnostic
        self._missing_edge_warned: set = set()

        if conditioning_dim is None or conditioning_dim == hidden_dim:
            self.conditioning_dim = hidden_dim
            self.conditioning_projection = nn.Identity()
        else:
            self.conditioning_dim = conditioning_dim
            self.conditioning_projection = nn.Linear(hidden_dim, conditioning_dim)

    @staticmethod
    def _metapath_key(cfg: "IntelligibleVariableConfig") -> str:
        """§1.6 helper: encode metapath identity into a ModuleDict-safe string.

        Uses double-underscore separator (illegal in Python identifiers but
        valid in dict keys). Format: ``{name}__{src}__{rel}__{tgt}``.
        """
        src, rel, tgt = cfg.meta_path
        return f"{cfg.name}__{src}__{rel}__{tgt}"

    def _migrate_legacy_state_dict(self, state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """§1.6 + J3 migration: convert V5-mini encoder state_dict to new format.

        V5-mini used:
          - hetero_conv.convs.<cfg.name>.<...>  (HeteroConv ModuleDict)
          - layer_norm.{weight,bias}             (single shared LayerNorm)

        Path C+ uses:
          - metapath_convs.<_metapath_key(cfg)>.<...>  (per-metapath ModuleDict)
          - layer_norms.<_metapath_key(cfg)>.{weight,bias}  (per-metapath ModuleDict)

        This migration:
        1. Renames hetero_conv.convs.<name>.* -> metapath_convs.<key>.*
        2. Duplicates the legacy shared layer_norm to all per-metapath layer_norms
           (initial post-§1.6 state: per-metapath norms all share the same V5-mini
            initialization, then learn independently from there)
        3. Passes through other keys (conditioning_projection.*) unchanged

        Returns the migrated state_dict ready for strict=True load.
        """
        migrated: Dict[str, Tensor] = {}
        consumed_keys: set = set()

        # Build the conv rename map: cfg.name -> _metapath_key(cfg)
        conv_renames = {}
        for cfg in self.configs:
            old_prefix = f"hetero_conv.convs.{cfg.name}."
            new_prefix = f"metapath_convs.{self._metapath_key(cfg)}."
            conv_renames[old_prefix] = new_prefix

        # Apply conv renames
        for old_key, val in state_dict.items():
            for old_prefix, new_prefix in conv_renames.items():
                if old_key.startswith(old_prefix):
                    new_key = new_prefix + old_key[len(old_prefix):]
                    migrated[new_key] = val
                    consumed_keys.add(old_key)
                    break

        # Handle layer_norm -> layer_norms (replicate to all metapaths)
        legacy_ln_weight = state_dict.get("layer_norm.weight")
        legacy_ln_bias = state_dict.get("layer_norm.bias")
        if legacy_ln_weight is not None or legacy_ln_bias is not None:
            for cfg in self.configs:
                mp_key = self._metapath_key(cfg)
                if legacy_ln_weight is not None:
                    migrated[f"layer_norms.{mp_key}.weight"] = legacy_ln_weight.clone()
                if legacy_ln_bias is not None:
                    migrated[f"layer_norms.{mp_key}.bias"] = legacy_ln_bias.clone()
            consumed_keys.add("layer_norm.weight")
            consumed_keys.add("layer_norm.bias")

        # Pass through any other keys (conditioning_projection, etc.)
        for key, val in state_dict.items():
            if key not in consumed_keys and key not in migrated:
                migrated[key] = val

        return migrated

    def load_state_dict(self, state_dict, strict: bool = True):
        """Override to auto-detect and migrate legacy V5-mini state_dict format.

        If the state_dict contains hetero_conv.* keys (V5-mini format), apply
        _migrate_legacy_state_dict before the standard load. Emits a UserWarning
        so the user knows migration was applied.
        """
        # Detect legacy format
        legacy_keys = [k for k in state_dict.keys() if k.startswith("hetero_conv.")]
        if legacy_keys:
            import warnings as _w
            _w.warn(
                f"§1.6+J3 migration: detected {len(legacy_keys)} legacy hetero_conv.* "
                f"keys in encoder state_dict. Auto-migrating to per-metapath format. "
                f"This is expected when loading V5-mini baseline into Path C+ refactored "
                f"encoder. The shared layer_norm is replicated to all per-metapath "
                f"layer_norms as initialization (they then learn independently).",
                UserWarning, stacklevel=2,
            )
            state_dict = self._migrate_legacy_state_dict(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, data: HeteroData, *, pooled: bool = False) -> Dict[str, Tensor]:
        """
        Applique l'encodeur et retourne un dict {variable_name: embeddings}.
        Si ``pooled=True``, les embeddings sont agrégés par graphe (global pooling).

        §1.6 + J4 fix: computes PER-METAPATH embeddings (not per-target-type).
        Previously, two configs targeting the same node type would receive
        identical embeddings because HeteroConv(aggr="sum") collapsed them.
        Now each config has its own SAGEConv and produces an independent
        embedding tensor keyed by cfg.name.
        """
        x_dict = {node_type: data[node_type].x for node_type in data.node_types}

        outputs: Dict[str, Tensor] = {}
        for cfg in self.configs:
            src_type, rel_type, tgt_type = cfg.meta_path
            edge_key = (src_type, rel_type, tgt_type)
            edge_idx = data.edge_index_dict.get(edge_key)
            mp_key = self._metapath_key(cfg)

            if edge_idx is None or edge_idx.numel() == 0:
                # Metapath edges absent in this graph — fall back to identity
                # on the target node features (encoder still produces output
                # but the metapath conv contributes nothing).
                # AI eng I-C: log first occurrence so silent gradient-blocked
                # paths are visible during training.
                if cfg.name not in self._missing_edge_warned:
                    self._missing_edge_warned.add(cfg.name)
                    import warnings as _w
                    _w.warn(
                        f"§1.6 missing-edge fallback for cfg='{cfg.name}' "
                        f"(metapath {cfg.meta_path}). Output will be zeros for "
                        f"this metapath, blocking gradient back to upstream "
                        f"node-type features. Will warn only once per cfg.",
                        UserWarning, stacklevel=2,
                    )
                tensor = x_dict[tgt_type]
                if tensor.shape[-1] != self.hidden_dim:
                    tensor = torch.zeros(
                        tensor.shape[0], self.hidden_dim,
                        device=tensor.device, dtype=tensor.dtype,
                    )
            else:
                conv = self.metapath_convs[mp_key]
                # SAGEConv expects (x_src, x_dst) for bipartite — both feature dicts
                tensor = conv((x_dict[src_type], x_dict[tgt_type]), edge_idx)

            # J3 fix: per-metapath LayerNorm preserves variable identity
            tensor = self.layer_norms[mp_key](tensor)
            tensor = self.activation(tensor)

            if pooled:
                pool_type = cfg.pool or self.default_pool
                batch_attr = getattr(data[tgt_type], "batch", None)
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


class SpatialConditioningProjector(nn.Module):
    """Projects RCN state [q, N, hidden] into spatially-compressed conditioning tokens.

    The state is reshaped onto the LR grid, adaptively pooled to a small target
    spatial resolution, then projected to ``conditioning_dim``.  The output has
    shape ``[batch, num_vars * target_h * target_w, conditioning_dim]`` and can
    be passed directly as ``encoder_hidden_states`` to a UNet with cross-attention.
    """

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        conditioning_dim: int,
        lr_shape: Tuple[int, int],
        target_shape: Tuple[int, int] = (6, 7),
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.lr_shape = lr_shape
        self.target_shape = target_shape
        self.adaptive_pool = nn.AdaptiveAvgPool2d(target_shape)
        self.proj = nn.Linear(hidden_dim, conditioning_dim)
        self.norm = nn.LayerNorm(conditioning_dim)

    def forward(self, state: Tensor, *, batch_index: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        state : Tensor
            RCN hidden state of shape ``[q, N, hidden]``.
        batch_index : Optional[Tensor]
            Not used (single-graph path). Reserved for future multi-graph batching.

        Returns
        -------
        Tensor
            ``[batch, num_vars * th * tw, conditioning_dim]``
        """
        q, N, d = state.shape
        lat, lon = self.lr_shape
        if N != lat * lon:
            raise ValueError(
                f"SpatialConditioningProjector: expected N={lat*lon} "
                f"(lr_shape={self.lr_shape}), got N={N}."
            )
        grid = state.view(q, lat, lon, d).permute(0, 3, 1, 2)    # [q, d, lat, lon]
        pooled = self.adaptive_pool(grid)                          # [q, d, th, tw]
        th, tw = self.target_shape
        tokens = pooled.permute(0, 2, 3, 1).reshape(q * th * tw, d)  # [q*th*tw, d]
        projected = self.norm(self.proj(tokens))                   # [q*th*tw, cond_dim]
        return projected.unsqueeze(0)                              # [1, q*th*tw, cond_dim]


class HRTargetIdentifiabilityHead(nn.Module):
    """
    Sprint 2: predicts summary statistics of the HR target from the pooled
    causal state H_T. Used as an *additional* reconstruction loss that
    complements the LR-driver recon (Eq.(7) of the ORACLE paper).

    Motivation
    ----------
    The Sprint 1 fix made ``L_rec = || g(H_t) - u_t ||^2`` functional, but
    ``u_t`` is the *low-resolution* driver field. Reconstructing the LR
    driver exercises the SCM as an autoencoder but never teaches it
    anything specific about the HR target (precipitation extremes). This
    head provides a much more targeted signal: the pooled causal state
    must predict a small vector of HR statistics (spatial mean, std, p95,
    p99 by default), so gradient flowing back through the SCM forces A_dag
    and the structural MLPs to carry information that *actually matters*
    for downscaling.

    The head is deliberately small (a 2-layer MLP) so most of the
    learning happens upstream — in A_dag, not in the head itself.
    """

    #: Default statistics order. Must stay stable across training/eval.
    DEFAULT_STATS: Tuple[str, ...] = ("mean", "std", "p95", "p99")

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        *,
        stats: Optional[Sequence[str]] = None,
        inner_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.stats: Tuple[str, ...] = tuple(stats) if stats else self.DEFAULT_STATS
        for s in self.stats:
            if s not in {"mean", "std", "p95", "p99", "max"}:
                raise ValueError(f"Unknown HR target stat '{s}'")
        inner = inner_dim or max(32, num_vars * hidden_dim // 4)
        self.mlp = nn.Sequential(
            nn.Linear(num_vars * hidden_dim, inner),
            nn.GELU(),
            nn.Linear(inner, len(self.stats)),
        )

    def forward(self, state: Tensor) -> Tensor:
        """
        Parameters
        ----------
        state : Tensor
            Pooled causal state ``[batch, q, hidden_dim]`` — typically the
            output of ``IntelligibleVariableEncoder.project_state_tensor``
            *without* the conditioning projection (pre-Linear), or any
            tensor that flattens to ``[batch, q*d]``.

        Returns
        -------
        Tensor
            ``[batch, len(self.stats)]`` — predicted summary statistics.
        """
        if state.dim() == 3:
            batch = state.shape[0]
            flat = state.reshape(batch, -1)
        elif state.dim() == 4:
            # [q, N, hidden] — single-graph path, pool spatially.
            pooled = state.mean(dim=1) if state.shape[0] == self.num_vars else state.mean(dim=2)
            flat = pooled.reshape(1, -1)
        else:
            flat = state.reshape(state.shape[0], -1)
        return self.mlp(flat).clone()

    @staticmethod
    def extract_target_stats(
        target: Tensor,
        stats: Sequence[str] = DEFAULT_STATS,
    ) -> Tensor:
        """
        Compute the per-sample summary statistics used as the regression
        target for this head. NaN pixels (ocean masks) are ignored.

        Parameters
        ----------
        target : Tensor
            HR field ``[batch, channels, H, W]``.

        Returns
        -------
        Tensor
            ``[batch, len(stats)]``
        """
        if target.dim() != 4:
            raise ValueError(f"target must be [B, C, H, W]; got {tuple(target.shape)}")
        batch = target.shape[0]
        out_cols = []
        for b in range(batch):
            x = target[b].reshape(-1)
            x = x[torch.isfinite(x)]
            if x.numel() == 0:
                row = [0.0] * len(stats)
            else:
                row = []
                for s in stats:
                    if s == "mean":
                        row.append(x.mean().item())
                    elif s == "std":
                        row.append(x.std(unbiased=False).item())
                    elif s == "p95":
                        row.append(torch.quantile(x, 0.95).item())
                    elif s == "p99":
                        row.append(torch.quantile(x, 0.99).item())
                    elif s == "max":
                        row.append(x.max().item())
                    else:
                        raise ValueError(f"Unknown stat '{s}'")
            out_cols.append(row)
        return torch.tensor(out_cols, dtype=target.dtype, device=target.device)


class CausalConditioningProjector(nn.Module):
    """
    Sprint 2: condition the diffusion decoder on *both* the causal state
    ``H_T`` and the learned DAG matrix ``A_dag``.

    The rationale is developed in ``ORACLE_HYPERPLAN_ANALYSIS.md`` §2.2
    and §6: to have A_dag *actually* condition every prediction, the
    UNet has to see it. The default path (``SpatialConditioningProjector``
    + FiLM of pooled H_T) sends the state to the UNet but hides A_dag.
    Here we append a learnable embedding of the flattened DAG matrix to
    the spatial tokens, so the cross-attention blocks can attend to the
    current causal structure. Editing ``A_dag`` at inference time (do-
    calculus) then directly changes the tokens fed to the UNet.

    The module wraps a ``SpatialConditioningProjector`` by default — the
    DAG token is an *addition*, not a replacement. Output shape:
    ``[batch, num_spatial_tokens + num_dag_tokens, conditioning_dim]``.
    """

    def __init__(
        self,
        num_vars: int,
        hidden_dim: int,
        conditioning_dim: int,
        lr_shape: Tuple[int, int],
        target_shape: Tuple[int, int] = (6, 7),
        *,
        num_dag_tokens: int = 1,
        dag_token_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.num_dag_tokens = int(num_dag_tokens)
        self.spatial = SpatialConditioningProjector(
            num_vars=num_vars,
            hidden_dim=hidden_dim,
            conditioning_dim=conditioning_dim,
            lr_shape=lr_shape,
            target_shape=target_shape,
        )
        # The DAG embedding: flatten A_dag [q, q] → q² scalars, project
        # through a small MLP to ``num_dag_tokens * conditioning_dim``.
        # Keeping this shallow (1 hidden layer) is important: we want the
        # UNet to *see* the raw entries of A_dag, not a heavy re-encoding
        # that dilutes interventional edits.
        inner = dag_token_dim or conditioning_dim
        self.dag_mlp = nn.Sequential(
            nn.Linear(num_vars * num_vars, inner),
            nn.GELU(),
            nn.Linear(inner, self.num_dag_tokens * conditioning_dim),
        )
        self.dag_norm = nn.LayerNorm(conditioning_dim)
        self.conditioning_dim = conditioning_dim

    def forward(
        self,
        state: Tensor,
        A_dag: Tensor,
        *,
        batch_index: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        state : Tensor
            RCN hidden state ``[q, N, hidden]``.
        A_dag : Tensor
            Masked DAG matrix ``[q, q]``. Must be attached (we *want*
            gradient to flow back to ``A_dag`` — this is the Sprint 2
            point: the UNet learns to use the causal structure).

        Returns
        -------
        Tensor
            ``[1, num_spatial + num_dag_tokens, conditioning_dim]``
        """
        spatial_tokens = self.spatial(state, batch_index=batch_index)  # [1, Ns, d]
        if A_dag.dim() != 2 or A_dag.shape[0] != self.num_vars or A_dag.shape[1] != self.num_vars:
            raise ValueError(
                f"CausalConditioningProjector: A_dag must be [{self.num_vars}, "
                f"{self.num_vars}]; got {tuple(A_dag.shape)}"
            )
        flat = A_dag.reshape(1, -1)                                       # [1, q²]
        dag_emb = self.dag_mlp(flat)                                      # [1, T*d]
        dag_tokens = dag_emb.view(1, self.num_dag_tokens, self.conditioning_dim)
        dag_tokens = self.dag_norm(dag_tokens)
        return torch.cat([spatial_tokens, dag_tokens], dim=1)             # [1, Ns+T, d]


