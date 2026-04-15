"""
ST-CDGM: Spatio-Temporal Causal Diffusion Generative Model

Package principal pour le modèle ST-CDGM.
"""

from .models.causal_rcn import RCNCell, RCNSequenceRunner
from .models.diffusion_decoder import CausalDiffusionDecoder, DiffusionOutput
from .models.intelligible_encoder import (
    IntelligibleVariableEncoder,
    IntelligibleVariableConfig,
    SpatialConditioningProjector,
    CausalConditioningProjector,
    HRTargetIdentifiabilityHead,
)
from .models.graph_builder import HeteroGraphBuilder
from .data.pipeline import NetCDFDataPipeline, ZarrDataPipeline, ResDiffIterableDataset
from .data.netcdf_utils import NetCDFToDataFrame
from .training.training_loop import (
    train_epoch,
    compute_rapsd_metric_from_batch,
    resolve_train_amp_mode,
)

__all__ = [
    # Models
    "RCNCell",
    "RCNSequenceRunner",
    "CausalDiffusionDecoder",
    "DiffusionOutput",
    "IntelligibleVariableEncoder",
    "IntelligibleVariableConfig",
    "SpatialConditioningProjector",
    "CausalConditioningProjector",
    "HRTargetIdentifiabilityHead",
    "HeteroGraphBuilder",
    # Data
    "NetCDFDataPipeline",
    "ZarrDataPipeline",
    "ResDiffIterableDataset",
    "NetCDFToDataFrame",
    # Training
    "train_epoch",
    "compute_rapsd_metric_from_batch",
    "resolve_train_amp_mode",
]

