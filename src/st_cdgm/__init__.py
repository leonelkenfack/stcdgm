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
from .models.regression_mean_predictor import (
    RegressionMeanPredictor,
    RegressionPredictorConfig,
)
from .models.graph_builder import HeteroGraphBuilder
from .data.pipeline import NetCDFDataPipeline, ZarrDataPipeline, ResDiffIterableDataset
from .data.netcdf_utils import NetCDFToDataFrame
from .training.training_loop import (
    train_epoch,
    compute_rapsd_metric_from_batch,
    resolve_train_amp_mode,
)
from .training.stage1_paths import (
    calibrate_sigma_data_variant,
    predict_mu_hr,
    precompute_stage1_outputs_variant,
    resolve_run_variant,
    train_epoch_stage1_noncausal,
    validate_stage1_gate,
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
    "RegressionMeanPredictor",
    "RegressionPredictorConfig",
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
    "calibrate_sigma_data_variant",
    "predict_mu_hr",
    "precompute_stage1_outputs_variant",
    "resolve_run_variant",
    "train_epoch_stage1_noncausal",
    "validate_stage1_gate",
]

