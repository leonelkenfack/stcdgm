"""
Modules d'entraînement pour ST-CDGM.
"""

from .training_loop import train_epoch
from .callbacks import EarlyStopping
from .stage1_paths import (
    calibrate_sigma_data_variant,
    predict_mu_hr,
    precompute_stage1_outputs_variant,
    resolve_run_variant,
    train_epoch_stage1_noncausal,
    validate_stage1_gate,
)

__all__ = [
    "train_epoch",
    "EarlyStopping",
    "calibrate_sigma_data_variant",
    "predict_mu_hr",
    "precompute_stage1_outputs_variant",
    "resolve_run_variant",
    "train_epoch_stage1_noncausal",
    "validate_stage1_gate",
]

