"""
Modules d'entraînement pour ST-CDGM.
"""

from .training_loop import train_epoch
from .callbacks import EarlyStopping

__all__ = [
    "train_epoch",
    "EarlyStopping",
]

