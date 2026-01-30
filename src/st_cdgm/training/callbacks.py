"""
Phase C2: Training callbacks for Early Stopping and Learning Rate Scheduling.

This module provides utilities for monitoring training and automatically
stopping when validation loss stops improving, as well as adaptive learning
rate scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch


@dataclass
class EarlyStopping:
    """
    Phase C2: Early stopping callback to stop training when validation loss stops improving.
    
    Monitors validation loss and stops training if no improvement is seen for
    a specified number of epochs (patience). Optionally restores the best model.
    
    Parameters
    ----------
    patience : int
        Number of epochs to wait before stopping if no improvement
    min_delta : float
        Minimum change in loss to qualify as an improvement
    mode : str
        'min' for loss (lower is better) or 'max' for metrics (higher is better)
    restore_best : bool
        If True, restores the best model weights at the end
    verbose : bool
        If True, prints messages when stopping or restoring
    """
    
    patience: int = 7
    min_delta: float = 0.0
    mode: str = "min"
    restore_best: bool = True
    verbose: bool = True
    
    def __post_init__(self):
        self.best_score: Optional[float] = None
        self.counter = 0
        self.best_weights: Optional[dict] = None
        self.early_stop = False
        
        if self.mode not in ["min", "max"]:
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode}")
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop.
        
        Parameters
        ----------
        score : float
            Current validation score (loss or metric)
        model : torch.nn.Module
            Model to monitor (weights will be saved if best)
        
        Returns
        -------
        bool
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model, score)
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model, score)
            if self.verbose:
                print(f"✓ EarlyStopping: New best score: {score:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs (best: {self.best_score:.6f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"EarlyStopping: Stopping training (patience={self.patience} reached)")
                if self.restore_best and self.best_weights is not None:
                    self._restore_checkpoint(model)
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best score."""
        if self.mode == "min":
            return current < (best - self.min_delta)
        else:  # mode == "max"
            return current > (best + self.min_delta)
    
    def _save_checkpoint(self, model: torch.nn.Module, score: float) -> None:
        """Save model weights checkpoint."""
        self.best_weights = {
            'state_dict': model.state_dict().copy(),
            'score': score,
        }
    
    def _restore_checkpoint(self, model: torch.nn.Module) -> None:
        """Restore best model weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights['state_dict'])
            if self.verbose:
                print(f"✓ EarlyStopping: Restored best model (score: {self.best_weights['score']:.6f})")
    
    def reset(self) -> None:
        """Reset early stopping state (useful for new training runs)."""
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False

