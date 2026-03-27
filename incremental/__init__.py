"""Incremental Gaussian Splatting update module.

This module provides functionality for incremental updates to Gaussian models
using change detection and 2D→3D lifting techniques.
"""

from .gaussian_adapter import GGSGaussianAdapter
from .trainer import IncrementalTrainer

__all__ = ["GGSGaussianAdapter", "IncrementalTrainer"]
