"""Lifter module for 2D to 3D lifting."""

try:
    from .sg_aware_lifter import SGAwareLifter, create_sg_aware_lifter, LiftResult
except ImportError:
    SGAwareLifter = None
    create_sg_aware_lifter = None
    LiftResult = None

from .pixel_gaussian_mapping import (
    PixelGaussianMapper,
    PixelGaussianMapping,
    create_pixel_gaussian_mapper,
)

__all__ = [
    "SGAwareLifter",
    "create_sg_aware_lifter",
    "LiftResult",
    "PixelGaussianMapper",
    "PixelGaussianMapping",
    "create_pixel_gaussian_mapper",
]
