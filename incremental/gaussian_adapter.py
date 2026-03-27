"""Gaussian Model Adapter.

Converts the GaussianModel from this project to the interface expected by
cl-splats for incremental updates.
"""

import torch
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from scene.gaussian_model import GaussianModel


@dataclass
class GaussianParams:
    """Container for Gaussian parameters in cl-splats format."""

    positions: torch.Tensor  # (N, 3)
    scales: torch.Tensor  # (N, 3) or (N, 1)
    quats: torch.Tensor  # (N, 4)
    sh_features: torch.Tensor  # (N, C, K)
    opacity: torch.Tensor  # (N, 1)
    sg_features: Optional[Dict[str, torch.Tensor]] = None  # SG feature dict


class GGSGaussianAdapter:
    """Adapter: Converts GaussianModel to cl-splats compatible interface.

    This adapter bridges the GaussianModel from this project with the
    CLGaussians interface expected by the cl-splats trainer.

    Supports both SH (Spherical Harmonics) and SG (Spherical Gaussians) features.
    """

    def __init__(
        self,
        gaussian_model: GaussianModel,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """Initialize the adapter with a GaussianModel.

        Args:
            gaussian_model: The GaussianModel to adapt
            optimizer: Optional optimizer (if not provided, model.optimizer is used)
        """
        self.model = gaussian_model
        self.optimizer = optimizer if optimizer is not None else gaussian_model.optimizer
        self.device = gaussian_model._xyz.device if gaussian_model._xyz.numel() > 0 else torch.device("cuda")

    @property
    def num_gaussians(self) -> int:
        """Return the number of Gaussians."""
        return self.model._xyz.shape[0]

    def get_positions(self) -> torch.Tensor:
        """Get Gaussian positions.

        Returns:
            positions: (N, 3) tensor of positions
        """
        return self.model._xyz

    def get_scales(self) -> torch.Tensor:
        """Get Gaussian scales (in linear space, not log).

        Returns:
            scales: (N, 3) tensor of scales
        """
        return torch.exp(self.model._scaling)

    def get_quats(self) -> torch.Tensor:
        """Get Gaussian quaternions.

        Returns:
            quats: (N, 4) tensor of quaternions (w, x, y, z)
        """
        return self.model.get_rotation

    def get_opacity(self) -> torch.Tensor:
        """Get Gaussian opacity (in linear space, not logit).

        Returns:
            opacity: (N, 1) tensor of opacity values
        """
        return self.model.get_opacity

    def get_sh_features(self) -> torch.Tensor:
        """Get Spherical Harmonics features.

        Returns:
            sh_features: (N, 3, K) tensor of SH features
        """
        return self.model.get_features

    def get_sg_features(self) -> Dict[str, torch.Tensor]:
        """Get Spherical Gaussian features.

        Returns:
            Dict with keys: 'axis' (N, D, 3), 'sharpness' (N, D), 'color' (N, D, 3)
        """
        return {
            "axis": self.model.get_sg_axis,
            "sharpness": self.model.get_sg_sharpness,
            "color": self.model.get_sg_color,
        }

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """Get 3D covariance matrices.

        Args:
            scaling_modifier: Optional modifier for scales

        Returns:
            covariance: (N, 6) upper triangular covariance matrices
        """
        return self.model.get_covariance(scaling_modifier)

    def get_params(self) -> GaussianParams:
        """Get all Gaussian parameters in cl-splats format.

        Returns:
            GaussianParams object with all parameters
        """
        sg_features = self.get_sg_features()

        return GaussianParams(
            positions=self.get_positions(),
            scales=self.get_scales(),
            quats=self.get_quats(),
            sh_features=self.get_sh_features(),
            opacity=self.get_opacity(),
            sg_features=sg_features,
        )

    def prune_gaussians(self, prune_mask: torch.Tensor) -> torch.Tensor:
        """Remove Gaussians where prune_mask[i] is True.

        Args:
            prune_mask: Boolean tensor, True indicates Gaussian to prune

        Returns:
            keep: Boolean mask indicating which Gaussians remain
        """
        if self.model.prune_points is None:
            raise NotImplementedError("GaussianModel does not implement prune_points")

        keep = self.model.prune_points(prune_mask)
        return keep

    def densify_and_split(
        self,
        max_grad: float,
        min_opacity: float,
        max_screen_size: int,
    ) -> None:
        """Densify by splitting high-gradient, large Gaussians.

        Args:
            max_grad: Maximum gradient threshold
            min_opacity: Minimum opacity threshold
            max_screen_size: Maximum screen size
        """
        if hasattr(self.model, "densify_and_split"):
            self.model.densify_and_split(max_grad, min_opacity, max_screen_size)

    def densify_and_clone(
        self,
        max_grad: float,
        min_opacity: float,
        max_screen_size: int,
    ) -> None:
        """Densify by cloning Gaussians with high gradient.

        Args:
            max_grad: Maximum gradient threshold
            min_opacity: Minimum opacity threshold
            max_screen_size: Maximum screen size
        """
        if hasattr(self.model, "densify_and_clone"):
            self.model.densify_and_clone(max_grad, min_opacity, max_screen_size)

    def step_optimizer(self) -> None:
        """Perform one optimizer step and zero gradients."""
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def get_gradient_accumulator(self) -> torch.Tensor:
        """Get the gradient accumulator for densification.

        Returns:
            gradient_accum: (N, 1) gradient accumulator
        """
        return self.model.xyz_gradient_accum

    def get_denom(self) -> torch.Tensor:
        """Get the denominator for densification.

        Returns:
            denom: (N, 1) denominator tensor
        """
        return self.model.denom

    def update_gradient_accum(self, grad: torch.Tensor) -> None:
        """Update gradient accumulator.

        Args:
            grad: Gradient tensor
        """
        self.model.xyz_gradient_accum += grad
        self.model.denom += 1

    def get_max_radii_2d(self) -> torch.Tensor:
        """Get 2D radii for densification.

        Returns:
            radii: (N,) tensor of 2D radii
        """
        return self.model.max_radii2D

    def update_max_radii_2d(self, radii: torch.Tensor) -> None:
        """Update 2D radii.

        Args:
            radii: (N,) tensor of 2D radii
        """
        # Only update if shapes match
        if radii.shape[0] == self.model.max_radii2D.shape[0]:
            self.model.max_radii2D = torch.maximum(self.model.max_radii2D, radii)

    def get_active_mask(self, opacity_threshold: float = 0.005) -> torch.Tensor:
        """Get mask of active (visible) Gaussians.

        Args:
            opacity_threshold: Minimum opacity to be considered active

        Returns:
            active_mask: Boolean tensor
        """
        opacity = self.get_opacity()
        return (opacity > opacity_threshold).squeeze(-1)

    def get_feature_dimensions(self) -> tuple:
        """Get the SH and SG feature dimensions.

        Returns:
            (sh_degree, sg_degree) tuple
        """
        return (self.model.max_sh_degree, self.model.max_sg_degree)

    def densify(
        self,
        max_grad: float = 0.0002,
        min_opacity: float = 0.005,
        max_screen_size: int = 20,
        scene_extent: float = 6.0,
        local_mask: Optional[torch.Tensor] = None,
    ) -> None:
        """Perform densification (both clone and split).

        Args:
            max_grad: Maximum gradient threshold
            min_opacity: Minimum opacity threshold
            max_screen_size: Maximum screen size
            scene_extent: Scene extent for densification
            local_mask: Optional (N,) boolean tensor - if provided, only densify
                       Gaussians where mask is True (for local region updates)
        """
        if self.num_gaussians == 0:
            return

        # Check if there are enough gradients accumulated
        denom = self.get_denom()
        if denom.numel() > 0 and denom.sum() < 10:
            # Not enough iterations to determine densification
            return

        # Use GaussianModel's densify_and_prune method
        if hasattr(self.model, "densify_and_prune"):
            try:
                old_count = self.num_gaussians

                # Capture state before densification
                before_state = {
                    'xyz': self.model._xyz.shape[0],
                    'filter_3D_size': self.model.filter_3D.shape[0] if hasattr(self.model, 'filter_3D') else 0
                }

                # Check if model supports local densification
                if hasattr(self.model, "_densify_unified") and local_mask is not None:
                    # Use local densification
                    grads = self.get_gradient_accumulator() / self.get_denom()
                    grads[grads.isnan()] = 0.0
                    grads_abs = self.model.xyz_gradient_accum_abs / self.get_denom()
                    grads_abs[grads_abs.isnan()] = 0.0
                    ratio = (torch.norm(grads, dim=-1) >= max_grad).float().mean()
                    Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

                    self.model._densify_unified(
                        grads=grads,
                        grad_threshold=max_grad,
                        grads_abs=grads_abs,
                        grad_abs_threshold=Q,
                        scene_extent=scene_extent,
                        min_opacity=min_opacity,
                        local_mask=local_mask,
                    )

                    # Prune low opacity Gaussians
                    prune_mask = (self.model.get_opacity < min_opacity).squeeze()
                    if prune_mask.any():
                        self.model.prune_points(prune_mask)
                else:
                    # Fall back to global densification
                    self.model.densify_and_prune(
                        max_grad=max_grad,
                        min_opacity=min_opacity,
                        extent=scene_extent,
                        max_screen_size=max_screen_size,
                    )

                new_count = self.num_gaussians

                # Reset filter_3D to zeros for new Gaussians only
                if hasattr(self.model, 'filter_3D') and new_count != before_state['filter_3D_size']:
                    old_size = before_state['filter_3D_size']
                    new_size = new_count
                    if new_size > old_size:
                        # Only reset the new portion
                        new_filter = torch.zeros((new_size - old_size, 1), device=self.model.filter_3D.device)
                        self.model.filter_3D = torch.cat([self.model.filter_3D[:old_size], new_filter], dim=0)
                    elif new_size < old_size:
                        self.model.filter_3D = self.model.filter_3D[:new_size]

                if new_count != old_count:
                    print(f"Densification: {old_count} -> {new_count} Gaussians")
            except Exception as e:
                print(f"Densification failed: {e}")

    def add_densification_stats(self, viewspace_points: torch.Tensor, visibility_filter: torch.Tensor) -> None:
        """Add densification statistics.

        Args:
            viewspace_points: Viewspace points tensor
            visibility_filter: Boolean mask of visible Gaussians
        """
        if hasattr(self.model, "add_densification_stats"):
            self.model.add_densification_stats(viewspace_points, visibility_filter)

    def prune(
        self,
        min_opacity: float = 0.005,
        max_screen_size: int = 20,
    ) -> None:
        """Prune invisible Gaussians.

        Args:
            min_opacity: Minimum opacity threshold
            max_screen_size: Maximum screen size
        """
        if self.num_gaussians == 0:
            return

        opacity = self.get_opacity()
        prune_mask = (opacity.squeeze(-1) < min_opacity)

        # Also prune based on 2D screen size
        if max_screen_size > 0:
            max_radii = self.get_max_radii_2d()
            prune_mask = torch.logical_or(prune_mask, max_radii > max_screen_size)

        if prune_mask.any():
            self.prune_gaussians(prune_mask)

    def get_xyz(self) -> torch.Tensor:
        """Get Gaussian positions (alias for get_positions).

        Returns:
            positions: (N, 3) tensor of positions
        """
        return self.model.get_xyz

    def get_rotation(self) -> torch.Tensor:
        """Get Gaussian rotations.

        Returns:
            rotations: (N, 4) tensor of quaternions
        """
        return self.model.get_rotation

    def get_scaling(self) -> torch.Tensor:
        """Get Gaussian scales (in log space).

        Returns:
            scaling: (N, 3) tensor of log scales
        """
        return self.model._scaling
