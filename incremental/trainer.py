"""Incremental Trainer.

Trainer that adapts cl-splats workflow to use the project's GaussianModel
with diff_gaussian_rasterization.
"""

import sys
import torch
import torch.nn.functional as F
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import logging
from collections import defaultdict

# Add cl-splats to path
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting/submodules/cl-splats')

from scene.gaussian_model import GaussianModel
from .gaussian_adapter import GGSGaussianAdapter
from .render_adapter import GGSRenderAdapter
from .lifter.depth_anything_lifter import DepthAnythingLifter

# Import cl-splats components
try:
    from clsplats.change_detection.dinov2_detector import DinoV2Detector
    from clsplats.config import CLSplatsConfig
    from clsplats.constraints.primitives import fit_primitives_for_active, union_distance
    from clsplats.dataset.cameras import Camera
    CL_SPLATS_AVAILABLE = True
except ImportError as e:
    CL_SPLATS_AVAILABLE = False
    DinoV2Detector = None
    CLSplatsConfig = None
    print(f"Warning: Failed to import cl-splats: {e}")

log = logging.getLogger(__name__)


@dataclass
class IncrementalTrainerConfig:
    """Configuration for incremental training."""

    # Change detection
    change_threshold: float = 0.8
    change_dilate_kernel: int = 31

    # Densification
    densify_enabled: bool = True
    densify_grad_threshold: float = 0.0002
    densify_opacity_threshold: float = 0.005
    densify_screen_size_threshold: int = 20

    # Pruning
    prune_enabled: bool = True
    prune_opacity_threshold: float = 0.005
    prune_screen_size_threshold: int = 20
    prune_every: int = 100

    # Training
    lr: float = 1e-3
    iters_per_timestep: int = 500
    log_interval: int = 10

    # SG features
    use_sg_features: bool = True
    use_sg_guidance: bool = True
    sg_lift_weight: float = 0.3

    # Constraints
    use_constraints: bool = True
    constraint_group_radius: float = 0.1

    # Paths
    output_dir: str = "./output_incremental"


@dataclass
class TrainingMetrics:
    """Container for training metrics."""

    loss: float = 0.0
    psnr: float = 0.0
    ssim: float = 0.0
    num_gaussians: int = 0
    num_active: int = 0


class IncrementalTrainer:
    """Incremental trainer using the project's GaussianModel.

    This trainer adapts the cl-splats workflow to use:
    - GaussianModel from this project (instead of CLGaussians)
    - diff_gaussian_rasterization (instead of gsplat)
    - SG feature support

    Supports change detection, 2D->3D lifting, and geometric constraints.
    """

    def __init__(
        self,
        gaussian_model: GaussianModel,
        cameras: List,
        cfg: Optional[IncrementalTrainerConfig] = None,
        clsplats_config: Optional[CLSplatsConfig] = None,
    ):
        """Initialize the incremental trainer.

        Args:
            gaussian_model: The GaussianModel to train
            cameras: List of camera objects
            cfg: Incremental trainer configuration
            clsplats_config: Optional cl-splats configuration
        """
        self.cfg = cfg or IncrementalTrainerConfig()
        self.clsplats_cfg = clsplats_config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gaussian_model = gaussian_model
        self.cameras = cameras

        # Create adapter for GaussianModel
        self.adapter = GGSGaussianAdapter(gaussian_model)

        # Create render adapter
        self.render_adapter = GGSRenderAdapter(gaussian_model)

        # Initialize change detector
        self.detector = None
        log.info(f"CL_SPLATS_AVAILABLE: {CL_SPLATS_AVAILABLE}")
        log.info(f"clsplats_config: {clsplats_config}")
        try:
            if CL_SPLATS_AVAILABLE and DinoV2Detector is not None:
                if self.clsplats_cfg is not None:
                    log.info("Creating DinoV2Detector with config...")
                    self.detector = DinoV2Detector(self.clsplats_cfg.change)
                    log.info("DinoV2Detector created successfully")
                else:
                    # Create minimal config
                    from dataclasses import dataclass
                    @dataclass
                    class MinimalChangeConfig:
                        threshold: float = 0.8
                        dilate_mask: bool = False
                        dilate_kernel_size: int = 31
                        upsample: bool = True
                    log.info("Creating DinoV2Detector with minimal config...")
                    self.detector = DinoV2Detector(MinimalChangeConfig())
                    log.info("DinoV2Detector created successfully")
        except Exception as e:
            log.error(f"Failed to create detector: {e}")

        # Initialize lifter (if cl-splats available)
        self.lifter = None
        self.sg_lifter = None
        if CL_SPLATS_AVAILABLE and self.clsplats_cfg is not None:
            from .lifter.depth_anything_lifter import DepthAnythingLifter
            base_lifter = DepthAnythingLifter(
                depth_model=self.clsplats_cfg.lifter.depth_model,
                k_nn=self.clsplats_cfg.lifter.k_nn,
                local_radius_thresh=self.clsplats_cfg.lifter.local_radius_thresh,
                depth_tol_abs=self.clsplats_cfg.lifter.depth_tol_abs,
                depth_tol_rel=self.clsplats_cfg.lifter.depth_tol_rel,
                lambda_seed=self.clsplats_cfg.lifter.lambda_seed,
                lambda_neg=self.clsplats_cfg.lifter.lambda_neg,
                min_visible_views=self.clsplats_cfg.lifter.min_visible_views,
                min_positive_views=self.clsplats_cfg.lifter.min_positive_views,
                min_seed_views=self.clsplats_cfg.lifter.min_seed_views,
                min_positive_ratio=self.clsplats_cfg.lifter.min_positive_ratio,
                final_thresh=self.clsplats_cfg.lifter.final_thresh,
            )
            # Use the new lifter directly
            self.sg_lifter = base_lifter

        # Training state
        self.timestep = 0
        self.active_mask: Optional[torch.Tensor] = None
        self._global_step = 0

        # Primitives for constraints
        self._primitives: list = []
        self._outside_counts = None

        # Create output directory
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def detect_changes(
        self,
        rendered_image: torch.Tensor,
        target_image: torch.Tensor,
    ) -> torch.Tensor:
        """Detect changes between rendered and target images.

        Args:
            rendered_image: (C, H, W) rendered image
            target_image: (C, H, W) target image

        Returns:
            change_mask: (H, W) binary mask of changes
        """
        if self.detector is None:
            # Fallback: simple difference-based detection
            diff = torch.abs(rendered_image - target_image).mean(dim=0)
            mask = diff > 0.1
            return mask

        # Use DINOv2 detector
        with torch.no_grad():
            change_mask = self.detector.predict_change_mask(
                rendered_image.permute(1, 2, 0),  # (H, W, C)
                target_image.permute(1, 2, 0),
            )
        return change_mask

    def lift_changes_to_3d(
        self,
        change_masks: List[torch.Tensor],
        cameras: List = None,
    ) -> torch.Tensor:
        """Lift 2D change masks to 3D Gaussian mask.

        Args:
            change_masks: List of change masks for each camera
            cameras: List of cameras corresponding to change_masks

        Returns:
            3D change mask: (N,) boolean mask for Gaussians
        """
        if self.sg_lifter is not None:
            # Use SG-aware lifter (use cameras param, not self.cameras)
            result = self.sg_lifter.lift(
                self.adapter,
                cameras,  # Use the filtered cameras that have change_masks
                change_masks,
            )
            return result.positive_mask

        elif self.lifter is not None:
            # Use base lifter
            result = self.lifter.lift(self.adapter, cameras, change_masks)
            return result.positive_mask

        else:
            # Fallback: use visibility from change detection cameras
            # Combine visibility filters from all cameras used for change detection
            # This limits updates to Gaussians visible in the change detection views
            log.info("Using simplified active mask from visibility filters")
            if cameras is not None:
                active_mask = torch.zeros(
                    self.adapter.num_gaussians,
                    dtype=torch.bool,
                    device=self.device,
                )
                for cam, change_mask in zip(cameras, change_masks):
                    # Render to get visibility
                    output = self.render_camera(cam)
                    visibility_filter = output.get("visibility_filter", None)
                    if visibility_filter is not None:
                        # Only mark Gaussians as active if they are:
                        # 1. Visible in this camera AND
                        # 2. Contributing to changed pixels
                        # For simplicity, we use visibility as proxy
                        active_mask |= visibility_filter
                        log.info(f"Camera {cam.colmap_id if hasattr(cam, 'colmap_id') else '?'}: {visibility_filter.sum()} visible Gaussians")
                return active_mask
            else:
                return self.adapter.get_active_mask()

    def compute_render_loss(
        self,
        rendered_image: torch.Tensor,
        target_image: torch.Tensor,
        active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute rendering loss.

        Args:
            rendered_image: (C, H, W) rendered image
            target_image: (C, H, W) target image
            active_mask: Optional mask for active Gaussians

        Returns:
            loss: Scalar loss value
        """
        # Ensure image sizes match by resizing
        if rendered_image.shape != target_image.shape:
            # Need to resize target to match rendered
            target_image = F.interpolate(
                target_image.unsqueeze(0),
                size=(rendered_image.shape[1], rendered_image.shape[2]),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        loss = F.mse_loss(rendered_image, target_image)

        # Add PSNR for logging
        mse = F.mse_loss(rendered_image, target_image)
        psnr = 10 * torch.log10(1.0 / mse.clamp(min=1e-10))

        return loss, psnr

    def render_camera(self, camera) -> Dict[str, torch.Tensor]:
        """Render from a single camera using base GaussianModel directly.

        Args:
            camera: Camera object

        Returns:
            Dict with 'image', 'depth', 'alpha', 'radii', 'viewspace_points', 'visibility_filter'
        """
        # Import gaussian_renderer here to avoid circular imports
        from gaussian_renderer import render as ggs_render

        # Create simple pipe params if not available
        class SimplePipeParams:
            convert_SHs_python = False
            compute_cov3D_python = False
            debug = False

        pipe = SimplePipeParams()
        bg_color = torch.zeros(3, device=self.device)
        kernel_size = 1

        try:
            result = ggs_render(
                viewpoint_camera=camera,
                pc=self.gaussian_model,
                pipe=pipe,
                bg_color=bg_color,
                kernel_size=kernel_size,
                scaling_modifier=1.0,
                require_depth=True,
            )
        except Exception as e:
            print(f"Render failed: {e}")
            raise

        return {
            "image": result["render"],
            "depth": result["median_depth"],
            "alpha": result["mask"],
            "radii": result["radii"],
            "viewspace_points": result.get("viewspace_points"),
            "visibility_filter": result.get("visibility_filter"),
        }

    def apply_constraints(self) -> None:
        """Apply geometric constraints to Gaussians."""
        # Skip constraints for now - they require more integration with cl-splats
        return

        if not self.cfg.use_constraints:
            return

        # Try to use cl-splats constraints
        try:
            if self._primitives is None or len(self._primitives) == 0:
                # Fit primitives to current Gaussians
                if self.active_mask is not None and self.active_mask.any():
                    active_indices = torch.where(self.active_mask)[0]
                    positions = self.adapter.get_positions()[active_indices]
                    scales = self.adapter.get_scales()[active_indices]

                    self._primitives = fit_primitives_for_active(
                        positions, scales, radius_frac=self.cfg.constraint_group_radius
                    )
                return

            # Compute distance to primitives and mark outliers
            positions = self.adapter.get_positions()
            outside_mask = union_distance(positions, self._primitives)

            # Gradually prune outliers
            if self._outside_counts is None:
                self._outside_counts = torch.zeros(positions.shape[0], device=self.device)

            self._outside_counts += outside_mask.float()

            # Prune after consecutive violations
            prune_threshold = 3
            prune_mask = self._outside_counts > prune_threshold

            if prune_mask.any():
                self.adapter.prune_gaussians(prune_mask)
                self._outside_counts = None  # Reset after pruning
        except Exception as e:
            print(f"Constraints application failed: {e}")

        # Gradually prune outliers
        if self._outside_counts is None:
            self._outside_counts = torch.zeros(positions.shape[0], device=self.device)

        self._outside_counts += outside_mask.float()

        # Prune after consecutive violations
        prune_threshold = 3
        prune_mask = self._outside_counts > prune_threshold

        if prune_mask.any():
            self.adapter.prune_gaussians(prune_mask)
            self._outside_counts = None  # Reset after pruning

    def train_step(
        self,
        camera,
        target_image: torch.Tensor,
    ) -> TrainingMetrics:
        """Perform one training step with incremental update.

        Uses base GaussianModel directly instead of adapter.

        Args:
            camera: Camera for rendering
            target_image: (C, H, W) target image

        Returns:
            TrainingMetrics for this step
        """
        # Forward: render
        output = self.render_camera(camera)
        rendered_image = output["image"]

        # Compute loss
        loss, psnr = self.compute_render_loss(rendered_image, target_image)

        # Backward
        loss.backward()

        # Apply gradient mask to only update active (changed) Gaussians
        if self.active_mask is not None:
            self._apply_gradient_mask()

        # Update optimizer directly
        self.gaussian_model.optimizer.step()
        self.gaussian_model.optimizer.zero_grad(set_to_none=True)

        # Update 2D radii directly
        if output["radii"] is not None:
            if self.gaussian_model.max_radii2D.shape[0] == output["radii"].shape[0]:
                self.gaussian_model.max_radii2D = torch.maximum(
                    self.gaussian_model.max_radii2D, output["radii"]
                )

        # Add densification stats directly
        if output.get("viewspace_points") is not None and output.get("visibility_filter") is not None:
            self.gaussian_model.add_densification_stats(
                output["viewspace_points"],
                output["visibility_filter"]
            )

        # Metrics
        num_gaussians = self.gaussian_model._xyz.shape[0]
        num_active = self.active_mask.sum().item() if self.active_mask is not None else num_gaussians
        metrics = TrainingMetrics(
            loss=loss.item(),
            psnr=psnr.item(),
            num_gaussians=num_gaussians,
            num_active=num_active,
        )

        self._global_step += 1
        return metrics

    def _apply_gradient_mask(self) -> None:
        """Apply gradient mask to only update active Gaussians.

        Sets gradients of non-active Gaussians to zero so they won't be updated.

        Note: active_mask should be kept in sync by densify_and_prune() and _prune_only().
        If there's a size mismatch, it indicates a bug - don't auto-reset.
        """
        if self.active_mask is None:
            return

        # Check if active_mask size matches current model size
        model = self.gaussian_model
        current_num_gaussians = model._xyz.shape[0]

        if self.active_mask.shape[0] != current_num_gaussians:
            # Size mismatch - this should not happen if densify/prune properly update active_mask
            # Log error but don't auto-reset (would lose information about active region)
            print(f"ERROR: Active mask size mismatch: {self.active_mask.shape[0]} vs {current_num_gaussians}")
            print("This indicates densify_and_prune or _prune_only didn't properly update active_mask")
            # Don't reset - let the code fail so we can debug
            return

        active_mask = self.active_mask

        # Get all parameter groups that need gradient masking
        param_groups = [
            ('_xyz', model._xyz),
            ('_features_dc', model._features_dc),
            ('_features_rest', model._features_rest),
            ('_opacity', model._opacity),
            ('_scaling', model._scaling),
            ('_rotation', model._rotation),
            ('_sg_axis', model._sg_axis),
            ('_sg_sharpness', model._sg_sharpness),
            ('_sg_color', model._sg_color),
        ]

        for param_name, param in param_groups:
            if param is None or param.numel() == 0:
                continue

            if not param.requires_grad or param.grad is None:
                continue

            # Create inverted mask (True = should be zeroed)
            # active_mask is boolean, we need to expand it properly
            param_shape = param.shape

            # Handle different parameter shapes
            if len(param_shape) == 2:
                # Parameters with feature dimension (N, D)
                # Only mask the first dimension (N)
                mask = active_mask.view(-1, 1).expand_as(param.grad)
            elif len(param_shape) == 3:
                # Parameters with (N, D1, D2)
                mask = active_mask.view(-1, 1, 1).expand_as(param.grad)
            else:
                # 1D parameters (N,)
                mask = active_mask.view(-1).expand_as(param.grad)

            # Zero out gradients for non-active Gaussians
            param.grad = param.grad * mask

    def densify_and_prune(self) -> None:
        """Perform densification and pruning ONLY within the active region.

        This method only densifies/prunes Gaussians that are in the active region,
        to reduce computation and preserve non-active Gaussians.
        """
        if self.gaussian_model._xyz.shape[0] == 0:
            return

        if self.active_mask is None:
            # No active mask - operate on all Gaussians
            self._densify_prune_all()
            return

        num_before = self.gaussian_model._xyz.shape[0]
        active_before = self.active_mask.sum().item()

        try:
            # Step 1: Extract only active Gaussians for densification
            active_indices = torch.where(self.active_mask)[0]
            print(f"Densifying {len(active_indices)} active Gaussians (total: {num_before})")

            # Step 2: Call the base model's densify_and_prune but only for active
            # We need to temporarily create a view of the model with only active Gaussians
            # The simplest way is to call a custom densification on active subset only

            self._densify_active_only(active_indices)

            # Step 3: Update active_mask
            num_after = self.gaussian_model._xyz.shape[0]
            if num_after != num_before:
                new_gaussians = num_after - num_before
                if new_gaussians > 0:
                    # New Gaussians added at end - mark them active
                    new_active = torch.ones(new_gaussians, dtype=torch.bool, device=self.device)
                    self.active_mask = torch.cat([self.active_mask[:num_after - new_gaussians], new_active])
                else:
                    # Gaussians removed - slice active_mask
                    self.active_mask = self.active_mask[:num_after].clone()

                print(f"Active mask updated: {num_before}->{num_after}, active {active_before}->{self.active_mask.sum()}")

        except Exception as e:
            print(f"Densification/pruning failed: {e}")
            import traceback
            traceback.print_exc()

    def _densify_active_only(self, active_indices):
        """Densify only the active Gaussians.

        This is a simplified version that operates only on the active set.
        """
        model = self.gaussian_model
        num_active = len(active_indices)

        if num_active == 0:
            return

        # Compute gradients for active Gaussians only
        grads = model.xyz_gradient_accum[active_indices] / model.denom[active_indices]
        grads[grads.isnan()] = 0.0
        grads_abs = model.xyz_gradient_accum_abs[active_indices] / model.denom[active_indices]

        scene_extent = getattr(self, '_cached_scene_extent', 6.0)

        # Compute threshold based on active Gaussians only
        ratio = (torch.norm(grads, dim=-1) >= self.cfg.densify_grad_threshold).float().mean()
        Q = torch.quantile(grads_abs.reshape(-1), 1 - ratio)

        # Densify and clone (small Gaussians)
        clone_mask = torch.where(
            torch.norm(grads, dim=-1) >= self.cfg.densify_grad_threshold, True, False
        )
        clone_mask = torch.logical_and(
            clone_mask,
            torch.max(model.get_scaling[active_indices], dim=1).values <= model.percent_dense * scene_extent
        )

        if clone_mask.any():
            # Get properties of Gaussians to clone
            sel_idx = active_indices[clone_mask]
            new_xyz = model._xyz[sel_idx]
            stds = model.get_scaling[sel_idx]
            means = torch.zeros((stds.size(0), 3), device=model.get_xyz.device)
            samples = torch.normal(mean=means, std=stds)
            from scene.gaussian_model import build_rotation
            rots = build_rotation(model._rotation[sel_idx])
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + model.get_xyz[sel_idx]

            new_features_dc = model._features_dc[sel_idx]
            new_features_rest = model._features_rest[sel_idx]
            new_opacities = model._opacity[sel_idx]
            new_scaling = model._scaling[sel_idx]
            new_rotation = model._rotation[sel_idx]
            new_sg_axis = model._sg_axis[sel_idx]
            new_sg_sharpness = model._sg_sharpness[sel_idx]
            new_sg_color = model._sg_color[sel_idx]

            model.densification_postfix(
                new_xyz, new_features_dc, new_features_rest, new_opacities,
                new_scaling, new_rotation, new_sg_axis, new_sg_sharpness, new_sg_color
            )
            print(f"  Cloned {clone_mask.sum()} Gaussians")

        # Densify and split (large Gaussians)
        grads_for_split = torch.zeros((model.get_xyz.shape[0], 1), device=model.get_xyz.device)
        grads_for_split[active_indices] = grads
        grads_abs_for_split = torch.zeros((model.get_xyz.shape[0], 1), device=model.get_xyz.device)
        grads_abs_for_split[active_indices] = grads_abs

        split_mask = torch.where(grads_for_split.squeeze() >= self.cfg.densify_grad_threshold, True, False)
        split_mask_abs = torch.where(grads_abs_for_split.squeeze() >= Q, True, False)
        split_mask = torch.logical_or(split_mask, split_mask_abs)
        split_mask = torch.logical_and(
            split_mask,
            torch.max(model.get_scaling, dim=1).values > model.percent_dense * scene_extent
        )

        # Only split Gaussians that are in the active set
        split_mask = split_mask & self.active_mask

        if split_mask.any():
            N = 2
            sel_idx = torch.where(split_mask)[0]
            stds = model.get_scaling[sel_idx].repeat(N, 1)
            means = torch.zeros((stds.size(0), 3), device=model.get_xyz.device)
            samples = torch.normal(mean=means, std=stds)
            from scene.gaussian_model import build_rotation
            rots = build_rotation(model._rotation[sel_idx]).repeat(N, 1, 1)
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + model.get_xyz[sel_idx].repeat(N, 1)

            new_scaling = model.scaling_inverse_activation(model.get_scaling[sel_idx].repeat(N, 1) / (0.8 * N))
            new_rotation = model._rotation[sel_idx].repeat(N, 1)
            new_features_dc = model._features_dc[sel_idx].repeat(N, 1, 1)
            new_features_rest = model._features_rest[sel_idx].repeat(N, 1, 1)
            new_opacities = model._opacity[sel_idx].repeat(N, 1)
            new_sg_axis = model._sg_axis[sel_idx].repeat(N, 1, 1)
            new_sg_sharpness = model._sg_sharpness[sel_idx].repeat(N, 1)
            new_sg_color = model._sg_color[sel_idx].repeat(N, 1, 1)

            model.densification_postfix(
                new_xyz, new_features_dc, new_features_rest, new_opacities,
                new_scaling, new_rotation, new_sg_axis, new_sg_sharpness, new_sg_color
            )
            print(f"  Split {split_mask.sum()} Gaussians into {N}x")

        print(f"Densification complete. Gaussians: {model._xyz.shape[0]}")

    def _densify_prune_all(self) -> None:
        """Internal method to run densification and pruning on all Gaussians."""
        # Compute scene extent from camera positions (cached)
        if not hasattr(self, '_cached_scene_extent'):
            scene_extent = 6.0
            if hasattr(self, 'cameras') and len(self.cameras) > 0:
                try:
                    cam_centers = []
                    for cam in self.cameras:
                        if hasattr(cam, 'camera_center'):
                            cam_centers.append(cam.camera_center)
                    if cam_centers:
                        cam_centers = torch.stack(cam_centers)
                        scene_extent = torch.norm(
                            cam_centers - cam_centers.mean(dim=0, keepdim=True), dim=1
                        ).max().item()
                except:
                    pass
            self._cached_scene_extent = scene_extent

        scene_extent = self._cached_scene_extent

        # Use base model's densify_and_prune directly
        # This handles:
        # 1. Gradient-based densification (clone/split)
        # 2. Opacity-based pruning
        # 3. filter_3D synchronization via densification_postfix
        self.gaussian_model.densify_and_prune(
            max_grad=self.cfg.densify_grad_threshold,
            min_opacity=self.cfg.densify_opacity_threshold,
            extent=scene_extent,
            max_screen_size=self.cfg.densify_screen_size_threshold,
        )
        print(f"Densification/pruning complete. Gaussians: {self.gaussian_model._xyz.shape[0]}")

    def train_timestep(
        self,
        cameras: List,
        target_images: List[torch.Tensor],
    ) -> List[TrainingMetrics]:
        """Train for one timestep.

        Args:
            cameras: List of cameras
            target_images: List of target images

        Returns:
            List of TrainingMetrics for each iteration
        """
        all_metrics = []

        # Training loop - NO pruning/densification inside the loop
        # Pruning inside the loop corrupts the autograd graph
        for i in range(self.cfg.iters_per_timestep):
            # Sample a camera
            idx = i % len(cameras)
            camera = cameras[idx]
            target = target_images[idx]

            # Training step
            metrics = self.train_step(camera, target)

            # Logging
            if i % self.cfg.log_interval == 0:
                active_count = self.active_mask.sum().item() if self.active_mask is not None else metrics.num_gaussians
                log.info(
                    f"Step {i}/{self.cfg.iters_per_timestep}: "
                    f"Loss={metrics.loss:.4f}, PSNR={metrics.psnr:.2f}, "
                    f"N={metrics.num_gaussians}, Active={active_count}"
                )

            all_metrics.append(metrics)

        # Pruning and densification AFTER the training loop is complete
        # This avoids corrupting the autograd graph
        if self.cfg.prune_enabled and self._global_step % self.cfg.prune_every == 0:
            self._prune_only()

        if self.cfg.densify_enabled:
            self.densify_and_prune()

        return all_metrics

    def _prune_only(self) -> None:
        """Prune only (without densification) using base GaussianModel directly.

        This method only prunes inactive Gaussians that meet prune criteria.
        Active Gaussians are protected from pruning to allow densification.
        """
        if not self.cfg.prune_enabled:
            return

        num_gaussians = self.gaussian_model._xyz.shape[0]
        if num_gaussians == 0:
            return

        try:
            # Get device from Gaussian model
            device = self.gaussian_model._xyz.device

            # Ensure active_mask is on the same device as Gaussians
            if self.active_mask is not None and self.active_mask.device != device:
                self.active_mask = self.active_mask.to(device=device)

            # Get opacity directly from model
            opacity = self.gaussian_model.get_opacity
            prune_mask = (opacity.squeeze(-1) < self.cfg.prune_opacity_threshold)

            # Also prune based on 2D screen size
            max_radii = self.gaussian_model.max_radii2D
            if max_radii is not None and max_radii.numel() > 0 and max_radii.shape[0] == prune_mask.shape[0]:
                # Ensure max_radii is on same device as prune_mask
                if max_radii.device != prune_mask.device:
                    max_radii = max_radii.to(device=prune_mask.device)
                prune_mask = torch.logical_or(prune_mask, max_radii > self.cfg.prune_screen_size_threshold)

            # Only prune inactive Gaussians that meet criteria
            # Active Gaussians are protected from pruning to allow densification
            if self.active_mask is not None and self.active_mask.shape[0] == num_gaussians:
                prune_mask = prune_mask & (~self.active_mask)  # Only prune inactive

            if prune_mask.any():
                valid_points_mask = ~prune_mask

                # Update active mask before pruning (we only keep valid_points_mask Gaussians)
                if self.active_mask is not None:
                    self.active_mask = self.active_mask[valid_points_mask]

                self.gaussian_model.prune_points(prune_mask)
                print(f"Pruned {prune_mask.sum()} Gaussians, active={self.active_mask.sum() if self.active_mask is not None else 'N/A'}")

        except Exception as e:
            print(f"Pruning failed: {e}")

    def run_incremental_update(
        self,
        new_cameras: List,
        new_images: List[torch.Tensor],
        detect_changes: bool = True,
        change_detection_camera_ids: List[int] = None,
    ) -> Dict[str, Any]:
        """Run incremental update with new observations.

        Args:
            new_cameras: List of cameras for training
            new_images: List of target images
            detect_changes: Whether to detect changes
            change_detection_camera_ids: List of camera IDs to use for change detection.
                                         If None, uses all cameras (backward compatible).

        Returns:
            Dict with training results
        """
        log.info(f"Starting incremental update with {len(new_cameras)} views")
        log.info(f"Change detection enabled: {detect_changes}")
        log.info(f"Detector available: {self.detector is not None}")
        log.info(f"Change detection camera IDs: {change_detection_camera_ids}")

        # Set up active mask (all Gaussians initially)
        N = self.adapter.num_gaussians
        self.active_mask = torch.ones(N, dtype=torch.bool, device=self.device)

        # If change detection is enabled (supports fallback if detector is None)
        if detect_changes:
            # Determine which cameras to use for change detection
            if change_detection_camera_ids is not None:
                # Filter to only use specified cameras for change detection
                cd_cameras = []
                cd_images = []
                for cam, img in zip(new_cameras, new_images):
                    if hasattr(cam, 'colmap_id') and cam.colmap_id in change_detection_camera_ids:
                        cd_cameras.append(cam)
                        cd_images.append(img)
                log.info(f"Filtered to {len(cd_cameras)} cameras for change detection")
            else:
                # Use all cameras (backward compatible)
                cd_cameras = new_cameras
                cd_images = new_images

            log.info("Running change detection...")
            if self.detector is None:
                log.info("Using fallback change detection (simple difference)")
            change_masks = []

            for i, (camera, target) in enumerate(zip(cd_cameras, cd_images)):
                # Render current state
                output = self.render_camera(camera)
                rendered = output["image"]

                # Detect changes
                change_mask = self.detect_changes(rendered, target)
                change_masks.append(change_mask)
                cam_id = camera.colmap_id if hasattr(camera, 'colmap_id') else i
                log.info(f"Camera {cam_id}: change mask sum = {change_mask.sum()}")

            # Lift changes to 3D
            log.info("Lifting 2D changes to 3D...")
            self.active_mask = self.lift_changes_to_3d(change_masks, cd_cameras)

            log.info(f"Active Gaussians after change detection: {self.active_mask.sum()} / {N}")
        else:
            log.info("Change detection skipped, using all Gaussians")

        # Train
        # Use only the cameras that were used for change detection to avoid
        # pulling active Gaussians toward views that don't see the changed regions
        if change_detection_camera_ids is not None:
            train_cameras = cd_cameras
            train_images = cd_images
            log.info(f"Training with {len(train_cameras)} cameras (change detection subset)")
        else:
            train_cameras = new_cameras
            train_images = new_images

        metrics = self.train_timestep(train_cameras, train_images)

        # Final densify and prune
        self.densify_and_prune()

        return {
            "metrics": metrics,
            "num_gaussians": self.adapter.num_gaussians,
            "num_active": self.active_mask.sum().item() if self.active_mask is not None else 0,
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model": self.gaussian_model.capture(),
            "step": self._global_step,
        }
        torch.save(checkpoint, path)
        log.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.gaussian_model.restore(checkpoint["model"])
        self._global_step = checkpoint["step"]
        log.info(f"Loaded checkpoint from {path}")


def create_trainer(
    gaussian_model: GaussianModel,
    cameras: List,
    config_path: Optional[str] = None,
) -> IncrementalTrainer:
    """Create an incremental trainer.

    Args:
        gaussian_model: The GaussianModel to train
        cameras: List of cameras
        config_path: Optional path to config

    Returns:
        IncrementalTrainer instance
    """
    cfg = IncrementalTrainerConfig()

    if config_path is not None:
        # Load config from file if provided
        import yaml
        with open(config_path) as f:
            config_dict = yaml.safe_load(f)
        # Update cfg with loaded values
        for key, value in config_dict.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)

    return IncrementalTrainer(gaussian_model, cameras, cfg)
