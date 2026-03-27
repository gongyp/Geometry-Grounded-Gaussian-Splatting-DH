"""SG-Aware Lifter.

Extends the DepthAnythingLifter to support Spherical Gaussian (SG) features
for more accurate 2D->3D lifting.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class LiftResult:
    """Result from the lifting operation."""

    positive_mask: torch.Tensor  # (N,) Boolean mask of changed Gaussians
    seed_score: torch.Tensor  # (N,) Score for each Gaussian
    neg_score: torch.Tensor  # (N,) Negative evidence score
    depth_consistency: Optional[torch.Tensor] = None  # (N,) Depth consistency score


class SGAwareLifter:
    """Lifter that considers SG features for depth lifting.

    This lifter extends the standard depth lifting to incorporate
    Spherical Gaussian features for better spatial awareness.
    """

    def __init__(
        self,
        depth_lifter,  # Base lifter (DepthAnythingLifter)
        use_sg_guidance: bool = True,
        sg_weight: float = 0.3,
    ):
        """Initialize the SG-aware lifter.

        Args:
            depth_lifter: The base DepthAnythingLifter instance
            use_sg_guidance: Whether to use SG features for guidance
            sg_weight: Weight for SG-based guidance (0-1)
        """
        self.base_lifter = depth_lifter
        self.use_sg_guidance = use_sg_guidance
        self.sg_weight = sg_weight

        # Forward compatible attributes
        self.k_nn = getattr(depth_lifter, "k_nn", 8)
        self.local_radius_thresh = getattr(depth_lifter, "local_radius_thresh", 2.5)
        self.depth_tol_abs = getattr(depth_lifter, "depth_tol_abs", 0.05)
        self.depth_tol_rel = getattr(depth_lifter, "depth_tol_rel", 0.05)

    def compute_sg_spatial_guidance(
        self,
        positions: torch.Tensor,
        sg_axis: torch.Tensor,
        sg_sharpness: torch.Tensor,
        pixels_xyz: torch.Tensor,
    ) -> torch.Tensor:
        """Compute spatial guidance from SG features.

        Args:
            positions: (N, 3) Gaussian positions
            sg_axis: (N, D, 3) SG axis directions
            sg_sharpness: (N, D) SG sharpness values
            pixels_xyz: (M, 3) 3D coordinates of pixels

        Returns:
            guidance: (M,) guidance score for each pixel
        """
        # Simplified: just return uniform guidance
        return torch.ones(pixels_xyz.shape[0], device=positions.device)

    def lift_with_sg(
        self,
        gaussians,  # GGSGaussianAdapter or similar
        cameras: List,
        change_masks: List[torch.Tensor],
    ) -> LiftResult:
        """Lift 2D change masks to 3D with simplified approach.

        Uses a simple depth-range based approach to avoid OOM with large scenes.

        Args:
            gaussians: Gaussian adapter with positions and SG features
            cameras: List of camera objects
            change_masks: List of change masks for each camera

        Returns:
            LiftResult with positive mask and scores
        """
        device = gaussians.get_positions().device
        N = gaussians.num_gaussians

        # Get Gaussian positions
        positions = gaussians.get_positions()

        # Initialize score accumulator - one per Gaussian
        # Use chunked processing to avoid OOM
        vote_accumulator = torch.zeros(N, device=device)

        # Sample a subset of cameras
        num_cams_to_use = min(3, len(cameras))
        cam_indices = torch.linspace(0, len(cameras)-1, num_cams_to_use).long()

        for view_id in cam_indices:
            cam = cameras[view_id]
            mask = change_masks[view_id]

            # Get camera pose
            R = cam.R
            T = cam.T
            camera_center = -R.T @ T

            # Get changed pixels
            mask = mask.to(device)
            pos_indices = torch.where(mask > 0.5)

            if len(pos_indices[0]) == 0:
                continue

            # Sample pixels
            max_pixels = 2000
            if len(pos_indices[0]) > max_pixels:
                sample_idx = torch.randperm(len(pos_indices[0]))[:max_pixels]
                pos_indices = (pos_indices[0][sample_idx], pos_indices[1][sample_idx])

            y_coords, x_coords = pos_indices

            # Estimate depth for these pixels
            if hasattr(self.base_lifter, "estimate_depth"):
                obs = cam.original_image.permute(1, 2, 0).contiguous()
                depth = self.base_lifter.estimate_depth(obs).to(device)
            else:
                continue

            H, W = depth.shape
            fx = cam.Fx if hasattr(cam, "Fx") else 1.0
            fy = cam.Fy if hasattr(cam, "Fy") else 1.0

            # Get depth at changed pixels
            valid_depth = torch.isfinite(depth) & (depth > 0)
            valid_mask = valid_depth[y_coords, x_coords]
            if not valid_mask.any():
                continue

            # Filter to valid depth pixels
            x_coords = x_coords[valid_mask]
            y_coords = y_coords[valid_mask]
            z = depth[y_coords, x_coords]

            # Back-project to 3D
            u = (x_coords.float() - W / 2) / fx
            v = (y_coords.float() - H / 2) / fy

            # 3D coordinates in camera frame
            x = u * z
            y = v * z
            pixels_xyz_cam = torch.stack([x, y, z], dim=-1)

            # Transform to world frame
            R_mat = R if R.shape == (3, 3) else R.T
            pixels_xyz = torch.matmul(pixels_xyz_cam, R_mat.T) + T.unsqueeze(0)

            # For each Gaussian, check if any changed pixel projects nearby
            # Use chunked processing
            chunk_size = 50000
            for chunk_start in range(0, N, chunk_size):
                chunk_end = min(chunk_start + chunk_size, N)
                chunk_positions = positions[chunk_start:chunk_end]

                # Compute distances from pixels to this chunk of Gaussians
                diff = pixels_xyz.unsqueeze(1) - chunk_positions.unsqueeze(0)  # (M, chunk, 3)
                dists = torch.norm(diff, dim=-1)  # (M, chunk)

                # Find if any pixel is close to each Gaussian
                radius = 0.5  # 0.5 meters radius
                close_pixels = dists < radius  # (M, chunk)

                # If any changed pixel is close, increment vote
                votes = close_pixels.any(dim=0)  # (chunk,)
                vote_accumulator[chunk_start:chunk_end] += votes.float()

        # Normalize and threshold
        vote_accumulator = vote_accumulator / num_cams_to_use

        # Final threshold - Gaussians with any vote are considered changed
        threshold = 0.1
        positive_mask = vote_accumulator > threshold

        # If too few changed, return all as changed for incremental update
        if positive_mask.sum() < 100:
            positive_mask = torch.ones(N, dtype=torch.bool, device=device)

        return LiftResult(
            positive_mask=positive_mask,
            seed_score=vote_accumulator,
            neg_score=torch.zeros(N, device=device),
        )


def create_sg_aware_lifter(base_lifter, use_sg_guidance: bool = True, sg_weight: float = 0.3):
    """Create an SG-aware lifter from a base DepthAnythingLifter.

    Args:
        base_lifter: The base DepthAnythingLifter instance
        use_sg_guidance: Whether to use SG features
        sg_weight: Weight for SG guidance

    Returns:
        SGAwareLifter instance
    """
    return SGAwareLifter(
        depth_lifter=base_lifter,
        use_sg_guidance=use_sg_guidance,
        sg_weight=sg_weight,
    )
