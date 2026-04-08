"""Ellipse Projection Lifter.

This lifter implements Method 6: Ellipse Projection Coverage Method.

Instead of using depth estimation to backproject 2D pixels to 3D and find
nearby Gaussians, this method:
1. Projects each 3D Gaussian to a 2D ellipse on the image plane
2. Determines which pixels are covered by each ellipse using Mahalanobis distance
3. Assigns change evidence from 2D change masks to covering Gaussians

This avoids the depth scale mismatch issue present in depth-based methods.
"""

import torch
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class EllipseLiftResult:
    """Result from ellipse projection lifting."""

    positive_mask: torch.Tensor  # (N,) Boolean mask of changed Gaussians
    seed_score: torch.Tensor  # (N,) Score for each Gaussian
    neg_score: torch.Tensor  # (N,) Negative evidence score
    coverage_count: torch.Tensor = None  # (N,) Number of pixels covering each Gaussian


class EllipseProjectionLifter:
    """Lifter using ellipse projection coverage method.

    This method projects 3D Gaussians to 2D ellipses and determines
    pixel-Gaussian coverage using Mahalanobis distance.
    """

    def __init__(
        self,
        k_nn: int = 8,
        coverage_threshold: float = 9.21,  # 3σ threshold for Mahalanobis distance
        min_visible_views: int = 1,
        min_positive_views: int = 1,
        min_seed_views: int = 1,
        min_positive_ratio: float = 0.1,
        final_thresh: float = 0.3,
        max_gaussians_per_batch: int = 512,  # Process Gaussians in batches
        use_opacity_weighting: bool = True,
    ):
        """Initialize the ellipse projection lifter.

        Args:
            k_nn: Number of nearest neighbors (unused, kept for compatibility)
            coverage_threshold: Mahalanobis distance threshold (default 9.21 ≈ 3σ)
            min_visible_views: Minimum number of views seeing a Gaussian
            min_positive_views: Minimum number of views with positive evidence
            min_seed_views: Minimum number of views with seed evidence
            min_positive_ratio: Minimum ratio of positive/total views
            final_thresh: Final threshold for positive mask
            max_gaussians_per_batch: Maximum Gaussians to process per batch
            use_opacity_weighting: Weight by opacity when accumulating scores
        """
        self.k_nn = k_nn
        self.coverage_threshold = coverage_threshold
        self.min_visible_views = min_visible_views
        self.min_positive_views = min_positive_views
        self.min_seed_views = min_seed_views
        self.min_positive_ratio = min_positive_ratio
        self.final_thresh = final_thresh
        self.max_gaussians_per_batch = min(max_gaussians_per_batch, 64)
        self.use_opacity_weighting = use_opacity_weighting

    def project_gaussians_to_2d(
        self,
        positions: torch.Tensor,  # (N, 3) Gaussian means in world coords
        scales: torch.Tensor,  # (N, 3) Gaussian scales
        Twc: torch.Tensor,  # (4, 4) World-to-camera transform
        fx: float, fy: float, cx: float, cy: float,  # Camera intrinsics
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project 3D Gaussians to 2D ellipses.

        Args:
            positions: (N, 3) Gaussian means in world coordinates
            scales: (N, 3) Gaussian scales
            Twc: (4, 4) World-to-camera transform matrix
            fx, fy, cx, cy: Camera intrinsic parameters

        Returns:
            means_2d: (N, 2) 2D ellipse centers (u, v)
            covs_2d: (N, 2, 2) 2D ellipse covariance matrices
        """
        N = positions.shape[0]
        device = positions.device

        # Transform means to camera coordinates
        ones = torch.ones(N, 1, device=device)
        mean_3d_h = torch.cat([positions, ones], dim=-1)  # (N, 4)
        mean_cam = mean_3d_h @ Twc.T  # (N, 4)

        # Get depth
        z = mean_cam[:, 2]
        inv_z = 1.0 / z.clamp(min=1e-6)

        # Project to 2D image plane
        x, y = mean_cam[:, 0], mean_cam[:, 1]
        u = fx * (x * inv_z) + cx
        v = fy * (y * inv_z) + cy

        means_2d = torch.stack([u, v], dim=-1)  # (N, 2)

        # Compute 2D covariance using first-order propagation
        scale_x = (fx * inv_z) ** 2
        scale_y = (fy * inv_z) ** 2

        # Build 2D covariance (diagonal approximation)
        cov_2d_xx = scales[:, 0] ** 2 * scale_x
        cov_2d_yy = scales[:, 1] ** 2 * scale_y
        cov_2d_xy = scales[:, 0] * scales[:, 1] * fx * fy * inv_z ** 2

        # Stack into (N, 2, 2) matrices
        covs_2d = torch.zeros(N, 2, 2, device=device)
        covs_2d[:, 0, 0] = cov_2d_xx
        covs_2d[:, 1, 1] = cov_2d_yy
        covs_2d[:, 0, 1] = cov_2d_xy
        covs_2d[:, 1, 0] = cov_2d_xy

        # Ensure positive definite (add small epsilon to diagonal)
        covs_2d = covs_2d + torch.eye(2, device=device).unsqueeze(0) * 1e-6

        return means_2d, covs_2d

    def lift(
        self,
        gaussians,  # GGSGaussianAdapter
        cameras: List,
        change_masks: List[torch.Tensor],
    ) -> EllipseLiftResult:
        """Lift 2D change masks to 3D using ellipse projection.

        Args:
            gaussians: GGSGaussianAdapter
            cameras: List of cameras
            change_masks: List of (H, W) change masks per camera

        Returns:
            EllipseLiftResult with positive mask and scores
        """
        device = gaussians.get_positions().device
        N = gaussians.num_gaussians

        # Initialize accumulators
        seed_score = torch.zeros(N, device=device)
        neg_score = torch.zeros(N, device=device)
        coverage_count = torch.zeros(N, dtype=torch.int32, device=device)
        visible_views = torch.zeros(N, dtype=torch.int32, device=device)
        positive_views = torch.zeros(N, dtype=torch.int32, device=device)

        positions = gaussians.get_positions()  # (N, 3)
        scales = gaussians.get_scales()  # (N, 3)

        # Get opacities
        if self.use_opacity_weighting:
            opacities = gaussians.get_opacity().squeeze(-1)  # (N,)
        else:
            opacities = torch.ones(N, device=device)

        threshold_sq = self.coverage_threshold ** 2

        for view_id, (cam, mask) in enumerate(zip(cameras, change_masks)):
            H, W = mask.shape

            # Get camera parameters
            fx, fy = cam.Fx, cam.Fy
            cx, cy = cam.Cx, cam.Cy
            Twc = cam.Twc.to(device)

            # Project all Gaussians to 2D ellipses
            means_2d, covs_2d = self.project_gaussians_to_2d(
                positions, scales, Twc, fx, fy, cx, cy
            )

            # Get positive and negative pixel coordinates
            pos_mask = mask > 0.5
            neg_mask = ~pos_mask & torch.isfinite(mask)

            pos_ys, pos_xs = torch.nonzero(pos_mask, as_tuple=True)
            neg_ys, neg_xs = torch.nonzero(neg_mask, as_tuple=True)

            if pos_xs.numel() == 0:
                continue

            # Stack pixel coordinates: [u, v] format (image coordinates)
            pos_pixel_coords = torch.stack([pos_xs.float(), pos_ys.float()], dim=-1)  # (M_pos, 2)
            neg_pixel_coords = torch.stack([neg_xs.float(), neg_ys.float()], dim=-1)  # (M_neg, 2)

            # Process positive pixels using Gaussian-centric approach
            self._accumulate_coverage(
                pos_pixel_coords, means_2d, covs_2d, opacities,
                threshold_sq, seed_score, coverage_count, positive_views, visible_views
            )

            # Process negative pixels
            if neg_xs.numel() > 0:
                self._accumulate_coverage(
                    neg_pixel_coords, means_2d, covs_2d, opacities,
                    threshold_sq, neg_score, None, None, visible_views, is_negative=True
                )

        # Combine evidence
        pos = seed_score
        neg = 0.25 * neg_score
        score = pos / (pos + neg + 1e-8)

        # Multi-view consistency filtering
        keep = (
            (visible_views >= self.min_visible_views)
            & (positive_views >= self.min_positive_views)
            & (seed_score >= self.min_seed_views)
            & (positive_views.float() / (visible_views.float() + 1e-8) >= self.min_positive_ratio)
        )
        score = torch.where(keep, score, torch.zeros_like(score))

        changed_gaussians = score > self.final_thresh

        # If too few changed, use fallback
        if changed_gaussians.sum() < 50:
            changed_gaussians = torch.ones(N, dtype=torch.bool, device=device)

        return EllipseLiftResult(
            positive_mask=changed_gaussians,
            seed_score=score,
            neg_score=neg_score,
            coverage_count=coverage_count,
        )

    def _accumulate_coverage(
        self,
        pixel_coords: torch.Tensor,  # (M, 2) pixel coordinates
        means_2d: torch.Tensor,  # (N, 2) ellipse centers
        covs_2d: torch.Tensor,  # (N, 2, 2) ellipse covariances
        opacities: torch.Tensor,  # (N,) opacities
        threshold_sq: float,
        score_accum: torch.Tensor,  # Score accumulator
        coverage_accum: torch.Tensor,  # Coverage count accumulator (optional)
        positive_views_accum: torch.Tensor,  # Positive views accumulator (optional)
        visible_views_accum: torch.Tensor,  # Visible views accumulator
        is_negative: bool = False,
    ):
        """Accumulate coverage evidence from pixels to Gaussians.

        Uses a Gaussian-centric approach: for each Gaussian, find which pixels
        it covers and accumulate evidence.
        """
        M, N = pixel_coords.shape[0], means_2d.shape[0]
        device = pixel_coords.device

        # Process Gaussians in batches to save memory
        for start in range(0, N, self.max_gaussians_per_batch):
            end = min(start + self.max_gaussians_per_batch, N)

            # Get batch covariances and compute inverse
            batch_covs = covs_2d[start:end]  # (B, 2, 2)
            batch_covs_inv = torch.inverse(batch_covs + torch.eye(2, device=device).unsqueeze(0) * 1e-6)

            # Get batch means and opacities
            batch_means = means_2d[start:end]  # (B, 2)
            batch_opacities = opacities[start:end]  # (B,)

            # For each Gaussian in batch, find pixels it covers
            for g in range(end - start):
                g_idx = start + g
                g_mean = batch_means[g]  # (2,)
                g_cov_inv = batch_covs_inv[g]  # (2, 2)
                g_opacity = batch_opacities[g]

                # Compute Mahalanobis distance from all pixels to this Gaussian
                diff = pixel_coords - g_mean.unsqueeze(0)  # (M, 2)
                diff_cov_inv = diff @ g_cov_inv  # (M, 2)
                dists_sq = (diff * diff_cov_inv).sum(dim=-1)  # (M,)

                # Find covered pixels
                covered_mask = dists_sq < threshold_sq
                num_covered = covered_mask.sum().item()

                if num_covered > 0:
                    weights = torch.exp(-0.5 * dists_sq[covered_mask]) * g_opacity
                    score_accum[g_idx] += weights.sum()

                    if coverage_accum is not None:
                        coverage_accum[g_idx] += 1
                    if positive_views_accum is not None:
                        positive_views_accum[g_idx] += 1
                    visible_views_accum[g_idx] += 1

            # Clear GPU memory between batches
            del batch_covs, batch_covs_inv, batch_means
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def create_ellipse_projection_lifter(**kwargs) -> EllipseProjectionLifter:
    """Create an EllipseProjectionLifter instance."""
    return EllipseProjectionLifter(**kwargs)
