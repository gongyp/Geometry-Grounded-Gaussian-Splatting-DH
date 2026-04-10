"""Pixel-Gaussian Mapping Module.

Computes the mapping relationship between image pixels and 3D Gaussians
using ellipse projection based on Gaussian covariance.

Each pixel in a camera view can be mapped to zero or more Gaussians that
contribute to its color via Gaussian splatting.
"""

import torch
from typing import List, Optional, Union
from dataclasses import dataclass
from scene.gaussian_model import GaussianModel


@dataclass
class PixelGaussianMapping:
    """Mapping result from pixel to Gaussians.

    Attributes:
        pixel_to_gaussians: 2D list [H][W] where pixel_to_gaussians[r][c] is
            a list of gaussian indices that contribute to pixel (r, c).
        gaussian_to_pixels: List of pixel indices for each gaussian. Each entry
            is a list of (pixel_r, pixel_c, mahalanobis_dist) tuples.
        height: Image height in pixels.
        width: Image width in pixels.
        num_gaussians: Number of Gaussians processed.
    """

    pixel_to_gaussians: List[List[List[int]]]
    gaussian_to_pixels: List[List[tuple]]
    height: int
    width: int
    num_gaussians: int


class PixelGaussianMapper:
    """Maps pixels in camera images to Gaussians in the model.

    Uses covariance-based ellipse projection to determine which Gaussians
    affect which pixels. A Gaussian affects a pixel if the pixel lies
    within the Gaussian's 2D projection ellipse (using Mahalanobis distance).

    The algorithm:
    1. Transform 3D Gaussian means to camera coordinate system
    2. Project to 2D image plane with perspective projection
    3. Compute 2D covariance via Jacobian of projection
    4. For each Gaussian, find all pixels within its ellipse (MD <= 1)
    """

    def __init__(
        self,
        gaussian_model: GaussianModel,
        sigma_scale: float = 3.0,
        batch_size: int = 4096,
        device: str = "cuda"
    ):
        """Initialize the mapper.

        Args:
            gaussian_model: The Gaussian model containing 3D Gaussians.
            sigma_scale: Number of sigmas to consider for ellipse radius.
            batch_size: Batch size for processing Gaussians.
            device: Device to use for computation.
        """
        self.gaussian_model = gaussian_model
        self.sigma_scale = sigma_scale
        self.batch_size = batch_size
        self.device = device

    def build_covariance_3d(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Build 3D covariance matrices from scales and rotations.

        Args:
            indices: Optional tensor of Gaussian indices. If None, uses all.

        Returns:
            Covariance matrices [N, 3, 3] in world coordinates.
        """
        if indices is not None:
            scales = self.gaussian_model._scaling[indices]
            rotation = self.gaussian_model._rotation[indices]
            means = self.gaussian_model._xyz[indices]
        else:
            scales = self.gaussian_model._scaling
            rotation = self.gaussian_model._rotation
            means = self.gaussian_model._xyz

        device = means.device
        scales = torch.exp(scales)
        rotation = rotation / rotation.norm(dim=1, keepdim=True)
        w, x, y, z = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]

        N = scales.shape[0]
        R = torch.zeros(N, 3, 3, device=device)
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)

        S_sq = torch.zeros(N, 3, 3, device=device)
        S_sq[:, 0, 0] = scales[:, 0] ** 2
        S_sq[:, 1, 1] = scales[:, 1] ** 2
        S_sq[:, 2, 2] = scales[:, 2] ** 2

        return R @ S_sq @ R.transpose(-1, -2)

    def map_camera(
        self,
        camera,
        gaussian_indices: Optional[torch.Tensor] = None,
    ) -> PixelGaussianMapping:
        """Compute pixel-to-gaussian mapping for a single camera.

        Args:
            camera: Camera object with image dimensions and pose info.
            gaussian_indices: Optional subset of Gaussian indices to consider.
                              If None, uses all Gaussians in the model.

        Returns:
            PixelGaussianMapping containing the mapping relationships.
        """
        H = camera.image_height
        W = camera.image_width

        if gaussian_indices is not None:
            means = self.gaussian_model._xyz[gaussian_indices]
            covs3d = self.build_covariance_3d(gaussian_indices)
            N = len(gaussian_indices)
            gaussian_offset = 0
        else:
            means = self.gaussian_model._xyz
            covs3d = self.build_covariance_3d()
            N = means.shape[0]
            gaussian_indices = torch.arange(N, device=self.device)
            gaussian_offset = 0

        K = torch.tensor(
            [[camera.Fx, 0, camera.Cx], [0, camera.Fy, camera.Cy], [0, 0, 1]],
            dtype=torch.float32,
            device=means.device
        )
        R_cam = camera.R
        T_cam = camera.T.reshape(3, 1)

        pixel_gaus = [[[] for _ in range(W)] for _ in range(H)]
        gaus_pixels: List[List[tuple]] = [[] for _ in range(N)]

        for b_start in range(0, N, self.batch_size):
            b_end = min(b_start + self.batch_size, N)
            B = b_end - b_start

            m = means[b_start:b_end]
            c = covs3d[b_start:b_end]

            with torch.no_grad():
                x_cam = m @ R_cam.T + T_cam.T
                x, y, z = x_cam.unbind(-1)
                valid = z > 0.01
                if not valid.any():
                    continue

                x_cam = x_cam[valid]
                c = c[valid]
                x, y, z = x_cam.unbind(-1)
                B = len(x)

                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]

                u = fx * x / z + cx
                v = fy * y / z + cy

                z2 = z ** 2
                J = torch.zeros(B, 2, 3, device=self.device)
                J[:, 0, 0] = fx / z
                J[:, 0, 2] = -fx * x / z2
                J[:, 1, 1] = fy / z
                J[:, 1, 2] = -fy * y / z2

                cov_cam = R_cam @ c @ R_cam.T
                cov2d = J @ cov_cam @ J.transpose(1, 2)

                vals, vecs = torch.linalg.eigh(cov2d)
                rx = self.sigma_scale * torch.sqrt(vals[:, 1].clamp(1e-8))
                ry = self.sigma_scale * torch.sqrt(vals[:, 0].clamp(1e-8))

                u0 = (u - rx).clamp(0, W - 1).long()
                u1 = (u + rx).clamp(0, W - 1).long()
                v0 = (v - ry).clamp(0, H - 1).long()
                v1 = (v + ry).clamp(0, H - 1).long()

                valid_indices = torch.where(valid)[0]

                for i in range(B):
                    actual_idx = valid_indices[i].item()
                    gaussian_idx = gaussian_offset + b_start + actual_idx

                    vv = torch.arange(v0[i], v1[i] + 1, device=self.device)
                    uu = torch.arange(u0[i], u1[i] + 1, device=self.device)
                    uu, vv = torch.meshgrid(uu, vv, indexing='ij')
                    uu = uu.flatten()
                    vv = vv.flatten()

                    du = uu - u[i]
                    dv = vv - v[i]
                    d = torch.stack([du, dv], dim=-1)

                    try:
                        inv_cov = torch.linalg.inv(cov2d[i])
                    except RuntimeError:
                        continue

                    maha = (d @ inv_cov * d).sum(-1)
                    mask = maha <= 1.0

                    uu = uu[mask]
                    vv = vv[mask]

                    if len(uu) == 0:
                        continue

                    maha_vals = maha[mask].cpu().tolist()
                    for px, py, md in zip(uu.cpu().tolist(), vv.cpu().tolist(), maha_vals):
                        pixel_gaus[py][px].append(gaussian_idx)
                        gaus_pixels[gaussian_idx].append((py, px, md))

        return PixelGaussianMapping(
            pixel_to_gaussians=pixel_gaus,
            gaussian_to_pixels=gaus_pixels,
            height=H,
            width=W,
            num_gaussians=N
        )

    def map_cameras(
        self,
        cameras: List,
        gaussian_indices: Optional[torch.Tensor] = None,
    ) -> List[PixelGaussianMapping]:
        """Compute pixel-to-gaussian mapping for multiple cameras.

        Args:
            cameras: List of camera objects.
            gaussian_indices: Optional subset of Gaussian indices to consider.

        Returns:
            List of PixelGaussianMapping, one per camera.
        """
        return [self.map_camera(cam, gaussian_indices) for cam in cameras]


def create_pixel_gaussian_mapper(
    gaussian_model: GaussianModel,
    sigma_scale: float = 3.0,
    batch_size: int = 4096,
    device: str = "cuda"
) -> PixelGaussianMapper:
    """Factory function to create a PixelGaussianMapper.

    Args:
        gaussian_model: The Gaussian model containing 3D Gaussians.
        sigma_scale: Number of sigmas to consider for ellipse radius.
        batch_size: Batch size for processing Gaussians.
        device: Device to use for computation.

    Returns:
        Configured PixelGaussianMapper instance.
    """
    return PixelGaussianMapper(
        gaussian_model=gaussian_model,
        sigma_scale=sigma_scale,
        batch_size=batch_size,
        device=device
    )
