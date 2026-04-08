#!/usr/bin/env python
"""Test ellipse_projection.py algorithm for pixel-to-gaussian mapping."""

import torch
import sys
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting')
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting/submodules/cl-splats')

from scene.gaussian_model import GaussianModel
from scene import Scene
from arguments import ModelParams, OptimizationParams
from argparse import ArgumentParser
import time

# Import the ellipse projection algorithms
from incremental.lifter.ellipse_projection import batch_project_gaussians_to_pixels
from incremental.lifter.ellipse_projection_scatter import batch_project_gaussians_to_pixels_scatter


def load_model(model_path, source_path, images):
    """Load GaussianModel."""
    parser = ArgumentParser(description="Model parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)

    old_argv = sys.argv
    sys.argv = [
        'script',
        '--model_path', model_path,
        '--source_path', source_path,
        '--images', images,
    ]
    try:
        args = parser.parse_args()
    finally:
        sys.argv = old_argv

    model_args = lp.extract(args)
    opt_args = op.extract(args)

    gaussian_model = GaussianModel(sh_degree=3, sg_degree=3)
    scene = Scene(model_args, gaussian_model, load_iteration=-1)
    gaussian_model.training_setup(opt_args)

    return gaussian_model, scene


def build_covariance_from_scales(scales, rotation):
    """Build 3D covariance from scales and rotation.

    Note: scales are stored as log(scales) in Gaussian Splatting.
    """
    N = scales.shape[0]
    device = scales.device

    # Exponential to get actual scales
    scales = torch.exp(scales)

    # Normalize quaternions
    rotation = rotation / rotation.norm(dim=1, keepdim=True)

    # Convert quaternion to rotation matrix
    w, x, y, z = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]

    # Build rotation matrices
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

    # Covariance = R @ diag(scales^2) @ R^T
    S_sq = torch.zeros(N, 3, 3, device=device)
    S_sq[:, 0, 0] = scales[:, 0] ** 2
    S_sq[:, 1, 1] = scales[:, 1] ** 2
    S_sq[:, 2, 2] = scales[:, 2] ** 2

    covs3d = R @ S_sq @ R.transpose(-1, -2)

    return covs3d


def test_single_camera(gaussian_model, cameras, camera_idx=0, max_gaussians=None, version='original'):
    """Test the ellipse projection on a single camera."""
    cam = cameras[camera_idx]
    H, W = cam.image_height, cam.image_width
    print(f"\n=== Testing camera {camera_idx} (H={H}, W={W}) ===")

    # Get Gaussian data
    means = gaussian_model._xyz
    scales = gaussian_model._scaling
    rotation = gaussian_model._rotation

    # Limit number of Gaussians for testing
    if max_gaussians is not None:
        means = means[:max_gaussians]
        scales = scales[:max_gaussians]
        rotation = rotation[:max_gaussians]
        print(f"Using only first {max_gaussians} Gaussians (for testing)")

    N = means.shape[0]
    print(f"Total Gaussians: {N}")

    # Build covariance
    print("Building 3D covariances...")
    t0 = time.time()
    covs3d = build_covariance_from_scales(scales, rotation)
    print(f"  Done in {time.time()-t0:.2f}s, shape: {covs3d.shape}")

    # Build camera intrinsics K
    fx, fy = cam.Fx, cam.Fy
    cx, cy = cam.Cx, cam.Cy
    K = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=means.device)

    # Get world to camera transform
    R_cam = cam.R
    T_cam = cam.T.reshape(3, 1)

    print(f"Camera: Fx={fx:.2f}, Fy={fy:.2f}, Cx={cx:.2f}, Cy={cy:.2f}")
    print(f"Image size: {H}x{W}")

    # Run ellipse projection
    print(f"\nRunning {version} version...")
    t0 = time.time()

    if version == 'scatter':
        pixel_to_gaussian, actual_counts = batch_project_gaussians_to_pixels_scatter(
            means, covs3d, K, R_cam, T_cam, H, W,
            batch_size=4096,
            sigma_scale=3.0,
            max_gaussians_per_pixel=50
        )
    else:
        pixel_to_gaussian, actual_counts = batch_project_gaussians_to_pixels(
            means, covs3d, K, R_cam, T_cam, H, W,
            batch_size=4096,
            sigma_scale=3.0
        )

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.2f}s")

    # Analyze results
    total_mappings = actual_counts.sum().item()
    pixels_with_gaussians = (actual_counts > 0).sum().item()
    max_gaussians_single_pixel = actual_counts.max().item()

    print(f"\n=== Results ===")
    print(f"pixel_to_gaussian shape: {pixel_to_gaussian.shape}")
    print(f"actual_counts shape: {actual_counts.shape}")
    print(f"Total pixel-gaussian mappings: {total_mappings}")
    print(f"Pixels with at least one Gaussian: {pixels_with_gaussians} / {H*W} ({100*pixels_with_gaussians/(H*W):.2f}%)")
    print(f"Max Gaussians for a single pixel: {max_gaussians_single_pixel}")
    if pixels_with_gaussians > 0:
        print(f"Average Gaussians per covered pixel: {total_mappings / pixels_with_gaussians:.2f}")

    # Show sample
    print("\nSample pixel mappings (first few non-empty):")
    count = 0
    for vp in range(min(10, H)):
        for up in range(min(10, W)):
            cnt = actual_counts[vp, up].item()
            if cnt > 0:
                gaussians = pixel_to_gaussian[vp, up, :cnt].tolist()
                print(f"  Pixel ({vp}, {up}): {cnt} Gaussians -> {gaussians}")
                count += 1
                if count >= 5:
                    break
        if count >= 5:
            break

    return pixel_to_gaussian, actual_counts


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="output/cone_pinhole")
    parser.add_argument("--source_path", type=str, default="eval_update/cl-splats-dataset/Real-World/cone_pinhole/t1")
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--camera_idx", type=int, default=0)
    parser.add_argument("--max_gaussians", type=int, default=100000)
    parser.add_argument("--version", type=str, default="original", choices=["original", "scatter"])
    args = parser.parse_args()

    print("Loading model...")
    gaussian_model, scene = load_model(args.model_path, args.source_path, args.images)
    cameras = scene.getTrainCameras(scale=1.0)
    print(f"Loaded {gaussian_model._xyz.shape[0]} Gaussians, {len(cameras)} cameras")

    pixel_to_gaussian, actual_counts = test_single_camera(
        gaussian_model, cameras,
        camera_idx=args.camera_idx,
        max_gaussians=args.max_gaussians,
        version=args.version
    )

    print("\n=== Test completed successfully ===")


if __name__ == "__main__":
    main()
