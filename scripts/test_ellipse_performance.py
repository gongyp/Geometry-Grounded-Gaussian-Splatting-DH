#!/usr/bin/env python
"""Test performance of different ellipse projection implementations."""

import torch
import sys
import importlib.util
import time

sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting')

from scene.gaussian_model import GaussianModel
from scene import Scene
from arguments import ModelParams, OptimizationParams
from argparse import ArgumentParser


def load_model(model_path, source_path, images):
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    sys.argv = ['script', '--model_path', model_path, '--source_path', source_path, '--images', images]
    args = parser.parse_args()
    model_args = lp.extract(args)
    opt_args = op.extract(args)

    gaussian_model = GaussianModel(sh_degree=3, sg_degree=3)
    scene = Scene(model_args, gaussian_model, load_iteration=-1)
    gaussian_model.training_setup(opt_args)
    return gaussian_model, scene


def build_covariance_from_scales(scales, rotation):
    device = scales.device
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


def main():
    print("Loading model...")
    gaussian_model, scene = load_model(
        "/data/Geometry-Grounded-Gaussian-Splatting/output/cone_pinhole",
        "/data/Geometry-Grounded-Gaussian-Splatting/eval_update/cl-splats-dataset/Real-World/cone_pinhole/t1",
        "images"
    )
    cameras = scene.getTrainCameras(scale=1.0)
    print(f"Loaded {gaussian_model._xyz.shape[0]} Gaussians")

    cam = cameras[0]
    H, W = cam.image_height, cam.image_width
    print(f"Camera 0: {H}x{W}")

    # Test with different sizes
    for N in [10000, 50000, 100000, 500000]:
        print(f"\n{'='*50}")
        print(f"Testing with {N} Gaussians")
        print(f"{'='*50}")

        means = gaussian_model._xyz[:N]
        covs3d = build_covariance_from_scales(gaussian_model._scaling[:N], gaussian_model._rotation[:N])

        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]],
                         dtype=torch.float32, device=means.device)
        R_cam = cam.R
        T_cam = cam.T.reshape(3, 1)

        # Import original
        spec = importlib.util.spec_from_file_location(
            "ellipse_projection_cuda",
            "/data/Geometry-Grounded-Gaussian-Splatting/incremental/lifter/ellipse_projection_cuda.py"
        )
        ellipse_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ellipse_module)
        fast_pixel2gaussians_original = ellipse_module.fast_pixel2gaussians

        # Import scatter version
        spec2 = importlib.util.spec_from_file_location(
            "ellipse_projection_scatter_v2",
            "/data/Geometry-Grounded-Gaussian-Splatting/incremental/lifter/ellipse_projection_scatter_v2.py"
        )
        scatter_module = importlib.util.module_from_spec(spec2)
        spec2.loader.exec_module(scatter_module)
        fast_pixel2gaussians_scatter = scatter_module.fast_pixel2gaussians_scatter
        fast_pixel2gaussians_minimal = scatter_module.fast_pixel2gaussians_minimal_cpu

        # Test original
        print(f"\nOriginal version:")
        t0 = time.time()
        result_orig = fast_pixel2gaussians_original(
            means, covs3d, K, R_cam, T_cam, H, W,
            batch_size=4096, sigma_scale=3.0
        )
        t_orig = time.time() - t0
        print(f"  Time: {t_orig:.2f}s")

        # Test scatter version
        print(f"\nScatter version (v2):")
        t0 = time.time()
        result_scatter = fast_pixel2gaussians_scatter(
            means, covs3d, K, R_cam, T_cam, H, W,
            batch_size=4096, sigma_scale=3.0
        )
        t_scatter = time.time() - t0
        print(f"  Time: {t_scatter:.2f}s")

        # Test minimal CPU version
        print(f"\nMinimal CPU version:")
        t0 = time.time()
        result_minimal = fast_pixel2gaussians_minimal(
            means, covs3d, K, R_cam, T_cam, H, W,
            batch_size=4096, sigma_scale=3.0
        )
        t_minimal = time.time() - t0
        print(f"  Time: {t_minimal:.2f}s")

        print(f"\nSpeedup vs Original:")
        print(f"  Scatter: {t_orig/t_scatter:.2f}x")
        print(f"  Minimal CPU: {t_orig/t_minimal:.2f}x")

        # Verify results match
        # Count pixels with gaussians
        def count_mappings(result):
            total = 0
            pixels = 0
            for r in range(H):
                for c in range(W):
                    g = result[r][c]
                    if len(g) > 0:
                        pixels += 1
                        total += len(g)
            return pixels, total

        p1, m1 = count_mappings(result_orig)
        p2, m2 = count_mappings(result_scatter)
        p3, m3 = count_mappings(result_minimal)

        print(f"\nMapping consistency:")
        print(f"  Original: {p1} pixels, {m1} mappings")
        print(f"  Scatter: {p2} pixels, {m2} mappings")
        print(f"  Minimal: {p3} pixels, {m3} mappings")


if __name__ == "__main__":
    main()
