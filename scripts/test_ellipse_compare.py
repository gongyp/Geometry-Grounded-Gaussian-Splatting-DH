#!/usr/bin/env python
"""Compare performance of ellipse projection implementations."""

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


def count_mappings(result, H, W):
    total = 0
    pixels = 0
    for r in range(H):
        for c in range(W):
            g = result[r][c]
            if len(g) > 0:
                pixels += 1
                total += len(g)
    return pixels, total


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

    # Import all versions
    versions = {}

    # Original
    spec = importlib.util.spec_from_file_location(
        "ellipse_projection_cuda",
        "/data/Geometry-Grounded-Gaussian-Splatting/incremental/lifter/ellipse_projection_cuda.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    versions['original'] = m.fast_pixel2gaussians

    # Scatter v2
    spec = importlib.util.spec_from_file_location(
        "scatter_v2",
        "/data/Geometry-Grounded-Gaussian-Splatting/incremental/lifter/ellipse_projection_scatter_v2.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    versions['scatter_v2'] = m.fast_pixel2gaussians_scatter

    # Fixed v3
    spec = importlib.util.spec_from_file_location(
        "fixed_v3",
        "/data/Geometry-Grounded-Gaussian-Splatting/incremental/lifter/ellipse_projection_fast_v3.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    versions['fixed_v3'] = m.fast_pixel2gaussians_fixed
    versions['sqrt_approx'] = m.fast_pixel2gaussians_sqrt_approx

    # Batch
    spec = importlib.util.spec_from_file_location(
        "batch",
        "/data/Geometry-Grounded-Gaussian-Splatting/incremental/lifter/ellipse_projection_batch.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    versions['batch'] = m.fast_pixel2gaussians_batch
    versions['scatter_full'] = m.fast_pixel2gaussians_scatter_full

    # Test sizes
    test_sizes = [10000, 50000]

    for N in test_sizes:
        print(f"\n{'='*60}")
        print(f"Testing with {N} Gaussians")
        print(f"{'='*60}")

        means = gaussian_model._xyz[:N]
        covs3d = build_covariance_from_scales(gaussian_model._scaling[:N], gaussian_model._rotation[:N])

        K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]],
                         dtype=torch.float32, device=means.device)
        R_cam = cam.R
        T_cam = cam.T.reshape(3, 1)

        baseline_time = None

        for name, func in versions.items():
            try:
                t0 = time.time()
                result = func(means, covs3d, K, R_cam, T_cam, H, W,
                            batch_size=4096, sigma_scale=3.0)
                elapsed = time.time() - t0

                p, m_count = count_mappings(result, H, W)

                if baseline_time is None:
                    baseline_time = elapsed
                    speedup = 1.0
                else:
                    speedup = baseline_time / elapsed

                print(f"{name:15s}: {elapsed:6.2f}s (speedup: {speedup:.2f}x) | {p} pixels, {m_count} mappings")

            except Exception as e:
                print(f"{name:15s}: ERROR - {str(e)[:50]}")

        print()


if __name__ == "__main__":
    main()
