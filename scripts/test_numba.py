#!/usr/bin/env python
"""Test Numba-accelerated ellipse projection."""

import torch
import sys
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting')
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting/submodules/cl-splats')

from scene.gaussian_model import GaussianModel
from scene import Scene
from arguments import ModelParams, OptimizationParams
from argparse import ArgumentParser
import time

# Import
from incremental.lifter.ellipse_projection_fast import batch_project_gaussians_to_pixels_fast


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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_gaussians", type=int, default=100000)
    args = parser.parse_args()

    print("Loading model...")
    gaussian_model, scene = load_model(
        "output/cone_pinhole",
        "eval_update/cl-splats-dataset/Real-World/cone_pinhole/t1",
        "images"
    )
    cameras = scene.getTrainCameras(scale=1.0)
    print(f"Loaded {gaussian_model._xyz.shape[0]} Gaussians")

    cam = cameras[0]
    H, W = cam.image_height, cam.image_width
    means = gaussian_model._xyz[:args.max_gaussians]
    covs3d = build_covariance_from_scales(gaussian_model._scaling[:args.max_gaussians], gaussian_model._rotation[:args.max_gaussians])

    K = torch.tensor([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]], dtype=torch.float32, device=means.device)
    R_cam = cam.R
    T_cam = cam.T.reshape(3, 1)

    print(f"\nTesting with {args.max_gaussians} Gaussians, camera 0 ({H}x{W})")

    t0 = time.time()
    pixel_to_gaussian, actual_counts = batch_project_gaussians_to_pixels_fast(
        means, covs3d, K, R_cam, T_cam, H, W,
        batch_size=8192,
        sigma_scale=3.0
    )
    elapsed = time.time() - t0

    total_mappings = actual_counts.sum().item()
    pixels_with = (actual_counts > 0).sum().item()
    print(f"Time: {elapsed:.2f}s")
    print(f"Total mappings: {total_mappings}")
    print(f"Pixels covered: {pixels_with} / {H*W} ({100*pixels_with/(H*W):.2f}%)")


if __name__ == "__main__":
    main()
