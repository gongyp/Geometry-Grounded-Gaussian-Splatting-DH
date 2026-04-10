#!/usr/bin/env python
"""Test the PixelGaussianMapping module."""

import torch
import sys
import time
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting')

from scene.gaussian_model import GaussianModel
from scene import Scene
from arguments import ModelParams, OptimizationParams
from argparse import ArgumentParser
import importlib.util
spec = importlib.util.spec_from_file_location(
    "pixel_gaussian_mapping",
    "/data/Geometry-Grounded-Gaussian-Splatting/incremental/lifter/pixel_gaussian_mapping.py"
)
pixel_gaussian_mapping_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pixel_gaussian_mapping_module)

PixelGaussianMapper = pixel_gaussian_mapping_module.PixelGaussianMapper
create_pixel_gaussian_mapper = pixel_gaussian_mapping_module.create_pixel_gaussian_mapper
PixelGaussianMapping = pixel_gaussian_mapping_module.PixelGaussianMapping


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


def main():
    print("Loading model...")
    gaussian_model, scene = load_model(
        "/data/Geometry-Grounded-Gaussian-Splatting/output/cone_pinhole",
        "/data/Geometry-Grounded-Gaussian-Splatting/eval_update/cl-splats-dataset/Real-World/cone_pinhole/t1",
        "images"
    )
    cameras = scene.getTrainCameras(scale=1.0)
    print(f"Loaded {gaussian_model._xyz.shape[0]} Gaussians")
    print(f"Number of cameras: {len(cameras)}")

    cam = cameras[0]
    H, W = cam.image_height, cam.image_width
    print(f"Camera 0: {H}x{W}")

    # Create mapper
    print("\nCreating PixelGaussianMapper...")
    mapper = create_pixel_gaussian_mapper(
        gaussian_model=gaussian_model,
        sigma_scale=3.0,
        batch_size=4096,
        device="cuda"
    )

    # Test with subset of Gaussians
    N = 50000
    gaussian_indices = torch.arange(N, device="cuda")

    print(f"\nComputing mapping for {N} Gaussians...")
    t0 = time.time()
    mapping = mapper.map_camera(cam, gaussian_indices=gaussian_indices)
    elapsed = time.time() - t0
    print(f"Time: {elapsed:.2f}s")

    # Analyze results
    total_mappings = 0
    pixels_covered = 0
    for r in range(H):
        for c in range(W):
            gaus = mapping.pixel_to_gaussians[r][c]
            if len(gaus) > 0:
                pixels_covered += 1
                total_mappings += len(gaus)

    print(f"\nResults:")
    print(f"  Pixels covered: {pixels_covered}/{H*W} ({100*pixels_covered/(H*W):.2f}%)")
    print(f"  Total pixel-gaussian mappings: {total_mappings}")
    print(f"  Average Gaussians per covered pixel: {total_mappings/pixels_covered:.2f}")

    # Show example: find a pixel with multiple Gaussians
    multi_gaus_pixels = []
    for r in range(H):
        for c in range(W):
            gaus = mapping.pixel_to_gaussians[r][c]
            if len(gaus) > 1:
                multi_gaus_pixels.append((r, c, gaus))

    if multi_gaus_pixels:
        r, c, gaus = multi_gaus_pixels[0]
        print(f"\nExample pixel ({r}, {c}) with multiple Gaussians:")
        print(f"  Gaussian indices: {gaus}")
        print(f"  Number of Gaussians: {len(gaus)}")

    # Show example: Gaussian that maps to multiple pixels
    gaus_with_many_pixels = []
    for idx, pixels in enumerate(mapping.gaussian_to_pixels):
        if len(pixels) > 10:
            gaus_with_many_pixels.append((idx, len(pixels), pixels[:5]))

    if gaus_with_many_pixels:
        idx, count, sample = gaus_with_many_pixels[0]
        print(f"\nExample Gaussian {idx} mapping to {count} pixels:")
        print(f"  First 5 pixel locations (r, c, maha_dist): {sample}")


if __name__ == "__main__":
    main()
