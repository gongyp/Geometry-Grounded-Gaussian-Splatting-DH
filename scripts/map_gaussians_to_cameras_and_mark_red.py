#!/usr/bin/env python
"""Map Gaussians to cameras using rasterizer visibility and mark visible ones red.

Uses the rasterizer's visibility mask (radii > 0) to determine which Gaussians
are visible in each camera view, then marks all visible Gaussians as red.
"""

import torch
import sys
import time
import math
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting')

from scene.gaussian_model import GaussianModel
from scene import Scene
from arguments import ModelParams, OptimizationParams
from argparse import ArgumentParser
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def get_rasterizer_visible(cameras, gaussian_model, kernel_size=1.0):
    """Use rasterizer to get visibility mask for each camera.

    Returns:
        camera_visibilities: List of boolean tensors, one per camera
        combined_any: Boolean tensor - visible in ANY camera (OR)
        combined_all: Boolean tensor - visible in ALL cameras (AND)
    """
    camera_visibilities = []
    bg = torch.zeros(3, device="cuda")

    for cam_idx, cam in enumerate(cameras):
        tanfovx = math.tan(cam.FoVx * 0.5)
        tanfovy = math.tan(cam.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(cam.image_height),
            image_width=int(cam.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            kernel_size=kernel_size,
            bg=bg,
            scale_modifier=1.0,
            viewmatrix=cam.world_view_transform,
            projmatrix=cam.full_proj_transform,
            sh_degree=gaussian_model.active_sh_degree,
            sg_degree=gaussian_model.active_sg_degree,
            campos=cam.camera_center,
            prefiltered=False,
            require_depth=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        screenspace_points = torch.zeros_like(gaussian_model.get_xyz, device="cuda")
        means3D = gaussian_model.get_xyz
        means2D = screenspace_points
        scales, opacity = gaussian_model.get_scaling_n_opacity_with_3D_filter
        rotations = gaussian_model.get_rotation
        shs = gaussian_model.get_features
        sg_axis = gaussian_model.get_sg_axis
        sg_sharpness = gaussian_model.get_sg_sharpness
        sg_color = gaussian_model.get_sg_color

        with torch.no_grad():
            _, radii, _, _, _ = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                sg_axis=sg_axis,
                sg_sharpness=sg_sharpness,
                sg_color=sg_color,
                colors_precomp=None,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3Ds_precomp=None,
            )

        visibility = radii > 0
        camera_visibilities.append(visibility)

        if cam_idx % 20 == 0:
            print(f"  Processed camera {cam_idx}/{len(cameras)}: {visibility.sum().item()} visible")

    # Combine
    combined_any = torch.zeros_like(camera_visibilities[0])
    combined_all = torch.ones_like(camera_visibilities[0])
    for vis in camera_visibilities:
        combined_any |= vis
        combined_all &= vis

    return camera_visibilities, combined_any, combined_all


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


def export_ply_with_colors(xyz, colors, output_path):
    """Export points as PLY file with RGB colors."""
    xyz_np = xyz.detach().cpu().numpy()
    colors_np = colors.detach().cpu().numpy()

    # Ensure colors are uint8 and correct shape
    if colors_np.ndim == 2:
        colors_np = colors_np.astype(np.uint8)
    else:
        colors_np = np.stack([colors_np, colors_np, colors_np], axis=1).astype(np.uint8)

    N = xyz_np.shape[0]
    vertices = np.empty(N, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                 ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertices['x'] = xyz_np[:, 0]
    vertices['y'] = xyz_np[:, 1]
    vertices['z'] = xyz_np[:, 2]
    vertices['red'] = colors_np[:, 0]
    vertices['green'] = colors_np[:, 1]
    vertices['blue'] = colors_np[:, 2]

    el = PlyElement.describe(vertices, 'vertex')
    PlyData([el]).write(output_path)
    print(f"Exported PLY to {output_path}")


def main():
    print("Loading model...")
    gaussian_model, scene = load_model(
        "/data/Geometry-Grounded-Gaussian-Splatting/output/cone_pinhole",
        "/data/Geometry-Grounded-Gaussian-Splatting/eval_update/cl-splats-dataset/Real-World/cone_pinhole/t1",
        "images"
    )
    cameras = scene.getTrainCameras(scale=1.0)
    N_total = gaussian_model._xyz.shape[0]
    print(f"Total Gaussians: {N_total}")
    print(f"Number of cameras: {len(cameras)}")

    # Step 1: Get visibility from rasterizer for all cameras
    print(f"\nStep 1: Computing rasterizer visibility for all cameras...")
    t0 = time.time()
    camera_vis, vis_any, vis_all = get_rasterizer_visible(cameras, gaussian_model)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Visible in at least one camera: {vis_any.sum().item()}/{N_total}")
    print(f"  Visible in all cameras: {vis_all.sum().item()}/{N_total}")

    # Step 2: Create colors
    xyz = gaussian_model._xyz
    original_colors = gaussian_model.get_features

    # Handle SH features - take DC component (first entry) for RGB
    if original_colors.ndim == 3:
        # Shape is [N, num_SH_coeffs, 3] - take DC component
        base_colors = (original_colors[:, 0, :] * 255).clamp(0, 255).to(torch.uint8)
    elif original_colors.shape[1] >= 3:
        base_colors = (original_colors[:, :3] * 255).clamp(0, 255).to(torch.uint8)
    else:
        base_colors = torch.ones(N_total, 3, device="cuda", dtype=torch.uint8) * 200

    red_color = torch.tensor([255, 0, 0], device="cuda", dtype=torch.uint8)
    colors = base_colors.clone()
    colors[vis_any] = red_color

    # Step 3: Export PLY
    output_path = "/data/Geometry-Grounded-Gaussian-Splatting/output/visible_gaussians_red.ply"
    print(f"\nStep 2: Exporting PLY...")
    export_ply_with_colors(xyz, colors, output_path)

    print(f"\nSummary:")
    print(f"  Total Gaussians: {N_total}")
    print(f"  Visible (marked RED): {vis_any.sum().item()}")
    print(f"  Not visible (gray): {N_total - vis_any.sum().item()}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
