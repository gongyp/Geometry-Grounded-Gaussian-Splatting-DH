#!/usr/bin/env python3
"""Compare PSNR between base model and incremental model."""

import os
import sys
import torch
import torch.nn.functional as F
from pathlib import Path
from argparse import ArgumentParser

# Add project root to path
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting')

from scene.gaussian_model import GaussianModel
from scene import Scene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, get_combined_args
from utils.general_utils import safe_state


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(1.0 / mse.clamp(min=1e-10))


def evaluate_model(model_path, iteration=-1, dataset_name="test"):
    """Evaluate a model and return average PSNR."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_path} (iteration {iteration})")
    print(f"{'='*60}")

    # Set up model params
    os.chdir('/data/Geometry-Grounded-Gaussian-Splatting')

    # Create parser
    parser = ArgumentParser(description="Model parameters")
    lp = ModelParams(parser)
    op = PipelineParams(parser)

    old_argv = sys.argv
    sys.argv = [
        'script',
        '--model_path', model_path,
        '--source_path', 'eval_tnt/GT_TNT_dataset/Meetingroom-Localupdate',
    ]

    try:
        args = parser.parse_args()
    finally:
        sys.argv = old_argv

    model_args = lp.extract(args)

    # Load model
    gaussians = GaussianModel(sh_degree=3, sg_degree=3)
    scene = Scene(model_args, gaussians, load_iteration=iteration, shuffle=False)

    # Setup
    bg_color = [1, 1, 1] if model_args.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    class SimplePipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    pipe = SimplePipe()

    # Get cameras
    if dataset_name == "test":
        cameras = scene.getTestCameras()
    else:
        cameras = scene.getTrainCameras()

    print(f"Number of {dataset_name} cameras: {len(cameras)}")

    psnr_values = []
    with torch.no_grad():
        for i, camera in enumerate(cameras):
            try:
                result = render(
                    viewpoint_camera=camera,
                    pc=gaussians,
                    pipe=pipe,
                    bg_color=background,
                    kernel_size=1,
                    scaling_modifier=1.0,
                    require_depth=False,
                )

                rendered = result["render"]
                gt = camera.original_image[0:3, :, :]

                # Resize if needed
                if rendered.shape != gt.shape:
                    rendered = F.interpolate(
                        rendered.unsqueeze(0),
                        size=(gt.shape[1], gt.shape[2]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                psnr = compute_psnr(rendered, gt)
                psnr_values.append(psnr.item())

                if i < 5:  # Print first few
                    print(f"  Camera {i}: PSNR = {psnr.item():.2f}")

            except Exception as e:
                print(f"  Camera {i}: Error - {e}")

    avg_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    print(f"\nAverage PSNR ({dataset_name}): {avg_psnr:.4f}")
    print(f"Number of images: {len(psnr_values)}")

    return avg_psnr, len(psnr_values)


def main():
    parser = ArgumentParser(description="Compare PSNR between base and incremental models")
    parser.add_argument("--base_model", type=str,
                        default="output/Meetingroom-Localupdate",
                        help="Base model path")
    parser.add_argument("--incremental_model", type=str,
                        default="output_incremental/final_model.ply",
                        help="Incremental model path (point cloud file)")
    parser.add_argument("--dataset", type=str, default="test",
                        choices=["train", "test"],
                        help="Dataset to evaluate on")
    args = parser.parse_args()

    print("=" * 60)
    print("PSNR Comparison: Base Model vs Incremental Model")
    print("=" * 60)

    # Evaluate base model
    base_psnr, base_count = evaluate_model(args.base_model, iteration=-1, dataset_name=args.dataset)

    # Evaluate incremental model (it's a .ply file, not a Scene directory)
    print(f"\n{'='*60}")
    print(f"Evaluating incremental model from: {args.incremental_model}")
    print(f"{'='*60}")

    # Load incremental model directly
    os.chdir('/data/Geometry-Grounded-Gaussian-Splatting')

    # Create parser
    parser2 = ArgumentParser(description="Model parameters")
    lp2 = ModelParams(parser2)

    old_argv = sys.argv
    sys.argv = [
        'script',
        '--model_path', args.base_model,  # Use same source path
        '--source_path', 'eval_tnt/GT_TNT_dataset/Meetingroom-Localupdate',
    ]

    try:
        args2 = parser2.parse_args()
    finally:
        sys.argv = old_argv

    model_args2 = lp2.extract(args2)

    # Load base scene to get cameras (for original images)
    gaussians_base = GaussianModel(sh_degree=3, sg_degree=3)
    scene_base = Scene(model_args2, gaussians_base, load_iteration=-1, shuffle=False)

    # Now load the incremental model
    print("Loading incremental model...")
    gaussians_base.load_ply(args.incremental_model)

    # Setup
    bg_color = [1, 1, 1] if model_args2.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    class SimplePipe:
        convert_SHs_python = False
        compute_cov3D_python = False
        debug = False

    pipe = SimplePipe()

    # Get cameras
    if args.dataset == "test":
        cameras = scene_base.getTestCameras()
    else:
        cameras = scene_base.getTrainCameras()

    print(f"Number of {args.dataset} cameras: {len(cameras)}")

    psnr_values = []
    with torch.no_grad():
        for i, camera in enumerate(cameras):
            try:
                result = render(
                    viewpoint_camera=camera,
                    pc=gaussians_base,
                    pipe=pipe,
                    bg_color=background,
                    kernel_size=1,
                    scaling_modifier=1.0,
                    require_depth=False,
                )

                rendered = result["render"]
                gt = camera.original_image[0:3, :, :]

                # Resize if needed
                if rendered.shape != gt.shape:
                    rendered = F.interpolate(
                        rendered.unsqueeze(0),
                        size=(gt.shape[1], gt.shape[2]),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)

                psnr = compute_psnr(rendered, gt)
                psnr_values.append(psnr.item())

                if i < 5:
                    print(f"  Camera {i}: PSNR = {psnr.item():.2f}")

            except Exception as e:
                print(f"  Camera {i}: Error - {e}")

    inc_psnr = sum(psnr_values) / len(psnr_values) if psnr_values else 0
    print(f"\nAverage PSNR ({args.dataset}): {inc_psnr:.4f}")
    print(f"Number of images: {len(psnr_values)}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Base Model PSNR:       {base_psnr:.4f}")
    print(f"Incremental Model PSNR: {inc_psnr:.4f}")
    print(f"Difference:             {inc_psnr - base_psnr:+.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
