#!/usr/bin/env python
"""Run incremental Gaussian Splatting update.

This script demonstrates how to use the incremental update pipeline
with a pre-trained GaussianModel.
"""

import argparse
import torch
import logging
import os
import sys
from pathlib import Path
from argparse import ArgumentParser

# Add project root and cl-splats submodule to path
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting')
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting/submodules/cl-splats')

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Run incremental Gaussian Splatting update")

    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to pre-trained GaussianModel (iteration folder)")
    parser.add_argument("--source_path", type=str, required=True,
                        help="Path to source data (COLMAP or similar)")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Path to model config YAML")
    parser.add_argument("--images", type=str, default="images",
                        help="Image folder name (e.g., 'images_add' for new data)")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./output_incremental",
                        help="Output directory")
    parser.add_argument("--iters", type=int, default=500,
                        help="Number of iterations per timestep")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--log_interval", type=int, default=10,
                        help="Logging interval")

    # Change detection
    parser.add_argument("--detect_changes", type=bool, default=True,
                        help="Enable change detection")
    parser.add_argument("--change_threshold", type=float, default=0.8,
                        help="Change detection threshold")

    # Densification
    parser.add_argument("--densify", action="store_true", default=False,
                        help="Enable densification")
    parser.add_argument("--densify_grad_thresh", type=float, default=0.0002,
                        help="Gradient threshold for densification")

    # SG features
    parser.add_argument("--use_sg", action="store_true", default=True,
                        help="Use SG features")
    parser.add_argument("--use_sg_guidance", action="store_true", default=True,
                        help="Use SG-based lifting guidance")

    # Other
    parser.add_argument("--eval", action="store_true",
                        help="Run in evaluation mode")
    parser.add_argument("--save_checkpoints", action="store_true",
                        help="Save checkpoints during training")

    return parser.parse_args()


def load_gaussian_model(model_path: str, source_path: str, images: str = "images"):
    """Load a pre-trained GaussianModel.

    Args:
        model_path: Path to model iteration folder (contains point_cloud/)
        source_path: Path to source data
        images: Image folder name (default: 'images')

    Returns:
        GaussianModel instance and Scene
    """
    from scene.gaussian_model import GaussianModel
    from scene import Scene
    from arguments import ModelParams, OptimizationParams

    # Create parser and get model params
    parser = ArgumentParser(description="Model parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)

    # Set arguments via sys.argv style
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

    # Create GaussianModel
    gaussian_model = GaussianModel(
        sh_degree=3,
        sg_degree=3,
    )

    # Create Scene and load checkpoint
    scene = Scene(model_args, gaussian_model, load_iteration=-1)
    gaussian_model.training_setup(opt_args)

    return gaussian_model, scene


def create_cameras_from_scene(scene):
    """Create camera list from Scene object.

    Args:
        scene: Scene object

    Returns:
        List of Camera objects
    """
    cameras = scene.getTrainCameras(scale=1.0)
    return cameras


def main():
    """Main entry point."""
    args = parse_args()

    log.info("Loading GaussianModel...")
    gaussian_model, scene = load_gaussian_model(
        args.model_path,
        args.source_path,
        args.images,
    )

    log.info(f"Loaded model with {gaussian_model._xyz.shape[0]} Gaussians")

    # Create cameras
    log.info("Creating cameras...")
    cameras = create_cameras_from_scene(scene)
    log.info(f"Created {len(cameras)} cameras")

    # Create trainer config
    from incremental.trainer import IncrementalTrainerConfig

    cfg = IncrementalTrainerConfig(
        output_dir=args.output_dir,
        iters_per_timestep=args.iters,
        lr=args.lr,
        log_interval=args.log_interval,
        densify_enabled=args.densify,
        densify_grad_threshold=args.densify_grad_thresh,
        use_sg_features=args.use_sg,
        use_sg_guidance=args.use_sg_guidance,
    )

    # Create trainer
    from incremental.trainer import IncrementalTrainer

    # Create minimal clsplats config for change detection
    from dataclasses import dataclass, field

    @dataclass
    class MinimalChangeConfig:
        threshold: float = 0.8
        dilate_mask: bool = False
        dilate_kernel_size: int = 31
        upsample: bool = True

    @dataclass
    class MinimalLifterConfig:
        depth_model: str = "depth-anything/Depth-Anything-V2-Small-hf"
        k_nn: int = 8
        local_radius_thresh: float = 2.5
        depth_tol_abs: float = 0.05
        depth_tol_rel: float = 0.05
        lambda_seed: float = 2.0
        lambda_neg: float = 0.25
        min_visible_views: int = 2
        min_positive_views: int = 2
        min_seed_views: int = 1
        min_positive_ratio: float = 0.3
        final_thresh: float = 0.6

    @dataclass
    class MinimalCLSplatsConfig:
        change: MinimalChangeConfig = field(default_factory=MinimalChangeConfig)
        lifter: MinimalLifterConfig = field(default_factory=MinimalLifterConfig)

    clsplats_cfg = MinimalCLSplatsConfig()

    trainer = IncrementalTrainer(
        gaussian_model=gaussian_model,
        cameras=cameras,
        cfg=cfg,
        clsplats_config=clsplats_cfg,
    )

    # Prepare target images
    target_images = []
    for cam in cameras:
        img = cam.original_image if hasattr(cam, "original_image") else cam.image
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
            import numpy as np
            img = np.array(img) / 255.0
            # Image is now (H, W, C)
            img = torch.from_numpy(img).float()
            # Convert to (C, H, W)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img.permute(2, 0, 1)
        target_images.append(img)

    # Run incremental update
    log.info("Starting incremental update...")
    results = trainer.run_incremental_update(
        new_cameras=cameras,
        new_images=target_images,
        detect_changes=args.detect_changes,
    )

    log.info(f"Training complete!")
    log.info(f"Final Gaussians: {results['num_gaussians']}")
    log.info(f"Active Gaussians: {results['num_active']}")

    # Save final model
    output_path = Path(args.output_dir) / "final_model.ply"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    log.info(f"Saving model to {output_path}")
    gaussian_model.save_ply(str(output_path))

    log.info("Done!")


if __name__ == "__main__":
    main()
