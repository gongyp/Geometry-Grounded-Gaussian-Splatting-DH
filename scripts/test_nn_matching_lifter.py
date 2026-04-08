#!/usr/bin/env python
"""Test script to compare original vs NN-matching change detection + lifting.

Usage:
    python scripts/test_nn_matching_lifter.py \
        --model_path output/cone_pinhole \
        --source_path eval_update/cl-splats-dataset/Real-World/cone_pinhole/t1 \
        --images images \
        --output_path output/cone_pinhole/nn_matching_test.ply \
        --new_camera_ids "2"
"""

import argparse
import torch
import logging
import sys
from pathlib import Path
from argparse import ArgumentParser
import numpy as np

sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting')
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting/submodules/cl-splats')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="Test NN-matching change detection + lifting")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--source_path", type=str, required=True)
    parser.add_argument("--images", type=str, default="images")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--new_camera_ids", type=str, default=None)
    parser.add_argument("--use_nn_matching", action="store_true", default=True,
                        help="Use NN-matching detector (default: True)")
    parser.add_argument("--save_images", action="store_true", default=True)
    return parser.parse_args()


def load_gaussian_model(model_path: str, source_path: str, images: str = "images"):
    """Load a pre-trained GaussianModel."""
    from scene.gaussian_model import GaussianModel
    from scene import Scene
    from arguments import ModelParams, OptimizationParams

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


def create_cameras_from_scene(scene):
    """Create camera list from Scene object."""
    cameras = scene.getTrainCameras(scale=1.0)
    return cameras


def save_debug_images(output_dir: Path, cam_id: int, image_name: str,
                      rendered: torch.Tensor, target: torch.Tensor, change_mask: torch.Tensor,
                      prefix: str = ""):
    """Save rendered, GT, and change detection images for debugging."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    cam_dir = output_dir / f"camera_{cam_id}"
    cam_dir.mkdir(exist_ok=True)

    def tensor_to_image(tensor, path):
        img = tensor.detach().cpu().numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        Image.fromarray(img).save(path)
        log.info(f"Saved: {path}")

    base_name = Path(image_name).stem if image_name else f"cam{cam_id}"
    rendered_path = cam_dir / f"{prefix}{base_name}_rendered.png"
    tensor_to_image(rendered, rendered_path)

    target_path = cam_dir / f"{prefix}{base_name}_target.png"
    tensor_to_image(target, target_path)

    mask_np = change_mask.detach().cpu().numpy()
    mask_np = np.clip(mask_np, 0, 1)
    mask_img = (mask_np * 255).astype(np.uint8)
    mask_path = cam_dir / f"{prefix}{base_name}_mask.png"
    Image.fromarray(mask_img).save(mask_path)
    log.info(f"Saved: {mask_path}")


def main():
    args = parse_args()

    log.info("Loading GaussianModel...")
    gaussian_model, scene = load_gaussian_model(
        args.model_path,
        args.source_path,
        args.images,
    )
    log.info(f"Loaded model with {gaussian_model._xyz.shape[0]} Gaussians")

    cameras = create_cameras_from_scene(scene)
    log.info(f"Created {len(cameras)} cameras")

    from incremental.trainer import IncrementalTrainer, IncrementalTrainerConfig
    from dataclasses import dataclass, field

    @dataclass
    class MinimalChangeConfig:
        threshold: float = 0.9
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
        use_nn_matching: bool = False

    clsplats_cfg = MinimalCLSplatsConfig()
    clsplats_cfg.use_nn_matching = args.use_nn_matching
    if args.use_nn_matching:
        clsplats_cfg.change.threshold = 0.7  # Better threshold for NN-matching
        log.info(f"Using NN-Matching detector with threshold={clsplats_cfg.change.threshold}")
    else:
        clsplats_cfg.change.threshold = 0.9
        log.info(f"Using original DinoV2 detector with threshold={clsplats_cfg.change.threshold}")

    cfg = IncrementalTrainerConfig(
        output_dir="./output_nn_matching_test",
        iters_per_timestep=1,
        log_interval=1,
    )

    trainer = IncrementalTrainer(
        gaussian_model=gaussian_model,
        cameras=cameras,
        cfg=cfg,
        clsplats_config=clsplats_cfg,
    )

    log.info(f"Detector type: {type(trainer.detector)}")
    log.info(f"Using NN-matching: {trainer.use_nn_matching}")

    # Parse new_camera_ids
    change_detection_camera_ids = None
    if args.new_camera_ids is not None:
        change_detection_camera_ids = [int(x.strip()) for x in args.new_camera_ids.split(',')]
        log.info(f"Change detection will use only camera IDs: {change_detection_camera_ids}")

    # Prepare target images
    target_images = []
    for cam in cameras:
        img = cam.original_image if hasattr(cam, "original_image") else cam.image
        if hasattr(img, 'convert'):
            img = img.convert('RGB')
            img = np.array(img) / 255.0
            img = torch.from_numpy(img).float()
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = img.permute(2, 0, 1)
        target_images.append(img)

    log.info("=" * 60)
    log.info("Running change detection...")
    log.info("=" * 60)

    N = trainer.adapter.num_gaussians
    trainer.active_mask = torch.ones(N, dtype=torch.bool, device=trainer.device)

    if change_detection_camera_ids is not None:
        cd_cameras = []
        cd_images = []
        for cam, img in zip(cameras, target_images):
            if hasattr(cam, 'colmap_id') and cam.colmap_id in change_detection_camera_ids:
                cd_cameras.append(cam)
                cd_images.append(img)
        log.info(f"Filtered to {len(cd_cameras)} cameras for change detection")
    else:
        cd_cameras = cameras
        cd_images = target_images

    image_output_dir = Path(args.output_path + "_images")
    log.info(f"Debug images will be saved to: {image_output_dir}")

    change_masks = []
    prefix = "nn_" if args.use_nn_matching else "orig_"

    for i, (camera, target) in enumerate(zip(cd_cameras, cd_images)):
        output = trainer.render_camera(camera)
        rendered = output["image"]

        change_mask = trainer.detect_changes(rendered, target)
        change_masks.append(change_mask)
        cam_id = camera.colmap_id if hasattr(camera, 'colmap_id') else i
        log.info(f"Camera {cam_id}: change mask sum = {change_mask.sum()} ({100*change_mask.sum().item()/change_mask.numel():.2f}%)")

        if args.save_images:
            image_name = getattr(camera, 'image_name', None)
            save_debug_images(image_output_dir, cam_id, image_name, rendered, target, change_mask, prefix=prefix)

    log.info("=" * 60)
    log.info("Lifting 2D changes to 3D...")
    log.info("=" * 60)

    trainer.active_mask = trainer.lift_changes_to_3d(change_masks, cd_cameras)

    active_count = trainer.active_mask.sum().item()
    log.info(f"Active Gaussians after change detection: {active_count} / {N} ({100*active_count/N:.2f}%)")

    # Modify model colors and save
    log.info("Modifying colors for active regions...")

    original_f_dc = gaussian_model._features_dc.clone()
    original_opacity = gaussian_model._opacity.clone()

    active_indices = trainer.active_mask.cpu().numpy()
    device = gaussian_model._features_dc.device

    with torch.no_grad():
        gaussian_model._features_dc.data[active_indices, :, :] = torch.tensor([[[1.0, 0.0, 0.0]]], device=device)
        gaussian_model._opacity.data[active_indices] = 0.5

    native_out = args.output_path + ".native.ply"
    log.info(f"Saving visualization PLY to {native_out}")
    gaussian_model.save_ply(native_out)

    gaussian_model._features_dc = original_f_dc
    gaussian_model._opacity = original_opacity

    log.info(f"Saved visualization with {active_count} active (red) Gaussians")
    log.info(f"Convert with: python scripts/SuperSplat_conversion.py --in_ply {native_out} --out_ply {args.output_path}")


if __name__ == "__main__":
    main()
