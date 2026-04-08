#!/usr/bin/env python
"""Test script to compare original DinoV2 detector with NN-matching detector.

Usage:
    python scripts/test_nn_matching_detector.py
"""

import argparse
import torch
import logging
import sys
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from PIL import Image

sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting')
sys.path.insert(0, '/data/Geometry-Grounded-Gaussian-Splatting/submodules/cl-splats')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = ArgumentParser(description="Test NN-matching change detector")
    parser.add_argument("--image_dir", type=str,
                        default="output/cone_pinhole/active_mask_debug.ply_images/camera_2",
                        help="Directory with debug images")
    parser.add_argument("--image_name", type=str, default="day_1_0066",
                        help="Base image name (without suffix)")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="Similarity threshold for NN-matching detector")
    return parser.parse_args()


def load_images(image_dir, image_name):
    """Load rendered and target images."""
    rendered_path = Path(image_dir) / f"{image_name}_rendered.png"
    target_path = Path(image_dir) / f"{image_name}_target.png"

    rendered = np.array(Image.open(rendered_path)).astype(np.float32) / 255.0
    target = np.array(Image.open(target_path)).astype(np.float32) / 255.0

    return rendered, target


def test_original_detector(rendered, target, threshold=0.9):
    """Test original DinoV2 detector."""
    from clsplats.change_detection.dinov2_detector import DinoV2Detector
    from dataclasses import dataclass

    @dataclass
    class Config:
        threshold: float = 0.9
        dilate_mask: bool = False
        dilate_kernel_size: int = 31
        upsample: bool = True

    detector = DinoV2Detector(Config())
    device = detector.device

    rendered_t = torch.from_numpy(rendered).to(device)
    target_t = torch.from_numpy(target).to(device)

    with torch.no_grad():
        mask = detector.predict_change_mask(rendered_t, target_t)

    return mask.cpu().numpy()


def test_nn_matching_detector(rendered, target, threshold=0.7, use_gpu=True):
    """Test NN-matching detector."""
    from incremental.change_detection.nn_matching_detector import (
        NNMatchingDetector, NNMatchingChangeConfig
    )

    cfg = NNMatchingChangeConfig(
        threshold=threshold,
        dilate_mask=True,
        dilate_kernel_size=5,
        upsample=True,
        use_faiss_gpu=use_gpu,
        k_nn=1,
        metric="cosine",
    )
    detector = NNMatchingDetector(cfg)
    device = detector.device

    rendered_t = torch.from_numpy(rendered).to(device)
    target_t = torch.from_numpy(target).to(device)

    with torch.no_grad():
        mask = detector.predict_change_mask(rendered_t, target_t)

    return mask.cpu().numpy()


def compute_metrics(mask, name):
    """Compute and print metrics for a mask."""
    changed = mask.sum()
    total = mask.size
    ratio = changed / total * 100
    print(f"{name}:")
    print(f"  Changed pixels: {changed}/{total} ({ratio:.2f}%)")
    return changed, ratio


def main():
    args = parse_args()

    log.info(f"Loading images from {args.image_dir}")
    rendered, target = load_images(args.image_dir, args.image_name)
    log.info(f"Image shape: {rendered.shape}")

    print("\n" + "="*60)
    print("Testing Original DinoV2 Detector (threshold=0.9)")
    print("="*60)

    mask_original = test_original_detector(rendered, target, threshold=0.9)
    compute_metrics(mask_original, "Original DinoV2")

    # Save original mask
    orig_mask_img = (mask_original * 255).astype(np.uint8)
    Image.fromarray(orig_mask_img).save(
        f"{args.image_dir}/test_original_mask.png"
    )
    log.info(f"Saved original mask to {args.image_dir}/test_original_mask.png")

    print("\n" + "="*60)
    print(f"Testing NN-Matching Detector (threshold={args.threshold})")
    print("="*60)

    mask_nn = test_nn_matching_detector(
        rendered, target, threshold=args.threshold, use_gpu=True
    )
    compute_metrics(mask_nn, "NN-Matching")

    # Save NN-matching mask
    nn_mask_img = (mask_nn * 255).astype(np.uint8)
    Image.fromarray(nn_mask_img).save(
        f"{args.image_dir}/test_nn_matching_mask.png"
    )
    log.info(f"Saved NN-matching mask to {args.image_dir}/test_nn_matching_mask.png")

    print("\n" + "="*60)
    print("Comparison")
    print("="*60)

    # Compute overlap
    overlap = (mask_original & mask_nn).sum()
    union = (mask_original | mask_nn).sum()
    iou = overlap / union * 100 if union > 0 else 0

    print(f"Overlap (both detected as changed): {overlap}")
    print(f"Union: {union}")
    print(f"IoU: {iou:.2f}%")

    # Show per-threshold analysis for NN-matching
    print("\n" + "="*60)
    print("NN-Matching threshold sensitivity analysis")
    print("="*60)

    for thresh in [0.5, 0.6, 0.7, 0.8, 0.9]:
        from incremental.change_detection.nn_matching_detector import (
            NNMatchingDetector, NNMatchingChangeConfig
        )
        cfg = NNMatchingChangeConfig(
            threshold=thresh,
            dilate_mask=True,
            dilate_kernel_size=5,
            upsample=True,
            use_faiss_gpu=True,
            k_nn=1,
            metric="cosine",
        )
        detector = NNMatchingDetector(cfg)
        device = detector.device
        rendered_t = torch.from_numpy(rendered).to(device)
        target_t = torch.from_numpy(target).to(device)

        with torch.no_grad():
            mask = detector.predict_change_mask(rendered_t, target_t)
        mask_np = mask.cpu().numpy()

        changed = mask_np.sum()
        ratio = changed / mask_np.size * 100
        print(f"  threshold={thresh}: changed={changed}/{mask_np.size} ({ratio:.2f}%)")


if __name__ == "__main__":
    main()
