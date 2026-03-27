'''
-----------------------------------------------------------------------------
Optimized PGSR preprocessing for TNT dataset
This version skips slow COLMAP feature extraction/matching and uses existing SfM results
-----------------------------------------------------------------------------
'''

import os
import numpy as np
import json
import sys
from pathlib import Path
from argparse import ArgumentParser
import trimesh

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from convert_data_to_json import export_to_json
from read_write_model import read_model, rotmat2qvec


def load_transformation(trans_file):
    """Load transformation matrix from file"""
    with open(trans_file) as f:
        lines = f.readlines()

    trans = np.eye(4)
    for idx, line in enumerate(lines):
        num = np.array([float(n) for n in line.split(' ')])
        trans[idx, :] = num

    return trans


def align_gt_with_cam(pts, trans):
    """Align ground truth point cloud with camera coordinates"""
    trans_inv = np.linalg.inv(trans)
    pts_aligned = pts @ trans_inv[:3, :3].transpose(-1, -2) + trans_inv[:3, -1]
    return pts_aligned


def compute_bound(pts):
    """Compute bounding sphere from points"""
    bounding_box = np.array([pts.min(axis=0), pts.max(axis=0)])
    center = bounding_box.mean(axis=0)
    radius = np.max(np.linalg.norm(pts - center, axis=-1)) * 1.01
    return center, radius, bounding_box.T.tolist()


def create_sparse_from_existing_sfm(scene_path, scene_name, image_dir):
    """Create sparse folder from existing COLMAP SfM results"""
    sfm_dir = os.path.join(scene_path, 'sparse')
    os.makedirs(sfm_dir, exist_ok=True)

    # Read existing images from database if available, otherwise use image directory
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    # Assume camera parameters from TNT dataset
    w, h = 1920, 1080
    fx, fy = 1152.0, 1152.0
    cx, cy = w / 2.0, h / 2.0

    # Write cameras.txt
    with open(os.path.join(sfm_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, FX, FY, CX, CY, RADIAl_K1, RADIAL_K2\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    # Read poses from the provided COLMAP_SfM.log
    log_file = os.path.join(scene_path, f'{scene_name}_COLMAP_SfM.log')
    poses = load_COLMAP_poses_from_log(log_file, image_dir)

    # Write images.txt
    with open(os.path.join(sfm_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for img_idx, img_name in enumerate(images):
            # Find matching pose
            if img_name in poses:
                pose = poses[img_name]
            else:
                pose = np.eye(4)

            # Extract rotation and translation
            R = pose[:3, :3]
            t = pose[:3, 3]

            # Convert rotation to quaternion (w, x, y, z)
            from scipy.spatial.transform import Rotation
            r = Rotation.from_matrix(R)
            qvec = r.as_quat()  # (x, y, z, w)
            qvec = [qvec[3], qvec[0], qvec[1], qvec[2]]  # (w, x, y, z)

            # Write image line
            f.write(f"{img_idx + 1} {qvec[0]:.6f} {qvec[1]:.6f} {qvec[2]:.6f} {qvec[3]:.6f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {img_name}\n")
            f.write("\n")

    # Create points3D.txt from existing COLMAP ply file
    colmap_ply_path = os.path.join(scene_path, f'{scene_name}_COLMAP.ply')
    if os.path.exists(colmap_ply_path):
        try:
            pcd = trimesh.load(colmap_ply_path)
            points = pcd.vertices
            colors = pcd.colors[:, :3] if pcd.colors.shape[1] >= 3 else np.ones((len(points), 3)) * 128

            with open(os.path.join(sfm_dir, 'points3D.txt'), 'w') as f:
                f.write("# 3D point list with one line per point:\n")
                f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
                for i, (pt, col) in enumerate(zip(points, colors)):
                    f.write(f"{i+1} {pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f} "
                            f"{int(col[0])} {int(col[1])} {int(col[2])} 0.0\n")
            print(f"Created points3D.txt from {colmap_ply_path}")
        except Exception as e:
            print(f"Warning: Failed to load PLY file: {e}")
            # Create empty points3D.txt
            with open(os.path.join(sfm_dir, 'points3D.txt'), 'w') as f:
                f.write("# 3D point list with one line per point:\n")
                f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    else:
        with open(os.path.join(sfm_dir, 'points3D.txt'), 'w') as f:
            f.write("# 3D point list with one line per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        print(f"Warning: {colmap_ply_path} not found")

    print(f"Created sparse folder at {sfm_dir}")
    return sfm_dir


def load_COLMAP_poses_from_log(cam_file, img_dir, tf='w2c'):
    """Load COLMAP poses from SfM log file"""
    names = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])

    with open(cam_file) as f:
        lines = f.readlines()

    poses = {}
    for idx, line in enumerate(lines):
        if idx % 5 == 0:  # header
            img_idx, valid, _ = line.split(' ')
            if valid != '-1':
                poses[int(img_idx)] = np.eye(4)
        else:
            if int(img_idx) in poses:
                num = np.array([float(n) for n in line.split(' ')])
                poses[int(img_idx)][idx % 5 - 1, :] = num

    if tf == 'c2w':
        return poses
    else:
        # convert to W2C (follow nerf convention)
        poses_w2c = {}
        for k, v in poses.items():
            if k < len(names):
                poses_w2c[names[k]] = np.linalg.inv(v)
        return poses_w2c


def init_colmap_fast(args):
    """Fast version that uses existing SfM results"""
    assert args.tnt_path, "Provide path to Tanks and Temples dataset"
    scene_list = os.listdir(args.tnt_path)

    for scene in scene_list:
        scene_path = os.path.join(args.tnt_path, scene)
        if not os.path.exists(f"{scene_path}/images_raw"):
            print(f"Warning: images_raw not found in {scene_path}, skipping...")
            continue

        print(f"\nProcessing scene: {scene}")

        # Create sparse folder from existing SfM
        print("Step 1: Creating sparse folder from existing SfM...")
        create_sparse_from_existing_sfm(scene_path, scene, os.path.join(scene_path, 'images_raw'))

        # Compute bounding info using GT point cloud alignment
        print("Step 2: Computing bounding info...")
        trans_file = os.path.join(scene_path, f'{scene}_trans.txt')
        ply_file = os.path.join(scene_path, f'{scene}.ply')

        if os.path.exists(trans_file) and os.path.exists(ply_file):
            trans = load_transformation(trans_file)
            pts = trimesh.load(ply_file)
            pts = pts.vertices
            pts_aligned = align_gt_with_cam(pts, trans)
            center, radius, bounding_box = compute_bound(pts_aligned[::100])
        else:
            # Fallback: use poses to compute bounding
            print("Warning: GT ply or trans file not found, using pose-based bounding")
            sfm_dir = os.path.join(scene_path, 'sparse')
            cameras, images, points3D = read_model(sfm_dir, ext='.txt')

            # Compute center from poses
            from convert_data_to_json import bound_by_pose
            center, radius, bounding_box = bound_by_pose(images)

        # Generate transforms.json using PGSR method
        print("Step 3: Generating transforms.json...")
        sfm_dir = os.path.join(scene_path, 'sparse')
        cameras, images, points3D = read_model(sfm_dir, ext='.txt')
        export_to_json(cameras, images, bounding_box, list(center), radius,
                      os.path.join(scene_path, 'transforms.json'))

        print(f"{scene} done!")
        print(f"transforms.json created at {scene_path}/transforms.json")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tnt_path', type=str, default=None, help='Path to tanks and temples dataset')

    args = parser.parse_args()

    init_colmap_fast(args)
