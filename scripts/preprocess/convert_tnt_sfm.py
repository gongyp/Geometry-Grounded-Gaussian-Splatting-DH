#!/usr/bin/env python3
"""
Convert existing COLMAP SfM results to training format for Geometry-Grounded Gaussian Splatting
"""

import os
import numpy as np
import json
from pathlib import Path
from scipy.spatial.transform import Rotation

def load_colmap_sfm_log(log_file):
    """Load COLMAP SfM log file"""
    with open(log_file, 'r') as f:
        lines = f.readlines()

    poses = {}
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue

        parts = line.split()
        if len(parts) == 3:
            img_idx, cam_idx, valid = int(parts[0]), int(parts[1]), int(parts[2])
            if valid == -1:
                i += 1
                continue

            # Read 4x4 transformation matrix
            pose = np.zeros((4, 4))
            for j in range(1, 5):
                row_vals = [float(x) for x in lines[i + j].strip().split()]
                pose[j - 1, :] = row_vals

            pose[3, 3] = 1.0
            poses[img_idx] = pose
            i += 5
        else:
            i += 1

    return poses

def create_sparse_folder(scene_path, scene_name, image_dir):
    """Create sparse folder with cameras.txt and images.txt"""
    sfm_dir = os.path.join(scene_path, 'sparse')
    os.makedirs(sfm_dir, exist_ok=True)

    log_file = os.path.join(scene_path, f'{scene_name}_COLMAP_SfM.log')
    poses = load_colmap_sfm_log(log_file)

    # Get image list
    images = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))])

    # Assume camera parameters (from original TNT dataset processing)
    # 1920x1080 images with fx=fy=0.6*1920=1152
    w, h = 1920, 1080
    fx, fy = 1152.0, 1152.0
    cx, cy = w / 2.0, h / 2.0

    # Write cameras.txt
    with open(os.path.join(sfm_dir, 'cameras.txt'), 'w') as f:
        f.write("# Camera list with one line per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, FX, FY, CX, CY, RADIAl_K1, RADIAL_K2\n")
        f.write(f"1 PINHOLE {w} {h} {fx} {fy} {cx} {cy}\n")

    # Write images.txt
    with open(os.path.join(sfm_dir, 'images.txt'), 'w') as f:
        f.write("# Image list with two lines per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")

        for img_idx, img_name in enumerate(images):
            # Find matching pose (img_idx should match)
            if img_idx in poses:
                pose = poses[img_idx]
            else:
                # Use identity if not found
                pose = np.eye(4)

            # Extract rotation and translation
            R = pose[:3, :3]
            t = pose[:3, 3]

            # Convert rotation to quaternion (w, x, y, z)
            r = Rotation.from_matrix(R)
            qvec = r.as_quat()  # (x, y, z, w)
            qvec = [qvec[3], qvec[0], qvec[1], qvec[2]]  # (w, x, y, z)

            # Write image line
            f.write(f"{img_idx + 1} {qvec[0]:.6f} {qvec[1]:.6f} {qvec[2]:.6f} {qvec[3]:.6f} "
                    f"{t[0]:.6f} {t[1]:.6f} {t[2]:.6f} 1 {img_name}\n")
            f.write("\n")

    # Create points3D.txt from COLMAP ply file
    colmap_ply_path = os.path.join(scene_path, f'{scene_name}_COLMAP.ply')
    if os.path.exists(colmap_ply_path):
        import trimesh
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
    else:
        # Create empty points3D.txt
        with open(os.path.join(sfm_dir, 'points3D.txt'), 'w') as f:
            f.write("# 3D point list with one line per point:\n")
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        print(f"Warning: {colmap_ply_path} not found, created empty points3D.txt")

    print(f"Created sparse folder at {sfm_dir}")
    return sfm_dir

def create_transforms_json(scene_path, scene_name):
    """Create transforms.json for training"""
    sfm_dir = os.path.join(scene_path, 'sparse')

    # Read images.txt to get camera poses
    images_file = os.path.join(sfm_dir, 'images.txt')
    with open(images_file, 'r') as f:
        lines = f.readlines()

    frames = []
    for i, line in enumerate(lines):
        if line.startswith('#') or not line.strip():
            continue

        parts = line.strip().split()
        if len(parts) < 9:
            continue

        img_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        img_name = parts[8]

        # Convert quaternion to rotation matrix
        qvec = [qx, qy, qz, qw]  # (x, y, z, w)
        r = Rotation.from_quat(qvec)
        R = r.as_matrix()

        # Build camera-to-world transform
        c2w = np.eye(4)
        c2w[:3, :3] = R
        c2w[:3, 3] = [tx, ty, tz]

        # Convert to NeRF format (flip Y and Z axes)
        # c2w[:3, 1:3] *= -1

        transform_matrix = c2w.tolist()

        frame = {
            "file_path": img_name.replace('.jpg', '').replace('.png', ''),
            "transform_matrix": transform_matrix
        }
        frames.append(frame)

    # Sort frames by image name
    frames.sort(key=lambda x: x['file_path'])

    transforms = {
        "camera_angle_x": 0.6911112070083618,  # ~2*atan(1152/1920)
        "frames": frames
    }

    json_path = os.path.join(scene_path, 'transforms.json')
    with open(json_path, 'w') as f:
        json.dump(transforms, f, indent=2)

    print(f"Created transforms.json at {json_path}")
    return json_path

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python convert_tnt_sfm.py <scene_dir>")
        print("Example: python convert_tnt_sfm.py /path/to/GT_TNT_dataset/Barn")
        sys.exit(1)

    scene_path = sys.argv[1]
    scene_name = os.path.basename(scene_path)
    image_dir = os.path.join(scene_path, 'images_raw')

    if not os.path.exists(image_dir):
        print(f"Error: images_raw folder not found at {image_dir}")
        sys.exit(1)

    # Create symlink for images
    images_link = os.path.join(scene_path, 'images')
    if not os.path.exists(images_link):
        os.symlink('images_raw', images_link)
        print(f"Created symlink: images -> images_raw")

    # Create sparse folder
    create_sparse_folder(scene_path, scene_name, image_dir)

    # Create transforms.json
    create_transforms_json(scene_path, scene_name)

    print("\nDone! You can now train with:")
    print(f"python train.py -s {scene_path} -m <output_dir> -r 2 --use_decoupled_appearance 3")
