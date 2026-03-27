#!/usr/bin/env python3
"""
Run COLMAP sparse reconstruction on images_add folder using automatic_reconstructor.
These images are from the same camera as the original images folder.
"""

import os
import sys
import subprocess
from pathlib import Path

# Paths
SCENE_PATH = "/data/Geometry-Grounded-Gaussian-Splatting/eval_tnt/GT_TNT_dataset/Meetingroom-Localupdate"
IMAGES_ADD_PATH = os.path.join(SCENE_PATH, "images_add")
WORKSPACE_PATH = os.path.join(SCENE_PATH, "workspace_images_add")


def run_command(cmd, description):
    """Run a shell command and print output"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"WARNING: Command failed with return code {result.returncode}")
        return False
    return True


def main():
    print("=" * 60)
    print("COLMAP Sparse Reconstruction for images_add")
    print("=" * 60)

    # Create workspace
    os.makedirs(WORKSPACE_PATH, exist_ok=True)

    # Use automatic_reconstructor with same camera parameters
    # Since images are from same camera, we use existing camera model
    cmd = [
        "colmap", "automatic_reconstructor",
        "--workspace_path", WORKSPACE_PATH,
        "--image_path", IMAGES_ADD_PATH,
        "--camera_model", "RADIAL",
        "--single_camera", "1",
        "--sparse", "1",
        "--灌",  # Skip dense reconstruction
    ]

    # Try with explicit parameters
    cmd = [
        "colmap", "automatic_reconstructor",
        "--workspace_path", WORKSPACE_PATH,
        "--image_path", IMAGES_ADD_PATH,
    ]

    print(f"\nRunning COLMAP automatic_reconstructor...")
    print(f"Workspace: {WORKSPACE_PATH}")
    print(f"Images: {IMAGES_ADD_PATH}")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        print(f"\nautomatic_reconstructor failed. Trying manual approach...")

        # Manual approach: create database, extract features, match, map
        database_path = os.path.join(WORKSPACE_PATH, "database.db")
        os.makedirs(database_path, exist_ok=False)

        # Step 1: Create database
        cmd = [
            "colmap", "database_creator",
            "--database_path", database_path,
        ]
        run_command(cmd, "Create Database")

        # Step 2: Feature extraction
        cmd = [
            "colmap", "feature_extractor",
            "--database_path", database_path,
            "--image_path", IMAGES_ADD_PATH,
            "--ImageReader.camera_model", "RADIAL",
            "--ImageReader.single_camera", "1",
        ]
        run_command(cmd, "Feature Extraction")

        # Step 3: Feature matching
        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", database_path,
        ]
        run_command(cmd, "Feature Matching")

        # Step 4: Create sparse folder
        sparse_path = os.path.join(WORKSPACE_PATH, "sparse")
        os.makedirs(sparse_path, exist_ok=True)

        # Step 5: Mapper
        cmd = [
            "colmap", "mapper",
            "--database_path", database_path,
            "--image_path", IMAGES_ADD_PATH,
            "--output_path", sparse_path,
        ]
        run_command(cmd, "Sparse Reconstruction (Mapper)")

    # Check results
    sparse_path = os.path.join(WORKSPACE_PATH, "sparse")
    if os.path.exists(sparse_path):
        model_dirs = sorted([d for d in os.listdir(sparse_path) if d.isdigit()])
        if model_dirs:
            model_path = os.path.join(sparse_path, model_dirs[0])
            print(f"\n{'='*60}")
            print(f"SUCCESS! Sparse reconstruction complete.")
            print(f"Model saved to: {model_path}")
            print(f"{'='*60}")

            # List model files
            print("\nModel contents:")
            for f in os.listdir(model_path):
                print(f"  {f}")

            # Copy to a more accessible location
            final_path = os.path.join(SCENE_PATH, "sparse_images_add")
            if os.path.exists(final_path):
                import shutil
                shutil.rmtree(final_path)
            shutil.copytree(model_path, final_path)
            print(f"\nCopied to: {final_path}")

            # Create cameras.txt in correct format
            # The model already has cameras.txt but in COLMAP format
            # We need to check the format
            cameras_file = os.path.join(final_path, "cameras.txt")
            if os.path.exists(cameras_file):
                print(f"\nCameras file exists at: {cameras_file}")
                with open(cameras_file, 'r') as f:
                    print("First 5 lines:")
                    for i, line in enumerate(f):
                        if i < 5:
                            print(f"  {line.strip()}")
        else:
            print("ERROR: No model output found in sparse directory")
    else:
        print("ERROR: Sparse directory not created")


if __name__ == "__main__":
    main()
