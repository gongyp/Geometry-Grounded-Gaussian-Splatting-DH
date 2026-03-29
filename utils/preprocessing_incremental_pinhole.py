"""
Incremental COLMAP preprocessing for multi-timestep datasets.
PINHOLE camera model variant - skips image_undistorter since PINHOLE has no distortion.

Intended for continual-learning 3DGS workflows where each timestep
represents a different acquisition time of the same scene.

Directory structure expected (input):
    input_dir/
        t0/images_raw/   (day_0_*.png/jpg)
        t1/images_raw/   (day_1_*.png/jpg)
        t2/images_raw/   (day_2_*.png/jpg)
        ...

Output structure (per timestep):
    input_dir/
        t0/
            images_raw/          original images
            images/              original images (PINHOLE has no distortion)
            sparse/
                0/               mapper output (cameras/images/points3D.bin)
                cameras.bin       corrected camera params from database
            stereo/              empty (no dense reconstruction)
            database.db
        t1/
            images_raw/          original images
            images/              t0 + t1 images
            sparse/
                0/               t0+t1 registered sparse
                cameras.bin       corrected camera params from database
            stereo/
            database.db
        t2/
            ...

Key design: the t0 sparse model is NEVER modified after initial construction.
Each subsequent timestep registers its images against t0's model, creating
an independent sparse reconstruction that contains t0 + tn.
"""

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_colmap(cmd: list, check=True, **kwargs):
    """Wrapper around subprocess.run that prints the command."""
    print(f"\n[COLMAP] {' '.join(cmd)}")
    subprocess.run(cmd, check=check, **kwargs)


def check_image_dir(img_dir: Path):
    """Verify img_dir contains only image files."""
    if not img_dir.is_dir():
        raise ValueError(f"Image directory not found: {img_dir}")
    files = list(img_dir.iterdir())
    if not files:
        raise ValueError(f"Image directory is empty: {img_dir}")
    for f in files:
        if not (f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]):
            raise ValueError(f"Non-image file found in {img_dir}: {f.name}")


def resize_images(src_dir: Path, dst_dir: Path, max_size: int = 2000):
    """
    Resize all images in src_dir to have max dimension = max_size, preserving aspect ratio.
    Saves resized images to dst_dir.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for img_path in src_dir.iterdir():
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if max(w, h) > max_size:
                scale = max_size / max(w, h)
                new_w, new_h = int(w * scale), int(h * scale)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            out_path = dst_dir / img_path.name
            img.save(out_path, quality=95)


def find_t_dirs(input_dir: Path):
    """Find and sort all t* subdirectories."""
    t_dirs = [d for d in input_dir.iterdir() if d.is_dir() and re.match(r"t\d+$", d.name)]
    if not t_dirs:
        raise ValueError("No subdirectories matching 't*' found.")
    t_dirs.sort(key=lambda d: int(d.name[1:]))
    return t_dirs


def create_image_symlinks(src_dirs: list[Path], dest_dir: Path):
    """
    Create symlinks for all images from src_dirs into dest_dir.
    dest_dir is created if needed.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src_dir in src_dirs:
        for img in src_dir.iterdir():
            target = dest_dir / img.name
            if not target.exists():
                os.symlink(img.resolve(), target)


def get_reconstruction_dir(sparse_dir: Path) -> Path:
    """Return the reconstruction directory (e.g. sparse/0).

    Handles two layouts:
    - sparse/0/  (mapper creates a subdirectory)
    - sparse/    (bundle_adjuster may output files directly here)
    """
    recon_dirs = sorted([d for d in sparse_dir.iterdir() if d.is_dir()])
    if recon_dirs:
        if len(recon_dirs) > 1:
            zero_dir = sparse_dir / "0"
            if zero_dir in recon_dirs:
                return zero_dir
            print(
                f"[preprocessing] Warning: multiple reconstructions found "
                f"({[d.name for d in recon_dirs]}), using '{recon_dirs[0].name}'.",
                file=sys.stderr,
            )
        return recon_dirs[0]

    required = ["cameras.bin", "images.bin", "points3D.bin"]
    if not all((sparse_dir / f).exists() for f in required):
        raise RuntimeError(f"No sparse reconstruction found in {sparse_dir}")
    return sparse_dir


def copy_spanning_tree(src_sparse: Path, dst_sparse: Path):
    """
    Copy t0 sparse model files (cameras.bin, images.bin, points3D.bin)
    to dst_sparse so that image_registrator can use t0 as input.
    """
    dst_sparse.mkdir(parents=True, exist_ok=True)
    for fname in ["cameras.bin", "images.bin", "points3D.bin"]:
        src = src_sparse / fname
        dst = dst_sparse / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def fix_cameras_bin_from_db(db_path: Path, sparse_dir: Path):
    """
    Create/replace sparse/cameras.bin with correct camera params from database.
    This fixes the PINHOLE cameras.bin bug where mapper outputs corrupted dimensions.
    """
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    cur = conn.execute("SELECT camera_id, model, width, height, params FROM cameras")
    row = cur.fetchone()
    conn.close()

    if row is None:
        raise RuntimeError(f"No camera found in database {db_path}")

    camera_id, model_id, width, height, params = row

    # Unpack params from bytes
    if params:
        n_params = len(params) // 8
        param_list = struct.unpack(f'<{n_params}d', params)
    else:
        # Params NULL in database - estimate
        param_list = []

    # Ensure sparse/ directory exists
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # PINHOLE model_id = 1, params: fx, fy, cx, cy
    # If params available in database, use them; otherwise estimate
    if param_list and len(param_list) >= 4:
        fx, fy, cx, cy = param_list[0], param_list[1], param_list[2], param_list[3]
    else:
        # Estimate: fx = fy = 1.2 * max(width, height), cx = width/2, cy = height/2
        fx = fy = 1.2 * max(width, height)
        cx = width / 2.0
        cy = height / 2.0

    # Write in COLMAP binary format (from reconstruction.cc):
    # num_cameras (uint64, 8 bytes)
    # For each camera:
    #   camera_id (int32, 4 bytes)
    #   model_id (int32, 4 bytes)
    #   width (uint64, 8 bytes)
    #   height (uint64, 8 bytes)
    #   params (double[4] for PINHOLE, 32 bytes)
    # Total per camera: 48 bytes
    with open(sparse_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))           # num_cameras = 1
        f.write(struct.pack("<i", camera_id))   # camera_id (int32)
        f.write(struct.pack("<i", model_id))    # model_id (int32, 1 = PINHOLE)
        f.write(struct.pack("<Q", width))       # width (uint64)
        f.write(struct.pack("<Q", height))     # height (uint64)
        f.write(struct.pack("<dddd", fx, fy, cx, cy))  # params (4 doubles)

    print(f"[preprocessing] Fixed cameras.bin: {width}x{height}, fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")


def create_stereo_placeholder(output_dir: Path):
    """Create empty stereo/ directory as placeholder."""
    stereo_dir = output_dir / "stereo"
    stereo_dir.mkdir(exist_ok=True)
    (stereo_dir / "README.txt").write_text(
        "Stereo reconstruction skipped for PINHOLE camera model.\n"
        "PINHOLE has no distortion parameters, so dense reconstruction is not needed.\n"
    )


# ---------------------------------------------------------------------------
# Stage 1: Process t0 (baseline)
# ---------------------------------------------------------------------------

def process_t0(t0_dir: Path, vocab_tree_path: str):
    """
    Full COLMAP pipeline for t0 with PINHOLE model:
    - feature_extractor
    - exhaustive_matcher
    - mapper
    - Skip image_undistorter (PINHOLE has no distortion)
    - Fix cameras.bin from database

    Output structure:
        t0_dir/
            images/              <- original images (PINHOLE: no distortion)
            sparse/
                0/               <- mapper output
                cameras.bin       <- corrected from database
            stereo/              <- placeholder
            database.db
    """
    print(f"\n{'='*60}")
    print(f"Processing t0 (baseline, PINHOLE)")
    print(f"{'='*60}")

    t0_images = t0_dir / "images_raw"
    check_image_dir(t0_images)

    # Output paths within t0_dir
    ws_images = t0_dir / "images"
    ws_db = t0_dir / "database.db"
    ws_sparse = t0_dir / "sparse"

    # Resize t0 images to max 2000px for COLMAP processing
    if ws_images.exists():
        shutil.rmtree(ws_images)
    resize_images(t0_images, ws_images, max_size=2000)

    # Feature extraction
    run_colmap([
        "colmap", "feature_extractor",
        "--database_path", str(ws_db),
        "--image_path", str(ws_images),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",
    ])

    # Matching
    run_colmap([
        "colmap", "exhaustive_matcher",
        "--database_path", str(ws_db),
    ])

    # Mapping (output to sparse/0/)
    ws_sparse.mkdir(parents=True, exist_ok=True)
    run_colmap([
        "colmap", "mapper",
        "--database_path", str(ws_db),
        "--image_path", str(ws_images),
        "--output_path", str(ws_sparse),
        "--Mapper.ba_global_function_tolerance", "0.000001",
    ])

    recon_dir = get_reconstruction_dir(ws_sparse)

    # For PINHOLE: skip image_undistorter (no distortion to remove)
    # Fix cameras.bin from database
    print(f"\n[PINHOLE] Skipping image_undistorter (PINHOLE has no distortion)")
    fix_cameras_bin_from_db(ws_db, ws_sparse)
    create_stereo_placeholder(t0_dir)

    print(f"\n[t0] Done. Sparse: {recon_dir}  |  Output: {t0_dir}")
    return recon_dir


# ---------------------------------------------------------------------------
# Stage N: Process tn (n >= 1) incrementally against t0
# ---------------------------------------------------------------------------

def process_tn(
    tn_dir: Path,
    t0_dir: Path,
    vocab_tree_path: str,
):
    """
    Incremental COLMAP pipeline for timestep tn with PINHOLE model:
    - Uses t0's sparse model as the base (NEVER modifies t0)
    - Registers tn images against t0's model
    - Creates a NEW sparse model containing t0 + tn images
    - Creates a NEW database.db containing t0 + tn image entries
    - Skips image_undistorter (PINHOLE has no distortion)

    Args:
        tn_dir:       Path to tn/ directory (contains images_raw/)
        t0_dir:       Path to t0/ directory (contains sparse/0/)
        vocab_tree_path: Path to COLMAP vocabulary tree file
    """
    print(f"\n{'='*60}")
    print(f"Processing {tn_dir.name} (incremental against t0, PINHOLE)")
    print(f"{'='*60}")

    tn_images = tn_dir / "images_raw"
    check_image_dir(tn_images)

    # Output paths within tn_dir
    ws_images = tn_dir / "images"
    ws_db = tn_dir / "database.db"
    ws_sparse = tn_dir / "sparse"

    # Resize tn images and copy t0 resized images to tn's images directory
    if ws_images.exists():
        shutil.rmtree(ws_images)
    ws_images.mkdir(parents=True)
    # Copy t0's already-resized images
    for img in (t0_dir / "images").iterdir():
        shutil.copy2(img, ws_images / img.name)
    # Resize and add tn images
    resize_images(tn_images, ws_images, max_size=2000)

    # Check if this timestep has already been fully processed.
    def _count_registered(db_path: Path) -> int:
        """Return number of registered images in the database."""
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            cur = conn.execute(
                "SELECT COUNT(*) FROM images WHERE camera_id IS NOT NULL"
            )
            count = cur.fetchone()[0]
            conn.close()
            return count
        except Exception:
            return 0

    def _sparse_has_tn_registered(sparse_dir: Path) -> bool:
        """Return True if sparse already contains tn images (not just t0)."""
        if not (sparse_dir / "0" / "images.bin").exists():
            return False
        t0_db = t0_dir / "database.db"
        t0_registered = _count_registered(t0_db)
        if ws_db.exists():
            tn_registered = _count_registered(ws_db)
            return tn_registered > t0_registered
        return False

    # Copy t0's database as base (contains t0 features, matches, etc.)
    t0_db = t0_dir / "database.db"
    if ws_db.exists():
        ws_db.unlink()
    shutil.copy2(t0_db, ws_db)

    # Check if already processed
    if _sparse_has_tn_registered(tn_dir / "sparse"):
        print(f"[{tn_dir.name}] Sparse already contains tn images — skipping COLMAP steps.")
        print(f"[{tn_dir.name}] Skipped. Output exists: {tn_dir}")
        return

    # Get t0's sparse reconstruction directory (sparse/0/)
    t0_sparse_recon = get_reconstruction_dir(t0_dir / "sparse")

    # Create sparse working directory in tn_dir
    if ws_sparse.exists():
        shutil.rmtree(ws_sparse)

    # Copy t0 sparse as starting point for this timestep (into sparse/0/ later by registrator)
    copy_spanning_tree(t0_sparse_recon, ws_sparse)

    # Step 1: Feature extraction for NEW images only
    run_colmap([
        "colmap", "feature_extractor",
        "--database_path", str(ws_db),
        "--image_path", str(ws_images),
        "--ImageReader.single_camera", "1",
        "--ImageReader.camera_model", "PINHOLE",
    ])

    # Step 2: Vocab tree matching FIRST to establish cross-timestep correspondences
    run_colmap([
        "colmap", "vocab_tree_matcher",
        "--database_path", str(ws_db),
        "--VocabTreeMatching.vocab_tree_path", vocab_tree_path,
    ])

    # Step 3: Register new images (tn) against existing t0 model
    run_colmap([
        "colmap", "image_registrator",
        "--database_path", str(ws_db),
        "--input_path", str(ws_sparse),
        "--output_path", str(ws_sparse),
    ])

    # Step 4: Bundle adjustment
    run_colmap([
        "colmap", "bundle_adjuster",
        "--input_path", str(ws_sparse),
        "--output_path", str(ws_sparse),
    ])

    # For PINHOLE: skip image_undistorter (no distortion to remove)
    # Fix cameras.bin from database
    print(f"\n[PINHOLE] Skipping image_undistorter (PINHOLE has no distortion)")
    fix_cameras_bin_from_db(ws_db, ws_sparse)
    create_stereo_placeholder(tn_dir)

    print(f"\n[{tn_dir.name}] Done. Sparse: {ws_sparse}  |  Output: {tn_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Incremental COLMAP preprocessing for PINHOLE camera model."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory containing t0/, t1/, ... subfolders.",
    )
    parser.add_argument(
        "--vocab-tree",
        type=str,
        default="/home/gongyp/.colmap/vocab_tree_flickr100K_words1M.bin",
        help="Path to COLMAP vocabulary tree file for vocab_tree_matcher.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    vocab_tree = args.vocab_tree

    if not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")

    t_dirs = find_t_dirs(input_dir)
    print(f"\nFound timesteps: {[d.name for d in t_dirs]}")
    print(f"Vocab tree: {vocab_tree}")
    print(f"Camera model: PINHOLE (no undistortion needed)")

    # -------------------------------------------------------------------------
    # Process t0 (baseline)
    # -------------------------------------------------------------------------
    t0_dir = t_dirs[0]
    process_t0(t0_dir, vocab_tree)

    # -------------------------------------------------------------------------
    # Process remaining timesteps incrementally
    # -------------------------------------------------------------------------
    for t_dir in t_dirs[1:]:
        process_tn(
            tn_dir=t_dir,
            t0_dir=t0_dir,
            vocab_tree_path=vocab_tree,
        )

    print(f"\n{'='*60}")
    print("All timesteps processed successfully.")
    print(f"Input dir: {input_dir}")
    print(f"{'='*60}")

    # Print summary
    for t_dir in t_dirs:
        db = t_dir / "database.db"
        sparse = t_dir / "sparse"
        images = t_dir / "images"
        stereo = t_dir / "stereo"

        n_images = len(list(images.glob("*"))) if images.exists() else 0
        n_stereo = len(list(stereo.rglob("*"))) if stereo.exists() else 0

        print(f"\n  {t_dir.name}:")
        print(f"    database.db : {'✓' if db.exists() else '✗'}")
        print(f"    sparse/0/   : {'✓' if (sparse / '0').exists() else '✗'}")
        print(f"    sparse/      : {'✓' if (sparse / 'cameras.bin').exists() else '✗'}")
        print(f"    images/      : {n_images} files")
        print(f"    stereo/      : {n_stereo} files")


if __name__ == "__main__":
    main()
