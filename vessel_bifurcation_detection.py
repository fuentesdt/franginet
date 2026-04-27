"""
Vessel Bifurcation Detection for CT Image Registration Validation
=================================================================
Uses Frangi vesselness filter to auto-identify candidate vessel
bifurcations as landmarks for pre-op / post-op CT registration validation.

Dependencies:
    pip install SimpleITK scikit-image scipy scikit-learn numpy

Usage:
    python vessel_bifurcation_detection.py
    Then edit the paths at the bottom of the file under "Entry point".
"""

import numpy as np
import SimpleITK as sitk
from skimage.filters import frangi
from skimage.morphology import skeletonize_3d
from scipy.ndimage import convolve
from sklearn.cluster import DBSCAN


# ---------------------------------------------------------------------------
# 1. I/O
# ---------------------------------------------------------------------------

def load_ct(path):
    """Load a NIfTI/DICOM CT image. Returns (array ZYX, spacing XYZ, sitk image)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)   # shape: (Z, Y, X)
    spacing = img.GetSpacing()          # (X, Y, Z)
    return arr, spacing, img


# ---------------------------------------------------------------------------
# 2. Frangi vesselness
# ---------------------------------------------------------------------------

def apply_frangi(arr, spacing,
                 scale_range=(1, 6), scale_step=0.5,
                 alpha=0.5, beta=0.5, gamma=15):
    """
    Apply 3-D Frangi vesselness filter.

    Parameters
    ----------
    scale_range : (min_mm, max_mm)
        Vessel radii (in voxels) to detect.
        For liver portal/hepatic branches use (1, 6).
    alpha, beta  : Frangi sensitivity parameters.
    gamma        : Sensitivity to background noise.
    black_ridges : False  -> bright vessels (contrast-enhanced CT).
    """
    # Clip to vessel-relevant HU range and normalise to [0, 1]
    arr_norm = np.clip(arr, -200, 400).astype(np.float32)
    arr_norm = (arr_norm - arr_norm.min()) / (arr_norm.max() - arr_norm.min())

    scales = np.arange(scale_range[0], scale_range[1], scale_step)

    vesselness = frangi(
        arr_norm,
        sigmas=scales,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=False   # vessels are bright post-contrast
    )
    return vesselness


# ---------------------------------------------------------------------------
# 3. Binary segmentation
# ---------------------------------------------------------------------------

def segment_vessels(vesselness, threshold=0.015):
    """Threshold vesselness map -> binary vessel mask."""
    return (vesselness > threshold).astype(np.uint8)


# ---------------------------------------------------------------------------
# 4. Skeletonisation
# ---------------------------------------------------------------------------

def skeletonize_vessels(binary_mask):
    """3-D thinning -> 1-voxel-wide centerline skeleton."""
    return skeletonize_3d(binary_mask)


# ---------------------------------------------------------------------------
# 5. Bifurcation detection
# ---------------------------------------------------------------------------

def detect_bifurcations(skeleton):
    """
    Bifurcation points = skeleton voxels with >= 3 neighbours
    (26-connectivity 3x3x3 kernel).

    Returns
    -------
    bif_coords   : (N, 3) int array  -- (Z, Y, X) voxel coordinates
    bif_volume   : bool volume same shape as skeleton
    """
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0                     # exclude centre voxel

    neighbor_count = convolve(
        skeleton.astype(np.uint8),
        kernel,
        mode='constant',
        cval=0
    )

    bifurcations = (skeleton > 0) & (neighbor_count >= 3)
    bif_coords = np.argwhere(bifurcations)  # shape (N, 3)
    return bif_coords, bifurcations


# ---------------------------------------------------------------------------
# 6. Clustering -- merge nearby detections into single points
# ---------------------------------------------------------------------------

def cluster_bifurcations(bif_coords, spacing, eps_mm=3.0, min_samples=1):
    """
    DBSCAN clustering in physical (mm) space so that duplicate voxels
    within eps_mm are merged to a single centroid.

    Parameters
    ----------
    eps_mm : float  -- merge radius in millimetres (default 3 mm).
    """
    if len(bif_coords) == 0:
        return np.array([])

    # (Z, Y, X) voxel -> mm  [spacing is (X, Y, Z)]
    coords_mm = bif_coords * np.array([spacing[2], spacing[1], spacing[0]])

    db = DBSCAN(eps=eps_mm, min_samples=min_samples).fit(coords_mm)

    clustered = []
    for label_id in set(db.labels_):
        if label_id == -1:
            continue                        # noise
        mask = db.labels_ == label_id
        centroid = bif_coords[mask].mean(axis=0)
        clustered.append(centroid)

    return np.array(clustered)


# ---------------------------------------------------------------------------
# 7. Restrict to liver ROI
# ---------------------------------------------------------------------------

def filter_to_liver(bif_coords, liver_mask):
    """Keep only bifurcation candidates that fall inside the liver mask."""
    inside = []
    for coord in bif_coords:
        z, y, x = np.round(coord).astype(int)
        # bounds check
        if (0 <= z < liver_mask.shape[0] and
                0 <= y < liver_mask.shape[1] and
                0 <= x < liver_mask.shape[2]):
            if liver_mask[z, y, x] > 0:
                inside.append(coord)
    return np.array(inside)


# ---------------------------------------------------------------------------
# 8. Rank by local vesselness strength
# ---------------------------------------------------------------------------

def rank_bifurcations(bif_coords, vesselness, top_n=25):
    """
    Score each candidate by mean vesselness in a small neighbourhood.
    Returns the top_n highest-confidence candidates.
    """
    if len(bif_coords) == 0:
        return bif_coords

    scores = []
    for coord in bif_coords:
        z, y, x = np.round(coord).astype(int)
        patch = vesselness[
            max(0, z - 2):z + 3,
            max(0, y - 2):y + 3,
            max(0, x - 2):x + 3
        ]
        scores.append(patch.mean())

    ranked_idx = np.argsort(scores)[::-1]
    return bif_coords[ranked_idx[:top_n]]


# ---------------------------------------------------------------------------
# 9. Export to 3D Slicer .fcsv fiducial file
# ---------------------------------------------------------------------------

def export_as_fcsv(coords, ref_image, output_path):
    """
    Write candidate bifurcations as a 3D Slicer Markups Fiducial file (.fcsv).
    Converts voxel (Z, Y, X) -> RAS mm coordinates expected by Slicer.

    Load in Slicer via: File -> Add Data -> select .fcsv
    """
    origin   = ref_image.GetOrigin()    # (X, Y, Z)
    spacing  = ref_image.GetSpacing()   # (X, Y, Z)

    lines = [
        "# Markups fiducial file version = 4.11",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
    ]

    for i, (z, y, x) in enumerate(coords):
        # Voxel -> LPS physical coordinates
        phys_x = origin[0] + x * spacing[0]
        phys_y = origin[1] + y * spacing[1]
        phys_z = origin[2] + z * spacing[2]
        # LPS -> RAS (negate X and Y for Slicer convention)
        ras_x, ras_y, ras_z = -phys_x, -phys_y, phys_z
        lines.append(
            f"vtkMRMLMarkupsFiducialNode_{i},"
            f"{ras_x:.3f},{ras_y:.3f},{ras_z:.3f},"
            f"0,0,0,1,1,1,0,BIF_{i:03d},,"
        )

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"[OK] Exported {len(coords)} bifurcation candidates -> {output_path}")


# ---------------------------------------------------------------------------
# 10. Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    ct_path,
    liver_mask_path,
    output_fcsv,
    scale_range=(1, 6),
    vesselness_threshold=0.015,
    cluster_eps_mm=3.0,
    top_n=25
):
    """
    End-to-end bifurcation detection pipeline.

    Parameters
    ----------
    ct_path                  : Path to pre-op (or post-op) CT NIfTI file.
    liver_mask_path          : Path to liver segmentation mask
                               (e.g. from TotalSegmentator).
    output_fcsv              : Output .fcsv file path for 3D Slicer.
    scale_range              : Frangi filter scale range in mm.
    vesselness_threshold     : Threshold for binary vessel mask.
    cluster_eps_mm           : DBSCAN merge radius in mm.
    top_n                    : Number of top candidates to export.
    """
    print("[1/7] Loading CT ...")
    arr, spacing, ref_img = load_ct(ct_path)

    print("[2/7] Loading liver mask ...")
    liver_arr, _, _ = load_ct(liver_mask_path)

    print("[3/7] Applying Frangi vesselness filter ...")
    vesselness = apply_frangi(arr, spacing, scale_range=scale_range)

    print("[4/7] Segmenting vessels ...")
    binary = segment_vessels(vesselness, threshold=vesselness_threshold)

    print("[5/7] Skeletonising ...")
    skeleton = skeletonize_vessels(binary)

    print("[6/7] Detecting & clustering bifurcations ...")
    bif_raw, _ = detect_bifurcations(skeleton)
    print(f"      Raw bifurcation voxels : {len(bif_raw)}")

    bif_clustered = cluster_bifurcations(bif_raw, spacing, eps_mm=cluster_eps_mm)
    print(f"      After clustering       : {len(bif_clustered)}")

    bif_liver = filter_to_liver(bif_clustered, liver_arr.astype(int))
    print(f"      Inside liver mask      : {len(bif_liver)}")

    bif_ranked = rank_bifurcations(bif_liver, vesselness, top_n=top_n)
    print(f"      Top candidates kept    : {len(bif_ranked)}")

    print("[7/7] Exporting .fcsv ...")
    export_as_fcsv(bif_ranked, ref_img, output_fcsv)
    print("[OK] Done.")


# ---------------------------------------------------------------------------
# Entry point -- edit paths here before running
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(
        ct_path              = "preop_ct.nii.gz",
        liver_mask_path      = "liver_mask.nii.gz",   # from TotalSegmentator
        output_fcsv          = "candidate_bifurcations_preop.fcsv",
        scale_range          = (1, 6),
        vesselness_threshold = 0.015,
        cluster_eps_mm       = 3.0,
        top_n                = 25,
    )
