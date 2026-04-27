"""
Combined Landmark Matching Pipeline for CT Registration Validation
==================================================================
Implements the full recommended workflow:

  Pre-op CT                        Post-op CT
      |                                 |
  Frangi -> bifurcations         Frangi -> bifurcations
      |                                 |
      +-------- ANTs warp --------------+
                    |
              Warped pre-op points
                    |
       Snap to nearest post-op bifurcation (search_radius_mm)
                    |
       Patch similarity filter (remove low-confidence pairs)
                    |
       Export matched .fcsv pair (matching label indices)
                    |
       Load BOTH in 3D Slicer Landmark Registration
                    |
       Reviewer spot-checks, corrects outliers
                    |
       Compute TRE on confirmed pairs

Dependencies:
    pip install SimpleITK scikit-image scipy scikit-learn numpy antspyx pandas

Usage:
    Edit the CONFIG section at the bottom and run:
        python landmark_matching_pipeline.py

Outputs:
    <output_dir>/preop_landmarks.fcsv       -- pre-op fiducials for Slicer
    <output_dir>/postop_landmarks.fcsv      -- matched post-op fiducials for Slicer
    <output_dir>/matched_pairs.csv          -- full match table with TRE/confidence
    <output_dir>/vesselness_preop.nii.gz    -- vesselness map (optional, for QC)
    <output_dir>/vesselness_postop.nii.gz   -- vesselness map (optional, for QC)
    <output_dir>/skeleton_preop.nii.gz      -- skeleton (optional, for QC)
    <output_dir>/skeleton_postop.nii.gz     -- skeleton (optional, for QC)
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from skimage.filters import frangi
from skimage.morphology import skeletonize_3d
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN

try:
    import ants
    ANTS_AVAILABLE = True
except ImportError:
    ANTS_AVAILABLE = False
    print("[WARN] antspyx not installed. ANTs warp propagation disabled.")
    print("       Install with: pip install antspyx")


# =============================================================================
# STEP 1 -- I/O helpers
# =============================================================================

def load_ct(path):
    """Load NIfTI/DICOM image. Returns (array ZYX float32, spacing XYZ, sitk image)."""
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (Z, Y, X)
    spacing = img.GetSpacing()                              # (X, Y, Z)
    return arr, spacing, img


def save_volume(arr, ref_img, path):
    """Save a numpy array as NIfTI using the reference image geometry."""
    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(ref_img)
    sitk.WriteImage(out, path)
    print(f"    Saved: {path}")


# =============================================================================
# STEP 2 -- Frangi vesselness
# =============================================================================

def apply_frangi(arr,
                 scale_range=(1.0, 6.0),
                 scale_step=0.5,
                 alpha=0.5,
                 beta=0.5,
                 gamma=15,
                 black_ridges=False):
    """
    Apply 3-D Frangi vesselness filter.

    Parameters
    ----------
    scale_range   : (min, max) vessel radii in voxels to detect.
                    For liver portal/hepatic branches use (1, 6).
    black_ridges  : False for bright vessels (contrast-enhanced CT).
    """
    arr_norm = np.clip(arr, -200, 400)
    arr_norm = (arr_norm - arr_norm.min()) / (arr_norm.max() - arr_norm.min() + 1e-8)

    scales = np.arange(scale_range[0], scale_range[1], scale_step)

    return frangi(
        arr_norm,
        sigmas=scales,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges
    )


# =============================================================================
# STEP 3 -- Vessel segmentation + skeletonisation
# =============================================================================

def segment_vessels(vesselness, threshold=0.015):
    return (vesselness > threshold).astype(np.uint8)


def skeletonize_vessels(binary_mask):
    return skeletonize_3d(binary_mask)


# =============================================================================
# STEP 4 -- Bifurcation detection
# =============================================================================

def detect_bifurcations(skeleton):
    """
    Skeleton voxels with >= 3 neighbours in a 26-connected neighbourhood
    are classified as bifurcation points.
    """
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0

    neighbor_count = convolve(skeleton.astype(np.uint8),
                              kernel, mode='constant', cval=0)

    bif_mask = (skeleton > 0) & (neighbor_count >= 3)
    bif_coords = np.argwhere(bif_mask)   # (N, 3)  Z Y X
    return bif_coords, bif_mask


# =============================================================================
# STEP 5 -- Cluster + filter + rank
# =============================================================================

def cluster_bifurcations(bif_coords, spacing, eps_mm=3.0):
    """Merge nearby detections using DBSCAN in physical mm space."""
    if len(bif_coords) == 0:
        return np.array([])

    coords_mm = bif_coords * np.array([spacing[2], spacing[1], spacing[0]])
    db = DBSCAN(eps=eps_mm, min_samples=1).fit(coords_mm)

    clustered = []
    for lid in set(db.labels_):
        if lid == -1:
            continue
        mask = db.labels_ == lid
        clustered.append(bif_coords[mask].mean(axis=0))

    return np.array(clustered)


def filter_to_mask(bif_coords, mask_arr):
    """Keep only points inside a binary mask (liver, etc.)."""
    inside = []
    for coord in bif_coords:
        z, y, x = np.round(coord).astype(int)
        if (0 <= z < mask_arr.shape[0] and
                0 <= y < mask_arr.shape[1] and
                0 <= x < mask_arr.shape[2]):
            if mask_arr[z, y, x] > 0:
                inside.append(coord)
    return np.array(inside) if inside else np.array([]).reshape(0, 3)


def rank_by_vesselness(bif_coords, vesselness, top_n=30):
    """Return top_n candidates ranked by mean local vesselness."""
    if len(bif_coords) == 0:
        return bif_coords

    scores = []
    for coord in bif_coords:
        z, y, x = np.round(coord).astype(int)
        patch = vesselness[max(0, z-2):z+3,
                           max(0, y-2):y+3,
                           max(0, x-2):x+3]
        scores.append(patch.mean())

    idx = np.argsort(scores)[::-1]
    return bif_coords[idx[:top_n]]


def run_bifurcation_pipeline(arr, spacing, liver_mask,
                              frangi_cfg, seg_threshold,
                              cluster_eps_mm, top_n,
                              label=""):
    """
    Full single-image bifurcation detection.
    Returns (bifurcation coords ZYX, vesselness map, skeleton).
    """
    tag = f"[{label}] " if label else ""

    print(f"  {tag}Frangi filter ...")
    ves = apply_frangi(arr, **frangi_cfg)

    print(f"  {tag}Segmenting vessels ...")
    binary = segment_vessels(ves, threshold=seg_threshold)

    print(f"  {tag}Skeletonising ...")
    skel = skeletonize_vessels(binary)

    print(f"  {tag}Detecting bifurcations ...")
    bif_raw, _ = detect_bifurcations(skel)
    print(f"      Raw detections         : {len(bif_raw)}")

    bif_c = cluster_bifurcations(bif_raw, spacing, eps_mm=cluster_eps_mm)
    print(f"      After clustering       : {len(bif_c)}")

    if liver_mask is not None:
        bif_c = filter_to_mask(bif_c, liver_mask)
        print(f"      Inside liver mask      : {len(bif_c)}")

    bif_ranked = rank_by_vesselness(bif_c, ves, top_n=top_n)
    print(f"      Top candidates kept    : {len(bif_ranked)}")

    return bif_ranked, ves, skel


# =============================================================================
# STEP 6 -- ANTs warp propagation
# =============================================================================

def voxel_to_lps(coords_zyx, spacing, origin):
    """Convert voxel (Z,Y,X) array to physical LPS mm coordinates."""
    pts = []
    for z, y, x in coords_zyx:
        pts.append([
            origin[0] + x * spacing[0],
            origin[1] + y * spacing[1],
            origin[2] + z * spacing[2]
        ])
    return np.array(pts)


def lps_to_voxel(pts_lps, spacing, origin):
    """Convert physical LPS mm to voxel (Z,Y,X)."""
    coords = []
    for px, py, pz in pts_lps:
        coords.append([
            (pz - origin[2]) / spacing[2],
            (py - origin[1]) / spacing[1],
            (px - origin[0]) / spacing[0]
        ])
    return np.array(coords)


def warp_points_ants(coords_zyx, spacing_preop, origin_preop,
                     fwd_transform_path):
    """
    Apply ANTs forward transform to a set of voxel coordinates.
    Returns warped coordinates in LPS mm (not yet in post-op voxels --
    use lps_to_voxel with post-op geometry afterward).
    """
    if not ANTS_AVAILABLE:
        raise RuntimeError("antspyx is not installed.")

    pts_lps = voxel_to_lps(coords_zyx, spacing_preop, origin_preop)
    df = pd.DataFrame(pts_lps, columns=['x', 'y', 'z'])

    warped_df = ants.apply_transforms_to_points(
        dim=3,
        points=df,
        transformlist=[fwd_transform_path],
        whichtoinvert=[False]
    )
    return warped_df[['x', 'y', 'z']].values   # LPS mm


def fallback_identity_warp(coords_zyx, spacing_preop, origin_preop):
    """
    When ANTs is unavailable, return the pre-op points converted to LPS
    as a naive (identity) estimate of post-op positions.
    Useful when pre/post images are already roughly aligned.
    """
    print("  [WARN] Using identity warp fallback (no ANTs transform).")
    return voxel_to_lps(coords_zyx, spacing_preop, origin_preop)


# =============================================================================
# STEP 7 -- Snap warped points to nearest post-op bifurcation
# =============================================================================

def snap_to_nearest_bifurcation(warped_lps, postop_bif_zyx,
                                 spacing_postop, origin_postop,
                                 search_radius_mm=8.0):
    """
    For each warped pre-op point, find the nearest post-op bifurcation
    within search_radius_mm. Falls back to the warped position if none found.

    Returns list of dicts with keys:
        warped_lps, matched_lps, matched_vox_zyx, dist_mm, snapped
    """
    postop_lps = voxel_to_lps(postop_bif_zyx, spacing_postop, origin_postop)

    results = []
    for wpt in warped_lps:
        if len(postop_lps) > 0:
            dists = np.linalg.norm(postop_lps - wpt, axis=1)
            nearest_idx = np.argmin(dists)
            nearest_dist = dists[nearest_idx]
        else:
            nearest_dist = np.inf

        if nearest_dist <= search_radius_mm:
            matched_lps = postop_lps[nearest_idx]
            matched_vox = postop_bif_zyx[nearest_idx]
            snapped = True
        else:
            matched_lps = wpt                                   # identity fallback
            matched_vox = lps_to_voxel([wpt],
                                        spacing_postop,
                                        origin_postop)[0]
            snapped = False

        results.append({
            'warped_lps'    : wpt,
            'matched_lps'   : matched_lps,
            'matched_vox_zyx': matched_vox,
            'dist_mm'       : nearest_dist if nearest_dist != np.inf else -1,
            'snapped'       : snapped
        })

    return results


# =============================================================================
# STEP 8 -- Patch similarity filter
# =============================================================================

def extract_patch(arr, coord_zyx, radius=5):
    z, y, x = [int(round(c)) for c in coord_zyx]
    patch = arr[max(0, z-radius):z+radius+1,
                max(0, y-radius):y+radius+1,
                max(0, x-radius):x+radius+1]
    flat = patch.flatten().astype(np.float32)
    std = flat.std()
    return (flat - flat.mean()) / std if std > 0 else flat - flat.mean()


def compute_patch_similarity(preop_coords_zyx, postop_coords_zyx,
                              preop_arr, postop_arr, radius=5):
    """
    Normalised cross-correlation between local intensity patches.
    Returns similarity scores in [0, 1] for each pair.
    """
    scores = []
    for pc, qc in zip(preop_coords_zyx, postop_coords_zyx):
        p = extract_patch(preop_arr, pc, radius)
        q = extract_patch(postop_arr, qc, radius)

        min_len = min(len(p), len(q))
        if min_len == 0:
            scores.append(0.0)
            continue

        p, q = p[:min_len], q[:min_len]
        cos_sim = 1.0 - cdist([p], [q], metric='cosine')[0, 0]
        scores.append(float(np.clip(cos_sim, 0, 1)))

    return np.array(scores)


# =============================================================================
# STEP 9 -- Build final matched pairs table
# =============================================================================

def build_matched_pairs(preop_bif_zyx, snap_results,
                         preop_arr, postop_arr,
                         preop_spacing, postop_spacing,
                         preop_origin, postop_origin,
                         patch_radius=5,
                         min_similarity=0.25):
    """
    Combine snap results with patch similarity scores.
    Returns a DataFrame sorted by confidence (similarity desc).
    """
    postop_coords = np.array([r['matched_vox_zyx'] for r in snap_results])
    preop_coords  = np.array(preop_bif_zyx)

    print("  Computing patch similarity scores ...")
    sim_scores = compute_patch_similarity(
        preop_coords, postop_coords, preop_arr, postop_arr,
        radius=patch_radius
    )

    rows = []
    for i, (r, sim) in enumerate(zip(snap_results, sim_scores)):
        pz, py, px = np.round(preop_coords[i]).astype(int)
        qz, qy, qx = np.round(r['matched_vox_zyx']).astype(int)

        # Physical distance between pre-op point and its warped estimate
        dist_warp = r['dist_mm']

        # Confidence label
        if sim >= 0.6 and r['snapped']:
            confidence = 'high'
        elif sim >= min_similarity:
            confidence = 'medium'
        else:
            confidence = 'low'

        rows.append({
            'label'           : f"LM_{i:03d}",
            'preop_z'         : pz,
            'preop_y'         : py,
            'preop_x'         : px,
            'postop_z'        : qz,
            'postop_y'        : qy,
            'postop_x'        : qx,
            'snap_dist_mm'    : round(dist_warp, 2),
            'patch_similarity': round(sim, 3),
            'snapped_to_bif'  : r['snapped'],
            'confidence'      : confidence,
        })

    df = pd.DataFrame(rows)
    # Filter out low-confidence, sort high-confidence first
    df = df[df['patch_similarity'] >= min_similarity].copy()
    order = {'high': 0, 'medium': 1, 'low': 2}
    df['_sort'] = df['confidence'].map(order)
    df = df.sort_values(['_sort', 'patch_similarity'],
                         ascending=[True, False]).drop(columns='_sort')
    df = df.reset_index(drop=True)

    # Re-label in confidence order
    df['label'] = [f"LM_{i:03d}" for i in range(len(df))]

    print(f"  Matched pairs kept         : {len(df)}"
          f"  (high={( df.confidence=='high').sum()}"
          f", medium={(df.confidence=='medium').sum()}"
          f", low={(df.confidence=='low').sum()})")

    return df


# =============================================================================
# STEP 10 -- Export .fcsv files
# =============================================================================

def coords_to_ras(z, y, x, spacing, origin):
    """Voxel (Z,Y,X) -> RAS mm (Slicer convention: negate X and Y)."""
    lps_x = origin[0] + x * spacing[0]
    lps_y = origin[1] + y * spacing[1]
    lps_z = origin[2] + z * spacing[2]
    return -lps_x, -lps_y, lps_z


def export_fcsv(df, ref_img, coord_cols_zyx, output_path, description=""):
    """
    Write a Slicer Markups .fcsv file from a DataFrame row set.
    coord_cols_zyx: tuple of (z_col, y_col, x_col) column names.
    """
    spacing = ref_img.GetSpacing()
    origin  = ref_img.GetOrigin()
    zc, yc, xc = coord_cols_zyx

    lines = [
        "# Markups fiducial file version = 4.11",
        f"# {description}",
        "# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID"
    ]

    for _, row in df.iterrows():
        rx, ry, rz = coords_to_ras(
            row[zc], row[yc], row[xc], spacing, origin
        )
        conf = row.get('confidence', '')
        lines.append(
            f"vtkMRMLMarkupsFiducialNode_{row['label']},"
            f"{rx:.3f},{ry:.3f},{rz:.3f},"
            f"0,0,0,1,1,1,0,{row['label']},{conf},"
        )

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"    Saved: {output_path}  ({len(df)} landmarks)")


# =============================================================================
# STEP 11 -- Optional TRE computation (post manual confirmation)
# =============================================================================

def compute_tre(df, preop_img, postop_img):
    """
    Compute Target Registration Error (TRE) in mm for each matched pair.
    Assumes the pre-op landmarks have already been warped to post-op space
    (i.e. warp quality is what we are measuring).

    In practice, call this after manual review of the .fcsv files in Slicer.
    Here it uses the auto-matched pairs as an approximation.
    """
    preop_spacing  = preop_img.GetSpacing()
    preop_origin   = preop_img.GetOrigin()
    postop_spacing = postop_img.GetSpacing()
    postop_origin  = postop_img.GetOrigin()

    tres = []
    for _, row in df.iterrows():
        pre_lps  = voxel_to_lps(
            [[row['preop_z'],  row['preop_y'],  row['preop_x']]],
            preop_spacing, preop_origin
        )[0]
        post_lps = voxel_to_lps(
            [[row['postop_z'], row['postop_y'], row['postop_x']]],
            postop_spacing, postop_origin
        )[0]
        tres.append(np.linalg.norm(pre_lps - post_lps))

    df = df.copy()
    df['tre_mm'] = np.round(tres, 3)
    return df


def print_tre_summary(df):
    if 'tre_mm' not in df.columns:
        return
    t = df['tre_mm']
    print("\n  --- TRE Summary (auto-matched pairs, pre manual review) ---")
    print(f"  N pairs     : {len(t)}")
    print(f"  Mean TRE    : {t.mean():.2f} mm")
    print(f"  Median TRE  : {t.median():.2f} mm")
    print(f"  95th pct    : {t.quantile(0.95):.2f} mm")
    print(f"  Max TRE     : {t.max():.2f} mm")
    print("  (Re-run after manual correction in Slicer for ground-truth TRE)")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(cfg):
    os.makedirs(cfg['output_dir'], exist_ok=True)
    out = lambda f: os.path.join(cfg['output_dir'], f)

    # ------------------------------------------------------------------
    print("\n[1/8] Loading images ...")
    preop_arr,  preop_spacing,  preop_img  = load_ct(cfg['preop_ct'])
    postop_arr, postop_spacing, postop_img = load_ct(cfg['postop_ct'])

    preop_mask = postop_mask = None
    if cfg.get('preop_liver_mask'):
        pm, _, _ = load_ct(cfg['preop_liver_mask'])
        preop_mask = (pm > 0).astype(np.uint8)
    if cfg.get('postop_liver_mask'):
        qm, _, _ = load_ct(cfg['postop_liver_mask'])
        postop_mask = (qm > 0).astype(np.uint8)

    # ------------------------------------------------------------------
    frangi_cfg = {
        'scale_range'  : cfg.get('scale_range', (1.0, 6.0)),
        'scale_step'   : cfg.get('scale_step',  0.5),
        'alpha'        : cfg.get('alpha',        0.5),
        'beta'         : cfg.get('beta',         0.5),
        'gamma'        : cfg.get('gamma',        15),
        'black_ridges' : cfg.get('black_ridges', False),
    }
    seg_thr     = cfg.get('vesselness_threshold', 0.015)
    eps_mm      = cfg.get('cluster_eps_mm',       3.0)
    top_n       = cfg.get('top_n',                30)

    print("\n[2/8] Pre-op bifurcation detection ...")
    preop_bif, preop_ves, preop_skel = run_bifurcation_pipeline(
        preop_arr, preop_spacing, preop_mask,
        frangi_cfg, seg_thr, eps_mm, top_n, label="pre-op"
    )

    print("\n[3/8] Post-op bifurcation detection ...")
    postop_bif, postop_ves, postop_skel = run_bifurcation_pipeline(
        postop_arr, postop_spacing, postop_mask,
        frangi_cfg, seg_thr, eps_mm, top_n * 2,   # keep more candidates as pool
        label="post-op"
    )

    # ------------------------------------------------------------------
    print("\n[4/8] Warping pre-op landmarks to post-op space ...")
    transform_path = cfg.get('ants_transform')

    if transform_path and ANTS_AVAILABLE:
        print("  Applying ANTs forward transform ...")
        warped_lps = warp_points_ants(
            preop_bif,
            preop_spacing,
            preop_img.GetOrigin(),
            transform_path
        )
    else:
        warped_lps = fallback_identity_warp(
            preop_bif,
            preop_spacing,
            preop_img.GetOrigin()
        )

    # ------------------------------------------------------------------
    print("\n[5/8] Snapping warped points to nearest post-op bifurcations ...")
    snap_results = snap_to_nearest_bifurcation(
        warped_lps,
        postop_bif,
        postop_spacing,
        postop_img.GetOrigin(),
        search_radius_mm=cfg.get('search_radius_mm', 8.0)
    )
    n_snapped = sum(r['snapped'] for r in snap_results)
    print(f"  Snapped to detected bifurcation : {n_snapped} / {len(snap_results)}")

    # ------------------------------------------------------------------
    print("\n[6/8] Computing patch similarity & building match table ...")
    matched_df = build_matched_pairs(
        preop_bif, snap_results,
        preop_arr, postop_arr,
        preop_spacing, postop_spacing,
        preop_img.GetOrigin(), postop_img.GetOrigin(),
        patch_radius=cfg.get('patch_radius', 5),
        min_similarity=cfg.get('min_similarity', 0.25)
    )

    # ------------------------------------------------------------------
    print("\n[7/8] Computing preliminary TRE ...")
    matched_df = compute_tre(matched_df, preop_img, postop_img)
    print_tre_summary(matched_df)

    # ------------------------------------------------------------------
    print("\n[8/8] Exporting results ...")

    matched_df.to_csv(out('matched_pairs.csv'), index=False)
    print(f"    Saved: {out('matched_pairs.csv')}")

    export_fcsv(
        matched_df, preop_img,
        coord_cols_zyx=('preop_z',  'preop_y',  'preop_x'),
        output_path=out('preop_landmarks.fcsv'),
        description="Pre-op bifurcation landmarks"
    )
    export_fcsv(
        matched_df, postop_img,
        coord_cols_zyx=('postop_z', 'postop_y', 'postop_x'),
        output_path=out('postop_landmarks.fcsv'),
        description="Post-op matched bifurcation landmarks"
    )

    if cfg.get('save_intermediate', False):
        save_volume(preop_ves.astype(np.float32),  preop_img,  out('vesselness_preop.nii.gz'))
        save_volume(postop_ves.astype(np.float32), postop_img, out('vesselness_postop.nii.gz'))
        save_volume(preop_skel.astype(np.uint8),   preop_img,  out('skeleton_preop.nii.gz'))
        save_volume(postop_skel.astype(np.uint8),  postop_img, out('skeleton_postop.nii.gz'))

    print("\n=== Pipeline complete ===")
    print(f"  Output directory : {cfg['output_dir']}")
    print(f"  Load in Slicer   : File -> Add Data -> select both .fcsv files")
    print(f"  Then use         : Modules -> Registration -> Landmark Registration")
    print(f"  to confirm / correct pairs before computing final TRE.\n")

    return matched_df


# =============================================================================
# CONFIG -- edit these paths and parameters before running
# =============================================================================

if __name__ == "__main__":

    CONFIG = {
        # ---- Required inputs ------------------------------------------------
        "preop_ct"           : "preop_ct.nii.gz",
        "postop_ct"          : "postop_ct.nii.gz",

        # ---- Optional liver masks (from TotalSegmentator) -------------------
        #      Strongly recommended -- greatly reduces false positives
        "preop_liver_mask"   : "preop_liver_mask.nii.gz",
        "postop_liver_mask"  : "postop_liver_mask.nii.gz",

        # ---- ANTs transform (pre-op -> post-op) -----------------------------
        #      Set to None to use identity warp fallback
        "ants_transform"     : "preop_to_postop_Warp.nii.gz",

        # ---- Output ---------------------------------------------------------
        "output_dir"         : "landmark_output",

        # ---- Frangi parameters (vessel detection) ---------------------------
        "scale_range"         : (1.0, 6.0),   # mm, adjust for vessel size
        "scale_step"          : 0.5,
        "alpha"               : 0.5,
        "beta"                : 0.5,
        "gamma"               : 15,
        "black_ridges"        : False,         # False = bright vessels (CE-CT)

        # ---- Segmentation + bifurcation parameters --------------------------
        "vesselness_threshold": 0.015,         # raise to reduce noise
        "cluster_eps_mm"      : 3.0,           # merge radius for nearby points
        "top_n"               : 25,            # max candidates per image

        # ---- Matching parameters --------------------------------------------
        "search_radius_mm"    : 8.0,           # snap search radius
        "patch_radius"        : 5,             # voxels for NCC patch
        "min_similarity"      : 0.25,          # discard pairs below this NCC

        # ---- Save intermediate volumes for QC in Slicer ---------------------
        "save_intermediate"   : True,
    }

    results = run_pipeline(CONFIG)
