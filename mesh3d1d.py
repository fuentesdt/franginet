"""
mesh3d1d.py — 3D tetrahedral mesh with vessel skeleton and surface constraints.

Label conventions (default)
---------------------------
  1 = liver
  2 = vessel

Pipeline
--------
  NIfTI label
    ├─ Vessel skeleton  (label==2)
    │    3-D topological thinning (Lee 1994).
    │    One node per skeleton voxel, positioned at world-coordinate voxel
    │    centre via NIfTI affine.  1-to-1 pixel correspondence is guaranteed;
    │    no smoothing or repositioning is applied to skeleton nodes.
    │
    └─ Outer surface  (label > 0)
         Marching cubes → largest CC → quadric decimation → Taubin smoothing
         → MeshFix × 2 → consistent normals.
         The surface may be decimated/smoothed to reduce mesh size.

  Both node sets are appended into a single pyvista PolyData passed to TetGen:
    • surface triangles define the domain boundary
    • skeleton nodes sit inside the domain as orphan (cell-less) points
  TetGen preserves ALL input points in its Delaunay triangulation, so every
  surface node and every skeleton node appears in the final tet mesh.
  Tetrahedra are naturally small near the densely-spaced skeleton nodes and
  larger in the open liver parenchyma — no explicit size metric file needed.

  Output PointData array 'node_set':
    0 = Steiner (inserted by TetGen for quality)
    1 = surface node  (from outer boundary mesh)
    2 = vessel skeleton node (1-to-1 with skeleton voxels)

Outputs
-------
  <out_dir>/liver_surface.vtp   — smoothed outer surface triangulation
  <out_dir>/vessel_skeleton.vtp — vessel skeleton point cloud
  <out_dir>/liver_tet.vtu       — tetrahedral volume mesh

Usage
-----
  python mesh3d1d.py prepost/PreTxArtLoRes.vessellabel.nii.gz
  python mesh3d1d.py prepost/PreTxArtLoRes.vessellabel.nii.gz \\
      --vessel-label 2 --decimate 0.75 --smooth-iter 50 \\
      --tet-quality 1.5 --tet-dihedral 20 --tet-maxvol 500 --output-dir newdata

Dependencies
------------
  pip install nibabel pyvista tetgen pymeshfix scikit-image numpy scipy
"""

import argparse
import os
import sys
import time

import numpy as np
import nibabel as nib
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import tetgen
import pymeshfix
from scipy.ndimage import binary_fill_holes, binary_closing
from scipy.spatial import KDTree
from skimage.morphology import skeletonize


def log(msg):
    print(f"[mesh3d1d] {msg}", flush=True)


# ── NIfTI loading ─────────────────────────────────────────────────────────────

def load_label_nii(path):
    """Return (data int array, affine float64)."""
    log(f"Loading  : {path}")
    nii     = nib.load(path)
    affine  = nii.affine.astype(np.float64)
    data    = np.asarray(nii.dataobj)
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    log(f"  Shape   : {data.shape}")
    log(f"  Labels  : {np.unique(data).tolist()}")
    log(f"  Spacing : {spacing.round(3).tolist()} mm")
    return data, affine


# ── Vessel skeleton ───────────────────────────────────────────────────────────

def vessel_skeleton_nodes(data, affine, vessel_label=2):
    """
    3-D topological skeletonization of the vessel mask (Lee 1994 iterative
    thinning via skimage).  Returns an N×3 float64 array of world-coordinate
    node positions — one node per non-zero skeleton voxel.

    The 1-to-1 correspondence between skeleton voxels and nodes is strictly
    preserved: nodes are placed at the voxel-centre world coordinate derived
    from the NIfTI affine, with no smoothing or repositioning.

    Also returns the boolean skeleton volume (same shape as data) so the
    caller can save it or use it for visualisation.
    """
    log(f"Skeletonizing vessels (label={vessel_label}) ...")
    binary = (data == vessel_label)
    log(f"  Vessel voxels   : {int(binary.sum())}")

    t0   = time.time()
    skel = skeletonize(binary)          # 3-D Lee 1994 thinning
    log(f"  Skeleton voxels : {int(skel.sum())}  ({time.time()-t0:.1f} s)")

    if skel.sum() == 0:
        log("  Warning: empty skeleton — vessel_label may be wrong")
        return np.zeros((0, 3), dtype=np.float64), skel

    # Voxel-index → world coordinates via NIfTI affine
    ijk = np.argwhere(skel).astype(np.float64)       # K × 3
    ones = np.ones((len(ijk), 1))
    world = (affine @ np.hstack([ijk, ones]).T).T[:, :3]

    bb_min, bb_max = world.min(axis=0), world.max(axis=0)
    log(f"  BBox x : {bb_min[0]:.1f} – {bb_max[0]:.1f} mm")
    log(f"  BBox y : {bb_min[1]:.1f} – {bb_max[1]:.1f} mm")
    log(f"  BBox z : {bb_min[2]:.1f} – {bb_max[2]:.1f} mm")

    return world, skel


# ── Surface extraction ────────────────────────────────────────────────────────

def extract_surface(data, affine, fill=True, close_iters=1):
    """
    Marching cubes surface of all labeled voxels (label > 0) in world mm.

    Using label > 0 captures the outer boundary of the combined liver+vessel
    domain.  fill=True removes enclosed cavities (bile ducts, etc.) before
    surface extraction.
    """
    log("Building outer surface mask (label > 0) ...")
    binary = (data > 0).astype(np.uint8)
    log(f"  Labeled voxels : {int(binary.sum())}")

    if fill:
        binary = binary_fill_holes(binary).astype(np.uint8)
        log(f"  After fill     : {int(binary.sum())} voxels")

    if close_iters > 0:
        binary = binary_closing(binary, iterations=close_iters).astype(np.uint8)
        log(f"  After closing  : {int(binary.sum())} voxels")

    log("Extracting surface (marching cubes) ...")
    grid            = pv.ImageData()
    grid.dimensions = binary.shape
    grid.spacing    = (1.0, 1.0, 1.0)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.point_data['mask'] = binary.flatten(order='F').astype(np.float32)

    surface = grid.contour([0.5], scalars='mask', method='marching_cubes')
    log(f"  Raw surface    : {surface.n_points} pts, {surface.n_cells} cells")
    if surface.n_points == 0:
        sys.exit("Error: marching cubes produced an empty surface")

    # Voxel space → world mm via NIfTI affine
    pts   = surface.points
    ones  = np.ones((len(pts), 1))
    world = (affine @ np.hstack([pts, ones]).T).T[:, :3]
    surface = surface.copy()
    surface.points = world

    bb = surface.bounds
    log(f"  BBox x : {bb[0]:.1f} – {bb[1]:.1f} mm")
    log(f"  BBox y : {bb[2]:.1f} – {bb[3]:.1f} mm")
    log(f"  BBox z : {bb[4]:.1f} – {bb[5]:.1f} mm")
    return surface


# ── Surface processing ────────────────────────────────────────────────────────

def _largest_connected_component(surface):
    cf = vtk.vtkPolyDataConnectivityFilter()
    cf.SetInputData(surface)
    cf.SetExtractionModeToLargestRegion()
    cf.Update()
    out = pv.wrap(cf.GetOutput())
    log(f"  Largest CC     : {out.n_points} pts, {out.n_cells} cells")
    return out


def _polys_only(vtk_pd):
    """
    Rebuild a vtkPolyData keeping only 3-point polygon cells.

    vtkCleanPolyData can collapse triangles to lines/vertices; in VTK 9.6
    vtkTriangleFilter.PassLinesOff() does not remove them, so we rebuild
    from the polygon connectivity array, keeping only n==3 entries.
    """
    polys     = vtk_pd.GetPolys()
    pts       = vtk_pd.GetPoints()
    new_polys = vtk.vtkCellArray()
    id_list   = vtk.vtkIdList()
    polys.InitTraversal()
    while polys.GetNextCell(id_list):
        if id_list.GetNumberOfIds() == 3:
            new_polys.InsertNextCell(id_list)
    new_pd = vtk.vtkPolyData()
    new_pd.SetPoints(pts)
    new_pd.SetPolys(new_polys)
    return pv.wrap(new_pd)


def process_surface(surface, decimate=0.75, smooth_iter=50, smooth_band=0.05):
    """
    Decimate → Taubin smooth → MeshFix × 2 → consistent normals.

    The surface node positions after this step define the domain boundary.
    These nodes ARE allowed to move (smoothing, decimation); they are not
    required to have 1-to-1 voxel correspondence.
    """
    log("Processing surface ...")
    surface = _largest_connected_component(surface)
    surface = surface.triangulate()

    log(f"Decimating  (reduction {decimate:.0%}) ...")
    surface = surface.decimate(decimate, progress_bar=False)
    log(f"  Post-decimate  : {surface.n_points} pts, {surface.n_cells} cells")

    log(f"Smoothing   (Taubin n_iter={smooth_iter}, pass_band={smooth_band}) ...")
    surface = surface.smooth_taubin(
        n_iter=smooth_iter, pass_band=smooth_band,
        boundary_smoothing=False, normalize_coordinates=False)

    surface = surface.clean(tolerance=1e-6)
    surface = surface.fill_holes(hole_size=100).triangulate()

    log("Repairing self-intersections (MeshFix pass 1) ...")
    mf = pymeshfix.MeshFix(surface); mf.repair(); surface = mf.mesh
    log(f"  Post-repair 1  : {surface.n_points} pts, {surface.n_cells} cells")

    log("Merging near-coincident vertices (tol=0.1 mm) ...")
    cv = vtk.vtkCleanPolyData()
    cv.SetInputData(surface); cv.SetToleranceIsAbsolute(True)
    cv.SetAbsoluteTolerance(0.1); cv.Update()
    surface = _polys_only(cv.GetOutput())
    log(f"  Post-merge     : {surface.n_points} pts, {surface.n_cells} cells")

    log("Repairing self-intersections (MeshFix pass 2) ...")
    mf2 = pymeshfix.MeshFix(surface); mf2.repair(); surface = mf2.mesh
    log(f"  Post-repair 2  : {surface.n_points} pts, {surface.n_cells} cells")

    surface = surface.compute_normals(
        consistent_normals=True, auto_orient_normals=True,
        flip_normals=False, cell_normals=True, point_normals=True)

    log(f"  Final surface  : {surface.n_points} pts, {surface.n_cells} cells")
    _report_quality(surface)
    return surface


def _report_quality(surface):
    try:
        q = vtk.vtkMeshQuality()
        q.SetInputData(surface)
        q.SetTriangleQualityMeasureToAspectRatio(); q.Update()
        ar = vtk_to_numpy(q.GetOutput().GetCellData().GetArray('Quality'))
        log(f"  Aspect ratio   : min={ar.min():.3f}  mean={ar.mean():.3f}  "
            f"max={ar.max():.3f}")
        q.SetTriangleQualityMeasureToMinAngle(); q.Update()
        ma = vtk_to_numpy(q.GetOutput().GetCellData().GetArray('Quality'))
        log(f"  Min angle      : min={ma.min():.1f}°  mean={ma.mean():.1f}°")
    except Exception as e:
        log(f"  (quality report skipped: {e})")


# ── Combine surface + skeleton for TetGen ─────────────────────────────────────

def combine_for_tetgen(surface, skel_nodes):
    """
    Append skeleton nodes as orphan (cell-less) points to the surface PolyData.

    TetGen reads ALL points in the input mesh (including those not referenced
    by any face) and preserves them in its Delaunay triangulation.  The surface
    triangles define the domain boundary; the skeleton nodes are interior
    constraint points that MUST appear in the output mesh.

    Natural mesh grading arises from point density:
      • Dense skeleton nodes (~voxel spacing)  → small tets near vessels
      • Decimated surface nodes               → medium tets near boundary
      • Open parenchyma between features      → large Steiner-filled tets

    Returns
    -------
    combined    : pyvista PolyData  (n_surf + n_skel points, n_surf triangles)
    n_surf      : int  number of surface nodes (indices 0 … n_surf-1)
    n_skel      : int  number of skeleton nodes (indices n_surf … n_surf+n_skel-1)
    """
    n_surf = surface.n_points
    n_skel = len(skel_nodes)

    # Stack all points; surface.faces references only the first n_surf indices
    all_pts  = np.vstack([surface.points, skel_nodes]) if n_skel else surface.points
    combined = pv.PolyData(all_pts, surface.faces)

    log(f"  Input to TetGen : {combined.n_points} pts "
        f"({n_surf} surface + {n_skel} skeleton), "
        f"{combined.n_cells} boundary triangles")
    return combined, n_surf, n_skel


# ── TetGen volume mesh ────────────────────────────────────────────────────────

def generate_tet_mesh(combined, min_ratio=1.5, min_dihedral=20.0, max_vol=None):
    """
    Tetrahedralise the combined surface+skeleton mesh.

    All input points are preserved.  TetGen's quality refinement inserts
    Steiner points to meet the quality constraints, but never removes or
    moves existing input points.

    Parameters
    ----------
    combined     : pyvista PolyData from combine_for_tetgen()
    min_ratio    : TetGen -q radius/edge ratio quality bound
    min_dihedral : minimum dihedral angle [degrees]
    max_vol      : maximum tet volume [mm³]; None = unconstrained
                   Set this to limit total mesh size in open regions.
    """
    log("Running TetGen ...")
    tet    = tetgen.TetGen(combined)
    kwargs = dict(order=1, mindihedral=min_dihedral, minratio=min_ratio, verbose=1)
    if max_vol is not None:
        kwargs['maxvolume'] = float(max_vol)
    tet.tetrahedralize(**kwargs)
    vol = tet.grid
    log(f"  Tet mesh       : {vol.n_points} nodes, {vol.n_cells} elements")
    return vol


# ── Node-set labelling ────────────────────────────────────────────────────────

def label_nodesets(vol, surface_pts, skel_nodes):
    """
    Tag each node in the tet mesh with its origin via PointData 'node_set':
      0 = Steiner (inserted by TetGen for quality)
      1 = surface node
      2 = vessel skeleton node  (1-to-1 with skeleton voxels)

    Matching uses coordinate identity: an output node is classified as a
    surface or skeleton node when its distance to the nearest input node of
    that type is < 1e-3 mm (floating-point equality after round-trip through
    TetGen's double-precision arithmetic).
    """
    log("Labelling node sets ...")
    out_pts = vol.points
    tags    = np.zeros(len(out_pts), dtype=np.int8)

    if len(surface_pts):
        d, _ = KDTree(surface_pts).query(out_pts, workers=-1)
        tags[d < 1e-3] = 1

    if len(skel_nodes):
        d, _ = KDTree(skel_nodes).query(out_pts, workers=-1)
        tags[d < 1e-3] = 2      # skeleton overwrites surface if coincident

    arr = numpy_to_vtk(tags, deep=True, array_type=vtk.VTK_SIGNED_CHAR)
    arr.SetName('node_set')
    vol.GetPointData().AddArray(arr)

    log(f"  Surface nodes   : {(tags == 1).sum()}")
    log(f"  Skeleton nodes  : {(tags == 2).sum()}")
    log(f"  Steiner nodes   : {(tags == 0).sum()}")
    return vol


# ── Output ────────────────────────────────────────────────────────────────────

def write_outputs(surface, skel_nodes, vol, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Outer surface
    srf_path = os.path.join(out_dir, 'liver_surface.vtp')
    surface.save(srf_path)
    log(f"Wrote surface      : {srf_path}")

    # Vessel skeleton point cloud  (1-to-1 with skeleton voxels)
    if len(skel_nodes):
        skel_pd = pv.PolyData(skel_nodes)
        node_ids = numpy_to_vtk(np.arange(len(skel_nodes), dtype=np.int32),
                                deep=True, array_type=vtk.VTK_INT)
        node_ids.SetName('skeleton_node_id')
        skel_pd.GetPointData().AddArray(node_ids)
        skel_path = os.path.join(out_dir, 'vessel_skeleton.vtp')
        skel_pd.save(skel_path)
        log(f"Wrote skeleton     : {skel_path}")

    # Tetrahedral volume mesh
    tet_path = os.path.join(out_dir, 'liver_tet.vtu')
    vol.save(tet_path)
    log(f"Wrote tet mesh     : {tet_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='3D tet mesh with vessel skeleton + surface constraints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    ap.add_argument('nii',
                    help='Label NIfTI (.nii or .nii.gz)')
    ap.add_argument('--vessel-label', type=int,   default=2,    metavar='INT',
                    help='Label value for vessel skeleton extraction (default 2)')
    ap.add_argument('--no-skeleton',  action='store_true',
                    help='Skip vessel skeleton extraction')
    ap.add_argument('--no-fill',      action='store_true',
                    help='Skip binary hole-filling of the outer surface mask')
    ap.add_argument('--close-iters',  type=int,   default=1,    metavar='N',
                    help='Morphological closing iterations for surface (default 1)')
    ap.add_argument('--decimate',     type=float, default=0.75, metavar='R',
                    help='Surface decimation ratio 0–1 (default 0.75 = keep 25%%)')
    ap.add_argument('--smooth-iter',  type=int,   default=50,   metavar='N',
                    help='Taubin smoothing iterations for surface (default 50)')
    ap.add_argument('--smooth-band',  type=float, default=0.05, metavar='F',
                    help='Taubin pass-band frequency (default 0.05)')
    ap.add_argument('--tet-quality',  type=float, default=1.5,  metavar='R',
                    help='TetGen min radius/edge ratio (default 1.5)')
    ap.add_argument('--tet-dihedral', type=float, default=20.0, metavar='DEG',
                    help='TetGen min dihedral angle [deg] (default 20)')
    ap.add_argument('--tet-maxvol',   type=float, default=None, metavar='MM3',
                    help='TetGen max tet volume [mm³]; limits mesh size in '
                         'open parenchyma (unconstrained by default)')
    ap.add_argument('--output-dir',   default='.',              metavar='DIR',
                    help='Output directory (default: .)')

    args     = ap.parse_args()
    nii_path = os.path.abspath(args.nii)
    if not os.path.exists(nii_path):
        sys.exit(f"Error: {nii_path} not found")

    # ── 1. Load ──────────────────────────────────────────────────────────────
    data, affine = load_label_nii(nii_path)

    # ── 2. Vessel skeleton (unmodified pixel positions) ───────────────────────
    if args.no_skeleton:
        skel_nodes = np.zeros((0, 3), dtype=np.float64)
        log("Skipping vessel skeleton (--no-skeleton)")
    else:
        skel_nodes, _ = vessel_skeleton_nodes(data, affine,
                                               vessel_label=args.vessel_label)

    # ── 3. Outer surface (may be decimated/smoothed) ──────────────────────────
    surface_raw = extract_surface(
        data, affine,
        fill=not args.no_fill,
        close_iters=args.close_iters,
    )
    surface = process_surface(
        surface_raw,
        decimate=args.decimate,
        smooth_iter=args.smooth_iter,
        smooth_band=args.smooth_band,
    )

    # ── 4. Combine surface + skeleton for TetGen ──────────────────────────────
    log("Combining surface and skeleton node sets ...")
    combined, n_surf, n_skel = combine_for_tetgen(surface, skel_nodes)

    # ── 5. Tetrahedral mesh ───────────────────────────────────────────────────
    vol = generate_tet_mesh(
        combined,
        min_ratio=args.tet_quality,
        min_dihedral=args.tet_dihedral,
        max_vol=args.tet_maxvol,
    )

    # ── 6. Tag node sets in output mesh ──────────────────────────────────────
    vol = label_nodesets(vol, surface.points, skel_nodes)

    # ── 7. Write outputs ──────────────────────────────────────────────────────
    write_outputs(surface, skel_nodes, vol, args.output_dir)
