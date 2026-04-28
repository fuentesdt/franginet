"""
meshliver.py — 3D tetrahedral mesh from a labeled NIfTI volume.

Extracts the outer surface of all labeled voxels (label > 0), decimates
and smooths it, repairs self-intersections, then tetrahedralises with TetGen.

Pipeline
--------
  label > 0 binary mask
    → fill enclosed cavities  (binary_fill_holes)
    → morphological closing   (optional)
    → marching cubes surface  (voxel space → world mm via NIfTI affine)
    → largest connected component
    → quadric decimation      (Quadric Error Metrics)
    → Taubin windowed-sinc smoothing  (volume-preserving vs Laplacian)
    → MeshFix × 2            (self-intersection / non-manifold repair)
    → absolute-tolerance point merge  (removes near-degenerate triangles)
    → consistent outward normals
    → TetGen tetrahedral volume mesh
  Outputs:
    <out_dir>/surface.vtp     closed triangulated surface
    <out_dir>/mesh.vtu         linear tetrahedral volume mesh

Usage
-----
  python meshliver.py prepost/PreTxArtLoRes.vessellabel.nii.gz
  python meshliver.py prepost/PreTxArtLoRes.vessellabel.nii.gz \\
      --decimate 0.75 --smooth-iter 50 \\
      --tet-quality 1.5 --tet-dihedral 20 --tet-maxvol 500 \\
      --output-dir newdata

Dependencies
------------
  pip install nibabel pyvista tetgen pymeshfix numpy scipy
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import pyvista as pv
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import tetgen
import pymeshfix
from scipy.ndimage import binary_fill_holes, binary_closing


def log(msg):
    print(f"[meshliver] {msg}", flush=True)


# ── NIfTI loading ─────────────────────────────────────────────────────────────

def load_nii(path):
    log(f"Loading  : {path}")
    nii     = nib.load(path)
    affine  = nii.affine.astype(np.float64)
    data    = np.asarray(nii.dataobj)
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    log(f"  Shape   : {data.shape}")
    log(f"  Labels  : {np.unique(data).tolist()}")
    log(f"  Spacing : {spacing.round(3).tolist()} mm")
    return data, affine


# ── Surface extraction ────────────────────────────────────────────────────────

def extract_surface(data, affine, fill=True, close_iters=1):
    """
    Marching cubes surface of all labeled voxels (label > 0) in world mm.
    fill=True removes enclosed cavities before surface extraction.
    """
    log("Building binary mask (label > 0) ...")
    binary = (data > 0).astype(np.uint8)
    log(f"  Labeled voxels : {int(binary.sum())}")

    if fill:
        binary = binary_fill_holes(binary).astype(np.uint8)
        log(f"  After fill     : {int(binary.sum())} voxels")

    if close_iters > 0:
        binary = binary_closing(binary, iterations=close_iters).astype(np.uint8)
        log(f"  After closing  : {int(binary.sum())} voxels")

    log("Extracting surface (marching cubes) ...")
    # Build in voxel-index space (spacing=1, origin=0); apply affine afterward
    # so any anisotropic spacing, axis flips, or rotation are handled uniformly.
    grid            = pv.ImageData()
    grid.dimensions = binary.shape
    grid.spacing    = (1.0, 1.0, 1.0)
    grid.origin     = (0.0, 0.0, 0.0)
    grid.point_data['mask'] = binary.flatten(order='F').astype(np.float32)

    surface = grid.contour([0.5], scalars='mask', method='marching_cubes')
    log(f"  Raw surface    : {surface.n_points} pts, {surface.n_cells} cells")
    if surface.n_points == 0:
        sys.exit("Error: marching cubes produced an empty surface")

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

def _largest_cc(surface):
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

    vtkCleanPolyData can collapse triangles into degenerate line/vertex cells.
    In VTK 9.6 vtkTriangleFilter.PassLinesOff() does not remove them, so we
    reconstruct the PolyData from the polygon array filtering to n_pts == 3.
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
    Produce a TetGen-ready closed triangulated surface.

    Steps
    -----
    1. Keep largest connected component (drop satellite fragments).
    2. Quadric Error Metric decimation.
    3. Taubin windowed-sinc smoothing — suppresses volume shrinkage vs Laplacian.
    4. MeshFix pass 1 — resolve self-intersections / non-manifold edges.
    5. Absolute-tolerance point merge (0.1 mm) — collapse near-degenerate
       triangles whose vertices are sub-mm apart; rebuild polys-only mesh to
       strip degenerate line/vertex cells left by vtkCleanPolyData (VTK 9.6).
    6. MeshFix pass 2 — repair any new crossings introduced by the merge.
    7. Consistent outward normals required by TetGen.
    """
    log("Processing surface ...")
    surface = _largest_cc(surface)
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
    mf = pymeshfix.MeshFix(surface)
    mf.repair()
    surface = mf.mesh
    log(f"  Post-repair 1  : {surface.n_points} pts, {surface.n_cells} cells")

    log("Merging near-coincident vertices (tol=0.1 mm) ...")
    cv = vtk.vtkCleanPolyData()
    cv.SetInputData(surface)
    cv.SetToleranceIsAbsolute(True)
    cv.SetAbsoluteTolerance(0.1)
    cv.Update()
    surface = _polys_only(cv.GetOutput())
    log(f"  Post-merge     : {surface.n_points} pts, {surface.n_cells} cells")

    log("Repairing self-intersections (MeshFix pass 2) ...")
    mf2 = pymeshfix.MeshFix(surface)
    mf2.repair()
    surface = mf2.mesh
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
        q.SetTriangleQualityMeasureToAspectRatio()
        q.Update()
        ar = vtk_to_numpy(q.GetOutput().GetCellData().GetArray('Quality'))
        log(f"  Aspect ratio   : min={ar.min():.3f}  mean={ar.mean():.3f}  "
            f"max={ar.max():.3f}  (1.0 = equilateral)")
        q.SetTriangleQualityMeasureToMinAngle()
        q.Update()
        ma = vtk_to_numpy(q.GetOutput().GetCellData().GetArray('Quality'))
        log(f"  Min angle      : min={ma.min():.1f}°  mean={ma.mean():.1f}°")
    except Exception as e:
        log(f"  (quality report skipped: {e})")


# ── TetGen volume mesh ────────────────────────────────────────────────────────

def generate_tet_mesh(surface, min_ratio=1.5, min_dihedral=20.0, max_vol=None):
    """
    Tetrahedralise the closed surface with TetGen.

    Parameters
    ----------
    min_ratio    : radius/edge-length quality bound (-q)
    min_dihedral : minimum dihedral angle [degrees]
    max_vol      : maximum tet volume [mm³]; None = unconstrained
    """
    log("Running TetGen ...")
    tet    = tetgen.TetGen(surface)
    kwargs = dict(order=1, mindihedral=min_dihedral, minratio=min_ratio, verbose=1)
    if max_vol is not None:
        kwargs['maxvolume'] = float(max_vol)
    tet.tetrahedralize(**kwargs)
    vol = tet.grid
    log(f"  Tet mesh       : {vol.n_points} nodes, {vol.n_cells} elements")
    return vol


# ── Output ────────────────────────────────────────────────────────────────────

def write_outputs(surface, vol, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    srf_path = os.path.join(out_dir, 'surface.vtp')
    tet_path = os.path.join(out_dir, 'mesh.vtu')
    surface.save(srf_path)
    log(f"Wrote surface  : {srf_path}")
    vol.save(tet_path)
    log(f"Wrote tet mesh : {tet_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='3D tet mesh from a labeled NIfTI volume (label > 0)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    ap.add_argument('nii',
                    help='Label NIfTI (.nii or .nii.gz)')
    ap.add_argument('--no-fill',      action='store_true',
                    help='Skip binary hole-filling of the label mask')
    ap.add_argument('--close-iters',  type=int,   default=1,    metavar='N',
                    help='Morphological closing iterations (default 1)')
    ap.add_argument('--decimate',     type=float, default=0.75, metavar='R',
                    help='Decimation reduction ratio 0–1 (default 0.75 = keep 25%%)')
    ap.add_argument('--smooth-iter',  type=int,   default=50,   metavar='N',
                    help='Taubin smoothing iterations (default 50)')
    ap.add_argument('--smooth-band',  type=float, default=0.05, metavar='F',
                    help='Taubin pass-band frequency (default 0.05)')
    ap.add_argument('--tet-quality',  type=float, default=1.5,  metavar='R',
                    help='TetGen min radius/edge ratio (default 1.5)')
    ap.add_argument('--tet-dihedral', type=float, default=20.0, metavar='DEG',
                    help='TetGen min dihedral angle [deg] (default 20)')
    ap.add_argument('--tet-maxvol',   type=float, default=None, metavar='MM3',
                    help='TetGen max tet volume [mm³] (unconstrained by default)')
    ap.add_argument('--output-dir',   default='.',              metavar='DIR',
                    help='Output directory (default: .)')

    args     = ap.parse_args()
    nii_path = os.path.abspath(args.nii)
    if not os.path.exists(nii_path):
        sys.exit(f"Error: {nii_path} not found")

    data, affine = load_nii(nii_path)

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

    vol = generate_tet_mesh(
        surface,
        min_ratio=args.tet_quality,
        min_dihedral=args.tet_dihedral,
        max_vol=args.tet_maxvol,
    )

    write_outputs(surface, vol, args.output_dir)
