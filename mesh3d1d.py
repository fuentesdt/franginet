"""
mesh3d1d.py — Generate a 3D tetrahedral mesh from a labeled NIfTI volume.

Label conventions (default)
---------------------------
  1 = liver   → outer surface extracted and tetrahedralised
  2 = vessel  → vessel lumens inside liver are filled before surface
                extraction so the output surface is the outer liver boundary

Pipeline
--------
  NIfTI label
    → binary liver mask  (fill enclosed cavities, optional morphological closing)
    → marching cubes surface  (voxel space → world mm via NIfTI affine)
    → largest connected component
    → quadric-decimation
    → Taubin windowed-sinc smoothing  (preserves volume vs. Laplacian)
    → mesh repair  (clean, fill holes, triangulate, consistent normals)
    → TetGen tetrahedral volume mesh
  Outputs: liver_surface.vtp   closed triangulated surface
           liver_tet.vtu        linear tetrahedral volume mesh

Usage
-----
  python mesh3d1d.py prepost/PreTxArtLoRes.vessellabel.nii.gz
  python mesh3d1d.py prepost/PreTxArtLoRes.vessellabel.nii.gz \\
      --liver-label 1 --decimate 0.9 --smooth-iter 50 \\
      --tet-quality 1.5 --tet-dihedral 20 --output-dir newdata

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


# ── Surface extraction ────────────────────────────────────────────────────────

def extract_surface(data, affine, liver_label=1, fill=True, close_iters=1):
    """
    Marching cubes surface of the liver label in world (mm) coordinates.

    fill=True fills enclosed cavities (vessel lumens, bile ducts) so the
    result is the outer liver boundary only.  Morphological closing bridges
    small segmentation gaps before surface extraction.
    """
    log(f"Building binary liver mask (label={liver_label}) ...")
    binary = (data == liver_label).astype(np.uint8)
    log(f"  Liver voxels  : {int(binary.sum())}")

    if fill:
        binary = binary_fill_holes(binary).astype(np.uint8)
        log(f"  After fill    : {int(binary.sum())} voxels")

    if close_iters > 0:
        binary = binary_closing(binary, iterations=close_iters).astype(np.uint8)
        log(f"  After closing : {int(binary.sum())} voxels")

    log("Extracting surface (marching cubes) ...")

    # Build pyvista ImageData in voxel-index space (spacing=1, origin=0).
    # The NIfTI affine (which may include anisotropic spacing, flips, or
    # rotation) is applied to the extracted points afterward, ensuring
    # correct world-coordinate positions regardless of acquisition geometry.
    grid            = pv.ImageData()
    grid.dimensions = binary.shape          # point-centred data: Nx × Ny × Nz
    grid.spacing    = (1.0, 1.0, 1.0)
    grid.origin     = (0.0, 0.0, 0.0)
    # VTK stores arrays in Fortran (column-major) order
    grid.point_data['mask'] = binary.flatten(order='F').astype(np.float32)

    surface = grid.contour([0.5], scalars='mask', method='marching_cubes')
    log(f"  Raw surface   : {surface.n_points} pts, {surface.n_cells} cells")
    if surface.n_points == 0:
        sys.exit("Error: marching cubes produced an empty surface — check liver_label")

    # Transform voxel-space points → world mm via the NIfTI affine
    pts   = surface.points                        # N × 3  (i, j, k) float
    ones  = np.ones((len(pts), 1))
    world = (affine @ np.hstack([pts, ones]).T).T[:, :3]
    surface = surface.copy()
    surface.points = world

    bb = surface.bounds   # (xmin, xmax, ymin, ymax, zmin, zmax)
    log(f"  BBox x : {bb[0]:.1f} – {bb[1]:.1f} mm")
    log(f"  BBox y : {bb[2]:.1f} – {bb[3]:.1f} mm")
    log(f"  BBox z : {bb[4]:.1f} – {bb[5]:.1f} mm")

    return surface


# ── Surface processing ────────────────────────────────────────────────────────

def _largest_connected_component(surface):
    """Use vtkPolyDataConnectivityFilter to keep the largest region."""
    cf = vtk.vtkPolyDataConnectivityFilter()
    cf.SetInputData(surface)
    cf.SetExtractionModeToLargestRegion()
    cf.Update()
    out = pv.wrap(cf.GetOutput())
    log(f"  Largest CC    : {out.n_points} pts, {out.n_cells} cells")
    return out


def _polys_only(vtk_pd):
    """
    Rebuild a vtkPolyData keeping only 3-point polygon cells.

    vtkCleanPolyData collapses near-coincident points; when a triangle is
    reduced to 2 or 1 unique vertices it becomes a line/vertex cell stored in
    GetLines() / GetVerts().  In VTK 9.6 vtkTriangleFilter.PassLinesOff()
    does not remove these, so we reconstruct the PolyData from scratch using
    only the polygon connectivity array and filtering to n==3.
    """
    polys = vtk_pd.GetPolys()
    pts   = vtk_pd.GetPoints()
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
    Decimate → smooth → repair surface mesh for TetGen compatibility.

    Taubin windowed-sinc smoothing is used instead of plain Laplacian
    to suppress volume shrinkage while regularising triangle shape.
    The repair sequence (clean → fill_holes → triangulate → normals)
    ensures a closed, manifold, consistently-oriented triangle mesh.
    """
    log("Processing surface ...")

    # 1. Largest connected component — discard satellite fragments
    surface = _largest_connected_component(surface)

    # 2. All-triangle cells required before decimation
    surface = surface.triangulate()

    # 3. Decimate with Quadric Error Metrics (better shape than DecimatePro)
    log(f"Decimating  (target reduction {decimate:.0%}) ...")
    surface = surface.decimate(decimate, progress_bar=False)
    log(f"  Post-decimate : {surface.n_points} pts, {surface.n_cells} cells")

    # 4. Taubin windowed-sinc smoothing
    log(f"Smoothing   (Taubin n_iter={smooth_iter}, pass_band={smooth_band}) ...")
    surface = surface.smooth_taubin(
        n_iter=smooth_iter,
        pass_band=smooth_band,
        boundary_smoothing=False,
        normalize_coordinates=False,
    )

    # 5. Clean: merge coincident points, remove degenerate (zero-area) cells
    surface = surface.clean(tolerance=1e-6)

    # 6. Fill small holes that can arise from decimation artefacts
    surface = surface.fill_holes(hole_size=100)

    # 7. Re-triangulate (fill_holes may emit quads)
    surface = surface.triangulate()

    # 8. Repair self-intersections and non-manifold edges with MeshFix.
    #    Decimation + smoothing can leave crossing triangles that cause TetGen
    #    to abort; MeshFix resolves them while keeping the surface closed.
    log("Repairing self-intersections (MeshFix) ...")
    mf = pymeshfix.MeshFix(surface)
    mf.repair()
    surface = mf.mesh
    log(f"  Post-repair   : {surface.n_points} pts, {surface.n_cells} cells")

    # 9. Absolute-tolerance point merge to remove near-degenerate triangles.
    #    vtkCleanPolyData collapses nearly-coincident vertices; triangles whose
    #    points merge to <3 unique vertices become degenerate line/vertex cells.
    #    vtkTriangleFilter.PassLinesOff() does not remove them in VTK 9.6, so
    #    we rebuild the PolyData from the polygon array keeping only 3-point
    #    cells.  A second MeshFix pass then resolves any self-intersections
    #    introduced when collapsing degenerate faces created crossing triangles.
    log("Merging near-coincident vertices (tol=0.1 mm) ...")
    clean_vtk = vtk.vtkCleanPolyData()
    clean_vtk.SetInputData(surface)
    clean_vtk.SetToleranceIsAbsolute(True)
    clean_vtk.SetAbsoluteTolerance(0.1)
    clean_vtk.Update()
    surface = _polys_only(clean_vtk.GetOutput())
    log(f"  Post-merge    : {surface.n_points} pts, {surface.n_cells} cells")

    log("Repairing remaining self-intersections (MeshFix pass 2) ...")
    mf2 = pymeshfix.MeshFix(surface)
    mf2.repair()
    surface = mf2.mesh
    log(f"  Post-repair 2 : {surface.n_points} pts, {surface.n_cells} cells")

    # 10. Consistent outward-facing normals — TetGen requires a correctly
    #    oriented closed surface to determine interior vs exterior
    surface = surface.compute_normals(
        consistent_normals=True,
        auto_orient_normals=True,
        flip_normals=False,
        cell_normals=True,
        point_normals=True,
    )

    log(f"  Final surface : {surface.n_points} pts, {surface.n_cells} cells")
    _report_quality(surface)
    return surface


def _report_quality(surface):
    """Log triangle aspect-ratio and minimum-angle statistics via vtkMeshQuality."""
    try:
        q = vtk.vtkMeshQuality()
        q.SetInputData(surface)

        q.SetTriangleQualityMeasureToAspectRatio()
        q.Update()
        ar = vtk_to_numpy(q.GetOutput().GetCellData().GetArray('Quality'))
        log(f"  Aspect ratio  : min={ar.min():.3f}  mean={ar.mean():.3f}  "
            f"max={ar.max():.3f}  (1.0 = equilateral)")

        q.SetTriangleQualityMeasureToMinAngle()
        q.Update()
        ma = vtk_to_numpy(q.GetOutput().GetCellData().GetArray('Quality'))
        log(f"  Min angle     : min={ma.min():.1f}°  mean={ma.mean():.1f}°")
    except Exception as e:
        log(f"  (quality report skipped: {e})")


# ── TetGen volume mesh ────────────────────────────────────────────────────────

def generate_tet_mesh(surface, min_ratio=1.5, min_dihedral=20.0, max_vol=None):
    """
    Tetrahedralise the closed liver surface using TetGen.

    Parameters
    ----------
    surface      : pyvista PolyData — closed, triangulated, manifold
    min_ratio    : float — TetGen -q radius/edge-length quality bound
    min_dihedral : float — minimum dihedral angle [degrees]
    max_vol      : float or None — max tet volume [mm³]; None = unconstrained

    Returns
    -------
    pyvista UnstructuredGrid of linear tetrahedra
    """
    log("Running TetGen ...")
    tet    = tetgen.TetGen(surface)
    kwargs = dict(order=1, mindihedral=min_dihedral, minratio=min_ratio, verbose=1)
    if max_vol is not None:
        kwargs['maxvolume'] = float(max_vol)
    tet.tetrahedralize(**kwargs)
    vol = tet.grid
    log(f"  Tet mesh      : {vol.n_points} nodes, {vol.n_cells} elements")
    return vol


# ── Output ────────────────────────────────────────────────────────────────────

def write_outputs(surface, vol, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    srf_path = os.path.join(out_dir, 'liver_surface.vtp')
    tet_path = os.path.join(out_dir, 'liver_tet.vtu')
    surface.save(srf_path)
    log(f"Wrote surface : {srf_path}")
    vol.save(tet_path)
    log(f"Wrote tet mesh: {tet_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='3D tetrahedral mesh from a labeled NIfTI volume',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    ap.add_argument('nii',
                    help='Label NIfTI (.nii or .nii.gz)')
    ap.add_argument('--liver-label',  type=int,   default=1,    metavar='INT',
                    help='Label value for liver voxels (default 1)')
    ap.add_argument('--vessel-label', type=int,   default=2,    metavar='INT',
                    help='Label value for vessel voxels (default 2; '
                         'informational — vessel lumens are filled implicitly '
                         'by --fill)')
    ap.add_argument('--no-fill',      action='store_true',
                    help='Skip binary hole-filling of the liver mask')
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

    args    = ap.parse_args()
    nii_path = os.path.abspath(args.nii)
    if not os.path.exists(nii_path):
        sys.exit(f"Error: {nii_path} not found")

    data, affine = load_label_nii(nii_path)

    surface_raw = extract_surface(
        data, affine,
        liver_label=args.liver_label,
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
