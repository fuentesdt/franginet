"""
convertparaview.py — Convert pipeline outputs to ParaView-readable formats.

Sub-commands
------------
vti   Convert a NIfTI label volume → VTK ImageData (.vti)
vtp   Convert resistance_graph.mat → VTK PolyData (.vtp) 1-D centerline
      with nodal pressure and per-edge radius / flow / conductance.

Usage
-----
    python convertparaview.py vti <input.nii[.gz]> [output.vti]
    python convertparaview.py vtp <resistance_graph.mat> [output.vtp]

If output path is omitted the file is written next to the input.

Dependencies
------------
    pip install nibabel vtk scipy
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def log(msg):
    print(f"[convertparaview] {msg}", flush=True)


# ---------------------------------------------------------------------------
def nifti_to_vtkImageData(nii_path):
    """
    Load a NIfTI file and return a vtkImageData in the NIfTI world coordinate
    system.  Origin, spacing, and direction cosines are taken from the affine.

    NIfTI affine convention (nibabel):
        world_xyz = affine[:3,:3] @ ijk + affine[:3,3]
    where ijk are 0-indexed voxel coordinates.

    VTK ImageData convention:
        world_xyz = D @ diag(spacing) @ ijk + origin
    where D is the 3×3 direction matrix whose columns are unit world vectors
    for each voxel axis.
    """
    nii      = nib.load(nii_path)
    affine   = nii.affine.astype(np.float64)          # 4×4
    data     = np.array(nii.dataobj)                  # writable copy

    log(f"  Shape   : {data.shape}  dtype: {data.dtype}")
    log(f"  Affine  :\n{affine.round(4)}")

    vox_vecs = affine[:3, :3]                                    # col j = world step for axis j
    spacing  = np.sqrt(np.sum(vox_vecs ** 2, axis=0))           # (3,) always positive
    origin   = affine[:3, 3].copy()                             # world pos of voxel (0,0,0)

    # VTK ImageData only supports positive spacing; it has no concept of a
    # flipped axis unless SetDirectionMatrix (VTK 9+) is used — and even then
    # ParaView does not always apply it when loading .vti files.
    #
    # For diagonal affines (standard CT/MRI): if a voxel step points in the
    # negative world direction, flip the data along that axis and move the
    # origin to the low-coordinate corner.  After this the spacing is positive
    # and no direction matrix is needed.
    for axis in range(3):
        if vox_vecs[axis, axis] < 0:                  # step is negative along this axis
            data   = np.flip(data, axis=axis).copy()
            origin = origin + vox_vecs[:, axis] * (data.shape[axis] - 1)

    bb_max = origin + spacing * (np.array(data.shape[:3]) - 1)
    log(f"  Origin  : {origin.round(3).tolist()}")
    log(f"  Spacing : {spacing.round(4).tolist()}")
    log(f"  BBox x  : {origin[0]:.2f} – {bb_max[0]:.2f} mm")
    log(f"  BBox y  : {origin[1]:.2f} – {bb_max[1]:.2f} mm")
    log(f"  BBox z  : {origin[2]:.2f} – {bb_max[2]:.2f} mm")

    # ── Build vtkImageData ──────────────────────────────────────────────────
    img = vtk.vtkImageData()
    nx, ny, nz = data.shape[:3]
    img.SetDimensions(nx, ny, nz)
    img.SetOrigin(origin.tolist())
    img.SetSpacing(spacing.tolist())

    # ── Scalar array ────────────────────────────────────────────────────────
    # Flatten in Fortran (column-major) order: first index (x) varies fastest,
    # which matches VTK's memory layout for vtkImageData point arrays.
    flat = data.flatten(order='F')

    dtype_to_vtk = {
        np.dtype('uint8'):   vtk.VTK_UNSIGNED_CHAR,
        np.dtype('int8'):    vtk.VTK_SIGNED_CHAR,
        np.dtype('uint16'):  vtk.VTK_UNSIGNED_SHORT,
        np.dtype('int16'):   vtk.VTK_SHORT,
        np.dtype('uint32'):  vtk.VTK_UNSIGNED_INT,
        np.dtype('int32'):   vtk.VTK_INT,
        np.dtype('float32'): vtk.VTK_FLOAT,
        np.dtype('float64'): vtk.VTK_DOUBLE,
    }
    vtk_type = dtype_to_vtk.get(flat.dtype, vtk.VTK_FLOAT)

    scalars = numpy_to_vtk(flat, deep=True, array_type=vtk_type)
    scalars.SetName('label')
    img.GetPointData().SetScalars(scalars)

    return img


# ---------------------------------------------------------------------------
def convert(nii_path, vti_path):
    log(f"Loading  : {nii_path}")
    img = nifti_to_vtkImageData(nii_path)
    log(f"Dims     : {img.GetDimensions()}")

    log(f"Writing  : {vti_path}")
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(vti_path)
    writer.SetInputData(img)
    writer.SetDataModeToAscii()     # plain-text XML — same as extract_centerline.py
    writer.Write()
    log(f"Done.")


# ---------------------------------------------------------------------------
def mat_to_vtp(mat_path, vtp_path, skel_path=None):
    """
    Load resistance_graph.mat (saved by resistanceLumping.m) and write a
    VTP polydata file of the 1-D resistance network for ParaView.

    Geometry
    --------
    Points = graph nodes (mm, world coordinates)
    Lines  = real edges  +  phantom gap-bridging edges

    If skel_path is given the node world coordinates are recomputed from the
    stored 0-indexed voxel indices (node_ijk0) via the NIfTI affine so that
    every VTP node lands exactly on the corresponding skeleton voxel centre
    when both files are loaded in ParaView.

    PointData
    ---------
    pressure_mmhg   — nodal pressure from the linear solve

    CellData (per edge/line cell)
    ---------
    radius_mm       — mean vessel radius along the edge
    length_mm       — path length along the skeleton
    flow_mm3s       — volumetric flow Q = G*(p_i - p_j)
    conductance_SI  — Hagen-Poiseuille conductance [m³ Pa⁻¹ s⁻¹]
    is_phantom      — 0 = real edge, 1 = gap-bridging phantom edge
    """
    import scipy.io as sio

    log(f"Loading  : {mat_path}")
    mat = sio.loadmat(mat_path, struct_as_record=False)
    r   = mat['results'].flat[0]          # unwrap the MATLAB struct

    def col(field, dtype=np.float64):
        """Extract a struct field as a flat 1-D numpy array."""
        return np.atleast_1d(np.asarray(getattr(r, field), dtype=dtype)).flatten()

    nodes    = np.asarray(r.nodes,       dtype=np.float64)   # N×3 mm
    edges    = np.asarray(r.edges,       dtype=np.int64)     # E×2 (1-indexed)
    if edges.ndim == 1:                   # single-edge corner case after loadmat
        edges = edges.reshape(1, -1)

    # Recompute node world coords from skeleton voxel indices + NIfTI affine
    # so every VTP point lands exactly on the corresponding voxel centre.
    if skel_path is not None:
        log(f"Aligning nodes to skeleton: {skel_path}")
        skel_nii = nib.load(skel_path)
        affine   = skel_nii.affine.astype(np.float64)
        ijk0 = np.asarray(r.node_ijk0, dtype=np.float64)   # N×3, 0-indexed
        ones = np.ones((ijk0.shape[0], 1), dtype=np.float64)
        nodes = (affine @ np.hstack([ijk0, ones]).T).T[:, :3]
        log(f"  Recomputed {nodes.shape[0]} node positions from nibabel affine")

    pressure = col('pressure_mmhg')       # N
    radii    = col('radii_mm')            # E
    lengths  = col('lengths_mm')          # E
    flows    = col('flow_mm3s')           # E
    conds    = col('conductances_SI')     # E

    ph_ni = col('phantom_ni',   np.int64)
    ph_nj = col('phantom_nj',   np.int64)
    ph_r  = col('phantom_radii_mm')
    ph_L  = col('phantom_lengths_mm')
    # empty phantom arrays arrive as [[]] after loadmat — normalise
    if ph_ni.size == 1 and ph_ni[0] == 0 and ph_nj.size == 1:
        ph_ni = np.empty(0, dtype=np.int64)
        ph_nj = np.empty(0, dtype=np.int64)
        ph_r  = np.empty(0)
        ph_L  = np.empty(0)

    n_nodes   = nodes.shape[0]
    n_edges   = edges.shape[0]
    n_phantom = len(ph_ni)

    log(f"  Nodes: {n_nodes}  Real edges: {n_edges}  Phantom: {n_phantom}")
    bb_min = nodes.min(axis=0)
    bb_max = nodes.max(axis=0)
    log(f"  BBox x : {bb_min[0]:.2f} – {bb_max[0]:.2f} mm")
    log(f"  BBox y : {bb_min[1]:.2f} – {bb_max[1]:.2f} mm")
    log(f"  BBox z : {bb_min[2]:.2f} – {bb_max[2]:.2f} mm")
    log(f"  Pressure : {pressure.min():.1f} – {pressure.max():.1f} mmHg")
    if n_edges:
        log(f"  |Flow|   : {np.abs(flows).min():.3e} – {np.abs(flows).max():.3e} mm³/s")

    # ── vtkPolyData ──────────────────────────────────────────────────────────
    pd = vtk.vtkPolyData()

    # Points
    pts = vtk.vtkPoints()
    pts.SetDataTypeToDouble()
    for i in range(n_nodes):
        pts.InsertNextPoint(float(nodes[i, 0]), float(nodes[i, 1]), float(nodes[i, 2]))
    pd.SetPoints(pts)

    # Lines: real edges then phantom edges (both 1-indexed → 0-indexed)
    lines = vtk.vtkCellArray()
    for k in range(n_edges):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(edges[k, 0]) - 1)
        line.GetPointIds().SetId(1, int(edges[k, 1]) - 1)
        lines.InsertNextCell(line)
    for k in range(n_phantom):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, int(ph_ni[k]) - 1)
        line.GetPointIds().SetId(1, int(ph_nj[k]) - 1)
        lines.InsertNextCell(line)
    pd.SetLines(lines)

    # ── PointData: nodal pressure ────────────────────────────────────────────
    pArr = numpy_to_vtk(pressure, deep=True, array_type=vtk.VTK_DOUBLE)
    pArr.SetName('pressure_mmhg')
    pd.GetPointData().SetScalars(pArr)

    # ── CellData: per-edge attributes ────────────────────────────────────────
    def cell_arr(real_data, phantom_data, name):
        v = numpy_to_vtk(
            np.concatenate([real_data, phantom_data]).astype(np.float64),
            deep=True, array_type=vtk.VTK_DOUBLE)
        v.SetName(name)
        return v

    nan_p  = np.full(n_phantom, np.nan)

    pd.GetCellData().AddArray(cell_arr(radii,  ph_r,  'radius_mm'))
    pd.GetCellData().AddArray(cell_arr(lengths, ph_L, 'length_mm'))
    pd.GetCellData().AddArray(cell_arr(flows,   nan_p, 'flow_mm3s'))
    pd.GetCellData().AddArray(cell_arr(conds,   nan_p, 'conductance_SI'))
    pd.GetCellData().SetActiveScalars('radius_mm')

    is_ph = np.array([0] * n_edges + [1] * n_phantom, dtype=np.int8)
    phArr = numpy_to_vtk(is_ph, deep=True, array_type=vtk.VTK_SIGNED_CHAR)
    phArr.SetName('is_phantom')
    pd.GetCellData().AddArray(phArr)

    # ── Write VTP ────────────────────────────────────────────────────────────
    log(f"Writing  : {vtp_path}")
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(vtp_path)
    w.SetInputData(pd)
    w.SetDataModeToAscii()    # plain-text XML — avoids binary encoding issues
    w.Write()
    log(f"Done.")


# ---------------------------------------------------------------------------
def _stem(path, suffixes):
    """Strip any of the given suffixes from path."""
    for s in suffixes:
        if path.endswith(s):
            return path[:-len(s)]
    return path


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Convert pipeline outputs to ParaView formats')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p_vti = sub.add_parser('vti', help='NIfTI → VTI image data')
    p_vti.add_argument('nii_path', help='Input .nii or .nii.gz')
    p_vti.add_argument('vti_path', nargs='?', help='Output .vti (default: beside input)')

    p_vtp = sub.add_parser('vtp', help='resistance_graph.mat → VTP centerline polydata')
    p_vtp.add_argument('mat_path', help='Input resistance_graph.mat')
    p_vtp.add_argument('vtp_path', nargs='?', help='Output .vtp (default: beside input)')
    p_vtp.add_argument('--skel', metavar='SKEL_NII',
                       help='Skeleton NIfTI used to pin node world coords to voxel centres')

    args = ap.parse_args()

    if args.cmd == 'vti':
        nii_path = os.path.abspath(args.nii_path)
        if not os.path.exists(nii_path):
            sys.exit(f"Error: {nii_path} not found")
        vti_path = args.vti_path or _stem(nii_path, ('.nii.gz', '.nii')) + '.vti'
        convert(nii_path, vti_path)

    elif args.cmd == 'vtp':
        mat_path = os.path.abspath(args.mat_path)
        if not os.path.exists(mat_path):
            sys.exit(f"Error: {mat_path} not found")
        skel_path = None
        if args.skel:
            skel_path = os.path.abspath(args.skel)
            if not os.path.exists(skel_path):
                sys.exit(f"Error: {skel_path} not found")
        vtp_path = args.vtp_path or _stem(mat_path, ('.mat',)) + '.vtp'
        mat_to_vtp(mat_path, vtp_path, skel_path=skel_path)
