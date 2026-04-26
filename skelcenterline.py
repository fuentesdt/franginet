"""
skelcenterline.py — Convert a skeletonized NIfTI label to a 1-D VTK centerline mesh.

Every non-zero voxel in the input becomes a mesh node positioned at its world-
coordinate voxel centre (derived from the NIfTI affine).  An undirected edge is
added between every pair of 26-adjacent non-zero voxels, so the mesh topology
exactly mirrors the skeleton connectivity in the NIfTI volume.

Usage
-----
    python skelcenterline.py <skel.nii[.gz]> [output.vtp]

If output path is omitted, <stem>_centerline.vtp is written beside the input.

Dependencies
------------
    pip install nibabel vtk numpy
"""

import argparse
import os
import sys

import numpy as np
import nibabel as nib
import vtk
from vtk.util.numpy_support import numpy_to_vtk


def log(msg):
    print(f"[skelcenterline] {msg}", flush=True)


# ---------------------------------------------------------------------------
def skel_to_vtp(nii_path, vtp_path):
    log(f"Loading  : {nii_path}")
    nii    = nib.load(nii_path)
    affine = nii.affine.astype(np.float64)
    data   = np.asarray(nii.dataobj) != 0      # any nonzero → skeleton voxel

    log(f"  Shape  : {data.shape}  dtype: {nii.get_data_dtype()}")
    n_vox = int(data.sum())
    log(f"  Non-zero voxels : {n_vox}")
    if n_vox == 0:
        sys.exit("Error: skeleton has no non-zero voxels")

    # ── 1. Nodes: every non-zero voxel → world-coordinate point ─────────────
    indices = np.argwhere(data)                 # N×3  (i, j, k)  0-indexed
    N       = len(indices)

    ijk0  = indices.astype(np.float64)
    world = (affine @ np.hstack([ijk0, np.ones((N, 1))]).T).T[:, :3]  # N×3 mm

    bb_min = world.min(axis=0)
    bb_max = world.max(axis=0)
    log(f"  BBox x : {bb_min[0]:.2f} – {bb_max[0]:.2f} mm")
    log(f"  BBox y : {bb_min[1]:.2f} – {bb_max[1]:.2f} mm")
    log(f"  BBox z : {bb_min[2]:.2f} – {bb_max[2]:.2f} mm")

    # ── 2. Edges: 26-adjacency restricted to skeleton voxels ─────────────────
    # Volume-sized lookup: voxel (i,j,k) → node ID  (−1 = background)
    node_id_vol = np.full(data.shape, -1, dtype=np.int64)
    node_id_vol[indices[:, 0], indices[:, 1], indices[:, 2]] = np.arange(N, dtype=np.int64)

    shape = data.shape
    edge_chunks = []

    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            for dk in (-1, 0, 1):
                if di == 0 and dj == 0 and dk == 0:
                    continue

                ni = indices[:, 0] + di
                nj = indices[:, 1] + dj
                nk = indices[:, 2] + dk

                in_bounds = ((ni >= 0) & (ni < shape[0]) &
                             (nj >= 0) & (nj < shape[1]) &
                             (nk >= 0) & (nk < shape[2]))

                src_ids = np.where(in_bounds)[0]
                nbr_ids = node_id_vol[ni[in_bounds], nj[in_bounds], nk[in_bounds]]

                # Keep valid neighbours; store each edge once (src < nbr)
                keep = (nbr_ids >= 0) & (nbr_ids > src_ids)
                if keep.any():
                    edge_chunks.append(
                        np.stack([src_ids[keep], nbr_ids[keep]], axis=1))

    if edge_chunks:
        edges = np.vstack(edge_chunks).astype(np.int64)
    else:
        edges = np.zeros((0, 2), dtype=np.int64)

    E = len(edges)
    log(f"  Nodes: {N}  Edges: {E}")

    # ── 3. Build vtkPolyData ─────────────────────────────────────────────────
    pd = vtk.vtkPolyData()

    # Points — set via numpy array for efficiency
    pts_arr = numpy_to_vtk(np.ascontiguousarray(world), deep=True, array_type=vtk.VTK_DOUBLE)
    pts     = vtk.vtkPoints()
    pts.SetData(pts_arr)
    pd.SetPoints(pts)

    # Lines — pack into legacy vtkCellArray connectivity format:
    #   [2, src0, dst0,  2, src1, dst1, ...]
    if E > 0:
        legacy      = np.empty(3 * E, dtype=np.int64)
        legacy[0::3] = 2
        legacy[1::3] = edges[:, 0]
        legacy[2::3] = edges[:, 1]
        id_arr = numpy_to_vtk(legacy, deep=True, array_type=vtk.VTK_ID_TYPE)
        lines  = vtk.vtkCellArray()
        lines.SetCells(E, id_arr)
    else:
        lines = vtk.vtkCellArray()
    pd.SetLines(lines)

    # ── 4. Write VTP ─────────────────────────────────────────────────────────
    log(f"Writing  : {vtp_path}")
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(vtp_path)
    w.SetInputData(pd)
    w.SetDataModeToAscii()
    w.Write()
    log(f"Done.")


# ---------------------------------------------------------------------------
def _stem(path, suffixes):
    for s in suffixes:
        if path.endswith(s):
            return path[:-len(s)]
    return path


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Convert a skeletonized NIfTI label to a 1-D VTK centerline mesh')
    ap.add_argument('nii_path', help='Input skeleton .nii or .nii.gz')
    ap.add_argument('vtp_path', nargs='?',
                    help='Output .vtp (default: <stem>_centerline.vtp beside input)')
    args = ap.parse_args()

    nii_path = os.path.abspath(args.nii_path)
    if not os.path.exists(nii_path):
        sys.exit(f"Error: {nii_path} not found")

    vtp_path = args.vtp_path or _stem(nii_path, ('.nii.gz', '.nii')) + '_centerline.vtp'
    skel_to_vtp(nii_path, vtp_path)
