"""
convertparaview.py — Convert a NIfTI label volume to VTK ImageData (.vti)
readable in ParaView, preserving the NIfTI world coordinate system.

Usage:
    python convertparaview.py <input.nii[.gz]> [output.vti]

If output path is omitted, writes <stem>.vti next to the input file.

Dependencies:
    pip install nibabel vtk

Encoding note: uses SetDataModeToAscii() — plain-text XML, no binary
encoding.  This is the same pattern as extract_centerline.py and avoids
the zlib / raw-binary parsing issues seen with the appended binary mode.
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
    nii    = nib.load(nii_path)
    affine = nii.affine.astype(np.float64)          # 4×4
    data   = np.asarray(nii.dataobj)

    log(f"  Shape   : {data.shape}  dtype: {data.dtype}")
    log(f"  Affine  :\n{affine.round(4)}")

    # Spacing = Euclidean norm of each column of the 3×3 sub-matrix
    spacing   = np.sqrt(np.sum(affine[:3, :3] ** 2, axis=0))   # (3,)

    # Origin = world position of voxel (0, 0, 0)
    origin    = affine[:3, 3]                                    # (3,)

    # Direction: each column is the unit world vector for one voxel axis
    direction = affine[:3, :3] / spacing[np.newaxis, :]          # 3×3

    log(f"  Origin  : {origin.round(3).tolist()}")
    log(f"  Spacing : {spacing.round(4).tolist()}")

    # ── Build vtkImageData ──────────────────────────────────────────────────
    img = vtk.vtkImageData()
    nx, ny, nz = data.shape[:3]
    img.SetDimensions(nx, ny, nz)
    img.SetOrigin(origin.tolist())
    img.SetSpacing(spacing.tolist())

    # Direction matrix — columns are world-space unit vectors per voxel axis.
    # Available since VTK 9.0 / ParaView 5.9; silently skipped on older builds.
    if hasattr(img, 'SetDirectionMatrix'):
        dm = vtk.vtkMatrix3x3()
        for r in range(3):
            for c in range(3):
                dm.SetElement(r, c, float(direction[r, c]))
        img.SetDirectionMatrix(dm)

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
if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='Convert NIfTI label volume to VTI for ParaView')
    ap.add_argument('nii_path',
                    help='Input NIfTI file (.nii or .nii.gz)')
    ap.add_argument('vti_path', nargs='?',
                    help='Output .vti path (default: same dir/stem as input)')
    args = ap.parse_args()

    nii_path = os.path.abspath(args.nii_path)
    if not os.path.exists(nii_path):
        sys.exit(f"Error: {nii_path} not found")

    if args.vti_path:
        vti_path = args.vti_path
    else:
        stem     = nii_path
        for ext in ('.nii.gz', '.nii'):
            if stem.endswith(ext):
                stem = stem[:-len(ext)]
                break
        vti_path = stem + '.vti'

    convert(nii_path, vti_path)
