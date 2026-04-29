"""
run_thermoembo.py — Prepare and launch a thermoembo1d coupled 1D-3D simulation.

Pipeline
--------
  1. Extract label==2 vessel mask → skeletonize → vessel_skeleton.nii.gz
  2. skelcenterline.py → centerline.vtp  (1-D vessel graph)
  3. meshliver.py     → mesh.vtu         (3-D liver tet mesh, coarse settings)
  4. Convert mesh.vtu → mesh.exo         (mm → m, meshio/netCDF4)
  5. Write vessel.vtk  (legacy VTK image from NIfTI label, mm coords)
  6. docker run thermoembo  thermoembo1d-...  with minimal test options

Inputs
------
  prepost/PreTxArtLoRes.vessellabel.nii.gz
    label == 1 : liver parenchyma
    label == 2 : vessel lumen

Outputs (written to --out-dir, default: thermoembo_run/)
  vessel_skeleton.nii.gz   skeletonised label-2 volume
  centerline.vtp           1-D vessel centerline mesh
  surface.vtp              decimated liver surface
  mesh.vtu                 tetrahedral volume mesh  (mm)
  mesh.exo                 Exodus mesh              (m, for PETSc/thermoembo1d)
  vessel.vtk               legacy VTK image (mm, material field)
  result*.vtu              thermoembo1d solution snapshots

Usage
-----
  python run_thermoembo.py prepost/PreTxArtLoRes.vessellabel.nii.gz
  python run_thermoembo.py prepost/PreTxArtLoRes.vessellabel.nii.gz \\
      --out-dir thermoembo_run --decimate 0.90 --maxvol 5000 --steps 5

Dependencies
------------
  pip install nibabel pyvista pymeshfix tetgen scipy scikit-image meshio netCDF4
  docker image: thermoembo (built from /home/fuentes/github/thermoembo/Dockerfile)
"""

import argparse
import os
import subprocess
import sys

import numpy as np
import nibabel as nib
import vtk
from vtk.util.numpy_support import numpy_to_vtk
import pyvista as pv
import meshio
from scipy.ndimage import binary_fill_holes
from skimage.morphology import skeletonize


THERMOEMBO_EXE = "/work/tutorials/thermoembo1d-3.10.2-xenial-gcc-5.4.0-opt"
DOCKER_IMAGE   = "thermoembo"


def log(msg):
    print(f"[run_thermoembo] {msg}", flush=True)


# ── Step 1: skeletonize label==2 ──────────────────────────────────────────────

def make_vessel_skeleton(nii_path, out_nii):
    """Extract label==2 from NIfTI, skeletonise, write result NIfTI."""
    log("Skeletonising vessel mask (label==2) ...")
    nii    = nib.load(nii_path)
    data   = np.asarray(nii.dataobj)
    binary = (data == 2)
    log(f"  Vessel voxels  : {int(binary.sum())}")
    if not binary.any():
        sys.exit("Error: no label==2 voxels found in input")

    skel = skeletonize(binary)          # Lee 1994 thinning (3-D)
    log(f"  Skeleton voxels: {int(skel.sum())}")

    out = nib.Nifti1Image(skel.astype(np.uint8), nii.affine, nii.header)
    nib.save(out, out_nii)
    log(f"  Written        : {out_nii}")


# ── Step 3: write legacy VTK image (material file) ────────────────────────────

def make_vessel_vtk(nii_path, out_vtk):
    """
    Write a zero-filled VTK legacy structured-points image covering the NIfTI
    extent.

    thermoembo1d uses this image via -vtk to initialise the phase field via
    analytic_phas(): phase(x) = image_value(x).  A zero image means
    "no embolic material deposited at t=0," which puts all tissue nodes in the
    free-DOF set (phase=0 < phasethresh=0.5).  Using the tissue label image
    would set phase=1 or 2 everywhere, making isnotdirichlet empty and leaving
    the coupled solver with no free degrees of freedom.

    vtkImageData requires positive axis-aligned spacing.  For NIfTI affines
    with negative diagonal elements (common RAS/LPS flip) we shift the origin
    to the minimum world corner so that ComputeStructuredCoordinates maps FEM
    world coordinates to correct voxel indices.
    """
    log("Writing VTK material image (zero initial phase field) ...")
    nii    = nib.load(nii_path)
    affine = nii.affine.astype(np.float64)

    signs   = np.sign(np.diag(affine[:3, :3]))
    spacing = np.abs(np.diag(affine[:3, :3]))   # always positive

    # Origin = minimum world corner (shift for negative-diagonal axes)
    origin = affine[:3, 3].copy()
    for ax in range(3):
        if signs[ax] < 0:
            n = nii.shape[ax]
            origin[ax] = affine[ax, 3] + signs[ax] * spacing[ax] * (n - 1)
    origin = np.minimum(origin, affine[:3, 3])

    log(f"  Origin  (mm)   : {origin.round(2).tolist()}")
    log(f"  Spacing (mm)   : {spacing.round(4).tolist()}")
    log(f"  Dimensions     : {list(nii.shape)}")

    # Zero-filled scalar array — phase=0 everywhere at t=0
    n_pts = int(np.prod(nii.shape))
    zeros = np.zeros(n_pts, dtype=np.float32)

    img = vtk.vtkImageData()
    img.SetDimensions(nii.shape)
    img.SetSpacing(spacing.tolist())
    img.SetOrigin(origin.tolist())

    arr = numpy_to_vtk(zeros, deep=True, array_type=vtk.VTK_FLOAT)
    arr.SetName("phase")
    img.GetPointData().SetScalars(arr)

    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(out_vtk)
    writer.SetInputData(img)
    writer.SetFileTypeToBinary()
    writer.Write()
    log(f"  Written        : {out_vtk}")


# ── Step 4: VTU → Exodus conversion (mm → m) ─────────────────────────────────

def vtu_to_exodus(vtu_path, exo_path):
    """
    Convert a pyvista VTU (tetrahedral, coordinates in mm) to Exodus II
    (coordinates in metres), as required by PETSc/thermoembo1d.
    """
    log("Converting VTU → Exodus (mm → m) ...")
    grid = pv.read(vtu_path)

    VTK_TETRA = 10
    if VTK_TETRA not in grid.cells_dict:
        sys.exit(f"Error: VTU mesh does not contain tetrahedral cells "
                 f"(found types: {list(grid.cells_dict.keys())})")

    points_m = grid.points * 1.0e-3               # mm → m
    cells    = [meshio.CellBlock('tetra', grid.cells_dict[VTK_TETRA])]
    mesh     = meshio.Mesh(points_m, cells)
    meshio.write(exo_path, mesh)

    log(f"  Nodes          : {grid.n_points}")
    log(f"  Elements       : {grid.n_cells}")
    bb = grid.bounds
    log(f"  BBox (m)       : x [{bb[0]*1e-3:.4f}, {bb[1]*1e-3:.4f}]  "
        f"y [{bb[2]*1e-3:.4f}, {bb[3]*1e-3:.4f}]  "
        f"z [{bb[4]*1e-3:.4f}, {bb[5]*1e-3:.4f}]")
    log(f"  Written        : {exo_path}")


# ── Step 5: run Docker ────────────────────────────────────────────────────────

def _docker_cmd(abs_out, steps, dt, vesselcoupling):
    """Return the shared argument list for both Docker invocations."""
    return [
        "docker", "run", "--rm",
        "-v", f"{abs_out}:/data",
        DOCKER_IMAGE,
        THERMOEMBO_EXE,
        # --- mesh ---
        "-dim",  "3",
        "-mesh", "/data/mesh.exo",
        # --- 1-D vessel centerline ---
        "-vtp1d",          "/data/centerline.vtp",
        "-vesselcoupling", str(vesselcoupling),
        # --- material image ---
        "-vtk", "/data/vessel.vtk",
        # --- output prefix (also defines phase-solution cache filename) ---
        "-o",   "/data/result",
        # --- FE degree (P1 for all fields) ---
        "-temp_petscspace_degree", "1",
        "-pres_petscspace_degree", "1",
        "-damg_petscspace_degree", "1",
        "-conc_petscspace_degree", "1",
        "-phas_petscspace_degree", "1",
        # --- time integration ---
        "-ts_type",      "beuler",
        "-ts_max_steps", str(steps),
        "-ts_dt",        str(dt),
        "-ts_monitor",
        "-modulowrite",  "1",           # write VTU every time step
        # --- main solve ---
        # The C code calls PCFieldSplitSetIS("p"/"s"/"u") then
        # PCFieldSplitGetSubKSP() unconditionally.  For a non-fieldsplit PC
        # this call is a no-op and the outer KSP runs unmodified.
        # GMRES with no preconditioner converges in the test case because the
        # initial condition (zero phase field everywhere) gives a zero RHS for
        # the first step, and subsequent steps remain well-conditioned.
        "-snes_type",   "ksponly",
        "-ksp_type",    "gmres",
        "-ksp_max_it",  "500",
        "-ksp_rtol",    "1e-3",
        "-pc_type",     "none",
        "-ksp_converged_reason",
        "-snes_converged_reason",
        "-ts_max_snes_failures", "-1",
        # --- phase-field presolve (run 1 only; result cached to .dat) ---
        "-phasepresolve_snes_type",              "ksponly",
        "-phasepresolve_ksp_type",               "preonly",
        "-phasepresolve_pc_type",                "fieldsplit",
        "-phasepresolve_pc_fieldsplit_type",      "additive",
        "-phasepresolve_fieldsplit_d_ksp_type",   "preonly",
        "-phasepresolve_fieldsplit_d_pc_type",    "none",
        "-phasepresolve_fieldsplit_1_ksp_type",   "preonly",
        "-phasepresolve_fieldsplit_1_pc_type",    "bjacobi",
        "-phasepresolve_ts_type",                "beuler",
        "-phasepresolve_ts_max_steps",           "3",
        "-phasepresolve_snes_converged_reason",
        "-phasepresolve_ts_max_snes_failures",   "-1",
        # --- misc ---
        "-disppressure",     "0.0",
        "-baselinepressure", "1.0",
    ]


def run_docker(out_dir, steps, dt, vesselcoupling):
    """
    Launch thermoembo1d in two Docker passes:
      Pass 1 — phase-field presolve: writes /data/result.0000.0001.dat
      Pass 2 — main coupled solve:   reads the .dat, runs <steps> time steps

    thermoembo1d detects which pass to run based on whether the .dat file
    exists (PetscTestFile); the two passes use the same command-line arguments.
    All files are mounted at /data inside the container.
    Solution VTU snapshots: /data/resultsolution000.NNNN.vtu
    """
    abs_out = os.path.abspath(out_dir)
    dat_file = os.path.join(out_dir, "result.0000.0001.dat")
    cmd = _docker_cmd(abs_out, steps, dt, vesselcoupling)

    log("Pass 1 — phase-field presolve ...")
    if os.path.exists(dat_file):
        log("  Phase solution cache found — skipping presolve.")
    else:
        log("  " + " ".join(cmd))
        r = subprocess.run(cmd)
        if r.returncode != 0:
            log(f"  Warning: pass 1 exited with code {r.returncode}")
        else:
            log(f"  Phase solution saved: {dat_file}")

    log("Pass 2 — main coupled solve ...")
    log("  " + " ".join(cmd))
    r = subprocess.run(cmd)
    if r.returncode != 0:
        log(f"Warning: thermoembo1d pass 2 exited with code {r.returncode}")
    else:
        log("thermoembo1d completed successfully")
        log(f"Solution VTU files written to: {out_dir}/resultsolution*.vtu")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Prepare meshes and run thermoembo1d via Docker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)

    ap.add_argument("nii",
                    help="Labeled NIfTI (.nii or .nii.gz) with label 1=liver, 2=vessel")
    ap.add_argument("--out-dir",       default="thermoembo_run",  metavar="DIR",
                    help="Output directory (default: thermoembo_run)")
    ap.add_argument("--decimate",      type=float, default=0.90,  metavar="R",
                    help="Surface decimation reduction ratio (default 0.90 = keep 10%%)")
    ap.add_argument("--maxvol",        type=float, default=5000,  metavar="MM3",
                    help="TetGen max tet volume mm³ (default 5000)")
    ap.add_argument("--tet-dihedral",  type=float, default=15.0,  metavar="DEG",
                    help="TetGen min dihedral angle (default 15)")
    ap.add_argument("--steps",         type=int,   default=5,     metavar="N",
                    help="thermoembo1d time steps (default 5)")
    ap.add_argument("--dt",            type=float, default=60.0,  metavar="S",
                    help="thermoembo1d time-step size in seconds (default 60)")
    ap.add_argument("--vesselcoupling",type=float, default=1.0e-4,metavar="G",
                    help="Vessel-tissue coupling conductance [1/s/atm] (default 1e-4)")
    ap.add_argument("--no-docker",     action="store_true",
                    help="Prepare meshes only — skip Docker run")

    args    = ap.parse_args()
    nii_path = os.path.abspath(args.nii)
    out_dir  = os.path.abspath(args.out_dir)

    if not os.path.exists(nii_path):
        sys.exit(f"Error: {nii_path} not found")

    os.makedirs(out_dir, exist_ok=True)
    here = os.path.dirname(os.path.abspath(__file__))

    # Paths inside out_dir
    skel_nii      = os.path.join(out_dir, "vessel_skeleton.nii.gz")
    centerline_vtp = os.path.join(out_dir, "centerline.vtp")
    mesh_vtu      = os.path.join(out_dir, "mesh.vtu")
    mesh_exo      = os.path.join(out_dir, "mesh.exo")
    vessel_vtk    = os.path.join(out_dir, "vessel.vtk")

    # ── 1. Skeletonise vessel ────────────────────────────────────────────────
    make_vessel_skeleton(nii_path, skel_nii)

    # ── 2. 1-D centerline mesh ───────────────────────────────────────────────
    log("Running skelcenterline.py ...")
    subprocess.run(
        [sys.executable,
         os.path.join(here, "skelcenterline.py"),
         skel_nii, centerline_vtp],
        check=True)

    # ── 3. 3-D liver tet mesh ────────────────────────────────────────────────
    log("Running meshliver.py (coarse settings for test) ...")
    subprocess.run(
        [sys.executable,
         os.path.join(here, "meshliver.py"),
         nii_path,
         "--decimate",     str(args.decimate),
         "--tet-maxvol",   str(args.maxvol),
         "--tet-dihedral", str(args.tet_dihedral),
         "--output-dir",   out_dir],
        check=True)

    # meshliver.py writes surface.vtp + mesh.vtu in out_dir
    if not os.path.exists(mesh_vtu):
        sys.exit(f"Error: expected {mesh_vtu} — meshliver.py may have failed")

    # ── 4. VTU → Exodus ─────────────────────────────────────────────────────
    vtu_to_exodus(mesh_vtu, mesh_exo)

    # ── 5. Material VTK image ────────────────────────────────────────────────
    make_vessel_vtk(nii_path, vessel_vtk)

    log("=" * 60)
    log("Mesh preparation complete:")
    for f in [centerline_vtp, mesh_vtu, mesh_exo, vessel_vtk]:
        size = os.path.getsize(f) / 1024
        log(f"  {os.path.basename(f):25s}  {size:8.1f} kB")
    log("=" * 60)

    if args.no_docker:
        log("--no-docker specified; skipping simulation run.")
        return

    # ── 6. Run thermoembo1d ─��────────────────────────────────────────────────
    run_docker(out_dir, args.steps, args.dt, args.vesselcoupling)


if __name__ == "__main__":
    main()
