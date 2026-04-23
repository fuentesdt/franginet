"""
Extract vessel centerlines from all Frangi vesselness maps produced by
porcine.m.  Reads Processed/manifest.csv, locates the matching
.frangi.nii.gz file for each row, thresholds at THRESHOLD, builds a
surface via marching cubes, and runs the VMTK centerline pipeline.

Outputs (per row, same directory as the .frangi file):
  {timepoint}.centerline.vtp        — centerline polydata (ParaView)
  {timepoint}.centerline_mask.vti   — thresholded binary mask in RAS

Run with:
  /opt/apps/slicer/Slicer-5.10.0-linux-amd64/Slicer \
      --no-main-window \
      --python-script /home/fuentes/github/franginet/extract_centerline.py
"""

import csv
import os
import sys
import slicer
import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

ROI_LABEL = 5   # label value that defines the vessel ROI in the .ml mask

# ── config ────────────────────────────────────────────────────────────────────
REPO_DIR  = "/home/fuentes/github/franginet"
MANIFEST  = os.path.join(REPO_DIR, "Processed", "manifest.csv")
THRESHOLD = 0.44   # match result.threshold from tuneFrangi_result.mat

# Decimation target: fraction of triangles to KEEP after quadric decimation.
# 1.0 = no decimation (skip step).  Lower values remove more triangles but
# can degrade topology.  Start at 0.5 and tune toward 0.2 if VMTK still
# reports degenerate triangles; move toward 1.0 if the centerline is jagged.
DECIMATE_TARGET_REDUCTION = 0.5   # e.g. 0.5 → keep 50 % of triangles


def log(msg):
    print(f"[centerline] {msg}", flush=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def polydata_points(pd):
    n = pd.GetNumberOfPoints()
    return np.array([pd.GetPoint(i) for i in range(n)], dtype=np.float64)


def pca_endpoints(pts):
    """Two surface points at opposite ends of the first principal axis."""
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[0]
    return pts[int(np.argmin(proj))], pts[int(np.argmax(proj))]


def closest_point_id(pd, xyz):
    loc = vtk.vtkPointLocator()
    loc.SetDataSet(pd)
    loc.BuildLocator()
    return loc.FindClosestPoint(xyz)


def make_id_list(*ids):
    il = vtk.vtkIdList()
    for i in ids:
        il.InsertNextId(i)
    return il


def save_vtp(pd, path):
    w = vtk.vtkXMLPolyDataWriter()
    w.SetFileName(path)
    w.SetInputData(pd)
    w.SetDataModeToAscii()   # plain-text XML; avoids zlib-encoding issues in ParaView 5.11
    w.Write()


def reslice_to_ras(imageData, ijkToRas, spacing):
    """Return a vtkImageData resampled onto an axis-aligned RAS grid."""
    rasToIjk = vtk.vtkMatrix4x4()
    vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk)

    dims = imageData.GetDimensions()
    m = np.array([[ijkToRas.GetElement(r, c) for c in range(4)] for r in range(4)])
    corners = np.array([
        [i, j, k, 1.0]
        for i in (0, dims[0] - 1)
        for j in (0, dims[1] - 1)
        for k in (0, dims[2] - 1)
    ])
    corners_ras = (m @ corners.T).T[:, :3]
    rasMin = corners_ras.min(axis=0)
    rasMax = corners_ras.max(axis=0)
    spacing = np.asarray(spacing)
    extent  = [int(round((rasMax[ax] - rasMin[ax]) / spacing[ax])) for ax in range(3)]

    xform = vtk.vtkMatrixToLinearTransform()
    xform.SetInput(rasToIjk)

    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(imageData)
    reslice.SetResliceTransform(xform)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.SetOutputOrigin(rasMin.tolist())
    reslice.SetOutputSpacing(spacing.tolist())
    reslice.SetOutputExtent(0, extent[0], 0, extent[1], 0, extent[2])
    reslice.Update()
    return reslice.GetOutput()


# ── per-file pipeline ─────────────────────────────────────────────────────────

def process_one(frangi_path, mask_path, out_vtp, out_vti):
    """Full pipeline for a single frangi vesselness volume."""

    # 1. load frangi volume into Slicer
    log(f"  Loading: {frangi_path}")
    frangiNode = slicer.util.loadVolume(frangi_path)
    if frangiNode is None:
        raise RuntimeError(f"slicer.util.loadVolume returned None for {frangi_path}")

    imageData = frangiNode.GetImageData()
    ijkToRas  = vtk.vtkMatrix4x4()
    frangiNode.GetIJKToRASMatrix(ijkToRas)
    spacing = frangiNode.GetSpacing()
    dims    = imageData.GetDimensions()
    log(f"  Volume: {dims}  spacing: {[round(s,3) for s in spacing]}")

    # 2. threshold → binary vtkImageData (still in IJK space)
    thresh = vtk.vtkImageThreshold()
    thresh.SetInputData(imageData)
    thresh.ThresholdByUpper(THRESHOLD)
    thresh.SetInValue(1)
    thresh.SetOutValue(0)
    thresh.SetOutputScalarTypeToUnsignedChar()
    thresh.Update()
    binaryArray = vtk_to_numpy(thresh.GetOutput().GetPointData().GetScalars())

    # 3. load label mask and restrict to ROI_LABEL (=5) voxels
    log(f"  Loading mask: {mask_path}  (ROI label={ROI_LABEL})")
    maskNode = slicer.util.loadLabelVolume(mask_path)
    if maskNode is None:
        raise RuntimeError(f"Could not load mask {mask_path}")
    maskDims = maskNode.GetImageData().GetDimensions()
    if maskDims != dims:
        raise RuntimeError(
            f"Mask dimensions {maskDims} differ from frangi dimensions {dims}")
    maskArray = vtk_to_numpy(maskNode.GetImageData().GetPointData().GetScalars())
    roiVoxels = int(np.sum(maskArray == ROI_LABEL))
    log(f"  ROI voxels (label={ROI_LABEL}): {roiVoxels}")
    if roiVoxels == 0:
        raise RuntimeError(
            f"Label {ROI_LABEL} not found in {mask_path} — "
            f"present labels: {np.unique(maskArray).tolist()}")

    # zero vesselness outside the ROI, then apply threshold
    maskedArray = (binaryArray * (maskArray == ROI_LABEL)).astype(np.uint8)

    binaryIJK = vtk.vtkImageData()
    binaryIJK.DeepCopy(thresh.GetOutput())
    scalars = numpy_to_vtk(maskedArray, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
    binaryIJK.GetPointData().SetScalars(scalars)

    nFg = int(maskedArray.sum())
    log(f"  Foreground voxels after ROI masking: {nFg}")
    if nFg == 0:
        raise RuntimeError("No foreground voxels inside ROI — check THRESHOLD and label")

    # 3. marching cubes → surface in IJK, then transform to RAS
    mc = vtk.vtkMarchingCubes()
    mc.SetInputData(binaryIJK)
    mc.SetValue(0, 0.5)
    mc.Update()

    xformFilter = vtk.vtkTransformPolyDataFilter()
    xform = vtk.vtkTransform()
    xform.SetMatrix(ijkToRas)
    xformFilter.SetInputData(mc.GetOutput())
    xformFilter.SetTransform(xform)
    xformFilter.Update()
    rawSurface = xformFilter.GetOutput()
    log(f"  Marching cubes: {rawSurface.GetNumberOfPoints()} pts")

    if rawSurface.GetNumberOfPoints() == 0:
        raise RuntimeError("Marching cubes produced empty surface")

    # 4. clean → fill holes → subdivide → smooth
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(rawSurface)
    cleaner.SetTolerance(0.0)
    cleaner.Update()

    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(cleaner.GetOutput())
    tri.Update()

    conn = vtk.vtkPolyDataConnectivityFilter()
    conn.SetInputData(tri.GetOutput())
    conn.SetExtractionModeToLargestRegion()
    conn.Update()

    fill = vtk.vtkFillHolesFilter()
    fill.SetInputData(conn.GetOutput())
    fill.SetHoleSize(10000.0)
    fill.Update()

    tri2 = vtk.vtkTriangleFilter()
    tri2.SetInputData(fill.GetOutput())
    tri2.Update()

    subdiv = vtk.vtkAdaptiveSubdivisionFilter()
    subdiv.SetInputData(tri2.GetOutput())
    subdiv.SetMaximumEdgeLength(0.3)
    subdiv.Update()
    pre_decimate = subdiv.GetOutput()
    log(f"  After subdivision : {pre_decimate.GetNumberOfPoints():>7} pts  "
        f"{pre_decimate.GetNumberOfCells():>7} tris")

    if DECIMATE_TARGET_REDUCTION < 1.0:
        decimate = vtk.vtkQuadricDecimation()
        decimate.SetInputData(pre_decimate)
        decimate.SetTargetReduction(DECIMATE_TARGET_REDUCTION)
        decimate.Update()
        post_decimate = decimate.GetOutput()
        log(f"  After decimation  : {post_decimate.GetNumberOfPoints():>7} pts  "
            f"{post_decimate.GetNumberOfCells():>7} tris  "
            f"(target reduction={DECIMATE_TARGET_REDUCTION:.2f})")
    else:
        post_decimate = pre_decimate
        log(f"  Decimation skipped (DECIMATE_TARGET_REDUCTION=1.0)")

    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(post_decimate)
    smoother.SetNumberOfIterations(20)
    smoother.SetPassBand(0.1)
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    surface = smoother.GetOutput()
    log(f"  After smoothing   : {surface.GetNumberOfPoints():>7} pts  "
        f"{surface.GetNumberOfCells():>7} tris")

    if surface.GetNumberOfPoints() == 0:
        raise RuntimeError("Surface pre-processing produced empty result")

    # 5. PCA endpoint detection
    pts = polydata_points(surface)
    srcXYZ, tgtXYZ = pca_endpoints(pts)
    srcId = closest_point_id(surface, srcXYZ)
    tgtId = closest_point_id(surface, tgtXYZ)
    log(f"  Source pt {srcId}: {srcXYZ.round(2)}")
    log(f"  Target pt {tgtId}: {tgtXYZ.round(2)}")

    # 6. compute consistent normals — required by VMTK's Voronoi builder
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(surface)
    normals.ConsistencyOn()       # flip inconsistent normals to agree with neighbours
    normals.SplittingOff()        # no new vertices at sharp edges
    normals.AutoOrientNormalsOn() # orient all normals outward
    normals.Update()
    surfaceWithNormals = normals.GetOutput()

    # 7. centerline extraction (backend A → B fallback)
    centerlinePolyData = None

    try:
        import vtkvmtkComputationalGeometryPython as vtkvmtk
        log("  Backend: vtkvmtkComputationalGeometryPython")
        clFilter = vtkvmtk.vtkvmtkPolyDataCenterlines()
        clFilter.SetInputData(surfaceWithNormals)
        clFilter.SetSourceSeedIds(make_id_list(srcId))
        clFilter.SetTargetSeedIds(make_id_list(tgtId))
        clFilter.SetRadiusArrayName("MaximumInscribedSphereRadius")
        clFilter.SetCostFunction("1/R")
        clFilter.SetFlipNormals(False)
        clFilter.SetAppendEndPointsToCenterlines(True)
        clFilter.SetSimplifyVoronoi(False)
        clFilter.Update()
        centerlinePolyData = clFilter.GetOutput()
        log(f"  Centerline: {centerlinePolyData.GetNumberOfPoints()} pts")
    except ImportError:
        log("  vtkvmtkComputationalGeometryPython not available, trying ExtractCenterline...")
    except Exception as e:
        log(f"  vtkvmtk backend failed ({e}), trying ExtractCenterline...")

    if centerlinePolyData is None or centerlinePolyData.GetNumberOfPoints() == 0:
        import ExtractCenterline
        log("  Backend: ExtractCenterlineLogic")
        ecLogic = ExtractCenterline.ExtractCenterlineLogic()

        # actual signature: extractCenterline(surfacePolyData, endPointsMarkupsNode,
        #                                     curveSamplingDistance=1.0)
        endpointsNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "Endpoints")
        endpointsNode.AddControlPoint(vtk.vtkVector3d(*srcXYZ))
        endpointsNode.AddControlPoint(vtk.vtkVector3d(*tgtXYZ))

        result = ecLogic.extractCenterline(surfaceWithNormals, endpointsNode)

        # result may be a model node, a polydata, or a dict depending on version
        if isinstance(result, vtk.vtkPolyData):
            centerlinePolyData = result
        elif hasattr(result, 'GetPolyData'):
            centerlinePolyData = result.GetPolyData()
        elif isinstance(result, dict):
            node = result.get('centerlineModelNode') or result.get('centerline')
            centerlinePolyData = node.GetPolyData() if hasattr(node, 'GetPolyData') else node
        else:
            # fall back: search the scene for any newly added model node
            for i in range(slicer.mrmlScene.GetNumberOfNodesByClass('vtkMRMLModelNode')):
                n = slicer.mrmlScene.GetNthNodeByClass(i, 'vtkMRMLModelNode')
                if n.GetPolyData() and n.GetPolyData().GetNumberOfLines() > 0:
                    centerlinePolyData = n.GetPolyData()
                    break

        if centerlinePolyData is not None:
            log(f"  Centerline: {centerlinePolyData.GetNumberOfPoints()} pts")

    if centerlinePolyData is None or centerlinePolyData.GetNumberOfPoints() == 0:
        raise RuntimeError("All backends returned empty centerline")

    # 7. save centerline VTP
    log(f"  Saving VTP  → {out_vtp}")
    save_vtp(centerlinePolyData, out_vtp)

    # 8. save thresholded binary mask as RAS-aligned VTI
    log(f"  Saving VTI  → {out_vti}")
    rasImage = reslice_to_ras(binaryIJK, ijkToRas, spacing)
    vtiWriter = vtk.vtkXMLImageDataWriter()
    vtiWriter.SetFileName(out_vti)
    vtiWriter.SetInputData(rasImage)
    vtiWriter.SetDataModeToAscii()   # avoid zlib-compressed binary; ParaView 5.11 parses this fine
    vtiWriter.Write()


# ── main ──────────────────────────────────────────────────────────────────────

log(f"Reading manifest: {MANIFEST}")
with open(MANIFEST, newline='') as f:
    rows = list(csv.DictReader(f))
log(f"  {len(rows)} rows\n")

errors = []

for row in rows:
    subject   = row['subject_id']
    session   = row['session']
    tp        = row['timepoint']
    img_path  = row['image']
    mask_path = row['mask']

    # derive frangi path from image path
    frangi_path = img_path.replace('.raw.nii.gz', '.frangi.nii.gz')
    if not os.path.exists(frangi_path):
        log(f"SKIP {subject}/{session}/{tp}: {frangi_path} not found")
        continue
    if not os.path.exists(mask_path):
        log(f"SKIP {subject}/{session}/{tp}: mask {mask_path} not found")
        continue

    out_dir = os.path.dirname(os.path.abspath(frangi_path))
    out_vtp = os.path.join(out_dir, f"{tp}.centerline.vtp")
    out_vti = os.path.join(out_dir, f"{tp}.centerline_mask.vti")

    log(f"=== {subject} / {session} / {tp} ===")
    try:
        process_one(frangi_path, mask_path, out_vtp, out_vti)
        log(f"  OK\n")
    except Exception as e:
        log(f"  ERROR: {e}\n")
        errors.append(f"{subject}/{session}/{tp}: {e}")
    finally:
        slicer.mrmlScene.Clear(0)   # free memory between volumes

# ── summary ───────────────────────────────────────────────────────────────────
n_ok = len(rows) - len(errors)
log(f"Finished: {n_ok}/{len(rows)} succeeded")
if errors:
    log(f"{len(errors)} error(s):")
    for e in errors:
        log(f"  {e}")

slicer.util.exit(0 if not errors else 1)
