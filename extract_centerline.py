"""
Extract vessel centerline from a binary NIfTI segmentation.

Two backends are tried in order:
  1. vtkvmtkComputationalGeometryPython  (C++ VMTK binding — preferred)
  2. ExtractCenterline logic node        (SlicerVMTK Python wrapper)

Endpoints are auto-detected via PCA on the surface point cloud.

Run with:
  /opt/apps/slicer/Slicer-5.10.0-linux-amd64/Slicer \
      --no-main-window \
      --python-script /home/fuentes/github/franginet/extract_centerline.py
"""

import os
import sys
import slicer
import vtk
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────────
REPO_DIR  = "/home/fuentes/github/franginet"
MASK_PATH = os.path.join(REPO_DIR, "bezier_training", "masks", "mask_001.nii")
OUT_DIR        = os.path.join(REPO_DIR, "bezier_training", "centerlines")
OUT_VTP        = os.path.join(OUT_DIR, "centerline_001.vtp")
OUT_MASK_VTI   = os.path.join(OUT_DIR, "mask_001_ras.vti")  # mask resampled into RAS, aligns with VTP in ParaView
OUT_FCSV       = os.path.join(OUT_DIR, "centerline_001.fcsv")

os.makedirs(OUT_DIR, exist_ok=True)


def log(msg):
    print(f"[centerline] {msg}", flush=True)


# ── helpers ────────────────────────────────────────────────────────────────────

def polydata_points(pd):
    n = pd.GetNumberOfPoints()
    return np.array([pd.GetPoint(i) for i in range(n)], dtype=np.float64)


def pca_endpoints(pts):
    """Two surface points at opposite ends of the principal axis."""
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
    w.Write()


# ── 1. load label map ──────────────────────────────────────────────────────────
log(f"Loading mask: {MASK_PATH}")
labelMapNode = slicer.util.loadLabelVolume(MASK_PATH)
if labelMapNode is None:
    sys.stderr.write(f"ERROR: could not load {MASK_PATH}\n")
    slicer.util.exit(1)


# ── 2. labelmap → segmentation → closed surface ───────────────────────────────
log("Converting labelmap → segmentation → closed surface...")
segNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
    labelMapNode, segNode
)
segNode.CreateClosedSurfaceRepresentation()

seg = segNode.GetSegmentation()
if seg.GetNumberOfSegments() == 0:
    sys.stderr.write("ERROR: segmentation is empty.\n")
    slicer.util.exit(1)
log(f"  segment id: {seg.GetNthSegmentID(0)}")


# ── 3. segmentation → model node ──────────────────────────────────────────────
log("Exporting surface to model node...")
shNode       = slicer.mrmlScene.GetSubjectHierarchyNode()
exportFolder = shNode.CreateFolderItem(shNode.GetSceneItemID(), "ModelExport")
slicer.modules.segmentations.logic().ExportAllSegmentsToModels(segNode, exportFolder)

childIds = vtk.vtkIdList()
shNode.GetItemChildren(exportFolder, childIds, True)
modelNode = None
for i in range(childIds.GetNumberOfIds()):
    node = shNode.GetItemDataNode(childIds.GetId(i))
    if node and node.IsA("vtkMRMLModelNode"):
        modelNode = node
        break

if modelNode is None:
    sys.stderr.write("ERROR: surface export produced no model nodes.\n")
    slicer.util.exit(1)

nPts = modelNode.GetPolyData().GetNumberOfPoints()
log(f"  surface: {nPts} points, {modelNode.GetPolyData().GetNumberOfCells()} cells")
if nPts == 0:
    sys.stderr.write("ERROR: exported surface is empty.\n")
    slicer.util.exit(1)


# ── 4. prepare surface (triangulate + keep largest region) ────────────────────
log("Pre-processing surface...")
tri = vtk.vtkTriangleFilter()
tri.SetInputData(modelNode.GetPolyData())
tri.Update()

conn = vtk.vtkPolyDataConnectivityFilter()
conn.SetInputData(tri.GetOutput())
conn.SetExtractionModeToLargestRegion()
conn.Update()
surface = conn.GetOutput()
log(f"  cleaned: {surface.GetNumberOfPoints()} pts")


# ── 5. PCA endpoint detection ─────────────────────────────────────────────────
log("Detecting endpoints via PCA...")
pts = polydata_points(surface)
srcXYZ, tgtXYZ = pca_endpoints(pts)
srcId = closest_point_id(surface, srcXYZ)
tgtId = closest_point_id(surface, tgtXYZ)
log(f"  source pt {srcId}: {srcXYZ.round(2)}")
log(f"  target pt {tgtId}: {tgtXYZ.round(2)}")


# ── 6. centerline extraction ───────────────────────────────────────────────────

centerlinePolyData = None

# ── backend A: vtkvmtkComputationalGeometryPython (C++ binding) ──────────────
try:
    import vtkvmtkComputationalGeometryPython as vtkvmtk
    log("Backend: vtkvmtkComputationalGeometryPython")

    clFilter = vtkvmtk.vtkvmtkPolyDataCenterlines()
    clFilter.SetInputData(surface)
    clFilter.SetSourceSeedIds(make_id_list(srcId))
    clFilter.SetTargetSeedIds(make_id_list(tgtId))
    clFilter.SetRadiusArrayName("MaximumInscribedSphereRadius")
    clFilter.SetCostFunction("1/R")
    clFilter.SetFlipNormals(False)
    clFilter.SetAppendEndPointsToCenterlines(True)
    clFilter.SetSimplifyVoronoi(False)
    clFilter.Update()
    centerlinePolyData = clFilter.GetOutput()
    log(f"  result: {centerlinePolyData.GetNumberOfPoints()} pts")

except ImportError:
    log("vtkvmtkComputationalGeometryPython not available, trying ExtractCenterline...")

except Exception as e:
    log(f"vtkvmtk backend failed ({e}), trying ExtractCenterline...")


# ── backend B: ExtractCenterlineLogic (SlicerVMTK wrapper) ───────────────────
if centerlinePolyData is None or centerlinePolyData.GetNumberOfPoints() == 0:
    try:
        import ExtractCenterline
        log("Backend: ExtractCenterlineLogic")

        ecLogic = ExtractCenterline.ExtractCenterlineLogic()

        # Feed PCA endpoints as markups fiducials
        endpointsNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsFiducialNode", "Endpoints"
        )
        endpointsNode.AddControlPoint(vtk.vtkVector3d(*srcXYZ))
        endpointsNode.AddControlPoint(vtk.vtkVector3d(*tgtXYZ))

        centerlineModelNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "CenterlineModel"
        )
        voronoiModelNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLModelNode", "VoronoiDiagram"
        )
        centerlineCurveNode = slicer.mrmlScene.AddNewNodeByClass(
            "vtkMRMLMarkupsCurveNode", "CenterlineCurve"
        )

        ecLogic.extractCenterline(
            modelNode,
            endpointsNode,
            centerlineModelNode,
            voronoiModelNode,
            centerlineCurveNode,
        )

        centerlinePolyData = centerlineModelNode.GetPolyData()
        log(f"  result: {centerlinePolyData.GetNumberOfPoints()} pts")

    except Exception as e:
        sys.stderr.write(f"ERROR: ExtractCenterline backend failed: {e}\n")
        slicer.util.exit(1)


if centerlinePolyData is None or centerlinePolyData.GetNumberOfPoints() == 0:
    sys.stderr.write("ERROR: all backends returned empty centerline.\n")
    slicer.util.exit(1)


# ── 7. save centerline VTP ────────────────────────────────────────────────────
log(f"Saving centerline VTP → {OUT_VTP}")
save_vtp(centerlinePolyData, OUT_VTP)

# ── 8. resample mask into axis-aligned RAS and write VTI ─────────────────────
# The centerline VTP is in Slicer's RAS frame. slicer.util.saveNode writes
# volumes via ITK in LPS (X/Y flipped), so we bypass ITK and reslice directly
# into the same RAS grid using vtkImageReslice.
log(f"Saving mask VTI (RAS-aligned) → {OUT_MASK_VTI}")

ijkToRas = vtk.vtkMatrix4x4()
labelMapNode.GetIJKToRASMatrix(ijkToRas)

rasToIjk = vtk.vtkMatrix4x4()
vtk.vtkMatrix4x4.Invert(ijkToRas, rasToIjk)

# Compute tight RAS bounding box from the 8 IJK corners
dims = labelMapNode.GetImageData().GetDimensions()
m = np.array([[ijkToRas.GetElement(r, c) for c in range(4)] for r in range(4)])
corners_ijk = np.array([
    [i, j, k, 1.0]
    for i in (0, dims[0] - 1)
    for j in (0, dims[1] - 1)
    for k in (0, dims[2] - 1)
])
corners_ras = (m @ corners_ijk.T).T[:, :3]
rasMin = corners_ras.min(axis=0)
rasMax = corners_ras.max(axis=0)

spacing = np.array(labelMapNode.GetSpacing())
extent  = [int(round((rasMax[ax] - rasMin[ax]) / spacing[ax])) for ax in range(3)]

resliceXform = vtk.vtkMatrixToLinearTransform()
resliceXform.SetInput(rasToIjk)

reslice = vtk.vtkImageReslice()
reslice.SetInputData(labelMapNode.GetImageData())
reslice.SetResliceTransform(resliceXform)
reslice.SetInterpolationModeToNearestNeighbor()   # binary mask — no blurring
reslice.SetOutputOrigin(rasMin.tolist())
reslice.SetOutputSpacing(spacing.tolist())
reslice.SetOutputExtent(0, extent[0], 0, extent[1], 0, extent[2])
reslice.Update()

vtiWriter = vtk.vtkXMLImageDataWriter()
vtiWriter.SetFileName(OUT_MASK_VTI)
vtiWriter.SetInputData(reslice.GetOutput())
vtiWriter.Write()

log(f"Saving FCSV → {OUT_FCSV}")
curveNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsCurveNode",
                                               "CenterlineCurve")
for i in range(centerlinePolyData.GetNumberOfPoints()):
    x, y, z = centerlinePolyData.GetPoint(i)
    curveNode.AddControlPoint(vtk.vtkVector3d(x, y, z))
slicer.util.saveNode(curveNode, OUT_FCSV)

log("Done.")
slicer.util.exit(0)
