# Hybrid Learnable-Frangi + U-Net — MATLAB Implementation

A differentiable vessel segmentation architecture combining a **learnable multi-scale Frangi vesselness filter** with a **U-Net encoder-decoder**, trained end-to-end with a combined Dice + BCE loss.

---

## Files

**Segmentation model**

| File | Purpose |
|---|---|
| `trainFrangiUNet.m` | Top-level training entry point |
| `buildFrangiUNet.m` | DAG network graph constructor |
| `learnableFrangiLayer.m` | Custom `nnet.layer` — differentiable Frangi |
| `dicePixelClassificationLayer.m` | Custom loss layer — Dice + BCE |
| `evaluateFrangiUNet.m` | Inference + Dice / clDice / AUC metrics |
| `demo_frangiUNet.m` | Self-contained demo on synthetic data |

**Vessel analysis pipeline**

| File | Purpose |
|---|---|
| `myskelotonize.m` | Batch skeletonization of binary vessel masks (MATLAB `bwskel`) |
| `resistanceLumping.m` | Hagen-Poiseuille resistance network solver on reduced skeleton graph (MATLAB) |
| `skelcenterline.py` | Convert skeleton NIfTI → dense 1-D VTP centerline mesh (every voxel = node) |
| `resistance_lumping.py` | Hagen-Poiseuille resistance lumping solver on a VTP centerline mesh (Python) |
| `convertparaview.py` | Convert NIfTI → VTI and `resistance_graph.mat` → VTP for ParaView |

---

## Requirements

- MATLAB R2021b or later
- Deep Learning Toolbox
- Image Processing Toolbox
- (Optional) Parallel Computing Toolbox — for GPU training

---

## Architecture

```
Input [H×W×1]
  │
  ├──────────────────────────────────────────────────────┐
  │                                                      │
  ▼                                                      │
LearnableFrangiLayer                                     │
  Learnable params:                                      │
    logSigmas [1×1×1×S]  — scale bank σ₁…σ_S            │
    logAlpha  (scalar)   — blob sensitivity              │
    logBeta   (scalar)   — background sensitivity        │
  Output: max-scale vesselness [H×W×1]                   │
  │                                                      │
  └──────────────► ConcatLayer ◄────────────────────────┘
                       │
                  [H×W×2]  (raw + vesselness)
                       │
             ┌─────────┴──────────┐
             │    U-Net Encoder   │
             │  Conv-BN-ReLU ×2  │
             │  MaxPool ×D        │
             └─────────┬──────────┘
                       │  Bottleneck
             ┌─────────┴──────────┐
             │    U-Net Decoder   │
             │  TransConv + Skip  │
             │  Conv-BN-ReLU ×2  │
             └─────────┬──────────┘
                       │
                  1×1 Conv → Sigmoid
                       │
               Dice + BCE Loss
```

---

## Quickstart

```matlab
% Run the self-contained demo (generates synthetic data)
demo_frangiUNet

% Train on your own data
opts.imgSize      = [512 512];
opts.numScales    = 6;
opts.sigmaMin     = 0.5;
opts.sigmaMax     = 6.0;
opts.encoderDepth = 4;
opts.initFilters  = 32;
opts.epochs       = 100;
opts.batchSize    = 8;

[net, info] = trainFrangiUNet('/path/to/images', '/path/to/labels', opts);

% Evaluate
results = evaluateFrangiUNet(net, '/path/to/test/images', '/path/to/test/labels');
```

---

---

## Vessel Analysis Pipeline

Two parallel workflows are supported downstream of segmentation.  Both start
from a binary vessel label volume (`.nii` / `.nii.gz`) and produce a 1-D
pressure/flow network that can be loaded directly in ParaView.

```
Binary vessel mask  (label.nii.gz)
         │
         ▼
  myskelotonize.m          ←── MATLAB skeletonization
         │
    skel.nii.gz
         │
    ┌────┴──────────────────────────────────────────────────────┐
    │  MATLAB workflow                                          │  Python workflow
    ▼                                                           ▼
resistanceLumping.m                                    skelcenterline.py
 • reduced graph (critical voxels only)                 • dense graph (every voxel = node)
 • resistance_graph.mat                                 • centerline.vtp
 • pressure_mmhg.nii.gz                                          │
         │                                             resistance_lumping.py
         ▼                                              • --label label.nii.gz
 convertparaview.py vtp                                 • pressure/flow on VTP
  • resistance_graph.mat → centerline.vtp                        │
 convertparaview.py vti                                          ▼
  • label.nii.gz         → label.vti               centerline_pressure.vtp
         │                                                       │
         └───────────────────────┬───────────────────────────────┘
                                 ▼
                             ParaView
```

---

### `myskelotonize.m` — Batch skeletonization

Reads a CSV manifest, extracts the binary vessel label from each NIfTI, runs
MATLAB `bwskel`, and writes compressed skeleton volumes under `newdata/`.

**Manifest format** (`manifest.csv`, no header required for this script):

| Column | Content |
|---|---|
| `label` | Path to the label NIfTI for that sample |

```matlab
% Edit MANIFEST / OUT_DIR / LABEL_VAL at the top of the file, then run:
myskelotonize
```

Outputs mirror the input directory structure under `newdata/`:
```
newdata/<original_dir>/<stem>_skel.nii.gz
```

**Requirements:** Image Processing Toolbox (`bwskel`, R2019a+)

---

### `skelcenterline.py` — Skeleton NIfTI → VTP centerline mesh

Every non-zero voxel becomes a mesh node at its world-coordinate voxel centre
(derived from the NIfTI affine).  Edges connect only 26-adjacent skeleton
voxels, so the mesh topology exactly mirrors the skeleton in the NIfTI.

```bash
python skelcenterline.py skel.nii.gz                   # → skel_centerline.vtp
python skelcenterline.py skel.nii.gz output.vtp        # explicit output path
```

**Dependencies:** `pip install nibabel vtk numpy`

---

### `resistance_lumping.py` — Hagen-Poiseuille solve on a VTP mesh

Reads the VTP from `skelcenterline.py`, looks up vessel radii from the label
NIfTI (or a pre-computed distance-transform), assembles a sparse Hagen-
Poiseuille conductance matrix, bridges disconnected components with phantom
gap edges, solves `K·p = q`, and writes pressure + flow back to a new VTP.

**Radius source (pick one):**

```bash
# Recommended — DT computed internally from the vessel label mask
python resistance_lumping.py centerline.vtp --label label.nii.gz

# Pre-computed DT NIfTI (radii in mm, e.g. exported from MATLAB bwdist × spacing)
python resistance_lumping.py centerline.vtp --dt dt_mm.nii.gz

# No label available — uniform radius fallback
python resistance_lumping.py centerline.vtp --default-radius 1.5
```

**All options:**

```
positional:
  vtp_in              Input .vtp from skelcenterline.py
  vtp_out             Output .vtp  [default: <stem>_pressure.vtp]

radius source:
  --label  FILE       Label NIfTI; DT of vessel mask computed internally
  --label-val INT     Vessel label value  [default: 2]
  --dt     FILE       Pre-computed DT NIfTI (radii in mm)
  --default-radius MM Uniform radius when --label / --dt omitted  [default: 1.0]

phantom gap edges:
  --gap-max MM        Maximum bridging distance  [default: 15]
  --alpha X           Phantom conductance penalty factor  [default: 10]
  --gap-mode          mst  — MST-minimal bridges, one per component pair (default)
                      knn  — all cross-component leaf pairs within --gap-max

solver:
  --p-in   MMHG       Inlet pressure   [default: 100]
  --p-out  MMHG       Outlet pressure  [default: 5]
  --mu     PA_S       Dynamic viscosity [default: 3.5e-3  (blood)]
```

**Output VTP arrays:**

| Array | Type | Description |
|---|---|---|
| `pressure_mmhg` | PointData | Nodal pressure (mmHg) |
| `radius_mm` | PointData | Vessel radius at each node (mm) |
| `flow_mm3s` | CellData | Volumetric flow per edge (mm³/s) |
| `conductance_SI` | CellData | Hagen-Poiseuille conductance (m³ Pa⁻¹ s⁻¹) |
| `radius_mm_edge` | CellData | Mean radius per edge (mm) |
| `length_mm` | CellData | Edge length (mm) |
| `is_phantom` | CellData | 0 = real edge, 1 = phantom gap edge |

**Dependencies:** `pip install nibabel vtk numpy scipy`

---

### `convertparaview.py` — MAT / NIfTI → ParaView formats

Used with the MATLAB workflow to convert `resistanceLumping.m` outputs to VTK
formats readable in ParaView.

```bash
# Convert label / skeleton NIfTI → VTK ImageData (.vti)
python convertparaview.py vti label.nii.gz
python convertparaview.py vti label.nii.gz output.vti

# Convert resistance_graph.mat → VTK PolyData (.vtp)
# Without --skel: straight lines between graph nodes
python convertparaview.py vtp resistance_graph.mat

# With --skel: polylines follow the actual skeleton voxel paths
# (nodes and edges co-registered with the VTI when both loaded in ParaView)
python convertparaview.py vtp resistance_graph.mat --skel skel.nii.gz
python convertparaview.py vtp resistance_graph.mat output.vtp --skel skel.nii.gz
```

The `--skel` flag re-derives node world coordinates from the stored 0-indexed
voxel indices via the NIfTI affine, and replaces each straight edge with a
`vtkPolyLine` tracing the full skeleton path — ensuring the VTP topology lies
exactly on the non-zero voxels of the VTI when both are registered in ParaView.

**Dependencies:** `pip install nibabel vtk numpy scipy`

---

## Learnable Frangi Layer — Design Notes

### Why log-space parameterisation?
`logSigmas = log(σ)` ensures σ > 0 without constraints. Gradients flow through
`σ = exp(logSigmas)`, which is smooth and well-conditioned.

### Eigenvalue computation
For a 2×2 symmetric Hessian H = [[Lxx Lxy]; [Lxy Lyy]], the closed-form eigenvalues are:

```
λ₁ = (Lxx+Lyy)/2 - sqrt(((Lxx-Lyy)/2)² + Lxy²)
λ₂ = (Lxx+Lyy)/2 + sqrt(((Lxx-Lyy)/2)² + Lxy²)
```

This is fully differentiable via `dlarray` arithmetic.

### Vesselness formula (2-D, bright vessels on dark background)
```
Rb = λ₁/λ₂                           (blob measure)
S² = λ₁² + λ₂²                        (structureness)
V  = exp(-Rb²/2α²) · (1 - exp(-S²/2β²))   [where λ₂ < 0]
```

### Kernel generation
Gaussian 2nd-derivative kernels are computed analytically from the (learned)
sigma values inside `forward()`. MATLAB's `dlconv` is used for convolution,
keeping the full computation graph intact.

---

## Loss Function

**Combined Dice + BCE:**

```
L = λ_dice · L_Dice + λ_bce · L_BCE
  = 0.7 · (1 - 2|P∩G|/(|P|+|G|)) + 0.3 · BCE(P, G)
```

Dice loss is essential for vessel segmentation due to extreme class imbalance
(vessels ≈ 3–10% of pixels). BCE alone leads to trivially empty predictions.

---

## Metrics

| Metric | Description |
|---|---|
| Dice | Standard F1 on binary masks |
| clDice | Centerline Dice — topology-aware, penalises missed branches |
| AUC-ROC | Threshold-independent ranking quality |

---

## Citation

If you use this code, please cite:

```
Frangi et al. (1998). Multiscale vessel enhancement filtering. MICCAI.
Ronneberger et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI.
Shit et al. (2021). clDice: A Novel Topology-Preserving Loss Function. CVPR.
```
