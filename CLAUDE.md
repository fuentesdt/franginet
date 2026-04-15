# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A MATLAB implementation of a hybrid vessel segmentation architecture that combines a **learnable multi-scale 3-D Frangi vesselness filter** with a **3-D U-Net encoder-decoder**, trained end-to-end via backpropagation. Designed for 3-D grayscale medical volumes (CT, MRI, etc.).

## Requirements

- MATLAB R2021b or later
- Deep Learning Toolbox
- Image Processing Toolbox (`imresize3`, `bwskel`)
- (Optional) Parallel Computing Toolbox — for GPU training

## Data format

Input volumes and labels are NIfTI files (`.nii` or `.nii.gz`), one per sample:
- `imgDir/*.nii` — single-precision float intensity volume `[H W D]`
- `labelDir/*.nii` — binary mask `[H W D]` (any numeric type; thresholded at 0.5 on load)

Files are paired by sorted order within each directory; filenames do not need to match.

## Running the code

```matlab
% Self-contained demo on synthetic 3-D data — no real data needed
demo_frangiUNet

% Train on your own data
[net, info] = trainFrangiUNet('/path/to/volumes', '/path/to/masks', opts);

% Evaluate a trained network
results = evaluateFrangiUNet(net, '/path/to/test/volumes', '/path/to/test/masks');
```

Default training opts (all fields optional, merged with defaults in `trainFrangiUNet`):
```matlab
opts.imgSize      = [64 64 32];  % spatial size [H W D]
opts.numScales    = 4;           % Frangi scale levels
opts.sigmaMin     = 1.0;         % minimum Gaussian sigma (voxels)
opts.sigmaMax     = 4.0;         % maximum Gaussian sigma (voxels)
opts.encoderDepth = 3;           % U-Net encoder depth (2^depth <= min(H,W,D))
opts.initFilters  = 16;          % filters in first encoder block
opts.epochs       = 50;
opts.batchSize    = 2;           % keep small — 3-D volumes are memory-heavy
opts.lr           = 1e-3;
opts.l2           = 1e-4;
opts.valFraction  = 0.15;
```

## Architecture

Two parallel branches feed into a shared 3-D U-Net:

1. **Raw input** `[H×W×D×1]` — passed directly to the concatenation layer.
2. **LearnableFrangiLayer** — computes differentiable 3-D multi-scale vesselness `[H×W×D×1]`.

These are concatenated along the channel dimension (dim 4) to `[H×W×D×2]` before entering the U-Net encoder.

**Encoder**: `encoderDepth` blocks of `Conv3D(3×3×3)`-BN-ReLU × 2, then `MaxPool3D(2×2×2)`. Filter count doubles each block starting from `initFilters`.

**Bottleneck**: same Conv3D-BN-ReLU × 2 pattern.

**Decoder**: `TransposedConv3D` upsample → skip-connection concat (along dim 4) → Conv3D-BN-ReLU × 2, mirroring the encoder.

**Output head**: `1×1×1 Conv3D` → Sigmoid → `dicePixelClassificationLayer` (Dice + BCE loss).

The DAG is assembled in `buildFrangiUNet.m` using `layerGraph` / `addLayers` / `connectLayers`.

## Key design decisions

### Learnable 3-D Frangi parameters (learnableFrangiLayer.m)
Four learnable parameters in log-space: `logSigmas` (scale bank), `logAlpha` (plate/tube discrimination), `logBeta` (blob suppression), `logC` (background/noise suppression). Log-space ensures positivity without constraints.

The 3-D Frangi formula requires 6 Hessian components (Lxx, Lxy, Lxz, Lyy, Lyz, Lzz), computed via 3-D Gaussian 2nd-derivative convolution (`dlConv3`/`dlconv`). Eigenvalues of the 3×3 symmetric Hessian are computed analytically per-voxel using **Cardano's trigonometric method** (`eigenvalues3x3sym`), which is fully differentiable via dlarray arithmetic.

Eigenvalues are returned value-sorted descending (ev1 ≥ ev2 ≥ ev3). For bright tubular structures: ev3 ≤ ev2 < 0 ≈ ev1. The vessel mask (ev2 < 0 AND ev3 < 0) and the eigenvalue sort use a straight-through pattern (`extractdata` breaks the auto-diff graph for those operations only — the gradient still flows through the eigenvalue expressions themselves).

### Loss function (dicePixelClassificationLayer.m)
Extends `nnet.layer.RegressionLayer`. Both `forwardLoss` and `backwardLoss` are implemented explicitly. Batch size is determined via `size(Y, ndims(Y))` so the layer is agnostic to spatial rank (works for both 2-D and 3-D tensors). Default split: 70% Dice + 30% BCE.

### Data loading (trainFrangiUNet.m)
Uses `fileDatastore` (with a custom `ReadFcn` that loads the first `.mat` variable) combined via `combine()` and `transform()`. This mirrors the 2-D `imageDatastore` pattern while supporting volumetric `.mat` files. Each preprocessed sample has shape `[H W D 1]`.

### Evaluation metrics (evaluateFrangiUNet.m)
- **Dice** — standard F1 on binary masks
- **clDice** — centerline Dice via `bwskel` (supports 3-D natively in MATLAB R2019a+); topology-aware, penalises missed vessel branches
- **AUC-ROC** — threshold-independent, uses MATLAB's `perfcurve`

Outputs are saved as `.mat` files (`_prob.mat`, `_mask.mat`) in `opts.outDir`.
