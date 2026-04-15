# Hybrid Learnable-Frangi + U-Net — MATLAB Implementation

A differentiable vessel segmentation architecture combining a **learnable multi-scale Frangi vesselness filter** with a **U-Net encoder-decoder**, trained end-to-end with a combined Dice + BCE loss.

---

## Files

| File | Purpose |
|---|---|
| `trainFrangiUNet.m` | Top-level training entry point |
| `buildFrangiUNet.m` | DAG network graph constructor |
| `learnableFrangiLayer.m` | Custom `nnet.layer` — differentiable Frangi |
| `dicePixelClassificationLayer.m` | Custom loss layer — Dice + BCE |
| `evaluateFrangiUNet.m` | Inference + Dice / clDice / AUC metrics |
| `demo_frangiUNet.m` | Self-contained demo on synthetic data |

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
