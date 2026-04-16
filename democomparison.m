%% democomparison.m
%  -----------------------------------------------------------------------
%  Hybrid Frangi-UNet vs plain U-Net — real NIfTI data comparison
%  -----------------------------------------------------------------------
%
%  Workflow:
%    1. Read a CSV manifest listing image / mask NIfTI pairs
%    2. Survey voxel spacings and resample everything to a common spacing
%    3. Split samples into a train set and a held-out test set
%    4. Train both architectures (same hyperparameters, same random seed)
%    5. Evaluate on the held-out test set
%    6. Print an accuracy table and show MIP visualisations
%
%  CSV format  (no header row):
%    column 1 — sample ID  (string or number; used for file naming)
%    column 2 — path to binary mask   (.nii or .nii.gz)
%    column 3 — path to intensity image  (.nii or .nii.gz)
%  Paths may be absolute or relative to the directory containing the CSV.
%
%  Requires: Deep Learning Toolbox, Image Processing Toolbox
%            (niftiread, niftiwrite, niftiinfo, imresize3)
%  -----------------------------------------------------------------------

clear; clc; close all;
addpath(fileparts(mfilename('fullpath')));

%% ── Configuration ────────────────────────────────────────────────────────
CSV_FILE    = 'trainingdata.csv';
OUTPUT_DIR  = 'frangi_comparison';

% Target voxel spacing (mm per axis).
% Leave empty [] to use the per-axis median across the dataset.
TARGET_SPACING = [];

% Network input patch size [H W D] fed to the U-Net.
% Leave empty [] to derive automatically from the median resampled volume.
TARGET_NET_SIZE = [];

% Fraction of samples withheld as a held-out test set.
TEST_FRACTION = 0.20;

% Shared hyperparameters for both training runs.
TRAIN_OPTS = struct( ...
    'numScales',    4,      ...
    'sigmaMin',     1.0,    ...
    'sigmaMax',     4.0,    ...
    'encoderDepth', 3,      ...
    'initFilters',  16,     ...
    'lr',           1e-3,   ...
    'epochs',       50,     ...
    'batchSize',    2,      ...
    'l2',           1e-4,   ...
    'valFraction',  0.15,   ...
    'plots',        'none'  ...
);

rng(42);

%% ── 1. Load CSV manifest ─────────────────────────────────────────────────
fprintf('=== Loading manifest: %s ===\n', CSV_FILE);

T = readtable(CSV_FILE, 'ReadVariableNames', false);
assert(width(T) >= 3, 'CSV must have at least 3 columns: id, mask, image.');

ids       = string(T{:,1});
maskPaths = string(T{:,2});
imgPaths  = string(T{:,3});
N         = height(T);

fprintf('  Found %d samples.\n', N);
assert(N >= 4, 'Need at least 4 samples for a meaningful train/test split.');

% Resolve relative paths against the CSV file's own directory.
csvDir    = fileparts(makeAbsolute(CSV_FILE));
imgPaths  = resolveFilePaths(imgPaths,  csvDir);
maskPaths = resolveFilePaths(maskPaths, csvDir);

%% ── 2. Survey voxel spacings ─────────────────────────────────────────────
fprintf('\n=== Surveying voxel spacings ===\n');

allSpacings = zeros(N, 3);
for k = 1:N
    allSpacings(k,:) = getVoxelSpacing(niftiinfo(imgPaths(k)));
end

if isempty(TARGET_SPACING)
    TARGET_SPACING = median(allSpacings, 1);
end
fprintf('  Min spacing  : [%.3f  %.3f  %.3f] mm\n', min(allSpacings));
fprintf('  Median spacing: [%.3f  %.3f  %.3f] mm\n', median(allSpacings));
fprintf('  Max spacing  : [%.3f  %.3f  %.3f] mm\n', max(allSpacings));
fprintf('  → Resampling to: [%.3f  %.3f  %.3f] mm\n', TARGET_SPACING);

%% ── 3. Resample and write organised train / test directories ─────────────
trainImgDir  = fullfile(OUTPUT_DIR, 'train', 'images');
trainMaskDir = fullfile(OUTPUT_DIR, 'train', 'labels');
testImgDir   = fullfile(OUTPUT_DIR, 'test',  'images');
testMaskDir  = fullfile(OUTPUT_DIR, 'test',  'labels');
for d = {trainImgDir, trainMaskDir, testImgDir, testMaskDir}
    mkdir(d{1});
end

% Reproducible shuffle → split
shuffleIdx = randperm(N);
nTest      = max(1, round(N * TEST_FRACTION));
nTrain     = N - nTest;
trainIdx   = sort(shuffleIdx(1:nTrain));
testIdx    = sort(shuffleIdx(nTrain+1:end));

fprintf('\n=== Resampling %d volumes → train: %d  test: %d ===\n', ...
        N, nTrain, nTest);

resampledSizes = zeros(N, 3);

for k = 1:N
    % Intensity image — trilinear interpolation
    [imgRs, ~] = resampleNifti(imgPaths(k),  TARGET_SPACING, 'linear');
    imgRs = single(imgRs);

    % Binary mask — nearest-neighbour, then re-threshold
    [mskRs, ~] = resampleNifti(maskPaths(k), TARGET_SPACING, 'nearest');
    mskRs = single(mskRs > 0.5);

    resampledSizes(k,:) = size(imgRs, [1 2 3]);

    % Normalise intensity to [0,1] per-volume
    lo = min(imgRs(:));  hi = max(imgRs(:));
    if hi > lo
        imgRs = (imgRs - lo) / (hi - lo);
    end

    fname = sprintf('%s.nii', ids(k));
    if ismember(k, trainIdx)
        niftiwrite(imgRs, fullfile(trainImgDir,  fname));
        niftiwrite(mskRs, fullfile(trainMaskDir, fname));
    else
        niftiwrite(imgRs, fullfile(testImgDir,   fname));
        niftiwrite(mskRs, fullfile(testMaskDir,  fname));
    end

    fprintf('  [%d/%d] %-20s → [%3d %3d %3d] vox\n', ...
            k, N, ids(k), resampledSizes(k,1), resampledSizes(k,2), resampledSizes(k,3));
end

%% ── 4. Determine network input size ─────────────────────────────────────
if isempty(TARGET_NET_SIZE)
    medSz = median(resampledSizes, 1);
    step  = 2^TRAIN_OPTS.encoderDepth;   % dims must be divisible by 2^depth
    TARGET_NET_SIZE = max(step, floor(medSz / step) * step);
end
fprintf('\nNetwork input size (imgSize): [%d %d %d]\n', TARGET_NET_SIZE);
TRAIN_OPTS.imgSize = TARGET_NET_SIZE;

%% ── 5. Train hybrid Frangi-UNet ──────────────────────────────────────────
fprintf('\n=== [1/2] Training hybrid Frangi-UNet ===\n');
rng(42);
opts_hybrid           = TRAIN_OPTS;
opts_hybrid.useFrangi = true;
[net_hybrid, info_hybrid] = trainFrangiUNet(trainImgDir, trainMaskDir, opts_hybrid);

%% ── 6. Train plain U-Net (control) ───────────────────────────────────────
fprintf('\n=== [2/2] Training plain U-Net (control) ===\n');
rng(42);
opts_plain           = TRAIN_OPTS;
opts_plain.useFrangi = false;
[net_plain, info_plain] = trainFrangiUNet(trainImgDir, trainMaskDir, opts_plain);

%% ── 7. Inspect learned Frangi parameters ────────────────────────────────
fprintf('\n=== Learned 3-D Frangi parameters ===\n');
layerIdx = find(strcmp({net_hybrid.Layers.Name}, 'frangi'), 1);
if ~isempty(layerIdx)
    fl = net_hybrid.Layers(layerIdx);
    fprintf('  sigmas : '); fprintf('%.3f  ', exp(double(fl.logSigmas(:)))); fprintf('\n');
    fprintf('  alpha  : %.4f  (plate/tube)\n',  exp(double(fl.logAlpha)));
    fprintf('  beta   : %.4f  (blob)\n',         exp(double(fl.logBeta)));
    fprintf('  c      : %.2f  (background)\n',   exp(double(fl.logC)));
end

%% ── 8. Evaluate on held-out test set ─────────────────────────────────────
fprintf('\n=== Evaluating on %d held-out test volumes ===\n', nTest);

evalOpts.imgSize   = TARGET_NET_SIZE;
evalOpts.threshold = 0.5;

evalOpts.outDir = fullfile(OUTPUT_DIR, 'preds_hybrid');
results_hybrid  = evaluateFrangiUNet(net_hybrid, testImgDir, testMaskDir, evalOpts);

evalOpts.outDir = fullfile(OUTPUT_DIR, 'preds_plain');
results_plain   = evaluateFrangiUNet(net_plain,  testImgDir, testMaskDir, evalOpts);

%% ── 9. Accuracy comparison table ─────────────────────────────────────────
fprintf('\n');
fprintf('╔══════════════════════╦══════════╦══════════╦══════════╗\n');
fprintf('║ Model                ║   Dice   ║  clDice  ║   AUC    ║\n');
fprintf('╠══════════════════════╬══════════╬══════════╬══════════╣\n');
fprintf('║ Hybrid Frangi-UNet   ║ %8.4f ║ %8.4f ║ %8.4f ║\n', ...
        results_hybrid.meanDice, results_hybrid.meanClDice, results_hybrid.meanAUC);
fprintf('║ Plain U-Net          ║ %8.4f ║ %8.4f ║ %8.4f ║\n', ...
        results_plain.meanDice,  results_plain.meanClDice,  results_plain.meanAUC);
fprintf('╠══════════════════════╬══════════╬══════════╬══════════╣\n');
fprintf('║ Δ (hybrid − plain)   ║ %+8.4f ║ %+8.4f ║ %+8.4f ║\n', ...
        results_hybrid.meanDice   - results_plain.meanDice,   ...
        results_hybrid.meanClDice - results_plain.meanClDice, ...
        results_hybrid.meanAUC    - results_plain.meanAUC);
fprintf('╚══════════════════════╩══════════╩══════════╩══════════╝\n');

%% ── 10. Per-sample Dice scatter ─────────────────────────────────────────
figure('Name','Per-sample Dice: hybrid vs plain','Color','w');
scatter(results_plain.dice, results_hybrid.dice, 60, 'filled'); hold on;
lims = [0 1];
plot(lims, lims, 'k--', 'LineWidth', 1);
xlabel('Plain U-Net Dice');
ylabel('Hybrid Frangi-UNet Dice');
title(sprintf('Per-sample Dice on %d test volumes\n(above diagonal = Frangi helps)', nTest));
grid on; axis square; xlim(lims); ylim(lims);

%% ── 11. MIP visualisation — best-delta test case ─────────────────────────
fprintf('\n=== Visualising test case with largest Dice improvement ===\n');

[~, bestIdx] = max(results_hybrid.dice - results_plain.dice);

testImgFiles  = sortedNiiList(testImgDir);
testMaskFiles = sortedNiiList(testMaskDir);
hybridProbs   = sortedNiiList(fullfile(OUTPUT_DIR, 'preds_hybrid'), '_prob');
plainProbs    = sortedNiiList(fullfile(OUTPUT_DIR, 'preds_plain'),  '_prob');

vol_t       = single(niftiread(fullfile(testImgDir,  testImgFiles(bestIdx).name)));
gt_mask     = single(niftiread(fullfile(testMaskDir, testMaskFiles(bestIdx).name)));
prob_hybrid = double(niftiread(fullfile(OUTPUT_DIR, 'preds_hybrid', hybridProbs(bestIdx).name)));
prob_plain  = double(niftiread(fullfile(OUTPUT_DIR, 'preds_plain',  plainProbs(bestIdx).name)));
pred_hybrid = prob_hybrid >= 0.5;
pred_plain  = prob_plain  >= 0.5;

cols  = {'Input', 'GT mask', 'Hybrid prob', 'Hybrid mask', 'Plain prob', 'Plain mask'};
vdata = {vol_t, gt_mask, prob_hybrid, pred_hybrid, prob_plain, pred_plain};
nC    = numel(cols);

figure('Name', sprintf('Test case %d — Dice: hybrid=%.3f  plain=%.3f', ...
       bestIdx, results_hybrid.dice(bestIdx), results_plain.dice(bestIdx)), 'Color','w');
for c = 1:nC
    v = vdata{c};
    subplot(3,nC,c);        imshow(squeeze(max(v,[],3)),[]); title([cols{c} ' ax']);
    subplot(3,nC,c+nC);     imshow(squeeze(max(v,[],2)),[]); title([cols{c} ' cor']);
    subplot(3,nC,c+2*nC);   imshow(squeeze(max(v,[],1)),[]); title([cols{c} ' sag']);
end

%% ── 12. Training-loss curves ─────────────────────────────────────────────
figure('Name','Training loss — hybrid vs plain','Color','w');
semilogy(info_hybrid.TrainingLoss,   'b-',  'LineWidth',1.5); hold on;
semilogy(info_plain.TrainingLoss,    'r-',  'LineWidth',1.5);
semilogy(info_hybrid.ValidationLoss, 'b--', 'LineWidth',1.5);
semilogy(info_plain.ValidationLoss,  'r--', 'LineWidth',1.5);
xlabel('Iteration'); ylabel('Loss (Dice + BCE, log scale)');
legend('Hybrid train','Plain train','Hybrid val','Plain val','Location','northeast');
grid on;
title('Training loss: Hybrid Frangi-UNet vs Plain U-Net');

fprintf('\nOutputs written to: %s\n', makeAbsolute(OUTPUT_DIR));
fprintf('Demo complete.\n');

%% =========================================================================
%% LOCAL HELPERS
%% =========================================================================

function paths = resolveFilePaths(paths, baseDir)
% Resolve each path: if it doesn't exist as given, try relative to baseDir.
    for k = 1:numel(paths)
        if ~isfile(paths(k))
            candidate = fullfile(baseDir, paths(k));
            if isfile(candidate)
                paths(k) = string(candidate);
            else
                warning('democomparison:missingFile', ...
                    'Cannot find file:\n  %s\n  (also tried: %s)', paths(k), candidate);
            end
        end
    end
end

% -------------------------------------------------------------------------
function spacing = getVoxelSpacing(info)
% Extract [dx dy dz] voxel spacing (mm) from a niftiinfo struct.
% Falls back to 1 mm for missing dimensions.
    pd = double(info.PixelDimensions);
    switch numel(pd)
        case 0,  spacing = [1 1 1];
        case 1,  spacing = [pd(1) pd(1) pd(1)];
        case 2,  spacing = [pd(1:2), 1];
        otherwise, spacing = pd(1:3);
    end
end

% -------------------------------------------------------------------------
function [vol_rs, outSpacing] = resampleNifti(filepath, targetSpacing, method)
% Resample a NIfTI volume to targetSpacing (mm) using the given interpolation.
    info      = niftiinfo(filepath);
    vol       = single(niftiread(filepath));

    inSpacing = getVoxelSpacing(info);
    inSize    = size(vol, [1 2 3]);
    outSize   = max(1, round(inSize .* inSpacing ./ targetSpacing));
    outSpacing = targetSpacing;

    if all(outSize == inSize)
        vol_rs = vol;   % already at target spacing
    else
        vol_rs = imresize3(vol, outSize, 'Method', method);
    end
end

% -------------------------------------------------------------------------
function p = makeAbsolute(p)
% Return the absolute path of p (resolves relative to pwd).
    [status, attr] = fileattrib(p);
    if status
        p = attr.Name;
    else
        p = fullfile(pwd, p);
    end
end

% -------------------------------------------------------------------------
function files = sortedNiiList(folder, suffix)
% Return dir() entries for .nii and .nii.gz files in folder,
% optionally filtered by a filename suffix string, sorted by name.
    if nargin < 2, suffix = ''; end
    files = [dir(fullfile(folder, ['*' suffix '*.nii'])); ...
             dir(fullfile(folder, ['*' suffix '*.nii.gz']))];
    if isempty(files), return; end
    [~, order] = sort({files.name});
    files = files(order);
end
