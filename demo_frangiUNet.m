%% demo_frangiUNet.m
%  Ablation demo: hybrid 3-D Frangi-UNet vs plain 3-D U-Net (control).
%
%  This script:
%   1. Generates a small synthetic 3-D vessel dataset (no real data needed)
%   2. Trains BOTH networks on the same data (same opts, same seed)
%   3. Evaluates each and prints a side-by-side accuracy comparison
%   4. Visualises MIP predictions from both models on the same test volume
%
%  Tested on MATLAB R2023b / R2025b with:
%   - Deep Learning Toolbox
%   - Image Processing Toolbox

clear; clc; close all;
addpath(fileparts(mfilename('fullpath')));

rng(42);

%% ── 0. Generate synthetic 3-D dataset ──────────────────────────────────
fprintf('=== Generating synthetic 3-D vessel volumes ===\n');

VOL_SIZE = [48 48 40];   % [H W D] — must be >= patchSize on each axis

nTrain = 20;
nVal   = 5;
nTotal = nTrain + nVal;

rootdir = '.'
imgDir   = fullfile(rootdir, 'frangi_demo3d', 'images');
labelDir = fullfile(rootdir, 'frangi_demo3d', 'labels');
mkdir(imgDir); mkdir(labelDir);

for i = 1:nTotal
    [vol, mask] = syntheticVesselVolume(VOL_SIZE);
    niftiwrite(single(vol),  fullfile(imgDir,   sprintf('%04d.nii', i)));
    niftiwrite(single(mask), fullfile(labelDir, sprintf('%04d.nii', i)));
end
fprintf('  Saved %d volume/label pairs.\n', nTotal);

%% ── 1. Inspect a sample ─────────────────────────────────────────────────
fprintf('\n=== Visualising a sample volume (MIPs) ===\n');

[vol_s, mask_s] = syntheticVesselVolume(VOL_SIZE);

figure('Name','Sample training pair (MIPs)','Color','w');
subplot(2,3,1); imshow(squeeze(max(vol_s,  [],3)),[]); title('Input  — axial MIP');
subplot(2,3,2); imshow(squeeze(max(vol_s,  [],2)),[]); title('Input  — coronal MIP');
subplot(2,3,3); imshow(squeeze(max(vol_s,  [],1)),[]); title('Input  — sagittal MIP');
subplot(2,3,4); imshow(squeeze(max(mask_s, [],3)),[]); title('GT mask — axial MIP');
subplot(2,3,5); imshow(squeeze(max(mask_s, [],2)),[]); title('GT mask — coronal MIP');
subplot(2,3,6); imshow(squeeze(max(mask_s, [],1)),[]); title('GT mask — sagittal MIP');

%% ── 2. Shared training options ──────────────────────────────────────────
base_opts = struct( ...
    'patchSize',          [32 32 32], ...  % 3-D patch — network is trained on this size
    'patchOverlap',       [4  4  4 ], ...  % inference overlap per side; stride = 48 48 48
    'patchesPerVolume',   4, ...           % patches per volume per epoch (demo: keep small)
    'foregroundFraction', 0.8, ...
    'sigmaMin',           1.0, ...
    'sigmaMax',           2.0, ...
    'encoderDepth',       2, ...    % shallow for demo speed
    'initFilters',        8, ...    % small for demo speed
    'lr',                 5e-3, ...
    'epochs',             50, ...
    'batchSize',          2, ...
    'l2',                 1e-5, ...
    'valFraction',        0.2, ...
    'plots',              'none', ...
    'numFrangiChannels',  2 ...
);

%% ── 2b. Estimate logC from training-set Hessian magnitude ───────────────
fprintf('\n=== Computing Hessian magnitude statistics (training set) ===\n');

sigma_ref = sqrt(base_opts.sigmaMin * base_opts.sigmaMax);   % geometric mean scale
S_max_vals = zeros(nTrain, 1);
for i = 1:nTrain
    vol_i = single(niftiread(fullfile(imgDir, sprintf('%04d.nii', i))));
    lo = min(vol_i(:));  hi = max(vol_i(:));
    if hi > lo, vol_i = (vol_i - lo) / (hi - lo); end
    S_max_vals(i) = hessianFrobeniusMax(vol_i, sigma_ref);
end
S_max_global       = max(S_max_vals);
logC_init          = log(0.5 * S_max_global);
base_opts.logCInit = logC_init;
fprintf('  sigma_ref=%.3f  S_max=%.4e  logC_init=%.4f  (c_init=%.4e)\n', ...
        sigma_ref, S_max_global, logC_init, exp(logC_init));

%% ── 3. Train selected architectures ─────────────────────────────────────
% To run a subset, set RUN_MODELS to a cell array of archMode strings, e.g.:
%   RUN_MODELS = {'frangi_threshold'};
%   RUN_MODELS = {'unet', 'frangi_unet'};
% Leave empty to run all five.
RUN_MODELS = {'frangi_threshold'};   % <── edit here to select a subset

ALL_MODELS = {
    'Plain U-Net',              struct('archMode','unet'); ...
    'Hybrid Frangi-UNet',       struct('archMode','frangi_unet'); ...
    'Frangi threshold',         struct('archMode','frangi_threshold'); ...
    'Frangi + linear (1×1×1)',  struct('archMode','frangi_linear'); ...
    'Frangi multichannel',      struct('archMode','frangi_multichannel'); ...
};

if isempty(RUN_MODELS)
    models = ALL_MODELS;
else
    keep = cellfun(@(s) ismember(s.archMode, RUN_MODELS), ALL_MODELS(:,2));
    models = ALL_MODELS(keep, :);
end
nModels = size(models, 1);
assert(nModels > 0, 'RUN_MODELS did not match any known archMode.');

nets      = cell(nModels, 1);
infos     = cell(nModels, 1);
nets_init = cell(nModels, 1);   % untrained snapshots (only set for frangi_threshold)

for m = 1:nModels
    fprintf('\n=== [%d/%d] Training: %s ===\n', m, nModels, models{m,1});
    rng(42);
    mopts = base_opts;
    extra = models{m,2};
    flds  = fieldnames(extra);
    for f = 1:numel(flds), mopts.(flds{f}) = extra.(flds{f}); end

    % For frangi_threshold: assemble the untrained network before any weight
    % updates so we can evaluate the raw initialised Frangi+threshold response.
    if strcmp(mopts.archMode, 'frangi_threshold')
        initOpts          = mopts;
        initOpts.imgSize  = mopts.patchSize;
        lgraph_init    = buildFrangiUNet(initOpts);
        nets_init{m}   = assembleNetwork(lgraph_init);
        fprintf('  (untrained network assembled for baseline evaluation)\n');
    end

    [nets{m}, infos{m}] = trainFrangiUNet(imgDir, labelDir, mopts);
end

%% ── 4. Inspect learned Frangi parameters (any Frangi model that was run) ─
fprintf('\n=== Learned 3-D Frangi parameters ===\n');
frangiArchs = {'frangi_unet','frangi_threshold','frangi_linear','frangi_multichannel'};
frangiIdx   = findModel(models, frangiArchs);
for m = frangiIdx
    layerIdx = find(strcmp({nets{m}.Layers.Name}, 'frangi'), 1);
    if isempty(layerIdx), continue; end
    fprintf('  -- %s --\n', models{m,1});
    fl = nets{m}.Layers(layerIdx);
    for ch = 1:fl.NumChannels
        fprintf('  [ch %d] sigma=%.3f  alpha=%.4f  beta=%.4f  c=%.2f\n', ch, ...
            exp(double(fl.logSigmas(ch))), ...
            exp(double(fl.logAlpha(ch))), ...
            exp(double(fl.logBeta(ch))), ...
            exp(double(fl.logC(ch))));
    end
    % Show learnable threshold params if present
    tIdx = find(strcmp({nets{m}.Layers.Name}, 'scaled_sigmoid'), 1);
    if ~isempty(tIdx)
        tl = nets{m}.Layers(tIdx);
        fprintf('  learnable_threshold: threshold=%.4f  scale=%.2f\n', ...
            double(tl.bias), double(tl.scale));
    end
end

%% ── 5. Evaluate all trained models (+ untrained baselines) ──────────────
fprintf('\n=== Evaluating %d model(s) ===\n', nModels);

evalOpts.patchSize    = base_opts.patchSize;
evalOpts.patchOverlap = base_opts.patchOverlap;
evalOpts.threshold    = 0.5;   % all archs output sigmoid probabilities

results      = cell(nModels, 1);
results_init = cell(nModels, 1);   % non-empty only where nets_init is set

for m = 1:nModels
    am = models{m,2}.archMode;
    evalOpts.outDir = fullfile(rootdir, 'frangi_demo3d', sprintf('preds_%s', am));
    results{m} = evaluateFrangiUNet(nets{m}, imgDir, labelDir, evalOpts);

    if ~isempty(nets_init{m})
        evalOpts.outDir = fullfile(rootdir, 'frangi_demo3d', sprintf('preds_%s_init', am));
        results_init{m} = evaluateFrangiUNet(nets_init{m}, imgDir, labelDir, evalOpts);
    end
end

%% ── 6. Accuracy comparison table ────────────────────────────────────────
unetBoundary = findModel(models, {'frangi_unet'});   % separator row index
fprintf('\n');
fprintf('╔══════════════════════════════╦══════════╦══════════╦══════════╗\n');
fprintf('║ Model                        ║   Dice   ║  clDice  ║   AUC    ║\n');
fprintf('╠══════════════════════════════╬══════════╬══════════╬══════════╣\n');
for m = 1:nModels
    % Print untrained baseline row immediately before the trained row
    if ~isempty(results_init{m})
        fprintf('║ %-28s ║ %8.4f ║ %8.4f ║ %8.4f ║\n', ...
            [models{m,1} ' (init)'], ...
            results_init{m}.meanDice, results_init{m}.meanClDice, results_init{m}.meanAUC);
    end
    fprintf('║ %-28s ║ %8.4f ║ %8.4f ║ %8.4f ║\n', ...
        models{m,1}, results{m}.meanDice, results{m}.meanClDice, results{m}.meanAUC);
    if ~isempty(unetBoundary) && m == unetBoundary
        fprintf('╠══════════════════════════════╬══════════╬══════════╬══════════╣\n');
    end
end
fprintf('╚══════════════════════════════╩══════════╩══════════╩══════════╝\n');

%% ── 7. MIP visualisation ─────────────────────────────────────────────────
fprintf('\n=== Visualising probability maps ===\n');

rng(0);
[vol_t, mask_t] = syntheticVesselVolume(VOL_SIZE);

vizOpts.patchSize    = base_opts.patchSize;
vizOpts.patchOverlap = base_opts.patchOverlap;

probs      = cell(nModels, 1);
probs_init = cell(nModels, 1);
for m = 1:nModels
    probs{m} = predictVolume(nets{m}, vol_t, vizOpts);
    if ~isempty(nets_init{m})
        probs_init{m} = predictVolume(nets_init{m}, vol_t, vizOpts);
    end
end

% Axial MIP strip — trained models, with init column inserted before each
% model that has an untrained baseline
cols  = {'Input', 'GT mask'};
vdata = {vol_t, mask_t};
for m = 1:nModels
    if ~isempty(probs_init{m})
        cols{end+1}  = [models{m,1} ' (init)'];   %#ok<SAGROW>
        vdata{end+1} = probs_init{m};              %#ok<SAGROW>
    end
    cols{end+1}  = models{m,1};   %#ok<SAGROW>
    vdata{end+1} = probs{m};      %#ok<SAGROW>
end
nC    = numel(cols);
figure('Name','All trained models — axial MIP','Color','w');
for c = 1:nC
    subplot(1, nC, c);
    imshow(squeeze(max(vdata{c}, [], 3)), []);
    title(cols{c}, 'Interpreter','none', 'FontSize',7);
end
sgtitle('Axial MIP — vesselness probability');

% Full ax/cor/sag figure — only if both U-Net models were trained
hiIdx   = findModel(models, {'frangi_unet'});
plainIdx = findModel(models, {'unet'});
if ~isempty(hiIdx) && ~isempty(plainIdx)
    figure('Name','U-Net models — full MIP comparison','Color','w');
    unetCols  = {'Input','GT mask','Hybrid Frangi-UNet','Plain U-Net'};
    unetVdata = {vol_t, mask_t, probs{hiIdx}, probs{plainIdx}};
    for c = 1:4
        v = unetVdata{c};
        subplot(3,4,c);   imshow(squeeze(max(v,[],3)),[]); title([unetCols{c} ' (ax)']);
        subplot(3,4,c+4); imshow(squeeze(max(v,[],2)),[]); title([unetCols{c} ' (cor)']);
        subplot(3,4,c+8); imshow(squeeze(max(v,[],1)),[]); title([unetCols{c} ' (sag)']);
    end
end

%% ── 8. Training-loss curves ──────────────────────────────────────────────
colors = lines(nModels);
styles = {'-','--'};

figure('Name','Training loss — all models','Color','w');
subplot(1,2,1); hold on;
subplot(1,2,2); hold on;

for m = 1:nModels
    tl = infos{m}.TrainingLoss;
    vl = infos{m}.ValidationLoss(~isnan(infos{m}.ValidationLoss));
    subplot(1,2,1);
    semilogy(tl, 'Color',colors(m,:), 'LineStyle',styles{1}, 'LineWidth',1.5, ...
             'DisplayName', models{m,1});
    subplot(1,2,2);
    semilogy(vl, 'Color',colors(m,:), 'LineStyle',styles{1}, 'LineWidth',1.5, ...
             'DisplayName', models{m,1});
end

subplot(1,2,1);
xlabel('Iteration'); ylabel('Loss'); title('Training loss'); grid on; legend('Location','northeast');
subplot(1,2,2);
xlabel('Validation check'); ylabel('Loss'); title('Validation loss'); grid on; legend('Location','northeast');

fprintf('\nDemo complete.\n');

%% =========================================================================
%% LOCAL HELPERS
%% =========================================================================

function idx = findModel(models, archModes)
% Return row indices in models whose archMode is in archModes (cell of strings).
% Returns empty if none of the requested archs were trained.
    if ischar(archModes), archModes = {archModes}; end
    idx = find(cellfun(@(s) ismember(s.archMode, archModes), models(:,2)));
end

%% ── synthetic 3-D vessel volume generator ───────────────────────────────
%% =========================================================================

function [vol, mask] = syntheticVesselVolume(sz)
% Generates a noisy 3-D volume with random tubular vessels (Bezier tubes).
%
%   sz   – [H W D] volume size
%   vol  – single [H W D] intensity volume in [0,1]
%   mask – single [H W D] binary vessel mask

    H = sz(1); W = sz(2); D = sz(3);

    mask = false(H, W, D);
    [gX, gY, gZ] = ndgrid(1:H, 1:W, 1:D);

    nVessels = randi([2 5]);
    for v = 1:nVessels
        radius = randi([1 2]);

        % Random quadratic Bezier control points in 3-D
        x0 = randi(H); y0 = randi(W); z0 = randi(D);
        x1 = randi(H); y1 = randi(W); z1 = randi(D);
        xm = clamp(round((x0+x1)/2 + randn*8),  1, H);
        ym = clamp(round((y0+y1)/2 + randn*8),  1, W);
        zm = clamp(round((z0+z1)/2 + randn*4),  1, D);

        t  = linspace(0, 1, 200);
        xp = clamp(round((1-t).^2*x0 + 2*(1-t).*t*xm + t.^2*x1), 1, H);
        yp = clamp(round((1-t).^2*y0 + 2*(1-t).*t*ym + t.^2*y1), 1, W);
        zp = clamp(round((1-t).^2*z0 + 2*(1-t).*t*zm + t.^2*z1), 1, D);

        for p = 1:numel(xp)
            dist = sqrt((gX-xp(p)).^2 + (gY-yp(p)).^2 + (gZ-zp(p)).^2);
            mask = mask | (dist <= radius);
        end
    end

    SNR = 25;
    mysigma = .05;
    bg  = 0.1 + mysigma *randn(H, W, D);
    vol = single(bg + SNR*mysigma*single(mask) + mysigma *randn(H, W, D));
    vol = max(0, min(10, vol));
    mask = single(mask);
end

function v = clamp(v, lo, hi)
    v = max(lo, min(hi, v));
end

%% ── Hessian Frobenius-norm maximum ──────────────────────────────────────

function S_max = hessianFrobeniusMax(vol, sigma)
% Returns the maximum scale-normalised Hessian Frobenius norm over the volume.
% S² = Lxx² + Lyy² + Lzz² + 2*(Lxy² + Lxz² + Lyz²)  (= sum of squared eigenvalues)
    ks = max(5, 2*ceil(3*sigma)+1);
    r  = floor(ks/2);
    [x, y, z] = meshgrid(-r:r, -r:r, -r:r);
    G   = exp(-(x.^2 + y.^2 + z.^2) / (2*sigma^2)) / (2*pi*sigma^2)^(3/2);
    sc  = sigma^2;   % scale normalisation

    Lxx = sc * imfilter(vol, G .* (x.^2/sigma^4 - 1/sigma^2), 'replicate');
    Lyy = sc * imfilter(vol, G .* (y.^2/sigma^4 - 1/sigma^2), 'replicate');
    Lzz = sc * imfilter(vol, G .* (z.^2/sigma^4 - 1/sigma^2), 'replicate');
    Lxy = sc * imfilter(vol, G .* (x.*y/sigma^4),              'replicate');
    Lxz = sc * imfilter(vol, G .* (x.*z/sigma^4),              'replicate');
    Lyz = sc * imfilter(vol, G .* (y.*z/sigma^4),              'replicate');

    S2    = Lxx.^2 + Lyy.^2 + Lzz.^2 + 2*(Lxy.^2 + Lxz.^2 + Lyz.^2);
    S_max = sqrt(double(max(S2(:))));
end
