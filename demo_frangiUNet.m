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

VOL_SIZE = [96 96 80];   % [H W D] — must be >= patchSize on each axis

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
    'patchSize',          [64 64 64], ...  % 3-D patch — network is trained on this size
    'patchOverlap',       [8  8  8 ], ...  % inference overlap per side; stride = 48 48 48
    'patchesPerVolume',   4, ...           % patches per volume per epoch (demo: keep small)
    'foregroundFraction', 0.8, ...
    'numScales',          3, ...
    'sigmaMin',           1.0, ...
    'sigmaMax',           3.0, ...
    'encoderDepth',       3, ...    % shallow for demo speed
    'initFilters',        16, ...    % small for demo speed
    'lr',                 5e-4, ...
    'epochs',             100, ...
    'batchSize',          2, ...
    'l2',                 1e-4, ...
    'valFraction',        0.2, ...
    'plots',              'none' ...
);

%% ── 3. Train hybrid Frangi-UNet ─────────────────────────────────────────
fprintf('\n=== [1/2] Training hybrid Frangi-UNet ===\n');
rng(42);   % same initialisation for fair comparison
opts_hybrid           = base_opts;
opts_hybrid.useFrangi = true;
[net_hybrid, info_hybrid] = trainFrangiUNet(imgDir, labelDir, opts_hybrid);

%% ── 4. Train plain U-Net (control) ─────────────────────────────────────
fprintf('\n=== [2/2] Training plain U-Net (control) ===\n');
rng(42);
opts_plain           = base_opts;
opts_plain.useFrangi = false;
[net_plain, info_plain] = trainFrangiUNet(imgDir, labelDir, opts_plain);

%% ── 5. Inspect learned Frangi parameters (hybrid only) ─────────────────
fprintf('\n=== Learned 3-D Frangi parameters ===\n');
layerIdx = find(strcmp({net_hybrid.Layers.Name}, 'frangi'), 1);
if ~isempty(layerIdx)
    fl = net_hybrid.Layers(layerIdx);
    learned_sigmas = exp(double(fl.logSigmas(:)));
    learned_alpha  = exp(double(fl.logAlpha));
    learned_beta   = exp(double(fl.logBeta));
    learned_c      = exp(double(fl.logC));

    fprintf('  Learned sigmas : '); fprintf('%.3f  ', learned_sigmas); fprintf('\n');
    fprintf('  Learned alpha  : %.4f  (plate/tube discrimination)\n', learned_alpha);
    fprintf('  Learned beta   : %.4f  (blob suppression)\n',          learned_beta);
    fprintf('  Learned c      : %.2f  (background suppression)\n',    learned_c);
end

%% ── 6. Evaluate both models ─────────────────────────────────────────────
fprintf('\n=== Evaluating both models ===\n');

evalOpts.patchSize    = base_opts.patchSize;
evalOpts.patchOverlap = base_opts.patchOverlap;
evalOpts.threshold    = 0.5;

evalOpts.outDir = fullfile(rootdir, 'frangi_demo3d', 'preds_hybrid');
results_hybrid  = evaluateFrangiUNet(net_hybrid, imgDir, labelDir, evalOpts);

evalOpts.outDir = fullfile(rootdir, 'frangi_demo3d', 'preds_plain');
results_plain   = evaluateFrangiUNet(net_plain,  imgDir, labelDir, evalOpts);

%% ── 7. Accuracy comparison table ────────────────────────────────────────
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
        results_hybrid.meanDice  - results_plain.meanDice, ...
        results_hybrid.meanClDice - results_plain.meanClDice, ...
        results_hybrid.meanAUC   - results_plain.meanAUC);
fprintf('╚══════════════════════╩══════════╩══════════╩══════════╝\n');

%% ── 8. Side-by-side MIP predictions ────────────────────────────────────
fprintf('\n=== Visualising side-by-side predictions ===\n');

rng(0);
[vol_t, mask_t] = syntheticVesselVolume(VOL_SIZE);

% Sliding-window inference on the full test volume (patchSize may differ from VOL_SIZE)
vizOpts.patchSize    = base_opts.patchSize;
vizOpts.patchOverlap = base_opts.patchOverlap;
prob_hybrid = predictVolume(net_hybrid, vol_t, vizOpts);
prob_plain  = predictVolume(net_plain,  vol_t, vizOpts);
pred_hybrid = prob_hybrid >= 0.5;
pred_plain  = prob_plain  >= 0.5;

cols   = {'Input', 'GT mask', 'Hybrid prob', 'Hybrid mask', 'Plain prob', 'Plain mask'};
vols   = {vol_t, mask_t, prob_hybrid, pred_hybrid, prob_plain, pred_plain};
nCols  = numel(cols);

figure('Name','Model comparison — axial / coronal / sagittal MIPs','Color','w');
for c = 1:nCols
    v = vols{c};
    subplot(3, nCols, c);          imshow(squeeze(max(v,[],3)),[]); title([cols{c} ' (ax)']);
    subplot(3, nCols, c + nCols);  imshow(squeeze(max(v,[],2)),[]); title([cols{c} ' (cor)']);
    subplot(3, nCols, c + 2*nCols);imshow(squeeze(max(v,[],1)),[]); title([cols{c} ' (sag)']);
end

%% ── 9. Training-loss curves (both models) ──────────────────────────────
figure('Name','Training curves — hybrid vs plain','Color','w');
semilogy(info_hybrid.TrainingLoss,   'b-',  'LineWidth',1.5); hold on;
semilogy(info_plain.TrainingLoss,    'r-',  'LineWidth',1.5);
semilogy(info_hybrid.ValidationLoss, 'b--', 'LineWidth',1.5);
semilogy(info_plain.ValidationLoss,  'r--', 'LineWidth',1.5);
xlabel('Iteration'); ylabel('Loss (Dice + BCE, log scale)');
legend('Hybrid train','Plain train','Hybrid val','Plain val', 'Location','northeast');
grid on;
title('Hybrid Frangi-UNet vs Plain U-Net — Training Loss');

fprintf('\nDemo complete.\n');

%% =========================================================================
%% LOCAL HELPER: synthetic 3-D vessel volume generator
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

    SNR = 3;
    mysigma = .05;
    bg  = 0.1 + mysigma *randn(H, W, D);
    vol = single(bg + SNR*mysigma*single(mask) + mysigma *randn(H, W, D));
    vol = max(0, min(1, vol));
    mask = single(mask);
end

function v = clamp(v, lo, hi)
    v = max(lo, min(hi, v));
end
