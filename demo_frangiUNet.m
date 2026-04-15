%% demo_frangiUNet.m
%  End-to-end demo of the hybrid learnable-Frangi + U-Net pipeline.
%
%  This script:
%   1. Generates a small synthetic vessel dataset (no real data needed)
%   2. Trains the hybrid network for a handful of epochs
%   3. Evaluates and visualises results
%
%  Tested on MATLAB R2023b with:
%   - Deep Learning Toolbox
%   - Image Processing Toolbox

clear; clc; close all;
addpath(fileparts(mfilename('fullpath')));

rng(42);

%% ── 0. Generate synthetic dataset ──────────────────────────────────────
fprintf('=== Generating synthetic vessel images ===\n');

nTrain = 40;
nVal   = 10;
nTotal = nTrain + nVal;

imgDir   = fullfile(tempdir, 'frangi_demo', 'images');
labelDir = fullfile(tempdir, 'frangi_demo', 'labels');
mkdir(imgDir); mkdir(labelDir);

for i = 1:nTotal
    [img, mask] = syntheticVesselImage(256, 256);
    imwrite(im2uint8(img),  fullfile(imgDir,   sprintf('%04d.png',i)));
    imwrite(im2uint8(mask), fullfile(labelDir, sprintf('%04d.png',i)));
end
fprintf('  Saved %d image/label pairs.\n', nTotal);

%% ── 1. Inspect a sample ─────────────────────────────────────────────────
figure('Name','Sample training pair','Color','w');
[img_s, mask_s] = syntheticVesselImage(256,256);

% Standalone Frangi for comparison
V = classicFrangi(img_s, [1 2 3 4]);

subplot(1,3,1); imshow(img_s,[]); title('Input image');
subplot(1,3,2); imshow(V,[]);     title('Classic Frangi (fixed \sigma)');
subplot(1,3,3); imshow(mask_s,[]); title('Ground truth mask');

%% ── 2. Train hybrid network ─────────────────────────────────────────────
fprintf('\n=== Training hybrid Frangi-UNet ===\n');

opts = struct( ...
    'imgSize',      [256 256], ...
    'numScales',    4, ...
    'sigmaMin',     1.0, ...
    'sigmaMax',     4.0, ...
    'encoderDepth', 3, ...    % depth 3 for 256px demo (depth 4 also fine)
    'initFilters',  16, ...   % smaller for demo speed
    'lr',           5e-4, ...
    'epochs',       10, ...   % increase to 50+ for real training
    'batchSize',    4, ...
    'l2',           1e-4, ...
    'valFraction',  0.2, ...
    'plots',        'none' ...
);

[trainedNet, trainInfo] = trainFrangiUNet(imgDir, labelDir, opts);

%% ── 3. Inspect learned Frangi parameters ───────────────────────────────
fprintf('\n=== Learned Frangi parameters ===\n');
frangiLayer = trainedNet.Layers(strcmp({trainedNet.Layers.Name},'frangi'));
if ~isempty(frangiLayer)
    learned_sigmas = exp(double(extractdata(frangiLayer.logSigmas(:))));
    learned_alpha  = exp(double(extractdata(frangiLayer.logAlpha)));
    learned_beta   = exp(double(extractdata(frangiLayer.logBeta)));

    fprintf('  Learned sigmas : '); fprintf('%.3f  ', learned_sigmas); fprintf('\n');
    fprintf('  Learned alpha  : %.4f\n', learned_alpha);
    fprintf('  Learned beta   : %.4f\n', learned_beta);
end

%% ── 4. Visualise predictions ────────────────────────────────────────────
fprintf('\n=== Visualising predictions ===\n');

[img_t, mask_t] = syntheticVesselImage(256,256);
X_t    = reshape(im2single(img_t), [256 256 1 1]);
prob_t = squeeze(double(extractdata(predict(trainedNet, X_t))));
pred_t = prob_t >= 0.5;

V_t = classicFrangi(img_t, [1 2 3 4]);

figure('Name','Prediction comparison','Color','w');
subplot(1,4,1); imshow(img_t,[]);    title('Input');
subplot(1,4,2); imshow(V_t,[]);      title('Classic Frangi');
subplot(1,4,3); imshow(prob_t,[]);   title('Hybrid net (prob)');
subplot(1,4,4); imshow(pred_t,[]);   title('Hybrid net (binary)');

%% ── 5. Evaluate ─────────────────────────────────────────────────────────
evalOpts.imgSize   = [256 256];
evalOpts.threshold = 0.5;
evalOpts.outDir    = fullfile(tempdir, 'frangi_demo', 'preds');

results = evaluateFrangiUNet(trainedNet, imgDir, labelDir, evalOpts);

%% ── 6. Training curve ───────────────────────────────────────────────────
figure('Name','Training curves','Color','w');
plot(trainInfo.TrainingLoss,   'b-',  'LineWidth',1.5); hold on;
plot(trainInfo.ValidationLoss, 'r--', 'LineWidth',1.5);
xlabel('Iteration'); ylabel('Loss (Dice + BCE)');
legend('Training','Validation'); grid on;
title('Hybrid Frangi–UNet Training Loss');

fprintf('\nDemo complete.\n');

%% =========================================================================
%% LOCAL HELPER: synthetic vessel generator
%% =========================================================================

function [img, mask] = syntheticVesselImage(H, W)
% Generates a noisy grayscale image with random tubular vessels.

    mask = false(H,W);

    % Draw 3-7 random line/curve vessels
    nVessels = randi([3 7]);
    for v = 1:nVessels
        width = randi([1 4]);
        % Random start/end with optional midpoint for curvature
        x0 = randi(W); y0 = randi(H);
        x1 = randi(W); y1 = randi(H);
        xm = round((x0+x1)/2 + randn*20);
        ym = round((y0+y1)/2 + randn*20);

        t  = linspace(0,1,500);
        xp = round((1-t).^2*x0 + 2*(1-t).*t*xm + t.^2*x1);
        yp = round((1-t).^2*y0 + 2*(1-t).*t*ym + t.^2*y1);

        % Clip to image
        xp = max(1,min(W,xp));
        yp = max(1,min(H,yp));

        for p = 1:numel(xp)
            y1r = max(1,yp(p)-width); y2r = min(H,yp(p)+width);
            x1r = max(1,xp(p)-width); x2r = min(W,xp(p)+width);
            mask(y1r:y2r, x1r:x2r) = true;
        end
    end

    % Noisy background + bright vessels
    bg  = 0.1 + 0.05*randn(H,W);
    img = bg + 0.6*double(mask) + 0.05*randn(H,W);
    img = mat2gray(img);
end

%% =========================================================================
%% LOCAL HELPER: classical (non-learnable) Frangi for comparison
%% =========================================================================

function V = classicFrangi(img, sigmas)
% Fixed-parameter multi-scale Frangi vesselness (Frangi 1998).
    alpha = 0.5; beta = 15;
    img   = im2double(img);
    V     = zeros(size(img));

    for s = 1:numel(sigmas)
        sig = sigmas(s);
        ks  = max(5, 2*ceil(3*sig)+1);
        r   = floor(ks/2);
        [x,y] = meshgrid(-r:r,-r:r);
        G   = exp(-(x.^2+y.^2)/(2*sig^2));
        Gxx = G.*(x.^2/sig^4-1/sig^2);
        Gyy = G.*(y.^2/sig^4-1/sig^2);
        Gxy = G.*(x.*y/sig^4);

        Lxx = sig^2 * imfilter(img,Gxx,'same','replicate');
        Lxy = sig^2 * imfilter(img,Gxy,'same','replicate');
        Lyy = sig^2 * imfilter(img,Gyy,'same','replicate');

        half = (Lxx+Lyy)/2;
        disc = sqrt(((Lxx-Lyy)/2).^2 + Lxy.^2 + 1e-8);
        l1   = half - disc;
        l2   = half + disc;

        Rb = l1 ./ (l2 + sign(l2)*1e-6 + 1e-8);
        S2 = l1.^2 + l2.^2;
        Vs = exp(-Rb.^2/(2*alpha^2)) .* (1-exp(-S2/(2*beta^2)));
        Vs(l2 > 0) = 0;
        V  = max(V, Vs);
    end
    V = mat2gray(V);
end
