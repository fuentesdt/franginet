function results = evaluateFrangiUNet(net, imgDir, labelDir, opts)
% EVALUATEFRANGUUNET  Run inference and compute segmentation metrics.
%
%   results = evaluateFrangiUNet(net, imgDir, labelDir)
%   results = evaluateFrangiUNet(net, imgDir, labelDir, opts)
%
%   OUTPUTS  results struct with fields:
%     .dice    – per-image Dice coefficient  [N×1]
%     .clDice  – per-image centerline Dice   [N×1]  (topology-aware)
%     .auc     – per-image ROC AUC           [N×1]
%     .meanDice, .meanClDice, .meanAUC
%
%   Also saves predicted probability maps and binary masks to opts.outDir.

    if nargin < 4, opts = struct(); end
    if ~isfield(opts,'imgSize'),   opts.imgSize  = [256 256]; end
    if ~isfield(opts,'threshold'), opts.threshold = 0.5;      end
    if ~isfield(opts,'outDir'),    opts.outDir   = './predictions'; end

    if ~exist(opts.outDir,'dir'), mkdir(opts.outDir); end

    imgFiles   = dir(fullfile(imgDir,   '*.png'));
    labelFiles = dir(fullfile(labelDir, '*.png'));

    % Fallback to tif
    if isempty(imgFiles)
        imgFiles   = dir(fullfile(imgDir,   '*.tif'));
        labelFiles = dir(fullfile(labelDir, '*.tif'));
    end

    N = numel(imgFiles);
    assert(N == numel(labelFiles), 'Image/label count mismatch.');

    dice_v   = zeros(N,1);
    clDice_v = zeros(N,1);
    auc_v    = zeros(N,1);

    fprintf('Running inference on %d images...\n', N);

    for i = 1:N
        img   = im2single(imread(fullfile(imgDir,   imgFiles(i).name)));
        label = imread(fullfile(labelDir, labelFiles(i).name));

        if size(img,3) > 1,   img   = rgb2gray(img);   end
        if size(label,3) > 1, label = rgb2gray(label);  end

        img_rs   = imresize(img,   opts.imgSize);
        label_rs = imresize(label, opts.imgSize, 'nearest') > 0;

        % ── Forward pass ─────────────────────────────────────────────────
        X    = reshape(img_rs, [opts.imgSize 1 1]);
        prob = predict(net, X);                   % [H W 1 1]
        prob = squeeze(double(extractdata(prob))); % [H W]

        % ── Binary prediction ─────────────────────────────────────────────
        pred = prob >= opts.threshold;

        % ── Metrics ──────────────────────────────────────────────────────
        dice_v(i)   = diceCoeff(pred, label_rs);
        clDice_v(i) = centerlineDice(pred, label_rs);
        auc_v(i)    = computeAUC(prob(:), double(label_rs(:)));

        % ── Save outputs ─────────────────────────────────────────────────
        [~,fname] = fileparts(imgFiles(i).name);
        imwrite(uint8(prob*255),  fullfile(opts.outDir, [fname '_prob.png']));
        imwrite(uint8(pred*255),  fullfile(opts.outDir, [fname '_mask.png']));

        if mod(i,10)==0
            fprintf('  [%d/%d] Dice=%.3f  clDice=%.3f  AUC=%.3f\n', ...
                     i, N, dice_v(i), clDice_v(i), auc_v(i));
        end
    end

    results.dice      = dice_v;
    results.clDice    = clDice_v;
    results.auc       = auc_v;
    results.meanDice  = mean(dice_v);
    results.meanClDice= mean(clDice_v);
    results.meanAUC   = mean(auc_v);

    fprintf('\n── Final metrics ───────────────────────────────\n');
    fprintf('  Mean Dice     : %.4f ± %.4f\n', mean(dice_v),   std(dice_v));
    fprintf('  Mean clDice   : %.4f ± %.4f\n', mean(clDice_v), std(clDice_v));
    fprintf('  Mean AUC      : %.4f ± %.4f\n', mean(auc_v),    std(auc_v));
    fprintf('────────────────────────────────────────────────\n');
end

% =========================================================================
% METRIC HELPERS
% =========================================================================

function d = diceCoeff(pred, gt)
    pred = logical(pred(:));
    gt   = logical(gt(:));
    d    = 2*sum(pred & gt) / (sum(pred) + sum(gt) + 1e-8);
end

% -------------------------------------------------------------------------
function cd = centerlineDice(pred, gt)
% CENTERLINEDICE  Topology-aware Dice on skeletonized centrelines.
%   Measures agreement on the 1-pixel-wide vessel skeletons, penalising
%   topological errors (missed branches, spurious loops) more heavily
%   than standard Dice.
%
%   Reference: Shit et al. (2021) "clDice – a Novel Topology-Preserving
%              Loss Function for Tubular Structure Segmentation", CVPR.

    skel_pred = bwskel(logical(pred));
    skel_gt   = bwskel(logical(gt));

    % Soft Dice on skeletons (no thresholding needed – already binary)
    Tprec = sum(skel_pred(:) & logical(pred(:))) / (sum(skel_pred(:)) + 1e-8);
    Tsens = sum(skel_gt(:)   & logical(gt(:)))   / (sum(skel_gt(:))   + 1e-8);
    cd    = 2*Tprec*Tsens / (Tprec + Tsens + 1e-8);
end

% -------------------------------------------------------------------------
function auc = computeAUC(scores, labels)
% COMPUTEAUC  Trapezoidal AUC-ROC.
    [~,~,~,auc] = perfcurve(labels, scores, 1);
end
