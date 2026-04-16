function results = evaluateFrangiUNet(net, imgDir, labelDir, opts)
% EVALUATEFRANGUUNET  Run patch-based 3-D inference and compute segmentation metrics.
%
%   results = evaluateFrangiUNet(net, imgDir, labelDir)
%   results = evaluateFrangiUNet(net, imgDir, labelDir, opts)
%
%   imgDir / labelDir must contain NIfTI files (.nii or .nii.gz).
%   Images should be single-precision float volumes; labels should be
%   binary masks (any numeric type — thresholded at 0.5 internally).
%
%   Large volumes are processed patch-by-patch via predictVolume.m
%   (sliding window + Gaussian blending) — the network is never asked to
%   handle a volume larger than opts.patchSize.
%
%   OPTS fields
%     patchSize    – [H W D] network input size         (default [64 64 64])
%     patchOverlap – overlap on each side (voxels)      (default [8 8 8])
%                    stride = patchSize - 2*patchOverlap
%     threshold    – binary decision threshold           (default 0.5)
%     outDir       – folder for _prob.nii / _mask.nii   (default './predictions')
%
%   OUTPUTS  results struct with fields:
%     .dice    – per-volume Dice coefficient  [N×1]
%     .clDice  – per-volume centerline Dice   [N×1]  (topology-aware)
%     .auc     – per-volume ROC AUC           [N×1]
%     .meanDice, .meanClDice, .meanAUC

    if nargin < 4, opts = struct(); end
    if ~isfield(opts,'patchSize'),    opts.patchSize    = [64 64 64]; end
    if ~isfield(opts,'patchOverlap'), opts.patchOverlap = [8  8  8 ]; end
    if ~isfield(opts,'threshold'),    opts.threshold    = 0.5;        end
    if ~isfield(opts,'outDir'),       opts.outDir       = './predictions'; end

    if ~exist(opts.outDir,'dir'), mkdir(opts.outDir); end

    imgFiles   = [dir(fullfile(imgDir,   '*.nii')); ...
                  dir(fullfile(imgDir,   '*.nii.gz'))];
    labelFiles = [dir(fullfile(labelDir, '*.nii')); ...
                  dir(fullfile(labelDir, '*.nii.gz'))];

    N = numel(imgFiles);
    assert(N == numel(labelFiles), 'NIfTI image/label count mismatch (%d vs %d).', ...
           N, numel(labelFiles));

    dice_v   = zeros(N,1);
    clDice_v = zeros(N,1);
    auc_v    = zeros(N,1);

    fprintf('Running patch-based 3-D inference on %d volumes (patchSize=[%d %d %d], overlap=[%d %d %d])...\n', ...
            N, opts.patchSize(1), opts.patchSize(2), opts.patchSize(3), ...
               opts.patchOverlap(1), opts.patchOverlap(2), opts.patchOverlap(3));

    for i = 1:N
        % ── Load ─────────────────────────────────────────────────────────
        vol   = im2single(niftiread(fullfile(imgDir,   imgFiles(i).name)));
        label = niftiread(fullfile(labelDir, labelFiles(i).name)) > 0;

        % ── Per-volume intensity normalisation (matches training) ─────────
        lo = min(vol(:));  hi = max(vol(:));
        if hi > lo, vol = (vol - lo) / (hi - lo); end

        % ── Sliding-window inference (predictVolume.m) ───────────────────
        fprintf('  [%d/%d] %s  vol=[%d %d %d]\n', ...
                i, N, imgFiles(i).name, size(vol,1), size(vol,2), size(vol,3));
        prob = predictVolume(net, vol, opts);
        pred = prob >= opts.threshold;

        % ── Metrics on full volume ────────────────────────────────────────
        dice_v(i)   = diceCoeff(pred, label);
        clDice_v(i) = centerlineDice(pred, label);
        auc_v(i)    = computeAUC(prob(:), double(label(:)));

        % ── Save outputs ─────────────────────────────────────────────────
        [~, fname] = fileparts(imgFiles(i).name);
        fname = regexprep(fname, '\.nii$', '');   % strip leftover .nii from .nii.gz
        niftiwrite(single(prob), fullfile(opts.outDir, [fname '_prob.nii']));
        niftiwrite(uint8(pred),  fullfile(opts.outDir, [fname '_mask.nii']));

        fprintf('          Dice=%.3f  clDice=%.3f  AUC=%.3f\n', ...
                 dice_v(i), clDice_v(i), auc_v(i));
    end

    results.dice       = dice_v;
    results.clDice     = clDice_v;
    results.auc        = auc_v;
    results.meanDice   = mean(dice_v);
    results.meanClDice = mean(clDice_v);
    results.meanAUC    = mean(auc_v);

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
%   Measures agreement on the 1-voxel-wide vessel skeletons, penalising
%   topological errors (missed branches, spurious loops) more heavily
%   than standard Dice.  bwskel supports both 2-D and 3-D inputs.
%
%   Reference: Shit et al. (2021) "clDice – a Novel Topology-Preserving
%              Loss Function for Tubular Structure Segmentation", CVPR.

    skel_pred = bwskel(logical(pred));
    skel_gt   = bwskel(logical(gt));

    Tprec = sum(skel_pred(:) & logical(pred(:))) / (sum(skel_pred(:)) + 1e-8);
    Tsens = sum(skel_gt(:)   & logical(gt(:)))   / (sum(skel_gt(:))   + 1e-8);
    cd    = 2*Tprec*Tsens / (Tprec + Tsens + 1e-8);
end

% -------------------------------------------------------------------------
function auc = computeAUC(scores, labels)
% COMPUTEAUC  Trapezoidal AUC-ROC.
    [~,~,~,auc] = perfcurve(labels, scores, 1);
end
