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

    hasFrangi = any(strcmp({net.Layers.Name}, 'frangi'));

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

    %for i = 1:N
    for i = 1:1
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

        if hasFrangi
            frangiVol = extractFrangiVolume(net, vol, opts);
            niftiwrite(single(frangiVol), fullfile(opts.outDir, [fname '_frangi.nii']));
        end

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

% =========================================================================
% FRANGI ACTIVATION EXTRACTION
% =========================================================================

function frangiVol = extractFrangiVolume(net, vol, opts)
% Sliding-window extraction of 'frangi' layer activations via activations().
% Uses the same Gaussian-blended patch tiling as predictVolume.
% Output is [H W D] if the frangi layer has 1 output channel, or [H W D nC]
% if it has nC > 1 (frangi_multichannel / ReduceMax=false).

    pSz    = opts.patchSize(:)';
    border = opts.patchOverlap(:)';
    stride = pSz - 2*border;

    origSz = size(vol, [1 2 3]);

    % Pad to cover last patch boundary exactly (same logic as predictVolume)
    padSz = zeros(1,3);
    for d = 1:3
        if origSz(d) <= pSz(d)
            padSz(d) = pSz(d);
        else
            nSteps   = ceil((origSz(d) - pSz(d)) / stride(d));
            padSz(d) = pSz(d) + nSteps * stride(d);
        end
    end
    padNeeded = padSz - origSz;
    if any(padNeeded > 0)
        vol = padarray(vol, padNeeded, 0, 'post');
    end

    % Probe one patch to determine the number of Frangi output channels
    probe   = reshape(im2single(vol(1:pSz(1), 1:pSz(2), 1:pSz(3))), [pSz 1]);
    actProbe = activations(net, probe, 'frangi');
    nC      = size(actProbe, 4);   % 1 when ReduceMax=true, numFrangiChannels otherwise

    actAcc  = zeros([padSz nC], 'single');
    wsumAcc = zeros(padSz,      'single');
    W       = gaussianWindow3D_local(pSz);

    starts1 = 1 : stride(1) : padSz(1)-pSz(1)+1;
    starts2 = 1 : stride(2) : padSz(2)-pSz(2)+1;
    starts3 = 1 : stride(3) : padSz(3)-pSz(3)+1;

    for i1 = starts1
        for i2 = starts2
            for i3 = starts3
                e1 = i1+pSz(1)-1;  e2 = i2+pSz(2)-1;  e3 = i3+pSz(3)-1;

                patch = vol(i1:e1, i2:e2, i3:e3);
                X     = reshape(im2single(patch), [pSz 1]);
                act   = single(squeeze(activations(net, X, 'frangi')));  % [H W D (nC)]

                if nC == 1
                    act = reshape(act, [pSz 1]);
                end

                for c = 1:nC
                    actAcc(i1:e1, i2:e2, i3:e3, c) = ...
                        actAcc(i1:e1, i2:e2, i3:e3, c) + act(:,:,:,c) .* W;
                end
                wsumAcc(i1:e1, i2:e2, i3:e3) = wsumAcc(i1:e1, i2:e2, i3:e3) + W;
            end
        end
    end

    wsum4     = repmat(max(wsumAcc, 1e-6), [1 1 1 nC]);
    frangiVol = actAcc ./ wsum4;
    frangiVol = frangiVol(1:origSz(1), 1:origSz(2), 1:origSz(3), :);

    if nC == 1
        frangiVol = squeeze(frangiVol);   % [H W D]
    end
end

% -------------------------------------------------------------------------
function W = gaussianWindow3D_local(sz)
    ax = cell(3,1);
    for d = 1:3
        t     = linspace(-1, 1, sz(d));
        ax{d} = exp(-2 * t.^2);
    end
    [A, B, C] = ndgrid(ax{1}, ax{2}, ax{3});
    W = single(A .* B .* C);
end
