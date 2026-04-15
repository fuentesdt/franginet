function results = evaluateFrangiUNet(net, imgDir, labelDir, opts)
% EVALUATEFRANGUUNET  Run 3-D inference and compute segmentation metrics.
%
%   results = evaluateFrangiUNet(net, imgDir, labelDir)
%   results = evaluateFrangiUNet(net, imgDir, labelDir, opts)
%
%   imgDir / labelDir must contain .mat files whose first variable is a
%   [H W D] single-precision volume or binary mask respectively.
%
%   OUTPUTS  results struct with fields:
%     .dice    – per-volume Dice coefficient  [N×1]
%     .clDice  – per-volume centerline Dice   [N×1]  (topology-aware)
%     .auc     – per-volume ROC AUC           [N×1]
%     .meanDice, .meanClDice, .meanAUC
%
%   Predicted probability volumes and binary masks are saved as .mat files
%   to opts.outDir.

    if nargin < 4, opts = struct(); end
    if ~isfield(opts,'imgSize'),   opts.imgSize   = [64 64 32]; end
    if ~isfield(opts,'threshold'), opts.threshold = 0.5;        end
    if ~isfield(opts,'outDir'),    opts.outDir    = './predictions'; end

    if ~exist(opts.outDir,'dir'), mkdir(opts.outDir); end

    imgFiles   = dir(fullfile(imgDir,   '*.mat'));
    labelFiles = dir(fullfile(labelDir, '*.mat'));

    N = numel(imgFiles);
    assert(N == numel(labelFiles), 'Image/label count mismatch (%d vs %d).', ...
           N, numel(labelFiles));

    dice_v   = zeros(N,1);
    clDice_v = zeros(N,1);
    auc_v    = zeros(N,1);

    fprintf('Running 3-D inference on %d volumes...\n', N);

    for i = 1:N
        % ── Load ─────────────────────────────────────────────────────────
        volData   = load(fullfile(imgDir,   imgFiles(i).name));
        maskData  = load(fullfile(labelDir, labelFiles(i).name));

        volFields  = fieldnames(volData);
        maskFields = fieldnames(maskData);

        vol   = im2single(volData.(volFields{1}));
        label = maskData.(maskFields{1});

        % ── Preprocess ───────────────────────────────────────────────────
        vol_rs   = imresize3(vol,   opts.imgSize);
        label_rs = imresize3(label, opts.imgSize, 'Method','nearest') > 0;

        % ── Forward pass ─────────────────────────────────────────────────
        X    = reshape(vol_rs, [opts.imgSize 1 1]);   % [H W D 1 1]
        prob = predict(net, X);                        % [H W D 1 1]
        prob = double(squeeze(extractdata(prob)));      % [H W D]

        % ── Binary prediction ─────────────────────────────────────────────
        pred = prob >= opts.threshold;

        % ── Metrics ──────────────────────────────────────────────────────
        dice_v(i)   = diceCoeff(pred, label_rs);
        clDice_v(i) = centerlineDice(pred, label_rs);
        auc_v(i)    = computeAUC(prob(:), double(label_rs(:)));

        % ── Save outputs ─────────────────────────────────────────────────
        [~,fname] = fileparts(imgFiles(i).name);
        prob_vol  = single(prob);                                   %#ok<NASGU>
        pred_mask = uint8(pred);                                    %#ok<NASGU>
        save(fullfile(opts.outDir, [fname '_prob.mat']), 'prob_vol');
        save(fullfile(opts.outDir, [fname '_mask.mat']), 'pred_mask');

        if mod(i,5)==0 || i==N
            fprintf('  [%d/%d] Dice=%.3f  clDice=%.3f  AUC=%.3f\n', ...
                     i, N, dice_v(i), clDice_v(i), auc_v(i));
        end
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
