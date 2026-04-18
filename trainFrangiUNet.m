function [trainedNet, info] = trainFrangiUNet(imgDir, labelDir, opts)
% TRAINFRANGUINET  Train the hybrid learnable-Frangi + U-Net on 3-D volumes.
%
%   [net, info] = trainFrangiUNet(imgDir, labelDir)
%   [net, info] = trainFrangiUNet(imgDir, labelDir, opts)
%
%   imgDir   – folder of NIfTI volumes (.nii or .nii.gz)
%   labelDir – folder of matching binary mask NIfTI files (same convention)
%   opts     – struct of training hyperparameters (see defaultOpts below)
%
%   PATCH-BASED TRAINING
%   Large volumes are never loaded into the network whole.  Instead,
%   opts.patchesPerVolume random 3-D patches of size opts.patchSize are
%   sampled from each volume every epoch.  With opts.foregroundFraction=0.8,
%   80 % of patches are centred on a random foreground voxel (with small
%   jitter), ensuring adequate positive-class coverage even in volumes with
%   sparse vessel labelling.  Volumes smaller than opts.patchSize are
%   zero-padded automatically.
%
%   INFERENCE on large volumes uses predictVolume (see predictVolume.m),
%   which applies the same network to overlapping patches and blends results
%   with a Gaussian window — so the network is never asked to process a
%   volume larger than opts.patchSize.
%
%   References:
%     Frangi et al. (1998) MICCAI – original vesselness filter
%     Ronneberger et al. (2015) MICCAI – U-Net
%
%   Requires: Deep Learning Toolbox, Image Processing Toolbox (imresize3 not needed)

    if nargin < 3, opts = struct(); end
    opts = defaultOpts(opts);

    %% ── 1. Datastore ────────────────────────────────────────────────────
    fprintf('[1/4] Building patch-based datastores...\n');
    ds = buildDatastore(imgDir, labelDir, opts);
    fprintf('      Train: %d patches/epoch (%d vols × %d patches)  |  Val: %d patches\n', ...
            ds.nTrainPatches, ds.nTrainVols, opts.patchesPerVolume, ds.nValVols);

    %% ── 2. Network ──────────────────────────────────────────────────────
    fprintf('[2/4] Assembling hybrid 3-D Frangi–UNet graph...\n');
    lgraph = buildFrangiUNet(opts);

    %% ── 3. Training options ─────────────────────────────────────────────
    fprintf('[3/4] Configuring training...\n');
    tOpts = trainingOptions('adam', ...
        'InitialLearnRate',     opts.lr, ...
        'MaxEpochs',            opts.epochs, ...
        'MiniBatchSize',        opts.batchSize, ...
        'Shuffle',              'every-epoch', ...
        'Plots',                opts.plots, ...
        'Verbose',              true, ...
        'VerboseFrequency',     10, ...
        'L2Regularization',     opts.l2, ...
        'LearnRateSchedule',    'piecewise', ...
        'LearnRateDropFactor',  0.5, ...
        'LearnRateDropPeriod',  floor(opts.epochs/3), ...
        'ValidationData',       ds.val, ...
        'ValidationFrequency',  20, ...
        'OutputNetwork',        'best-validation-loss');

    %% ── 4. Train ────────────────────────────────────────────────────────
    fprintf('[4/4] Training...\n');
    [trainedNet, info] = trainNetwork(ds.train, lgraph, tOpts);

    fprintf('Done. Best validation loss: %.4f\n', min(info.ValidationLoss));
end

% =========================================================================
function opts = defaultOpts(opts)
    defaults = struct( ...
        'patchSize',          [64 64 64], ...  % 3-D patch extracted per sample [H W D]
        'patchesPerVolume',   8, ...            % independent patches per volume per epoch
        'foregroundFraction', 0.8, ...          % probability a patch is fg-centred
        'sigmaMin',           1.0, ...          % minimum Gaussian sigma (voxels)
        'sigmaMax',           4.0, ...          % maximum Gaussian sigma (voxels)
        'encoderDepth',       3, ...            % U-Net encoder depth (2^depth <= min patch dim)
        'initFilters',        16, ...           % filters in first encoder block
        'lr',                 1e-3, ...
        'epochs',             50, ...
        'batchSize',          2, ...            % small batches — 3-D patches are memory-heavy
        'l2',                 1e-4, ...
        'valFraction',        0.15, ...
        'plots',              'training-progress', ...
        'useFrangi',          true, ...
        'archMode',           '', ...   % '' = derive from useFrangi
        'numFrangiChannels',  1, ...
        'augFlip',            true, ...   % random 50/50 flip along each axis
        'augNoiseStd',        0.02, ...   % additive Gaussian noise std (post-norm scale)
        'augIntensityScale',  [0.9 1.1] ... % multiplicative intensity jitter range
    );
    fields = fieldnames(defaults);
    for k = 1:numel(fields)
        f = fields{k};
        if ~isfield(opts, f), opts.(f) = defaults.(f); end
    end
    % imgSize drives buildFrangiUNet — always derived from patchSize
    opts.imgSize = opts.patchSize;
end

% =========================================================================
function ds = buildDatastore(imgDir, labelDir, opts)
% Build train / val datastores with random foreground-biased patch extraction.
%
%   Training files are replicated patchesPerVolume times so that each epoch
%   draws that many independent random patches per volume.  Because extractPatch
%   is stateless (randomness via rand/randi at read-time), each call returns a
%   different patch even for the same file.

    niiExts = {'.nii', '.nii.gz'};

    imgDs   = fileDatastore(imgDir,   'ReadFcn', @loadNifti, 'FileExtensions', niiExts);
    labelDs = fileDatastore(labelDir, 'ReadFcn', @loadNifti, 'FileExtensions', niiExts);

    n      = numel(imgDs.Files);
    nVal   = max(1, round(n * opts.valFraction));
    nTrain = n - nVal;

    assert(n == numel(labelDs.Files), ...
        'Image/label NIfTI file count mismatch (%d vs %d).', n, numel(labelDs.Files));

    % ── Training: replicate file lists for patchesPerVolume patches per volume ──
    nP            = opts.patchesPerVolume;
    trainImgFiles = repmat(imgDs.Files(1:nTrain),   nP, 1);
    trainLblFiles = repmat(labelDs.Files(1:nTrain), nP, 1);

    trainImgDs = fileDatastore(trainImgFiles, 'ReadFcn', @loadNifti, ...
                               'FileExtensions', niiExts);
    trainLblDs = fileDatastore(trainLblFiles, 'ReadFcn', @loadNifti, ...
                               'FileExtensions', niiExts);
    trainCds   = combine(trainImgDs, trainLblDs);
    ds.train   = transform(trainCds, @(x) extractPatch(x, opts, true));

    % ── Validation: one random patch per volume (no fg-bias for honest reporting) ──
    valImgFiles = imgDs.Files(nTrain+1:n);
    valLblFiles = labelDs.Files(nTrain+1:n);

    valImgDs = fileDatastore(valImgFiles, 'ReadFcn', @loadNifti, ...
                             'FileExtensions', niiExts);
    valLblDs = fileDatastore(valLblFiles, 'ReadFcn', @loadNifti, ...
                             'FileExtensions', niiExts);
    valCds   = combine(valImgDs, valLblDs);
    ds.val   = transform(valCds, @(x) extractPatch(x, opts, false));

    ds.nTrainVols    = nTrain;
    ds.nTrainPatches = nTrain * nP;
    ds.nValVols      = nVal;
end

% =========================================================================
function vol = loadNifti(filename)
    vol = single(niftiread(filename));
end

% =========================================================================
function out = extractPatch(data, opts, useForegroundBias)
% Extract a random 3-D patch, optionally biased toward foreground voxels.
%
%   When useForegroundBias is true and the volume has at least one foreground
%   voxel, a patch is centred on a uniformly-random foreground voxel with
%   independent uniform jitter of ±patchSize/4 on each axis.  Otherwise a
%   fully-random patch origin is drawn.  Volumes smaller than patchSize on
%   any axis are zero-padded before sampling.

    vol  = im2single(data{1});
    mask = single(data{2} > 0.5);
    pSz  = opts.patchSize(:)';

    volSz = size(vol, [1 2 3]);

    % Zero-pad if any volume dimension is smaller than the patch size
    padNeeded = max(0, pSz - volSz);
    if any(padNeeded > 0)
        vol   = padarray(vol,  padNeeded, 0, 'post');
        mask  = padarray(mask, padNeeded, 0, 'post');
        volSz = volSz + padNeeded;
    end

    maxStart = max(1, volSz - pSz + 1);   % largest valid top-left corner index

    % ── Choose patch origin ────────────────────────────────────────────────
    if useForegroundBias && rand() < opts.foregroundFraction && any(mask(:) > 0.5)
        % Centre on a random foreground voxel, then jitter by ±patchSize/4
        fgIdx = find(mask > 0.5);
        sel   = fgIdx(randi(numel(fgIdx)));
        [cx, cy, cz] = ind2sub(volSz, sel);

        jMax = max(1, floor(pSz / 4));    % jitter radius per dimension
        cx   = cx + randi([-jMax(1), jMax(1)]);
        cy   = cy + randi([-jMax(2), jMax(2)]);
        cz   = cz + randi([-jMax(3), jMax(3)]);

        r1 = [cx - floor(pSz(1)/2), ...
              cy - floor(pSz(2)/2), ...
              cz - floor(pSz(3)/2)];
    else
        r1 = [randi([1, maxStart(1)]), ...
              randi([1, maxStart(2)]), ...
              randi([1, maxStart(3)])];
    end

    % Clamp top-left corner so patch stays in bounds, then extract
    r1 = max(1, min(r1, maxStart));
    r2 = r1 + pSz - 1;

    volPatch  = vol( r1(1):r2(1), r1(2):r2(2), r1(3):r2(3) );
    maskPatch = mask(r1(1):r2(1), r1(2):r2(2), r1(3):r2(3) );

    % Per-patch intensity normalisation to [0, 1]
    lo = min(volPatch(:));  hi = max(volPatch(:));
    if hi > lo
        volPatch = (volPatch - lo) / (hi - lo);
    end

    % ── Augmentation (training only) ────────────────────────────────────
    if useForegroundBias

        % Random flip along each spatial axis independently
        if opts.augFlip
            for ax = 1:3
                if rand() < 0.5
                    volPatch  = flip(volPatch,  ax);
                    maskPatch = flip(maskPatch, ax);
                end
            end
        end

        % Multiplicative intensity scaling (image only)
        if opts.augIntensityScale(2) > opts.augIntensityScale(1)
            scale    = opts.augIntensityScale(1) + ...
                       rand() * diff(opts.augIntensityScale);
            volPatch = volPatch * scale;
        end

        % Additive Gaussian noise (image only)
        if opts.augNoiseStd > 0
            volPatch = volPatch + opts.augNoiseStd * randn(size(volPatch), 'single');
        end

        % Clamp to [0, 1] after intensity perturbations
        volPatch = max(0, min(1, volPatch));
    end

    out = {reshape(volPatch,  [pSz 1]), ...
           reshape(maskPatch, [pSz 1])};
end
