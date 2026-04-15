function [trainedNet, info] = trainFrangiUNet(imgDir, labelDir, opts)
% TRAINFRANGUINET  Train the hybrid learnable-Frangi + U-Net on 3-D volumes.
%
%   [net, info] = trainFrangiUNet(imgDir, labelDir)
%   [net, info] = trainFrangiUNet(imgDir, labelDir, opts)
%
%   imgDir   – folder of 3-D volume .mat files  (each contains a single
%              variable: a [H W D] single-precision array)
%   labelDir – folder of matching binary mask .mat files (same convention)
%   opts     – struct of training hyperparameters (see defaultOpts below)
%
%   The network graph is:
%
%     Input [H×W×D×1]
%       └─> LearnableFrangiLayer  (differentiable 3-D multi-scale vesselness)
%       └─> [concat with raw input]  →  [H×W×D×2]
%             └─> 3-D U-Net encoder-decoder with skip connections
%                   └─> Sigmoid output
%
%   References:
%     Frangi et al. (1998) MICCAI – original vesselness filter
%     Ronneberger et al. (2015) MICCAI – U-Net
%
%   Requires: Deep Learning Toolbox, Image Processing Toolbox (for imresize3)

    if nargin < 3, opts = struct(); end
    opts = defaultOpts(opts);

    %% ── 1. Datastore ────────────────────────────────────────────────────
    fprintf('[1/4] Building datastores...\n');
    ds = buildDatastore(imgDir, labelDir, opts);

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
        'imgSize',      [64 64 32], ...  % spatial size [H W D]
        'numScales',    4, ...           % Frangi scale levels
        'sigmaMin',     1.0, ...         % minimum Gaussian sigma (voxels)
        'sigmaMax',     4.0, ...         % maximum Gaussian sigma (voxels)
        'encoderDepth', 3, ...           % U-Net encoder depth (reduce for small volumes)
        'initFilters',  16, ...          % filters in first encoder block
        'lr',           1e-3, ...
        'epochs',       50, ...
        'batchSize',    2, ...           % small batches for 3-D memory budget
        'l2',           1e-4, ...
        'valFraction',  0.15, ...
        'plots',        'training-progress', ...
        'useFrangi',    true ...         % false → plain U-Net control
    );
    fields = fieldnames(defaults);
    for k = 1:numel(fields)
        f = fields{k};
        if ~isfield(opts, f), opts.(f) = defaults.(f); end
    end
end

% =========================================================================
function ds = buildDatastore(imgDir, labelDir, opts)
% Accepts NIfTI files (.nii or .nii.gz) in imgDir and labelDir.
% Files are paired in sorted order; names need not match as long as counts do.

    niiExts = {'.nii', '.nii.gz'};

    imgDs   = fileDatastore(imgDir,   'ReadFcn', @loadNifti, ...
                            'FileExtensions', niiExts);
    labelDs = fileDatastore(labelDir, 'ReadFcn', @loadNifti, ...
                            'FileExtensions', niiExts);

    n      = numel(imgDs.Files);
    nVal   = max(1, round(n * opts.valFraction));
    nTrain = n - nVal;

    assert(n == numel(labelDs.Files), ...
        'Image/label NIfTI file count mismatch (%d vs %d).', n, numel(labelDs.Files));

    cds = combine(imgDs, labelDs);
    cds = transform(cds, @(x) preprocessPair(x, opts));

    ds.train = subset(cds, 1:nTrain);
    ds.val   = subset(cds, nTrain+1:n);
end

% =========================================================================
function vol = loadNifti(filename)
% Read a NIfTI file (.nii or .nii.gz) and return a single-precision array.
    vol = single(niftiread(filename));
end

% =========================================================================
function out = preprocessPair(data, opts)
% Resize volume and mask to opts.imgSize, add channel dim.

    vol  = im2single(imresize3(data{1}, opts.imgSize));
    mask = imresize3(data{2}, opts.imgSize, 'Method', 'nearest');

    % Ensure single-channel: collapse any extra dims
    vol  = vol(:,:,:,1);
    mask = mask(:,:,:,1);

    % Add channel dimension: [H W D] -> [H W D 1]
    vol  = reshape(vol,           [opts.imgSize 1]);
    mask = reshape(single(mask > 0.5), [opts.imgSize 1]);

    out = {vol, mask};
end
