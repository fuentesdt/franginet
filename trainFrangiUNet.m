function [trainedNet, info] = trainFrangiUNet(imgDir, labelDir, opts)
% TRAINFRANGUINET  Train the hybrid learnable-Frangi + U-Net segmentation model.
%
%   [net, info] = trainFrangiUNet(imgDir, labelDir)
%   [net, info] = trainFrangiUNet(imgDir, labelDir, opts)
%
%   imgDir   – folder of 2-D grayscale images  (.png / .tif)
%   labelDir – folder of matching binary vessel masks
%   opts     – struct of training hyperparameters (see defaultOpts below)
%
%   The network graph is:
%
%     Input
%       └─> LearnableFrangiLayer  (differentiable multi-scale vesselness)
%       └─> [concat with raw input]
%             └─> U-Net encoder-decoder with skip connections
%                   └─> Sigmoid output
%
%   References:
%     Frangi et al. (1998) MICCAI – original vesselness filter
%     Ronneberger et al. (2015) MICCAI – U-Net
%
%   Requires: Deep Learning Toolbox, Image Processing Toolbox

    if nargin < 3, opts = struct(); end
    opts = defaultOpts(opts);

    %% ── 1. Datastore ────────────────────────────────────────────────────
    fprintf('[1/4] Building datastores...\n');
    ds = buildDatastore(imgDir, labelDir, opts);

    %% ── 2. Network ──────────────────────────────────────────────────────
    fprintf('[2/4] Assembling hybrid Frangi–UNet graph...\n');
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
        'imgSize',      [256 256], ...   % spatial size (H x W)
        'numScales',    4, ...           % Frangi scale levels
        'sigmaMin',     1.0, ...         % minimum Gaussian sigma (px)
        'sigmaMax',     4.0, ...         % maximum Gaussian sigma (px)
        'encoderDepth', 4, ...           % U-Net encoder depth
        'initFilters',  32, ...          % filters in first encoder block
        'lr',           1e-3, ...
        'epochs',       50, ...
        'batchSize',    8, ...
        'l2',           1e-4, ...
        'valFraction',  0.15, ...
        'plots',        'training-progress' ...
    );
    fields = fieldnames(defaults);
    for k = 1:numel(fields)
        f = fields{k};
        if ~isfield(opts, f), opts.(f) = defaults.(f); end
    end
end

% =========================================================================
function ds = buildDatastore(imgDir, labelDir, opts)
    imgFiles   = imageDatastore(imgDir,   'FileExtensions', {'.png','.tif','.tiff','.jpg'});
    labelFiles = imageDatastore(labelDir, 'FileExtensions', {'.png','.tif','.tiff','.jpg'});

    n      = numel(imgFiles.Files);
    nVal   = max(1, round(n * opts.valFraction));
    nTrain = n - nVal;

    % Combine into pixel-label compatible combined datastore
    cds = combine(imgFiles, labelFiles);
    cds = transform(cds, @(x) preprocessPair(x, opts));

    ds.train = subset(cds, 1:nTrain);
    ds.val   = subset(cds, nTrain+1:n);
end

% =========================================================================
function out = preprocessPair(data, opts)
    img   = im2single(imresize(data{1}, opts.imgSize));
    label = imresize(data{2}, opts.imgSize, 'nearest');

    if size(img,3) > 1,   img   = rgb2gray(img);   end
    if size(label,3) > 1, label = rgb2gray(label);  end

    label = single(label > 0.5);          % binary mask -> single [0,1]
    out   = {img, label};
end
