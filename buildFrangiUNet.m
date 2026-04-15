function lgraph = buildFrangiUNet(opts)
% BUILDFRANGUINET  Construct the hybrid learnable-Frangi + U-Net DAG.
%
%   The architecture has two parallel input branches that are concatenated
%   before entering the U-Net encoder:
%
%     ┌── raw input ─────────────────────────────────────┐
%     │                                                  ↓
%   input ──> LearnableFrangiLayer ──> frangi response → concat → U-Net
%
%   Each encoder block: Conv-BN-ReLU → Conv-BN-ReLU → MaxPool
%   Each decoder block: TransConv (upsample) → concat(skip) → Conv-BN-ReLU × 2
%   Output:             1×1 Conv → Sigmoid → pixel classification loss

    H  = opts.imgSize(1);
    W  = opts.imgSize(2);
    nF = opts.initFilters;
    D  = opts.encoderDepth;    % must be 1..4 for 256px input

    layers  = {};
    connect = {};   % {src, dst} pairs

    % ── Input ────────────────────────────────────────────────────────────
    layers{end+1} = imageInputLayer([H W 1], 'Name','input', ...
                                    'Normalization','none');

    % ── Learnable Frangi branch ──────────────────────────────────────────
    layers{end+1} = learnableFrangiLayer(opts.numScales, ...
                                         opts.sigmaMin, ...
                                         opts.sigmaMax, ...
                                         'Name','frangi');
    connect{end+1} = {'input','frangi'};

    % ── Concatenate raw + vesselness ─────────────────────────────────────
    layers{end+1} = concatenationLayer(3, 2, 'Name','cat_input');
    connect{end+1} = {'input',  'cat_input/in1'};
    connect{end+1} = {'frangi', 'cat_input/in2'};

    % ── Encoder ──────────────────────────────────────────────────────────
    skipNames = cell(1, D);
    prevName  = 'cat_input';
    inCh      = 2;    % raw (1ch) + frangi (1ch)

    for d = 1:D
        outCh    = nF * 2^(d-1);
        blkName  = sprintf('enc%d', d);
        [layers, connect, lastLayer] = addEncoderBlock(layers, connect, ...
                                        prevName, inCh, outCh, blkName);
        skipNames{d} = lastLayer;   % pre-pool feature map

        poolName = sprintf('pool%d', d);
        layers{end+1} = maxPooling2dLayer(2, 'Stride',2, 'Name',poolName);
        connect{end+1} = {lastLayer, poolName};

        prevName = poolName;
        inCh     = outCh;
    end

    % ── Bottleneck ───────────────────────────────────────────────────────
    btCh = nF * 2^D;
    [layers, connect, prevName] = addEncoderBlock(layers, connect, ...
                                    prevName, inCh, btCh, 'bottleneck');

    % ── Decoder ──────────────────────────────────────────────────────────
    for d = D:-1:1
        outCh    = nF * 2^(d-1);
        upName   = sprintf('up%d', d);
        catName  = sprintf('cat%d', d);
        blkName  = sprintf('dec%d', d);

        % Transposed conv upsample
        layers{end+1} = transposedConv2dLayer(2, outCh, 'Stride',2, ...
                            'Name',upName);
        connect{end+1} = {prevName, upName};

        % Concatenate with skip
        layers{end+1} = concatenationLayer(3, 2, 'Name',catName);
        connect{end+1} = {upName,          sprintf('%s/in1',catName)};
        connect{end+1} = {skipNames{d},    sprintf('%s/in2',catName)};

        % Decoder conv block
        [layers, connect, prevName] = addDecoderBlock(layers, connect, ...
                                        catName, outCh*2, outCh, blkName);
    end

    % ── Output head ──────────────────────────────────────────────────────
    layers{end+1} = convolution2dLayer(1, 1, 'Name','conv_out');
    connect{end+1} = {prevName, 'conv_out'};

    layers{end+1} = sigmoidLayer('Name','sigmoid');
    connect{end+1} = {'conv_out','sigmoid'};

    layers{end+1} = dicePixelClassificationLayer('Name','loss');
    connect{end+1} = {'sigmoid','loss'};

    % ── Assemble DAG ─────────────────────────────────────────────────────
    lgraph = layerGraph();
    lgraph = addLayers(lgraph, [layers{:}]);
    for k = 1:numel(connect)
        lgraph = connectLayers(lgraph, connect{k}{1}, connect{k}{2});
    end
end

% =========================================================================
function [layers, connect, postConvName] = addEncoderBlock(layers, connect, ...
                                            prevName, inCh, outCh, prefix)
    n1 = [prefix '_c1'];
    n2 = [prefix '_c2'];

    layers{end+1} = convolution2dLayer(3, outCh, 'Padding','same', 'Name',n1);
    connect{end+1} = {prevName, n1};

    layers{end+1} = batchNormalizationLayer('Name',[prefix '_bn1']);
    connect{end+1} = {n1, [prefix '_bn1']};

    layers{end+1} = reluLayer('Name',[prefix '_r1']);
    connect{end+1} = {[prefix '_bn1'], [prefix '_r1']};

    layers{end+1} = convolution2dLayer(3, outCh, 'Padding','same', 'Name',n2);
    connect{end+1} = {[prefix '_r1'], n2};

    layers{end+1} = batchNormalizationLayer('Name',[prefix '_bn2']);
    connect{end+1} = {n2, [prefix '_bn2']};

    layers{end+1} = reluLayer('Name',[prefix '_r2']);
    connect{end+1} = {[prefix '_bn2'], [prefix '_r2']};

    postConvName = [prefix '_r2'];
end

% =========================================================================
function [layers, connect, lastName] = addDecoderBlock(layers, connect, ...
                                         prevName, inCh, outCh, prefix)
    [layers, connect, lastName] = addEncoderBlock(layers, connect, ...
                                    prevName, inCh, outCh, prefix);
end
