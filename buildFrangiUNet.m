function lgraph = buildFrangiUNet(opts)
% BUILDFRANGUINET  Construct the hybrid learnable-Frangi + U-Net DAG for 3-D volumes.
%
%   opts.useFrangi (default true) controls the architecture:
%
%     true  — Hybrid: two branches concatenated before the encoder
%               ┌── raw input ─────────────────────────────────────┐
%               │                                                  ↓
%             input ──> LearnableFrangiLayer ──> frangi → concat → U-Net
%
%     false — Plain U-Net control (no Frangi branch):
%             input ──────────────────────────────────────────────> U-Net
%
%   Each encoder block: Conv3D-BN-ReLU → Conv3D-BN-ReLU → MaxPool3D
%   Each decoder block: TransConv3D (upsample) → concat(skip) → Conv3D-BN-ReLU × 2
%   Output:             1×1×1 Conv3D → Sigmoid → pixel classification loss
%
%   opts.imgSize must be a 3-element vector [H W D].

    H  = opts.imgSize(1);
    W  = opts.imgSize(2);
    D  = opts.imgSize(3);
    nF = opts.initFilters;
    Dp = opts.encoderDepth;   % must satisfy 2^Dp <= min(H,W,D)

    layers  = {};
    connect = {};   % {src, dst} pairs

    % ── Input ────────────────────────────────────────────────────────────
    % image3dInputLayer (introduced R2019b) is required for volumetric data.
    % imageInputLayer only accepts 2-D sizes ([h w] or [h w c]).
    layers{end+1} = image3dInputLayer([H W D 1], 'Name','input', ...
                                      'Normalization','none');

    % ── Optional Frangi branch ───────────────────────────────────────────
    useFrangi = ~isfield(opts,'useFrangi') || opts.useFrangi;

    if useFrangi
        layers{end+1} = learnableFrangiLayer(opts.numScales, ...
                                             opts.sigmaMin, ...
                                             opts.sigmaMax, ...
                                             'Name','frangi');
        connect{end+1} = {'input','frangi'};

        % Concatenate raw + vesselness along channel dim 4: [H W D C B]
        layers{end+1} = concatenationLayer(4, 2, 'Name','cat_input');
        connect{end+1} = {'input',  'cat_input/in1'};
        connect{end+1} = {'frangi', 'cat_input/in2'};

        prevName = 'cat_input';
        inCh     = 2;   % raw (1ch) + frangi (1ch)
    else
        % Plain U-Net: raw input feeds directly into the encoder
        prevName = 'input';
        inCh     = 1;
    end

    % ── Encoder ──────────────────────────────────────────────────────────
    skipNames = cell(1, Dp);

    for d = 1:Dp
        %outCh   = nF * 2^(d-1);
        outCh   = nF ; % pocketnet
        blkName = sprintf('enc%d', d);
        [layers, connect, lastLayer] = addEncoderBlock(layers, connect, ...
                                        prevName, inCh, outCh, blkName);
        skipNames{d} = lastLayer;   % pre-pool feature map

        poolName = sprintf('pool%d', d);
        layers{end+1} = maxPooling3dLayer(2, 'Stride',2, 'Name',poolName);
        connect{end+1} = {lastLayer, poolName};

        prevName = poolName;
        inCh     = outCh;
    end

    % ── Bottleneck ───────────────────────────────────────────────────────
    btCh = nF * 2^Dp;
    [layers, connect, prevName] = addEncoderBlock(layers, connect, ...
                                    prevName, inCh, btCh, 'bottleneck');

    % ── Decoder ──────────────────────────────────────────────────────────
    for d = Dp:-1:1
        outCh   = nF * 2^(d-1);
        upName  = sprintf('up%d', d);
        catName = sprintf('cat%d', d);
        blkName = sprintf('dec%d', d);

        % Transposed conv upsample
        layers{end+1} = transposedConv3dLayer(2, outCh, 'Stride',2, ...
                            'Name',upName);
        connect{end+1} = {prevName, upName};

        % Concatenate with skip
        layers{end+1} = concatenationLayer(4, 2, 'Name',catName);
        connect{end+1} = {upName,       sprintf('%s/in1',catName)};
        connect{end+1} = {skipNames{d}, sprintf('%s/in2',catName)};

        % Decoder conv block
        [layers, connect, prevName] = addDecoderBlock(layers, connect, ...
                                        catName, outCh*2, outCh, blkName);
    end

    % ── Output head ──────────────────────────────────────────────────────
    layers{end+1} = convolution3dLayer(1, 1, 'Name','conv_out');
    connect{end+1} = {prevName, 'conv_out'};

    layers{end+1} = sigmoidLayer('Name','sigmoid');
    connect{end+1} = {'conv_out','sigmoid'};

    layers{end+1} = dicePixelClassificationLayer('Name','loss');
    connect{end+1} = {'sigmoid','loss'};

    % ── Assemble DAG ─────────────────────────────────────────────────────
    % Add each layer individually to avoid the auto-connect behaviour that
    % addLayers triggers when given an array (it would connect them in series,
    % causing "connection already exists" errors when we wire the DAG below).
    lgraph = layerGraph();
    for k = 1:numel(layers)
        lgraph = addLayers(lgraph, layers{k});
    end
    for k = 1:numel(connect)
        lgraph = connectLayers(lgraph, connect{k}{1}, connect{k}{2});
    end
end

% =========================================================================
function [layers, connect, postConvName] = addEncoderBlock(layers, connect, ...
                                            prevName, inCh, outCh, prefix)
    n1 = [prefix '_c1'];
    n2 = [prefix '_c2'];

    layers{end+1} = convolution3dLayer(3, outCh, 'Padding','same', 'Name',n1);
    connect{end+1} = {prevName, n1};

    layers{end+1} = batchNormalizationLayer('Name',[prefix '_bn1']);
    connect{end+1} = {n1, [prefix '_bn1']};

    layers{end+1} = reluLayer('Name',[prefix '_r1']);
    connect{end+1} = {[prefix '_bn1'], [prefix '_r1']};

    layers{end+1} = convolution3dLayer(3, outCh, 'Padding','same', 'Name',n2);
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
