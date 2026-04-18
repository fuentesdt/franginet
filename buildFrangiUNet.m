function lgraph = buildFrangiUNet(opts)
% BUILDFRANGUINET  Construct a 3-D segmentation network DAG.
%
%   Architecture is controlled by opts.archMode (or backwards-compat
%   opts.useFrangi).  Supported modes:
%
%   'unet'               — Plain U-Net (no Frangi branch)
%   'frangi_unet'        — Hybrid: Frangi(max) concatenated with raw input
%                          before the U-Net encoder (default when useFrangi=true)
%   'frangi_threshold'   — Arch 2: Frangi(max) → Sigmoid → loss
%   'frangi_linear'      — Arch 3: Frangi(max) → 1×1×1 Conv → Sigmoid → loss
%   'frangi_multichannel'— Arch 4: Frangi(all channels) → 1×1×1 Conv → Sigmoid → loss
%
%   opts.imgSize must be [H W D].

    archMode = resolveArchMode(opts);

    H  = opts.imgSize(1);
    W  = opts.imgSize(2);
    D  = opts.imgSize(3);
    nFrangiCh = opts.numFrangiChannels;

    layers  = {};
    connect = {};

    layers{end+1} = image3dInputLayer([H W D 1], 'Name','input', ...
                                      'Normalization','none');

    % ── Frangi-only architectures (no U-Net) ─────────────────────────────
    switch archMode
        case 'frangi_threshold'
            % Arch 2: Frangi(max) → loss directly.
            % Vesselness is already in [0,1]; no sigmoid needed or wanted —
            % any sigmoid with non-negative input sits at >= 0.5 everywhere.
            layers{end+1} = learnableFrangiLayer(nFrangiCh, opts.sigmaMin, ...
                                opts.sigmaMax, 'ReduceMax',true, 'Name','frangi');
            connect{end+1} = {'input',  'frangi'};
            connect{end+1} = {'frangi', 'loss'};

        case 'frangi_linear'
            % Arch 3: Frangi(max) → 1×1×1 Conv → Sigmoid → loss
            layers{end+1} = learnableFrangiLayer(nFrangiCh, opts.sigmaMin, ...
                                opts.sigmaMax, 'ReduceMax',true, 'Name','frangi');
            layers{end+1} = convolution3dLayer(1, 1, 'Name','conv_out');
            connect{end+1} = {'input',    'frangi'};
            connect{end+1} = {'frangi',   'conv_out'};
            connect{end+1} = {'conv_out', 'sigmoid'};

        case 'frangi_multichannel'
            % Arch 4: Frangi(all channels) → 1×1×1 Conv → Sigmoid → loss
            layers{end+1} = learnableFrangiLayer(nFrangiCh, opts.sigmaMin, ...
                                opts.sigmaMax, 'ReduceMax',false, 'Name','frangi');
            layers{end+1} = convolution3dLayer(1, 1, 'Name','conv_out');
            connect{end+1} = {'input',    'frangi'};
            connect{end+1} = {'frangi',   'conv_out'};
            connect{end+1} = {'conv_out', 'sigmoid'};
    end

    if ismember(archMode, {'frangi_threshold','frangi_linear','frangi_multichannel'})
        layers{end+1} = dicePixelClassificationLayer('Name','loss');
        if ~strcmp(archMode, 'frangi_threshold')
            % frangi_linear / frangi_multichannel: conv_out → sigmoid → loss
            layers{end+1} = sigmoidLayer('Name','sigmoid');
            connect{end+1} = {'conv_out', 'sigmoid'};
            connect{end+1} = {'sigmoid',  'loss'};
        end
        % frangi_threshold: frangi → loss already connected in switch above
        lgraph = assembleDag(layers, connect);
        return
    end

    % ── U-Net architectures ('unet' and 'frangi_unet') ───────────────────
    nF = opts.initFilters;
    Dp = opts.encoderDepth;

    if strcmp(archMode, 'frangi_unet')
        layers{end+1} = learnableFrangiLayer(nFrangiCh, opts.sigmaMin, ...
                            opts.sigmaMax, 'ReduceMax',true, 'Name','frangi');
        connect{end+1} = {'input', 'frangi'};

        layers{end+1} = concatenationLayer(4, 2, 'Name','cat_input');
        connect{end+1} = {'input',  'cat_input/in1'};
        connect{end+1} = {'frangi', 'cat_input/in2'};

        prevName = 'cat_input';
        inCh     = 2;
    else   % 'unet'
        prevName = 'input';
        inCh     = 1;
    end

    % ── Encoder ──────────────────────────────────────────────────────────
    skipNames = cell(1, Dp);

    for d = 1:Dp
        outCh   = nF;
        blkName = sprintf('enc%d', d);
        [layers, connect, lastLayer] = addEncoderBlock(layers, connect, ...
                                        prevName, inCh, outCh, blkName);
        skipNames{d} = lastLayer;

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

        layers{end+1} = transposedConv3dLayer(2, outCh, 'Stride',2, 'Name',upName);
        connect{end+1} = {prevName, upName};

        layers{end+1} = concatenationLayer(4, 2, 'Name',catName);
        connect{end+1} = {upName,       sprintf('%s/in1',catName)};
        connect{end+1} = {skipNames{d}, sprintf('%s/in2',catName)};

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

    lgraph = assembleDag(layers, connect);
end

% =========================================================================
function mode = resolveArchMode(opts)
    if isfield(opts,'archMode') && ~isempty(opts.archMode)
        mode = opts.archMode;
    elseif isfield(opts,'useFrangi') && ~opts.useFrangi
        mode = 'unet';
    else
        mode = 'frangi_unet';
    end
end

% =========================================================================
function lgraph = assembleDag(layers, connect)
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
