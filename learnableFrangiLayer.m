classdef learnableFrangiLayer < nnet.layer.Layer & nnet.layer.Formattable
% LEARNABLEFRANGUYLAYER  Differentiable 3-D Frangi vesselness, one scale per channel.
%
%   Each output channel applies a single-scale Frangi filter with its own
%   independent set of learnable parameters.  numFrangiChannels channels are
%   produced; downstream frangiSoftmaxPoolingLayer collapses them to 1 channel
%   via learnable softmax-weighted pooling.
%
%   LEARNABLE PARAMETERS  (leading dimension = numChannels)
%   ────────────────────
%   logSigmas  [numChannels 1] — log Gaussian scale per channel
%   logAlpha   [numChannels 1] — log plate/tube discrimination coefficient
%   logBeta    [numChannels 1] — log blob-suppression coefficient
%   logC       [numChannels 1] — log background/noise suppression coefficient
%
%   OUTPUT: [H W D 1 B]  format 'SSSCB'  — pixelwise max across all channels

    properties (Learnable)
        logSigmas   % [numChannels 1]
        logAlpha    % [numChannels 1]
        logBeta     % [numChannels 1]
        logC        % [numChannels 1]
    end

    properties
        NumChannels
        ReduceMax   % logical — true: pixelwise max across channels (1-ch output)
                    %           false: return all channels ([H W D nC B])
    end

    methods
        function layer = learnableFrangiLayer(numChannels, sigmaMin, sigmaMax, varargin)
            layer.NumChannels = numChannels;
            layer.ReduceMax   = true;
            layer.Name        = 'frangi';
            layer.Description = 'Learnable 3-D single-scale-per-channel Frangi vesselness';

            % Initialise sigmas log-uniformly across [sigmaMin, sigmaMax]
            sigmas = exp(linspace(log(sigmaMin), log(sigmaMax), numChannels));
            layer.logSigmas = dlarray(single(log(sigmas(:))));   % [nC 1]

            layer.logAlpha = dlarray(log(0.5)   * ones(numChannels, 1, 'single'));
            layer.logBeta  = dlarray(log(0.5)   * ones(numChannels, 1, 'single'));
            layer.logC     = dlarray(log(500.0) * ones(numChannels, 1, 'single'));

            for k = 1:2:numel(varargin)
                if strcmpi(varargin{k},'Name')
                    layer.Name = varargin{k+1};
                elseif strcmpi(varargin{k},'ReduceMax')
                    layer.ReduceMax = varargin{k+1};
                end
            end
        end

        % -----------------------------------------------------------------
        function Z = predict(layer, X)
        % PREDICT  One Frangi response per channel; output [H W D nC B].

            nC = layer.NumChannels;
            sz = size(X);
            has_batch = (numel(sz) == 5);

            if has_batch
                H = sz(1); W = sz(2); D = sz(3); B = sz(5);
                if sz(4) > 1, X = X(:,:,:,1,:); end
                X5 = X;
            else
                H = sz(1); W = sz(2); D = sz(3);
                X5 = dlarray(reshape(stripdims(X), H, W, D, 1, 1), 'SSSCB');
                B  = 1;
            end

            channels = cell(1, nC);
            for ch = 1:nC
                sig   = exp(layer.logSigmas(ch));
                alpha = exp(layer.logAlpha(ch));
                beta  = exp(layer.logBeta(ch));
                c     = exp(layer.logC(ch));
                channels{ch} = frangiScaleResponse3D(X5, sig, alpha, beta, c);
            end

            Z_all = cat(4, channels{:});   % [H W D nC B]

            if layer.ReduceMax
                % Pixelwise max — gradients flow to the winning channel
                Z_raw = max(stripdims(Z_all), [], 4);   % [H W D 1 B]
                if has_batch
                    Z = dlarray(Z_raw, 'SSSCB');
                else
                    Z = dlarray(reshape(Z_raw, H, W, D, 1), 'SSSC');
                end
            else
                % Return all channels for downstream learned combination
                if has_batch
                    Z = Z_all;
                else
                    Z = dlarray(reshape(stripdims(Z_all), H, W, D, nC), 'SSSC');
                end
            end
        end
    end
end

% =========================================================================
% LOCAL HELPERS
% =========================================================================

function V = frangiScaleResponse3D(X, sigma, alpha, beta, c)
    [Lxx, Lxy, Lxz, Lyy, Lyz, Lzz] = hessianResponse3D(X, sigma);
    [ev1, ev2, ev3] = eigenvalues3x3sym(Lxx, Lxy, Lxz, Lyy, Lyz, Lzz);

    abs_ev2 = abs(ev2);
    abs_ev3 = abs(ev3) + 1e-8;
    abs_ev1 = abs(ev1);

    RA = abs_ev2 ./ abs_ev3;
    RB = abs_ev1 ./ (sqrt(abs_ev2 .* abs_ev3) + 1e-8);
    S2 = ev1.^2 + ev2.^2 + ev3.^2;

    V = (1 - exp(-RA.^2 ./ (2*alpha^2))) ...
      .*  exp(-RB.^2 ./ (2*beta^2))      ...
      .* (1 - exp(-S2  ./ (2*c^2)));

    ev2_data = double(extractdata(ev2));
    ev3_data = double(extractdata(ev3));
    vessel_mask = dlarray(single((ev2_data < 0) & (ev3_data < 0)));
    V = V .* vessel_mask;
end

% -------------------------------------------------------------------------
function [Lxx, Lxy, Lxz, Lyy, Lyz, Lzz] = hessianResponse3D(X, sigma)
    sig_val = double(extractdata(sigma));
    ks      = max(5, 2*ceil(3*sig_val)+1);
    [Gxx, Gxy, Gxz, Gyy, Gyz, Gzz] = gaussianHessianKernels3D(sig_val, ks);
    scale = sig_val^2;
    Lxx = scale * dlConv3(X, dlarray(single(Gxx)));
    Lxy = scale * dlConv3(X, dlarray(single(Gxy)));
    Lxz = scale * dlConv3(X, dlarray(single(Gxz)));
    Lyy = scale * dlConv3(X, dlarray(single(Gyy)));
    Lyz = scale * dlConv3(X, dlarray(single(Gyz)));
    Lzz = scale * dlConv3(X, dlarray(single(Gzz)));
end

% -------------------------------------------------------------------------
function [Gxx, Gxy, Gxz, Gyy, Gyz, Gzz] = gaussianHessianKernels3D(sigma, ks)
    r = floor(ks/2);
    [x, y, z] = meshgrid(-r:r, -r:r, -r:r);
    G = exp(-(x.^2 + y.^2 + z.^2) / (2*sigma^2)) / (2*pi*sigma^2)^(3/2);
    Gxx = G .* (x.^2/sigma^4 - 1/sigma^2);
    Gyy = G .* (y.^2/sigma^4 - 1/sigma^2);
    Gzz = G .* (z.^2/sigma^4 - 1/sigma^2);
    Gxy = G .* (x.*y/sigma^4);
    Gxz = G .* (x.*z/sigma^4);
    Gyz = G .* (y.*z/sigma^4);
end

% -------------------------------------------------------------------------
function [ev1, ev2, ev3] = eigenvalues3x3sym(Lxx, Lxy, Lxz, Lyy, Lyz, Lzz)
    q  = (Lxx + Lyy + Lzz) / 3;
    p1 = Lxy.^2 + Lxz.^2 + Lyz.^2;
    p2 = (Lxx - q).^2 + (Lyy - q).^2 + (Lzz - q).^2 + 2*p1;
    p  = sqrt(p2 / 6 + 1e-10);

    inv_p = 1 ./ (p + 1e-10);
    Bxx = (Lxx - q) .* inv_p;
    Byy = (Lyy - q) .* inv_p;
    Bzz = (Lzz - q) .* inv_p;
    Bxy = Lxy .* inv_p;
    Bxz = Lxz .* inv_p;
    Byz = Lyz .* inv_p;

    detB = Bxx .* (Byy.*Bzz - Byz.^2) ...
         - Bxy .* (Bxy.*Bzz - Byz.*Bxz) ...
         + Bxz .* (Bxy.*Byz - Byy.*Bxz);
    r   = min(max(detB/2, -1 + 1e-7), 1 - 1e-7);
    phi = acos(r) / 3;

    ev1 = q + 2*p .* cos(phi);
    ev3 = q + 2*p .* cos(phi + 2*pi/3);
    ev2 = 3*q - ev1 - ev3;
end

% -------------------------------------------------------------------------
function Y = dlConv3(X, K)
    kSz = size(K);
    kH = kSz(1); kW = kSz(2); kD = kSz(3);
    K5D = reshape(K, [kH kW kD 1 1]);
    Y = dlconv(X, K5D, 0, 'Padding', [floor(kH/2) floor(kH/2) ...
                                       floor(kW/2) floor(kW/2) ...
                                       floor(kD/2) floor(kD/2)]);
end
