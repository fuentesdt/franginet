classdef learnableFrangiLayer < nnet.layer.Layer & nnet.layer.Formattable
% LEARNABLEFRANGUYLAYER  Differentiable multi-scale 3-D Frangi vesselness filter.
%
%   Each output channel has its own independent set of learnable parameters,
%   letting the network discover multiple vessel-appearance modes end-to-end.
%
%   LEARNABLE PARAMETERS  (each has a leading numChannels dimension)
%   ────────────────────
%   logSigmas  – log Gaussian scales, shape [numChannels numScales]
%   logAlpha   – log plate/tube coefficient,  shape [numChannels 1]
%   logBeta    – log blob-suppression coeff,  shape [numChannels 1]
%   logC       – log background/noise coeff,  shape [numChannels 1]
%
%   FORWARD PASS
%   ────────────
%   For each channel ch and scale σ_k:
%     1. Convolve input with 6 Gaussian 2nd-order derivative kernels.
%     2. Compute eigenvalues (λ1 ≥ λ2 ≥ λ3) via Cardano's method.
%     3. Compute 3-D Frangi vesselness (condition: λ2 < 0 AND λ3 < 0).
%   Maximum vesselness across scales is kept per channel.
%   Output: [H W D numChannels B] (format 'SSSCB').

    properties (Learnable)
        logSigmas   % [numChannels numScales] – log-scale parameters
        logAlpha    % [numChannels 1]         – plate/tube discrimination
        logBeta     % [numChannels 1]         – blob suppression
        logC        % [numChannels 1]         – background/noise suppression
    end

    properties
        NumScales
        NumChannels
    end

    methods
        function layer = learnableFrangiLayer(numChannels, numScales, sigmaMin, sigmaMax, varargin)
            layer.NumChannels = numChannels;
            layer.NumScales   = numScales;
            layer.Name        = 'frangi';
            layer.Description = 'Learnable 3-D multi-scale Frangi vesselness';

            % Initialise sigmas log-uniformly; each channel gets the same init
            sigmas = exp(linspace(log(sigmaMin), log(sigmaMax), numScales));
            layer.logSigmas = dlarray(repmat(log(sigmas), numChannels, 1));   % [nC nS]

            % Per-channel scalar params — same init for all channels
            layer.logAlpha = dlarray(log(0.5)  * ones(numChannels, 1, 'single'));
            layer.logBeta  = dlarray(log(0.5)  * ones(numChannels, 1, 'single'));
            layer.logC     = dlarray(log(500.0)* ones(numChannels, 1, 'single'));

            for k = 1:2:numel(varargin)
                if strcmpi(varargin{k},'Name')
                    layer.Name = varargin{k+1};
                end
            end
        end

        % -----------------------------------------------------------------
        function Z = predict(layer, X)
        % PREDICT  Forward pass – returns multi-channel vesselness volume.
        %
        %   Output: [H W D numChannels B]  'SSSCB'

            nC = layer.NumChannels;
            nS = layer.NumScales;

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
                alpha_ch = exp(layer.logAlpha(ch));
                beta_ch  = exp(layer.logBeta(ch));
                c_ch     = exp(layer.logC(ch));

                V_max = dlarray(zeros(H, W, D, 1, B, 'single'), 'SSSCB');
                for s = 1:nS
                    sig   = exp(layer.logSigmas(ch, s));
                    V_s   = frangiScaleResponse3D(X5, sig, alpha_ch, beta_ch, c_ch);
                    V_max = max(V_max, V_s);
                end
                channels{ch} = V_max;   % [H W D 1 B]
            end

            Z_batch = cat(4, channels{:});   % [H W D nC B]  'SSSCB'

            if has_batch
                Z = Z_batch;
            else
                Z = dlarray(reshape(stripdims(Z_batch), H, W, D, nC), 'SSSC');
            end
        end
    end
end

% =========================================================================
% LOCAL HELPERS (file-local, not class methods)
% =========================================================================

function V = frangiScaleResponse3D(X, sigma, alpha, beta, c)
% FRANGIKCALERESPONSE3D  Single-scale differentiable 3-D vesselness.

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
% Cardano's method — returns ev1 >= ev2 >= ev3 (value-sorted descending).

    q = (Lxx + Lyy + Lzz) / 3;
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
    r = detB / 2;

    r   = min(max(r, -1 + 1e-7), 1 - 1e-7);
    phi = acos(r) / 3;

    ev1 = q + 2*p .* cos(phi);
    ev3 = q + 2*p .* cos(phi + 2*pi/3);
    ev2 = 3*q - ev1 - ev3;
end

% -------------------------------------------------------------------------
function Y = dlConv3(X, K)
% DLCONV3  3-D convolution of dlarray X with kernel K (same-padding).
%   X : [H W D 1 B]  dlarray (format 'SSSCB')
%   K : [kH kW kD]   dlarray kernel (reshaped to [kH kW kD 1 1])

    kSz = size(K);
    kH = kSz(1);  kW = kSz(2);  kD = kSz(3);
    K5D = reshape(K, [kH kW kD 1 1]);
    padH = floor(kH/2);
    padW = floor(kW/2);
    padD = floor(kD/2);
    Y = dlconv(X, K5D, 0, 'Padding', [padH padH padW padW padD padD]);
end
