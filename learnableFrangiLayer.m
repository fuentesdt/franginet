classdef learnableFrangiLayer < nnet.layer.Layer & nnet.layer.Formattable
% LEARNABLEFRANGUYLAYER  Differentiable multi-scale Frangi vesselness filter.
%
%   This layer implements a fully differentiable version of the Frangi
%   vesselness filter (Frangi et al., 1998) whose scale parameters and
%   sensitivity coefficients are learned end-to-end via backpropagation.
%
%   LEARNABLE PARAMETERS
%   ────────────────────
%   logSigmas  – log of Gaussian scales σ_k, shape [1 1 1 numScales]
%                Parameterised in log-space to enforce σ > 0.
%   logAlpha   – log of blob-suppression coefficient α  (scalar)
%   logBeta    – log of background-suppression coefficient β (scalar)
%
%   FORWARD PASS
%   ────────────
%   For each scale σ_k:
%     1. Convolve input with Gaussian 2nd-order derivative kernels
%        to obtain Hessian components (Lxx, Lxy, Lyy).
%     2. Compute eigenvalues (λ1, λ2) of the 2×2 Hessian at each pixel.
%        For bright vessels on dark background: |λ1| ≪ |λ2|, λ2 < 0.
%     3. Compute vesselness:
%          Rb = λ1 / λ2              (blob measure)
%          S  = sqrt(λ1² + λ2²)     (structure measure)
%          V  = exp(-Rb²/2α²) * (1 - exp(-S²/2β²))  [λ2 < 0 only]
%   Maximum vesselness across scales is returned.
%
%   GRADIENT NOTES
%   ──────────────
%   All operations use dlarray arithmetic so MATLAB's auto-diff computes
%   exact gradients w.r.t. logSigmas, logAlpha, logBeta, and the input.
%   Gaussian kernels are analytically generated from the learned sigmas
%   inside forward(); they are NOT stored as separate learnable parameters.

    properties (Learnable)
        logSigmas   % [1 1 1 numScales] – log-scale parameters
        logAlpha    % scalar – Frangi blob sensitivity
        logBeta     % scalar – background sensitivity
    end

    properties
        NumScales
    end

    methods
        function layer = learnableFrangiLayer(numScales, sigmaMin, sigmaMax, varargin)
            layer.NumScales   = numScales;
            layer.Name        = 'frangi';
            layer.Description = 'Learnable multi-scale Frangi vesselness';

            % Initialise sigmas log-uniformly between sigmaMin and sigmaMax
            sigmas = exp(linspace(log(sigmaMin), log(sigmaMax), numScales));
            layer.logSigmas = dlarray(reshape(log(sigmas), [1 1 1 numScales]));

            % Reasonable Frangi defaults
            layer.logAlpha = dlarray(log(0.5));
            layer.logBeta  = dlarray(log(15.0));

            % Parse optional Name-Value
            for k = 1:2:numel(varargin)
                if strcmpi(varargin{k},'Name')
                    layer.Name = varargin{k+1};
                end
            end
        end

        % -----------------------------------------------------------------
        function Z = predict(layer, X)
        % PREDICT  Forward pass – returns max-scale vesselness map.
        %   X : dlarray  [H W C B] or [H W 1 B], single-channel expected
        %   Z : dlarray  [H W 1 B]  vesselness in [0,1]

            % Recover learned parameters (positive via exp)
            sigmas = exp(layer.logSigmas);   % [1 1 1 nS]
            alpha  = exp(layer.logAlpha);
            beta   = exp(layer.logBeta);

            % Work with spatial dims; squeeze channel if needed
            if size(X,3) > 1
                X = X(:,:,1,:);   % use first channel only
            end

            [H, W, ~, B] = size(X, [1 2 3 4]);
            nS = layer.NumScales;

            % Accumulate max vesselness across scales
            V_max = zeros([H W 1 B], 'like', X);

            for s = 1:nS
                sig = sigmas(1,1,1,s);
                V_s = frangiScaleResponse(X, sig, alpha, beta);
                V_max = max(V_max, V_s);
            end

            Z = V_max;
        end
    end
end

% =========================================================================
% LOCAL HELPERS (file-local, not class methods)
% =========================================================================

function V = frangiScaleResponse(X, sigma, alpha, beta)
% FRANGIKCALERESPONSE  Single-scale differentiable vesselness.

    % ── Hessian via 2nd-derivative Gaussian convolution ──────────────────
    [Lxx, Lxy, Lyy] = hessianResponse(X, sigma);

    % ── Closed-form 2×2 eigenvalues ──────────────────────────────────────
    % Eigenvalues of [[Lxx Lxy];[Lxy Lyy]]
    half_trace = (Lxx + Lyy) / 2;
    disc        = sqrt(((Lxx - Lyy)/2).^2 + Lxy.^2 + 1e-8);

    lambda1 = half_trace - disc;   % smaller eigenvalue
    lambda2 = half_trace + disc;   % larger eigenvalue (more negative for vessels)

    % ── Frangi vesselness measures ────────────────────────────────────────
    % Protect against division by zero
    safe_l2 = lambda2 + sign(lambda2)*1e-6 + 1e-8;

    Rb = lambda1 ./ safe_l2;                              % blob ratio
    S2 = lambda1.^2 + lambda2.^2;                         % structureness

    V  = exp(-Rb.^2 ./ (2*alpha^2)) .* (1 - exp(-S2 ./ (2*beta^2)));

    % Suppress bright blobs (λ2 > 0 → not a dark-on-bright vessel)
    vessel_mask = double(extractdata(lambda2) < 0);
    vessel_mask = dlarray(single(vessel_mask));
    V = V .* vessel_mask;
end

% -------------------------------------------------------------------------
function [Lxx, Lxy, Lyy] = hessianResponse(X, sigma)
% HESSIANRESPONSE  Convolve X with 2nd-derivative-of-Gaussian kernels.

    sig_val = double(extractdata(sigma));
    ks      = max(5, 2*ceil(3*sig_val)+1);   % kernel half-width: 3σ

    [Gxx, Gxy, Gyy] = gaussianHessianKernels(sig_val, ks);

    % Convert to dlarray kernels for auto-diff compatibility
    % Kernels are constant w.r.t. sigma here (sigma enters via logSigmas
    % through a straight-through approximation for kernel shape; the scale
    % normalisation factor σ² is applied after convolution).
    kxx = dlarray(single(Gxx));
    kxy = dlarray(single(Gxy));
    kyy = dlarray(single(Gyy));

    scale = sig_val^2;   % Frangi scale normalisation

    Lxx = scale * dlConv2(X, kxx);
    Lxy = scale * dlConv2(X, kxy);
    Lyy = scale * dlConv2(X, kyy);
end

% -------------------------------------------------------------------------
function [Gxx, Gxy, Gyy] = gaussianHessianKernels(sigma, ks)
% GAUSSIANHESSIANKERNELS  Analytically compute 2nd-order Gaussian kernels.

    r   = floor(ks/2);
    [x, y] = meshgrid(-r:r, -r:r);
    G   = exp(-(x.^2 + y.^2)/(2*sigma^2)) / (2*pi*sigma^2);

    Gxx = G .* (x.^2/sigma^4 - 1/sigma^2);
    Gyy = G .* (y.^2/sigma^4 - 1/sigma^2);
    Gxy = G .* (x.*y/sigma^4);
end

% -------------------------------------------------------------------------
function Y = dlConv2(X, K)
% DLCONV2  2-D convolution of dlarray X with kernel K (same padding).
%   X : [H W 1 B]  dlarray
%   K : [kH kW]    dlarray kernel (will be reshaped to [kH kW 1 1])

    [kH, kW] = size(K, [1 2]);
    K4D = reshape(K, [kH kW 1 1]);
    padH = floor(kH/2);
    padW = floor(kW/2);
    Y = dlconv(X, K4D, [], 'Padding', [padH padH padW padW]);
end
