classdef learnableFrangiLayer < nnet.layer.Layer & nnet.layer.Formattable
% LEARNABLEFRANGUYLAYER  Differentiable multi-scale 3-D Frangi vesselness filter.
%
%   This layer implements a fully differentiable 3-D version of the Frangi
%   vesselness filter (Frangi et al., 1998) whose scale parameters and
%   sensitivity coefficients are learned end-to-end via backpropagation.
%
%   LEARNABLE PARAMETERS
%   ────────────────────
%   logSigmas  – log of Gaussian scales σ_k, shape [1 1 1 1 numScales]
%                Parameterised in log-space to enforce σ > 0.
%   logAlpha   – log of plate/tube discrimination coefficient α (scalar)
%   logBeta    – log of blob-suppression coefficient β (scalar)
%   logC       – log of background/noise suppression coefficient c (scalar)
%
%   FORWARD PASS
%   ────────────
%   For each scale σ_k:
%     1. Convolve input with 6 Gaussian 2nd-order derivative kernels to
%        obtain Hessian components (Lxx, Lxy, Lxz, Lyy, Lyz, Lzz).
%     2. Compute eigenvalues (λ1 ≥ λ2 ≥ λ3) of the 3×3 Hessian at each
%        voxel using Cardano's closed-form trigonometric method.
%        For bright vessels on dark background: λ3 ≤ λ2 < 0 ≈ λ1.
%     3. Compute 3-D Frangi vesselness (condition: λ2 < 0 AND λ3 < 0):
%          RA = |λ2|/|λ3|              (plate vs tube: ~1 for tubes)
%          RB = |λ1|/sqrt(|λ2·λ3|)   (blob measure: ~0 for tubes)
%          S² = λ1² + λ2² + λ3²       (structureness)
%          V  = (1-exp(-RA²/2α²)) · exp(-RB²/2β²) · (1-exp(-S²/2c²))
%   Maximum vesselness across scales is returned.
%
%   GRADIENT NOTES
%   ──────────────
%   All operations use dlarray arithmetic so MATLAB's auto-diff computes
%   exact gradients w.r.t. logSigmas, logAlpha, logBeta, logC, and the input.
%   Gaussian kernels are analytically generated from the learned sigmas
%   inside predict(); they are NOT stored as separate learnable parameters.
%   The vessel indicator (λ2 < 0 AND λ3 < 0) and eigenvalue sort use a
%   straight-through pattern (extractdata breaks the graph for those masks).

    properties (Learnable)
        logSigmas   % [1 1 1 1 numScales] – log-scale parameters
        logAlpha    % scalar – plate/tube discrimination
        logBeta     % scalar – blob suppression
        logC        % scalar – background/noise suppression
    end

    properties
        NumScales
    end

    methods
        function layer = learnableFrangiLayer(numScales, sigmaMin, sigmaMax, varargin)
            layer.NumScales   = numScales;
            layer.Name        = 'frangi';
            layer.Description = 'Learnable 3-D multi-scale Frangi vesselness';

            % Initialise sigmas log-uniformly between sigmaMin and sigmaMax
            sigmas = exp(linspace(log(sigmaMin), log(sigmaMax), numScales));
            layer.logSigmas = dlarray(reshape(log(sigmas), [1 1 1 1 numScales]));

            % Reasonable 3-D Frangi defaults
            layer.logAlpha = dlarray(log(0.5));
            layer.logBeta  = dlarray(log(0.5));
            layer.logC     = dlarray(log(500.0));

            % Parse optional Name-Value
            for k = 1:2:numel(varargin)
                if strcmpi(varargin{k},'Name')
                    layer.Name = varargin{k+1};
                end
            end
        end

        % -----------------------------------------------------------------
        function Z = predict(layer, X)
        % PREDICT  Forward pass – returns max-scale vesselness volume.
        %
        %   image3dInputLayer([H W D 1]) delivers:
        %     [H W D 1 B]  'SSSCB'  during training  (5-D, explicit batch)
        %     [H W D 1]    'SSSC'   during network validation  (4-D, no batch)
        %   Both ranks are handled identically; Z matches the input rank.

            sigmas = exp(layer.logSigmas);   % [1 1 1 1 nS]
            alpha  = exp(layer.logAlpha);
            beta   = exp(layer.logBeta);
            c      = exp(layer.logC);

            sz = size(X);
            has_batch = (numel(sz) == 5);   % false during validation

            if has_batch
                H = sz(1); W = sz(2); D = sz(3); B = sz(5);
                if sz(4) > 1, X = X(:,:,:,1,:); end
                X5 = X;   % already [H W D 1 B]
            else
                % Validation: [H W D 1] — wrap a singleton batch dim
                H = sz(1); W = sz(2); D = sz(3);
                X5 = dlarray(reshape(stripdims(X), H, W, D, 1, 1), 'SSSCB');
                B  = 1;
            end

            nS = layer.NumScales;

            % Initialise accumulator as a labelled dlarray of zeros so that
            % max() between accumulator and V_s preserves format labels.
            V_max = dlarray(zeros(H, W, D, 1, B, 'single'), 'SSSCB');

            for s = 1:nS
                sig = sigmas(1,1,1,1,s);
                V_s = frangiScaleResponse3D(X5, sig, alpha, beta, c);
                V_max = max(V_max, V_s);
            end

            % Return same rank as the input so the downstream
            % concatenationLayer sees matching shapes from both branches.
            if has_batch
                Z = V_max;   % [H W D 1 B]  'SSSCB'
            else
                Z = dlarray(reshape(stripdims(V_max), H, W, D, 1), 'SSSC');
            end
        end
    end
end

% =========================================================================
% LOCAL HELPERS (file-local, not class methods)
% =========================================================================

function V = frangiScaleResponse3D(X, sigma, alpha, beta, c)
% FRANGIKCALERESPONSE3D  Single-scale differentiable 3-D vesselness.

    % ── Hessian via 2nd-derivative Gaussian convolution ──────────────────
    [Lxx, Lxy, Lxz, Lyy, Lyz, Lzz] = hessianResponse3D(X, sigma);

    % ── Eigenvalues (Cardano): ev1 >= ev2 >= ev3 ─────────────────────────
    % For bright vessels: ev3 <= ev2 < 0 ≈ ev1
    [ev1, ev2, ev3] = eigenvalues3x3sym(Lxx, Lxy, Lxz, Lyy, Lyz, Lzz);

    % ── 3-D Frangi vesselness measures ───────────────────────────────────
    % |λ1| = |ev1|  (smallest abs, ~0 along tube axis)
    % |λ2| = |ev2|  (medium, negative for vessels)
    % |λ3| = |ev3|  (largest abs, most negative)
    abs_ev2 = abs(ev2);
    abs_ev3 = abs(ev3) + 1e-8;
    abs_ev1 = abs(ev1);

    RA = abs_ev2 ./ abs_ev3;                                 % plate vs tube [0,1]
    RB = abs_ev1 ./ (sqrt(abs_ev2 .* abs_ev3) + 1e-8);      % blob measure
    S2 = ev1.^2 + ev2.^2 + ev3.^2;                          % structureness

    V = (1 - exp(-RA.^2 ./ (2*alpha^2))) ...
      .*  exp(-RB.^2 ./ (2*beta^2))      ...
      .* (1 - exp(-S2  ./ (2*c^2)));

    % Suppress voxels where λ2 >= 0 or λ3 >= 0 (not tubular/dark background)
    ev2_data = double(extractdata(ev2));
    ev3_data = double(extractdata(ev3));
    vessel_mask = dlarray(single((ev2_data < 0) & (ev3_data < 0)));
    V = V .* vessel_mask;
end

% -------------------------------------------------------------------------
function [Lxx, Lxy, Lxz, Lyy, Lyz, Lzz] = hessianResponse3D(X, sigma)
% HESSIANRESPONSE3D  Convolve X with 6 3-D 2nd-derivative-of-Gaussian kernels.

    sig_val = double(extractdata(sigma));
    ks      = max(5, 2*ceil(3*sig_val)+1);   % kernel size: ~6σ span

    [Gxx, Gxy, Gxz, Gyy, Gyz, Gzz] = gaussianHessianKernels3D(sig_val, ks);

    scale = sig_val^2;   % Frangi scale normalisation

    Lxx = scale * dlConv3(X, dlarray(single(Gxx)));
    Lxy = scale * dlConv3(X, dlarray(single(Gxy)));
    Lxz = scale * dlConv3(X, dlarray(single(Gxz)));
    Lyy = scale * dlConv3(X, dlarray(single(Gyy)));
    Lyz = scale * dlConv3(X, dlarray(single(Gyz)));
    Lzz = scale * dlConv3(X, dlarray(single(Gzz)));
end

% -------------------------------------------------------------------------
function [Gxx, Gxy, Gxz, Gyy, Gyz, Gzz] = gaussianHessianKernels3D(sigma, ks)
% GAUSSIANHESSIANKERNELS3D  Analytically compute 3-D 2nd-order Gaussian kernels.

    r = floor(ks/2);
    [x, y, z] = meshgrid(-r:r, -r:r, -r:r);   % note: meshgrid→ x varies along dim2
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
% EIGENVALUES3X3SYM  Cardano's method for 3×3 symmetric matrices, voxelwise.
%
%   Returns eigenvalues with ev1 >= ev2 >= ev3 (value-sorted descending).
%   For bright tubular structures: ev3 <= ev2 < 0 <= ev1.
%
%   Reference: Smith (1961) "Eigenvalues of a symmetric 3×3 matrix",
%              Communications of the ACM.

    % Mean of diagonal (= trace/3)
    q = (Lxx + Lyy + Lzz) / 3;

    % Off-diagonal energy
    p1 = Lxy.^2 + Lxz.^2 + Lyz.^2;

    % Scale factor for normalised matrix B
    p2 = (Lxx - q).^2 + (Lyy - q).^2 + (Lzz - q).^2 + 2*p1;
    p  = sqrt(p2 / 6 + 1e-10);   % add eps to avoid sqrt(0)

    % Normalised traceless matrix B = (A - qI)/p
    inv_p = 1 ./ (p + 1e-10);
    Bxx = (Lxx - q) .* inv_p;
    Byy = (Lyy - q) .* inv_p;
    Bzz = (Lzz - q) .* inv_p;
    Bxy = Lxy .* inv_p;
    Bxz = Lxz .* inv_p;
    Byz = Lyz .* inv_p;

    % det(B)/2  (analytical 3×3 determinant, symmetric B so Bji=Bij)
    detB = Bxx .* (Byy.*Bzz - Byz.^2) ...
         - Bxy .* (Bxy.*Bzz - Byz.*Bxz) ...
         + Bxz .* (Bxy.*Byz - Byy.*Bxz);
    r = detB / 2;

    % Clamp r to (-1, 1) for acos numerical stability
    r = min(max(r, -1 + 1e-7), 1 - 1e-7);
    phi = acos(r) / 3;

    % Eigenvalues in descending order: ev1 >= ev2 >= ev3
    %   cos(phi)         in [0.5, 1]     → ev1 is largest
    %   cos(phi+2π/3)    in [-1, -0.5]   → ev3 is smallest
    %   ev2 from trace constraint
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
