classdef dicePixelClassificationLayer < nnet.layer.RegressionLayer
% DICEPIXELCLASSIFICATIONLAYER  Differentiable Dice + BCE combined loss.
%
%   Loss = λ_dice * L_dice + λ_bce * L_bce
%
%   Dice loss is preferred for vessel segmentation because it is
%   intrinsically robust to class imbalance (foreground vessels are a
%   small fraction of image pixels).
%
%   INPUTS
%     Y  – network output, sigmoid-activated, shape [H W 1 B], values ∈ (0,1)
%     T  – ground-truth binary mask,           shape [H W 1 B], values ∈ {0,1}
%
%   PARAMETERS (set via constructor or properties)
%     lambdaDice  – weight for Dice term  (default 0.7)
%     lambdaBCE   – weight for BCE term   (default 0.3)
%     smooth      – Laplace smoothing for Dice (default 1e-5)

    properties
        lambdaDice = 0.7
        lambdaBCE  = 0.3
        smooth     = 1e-5
    end

    methods
        function layer = dicePixelClassificationLayer(varargin)
            layer.Name        = 'loss';
            layer.Description = 'Dice + BCE vessel segmentation loss';

            for k = 1:2:numel(varargin)
                switch lower(varargin{k})
                    case 'name',        layer.Name        = varargin{k+1};
                    case 'lambdadice',  layer.lambdaDice  = varargin{k+1};
                    case 'lambdabce',   layer.lambdaBCE   = varargin{k+1};
                    case 'smooth',      layer.smooth      = varargin{k+1};
                end
            end
        end

        % -----------------------------------------------------------------
        function loss = forwardLoss(layer, Y, T)
        % FORWARDLOSS  Compute combined Dice + BCE loss.
        %   Y, T : [H W 1 B]

            eps = layer.smooth;

            % ── Dice loss ─────────────────────────────────────────────────
            % Flatten spatial dims per batch element
            Y_flat = reshape(Y, [], size(Y,4));   % [H*W, B]
            T_flat = reshape(T, [], size(T,4));

            intersection = sum(Y_flat .* T_flat, 1);
            union        = sum(Y_flat, 1) + sum(T_flat, 1);
            dice_score   = (2*intersection + eps) ./ (union + eps);
            L_dice       = mean(1 - dice_score);

            % ── Binary cross-entropy loss ─────────────────────────────────
            Y_clip = max(min(Y, 1-1e-7), 1e-7);   % numerical stability
            L_bce  = -mean(T .* log(Y_clip) + (1-T) .* log(1-Y_clip), 'all');

            % ── Combined ─────────────────────────────────────────────────
            loss = layer.lambdaDice * L_dice + layer.lambdaBCE * L_bce;
        end

        % -----------------------------------------------------------------
        function dX = backwardLoss(layer, Y, T, dLdY_upstream)
        % BACKWARDLOSS  Analytical gradients (optional; auto-diff fallback
        %               is used when this method is absent, but explicit
        %               gradients are faster and more stable).

            eps  = layer.smooth;
            B    = size(Y, 4);
            N    = numel(Y) / B;    % pixels per image

            Y_flat = reshape(Y, N, B);
            T_flat = reshape(T, N, B);

            % ── Dice gradient ─────────────────────────────────────────────
            intersection = sum(Y_flat .* T_flat, 1);       % [1 B]
            sum_Y        = sum(Y_flat, 1);
            sum_T        = sum(T_flat, 1);
            denom        = (sum_Y + sum_T + eps).^2;

            dDice_dY = reshape(...
                (2*T_flat .* (sum_Y + sum_T + eps) - 2*(2*intersection + eps)) ...
                ./ (N * denom), size(Y));

            % ── BCE gradient ──────────────────────────────────────────────
            Y_clip      = max(min(Y, 1-1e-7), 1e-7);
            nPix        = numel(Y);
            dBCE_dY     = (- T ./ Y_clip + (1-T) ./ (1-Y_clip)) / nPix;

            % ── Combined ─────────────────────────────────────────────────
            dX = layer.lambdaDice * dDice_dY + layer.lambdaBCE * dBCE_dY;

            if nargin >= 4 && ~isempty(dLdY_upstream)
                dX = dX * dLdY_upstream;
            end
        end
    end
end
