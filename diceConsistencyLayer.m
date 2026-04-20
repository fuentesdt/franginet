classdef diceConsistencyLayer < nnet.layer.RegressionLayer
% DICECONSISTENCYLAYER  Dice + Frangi consistency + vessel-break penalty.
%
%   Expects a 2-channel network output Y = [H W D 2 B]:
%     channel 1 — sigmoid UNet prediction  (Y_pred)
%     channel 2 — learnable Frangi vesselness V
%   Target T = [H W D 1 B] — GT binary mask.
%
%   Loss = Dice(Y_pred, T)
%        + lambdaConsistency * MSE(Y_pred, V)
%        + lambdaBreak       * mean( |∇V| * (1 - Y_pred) )
%
%   BREAK PENALTY RATIONALE
%   At a vessel break the A·B·C vesselness is low *at* the break, but high
%   on both sides.  Consequently |∇V| (the spatial gradient magnitude of
%   the vesselness field) is large at the break edges.  Weighting
%   (1 - Y_pred) by |∇V| penalises low network confidence at exactly
%   those voxels that bound a discontinuity, pushing the UNet to close
%   gaps.  |∇V| is computed by finite differences and detached from the
%   computation graph (treated as a fixed importance map), so gradients
%   flow only through Y_pred → UNet/Frangi parameters.
%
%   PARAMETERS
%     lambdaConsistency  – MSE weight             (default 0.1)
%     lambdaBreak        – break-penalty weight   (default 0.1)
%     smooth             – Laplace smoothing       (default 1e-5)

    properties
        lambdaConsistency = 0.1
        lambdaBreak       = 0.1
        smooth            = 1e-5
    end

    methods
        function layer = diceConsistencyLayer(varargin)
            layer.Name        = 'loss';
            layer.Description = 'Dice + Frangi consistency + break penalty';
            for k = 1:2:numel(varargin)
                switch lower(varargin{k})
                    case 'name',               layer.Name              = varargin{k+1};
                    case 'lambdaconsistency',  layer.lambdaConsistency = varargin{k+1};
                    case 'lambdabreak',        layer.lambdaBreak       = varargin{k+1};
                    case 'smooth',             layer.smooth            = varargin{k+1};
                end
            end
        end

        % -----------------------------------------------------------------
        function loss = forwardLoss(layer, Y, T)
            eps    = layer.smooth;
            B      = size(Y, ndims(Y));

            Y_pred = Y(:,:,:,1,:);   % [H W D 1 B]
            V      = Y(:,:,:,2,:);   % [H W D 1 B]

            % ── Dice loss ─────────────────────────────────────────────────
            Yf = reshape(Y_pred, [], B);
            Tf = reshape(T,      [], B);
            intersection = sum(Yf .* Tf, 1);
            union        = sum(Yf, 1) + sum(Tf, 1);
            L_dice = mean(1 - (2*intersection + eps) ./ (union + eps));

            % ── Consistency loss ───────────────────────────────────────────
            L_cons = mean((Y_pred - V).^2, 'all');

            % ── Break penalty ─────────────────────────────────────────────
            % |∇V| detached — used as fixed voxel-importance weight
            gradV  = vesselGradMag(single(V));          % [H W D 1 B]
            L_break = mean(gradV .* (1 - Y_pred), 'all');

            loss = L_dice ...
                 + layer.lambdaConsistency * L_cons  ...
                 + layer.lambdaBreak       * L_break;
        end

        % -----------------------------------------------------------------
        function dX = backwardLoss(layer, Y, T)
            eps  = layer.smooth;
            B    = size(Y, ndims(Y));
            N    = numel(Y(:,:,:,1,:)) / B;

            Y_pred = Y(:,:,:,1,:);
            V      = Y(:,:,:,2,:);

            % ── Dice gradient w.r.t. Y_pred ───────────────────────────────
            Yf = reshape(Y_pred, N, B);
            Tf = reshape(T,      N, B);
            intersection = sum(Yf .* Tf, 1);
            sum_Y        = sum(Yf, 1);
            sum_T        = sum(Tf, 1);
            denom        = (sum_Y + sum_T + eps).^2;
            dDice = reshape( ...
                (2*Tf .* (sum_Y + sum_T + eps) - 2*(2*intersection + eps)) ...
                ./ (N * denom), size(Y_pred));

            % ── Consistency gradients ──────────────────────────────────────
            nVox         = numel(Y_pred);
            delta        = Y_pred - V;
            dCons_dYpred =  2 * delta / nVox;
            dCons_dV     = -2 * delta / nVox;

            % ── Break-penalty gradient w.r.t. Y_pred ──────────────────────
            % gradV is detached — no gradient flows through it to V
            gradV         = vesselGradMag(single(V));   % fixed weight
            dBreak_dYpred = -gradV / nVox;

            % ── Assemble ──────────────────────────────────────────────────
            dX = cat(4, ...
                dDice ...
                + layer.lambdaConsistency * dCons_dYpred ...
                + layer.lambdaBreak       * dBreak_dYpred, ...
                layer.lambdaConsistency   * dCons_dV);
        end
    end
end

% =========================================================================
function gm = vesselGradMag(V)
% Gradient magnitude of V via central finite differences.
% V: [H W D 1 B]  →  gm: [H W D 1 B]  (same size, values ≥ 0)
% Detached from the dlarray computation graph — used as a fixed weight.

    H = size(V,1);  W = size(V,2);  D = size(V,3);  B = size(V,5);
    gm = zeros(H, W, D, 1, B, 'single');

    for b = 1:B
        Vb = V(:,:,:,1,b);   % [H W D]

        % Central differences with reflected (replicated) boundary
        dx = cat(1, Vb(2,:,:)   - Vb(1,:,:), ...
                    (Vb(3:end,:,:) - Vb(1:end-2,:,:)) / 2, ...
                    Vb(end,:,:) - Vb(end-1,:,:));

        dy = cat(2, Vb(:,2,:)   - Vb(:,1,:), ...
                    (Vb(:,3:end,:) - Vb(:,1:end-2,:)) / 2, ...
                    Vb(:,end,:) - Vb(:,end-1,:));

        dz = cat(3, Vb(:,:,2)   - Vb(:,:,1), ...
                    (Vb(:,:,3:end) - Vb(:,:,1:end-2)) / 2, ...
                    Vb(:,:,end) - Vb(:,:,end-1));

        gm(:,:,:,1,b) = sqrt(dx.^2 + dy.^2 + dz.^2);
    end
end
