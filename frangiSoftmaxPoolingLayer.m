classdef frangiSoftmaxPoolingLayer < nnet.layer.Layer & nnet.layer.Formattable
% FRANGISOFTMAXPOOLINGLAYER  Learnable softmax-weighted pooling of Frangi channels.
%
%   Collapses [H W D numChannels B] → [H W D 1 B] via a weighted sum whose
%   coefficients are the softmax of a learnable logit vector.
%
%   LEARNABLE PARAMETER
%   ───────────────────
%   logWeights  [numChannels 1] — pre-softmax channel logits
%
%   BEHAVIOUR
%   ─────────
%   pooled = sum_ch( softmax(logWeights)_ch * X(:,:,:,ch,:) )
%
%   When all logWeights are equal the output is a simple channel average.
%   As one logWeight grows large relative to the others the output approaches
%   that channel's response (soft-max behaviour).

    properties (Learnable)
        logWeights   % [numChannels 1]
    end

    properties
        NumChannels
    end

    methods
        function layer = frangiSoftmaxPoolingLayer(numChannels, varargin)
            layer.NumChannels = numChannels;
            layer.Name        = 'frangi_pool';
            layer.Description = 'Learnable softmax pooling of Frangi channels';

            % Initialise to uniform average (all logits equal)
            layer.logWeights = dlarray(zeros(numChannels, 1, 'single'));

            for k = 1:2:numel(varargin)
                if strcmpi(varargin{k},'Name')
                    layer.Name = varargin{k+1};
                end
            end
        end

        % -----------------------------------------------------------------
        function Z = predict(layer, X)
        % PREDICT  Softmax-weighted sum across channel dim 4.
        %
        %   X : [H W D nC B]  'SSSCB'  or  [H W D nC]  'SSSC'
        %   Z : [H W D 1  B]  'SSSCB'  or  [H W D 1]   'SSSC'

            nC = layer.NumChannels;

            % Numerically stable softmax (shift by max; max() only for the constant)
            w       = layer.logWeights;                   % [nC 1] tracked
            w_shift = w - max(extractdata(w));            % stable shift (double scalar)
            w_exp   = exp(w_shift);
            w_soft  = w_exp ./ sum(w_exp);               % [nC 1] tracked

            sz        = size(X);
            has_batch = (numel(sz) == 5);
            X_raw     = stripdims(X);                    % remove format labels; graph intact

            if has_batch
                H = sz(1); W = sz(2); D = sz(3); B = sz(5);
                w_bcast = reshape(w_soft, [1 1 1 nC 1]);
                Z = dlarray(sum(X_raw .* w_bcast, 4), 'SSSCB');
            else
                H = sz(1); W = sz(2); D = sz(3);        %#ok<NASGU>
                w_bcast = reshape(w_soft, [1 1 1 nC]);
                Z = dlarray(sum(X_raw .* w_bcast, 4), 'SSSC');
            end
        end
    end
end
