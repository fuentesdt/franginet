classdef learnableThresholdLayer < nnet.layer.Layer & nnet.layer.Formattable
% LEARNABLETHRESHOLDLAYER  Differentiable soft threshold on scalar inputs.
%
%   Z = sigmoid( exp(logScale) .* (X - threshold) )
%
%   LEARNABLE PARAMETERS
%   ────────────────────
%   logScale   scalar — log of sharpness; exp(logScale) > 0 always
%   threshold  scalar — decision boundary in the input domain
%
%   At initialisation the layer approximates a step function at
%   initThreshold.  Training shifts the threshold and adjusts sharpness.
%   Output is in (0,1) so downstream loss layers receive valid probabilities
%   and binary predictions at 0.5 are meaningful.

    properties (Learnable)
        logScale    % scalar, init log(10) → scale=10 (sharp initial boundary)
        threshold   % scalar, init from constructor argument
    end

    methods
        function layer = learnableThresholdLayer(initThreshold, varargin)
            if nargin < 1, initThreshold = 0.1; end
            layer.Name        = 'learnable_threshold';
            layer.Description = 'Learnable sigmoid threshold';
            layer.logScale    = dlarray(single(0.0));   % scale=1 keeps sigmoid linear at init
            layer.threshold   = dlarray(single(initThreshold));

            for k = 1:2:numel(varargin)
                if strcmpi(varargin{k}, 'Name')
                    layer.Name = varargin{k+1};
                end
            end
        end

        function Z = predict(layer, X)
            scale = exp(layer.logScale);
            Z_raw = sigmoid(scale .* (stripdims(X) - layer.threshold));
            if ~isempty(dims(X))
                Z = dlarray(Z_raw, dims(X));
            else
                Z = Z_raw;
            end
        end
    end
end
