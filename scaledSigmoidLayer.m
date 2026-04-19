classdef scaledSigmoidLayer < nnet.layer.Layer & nnet.layer.Formattable
% SCALEDSIGMOIDLAYER  Sigmoid with learnable scale and fixed bias.
%
%   Z = sigmoid(scale .* X + bias)
%
%   LEARNABLE PARAMETERS
%   ────────────────────
%   scale  – scalar sharpness (init 1.0); controls steepness of the gate
%
%   FIXED PARAMETERS
%   ────────────────
%   bias   – fixed at -0.003; zero-crossing at X = 0.003 when scale=1,
%            matching the known-good Frangi vesselness threshold.

    properties (Learnable)
        scale   % scalar — learnable sharpness
    end

    properties
        bias = single(-0.003)   % fixed offset; zero-crossing at x = 0.003 when scale=1
    end

    methods
        function layer = scaledSigmoidLayer(varargin)
            layer.Name        = 'scaled_sigmoid';
            layer.Description = 'Sigmoid with learnable scale, fixed bias=-0.003';

            layer.scale = dlarray(single(1.0));

            for k = 1:2:numel(varargin)
                if strcmpi(varargin{k}, 'Name')
                    layer.Name = varargin{k+1};
                end
            end
        end

        function Z = predict(layer, X)
            Z = sigmoid(layer.scale .* stripdims(X) + layer.bias);
            % Re-attach format labels from X so downstream layers see 'SSSCB'/'SSSC'
            if ~isempty(dims(X))
                Z = dlarray(Z, dims(X));
            end
        end
    end
end
