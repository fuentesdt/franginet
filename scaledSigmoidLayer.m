classdef scaledSigmoidLayer < nnet.layer.Layer & nnet.layer.Formattable
% SCALEDSIGMOIDLAYER  Sigmoid with learnable affine pre-transform.
%
%   Z = sigmoid(scale .* X + bias)
%
%   LEARNABLE PARAMETERS
%   ────────────────────
%   scale  – scalar multiplicative factor   (init 1.0,  unconstrained)
%   bias   – scalar additive offset         (init -2.0, unconstrained)
%
%   Initialising bias to -2 means sigmoid(-2) ≈ 0.12 at zero input, so the
%   network starts with a high effective threshold and must learn to lower it
%   only for voxels with strong Frangi response.  This avoids the degenerate
%   all-positive output that occurs when bias = 0 and all Frangi values ≥ 0.

    properties (Learnable)
        scale   % scalar
        bias    % scalar
    end

    methods
        function layer = scaledSigmoidLayer(varargin)
            layer.Name        = 'scaled_sigmoid';
            layer.Description = 'Sigmoid with learnable scale and bias';

            layer.scale = dlarray(single( 1.0));
            layer.bias  = dlarray(single(-2.0));

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
