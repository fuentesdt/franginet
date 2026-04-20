function prob = predictVolume(net, vol, opts)
% PREDICTVOLUME  Patch-based sliding-window inference on a 3-D volume.
%
%   Processes large volumes as overlapping 3-D patches — each patch is run
%   through the network independently, then results are blended with a
%   Gaussian window to suppress boundary artefacts.  This is the inference
%   counterpart of the foreground-biased patch sampling used during training,
%   inspired by the blockedImage / apply pattern.
%
%   prob = predictVolume(net, vol)
%   prob = predictVolume(net, vol, opts)
%
%   INPUTS
%     net              – trained DAGNetwork / SeriesNetwork
%     vol              – single [H W D] intensity volume in [0,1]
%     opts.patchSize   – [H W D] network input size          (default [64 64 64])
%     opts.patchOverlap– overlap (voxels) on each side       (default [8 8 8])
%                        stride = patchSize - 2*patchOverlap
%
%   OUTPUT
%     prob  – single [H W D] probability map, range [0,1]
%
%   BLENDING STRATEGY
%   Each output voxel is the weighted average of all patches that cover it,
%   where the weight is a 3-D Gaussian that peaks at 1 in the patch centre
%   and decays toward the edges.  Patches with higher overlap give smoother
%   results at the cost of more forward passes.

    if nargin < 3, opts = struct(); end
    if ~isfield(opts,'patchSize'),    opts.patchSize    = [64 64 64]; end
    if ~isfield(opts,'patchOverlap'), opts.patchOverlap = [8  8  8 ]; end

    pSz    = opts.patchSize(:)';      % [1×3]
    border = opts.patchOverlap(:)';   % [1×3]
    stride = pSz - 2*border;
    assert(all(stride >= 1), ...
        'patchOverlap must satisfy 2*patchOverlap < patchSize on every dimension.');

    origSz = size(vol, [1 2 3]);

    % ── Pad volume to cover last patch starting positions exactly ────────────
    %   padSz(d) = pSz(d) + nSteps(d)*stride(d)
    %   nSteps   = ceil(max(0, origSz(d)-pSz(d)) / stride(d))
    padSz = zeros(1,3);
    for d = 1:3
        if origSz(d) <= pSz(d)
            padSz(d) = pSz(d);
        else
            nSteps   = ceil((origSz(d) - pSz(d)) / stride(d));
            padSz(d) = pSz(d) + nSteps * stride(d);
        end
    end

    padNeeded = padSz - origSz;
    if any(padNeeded > 0)
        vol = padarray(vol, padNeeded, 0, 'post');
    end

    probAcc = zeros(padSz, 'single');
    wsumAcc = zeros(padSz, 'single');

    % Gaussian blending window — peaks at 1 in centre, tapers smoothly to edges
    W = gaussianWindow3D(pSz);

    % ── Sliding-window forward passes ────────────────────────────────────────
    starts1 = 1 : stride(1) : padSz(1)-pSz(1)+1;
    starts2 = 1 : stride(2) : padSz(2)-pSz(2)+1;
    starts3 = 1 : stride(3) : padSz(3)-pSz(3)+1;

    nPatches = numel(starts1) * numel(starts2) * numel(starts3);
    cnt      = 0;

    for i1 = starts1
        for i2 = starts2
            for i3 = starts3
                e1 = i1 + pSz(1) - 1;
                e2 = i2 + pSz(2) - 1;
                e3 = i3 + pSz(3) - 1;

                patch = vol(i1:e1, i2:e2, i3:e3);
                X     = reshape(im2single(patch), [pSz 1 1]);    % [H W D 1 1]
                p_raw = single(predict(net, X));
                p     = squeeze(p_raw(:,:,:,1,:));               % [H W D] — ch1 only

                probAcc(i1:e1, i2:e2, i3:e3) = probAcc(i1:e1, i2:e2, i3:e3) + p .* W;
                wsumAcc(i1:e1, i2:e2, i3:e3) = wsumAcc(i1:e1, i2:e2, i3:e3) + W;

                cnt = cnt + 1;
                %if cnt == 1 || mod(cnt, max(1, floor(nPatches/5))) == 0 || cnt == nPatches
                %    fprintf('    patch %d/%d\n', cnt, nPatches);
                %end
            end
        end
    end

    % ── Normalise and crop back to original volume size ──────────────────────
    prob = probAcc ./ max(wsumAcc, 1e-6);
    prob = prob(1:origSz(1), 1:origSz(2), 1:origSz(3));
end

% -------------------------------------------------------------------------
function W = gaussianWindow3D(sz)
% 3-D Gaussian window, peak value 1 at the centre.
    ax = cell(3,1);
    for d = 1:3
        t     = linspace(-1, 1, sz(d));
        ax{d} = exp(-2 * t.^2);   % σ ≈ 0.5 in normalised coords
    end
    [A, B, C] = ndgrid(ax{1}, ax{2}, ax{3});
    W = single(A .* B .* C);
end
