%% tuneBezierTraining.m
%  -----------------------------------------------------------------------
%  Generate N synthetic 32^3 NIfTI training patches from random 3-D cubic
%  Bezier curves.  Each patch has a matching binary mask at the known curve
%  location.  SNR = signal_peak / noise_std (default 10).
%
%  Output layout
%    bezier_training/
%      images/  patch_001.nii  ...  patch_N.nii
%      masks/   mask_001.nii   ...  mask_N.nii
%      manifest.csv   (id, mask_path, image_path — democomparison.m format)
%
%  The Bezier tube is rendered as a binary cylinder of radius TUBE_RADIUS
%  voxels, then a Gaussian cross-section profile is applied so the image
%  intensity falls off smoothly from the centreline.
%  -----------------------------------------------------------------------

clear; clc;

%% ── Parameters ───────────────────────────────────────────────────────────
N               = 12;           % number of patches
PATCH_SZ        = [32 32 32];   % voxels
SNR             = 10;           % signal peak / noise std
N_CURVES_RANGE  = [5 10];       % random number of Bezier curves per patch
RADIUS_RANGE    = [1.0 5.0];    % per-curve tube radius range (voxels)
N_BEZIER_T      = 2000;         % Bezier sample density (≥ 50 × diagonal)
OUT_DIR         = 'bezier_training';
rng(42);                        % reproducible

%% ── Directory setup ──────────────────────────────────────────────────────
imgDir = fullfile(OUT_DIR, 'images');
mskDir = fullfile(OUT_DIR, 'masks');
for d = {OUT_DIR, imgDir, mskDir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

csvPath = fullfile(OUT_DIR, 'manifest.csv');
fid     = fopen(csvPath, 'w');

%% ── Voxel grid (column vectors for broadcasting) ─────────────────────────
[GX, GY, GZ] = ndgrid(1:PATCH_SZ(1), 1:PATCH_SZ(2), 1:PATCH_SZ(3));
gx = GX(:);  gy = GY(:);  gz = GZ(:);   % [V×1]

%% ── Generate patches ─────────────────────────────────────────────────────
fprintf('Generating %d Bezier training patches (%d^3, SNR=%g, %d-%d curves/patch, r=%.0f-%.0f vox)...\n', ...
        N, PATCH_SZ(1), SNR, N_CURVES_RANGE(1), N_CURVES_RANGE(2), ...
        RADIUS_RANGE(1), RADIUS_RANGE(2));

noise_std = 1.0 / SNR;

for i = 1:N

    nCurves = randi(N_CURVES_RANGE);   % number of curves in this patch
    signal  = zeros(PATCH_SZ, 'single');
    mask    = false(PATCH_SZ);

    for c = 1:nCurves

        %% Random cubic Bezier control points in [15%, 85%] of volume
        lo = 0.15 * PATCH_SZ;
        hi = 0.85 * PATCH_SZ;
        P  = lo + rand(4, 3) .* (hi - lo);   % [4×3] P0..P3, cols=x,y,z

        %% Per-curve radius drawn uniformly from RADIUS_RANGE
        r = RADIUS_RANGE(1) + rand() * diff(RADIUS_RANGE);

        %% Sample curve: B(t), t in [0,1]
        t = linspace(0, 1, N_BEZIER_T)';
        B = (1-t).^3       .* P(1,:) + ...
            3*(1-t).^2.*t  .* P(2,:) + ...
            3*(1-t)   .*t.^2 .* P(3,:) + ...
            t.^3            .* P(4,:);      % [T×3]

        %% Minimum distance from every voxel to this curve  [V×1]
        dx2 = (gx - B(:,1)').^2;
        dy2 = (gy - B(:,2)').^2;
        dz2 = (gz - B(:,3)').^2;
        minDist = sqrt(min(dx2 + dy2 + dz2, [], 2));
        minDist = reshape(minDist, PATCH_SZ);

        %% Accumulate mask and Gaussian-profile signal for this curve
        inTube   = minDist <= r;
        mask     = mask | inTube;
        sigma_c  = r / 2;
        profile  = single(exp(-minDist.^2 / (2 * sigma_c^2)));
        profile(~inTube) = 0;
        signal   = max(signal, profile);   % max-composite keeps peaks sharp
    end

    %% Add noise
    img = signal + noise_std * single(randn(PATCH_SZ));

    %% Write NIfTI
    imgName = sprintf('patch_%03d.nii', i);
    mskName = sprintf('mask_%03d.nii',  i);

    niftiwrite(img,          fullfile(imgDir, imgName));
    niftiwrite(uint8(mask),  fullfile(mskDir, mskName));

    %% CSV row: id, mask_path, image_path  (democomparison.m column order)
    relImg = ['images/' imgName];
    relMsk = ['masks/'  mskName];
    fprintf(fid, '%d,%s,%s\n', i, relMsk, relImg);

    fprintf('  [%2d/%d]  curves=%d  vessel_vox=%d (%.1f%%)\n', ...
            i, N, nCurves, sum(mask(:)), 100*mean(mask(:)));
end

fclose(fid);
fprintf('\nDone.  Manifest: %s\n', csvPath);
