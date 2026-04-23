%% porcine.m
%  -----------------------------------------------------------------------
%  Apply tuned Frangi vesselness to every image listed in a manifest CSV.
%
%  Inputs
%    MANIFEST   — CSV produced by the Processed pipeline (with header row)
%                 columns: subject_id, session, timepoint, image, mask
%    RESULT_MAT — .mat saved by tuneFrangi.m; contains a 'result' struct
%                 with fields: sigmaMin, sigmaMax, numScales, alpha, beta, C
%
%  Output
%    For each input image  <dir>/<stem>.raw.nii.gz
%    writes vesselness map <dir>/<stem>.frangi.nii.gz
%    preserving the NIfTI spatial header of the source image.
%  -----------------------------------------------------------------------

clear; clc;

MANIFEST   = fullfile('Processed', 'manifest.csv');
RESULT_MAT = 'tuneFrangi_result.mat';

%% ── 1. Load tuned parameters ─────────────────────────────────────────────
fprintf('Loading tuned Frangi parameters from %s\n', RESULT_MAT);
tmp = load(RESULT_MAT, 'result');
p   = tmp.result;

sigmas = exp(linspace(log(p.sigmaMin), log(p.sigmaMax), p.numScales));

fprintf('  sigmaMin=%.4f  sigmaMax=%.4f  numScales=%d\n', ...
        p.sigmaMin, p.sigmaMax, p.numScales);
fprintf('  alpha=%.4f  beta=%.4f  C=%.4f\n', p.alpha, p.beta, p.C);
fprintf('  threshold=%.4f\n\n', p.threshold);

%% ── 2. Load manifest ─────────────────────────────────────────────────────
fprintf('Reading manifest: %s\n', MANIFEST);
% Skip header row; columns: subject_id(1), session(2), timepoint(3), image(4), mask(5)
T = readcell(MANIFEST, 'Delimiter', ',', 'NumHeaderLines', 1);
N = size(T, 1);
fprintf('  %d rows found.\n\n', N);

%% ── 3. Process each image ────────────────────────────────────────────────
for i = 1:N
    subject = char(T{i,1});
    session = char(T{i,2});
    tp      = char(T{i,3});
    imgPath = char(T{i,4});

    fprintf('[%d/%d]  %s / %s / %s\n', i, N, subject, session, tp);
    fprintf('  Reading: %s\n', imgPath);

    vol  = single(niftiread(imgPath));
    info = niftiinfo(imgPath);

    fprintf('  Volume size: %s  (%.1f MB)\n', ...
            mat2str(size(vol)), numel(vol)*4/1e6);

    %% Compute Frangi vesselness
    tic;
    V = frangiVesselness3D(vol, sigmas, p.alpha, p.beta, p.C);
    elapsed = toc;
    fprintf('  done  %.1f s   range=[%.4f, %.4f]\n', elapsed, min(V(:)), max(V(:)));

    %% Build output path: replace .raw.nii.gz → .frangi.nii.gz
    outPath = regexprep(imgPath, '\.raw\.nii(\.gz)?$', '.frangi.nii$1');
    if strcmp(outPath, imgPath)
        % Fallback: image didn't match .raw pattern — append suffix
        outPath = regexprep(imgPath, '\.nii(\.gz)?$', '.frangi.nii$1');
    end

    %% Write — preserve the source NIfTI header, change datatype to single
    info.Datatype    = 'single';
    info.BitsPerPixel = 32;
    info.Filename    = '';   % niftiwrite will set this

    isGz = endsWith(outPath, '.gz');
    outBase = outPath;
    if isGz
        outBase = outPath(1:end-3);  % strip .gz for niftiwrite
    end

    niftiwrite(V, outBase, info, 'Compressed', isGz);
    fprintf('  Saved : %s\n\n', outPath);
end

fprintf('=== Done: processed %d volumes ===\n', N);

% =========================================================================
% LOCAL FUNCTIONS  (identical to tuneFrangi.m — max-over-scales 3-D Frangi)
% =========================================================================

function V = frangiVesselness3D(vol, sigmas, alpha, beta, C)
    nS  = numel(sigmas);
    BAR = 30;
    V   = zeros(size(vol), 'single');
    fprintf('  Computing vesselness  [%s] 0/%d  σ=%.2f', ...
            repmat(' ', 1, BAR), nS, sigmas(1));
    for k = 1:nS
        Vk     = frangiScaleResponse(vol, sigmas(k), alpha, beta, C);
        V      = max(V, Vk);
        filled = round(k / nS * BAR);
        next_s = sigmas(min(k+1, nS));
        fprintf('\r  Computing vesselness  [%s%s] %d/%d  σ=%.2f', ...
                repmat('#', 1, filled), repmat(' ', 1, BAR - filled), ...
                k, nS, next_s);
    end
end

% -------------------------------------------------------------------------
function V = frangiScaleResponse(vol, sigma, alpha, beta, C)
    [Lxx, Lxy, Lxz, Lyy, Lyz, Lzz] = gaussHessian3D(vol, sigma);
    [ev1, ev2, ev3] = cardanoEig3(Lxx, Lxy, Lxz, Lyy, Lyz, Lzz);

    vessel  = (ev2 < 0) & (ev3 < 0);
    abs_ev2 = abs(ev2);
    abs_ev3 = abs(ev3) + 1e-8;
    abs_ev1 = abs(ev1);

    RA = abs_ev2 ./ abs_ev3;
    RB = abs_ev1 ./ (sqrt(abs_ev2 .* abs_ev3) + 1e-8);
    S2 = ev1.^2 + ev2.^2 + ev3.^2;

    V = (1 - exp(-RA.^2 ./ (2*alpha^2))) ...
      .*  exp(-RB.^2 ./ (2*beta^2))      ...
      .* (1 - exp(-S2  ./ (2*C^2)));
    V = V .* single(vessel);
    V(~isfinite(V)) = 0;
end

% -------------------------------------------------------------------------
function [Lxx, Lxy, Lxz, Lyy, Lyz, Lzz] = gaussHessian3D(vol, sigma)
    ks  = max(5, 2*ceil(3*sigma)+1);
    r   = floor(ks/2);
    [x, y, z] = meshgrid(-r:r, -r:r, -r:r);
    G   = exp(-(x.^2 + y.^2 + z.^2) / (2*sigma^2)) / (2*pi*sigma^2)^(3/2);
    sc  = sigma^2;

    Lxx = single(sc * imfilter(double(vol), G .* (x.^2/sigma^4 - 1/sigma^2), 'replicate'));
    Lyy = single(sc * imfilter(double(vol), G .* (y.^2/sigma^4 - 1/sigma^2), 'replicate'));
    Lzz = single(sc * imfilter(double(vol), G .* (z.^2/sigma^4 - 1/sigma^2), 'replicate'));
    Lxy = single(sc * imfilter(double(vol), G .* (x.*y/sigma^4),              'replicate'));
    Lxz = single(sc * imfilter(double(vol), G .* (x.*z/sigma^4),              'replicate'));
    Lyz = single(sc * imfilter(double(vol), G .* (y.*z/sigma^4),              'replicate'));
end

% -------------------------------------------------------------------------
function [ev1, ev2, ev3] = cardanoEig3(Lxx, Lxy, Lxz, Lyy, Lyz, Lzz)
    q  = (Lxx + Lyy + Lzz) / 3;
    p1 = Lxy.^2 + Lxz.^2 + Lyz.^2;
    p2 = (Lxx - q).^2 + (Lyy - q).^2 + (Lzz - q).^2 + 2*p1;
    p  = sqrt(p2 / 6 + 1e-10);

    inv_p = 1 ./ (p + 1e-10);
    Bxx   = (Lxx - q) .* inv_p;
    Byy   = (Lyy - q) .* inv_p;
    Bzz   = (Lzz - q) .* inv_p;
    Bxy   = Lxy .* inv_p;
    Bxz   = Lxz .* inv_p;
    Byz   = Lyz .* inv_p;

    detB = Bxx .* (Byy.*Bzz - Byz.^2) ...
         - Bxy .* (Bxy.*Bzz - Byz.*Bxz) ...
         + Bxz .* (Bxy.*Byz - Byy.*Bxz);
    r    = min(max(detB / 2, -1 + 1e-7), 1 - 1e-7);
    phi  = acos(r) / 3;

    ev1 = q + 2*p .* cos(phi);
    ev3 = q + 2*p .* cos(phi + 2*pi/3);
    ev2 = 3*q - ev1 - ev3;
end

% -------------------------------------------------------------------------
function tf = endsWith(str, suffix)
    if length(str) < length(suffix), tf = false; return; end
    tf = strcmp(str(end-length(suffix)+1:end), suffix);
end
