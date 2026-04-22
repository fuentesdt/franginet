%% tuneFrangi.m
%  -----------------------------------------------------------------------
%  Optimise classical 3-D Frangi vesselness parameters to maximise Dice
%  against binary vessel masks.
%
%  Loaded from: a manifest CSV produced by tuneBezierTraining.m
%    col 1 — sample id
%    col 2 — relative path to mask  (.nii)
%    col 3 — relative path to image (.nii)
%  Paths are relative to the directory containing the CSV.
%
%  Parameters optimised (Nelder-Mead, unconstrained):
%    sigmaMin, sigmaMax  — Gaussian scale range (voxels)
%    alpha               — plate/tube discrimination (RA term)
%    beta                — blob suppression (RB term)
%    C                   — background suppression, initialised from
%                          max Hessian Frobenius norm over training set
%
%  A fixed numScales=4 log-spaced scales between sigmaMin and sigmaMax are
%  used during optimisation; the influence of numScales is swept post-hoc.
%
%  Objective during optimisation: mean soft-Dice (smooth surrogate).
%  Final reporting: hard-Dice at optimal threshold (swept over vesselness).
%
%  Outputs
%    tuneFrangi_result.mat  — optimal params, per-patch Dice, threshold
%  -----------------------------------------------------------------------

clear; clc;

%% ── Configuration ────────────────────────────────────────────────────────
CSV_FILE   = fullfile('bezier_training', 'manifest.csv');
NUM_SCALES = 4;        % fixed for optimisation; swept post-hoc
MAX_ITER   = 400;      % fminsearch MaxFunEvals / MaxIter
RESULT_MAT = 'tuneFrangi_result.mat';

%% ── 1. Load CSV and read volumes ─────────────────────────────────────────
fprintf('=== Loading training data from %s ===\n', CSV_FILE);
csvDir = fileparts(CSV_FILE);
T      = readcell(CSV_FILE, 'Delimiter', ',', 'NumHeaderLines', 0);

N   = size(T, 1);
imgs  = cell(N, 1);
masks = cell(N, 1);

for i = 1:N
    mskRel = T{i,2};
    imgRel = T{i,3};
    vol  = single(niftiread(fullfile(csvDir, imgRel)));
    msk  = niftiread(fullfile(csvDir, mskRel)) > 0;

    imgs{i}  = vol;
    masks{i} = logical(msk);
end
fprintf('  Loaded %d volumes.\n', N);

%% ── 2. Initialise C from Hessian Frobenius norm ──────────────────────────
fprintf('=== Computing Hessian magnitude statistics ===\n');
sigma_ref  = 1.5;   % representative mid-range scale for statistics
S_max_vals = zeros(N, 1);
for i = 1:N
    S_max_vals(i) = hessianFrobeniusMax(imgs{i}, sigma_ref);
    fprintf('  [%d/%d]  S_max = %.4f\n', i, N, S_max_vals(i));
end
S_max_global = max(S_max_vals);
C_init       = 0.5 * S_max_global;
fprintf('  Global S_max = %.4f  →  C_init = %.4f\n', S_max_global, C_init);

%% ── 3. Optimisation setup ────────────────────────────────────────────────
%  Unconstrained parameters in log-space:
%    x(1) = log(sigmaMin)   init: log(1.0)
%    x(2) = log(sigmaMax)   init: log(5.0)
%    x(3) = log(alpha)      init: log(0.5)
%    x(4) = log(beta)       init: log(0.5)
%    x(5) = log(C)          init: log(C_init)

x0 = [log(1.0), log(5.0), log(0.5), log(0.5), log(C_init + 1e-8)];

fprintf('\n=== Starting Nelder-Mead optimisation (%d patches, %d scales) ===\n', ...
        N, NUM_SCALES);
fprintf('  Initial params: sigmaMin=%.3f  sigmaMax=%.3f  alpha=%.3f  beta=%.3f  C=%.4f\n', ...
        exp(x0(1)), exp(x0(2)), exp(x0(3)), exp(x0(4)), exp(x0(5)));

objFun = @(x) negSoftDice(x, imgs, masks, NUM_SCALES);

opts_nm = optimset('fminsearch');
opts_nm.MaxFunEvals = MAX_ITER * numel(x0);
opts_nm.MaxIter     = MAX_ITER;
opts_nm.Display     = 'iter';
opts_nm.TolFun      = 1e-4;
opts_nm.TolX        = 1e-4;

[x_opt, fval] = fminsearch(objFun, x0, opts_nm);

sigmaMin_opt = exp(x_opt(1));
sigmaMax_opt = exp(x_opt(2));
alpha_opt    = exp(x_opt(3));
beta_opt     = exp(x_opt(4));
C_opt        = exp(x_opt(5));

fprintf('\n=== Optimisation converged  (soft-Dice = %.4f) ===\n', -fval);
fprintf('  sigmaMin = %.4f\n  sigmaMax = %.4f\n  alpha    = %.4f\n', ...
        sigmaMin_opt, sigmaMax_opt, alpha_opt);
fprintf('  beta     = %.4f\n  C        = %.4f\n', beta_opt, C_opt);

%% ── 4. numScales sweep (post-hoc) ────────────────────────────────────────
fprintf('\n=== numScales sweep (sigmas fixed to optimal) ===\n');
ns_candidates = [1 2 3 4 6 8];
softDice_ns   = zeros(size(ns_candidates));
for ki = 1:numel(ns_candidates)
    ns = ns_candidates(ki);
    sd = -negSoftDice(x_opt, imgs, masks, ns);
    softDice_ns(ki) = sd;
    fprintf('  numScales=%d  soft-Dice=%.4f\n', ns, sd);
end
[~, best_ns_idx] = max(softDice_ns);
numScales_opt = ns_candidates(best_ns_idx);
fprintf('  Best numScales = %d\n', numScales_opt);

%% ── 5. Compute vesselness for x0 and x_opt; sweep thresholds for both ────
fprintf('\n=== Threshold sweep: x0 (initial) vs x_opt (optimised) ===\n');
thresholds = 0.01 : 0.01 : 0.99;

[allV_x0,  thr_x0,  bestDice_x0,  dice_hard_x0]  = evalParams(x0,    imgs, masks, NUM_SCALES, thresholds);
[allV_opt, thr_opt, bestDice_opt, dice_hard_opt] = evalParams(x_opt, imgs, masks, numScales_opt, thresholds);

fprintf('  x0   threshold=%.2f  mean hard-Dice=%.4f\n', thr_x0,  bestDice_x0);
fprintf('  x_opt threshold=%.2f  mean hard-Dice=%.4f\n', thr_opt, bestDice_opt);

%% ── 6. Per-patch comparison table ───────────────────────────────────────
SEP = repmat('-', 1, 72);
fprintf('\n%s\n', SEP);
fprintf('  %-6s  %-12s  %-12s  %-12s  %-12s\n', ...
        'Patch', 'SoftDice_x0', 'SoftDice_opt', 'HardDice_x0', 'HardDice_opt');
fprintf('%s\n', SEP);
for i = 1:N
    sd0  = softDice1(allV_x0{i},  single(masks{i}));
    sdOp = softDice1(allV_opt{i}, single(masks{i}));
    fprintf('  %-6d  %-12.4f  %-12.4f  %-12.4f  %-12.4f\n', ...
            i, sd0, sdOp, dice_hard_x0(i), dice_hard_opt(i));
end
fprintf('%s\n', SEP);
fprintf('  %-6s  %-12.4f  %-12.4f  %-12.4f  %-12.4f\n', 'MEAN', ...
        -negSoftDice(x0, imgs, masks, NUM_SCALES), -fval, ...
        mean(dice_hard_x0), mean(dice_hard_opt));
fprintf('%s\n', SEP);
fprintf('  Improvement: soft-Dice %+.4f   hard-Dice %+.4f\n', ...
        (-fval) - (-negSoftDice(x0, imgs, masks, NUM_SCALES)), ...
        mean(dice_hard_opt) - mean(dice_hard_x0));

%% ── 7. Save results ──────────────────────────────────────────────────────
result = struct( ...
    'sigmaMin',       sigmaMin_opt, ...
    'sigmaMax',       sigmaMax_opt, ...
    'numScales',      numScales_opt, ...
    'alpha',          alpha_opt, ...
    'beta',           beta_opt, ...
    'C',              C_opt, ...
    'threshold',      thr_opt, ...
    'meanSoftDice',  -fval, ...
    'meanHardDice',   mean(dice_hard_opt), ...
    'dicePerPatch',   dice_hard_opt, ...
    'x0_meanHardDice', mean(dice_hard_x0), ...
    'x0_dicePerPatch', dice_hard_x0);
save(RESULT_MAT, 'result');
fprintf('\nResults saved to %s\n', RESULT_MAT);

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function val = negSoftDice(x, imgs, masks, numScales)
% Negative mean soft-Dice over all patches for given log-space params.
    sigmaMin = exp(x(1));
    sigmaMax = max(exp(x(2)), sigmaMin * 1.01);
    alpha    = exp(x(3));
    beta     = exp(x(4));
    C        = exp(x(5));
    sigmas   = exp(linspace(log(sigmaMin), log(sigmaMax), numScales));

    N = numel(imgs);
    d = zeros(N, 1);
    for i = 1:N
        V    = frangiVesselness3D(imgs{i}, sigmas, alpha, beta, C);
        d(i) = softDice1(V, single(masks{i}));
    end
    val = -mean(d);
end

% -------------------------------------------------------------------------
function [allV, thr_best, dice_best, dice_hard] = evalParams(x, imgs, masks, numScales, thresholds)
% Compute vesselness for all patches, find best hard-Dice threshold.
    sigmaMin = exp(x(1));
    sigmaMax = max(exp(x(2)), sigmaMin * 1.01);
    alpha    = exp(x(3));
    beta     = exp(x(4));
    C        = exp(x(5));
    sigmas   = exp(linspace(log(sigmaMin), log(sigmaMax), numScales));

    N    = numel(imgs);
    allV = cell(N, 1);
    for i = 1:N
        allV{i} = frangiVesselness3D(imgs{i}, sigmas, alpha, beta, C);
    end

    meanDice_thr = zeros(size(thresholds));
    for ti = 1:numel(thresholds)
        d = zeros(N, 1);
        for i = 1:N
            d(i) = diceCoeff(allV{i} >= thresholds(ti), masks{i});
        end
        meanDice_thr(ti) = mean(d);
    end

    [dice_best, best_ti] = max(meanDice_thr);
    thr_best  = thresholds(best_ti);
    dice_hard = zeros(N, 1);
    for i = 1:N
        dice_hard(i) = diceCoeff(allV{i} >= thr_best, masks{i});
    end
end

% -------------------------------------------------------------------------
function V = frangiVesselness3D(vol, sigmas, alpha, beta, C)
% Max-over-scales classical 3-D Frangi vesselness (no dlarray).
    V = zeros(size(vol), 'single');
    for k = 1:numel(sigmas)
        Vk = frangiScaleResponse(vol, sigmas(k), alpha, beta, C);
        V  = max(V, Vk);
    end
end

% -------------------------------------------------------------------------
function V = frangiScaleResponse(vol, sigma, alpha, beta, C)
    [Lxx, Lxy, Lxz, Lyy, Lyz, Lzz] = gaussHessian3D(vol, sigma);
    [ev1, ev2, ev3] = cardanoEig3(Lxx, Lxy, Lxz, Lyy, Lyz, Lzz);

    % Vessel mask: ev2 < 0 AND ev3 < 0 (bright tubular, value-sorted desc)
    vessel = (ev2 < 0) & (ev3 < 0);

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
% Analytical eigenvalues of 3×3 symmetric Hessian (Cardano / Schur method).
% Eigenvalues returned value-sorted descending: ev1 >= ev2 >= ev3.
% Matches the convention in learnableFrangiLayer.m.
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
function d = softDice1(V, G)
% Soft Dice between vesselness map V in [0,1] and binary mask G in {0,1}.
    eps = 1e-5;
    Vf  = V(:);  Gf = G(:);
    d   = (2*sum(Vf .* Gf) + eps) / (sum(Vf) + sum(Gf) + eps);
end

% -------------------------------------------------------------------------
function d = diceCoeff(pred, gt)
    pred = logical(pred(:));  gt = logical(gt(:));
    d    = 2*sum(pred & gt) / (sum(pred) + sum(gt) + 1e-8);
end
