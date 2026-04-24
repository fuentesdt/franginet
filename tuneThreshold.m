%% tuneThreshold.m
%  -----------------------------------------------------------------------
%  Optimise a two-sided intensity threshold (lo_HU, hi_HU) on the raw CT
%  image to maximise hard-Dice agreement with binary vessel masks.
%
%  Loaded from: the same manifest CSV used by tuneFrangi.m
%    col 1 — sample id
%    col 2 — relative path to mask  (.nii)
%    col 3 — relative path to image (.nii)
%    col 4 — (optional) ROI mask path
%  Paths are relative to the directory containing the CSV.
%
%  Binary prediction at threshold pair (lo, hi):
%    pred = img >= lo_HU  &  img <= hi_HU
%    (restricted to ROI voxels when col 4 is provided)
%
%  Method:
%    1. Collect vessel-positive voxel HU values to set search bounds.
%    2. Build per-sample cumulative histograms for O(1) Dice evaluation.
%    3. Coarse N_GRID × N_GRID grid search (upper-triangle: hi > lo only).
%    4. Fine N_FINE × N_FINE grid search centred on the coarse optimum.
%    5. Per-sample comparison: initial (percentile-based) vs optimised.
%    6. Sensitivity sweep: Dice vs lo (hi fixed) and vs hi (lo fixed).
%
%  Outputs
%    tuneThreshold_result.mat — lo_HU, hi_HU, meanHardDice, dicePerSample
%  -----------------------------------------------------------------------

clear; clc;

%% ── Configuration ────────────────────────────────────────────────────────
CSV_FILE   = fullfile('preprocessed', 'manifest.csv');
N_GRID     = 60;      % coarse grid: N_GRID × N_GRID pairs, upper-triangle only
N_FINE     = 50;      % fine grid: N_FINE × N_FINE centred on coarse optimum
N_BINS     = 2000;    % histogram bins (determines HU resolution of grid search)
N_SENS     = 60;      % points in sensitivity sweep
RESULT_MAT = 'tuneThreshold_result.mat';

%% ── 1. Load CSV and read volumes ─────────────────────────────────────────
fprintf('=== Loading training data from %s ===\n', CSV_FILE);
csvDir = fileparts(CSV_FILE);
T      = readcell(CSV_FILE, 'Delimiter', ',', 'NumHeaderLines', 0);

N      = size(T, 1);
hasROI = size(T, 2) >= 4;
imgs   = cell(N, 1);
masks  = cell(N, 1);
rois   = cell(N, 1);

for i = 1:N
    mskRel   = T{i,2};
    imgRel   = T{i,3};
    imgs{i}  = single(niftiread(fullfile(csvDir, imgRel)));
    masks{i} = logical(niftiread(fullfile(csvDir, mskRel)) > 0);
    if hasROI
        rois{i} = logical(niftiread(fullfile(csvDir, T{i,4})) > 0);
    end
end

if hasROI
    fprintf('  Loaded %d volumes with ROI masks.\n', N);
else
    fprintf('  Loaded %d volumes (no ROI column — whole-volume Dice).\n', N);
end

%% ── 2. Estimate search bounds from vessel-positive voxel HU distribution ─
fprintf('\n=== Estimating HU range from vessel voxels ===\n');

pos_vals = [];
for i = 1:N
    if ~isempty(rois{i})
        px = double(imgs{i}(masks{i} & rois{i}));
    else
        px = double(imgs{i}(masks{i}));
    end
    pos_vals = [pos_vals; px(:)];   %#ok<AGROW>
end

pct = prctile(pos_vals, [1 10 50 90 99]);
fprintf('  Vessel HU percentiles  P1=%.0f  P10=%.0f  P50=%.0f  P90=%.0f  P99=%.0f\n', ...
        pct(1), pct(2), pct(3), pct(4), pct(5));

% Search range: 20 %% margin beyond P1 and P99
margin = max(50, (pct(5) - pct(1)) * 0.20);
HU_MIN = pct(1) - margin;
HU_MAX = pct(5) + margin;

% Initial thresholds: P10 / P90 of vessel voxels (percentile-based baseline)
lo0 = pct(2);
hi0 = pct(4);
fprintf('  Search range : [%.0f, %.0f] HU\n', HU_MIN, HU_MAX);
fprintf('  Initial pair : lo=%.0f  hi=%.0f HU\n', lo0, hi0);

%% ── 3. Build cumulative histograms ────────────────────────────────────────
fprintf('\n=== Building cumulative histograms (N_BINS=%d) ===\n', N_BINS);

edges = linspace(HU_MIN, HU_MAX, N_BINS + 1);
bw    = edges(2) - edges(1);

% cum_pos(i,k): # vessel voxels in sample i with HU in bins 1..k-1
%               i.e. HU < edges(k)
% cum_all(i,k): same for all ROI voxels (vessel + background)
cum_pos   = zeros(N, N_BINS + 1);
cum_all   = zeros(N, N_BINS + 1);
total_pos = zeros(N, 1);

for i = 1:N
    if ~isempty(rois{i})
        roi_mask = rois{i};
    else
        roi_mask = true(size(imgs{i}));
    end
    msk_roi = masks{i} & roi_mask;

    h_pos = histcounts(double(imgs{i}(msk_roi)),      edges);
    h_all = histcounts(double(imgs{i}(roi_mask(:))),  edges);

    cum_pos(i,:)  = [0, cumsum(h_pos)];
    cum_all(i,:)  = [0, cumsum(h_all)];
    total_pos(i)  = sum(msk_roi(:));
    fprintf('  [%d/%d]  vessel=%d  ROI=%d  voxels\n', ...
            i, N, total_pos(i), sum(roi_mask(:)));
end

%% ── 4. Coarse grid search ─────────────────────────────────────────────────
fprintf('\n=== Coarse grid search (%d × %d) ===\n', N_GRID, N_GRID);

lv_c     = linspace(HU_MIN, HU_MAX, N_GRID);
hv_c     = linspace(HU_MIN, HU_MAX, N_GRID);
dice_c   = zeros(N_GRID, N_GRID);
n_eval   = 0;

for li = 1:N_GRID
    for hi_idx = li+1 : N_GRID        % upper-triangle: hi > lo
        dice_c(li, hi_idx) = histDice(lv_c(li), hv_c(hi_idx), ...
                                       cum_pos, cum_all, total_pos, ...
                                       edges, bw, N);
        n_eval = n_eval + 1;
    end
end
fprintf('  Evaluated %d (lo, hi) pairs.\n', n_eval);

[best_c, best_ci]   = max(dice_c(:));
[lo_ci, hi_ci]      = ind2sub([N_GRID N_GRID], best_ci);
fprintf('  Best coarse : lo=%.1f  hi=%.1f  mean-Dice=%.4f\n', ...
        lv_c(lo_ci), hv_c(hi_ci), best_c);

%% ── 5. Fine grid search centred on coarse optimum ────────────────────────
fprintf('\n=== Fine grid search (%d × %d) ===\n', N_FINE, N_FINE);

cs       = lv_c(2) - lv_c(1);    % coarse step size
lv_f     = linspace(lv_c(lo_ci) - 2*cs, lv_c(lo_ci) + 2*cs, N_FINE);
hv_f     = linspace(hv_c(hi_ci) - 2*cs, hv_c(hi_ci) + 2*cs, N_FINE);
lv_f     = max(lv_f, HU_MIN);
hv_f     = min(hv_f, HU_MAX);
dice_f   = zeros(N_FINE, N_FINE);

for li = 1:N_FINE
    for hi_idx = 1:N_FINE
        if hv_f(hi_idx) <= lv_f(li), continue; end
        dice_f(li, hi_idx) = histDice(lv_f(li), hv_f(hi_idx), ...
                                       cum_pos, cum_all, total_pos, ...
                                       edges, bw, N);
    end
end

[best_f, best_fi] = max(dice_f(:));
[lo_fi, hi_fi]    = ind2sub([N_FINE N_FINE], best_fi);
lo_opt            = lv_f(lo_fi);
hi_opt            = hv_f(hi_fi);
fprintf('  Best fine   : lo=%.1f  hi=%.1f  mean-Dice=%.4f\n', ...
        lo_opt, hi_opt, best_f);

%% ── 6. Per-sample comparison table ──────────────────────────────────────
fprintf('\n=== Per-sample Dice comparison ===\n');

dice_init = zeros(N, 1);
dice_opt  = zeros(N, 1);
for i = 1:N
    if ~isempty(rois{i})
        roi_mask = rois{i};
    else
        roi_mask = true(size(imgs{i}));
    end
    dice_init(i) = diceCoeff(imgs{i} >= lo0    & imgs{i} <= hi0,    masks{i}, roi_mask);
    dice_opt(i)  = diceCoeff(imgs{i} >= lo_opt & imgs{i} <= hi_opt, masks{i}, roi_mask);
end

SEP = repmat('-', 1, 60);
fprintf('%s\n', SEP);
fprintf('  %-8s  %-14s  %-14s\n', 'Sample', 'Dice_initial', 'Dice_optimised');
fprintf('%s\n', SEP);
for i = 1:N
    fprintf('  %-8s  %-14.4f  %-14.4f\n', char(T{i,1}), dice_init(i), dice_opt(i));
end
fprintf('%s\n', SEP);
fprintf('  %-8s  %-14.4f  %-14.4f\n', 'MEAN', mean(dice_init), mean(dice_opt));
fprintf('%s\n', SEP);
fprintf('  Improvement : %+.4f\n', mean(dice_opt) - mean(dice_init));
fprintf('  Initial     : lo=%.1f  hi=%.1f HU\n', lo0, hi0);
fprintf('  Optimised   : lo=%.1f  hi=%.1f HU\n', lo_opt, hi_opt);
if hasROI
    fprintf('  (Dice computed within ROI mask)\n');
end

%% ── 7. Sensitivity sweep ─────────────────────────────────────────────────
fprintf('\n=== Sensitivity sweep ===\n');

lo_sweep = linspace(lv_f(1), lv_f(end), N_SENS);
hi_sweep = linspace(hv_f(1), hv_f(end), N_SENS);

dice_lo_sweep = zeros(N_SENS, 1);
dice_hi_sweep = zeros(N_SENS, 1);
for k = 1:N_SENS
    if lo_sweep(k) < hi_opt
        dice_lo_sweep(k) = histDice(lo_sweep(k), hi_opt, ...
                                     cum_pos, cum_all, total_pos, edges, bw, N);
    end
    if hv_f(end) > lo_opt && hi_sweep(k) > lo_opt
        dice_hi_sweep(k) = histDice(lo_opt, hi_sweep(k), ...
                                     cum_pos, cum_all, total_pos, edges, bw, N);
    end
end

fprintf('  lo sweep (hi=%.0f fixed):\n', hi_opt);
fprintf('    %-8s  %s\n', 'lo (HU)', 'mean-Dice');
for k = 1:N_SENS
    marker = '';
    if abs(lo_sweep(k) - lo_opt) < (lo_sweep(2)-lo_sweep(1))/2, marker = ' ◄ opt'; end
    fprintf('    %-8.1f  %.4f%s\n', lo_sweep(k), dice_lo_sweep(k), marker);
end

fprintf('\n  hi sweep (lo=%.0f fixed):\n', lo_opt);
fprintf('    %-8s  %s\n', 'hi (HU)', 'mean-Dice');
for k = 1:N_SENS
    marker = '';
    if abs(hi_sweep(k) - hi_opt) < (hi_sweep(2)-hi_sweep(1))/2, marker = ' ◄ opt'; end
    fprintf('    %-8.1f  %.4f%s\n', hi_sweep(k), dice_hi_sweep(k), marker);
end

%% ── 8. Save results ──────────────────────────────────────────────────────
result = struct( ...
    'lo_HU',             lo_opt,          ...
    'hi_HU',             hi_opt,          ...
    'meanHardDice',      mean(dice_opt),  ...
    'dicePerSample',     dice_opt,        ...
    'lo0_HU',            lo0,             ...
    'hi0_HU',            hi0,             ...
    'x0_meanHardDice',   mean(dice_init), ...
    'x0_dicePerSample',  dice_init);
save(RESULT_MAT, 'result');
fprintf('\nResults saved to %s\n', RESULT_MAT);

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function d = histDice(lo, hi, cum_pos, cum_all, total_pos, edges, bw, N)
% Fast mean Dice over N samples using pre-built cumulative histograms.
%
% cum_pos(i,k): # vessel voxels in sample i with HU < edges(k)
% cum_all(i,k): # ROI    voxels in sample i with HU < edges(k)
%
% For the band [lo, hi]:
%   TP  = vessel voxels in band  = cum_pos(i,b+1) - cum_pos(i,a)
%   |pred| = all ROI voxels in band = cum_all(i,b+1) - cum_all(i,a)
%   Dice = 2*TP / (|pred| + total_pos(i))
    a = max(1, min(size(cum_pos,2)-1, floor((lo - edges(1)) / bw) + 1));
    b = max(1, min(size(cum_pos,2)-1, floor((hi - edges(1)) / bw) + 1));
    if b < a
        d = 0; return;
    end
    d_vec = zeros(N, 1);
    for i = 1:N
        tp  = cum_pos(i, b+1) - cum_pos(i, a);
        np  = cum_all(i, b+1) - cum_all(i, a);     % |pred|
        d_vec(i) = 2*tp / (np + total_pos(i) + 1e-8);
    end
    d = mean(d_vec);
end

% -------------------------------------------------------------------------
function d = diceCoeff(pred, gt, roi)
    if nargin >= 3 && ~isempty(roi)
        r    = logical(roi(:));
        pred = logical(pred(r));  gt = logical(gt(r));
    else
        pred = logical(pred(:));  gt = logical(gt(:));
    end
    d = 2*sum(pred & gt) / (sum(pred) + sum(gt) + 1e-8);
end
