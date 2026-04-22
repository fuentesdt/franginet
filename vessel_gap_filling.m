%% vessel_gap_filling.m
%
% Gradient-guided gap filling for vessel centreline discontinuities.
%
% Uses the spatial gradients of Frangi filter factors A (plate suppressor)
% and B (blob suppressor) as a geometric flow field to bridge breaks in
% a thresholded vesselness mask.
%
% INPUTS:
%   image.nii.gz      - 3D intensity volume (any modality; normalised to [0,1])
%   Frangi parameters are loaded automatically from tuneFrangi_result.mat
%                       (produced by tuneFrangi.m)
%
% OUTPUTS:
%   filled_mask.nii.gz     - binary vessel mask with gaps bridged
%   gap_report.mat         - struct array describing each detected gap
%   gap_markers.fcsv       - 3D Slicer fiducial markup for review cases
%
% DEPENDENCIES:
%   Tools for NIfTI and ANALYZE (Jimmy Shen) - niftiread/niftiwrite
%   or Image Processing Toolbox (R2017b+) for niftiread/niftiwrite
%   bwskel (Image Processing Toolbox R2019a+)
%   msfm3d (fast marching) - from Dirk-Jan Kroon's FEX submission
%       https://www.mathworks.com/matlabcentral/fileexchange/24531
%
% USAGE:
%   vessel_gap_filling('image.nii.gz')
%   vessel_gap_filling('image.nii.gz', 'result_mat', 'my_tuneFrangi_result.mat')
%   vessel_gap_filling('image.nii.gz', 'max_gap_mm', 12.0)
%
% PARAMETERS (name-value pairs, all optional):
%   result_mat     path to tuneFrangi_result.mat                default 'tuneFrangi_result.mat'
%                  Supplies: sigmaMin, sigmaMax, numScales, alpha, beta, C, threshold
%   normalize      normalise image intensity to [0,1] before   default true
%                  computing vesselness (disable for pre-normalised inputs)
%   label_file     path to GT binary mask NIfTI for DSC eval   default '' (skip)
%                  reports DSC before and after gap filling
%   t_vessel       vesselness threshold for mask (overrides result.threshold) default from result
%   t_resume       vesselness threshold to declare resume     default 0.8 * t_vessel
%   t_break        vesselness level considered background     default 0.3 * t_vessel
%   max_gap_mm     maximum gap distance to attempt bridging   default 8.0
%   step_mm        flow integration step size (mm)            default 0.5
%   alpha_blend    blend weight: F_flow vs v1 axis direction  default 0.60
%   w_A            weight of grad_A in flow field             default 1.0
%   w_B            weight of grad_B in flow field             default 1.0
%   w_v            weight of vesselness in cost volume        default 0.40
%   w_geo          weight of A*B geometry in cost volume      default 0.35
%   w_hu           weight of HU plausibility in cost volume   default 0.25
%   hu_vessel      expected HU of contrast-enhanced artery    default 250
%   conf_auto      confidence threshold for auto-fill         default 0.65
%   conf_review    confidence threshold to flag for review    default 0.35
%   output_dir     directory for output files                 default './'

function vessel_gap_filling(image_file, varargin)

%% -----------------------------------------------------------------------
% 0. Parse inputs
% -----------------------------------------------------------------------
p = inputParser;
addRequired(p,  'image_file', @ischar);
addParameter(p, 'result_mat',   'tuneFrangi_result.mat');
addParameter(p, 't_vessel',     []);    % filled from result below
addParameter(p, 't_resume',     []);
addParameter(p, 't_break',      []);
addParameter(p, 'max_gap_mm',   8.0);
addParameter(p, 'step_mm',      0.5);
addParameter(p, 'alpha_blend',  0.60);
addParameter(p, 'w_A',          1.0);
addParameter(p, 'w_B',          1.0);
addParameter(p, 'w_v',          0.40);
addParameter(p, 'w_geo',        0.35);
addParameter(p, 'w_hu',         0.25);
addParameter(p, 'hu_vessel',    250);
addParameter(p, 'conf_auto',    0.65);
addParameter(p, 'conf_review',  0.35);
addParameter(p, 'normalize',    true);
addParameter(p, 'label_file',   '');    % optional GT mask for DSC evaluation
addParameter(p, 'output_dir',   './');
parse(p, image_file, varargin{:});
opt = p.Results;

%% -----------------------------------------------------------------------
% 0b. Load Frangi parameters (fall back to Frangi 1998 defaults if absent)
% -----------------------------------------------------------------------
FRANGI_DEFAULTS = struct('sigmaMin', 1.0, 'sigmaMax', 4.0, 'numScales', 4, ...
                         'alpha', 0.5, 'beta', 0.5, 'C', 0.25, 'threshold', 0.15);

if exist(opt.result_mat, 'file')
    fprintf('=== Loading Frangi parameters from %s ===\n', opt.result_mat);
    r  = load(opt.result_mat, 'result');
    fr = r.result;
else
    warning('vessel_gap_filling:noResultMat', ...
        'Result file ''%s'' not found — using Frangi 1998 defaults.', opt.result_mat);
    fr = FRANGI_DEFAULTS;
end

opt.alpha     = fr.alpha;
opt.beta      = fr.beta;
opt.C         = fr.C;
opt.sigmas_mm = exp(linspace(log(fr.sigmaMin), log(fr.sigmaMax), fr.numScales));

if isempty(opt.t_vessel), opt.t_vessel = fr.threshold;        end
if isempty(opt.t_resume), opt.t_resume = 0.80 * opt.t_vessel; end
if isempty(opt.t_break),  opt.t_break  = 0.30 * opt.t_vessel; end

fprintf('   sigmas_mm  : %s\n', num2str(opt.sigmas_mm, '%.2f  '));
fprintf('   alpha=%.4f  beta=%.4f  C=%.4f\n', opt.alpha, opt.beta, opt.C);
fprintf('   t_vessel=%.4f  t_resume=%.4f  t_break=%.4f\n', ...
        opt.t_vessel, opt.t_resume, opt.t_break);

fprintf('=== Vessel gap filling: gradient-guided fast marching ===\n');
if ~exist(opt.output_dir, 'dir'), mkdir(opt.output_dir); end

%% -----------------------------------------------------------------------
% 1. Load volumes
% -----------------------------------------------------------------------
fprintf('[1/9] Loading image volume...\n');

info_img = niftiinfo(image_file);
I        = double(niftiread(image_file));

% Per-volume normalisation to [0,1] (matches tuneFrangi training convention)
if opt.normalize
    lo = min(I(:));  hi = max(I(:));
    if hi > lo, I = (I - lo) / (hi - lo); end
    fprintf('   Normalised intensity to [0, 1].\n');
else
    fprintf('   Normalisation skipped (normalize=false).\n');
end

% Voxel size in mm [dx, dy, dz]
vox = abs(diag(info_img.Transform.T(1:3,1:3)))';
sz  = size(I);
fprintf('   Volume size : %d x %d x %d voxels\n', sz(1), sz(2), sz(3));
fprintf('   Voxel size  : %.3f x %.3f x %.3f mm\n', vox(1), vox(2), vox(3));

% Optional ground-truth label mask
GT = [];
if ~isempty(opt.label_file)
    GT = niftiread(opt.label_file) > 0;
    assert(isequal(size(GT), sz), ...
        'Label mask size %s does not match image size %s.', ...
        mat2str(size(GT)), mat2str(sz));
    fprintf('   GT label loaded: %d vessel voxels (%.1f%%)\n', ...
            sum(GT(:)), 100*mean(GT(:)));
end

%% -----------------------------------------------------------------------
% 2. Compute Hessian eigenvalues and eigenvectors at dominant scale
% -----------------------------------------------------------------------
fprintf('[2/9] Computing multi-scale Hessian (scales: %s mm)...\n', ...
    num2str(opt.sigmas_mm, '%.2f  '));

% Allocate arrays for scale-aggregated outputs
lam1 = zeros(sz); lam2 = zeros(sz); lam3 = zeros(sz);
v1        = zeros([sz 3]);
A_map     = zeros(sz);
B_map     = zeros(sz);
V_scale_max = zeros(sz);

for s = opt.sigmas_mm
    sig_vox = s ./ vox;   % sigma in voxels per axis

    [L1s, L2s, L3s, ev1s] = hessian_eigvals(I, sig_vox, s);
    [L1s, L2s, L3s, ev1s] = sort_eigenvalues(L1s, L2s, L3s, ev1s);

    RA = abs(L2s) ./ (abs(L3s) + 1e-9);
    RB = abs(L1s) ./ (sqrt(abs(L2s) .* abs(L3s)) + 1e-9);
    S2 = L1s.^2 + L2s.^2 + L3s.^2;

    As = 1 - exp(-RA.^2 / (2 * opt.alpha^2));
    Bs =     exp(-RB.^2 / (2 * opt.beta^2));
    Cs = 1 - exp(-S2    / (2 * opt.C^2));     % C loaded from tuneFrangi result

    Vs = As .* Bs .* Cs;
    Vs(L2s > 0 | L3s > 0) = 0;

    update = Vs > V_scale_max;
    V_scale_max(update) = Vs(update);
    lam1(update) = L1s(update);
    lam2(update) = L2s(update);
    lam3(update) = L3s(update);
    A_map(update) = As(update);
    B_map(update) = Bs(update);
    for d = 1:3
        tmp = v1(:,:,:,d);
        e_d = ev1s(:,:,:,d);
        tmp(update) = e_d(update);
        v1(:,:,:,d) = tmp;
    end
end
clear L1s L2s L3s Vs As Bs Cs RA RB S2 update ev1s;

% Vesselness map derived entirely from loaded Frangi parameters
V = V_scale_max;
fprintf('   Hessian complete.  vesselness range [%.4f, %.4f]\n', min(V(:)), max(V(:)));

%% -----------------------------------------------------------------------
% 3. Compute spatial gradients of A and B
% -----------------------------------------------------------------------
fprintf('[3/9] Computing grad_A and grad_B...\n');

% gradient() returns dA/d(index); convert to mm using voxel spacing
[dA_x, dA_y, dA_z] = gradient(A_map, vox(1), vox(2), vox(3));
[dB_x, dB_y, dB_z] = gradient(B_map, vox(1), vox(2), vox(3));

% Geometric flow field: F = w_A * grad_A - w_B * grad_B
% Points toward high eccentricity and low blobness simultaneously
Fx = opt.w_A * dA_x - opt.w_B * dB_x;
Fy = opt.w_A * dA_y - opt.w_B * dB_y;
Fz = opt.w_A * dA_z - opt.w_B * dB_z;

% Normalise to unit vectors
F_norm = sqrt(Fx.^2 + Fy.^2 + Fz.^2) + 1e-9;
Fx = Fx ./ F_norm;
Fy = Fy ./ F_norm;
Fz = Fz ./ F_norm;

clear dA_x dA_y dA_z dB_x dB_y dB_z F_norm;

%% -----------------------------------------------------------------------
% 4. Build combined cost volume for fast marching
% -----------------------------------------------------------------------
fprintf('[4/9] Building cost volume...\n');

% HU plausibility: sigmoid centred on expected artery HU
hu_norm = 1 ./ (1 + exp(-(I - opt.hu_vessel) / 50));

% Geometry-only vesselness (no C gate — works in low-contrast regions)
V_geo = A_map .* B_map;

% Combined cost: low cost where vessel is geometrically plausible
V_combined = (opt.w_v   * V       ...
            + opt.w_geo * V_geo   ...
            + opt.w_hu  * hu_norm) ...
           / (opt.w_v + opt.w_geo + opt.w_hu);
V_combined = max(min(V_combined, 1), 0);

cost_vol = max(1 - V_combined, 1e-3);  % invert: low cost = likely vessel

%% -----------------------------------------------------------------------
% 5. Threshold vesselness map and skeletonise
% -----------------------------------------------------------------------
fprintf('[5/9] Thresholding and skeletonising...\n');

binary_vesselness = V > opt.t_vessel;   % kept for output and pre-fill DSC
binary_mask = bwareaopen(binary_vesselness, 50);  % remove small islands

% Pre-fill DSC against GT (if supplied)
dsc_pre = NaN;
if ~isempty(GT)
    dsc_pre = dice_coeff(binary_mask, GT);
    fprintf('   DSC (binarized vesselness vs GT) : %.4f\n', dsc_pre);
end

% 3D thinning to 1-voxel-wide skeleton
skeleton = bwskel(binary_mask, 'MinBranchLength', 3);

% Label connected skeleton voxels for graph analysis
skel_cc = bwconncomp(skeleton, 26);
fprintf('   Skeleton connected components: %d\n', skel_cc.NumObjects);

%% -----------------------------------------------------------------------
% 6. Detect break tips (degree-1 skeleton endpoints)
% -----------------------------------------------------------------------
fprintf('[6/9] Detecting break tips...\n');

% Degree of each skeleton voxel = number of 26-connected skeleton neighbours
break_tips = detect_break_tips(skeleton, V, opt.t_vessel, opt.t_break, sz);
fprintf('   Break tip candidates: %d\n', length(break_tips));

%% -----------------------------------------------------------------------
% 7. Pair tips into gap candidates
% -----------------------------------------------------------------------
fprintf('[7/9] Pairing tips into gap candidates...\n');

max_gap_vox = opt.max_gap_mm / mean(vox);

% Collect tip coordinates and axis directions
n_tips = length(break_tips);
tip_coords = zeros(n_tips, 3);
tip_v1     = zeros(n_tips, 3);
for k = 1:n_tips
    tip_coords(k,:) = break_tips(k).coord;
    c = num2cell(break_tips(k).coord);
    tip_v1(k,:) = squeeze(v1(c{1}, c{2}, c{3}, :))';
end

gap_candidates = struct([]);
n_gaps = 0;

for i = 1:n_tips
    for j = i+1:n_tips
        dist = norm(tip_coords(i,:) - tip_coords(j,:));
        if dist > max_gap_vox, continue; end

        % Antiparallel test: tips should face each other
        dot_ij = dot(tip_v1(i,:), tip_v1(j,:));
        if dot_ij > -0.4, continue; end  % not antiparallel

        n_gaps = n_gaps + 1;
        gap_candidates(n_gaps).tip_a     = break_tips(i);
        gap_candidates(n_gaps).tip_b     = break_tips(j);
        gap_candidates(n_gaps).dist_mm   = dist * mean(vox);
        gap_candidates(n_gaps).bridge    = [];
        gap_candidates(n_gaps).method    = '';
        gap_candidates(n_gaps).confidence = 0;
        gap_candidates(n_gaps).action    = '';
    end
end
fprintf('   Gap candidates: %d\n', n_gaps);

%% -----------------------------------------------------------------------
% 8. Trace gradient-guided bridge for each gap candidate
% -----------------------------------------------------------------------
fprintf('[8/9] Bridging gaps...\n');

n_auto = 0; n_review = 0; n_reject = 0;

for g = 1:n_gaps
    ta = gap_candidates(g).tip_a.coord;
    tb = gap_candidates(g).tip_b.coord;

    % --- Flow-guided trace from tip A forward ---
    [path_a, found_a] = trace_flow(ta, Fx, Fy, Fz, v1, V, ...
        opt.t_resume, opt.step_mm, vox, opt.alpha_blend, ...
        opt.max_gap_mm, sz, +1);

    % --- Flow-guided trace from tip B (reverse flow direction) ---
    [path_b, found_b] = trace_flow(tb, Fx, Fy, Fz, v1, V, ...
        opt.t_resume, opt.step_mm, vox, opt.alpha_blend, ...
        opt.max_gap_mm, sz, -1);

    % Check convergence: do the two path endpoints meet?
    converged = false;
    if ~isempty(path_a) && ~isempty(path_b)
        d_ends = norm(path_a(end,:) - path_b(end,:));
        converged = d_ends < 2.0;
    end

    if converged
        bridge = [path_a; flipud(path_b)];
        method = 'flow_converged';
    else
        % Fallback: fast marching cost-weighted shortest path
        bridge = fast_march_bridge(cost_vol, ta, tb, sz, vox);
        method = 'fast_march_fallback';
    end

    % Score the proposed bridge
    conf = score_bridge(bridge, A_map, B_map, V, I, ...
                        opt.hu_vessel, vox, sz);

    gap_candidates(g).bridge     = bridge;
    gap_candidates(g).method     = method;
    gap_candidates(g).confidence = conf;

    % Decide action based on confidence
    if conf >= opt.conf_auto
        gap_candidates(g).action = 'auto_filled';
        % Paint bridge into binary mask
        for k = 1:size(bridge,1)
            idx = max(1, min(sz, round(bridge(k,:))));
            binary_mask(idx(1), idx(2), idx(3)) = true;
        end
        n_auto = n_auto + 1;

    elseif conf >= opt.conf_review
        gap_candidates(g).action = 'flagged_for_review';
        n_review = n_review + 1;

    else
        gap_candidates(g).action = 'rejected';
        n_reject = n_reject + 1;
    end
end

fprintf('   Auto-filled: %d  |  Flagged: %d  |  Rejected: %d\n', ...
    n_auto, n_review, n_reject);

%% -----------------------------------------------------------------------
% 9. Save outputs
% -----------------------------------------------------------------------
fprintf('[9/9] Saving outputs...\n');

nii_hdr           = info_img;
nii_hdr.Filename  = '';   % suppress stale path warnings

% Post-fill DSC against GT (if supplied)
dsc_post = NaN;
if ~isempty(GT)
    dsc_post = dice_coeff(binary_mask, GT);
    fprintf('\n── DSC evaluation ──────────────────────────────\n');
    fprintf('   Before gap filling (binarized vesselness) : %.4f\n', dsc_pre);
    fprintf('   After  gap filling                        : %.4f\n', dsc_post);
    fprintf('   Improvement                               : %+.4f\n', dsc_post - dsc_pre);
    fprintf('────────────────────────────────────────────────\n\n');
end

% vesselness.nii.gz
ves_hdr          = nii_hdr;
ves_hdr.Datatype = 'single';
ves_file         = fullfile(opt.output_dir, 'vesselness.nii.gz');
niftiwrite(single(V),                  ves_file, ves_hdr, 'Compressed', true);

% binary_vesselness.nii.gz
bv_hdr          = nii_hdr;
bv_hdr.Datatype = 'uint8';
bv_file         = fullfile(opt.output_dir, 'binary_vesselness.nii.gz');
niftiwrite(uint8(binary_vesselness),   bv_file,  bv_hdr,  'Compressed', true);

% filled_mask.nii.gz
fm_hdr          = nii_hdr;
fm_hdr.Datatype = 'uint8';
fm_file         = fullfile(opt.output_dir, 'filled_mask.nii.gz');
niftiwrite(uint8(binary_mask),         fm_file,  fm_hdr,  'Compressed', true);

% gap_report.mat
report_file = fullfile(opt.output_dir, 'gap_report.mat');
save(report_file, 'gap_candidates', 'opt', 'dsc_pre', 'dsc_post');

% 3D Slicer FCSV markup for review cases
fcsv_file = fullfile(opt.output_dir, 'gap_markers.fcsv');
write_fcsv(gap_candidates, fcsv_file, vox, info_img);

fprintf('=== Done ===\n');
fprintf('   vesselness.nii.gz        -> %s\n', ves_file);
fprintf('   binary_vesselness.nii.gz -> %s\n', bv_file);
fprintf('   filled_mask.nii.gz       -> %s\n', fm_file);
fprintf('   gap_report.mat           -> %s\n', report_file);
fprintf('   gap_markers.fcsv         -> %s\n', fcsv_file);
end


%% =======================================================================
% LOCAL FUNCTIONS
% =======================================================================

function d = dice_coeff(pred, gt)
pred = logical(pred(:));  gt = logical(gt(:));
d    = 2*sum(pred & gt) / (sum(pred) + sum(gt) + 1e-8);
end

function [L1, L2, L3, ev1] = hessian_eigvals(I, sig_vox, sigma_mm)
%HESSIAN_EIGVALS Compute scale-normalised Hessian eigenvalues and
% the principal axis eigenvector at each voxel.
%
% Inputs:
%   I        - 3D volume (double)
%   sig_vox  - [sx sy sz] Gaussian sigma in voxels
%   sigma_mm - scalar sigma in mm (for scale normalisation)
%
% Outputs:
%   L1, L2, L3 - eigenvalue volumes (|L1|<=|L2|<=|L3|)
%   ev1        - [X Y Z 3] axis eigenvector (for L1)

sz = size(I);

% Gaussian smoothing
Ig = imgaussfilt3(I, sig_vox);

% Second-order derivatives (scale-normalised by sigma^2)
s2 = sigma_mm^2;
Ixx = s2 * del2_along(Ig, 1);
Iyy = s2 * del2_along(Ig, 2);
Izz = s2 * del2_along(Ig, 3);
Ixy = s2 * mixed_deriv(Ig, 1, 2);
Ixz = s2 * mixed_deriv(Ig, 1, 3);
Iyz = s2 * mixed_deriv(Ig, 2, 3);

% Eigendecomposition voxel-by-voxel (vectorised over batch)
L1 = zeros(sz); L2 = zeros(sz); L3 = zeros(sz);
ev1 = zeros([sz 3]);

% Reshape to [N 6] matrix of unique Hessian entries
N   = prod(sz);
H6  = [Ixx(:), Iyy(:), Izz(:), Ixy(:), Ixz(:), Iyz(:)];

% Process in chunks to manage memory
chunk = 50000;
for start = 1:chunk:N
    idx = start:min(start+chunk-1, N);
    for k = idx
        H = [H6(k,1) H6(k,4) H6(k,5);
             H6(k,4) H6(k,2) H6(k,6);
             H6(k,5) H6(k,6) H6(k,3)];
        [V_eig, D_eig] = eig(H, 'vector');
        [~, order] = sort(abs(D_eig));
        D_eig = D_eig(order);
        V_eig = V_eig(:, order);
        L1(k) = D_eig(1);
        L2(k) = D_eig(2);
        L3(k) = D_eig(3);
        ev1(k) = V_eig(1,1);
        ev1(k + N) = V_eig(2,1);
        ev1(k + 2*N) = V_eig(3,1);
    end
end
end


function d2 = del2_along(I, dim)
%DEL2_ALONG Second derivative along a single dimension using central diff.
sz = size(I);
d2 = zeros(sz);
idx_m = {':',':',':'};
idx_0 = {':',':',':'};
idx_p = {':',':',':'};
idx_m{dim} = 1:sz(dim)-2;
idx_0{dim} = 2:sz(dim)-1;
idx_p{dim} = 3:sz(dim);
d2(idx_0{:}) = I(idx_p{:}) - 2*I(idx_0{:}) + I(idx_m{:});
end


function dxy = mixed_deriv(I, d1, d2)
%MIXED_DERIV Mixed second derivative d^2I / (d(d1) d(d2)).
dxy = gradient_along(gradient_along(I, d1), d2);
end


function dI = gradient_along(I, dim)
%GRADIENT_ALONG Central difference gradient along one dimension.
sz = size(I);
dI = zeros(sz);
idx_m = {':',':',':'};
idx_p = {':',':',':'};
idx_m{dim} = 1:sz(dim)-2;
idx_p{dim} = 3:sz(dim);
idx_c = {':',':',':'};
idx_c{dim} = 2:sz(dim)-1;
dI(idx_c{:}) = (I(idx_p{:}) - I(idx_m{:})) / 2;
end


function [L1, L2, L3, ev1] = sort_eigenvalues(L1, L2, L3, ev1)
%SORT_EIGENVALUES Ensure |L1| <= |L2| <= |L3| everywhere.
sz   = size(L1);
nvox = prod(sz);

% Gather: for each voxel, reorder [L1 L2 L3] by ascending |eigenvalue|
Lall   = reshape(cat(4, L1, L2, L3), nvox, 3);   % [nvox × 3]
absL   = abs(Lall);
[~, ord] = sort(absL, 2);                          % [nvox × 3] col-permutation

% Vectorised index gather: Lsorted(i,d) = Lall(i, ord(i,d))
row_idx  = repmat((1:nvox)', 1, 3);
lin      = sub2ind([nvox, 3], row_idx, ord);
Lsorted  = reshape(Lall(lin), [sz 3]);

L1 = Lsorted(:,:,:,1);
L2 = Lsorted(:,:,:,2);
L3 = Lsorted(:,:,:,3);
% ev1 remains aligned with the pre-sorted L1 from hessian_eigvals
end


function break_tips = detect_break_tips(skeleton, V, t_vessel, t_break, sz)
%DETECT_BREAK_TIPS Find degree-1 skeleton endpoints that indicate breaks.
%
% A break tip has:
%   - exactly 1 skeleton neighbour (degree 1)
%   - vesselness behind it (inside branch) > t_vessel
%   - is not at image boundary

[xi, yi, zi] = ind2sub(sz, find(skeleton));
coords = [xi yi zi];
n = size(coords, 1);
break_tips = struct('coord', {}, 'degree', {});

% 26-connectivity kernel offsets
[dx, dy, dz] = ndgrid(-1:1, -1:1, -1:1);
offsets = [dx(:) dy(:) dz(:)];
offsets(all(offsets==0, 2), :) = [];  % remove self

n_bt = 0;
for k = 1:n
    c = coords(k,:);

    % Skip boundary voxels
    if any(c <= 2) || any(c >= sz - 1), continue; end

    % Count 26-connected skeleton neighbours
    deg = 0;
    for o = 1:size(offsets,1)
        nb = c + offsets(o,:);
        if skeleton(nb(1), nb(2), nb(3)), deg = deg + 1; end
    end

    if deg ~= 1, continue; end  % not an endpoint

    % Sample vesselness 3 voxels behind (inside branch)
    % "behind" = toward the one neighbour
    nb_coord = [];
    for o = 1:size(offsets,1)
        nb = c + offsets(o,:);
        if skeleton(nb(1), nb(2), nb(3))
            nb_coord = nb; break;
        end
    end
    if isempty(nb_coord), continue; end

    v_behind = V(nb_coord(1), nb_coord(2), nb_coord(3));
    v_at_tip = V(c(1), c(2), c(3));

    % Break signature: vessel was present, now stops
    if v_behind > t_vessel && v_at_tip < t_vessel + 0.05
        n_bt = n_bt + 1;
        break_tips(n_bt).coord  = c;
        break_tips(n_bt).degree = deg;
    end
end
end


function [path, found] = trace_flow(start_coord, Fx, Fy, Fz, v1, V, ...
    t_resume, step_mm, vox, alpha_blend, max_gap_mm, sz, flow_sign)
%TRACE_FLOW Integrate the geometric flow field from a break tip.
%
% At each step the march direction blends:
%   flow_sign * F_unit   : geometric pull toward vessel-like structure
%   v1 at tip            : maintains axial alignment
%
% flow_sign = +1 for tip_a, -1 for tip_b (so both trace toward each other)

max_steps = ceil(max_gap_mm / step_mm);
step_vox  = step_mm ./ vox;            % step size in voxels per axis

pos  = double(start_coord);
path = pos;
found = false;

% Axis direction fixed from starting tip
c0 = num2cell(round(start_coord));
ax = squeeze(v1(c0{1}, c0{2}, c0{3}, :))';

for iter = 1:max_steps
    ci = max(1, min(sz, round(pos)));

    % Sample flow field at current position
    f = flow_sign * [Fx(ci(1),ci(2),ci(3)), ...
                     Fy(ci(1),ci(2),ci(3)), ...
                     Fz(ci(1),ci(2),ci(3))];

    % Ensure F hemisphere aligns with ax
    if dot(f, ax) < 0, f = -f; end
    f_norm = norm(f);
    if f_norm < 1e-6, f = ax; else f = f / f_norm; end

    % Blend flow with fixed axis direction
    direction = alpha_blend * f + (1 - alpha_blend) * ax;
    d_norm = norm(direction);
    if d_norm < 1e-9, break; end
    direction = direction / d_norm;

    % Advance position (step in voxels, scaled per axis)
    pos = pos + direction .* step_vox;

    % Bounds check
    if any(pos < 1) || any(pos > sz), break; end

    path = [path; pos]; %#ok<AGROW>

    % Check if vesselness has resumed
    ci = max(1, min(sz, round(pos)));
    if V(ci(1), ci(2), ci(3)) > t_resume
        found = true;
        break;
    end
end
end


function bridge = fast_march_bridge(cost_vol, ta, tb, sz, vox)
%FAST_MARCH_BRIDGE Cost-weighted shortest path via fast marching.
%
% Uses msfm3d (Dirk-Jan Kroon, FEX #24531).
% Falls back to straight-line interpolation if msfm3d not available.

try
    % msfm3d convention: source = single seed, extract path to target
    source_mask = false(sz);
    source_mask(ta(1), ta(2), ta(3)) = true;

    T = msfm3d(cost_vol, double(source_mask), true, true);

    % Back-trace gradient descent from tb to ta through arrival time T
    bridge = backtrack_path(T, ta, tb, sz, vox);

catch
    % Fallback: linear interpolation
    n_pts = max(10, ceil(norm(tb - ta)));
    t_lin = linspace(0, 1, n_pts)';
    bridge = ta + t_lin .* (tb - ta);
end
end


function path = backtrack_path(T, ta, tb, sz, vox)
%BACKTRACK_PATH Gradient descent in arrival time T from tb back to ta.
MAX_STEPS = 500;
pos  = double(tb);
path = pos;

for iter = 1:MAX_STEPS
    ci = max(1, min(sz, round(pos)));

    % Numerical gradient of T at current position (6-connectivity)
    gT = arrival_gradient(T, ci, sz, vox);
    g_norm = norm(gT);
    if g_norm < 1e-9, break; end
    step_dir = -gT / g_norm;       % descend arrival time toward source

    pos = pos + step_dir .* (vox / mean(vox));
    path = [path; pos]; %#ok<AGROW>

    % Stop when close to ta
    if norm(pos - double(ta)) < 1.5, break; end
    if any(pos < 1) || any(pos > sz), break; end
end
end


function gT = arrival_gradient(T, ci, sz, vox)
%ARRIVAL_GRADIENT Central difference gradient of arrival time T.
gT = zeros(1,3);
for d = 1:3
    ip = ci; im = ci;
    ip(d) = min(sz(d), ci(d)+1);
    im(d) = max(1,      ci(d)-1);
    gT(d) = (T(ip(1),ip(2),ip(3)) - T(im(1),im(2),im(3))) / (2*vox(d));
end
end


function conf = score_bridge(bridge, A_map, B_map, V, I, ...
    hu_vessel, vox, sz)
%SCORE_BRIDGE Compute confidence score [0,1] for a proposed bridge path.
%
% Four components:
%   geo        = mean(A * B) along path        (tubular geometry)
%   smoothness = 1 / (1 + var(dA/ds))          (smooth eccentricity)
%   hu         = mean sigmoid((HU - hu_vessel)) (intensity plausibility)
%   straight   = mean cos(angle between steps)  (path curvature)

if isempty(bridge) || size(bridge,1) < 2
    conf = 0; return;
end

n = size(bridge, 1);
a_vals = zeros(n,1); b_vals = zeros(n,1); hu_vals = zeros(n,1);

for k = 1:n
    ci = max(1, min(sz, round(bridge(k,:))));
    a_vals(k)  = A_map(ci(1), ci(2), ci(3));
    b_vals(k)  = B_map(ci(1), ci(2), ci(3));
    hu_vals(k) = 1 / (1 + exp(-(I(ci(1),ci(2),ci(3)) - hu_vessel)/50));
end

geo        = mean(a_vals .* b_vals);
smoothness = 1 / (1 + var(diff(a_vals)) * 100);
hu         = mean(hu_vals);

% Path straightness: mean dot product of consecutive unit step vectors
vecs  = diff(bridge, 1, 1);
norms = sqrt(sum(vecs.^2, 2)) + 1e-9;
uvecs = vecs ./ norms;
if size(uvecs,1) > 1
    dots = sum(uvecs(1:end-1,:) .* uvecs(2:end,:), 2);
    straightness = mean(dots);
else
    straightness = 1.0;
end

conf = 0.35*geo + 0.25*smoothness + 0.25*hu + 0.15*straightness;
conf = max(0, min(1, conf));
end


function write_fcsv(gap_candidates, filename, vox, info_img)
%WRITE_FCSV Write 3D Slicer FCSV markup file for review-flagged gaps.
%
% Each flagged gap gets two fiducial points (tip_a and tip_b midpoint)
% labelled with confidence and distance information.

fid = fopen(filename, 'w');
fprintf(fid, '# Markups fiducial file version = 4.11\n');
fprintf(fid, '# CoordinateSystem = LPS\n');
fprintf(fid, ['# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,' ...
              'lock,label,desc,associatedNodeID\n']);

n_written = 0;
for g = 1:length(gap_candidates)
    if ~strcmp(gap_candidates(g).action, 'flagged_for_review'), continue; end

    ta = gap_candidates(g).tip_a.coord;
    tb = gap_candidates(g).tip_b.coord;
    mid = (ta + tb) / 2;

    % Convert voxel coords to LPS mm
    % T matrix: voxel -> RAS, then negate x,y for LPS
    T = info_img.Transform.T;
    ras_mid = T(1:3,1:3)' * (mid' - 1) + T(1:3,4);
    lps_mid = [-ras_mid(1); -ras_mid(2); ras_mid(3)];

    n_written = n_written + 1;
    label = sprintf('gap_%03d_conf%.2f_%.1fmm', g, ...
        gap_candidates(g).confidence, gap_candidates(g).dist_mm);
    fprintf(fid, 'vtkMRMLMarkupsFiducialNode_%d,%.4f,%.4f,%.4f,', ...
        n_written, lps_mid(1), lps_mid(2), lps_mid(3));
    fprintf(fid, '0,0,0,1,1,1,0,%s,,\n', label);
end

fclose(fid);
fprintf('   Written %d review markers to %s\n', n_written, filename);
end
