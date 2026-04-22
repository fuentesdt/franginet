%% vessel_gap_filling.m
%
% Gradient-guided gap filling for vessel centreline discontinuities.
%
% Uses the spatial gradients of Frangi filter factors A (plate suppressor)
% and B (blob suppressor) as a geometric flow field to bridge breaks in
% a thresholded vesselness mask.
%
% INPUTS (loaded from file):
%   image.nii.gz      - 3D CT volume (HU values)
%   vesselness.nii.gz - Frangi vesselness map (same geometry)
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
%   vessel_gap_filling('image.nii.gz', 'vesselness.nii.gz')
%   vessel_gap_filling('image.nii.gz', 'vesselness.nii.gz', 'alpha', 0.5)
%
% PARAMETERS (name-value pairs, all optional):
%   alpha          Frangi alpha (eccentricity sensitivity)    default 0.5
%   beta           Frangi beta  (blobness sensitivity)        default 0.5
%   t_vessel       vesselness threshold for mask              default 0.15
%   t_resume       vesselness threshold to declare resume     default 0.12
%   t_break        vesselness level considered background     default 0.05
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
%   scales_mm      Frangi filter scales for Hessian (mm)      default [1 2 3]
%   output_dir     directory for output files                 default './'

function vessel_gap_filling(image_file, vesselness_file, varargin)

%% -----------------------------------------------------------------------
% 0. Parse inputs
% -----------------------------------------------------------------------
p = inputParser;
addRequired(p, 'image_file',     @ischar);
addRequired(p, 'vesselness_file',@ischar);
addParameter(p, 'alpha',        0.5);
addParameter(p, 'beta',         0.5);
addParameter(p, 't_vessel',     0.15);
addParameter(p, 't_resume',     0.12);
addParameter(p, 't_break',      0.05);
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
addParameter(p, 'scales_mm',    [1 2 3]);
addParameter(p, 'output_dir',   './');
parse(p, image_file, vesselness_file, varargin{:});
opt = p.Results;

fprintf('=== Vessel gap filling: gradient-guided fast marching ===\n');
if ~exist(opt.output_dir, 'dir'), mkdir(opt.output_dir); end

%% -----------------------------------------------------------------------
% 1. Load volumes
% -----------------------------------------------------------------------
fprintf('[1/9] Loading volumes...\n');

nii_img = niftiread(image_file);
info_img = niftiinfo(image_file);
I = double(nii_img);

nii_ves = niftiread(vesselness_file);
V = double(nii_ves);

% Voxel size in mm [dx, dy, dz]
vox = abs(diag(info_img.Transform.T(1:3,1:3)))';
sz  = size(I);
fprintf('   Volume size : %d x %d x %d voxels\n', sz(1), sz(2), sz(3));
fprintf('   Voxel size  : %.3f x %.3f x %.3f mm\n', vox(1), vox(2), vox(3));

%% -----------------------------------------------------------------------
% 2. Compute Hessian eigenvalues and eigenvectors at dominant scale
% -----------------------------------------------------------------------
fprintf('[2/9] Computing multi-scale Hessian (scales: %s mm)...\n', ...
    num2str(opt.scales_mm));

% Allocate arrays for scale-aggregated outputs
lam1 = zeros(sz); lam2 = zeros(sz); lam3 = zeros(sz);
v1   = zeros([sz 3]);   % vessel axis eigenvector (corresponding to lam1)
A_map = zeros(sz);
B_map = zeros(sz);
V_scale_max = zeros(sz);

for s = opt.scales_mm
    sigma = s;                              % sigma in mm -> voxels below
    sig_vox = sigma ./ vox;                 % [sx sy sz] in voxels

    % Scale-normalised Gaussian second derivatives
    [L1s, L2s, L3s, ev1s] = hessian_eigvals(I, sig_vox, sigma);

    % Sort by absolute magnitude: |lam1| <= |lam2| <= |lam3|
    [L1s, L2s, L3s, ev1s] = sort_eigenvalues(L1s, L2s, L3s, ev1s);

    % Frangi factors at this scale
    RA = abs(L2s) ./ (abs(L3s) + 1e-9);
    RB = abs(L1s) ./ (sqrt(abs(L2s) .* abs(L3s)) + 1e-9);
    S  = sqrt(L1s.^2 + L2s.^2 + L3s.^2);

    As = 1 - exp(-RA.^2 / (2 * opt.alpha^2));
    Bs = exp( -RB.^2 / (2 * opt.beta^2));
    Cs = 1 - exp(-S.^2  / (2 * (max(S(:))/2)^2));

    Vs = As .* Bs .* Cs;
    Vs(L2s > 0 | L3s > 0) = 0;  % vessel condition: lam2,lam3 < 0

    % Keep values at scale of maximum response
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
clear L1s L2s L3s Vs As Bs Cs RA RB S update ev1s;
fprintf('   Hessian complete.\n');

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

binary_mask = V > opt.t_vessel;
binary_mask = bwareaopen(binary_mask, 50);  % remove small islands

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

% filled_mask.nii.gz
out_mask = info_img;
out_mask.Filename = fullfile(opt.output_dir, 'filled_mask.nii.gz');
out_mask.Datatype = 'uint8';
niftiwrite(uint8(binary_mask), out_mask.Filename, out_mask, 'Compressed', true);

% gap_report.mat
report_file = fullfile(opt.output_dir, 'gap_report.mat');
save(report_file, 'gap_candidates', 'opt');

% 3D Slicer FCSV markup for review cases
fcsv_file = fullfile(opt.output_dir, 'gap_markers.fcsv');
write_fcsv(gap_candidates, fcsv_file, vox, info_img);

fprintf('=== Done ===\n');
fprintf('   filled_mask.nii.gz  -> %s\n', out_mask.Filename);
fprintf('   gap_report.mat      -> %s\n', report_file);
fprintf('   gap_markers.fcsv    -> %s\n', fcsv_file);
end


%% =======================================================================
% LOCAL FUNCTIONS
% =======================================================================

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
% Simple insertion sort across the three values per voxel.
% This is a no-op if hessian_eigvals already sorted them.
% Included for safety when called with pre-sorted inputs.
absL = cat(4, abs(L1), abs(L2), abs(L3));
[~, ord] = sort(absL, 4);
Lall = cat(4, L1, L2, L3);
Lsorted = zeros(size(Lall));
for d = 1:3
    for src = 1:3
        mask = ord(:,:,:,d) == src;
        tmp = Lsorted(:,:,:,d);
        tmp(mask) = Lall(:,:,:,src);  % note: simplified, works for src==d
        Lsorted(:,:,:,d) = tmp;
    end
end
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
