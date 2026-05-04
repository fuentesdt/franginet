%% liver_phantom.m
%
% Generates 3-D NIfTI label images with:
%   Label 0 — background
%   Label 1 — liver parenchyma
%   Label 2 — inflow vessels    (portal vein + hepatic artery, CCO-inspired)
%   Label 3 — outflow vessels   (hepatic veins draining to IVC)
%
% -------------------------------------------------------------------------
% TWO OPERATING MODES
% -------------------------------------------------------------------------
%
%   1. MATHEMATICAL PHANTOM (default — no 'csv' argument):
%      Synthetic liver built from a superellipsoid + spherical-harmonic
%      shape model, calibrated to human-average adult anatomy.
%
%   2. REAL-DATA MODE ('csv' argument provided):
%      Reads a CSV file (columns: id, label) where each label column points
%      to a NIfTI segmentation of the liver boundary.  For each entry:
%        a. Load the NIfTI, binarize (threshold > 0.5).
%        b. Take the largest 26-connected component as the liver boundary.
%        c. Grow anatomically motivated CCO inflow (label 2) and outflow
%           (label 3) vessel trees constrained to that boundary.
%      The random seed for sample k is  opt.seed + (k-1)  so every case
%      produces distinct but reproducible vessel paths.
%
% -------------------------------------------------------------------------
% CSV FORMAT (first row is a header):
%   id,label
%   001,/path/to/sample001.nii.gz
%   002,/path/to/sample002.nii.gz
%
% -------------------------------------------------------------------------
% USAGE:
%   liver_phantom()
%   liver_phantom('voxel_mm', 1.5)
%   liver_phantom('csv', 'phantom.csv', 'outDir', './output')
%   liver_phantom('csv', 'phantom.csv', 'seed', 123, 'n_terminal', 150)
%
% -------------------------------------------------------------------------
% PARAMETERS (name-value):
%   csv           path to CSV file for real-data mode              default ''
%   outDir        output directory (real-data mode)                default '.'
%   voxel_mm      isotropic voxel size for math phantom (mm)       default 1.0
%   fov_mm        [Lx Ly Lz] field of view for math phantom (mm)   default [260 180 180]
%   n_terminal    terminal nodes per vascular tree                 default 200
%   r_vessel_min  minimum vessel radius to rasterise (mm)          default 0.8
%   seed          base random seed (math phantom) or per-sample    default 42
%                 base (real-data: sample k uses seed + k - 1)
%   output        output filename (math phantom mode only)         default 'liver_phantom.nii.gz'
%   verbose       print progress                                   default true
%
% -------------------------------------------------------------------------
% OUTPUT (real-data mode) — three files written per CSV row:
%
%   <outDir>/<id>_vessel_phantom.nii.gz
%     uint8 label volume in the same voxel grid as the input NIfTI.
%       0  background
%       1  liver parenchyma
%       2  inflow vessels  (portal vein + hepatic artery)
%       3  outflow vessels (right / middle / left hepatic veins → IVC)
%       4  tumor           (spiculated, spherical-harmonic shape)
%
%   <outDir>/<id>_skel_seed.nii.gz
%     uint8 skeleton buffer, same voxel grid.
%       0  background
%       1  1-voxel-wide inflow skeleton (bwskel of label 2)
%       5  single seed voxel on the hepatic-artery branch nearest the tumor
%
%   <outDir>/<id>_seed.fcsv
%     3D Slicer Markups fiducial file (LPS world coordinates).
%     Contains the label-5 seed point; use as catheter entry for
%     interventional planning (e.g. thermal embolisation).
%
% -------------------------------------------------------------------------
% DEPENDENCIES:
%   Image Processing Toolbox (bwdist, bwconncomp, imfill)
%   niftiwrite / niftiread / niftiinfo  (MATLAB R2017b+)

function liver_phantom(varargin)

p = inputParser;
addParameter(p, 'csv',          '');
addParameter(p, 'outDir',       '.');
addParameter(p, 'voxel_mm',     1.0);
addParameter(p, 'fov_mm',       [260 180 180]);
addParameter(p, 'n_terminal',   200);
addParameter(p, 'r_vessel_min', 0.8);
addParameter(p, 'seed',         42);
addParameter(p, 'output',       'liver_phantom.nii.gz');
addParameter(p, 'verbose',      true);
parse(p, varargin{:});
opt = p.Results;

if ~isempty(opt.csv)
    run_csv_mode(opt);
else
    run_math_phantom(opt);
end
end


%% ==========================================================================
%  MODE 1: REAL-DATA / CSV MODE
%% ==========================================================================

function run_csv_mode(opt)
%RUN_CSV_MODE  Process each NIfTI entry in the CSV as a liver boundary.

csv_path = opt.csv;
if ~isabs_path(csv_path)
    csv_path = fullfile(pwd, csv_path);
end
csv_dir = fileparts(csv_path);

% Parse CSV (first row is header: id, label)
[ids, paths] = read_csv(csv_path);
n_samples = numel(ids);

vprint(opt, '=== liver_phantom: real-data mode, %d samples ===', n_samples);

if ~exist(opt.outDir, 'dir')
    mkdir(opt.outDir);
end

for k = 1:n_samples
    sid   = char(ids(k));
    npath = char(paths(k));

    % Resolve relative paths against the directory containing the CSV
    if ~isabs_path(npath)
        npath = fullfile(csv_dir, npath);
    end

    vprint(opt, '\n[%d/%d] id=%s', k, n_samples, sid);
    vprint(opt, '   file: %s', npath);

    process_nifti_phantom(npath, sid, opt, opt.seed + (k - 1));
end

vprint(opt, '\n=== Done: %d samples written to %s ===', n_samples, opt.outDir);
end


function process_nifti_phantom(nii_path, sample_id, opt, seed_val)
%PROCESS_NIFTI_PHANTOM  Generate vessel labels for one NIfTI liver boundary.
%
% Vessel path randomness is controlled by seed_val so different calls with
% different seeds produce anatomically distinct but reproducible trees.

rng(seed_val, 'twister');

%% ---- Load and binarize ----
if ~exist(nii_path, 'file')
    warning('liver_phantom: file not found — %s', nii_path);
    return;
end
try
    info_in = niftiinfo(nii_path);
    vol     = single(niftiread(info_in));
catch ME
    warning('liver_phantom: cannot read %s — %s', nii_path, ME.message);
    return;
end

bin_mask = vol > 0.5;

%% ---- Largest 26-connected component → liver boundary ----
cc = bwconncomp(bin_mask, 26);
if cc.NumObjects == 0
    warning('liver_phantom: no foreground voxels in %s', nii_path);
    return;
end
comp_sizes = cellfun(@numel, cc.PixelIdxList);
[~, largest_idx] = max(comp_sizes);
liver_mask = false(size(vol));
liver_mask(cc.PixelIdxList{largest_idx}) = true;

sz     = size(liver_mask);
pixdim = double(info_in.PixelDimensions(1:3));
vx     = mean(pixdim);    % representative isotropic size for CCO distances

vprint(opt, '   Grid   : %d x %d x %d  |  voxel %.2f x %.2f x %.2f mm', ...
    sz(1), sz(2), sz(3), pixdim(1), pixdim(2), pixdim(3));

%% ---- Physical coordinate arrays (mm), origin at volume centre ----
% Dimension 1 = x (R), dimension 2 = y (A), dimension 3 = z (S) — RAS.
xv = ((1:sz(1)) - (sz(1)+1)/2) * pixdim(1);
yv = ((1:sz(2)) - (sz(2)+1)/2) * pixdim(2);
zv = ((1:sz(3)) - (sz(3)+1)/2) * pixdim(3);

%% ---- Liver geometry: centroid and bounding box in mm ----
[ix, iy, iz] = ind2sub(sz, find(liver_mask));
cx_mm = mean(xv(ix));
cy_mm = mean(yv(iy));
cz_mm = mean(zv(iz));                          %#ok<NASGU>

x_lo = xv(min(ix));  x_hi = xv(max(ix));  x_span = x_hi - x_lo;
y_lo = yv(min(iy));  y_hi = yv(max(iy));  y_span = y_hi - y_lo;
z_lo = zv(min(iz));  z_hi = zv(max(iz));  z_span = z_hi - z_lo;

vol_cm3 = numel(ix) * prod(pixdim) / 1000;
vprint(opt, '   Volume : %.0f cm³', vol_cm3);
vprint(opt, '   BBox   : X=[%.0f %.0f] Y=[%.0f %.0f] Z=[%.0f %.0f] mm', ...
    x_lo, x_hi, y_lo, y_hi, z_lo, z_hi);

%% ---- Root placement (anatomy-motivated, scaled to liver bounding box) ----
%
% NIfTI standard orientation is RAS+:
%   x increases to the Right, y increases Anteriorly, z increases Superiorly.
%
% Portal vein / hepatic artery enter at the hepatic hilum (porta hepatis):
%   inferior face (z near z_lo), anterior border (y near y_hi),
%   slightly right of centroid.
%
% Hepatic veins exit at the IVC confluence:
%   superior face (z near z_hi), posterior border (y near y_lo),
%   split right / middle / left along x.
%
% Fractional offsets are used so root positions scale with liver size.

% ---- Inflow (portal vein + hepatic artery) ----
root_pv = [cx_mm + 0.05*x_span, ...
           y_lo  + 0.65*y_span, ...
           z_lo  + 0.15*z_span];
r_root_pv = 5.5;   % human-average portal vein trunk radius (mm)

root_ha = [cx_mm - 0.08*x_span, ...
           y_lo  + 0.72*y_span, ...
           z_lo  + 0.18*z_span];
r_root_ha = 2.5;   % proper hepatic artery radius (mm)

% ---- Outflow (hepatic veins draining to IVC) ----
z_hv = z_lo + 0.90 * z_span;   % near superior face
y_hv = y_lo + 0.20 * y_span;   % near posterior (small y in RAS)

root_rhv = [x_lo + 0.78*x_span,  y_hv,  z_hv];
root_mhv = [cx_mm,                y_hv,  z_hv];
root_lhv = [x_lo + 0.22*x_span,  y_hv,  z_hv];

r_root_hv_R = 6.0;
r_root_hv_M = 5.0;
r_root_hv_L = 4.0;

vprint(opt, '   PV root : [%.0f %.0f %.0f] mm  (seed %d)', ...
    root_pv(1), root_pv(2), root_pv(3), seed_val);

%% ---- Terminal node sampling (portal triad positions) ----
% Keep terminals at least 12 mm from the liver surface.
dist_from_edge = bwdist(~liver_mask) * vx;   % approximate mm distance
interior_mask  = dist_from_edge > 12.0;

[ixi, iyi, izi] = ind2sub(sz, find(interior_mask));
n_avail = numel(ixi);

if n_avail < 4
    warning('liver_phantom: liver interior too small for sampling — %s', nii_path);
    return;
end

Nt = min(opt.n_terminal, n_avail);
idx_sel     = randperm(n_avail, Nt);
terminal_mm = [xv(ixi(idx_sel))', yv(iyi(idx_sel))', zv(izi(idx_sel))'];

vprint(opt, '   Terminals : %d', Nt);

%% ---- Build inflow trees (portal vein + hepatic artery) ----
pv_tree = build_cco_tree(root_pv, terminal_mm, r_root_pv, Nt);
ha_tree = build_cco_tree(root_ha, terminal_mm, r_root_ha, Nt);

%% ---- Build outflow trees (right / middle / left hepatic veins) ----
% Split terminals into thirds by x-position for anatomical drainage zones.
term_x = terminal_mm(:,1);
q1 = quantile(term_x, 1/3);
q2 = quantile(term_x, 2/3);

term_R = terminal_mm(term_x >= q2, :);
term_M = terminal_mm(term_x >= q1 & term_x < q2, :);
term_L = terminal_mm(term_x <  q1, :);

% Guard: ensure minimum 2 terminals per subtree
if size(term_R,1) < 2,  term_R = terminal_mm(1:min(2,Nt),:);  end
if size(term_M,1) < 2,  term_M = terminal_mm(1:min(2,Nt),:);  end
if size(term_L,1) < 2,  term_L = terminal_mm(1:min(2,Nt),:);  end

rhv_tree = build_cco_tree(root_rhv, term_R, r_root_hv_R, size(term_R,1));
mhv_tree = build_cco_tree(root_mhv, term_M, r_root_hv_M, size(term_M,1));
lhv_tree = build_cco_tree(root_lhv, term_L, r_root_hv_L, size(term_L,1));

vprint(opt, '   Inflow  : %d PV + %d HA segs', ...
    size(pv_tree.segments,1), size(ha_tree.segments,1));
vprint(opt, '   Outflow : %d RHV + %d MHV + %d LHV segs', ...
    size(rhv_tree.segments,1), size(mhv_tree.segments,1), size(lhv_tree.segments,1));

%% ---- Rasterize into label volume ----
label = zeros(sz, 'uint8');
label(liver_mask) = 1;

label = rasterise_tree(pv_tree,  label, xv, yv, zv, sz, 2, opt.r_vessel_min,       liver_mask);
label = rasterise_tree(ha_tree,  label, xv, yv, zv, sz, 2, opt.r_vessel_min * 0.6, liver_mask);
label = rasterise_tree(rhv_tree, label, xv, yv, zv, sz, 3, opt.r_vessel_min,       liver_mask);
label = rasterise_tree(mhv_tree, label, xv, yv, zv, sz, 3, opt.r_vessel_min,       liver_mask);
label = rasterise_tree(lhv_tree, label, xv, yv, zv, sz, 3, opt.r_vessel_min,       liver_mask);

%% ---- Tumor (label 4) ----
[label, tumor_ctr] = add_tumor(label, liver_mask, xv, yv, zv, sz, vx, opt);

%% ---- Inflow skeleton + upstream seed (label 5) ----
[skel_vol, seed_vox] = skeletonise_inflow(label, xv, yv, zv, sz, vx, root_ha, tumor_ctr, opt);

for lab = 1:4
    names = {'liver','inflow vessels','outflow vessels','tumor'};
    n = sum(label(:) == lab);
    vprint(opt, '   Label %d (%s): %d voxels (%.1f cm³)', ...
        lab, names{lab}, n, n * prod(pixdim) / 1000);
end

%% ---- Write output NIfTI (same geometry as input) ----
info_out = make_output_nifti_info(info_in, sz);
out_base = fullfile(opt.outDir, sprintf('%s_vessel_phantom.nii', sample_id));
niftiwrite(label, out_base, info_out, 'Compressed', true);
vprint(opt, '   Written : %s.gz', out_base);
skel_out = fullfile(opt.outDir, sprintf('%s_skel_seed.nii', sample_id));
niftiwrite(skel_vol, skel_out, info_out, 'Compressed', true);
vprint(opt, '   Skel buffer: %s.gz', skel_out);
fcsv_out   = fullfile(opt.outDir, sprintf('%s_seed.fcsv', sample_id));
seed_world = vox_to_world_ras(seed_vox, info_out);
write_seed_fcsv(seed_world, fcsv_out, sample_id, opt);
end


%% ==========================================================================
%  MODE 2: MATHEMATICAL PHANTOM (original behaviour, unchanged)
%% ==========================================================================

function run_math_phantom(opt)
%RUN_MATH_PHANTOM  Original superellipsoid + CCO phantom.

rng(opt.seed);
vx  = opt.voxel_mm;
fov = opt.fov_mm;

Nx = round(fov(1)/vx);
Ny = round(fov(2)/vx);
Nz = round(fov(3)/vx);
sz = [Nx Ny Nz];

xv = linspace(-fov(1)/2, fov(1)/2, Nx);
yv = linspace(-fov(2)/2, fov(2)/2, Ny);
zv = linspace(-fov(3)/2, fov(3)/2, Nz);
[X, Y, Z] = meshgrid(xv, yv, zv);
X = permute(X,[2 1 3]);
Y = permute(Y,[2 1 3]);
Z = permute(Z,[2 1 3]);

label = zeros(sz, 'uint8');

vprint(opt, '=== Liver phantom generator ===');
vprint(opt, 'Grid: %d x %d x %d  |  voxel %.1f mm  |  FOV %d x %d x %d mm', ...
    Nx,Ny,Nz, vx, fov(1),fov(2),fov(3));

%% [1/5] Liver shape
vprint(opt, '[1/5] Building liver shape (human-average dimensions)...');

cx = 20;   cy =  0;   cz = 10;
a = 100;   b  =  55;  c  =  72;
nx = 2.8;  ny = 2.4;  nz = 2.3;

Xs = X - cx;  Ys = Y - cy;  Zs = Z - cz;

a_right = 105;  a_left = 70;
a_field = a_right * ones(sz, 'single');
a_field(Xs < 0) = a_left;

phi = (abs(Xs) ./ a_field).^nx + (abs(Ys) / b).^ny + (abs(Zs) / c).^nz - 1;

% IVC notch
x_ivc = 35;  y_ivc = -35;  z_ivc = 38;
w_ivc = 20;  d_ivc = 14;   h_ivc = 22;
phi_ivc = ((Xs-x_ivc)/w_ivc).^2 + ((Ys-y_ivc)/d_ivc).^2 + ((Zs-z_ivc)/h_ivc).^2 - 1;
phi = phi + max(0, -phi_ivc) * 0.40;

% Porta hepatis / gallbladder fossa
x_gb =  5;  y_gb = 42;  z_gb = -48;
phi_gb = ((Xs-x_gb)/30).^2 + ((Ys-y_gb)/12).^2 + ((Zs-z_gb)/12).^2 - 1;
phi = phi + max(0, -phi_gb) * 0.30;

% Caudate lobe
x_caud = -10;  y_caud = -30;  z_caud = 15;
phi_caud = ((Xs-x_caud)/18).^2 + ((Ys-y_caud)/12).^2 + ((Zs-z_caud)/12).^2 - 1;
phi = phi - max(0, -phi_caud) * 0.28;

% Inferior right-lobe notch
x_rn = 55;  y_rn = 20;  z_rn = -55;
phi_rn = ((Xs-x_rn)/22).^2 + ((Ys-y_rn)/15).^2 + ((Zs-z_rn)/14).^2 - 1;
phi = phi + max(0, -phi_rn) * 0.20;

% Spherical harmonic perturbations
r_sph  = sqrt(Xs.^2 + Ys.^2 + Zs.^2) + 1e-9;
theta  = acos(max(-1, min(1, Zs ./ r_sph)));
phi_sp = atan2(Ys, Xs);
Y10 = cos(theta);
Y20 = 0.5 * (3*cos(theta).^2 - 1);
Y22 = sin(theta).^2 .* cos(2*phi_sp);
Y30 = 0.5 * cos(theta) .* (5*cos(theta).^2 - 3);
phi = phi + (-0.035)*Y10 + (0.025)*Y20 + (-0.045)*Y22 + (-0.015)*Y30;

liver_mask = phi <= 0;
liver_mask = imfill(liver_mask, 'holes');
label(liver_mask) = 1;

n_liver  = sum(liver_mask(:));
vol_cm3  = n_liver * vx^3 / 1000;
vol_err  = (vol_cm3 - 1500) / 1500 * 100;
vprint(opt, '   Liver voxels : %d', n_liver);
vprint(opt, '   Liver volume : %.0f cm³  (target ~1500 cm³, error %+.1f%%)', vol_cm3, vol_err);
if abs(vol_err) > 20
    warning(['liver_phantom: volume %.0f cm³ deviates >20%% from human ' ...
             'average (1500 cm³).'], vol_cm3);
end

[ix_l,iy_l,iz_l] = ind2sub(sz, find(liver_mask));
vprint(opt, '   Bounding box : RL=%.0f mm  AP=%.0f mm  CC=%.0f mm', ...
    (max(ix_l)-min(ix_l))*vx, (max(iy_l)-min(iy_l))*vx, (max(iz_l)-min(iz_l))*vx);
vprint(opt, '   Reference    : RL~215 mm   AP~110 mm   CC~150 mm');

%% [2/5] Terminal nodes
vprint(opt, '[2/5] Placing portal triad terminal nodes...');

dist_from_edge = bwdist(~liver_mask) * vx;
interior_mask  = dist_from_edge > 12.0;
[ixi, iyi, izi] = ind2sub(sz, find(interior_mask));
n_avail = length(ixi);
Nt = min(opt.n_terminal, n_avail);
idx_sel = randperm(n_avail, Nt);
terminal_mm = [xv(ixi(idx_sel))', yv(iyi(idx_sel))', zv(izi(idx_sel))'];
vprint(opt, '   Terminal nodes: %d', Nt);

%% [3/5] Inflow trees
vprint(opt, '[3/5] Growing inflow vessel trees (portal vein + hepatic artery)...');

root_pv = [cx - 8,   45,  cz - 55];
root_ha = [cx - 18,  48,  cz - 52];
r_root_pv = 5.5;
r_root_ha = 2.5;

pv_tree = build_cco_tree(root_pv, terminal_mm, r_root_pv, Nt);
ha_tree = build_cco_tree(root_ha, terminal_mm, r_root_ha, Nt);

label = rasterise_tree(pv_tree, label, xv, yv, zv, sz, 2, opt.r_vessel_min,       liver_mask);
label = rasterise_tree(ha_tree, label, xv, yv, zv, sz, 2, opt.r_vessel_min * 0.6, liver_mask);
vprint(opt, '   Inflow segments: %d PV + %d HA', ...
    size(pv_tree.segments,1), size(ha_tree.segments,1));

%% [4/5] Outflow tree
vprint(opt, '[4/5] Growing outflow vessel tree (hepatic veins)...');

root_rhv = [cx + 55,  -42,  cz + 52];
root_mhv = [cx +  8,  -38,  cz + 55];
root_lhv = [cx - 30,  -35,  cz + 50];
r_root_hv_R = 6.0;
r_root_hv_M = 5.0;
r_root_hv_L = 4.0;

term_x = terminal_mm(:,1);
q1 = quantile(term_x, 1/3);
q2 = quantile(term_x, 2/3);
term_R = terminal_mm(term_x >= q2, :);
term_M = terminal_mm(term_x >= q1 & term_x < q2, :);
term_L = terminal_mm(term_x <  q1, :);

if size(term_R,1) < 2, term_R = terminal_mm(1:2,:); end
if size(term_M,1) < 2, term_M = terminal_mm(1:2,:); end
if size(term_L,1) < 2, term_L = terminal_mm(1:2,:); end

rhv_tree = build_cco_tree(root_rhv, term_R, r_root_hv_R, size(term_R,1));
mhv_tree = build_cco_tree(root_mhv, term_M, r_root_hv_M, size(term_M,1));
lhv_tree = build_cco_tree(root_lhv, term_L, r_root_hv_L, size(term_L,1));

label = rasterise_tree(rhv_tree, label, xv, yv, zv, sz, 3, opt.r_vessel_min, liver_mask);
label = rasterise_tree(mhv_tree, label, xv, yv, zv, sz, 3, opt.r_vessel_min, liver_mask);
label = rasterise_tree(lhv_tree, label, xv, yv, zv, sz, 3, opt.r_vessel_min, liver_mask);
vprint(opt, '   Outflow segments: %d RHV + %d MHV + %d LHV', ...
    size(rhv_tree.segments,1), size(mhv_tree.segments,1), size(lhv_tree.segments,1));

%% [5/6] Tumor
vprint(opt, '[5/6] Placing tumor...');
[label, tumor_ctr] = add_tumor(label, liver_mask, xv, yv, zv, sz, vx, opt);

%% [5b/6] Inflow skeleton + upstream seed (label 5)
[skel_vol, seed_vox] = skeletonise_inflow(label, xv, yv, zv, sz, vx, root_ha, tumor_ctr, opt);

%% [6/6] Write NIfTI
vprint(opt, '[6/6] Writing NIfTI: %s', opt.output);
info = build_nifti_info(sz, vx, fov);
outfile = opt.output;
if endsWith(outfile, '.gz')
    niftiwrite(label, outfile(1:end-3), info, 'Compressed', true);
else
    niftiwrite(label, outfile, info, 'Compressed', false);
end
stem     = outfile(1:end-7);   % strip .nii.gz
skel_base = [stem '_skel_seed.nii'];
niftiwrite(skel_vol, skel_base, info, 'Compressed', true);
vprint(opt, '   Skel buffer: %s.gz', skel_base);
seed_world = vox_to_world_ras(seed_vox, info);
write_seed_fcsv(seed_world, [stem '_seed.fcsv'], 'seed', opt);

for lab = 1:4
    names = {'liver','inflow vessels','outflow vessels','tumor'};
    n = sum(label(:) == lab);
    vprint(opt, '   Label %d (%s): %d voxels (%.1f cm³)', ...
        lab, names{lab}, n, n*vx^3/1000);
end
vprint(opt, '=== Done: %s ===', opt.output);
end


%% ==========================================================================
%  SHARED HELPER FUNCTIONS
%% ==========================================================================

function [label, tumor_ctr] = add_tumor(label, liver_mask, xv, yv, zv, sz, vx, opt)
%ADD_TUMOR  Place one randomly shaped malignant-looking tumor (label 4).
%
% Shape model  r(θ,φ) = r0 · [1 + SH_deform(θ,φ) + spicule_bumps(θ,φ)]
%
%   SH_deform   : real spherical harmonic expansion degrees 1–6 with random
%                 coefficients; amplitude decreases with degree so low-order
%                 modes drive the gross shape and high-order modes add texture.
%
%   spicule_bumps: N narrow Gaussian bumps placed at random directions on the
%                 unit sphere (geodesic-distance kernel); mimics the spiculated
%                 margin of hepatocellular carcinoma / metastases.
%
% Constraints
%   • Vessel voxels (label 2, 3) are preserved — tumor does not overwrite them.
%   • Tumor voxels outside liver_mask are forced to 0 (background).

%% ---- Random size and placement ----
diam_mm = 10 + 20 * rand();     % uniform in [10, 30] mm
r0      = diam_mm / 2;          % base radius (mm)

% Require tumour centre to be at least (r0 + 5 mm) from liver surface
margin_mm  = r0 + 5;
dist_edge  = bwdist(~liver_mask) * vx;   % mm to nearest non-liver voxel
place_mask = dist_edge > margin_mm;

[pxi, pyi, pzi] = ind2sub(sz, find(place_mask));
if isempty(pxi)
    % Relax margin to half-radius if liver is small
    [pxi, pyi, pzi] = ind2sub(sz, find(dist_edge > r0 * 0.5));
end
if isempty(pxi)
    warning('liver_phantom: cannot place tumour — liver interior too small');
    return;
end

sel = randi(numel(pxi));
cx  = xv(pxi(sel));
cy  = yv(pyi(sel));
cz  = zv(pzi(sel));

%% ---- Spherical-harmonic deformation coefficients (degrees 1–6) ----
% Draw raw N(0,1) coefficients then rescale so overall RMS = sh_rms.
% Amplitude is further weighted by 1/(l+1) so coarse modes dominate.
sh_rms = 0.05 + 0.12 * rand();    % 5–17 % of r0 RMS deformation

sh_a   = zeros(7, 8);   % sh_a(l, m+1)  — cosine terms, l=1..6
sh_b   = zeros(7, 7);   % sh_b(l, m)    — sine  terms,  l=1..6, m=1..l
raw    = zeros(1, sum(1:6)*2 + 6);   % pre-allocated; overwritten below
cnt    = 0;

for l = 1:6
    w = 1 / (l + 1);    % downweight higher degrees
    for m = 0:l
        c = w * randn();
        sh_a(l, m+1) = c;
        cnt = cnt + 1;  raw(cnt) = c;
        if m > 0
            s = w * randn();
            sh_b(l, m) = s;
            cnt = cnt + 1;  raw(cnt) = s;
        end
    end
end

raw   = raw(1:cnt);
scale = sh_rms / (sqrt(mean(raw.^2)) + eps);
sh_a  = sh_a * scale;
sh_b  = sh_b * scale;

%% ---- Spicule parameters ----
n_spic     = randi([3, 10]);
% Uniform distribution on the unit sphere via inverse-CDF of cos θ
spic_theta = acos(2*rand(n_spic,1) - 1);
spic_phi   = 2*pi * rand(n_spic,1);
spic_amp   = 0.15 + 0.35 * rand(n_spic,1);    % 15–50 % of r0 extra height
spic_sig   = 0.10 + 0.20 * rand(n_spic,1);    % angular half-width σ (radians)

vprint(opt, '   Tumour: diam=%.1f mm  SH-rms=%.2f  spicules=%d  centre=[%.0f %.0f %.0f] mm', ...
    diam_mm, sh_rms, n_spic, cx, cy, cz);

%% ---- Build sub-volume bounding box ----
r_max  = r0 * (1 + 3*sh_rms + max(spic_amp));   % generous upper bound
dx_vox = xv(2) - xv(1);
r_pad  = ceil(r_max / dx_vox) + 2;

xi_c = nearest_idx(cx, xv, sz(1));
yi_c = nearest_idx(cy, yv, sz(2));
zi_c = nearest_idx(cz, zv, sz(3));

xi1 = max(1, xi_c-r_pad);  xi2 = min(sz(1), xi_c+r_pad);
yi1 = max(1, yi_c-r_pad);  yi2 = min(sz(2), yi_c+r_pad);
zi1 = max(1, zi_c-r_pad);  zi2 = min(sz(3), zi_c+r_pad);

xs = xv(xi1:xi2) - cx;
ys = yv(yi1:yi2) - cy;
zs = zv(zi1:zi2) - cz;

[Xs, Ys, Zs] = meshgrid(xs, ys, zs);
Xs = permute(Xs, [2 1 3]);
Ys = permute(Ys, [2 1 3]);
Zs = permute(Zs, [2 1 3]);

R     = sqrt(Xs.^2 + Ys.^2 + Zs.^2);
R_eps = max(R, 1e-9);
COS_T = max(-1, min(1, Zs ./ R_eps));
THETA = acos(COS_T);
PHI   = atan2(Ys, Xs);

subsz = [length(xi1:xi2), length(yi1:yi2), length(zi1:zi2)];

%% ---- Evaluate SH deformation over the sub-volume ----
SH_DEF = zeros(subsz);

for l = 1:6
    % legendre(l, X) returns [l+1, numel(X)] for row-vector X
    cos_flat = COS_T(:)';                          % [1, N]
    P = legendre(l, cos_flat);                     % [l+1, N]
    P = reshape(P, [l+1, subsz(1), subsz(2), subsz(3)]);

    for m = 0:l
        Pm = reshape(P(m+1,:,:,:), subsz);

        % Log-space normalization to avoid overflow at high (l,m)
        log_nf = 0.5*(log(2*l+1) - log(4*pi) + ...
                      gammaln(l-m+1) - gammaln(l+m+1));
        nf = exp(log_nf);

        if m == 0
            SH_DEF = SH_DEF + sh_a(l,1) * nf * Pm;
        else
            nf2 = nf * sqrt(2);
            SH_DEF = SH_DEF + nf2 * Pm .* ...
                (sh_a(l,m+1) .* cos(m*PHI) + sh_b(l,m) .* sin(m*PHI));
        end
    end
end

%% ---- Evaluate spicule bumps ----
SPIC_DEF = zeros(subsz);

for k = 1:n_spic
    % Geodesic angular distance between each voxel direction and spicule centre
    cos_ang  = sin(THETA).*sin(spic_theta(k)).*cos(PHI - spic_phi(k)) + ...
               cos(THETA).*cos(spic_theta(k));
    ang_dist = acos(max(-1, min(1, cos_ang)));
    SPIC_DEF = SPIC_DEF + spic_amp(k) .* exp(-ang_dist.^2 / (2*spic_sig(k)^2));
end

%% ---- Tumor surface radius and classification ----
R_SURFACE = max(r0 .* (1 + SH_DEF + SPIC_DEF), 0);
tumor_sub = R <= R_SURFACE;

%% ---- Apply label constraints ----
sub_lbl    = label(xi1:xi2, yi1:yi2, zi1:zi2);
sub_liver  = liver_mask(xi1:xi2, yi1:yi2, zi1:zi2);
sub_vessel = (sub_lbl == 2) | (sub_lbl == 3);

% Paint label 4 only where inside tumor AND inside liver AND not a vessel
sub_lbl(tumor_sub & sub_liver & ~sub_vessel) = 4;

% Tumor voxels outside liver boundary → background
sub_lbl(tumor_sub & ~sub_liver) = 0;

label(xi1:xi2, yi1:yi2, zi1:zi2) = sub_lbl;
tumor_ctr = [cx, cy, cz];   % mm world coordinates, returned to caller
end


function [skel_vol, seed_vox] = skeletonise_inflow(label, xv, yv, zv, sz, vx, root_ha_mm, tumor_ctr, opt)
%SKELETONISE_INFLOW  Skeleton buffer for inflow vessels with upstream seed.
%
% Returns skel_vol (uint8, same size as label):
%   0 — background
%   1 — inflow skeleton voxel  (bwskel of label==2)
%   5 — upstream seed (1×1×1 voxel, randomly chosen between the hepatic-
%        artery root and the tumour-nearest skeleton voxel)
%
% "Upstream" is defined by Euclidean distance to root_ha_mm: candidates
% must be closer to the HA root than the tumour-nearest skeleton voxel,
% ensuring the seed lies on the feeding-artery side of the tumour.

%% Skeletonise inflow (label 2)
inflow_mask = label == 2;
skel_vol = zeros(sz, 'uint8');
seed_vox = [];   % returned empty if placement fails

if ~any(inflow_mask(:))
    warning('liver_phantom: no inflow voxels (label 2) — skeleton is empty');
    return;
end

skel = bwskel(inflow_mask);
skel_vol(skel) = 1;

%% Skeleton voxel coordinates in mm
[si, sj, sk_] = ind2sub(sz, find(skel));
if isempty(si)
    warning('liver_phantom: bwskel returned empty skeleton');
    return;
end
skel_mm = [xv(si)', yv(sj)', zv(sk_)'];   % [N × 3]

%% Tumour centroid in mm
if ~isempty(tumor_ctr) && numel(tumor_ctr) == 3
    tc_mm = tumor_ctr(:)';
else
    [ti, tj, tk_t] = ind2sub(sz, find(label == 4));
    if isempty(ti)
        vprint(opt, '   Seed: no tumour found — skipping seed placement');
        return;
    end
    tc_mm = [mean(xv(ti)), mean(yv(tj)), mean(zv(tk_t))];
end

%% T_skel: skeleton voxel nearest to tumour centroid
d_tumor  = sqrt(sum((skel_mm - tc_mm).^2, 2));
[~, T_idx] = min(d_tumor);

%% Upstream candidates: skeleton voxels closer to HA root than T_skel
d_root   = sqrt(sum((skel_mm - root_ha_mm).^2, 2));
d_T_root = d_root(T_idx);

% Must not be at the very root entry (leave at least 30 % margin)
d_min    = max(5, 0.30 * d_T_root);
upstream = (d_root < d_T_root) & (d_root > d_min);

if any(upstream)
    cands   = find(upstream);
    pick    = cands(randi(numel(cands)));
else
    % Fallback: place seed at T_skel itself
    pick = T_idx;
    vprint(opt, '   Seed: no upstream candidates — seed at tumour-nearest voxel');
end

%% Place 1×1×1 seed (label 5)
seed_vi = si(pick);
seed_vj = sj(pick);
seed_vk = sk_(pick);
skel_vol(seed_vi, seed_vj, seed_vk) = 5;
seed_vox = [seed_vi, seed_vj, seed_vk];   % 1-indexed MATLAB voxel, returned to caller

vprint(opt, '   Seed (label 5): vox=[%d %d %d]  d_root=%.0f mm  d_tumour=%.0f mm', ...
    seed_vi, seed_vj, seed_vk, d_root(pick), d_tumor(pick));
end


function tree = build_cco_tree(root_mm, terminals_mm, r_root, n_term)
%BUILD_CCO_TREE  Greedy nearest-neighbour CCO tree with Murray's-law radii.
%
%   tree.segments  [N x 7]  [x1 y1 z1 x2 y2 z2 radius_mm]
%   tree.nodes     [M x 3]  node coordinates (mm)

Nt = min(n_term, size(terminals_mm,1));
if Nt < 1
    tree.segments = zeros(0,7);
    tree.nodes = root_mm;
    return;
end

nodes    = root_mm;
parent   = 0;
children = {};
radii    = r_root;
flow     = ones(1,1);

nodes  = [nodes; terminals_mm(1,:)];
parent = [parent; 1];
children{1} = [2];
children{2} = [];
radii  = [radii; r_root * 0.7];
flow   = [flow; 1];

for k = 2:Nt
    new_term = terminals_mm(k,:);
    dists = sqrt(sum((nodes - new_term).^2, 2));
    [~, nearest_idx] = min(dists);
    par_idx = parent(nearest_idx);

    if par_idx == 0
        bif_pt = (root_mm + new_term) / 2;
    else
        bif_pt = (nodes(par_idx,:) + nodes(nearest_idx,:)) / 2;
    end

    n_bif = size(nodes,1) + 1;
    nodes  = [nodes; bif_pt];           %#ok<AGROW>
    parent = [parent; par_idx];         %#ok<AGROW>
    children{n_bif} = [nearest_idx, size(nodes,1)+1]; %#ok<AGROW>
    radii  = [radii; radii(nearest_idx)];  %#ok<AGROW>
    flow   = [flow; flow(nearest_idx)];    %#ok<AGROW>

    parent(nearest_idx) = n_bif;
    if par_idx > 0
        ch = children{par_idx};
        ch(ch == nearest_idx) = n_bif;
        children{par_idx} = ch;
    end

    n_new = size(nodes,1) + 1;
    nodes  = [nodes; new_term];          %#ok<AGROW>
    parent = [parent; n_bif];            %#ok<AGROW>
    children{n_new} = [];                %#ok<AGROW>
    radii  = [radii; radii(nearest_idx) * 0.65]; %#ok<AGROW>
    flow   = [flow; 1];                  %#ok<AGROW>
end

n_nodes   = size(nodes,1);
node_flow = ones(n_nodes,1);
order     = topological_order(parent, n_nodes);

for k = fliplr(order)
    ch = children{k};
    if ~isempty(ch)
        node_flow(k) = sum(node_flow(ch));
    end
end

r_scaled = r_root * (node_flow / node_flow(1)).^(1/3);

segs      = zeros(n_nodes-1, 7);
seg_count = 0;
for k = 2:n_nodes
    par = parent(k);
    if par == 0, continue; end
    seg_count = seg_count + 1;
    r_seg = min(r_scaled(par), r_scaled(k));
    segs(seg_count,:) = [nodes(par,:), nodes(k,:), r_seg];
end

tree.segments = segs(1:seg_count,:);
tree.nodes    = nodes;
end


function order = topological_order(parent, n_nodes)
%TOPOLOGICAL_ORDER  BFS from root node(s).
visited = false(n_nodes,1);
queue   = find(parent == 0);
order   = [];
while ~isempty(queue)
    k = queue(1); queue(1) = [];
    if visited(k), continue; end
    visited(k) = true;
    order(end+1) = k; %#ok<AGROW>
    queue = [queue, find(parent == k)']; %#ok<AGROW>
end
end


function label = rasterise_tree(tree, label, xv, yv, zv, sz, lab, r_min, liver_mask)
%RASTERISE_TREE  Paint vessel segments into the label volume (liver-clipped).

segs = tree.segments;
if isempty(segs), return; end

vx_mm = xv(2) - xv(1);   % assumes approximately isotropic

for k = 1:size(segs,1)
    p1 = segs(k,1:3);
    p2 = segs(k,4:6);
    r  = max(segs(k,7), r_min);

    seg_len = norm(p2 - p1);
    if seg_len < 0.1, continue; end

    n_steps = max(2, ceil(seg_len / (vx_mm * 0.5)));
    t_vals  = linspace(0, 1, n_steps);

    for t = t_vals
        pt = p1 + t * (p2 - p1);

        xi = nearest_idx(pt(1), xv, sz(1));
        yi = nearest_idx(pt(2), yv, sz(2));
        zi = nearest_idx(pt(3), zv, sz(3));

        r_vox = max(1, ceil(r / vx_mm));
        x_lo = max(1, xi-r_vox);  x_hi = min(sz(1), xi+r_vox);
        y_lo = max(1, yi-r_vox);  y_hi = min(sz(2), yi+r_vox);
        z_lo = max(1, zi-r_vox);  z_hi = min(sz(3), zi+r_vox);

        xs = xv(x_lo:x_hi);  ys = yv(y_lo:y_hi);  zs = zv(z_lo:z_hi);
        [Xs, Ys, Zs] = meshgrid(xs, ys, zs);
        Xs = permute(Xs,[2 1 3]);
        Ys = permute(Ys,[2 1 3]);
        Zs = permute(Zs,[2 1 3]);

        in_sphere = (Xs-pt(1)).^2 + (Ys-pt(2)).^2 + (Zs-pt(3)).^2 <= r^2;
        in_liver  = liver_mask(x_lo:x_hi, y_lo:y_hi, z_lo:z_hi);

        sub = label(x_lo:x_hi, y_lo:y_hi, z_lo:z_hi);
        sub(in_sphere & in_liver) = uint8(lab);
        label(x_lo:x_hi, y_lo:y_hi, z_lo:z_hi) = sub;
    end
end
end


function idx = nearest_idx(val, vec, n)
%NEAREST_IDX  Nearest index in a coordinate vector.
[~, idx] = min(abs(vec - val));
idx = max(1, min(n, idx));
end


function info_out = make_output_nifti_info(info_in, sz)
%MAKE_OUTPUT_NIFTI_INFO  Clone NIfTI header from input; set uint8 datatype.
info_out = info_in;
info_out.Datatype                 = 'uint8';
info_out.BitsPerPixel             = 8;
info_out.ImageSize                = sz;
info_out.AdditiveOffset           = 0;
info_out.MultiplicativeScaling    = 0;
info_out.DisplayIntensityRange    = [0 4];
if isfield(info_out, 'raw')
    info_out.raw.datatype = 2;   % uint8
    info_out.raw.bitpix   = 8;
    info_out.raw.dim(2:4) = sz(:)';
end
end


function info = build_nifti_info(sz, vx_mm, fov_mm)
%BUILD_NIFTI_INFO  Minimal NIfTI info for math phantom (diagonal RAS affine).
info = struct();
info.Filename              = '';
info.Filemoddate           = datestr(now);       %#ok<TNOW1,DATST>
info.FileSize              = 0;
info.Version               = 'NIfTI1';
info.Description           = '';
info.ImageSize             = sz;
info.PixelDimensions       = [vx_mm vx_mm vx_mm];
info.Datatype              = 'uint8';
info.BitsPerPixel          = 8;
info.SpaceUnits            = 'Millimeter';
info.TimeUnits             = 'Second';
info.AdditiveOffset        = 0;
info.MultiplicativeScaling = 0;
info.TimeOffset            = 0;
info.SliceCode             = 'Unknown';
info.FrequencyDimension    = 0;
info.PhaseDimension        = 0;
info.SpatialDimension      = 0;
info.DisplayIntensityRange = [0 0];
info.Qfactor               = 1;
T = eye(4);
T(1,1) =  vx_mm;  T(1,4) = -fov_mm(1)/2;
T(2,2) =  vx_mm;  T(2,4) = -fov_mm(2)/2;
T(3,3) =  vx_mm;  T(3,4) = -fov_mm(3)/2;
info.Transform         = affine3d(T');
info.TransformName     = 'Sform';
info.raw.sform_code    = 1;
info.raw.qform_code    = 1;
info.raw.pixdim        = [1 vx_mm vx_mm vx_mm 0 0 0 0];
info.raw.srow_x        = T(1,:);
info.raw.srow_y        = T(2,:);
info.raw.srow_z        = T(3,:);
info.raw.dim           = [3 sz(1) sz(2) sz(3) 1 1 1 1];
info.raw.datatype      = 2;
info.raw.bitpix        = 8;
info.raw.xyzt_units    = 2;
end


function world_ras = vox_to_world_ras(vox_1idx, nii_info)
%VOX_TO_WORLD_RAS  NIfTI sform/qform → world RAS, 1-indexed voxel input.
if isempty(vox_1idx), world_ras = []; return; end
v0 = double(vox_1idx(:)) - 1;    % 1-indexed → 0-indexed

% sform takes priority (sform_code > 0)
if isfield(nii_info,'raw') && isfield(nii_info.raw,'sform_code') && ...
        nii_info.raw.sform_code > 0
    T = [nii_info.raw.srow_x(:)'; nii_info.raw.srow_y(:)'; nii_info.raw.srow_z(:)'];
    world_ras = (T * [v0; 1])';
    return;
end

% Fall back to qform (qform_code > 0)
if isfield(nii_info,'raw') && isfield(nii_info.raw,'qform_code') && ...
        nii_info.raw.qform_code > 0
    b = double(nii_info.raw.quatern_b);
    c = double(nii_info.raw.quatern_c);
    d = double(nii_info.raw.quatern_d);
    a = sqrt(max(0, 1 - b^2 - c^2 - d^2));
    R = [a^2+b^2-c^2-d^2,  2*(b*c-a*d),    2*(b*d+a*c); ...
         2*(b*c+a*d),       a^2-b^2+c^2-d^2, 2*(c*d-a*b); ...
         2*(b*d-a*c),       2*(c*d+a*b),    a^2-b^2-c^2+d^2];
    pd   = double(nii_info.PixelDimensions(1:3));
    qfac = sign(double(nii_info.raw.pixdim(1)));
    if qfac == 0, qfac = 1; end
    RS   = R * diag([pd(1), pd(2), pd(3)*qfac]);
    offs = [double(nii_info.raw.qoffset_x); ...
            double(nii_info.raw.qoffset_y); ...
            double(nii_info.raw.qoffset_z)];
    world_ras = (RS * v0 + offs)';
    return;
end

% Last resort: PixelDimensions only (no rotation, origin at 0)
pd = double(nii_info.PixelDimensions(1:3));
world_ras = (pd(:) .* v0)';
end


function write_seed_fcsv(seed_mm, fcsv_path, label_str, opt)
%WRITE_SEED_FCSV  Write a single-point Slicer Markups Fiducial (.fcsv) file.
%
% Coordinate convention: .fcsv uses LPS.  seed_mm is in RAS (as used
% throughout liver_phantom), so x and y are negated on output.
%
% File format matches gap_markers.fcsv (Markups fiducial version 4.11).

if isempty(seed_mm)
    vprint(opt, '   FCSV: no seed position — skipping %s', fcsv_path);
    return;
end

% RAS → LPS
lps = [-seed_mm(1), -seed_mm(2), seed_mm(3)];

fid = fopen(fcsv_path, 'w');
if fid < 0
    warning('liver_phantom: cannot open %s for writing', fcsv_path);
    return;
end

fprintf(fid, '# Markups fiducial file version = 4.11\n');
fprintf(fid, '# CoordinateSystem = LPS\n');
fprintf(fid, '# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n');
fprintf(fid, 'vtkMRMLMarkupsFiducialNode_0,%.4f,%.4f,%.4f,0,0,0,1,1,1,0,%s,,\n', ...
    lps(1), lps(2), lps(3), label_str);

fclose(fid);
vprint(opt, '   Seed FCSV : %s', fcsv_path);
end


function [ids, paths] = read_csv(csv_path)
%READ_CSV  Parse a two-column CSV with a header row (id, label).
fid = fopen(csv_path, 'r');
if fid < 0
    error('liver_phantom: cannot open CSV — %s', csv_path);
end
header = fgetl(fid);   %#ok<NASGU>  % skip header line
ids   = {};
paths = {};
while ~feof(fid)
    line = fgetl(fid);
    if ~ischar(line), continue; end
    line = strtrim(line);
    if isempty(line), continue; end
    parts = strsplit(line, ',');
    if numel(parts) < 2, continue; end
    ids{end+1}   = strtrim(parts{1});   %#ok<AGROW>
    paths{end+1} = strtrim(parts{2});   %#ok<AGROW>
end
fclose(fid);
ids   = ids(:);
paths = paths(:);
end


function tf = isabs_path(p)
%ISABS_PATH  True if p is an absolute filesystem path.
tf = ~isempty(p) && (p(1)=='/' || p(1)=='\' || (numel(p)>2 && p(2)==':'));
end


function tf = endsWith(str, suffix)
%ENDSWITH  Pre-R2016b compatible endsWith.
if length(str) < length(suffix)
    tf = false;
else
    tf = strcmp(str(end-length(suffix)+1:end), suffix);
end
end


function vprint(opt, fmt, varargin)
%VPRINT  Print only when verbose is true.
if opt.verbose
    fprintf([fmt '\n'], varargin{:});
end
end
