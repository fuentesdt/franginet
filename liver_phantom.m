%% liver_phantom.m
%
% Generates a 3D NIfTI label image of a mathematical liver phantom with:
%   Label 0 — background
%   Label 1 — liver parenchyma  (superellipsoid + spherical harmonic bumps)
%   Label 2 — inflow vessels    (portal vein + hepatic artery, CCO-inspired)
%   Label 3 — outflow vessels   (hepatic veins draining to IVC)
%
% All geometric dimensions are calibrated to the HUMAN AVERAGE adult liver:
%
%   Morphometric reference (Umar et al. 2019; Poovathumkadavi et al. 2014;
%   Chandramohan et al. 2018):
%     Total volume      : 1400 – 1600 cm³  (mean ~1500 cm³)
%     Right-left span   : 20 – 23 cm       (mean ~21.5 cm, right lobe dominant)
%     Antero-posterior  : 10 – 13 cm       (mean ~11 cm)
%     Cranio-caudal     : 14 – 17 cm       (mean ~15 cm, right mid-clavicular line)
%     Right lobe        : ~65% of volume
%     Left lobe         : ~35% of volume
%
%   Vascular reference (Lafortune et al. 1987; Kang et al. 2014):
%     Portal vein trunk diameter  : 10 – 13 mm  (mean ~11 mm)
%     Hepatic artery diameter     : 4 –  6 mm   (mean ~5 mm, proper hepatic)
%     Right hepatic vein diameter : 10 – 14 mm  (mean ~12 mm at IVC confluence)
%     Middle hepatic vein         : 8  – 12 mm  (mean ~10 mm)
%     Left hepatic vein           : 7  – 10 mm  (mean ~8  mm)
%
% The phantom is anatomically motivated:
%   - Liver shape: 3D superellipsoid with right-lobe dominance,
%     IVC notch, and low-order spherical harmonic perturbations
%   - Inflow trees: two binary trees (portal vein, hepatic artery) grown
%     by constrained constructive optimisation (CCO) with Murray's law
%     bifurcation radii; trees share terminal node locations (portal triads)
%   - Outflow tree: hepatic vein binary tree whose terminal nodes are
%     co-located with inflow terminals (sinusoidal coupling); drains
%     toward the IVC notch on the posterior-superior surface
%
% OUTPUT:
%   liver_phantom.nii.gz   — uint8 label volume, 1 mm isotropic
%
% USAGE:
%   liver_phantom()                        % default 260x180x180 mm FOV
%   liver_phantom('voxel_mm', 1.5)         % coarser grid (faster)
%   liver_phantom('n_terminal', 300)       % fewer terminal nodes
%   liver_phantom('output', 'my_liver.nii.gz')
%
% PARAMETERS (name-value):
%   voxel_mm      isotropic voxel size (mm)              default 1.0
%   fov_mm        [Lx Ly Lz] field of view (mm)          default [260 180 180]
%   n_terminal    terminal nodes per vascular tree       default 200
%   r_vessel_min  minimum vessel radius to rasterise(mm) default 0.8
%   seed          random seed for reproducibility        default 42
%   output        output filename                        default 'liver_phantom.nii.gz'
%   verbose       print progress                         default true
%
% DEPENDENCIES:
%   Image Processing Toolbox (bwdist, imfill)
%   niftiwrite / niftiinfo  (MATLAB R2017b+ Image Processing Toolbox)
%   No external toolboxes required.

function liver_phantom(varargin)

%% -----------------------------------------------------------------------
% 0. Parse parameters
% -----------------------------------------------------------------------
p = inputParser;
addParameter(p, 'voxel_mm',     1.0);
addParameter(p, 'fov_mm',       [260 180 180]);
addParameter(p, 'n_terminal',   200);
addParameter(p, 'r_vessel_min', 0.8);
addParameter(p, 'seed',         42);
addParameter(p, 'output',       'liver_phantom.nii.gz');
addParameter(p, 'verbose',      true);
parse(p, varargin{:});
opt = p.Results;

rng(opt.seed);
vx = opt.voxel_mm;
fov = opt.fov_mm;

% Grid dimensions
Nx = round(fov(1)/vx);
Ny = round(fov(2)/vx);
Nz = round(fov(3)/vx);
sz = [Nx Ny Nz];

% Physical coordinate arrays (mm), origin at centre
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

%% -----------------------------------------------------------------------
% 1. Liver shape — superellipsoid + spherical harmonic perturbations
%    All dimensions calibrated to human adult average anatomy.
% -----------------------------------------------------------------------
vprint(opt, '[1/5] Building liver shape (human-average dimensions)...');

% -----------------------------------------------------------------------
% Human-average liver morphometry
% Reference: Umar et al. (2019) Folia Morphologica;
%            Poovathumkadavi et al. (2014) IJAV;
%            Chandramohan et al. (2018) IJAM
%
% Mean total volume  : ~1500 cm³ (range 1400–1600 cm³)
% RL span            : ~215 mm   (right lobe ~140 mm, left lobe ~75 mm)
% AP depth           : ~110 mm
% CC height          : ~150 mm   (right lobe at mid-clavicular line)
%   The CC dimension is the dominant axis; right-lobe height >> left lobe.
% Right lobe volume  : ~65%
% Left lobe volume   : ~35%
% -----------------------------------------------------------------------

% Centre of liver in FOV (mm): shifted right (+x) to reflect subphrenic
% position and slightly superior (+z) in the FOV
cx = 20;   % rightward shift (liver lies right of midline)
cy =  0;   % centred AP
cz = 10;   % slightly superior in the FOV

% Semi-axes of the encompassing superellipsoid (half-dimensions)
% a: half the total RL span = 215/2 = 107.5 mm.  But the superellipsoid
%    with nx>2 is smaller than its bounding box, so we use a slightly
%    larger value and rely on the asymmetry field below.
a = 100;   % RL half-span (mm); right side extends ~140mm, left ~75mm
b =  55;   % AP half-depth (110/2 = 55 mm)
c =  72;   % CC half-height (145/2 ≈ 72 mm — dominant craniocaudal axis)

% Superellipsoid exponents: n>2 flattens the sides (more liver-like than
% a pure ellipse).  nz slightly lower to keep the dome shape superiorly.
nx = 2.8;  % RL
ny = 2.4;  % AP (slightly more rounded anteriorly)
nz = 2.3;  % CC (softer superior dome)

% Shifted coordinates relative to liver centre
Xs = X - cx;
Ys = Y - cy;
Zs = Z - cz;

% Right-lobe asymmetry: the right half-span is ~140 mm, the left ~75 mm.
% We implement this as a spatially-varying semi-axis a_field.
%   Right side (Xs >= 0): a_right = 140 mm  (whole right lobe)
%   Left  side (Xs <  0): a_left  = 75  mm  (left lobe)
a_right = 105;   % slightly > 100 so right edge sits at ~105 mm from cx
a_left  =  70;   % left lobe half-width
a_field = a_right * ones(sz, 'single');
a_field(Xs < 0) = a_left;

% Superellipsoid implicit function: phi <= 0 defines interior
phi = (abs(Xs) ./ a_field).^nx + (abs(Ys) / b).^ny + (abs(Zs) / c).^nz - 1;

% -----------------------------------------------------------------------
% Anatomical surface features
% -----------------------------------------------------------------------

% IVC / hepatic vein notch: posterior-superior indentation where the IVC
% runs in a groove along the bare area.
% Location: right-posterior surface at approximately z = +40 mm (superior),
%           y = -35 mm (posterior), x = +35 mm (right of centre).
% Size: ~20 mm wide, 15 mm deep.
x_ivc = 35;  y_ivc = -35;  z_ivc = 38;
w_ivc = 20;  d_ivc = 14;   h_ivc = 22;
phi_ivc = ((Xs - x_ivc)/w_ivc).^2 + ((Ys - y_ivc)/d_ivc).^2 + ...
          ((Zs - z_ivc)/h_ivc).^2 - 1;
phi = phi + max(0, -phi_ivc) * 0.40;   % positive addition = remove tissue

% Porta hepatis / gallbladder fossa: inferior surface indent between
% right and left lobes.  Centred at approximately x=+5, y=+40 (anterior),
% z=-50 (inferior).  Ellipse ~30 x 12 x 12 mm.
x_gb =  5;  y_gb = 42;  z_gb = -48;
phi_gb = ((Xs - x_gb)/30).^2 + ((Ys - y_gb)/12).^2 + ((Zs - z_gb)/12).^2 - 1;
phi = phi + max(0, -phi_gb) * 0.30;

% Caudate lobe (segment I): posterior protrusion between IVC and porta
% hepatis.  Approximately 30 x 20 x 20 mm, centred at x=-10, y=-30, z=+15.
x_caud = -10;  y_caud = -30;  z_caud = 15;
phi_caud = ((Xs - x_caud)/18).^2 + ((Ys - y_caud)/12).^2 + ...
           ((Zs - z_caud)/12).^2 - 1;
phi = phi - max(0, -phi_caud) * 0.28;  % subtract = add tissue

% Inferior notch of the right lobe: slight concavity along the inferior
% border where the hepatic flexure of the colon contacts the liver.
x_rn = 55;  y_rn = 20;  z_rn = -55;
phi_rn = ((Xs - x_rn)/22).^2 + ((Ys - y_rn)/15).^2 + ((Zs - z_rn)/14).^2 - 1;
phi = phi + max(0, -phi_rn) * 0.20;

% -----------------------------------------------------------------------
% Spherical harmonic perturbations (low-order anatomical shape modes)
% -----------------------------------------------------------------------
r_sph  = sqrt(Xs.^2 + Ys.^2 + Zs.^2) + 1e-9;
theta  = acos(max(-1, min(1, Zs ./ r_sph)));  % polar angle [0, pi]
phi_sp = atan2(Ys, Xs);                        % azimuth [-pi, pi]

% Y_1^0 (cos theta): inferior tilt — the liver's inferior surface tilts
%        anteriorly by ~10–15 degrees.
Y10 = cos(theta);

% Y_2^0 (3cos²θ - 1)/2: slight oblate flattening (AP < CC < RL).
Y20 = 0.5 * (3*cos(theta).^2 - 1);

% Y_2^2 (sin²θ cos 2φ): left-right elongation asymmetry.
Y22 = sin(theta).^2 .* cos(2*phi_sp);

% Y_3^0: third-order superior dome rounding (more rounded superiorly).
Y30 = 0.5 * cos(theta) .* (5*cos(theta).^2 - 3);

% Perturbation amplitudes tuned so total volume stays near 1500 cm³
phi = phi + (-0.035)*Y10 + (0.025)*Y20 + (-0.045)*Y22 + (-0.015)*Y30;

% -----------------------------------------------------------------------
% Finalise liver mask
% -----------------------------------------------------------------------
liver_mask = phi <= 0;
liver_mask = imfill(liver_mask, 'holes');

label(liver_mask) = 1;

% -----------------------------------------------------------------------
% Volume validation against human average
% -----------------------------------------------------------------------
n_liver   = sum(liver_mask(:));
vol_cm3   = n_liver * vx^3 / 1000;
vol_target = 1500;   % cm³, human average
vol_err    = (vol_cm3 - vol_target) / vol_target * 100;

vprint(opt, '   Liver voxels : %d', n_liver);
vprint(opt, '   Liver volume : %.0f cm³  (target ~%d cm³, error %+.1f%%)', ...
    vol_cm3, vol_target, vol_err);

if abs(vol_err) > 20
    warning(['liver_phantom: volume %.0f cm³ deviates >20%% from human ' ...
             'average (%.0f cm³). Consider adjusting semi-axes a,b,c.'], ...
             vol_cm3, vol_target);
end

% Bounding box in mm — verify spans against morphometric references
[ix_l,iy_l,iz_l] = ind2sub(sz, find(liver_mask));
rl_span = (max(ix_l) - min(ix_l)) * vx;
ap_span = (max(iy_l) - min(iy_l)) * vx;
cc_span = (max(iz_l) - min(iz_l)) * vx;
vprint(opt, '   Bounding box : RL=%.0f mm  AP=%.0f mm  CC=%.0f mm', ...
    rl_span, ap_span, cc_span);
vprint(opt, '   Reference    : RL~215 mm   AP~110 mm   CC~150 mm');

%% -----------------------------------------------------------------------
% 2. Sample terminal node positions within liver
% -----------------------------------------------------------------------
vprint(opt, '[2/5] Placing portal triad terminal nodes...');

% Find all liver voxel coordinates
[ix, iy, iz] = ind2sub(sz, find(liver_mask));
liver_coords_vox = [ix iy iz];
liver_coords_mm  = [xv(ix)', yv(iy)', zv(iz)'];

% Subsample to n_terminal random positions (portal triad locations)
% Exclude a margin near the liver surface for interior terminals.
% Use 12 mm erosion — deeper margin appropriate for larger human liver
% (keeps terminals away from the capsule where vessel density is lower).
dist_from_edge = bwdist(~liver_mask) * vx;  % mm from surface
interior_mask  = dist_from_edge > 12.0;
[ixi, iyi, izi] = ind2sub(sz, find(interior_mask));
n_avail = length(ixi);

Nt = min(opt.n_terminal, n_avail);
idx_sel = randperm(n_avail, Nt);
terminal_mm = [xv(ixi(idx_sel))', yv(iyi(idx_sel))', zv(izi(idx_sel))'];

vprint(opt, '   Terminal nodes: %d', Nt);

%% -----------------------------------------------------------------------
% 3. Build inflow trees (portal vein + hepatic artery)
%    CCO-inspired binary tree with Murray's law radii
%    Root enters from inferior-medial border
% -----------------------------------------------------------------------
vprint(opt, '[3/5] Growing inflow vessel trees (portal vein + hepatic artery)...');

% Portal vein root: enters at the porta hepatis (hepatoduodenal ligament),
% inferior-medial surface.  Human PV trunk diameter ~11 mm (radius 5.5 mm).
% Entry point at approximately x=cx-8, y=+45 (anterior), z=cz-55 (inferior).
root_pv = [cx - 8,  45,  cz - 55];   % x,y,z in mm

% Hepatic artery proper: runs alongside PV in the hepatoduodenal ligament,
% slightly anterior and left.  Proper hepatic artery diameter ~5 mm (r=2.5).
root_ha = [cx - 18,  48,  cz - 52];

% Root radii calibrated to human averages:
%   Portal vein trunk       : 11 mm diameter -> 5.5 mm radius
%   Proper hepatic artery   :  5 mm diameter -> 2.5 mm radius
r_root_pv = 5.5;
r_root_ha = 2.5;

% Total flow proportional to r^3 (Murray)
% Build tree via recursive binary subdivision
pv_tree = build_cco_tree(root_pv, terminal_mm, r_root_pv, Nt);
% HA tree uses same terminals; scale radius so HA/PV ratio ~0.45 at root
ha_tree = build_cco_tree(root_ha, terminal_mm, r_root_ha, Nt);

% Rasterise both trees into label volume as label 2
label = rasterise_tree(pv_tree, label, xv, yv, zv, sz, 2, opt.r_vessel_min, liver_mask);
label = rasterise_tree(ha_tree, label, xv, yv, zv, sz, 2, opt.r_vessel_min * 0.6, liver_mask);

vprint(opt, '   Inflow segments: %d PV + %d HA', ...
    size(pv_tree.segments,1), size(ha_tree.segments,1));

%% -----------------------------------------------------------------------
% 4. Build outflow tree (hepatic veins)
%    Roots at IVC entry (posterior-superior), terminals co-located with
%    portal triad terminals (sinusoidal coupling)
% -----------------------------------------------------------------------
vprint(opt, '[4/5] Growing outflow vessel tree (hepatic veins)...');

% Hepatic vein roots at IVC confluence on posterior-superior surface.
% Human average diameters at IVC entry:
%   Right hepatic vein (RHV)   : ~12 mm diameter -> 6.0 mm radius
%   Middle hepatic vein (MHV)  : ~10 mm diameter -> 5.0 mm radius
%   Left hepatic vein (LHV)    :  ~8 mm diameter -> 4.0 mm radius
%
% Anatomical entry positions (posterior = negative y, superior = positive z):
%   RHV enters IVC at right posterior-superior, x~+55, y~-42, z~+55
%   MHV enters at midline, often shared ostium with LHV
%   LHV enters left posterior-superior
root_rhv = [cx + 55,  -42,  cz + 52];   % right hepatic vein
root_mhv = [cx +  8,  -38,  cz + 55];   % middle
root_lhv = [cx - 30,  -35,  cz + 50];   % left

r_root_hv_R = 6.0;   % right HV radius (mm)
r_root_hv_M = 5.0;   % middle HV radius (mm)
r_root_hv_L = 4.0;   % left HV radius (mm)

% Split terminals roughly into thirds by x-position for anatomical realism
term_x = terminal_mm(:,1);
q1 = quantile(term_x, 1/3);
q2 = quantile(term_x, 2/3);

term_R = terminal_mm(term_x >= q2, :);
term_M = terminal_mm(term_x >= q1 & term_x < q2, :);
term_L = terminal_mm(term_x <  q1, :);

% Ensure minimum 2 terminals per subtree
if size(term_R,1) < 2, term_R = terminal_mm(1:2,:); end
if size(term_M,1) < 2, term_M = terminal_mm(1:2,:); end
if size(term_L,1) < 2, term_L = terminal_mm(1:2,:); end

rhv_tree = build_cco_tree(root_rhv, term_R, r_root_hv_R, size(term_R,1));
mhv_tree = build_cco_tree(root_mhv, term_M, r_root_hv_M, size(term_M,1));
lhv_tree = build_cco_tree(root_lhv, term_L, r_root_hv_L, size(term_L,1));

% Rasterise as label 3
label = rasterise_tree(rhv_tree, label, xv, yv, zv, sz, 3, opt.r_vessel_min, liver_mask);
label = rasterise_tree(mhv_tree, label, xv, yv, zv, sz, 3, opt.r_vessel_min, liver_mask);
label = rasterise_tree(lhv_tree, label, xv, yv, zv, sz, 3, opt.r_vessel_min, liver_mask);

vprint(opt, '   Outflow segments: %d RHV + %d MHV + %d LHV', ...
    size(rhv_tree.segments,1), size(mhv_tree.segments,1), size(lhv_tree.segments,1));

%% -----------------------------------------------------------------------
% 5. Write NIfTI
% -----------------------------------------------------------------------
vprint(opt, '[5/5] Writing NIfTI: %s', opt.output);

% Build NIfTI info struct: 1mm isotropic, LPS orientation
info = build_nifti_info(sz, vx, fov);

% niftiwrite requires no .gz extension in the filename argument;
% pass 'Compressed' true for .nii.gz
outfile = opt.output;
if endsWith(outfile, '.gz')
    outfile_base = outfile(1:end-3);  % strip .gz
    niftiwrite(label, outfile_base, info, 'Compressed', true);
else
    niftiwrite(label, outfile, info, 'Compressed', false);
end

% Print label statistics
for lab = 1:3
    names = {'liver','inflow vessels','outflow vessels'};
    n = sum(label(:) == lab);
    vprint(opt, '   Label %d (%s): %d voxels (%.1f cm³)', ...
        lab, names{lab}, n, n*vx^3/1000);
end

vprint(opt, '=== Done: %s ===', opt.output);
end


%% =======================================================================
%  LOCAL FUNCTIONS
%% =======================================================================

function tree = build_cco_tree(root_mm, terminals_mm, r_root, n_term)
%BUILD_CCO_TREE  Build a binary vascular tree using a simplified CCO
% strategy: greedy nearest-neighbour with Murray's law radii.
%
% The tree is stored as a segment list:
%   tree.segments  [N x 7]  [x1 y1 z1 x2 y2 z2 radius_mm]
%   tree.nodes     [M x 3]  node coordinates (mm)
%
% Algorithm:
%   1. Start with root node.
%   2. For each terminal, find the nearest existing node.
%   3. Insert a new bifurcation point between the nearest node and its
%      parent; add the new terminal as a sibling branch.
%   4. Update radii using Murray's law bottom-up.
%
% This is a deterministic nearest-neighbour approximation to full CCO,
% which avoids the O(N^2) optimisation per terminal and is efficient
% for phantom generation.

Nt = min(n_term, size(terminals_mm,1));
if Nt < 1
    tree.segments = zeros(0,7);
    tree.nodes = root_mm;
    return;
end

% Initialise with root connected to first terminal
nodes    = root_mm;                    % [M x 3] node positions
parent   = 0;                          % parent index (0 = none, root)
children = {};                         % cell array of child index lists
radii    = r_root;                     % radius of each NODE (to its parent)
flow     = ones(1,1);                  % relative flow at each node

% Connect root to first terminal
nodes  = [nodes; terminals_mm(1,:)];
parent = [parent; 1];
children{1} = [2];
children{2} = [];
radii  = [radii; r_root * 0.7];
flow   = [flow; 1];

% Grow tree by adding remaining terminals
for k = 2:Nt
    new_term = terminals_mm(k,:);

    % Find nearest existing node
    dists = sqrt(sum((nodes - new_term).^2, 2));
    [~, nearest_idx] = min(dists);

    % Insert bifurcation: split the segment (parent[nearest] -> nearest)
    % by adding a midpoint, then attach new terminal from midpoint
    par_idx = parent(nearest_idx);

    if par_idx == 0
        % nearest IS the root — just add terminal as second root child
        bif_pt = (root_mm + new_term) / 2;
    else
        bif_pt = (nodes(par_idx,:) + nodes(nearest_idx,:)) / 2;
    end

    % Add bifurcation node
    n_bif = size(nodes,1) + 1;
    nodes  = [nodes; bif_pt];          %#ok<AGROW>
    parent = [parent; par_idx];        %#ok<AGROW>
    children{n_bif} = [nearest_idx, size(nodes,1)+1]; %#ok<AGROW>
    radii  = [radii; radii(nearest_idx)]; %#ok<AGROW>
    flow   = [flow; flow(nearest_idx)];   %#ok<AGROW>

    % Update nearest_idx parent to bif
    parent(nearest_idx) = n_bif;
    if par_idx > 0
        ch = children{par_idx};
        ch(ch == nearest_idx) = n_bif;
        children{par_idx} = ch;
    end

    % Add new terminal
    n_new = size(nodes,1) + 1;
    nodes  = [nodes; new_term];         %#ok<AGROW>
    parent = [parent; n_bif];           %#ok<AGROW>
    children{n_new} = [];               %#ok<AGROW>
    radii  = [radii; radii(nearest_idx) * 0.65]; %#ok<AGROW>
    flow   = [flow; 1];                 %#ok<AGROW>
end

% Recompute radii using Murray's law bottom-up
% Leaves have flow=1; interior nodes sum children flows
n_nodes = size(nodes,1);
node_flow = ones(n_nodes,1);

% Topological sort: leaves first (nodes with no children)
is_leaf = cellfun(@isempty, children(1:n_nodes));
order   = topological_order(parent, n_nodes);

for k = fliplr(order)
    ch = children{k};
    if ~isempty(ch)
        node_flow(k) = sum(node_flow(ch));
    end
end

% Murray's law: r^3 proportional to flow
% r_root is fixed; scale all radii by (flow/flow_root)^(1/3)
r_scaled = r_root * (node_flow / node_flow(1)).^(1/3);

% Build segment list [x1 y1 z1 x2 y2 z2 radius]
segs = zeros(n_nodes-1, 7);
seg_count = 0;
for k = 2:n_nodes
    par = parent(k);
    if par == 0, continue; end
    seg_count = seg_count + 1;
    r_seg = min(r_scaled(par), r_scaled(k));  % use child radius
    segs(seg_count,:) = [nodes(par,:), nodes(k,:), r_seg];
end
segs = segs(1:seg_count,:);

tree.segments = segs;
tree.nodes    = nodes;
end


function order = topological_order(parent, n_nodes)
%TOPOLOGICAL_ORDER  Returns node indices in BFS order from root.
visited = false(n_nodes,1);
queue   = find(parent == 0);  % roots
order   = [];
while ~isempty(queue)
    k = queue(1); queue(1) = [];
    if visited(k), continue; end
    visited(k) = true;
    order(end+1) = k; %#ok<AGROW>
    % Find children
    ch_idx = find(parent == k);
    queue  = [queue, ch_idx']; %#ok<AGROW>
end
end


function label = rasterise_tree(tree, label, xv, yv, zv, sz, lab, r_min, liver_mask)
%RASTERISE_TREE  Paint vessel segments into the label volume.
%
% For each segment, march along its axis and stamp a sphere of the
% segment's radius at each step.  Uses nearest-voxel indexing for speed.
%
% Only voxels inside liver_mask are painted (no extrahepatic vessels).

segs = tree.segments;    % [N x 7]: [x1 y1 z1 x2 y2 z2 r]
if isempty(segs), return; end

vx_mm = xv(2) - xv(1);  % assumes isotropic

for k = 1:size(segs,1)
    p1 = segs(k,1:3);
    p2 = segs(k,4:6);
    r  = max(segs(k,7), r_min);

    seg_len = norm(p2 - p1);
    if seg_len < 0.1, continue; end

    % Number of steps: at least one per voxel along the segment
    n_steps = max(2, ceil(seg_len / (vx_mm * 0.5)));
    t_vals  = linspace(0, 1, n_steps);

    for t = t_vals
        pt = p1 + t * (p2 - p1);

        % Find nearest grid index
        xi = nearest_idx(pt(1), xv, sz(1));
        yi = nearest_idx(pt(2), yv, sz(2));
        zi = nearest_idx(pt(3), zv, sz(3));

        % Stamp sphere of radius r in voxels
        r_vox = ceil(r / vx_mm);
        r_vox = max(1, r_vox);

        x_lo = max(1, xi-r_vox); x_hi = min(sz(1), xi+r_vox);
        y_lo = max(1, yi-r_vox); y_hi = min(sz(2), yi+r_vox);
        z_lo = max(1, zi-r_vox); z_hi = min(sz(3), zi+r_vox);

        % Sub-volume coordinate arrays
        xs = xv(x_lo:x_hi);
        ys = yv(y_lo:y_hi);
        zs = zv(z_lo:z_hi);

        [Xs, Ys, Zs] = meshgrid(xs, ys, zs);
        Xs = permute(Xs,[2 1 3]);
        Ys = permute(Ys,[2 1 3]);
        Zs = permute(Zs,[2 1 3]);

        in_sphere = (Xs - pt(1)).^2 + (Ys - pt(2)).^2 + (Zs - pt(3)).^2 <= r^2;
        in_liver  = liver_mask(x_lo:x_hi, y_lo:y_hi, z_lo:z_hi);

        % Paint only inside liver
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


function info = build_nifti_info(sz, vx_mm, fov_mm)
%BUILD_NIFTI_INFO  Create a minimal NIfTI info struct for niftiwrite.
%
% Uses a diagonal affine (RAS orientation): voxel origin at the
% physical centre of the volume (negative corner at -FOV/2).

info = struct();
info.Filename              = '';
info.Filemoddate           = datestr(now);
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

% Affine matrix: voxel [1,1,1] maps to physical [-FOV/2, -FOV/2, -FOV/2]
% RAS convention (x=R, y=A, z=S)
T = eye(4);
T(1,1) =  vx_mm;
T(2,2) =  vx_mm;
T(3,3) =  vx_mm;
T(1,4) = -fov_mm(1)/2;  % x origin
T(2,4) = -fov_mm(2)/2;  % y origin
T(3,4) = -fov_mm(3)/2;  % z origin

info.Transform         = affine3d(T');
info.TransformName     = 'Sform';
info.raw.sform_code    = 1;
info.raw.qform_code    = 1;
info.raw.pixdim        = [1 vx_mm vx_mm vx_mm 0 0 0 0];
info.raw.srow_x        = T(1,:);
info.raw.srow_y        = T(2,:);
info.raw.srow_z        = T(3,:);
info.raw.dim           = [3 sz(1) sz(2) sz(3) 1 1 1 1];
info.raw.datatype      = 2;  % uint8
info.raw.bitpix        = 8;
info.raw.xyzt_units    = 2;  % mm
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
%VPRINT  Print only if verbose flag is set.
if opt.verbose
    fprintf([fmt '\n'], varargin{:});
end
end
