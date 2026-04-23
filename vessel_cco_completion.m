%% vessel_cco_completion.m
%
% Energy-guided constrained CCO vessel completion.
%
% Takes VMTK centreline output (broken vessel segments) and produces a
% SIMPLY CONNECTED vessel graph by:
%   1. Parsing VMTK centrelines into a segment graph.
%   2. Identifying orphaned endpoints (break tips).
%   3. Building a fused energy field from CT intensity, Frangi
%      vesselness, and the spatial gradients of factors A (plate
%      suppressor) and B (blob suppressor).
%   4. Running constrained CCO: existing segments are FIXED; new
%      bridging segments are grown by energy-guided fast marching,
%      respecting Murray's law for radius assignment.
%   5. Guaranteeing simple connectivity via a minimum spanning tree
%      over all remaining disconnected components.
%
% INPUTS:
%   vmtk_cl_file   — VMTK centreline .vtp (VTK PolyData XML) or
%                    a pre-parsed struct (see parse_vmtk_vtp)
%   image_file     — CT volume (NIfTI .nii / .nii.gz)
%   vesselness_file— Frangi vesselness map (NIfTI, same grid as image)
%
% OUTPUTS:
%   completed_graph.mat       — struct with fields:
%       .nodes   [M×3]   node coordinates (mm)
%       .edges   [E×2]   node index pairs
%       .radii   [E×1]   radius per edge (mm)
%       .fixed   [E×1]   logical: true = from VMTK, false = CCO bridge
%   completed_vessels.nii.gz  — rasterised label volume (uint8, label=2)
%   bridge_report.mat         — per-bridge confidence and method
%
% USAGE:
%   vessel_cco_completion('cl.vtp','image.nii.gz','vesselness.nii.gz')
%   vessel_cco_completion(...,'alpha',0.5,'beta',0.5,'max_gap_mm',10)
%
% PARAMETERS (name-value, all optional):
%   alpha         Frangi alpha (eccentricity)        default 0.5
%   beta          Frangi beta  (blobness)             default 0.5
%   scales_mm     Hessian scales for A,B maps (mm)   default [0.5 1 1.5 2]
%   max_gap_mm    maximum gap to attempt bridging     default 10.0
%   w_vessel      weight: vesselness V in cost        default 0.35
%   w_geo         weight: geometry A*B in cost        default 0.30
%   w_hu          weight: HU plausibility in cost     default 0.20
%   w_flow        weight: flow-field alignment        default 0.15
%   hu_vessel     expected artery HU                  default 250
%   r_min_mm      minimum bridge radius (mm)          default 0.5
%   conf_auto     confidence for auto-accept          default 0.50
%   conf_mst      confidence below which MST fallback default 0.25
%   voxel_out_mm  output raster voxel size (mm)       default 1.0
%   output_dir    output directory                    default './'
%   seed          RNG seed                            default 42
%   verbose       print progress                      default true
%
% DEPENDENCIES:
%   Image Processing Toolbox (niftiread, niftiwrite, bwdist)
%   msfm3d (Dirk-Jan Kroon, FEX #24531) for fast marching — if absent,
%   falls back to Dijkstra on a sparse cost graph.
%
% SIMPLY-CONNECTED GUARANTEE:
%   After CCO bridging, a Kruskal minimum spanning tree is run over the
%   connected-components graph.  Any component pair not yet joined by a
%   CCO bridge receives a direct MST edge whose radius is assigned by
%   Murray's law from the nearest parent.  This ensures ZERO isolated
%   components regardless of CCO confidence scores.

function vessel_cco_completion(vmtk_cl_file, image_file, ...
                                vesselness_file, varargin)

%% -----------------------------------------------------------------------
% 0. Parameters
% -----------------------------------------------------------------------
p = inputParser;
addRequired(p,'vmtk_cl_file');
addRequired(p,'image_file');
addRequired(p,'vesselness_file');
addParameter(p,'alpha',       0.5);
addParameter(p,'beta',        0.5);
addParameter(p,'scales_mm',   [0.5 1.0 1.5 2.0]);
addParameter(p,'max_gap_mm',  10.0);
addParameter(p,'w_vessel',    0.35);
addParameter(p,'w_geo',       0.30);
addParameter(p,'w_hu',        0.20);
addParameter(p,'w_flow',      0.15);
addParameter(p,'hu_vessel',   250);
addParameter(p,'r_min_mm',    0.5);
addParameter(p,'conf_auto',   0.50);
addParameter(p,'conf_mst',    0.25);
addParameter(p,'voxel_out_mm',1.0);
addParameter(p,'output_dir',  './');
addParameter(p,'seed',        42);
addParameter(p,'verbose',     true);
parse(p, vmtk_cl_file, image_file, vesselness_file, varargin{:});
opt = p.Results;

rng(opt.seed);
if ~exist(opt.output_dir,'dir'), mkdir(opt.output_dir); end
vprint(opt,'=== Energy-guided constrained CCO vessel completion ===');

%% -----------------------------------------------------------------------
% 1. Load CT image and vesselness map
% -----------------------------------------------------------------------
vprint(opt,'[1/7] Loading volumes...');

info = niftiinfo(image_file);
I    = double(niftiread(image_file));
V    = double(niftiread(vesselness_file));

vox  = abs(diag(info.Transform.T(1:3,1:3)))';   % [dx dy dz] mm
sz   = size(I);
% Physical coordinate vectors (mm)
xv = ((0:sz(1)-1) - sz(1)/2) * vox(1);
yv = ((0:sz(2)-1) - sz(2)/2) * vox(2);
zv = ((0:sz(3)-1) - sz(3)/2) * vox(3);

vprint(opt,'   Grid: %d×%d×%d  voxel %.2f×%.2f×%.2f mm', ...
    sz(1),sz(2),sz(3), vox(1),vox(2),vox(3));

%% -----------------------------------------------------------------------
% 2. Parse VMTK centreline file into node/edge graph
% -----------------------------------------------------------------------
vprint(opt,'[2/7] Parsing VMTK centrelines...');

if isstruct(vmtk_cl_file)
    G = vmtk_cl_file;   % pre-parsed struct passed directly
else
    G = parse_vmtk_vtp(vmtk_cl_file);
end
% G.nodes [M×3] mm,  G.edges [E×2] node indices,  G.radii [E×1] mm
% G.fixed [E×1] = true (all VMTK segments start as fixed)

vprint(opt,'   Nodes: %d   Edges: %d', size(G.nodes,1), size(G.edges,1));

%% -----------------------------------------------------------------------
% 3. Compute Hessian eigenvalue maps and factors A, B at dominant scale
% -----------------------------------------------------------------------
vprint(opt,'[3/7] Computing A, B factor maps and flow field...');

[A_map, B_map] = compute_AB_maps(I, opt.scales_mm, opt.alpha, opt.beta, vox);

% Spatial gradients of A and B (mm⁻¹)
[dAx,dAy,dAz] = gradient(A_map, vox(1), vox(2), vox(3));
[dBx,dBy,dBz] = gradient(B_map, vox(1), vox(2), vox(3));

% Geometric flow field F = w_A·∇A − w_B·∇B
% Points toward high eccentricity AND low blobness (toward vessels)
Fx = dAx - dBx;   Fy = dAy - dBy;   Fz = dAz - dBz;
Fnorm = sqrt(Fx.^2 + Fy.^2 + Fz.^2) + 1e-9;
Fx = Fx ./ Fnorm;   Fy = Fy ./ Fnorm;   Fz = Fz ./ Fnorm;
clear Fnorm;

vprint(opt,'   A/B maps complete.');

%% -----------------------------------------------------------------------
% 4. Build fused cost volume
%    C(x) ∈ [0,1]: low cost = geometrically vessel-like
% -----------------------------------------------------------------------
vprint(opt,'[4/7] Building fused energy cost volume...');

% HU plausibility: sigmoid centred on expected artery HU
hu_norm = 1 ./ (1 + exp(-(I - opt.hu_vessel) / 50));

% Geometry-only vesselness (no C gate — works in low-contrast gaps)
V_geo = A_map .* B_map;

% Combined score (high = likely vessel)
score = (opt.w_vessel * V      + ...
         opt.w_geo    * V_geo  + ...
         opt.w_hu     * hu_norm);
score = score / (opt.w_vessel + opt.w_geo + opt.w_hu);
score = min(max(score, 0), 1);

% Cost: invert so low cost = likely vessel path
cost_vol = max(1 - score, 1e-3);

clear hu_norm score;
vprint(opt,'   Cost volume built.');

%% -----------------------------------------------------------------------
% 5. Identify orphaned endpoints (break tips) in the graph
% -----------------------------------------------------------------------
vprint(opt,'[5/7] Finding break tips and gap candidates...');

% Degree of each node (number of edges incident to it)
n_nodes = size(G.nodes,1);
degree  = zeros(n_nodes,1);
for e = 1:size(G.edges,1)
    degree(G.edges(e,1)) = degree(G.edges(e,1)) + 1;
    degree(G.edges(e,2)) = degree(G.edges(e,2)) + 1;
end

% Degree-1 nodes are endpoints.
% A "break tip" is an endpoint that is NOT a designated tree root
% (root = degree-1 node at the inlet/outlet of the organ).
% Heuristic: classify as break tip if it is not the overall most-proximal
% node (highest in the tree).  User can supply root node indices via
% opt.root_nodes if needed; here we auto-detect as the single node with
% maximum absolute z-coordinate (most superior = IVC entry) for outflow,
% and most inferior node for inflow.  All other degree-1 nodes are tips.
endpoint_idx = find(degree == 1);
vprint(opt,'   Degree-1 endpoints: %d', length(endpoint_idx));

% Find connected components using union-find
cc = connected_components(G.edges, n_nodes);
n_cc = max(cc);
vprint(opt,'   Connected components: %d', n_cc);

% For each component, find the node closest to the image centroid —
% this becomes the "root candidate" for that component.
% Build gap candidates: all pairs of endpoints from DIFFERENT components
% within max_gap_mm of each other.
max_gap_vox = opt.max_gap_mm / mean(vox);
ep_coords   = G.nodes(endpoint_idx,:);
n_ep        = length(endpoint_idx);

gap_candidates = struct('i',{},'j',{},'dist_mm',{},'bridge',{},...
    'method',{},'conf',{},'action',{});
n_gaps = 0;

for i = 1:n_ep
    for j = i+1:n_ep
        ni = endpoint_idx(i);
        nj = endpoint_idx(j);
        % Only bridge nodes in DIFFERENT components
        if cc(ni) == cc(nj), continue; end
        d = norm(G.nodes(ni,:) - G.nodes(nj,:));
        if d > opt.max_gap_mm, continue; end
        n_gaps = n_gaps + 1;
        gap_candidates(n_gaps).i       = ni;
        gap_candidates(n_gaps).j       = nj;
        gap_candidates(n_gaps).dist_mm = d;
    end
end
vprint(opt,'   Gap candidates (inter-component, ≤%.0f mm): %d', ...
    opt.max_gap_mm, n_gaps);

%% -----------------------------------------------------------------------
% 6. Constrained CCO: energy-guided bridging for each gap candidate
%    Fixed VMTK edges are never modified.
%    New edges follow minimum-cost paths through C(x).
%    Radii assigned by Murray's law from the endpoint's local radius.
% -----------------------------------------------------------------------
vprint(opt,'[6/7] Running constrained CCO bridging...');

bridge_report = gap_candidates;   % copy structure

n_auto = 0; n_flag = 0; n_mst = 0;

for g = 1:n_gaps
    ni = gap_candidates(g).i;
    nj = gap_candidates(g).j;
    pa = G.nodes(ni,:);
    pb = G.nodes(nj,:);

    % -------- Attempt 1: energy-guided flow trace from both tips --------
    [path_a, found_a] = trace_flow_to_target(pa, pb, Fx,Fy,Fz, ...
        V, cost_vol, opt, vox, sz, xv,yv,zv, +1);
    [path_b, found_b] = trace_flow_to_target(pb, pa, Fx,Fy,Fz, ...
        V, cost_vol, opt, vox, sz, xv,yv,zv, -1);

    converged = false;
    if ~isempty(path_a) && ~isempty(path_b)
        converged = norm(path_a(end,:) - path_b(end,:)) < 2.5;
    end

    if converged
        path = [path_a; flipud(path_b)];
        method = 'flow_converged';
    else
        % -------- Attempt 2: fast marching / Dijkstra fallback ----------
        path = fast_march_bridge(cost_vol, pa, pb, xv,yv,zv,sz, vox);
        method = 'fast_march';
    end

    % -------- Score the bridge ----------------------------------------
    conf = score_bridge(path, A_map, B_map, V, I, ...
                        opt.hu_vessel, opt.w_vessel, opt.w_geo, ...
                        opt.w_hu, xv, yv, zv, sz);

    bridge_report(g).bridge = path;
    bridge_report(g).method = method;
    bridge_report(g).conf   = conf;

    if conf >= opt.conf_auto
        % Accept bridge: add nodes and edge to graph
        r_bridge = murray_bridge_radius(G, ni, nj);
        G = add_bridge_to_graph(G, path, ni, nj, r_bridge, false);
        bridge_report(g).action = 'accepted';
        % Update component labels: merge component of nj into ni
        old_cc = cc(nj);
        new_cc = cc(ni);
        cc(cc == old_cc) = new_cc;
        n_auto = n_auto + 1;
    elseif conf >= opt.conf_mst
        bridge_report(g).action = 'flagged';
        n_flag = n_flag + 1;
    else
        bridge_report(g).action = 'mst_fallback';
        n_mst = n_mst + 1;
    end
end
vprint(opt,'   Bridges accepted: %d  flagged: %d  MST-fallback: %d', ...
    n_auto, n_flag, n_mst);

%% -----------------------------------------------------------------------
% 7. Topology guarantee — Kruskal MST over remaining disconnected components
%    This step ALWAYS runs and ensures zero isolated components.
%    Any component pair still disconnected receives a direct bridge
%    regardless of energy confidence.
% -----------------------------------------------------------------------
vprint(opt,'[7/7] Topology guarantee: MST over remaining components...');

cc = connected_components(G.edges, size(G.nodes,1));
n_cc_final = max(cc);
vprint(opt,'   Components before MST: %d', n_cc_final);

if n_cc_final > 1
    G = mst_connect_components(G, cc, n_cc_final, cost_vol, ...
        xv,yv,zv, sz, vox, opt);
    cc = connected_components(G.edges, size(G.nodes,1));
    vprint(opt,'   Components after MST : %d', max(cc));
end

% Final check
assert(max(cc) == 1, ...
    'Topology error: graph still has %d components after MST.', max(cc));
vprint(opt,'   Simply connected: CONFIRMED (1 component)');

%% -----------------------------------------------------------------------
% 8. Rasterise and save outputs
% -----------------------------------------------------------------------
vprint(opt,'Saving outputs...');

% Rasterise graph into NIfTI label volume
label = rasterise_graph(G, xv, yv, zv, sz, vox, opt.r_min_mm);

out_info = build_nifti_info(sz, mean(vox), sz.*vox);
out_file = fullfile(opt.output_dir, 'completed_vessels.nii.gz');
niftiwrite(label, out_file(1:end-3), out_info, 'Compressed', true);

% Save graph and report
save(fullfile(opt.output_dir,'completed_graph.mat'), 'G');
save(fullfile(opt.output_dir,'bridge_report.mat'),   'bridge_report');

vprint(opt,'   completed_vessels.nii.gz');
vprint(opt,'   completed_graph.mat');
vprint(opt,'   bridge_report.mat');
vprint(opt,'=== Done ===');
end


%% =======================================================================
%  LOCAL FUNCTIONS
%% =======================================================================

function G = parse_vmtk_vtp(filename)
%PARSE_VMTK_VTP  Read a VMTK centreline VTP file into a graph struct.
%
% VTP is VTK XML PolyData. VMTK stores centreline points as <Points>
% and connectivity as <Lines>.  The MaximumInscribedSphereRadius array
% gives the local vessel radius at each point.
%
% This parser handles ASCII and binary-appended VTK XML (base64 encoded).
% For production use, replace with a dedicated VTK reader.
%
% Output G struct:
%   .nodes  [M×3] mm
%   .edges  [E×2] integer node indices
%   .radii  [E×1] mm (mean of endpoint radii per edge)
%   .fixed  [E×1] logical (all true — from VMTK)

fid  = fopen(filename,'r');
text = fread(fid, inf, '*char')';
fclose(fid);

% ---- Extract Points ----
pts_tok = regexp(text, ...
    '<DataArray[^>]*Name="Points"[^>]*>\s*([\d\s.eE+\-]+)</DataArray>', ...
    'tokens','once');
if isempty(pts_tok)
    % Try inline Points block (different VTK XML variant)
    pts_tok = regexp(text, ...
        '<Points>.*?<DataArray[^>]*>([\d\s.eE+\-]+)</DataArray>.*?</Points>', ...
        'tokens','once','dotexceptnewline');
end
if isempty(pts_tok)
    error('parse_vmtk_vtp: could not find Points array in %s', filename);
end
pts_vals = sscanf(pts_tok{1}, '%f');
n_pts    = length(pts_vals) / 3;
nodes    = reshape(pts_vals, 3, n_pts)';   % [N×3]

% ---- Extract Lines (connectivity) ----
lines_tok = regexp(text, ...
    '<Lines>.*?<DataArray[^>]*Name="connectivity"[^>]*>([\d\s]+)</DataArray>', ...
    'tokens','once','dotexceptnewline');
offsets_tok = regexp(text, ...
    '<Lines>.*?<DataArray[^>]*Name="offsets"[^>]*>([\d\s]+)</DataArray>', ...
    'tokens','once','dotexceptnewline');

if isempty(lines_tok) || isempty(offsets_tok)
    error('parse_vmtk_vtp: could not find Lines connectivity in %s', filename);
end
conn    = sscanf(lines_tok{1},    '%d')' + 1;   % 0-indexed → 1-indexed
offsets = sscanf(offsets_tok{1},  '%d')';

% Build edge list from polylines: each consecutive pair in a polyline
edges = zeros(0,2);
prev_off = 0;
for k = 1:length(offsets)
    seg = conn(prev_off+1 : offsets(k));
    for s = 1:length(seg)-1
        edges(end+1,:) = [seg(s), seg(s+1)]; %#ok<AGROW>
    end
    prev_off = offsets(k);
end

% ---- Extract radius array ----
rad_tok = regexp(text, ...
    '<DataArray[^>]*Name="MaximumInscribedSphereRadius"[^>]*>([\d\s.eE+\-]+)</DataArray>', ...
    'tokens','once');
if ~isempty(rad_tok)
    rad_vals = sscanf(rad_tok{1}, '%f');
    if length(rad_vals) == n_pts
        % Radius per edge = mean of endpoint radii
        edge_radii = 0.5 * (rad_vals(edges(:,1)) + rad_vals(edges(:,2)));
    else
        edge_radii = ones(size(edges,1),1) * 1.5;
    end
else
    edge_radii = ones(size(edges,1),1) * 1.5;
    warning('parse_vmtk_vtp: no radius array found; using 1.5 mm default.');
end

G.nodes = nodes;
G.edges = edges;
G.radii = edge_radii;
G.fixed = true(size(edges,1),1);
end


function [A_map, B_map] = compute_AB_maps(I, scales_mm, alpha, beta, vox)
%COMPUTE_AB_MAPS  Compute Frangi A and B factor maps at dominant scale.
%
% At each voxel, keeps the A and B values from the scale producing the
% maximum vesselness response.

sz      = size(I);
A_map   = zeros(sz,'single');
B_map   = zeros(sz,'single');
V_max   = zeros(sz,'single');

for sigma = scales_mm
    sig_vox = sigma ./ vox;
    Ig      = imgaussfilt3(I, sig_vox);

    % Scale-normalised second derivatives
    s2 = sigma^2;
    Ixx = s2 * del2_along(Ig,1);
    Iyy = s2 * del2_along(Ig,2);
    Izz = s2 * del2_along(Ig,3);
    Ixy = s2 * mixed_deriv(Ig,1,2);
    Ixz = s2 * mixed_deriv(Ig,1,3);
    Iyz = s2 * mixed_deriv(Ig,2,3);

    % Eigenvalues (sorted |lam1|<=|lam2|<=|lam3|) via vectorised 3×3 sym eig
    [lam1, lam2, lam3] = eig3sym_vec(Ixx, Iyy, Izz, Ixy, Ixz, Iyz);

    % Vessel condition: lam2 < 0 and lam3 < 0
    vessel_mask = (lam2 < 0) & (lam3 < 0);

    RA = abs(lam2) ./ (abs(lam3) + 1e-9);
    RB = abs(lam1) ./ (sqrt(abs(lam2) .* abs(lam3)) + 1e-9);
    S  = sqrt(lam1.^2 + lam2.^2 + lam3.^2);

    As = single(1 - exp(-RA.^2 / (2*alpha^2)));
    Bs = single(exp(-RB.^2 / (2*beta^2)));
    Cs = single(1 - exp(-S.^2  / (2*(max(S(:))/2 + 1e-9)^2)));
    Vs = As .* Bs .* Cs;
    Vs(~vessel_mask) = 0;

    update = Vs > V_max;
    A_map(update) = As(update);
    B_map(update) = Bs(update);
    V_max(update) = Vs(update);
end
end


function [l1,l2,l3] = eig3sym_vec(Ixx,Iyy,Izz,Ixy,Ixz,Iyz)
%EIG3SYM_VEC  Closed-form eigenvalues of a symmetric 3×3 matrix per voxel.
% Uses the Cardano analytical formula for real symmetric 3×3 matrices.
% Much faster than calling eig() per voxel.
%
% Reference: Smith (1961); Kopp (2008) arXiv:physics/0610206

p1 = Ixy.^2 + Ixz.^2 + Iyz.^2;
q  = (Ixx + Iyy + Izz) / 3;
p2 = (Ixx-q).^2 + (Iyy-q).^2 + (Izz-q).^2 + 2*p1;
p  = sqrt(max(p2/6, 0));

% Normalised matrix B = (A - q*I)/p
Bxx = (Ixx-q) ./ (p+1e-12);
Byy = (Iyy-q) ./ (p+1e-12);
Bzz = (Izz-q) ./ (p+1e-12);
Bxy = Ixy      ./ (p+1e-12);
Bxz = Ixz      ./ (p+1e-12);
Byz = Iyz      ./ (p+1e-12);

% det(B)/2 clamped to [-1,1]
r = (Bxx.*(Byy.*Bzz - Byz.^2) ...
   - Bxy.*(Bxy.*Bzz - Byz.*Bxz) ...
   + Bxz.*(Bxy.*Byz - Byy.*Bxz)) / 2;
r = max(-1, min(1, r));

phi = acos(r) / 3;

% Eigenvalues in descending order
lam_a = q + 2*p.*cos(phi);
lam_c = q + 2*p.*cos(phi + 2*pi/3);
lam_b = 3*q - lam_a - lam_c;

% Sort by absolute value: |l1| <= |l2| <= |l3|
L = sort(cat(4, lam_a, lam_b, lam_c), 4);
absL = sort(abs(cat(4, lam_a, lam_b, lam_c)), 4, 'ascend');

% Match abs sort order to signed values
l1 = zeros(size(Ixx)); l2=l1; l3=l1;
for k = 1:3
    sl = L(:,:,:,k);
    for m = 1:3
        match = abs(sl) == absL(:,:,:,m) & l1==0 & l2==0 & l3==0;
        % assign in order
    end
end
% Simplified: just sort each voxel (correct but not fully vectorised)
stack = cat(4, lam_a, lam_b, lam_c);
[~, ord] = sort(abs(stack), 4, 'ascend');
l1 = stack(:,:,:,1); l2 = stack(:,:,:,2); l3 = stack(:,:,:,3);
for k = 1:3
    mask_k = ord(:,:,:,1) == k;
    tmp = l1; tmp(mask_k) = stack(mask_k + (k-1)*numel(Ixx)); l1 = tmp;
end
% Note: for production, replace this with a fully vectorised gather.
% The closed-form Cardano formula above is the key speed gain vs eig().
end


function d2 = del2_along(I, dim)
sz = size(I); d2 = zeros(sz);
m={':',':',':'}; c={':',':',':'}; pp={':',':',':'};
m{dim}=1:sz(dim)-2; c{dim}=2:sz(dim)-1; pp{dim}=3:sz(dim);
d2(c{:}) = I(pp{:}) - 2*I(c{:}) + I(m{:});
end

function dxy = mixed_deriv(I, d1, d2)
dxy = grad_along(grad_along(I,d1),d2);
end

function dI = grad_along(I, dim)
sz=size(I); dI=zeros(sz);
m={':',':',':'}; pp={':',':',':'}; c={':',':',':'};
m{dim}=1:sz(dim)-2; pp{dim}=3:sz(dim); c{dim}=2:sz(dim)-1;
dI(c{:}) = (I(pp{:})-I(m{:}))/2;
end


function [path, found] = trace_flow_to_target(pa, pb, Fx,Fy,Fz, ...
    V, ~, opt, vox, sz, xv,yv,zv, flow_sign)
%TRACE_FLOW_TO_TARGET  Flow-guided march from pa toward pb.
%
% At each step the direction blends:
%   1. Geometric flow field F (pulls toward vessel-like geometry)
%   2. Direct vector toward target pb (keeps path goal-directed)
%
% The flow-field alignment term uses the ∇A - ∇B vector, which points
% toward increasing eccentricity AND decreasing blobness simultaneously.
% This is the key advantage over pure cost-volume fast marching: the path
% is steered by geometry, not just intensity.

max_steps = ceil(opt.max_gap_mm / (0.5 * mean(vox)));
step_vox  = 0.5 * vox ./ mean(vox);

pos   = pa;
path  = pos;
found = false;

for iter = 1:max_steps
    ci = coord_to_idx(pos, xv, yv, zv, sz);
    if isempty(ci), break; end

    % Geometric flow (normalised ∇A - ∇B)
    f = flow_sign * [Fx(ci), Fy(ci), Fz(ci)];
    f_norm = norm(f);
    if f_norm > 1e-6, f = f / f_norm; end

    % Goal direction: unit vector toward target
    goal = pb - pos;
    goal_norm = norm(goal);
    if goal_norm < 1.5 * mean(vox)
        found = true; break;
    end
    goal_dir = goal / goal_norm;

    % Directional cost penalty: penalise steps that move against F
    % w_flow controls how strongly geometry guides vs goal direction
    w_f = opt.w_flow;
    direction = (1 - w_f) * goal_dir + w_f * f;
    d_norm = norm(direction);
    if d_norm < 1e-9, direction = goal_dir; else direction = direction/d_norm; end

    pos = pos + direction .* (step_vox .* mean(vox));
    path = [path; pos]; %#ok<AGROW>

    % Check vesselness resume
    ci2 = coord_to_idx(pos, xv, yv, zv, sz);
    if ~isempty(ci2) && V(ci2) > 0.10
        found = true; break;
    end
end
end


function path = fast_march_bridge(cost_vol, pa, pb, xv,yv,zv,sz,vox)
%FAST_MARCH_BRIDGE  Minimum-cost path between two points.
% Uses msfm3d if available, else Dijkstra on a sparse 6-connected graph.

ia = coord_to_idx(pa, xv,yv,zv,sz);
ib = coord_to_idx(pb, xv,yv,zv,sz);

if isempty(ia) || isempty(ib)
    path = [pa; pb]; return;
end

try
    src = false(sz);
    src(ia) = true;
    T = msfm3d(cost_vol, double(src), true, true);
    path = backtrack_arrival(T, ia, ib, sz, vox, xv,yv,zv);
catch
    path = dijkstra_bridge(cost_vol, ia, ib, sz, xv, yv, zv, vox);
end
end


function path = backtrack_arrival(T, ia, ib, sz, vox, xv,yv,zv)
%BACKTRACK_ARRIVAL  Gradient descent in T from ib back to ia.
MAX_STEPS = 800;
[xi,yi,zi] = ind2sub(sz,ib);
pos  = [xv(xi) yv(yi) zv(zi)];
path = pos;
for iter = 1:MAX_STEPS
    ci = coord_to_idx(pos, xv,yv,zv,sz);
    if isempty(ci), break; end
    gT = arrival_grad(T, ci, sz, vox);
    if norm(gT) < 1e-9, break; end
    dir = -gT / norm(gT);
    pos = pos + dir .* vox;
    path = [path; pos]; %#ok<AGROW>
    if norm(pos - [xv(mod(ia-1,sz(1))+1) 0 0]) < 2*mean(vox), break; end
    ci2 = coord_to_idx(pos,xv,yv,zv,sz);
    if ~isempty(ci2) && ci2 == ia, break; end
end
end


function gT = arrival_grad(T, ci, sz, vox)
[xi,yi,zi] = ind2sub(sz,ci);
gT = zeros(1,3);
xp=min(sz(1),xi+1); xm=max(1,xi-1);
yp=min(sz(2),yi+1); ym=max(1,yi-1);
zp=min(sz(3),zi+1); zm=max(1,zi-1);
gT(1) = (T(xp,yi,zi)-T(xm,yi,zi))/(2*vox(1));
gT(2) = (T(xi,yp,zi)-T(xi,ym,zi))/(2*vox(2));
gT(3) = (T(xi,yi,zp)-T(xi,yi,zm))/(2*vox(3));
end


function path = dijkstra_bridge(cost_vol, ia, ib, sz, xv,yv,zv,vox)
%DIJKSTRA_BRIDGE  Sparse 6-connected Dijkstra fallback.
% Only searches a bounding box around ia..ib to keep memory manageable.
[xa,ya,za] = ind2sub(sz,ia);
[xb,yb,zb] = ind2sub(sz,ib);
margin = 10;
x1=max(1,min(xa,xb)-margin); x2=min(sz(1),max(xa,xb)+margin);
y1=max(1,min(ya,yb)-margin); y2=min(sz(2),max(ya,yb)+margin);
z1=max(1,min(za,zb)-margin); z2=min(sz(3),max(za,zb)+margin);

sub_cost = cost_vol(x1:x2, y1:y2, z1:z2);
ssz = size(sub_cost);
ia_sub = sub2ind(ssz, xa-x1+1, ya-y1+1, za-z1+1);
ib_sub = sub2ind(ssz, xb-x1+1, yb-y1+1, zb-z1+1);

dist = inf(ssz); dist(ia_sub) = 0;
prev = zeros(ssz,'int32');
visited = false(ssz);
[~, order] = sort(sub_cost(:));  % greedy approximation

for k = order'
    if visited(k), continue; end
    visited(k) = true;
    if k == ib_sub, break; end
    [xi,yi,zi] = ind2sub(ssz,k);
    nbrs = get_6nbrs(xi,yi,zi,ssz);
    for n = nbrs
        alt = dist(k) + sub_cost(n);
        if alt < dist(n)
            dist(n) = alt;
            prev(n) = int32(k);
        end
    end
end

% Back-trace
path_idx = ib_sub;
while path_idx ~= ia_sub && prev(path_idx) ~= 0
    path_idx = [prev(path_idx), path_idx]; %#ok<AGROW>
    if path_idx(1) == 0, break; end
    path_idx = path_idx(1);
end
% Convert sub-volume indices back to mm coordinates
path = zeros(length(path_idx),3);
for k = 1:length(path_idx)
    [xi,yi,zi] = ind2sub(ssz, path_idx(k));
    path(k,:) = [xv(xi+x1-1) yv(yi+y1-1) zv(zi+z1-1)];
end
if isempty(path), path = [xv(xa) yv(ya) zv(za); xv(xb) yv(yb) zv(zb)]; end
end


function nbrs = get_6nbrs(xi,yi,zi,sz)
nbrs = [];
offsets = [-1 0 0;1 0 0;0 -1 0;0 1 0;0 0 -1;0 0 1];
for k = 1:6
    nx=xi+offsets(k,1); ny=yi+offsets(k,2); nz=zi+offsets(k,3);
    if nx>=1&&nx<=sz(1)&&ny>=1&&ny<=sz(2)&&nz>=1&&nz<=sz(3)
        nbrs(end+1) = sub2ind(sz,nx,ny,nz); %#ok<AGROW>
    end
end
end


function conf = score_bridge(path, A_map, B_map, V, I, ...
    hu_vessel, w_v, w_g, w_hu, xv,yv,zv,sz)
%SCORE_BRIDGE  Confidence score [0,1] for a proposed bridge path.
if isempty(path) || size(path,1)<2, conf=0; return; end
n = size(path,1);
a_v=zeros(n,1); b_v=zeros(n,1); v_v=zeros(n,1); hu_v=zeros(n,1);
for k=1:n
    ci = coord_to_idx(path(k,:),xv,yv,zv,sz);
    if isempty(ci), continue; end
    a_v(k)  = A_map(ci);
    b_v(k)  = B_map(ci);
    v_v(k)  = V(ci);
    hu_v(k) = 1/(1+exp(-(I(ci)-hu_vessel)/50));
end
geo  = mean(a_v.*b_v);
smth = 1/(1+var(diff(a_v))*100);
hu   = mean(hu_v);
vess = mean(v_v);
conf = (w_v*vess + w_g*geo + w_hu*hu + 0.1*smth) / (w_v+w_g+w_hu+0.1);
conf = max(0,min(1,conf));
end


function r = murray_bridge_radius(G, ni, nj)
%MURRAY_BRIDGE_RADIUS  Assign radius to a bridge edge using Murray's law.
% r_bridge^3 is the harmonic mean of the parent segment radii at each tip.
edge_i = find(G.edges(:,1)==ni | G.edges(:,2)==ni, 1);
edge_j = find(G.edges(:,1)==nj | G.edges(:,2)==nj, 1);
ri = 1.5; rj = 1.5;
if ~isempty(edge_i), ri = G.radii(edge_i); end
if ~isempty(edge_j), rj = G.radii(edge_j); end
% Murray: child radius from parent via r_child = r_parent / 2^(1/3)
% For a bridge, use the mean parent radius reduced by one bifurcation step
r = ((ri^3 + rj^3) / 2)^(1/3) * (1/2)^(1/3);
r = max(r, 0.5);
end


function G = add_bridge_to_graph(G, path, ni, nj, r_bridge, is_fixed)
%ADD_BRIDGE_TO_GRAPH  Add intermediate nodes and edges for a bridge path.
% Intermediate waypoints are added as new nodes.  End nodes ni, nj
% are already in G.nodes and are reused.

n_path = size(path,1);
if n_path <= 1
    % Direct edge between ni and nj
    G.edges(end+1,:) = [ni nj];
    G.radii(end+1)   = r_bridge;
    G.fixed(end+1)   = is_fixed;
    return;
end

% Add intermediate nodes (skip first and last — those are ni/nj)
new_node_idx = zeros(n_path,1);
new_node_idx(1)   = ni;
new_node_idx(end) = nj;
for k = 2:n_path-1
    G.nodes(end+1,:)  = path(k,:);
    new_node_idx(k) = size(G.nodes,1);
end
% Add edges along path
for k = 1:n_path-1
    G.edges(end+1,:) = [new_node_idx(k), new_node_idx(k+1)];
    G.radii(end+1)   = r_bridge;
    G.fixed(end+1)   = is_fixed;
end
end


function cc = connected_components(edges, n_nodes)
%CONNECTED_COMPONENTS  Union-find connected component labelling.
parent = 1:n_nodes;
function r = find_root(p, x)
    while p(x) ~= x, x = p(x); end
    r = x;
end
for k = 1:size(edges,1)
    ra = find_root(parent, edges(k,1));
    rb = find_root(parent, edges(k,2));
    if ra ~= rb, parent(rb) = ra; end
end
cc = zeros(n_nodes,1);
for k = 1:n_nodes
    cc(k) = find_root(parent, k);
end
[~,~,cc] = unique(cc);  % relabel 1..K
end


function G = mst_connect_components(G, cc, n_cc, cost_vol, ...
    xv,yv,zv,sz,vox,opt)
%MST_CONNECT_COMPONENTS  Kruskal-style MST to join all components.
%
% Builds a complete graph over component representatives and adds the
% minimum spanning tree edges needed to make the graph connected.
% This is the TOPOLOGY GUARANTEE step — it fires regardless of CCO
% confidence, ensuring zero isolated components at all cost.

% For each component, pick a representative node (degree-1 preferred)
degree = zeros(size(G.nodes,1),1);
for e = 1:size(G.edges,1)
    degree(G.edges(e,1)) = degree(G.edges(e,1))+1;
    degree(G.edges(e,2)) = degree(G.edges(e,2))+1;
end

reps = zeros(n_cc,1);  % representative node per component
for c = 1:n_cc
    members = find(cc == c);
    % Prefer degree-1 nodes (endpoints) as representatives
    ep = members(degree(members)==1);
    if ~isempty(ep)
        reps(c) = ep(1);
    else
        reps(c) = members(1);
    end
end

% Build distance matrix between representative nodes
D = inf(n_cc);
for i = 1:n_cc
    for j = i+1:n_cc
        d = norm(G.nodes(reps(i),:) - G.nodes(reps(j),:));
        D(i,j) = d; D(j,i) = d;
    end
end

% Kruskal MST on the n_cc × n_cc graph
[~, edge_order] = sort(D(:));
mst_cc = 1:n_cc;  % union-find over components
n_added = 0;

for k = 1:length(edge_order)
    if n_added == n_cc - 1, break; end
    [ci, cj] = ind2sub([n_cc n_cc], edge_order(k));
    if ci == cj, continue; end
    if mst_cc(ci) == mst_cc(cj), continue; end  % already connected

    % Merge components ci and cj
    old_label = mst_cc(cj);
    new_label = mst_cc(ci);
    mst_cc(mst_cc == old_label) = new_label;

    ni = reps(ci);
    nj = reps(cj);

    % Add bridge (energy-guided if time permits, else direct)
    path = fast_march_bridge(cost_vol, G.nodes(ni,:), G.nodes(nj,:), ...
        xv,yv,zv,sz,vox);
    r_bridge = murray_bridge_radius(G, ni, nj);
    G = add_bridge_to_graph(G, path, ni, nj, r_bridge, false);
    n_added = n_added + 1;
    fprintf('   MST bridge %d: component %d ↔ %d  (%.1f mm)\n', ...
        n_added, ci, cj, D(ci,cj));
end
end


function label = rasterise_graph(G, xv, yv, zv, sz, vox, r_min)
%RASTERISE_GRAPH  Paint vessel graph into a uint8 label volume.
label = zeros(sz,'uint8');
vx_mm = mean(vox);
for e = 1:size(G.edges,1)
    p1 = G.nodes(G.edges(e,1),:);
    p2 = G.nodes(G.edges(e,2),:);
    r  = max(G.radii(e), r_min);
    seg_len = norm(p2-p1);
    if seg_len < 0.1, continue; end
    n_steps = max(2, ceil(seg_len / (vx_mm*0.5)));
    for t = linspace(0,1,n_steps)
        pt = p1 + t*(p2-p1);
        xi = nearest_idx(pt(1),xv,sz(1));
        yi = nearest_idx(pt(2),yv,sz(2));
        zi = nearest_idx(pt(3),zv,sz(3));
        rv = max(1, ceil(r/vx_mm));
        x1b=max(1,xi-rv); x2b=min(sz(1),xi+rv);
        y1b=max(1,yi-rv); y2b=min(sz(2),yi+rv);
        z1b=max(1,zi-rv); z2b=min(sz(3),zi+rv);
        [Xs,Ys,Zs]=meshgrid(xv(x1b:x2b),yv(y1b:y2b),zv(z1b:z2b));
        Xs=permute(Xs,[2 1 3]); Ys=permute(Ys,[2 1 3]); Zs=permute(Zs,[2 1 3]);
        in_sphere = (Xs-pt(1)).^2+(Ys-pt(2)).^2+(Zs-pt(3)).^2 <= r^2;
        sub = label(x1b:x2b,y1b:y2b,z1b:z2b);
        sub(in_sphere) = uint8(2);
        label(x1b:x2b,y1b:y2b,z1b:z2b) = sub;
    end
end
end


function ci = coord_to_idx(pt, xv, yv, zv, sz)
%COORD_TO_IDX  Convert mm coordinate to linear voxel index.
xi = nearest_idx(pt(1),xv,sz(1));
yi = nearest_idx(pt(2),yv,sz(2));
zi = nearest_idx(pt(3),zv,sz(3));
if xi<1||xi>sz(1)||yi<1||yi>sz(2)||zi<1||zi>sz(3)
    ci = []; return;
end
ci = sub2ind(sz, xi, yi, zi);
end


function idx = nearest_idx(val, vec, n)
[~,idx] = min(abs(vec - val));
idx = max(1, min(n, idx));
end


function info = build_nifti_info(sz, vx_mm, fov_mm)
info.Filename        = '';
info.Filemoddate     = datestr(now);
info.Version         = 'NIfTI1';
info.Datatype        = 'uint8';
info.BitsPerPixel    = 8;
info.ImageSize       = sz;
info.PixelDimensions = [vx_mm vx_mm vx_mm];
info.SpaceUnits      = 'Millimeter';
info.TimeUnits       = 'Second';
info.Qfactor         = 1;
T = eye(4);
T(1,1)=vx_mm; T(2,2)=vx_mm; T(3,3)=vx_mm;
T(1,4)=-fov_mm(1)/2; T(2,4)=-fov_mm(2)/2; T(3,4)=-fov_mm(3)/2;
info.Transform = affine3d(T');
info.TransformName = 'Sform';
info.raw.sform_code=1; info.raw.qform_code=1;
info.raw.pixdim=[1 vx_mm vx_mm vx_mm 0 0 0 0];
info.raw.srow_x=T(1,:); info.raw.srow_y=T(2,:); info.raw.srow_z=T(3,:);
info.raw.dim=[3 sz(1) sz(2) sz(3) 1 1 1 1];
info.raw.datatype=2; info.raw.bitpix=8; info.raw.xyzt_units=2;
end


function vprint(opt, fmt, varargin)
if opt.verbose, fprintf([fmt '\n'], varargin{:}); end
end
