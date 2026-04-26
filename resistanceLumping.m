%% resistanceLumping.m
%
% Hagen-Poiseuille resistance network on a vessel skeleton with
% phantom edges bridging disconnected components (resistance lumping).
%
% APPROACH:
%   Every detected gap between two endpoints from different connected
%   components is assigned a phantom edge with conductance:
%
%     G_real    = pi * r^4      / (8 * mu * L)
%     G_phantom = pi * r_mean^4 / (8 * mu * alpha * L_gap)
%
%   The global conductance matrix K is assembled (sparse, symmetric,
%   positive-definite), boundary pressures are applied at inlet/outlet
%   leaf nodes, and the linear system K*p = q is solved for nodal
%   pressures.  Flow rates follow from Q_ij = G_ij*(p_i - p_j).
%
% INPUTS:
%   skel_nii  — binary skeleton NIfTI (output of myskelotonize.m)
%   label_nii — label NIfTI used to estimate vessel radii via distance
%               transform of voxels where label == label_val (default 2)
%
% PARAMETERS (name-value):
%   alpha        gap penalty factor (>1, typically 5–20)  default 10
%   mu_pas       dynamic viscosity, Pa·s (blood ~3.5e-3)  default 3.5e-3
%   p_in_mmhg    inlet pressure  (mmHg)                   default 100
%   p_out_mmhg   outlet pressure (mmHg)                   default 5
%   gap_max_mm   max gap distance for phantom edges (mm)  default 15
%   label_val    vessel label value for radius estimation  default 2
%   output_dir   directory for output files               default './'
%   verbose      print progress                           default true
%
% OUTPUTS (written to output_dir):
%   pressure_mmhg.nii.gz   — nodal pressure mapped to skeleton voxels
%   resistance_graph.mat   — full results struct
%
% USAGE:
%   R = resistanceLumping('skeleton.nii.gz','label.nii.gz')
%   R = resistanceLumping(...,'alpha',15,'p_in_mmhg',120,'gap_max_mm',20)

function results = resistanceLumping(skel_nii, label_nii, varargin)

%% ── Parameters ──────────────────────────────────────────────────────────
p = inputParser;
addRequired(p,  'skel_nii');
addRequired(p,  'label_nii');
addParameter(p, 'alpha',       10);
addParameter(p, 'mu_pas',      3.5e-3);
addParameter(p, 'p_in_mmhg',   100);
addParameter(p, 'p_out_mmhg',  5);
addParameter(p, 'gap_max_mm',  15);
addParameter(p, 'label_val',   2);
addParameter(p, 'output_dir',  './');
addParameter(p, 'verbose',     true);
parse(p, skel_nii, label_nii, varargin{:});
opt = p.Results;

if ~exist(opt.output_dir,'dir'), mkdir(opt.output_dir); end
vp = @(fmt,varargin) deal(opt.verbose && fprintf([fmt '\n'], varargin{:}));

vp('=== Resistance lumping ===');
vp('  skeleton  : %s', skel_nii);
vp('  label     : %s', label_nii);
vp('  alpha=%.0f  mu=%.2e Pa·s  p_in=%.0f  p_out=%.0f mmHg  gap_max=%.0f mm', ...
    opt.alpha, opt.mu_pas, opt.p_in_mmhg, opt.p_out_mmhg, opt.gap_max_mm);

%% ── 1. Load volumes ──────────────────────────────────────────────────────
vp('[1/6] Loading volumes...');

info_skel = niftiinfo(skel_nii);
skel      = logical(niftiread(skel_nii));
label_vol = niftiread(label_nii);

sz  = size(skel);

% MATLAB affine3d convention: [x y z 1] = [i j k 1] * A  (1-indexed i,j,k)
% xv(n) = world x of MATLAB row-index n  →  matches NIfTI world coords exactly.
% vox uses abs() for distance-transform scaling (always positive).
A   = info_skel.Transform.T;
vox = abs([A(1,1), A(2,2), A(3,3)]);
xv  = A(1,1)*(1:sz(1)) + A(4,1);
yv  = A(2,2)*(1:sz(2)) + A(4,2);
zv  = A(3,3)*(1:sz(3)) + A(4,3);

% Radius map from distance transform of vessel binary mask
binary = label_vol == opt.label_val;
dt_mm  = bwdist(~binary) .* mean(vox);   % local radius in mm
vp('   Skeleton voxels : %d  |  vessel voxels : %d', sum(skel(:)), sum(binary(:)));

%% ── 2. Extract skeleton graph ────────────────────────────────────────────
vp('[2/6] Extracting skeleton graph...');

[nodes, edges, radii, seg_vox, node_ijk0] = extract_graph_from_skel(skel, dt_mm, vox, xv, yv, zv, sz);
n_nodes = size(nodes, 1);
n_edges = size(edges, 1);
vp('   Nodes : %d  |  Edges : %d', n_nodes, n_edges);

if n_edges == 0
    warning('resistanceLumping: skeleton has no edges — cannot build network.');
    results = struct(); return;
end
vp('   Radius range : %.2f – %.2f mm', min(radii), max(radii));

%% ── 3. Edge lengths and real conductances (Hagen-Poiseuille) ─────────────
vp('[3/6] Computing conductances...');

% Path lengths along skeleton (mm) — more accurate than Euclidean
lengths_mm = zeros(n_edges, 1);
for e = 1:n_edges
    vx_path = seg_vox{e};
    if numel(vx_path) >= 2
        [px,py,pz] = ind2sub(sz, vx_path(:));
        coords     = [xv(px)', yv(py)', zv(pz)'];
        lengths_mm(e) = sum(sqrt(sum(diff(coords,1,1).^2, 2)));
    else
        lengths_mm(e) = norm(nodes(edges(e,1),:) - nodes(edges(e,2),:));
    end
    lengths_mm(e) = max(lengths_mm(e), 1e-3);
end

% Conductance in SI: r (m), L (m) → G (m³ Pa⁻¹ s⁻¹)
r_m  = radii(:)      * 1e-3;   % force column to prevent broadcast expansion
L_m  = lengths_mm(:) * 1e-3;
G_real = (pi .* r_m.^4) ./ (8 * opt.mu_pas .* L_m);

vp('   Resistance range (real) : %.2e – %.2e Pa·s/m³', ...
    min(1./G_real), max(1./G_real));

%% ── 4. Phantom edges across gaps ─────────────────────────────────────────
vp('[4/6] Identifying gaps and building phantom edges...');

cc     = connected_components(edges, n_nodes);
n_cc   = max(cc);
vp('   Connected components : %d', n_cc);

degree = accumarray(edges(:), 1, [n_nodes 1]);   % degree of each node
ep_idx = find(degree == 1);
vp('   Leaf endpoints       : %d', numel(ep_idx));

% Representative edge radius for each endpoint
ep_r = arrayfun(@(n) endpoint_radius(edges, radii, n), ep_idx);

% Build phantom edge list — inter-component pairs within gap_max_mm
ph_ni  = [];  ph_nj  = [];
ph_r   = [];  ph_L   = [];

for i = 1:numel(ep_idx)
    for j = i+1 : numel(ep_idx)
        ni = ep_idx(i);  nj = ep_idx(j);
        if cc(ni) == cc(nj), continue; end
        d = norm(nodes(ni,:) - nodes(nj,:));
        if d > opt.gap_max_mm, continue; end
        ph_ni(end+1) = ni;         %#ok<AGROW>
        ph_nj(end+1) = nj;         %#ok<AGROW>
        ph_r(end+1)  = (ep_r(i) + ep_r(j)) / 2;   %#ok<AGROW>
        ph_L(end+1)  = max(d, 1e-3);               %#ok<AGROW>
    end
end
n_phantom = numel(ph_ni);
vp('   Phantom edges         : %d  (alpha=%.0f)', n_phantom, opt.alpha);

% Phantom conductances
if n_phantom > 0
    rp_m = ph_r(:) * 1e-3;
    Lp_m = ph_L(:) * 1e-3;
    G_phantom = (pi .* rp_m.^4) ./ (8 * opt.mu_pas * opt.alpha .* Lp_m);
    vp('   Resistance range (phantom) : %.2e – %.2e Pa·s/m³', ...
        min(1./G_phantom), max(1./G_phantom));
else
    G_phantom = zeros(0,1);
end

%% ── 5. Assemble K, apply BCs, solve K·p = q ──────────────────────────────
vp('[5/6] Assembling K and solving...');

all_e1 = [edges(:,1);   ph_ni(:)];
all_e2 = [edges(:,2);   ph_nj(:)];
all_G  = [G_real(:);    G_phantom(:)];

% Sparse assembly: K(i,i) += G, K(j,j) += G, K(i,j) -= G, K(j,i) -= G
ii = [all_e1; all_e2; all_e1; all_e2];
jj = [all_e1; all_e2; all_e2; all_e1];
vv = [all_G;  all_G; -all_G; -all_G];
K  = sparse(ii, jj, vv, n_nodes, n_nodes);

% Boundary conditions — auto-detect:
%   inlet  = leaf node with the largest connected-edge radius
%   outlets = all remaining leaf nodes
[~, best]   = max(ep_r);
inlet_node  = ep_idx(best);
outlet_nodes = ep_idx(ep_idx ~= inlet_node);
vp('   Inlet  node : %d  (r=%.2f mm)', inlet_node, ep_r(best));
vp('   Outlet nodes: %d', numel(outlet_nodes));

p_in_pa   = opt.p_in_mmhg  * 133.322;   % mmHg → Pa
p_out_pa  = opt.p_out_mmhg * 133.322;
p_mean_pa = (p_in_pa + p_out_pa) / 2;

% Dirichlet BC: row/col elimination (preserves symmetry)
q = zeros(n_nodes, 1);
bc_nodes = [inlet_node; outlet_nodes(:)];
bc_vals  = [p_in_pa;    repmat(p_out_pa, numel(outlet_nodes), 1)];

% Pin one node per floating component (no BC node reachable in combined graph)
if ~isempty(all_e1)
    cc_all = connected_components(double([all_e1(:), all_e2(:)]), n_nodes);
else
    cc_all = (1:n_nodes)';
end
bc_set  = unique(bc_nodes);
n_float = 0;
for comp_id = 1 : max(cc_all)
    comp_nodes = find(cc_all == comp_id);
    if any(ismember(comp_nodes, bc_set)), continue; end
    bc_nodes(end+1) = comp_nodes(1);   %#ok<AGROW>
    bc_vals(end+1)  = p_mean_pa;       %#ok<AGROW>
    n_float = n_float + 1;
end
if n_float > 0
    vp('   Pinned %d floating component(s) at mean pressure (%.1f mmHg)', ...
        n_float, p_mean_pa / 133.322);
end

for k = 1:numel(bc_nodes)
    d  = bc_nodes(k);
    pd = bc_vals(k);
    q  = q - K(:,d) * pd;
    K(:,d) = 0;   K(d,:) = 0;
    K(d,d) = 1;   q(d)   = pd;
end

% Regularise any remaining zero-diagonal entries (isolated degree-0 nodes)
iso_diag = find(diag(K) == 0);
for k = 1:numel(iso_diag)
    d      = iso_diag(k);
    K(d,d) = 1;
    q(d)   = p_mean_pa;
end
if ~isempty(iso_diag)
    vp('   Regularised %d isolated node(s)', numel(iso_diag));
end

pressure_pa   = K \ q;
pressure_mmhg = pressure_pa / 133.322;

vp('   Pressure range : %.1f – %.1f mmHg', ...
    min(pressure_mmhg), max(pressure_mmhg));

%% ── 6. Flows, pressure NIfTI, save ──────────────────────────────────────
vp('[6/6] Computing flows, writing outputs...');

% Q_ij = G_ij * (p_i - p_j)  [m³/s → mm³/s = × 1e9]
Q_m3s  = G_real .* (pressure_pa(edges(:,1)) - pressure_pa(edges(:,2)));
Q_mm3s = Q_m3s * 1e9;
vp('   Flow range (real edges) : %.3e – %.3e mm³/s', ...
    min(abs(Q_mm3s)), max(abs(Q_mm3s)));

% ── Map nodal pressure to skeleton voxels ────────────────────────────────
pressure_vol = zeros(sz, 'single');
for e = 1:n_edges
    ni      = edges(e,1);  nj = edges(e,2);
    vx_path = seg_vox{e};
    n_path  = numel(vx_path);
    for k = 1:n_path
        t = (k-1) / max(1, n_path-1);
        pressure_vol(vx_path(k)) = single((1-t)*pressure_mmhg(ni) + t*pressure_mmhg(nj));
    end
end

out_info              = info_skel;
out_info.Datatype     = 'single';
out_info.BitsPerPixel = 32;
out_info.Filename     = '';
pnii = fullfile(opt.output_dir, 'pressure_mmhg.nii.gz');
niftiwrite(pressure_vol, pnii(1:end-3), out_info, 'Compressed', true);
vp('   pressure_mmhg.nii.gz');

% ── Pack seg_vox as CSR for Python ───────────────────────────────────────
% seg_vox_flat : 0-indexed linear voxel indices (all paths concatenated)
% seg_vox_ptr  : CSR pointers, length n_edges+1; edge e occupies flat[ptr(e):ptr(e+1)]
% skel_size    : [nx ny nz] so Python can convert linear→(i,j,k)
if ~isempty(seg_vox)
    seg_lens     = int32(cellfun(@numel, seg_vox(:)));
    seg_vox_flat = int32(vertcat(seg_vox{:})) - 1;   % 0-indexed
    seg_vox_ptr  = int32([0; cumsum(seg_lens)]);
else
    seg_vox_flat = int32(zeros(0, 1));
    seg_vox_ptr  = int32(zeros(1, 1));
end
skel_size = int32(sz(:)');

% ── Save MAT ─────────────────────────────────────────────────────────────
results = struct( ...
    'nodes',             nodes,          ...
    'node_ijk0',         node_ijk0,      ...
    'skel_size',         skel_size,      ...
    'seg_vox_flat',      seg_vox_flat,   ...
    'seg_vox_ptr',       seg_vox_ptr,    ...
    'edges',             edges,          ...
    'radii_mm',          radii,          ...
    'lengths_mm',        lengths_mm,     ...
    'conductances_SI',   G_real,         ...
    'pressure_pa',       pressure_pa,    ...
    'pressure_mmhg',     pressure_mmhg,  ...
    'flow_mm3s',         Q_mm3s,         ...
    'phantom_ni',        ph_ni(:),       ...
    'phantom_nj',        ph_nj(:),       ...
    'phantom_radii_mm',  ph_r(:),        ...
    'phantom_lengths_mm',ph_L(:),        ...
    'alpha',             opt.alpha,      ...
    'mu_pas',            opt.mu_pas,     ...
    'inlet_node',        inlet_node,     ...
    'outlet_nodes',      outlet_nodes);

save(fullfile(opt.output_dir, 'resistance_graph.mat'), 'results');
vp('   resistance_graph.mat');
vp('=== Done ===');
end


%% =======================================================================
%  LOCAL FUNCTIONS
%% =======================================================================

% -------------------------------------------------------------------------
function [nodes, edges, radii, seg_vox, node_ijk0] = ...
        extract_graph_from_skel(skel, dt_mm, vox, xv, yv, zv, sz)
% Extract node/edge graph from a binary skeleton with radius from dt_mm.
% seg_vox{e} = linear voxel indices along the skeleton path for edge e.

% 1. Count 26-connected neighbours using zero-padding (correct at borders)
kern = ones(3,3,3,'double');  kern(2,2,2) = 0;
nbr_count = round(imfilter(single(skel), kern, 0));
nbr_count(~skel) = 0;

% 2. Critical voxels
ep_mask   = skel & (nbr_count == 1);
jn_mask   = skel & (nbr_count >= 3);
iso_mask  = skel & (nbr_count == 0);
crit_mask = ep_mask | jn_mask | iso_mask;

crit_idx = find(crit_mask);
n_nodes  = numel(crit_idx);

if n_nodes == 0
    skel_idx  = find(skel);
    [~, best] = max(dt_mm(skel_idx));
    crit_idx  = skel_idx(best);
    n_nodes   = 1;
    crit_mask = false(sz);
    crit_mask(crit_idx) = true;
end

[cx,cy,cz] = ind2sub(sz, crit_idx(:));
nodes     = [reshape(xv(cx(:)),[],1), reshape(yv(cy(:)),[],1), reshape(zv(cz(:)),[],1)];
node_ijk0 = int32([cx(:)-1, cy(:)-1, cz(:)-1]);   % 0-indexed voxel coords for nibabel

fprintf('   Critical voxels: %d endpoints, %d junctions\n', ...
    sum(ep_mask(:)), sum(jn_mask(:)));

node_map = zeros(sz, 'int32');
for k = 1:n_nodes
    node_map(crit_idx(k)) = int32(k);
end

offsets = precompute_26_offsets(sz);
n_off   = numel(offsets);

visited = ~skel;
visited(crit_mask) = true;

edges    = zeros(0, 2, 'int32');
e_radii  = zeros(0, 1, 'single');
seg_vox  = {};

for ni = 1:n_nodes
    si = crit_idx(ni);

    % Walk from each unvisited skeleton neighbour
    for d = 1:n_off
        nbr = int32(si) + offsets(d);
        if nbr < 1 || nbr > numel(skel), continue; end
        if visited(nbr), continue; end

        path_lin = [si; nbr];
        path_r   = single([dt_mm(si); dt_mm(nbr)]);
        current  = nbr;
        visited(current) = true;

        for step = 1:20000
            end_node = int32(0);
            next_vox = int32(0);
            for d2 = 1:n_off
                nc = int32(current) + offsets(d2);
                if nc < 1 || nc > numel(skel), continue; end
                if ~skel(nc), continue; end
                if crit_mask(nc)
                    end_node = node_map(nc); break;
                elseif ~visited(nc)
                    next_vox = nc;
                end
            end

            if end_node > 0
                nj = end_node;
                if nj ~= ni
                    edges(end+1,:)   = int32([ni, nj]);   %#ok<AGROW>
                    e_radii(end+1)   = mean(path_r);       %#ok<AGROW>
                    seg_vox{end+1}   = path_lin;           %#ok<AGROW>
                end
                break;
            elseif next_vox > 0
                current = next_vox;
                visited(current) = true;
                path_lin(end+1) = current;              %#ok<AGROW>
                path_r(end+1)   = single(dt_mm(current)); %#ok<AGROW>
            else
                break;
            end
        end
    end

    % Direct critical–critical edges
    for d = 1:n_off
        nbr = int32(si) + offsets(d);
        if nbr < 1 || nbr > numel(skel), continue; end
        if ~crit_mask(nbr), continue; end
        nj = node_map(nbr);
        if nj > ni
            edges(end+1,:) = int32([ni, nj]);              %#ok<AGROW>
            e_radii(end+1) = single((dt_mm(si)+dt_mm(nbr))/2); %#ok<AGROW>
            seg_vox{end+1} = [si; nbr];                    %#ok<AGROW>
        end
    end
end

% Deduplicate
if ~isempty(edges)
    [~, ia] = unique(sort(double(edges),2), 'rows', 'stable');
    edges   = double(edges(ia,:));
    e_radii = e_radii(ia);
    seg_vox = seg_vox(ia);
else
    edges   = zeros(0,2);
    e_radii = zeros(0,1,'single');
    seg_vox = {};
end

radii = max(double(e_radii(:)), 0.1);   % (:) ensures column regardless of accumulation order
end


% -------------------------------------------------------------------------
function offsets = precompute_26_offsets(sz)
offsets = zeros(26,1,'int32');
k = 0;
for dz = -1:1; for dy = -1:1; for dx = -1:1
    if dx==0 && dy==0 && dz==0, continue; end
    k = k+1;
    offsets(k) = int32(dx + dy*sz(1) + dz*sz(1)*sz(2));
end; end; end
offsets = offsets(1:k);
end


% -------------------------------------------------------------------------
function cc = connected_components(edges, n_nodes)
parent = 1:n_nodes;
    function r = find_root(p,x)
        while p(x)~=x, x=p(x); end; r=x;
    end
for k = 1:size(edges,1)
    ra = find_root(parent,edges(k,1)); rb = find_root(parent,edges(k,2));
    if ra~=rb, parent(rb)=ra; end
end
cc = zeros(n_nodes,1);
for k = 1:n_nodes, cc(k) = find_root(parent,k); end
[~,~,cc] = unique(cc);
end


% -------------------------------------------------------------------------
function r = endpoint_radius(edges, radii, node_idx)
% Radius of the first edge connected to node_idx, or 1.5 mm if isolated.
ei = find(edges(:,1)==node_idx | edges(:,2)==node_idx, 1);
if ~isempty(ei), r = radii(ei); else, r = 1.5; end
end


