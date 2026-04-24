%% vessel_frangi_completion.m
%
% Energy-guided constrained CCO vessel completion from a Frangi
% vesselness image — no pre-computed centreline required.
%
% PIPELINE:
%   1. Load Frangi vesselness and CT image.
%   2. Threshold Frangi at frangi_threshold → binary mask.
%      Skeletonise (bwskel) and extract a node/edge graph with radii
%      from the distance transform.  All extracted segments are FIXED.
%   3. Compute Hessian eigenvalue maps; build factors A (plate suppressor)
%      and B (blob suppressor) and their spatial gradients.
%   4. Fuse CT intensity, Frangi vesselness, and A×B geometry into a
%      scalar cost volume C(x) ∈ [0,1] (low = vessel-like).
%   5. Identify all degree-1 skeleton endpoints (break tips).
%   6. Constrained CCO bridging: energy-guided fast marching + Murray
%      radius assignment; fixed segments never modified.
%   7. Kruskal MST topology guarantee: zero isolated components.
%   8. Rasterise and write outputs.
%
% INPUTS:
%   frangi_file  — Frangi vesselness map  (NIfTI .nii/.nii.gz)
%   image_file   — CT volume              (NIfTI .nii/.nii.gz)
%
% OUTPUTS (written to output_dir):
%   completed_vessels.nii.gz  — rasterised label volume (uint8, label=2)
%   completed_graph.mat       — struct: .nodes .edges .radii .fixed
%   bridge_report.mat         — per-bridge confidence and method
%
% USAGE:
%   vessel_frangi_completion('vesselness.nii.gz','ct.nii.gz')
%   vessel_frangi_completion(...,'frangi_threshold',0.44,'max_gap_mm',15)
%
% PARAMETERS (name-value, all optional):
%   frangi_threshold  binarisation threshold on vesselness     default 0.44
%   alpha             Frangi alpha (eccentricity)              default 0.5
%   beta              Frangi beta  (blobness)                  default 0.5
%   scales_mm         Hessian scales (mm)                      default [0.5 1 1.5 2]
%   max_gap_mm        maximum inter-component gap to bridge    default 10.0
%   w_vessel          cost weight: Frangi vesselness           default 0.35
%   w_geo             cost weight: geometry A×B               default 0.30
%   w_hu              cost weight: HU plausibility            default 0.20
%   w_flow            cost weight: flow-field alignment        default 0.15
%   hu_vessel         expected artery HU                       default 250
%   r_min_mm          minimum bridge radius (mm)               default 0.5
%   conf_auto         confidence threshold → auto-accept       default 0.50
%   conf_mst          confidence threshold → MST fallback      default 0.25
%   voxel_out_mm      output raster voxel size (mm)            default 1.0
%   output_dir        output directory                         default './'
%   seed              RNG seed                                 default 42
%   verbose           print progress                           default true

function vessel_frangi_completion(frangi_file, image_file, varargin)

%% -----------------------------------------------------------------------
% 0. Parameters
% -----------------------------------------------------------------------
p = inputParser;
addRequired(p,'frangi_file');
addRequired(p,'image_file');
addParameter(p,'frangi_threshold', 0.44);
addParameter(p,'min_island_vox',  10);   % drop CC with fewer voxels than this
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
parse(p, frangi_file, image_file, varargin{:});
opt = p.Results;

rng(opt.seed);
if ~exist(opt.output_dir,'dir'), mkdir(opt.output_dir); end
vprint(opt,'=== Frangi-CCO vessel completion ===');

%% -----------------------------------------------------------------------
% 1. Load volumes
% -----------------------------------------------------------------------
vprint(opt,'[1/7] Loading volumes...');

info = niftiinfo(image_file);
I    = double(niftiread(image_file));
V    = double(niftiread(frangi_file));

vox  = abs(diag(info.Transform.T(1:3,1:3)))';   % [dx dy dz] mm
sz   = size(I);

xv = ((0:sz(1)-1) - sz(1)/2) * vox(1);
yv = ((0:sz(2)-1) - sz(2)/2) * vox(2);
zv = ((0:sz(3)-1) - sz(3)/2) * vox(3);

vprint(opt,'   Grid: %d×%d×%d  voxel %.2f×%.2f×%.2f mm', ...
    sz(1),sz(2),sz(3), vox(1),vox(2),vox(3));
vprint(opt,'   Vesselness range: [%.4f, %.4f]', min(V(:)), max(V(:)));

%% -----------------------------------------------------------------------
% 2. Threshold Frangi → binary mask → skeleton → graph
% -----------------------------------------------------------------------
vprint(opt,'[2/7] Extracting vessel graph from Frangi (threshold=%.2f)...', ...
    opt.frangi_threshold);

G = extract_graph_from_frangi(V, opt.frangi_threshold, opt.min_island_vox, vox, xv, yv, zv, sz);

vprint(opt,'   Skeleton nodes : %d', size(G.nodes,1));
vprint(opt,'   Skeleton edges : %d', size(G.edges,1));
vprint(opt,'   Radius range   : %.2f – %.2f mm', ...
    min(G.radii), max(G.radii));

%% -----------------------------------------------------------------------
% 3. Compute A, B factor maps and flow field
% -----------------------------------------------------------------------
vprint(opt,'[3/7] Computing A, B factor maps and flow field...');

[A_map, B_map] = compute_AB_maps(I, opt.scales_mm, opt.alpha, opt.beta, vox);

[dAx,dAy,dAz] = gradient(A_map, vox(1), vox(2), vox(3));
[dBx,dBy,dBz] = gradient(B_map, vox(1), vox(2), vox(3));

% Geometric flow: ∇A − ∇B  (toward high eccentricity, low blobness)
Fx = dAx - dBx;   Fy = dAy - dBy;   Fz = dAz - dBz;
Fnorm = sqrt(Fx.^2 + Fy.^2 + Fz.^2) + 1e-9;
Fx = Fx./Fnorm;   Fy = Fy./Fnorm;   Fz = Fz./Fnorm;
clear Fnorm dAx dAy dAz dBx dBy dBz;
vprint(opt,'   A/B maps complete.');

%% -----------------------------------------------------------------------
% 4. Build fused cost volume
% -----------------------------------------------------------------------
vprint(opt,'[4/7] Building fused energy cost volume...');

hu_norm = 1 ./ (1 + exp(-(I - opt.hu_vessel) / 50));
V_geo   = A_map .* B_map;

score = (opt.w_vessel * V      + ...
         opt.w_geo    * V_geo  + ...
         opt.w_hu     * hu_norm);
score    = score / (opt.w_vessel + opt.w_geo + opt.w_hu);
score    = min(max(score, 0), 1);
cost_vol = max(1 - score, 1e-3);

clear hu_norm score;
vprint(opt,'   Cost volume built.');

%% -----------------------------------------------------------------------
% 5. Identify endpoints (degree-1 nodes) and connected components
% -----------------------------------------------------------------------
vprint(opt,'[5/7] Finding break tips and gap candidates...');

n_nodes = size(G.nodes,1);

% Drop any edges whose node indices exceed the node count (guards against
% the closed-loop fallback producing a mis-indexed crit_idx).
if ~isempty(G.edges)
    valid_e  = all(G.edges >= 1 & G.edges <= n_nodes, 2);
    n_pruned = sum(~valid_e);
    if n_pruned > 0
        warning('vessel_frangi_completion:badEdges', ...
            'Pruned %d edges with out-of-range node indices (n_nodes=%d).', ...
            n_pruned, n_nodes);
        G.edges = G.edges(valid_e, :);
        G.radii = G.radii(valid_e);
        G.fixed = G.fixed(valid_e);
    end
end

degree  = zeros(n_nodes,1);
for e = 1:size(G.edges,1)
    degree(G.edges(e,1)) = degree(G.edges(e,1)) + 1;
    degree(G.edges(e,2)) = degree(G.edges(e,2)) + 1;
end

endpoint_idx = find(degree == 1);
vprint(opt,'   Degree-1 endpoints : %d', numel(endpoint_idx));

cc     = connected_components(G.edges, n_nodes);
n_cc   = max(cc);
vprint(opt,'   Connected components: %d', n_cc);

% Build inter-component gap candidates within max_gap_mm
n_ep  = numel(endpoint_idx);
gap_candidates = struct('i',{},'j',{},'dist_mm',{},'bridge',{},...
    'method',{},'conf',{},'action',{});
n_gaps = 0;

for i = 1:n_ep
    for j = i+1:n_ep
        ni = endpoint_idx(i);
        nj = endpoint_idx(j);
        if cc(ni) == cc(nj), continue; end
        d = norm(G.nodes(ni,:) - G.nodes(nj,:));
        if d > opt.max_gap_mm, continue; end
        n_gaps = n_gaps + 1;
        gap_candidates(n_gaps).i       = ni;
        gap_candidates(n_gaps).j       = nj;
        gap_candidates(n_gaps).dist_mm = d;
    end
end
vprint(opt,'   Gap candidates (≤%.0f mm, inter-component): %d', ...
    opt.max_gap_mm, n_gaps);

%% -----------------------------------------------------------------------
% 6. Constrained CCO: energy-guided bridging
% -----------------------------------------------------------------------
vprint(opt,'[6/7] Running constrained CCO bridging...');

bridge_report = gap_candidates;
n_auto = 0; n_flag = 0; n_mst = 0;

for g = 1:n_gaps
    ni = gap_candidates(g).i;
    nj = gap_candidates(g).j;
    pa = G.nodes(ni,:);
    pb = G.nodes(nj,:);

    [path_a, ~] = trace_flow_to_target(pa, pb, Fx,Fy,Fz, ...
        V, cost_vol, opt, vox, sz, xv,yv,zv, +1);
    [path_b, ~] = trace_flow_to_target(pb, pa, Fx,Fy,Fz, ...
        V, cost_vol, opt, vox, sz, xv,yv,zv, -1);

    converged = ~isempty(path_a) && ~isempty(path_b) && ...
                norm(path_a(end,:) - path_b(end,:)) < 2.5;

    if converged
        path   = [path_a; flipud(path_b)];
        method = 'flow_converged';
    else
        path   = fast_march_bridge(cost_vol, pa, pb, xv,yv,zv,sz,vox);
        method = 'fast_march';
    end

    conf = score_bridge(path, A_map, B_map, V, I, ...
                        opt.hu_vessel, opt.w_vessel, opt.w_geo, ...
                        opt.w_hu, xv, yv, zv, sz);

    bridge_report(g).bridge = path;
    bridge_report(g).method = method;
    bridge_report(g).conf   = conf;

    if conf >= opt.conf_auto
        r_bridge = murray_bridge_radius(G, ni, nj);
        G = add_bridge_to_graph(G, path, ni, nj, r_bridge, false);
        old_cc = cc(nj); cc(cc == old_cc) = cc(ni);
        bridge_report(g).action = 'accepted';
        n_auto = n_auto + 1;
    elseif conf >= opt.conf_mst
        bridge_report(g).action = 'flagged';
        n_flag = n_flag + 1;
    else
        bridge_report(g).action = 'mst_fallback';
        n_mst = n_mst + 1;
    end
end
vprint(opt,'   Accepted: %d  flagged: %d  MST-fallback: %d', ...
    n_auto, n_flag, n_mst);

%% -----------------------------------------------------------------------
% 7. Topology guarantee: Kruskal MST over remaining components
% -----------------------------------------------------------------------
vprint(opt,'[7/7] Topology guarantee: MST over remaining components...');

cc       = connected_components(G.edges, size(G.nodes,1));
n_cc_fin = max(cc);
vprint(opt,'   Components before MST: %d', n_cc_fin);

if n_cc_fin > 1
    G  = mst_connect_components(G, cc, n_cc_fin, cost_vol, ...
                                 xv,yv,zv,sz,vox,opt);
    cc = connected_components(G.edges, size(G.nodes,1));
    vprint(opt,'   Components after MST : %d', max(cc));
end

assert(max(cc) == 1, ...
    'Topology error: %d components remain after MST.', max(cc));
vprint(opt,'   Simply connected: CONFIRMED');

%% -----------------------------------------------------------------------
% 8. Rasterise and save outputs
% -----------------------------------------------------------------------
vprint(opt,'Saving outputs...');

label = rasterise_graph(G, xv, yv, zv, sz, vox, opt.r_min_mm);

% Fallback: if the rasterised graph is empty (no CCO improvement possible),
% save the raw binary Frangi threshold so the output is never all-zeros.
if ~any(label(:))
    vprint(opt,'   WARNING: rasterised graph is empty — falling back to binary Frangi threshold (%.2f).', ...
        opt.frangi_threshold);
    label = uint8(V >= opt.frangi_threshold);
    out_info = info;
    out_info.Datatype     = 'uint8';
    out_info.BitsPerPixel = 8;
    out_info.Filename     = '';
    out_info.DisplayIntensityRange = [0 1];
else
    out_info = build_nifti_info(sz, mean(vox), sz.*vox);
end

% ── Post-processing: enforce a single 3D connected component ─────────────
% "Simply connected" in the graph does not guarantee voxel-level connectivity
% (MST bridge tubes can be thinner than the voxel gap they span).
% Keep the largest 3D component; report what is dropped.
cc_post = bwconncomp(label > 0, 6);
if cc_post.NumObjects > 1
    comp_sz  = cellfun(@numel, cc_post.PixelIdxList);
    [~, L]   = max(comp_sz);
    dropped  = sum(label(:) > 0) - comp_sz(L);
    label_lc = zeros(sz, 'uint8');
    label_lc(cc_post.PixelIdxList{L}) = label(cc_post.PixelIdxList{L});
    label    = label_lc;
    vprint(opt,'   3D post-processing: %d→1 component, %d voxels dropped (%.1f%%)', ...
        cc_post.NumObjects, dropped, 100*dropped/sum(label(:)>0 | dropped>0));
else
    vprint(opt,'   3D post-processing: already 1 component, nothing dropped.');
end

out_nii = fullfile(opt.output_dir,'completed_vessels.nii.gz');
niftiwrite(label, out_nii(1:end-3), out_info, 'Compressed', true);

skel_info = info;
skel_info.Datatype     = 'uint8';
skel_info.BitsPerPixel = 8;
skel_info.Filename     = '';
skel_nii = fullfile(opt.output_dir,'skeleton.nii.gz');
niftiwrite(uint8(G.skel), skel_nii(1:end-3), skel_info, 'Compressed', true);

save(fullfile(opt.output_dir,'completed_graph.mat'),  'G');
save(fullfile(opt.output_dir,'bridge_report.mat'),     'bridge_report');

vprint(opt,'   completed_vessels.nii.gz  (%d vessel voxels)', sum(label(:) > 0));
vprint(opt,'   skeleton.nii.gz           (%d skeleton voxels)', sum(G.skel(:)));
vprint(opt,'   completed_graph.mat');
vprint(opt,'   bridge_report.mat');
vprint(opt,'=== Done ===');
end


%% =======================================================================
%  LOCAL FUNCTIONS
%% =======================================================================

% -------------------------------------------------------------------------
function G = extract_graph_from_frangi(V_map, threshold, min_island_vox, vox, xv, yv, zv, sz)
%EXTRACT_GRAPH_FROM_FRANGI
%  Threshold the Frangi map, skeletonise with bwskel, and extract a
%  node/edge graph.  Radii are estimated from the distance transform of
%  the binary mask.
%
%  Graph convention:
%    G.nodes  [M×3]  RAS mm coordinates of skeleton branch/end points
%    G.edges  [E×2]  node index pairs
%    G.radii  [E×1]  mean local radius along segment (mm)
%    G.fixed  [E×1]  true = original Frangi segment (never modified)

% 1. Threshold; keep all connected components above min_island_vox
binary = V_map >= threshold;
cc_bin = bwconncomp(binary, 6);
if cc_bin.NumObjects == 0
    error('extract_graph_from_frangi: no voxels above threshold %.2f', threshold);
end
comp_sizes = cellfun(@numel, cc_bin.PixelIdxList);
keep_mask  = comp_sizes >= min_island_vox;
binary_clean = false(sz);
for k = find(keep_mask)
    binary_clean(cc_bin.PixelIdxList{k}) = true;
end

fprintf('   Components: %d total, %d kept (>= %d vox), %d dropped as islands\n', ...
    cc_bin.NumObjects, sum(keep_mask), min_island_vox, sum(~keep_mask));
fprintf('   Binary mask: %d voxels retained\n', sum(binary_clean(:)));

% 2. Skeletonise (requires Image Processing Toolbox R2019a+)
skel = bwskel(binary_clean);
fprintf('   Skeleton: %d voxels\n', sum(skel(:)));

% 3. Distance transform → local radius in mm
dt = bwdist(~binary_clean) .* mean(vox);   % mm

% 4. Count 26-connected skeleton neighbours at each skeleton voxel
%    via convolution with a 3×3×3 all-ones kernel
kern      = ones(3,3,3,'double');
kern(2,2,2) = 0;
nbr_count = round(imfilter(single(skel), kern, 'replicate'));
nbr_count(~skel) = 0;

% 5. Critical voxels: endpoints (1 neighbour) or junctions (≥ 3)
endpoint_mask  = skel & (nbr_count == 1);
junction_mask  = skel & (nbr_count >= 3);
isolated_mask  = skel & (nbr_count == 0);   % lone voxels → treat as endpoint
critical_mask  = endpoint_mask | junction_mask | isolated_mask;

crit_idx = find(critical_mask);
n_nodes  = numel(crit_idx);

if n_nodes == 0
    % Whole skeleton is a single closed loop — add one artificial endpoint.
    % max(dt(skel)) returns an index into the sub-vector, not the volume;
    % use find(skel) to map back to a proper linear volume index.
    skel_idx = find(skel);
    [~, best] = max(dt(skel_idx));
    crit_idx = skel_idx(best);
    n_nodes = 1;
    critical_mask = false(sz);
    critical_mask(crit_idx) = true;
end

[cx, cy, cz] = ind2sub(sz, crit_idx(:));
% reshape forces n_nodes×1 regardless of whether xv(cx(:)) returns row or col
nodes = [reshape(xv(cx(:)), [], 1), ...
         reshape(yv(cy(:)), [], 1), ...
         reshape(zv(cz(:)), [], 1)];   % [n_nodes × 3]

fprintf('   Critical voxels: %d endpoints, %d junctions\n', ...
    sum(endpoint_mask(:)), sum(junction_mask(:)));

% 6. Node index map
node_map = zeros(sz, 'int32');
for k = 1:n_nodes
    node_map(crit_idx(k)) = int32(k);
end

% 7. Trace skeleton segments between critical voxels
%    Non-critical skeleton voxels form path segments; we walk from each
%    critical voxel along non-critical neighbours until reaching another
%    critical voxel.
offsets = precompute_26_offsets(sz);
n_off   = numel(offsets);

visited    = ~skel;           % non-skeleton = already visited
visited(critical_mask) = true; % critical voxels = visited (entry points)

edges      = zeros(0, 2, 'int32');
edge_radii = zeros(0, 1, 'single');

for ni = 1:n_nodes
    si = crit_idx(ni);

    for d = 1:n_off
        nbr = int32(si) + offsets(d);
        if nbr < 1 || nbr > numel(skel), continue; end
        if visited(nbr), continue; end

        % Walk from nbr until a critical voxel is reached
        path_r  = single([dt(si); dt(nbr)]);
        current = nbr;
        visited(current) = true;
        found   = false;

        for step = 1:10000
            end_node = int32(0);
            next_vox = int32(0);
            for d2 = 1:n_off
                nc = int32(current) + offsets(d2);
                if nc < 1 || nc > numel(skel), continue; end
                if ~skel(nc), continue; end
                if critical_mask(nc)
                    end_node = node_map(nc);
                    break;
                elseif ~visited(nc)
                    next_vox = nc;
                end
            end

            if end_node > 0
                nj = end_node;
                if nj ~= ni
                    edges(end+1,:)      = int32([ni, nj]);   %#ok<AGROW>
                    edge_radii(end+1)   = mean(path_r);       %#ok<AGROW>
                end
                found = true; break;
            elseif next_vox > 0
                current = next_vox;
                visited(current) = true;
                path_r(end+1) = single(dt(current));         %#ok<AGROW>
            else
                break;   % dead end
            end
        end
    end

    % Direct edges: this critical voxel adjacent to another critical voxel
    for d = 1:n_off
        nbr = int32(si) + offsets(d);
        if nbr < 1 || nbr > numel(skel), continue; end
        if ~critical_mask(nbr), continue; end
        nj = node_map(nbr);
        if nj > ni
            edges(end+1,:)    = int32([ni, nj]);             %#ok<AGROW>
            edge_radii(end+1) = single((dt(si)+dt(nbr))/2);  %#ok<AGROW>
        end
    end
end

% Closed-loop fallback: BFS produced zero edges because the skeleton is a
% single closed curve with no degree-1 or degree-3 voxels.  Trace the full
% skeleton as an ordered BFS chain so that rasterise_graph has something to
% draw.  Decimated to one node per voxel (may be large but always correct).
if isempty(edges) && any(skel(:))
    fprintf('   Skeleton has no endpoints/junctions — tracing as closed-loop chain.\n');
    queue   = crit_idx(1);
    vis2    = false(sz);
    vis2(queue) = true;
    order   = queue;
    head    = 1;
    while head <= numel(order)
        curr = order(head);  head = head + 1;
        for d = 1:n_off
            nb = int32(curr) + offsets(d);
            if nb < 1 || nb > numel(skel), continue; end
            if ~skel(nb) || vis2(nb),       continue; end
            vis2(nb) = true;
            order(end+1) = nb;   %#ok<AGROW>
        end
    end
    [ox, oy, oz] = ind2sub(sz, order(:));
    nodes        = [reshape(xv(ox(:)), [], 1), ...
                    reshape(yv(oy(:)), [], 1), ...
                    reshape(zv(oz(:)), [], 1)];
    n_nodes      = numel(order);
    n_chain      = n_nodes - 1;
    edges        = [(1:n_chain)', (2:n_chain+1)'];
    edge_radii   = zeros(n_chain, 1, 'single');
    for k = 1:n_chain
        edge_radii(k) = single((dt(order(k)) + dt(order(k+1))) / 2 * mean(vox));
    end
    fprintf('   Chain: %d nodes, %d edges.\n', n_nodes, n_chain);
end

% Remove duplicate edges
if ~isempty(edges)
    sorted_e  = sort(double(edges), 2);
    [~, ia]   = unique(sorted_e, 'rows', 'stable');
    edges      = double(edges(ia,:));
    edge_radii = edge_radii(ia);
else
    edges      = zeros(0,2);
    edge_radii = zeros(0,1,'single');
end

G.nodes = nodes;
G.edges = edges;
G.radii = max(double(edge_radii), 0.1);
G.fixed = true(size(edges,1),1);
G.skel  = skel;
end


% -------------------------------------------------------------------------
function offsets = precompute_26_offsets(sz)
%PRECOMPUTE_26_OFFSETS  Linear-index offsets for 26-connected neighbourhood.
offsets = zeros(26, 1, 'int32');
k = 0;
for dz = -1:1
    for dy = -1:1
        for dx = -1:1
            if dx==0 && dy==0 && dz==0, continue; end
            k = k+1;
            offsets(k) = int32(dx + dy*sz(1) + dz*sz(1)*sz(2));
        end
    end
end
offsets = offsets(1:k);
end


% -------------------------------------------------------------------------
function [A_map, B_map] = compute_AB_maps(I, scales_mm, alpha, beta, vox)
sz    = size(I);
A_map = zeros(sz,'single');
B_map = zeros(sz,'single');
V_max = zeros(sz,'single');
nS    = numel(scales_mm);
BAR   = 30;
fprintf('   Computing A/B maps  [%s] 0/%d  σ=%.2f mm', ...
    repmat(' ',1,BAR), nS, scales_mm(1));

for k = 1:nS
    sigma   = scales_mm(k);
    sig_vox = sigma ./ vox;
    Ig      = imgaussfilt3(I, sig_vox);
    s2      = sigma^2;

    Ixx = s2 * del2_along(Ig,1);
    Iyy = s2 * del2_along(Ig,2);
    Izz = s2 * del2_along(Ig,3);
    Ixy = s2 * mixed_deriv(Ig,1,2);
    Ixz = s2 * mixed_deriv(Ig,1,3);
    Iyz = s2 * mixed_deriv(Ig,2,3);

    [lam1,lam2,lam3] = eig3sym_vec(Ixx,Iyy,Izz,Ixy,Ixz,Iyz);

    vessel_mask = (lam2 < 0) & (lam3 < 0);

    RA = abs(lam2) ./ (abs(lam3) + 1e-9);
    RB = abs(lam1) ./ (sqrt(abs(lam2).*abs(lam3)) + 1e-9);
    S  = sqrt(lam1.^2 + lam2.^2 + lam3.^2);
    C_gate = max(S(:))/2 + 1e-9;

    As = single(1 - exp(-RA.^2/(2*alpha^2)));
    Bs = single(exp(-RB.^2/(2*beta^2)));
    Cs = single(1 - exp(-S.^2/(2*C_gate^2)));
    Vs = As .* Bs .* Cs;
    Vs(~vessel_mask) = 0;

    update = Vs > V_max;
    A_map(update) = As(update);
    B_map(update) = Bs(update);
    V_max(update) = Vs(update);

    filled = round(k / nS * BAR);
    next_s = scales_mm(min(k+1, nS));
    fprintf('\r   Computing A/B maps  [%s%s] %d/%d  σ=%.2f mm', ...
        repmat('#',1,filled), repmat(' ',1,BAR-filled), k, nS, next_s);
end
fprintf('\n');
end


% -------------------------------------------------------------------------
function [l1,l2,l3] = eig3sym_vec(Ixx,Iyy,Izz,Ixy,Ixz,Iyz)
p1 = Ixy.^2 + Ixz.^2 + Iyz.^2;
q  = (Ixx + Iyy + Izz) / 3;
p2 = (Ixx-q).^2 + (Iyy-q).^2 + (Izz-q).^2 + 2*p1;
p  = sqrt(max(p2/6,0));

inv_p = 1./(p+1e-12);
Bxx=(Ixx-q).*inv_p; Byy=(Iyy-q).*inv_p; Bzz=(Izz-q).*inv_p;
Bxy=Ixy.*inv_p;     Bxz=Ixz.*inv_p;     Byz=Iyz.*inv_p;

r = (Bxx.*(Byy.*Bzz-Byz.^2) ...
   - Bxy.*(Bxy.*Bzz-Byz.*Bxz) ...
   + Bxz.*(Bxy.*Byz-Byy.*Bxz)) / 2;
r = max(-1,min(1,r));
phi = acos(r)/3;

lam_a = q + 2*p.*cos(phi);
lam_c = q + 2*p.*cos(phi + 2*pi/3);
lam_b = 3*q - lam_a - lam_c;

stack = cat(4,lam_a,lam_b,lam_c);
[~,ord] = sort(abs(stack),4,'ascend');

% Gather sorted eigenvalues: stack is [H W D 3], linear index for
% spatial voxel k (0-based) with channel c (1-based) = k+1 + (c-1)*n
n = numel(Ixx);
base = reshape(0:n-1, size(Ixx));
o1 = reshape(ord(:,:,:,1), size(Ixx));
o2 = reshape(ord(:,:,:,2), size(Ixx));
o3 = reshape(ord(:,:,:,3), size(Ixx));
l1 = reshape(stack(base + 1 + (o1-1)*n), size(Ixx));
l2 = reshape(stack(base + 1 + (o2-1)*n), size(Ixx));
l3 = reshape(stack(base + 1 + (o3-1)*n), size(Ixx));
end


% -------------------------------------------------------------------------
function d2 = del2_along(I,dim)
sz=size(I); d2=zeros(sz);
m={':',':',':'}; c={':',':',':'}; pp={':',':',':'};
m{dim}=1:sz(dim)-2; c{dim}=2:sz(dim)-1; pp{dim}=3:sz(dim);
d2(c{:}) = I(pp{:}) - 2*I(c{:}) + I(m{:});
end

function dxy = mixed_deriv(I,d1,d2)
dxy = grad_along(grad_along(I,d1),d2);
end

function dI = grad_along(I,dim)
sz=size(I); dI=zeros(sz);
m={':',':',':'}; pp={':',':',':'}; c={':',':',':'};
m{dim}=1:sz(dim)-2; pp{dim}=3:sz(dim); c{dim}=2:sz(dim)-1;
dI(c{:}) = (I(pp{:})-I(m{:}))/2;
end


% -------------------------------------------------------------------------
function [path, found] = trace_flow_to_target(pa, pb, Fx,Fy,Fz, ...
    V, ~, opt, vox, sz, xv,yv,zv, flow_sign)
max_steps = ceil(opt.max_gap_mm / (0.5*mean(vox)));
step_sz   = 0.5 * mean(vox);
pos  = pa;  path = pos;  found = false;

for iter = 1:max_steps
    ci = coord_to_idx(pos,xv,yv,zv,sz);
    if isempty(ci), break; end

    f = flow_sign * [Fx(ci), Fy(ci), Fz(ci)];
    fn = norm(f); if fn > 1e-6, f=f/fn; end

    goal = pb - pos; gn = norm(goal);
    if gn < 1.5*mean(vox), found=true; break; end
    gd = goal/gn;

    dir = (1-opt.w_flow)*gd + opt.w_flow*f;
    dn  = norm(dir); if dn<1e-9, dir=gd; else dir=dir/dn; end

    pos  = pos + dir*step_sz;
    path = [path; pos];  %#ok<AGROW>

    ci2 = coord_to_idx(pos,xv,yv,zv,sz);
    if ~isempty(ci2) && V(ci2) > 0.10, found=true; break; end
end
end


% -------------------------------------------------------------------------
function path = fast_march_bridge(cost_vol, pa, pb, xv,yv,zv,sz,vox)
ia = coord_to_idx(pa,xv,yv,zv,sz);
ib = coord_to_idx(pb,xv,yv,zv,sz);
if isempty(ia)||isempty(ib), path=[pa;pb]; return; end
try
    src=false(sz); src(ia)=true;
    T = msfm3d(cost_vol, double(src), true, true);
    path = backtrack_arrival(T,ia,ib,sz,vox,xv,yv,zv);
catch
    path = dijkstra_bridge(cost_vol,ia,ib,sz,xv,yv,zv,vox);
end
end


% -------------------------------------------------------------------------
function path = backtrack_arrival(T,ia,ib,sz,vox,xv,yv,zv)
MAX_STEPS=800;
[xi,yi,zi]=ind2sub(sz,ib);
pos=[ xv(xi) yv(yi) zv(zi) ]; path=pos;
for iter=1:MAX_STEPS
    ci=coord_to_idx(pos,xv,yv,zv,sz); if isempty(ci),break; end
    gT=arrival_grad(T,ci,sz,vox);
    if norm(gT)<1e-9,break; end
    pos=pos-gT/norm(gT).*vox;
    path=[path;pos]; %#ok<AGROW>
    ci2=coord_to_idx(pos,xv,yv,zv,sz);
    if ~isempty(ci2)&&ci2==ia,break; end
end
end

function gT=arrival_grad(T,ci,sz,vox)
[xi,yi,zi]=ind2sub(sz,ci); gT=zeros(1,3);
xp=min(sz(1),xi+1); xm=max(1,xi-1);
yp=min(sz(2),yi+1); ym=max(1,yi-1);
zp=min(sz(3),zi+1); zm=max(1,zi-1);
gT(1)=(T(xp,yi,zi)-T(xm,yi,zi))/(2*vox(1));
gT(2)=(T(xi,yp,zi)-T(xi,ym,zi))/(2*vox(2));
gT(3)=(T(xi,yi,zp)-T(xi,yi,zm))/(2*vox(3));
end


% -------------------------------------------------------------------------
function path = dijkstra_bridge(cost_vol,ia,ib,sz,xv,yv,zv,vox)
[xa,ya,za]=ind2sub(sz,ia); [xb,yb,zb]=ind2sub(sz,ib);
margin=10;
x1=max(1,min(xa,xb)-margin); x2=min(sz(1),max(xa,xb)+margin);
y1=max(1,min(ya,yb)-margin); y2=min(sz(2),max(ya,yb)+margin);
z1=max(1,min(za,zb)-margin); z2=min(sz(3),max(za,zb)+margin);

sub_cost=cost_vol(x1:x2,y1:y2,z1:z2);
ssz=size(sub_cost);
ia_s=sub2ind(ssz,xa-x1+1,ya-y1+1,za-z1+1);
ib_s=sub2ind(ssz,xb-x1+1,yb-y1+1,zb-z1+1);

dist=inf(ssz); dist(ia_s)=0;
prev=zeros(ssz,'int32'); visited=false(ssz);
[~,order]=sort(sub_cost(:));

for k=order'
    if visited(k),continue; end; visited(k)=true;
    if k==ib_s,break; end
    [xi,yi,zi]=ind2sub(ssz,k);
    for n=get_6nbrs(xi,yi,zi,ssz)
        alt=dist(k)+sub_cost(n);
        if alt<dist(n), dist(n)=alt; prev(n)=int32(k); end
    end
end

path_idx=ib_s;
cur=ib_s;
while cur~=ia_s && prev(cur)~=0
    cur=prev(cur); path_idx=[cur path_idx]; %#ok<AGROW>
end
path=zeros(numel(path_idx),3);
for k=1:numel(path_idx)
    [xi,yi,zi]=ind2sub(ssz,path_idx(k));
    path(k,:)=[xv(xi+x1-1) yv(yi+y1-1) zv(zi+z1-1)];
end
if isempty(path)||size(path,1)<2
    path=[xv(xa) yv(ya) zv(za); xv(xb) yv(yb) zv(zb)];
end
end

function nbrs=get_6nbrs(xi,yi,zi,sz)
nbrs=[];
for off=[-1 0 0;1 0 0;0 -1 0;0 1 0;0 0 -1;0 0 1]'
    nx=xi+off(1); ny=yi+off(2); nz=zi+off(3);
    if nx>=1&&nx<=sz(1)&&ny>=1&&ny<=sz(2)&&nz>=1&&nz<=sz(3)
        nbrs(end+1)=sub2ind(sz,nx,ny,nz); %#ok<AGROW>
    end
end
end


% -------------------------------------------------------------------------
function conf=score_bridge(path,A_map,B_map,V,I,hu_vessel, ...
    w_v,w_g,w_hu,xv,yv,zv,sz)
if isempty(path)||size(path,1)<2, conf=0; return; end
n=size(path,1);
a_v=zeros(n,1); b_v=zeros(n,1); v_v=zeros(n,1); hu_v=zeros(n,1);
for k=1:n
    ci=coord_to_idx(path(k,:),xv,yv,zv,sz); if isempty(ci),continue; end
    a_v(k)=A_map(ci); b_v(k)=B_map(ci);
    v_v(k)=V(ci);
    hu_v(k)=1/(1+exp(-(I(ci)-hu_vessel)/50));
end
geo  = mean(a_v.*b_v);
smth = 1/(1+var(diff(a_v))*100);
conf = (w_v*mean(v_v) + w_g*geo + w_hu*mean(hu_v) + 0.1*smth) / ...
       (w_v+w_g+w_hu+0.1);
conf = max(0,min(1,conf));
end


% -------------------------------------------------------------------------
function r=murray_bridge_radius(G,ni,nj)
ei=find(G.edges(:,1)==ni|G.edges(:,2)==ni,1);
ej=find(G.edges(:,1)==nj|G.edges(:,2)==nj,1);
ri=1.5; rj=1.5;
if ~isempty(ei), ri=G.radii(ei); end
if ~isempty(ej), rj=G.radii(ej); end
r = ((ri^3+rj^3)/2)^(1/3) * (1/2)^(1/3);
r = max(r, 0.5);
end


% -------------------------------------------------------------------------
function G=add_bridge_to_graph(G,path,ni,nj,r_bridge,is_fixed)
n_path=size(path,1);
if n_path<=1
    G.edges(end+1,:)=[ni nj]; G.radii(end+1)=r_bridge;
    G.fixed(end+1)=is_fixed; return;
end
new_idx=zeros(n_path,1); new_idx(1)=ni; new_idx(end)=nj;
for k=2:n_path-1
    G.nodes(end+1,:)=path(k,:); new_idx(k)=size(G.nodes,1);
end
for k=1:n_path-1
    G.edges(end+1,:)=[new_idx(k) new_idx(k+1)];
    G.radii(end+1)=r_bridge; G.fixed(end+1)=is_fixed;
end
end


% -------------------------------------------------------------------------
function cc=connected_components(edges,n_nodes)
parent=1:n_nodes;
    function r=find_root(p,x)
        while p(x)~=x, x=p(x); end; r=x;
    end
for k=1:size(edges,1)
    ra=find_root(parent,edges(k,1)); rb=find_root(parent,edges(k,2));
    if ra~=rb, parent(rb)=ra; end
end
cc=zeros(n_nodes,1);
for k=1:n_nodes, cc(k)=find_root(parent,k); end
[~,~,cc]=unique(cc);
end


% -------------------------------------------------------------------------
function G=mst_connect_components(G,cc,n_cc,cost_vol,xv,yv,zv,sz,vox,opt)
degree=zeros(size(G.nodes,1),1);
for e=1:size(G.edges,1)
    degree(G.edges(e,1))=degree(G.edges(e,1))+1;
    degree(G.edges(e,2))=degree(G.edges(e,2))+1;
end
reps=zeros(n_cc,1);
for c=1:n_cc
    members=find(cc==c); ep=members(degree(members)==1);
    if ~isempty(ep), reps(c)=ep(1); else, reps(c)=members(1); end
end
D=inf(n_cc);
for i=1:n_cc
    for j=i+1:n_cc
        d=norm(G.nodes(reps(i),:)-G.nodes(reps(j),:));
        D(i,j)=d; D(j,i)=d;
    end
end
[~,edge_order]=sort(D(:)); mst_cc=1:n_cc; n_added=0;
for k=1:numel(edge_order)
    if n_added==n_cc-1, break; end
    [ci,cj]=ind2sub([n_cc n_cc],edge_order(k));
    if ci==cj||mst_cc(ci)==mst_cc(cj), continue; end
    old=mst_cc(cj); mst_cc(mst_cc==old)=mst_cc(ci);
    ni=reps(ci); nj=reps(cj);
    path=fast_march_bridge(cost_vol,G.nodes(ni,:),G.nodes(nj,:), ...
        xv,yv,zv,sz,vox);
    r=murray_bridge_radius(G,ni,nj);
    G=add_bridge_to_graph(G,path,ni,nj,r,false);
    n_added=n_added+1;
    fprintf('   MST bridge %d: comp %d↔%d  (%.1f mm)\n', ...
        n_added,ci,cj,D(ci,cj));
end
end


% -------------------------------------------------------------------------
function label=rasterise_graph(G,xv,yv,zv,sz,vox,r_min)
label=zeros(sz,'uint8'); vx_mm=mean(vox);
for e=1:size(G.edges,1)
    p1=G.nodes(G.edges(e,1),:); p2=G.nodes(G.edges(e,2),:);
    r=max(G.radii(e),r_min); seg_len=norm(p2-p1);
    if seg_len<0.1, continue; end
    n_steps=max(2,ceil(seg_len/(vx_mm*0.5)));
    for t=linspace(0,1,n_steps)
        pt=p1+t*(p2-p1);
        xi=nearest_idx(pt(1),xv,sz(1));
        yi=nearest_idx(pt(2),yv,sz(2));
        zi=nearest_idx(pt(3),zv,sz(3));
        rv=max(1,ceil(r/vx_mm));
        x1b=max(1,xi-rv); x2b=min(sz(1),xi+rv);
        y1b=max(1,yi-rv); y2b=min(sz(2),yi+rv);
        z1b=max(1,zi-rv); z2b=min(sz(3),zi+rv);
        [Xs,Ys,Zs]=meshgrid(xv(x1b:x2b),yv(y1b:y2b),zv(z1b:z2b));
        Xs=permute(Xs,[2 1 3]); Ys=permute(Ys,[2 1 3]); Zs=permute(Zs,[2 1 3]);
        in_s=(Xs-pt(1)).^2+(Ys-pt(2)).^2+(Zs-pt(3)).^2<=r^2;
        sub=label(x1b:x2b,y1b:y2b,z1b:z2b);
        sub(in_s)=uint8(2);
        label(x1b:x2b,y1b:y2b,z1b:z2b)=sub;
    end
end
end


% -------------------------------------------------------------------------
function ci=coord_to_idx(pt,xv,yv,zv,sz)
xi=nearest_idx(pt(1),xv,sz(1));
yi=nearest_idx(pt(2),yv,sz(2));
zi=nearest_idx(pt(3),zv,sz(3));
if xi<1||xi>sz(1)||yi<1||yi>sz(2)||zi<1||zi>sz(3), ci=[]; return; end
ci=sub2ind(sz,xi,yi,zi);
end

function idx=nearest_idx(val,vec,n)
[~,idx]=min(abs(vec-val)); idx=max(1,min(n,idx));
end


% -------------------------------------------------------------------------
function info=build_nifti_info(sz,vx_mm,fov_mm)
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
T=eye(4); T(1,1)=vx_mm; T(2,2)=vx_mm; T(3,3)=vx_mm;
T(1,4)=-fov_mm(1)/2; T(2,4)=-fov_mm(2)/2; T(3,4)=-fov_mm(3)/2;
info.Transform             = affine3d(T');
info.TransformName         = 'Sform';
info.raw.sform_code=1; info.raw.qform_code=1;
info.raw.pixdim=[1 vx_mm vx_mm vx_mm 0 0 0 0];
info.raw.srow_x=T(1,:); info.raw.srow_y=T(2,:); info.raw.srow_z=T(3,:);
info.raw.dim=[3 sz(1) sz(2) sz(3) 1 1 1 1];
info.raw.datatype=2; info.raw.bitpix=8; info.raw.xyzt_units=2;
end


% -------------------------------------------------------------------------
function vprint(opt,fmt,varargin)
if opt.verbose, fprintf([fmt '\n'],varargin{:}); end
end
