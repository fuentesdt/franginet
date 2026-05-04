% visualize_vessel_phantoms.m
% Renders a 4x3 dashboard of 3-D isosurface views for 00?_vessel_phantom.nii.gz.
% Output: vessel_phantom_dashboard.png saved alongside this script.
%
% Run from any directory:
%   >> visualize_vessel_phantoms

%% ── 0. Configuration ─────────────────────────────────────────────────────

SCRIPT_DIR = fileparts(mfilename('fullpath'));
OUT_FILE   = fullfile(SCRIPT_DIR, 'vessel_phantom_dashboard.png');
TARGET_MM  = 2.0;   % resample to isotropic 2 mm before isosurface

% Colours (RGB) and opacities indexed by label value 1-4
COLORS = [0.85 0.55 0.35;   % 1 liver      – tan
          0.20 0.60 1.00;   % 2 inflow     – blue
          0.90 0.30 0.30;   % 3 outflow    – red
          1.00 0.85 0.00];  % 4 tumor      – yellow
ALPHAS     = [0.25; 1.0; 1.0; 1.0];

% reducepatch face-count ceilings per label
MAX_FACES  = [30000; 8000; 8000; 4000];

% Three camera orientations [azimuth, elevation] for dashboard columns
VIEWS      = [-45 25; 30 35; 135 20];

% smooth3 parameters per label (wider kernel for liver to avoid aliasing at 2mm)
SMOOTH_K   = {[5 5 5]; [3 3 3]; [3 3 3]; [3 3 3]};   % kernel size
SMOOTH_S   = [1.5; 1.0; 1.0; 1.0];                    % Gaussian sigma

%% ── 1. Locate phantom files ───────────────────────────────────────────────

listing = dir(fullfile(SCRIPT_DIR, '00*_vessel_phantom.nii.gz'));
if isempty(listing)
    error('No 00?_vessel_phantom.nii.gz files found in %s', SCRIPT_DIR);
end
listing = listing(~[listing.isdir]);
listing = listing(~cellfun(@isempty, regexp({listing.name}, '^00\d_vessel_phantom\.nii\.gz$')));
[~, ord] = sort({listing.name});
listing  = listing(ord);
nP       = numel(listing);
fprintf('Found %d phantom file(s).\n', nP);

%% ── 2. Pre-compute isosurfaces ────────────────────────────────────────────

patches = struct('id', cell(nP,1), 'fv', cell(nP,1), 'vol_sz', cell(nP,1));

for p = 1:nP
    fpath = fullfile(listing(p).folder, listing(p).name);
    id    = listing(p).name(1:3);
    fprintf('[%s] Loading %s ...\n', id, listing(p).name);

    info   = niftiinfo(fpath);
    vol    = niftiread(info);                       % uint8, [nx ny nz]
    pixdim = double(info.PixelDimensions(1:3));     % voxel size in mm

    % Resample to TARGET_MM isotropic (nearest-neighbour preserves discrete labels)
    in_sz  = size(vol, [1 2 3]);
    out_sz = max(1, round(double(in_sz) .* pixdim / TARGET_MM));
    vol_ds = uint8(imresize3(single(vol), out_sz, 'Method', 'nearest'));

    patches(p).id     = id;
    patches(p).vol_sz = out_sz;
    patches(p).fv     = cell(4, 1);

    for lab = 1:4
        bin = double(vol_ds == lab);
        if ~any(bin(:))
            continue;
        end

        % Smooth the binary mask
        sm = smooth3(bin, 'gaussian', SMOOTH_K{lab}, SMOOTH_S(lab));

        % Pad one voxel of zeros on every face to seal open boundaries
        sm_pad = padarray(sm, [1 1 1], 0, 'both');

        % Extract isosurface in padded space
        fv = isosurface(sm_pad, 0.5);
        if isempty(fv.vertices)
            continue;
        end

        % Compute per-vertex normals from the smooth scalar field (before any offset)
        fv.normals = isonormals(sm_pad, fv.vertices);

        % Decimate if needed; recompute normals afterwards
        if size(fv.faces, 1) > MAX_FACES(lab)
            fv = reducepatch(fv, MAX_FACES(lab));
            fv.normals = isonormals(sm_pad, fv.vertices);
        end

        % Remove the 1-voxel pad offset from vertex coordinates
        fv.vertices = fv.vertices - 1;

        patches(p).fv{lab} = fv;
        fprintf('  label %d: %d faces\n', lab, size(fv.faces, 1));
    end
end

%% ── 3. Build dashboard figure ─────────────────────────────────────────────

fprintf('Rendering dashboard ...\n');

fig = figure('Color', 'k', ...
             'Units', 'inches', ...
             'Position', [0 0 15 12], ...
             'Visible', 'off');
set(fig, 'Renderer', 'opengl');

t = tiledlayout(fig, nP, 3, 'TileSpacing', 'tight', 'Padding', 'compact');

for p = 1:nP
    for v = 1:3
        ax = nexttile(t);
        hold(ax, 'on');

        % Hide axis lines and ticks while keeping title visible
        ax.Color  = 'k';
        ax.XColor = 'none';
        ax.YColor = 'none';
        ax.ZColor = 'none';
        ax.DataAspectRatio = [1 1 1];
        ax.Projection      = 'perspective';

        % Draw opaque labels first (4→3→2), semi-transparent liver (1) last
        for lab = [4 3 2 1]
            fv = patches(p).fv{lab};
            if isempty(fv)
                continue;
            end
            patch(ax, ...
                'Vertices',           fv.vertices, ...
                'Faces',              fv.faces, ...
                'VertexNormals',      fv.normals, ...
                'FaceColor',          COLORS(lab, :), ...
                'FaceAlpha',          ALPHAS(lab), ...
                'EdgeColor',          'none', ...
                'VertexNormalsMode',  'manual', ...
                'AmbientStrength',    0.30, ...
                'DiffuseStrength',    0.70, ...
                'SpecularStrength',   0.40, ...
                'SpecularExponent',   20);
        end

        % Camera orientation first, then lights (headlight tracks camera)
        view(ax, VIEWS(v, 1), VIEWS(v, 2));
        camlight(ax, 'headlight');
        camlight(ax, 30, 10);
        lighting(ax, 'gouraud');
        material(ax, 'shiny');

        % Zoom to actual geometry, lock 3-D aspect so rotation doesn't rescale
        axis(ax, 'tight');
        axis(ax, 'vis3d');

        % Two-line tile title: phantom file on line 1, view angles on line 2
        title(ax, {sprintf('%s\\_vessel\\_phantom', patches(p).id), ...
                   sprintf('az=%d°, el=%d°', VIEWS(v,1), VIEWS(v,2))}, ...
            'Color', 'w', 'FontSize', 7, 'FontWeight', 'bold');

        hold(ax, 'off');
    end
end

%% ── 4. Colour legend ─────────────────────────────────────────────────────

label_names = {'Liver (\alpha=0.25)', 'Inflow vessels', 'Outflow vessels', 'Tumor'};
n_leg = numel(label_names);
swatch_w = 0.03;
swatch_h = 0.012;
gap      = (1 - n_leg * (swatch_w + 0.11)) / 2;

for lab = 1:n_leg
    x0 = gap + (lab - 1) * (swatch_w + 0.11);
    annotation(fig, 'rectangle', [x0, 0.005, swatch_w, swatch_h], ...
        'FaceColor', COLORS(lab, :), 'EdgeColor', 'none');
    annotation(fig, 'textbox', [x0 + swatch_w + 0.005, 0.001, 0.10, 0.020], ...
        'String', label_names{lab}, ...
        'Color', 'w', 'FontSize', 7, ...
        'EdgeColor', 'none', 'BackgroundColor', 'none', ...
        'VerticalAlignment', 'middle');
end

%% ── 5. Export ────────────────────────────────────────────────────────────

exportgraphics(fig, OUT_FILE, 'Resolution', 150, 'BackgroundColor', 'k');
fprintf('Saved: %s\n', OUT_FILE);
close(fig);
