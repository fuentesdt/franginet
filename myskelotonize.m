%% myskelotonize.m
%  Read manifest.csv, extract label==2 from each NIfTI in the 'label'
%  column, compute bwskel, and write skeletons to 'newdata/'.
%
%  Output paths mirror the input directory structure under newdata/:
%    <label_path>  →  newdata/<label_path_stem>_skel.nii.gz
%
%  Requires: Image Processing Toolbox (bwskel, R2019a+)

clear; clc;

MANIFEST  = 'manifest.csv';
OUT_DIR   = 'newdata';
LABEL_VAL = 2;

if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

%% ── 1. Read manifest and locate the 'label' column ──────────────────────
raw = readcell(MANIFEST, 'Delimiter', ',');
hdr = strtrim(raw(1,:));

label_col = find(strcmpi(hdr, 'label'));
if isempty(label_col)
    error('myskelotonize: no "label" column in %s.\nAvailable columns: %s', ...
        MANIFEST, strjoin(hdr, ', '));
end

data = raw(2:end, :);
N    = size(data, 1);
fprintf('Manifest: %d rows  |  label column index = %d\n\n', N, label_col);

%% ── 2. Process each row ──────────────────────────────────────────────────
n_done = 0;
for i = 1:N
    lbl_path = strtrim(char(data{i, label_col}));
    fprintf('[%d/%d]  %s\n', i, N, lbl_path);

    if ~isfile(lbl_path)
        fprintf('  SKIP: file not found.\n\n');
        continue;
    end

    %% Load
    info = niftiinfo(lbl_path);
    vol  = niftiread(lbl_path);

    binary = vol == LABEL_VAL;
    n_vox  = sum(binary(:));
    if n_vox == 0
        fprintf('  SKIP: no voxels with label=%d.\n\n', LABEL_VAL);
        continue;
    end
    fprintf('  label=%d voxels : %d\n', LABEL_VAL, n_vox);

    %% Skeletonise
    tic;
    skel    = bwskel(logical(binary));
    elapsed = toc;
    fprintf('  skeleton voxels : %d   (%.1f s)\n', sum(skel(:)), elapsed);

    %% Build output path: mirror directory structure under OUT_DIR
    [fdir, fname, fext] = fileparts(lbl_path);
    if strcmpi(fext, '.gz')
        [~, fname] = fileparts(fname);   % strip inner .nii → bare stem
    end
    out_subdir = fullfile(OUT_DIR, fdir);
    if ~exist(out_subdir, 'dir'), mkdir(out_subdir); end
    out_path = fullfile(out_subdir, [fname, '_skel.nii.gz']);

    %% Write — preserve spatial header, switch to uint8
    out_info              = info;
    out_info.Datatype     = 'uint8';
    out_info.BitsPerPixel = 8;
    out_info.Filename     = '';

    niftiwrite(uint8(skel), out_path(1:end-3), out_info, 'Compressed', true);
    fprintf('  saved : %s\n\n', out_path);
    n_done = n_done + 1;
end

fprintf('=== Done: %d / %d volumes skeletonised ===\n', n_done, N);
