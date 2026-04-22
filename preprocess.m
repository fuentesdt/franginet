%% preprocess.m
%  -----------------------------------------------------------------------
%  Preprocess liver/vessel NIfTI pairs for use with tuneFrangi.m.
%
%  Input CSV  (fullpath.csv, no header):
%    col 1 — sample ID
%    col 2 — full path to label mask  (.nii or .nii.gz)
%             label=0 background, label=1 liver, label=2 blood vessels
%    col 3 — full path to intensity image (.nii or .nii.gz)
%
%  Processing per sample:
%    1. Load image and label.
%    2. Find bounding box of label > 0 (liver + vessel union).
%    3. Crop image and label to bounding box — original intensities preserved.
%    4. Extract binary vessel mask  (label == 2 → 1, all else → 0).
%    5. Update NIfTI spatial transform to reflect crop offset.
%    6. Write cropped image and binary vessel mask as .nii.gz.
%
%  Output CSV  (OUT_DIR/manifest.csv, same column order as tuneFrangi.m):
%    col 1 — sample ID
%    col 2 — relative path to binary vessel mask
%    col 3 — relative path to cropped image
%  -----------------------------------------------------------------------

clear; clc;

%% ── Configuration ────────────────────────────────────────────────────────
IN_CSV  = 'fullpath.csv';    % full-path manifest (input)
OUT_DIR = 'preprocessed';    % root output directory
IMG_SUBDIR = 'images';
MSK_SUBDIR = 'masks';

%% ── Directory setup ──────────────────────────────────────────────────────
imgOutDir = fullfile(OUT_DIR, IMG_SUBDIR);
mskOutDir = fullfile(OUT_DIR, MSK_SUBDIR);
for d = {OUT_DIR, imgOutDir, mskOutDir}
    if ~exist(d{1}, 'dir'), mkdir(d{1}); end
end

outCsvPath = fullfile(OUT_DIR, 'manifest.csv');
fid = fopen(outCsvPath, 'w');

%% ── Read input CSV ───────────────────────────────────────────────────────
fprintf('Reading %s ...\n', IN_CSV);
T = readcell(IN_CSV, 'Delimiter', ',', 'NumHeaderLines', 0);
N = size(T, 1);
fprintf('  %d samples found.\n\n', N);

%% ── Process each sample ──────────────────────────────────────────────────
for i = 1:N
    sampleID  = T{i,1};
    labelPath = strtrim(T{i,2});
    imagePath = strtrim(T{i,3});

    fprintf('[%d/%d] ID=%s\n', i, N, num2str(sampleID));
    fprintf('  image : %s\n', imagePath);
    fprintf('  label : %s\n', labelPath);

    %% Load volumes
    imgInfo = niftiinfo(imagePath);
    img     = niftiread(imgInfo);          % keep original type / values

    lblInfo = niftiinfo(labelPath);
    lbl     = niftiread(lblInfo);          % integer label map

    assert(isequal(size(img), size(lbl)), ...
        'Image and label sizes differ for sample %s (%s vs %s).', ...
        num2str(sampleID), mat2str(size(img)), mat2str(size(lbl)));

    sz = size(img);

    %% Bounding box of label > 0  (liver + vessel union)
    roi = lbl > 0;
    [rx, ry, rz] = ind2sub(sz, find(roi));

    if isempty(rx)
        warning('preprocess:emptyLabel', ...
            'Sample %s has no foreground voxels — skipping.', num2str(sampleID));
        continue
    end

    x1 = min(rx);  x2 = max(rx);
    y1 = min(ry);  y2 = max(ry);
    z1 = min(rz);  z2 = max(rz);

    fprintf('  BB : [%d:%d, %d:%d, %d:%d]  (size %dx%dx%d)\n', ...
            x1,x2, y1,y2, z1,z2, x2-x1+1, y2-y1+1, z2-z1+1);

    %% Crop image and label to bounding box
    imgCrop = img(x1:x2, y1:y2, z1:z2);   % original intensities preserved
    lblCrop = lbl(x1:x2, y1:y2, z1:z2);

    %% Binary vessel mask  (label == 2)
    vesselMask = uint8(lblCrop == 2);
    fprintf('  Vessel voxels: %d (%.1f%% of BB)\n', ...
            sum(vesselMask(:)), 100*mean(vesselMask(:)));

    %% Update NIfTI spatial transform for crop offset
    imgInfoCrop = update_nifti_info(imgInfo, [x1 y1 z1], size(imgCrop));
    lblInfoCrop = update_nifti_info(lblInfo, [x1 y1 z1], size(vesselMask));

    %% Output filenames
    idStr   = sprintf('%s', num2str(sampleID));
    imgName = sprintf('image_%s.nii.gz',  idStr);
    mskName = sprintf('mask_%s.nii.gz',   idStr);

    imgOutPath = fullfile(imgOutDir, imgName);
    mskOutPath = fullfile(mskOutDir, mskName);

    %% Write NIfTIs
    niftiwrite(imgCrop,    imgOutPath, imgInfoCrop, 'Compressed', true);
    niftiwrite(vesselMask, mskOutPath, lblInfoCrop, 'Compressed', true);

    fprintf('  -> %s\n', imgOutPath);
    fprintf('  -> %s\n\n', mskOutPath);

    %% Write CSV row: id, relative_mask, relative_image
    relImg = [IMG_SUBDIR '/' imgName];
    relMsk = [MSK_SUBDIR '/' mskName];
    fprintf(fid, '%s,%s,%s\n', num2str(sampleID), relMsk, relImg);
end

fclose(fid);
fprintf('=== Done. Manifest written to %s ===\n', outCsvPath);

% =========================================================================
% LOCAL FUNCTIONS
% =========================================================================

function new_info = update_nifti_info(info, bb_start, new_size)
%UPDATE_NIFTI_INFO  Adjust NIfTI header after cropping.
%
%   bb_start  — [i1 j1 k1] 1-based voxel index of the crop origin
%   new_size  — [ni nj nk] dimensions of the cropped volume
%
%   The spatial transform is updated so that the new voxel [1,1,1]
%   maps to the world coordinate of the original voxel bb_start.
%   Voxel spacing and orientation are unchanged.

new_info = info;
new_info.ImageSize = new_size(:)';

% affine3d.T convention (MATLAB): world = voxel_0based * T(1:3,1:3) + T(4,1:3)
% where voxel_0based = voxel_1based - 1.
% New origin = world coord of (bb_start - 1) in 0-based coords.
T       = info.Transform.T;                  % 4×4
offset  = double(bb_start(:)') - 1;         % 0-based offset [1×3]
new_T   = T;
new_T(4,1:3) = offset * T(1:3,1:3) + T(4,1:3);
new_info.Transform = affine3d(new_T);
end
