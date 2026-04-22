%% gapdriver.m
%  -----------------------------------------------------------------------
%  Batch driver: applies vessel_gap_filling to every image in a manifest
%  CSV and organises per-sample outputs under a single results directory.
%
%  Uses the same CSV_FILE and RESULT_MAT as tuneFrangi.m.
%
%  CSV format  (no header, paths relative to CSV directory):
%    col 1 — sample ID
%    col 2 — relative path to binary vessel mask  (used as GT for DSC)
%    col 3 — relative path to intensity image
%    col 4 — relative path to liver mask           (optional)
%
%  Output layout
%    OUT_DIR/
%      <id>/
%        vesselness.nii.gz
%        binary_vesselness.nii.gz
%        filled_mask.nii.gz
%        gap_report.mat
%        gap_markers.fcsv
%      summary.csv   — per-sample DSC and gap counts
%  -----------------------------------------------------------------------

clear; clc;

%% ── Configuration ────────────────────────────────────────────────────────
CSV_FILE   = fullfile('preprocessed', 'manifest.csv');
RESULT_MAT = 'tuneFrangi_result.mat';
OUT_DIR    = 'gap_results';

% vessel_gap_filling options passed to every sample (name-value pairs)
GAP_OPTS = { ...
    'result_mat',  RESULT_MAT, ...
    'normalize',   true,       ...
    'max_gap_mm',  8.0,        ...
    'conf_auto',   0.65,       ...
    'conf_review', 0.35        ...
};

%% ── Setup ────────────────────────────────────────────────────────────────
if ~exist(OUT_DIR, 'dir'), mkdir(OUT_DIR); end

csvDir = fileparts(CSV_FILE);
if isempty(csvDir), csvDir = '.'; end

T      = readcell(CSV_FILE, 'Delimiter', ',', 'NumHeaderLines', 0);
N      = size(T, 1);
hasROI = size(T, 2) >= 4;

fprintf('=== Gap filling driver: %d samples ===\n', N);
fprintf('    CSV        : %s\n', CSV_FILE);
fprintf('    result_mat : %s\n', RESULT_MAT);
fprintf('    output     : %s\n', OUT_DIR);
fprintf('    liver ROI  : %s\n\n', mat2str(hasROI));

%% ── Summary accumulators ─────────────────────────────────────────────────
sumFid = fopen(fullfile(OUT_DIR, 'summary.csv'), 'w');
fprintf(sumFid, 'id,dsc_pre,dsc_post,improvement,n_auto,n_review,n_reject,status\n');

ids       = cell(N,1);
dsc_pre   = nan(N,1);
dsc_post  = nan(N,1);
n_auto_v  = zeros(N,1);
n_review_v= zeros(N,1);
n_reject_v= zeros(N,1);
statuses  = cell(N,1);

%% ── Per-sample loop ──────────────────────────────────────────────────────
for i = 1:N
    sampleID  = T{i,1};
    idStr     = num2str(sampleID);
    mskRel    = strtrim(T{i,2});
    imgRel    = strtrim(T{i,3});

    imgPath = fullfile(csvDir, imgRel);
    mskPath = fullfile(csvDir, mskRel);

    sampleOutDir = fullfile(OUT_DIR, idStr);
    if ~exist(sampleOutDir, 'dir'), mkdir(sampleOutDir); end

    fprintf('[%d/%d]  ID=%s\n', i, N, idStr);

    % Build per-call options: add label_file for DSC, output_dir
    callOpts = [GAP_OPTS, {'label_file', mskPath, 'output_dir', sampleOutDir}];

    ids{i} = idStr;
    try
        vessel_gap_filling(imgPath, callOpts{:});

        % Load per-sample results from gap_report.mat
        rpt = load(fullfile(sampleOutDir, 'gap_report.mat'), ...
                   'gap_candidates', 'dsc_pre', 'dsc_post');

        dsc_pre(i)    = rpt.dsc_pre;
        dsc_post(i)   = rpt.dsc_post;

        actions       = {rpt.gap_candidates.action};
        n_auto_v(i)   = sum(strcmp(actions, 'auto_filled'));
        n_review_v(i) = sum(strcmp(actions, 'flagged_for_review'));
        n_reject_v(i) = sum(strcmp(actions, 'rejected'));
        statuses{i}   = 'ok';

    catch ME
        warning('gapdriver:sampleFailed', ...
            'Sample %s failed: %s', idStr, ME.message);
        statuses{i} = ['ERROR: ' ME.message];
    end

    % Write summary row (NaN shown as blank for failed samples)
    fprintf(sumFid, '%s,%.4f,%.4f,%.4f,%d,%d,%d,%s\n', ...
        idStr, dsc_pre(i), dsc_post(i), dsc_post(i)-dsc_pre(i), ...
        n_auto_v(i), n_review_v(i), n_reject_v(i), statuses{i});

    fprintf('\n');
end

fclose(sumFid);

%% ── Cohort summary table ─────────────────────────────────────────────────
ok = strcmp(statuses, 'ok');
SEP = repmat('-', 1, 68);
fprintf('\n%s\n', SEP);
fprintf('  %-8s  %-8s  %-8s  %-10s  %-6s  %-6s  %-6s\n', ...
        'ID', 'DSC_pre', 'DSC_post', 'Improve', 'Auto', 'Review', 'Reject');
fprintf('%s\n', SEP);
for i = 1:N
    if ok(i)
        fprintf('  %-8s  %-8.4f  %-8.4f  %-10.4f  %-6d  %-6d  %-6d\n', ...
            ids{i}, dsc_pre(i), dsc_post(i), dsc_post(i)-dsc_pre(i), ...
            n_auto_v(i), n_review_v(i), n_reject_v(i));
    else
        fprintf('  %-8s  %-8s  %-8s  %-10s  %s\n', ids{i}, ...
            'N/A', 'N/A', 'N/A', statuses{i});
    end
end
fprintf('%s\n', SEP);
if any(ok)
    fprintf('  %-8s  %-8.4f  %-8.4f  %-10.4f  %-6d  %-6d  %-6d\n', ...
        'MEAN', mean(dsc_pre(ok)), mean(dsc_post(ok)), ...
        mean(dsc_post(ok)-dsc_pre(ok)), ...
        sum(n_auto_v(ok)), sum(n_review_v(ok)), sum(n_reject_v(ok)));
end
fprintf('%s\n', SEP);
fprintf('  Completed: %d/%d   Failed: %d\n', sum(ok), N, sum(~ok));
fprintf('  Summary   : %s\n', fullfile(OUT_DIR, 'summary.csv'));
fprintf('=== Done ===\n');
