function ROI_Decoding_Complete_foldwise_zscore(classifier, fold_method, pipeline, confounds, cohort, opts)
% ROI_Decoding_Complete_foldwise_zscore  Unified ROI LSS decoding for SPM/DeepPrep/fMRIPrep.
%
% This script merges the original four decoding variants:
%   ROI_Decoding_Parallelized_LIBSVM.m
%   ROI_Decoding_Parallelized_LIBSVM_Unbalanced.m
%   ROI_Decoding_Parallelized_SampleBalanced.m
%   ROI_Decoding_Parallelized_SampleUnBalanced.m
% and uses the input/path conventions from:
%   ROI_Decoding_Parallelized.m
%   ROI_Decoding_LSS.m
%
% USAGE EXAMPLES
% -------------------------------------------------------------------------
% DeepPrep, 6 confounds, 16 participants, MATLAB fitcsvm, stratified folds:
%   ROI_Decoding_Complete_foldwise_zscore('fitcsvm','balanced','deepprep',6)
%
% fMRIPrep, 29 confounds, 16 participants, LIBSVM, non-stratified folds:
%   ROI_Decoding_Complete_foldwise_zscore('libsvm','unbalanced','fmriprep',29)
%
% SPM 6-conf, LIBSVM balanced, run BOTH cohorts: all20 and sub16:
%   ROI_Decoding_Complete_foldwise_zscore('libsvm','balanced','spm',6)
%
% SPM 6-conf, fitcsvm unbalanced, only original 16-subject subset:
%   ROI_Decoding_Complete_foldwise_zscore('fitcsvm','unbalanced','spm',6,'sub16')
%
% Optional override structure:
%   opts.data_root = '/orange/ruogu.fang/pateld3/data/LSS';
%   opts.spm_root  = '/orange/ruogu.fang/pateld3/data/LSS';
%   opts.out_dir   = '/orange/ruogu.fang/pateld3/data/LSS/lss_decoding_results';
%   opts.ncpu      = 16;
%   opts.K         = 10;
%   opts.repeats   = 100;
%   opts.use_parallel = true;
%   ROI_Decoding_Complete_foldwise_zscore('libsvm','balanced','deepprep',6,'sub16',opts)
%
% PARAMETERS
% -------------------------------------------------------------------------
% classifier  : 'fitcsvm' or 'libsvm'
% fold_method : 'balanced' or 'unbalanced'
%               fitcsvm/balanced   -> cvpartition(...,'Stratify',true)
%               fitcsvm/unbalanced -> manual per-class chunk folds, matching
%                                     SampleUnBalanced.m naming/logic
%               libsvm/balanced    -> manual balanced class folds + LIBSVM
%               libsvm/unbalanced  -> random non-stratified folds + LIBSVM
%               LIBSVM modes standardize inside each fold using training-only
%               mean/std, then apply those parameters to the held-out fold.
% pipeline    : 'deepprep', 'fmriprep', 'spm', or a full config string such
%               as 'deepprep_6conf'. Full config strings infer pipeline.
% confounds   : 6 or 29; ignored if pipeline already includes '_6conf'/'_29conf'
% cohort      : 'auto', 'sub16', or 'all20'
%               auto -> SPM runs both sub16 and all20; deepprep/fmriprep run sub16
% opts        : optional struct for path/runtime overrides
%
% DATA EXPECTED
% -------------------------------------------------------------------------
% Pl#/Nt#/Up#.mat files containing variables Pl, Nt, Up respectively.
% Matrices are expected as voxels x trials. ROI masks are NIfTI files named
% maxprob_vol_lh_#.nii and maxprob_vol_rh_#.nii, with ROIfiles_Labeling.txt.
%
% NOTES
% -------------------------------------------------------------------------
% Requires SPM for spm_vol/spm_read_vols. Requires Statistics and Machine
% Learning Toolbox for fitcsvm modes. Requires LIBSVM on path for libsvm modes.
% No significance/permutation testing is performed.

%% ----- Defaults / environment fallbacks -----
if nargin < 1 || isempty(classifier),  classifier  = getenv('LSS_CLASSIFIER'); end
if nargin < 2 || isempty(fold_method), fold_method = getenv('LSS_METHOD'); end
if nargin < 3 || isempty(pipeline),    pipeline    = getenv('LSS_PIPELINE'); end
if nargin < 4 || isempty(confounds)
    env_conf = getenv('LSS_CONFOUNDS');
    if isempty(env_conf), confounds = []; else, confounds = str2double(env_conf); end
end
if nargin < 5 || isempty(cohort), cohort = getenv('LSS_COHORT'); end
if nargin < 6 || isempty(opts), opts = struct(); end

if isempty(classifier),  classifier  = 'fitcsvm'; end
if isempty(fold_method), fold_method = 'balanced'; end
if isempty(pipeline),    pipeline    = 'deepprep'; end
if isempty(cohort),      cohort      = 'auto'; end
if isempty(confounds) || isnan(confounds), confounds = 6; end

classifier  = lower(strtrim(classifier));
fold_method = lower(strtrim(fold_method));
pipeline    = lower(strtrim(pipeline));
cohort      = lower(strtrim(cohort));

assert(any(strcmp(classifier, {'fitcsvm','libsvm'})), ...
    'classifier must be ''fitcsvm'' or ''libsvm''');
assert(any(strcmp(fold_method, {'balanced','unbalanced'})), ...
    'fold_method must be ''balanced'' or ''unbalanced''');
assert(any(strcmp(cohort, {'auto','sub16','all20'})), ...
    'cohort must be ''auto'', ''sub16'', or ''all20''');

%% ----- Runtime parameters -----
K           = get_opt(opts, 'K', 10);
NUM_REPEATS = get_opt(opts, 'repeats', 100);
use_parallel = get_opt(opts, 'use_parallel', true);

roi_names = get_opt(opts, 'roi_names', {'V1v','V1d','V2v','V2d','V3v','V3d','hV4','VO1','VO2', ...
             'PHC1','PHC2','hMT','LO2','LO1','V3b','V3a','IPS'});

%% ----- Resolve config, paths, and cohorts -----
[config, pipeline_family] = resolve_config(pipeline, confounds);
paths = resolve_paths(config, pipeline_family, opts);
cohort_subjects = containers.Map();
cohort_subjects('all20') = 1:20;
cohort_subjects('sub16') = [1 2 3 4 5 6 7 8 9 14 15 16 17 18 19 20];

if strcmp(pipeline_family, 'spm')
    if strcmp(cohort, 'auto')
        cohort_list = {'sub16','all20'};
    else
        cohort_list = {cohort};
    end
else
    assert(~strcmp(cohort, 'all20'), 'all20 is only valid for pipeline=''spm''.');
    cohort_list = {'sub16'};
    cohort_subjects('sub16') = 1:16;  % DeepPrep/fMRIPrep files are renumbered Pl1..Pl16.
end

if ~exist(paths.out_dir, 'dir'), mkdir(paths.out_dir); end

fprintf('=== ROI_Decoding_Complete_foldwise_zscore ===\n');
fprintf('classifier=%s | fold_method=%s | pipeline=%s | config=%s\n', ...
    classifier, fold_method, pipeline_family, config);
fprintf('data_dir : %s\nroiDir   : %s\nout_dir  : %s\n', ...
    paths.data_dir, paths.roiDir, paths.out_dir);

%% ----- Dependency checks -----
assert(exist('spm_vol', 'file') > 0 && exist('spm_read_vols', 'file') > 0, ...
    'SPM functions spm_vol/spm_read_vols must be on the MATLAB path.');
if strcmp(classifier, 'libsvm')
    assert(exist('svmtrain', 'file') > 0 && exist('svmpredict', 'file') > 0, ...
        'LIBSVM svmtrain/svmpredict must be on the MATLAB path for classifier=''libsvm''.');
else
    assert(exist('fitcsvm', 'file') > 0, ...
        'fitcsvm requires the Statistics and Machine Learning Toolbox.');
end

%% ----- Start pool -----
if use_parallel
    ncpu = get_opt(opts, 'ncpu', []);
    if isempty(ncpu)
        ncpu = str2double(getenv('SLURM_CPUS_PER_TASK'));
        if isnan(ncpu) || ncpu < 1, ncpu = 16; end
    end
    if isempty(gcp('nocreate'))
        try
            parpool(ncpu);
        catch ME
            warning('parpool failed (%s). Continuing; parfor may run serially.', ME.message);
        end
    end
end

%% ----- Parse ROI label file once -----
roiLabelFile = fullfile(paths.roiDir, 'ROIfiles_Labeling.txt');
roi_name_to_num = parse_roi_labels(roiLabelFile);

%% ----- Run requested cohort(s) -----
for ci = 1:numel(cohort_list)
    cohort_tag = cohort_list{ci};
    subjects = cohort_subjects(cohort_tag);
    save_file = fullfile(paths.out_dir, sprintf('DecodingResults_%s_%s_%s_%s.mat', ...
        config, classifier, fold_method, cohort_tag));

    fprintf('\n--- Cohort %s (%d subjects) -> %s ---\n', ...
        cohort_tag, numel(subjects), save_file);

    decode_cohort(subjects, save_file, paths.data_dir, paths.roiDir, ...
        roi_name_to_num, roi_names, classifier, fold_method, K, NUM_REPEATS, ...
        config, pipeline_family, cohort_tag);
end
end

%% ========================================================================
function decode_cohort(subjects, save_file, data_dir, roiDir, roi_name_to_num, ...
    roi_names, classifier, fold_method, K, NUM_REPEATS, config, pipeline_family, cohort_tag)
%DECODE_COHORT Run every available subject-by-ROI job for one cohort.

job_list = [];
job_counter = 1;
for subj = subjects
    pl_file = fullfile(data_dir, sprintf('Pl%d.mat', subj));
    nt_file = fullfile(data_dir, sprintf('Nt%d.mat', subj));
    up_file = fullfile(data_dir, sprintf('Up%d.mat', subj));
    if ~isfile(pl_file) || ~isfile(nt_file) || ~isfile(up_file)
        warning('Missing Pl/Nt/Up files for Subject %d. Skipping.', subj);
        continue;
    end
    for r = 1:numel(roi_names)
        job_list(job_counter, :) = [subj, r]; %#ok<AGROW>
        job_counter = job_counter + 1;
    end
end
fprintf('Processing %d participant-ROI combinations...\n', size(job_list, 1));

results_cell = cell(size(job_list, 1), 1);

parfor job_idx = 1:size(job_list, 1)
    subj = job_list(job_idx, 1);
    r = job_list(job_idx, 2);
    roi = roi_names{r};

    fprintf('Processing Subject %d, ROI %s\n', subj, roi);

    try
        Pl = load(fullfile(data_dir, sprintf('Pl%d.mat', subj))).Pl;
        Nt = load(fullfile(data_dir, sprintf('Nt%d.mat', subj))).Nt;
        Up = load(fullfile(data_dir, sprintf('Up%d.mat', subj))).Up;
    catch ME
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, ...
            'status', 'load_error', 'error_msg', ME.message);
        continue;
    end

    [mask, mask_status] = build_roi_mask(roi, roiDir, roi_name_to_num);
    if ~strcmp(mask_status, 'success')
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, 'status', mask_status);
        continue;
    end

    roi_mask = mask(:);
    if numel(roi_mask) ~= size(Pl, 1) || numel(roi_mask) ~= size(Nt, 1) || numel(roi_mask) ~= size(Up, 1)
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, ...
            'status', 'mask_size_mismatch', 'mask_voxels', numel(roi_mask), ...
            'pl_voxels', size(Pl, 1), 'nt_voxels', size(Nt, 1), ...
            'up_voxels', size(Up, 1));
        continue;
    end

    Pl_roi = Pl(roi_mask, :)';
    Nt_roi = Nt(roi_mask, :)';
    Up_roi = Up(roi_mask, :)';

    valid_voxels = ~(all(isnan(Pl_roi), 1) & all(isnan(Nt_roi), 1) & all(isnan(Up_roi), 1));
    if sum(valid_voxels) == 0
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, 'status', 'no_valid_voxels');
        continue;
    end
    Pl_roi = Pl_roi(:, valid_voxels);
    Nt_roi = Nt_roi(:, valid_voxels);
    Up_roi = Up_roi(:, valid_voxels);

    data_plnt   = [Pl_roi; Nt_roi];
    data_upnt   = [Up_roi; Nt_roi];
    labels_plnt = [ones(size(Pl_roi, 1), 1); -ones(size(Nt_roi, 1), 1)];
    labels_upnt = [ones(size(Up_roi, 1), 1); -ones(size(Nt_roi, 1), 1)];

    [acc1, clean_plnt, status_plnt, error_plnt] = run_decoder_safe( ...
        data_plnt, labels_plnt, classifier, fold_method, K, NUM_REPEATS);
    [acc2, clean_upnt, status_upnt, error_upnt] = run_decoder_safe( ...
        data_upnt, labels_upnt, classifier, fold_method, K, NUM_REPEATS);

    results_cell{job_idx} = struct('subj', subj, 'roi', roi, ...
        'pleasant_vs_neutral', acc1, 'unpleasant_vs_neutral', acc2, ...
        'pleasant_vs_neutral_status', status_plnt, ...
        'unpleasant_vs_neutral_status', status_upnt, ...
        'pleasant_vs_neutral_error_msg', error_plnt, ...
        'unpleasant_vs_neutral_error_msg', error_upnt, ...
        'nvox', min(clean_plnt.n_features, clean_upnt.n_features), ...
        'nvox_after_all_nan_filter', sum(valid_voxels), ...
        'nvox_pleasant_vs_neutral', clean_plnt.n_features, ...
        'nvox_unpleasant_vs_neutral', clean_upnt.n_features, ...
        'n_samples_pleasant_vs_neutral', clean_plnt.n_samples, ...
        'n_samples_unpleasant_vs_neutral', clean_upnt.n_samples, ...
        'n_pos_pleasant_vs_neutral', clean_plnt.n_pos, ...
        'n_neg_pleasant_vs_neutral', clean_plnt.n_neg, ...
        'n_pos_unpleasant_vs_neutral', clean_upnt.n_pos, ...
        'n_neg_unpleasant_vs_neutral', clean_upnt.n_neg, ...
        'n_pl', size(Pl_roi,1), ...
        'n_nt', size(Nt_roi,1), 'n_up', size(Up_roi,1), 'status', 'success');
end

results = struct();
skipped = {};
for i = 1:numel(results_cell)
    result = results_cell{i};
    if isempty(result), continue; end
    if ~isfield(result, 'status') || ~strcmp(result.status, 'success')
        skipped{end+1} = result; %#ok<AGROW>
        if isfield(result, 'subj') && isfield(result, 'roi')
            fprintf('Skipped Subject %d, ROI %s: %s\n', result.subj, result.roi, result.status);
        end
        continue;
    end
    subj_id = sprintf('Subj%d', result.subj);
    if ~isfield(results, subj_id), results.(subj_id) = struct(); end
    results.(subj_id).(result.roi).pleasant_vs_neutral   = result.pleasant_vs_neutral;
    results.(subj_id).(result.roi).unpleasant_vs_neutral = result.unpleasant_vs_neutral;
    results.(subj_id).(result.roi).pleasant_vs_neutral_status = result.pleasant_vs_neutral_status;
    results.(subj_id).(result.roi).unpleasant_vs_neutral_status = result.unpleasant_vs_neutral_status;
    results.(subj_id).(result.roi).pleasant_vs_neutral_error_msg = result.pleasant_vs_neutral_error_msg;
    results.(subj_id).(result.roi).unpleasant_vs_neutral_error_msg = result.unpleasant_vs_neutral_error_msg;
    results.(subj_id).(result.roi).nvox = result.nvox;
    results.(subj_id).(result.roi).nvox_after_all_nan_filter = result.nvox_after_all_nan_filter;
    results.(subj_id).(result.roi).nvox_pleasant_vs_neutral = result.nvox_pleasant_vs_neutral;
    results.(subj_id).(result.roi).nvox_unpleasant_vs_neutral = result.nvox_unpleasant_vs_neutral;
    results.(subj_id).(result.roi).n_samples_pleasant_vs_neutral = result.n_samples_pleasant_vs_neutral;
    results.(subj_id).(result.roi).n_samples_unpleasant_vs_neutral = result.n_samples_unpleasant_vs_neutral;
    results.(subj_id).(result.roi).n_pos_pleasant_vs_neutral = result.n_pos_pleasant_vs_neutral;
    results.(subj_id).(result.roi).n_neg_pleasant_vs_neutral = result.n_neg_pleasant_vs_neutral;
    results.(subj_id).(result.roi).n_pos_unpleasant_vs_neutral = result.n_pos_unpleasant_vs_neutral;
    results.(subj_id).(result.roi).n_neg_unpleasant_vs_neutral = result.n_neg_unpleasant_vs_neutral;
    results.(subj_id).(result.roi).n_pl = result.n_pl;
    results.(subj_id).(result.roi).n_nt = result.n_nt;
    results.(subj_id).(result.roi).n_up = result.n_up;
end

metadata = struct();
metadata.config = config;
metadata.pipeline_family = pipeline_family;
metadata.cohort = cohort_tag;
metadata.subjects = subjects;
metadata.roi_names = roi_names;
metadata.classifier = classifier;
metadata.fold_method = fold_method;
metadata.K = K;
metadata.NUM_REPEATS = NUM_REPEATS;
metadata.script_name = 'ROI_Decoding_Complete_foldwise_zscore';
metadata.clean_rows = ['feature-wise finite filtering before row cleanup; ', ...
    'nvox is min(final pleasant-vs-neutral features, final unpleasant-vs-neutral features)'];
metadata.contrast_status = ['each contrast stores status separately: ', ...
    'success, nan_accuracy, or decode_error'];
metadata.libsvm_standardization = 'training-fold-only zscore, applied to held-out fold';
metadata.date_saved = datestr(now);
metadata.skipped = skipped;

save(save_file, 'results', 'metadata');
fprintf('Decoding complete. Saved: %s\n', save_file);
end

%% ========================================================================
function [acc, clean_info] = run_decoder(data, labels, classifier, fold_method, k, repeats)
%RUN_DECODER Dispatch to the selected classifier and fold implementation.
if strcmp(classifier, 'libsvm')
    if strcmp(fold_method, 'balanced')
        [acc, clean_info] = svm_repeat_libsvm_balanced(data, labels, k, repeats);
    else
        [acc, clean_info] = svm_repeat_libsvm_unbalanced(data, labels, k, repeats);
    end
else
    if strcmp(fold_method, 'balanced')
        [acc, clean_info] = svm_repeat_fitcsvm_balanced(data, labels, k, repeats);
    else
        [acc, clean_info] = svm_repeat_fitcsvm_manual_unbalanced(data, labels, k, repeats);
    end
end
end

function [acc, clean_info, status, error_msg] = run_decoder_safe(data, labels, classifier, fold_method, k, repeats)
%RUN_DECODER_SAFE Record a failed ROI without aborting the full cohort.
try
    [acc, clean_info] = run_decoder(data, labels, classifier, fold_method, k, repeats);
    if isnan(acc)
        status = 'nan_accuracy';
    else
        status = 'success';
    end
    error_msg = '';
catch ME
    acc = NaN;
    clean_info = clean_info_from_raw(data, labels);
    status = 'decode_error';
    error_msg = ME.message;
end
end

%% ----- fitcsvm balanced: SampleBalanced.m methodology -----
function [avg_acc, clean_info] = svm_repeat_fitcsvm_balanced(data, labels, k, repeats)
%SVM_REPEAT_FITCSVM_BALANCED Use repeated stratified fitcsvm folds.
[data, labels, clean_info] = clean_rows(data, labels);
unique_labels = unique(labels);
if numel(unique_labels) < 2, avg_acc = NaN; return; end
class_counts = arrayfun(@(x) sum(labels == x), unique_labels);
if min(class_counts) < k, avg_acc = NaN; return; end

rng(1);
accs = NaN(repeats, 1);
for r = 1:repeats
    cv = cvpartition(labels, 'KFold', k, 'Stratify', true);
    acc = NaN(k, 1);
    for i = 1:k
        trainIdx = training(cv, i);
        testIdx  = test(cv, i);
        X_train = data(trainIdx, :); Y_train = labels(trainIdx);
        X_test  = data(testIdx, :);  Y_test  = labels(testIdx);
        if numel(unique(Y_train)) < 2 || isempty(X_test), continue; end
        [X_train, X_test, ok_features] = remove_constant_train_features(X_train, X_test);
        if ~ok_features, continue; end
        try
            model = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear', 'Standardize', true);
            pred = predict(model, X_test);
            acc(i) = mean(pred == Y_test);
        catch
            continue;
        end
    end
    if any(~isnan(acc)), accs(r) = mean(acc(~isnan(acc))); end
end
avg_acc = mean_or_nan(accs);
end

%% ----- fitcsvm unbalanced: SampleUnBalanced.m manual class chunk folds -----
function [avg_acc, clean_info] = svm_repeat_fitcsvm_manual_unbalanced(data, labels, k, repeats)
%SVM_REPEAT_FITCSVM_MANUAL_UNBALANCED Reproduce manual class-chunk folds.
[data, labels, clean_info] = clean_rows(data, labels);
unique_labels = unique(labels);
if numel(unique_labels) < 2, avg_acc = NaN; return; end

rng(1);
accs = NaN(repeats, 1);
for r = 1:repeats
    class1_idx = find(labels == unique_labels(1));
    class2_idx = find(labels == unique_labels(2));
    class1_idx = class1_idx(randperm(length(class1_idx)));
    class2_idx = class2_idx(randperm(length(class2_idx)));
    fold_size1 = floor(length(class1_idx) / k);
    fold_size2 = floor(length(class2_idx) / k);
    if fold_size1 == 0 || fold_size2 == 0, continue; end

    acc = NaN(k, 1);
    for i = 1:k
        test_start1 = (i-1) * fold_size1 + 1;
        test_end1   = min(i * fold_size1, length(class1_idx));
        test_start2 = (i-1) * fold_size2 + 1;
        test_end2   = min(i * fold_size2, length(class2_idx));
        test_idx = [class1_idx(test_start1:test_end1); class2_idx(test_start2:test_end2)];
        train_idx = setdiff(1:length(labels), test_idx);

        X_train = data(train_idx, :); Y_train = labels(train_idx);
        X_test  = data(test_idx, :);  Y_test  = labels(test_idx);
        if numel(unique(Y_train)) < 2 || isempty(X_test), continue; end
        [X_train, X_test, ok_features] = remove_constant_train_features(X_train, X_test);
        if ~ok_features, continue; end
        try
            model = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear', 'Standardize', true);
            pred = predict(model, X_test);
            acc(i) = mean(pred == Y_test);
        catch
            continue;
        end
    end
    if any(~isnan(acc)), accs(r) = mean(acc(~isnan(acc))); end
end
avg_acc = mean_or_nan(accs);
end

%% ----- LIBSVM balanced: ROI_Decoding_Parallelized_LIBSVM.m methodology -----
function [avg_acc, clean_info] = svm_repeat_libsvm_balanced(data, labels, k, repeats)
%SVM_REPEAT_LIBSVM_BALANCED Use balanced folds and train-only scaling.
[data, labels, clean_info] = clean_rows(data, labels);
unique_labels = unique(labels);
if numel(unique_labels) < 2, avg_acc = NaN; return; end
assert_libsvm_on_worker();
rng(1);
accs = NaN(repeats, 1);
for r = 1:repeats
    class1_idx = find(labels == unique_labels(1));
    class2_idx = find(labels == unique_labels(2));
    class1_idx = class1_idx(randperm(length(class1_idx)));
    class2_idx = class2_idx(randperm(length(class2_idx)));
    fold_size1 = floor(length(class1_idx) / k);
    fold_size2 = floor(length(class2_idx) / k);
    if fold_size1 == 0 || fold_size2 == 0, continue; end

    acc = NaN(k, 1);
    for i = 1:k
        test_start1 = (i-1) * fold_size1 + 1;
        test_end1   = min(i * fold_size1, length(class1_idx));
        test_start2 = (i-1) * fold_size2 + 1;
        test_end2   = min(i * fold_size2, length(class2_idx));
        test_idx = [class1_idx(test_start1:test_end1); class2_idx(test_start2:test_end2)];
        train_idx = setdiff(1:length(labels), test_idx);

        X_train = data(train_idx, :); Y_train = labels(train_idx);
        X_test  = data(test_idx, :);  Y_test  = labels(test_idx);
        if numel(unique(Y_train)) < 2 || isempty(X_test), continue; end

        [X_train, X_test, ok_scale] = standardize_train_apply_test(X_train, X_test);
        if ~ok_scale, continue; end

        try
            model = svmtrain(double(Y_train), double(X_train), '-t 0 -q'); %#ok<SVMTRAIN>
            pred = svmpredict(double(Y_test), double(X_test), model, '-q'); %#ok<SVMPREDICT>
            acc(i) = mean(pred == Y_test);
        catch
            continue;
        end
    end
    if any(~isnan(acc)), accs(r) = mean(acc(~isnan(acc))); end
end
avg_acc = mean_or_nan(accs);
end

%% ----- LIBSVM unbalanced: LIBSVM_Unbalanced.m random non-stratified folds -----
function [avg_acc, clean_info] = svm_repeat_libsvm_unbalanced(data, labels, k, repeats)
%SVM_REPEAT_LIBSVM_UNBALANCED Use random folds and train-only scaling.
[data, labels, clean_info] = clean_rows(data, labels);
if numel(unique(labels)) < 2, avg_acc = NaN; return; end
assert_libsvm_on_worker();

n_samples = size(data, 1);
fold_size = floor(n_samples / k);
if fold_size < 1, avg_acc = NaN; return; end

rng(1);
accs = NaN(repeats, 1);
for r = 1:repeats
    shuffled_indices = randperm(n_samples);
    acc = NaN(k, 1);
    for i = 1:k
        test_start = (i-1) * fold_size + 1;
        if i == k
            test_end = n_samples;
        else
            test_end = min(i * fold_size, n_samples);
        end
        test_idx = shuffled_indices(test_start:test_end);
        train_idx = shuffled_indices(setdiff(1:n_samples, test_start:test_end));

        X_train = data(train_idx, :); Y_train = labels(train_idx);
        X_test  = data(test_idx, :);  Y_test  = labels(test_idx);
        if numel(unique(Y_train)) < 2 || isempty(X_test), continue; end

        [X_train, X_test, ok_scale] = standardize_train_apply_test(X_train, X_test);
        if ~ok_scale, continue; end

        try
            model = svmtrain(double(Y_train), double(X_train), '-t 0 -q'); %#ok<SVMTRAIN>
            pred = svmpredict(double(Y_test), double(X_test), model, '-q'); %#ok<SVMPREDICT>
            acc(i) = mean(pred == Y_test);
        catch
            continue;
        end
    end
    if any(~isnan(acc)), accs(r) = mean(acc(~isnan(acc))); end
end
avg_acc = mean_or_nan(accs);
end

%% ========================================================================
function [data, labels, clean_info] = clean_rows(data, labels)
% Drop unstable voxels/features before dropping samples. fMRIPrep/DeepPrep
% beta images can contain sparse nonfinite voxels inside otherwise valid ROIs.
clean_info = struct();
clean_info.n_features_initial = size(data, 2);
clean_info.n_samples_initial = size(data, 1);

valid_features = all(isfinite(data), 1);
data = data(:, valid_features);

valid_rows = ~any(~isfinite(data), 2);
data = double(data(valid_rows, :));
labels = double(labels(valid_rows));

clean_info.n_features = size(data, 2);
clean_info.n_samples = size(data, 1);
clean_info.n_pos = sum(labels == 1);
clean_info.n_neg = sum(labels == -1);
clean_info.n_features_removed = clean_info.n_features_initial - clean_info.n_features;
clean_info.n_samples_removed = clean_info.n_samples_initial - clean_info.n_samples;
end

function clean_info = clean_info_from_raw(data, labels)
%CLEAN_INFO_FROM_RAW Recover diagnostics after a decoding exception.
clean_info = struct();
clean_info.n_features_initial = size(data, 2);
clean_info.n_samples_initial = size(data, 1);

valid_features = all(isfinite(data), 1);
data = data(:, valid_features);
valid_rows = ~any(~isfinite(data), 2);
labels = double(labels(valid_rows));

clean_info.n_features = size(data, 2);
clean_info.n_samples = sum(valid_rows);
clean_info.n_pos = sum(labels == 1);
clean_info.n_neg = sum(labels == -1);
clean_info.n_features_removed = clean_info.n_features_initial - clean_info.n_features;
clean_info.n_samples_removed = clean_info.n_samples_initial - clean_info.n_samples;
end

function [X_train_clean, X_test_clean, ok] = remove_constant_train_features(X_train, X_test)
% fitcsvm standardizes inside the model; remove fold-wise constant/invalid
% training features first so standardization cannot create NaNs.
ok = false;
if isempty(X_train) || isempty(X_test) || size(X_train, 2) == 0
    X_train_clean = [];
    X_test_clean = [];
    return;
end

X_train = double(X_train);
X_test = double(X_test);
sigma = std(X_train, 0, 1);
valid_features = isfinite(sigma) & sigma > 0 & ...
    all(isfinite(X_train), 1);

if ~any(valid_features)
    X_train_clean = [];
    X_test_clean = [];
    return;
end

X_train_clean = X_train(:, valid_features);
X_test_clean = X_test(:, valid_features);
ok = true;
end

function [X_train_z, X_test_z, ok] = standardize_train_apply_test(X_train, X_test)
% Training-fold-only standardization for LIBSVM.
% This avoids leakage from test samples that would occur with zscore(data)
% before cross-validation. Constant/invalid features are identified using
% training data only, removed from both train and test, and each held-out
% fold is scaled with the corresponding training mean/std.
ok = false;

if isempty(X_train) || isempty(X_test) || size(X_train, 2) == 0
    X_train_z = [];
    X_test_z = [];
    return;
end

X_train = double(X_train);
X_test = double(X_test);

mu = mean(X_train, 1);
sigma = std(X_train, 0, 1);
valid_features = isfinite(mu) & isfinite(sigma) & sigma > 0;

if ~any(valid_features)
    X_train_z = [];
    X_test_z = [];
    return;
end

X_train = X_train(:, valid_features);
X_test  = X_test(:, valid_features);
mu      = mu(valid_features);
sigma   = sigma(valid_features);

X_train_z = bsxfun(@rdivide, bsxfun(@minus, X_train, mu), sigma);
X_test_z  = bsxfun(@rdivide, bsxfun(@minus, X_test,  mu), sigma);

if any(~isfinite(X_train_z(:))) || any(~isfinite(X_test_z(:)))
    X_train_z = [];
    X_test_z = [];
    return;
end

ok = true;
end

function m = mean_or_nan(x)
%MEAN_OR_NAN Average valid repetitions without an omitnan dependency.
valid = ~isnan(x);
if any(valid), m = mean(x(valid)); else, m = NaN; end
end

function assert_libsvm_on_worker()
%ASSERT_LIBSVM_ON_WORKER Verify LIBSVM is visible inside parallel workers.
assert(exist('svmtrain', 'file') > 0 && exist('svmpredict', 'file') > 0, ...
    'LIBSVM svmtrain/svmpredict are not available on this MATLAB worker.');
end

%% ========================================================================
function [mask, status] = build_roi_mask(roi, roiDir, roi_name_to_num)
%BUILD_ROI_MASK Combine hemispheres and aggregate IPS0-IPS5 when requested.
mask = [];
status = 'success';
if strcmp(roi, 'IPS')
    for kk = 0:5
        roi_k = sprintf('IPS%d', kk);
        if ~isKey(roi_name_to_num, roi_k), continue; end
        roi_num = roi_name_to_num(roi_k);
        lh_file = fullfile(roiDir, sprintf('maxprob_vol_lh_%d.nii', roi_num));
        rh_file = fullfile(roiDir, sprintf('maxprob_vol_rh_%d.nii', roi_num));
        lh = [];
        rh = [];
        if isfile(lh_file), lh = spm_read_vols(spm_vol(lh_file)) > 0; end
        if isfile(rh_file), rh = spm_read_vols(spm_vol(rh_file)) > 0; end
        if isempty(lh) && isempty(rh), continue; end
        if isempty(lh), combined = rh; elseif isempty(rh), combined = lh; else, combined = lh | rh; end
        if isempty(mask), mask = combined; else, mask = mask | combined; end
    end
else
    if ~isKey(roi_name_to_num, roi)
        status = 'no_roi_mapping'; return;
    end
    roi_num = roi_name_to_num(roi);
    lh_file = fullfile(roiDir, sprintf('maxprob_vol_lh_%d.nii', roi_num));
    rh_file = fullfile(roiDir, sprintf('maxprob_vol_rh_%d.nii', roi_num));
    if ~isfile(lh_file) || ~isfile(rh_file)
        status = 'missing_roi_files'; return;
    end
    lh = spm_read_vols(spm_vol(lh_file)) > 0;
    rh = spm_read_vols(spm_vol(rh_file)) > 0;
    mask = lh | rh;
end
if isempty(mask) || sum(mask(:)) == 0
    status = 'empty_mask';
end
end

function roi_name_to_num = parse_roi_labels(roiLabelFile)
%PARSE_ROI_LABELS Map atlas names to numeric mask-file identifiers.
assert(isfile(roiLabelFile), 'ROI label file not found: %s', roiLabelFile);
roi_name_to_num = containers.Map();
fid = fopen(roiLabelFile, 'r');
assert(fid > 0, 'Could not open ROI label file: %s', roiLabelFile);
cleanupObj = onCleanup(@() fclose(fid)); %#ok<NASGU>
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if startsWith(line, '%')
        tokens = strsplit(strtrim(line(2:end)), '-');
        if numel(tokens) == 2
            roi_name = strtrim(tokens{2});
            roi_num = str2double(strtrim(tokens{1}));
            if ~isnan(roi_num)
                roi_name_to_num(roi_name) = roi_num;
            end
        end
    end
end
end

%% ========================================================================
function [config, pipeline_family] = resolve_config(pipeline, confounds)
%RESOLVE_CONFIG Normalize a pipeline family or fully qualified config name.
if contains(pipeline, '_')
    config = pipeline;
    if startsWith(config, 'deepprep')
        pipeline_family = 'deepprep';
    elseif startsWith(config, 'fmriprep')
        pipeline_family = 'fmriprep';
    elseif startsWith(config, 'spm')
        pipeline_family = 'spm';
    else
        error('Could not infer pipeline family from config: %s', config);
    end
else
    assert(any(strcmp(pipeline, {'deepprep','fmriprep','spm'})), ...
        'pipeline must be deepprep, fmriprep, spm, or a full config string.');
    pipeline_family = pipeline;
    config = sprintf('%s_%dconf', pipeline, confounds);
end
end

function paths = resolve_paths(config, pipeline_family, opts)
%RESOLVE_PATHS Select pipeline defaults and apply optional path overrides.
paths = struct();
if strcmp(pipeline_family, 'spm')
    spm_root = get_opt(opts, 'spm_root', '/orange/ruogu.fang/pateld3/data/LSS');
    paths.data_dir = get_opt(opts, 'data_dir', fullfile(spm_root, 'lss_extracted', config));
    paths.roiDir   = get_opt(opts, 'roiDir',   fullfile(spm_root, 'masks'));
    default_out    = fullfile(spm_root, 'results');
else
    data_root = get_opt(opts, 'data_root', '/orange/ruogu.fang/pateld3/data/LSS');
    paths.data_dir = get_opt(opts, 'data_dir', fullfile(data_root, 'lss_extracted', config));
    paths.roiDir   = get_opt(opts, 'roiDir',   fullfile(data_root, 'lss_roi_masks', config));
    default_out    = fullfile(data_root, 'lss_decoding_results');
end
paths.out_dir = get_opt(opts, 'out_dir', default_out);
end

function value = get_opt(opts, name, default_value)
%GET_OPT Return a nonempty structure field or its default.
if isstruct(opts) && isfield(opts, name) && ~isempty(opts.(name))
    value = opts.(name);
else
    value = default_value;
end
end
