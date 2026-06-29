%% ROI_Decoding_Parallelized_LIBSVM_Unbalanced - Debug random-fold decoder
% Diagnostic version of the LIBSVM decoder using non-stratified random folds.
% It intentionally processes only three subjects and two ROIs, runs serially,
% and prints fold diagnostics. Expand the subject/ROI lists and restore the
% parallel pool only after validating the data and LIBSVM installation.
%
% The historical method globally z-scores each ROI dataset before splitting.

clear; clc;
%% Parameters and paths
data_dir = 'N:\Experimental_Data\Max Lobel\IAPS\fMRI Analysis\Decoding Analysis\BetaS2LSS_Output\Reshaped Files Presentation Order';
roiDir = 'N:\Experimental_Data\Ke Bo\Project_IAPS\fMRI\roi\new\';
roiLabelFile = fullfile(roiDir, 'ROIfiles_Labeling.txt');
save_file = 'N:\Experimental_Data\Max Lobel\IAPS\fMRI Analysis\Decoding Analysis\Replicating Ke Bo 3B\Output\DecodingResults_LIBSVM_nostrat_debug_v5.mat';
K = 10;
NUM_REPEATS = 100;
subjects = 1:3; % Test with just first 3 subjects for debugging
roi_names = {'V1v','V1d'}; % Test with just 2 ROIs for debugging

% Start parallel pool (disable for debugging)
% if isempty(gcp('nocreate'))
%     parpool(18);
% end

%% Parse ROI label file
roi_name_to_num = containers.Map();
fid = fopen(roiLabelFile, 'r');
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if startsWith(line, '%')
        tokens = strsplit(strtrim(line(2:end)), '-');
        if numel(tokens) == 2
            roi_name = strtrim(tokens{2});
            roi_num = str2double(strtrim(tokens{1}));
            roi_name_to_num(roi_name) = roi_num;
        end
    end
end
fclose(fid);

%% Create subject-by-ROI job list
job_list = [];
job_counter = 1;
for subj = subjects
    pl_file = fullfile(data_dir, sprintf('Pl%d.mat', subj));
    nt_file = fullfile(data_dir, sprintf('Nt%d.mat', subj));
    up_file = fullfile(data_dir, sprintf('Up%d.mat', subj));
    
    if ~isfile(pl_file) || ~isfile(nt_file) || ~isfile(up_file)
        warning('Missing files for Subject %d. Skipping.', subj);
        continue;
    end
    
    for r = 1:length(roi_names)
        job_list(job_counter, :) = [subj, r];
        job_counter = job_counter + 1;
    end
end

fprintf('Processing %d participant-ROI combinations...\n', size(job_list, 1));

%% Decode jobs sequentially for debugging
results_cell = cell(size(job_list, 1), 1);

for job_idx = 1:size(job_list, 1)
    subj = job_list(job_idx, 1);
    r = job_list(job_idx, 2);
    roi = roi_names{r};
    
    fprintf('\n=== Processing Subject %d, ROI %s ===\n', subj, roi);
    
    % Load data
    pl_file = fullfile(data_dir, sprintf('Pl%d.mat', subj));
    nt_file = fullfile(data_dir, sprintf('Nt%d.mat', subj));
    up_file = fullfile(data_dir, sprintf('Up%d.mat', subj));
    
    Pl = load(pl_file).Pl;
    Nt = load(nt_file).Nt;
    Up = load(up_file).Up;
    
    % Build mask
    mask = [];
    if strcmp(roi, 'IPS')
        for k = 0:5
            roi_k = sprintf('IPS%d', k);
            if ~isKey(roi_name_to_num, roi_k), continue; end
            roi_num = roi_name_to_num(roi_k);
            lh_file = fullfile(roiDir, sprintf('maxprob_vol_lh_%d.nii', roi_num));
            rh_file = fullfile(roiDir, sprintf('maxprob_vol_rh_%d.nii', roi_num));
            if isfile(lh_file), lh = spm_read_vols(spm_vol(lh_file)) > 0; else, lh = []; end
            if isfile(rh_file), rh = spm_read_vols(spm_vol(rh_file)) > 0; else, rh = []; end
            combined = lh | rh;
            if isempty(mask), mask = combined; else, mask = mask | combined; end
        end
    else
        if ~isKey(roi_name_to_num, roi)
            fprintf('No ROI mapping for %s\n', roi);
            results_cell{job_idx} = struct('subj', subj, 'roi', roi, 'status', 'no_roi_mapping');
            continue;
        end
        roi_num = roi_name_to_num(roi);
        lh_file = fullfile(roiDir, sprintf('maxprob_vol_lh_%d.nii', roi_num));
        rh_file = fullfile(roiDir, sprintf('maxprob_vol_rh_%d.nii', roi_num));
        if ~isfile(lh_file) || ~isfile(rh_file)
            fprintf('Missing ROI files for %s\n', roi);
            results_cell{job_idx} = struct('subj', subj, 'roi', roi, 'status', 'missing_roi_files');
            continue;
        end
        lh = spm_read_vols(spm_vol(lh_file)) > 0;
        rh = spm_read_vols(spm_vol(rh_file)) > 0;
        mask = lh | rh;
    end
    
    if isempty(mask) || sum(mask(:)) == 0
        fprintf('Empty mask for %s\n', roi);
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, 'status', 'empty_mask');
        continue;
    end
    
    roi_mask = mask(:);
    fprintf('ROI mask has %d voxels\n', sum(roi_mask));
    
    Pl_roi = Pl(roi_mask, :)';
    Nt_roi = Nt(roi_mask, :)';
    Up_roi = Up(roi_mask, :)';
    
    % Remove fully-NaN voxels
    valid_voxels = ~(all(isnan(Pl_roi), 1) & all(isnan(Nt_roi), 1) & all(isnan(Up_roi), 1));
    fprintf('Valid voxels: %d/%d\n', sum(valid_voxels), length(valid_voxels));
    
    if sum(valid_voxels) == 0
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, 'status', 'no_valid_voxels');
        continue;
    end
    
    Pl_roi = Pl_roi(:, valid_voxels);
    Nt_roi = Nt_roi(:, valid_voxels);
    Up_roi = Up_roi(:, valid_voxels);
    
    % Prepare data
    data_plnt = [Pl_roi; Nt_roi];
    data_upnt = [Up_roi; Nt_roi];
    labels_plnt = [ones(100,1); -ones(100,1)];
    labels_upnt = [ones(100,1); -ones(100,1)];
    
    fprintf('Data shapes - Pl+Nt: %s, Up+Nt: %s\n', mat2str(size(data_plnt)), mat2str(size(data_upnt)));
    
    % Run SVM
    try
        fprintf('\n--- Pleasant vs Neutral ---\n');
        acc1 = svm_repeat_libsvm_random_debug(data_plnt, labels_plnt, K, NUM_REPEATS);
        fprintf('\n--- Unpleasant vs Neutral ---\n');
        acc2 = svm_repeat_libsvm_random_debug(data_upnt, labels_upnt, K, NUM_REPEATS);
        
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, ...
                                     'pleasant_vs_neutral', acc1, ...
                                     'unpleasant_vs_neutral', acc2, ...
                                     'status', 'success');
        
        fprintf('RESULTS: Pleasant vs Neutral = %.3f, Unpleasant vs Neutral = %.3f\n', acc1, acc2);
        
    catch ME
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, ...
                                     'status', 'error', 'error_msg', ME.message);
        fprintf('Error for Subject %d, ROI %s: %s\n', subj, roi, ME.message);
    end
end

% ------------------------------
% Debug Function
% ------------------------------
function avg_acc = svm_repeat_libsvm_random_debug(data, labels, k, repeats)
    % Remove NaN rows first
    valid_rows = ~any(isnan(data), 2);
    data = data(valid_rows, :);
    labels = labels(valid_rows);
    
    unique_labels = unique(labels);
    fprintf('  Data after NaN removal: %d samples, classes: %s\n', ...
            size(data,1), mat2str(unique_labels));
    
    if length(unique_labels) < 2
        fprintf('  ERROR: Less than 2 classes available\n');
        avg_acc = NaN;
        return;
    end
    
    % Check for constant features (will cause zscore to fail)
    data_std = std(data, 0, 1);
    constant_features = data_std == 0;
    if any(constant_features)
        fprintf('  WARNING: %d constant features found, removing them\n', sum(constant_features));
        data = data(:, ~constant_features);
    end
    
    % Standardize data
    if size(data, 2) > 0
        data = zscore(data);
    else
        fprintf('  ERROR: No features left after removing constants\n');
        avg_acc = NaN;
        return;
    end
    
    n_samples = size(data, 1);
    fold_size = floor(n_samples / k);
    
    fprintf('  Total samples: %d, Fold size: %d\n', n_samples, fold_size);
    
    if fold_size < 2
        fprintf('  ERROR: Fold size too small\n');
        avg_acc = NaN;
        return;
    end
    
    rng(1);
    accs = NaN(repeats, 1);
    
    % Just test the first repeat to see what's happening
    r = 1;
    fprintf('  Testing Repeat %d:\n', r);
    shuffled_indices = randperm(n_samples);
    acc = NaN(k, 1);
    
    for i = 1:k
        test_start = (i-1) * fold_size + 1;
        test_end = min(i * fold_size, n_samples);
        
        if i == k
            test_end = n_samples;
        end
        
        test_idx = shuffled_indices(test_start:test_end);
        train_idx = shuffled_indices(setdiff(1:n_samples, test_start:test_end));
        
        X_train = data(train_idx, :);
        Y_train = labels(train_idx);
        X_test = data(test_idx, :);
        Y_test = labels(test_idx);
        
        % Debug class distributions
        train_classes = unique(Y_train);
        test_classes = unique(Y_test);
        train_counts = [sum(Y_train == -1), sum(Y_train == 1)];
        test_counts = [sum(Y_test == -1), sum(Y_test == 1)];
        
        fprintf('    Fold %d: Train [-1,+1]=[%d,%d], Test [-1,+1]=[%d,%d]\n', ...
               i, train_counts(1), train_counts(2), test_counts(1), test_counts(2));
        
        if numel(unique(Y_train)) < 2 || isempty(X_test)
            fprintf('    Fold %d: SKIPPED - insufficient classes or empty test\n', i);
            continue;
        end
        
        try
            model = svmtrain(Y_train, X_train, '-t 0 -q');
            pred = svmpredict(Y_test, X_test, model, '-q');
            acc(i) = mean(pred == Y_test);
            fprintf('    Fold %d: Accuracy = %.3f\n', i, acc(i));
        catch ME
            fprintf('    Fold %d: ERROR - %s\n', i, ME.message);
            continue;
        end
    end
    
    valid_folds = ~isnan(acc);
    fprintf('    Valid folds: %d/%d\n', sum(valid_folds), k);
    if sum(valid_folds) > 0
        accs(r) = mean(acc(valid_folds));
        fprintf('    Repeat accuracy: %.3f\n', accs(r));
    else
        fprintf('    Repeat accuracy: NaN (no valid folds)\n');
    end
    
    % If first repeat worked, run all repeats
    if ~isnan(accs(1))
        fprintf('  First repeat worked, running remaining %d repeats...\n', repeats-1);
        for r = 2:repeats
            shuffled_indices = randperm(n_samples);
            acc = NaN(k, 1);
            
            for i = 1:k
                test_start = (i-1) * fold_size + 1;
                test_end = min(i * fold_size, n_samples);
                
                if i == k
                    test_end = n_samples;
                end
                
                test_idx = shuffled_indices(test_start:test_end);
                train_idx = shuffled_indices(setdiff(1:n_samples, test_start:test_end));
                
                X_train = data(train_idx, :);
                Y_train = labels(train_idx);
                X_test = data(test_idx, :);
                Y_test = labels(test_idx);
                
                if numel(unique(Y_train)) < 2 || isempty(X_test)
                    continue;
                end
                
                try
                    model = svmtrain(Y_train, X_train, '-t 0 -q');
                    pred = svmpredict(Y_test, X_test, model, '-q');
                    acc(i) = mean(pred == Y_test);
                catch
                    continue;
                end
            end
            
            valid_folds = ~isnan(acc);
            if sum(valid_folds) > 0
                accs(r) = mean(acc(valid_folds));
            end
        end
    end
    
    valid_repeats = ~isnan(accs);
    if sum(valid_repeats) > 0
        avg_acc = mean(accs(valid_repeats));
        fprintf('  Final average accuracy: %.3f (%d valid repeats)\n', avg_acc, sum(valid_repeats));
    else
        avg_acc = NaN;
        fprintf('  FINAL RESULT: NaN (no valid repeats)\n');
    end
end
