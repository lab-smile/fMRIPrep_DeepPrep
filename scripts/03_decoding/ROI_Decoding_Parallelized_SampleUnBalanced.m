%% ROI_Decoding_Parallelized_SampleUnBalanced - Manual fitcsvm folds
% Runs both affective contrasts for every subject/ROI pair with MATLAB
% fitcsvm. Each repetition shuffles the two classes separately, then assigns
% equal-sized chunks from each class to a fold. Any remainder stays in the
% training set, matching the historical "unbalanced" implementation.
%
% Input matrices are voxels-by-trials; classifier matrices are transposed to
% trials-by-voxels. IPS0-IPS5 and both hemispheres are combined into one ROI.

clear; clc;
%% Parameters and paths
data_dir = 'N:\Experimental_Data\Max Lobel\IAPS\fMRI Analysis\Decoding Analysis\BetaS2LSS_Output\Reshaped Files Presentation Order';
roiDir = 'N:\Experimental_Data\Ke Bo\Project_IAPS\fMRI\roi\new\';
roiLabelFile = fullfile(roiDir, 'ROIfiles_Labeling.txt');
save_file = 'N:\Experimental_Data\Max Lobel\IAPS\fMRI Analysis\Decoding Analysis\Replicating Ke Bo 3B\Output\DecodingResults_FITCSVMLib_v5_nostrat.mat';
K = 10;
NUM_REPEATS = 100;
subjects = 1:20;
roi_names = {'V1v','V1d','V2v','V2d','V3v','V3d','hV4','VO1','VO2',...
             'PHC1','PHC2','hMT','LO2','LO1','V3b','V3a','IPS'};

% Start parallel pool
if isempty(gcp('nocreate'))
    parpool(16);
end

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

%% Decode jobs in parallel
results_cell = cell(size(job_list, 1), 1);

parfor job_idx = 1:size(job_list, 1)
    subj = job_list(job_idx, 1);
    r = job_list(job_idx, 2);
    roi = roi_names{r};
    
    fprintf('Processing Subject %d, ROI %s\n', subj, roi);
    
    % Load data
    pl_file = fullfile(data_dir, sprintf('Pl%d.mat', subj));
    nt_file = fullfile(data_dir, sprintf('Nt%d.mat', subj));
    up_file = fullfile(data_dir, sprintf('Up%d.mat', subj));
    
    Pl = load(pl_file).Pl;
    Nt = load(nt_file).Nt;
    Up = load(up_file).Up;
    
    % Build mask (same as before)
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
            results_cell{job_idx} = struct('subj', subj, 'roi', roi, 'status', 'no_roi_mapping');
            continue;
        end
        roi_num = roi_name_to_num(roi);
        lh_file = fullfile(roiDir, sprintf('maxprob_vol_lh_%d.nii', roi_num));
        rh_file = fullfile(roiDir, sprintf('maxprob_vol_rh_%d.nii', roi_num));
        if ~isfile(lh_file) || ~isfile(rh_file)
            results_cell{job_idx} = struct('subj', subj, 'roi', roi, 'status', 'missing_roi_files');
            continue;
        end
        lh = spm_read_vols(spm_vol(lh_file)) > 0;
        rh = spm_read_vols(spm_vol(rh_file)) > 0;
        mask = lh | rh;
    end
    
    if isempty(mask) || sum(mask(:)) == 0
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, 'status', 'empty_mask');
        continue;
    end
    
    roi_mask = mask(:);
    Pl_roi = Pl(roi_mask, :)';
    Nt_roi = Nt(roi_mask, :)';
    Up_roi = Up(roi_mask, :)';
    
    % Remove fully-NaN voxels
    valid_voxels = ~(all(isnan(Pl_roi), 1) & all(isnan(Nt_roi), 1) & all(isnan(Up_roi), 1));
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
    
    % Run SVM
    try
        acc1 = svm_repeat_manual_balance(data_plnt, labels_plnt, K, NUM_REPEATS);
        acc2 = svm_repeat_manual_balance(data_upnt, labels_upnt, K, NUM_REPEATS);
        
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, ...
                                     'pleasant_vs_neutral', acc1, ...
                                     'unpleasant_vs_neutral', acc2, ...
                                     'status', 'success');
    catch ME
        results_cell{job_idx} = struct('subj', subj, 'roi', roi, ...
                                     'status', 'error', 'error_msg', ME.message);
        fprintf('Error for Subject %d, ROI %s: %s\n', subj, roi, ME.message);
    end
end

% ------------------------------
% Reorganize results
% ------------------------------
results = struct();
for i = 1:length(results_cell)
    result = results_cell{i};
    if ~isfield(result, 'status') || ~strcmp(result.status, 'success')
        fprintf('Skipped Subject %d, ROI %s: %s\n', result.subj, result.roi, result.status);
        continue;
    end
    
    subj_id = sprintf('Subj%d', result.subj);
    if ~isfield(results, subj_id)
        results.(subj_id) = struct();
    end
    results.(subj_id).(result.roi).pleasant_vs_neutral = result.pleasant_vs_neutral;
    results.(subj_id).(result.roi).unpleasant_vs_neutral = result.unpleasant_vs_neutral;
end

save(save_file, 'results');
disp('Decoding complete.');

% ------------------------------
% Manual Balance Function
% ------------------------------
function avg_acc = svm_repeat_manual_balance(data, labels, k, repeats)
    % Remove NaN rows first
    valid_rows = ~any(isnan(data), 2);
    data = data(valid_rows, :);
    labels = labels(valid_rows);
    
    unique_labels = unique(labels);
    if length(unique_labels) < 2
        avg_acc = NaN;
        return;
    end
    
    rng(1);
    accs = NaN(repeats, 1);
    
    for r = 1:repeats
        % Manual balanced fold creation
        class1_idx = find(labels == unique_labels(1));
        class2_idx = find(labels == unique_labels(2));
        
        % Shuffle indices
        class1_idx = class1_idx(randperm(length(class1_idx)));
        class2_idx = class2_idx(randperm(length(class2_idx)));
        
        % Create balanced folds
        fold_size1 = floor(length(class1_idx) / k);
        fold_size2 = floor(length(class2_idx) / k);
        
        if fold_size1 == 0 || fold_size2 == 0
            continue;
        end
        
        acc = NaN(k, 1);
        
        for i = 1:k
            % Create test indices for this fold
            test_start1 = (i-1) * fold_size1 + 1;
            test_end1 = min(i * fold_size1, length(class1_idx));
            test_start2 = (i-1) * fold_size2 + 1;
            test_end2 = min(i * fold_size2, length(class2_idx));
            
            test_idx = [class1_idx(test_start1:test_end1); class2_idx(test_start2:test_end2)];
            train_idx = setdiff(1:length(labels), test_idx);
            
            X_train = data(train_idx, :);
            Y_train = labels(train_idx);
            X_test = data(test_idx, :);
            Y_test = labels(test_idx);
            
            if numel(unique(Y_train)) < 2 || isempty(X_test)
                continue;
            end
            
            try
                model = fitcsvm(X_train, Y_train, 'KernelFunction', 'linear', 'Standardize', true);
                pred = predict(model, X_test);
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
    
    valid_repeats = ~isnan(accs);
    if sum(valid_repeats) > 0
        avg_acc = mean(accs(valid_repeats));
    else
        avg_acc = NaN;
    end
end
