%% SingleTrialDecodingv3 - ROI decoding with subject-level null distributions
% Reproduces the single-trial MVPA stage for pleasant-vs-neutral and
% unpleasant-vs-neutral contrasts. For each subject and ROI, the script runs
% repeated stratified linear-SVM cross-validation and label-shuffle nulls.
%
% Inputs:
%   Pl#/Nt#/Up#.mat   voxels-by-trials condition matrices
%   roi_masks.mat     masks on the same voxel grid as the beta matrices
%
% Outputs include accuracy/null arrays, group summaries, and ROI boxplots.

%% Configuration
% Define file paths
data_dir = '/orange/ruogu.fang/pateld3/SPM_Preprocessed_fMRI_20Subjects/betas/extracted_betas';
roi_masks_file = '/orange/ruogu.fang/pateld3/SPM_Preprocessed_fMRI_20Subjects/betas/extracted_betas/roi_masks.mat'; % Precomputed ROI masks file
output_dir = '/orange/ruogu.fang/pateld3/SPM_Preprocessed_fMRI_20Subjects/single_mvpa_results/'; % <— Output folder

% Ensure output folder exists
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

% Load ROI masks
roi_masks = load(roi_masks_file);

% --- Combine IPS0–IPS5 into one IPS ROI if available ---
ips_parts = {'IPS0','IPS1','IPS2','IPS3','IPS4','IPS5'};
have_all = all(isfield(roi_masks, ips_parts));
if have_all
    m = false(size(roi_masks.(ips_parts{1})));
    for p = 1:numel(ips_parts)
        m = m | (roi_masks.(ips_parts{p}) > 0);
    end
    roi_masks.IPS = m;
end

% ROIs of interest (IPS unified)
roi_names_of_interest = {'whole_brain', 'V1v', 'V1d', 'V2v', 'V2d', 'V3v', 'V3d', 'hV4', ...
                         'VO1', 'VO2', 'PHC1', 'PHC2', 'hMT', 'LO1', 'LO2', 'V3a', 'V3b', 'IPS'};

num_subjects = 20;
k_folds = 5;
n_reps  = 100;
n_shuffles = 100;

PlNt_acc  = NaN(num_subjects, numel(roi_names_of_interest));
UpNt_acc  = NaN(num_subjects, numel(roi_names_of_interest));
PlNt_null = NaN(num_subjects, numel(roi_names_of_interest), n_shuffles);
UpNt_null = NaN(num_subjects, numel(roi_names_of_interest), n_shuffles);

%% ===================== MAIN LOOP =====================
for roi_idx = 1:numel(roi_names_of_interest)
    roi_name = roi_names_of_interest{roi_idx};
    fprintf('Processing ROI: %s\n', roi_name);

    % ROI mask
    use_whole_brain = strcmpi(roi_name, 'whole_brain');
    if ~use_whole_brain
        if ~isfield(roi_masks, roi_name)
            warning('ROI "%s" not found. Skipping...', roi_name);
            continue;
        end
        roi_mask = roi_masks.(roi_name)(:)>0;
    else
        roi_mask = [];
    end

    for subj = 1:num_subjects
        pl_file = fullfile(data_dir, sprintf('Pl%d.mat', subj));
        nt_file = fullfile(data_dir, sprintf('Nt%d.mat', subj));
        up_file = fullfile(data_dir, sprintf('Up%d.mat', subj));

        if ~isfile(pl_file) || ~isfile(nt_file) || ~isfile(up_file)
            warning('Missing files for Subject %d. Skipping...', subj);
            continue;
        end

        pl_trials = load(pl_file); pl_trials = pl_trials.Pl;
        nt_trials = load(nt_file); nt_trials = nt_trials.Nt;
        up_trials = load(up_file); up_trials = up_trials.Up;

        if ~use_whole_brain
            if numel(roi_mask) ~= size(pl_trials,1)
                roi_mask = roi_mask(:);
                if numel(roi_mask) ~= size(pl_trials,1)
                    warning('ROI mask size mismatch for %s, subj %d. Skipping...', roi_name, subj);
                    continue;
                end
            end
            pl_trials = pl_trials(roi_mask,:);
            nt_trials = nt_trials(roi_mask,:);
            up_trials = up_trials(roi_mask,:);
        end

        valid_voxels = ~(all(isnan(pl_trials),2) & all(isnan(nt_trials),2) & all(isnan(up_trials),2));
        pl_trials = pl_trials(valid_voxels,:);
        nt_trials = nt_trials(valid_voxels,:);
        up_trials = up_trials(valid_voxels,:);

        fprintf('Subj %d (%s): removed %d empty voxels, kept %d\n', ...
            subj, roi_name, sum(~valid_voxels), sum(valid_voxels));

        if any([isempty(pl_trials), isempty(nt_trials), isempty(up_trials)])
            warning('No valid voxels left for subj %d, %s', subj, roi_name);
            continue;
        end

        % Pleasant vs Neutral
        PlNt_data = [pl_trials, nt_trials]';
        PlNt_labels = [ones(size(pl_trials,2),1); -ones(size(nt_trials,2),1)];

        % Unpleasant vs Neutral
        UpNt_data = [up_trials, nt_trials]';
        UpNt_labels = [ones(size(up_trials,2),1); -ones(size(nt_trials,2),1)];

        if numel(unique(PlNt_labels))>1
            PlNt_acc(subj,roi_idx) = svm_kfold_repeated(PlNt_data, PlNt_labels, k_folds, n_reps);
        end
        if numel(unique(UpNt_labels))>1
            UpNt_acc(subj,roi_idx) = svm_kfold_repeated(UpNt_data, UpNt_labels, k_folds, n_reps);
        end

        % Null distributions: 100 label-shuffle runs per subject per ROI
        pn_null = zeros(1, n_shuffles);
        un_null = zeros(1, n_shuffles);
        for sh = 1:n_shuffles
            pn_labels_shuf = PlNt_labels(randperm(length(PlNt_labels)));
            un_labels_shuf = UpNt_labels(randperm(length(UpNt_labels)));
            if numel(unique(pn_labels_shuf))>1
                pn_null(sh) = svm_kfold_repeated(PlNt_data, pn_labels_shuf, k_folds, n_reps);
            end
            if numel(unique(un_labels_shuf))>1
                un_null(sh) = svm_kfold_repeated(UpNt_data, un_labels_shuf, k_folds, n_reps);
            end
        end
        PlNt_null(subj, roi_idx, :) = pn_null;
        UpNt_null(subj, roi_idx, :) = un_null;
    end
end

%% ===================== SUMMARY STATS =====================
mean_PN_acc = nanmean(PlNt_acc);
mean_UN_acc = nanmean(UpNt_acc);

sem_PN_acc = nanstd(PlNt_acc) ./ sqrt(sum(~isnan(PlNt_acc)));
sem_UN_acc = nanstd(UpNt_acc) ./ sqrt(sum(~isnan(UpNt_acc)));

for r = 1:numel(roi_names_of_interest)
    fprintf('%s - PvsN: %.3f ± %.3f | UvsN: %.3f ± %.3f\n', ...
        roi_names_of_interest{r}, mean_PN_acc(r), sem_PN_acc(r), ...
        mean_UN_acc(r), sem_UN_acc(r));
end

save(fullfile(output_dir, 'decoding_results_k5x100_v4.mat'), ...
     'PlNt_acc','UpNt_acc','PlNt_null','UpNt_null', ...
     'mean_PN_acc','sem_PN_acc','mean_UN_acc','sem_UN_acc','roi_names_of_interest','k_folds','n_reps','n_shuffles');

%% ===================== PLOTS =====================
% Generate a colormap for distinct ROI colors
colors = lines(numel(roi_names_of_interest)); % bright, distinct colors

% ----- Pleasant vs Neutral -----
figure('Name','Pleasant vs Neutral');
boxplot(PlNt_acc,'Labels',roi_names_of_interest,'Whisker',1.5);
title(sprintf('Pleasant vs Neutral (%dx%d)',k_folds,n_reps));
xlabel('ROI'); ylabel('Decoding accuracy (%)'); set(gca,'XTickLabelRotation',45);
ylim([.45 .8]); grid on;

% Apply box colors
h = findobj(gca,'Tag','Box');
for j = 1:length(h)
    patch(get(h(j),'XData'), get(h(j),'YData'), colors(length(h)-j+1,:), 'FaceAlpha', 0.8);
end
set(gca,'FontSize',12,'LineWidth',1.2);
savefig(fullfile(output_dir,'PlNtROI_k5x100_v4.fig'));

% ----- Unpleasant vs Neutral -----
figure('Name','Unpleasant vs Neutral');
boxplot(UpNt_acc,'Labels',roi_names_of_interest,'Whisker',1.5);
title(sprintf('Unpleasant vs Neutral (%dx%d)',k_folds,n_reps));
xlabel('ROI'); ylabel('Decoding accuracy (%)'); set(gca,'XTickLabelRotation',45);
ylim([.45 .8]); grid on;

% Apply box colors
h = findobj(gca,'Tag','Box');
for j = 1:length(h)
    patch(get(h(j),'XData'), get(h(j),'YData'), colors(length(h)-j+1,:), 'FaceAlpha', 0.8);
end
set(gca,'FontSize',12,'LineWidth',1.2);
savefig(fullfile(output_dir,'UpNtROI_k5x100_v4.fig'));

%% ===================== HELPER =====================
function macc = svm_kfold_repeated(data, labels, k, n_reps)
    %SVM_KFOLD_REPEATED Average repeated stratified linear-SVM accuracy.
    acc_all = NaN(n_reps,1);
    for rep = 1:n_reps
        cv = cvpartition(labels,'KFold',k);
        fold_acc = NaN(k,1);
        for f = 1:k
            tr = training(cv,f); te = test(cv,f);
            Xtr = data(tr,:); ytr = labels(tr);
            Xte = data(te,:); yte = labels(te);
            Xtr = Xtr(all(~isnan(Xtr),2),:); ytr = ytr(all(~isnan(Xtr),2));
            Xte = Xte(all(~isnan(Xte),2),:); yte = yte(all(~isnan(Xte),2));
            if numel(unique(ytr))<2 || isempty(Xte), continue; end
            model = fitcsvm(Xtr,ytr,'KernelFunction','linear','Standardize',true);
            if isempty(model.SupportVectors), continue; end
            yhat = predict(model,Xte);
            fold_acc(f) = mean(yhat==yte);
        end
        acc_all(rep) = mean(fold_acc,'omitnan');
    end
    macc = mean(acc_all,'omitnan');
end
