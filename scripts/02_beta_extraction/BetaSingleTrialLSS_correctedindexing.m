%% BetaSingleTrialLSS_correctedindexing - Collect LSS betas by condition
% Reads beta_0001 from every trial-specific GLM and writes one voxels-by-
% trials matrix for each condition and subject. Trial folders are named in
% condition-block design order, so onset times are used to restore actual
% presentation order before matrices are assembled.
%
% Outputs:
%   Pl#.mat, Nt#.mat, Up#.mat, each containing 100 trial columns.

clear;
mainpath = 'N:\Experimental_Data\Max Lobel\IAPS\fMRI Analysis\Decoding Analysis\BetaS2LSS_Output';
maindir = dir(mainpath);

% Filter to only get beta_series directories
is_beta_series = false(length(maindir), 1);
for i = 1:length(maindir)
    if maindir(i).isdir && startsWith(maindir(i).name, 'beta_series_')
        is_beta_series(i) = true;
    end
end

sublist = {maindir(is_beta_series).name};
total_subjects = length(sublist);

behavdir = dir('N:\Experimental_Data\Ben Yin\IAPS\fMRI Analysis\Decoding Analysis\SingleTrialGLMs\NewStimuluesSetting\*.mat');

% Define output directory
output_dir = fullfile(mainpath, 'Reshaped Files Presentation Order');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('Found %d subjects to process\n', total_subjects);

for sub = 1:total_subjects
    subject = sublist{sub};
    subject_number = regexp(subject, '\d+$', 'match', 'once');
    
    if isempty(subject_number)
        error(['Could not extract subject number from: ', subject]);
    end
    
    disp(['Processing subject: ', subject, ' (', subject_number, ')']);
    
    % Initialize condition matrices
    Pl = [];
    Nt = [];
    Up = [];
    
    subject_path = fullfile(mainpath, subject);
    
    % Process each run
    for run = 1:5
        fprintf('  Processing run %d...\n', run);
        
        % The onset-file listing is assumed to contain five consecutive
        % entries per subject in run order.
        onset_file_idx = (str2double(subject_number) - 1) * 5 + run;
        load(fullfile('N:\Experimental_Data\Ke Bo\Project_IAPS\fMRI\NewStimuluesSetting', ...
                      behavdir(onset_file_idx).name));
        
        % Create trial information with design matrix indices
        trial_info = [];
        idx = 1;
        
        % Pleasant trials (design indices 1-20)
        for t = 1:20
            trial_info(idx).onset = Onset(t, 2);
            trial_info(idx).condition = 'Pl';
            trial_info(idx).design_idx = t;  % This is what the folder is named by
            idx = idx + 1;
        end
        
        % Neutral trials (design indices 21-40, but labeled as 21-40 in folders)
        for t = 1:20
            trial_info(idx).onset = Onset(t, 3);
            trial_info(idx).condition = 'Nt';
            trial_info(idx).design_idx = t + 20;  % Folders are named Nt21-Nt40
            idx = idx + 1;
        end
        
        % Unpleasant trials (design indices 41-60)
        for t = 1:20
            trial_info(idx).onset = Onset(t, 4);
            trial_info(idx).condition = 'Up';
            trial_info(idx).design_idx = t + 40;  % Folders are named Up41-Up60
            idx = idx + 1;
        end
        
        % Sort by onset time to get presentation order
        [~, sort_idx] = sort([trial_info.onset]);
        trial_info_ordered = trial_info(sort_idx);
        
        % Extract betas in presentation order
        for t = 1:60
            trial = trial_info_ordered(t);
            
            % Construct directory name as created in LSS script
            trial_dir_name = sprintf('%s%d_run%d', trial.condition, trial.design_idx, run);
            trial_dir = fullfile(subject_path, trial_dir_name);
            
            % Find the beta file
            beta_file = fullfile(trial_dir, 'beta_0001.nii');
            
            if ~exist(beta_file, 'file')
                warning(['Beta file not found: ', beta_file]);
                continue;
            end
            
            % Read the beta image
            vol_info = spm_vol(beta_file);
            img_data = spm_read_vols(vol_info);
            
            % Reshape to column vector
            img_vector = img_data(:);
            
            % Append to appropriate condition matrix (in presentation order)
            switch trial.condition
                case 'Pl'
                    Pl = [Pl, img_vector];
                case 'Nt'
                    Nt = [Nt, img_vector];
                case 'Up'
                    Up = [Up, img_vector];
            end
        end
    end
    
    % Display matrix dimensions
    fprintf('  Final matrix dimensions - Pl: %d, Nt: %d, Up: %d\n', ...
            size(Pl, 2), size(Nt, 2), size(Up, 2));
    
    if size(Pl, 2) ~= 100 || size(Nt, 2) ~= 100 || size(Up, 2) ~= 100
        warning('Unexpected number of trials for subject %s', subject);
    end
    
    % Save processed data
    save(fullfile(output_dir, ['Pl', subject_number, '.mat']), 'Pl');
    save(fullfile(output_dir, ['Nt', subject_number, '.mat']), 'Nt');
    save(fullfile(output_dir, ['Up', subject_number, '.mat']), 'Up');
    
    disp(['Finished processing subject: ', subject]);
end

disp('All subjects processed.');
disp(['Results saved to: ', output_dir]);
