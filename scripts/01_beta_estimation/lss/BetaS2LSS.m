%% Legacy native-SPM least-squares-separate (LSS) beta estimation
% Fits one GLM per trial. The target trial receives its own regressor and all
% remaining trials share a nuisance regressor, producing beta_0001 for the
% target trial. Paths and subject ordering reflect the original workstation.

clear;
spm('defaults', 'FMRI');

global defaults

mainpath = dir('N:\Experimental_Data\Ke Bo\Project_IAPS\fMRI\code\fMRI\FMRIPreprocess_All\Newdata');

runs = {'run1', 'run2', 'run3', 'run4', 'run5'};
behavdir = dir('N:\Experimental_Data\Ke Bo\Project_IAPS\fMRI\NewStimuluesSetting\*.mat');
nscan = 206;
TR = 1.98;

% Define condition labels
condition_labels = [repmat("Pl", 1, 20), repmat("Nt", 1, 20), repmat("Up", 1, 20)];

for i = 1:20
    % Create participant-level output directory
    participant_dir = fullfile('N:\Experimental_Data\Max Lobel\IAPS\fMRI Analysis\Decoding Analysis\BetaS2LSS_Output', ['beta_series_', num2str(i)]);
    if ~exist(participant_dir, 'dir')
        mkdir(participant_dir);
    end
    
    for j = 1:5
        currDir = fullfile('N:\Experimental_Data\Ke Bo\Project_IAPS\fMRI\code\fMRI\FMRIPreprocess_All\Newdata', mainpath(2 * i + 1).name);
        
        % Load stimulus onset timings
        onsetDir = 'N:\Experimental_Data\Ke Bo\Project_IAPS\fMRI\NewStimuluesSetting';
        load(fullfile(onsetDir, behavdir((i - 1) * 5 + j).name));
        
        dirPath = fullfile(currDir, runs{j});
        tmp = spm_select('fplist', dirPath, '^swars.*\.img');
        
        if isempty(tmp)
            warning('No functional images found in %s', dirPath);
            continue;
        end

        % The onset matrix stores 20 trials per valence in columns 2-4.
        % Concatenation fixes design indices as Pl=1:20, Nt=21:40, Up=41:60.
        Beta = [Onset(1:20, 2); Onset(1:20, 3); Onset(1:20, 4)];

        % Load motion parameters
        motion_file = spm_select('fplist', dirPath, '^rp.*\.txt');
        if isempty(motion_file)
            warning('No motion parameters found in %s', dirPath);
            continue;
        end
        motion_params = load(motion_file);

        % --- TRUE LSS: Run a separate GLM for each trial ---
        for trial = 1:length(Beta)
            clear SPM;  % Reset SPM for each trial
            
            % Generate condition-specific folder with run number
            trial_condition = sprintf('%s%d_run%d', condition_labels(trial), trial, j); % Example: "Pl1_run1", "Nt3_run2"
            trial_dir = fullfile(participant_dir, trial_condition);
            if ~exist(trial_dir, 'dir')
                mkdir(trial_dir);
            end

            % SPM design setup
            SPM.xBF.name = 'hrf';
            SPM.xBF.length = 32.0513;
            SPM.xBF.order = 1;
            SPM.xBF.T = 36;
            SPM.xBF.T0 = 18;
            SPM.xBF.UNITS = 'scans';
            SPM.xBF.Volterra = 1;

            SPM.nscan = nscan;
            SPM.xGX.iGXcalc = 'Scaling';
            SPM.xX.K.HParam = 128;
            SPM.xVi.form = 'AR(1)';
            SPM.xY.P = tmp;
            SPM.xY.RT = TR;

            % Define regressor for the trial of interest
            SPM.Sess.U(1).name = {trial_condition};
            SPM.Sess.U(1).ons = Beta(trial);
            SPM.Sess.U(1).dur = 1.5152;
            SPM.Sess.U(1).P.name = 'none';

            % LSS isolates the target by collapsing the other 59 events into
            % one regressor rather than estimating all trials independently.
            other_trials = setdiff(1:length(Beta), trial);
            if ~isempty(other_trials)
                SPM.Sess.U(2).name = {'[REDACTED]'};
                SPM.Sess.U(2).ons = Beta(other_trials);
                SPM.Sess.U(2).dur = 1.5152;
                SPM.Sess.U(2).P.name = 'none';
            end

            % Add motion parameters
            SPM.Sess.C.C = motion_params;
            SPM.Sess.C.name = {'X', 'Y', 'Z', 'x', 'y', 'z'};

            % Estimate separate GLM for this trial
            cd(trial_dir);
            SPM = spm_fmri_spm_ui(SPM);
            SPM = spm_spm(SPM);

            % Save GLM for this trial in its corresponding folder
            save(fullfile(trial_dir, sprintf('SPM_run%d.mat', j)), 'SPM');
        end
    end
end
