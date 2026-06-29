%% BetaS2_fmriprep - Estimate LSA betas from fMRIPrep derivatives
% Uses fMRIPrep MNI152NLin2009cAsym res-2 output for 16 subjects.
% Trims first 6 dummy volumes, applies 8mm FWHM smoothing.
% Each subject is modeled as five SPM sessions with one regressor per trial.
% Inputs and output paths below are cluster-specific configuration.

clear;
spm('defaults','FMRI');
global defaults;

% -------------------------------------------------------------------------
% Paths
% -------------------------------------------------------------------------
fmriprep_dir = '/orange/ruogu.fang/pateld3/data/fmriprep';
tmp_base     = '/orange/ruogu.fang/pateld3/data/fmriprep-tmp';
betas_base   = '/orange/ruogu.fang/pateld3/data/fmriprep-betas';
onset_dir    = '/blue/ruogu.fang/bradystemac/mvpa_project/IAPS_raw_fMRI_20subjects/NewStimuluesSetting/NewStimuluesSetting/';
spm_path     = '/blue/ruogu.fang/bradystemac/spm12';
addpath(spm_path);

% -------------------------------------------------------------------------
% Subject mapping: 16 subjects, new index 1..16 -> original IDs
% (Sub-010..Sub-013 from the original 20 were excluded — no T1)
% -------------------------------------------------------------------------
ORIG_SUB_IDS = [1,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20];
N_SUBS  = 16;
N_RUNS  = 5;
TR      = 1.98;
N_SCANS = 206;

% -------------------------------------------------------------------------
% Main loop
% -------------------------------------------------------------------------
for i = 1:N_SUBS

    sid     = sprintf('%03d', i);
    orig_id = ORIG_SUB_IDS(i);
    fprintf('\n===== Subject sub-%s (orig %02d) =====\n', sid, orig_id);

    % Per-subject directories
    sub_fmriprep = fullfile(fmriprep_dir, ['sub-' sid], 'func');
    tmp_sub_dir  = fullfile(tmp_base, ['sub-' sid]);
    beta_out     = fullfile(betas_base, ['beta_series_sub-' sid]);
    mkdir(tmp_sub_dir);
    mkdir(beta_out);

    % Clear session-level SPM struct for this subject
    clear SPM;

    % SPM basis function parameters
    SPM.xBF.name     = 'hrf';
    SPM.xBF.length   = 32.0513;
    SPM.xBF.order    = 1;
    SPM.xBF.T        = 36;
    SPM.xBF.T0       = 18;
    SPM.xBF.UNITS    = 'scans';
    SPM.xBF.Volterra = 1;
    SPM.nscan        = ones(1, N_RUNS) * N_SCANS;
    SPM.xY.RT        = TR;
    SPM.xGX.iGXcalc  = 'Scaling';
    SPM.xVi.form     = 'AR(1)';

    tmp = cell(1, N_RUNS);

    for j = 1:N_RUNS

        run_label   = sprintf('%02d', j);
        tmp_run_dir = fullfile(tmp_sub_dir, ['run-' run_label]);
        mkdir(tmp_run_dir);

        % -----------------------------------------------------------------
        % 1. Locate fMRIPrep BOLD and confounds
        % -----------------------------------------------------------------
        bold_fname = sprintf('sub-%s_task-emotion_run-%s_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz', ...
                             sid, run_label);
        bold_gz    = fullfile(sub_fmriprep, bold_fname);

        confounds_tsv = fullfile(sub_fmriprep, ...
            sprintf('sub-%s_task-emotion_run-%s_desc-confounds_timeseries.tsv', sid, run_label));

        fprintf('Run %d: loading %s\n', j, bold_gz);

        % -----------------------------------------------------------------
        % 2. Gunzip to tmp run dir
        % -----------------------------------------------------------------
        gunzip(bold_gz, tmp_run_dir);
        bold_nii_name = strrep(bold_fname, '.gz', '');
        bold_nii      = fullfile(tmp_run_dir, bold_nii_name);

        % -----------------------------------------------------------------
        % 3. Normalize both possible derivative lengths to 206 scans
        % -----------------------------------------------------------------
        V      = spm_vol(bold_nii);
        n_vols = length(V);
        fprintf('  Volume count: %d\n', n_vols);

        if n_vols == 212
            n_skip = 6;
        elseif n_vols == N_SCANS
            n_skip = 0;
        else
            error('Unexpected volume count %d for sub-%s run-%s', n_vols, sid, run_label);
        end

        fprintf('  n_skip = %d\n', n_skip);

        if n_skip > 0
            V_trim       = V(n_skip+1:end);
            trimmed_name = ['trimmed_' bold_nii_name];
            trimmed_nii  = fullfile(tmp_run_dir, trimmed_name);
            spm_file_merge(V_trim, trimmed_nii, 0, TR);
            bold_for_smooth = trimmed_nii;
        else
            bold_for_smooth = bold_nii;
        end

        % -----------------------------------------------------------------
        % 4. Apply 8mm FWHM spatial smoothing
        % -----------------------------------------------------------------
        [~, smooth_base, smooth_ext] = fileparts(bold_for_smooth);
        smoothed_nii = fullfile(tmp_run_dir, ['s' smooth_base smooth_ext]);
        spm_smooth(bold_for_smooth, smoothed_nii, [8 8 8]);

        % Collect smoothed volume list for SPM
        [~, sname, ~]  = fileparts(smoothed_nii);
        smooth_pattern = ['^' sname '\.nii$'];
        tmp{j} = spm_select('ExtFPList', tmp_run_dir, smooth_pattern, Inf);
        fprintf('  Smoothed volumes collected: %d\n', size(tmp{j}, 1));

        % -----------------------------------------------------------------
        % 5. Load confounds (6 motion parameters)
        % -----------------------------------------------------------------
        T_conf = readtable(confounds_tsv, 'FileType', 'text', 'Delimiter', '\t');
        motion = [T_conf.trans_x, T_conf.trans_y, T_conf.trans_z, ...
                  T_conf.rot_x,   T_conf.rot_y,   T_conf.rot_z];
        motion = motion(n_skip+1:end, :);
        motion(isnan(motion)) = 0;
        fprintf('  Confounds shape after trim: [%d x %d]\n', size(motion,1), size(motion,2));

        SPM.Sess(j).C.C    = motion;
        SPM.Sess(j).C.name = {'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'};
        SPM.xX.K(j).HParam = 128;

        % -----------------------------------------------------------------
        % 6. Load onset file and build design matrix (60 regressors/run)
        % -----------------------------------------------------------------
        if orig_id >= 12
            onset_name = sprintf('Sub%02d_run%d.mat', orig_id, j);
        else
            onset_name = sprintf('Sub%02drun%d.mat', orig_id, j);
        end
        onset_file = fullfile(onset_dir, onset_name);
        fprintf('  Loading onsets: %s\n', onset_file);
        load(onset_file);   % loads variable: Onset [60 x 4]

        % Beta vector: col 2 = Pl, col 3 = Nt, col 4 = Up (20 trials each)
        Beta(1:20)  = Onset(1:20, 2);
        Beta(21:40) = Onset(1:20, 3);
        Beta(41:60) = Onset(1:20, 4);

        % Condition names
        condition_name = cell(1, 60);
        for Nam = 1:20
            condition_name{Nam}    = sprintf('Pl%d', Nam);
            condition_name{Nam+20} = sprintf('Nt%d', Nam);
            condition_name{Nam+40} = sprintf('Up%d', Nam);
        end

        % Build 60 single-trial regressors
        for c = 1:60
            SPM.Sess(j).U(c).name      = {condition_name{c}};
            SPM.Sess(j).U(c).ons       = Beta(c);
            SPM.Sess(j).U(c).dur       = 1.5152;
            SPM.Sess(j).U(c).P(1).name = 'none';
        end

        clear Beta Onset;

    end % run loop

    % ---------------------------------------------------------------------
    % 7. Estimate GLM
    % ---------------------------------------------------------------------
    SPM.xY.P = cat(1, tmp{:});
    cd(beta_out);
    SPM = spm_fmri_spm_ui(SPM);
    SPM = spm_spm(SPM);
    clear SPM;

    fprintf('Subject sub-%s done. Betas in: %s\n', sid, beta_out);

end % subject loop

fprintf('\nAll subjects complete.\n');
