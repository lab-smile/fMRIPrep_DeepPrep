function BetaLSS_spmspace(pipeline, confound_count)
% BETALSS_SPMSPACE Estimate LSS betas on the native SPM 3 mm grid.
%   BETALSS_SPMSPACE(PIPELINE, CONFOUND_COUNT) fits one GLM per trial for
%   each subject and run. PIPELINE is 'deepprep' or 'fmriprep';
%   CONFOUND_COUNT is 6 or 29.
%
%   With omitted inputs, LSS_PIPELINE and LSS_CONFOUNDS are read from the
%   environment. SLURM_ARRAY_TASK_ID restricts execution to one subject.
%
% -------------------------------------------------------------------------
% Input BOLD must already be conformed by conform_preproc_to_spm_space.sh:
%   <LSS_ROOT>/conformed_spmspace/<pipeline>/sub-XXX/func/
%   sub-XXX_task-emotion_run-YY_space-SPM3mm_desc-preproc_bold.nii.gz
%
% This keeps the original LSS model but estimates betas directly on the
% 53x63x46 @3mm grid used by the old SPM/Wang ROI masks. Confounds are still
% read from the original DeepPrep/fMRIPrep TSVs and trimmed to match 206 scans.
%
% Usage:
%   BetaLSS_spmspace('deepprep', 6)
%   BetaLSS_spmspace('fmriprep', 29)
% or set LSS_PIPELINE and LSS_CONFOUNDS in the SLURM environment.
% -------------------------------------------------------------------------

if nargin < 1 || isempty(pipeline), pipeline = getenv('LSS_PIPELINE'); end
if nargin < 2 || isempty(confound_count)
    confound_count = str2double(getenv('LSS_CONFOUNDS'));
end
if isempty(pipeline), pipeline = 'deepprep'; end
if isnan(confound_count) || isempty(confound_count), confound_count = 6; end

pipeline = lower(char(pipeline));
assert(any(strcmp(pipeline, {'deepprep','fmriprep'})), ...
    'pipeline must be deepprep or fmriprep');
assert(any(confound_count == [6 29]), 'confound_count must be 6 or 29');

spm('defaults','FMRI');
global defaults; %#ok<NUSED>

data_root = getenv_default('DATA_ROOT', '/orange/ruogu.fang/pateld3/data');
lss_root  = getenv_default('LSS_ROOT', fullfile(data_root, 'LSS'));
spm_path  = getenv_default('SPM_PATH', '/home/pateld3/spm');
addpath(spm_path);

config        = sprintf('%s_spmspace_%dconf', pipeline, confound_count);
conformed_dir = fullfile(lss_root, 'conformed_spmspace', pipeline);
tmp_base      = fullfile(lss_root, 'tmp_spmspace', config);
betas_base    = fullfile(lss_root, 'lss_betas_spmspace', config);
onset_dir     = getenv_default('ONSET_DIR', '/home/pateld3/NewStimuluesSetting');

ORIG_SUB_IDS = [1,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20];
N_SUBS  = 16;
N_RUNS  = 5;
TR      = 1.98;
N_SCANS = 206;
DUR     = 1.5152;
EXPECTED_DIM = [53 63 46];
condition_labels = [repmat({'Pl'},1,20), repmat({'Nt'},1,20), repmat({'Up'},1,20)];

arr = getenv('SLURM_ARRAY_TASK_ID');
if isempty(arr)
    sub_list = 1:N_SUBS;
else
    sub_list = str2double(arr);
    fprintf('SLURM array task: processing subject index %d only\n', sub_list);
end

fprintf('SPM-space LSS config=%s\n  conformed=%s\n  betas=%s\n', ...
    config, conformed_dir, betas_base);

for i = sub_list
    sid     = sprintf('%03d', i);
    orig_id = ORIG_SUB_IDS(i);
    fprintf('\n===== Subject sub-%s (orig %02d) =====\n', sid, orig_id);

    confound_func = original_func_dir(data_root, pipeline, sid);
    conformed_func = fullfile(conformed_dir, ['sub-' sid], 'func');
    tmp_sub_dir = fullfile(tmp_base, ['sub-' sid]);
    beta_out = fullfile(betas_base, ['beta_series_sub-' sid]);
    if ~exist(tmp_sub_dir,'dir'), mkdir(tmp_sub_dir); end
    if ~exist(beta_out,'dir'), mkdir(beta_out); end

    for run = 1:N_RUNS
        run_label = sprintf('%02d', run);
        tmp_run_dir = fullfile(tmp_sub_dir, ['run-' run_label]);
        if ~exist(tmp_run_dir,'dir'), mkdir(tmp_run_dir); end

        bold_fname = sprintf('sub-%s_task-emotion_run-%s_space-SPM3mm_desc-preproc_bold.nii.gz', ...
            sid, run_label);
        bold_gz = fullfile(conformed_func, bold_fname);
        assert(isfile(bold_gz), 'Missing conformed BOLD: %s', bold_gz);

        confounds_tsv = fullfile(confound_func, ...
            sprintf('sub-%s_task-emotion_run-%s_desc-confounds_timeseries.tsv', sid, run_label));
        assert(isfile(confounds_tsv), 'Missing confounds TSV: %s', confounds_tsv);

        fprintf('Run %d: loading %s\n', run, bold_gz);
        gunzip(bold_gz, tmp_run_dir);
        bold_nii_name = strrep(bold_fname, '.gz', '');
        bold_nii = fullfile(tmp_run_dir, bold_nii_name);

        V = spm_vol(bold_nii);
        assert(isequal(V(1).dim, EXPECTED_DIM), ...
            'Expected SPM-space BOLD dim [%d %d %d], got [%d %d %d] for %s', ...
            EXPECTED_DIM(1), EXPECTED_DIM(2), EXPECTED_DIM(3), ...
            V(1).dim(1), V(1).dim(2), V(1).dim(3), bold_nii);
        n_vols = length(V);
        if n_vols == N_SCANS
            n_skip_bold = 0;
        elseif n_vols == 212
            n_skip_bold = 6;
        else
            n_skip_bold = n_vols - N_SCANS;
            if n_skip_bold < 0 || n_skip_bold > 12
                error('Unexpected volume count %d for %s', n_vols, bold_nii);
            end
        end
        fprintf('  Volume count: %d; n_skip_bold=%d\n', n_vols, n_skip_bold);

        if n_skip_bold > 0
            V_trim = V(n_skip_bold+1:end);
            trimmed_nii = fullfile(tmp_run_dir, ['trimmed_' bold_nii_name]);
            spm_file_merge(V_trim, trimmed_nii, 0, TR);
            bold_for_smooth = trimmed_nii;
        else
            bold_for_smooth = bold_nii;
        end

        [~, smooth_base, smooth_ext] = fileparts(bold_for_smooth);
        smoothed_nii = fullfile(tmp_run_dir, ['s' smooth_base smooth_ext]);
        spm_smooth(bold_for_smooth, smoothed_nii, [8 8 8]);
        [~, sname, ~] = fileparts(smoothed_nii);
        run_vols = spm_select('ExtFPList', tmp_run_dir, ['^' sname '\.nii$'], Inf);
        fprintf('  Smoothed volumes collected: %d\n', size(run_vols,1));
        assert(size(run_vols,1) == N_SCANS, 'Expected %d smoothed scans.', N_SCANS);
        V_smooth = spm_vol(deblank(run_vols(1,:)));
        assert(isequal(V_smooth.dim, EXPECTED_DIM), ...
            'Expected smoothed SPM-space dim [%d %d %d], got [%d %d %d] for %s', ...
            EXPECTED_DIM(1), EXPECTED_DIM(2), EXPECTED_DIM(3), ...
            V_smooth.dim(1), V_smooth.dim(2), V_smooth.dim(3), deblank(run_vols(1,:)));

        T_conf = readtable(confounds_tsv, 'FileType','text', 'Delimiter','\t');
        n_skip_conf = height(T_conf) - N_SCANS;
        if n_skip_conf < 0 || n_skip_conf > 12
            error('Unexpected confound rows %d for %s', height(T_conf), confounds_tsv);
        end
        [confounds, confound_names] = build_confounds(T_conf, confound_count, n_skip_conf, N_SCANS);
        fprintf('  Confounds shape after trim: [%d x %d]\n', size(confounds,1), size(confounds,2));

        if orig_id >= 12
            onset_name = sprintf('Sub%02d_run%d.mat', orig_id, run);
        else
            onset_name = sprintf('Sub%02drun%d.mat', orig_id, run);
        end
        load(fullfile(onset_dir, onset_name)); %#ok<LOAD>
        Beta = [Onset(1:20,2); Onset(1:20,3); Onset(1:20,4)]; %#ok<NODEF>
        clear Onset;

        for trial = 1:60
            clear SPM;
            trial_name = sprintf('%s%d_run%d', condition_labels{trial}, trial, run);
            trial_dir = fullfile(beta_out, trial_name);
            if ~exist(trial_dir,'dir'), mkdir(trial_dir); end

            SPM.xBF.name     = 'hrf';
            SPM.xBF.length   = 32.0513;
            SPM.xBF.order    = 1;
            SPM.xBF.T        = 36;
            SPM.xBF.T0       = 18;
            SPM.xBF.UNITS    = 'scans';
            SPM.xBF.Volterra = 1;
            SPM.nscan        = N_SCANS;
            SPM.xGX.iGXcalc  = 'Scaling';
            SPM.xX.K.HParam  = 128;
            SPM.xVi.form     = 'AR(1)';
            SPM.xY.P         = run_vols;
            SPM.xY.RT        = TR;

            SPM.Sess.U(1).name   = {trial_name};
            SPM.Sess.U(1).ons    = Beta(trial);
            SPM.Sess.U(1).dur    = DUR;
            SPM.Sess.U(1).P.name = 'none';

            other = setdiff(1:60, trial);
            SPM.Sess.U(2).name   = {'other_trials'};
            SPM.Sess.U(2).ons    = Beta(other);
            SPM.Sess.U(2).dur    = DUR;
            SPM.Sess.U(2).P.name = 'none';

            SPM.Sess.C.C    = confounds;
            SPM.Sess.C.name = confound_names;

            cd(trial_dir);
            SPM = spm_fmri_spm_ui(SPM);
            SPM = spm_spm(SPM);
        end

        clear Beta;
        fprintf('  Run %d done: 60 trial GLMs estimated.\n', run);
    end

    fprintf('Subject sub-%s done. Betas in: %s\n', sid, beta_out);
end

fprintf('\nAll requested subjects complete for %s.\n', config);
end

function path_value = getenv_default(name, default_value)
%GETENV_DEFAULT Return an environment value or a configured fallback.
path_value = getenv(name);
if isempty(path_value), path_value = default_value; end
end

function func_dir = original_func_dir(data_root, pipeline, sid)
%ORIGINAL_FUNC_DIR Locate native derivative confounds for one subject.
if strcmp(pipeline, 'deepprep')
    func_dir = fullfile(data_root, 'deepprep', 'BOLD', ['sub-' sid], 'func');
else
    func_dir = fullfile(data_root, 'fmriprep', ['sub-' sid], 'func');
end
end

function [confounds, confound_names] = build_confounds(T_conf, confound_count, n_skip, n_scans)
%BUILD_CONFOUNDS Select, trim, and sanitize the requested nuisance model.
if confound_count == 6
    % The compact model contains rigid-body translations and rotations.
    confounds = [T_conf.trans_x, T_conf.trans_y, T_conf.trans_z, ...
                 T_conf.rot_x,   T_conf.rot_y,   T_conf.rot_z];
    confound_names = {'trans_x','trans_y','trans_z','rot_x','rot_y','rot_z'};
else
    % The expanded model adds motion derivatives/squares and five aCompCor
    % components, yielding 29 nuisance regressors in total.
    hmp24 = [T_conf.trans_x,        T_conf.trans_x_derivative1, ...
             T_conf.trans_x_power2, T_conf.trans_x_derivative1_power2, ...
             T_conf.trans_y,        T_conf.trans_y_derivative1, ...
             T_conf.trans_y_power2, T_conf.trans_y_derivative1_power2, ...
             T_conf.trans_z,        T_conf.trans_z_derivative1, ...
             T_conf.trans_z_power2, T_conf.trans_z_derivative1_power2, ...
             T_conf.rot_x,          T_conf.rot_x_derivative1, ...
             T_conf.rot_x_power2,   T_conf.rot_x_derivative1_power2, ...
             T_conf.rot_y,          T_conf.rot_y_derivative1, ...
             T_conf.rot_y_power2,   T_conf.rot_y_derivative1_power2, ...
             T_conf.rot_z,          T_conf.rot_z_derivative1, ...
             T_conf.rot_z_power2,   T_conf.rot_z_derivative1_power2];
    acompcor = [T_conf.a_comp_cor_00, T_conf.a_comp_cor_01, T_conf.a_comp_cor_02, ...
                T_conf.a_comp_cor_03, T_conf.a_comp_cor_04];
    confounds = [hmp24, acompcor];
    confound_names = {'trans_x','trans_x_d1','trans_x_sq','trans_x_d1sq', ...
                      'trans_y','trans_y_d1','trans_y_sq','trans_y_d1sq', ...
                      'trans_z','trans_z_d1','trans_z_sq','trans_z_d1sq', ...
                      'rot_x','rot_x_d1','rot_x_sq','rot_x_d1sq', ...
                      'rot_y','rot_y_d1','rot_y_sq','rot_y_d1sq', ...
                      'rot_z','rot_z_d1','rot_z_sq','rot_z_d1sq', ...
                      'aCC00','aCC01','aCC02','aCC03','aCC04'};
end

confounds = confounds(n_skip+1:end, :);
confounds(isnan(confounds)) = 0;
assert(size(confounds,1) == n_scans, 'Confounds do not match %d scans.', n_scans);
end
