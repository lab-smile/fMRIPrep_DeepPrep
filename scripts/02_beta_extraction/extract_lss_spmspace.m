function extract_lss_spmspace(pipeline, confound_count)
% EXTRACT_LSS_SPMSPACE Collect trial betas into condition matrices.
%   EXTRACT_LSS_SPMSPACE(PIPELINE, CONFOUND_COUNT) reads trial-specific
%   beta_0001 images and saves Pl#/Nt#/Up#.mat for decoding. PIPELINE is
%   'deepprep' or 'fmriprep'; CONFOUND_COUNT is 6 or 29.
%
%   Omitted inputs fall back to LSS_PIPELINE and LSS_CONFOUNDS.
% Output matrices are voxels x trials and should have 53*63*46 rows when the
% conformed beta branch is on the native SPM 3 mm grid.

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

data_root = getenv_default('DATA_ROOT', '/orange/ruogu.fang/pateld3/data');
lss_root  = getenv_default('LSS_ROOT', fullfile(data_root, 'LSS'));
onset_dir = getenv_default('ONSET_DIR', '/home/pateld3/NewStimuluesSetting');

config = sprintf('%s_spmspace_%dconf', pipeline, confound_count);
beta_root = fullfile(lss_root, 'lss_betas_spmspace', config);
out_dir   = fullfile(lss_root, 'lss_extracted_spmspace', config);

ORIG_SUB_IDS = [1,2,3,4,5,6,7,8,9,14,15,16,17,18,19,20];
N_SUBS = 16;
N_RUNS = 5;

if ~exist(out_dir,'dir'), mkdir(out_dir); end

fprintf('Extracting SPM-space LSS config=%s\n  betas=%s\n  out=%s\n', ...
    config, beta_root, out_dir);

for s = 1:N_SUBS
    sid = sprintf('%03d', s);
    orig_id = ORIG_SUB_IDS(s);
    fprintf('\n=== Subject sub-%s (orig %02d) ===\n', sid, orig_id);

    beta_dir = fullfile(beta_root, ['beta_series_sub-' sid]);
    if ~isfolder(beta_dir)
        warning('Missing beta dir: %s. Skipping subject %d.', beta_dir, s);
        continue;
    end

    Pl = []; Nt = []; Up = []; %#ok<NASGU>

    for run = 1:N_RUNS
        if orig_id >= 12
            onset_name = sprintf('Sub%02d_run%d.mat', orig_id, run);
        else
            onset_name = sprintf('Sub%02drun%d.mat', orig_id, run);
        end
        load(fullfile(onset_dir, onset_name)); %#ok<LOAD>

        % Folder names use condition-block design indices, but downstream
        % matrices must follow presentation order. Preserve both values here.
        trial_info = struct('onset',{},'condition',{},'design_idx',{});
        idx = 1;
        for t = 1:20
            trial_info(idx) = struct('onset',Onset(t,2),'condition','Pl','design_idx',t); %#ok<NODEF>
            idx = idx + 1;
        end
        for t = 1:20
            trial_info(idx) = struct('onset',Onset(t,3),'condition','Nt','design_idx',t+20);
            idx = idx + 1;
        end
        for t = 1:20
            trial_info(idx) = struct('onset',Onset(t,4),'condition','Up','design_idx',t+40);
            idx = idx + 1;
        end
        clear Onset;

        % Sorting all conditions together restores the temporal trial order
        % expected by the repeated cross-validation scripts.
        [~, order] = sort([trial_info.onset]);
        trial_info = trial_info(order);

        for t = 1:60
            ti = trial_info(t);
            trial_dir = fullfile(beta_dir, sprintf('%s%d_run%d', ti.condition, ti.design_idx, run));
            beta_file = fullfile(trial_dir, 'beta_0001.nii');
            if ~isfile(beta_file)
                beta_img = fullfile(trial_dir, 'beta_0001.img');
                if isfile(beta_img)
                    beta_file = beta_img;
                else
                    warning('Missing beta: %s', beta_file);
                    continue;
                end
            end
            % MATLAB linear indexing matches the flattened Wang atlas masks.
            vec = single(reshape(spm_read_vols(spm_vol(beta_file)), [], 1));
            switch ti.condition
                case 'Pl'; Pl = [Pl, vec]; %#ok<AGROW>
                case 'Nt'; Nt = [Nt, vec]; %#ok<AGROW>
                case 'Up'; Up = [Up, vec]; %#ok<AGROW>
            end
        end
        fprintf('  Run %d done.\n', run);
    end

    fprintf('  Totals: Pl=%d Nt=%d Up=%d voxels=%d\n', ...
        size(Pl,2), size(Nt,2), size(Up,2), size(Pl,1));
    if size(Pl,1) ~= 53*63*46
        warning('Subject %d: expected 153594 voxels for 53x63x46, got %d.', s, size(Pl,1));
    end
    if size(Pl,2)~=100 || size(Nt,2)~=100 || size(Up,2)~=100
        warning('Subject %d: expected 100 trials/condition.', s);
    end

    save(fullfile(out_dir, sprintf('Pl%d.mat', s)), 'Pl', '-v7.3');
    save(fullfile(out_dir, sprintf('Nt%d.mat', s)), 'Nt', '-v7.3');
    save(fullfile(out_dir, sprintf('Up%d.mat', s)), 'Up', '-v7.3');
    fprintf('  Saved Pl%d/Nt%d/Up%d.mat\n', s, s, s);
end

fprintf('\nAll subjects extracted to %s\n', out_dir);
end

function path_value = getenv_default(name, default_value)
%GETENV_DEFAULT Return an environment value or a configured fallback.
path_value = getenv(name);
if isempty(path_value), path_value = default_value; end
end
