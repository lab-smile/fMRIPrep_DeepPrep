%% BetaS2 - Estimate native-SPM least-squares-all beta series
% Builds one five-session first-level model per subject using the original
% SPM-preprocessed images. Each run contains 60 trial regressors ordered as
% pleasant (1:20), neutral (21:40), and unpleasant (41:60), plus six motion
% regressors. The resulting beta images feed the extraction/decoding stages.
%
% Data assumptions:
%   * 20 subject directories named Sub*
%   * five runs and 206 scans per run
%   * 100 onset MAT files sorted by subject and run
%   * onset values and durations are expressed in scan units

clear;
spm('defaults','FMRI')

global defaults

%% Discover subjects and onset files
% Define the main directory containing subject folders.
main_dir = '/orange/ruogu.fang/pateld3/SPM_Preprocessed_fMRI_20Subjects/20_subs';

% Get all subject folders that start with 'Sub'
mainpath = dir(fullfile(main_dir, 'Sub*'));
mainpath = mainpath([mainpath.isdir]);  % keep only directories

% Sort alphabetically so Sub01 -> Sub02 -> ... -> Sub20
mainpath = sortrows(struct2table(mainpath), 'name');
mainpath = table2struct(mainpath);


runs = ['run1';
        'run2';
        'run3';
        'run4';
        'run5'];
   
behavdir = dir(fullfile('/home/pateld3/NewStimuluesSetting','Sub*run*.mat'));  % Only Sub##run#.mat files
behavdir = struct2table(behavdir);
behavdir = sortrows(behavdir, 'name');  % Sort alphabetically
behavdir = table2struct(behavdir);
fprintf('Found %d onset files (expecting 100)\n', length(behavdir));
if length(behavdir) ~= 100
    error('Expected 100 onset files (20 subjects x 5 runs), found %d', length(behavdir));
end
        
% sessionID = 1;                   % The session number from 1 -> 3; change it accordingly
nscan = [206;206;206;206;206];           % Number of scans for each of the three sessions
nses = 5;                       % Total number of sessions/runs to be modeled, 3 in this case, habituation, acquisition, and extinction; each modeled separately
TR = 1.98;

%% Estimate one five-session model per subject
for i = 1:20
       
    for j = 1:5
        % Set basis function parameters
        SPM.xBF.name = 'hrf';
        SPM.xBF.length = 32.0513;
        SPM.xBF.order = 1;
        SPM.xBF.T = 36;
        SPM.xBF.T0 = 18;
        SPM.xBF.UNITS = 'scans';
        SPM.xBF.Volterra = 1;

        % Correctly construct current subject directory
        currDir = fullfile(main_dir, mainpath(i).name);
        cd(currDir);

        % Create output folder once per subject
        if j == 1
            outputdir = fullfile('/orange/ruogu.fang/pateld3/SPM_Preprocessed_fMRI_20Subjects/betas', ...
                ['beta_series_' num2str(i)]);
            mkdir(outputdir);
        end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Basis functions and timing parameters %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
        SPM.nscan          = ones(1,nses)*206;
        
        
%         SPM.xBF.dt % length of time bin in seconds
%         SPM.xBF.bf % basis set matrix

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Trial Specification: Design Matrix %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        onsetDir = '/home/pateld3/NewStimuluesSetting';
        onsetFile = fullfile(onsetDir, behavdir((i-1)*5+j).name);
        fprintf('Loading: %s\n', onsetFile);
        load(onsetFile);    % Load the .mat file for stimulus onset timings
        runDir{j} = strcat(currDir,'/run',num2str(j));
        tmp{j} = spm_select('fplist',runDir{j},'^swars.*\.img');
        % Columns 2-4 contain condition-specific trial onsets. Their fixed
        % block order determines which beta index belongs to each condition.
        Onset1=Onset(:,1);
        Beta(1:20)=Onset(1:20,2);
        Beta(21:40)=Onset(1:20,3);
        Beta(41:60)=Onset(1:20,4);
 
%         Dur1=Dur(:,1);
%         BetaDur(1:20)=Dur(1:20,2);
%         BetaDur(21:40)=Dur(1:20,3);
%         BetaDur(41:60)=Dur(1:20,4);
%         time_by_Order(201:240) = [];
%         time_by_Order(141:180) = [];
%         time_by_Order(81:120) = [];
% 
% 
%         Condi_number = size (time_by_Order,2);
% %         for n = 1:Condi_number
% %             if n<=length(sots{1})
% %                 time_by_Order(2,n) = -1;
% %             else if n<=length(sots{1})+length(sots{2})
% %                     time_by_Order(2,n) = 1;
% %                 else if n<=Condi_number
% %                         time_by_Order(2,n) = 0;
% %                     end
% %                 end
% %             end
% %         end
%         for n=1:60
%             time_by_Order(2,i)=0;
%         end
%         for n=61:80
%             time_by_Order(2,i)=1;
%         end
%         for n=81:100
%             time_by_Order(2,i)=-1;
%         end
%         for n=101:120
%             time_by_Order(2,i)=0;
%         end

%         Sorted_designmatrix = sortrows(time_by_Order',1);
%         Sorted_designmatrix = Sorted_designmatrix';
        
        %condition_name{1} = 'Fixation';
        for Nam=1:20
            condition_name{Nam}=strcat('Pl',num2str(Nam));
        end
        for Nam=1:20
            condition_name{Nam+20}=strcat('Nt',num2str(Nam));
        end
        for Nam=1:20
            condition_name{Nam+40}=strcat('Up',num2str(Nam));
        end
        
          %  SPM.Sess(j).U(1).name = {condition_name{1,1}};    
          %  SPM.Sess(j).U(1).ons = Onset1;
          %  SPM.Sess(j).U(1).dur = Dur1;
          %  SPM.Sess(j).U(1).P(1).name = 'none';  
        for c = 1:20
            SPM.Sess(j).U(c).name = {condition_name{1,c}};    
            SPM.Sess(j).U(c).ons = Beta(c);
            SPM.Sess(j).U(c).dur = 1.5152;
%             SPM.Sess(j).U(c).dur = 0;
            SPM.Sess(j).U(c).P(1).name = 'none';       % Parametric Modulation; 'none' for now
        end
        
         for c = 1:20
            SPM.Sess(j).U(c+20).name = {condition_name{1,c+20}};    
            SPM.Sess(j).U(c+20).ons = Beta(c+20);
           SPM.Sess(j).U(c+20).dur = 1.5152;
%             SPM.Sess(j).U(c+20).dur = 0;
            SPM.Sess(j).U(c+20).P(1).name = 'none';       % Parametric Modulation; 'none' for now
        end
        for c = 1:20
            SPM.Sess(j).U(c+40).name = {condition_name{1,c+40}};    
            SPM.Sess(j).U(c+40).ons = Beta(c+40);
            SPM.Sess(j).U(c+40).dur = 1.5152;
%             SPM.Sess(j).U(c+40).dur = 0;
            SPM.Sess(j).U(c+40).P(1).name = 'none';       % Parametric Modulation; 'none' for now
        end
    

    % Include movement parameters
        rpDir = fullfile(currDir, ['run' num2str(j)]);
        cd(rpDir);
        rnam = {'X','Y','Z','x','y','z'};
        fn = spm_select('list',rpDir,'^rp.*\.txt');
        [r1,r2,r3,r4,r5,r6] = textread(fn,'%f%f%f%f%f%f');
        SPM.Sess(j).C.C = [r1 r2 r3 r4 r5 r6];
        SPM.Sess(j).C.name = rnam;

    % Global Normalization: OPTIONS: 'Scaling'
  
    %-------------------------------------------------------------
        SPM.xGX.iGXcalc = 'Scaling';
        
    % low frequency confound: high-pass cutoff (secs) [Inf = no filtering]
    %-------------------------------------------------------------
        SPM.xX.K(j).HParam = 128;
        
    % intrinsic autocorrelations: OPTIONS: 'none'|'AR(1) + w'
    %-------------------------------------------------------------
        SPM.xVi.form       = 'AR(1)';
        
    % Specify data image files
       
     end
 
        SPM.xY.P = cat(1,tmp{:});
        SPM.xY.RT = TR;
        SPM = spm_fmri_spm_ui(SPM);
        cd(outputdir);
        SPM = spm_spm(SPM);
  
    
    % Configure design matrix
        

    % Estimate parameters
        
        
 
        clear SPM ;
    
    %%  Save the design matrix
%     Savename= strcat('DesignMatrix_',mainpath(i).name(23:27),runs(j,:));
%     eval([Savename '=Sorted_designmatrix']); 
%     savepath = '/orange/ruogu.fang/pateld3/SPM_Preprocessed_fMRI_20Subjects/betas';
%     EEG_BOLD = [savepath 'Designmatrix.mat'];
%     save(EEG_BOLD,Savename,'-append');
    
 end
