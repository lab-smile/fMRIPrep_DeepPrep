# Grouped Final Comparisons

Each image places three boxes side by side within every ROI. Separate images
are provided for Pleasant vs Neutral and Unpleasant vs Neutral.

## Comparisons

1. Paper vs Max vs Our SPM, n=20
2. Paper vs Max vs Our SPM, n=16
3. DeepPrep vs fMRIPrep vs Max's SPM, n=16
4. DeepPrep vs fMRIPrep vs Our SPM, n=16
5. DeepPrep vs fMRIPrep vs Paper

All subject-level comparisons use the balanced LIBSVM result files with
6 confounds and no global z-score. The DeepPrep and fMRIPrep files use the
SPM-space LSS results.

## Input Files

- DeepPrep:
  `01_our_deepprep_fmriprep_lss_spmspace_no_global_zscore/DecodingResults_deepprep_spmspace_6conf_libsvm_balanced_sub16.mat`
- fMRIPrep:
  `01_our_deepprep_fmriprep_lss_spmspace_no_global_zscore/DecodingResults_fmriprep_spmspace_6conf_libsvm_balanced_sub16.mat`
- Max's SPM:
  `02_max_spm_results/DecodingResults_LIBSVM_v5.mat`
- Our SPM n=20:
  `03_our_spm_results_no_global_zscore/DecodingResults_spm_6conf_libsvm_balanced_all20.mat`
- Our SPM n=16:
  `03_our_spm_results_no_global_zscore/DecodingResults_spm_6conf_libsvm_balanced_sub16.mat`
- Paper:
  `04_published_paper/Bo2021_Figure3B_<contrast>_estimates.mat`

## Box Interpretation

- Subject-level results: standard boxplots across subjects, with 1.5-IQR
  whiskers and `+` markers for outliers.
- Paper results: approximate boxes and visible whisker-cap endpoints
  digitized from Bo et al. 2021 Figure 3B. No paper outliers or subject
  values are inferred.
- Dashed horizontal lines mark chance at 50% and the paper's statistical
  threshold at 54%.

## Cohorts

- Max n=20 uses all subjects in `DecodingResults_LIBSVM_v5.mat`.
- Max n=16 uses original subject IDs `1-9, 14-20`.
- Our SPM n=16 uses the matching `sub16` file with those original IDs.
- DeepPrep and fMRIPrep use their 16 retained subjects.
- The published paper boxplots are group summaries and are unchanged
  between the n=20 and n=16 comparisons.

## Reproduction

From the repository root:

```bash
python3 plots/generate_grouped_comparison_plots.py
```
