#!/bin/bash
# Step 4 -- SPM-space 6-conf ROI decoding, no global z-score.
#
# Uses ROI_Decoding_Complete_foldwise_zscore.m on extracted SPM-space LSS betas:
#   lss_extracted_spmspace/{deepprep,fmriprep}_spmspace_6conf
#
# Array tasks:
#   1 = deepprep  libsvm  balanced
#   2 = deepprep  libsvm  unbalanced
#   3 = deepprep  fitcsvm balanced
#   4 = deepprep  fitcsvm unbalanced
#   5 = fmriprep  libsvm  balanced
#   6 = fmriprep  libsvm  unbalanced
#   7 = fmriprep  fitcsvm balanced
#   8 = fmriprep  fitcsvm unbalanced
#
# Submit after extraction finishes:
#   sbatch run_decode_spmspace_6conf_no_global_zscore.sbatch

#SBATCH --job-name=dec_spm6_noglob
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pateld3@ufl.edu
#SBATCH --array=1-8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120gb
#SBATCH --time=48:00:00
#SBATCH --output=/orange/ruogu.fang/pateld3/data/LSS/logs/%x_%A_%a.log

set -euo pipefail

module load matlab

SCRIPT_DIR="${SCRIPT_DIR:-/orange/ruogu.fang/pateld3/data/LSS/scripts}"
SPM_PATH="${SPM_PATH:-/home/pateld3/spm}"
DATA_ROOT="${DATA_ROOT:-/orange/ruogu.fang/pateld3/data}"
LSS_ROOT="${LSS_ROOT:-${DATA_ROOT}/LSS}"
OUT_DIR="${OUT_DIR:-${LSS_ROOT}/lss_decoding_results_spmspace_no_global_zscore}"

# LIBSVM folder containing svmtrain.m and svmpredict.m.
# One Might need to compile the svmtrain and svmpredict files
LIBSVM_PATH="${LIBSVM_PATH:-/home/pateld3/libsvm-3.37/matlab}"

case "${SLURM_ARRAY_TASK_ID}" in
  1) CONFIG="deepprep_spmspace_6conf"; CLASSIFIER="libsvm";  FOLD_METHOD="balanced" ;;
  2) CONFIG="deepprep_spmspace_6conf"; CLASSIFIER="libsvm";  FOLD_METHOD="unbalanced" ;;
  3) CONFIG="deepprep_spmspace_6conf"; CLASSIFIER="fitcsvm"; FOLD_METHOD="balanced" ;;
  4) CONFIG="deepprep_spmspace_6conf"; CLASSIFIER="fitcsvm"; FOLD_METHOD="unbalanced" ;;
  5) CONFIG="fmriprep_spmspace_6conf"; CLASSIFIER="libsvm";  FOLD_METHOD="balanced" ;;
  6) CONFIG="fmriprep_spmspace_6conf"; CLASSIFIER="libsvm";  FOLD_METHOD="unbalanced" ;;
  7) CONFIG="fmriprep_spmspace_6conf"; CLASSIFIER="fitcsvm"; FOLD_METHOD="balanced" ;;
  8) CONFIG="fmriprep_spmspace_6conf"; CLASSIFIER="fitcsvm"; FOLD_METHOD="unbalanced" ;;
  *) echo "Invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2; exit 1 ;;
esac

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "Job        : ${SLURM_JOB_ID}.${SLURM_ARRAY_TASK_ID}"
echo "Config     : ${CONFIG}"
echo "Classifier : ${CLASSIFIER}"
echo "Fold method: ${FOLD_METHOD}"
echo "Data dir   : ${LSS_ROOT}/lss_extracted_spmspace/${CONFIG}"
echo "ROI dir    : ${LSS_ROOT}/masks"
echo "Output dir : ${OUT_DIR}"
echo "LIBSVM path: ${LIBSVM_PATH:-<MATLAB path/default>}"
echo "CPUs       : ${SLURM_CPUS_PER_TASK}"
echo "Node       : $(hostname)"
echo "Start      : $(date)"
echo "============================================================"

MATLAB_CMD="try; "
MATLAB_CMD="${MATLAB_CMD}addpath('${SPM_PATH}'); addpath('${SCRIPT_DIR}'); "
if [[ -n "${LIBSVM_PATH}" ]]; then
    MATLAB_CMD="${MATLAB_CMD}addpath(genpath('${LIBSVM_PATH}')); "
fi
MATLAB_CMD="${MATLAB_CMD}opts=struct(); "
MATLAB_CMD="${MATLAB_CMD}opts.data_root='${LSS_ROOT}'; "
MATLAB_CMD="${MATLAB_CMD}opts.data_dir=fullfile('${LSS_ROOT}','lss_extracted_spmspace','${CONFIG}'); "
MATLAB_CMD="${MATLAB_CMD}opts.roiDir=fullfile('${LSS_ROOT}','masks'); "
MATLAB_CMD="${MATLAB_CMD}opts.out_dir='${OUT_DIR}'; "
MATLAB_CMD="${MATLAB_CMD}opts.ncpu=str2double(getenv('SLURM_CPUS_PER_TASK')); "
MATLAB_CMD="${MATLAB_CMD}ROI_Decoding_Complete_foldwise_zscore('${CLASSIFIER}','${FOLD_METHOD}','${CONFIG}',6,'sub16',opts); "
MATLAB_CMD="${MATLAB_CMD}catch ME; disp(getReport(ME,'extended')); exit(1); end"

matlab -batch "${MATLAB_CMD}"

exit_code=$?

echo "============================================================"
echo "End       : $(date)"
echo "Exit code : ${exit_code}"
echo "============================================================"

exit "${exit_code}"
