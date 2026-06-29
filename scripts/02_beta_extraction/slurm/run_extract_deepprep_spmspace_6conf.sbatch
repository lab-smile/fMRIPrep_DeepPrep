#!/bin/bash
# Step 3 -- extract DeepPrep SPM-space LSS betas, 6 confounds.

#SBATCH --job-name=ext_dp_spm6
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pateld3@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30gb
#SBATCH --time=10:00:00
#SBATCH --output=/orange/ruogu.fang/pateld3/data/LSS/logs/%x_%j.log

set -euo pipefail

module load matlab

SCRIPT_DIR="${SCRIPT_DIR:-/orange/ruogu.fang/pateld3/data/LSS/scripts}"
SPM_PATH="${SPM_PATH:-/home/pateld3/spm}"

matlab -nodisplay -nosplash -r \
    "try; addpath('${SPM_PATH}'); addpath('${SCRIPT_DIR}'); extract_lss_spmspace('deepprep', 6); catch ME; disp(getReport(ME,'extended')); exit(1); end; exit(0);"

