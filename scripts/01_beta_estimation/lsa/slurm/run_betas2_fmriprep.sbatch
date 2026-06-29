#!/bin/bash
#SBATCH --job-name=mvpa_betas
#SBATCH --output=/blue/ruogu.fang/bradystemac/mvpa_project/logs/betas_%j.log
#SBATCH --error=/blue/ruogu.fang/bradystemac/mvpa_project/logs/betas_%j.err
#SBATCH --time=24:00:00
#SBATCH --mail-user=bradystemac@ufl.edu
#SBATCH --mem=64gb
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --account=ruogu.fang

# ============================================================
# Run Step 1: Beta Series GLM (fMRIPrep version)
# ============================================================

echo "Job started: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Create required directories if they don't exist
mkdir -p /blue/ruogu.fang/bradystemac/mvpa_project/betas
mkdir -p /blue/ruogu.fang/bradystemac/mvpa_project/tmp
mkdir -p /blue/ruogu.fang/bradystemac/mvpa_project/logs

# Load MATLAB module
module load matlab/2025b

# Add SPM to path and run the script
matlab -nodisplay -nosplash -r " addpath('/blue/ruogu.fang/bradystemac/spm12'); run('/blue/ruogu.fang/bradystemac/mvpa_project/BetaS2_fmriprep.m'); exit;"

echo "Job finished: $(date)"