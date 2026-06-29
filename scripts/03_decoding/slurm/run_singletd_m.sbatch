#!/bin/bash

#SBATCH --job-name=matlab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pateld3@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30gb
#SBATCH --time=4-00:00:00
#SBATCH --output=%x_%j.log

module load matlab
SPM_PATH="/home/pateld3/spm"

matlab -nodisplay -nosplash -r "addpath('$SPM_PATH'); run('SingleTrialDecodingv3.m'); exit;"
