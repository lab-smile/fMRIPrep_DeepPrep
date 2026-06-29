#!/bin/bash

#SBATCH --job-name=matlab
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=email@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=30gb
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.log

module load matlab
SPM_PATH="/path/to/spm"

matlab -nodisplay -nosplash -r "addpath('$SPM_PATH'); run('BetaS2.m'); exit;"

