#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=hpg-turin
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email@ufl.edu>
pwd;
date;

module load fmriprep/23.1.4

export FREESURFER_LICENSE=/path/to/license.txt
export BIDS_DIR=/path/to/bids-data
export OUTPUT_DIR=/path/to/output/fmriprep/
export WORK_DIR=/path/to/workdir/fmriprep_work/
export TEMPLATEFLOW_HOME=/path/to/templateflow

echo "Starting fMRIprep"
mkdir -p ${OUTPUT_DIR}
mkdir -p ${WORK_DIR}

fmriprep \
    ${BIDS_DIR} ${OUTPUT_DIR} participant \
    --task-id emotion \
    --fs-license-file ${FREESURFER_LICENSE} \
    --ignore fieldmaps \
    --output-spaces MNI152NLin2009cAsym:res-2 \
    --dummy-scans 6 \
    --nthreads 16 \
    --omp-nthreads 8 \
    --mem-mb 160000 \
    --work-dir ${WORK_DIR}

echo "Done"
date