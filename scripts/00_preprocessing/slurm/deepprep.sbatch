#!/bin/bash
#SBATCH --job-name=deepprep
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=10GB
#SBATCH --partition=hpg-turin
#SBATCH --gpus=1
#SBATCH --time=3-00:00:00
#SBATCH --output=%x_%j.log
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<email@ufl.edu>

pwd;
date;

module load singularity

export FREESURFER_LICENSE=/path/to/license.txt
export BIDS_DIR=/path/to/bids-data
export DEEPPREP_PATH=/apps/deepprep/25.1
export OUTPUT_DIR=/path/to/output/deepprep/
export NXF_HOME=/path/to/output/deepprep/

echo "Starting DeepPrep NextFlow Download"
mkdir -p ${OUTPUT_DIR}/.nextflow
singularity exec --cleanenv ${DEEPPREP_PATH}/deepprep.sif rsync -arv /home/deepprep/.nextflow ${OUTPUT_DIR}/.nextflow

echo "Starting DeepPrep"
singularity run --cleanenv --nv -B ${BIDS_DIR}:/input -B ${OUTPUT_DIR}:/output -B ${FREESURFER_LICENSE}:/fs_license.txt ${DEEPPREP_PATH}/deepprep.sif /input /output participant --bold_task_type emotion --fs_license_file /fs_license.txt --device gpu --bold_skip_frame 6 --bold_sdc False --bold_volume_space MNI152NLin2009cAsym --bold_volume_res 02
echo "Done"

date