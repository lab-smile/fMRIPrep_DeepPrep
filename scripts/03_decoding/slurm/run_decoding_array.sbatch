#!/bin/bash
#SBATCH --job-name=decode_array
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pateld3@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=15gb
#SBATCH --time=08:00:00
#SBATCH --array=1-9
#SBATCH --output=/orange/ruogu.fang/pateld3/data/deepprep_logs/deepprep-9-decode_%A_%a.log


# ─────────────────────────────────────────────────────────────────────────────
# run_decoding_array.sbatch
#
# SLURM array job: one job per subject, all subjects run in parallel.
# Each job calls decode_subject.py for one subject with N_JOBS CPUs.
#
# After ALL array jobs finish, run merge_results.py to assemble the final .mat:
#   python pipeline/step4_decoding/merge_results.py
#
# Submit:
#   sbatch pipeline/step4_decoding/run_decoding_array.sbatch
# ─────────────────────────────────────────────────────────────────────────────
# ─── Environment ─────────────────────────────────────────────────────────────
#mkdir -p logs

# Use the project virtualenv (numpy / scipy / scikit-learn / joblib required)
source ~/.venv/bin/activate

# Propagate SLURM array task ID as the subject index
SUBJECT=${SLURM_ARRAY_TASK_ID}
N_JOBS=${SLURM_CPUS_PER_TASK}   # = 8; matches --cpus-per-task above

echo "============================================================"
echo "Array job ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Subject : ${SUBJECT}"
echo "CPUs    : ${N_JOBS}"
echo "Node    : $(hostname)"
echo "Start   : $(date)"
echo "============================================================"

python decode_subject.py \
    --subject    "${SUBJECT}"  \
    --n-jobs     "${N_JOBS}"   \
    --skip-existing

EXIT_CODE=$?

echo "============================================================"
echo "End : $(date)"
echo "Exit code : ${EXIT_CODE}"
echo "============================================================"

exit ${EXIT_CODE}
