#!/bin/bash
#SBATCH --job-name=group_validation
#SBATCH --output=/orange/ruogu.fang/pateld3/data/deepprep_group_mvpa_results/logs/group_validation_%j.log
#SBATCH --error=/orange/ruogu.fang/pateld3/data/deepprep_group_mvpa_results/logs/group_validation_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pateld3@ufl.edu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20gb
#SBATCH --time=04:00:00

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "Start Time: $(date)"
echo "=================================================="
echo ""

# Load required modules
module purge
#module load conda
module load python/3.10

# Activate conda environment (adjust to your environment name)
# conda activate neuroimaging  # Uncomment and adjust if using conda env

# Set paths
SCRIPT_DIR="/orange/ruogu.fang/pateld3/data/"
RESULTS_FILE="/orange/ruogu.fang/pateld3/data/single_mvpa_results/decoding_results_k5x100_v4.mat"
OUTPUT_DIR="/orange/ruogu.fang/pateld3/data/deepprep_group_mvpa_results"
CHECKPOINT_DIR="/orange/ruogu.fang/pateld3/data/deepprep_group_mvpa_results/checkpoints"

# Create necessary directories
mkdir -p $OUTPUT_DIR
mkdir -p $CHECKPOINT_DIR
mkdir -p /orange/ruogu.fang/pateld3/data/deepprep_group_mvpa_results/logs

# Install required packages if not already installed
#pip install --user numpy scipy matplotlib seaborn 2>&1 | tail -n 5
source ~/.venv/bin/activate


echo ""
echo "=================================================="
echo "Running Group-Level Validation Script"
echo "=================================================="
echo "Results file: $RESULTS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo ""

# Run the validation script with checkpointing
# The script will automatically resume from the last checkpoint if interrupted
python group_level_validation.py \
    --results-file "$RESULTS_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --n-permutations 100000 \
    --checkpoint-interval 5000 \
    --alpha 0.001 \
    --comparison both

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=================================================="
echo "Job completed"
echo "Exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=================================================="

# Exit with the script's exit code
exit $EXIT_CODE
