#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --mem=12GB
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --partition=all
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hr0283@princeton.edu
#SBATCH -J mne
#SBATCH --output logs/log_%A-%a.txt
#SBATCH --error logs/error_%A-%a.txt
#SBATCH --array=1-43

# ============================================================================
# SLURM Array Job for MNE-OPM Pipeline
# ============================================================================
# 
# Usage:
#   sbatch slurm_mne-opm.sh
#
# The SLURM array index ($SLURM_ARRAY_TASK_ID) is zero-padded to create
# the subject number (e.g., array index 1 -> subject "001").
#
# To run specific subjects, modify the --array parameter above:
#   --array=1-43       # subjects 001-043
#   --array=1,5,10     # subjects 001, 005, 010
#   --array=1-10%4     # subjects 001-010, max 4 concurrent jobs
#
# ============================================================================

# ==== EDITABLE DEFAULTS ====
# Modify these parameters for your analysis

PIPELINE="func"                 # pipeline: bids, coreg, freesurfer, preproc, sensor, source, func, anat
EXPERIMENT="TSXpilot"           # experiment name
ANALYSIS="trial"                # analysis name
SESSION="01"                    # session number

# Paths (adjust for your cluster environment)
CONFIG_BASE="/Users/hr0283/Projects/TSX_OPM/analysis/config"
DATA_BASE="/Users/hr0283/Projects/TSX_OPM/data"
FREESURFER_HOME="/Applications/freesurfer/8.1.0"

# Processing options
FAIL_ON_FIRST_CRASH=1
MAX_WORKERS=8                   # match SLURM -n parameter

# ==== END EDITABLE DEFAULTS ====


# ============================================================================
# Derived variables (do not edit below unless necessary)
# ============================================================================

# Zero-pad the SLURM array task ID to 3 digits for subject number
SUBJECT=$(printf "%03d" $SLURM_ARRAY_TASK_ID)

# Construct paths
SUBJECTS_DIR="$DATA_BASE/$EXPERIMENT/freesurfer"
T1W_PATH="$SUBJECTS_DIR/anat/sub-$SUBJECT/anat_ses-01_T1w_acq-0.8mm-MPR_t1w.nii.gz"
T2W_PATH="$SUBJECTS_DIR/anat/sub-$SUBJECT/anat_ses-01_T2w_acq-0.8mm-MPR_t2w.nii.gz"

# Path to the mne-opm.sh script (assumes this script is in src/run/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MNE_OPM_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# ============================================================================
# Environment setup
# ============================================================================

# Create logs directory if it doesn't exist
mkdir -p "$MNE_OPM_DIR/logs"

# Print job info
echo "============================================"
echo "MNE-OPM SLURM Job"
echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Subject:       $SUBJECT"
echo "Pipeline:      $PIPELINE"
echo "Experiment:    $EXPERIMENT"
echo "Analysis:      $ANALYSIS"
echo "Session:       $SESSION"
echo "Node:          $SLURM_NODELIST"
echo "CPUs:          $SLURM_CPUS_ON_NODE"
echo "Start time:    $(date)"
echo "============================================"

# ============================================================================
# Run the pipeline
# ============================================================================

cd "$MNE_OPM_DIR"

# Call mne-opm.sh with all parameters
bash mne-opm.sh "$PIPELINE" \
    --exp "$EXPERIMENT" \
    --sub "$SUBJECT" \
    --analysis "$ANALYSIS" \
    --session "$SESSION" \
    --data "$DATA_BASE" \
    --config "$CONFIG_BASE" \
    --fs "$FREESURFER_HOME" \
    --t1w "$T1W_PATH" \
    --t2w "$T2W_PATH" \
    --workers "$MAX_WORKERS"

# Capture exit status
EXIT_STATUS=$?

# ============================================================================
# Cleanup and reporting
# ============================================================================

echo "============================================"
echo "End time: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "============================================"

exit $EXIT_STATUS
