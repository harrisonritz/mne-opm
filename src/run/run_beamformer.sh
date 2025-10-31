#!/bin/bash
# LCMV Beamformer Source Reconstruction
# Harrison Ritz 2025


## activate environment ----------------------------------------
# conda activate mne-opm

## fixed variables
export MPLBACKEND=agg

# set config
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"


# fail on first crash
if [ "${FAIL_ON_FIRST_CRASH}" = "1" ]; then
	set -euo pipefail
	trap 'echo "[FAIL-FAST] Error at line $LINENO. Exiting." >&2' ERR
	echo "[FAIL-FAST] Enabled: pipeline will stop on first failure."
fi


## RUN BEAMFORMER ------------------
echo ""
echo "RUNNING BEAMFORMER ANALYSIS --------------------------"
echo "ROOT_DIR: $ROOT_DIR"
echo "EXPERIMENT: $EXPERIMENT"
echo "ANALYSIS: $ANALYSIS"
echo "SUBJECT: $SUBJECT"
echo "CONFIG_PATH: $CONFIG_PATH"
echo "RAW_DIR: $RAW_DIR"
echo "BIDS_DIR: $BIDS_DIR"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "--------------------------"
echo ""


# Run both time-locked and power beamformer analyses
# Time-locked is PRIMARY, Power is SECONDARY
echo ""
echo "======================= BEAMFORMER: Time-Locked + Power =============================================="
echo ""
python "$ROOT_DIR/src/custom/run_beamformer.py" --config="$CONFIG_PATH"
