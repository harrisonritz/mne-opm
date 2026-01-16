# run freesurfer
# Harrison Ritz 2025

# fail on first crash
if [ "${FAIL_ON_FIRST_CRASH}" = "1" ]; then
	set -eo pipefail
	trap 'echo "[FAIL-FAST] Error at line $LINENO. Exiting." >&2' ERR
	echo "[FAIL-FAST] Enabled: pipeline will stop on first failure."
fi

# set config
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"

# set fs variables
old_sub=$SUBJECT
export SUBJECT=${SUBJECT_NUM}_ses-${SESSION}
export FS_ALLOW_DEEP=1

echo "fs home: ${FREESURFER_HOME}"
source $FREESURFER_HOME/SetUpFreeSurfer.sh
echo "freesurfer version: $(recon-all --version)"
# ------------------------------



## RUN FREESURFER ------------------
echo ""
echo "RUNNING FREESURFER --------------------------"
echo "ROOT_DIR: $ROOT_DIR"
echo "EXPERIMENT: $EXPERIMENT"
echo "SUBJECT: $SUBJECT"
echo "T1W_PATH: $T1W_PATH"
echo "T2W_PATH: $T2W_PATH"
echo "FREESURFER_HOME: $FREESURFER_HOME"
echo "RAW_DIR: $RAW_DIR"
echo "BIDS_DIR: $BIDS_DIR"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "--------------------------"
echo ""

# Check whether SUBJECTS_DIR exists
if [ ! -d "$SUBJECTS_DIR" ]; then
    echo "Creating SUBJECTS_DIR at: $SUBJECTS_DIR"
    mkdir -p "$SUBJECTS_DIR"
fi

# Check if freesurfer has already been run
if [ -d "$SUBJECTS_DIR/$SUBJECT/mri" ]; then
    echo "FreeSurfer has already been run for subject: $SUBJECT"
    return
fi

# Check if T1W_PATH exists
if [ -z "$T1W_PATH" ] || [ ! -f "$T1W_PATH" ]; then
    echo "ERROR: T1w image not found at path: $T1W_PATH"
    echo "FreeSurfer requires a T1w image to run. Please check the file path."
    exit 1
fi

# Check if T2W_PATH also exists and run appropriate recon-all command
if [ -n "$T2W_PATH" ] && [ -f "$T2W_PATH" ]; then
    echo "T2w image found at: $T2W_PATH"
    echo "Running FreeSurfer with both T1w and T2w inputs ----------------------------"
    echo ""
    echo "Command: recon-all -i $T1W_PATH -T2 $T2W_PATH -T2pial -s $SUBJECT -parallel -openmp $MAX_WORKERS -all"
    recon-all -i "${T1W_PATH}" -T2 "${T2W_PATH}" -T2pial -s $SUBJECT -parallel -openmp $MAX_WORKERS -all
else
    echo "T2w image not found, running FreeSurfer with T1w only..."
    recon-all -i "$T1W_PATH" -s $SUBJECT -parallel -openmp $MAX_WORKERS -all
fi

# build BEM ------------------
# create boundary element model
echo "\n\nBuilding watershed bem..."
mne watershed_bem --subject=$SUBJECT --subjects-dir="$SUBJECTS_DIR" --overwrite --atlas --gcaatlas --verbose

# # construct hi-res head surfaces
echo "\n\nMaking hi-res scalp surface..."
mne make_scalp_surfaces --subject=$SUBJECT --subjects-dir="$SUBJECTS_DIR" --overwrite --force --verbose


# revert to original subject formatting
export SUBJECT=$old_sub