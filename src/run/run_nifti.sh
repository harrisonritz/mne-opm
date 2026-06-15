# convert dicom to nifti
# Harrison Ritz 2025


# fail on first crash
if [ "${FAIL_ON_FIRST_CRASH}" = "1" ]; then
	set -eo pipefail
	trap 'echo "[FAIL-FAST] Error at line $LINENO. Exiting." >&2' ERR
	echo "[FAIL-FAST] Enabled: pipeline will stop on first failure."
fi

# set config
export CONFIG_PATH="$CONFIG_DIR/config-$ANALYSIS.py"


SUBJ_RAW_PATH=$(find "$RAW_DIR" -type d -path "*$SUBJECT")
DICOM_PATH="$SUBJ_RAW_PATH/dicom"
NIFTI_PATH="$SUBJ_RAW_PATH/anat"

# overwrite existing folder
rm -rf "$NIFTI_PATH"
mkdir -p "$NIFTI_PATH"


## BIDS FORMATTING ----------------------------------------
echo ""
echo "RUNNING BIDS FORMATTING --------------------------"
echo "ROOT_DIR: $ROOT_DIR"
echo "EXPERIMENT: $EXPERIMENT"
echo "SUBJECT: $SUBJECT"
echo "RAW_DIR: $RAW_DIR"
echo "BIDS_DIR: $BIDS_DIR"
echo "SUBJ_RAW_PATH: $SUBJ_RAW_PATH"
echo "DICOM_PATH: $DICOM_PATH"
echo "NIFTI_PATH: $NIFTI_PATH"
echo "--------------------------------------------------"
echo ""



# convert dicom to nifti ---------------------------
# using dcm2niix
# -a: append all conversions into single folder
# -z: compress output files
# -f: specify output filename format
# -o: output directory
# input directory
dcm2niix -a y -z y -f %p -o "$NIFTI_PATH" "$DICOM_PATH"

# append '_t1w' to any files containing 'T1w' in the name -----------
# convert nii           
for file in "$NIFTI_PATH"/*T1w*.nii.gz; do
    if [[ -f "$file" ]]; then
        new_name="${file%.nii.gz}_t1w.nii.gz"
        mv "$file" "$new_name"
        echo "Renamed $file to $new_name"
    else
        echo "No T1w files found in $NIFTI_PATH"
    fi
done

# convert sidecar json
for file in "$NIFTI_PATH"/*T1w*.json; do
    if [[ -f "$file" ]]; then
        new_name="${file%.json}_t1w.json"
        mv "$file" "$new_name"
        echo "Renamed $file to $new_name"
    else
        echo "No json files found in $NIFTI_PATH"
    fi
done



# append '_t2w' to any files containing 'T2w' in the name -----------
# convert nii           
for file in "$NIFTI_PATH"/*T2w*.nii.gz; do
    if [[ -f "$file" ]]; then        
        new_name="${file%.nii.gz}_t2w.nii.gz"
        mv "$file" "$new_name"
        echo "Renamed $file to $new_name"
    else
        echo "No T2w files found in $NIFTI_PATH"
    fi
done

# convert sidecar json
for file in "$NIFTI_PATH"/*T2w*.json; do
    if [[ -f "$file" ]]; then
        new_name="${file%.json}_t2w.json"
        mv "$file" "$new_name"

        echo "Renamed $file to $new_name"
    else
        echo "No json files found in $NIFTI_PATH"
    fi
done
