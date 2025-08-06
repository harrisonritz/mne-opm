# convert dicom
# Harrison Ritz 2025


## activate environment ----------------------------------------
# source activate base
conda activate mne-opm



SUBJ_RAW_PATH=$(find $RAW_DIR -type d -path "*$SUBJECT")
DICOM_PATH=$SUBJ_RAW_PATH/dcm
NIFTI_PATH=$SUBJ_RAW_PATH/anat
mkdir -p $NIFTI_PATH


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



dcm2niix -z y -f %p -o $NIFTI_PATH $DICOM_PATH


# append 't1w' to any files containting 'T1w' in the name
for file in $NIFTI_PATH/*T1w*.nii.gz; do
    if [[ -f "$file" ]]; then
        new_name="${file%.nii.gz}_t1w.nii.gz"
        mv "$file" "$new_name"
        echo "Renamed $file to $new_name"
    else
        echo "No T1w files found in $NIFTI_PATH"
    fi
done

# append 't2w' to any files containting 'T2w' in the name
for file in $NIFTI_PATH/*T2w*.nii.gz; do
    if [[ -f "$file" ]]; then
        new_name="${file%.nii.gz}_t2w.nii.gz"
        mv "$file" "$new_name"
        echo "Renamed $file to $new_name"
    else
        echo "No T2w files found in $NIFTI_PATH"
    fi
done
