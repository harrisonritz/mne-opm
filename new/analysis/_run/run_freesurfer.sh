## activate environment ----------------------------------------
conda activate mne-opm


# set parameters ---------------
subject_num=sub-004
export SUBJECT=sub-004_ses-01

export T1W_PATH=/Users/hr0283/Projects/TSX_OPM/data/bids/${subject_num}/ses-01/anat/${subject_num}_ses-01_T1w.nii.gz


export FREESURFER_HOME=/Applications/freesurfer/8.0.0
export FS_ALLOW_DEEP=1
source $FREESURFER_HOME/SetUpFreeSurfer.sh

export SUBJECTS_DIR=/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/freesurfer/subjects # save location


n_cpus=16
# ------------------------------



# # run freesurfer ------------------
echo "Running freesurfer pipeline..."
recon-all -i $T1W_PATH -s $SUBJECT -parallel -openmp $n_cpus -all


# build BEM ------------------
# create boundary element model
echo "Building watershed bem..."
mne watershed_bem --subject=$SUBJECT --subjects-dir=$SUBJECTS_DIR --overwrite --atlas --gcaatlas --verbose

# # construct hi-res head surfaces
echo "Making hi-res scalp surface..."
mne make_scalp_surfaces --subject=$SUBJECT --subjects-dir=$SUBJECTS_DIR --overwrite --force --verbose