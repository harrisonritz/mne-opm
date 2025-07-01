# run freesurfer
# Harrison Ritz 2025


## activate environment ----------------------------------------
conda activate mne-opm


# ## import variables
# export EXPERIMENT=$1
# if [ ! ${EXPERIMENT} ]; then
#   echo "Error: please provide a experiment name"
#   exit 1
# fi

# export SUBJECT_NUM=sub-$2
# if [ ! ${SUBJECT} ]; then
#   echo "Error: please provide a subject number"
#   exit 1
# fi


# ## activate environment file
# source ../{$EXPERIMENT}.env
# if [ ! ${ROOT_DIR} ]; then
#   echo "Error: please check that the environmental variables are available at '../{$EXPERIMENT}.env'"
#   exit 1
# fi


## fixed variables
# export RAW_DIR=$(eval echo $RAW_DIR)
# export BIDS_DIR=$(eval echo $BIDS_DIR)
# export SUBJECTS_DIR=$(eval echo $SUBJECTS_DIR)

old_sub=$SUBJECT
export SUBJECT=${SUBJECT_NUM}_ses-${SESSION}
export FS_ALLOW_DEEP=1
source $FREESURFER_HOME/SetUpFreeSurfer.sh


# ------------------------------



## RUN FREESURFER ------------------
echo ""
echo "RUNNING FREESURFER --------------------------"
echo "ROOT_DIR: $ROOT_DIR"
echo "EXPERIMENT: $EXPERIMENT"
echo "SUBJECT: $SUBJECT"
echo "T1W_PATH: $T1W_PATH"
echo "FREESURFER_HOME: $FREESURFER_HOME"
echo "RAW_DIR: $RAW_DIR"
echo "BIDS_DIR: $BIDS_DIR"
echo "SUBJECTS_DIR: $SUBJECTS_DIR"
echo "--------------------------"
echo ""

recon-all -i $T1W_PATH -s $SUBJECT -parallel -openmp $MAX_WORKERS -all

# build BEM ------------------
# create boundary element model
echo "Building watershed bem..."
mne watershed_bem --subject=$SUBJECT --subjects-dir=$SUBJECTS_DIR --overwrite --atlas --gcaatlas --verbose

# # construct hi-res head surfaces
echo "Making hi-res scalp surface..."
mne make_scalp_surfaces --subject=$SUBJECT --subjects-dir=$SUBJECTS_DIR --overwrite --force --verbose

# revert to old subject just in case
export SUBJECT=$old_sub