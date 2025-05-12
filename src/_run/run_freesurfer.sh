## activate environment ----------------------------------------
conda activate mne-opm


# set parameters ---------------
export SUBJECT_NUM=sub-$1
export SUBJECT=sub-$1_ses-01



export FREESURFER_HOME=/Applications/freesurfer/8.0.0
export FS_ALLOW_DEEP=1
source $FREESURFER_HOME/SetUpFreeSurfer.sh


export SUBJECTS_DIR=path/to/freesurfer/subjects
export T1W_PATH=path/to/${SUBJECT_NUM}_ses-01_T1w.nii.gz


n_cpus=4
# ------------------------------



# # run freesurfer ------------------
echo "Running freesurfer pipeline ----------------------"
recon-all -i $T1W_PATH -s $SUBJECT -parallel -openmp $n_cpus -all


# build BEM ------------------
# create boundary element model
echo "Building watershed bem ----------------------"
mne watershed_bem --subject=$SUBJECT --subjects-dir=$SUBJECTS_DIR --overwrite --atlas --gcaatlas --verbose

# # construct hi-res head surfaces
echo "Making hi-res scalp surface ----------------------"
mne make_scalp_surfaces --subject=$SUBJECT --subjects-dir=$SUBJECTS_DIR --overwrite --force --verbose