#  RUN COREG

HAIR_GROW = 5.0
OMIT_DISTANCE = 2.5/1e3
N_ROUNDS = 3


# %% imports
from dataclasses import dataclass
import os
from click import pause
import mne
import numpy as np
from mne.io import read_info
import mne_bids
from mne_bids import (
    BIDSPath,
    get_anat_landmarks,
    write_anat,
)
from dotenv import load_dotenv, find_dotenv
from glob import glob


# %% get info

BIDS_DIR = f"{os.environ.get('BIDS_DIR')}"
SUBJECTS_DIR = f"{os.environ.get('SUBJECTS_DIR')}" 

# get SUBJECT enviromental veriable
SUBJECT = f"{os.environ.get('SUBJECT')}"
if SUBJECT is None:
    raise ValueError("Please set the SUBJECT environment variable.")
SUBJECT_NUM = SUBJECT.split('_')[0].split('-')[1]

TASK = os.environ.get("EXPERIMENT")
if TASK is None:
    raise ValueError("Please set the TASK environment variable.")

SESSION = os.environ.get("SESSION")
if SESSION is None:
    raise ValueError("Please set the SESSION environment variable.")


# check if landmarks already written
if BIDSPath(
    subject=SUBJECT_NUM, 
    session=SESSION, 
    root=BIDS_DIR, 
    suffix="T1w",
    extension=".json",
    ):
    print(f"Landmarks already written for {SUBJECT_NUM}/{SUBJECT} with task {TASK} and session {SESSION}")
    return
else:
    print(f"Running coreg for {SUBJECT_NUM}/{SUBJECT} with task {TASK} and session {SESSION}")


# get info
fname_raw = mne_bids.find_matching_paths(
    root=BIDS_DIR,
    subjects=SUBJECT_NUM,
    sessions=SESSION,
    tasks=TASK,
    ignore_nosub=True,
    extensions=".fif",
)[0]

print('paths: ', mne_bids.find_matching_paths(
    root=BIDS_DIR,
    subjects=SUBJECT_NUM,
    sessions=SESSION,
    tasks=TASK,
    ignore_nosub=True,
    extensions=".fif",
))


info = read_info(fname_raw)

# %% inspect dataset

# mne_bids.inspect_dataset(bids_path=BIDS_DIR)


# %% run coreg

plot_kwargs = dict(
    subject=SUBJECT,
    subjects_dir=SUBJECTS_DIR,
    surfaces=dict(seghead=0.95),
    dig=True,
    eeg=[],
    meg="sensors",
    show_axes=True,
    coord_frame="auto",
)



# automatic coregistration
if fname_raw
try:
    coreg = mne.gui.coregistration(inst=fname_raw, subject=SUBJECT, subjects_dir=SUBJECTS_DIR, block=True)
except Exception as e:
    print(f"------ Error in GUI coregistration: {e}")

coreg = mne.coreg.Coregistration(info, SUBJECT, SUBJECTS_DIR, fiducials='estimated')

# fit fiducials
coreg.set_scale_mode('Uniform')
coreg.fit_fiducials(verbose=True)

# fit head shape points
coreg.set_scale_mode('3-axis')
coreg.set_grow_hair(HAIR_GROW)


for rr in range(N_ROUNDS):
    coreg.omit_head_shape_points(distance=OMIT_DISTANCE)  # distance is in meters
    coreg.fit_icp(n_iterations=100, verbose=True)

    dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
    print(
        f"[round {rr}] Distance between HSP and MRI (mean/median/max):"
        f"\n{np.mean(dists):.2f} mm "
        f"/ {np.median(dists):.2f} mm / {np.max(dists):.2f} mm"
    )

fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
fig.savefig(os.path.join(SUBJECTS_DIR, SUBJECT, "mri", "coregistration.png"))

# pause("\n\nPress any key to continue after inspecting the coregistration plot...\n")




# %% save t1w info


# fs T1 path (avoid bad mri datatypes)
t1w_fs_path = os.path.join(SUBJECTS_DIR, SUBJECT, "mri", "T1.mgz")

# BIDS T1 path
t1w_bids_path = BIDSPath(
    subject=SUBJECT_NUM, 
    session=SESSION, 
    root=BIDS_DIR, 
    suffix="T1w",
    datatype="anat",
    extension=".nii",
    )

# BIDS anat dir
anat_bids_path = BIDSPath(
    subject=SUBJECT_NUM, 
    session=SESSION, 
    root=BIDS_DIR, 
    suffix="T1w",
    datatype="anat",
    )

# use ``trans`` to transform landmarks from the ``raw`` file to
# the voxel space of the image
landmarks = get_anat_landmarks(
    t1w_fs_path,  # path to the MRI scan
    info=info,  # the MEG data file info from the same SUBJECT as the MRI
    trans=coreg.trans,  # our transformation matrix
    fs_subject=SUBJECT,  # FreeSurfer SUBJECT
    fs_subjects_dir=SUBJECTS_DIR,  # FreeSurfer subjects directory
)

# We use the write_anat function
t1w_bids_path = write_anat(
    image=t1w_fs_path,  # path to the MRI scan
    bids_path=anat_bids_path,
    landmarks=landmarks,  # the landmarks in MRI voxel space
    verbose=True,  # this will print out the sidecar file
    overwrite=True,  # overwrite the file if it exists
)
