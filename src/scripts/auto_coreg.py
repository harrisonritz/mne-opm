#  RUN COREG

# %% imports
from dataclasses import dataclass
import os
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
found_env = load_dotenv(find_dotenv('TSX_env.env', usecwd=True), verbose=True, override=True)
if found_env:
    BIDS_DIR = f"{os.environ.get('BIDS_DIR')}"
    SUBJECTS_DIR = f"{os.environ.get('SUBJECTS_DIR')}" 
    print('data & bids dirs from environment variables')
else:
    raise ValueError("Please set the TSX_BIDS environment variable.")
    # BIDS_DIR = "/Volumes/hritz/2025_TSX_Pilot/bids"
    # print('data & bids dir from local variables')

# get SUBJECT enviromental veriable
SUBJECT = os.environ.get("SUBJECT")
if SUBJECT is None:
    # SUBJECT='sub-003'
    raise ValueError("Please set the SUBJECT environment variable.")
SUBJECT_NUM = SUBJECT.split('_')[0].split('-')[1]

TASK = os.environ.get("TASK")
if TASK is None:
    # TASK = "TSXpilot"
    raise ValueError("Please set the TASK environment variable.")

SESSION = os.environ.get("SESSION")
if SESSION is None:
    # SESSION = "01"
    raise ValueError("Please set the SESSION environment variable.")

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
    surfaces="head-dense",
    dig=True,
    eeg=[],
    meg="sensors",
    show_axes=True,
    coord_frame="auto",
)



# automatic coregistration
try:
    coreg = mne.gui.coregistration(inst=fname_raw, subject=SUBJECT, subjects_dir=SUBJECTS_DIR, block=True)
except Exception as e:
    print(f"------ Error in GUI coregistration: {e}")

coreg = mne.coreg.Coregistration(info, SUBJECT, SUBJECTS_DIR, fiducials='estimated')

coreg.set_scale_mode('Uniform')
coreg.fit_fiducials(verbose=True)
coreg.omit_head_shape_points(distance=5.0 / 1e3)  # distance is in meters

coreg.set_scale_mode('3-axis')
coreg.fit_icp(n_iterations=100, verbose=True)
fig = mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)
# fig.save(
#     os.path.join(
#         BIDS_DIR,
#         "derivatives",
#         "figures",
#         f"coreg_{SUBJECT}.png",
#     ),
#     dpi=300,
# )

dists = coreg.compute_dig_mri_distances() * 1e3  # in mm
print(
    f"Distance between HSP and MRI (mean/min/max):\n{np.mean(dists):.2f} mm "
    f"/ {np.min(dists):.2f} mm / {np.max(dists):.2f} mm"
)

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
