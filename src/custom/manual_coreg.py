

# %%


import mne



# %% setup

# 019: minor
# 020: major
# 021: major
# 022: medium
# 025: medium
# 026: minor
# 028: minor
# 029: major
# 031: major
# 034: medium


root_dir = "/Volumes/fileset-NDAW/harrison_ritz/TSX/data/TSX"
model_name = "trial__PrePost_maxwell-8-3-95_ica-96_gesd-99"
fs_dir = f"{root_dir}/freesurfer"
bids_dir = f"{root_dir}/bids"


plot_kwargs = dict(
    subjects_dir=fs_dir,
    surfaces=dict(seghead=0.67),
    dig=True,
    meg=dict(sensors=.67),
    show_axes=True,
    coord_frame="meg",
    mri_fiducials=True,
)

# print all the subject numbers in the bids directory, sorted by subject number
import os
sub_dirs = [d for d in os.listdir(bids_dir) if d.startswith("sub-")]
sub_dirs = sorted(sub_dirs, key=lambda x: int(x.split("-")[1]))
print(sub_dirs)
print(f"Number of subjects: {len(sub_dirs)}")


# %% plot
# Here we look at the dense head, which isn't used for BEM computations but
# is useful for coregistration.



# loop over subjects and plot the alignment for each subject
for sub_id in sub_dirs:
    subject = f"{sub_id}_ses-01"
    trans = f"{bids_dir}/derivatives/{model_name}/{sub_id}/ses-01/meg/{sub_id}_ses-01_task-TSX_trans.fif"
    epo_fname = f"{bids_dir}/derivatives/{model_name}/{sub_id}/ses-01/meg/{sub_id}_ses-01_task-TSX_epo.fif"
    fwd_fname = f"{bids_dir}/derivatives/{model_name}/{sub_id}/ses-01/meg/{sub_id}_ses-01_task-TSX_fwd.fif"

    info = mne.io.read_info(epo_fname)

    coreg = mne.coreg.Coregistration(info, subject, fs_dir, fiducials="auto")
    




# %% plot coreg options
