

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



sub_id = "sub-036"

fs_dir = "/Volumes/fileset-NDAW/harrison_ritz/TSX/data/TSX/freesurfer"
data_dir = "/Volumes/fileset-NDAW/harrison_ritz/TSX/data/TSX/bids/derivatives/trial__PrePost_maxwell-8-3-95_ica-96_gesd-99"

subject = f"{sub_id}_ses-01"
trans = f"{data_dir}/{sub_id}/ses-01/meg/{sub_id}_ses-01_task-TSX_trans.fif"
epo_fname = f"{data_dir}/{sub_id}/ses-01/meg/{sub_id}_ses-01_task-TSX_epo.fif"
fwd_fname = f"{data_dir}/{sub_id}/ses-01/meg/{sub_id}_ses-01_task-TSX_fwd.fif"

info = mne.io.read_info(epo_fname)


plot_kwargs = dict(
    subject=subject,
    subjects_dir=fs_dir,
    surfaces=dict(seghead=0.67),
    dig=True,
    meg=dict(sensors=.67),
    show_axes=True,
    coord_frame="meg",
    mri_fiducials=True,
)


# %% plot
# Here we look at the dense head, which isn't used for BEM computations but
# is useful for coregistration.
mne.viz.plot_alignment(
    info,
    trans,
    **plot_kwargs
)



# %% plot coreg options
coreg = mne.coreg.Coregistration(info, subject, fs_dir, fiducials="auto")
coreg.fit_fiducials(verbose=True)
mne.viz.plot_alignment(info, trans=coreg.trans, **plot_kwargs)