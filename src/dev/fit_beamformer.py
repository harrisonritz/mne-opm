# %% 
import mne
from mne.beamformer import apply_lcmv, make_lcmv

# %% param

subject = '004'


fwd_path = f'/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/trial-repeat/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-TSXpilot_fwd.fif'
epo_path = f'/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/trial-repeat/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-TSXpilot_proc-clean_epo.fif'
noise_covPath = f'/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/trial-repeat/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-noise_proc-clean_cov.fif'

# %% load fwd and epochs
fwd = mne.read_forward_solution(fwd_path)
epochs = mne.read_epochs(epo_path, preload=True)


# %% get evoked
# for speed purposes, cut to a window of interest
evoked = epochs.interpolate_bads().average(picks='mag', by_event_type=True)

contrast = mne.combine_evoked(evoked, weights=[1, 1, -1, -1]) # listen - read
del evoked


contrast.plot_joint(topomap_args = dict(vlim=(-200,200)))

# %% noise cov

noise_cov = mne.read_cov(noise_covPath)
data_cov = mne.compute_covariance(epochs, method="shrunk")
# del epochs


# %% forward model
forward = mne.read_forward_solution(fwd_path)


# %% make beamformer

filters = make_lcmv(
    contrast.info,
    forward,
    data_cov,
    reg=0.05,
    noise_cov=noise_cov,
    pick_ori="max-power",
    weight_norm="nai",
)


# You can save the filter for later use with:
# filters.save('filters-lcmv.h5')

# %% apply beamformer

stc = apply_lcmv(contrast, filters)
# del filters

# %% plot

stc.plot(
    subjects_dir='/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/freesurfer/subjects',
    initial_time=0.1,
    time_viewer=True,
    hemi='both',
    views=['lateral', 'medial'],
)

# %%
