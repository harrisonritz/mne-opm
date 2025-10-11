# %% 
import mne
from mne.beamformer import apply_lcmv, make_lcmv,apply_lcmv_cov
import matplotlib.pyplot as plt

# %% param

subject = '003'


# preproc = 'response_st-120_hpf-1'
# preproc = 'response_no-mf'
# preproc = 'response_hfc'
# preproc = 'response_keep-bads'

# preproc= 'trial'
preproc = 'trial-baseline'

# conds = ['left', 'right']

conds = ['listen_listen', 'read_read']

plot_cov = False
plot_time = True


fwd_path = f'/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/{preproc}/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-TSXpilot_fwd.fif'
epo_path = f'/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/{preproc}/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-TSXpilot_proc-clean_epo.fif'
noise_covPath = f'/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/{preproc}/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-noise_proc-clean_cov.fif'
subjects_dir = '/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/freesurfer/subjects'

# %% load fwd and epochs
fwd = mne.read_forward_solution(fwd_path)
epochs = mne.read_epochs(epo_path, preload=True)
cov_noise = mne.read_cov(noise_covPath)


# %% get evoked
# for speed purposes, cut to a window of interest 
# evoked = epochs.average(picks='mag', by_event_type=True)
# print(evoked)

# contrast = mne.combine_evoked(evoked, weights=[1, -1])
# del evoked

# contrast.plot_joint(topomap_args = dict(vlim=(-200,200)))
# plt.show()

# %% NERUAL ACTIVATION
# https://github.com/Neuronal-Oscillations/FLUX/blob/main/MNEPython/LCMV.ipynb

if plot_cov:
    print('\n\n ---------- Plotting Source-Localized Power ---------- ')

    cov_A = mne.compute_covariance(epochs[conds[0]], method="shrunk", tmin=-.1, tmax=.1)
    cov_B = mne.compute_covariance(epochs[conds[1]], method="shrunk", tmin=-.1, tmax=.1)

    total_cov = cov_A + cov_B


    filter = make_lcmv(
        epochs.info,
        fwd,
        total_cov,
        reg=0.05,
        noise_cov=cov_noise,
        pick_ori="max-power",
        weight_norm="nai",
    )


    stc_A = apply_lcmv_cov(cov_A, filter)
    stc_B = apply_lcmv_cov(cov_B, filter)

    stc_AB_diff = (stc_A - stc_B) / (stc_A + stc_B)


    stc_AB_diff.plot(
        subjects_dir=subjects_dir,
        surface='inflated',
        hemi='split',
        cortex='classic',
    )


# stc_noise = apply_lcmv_cov(cov_noise, filter)
# cov_noise = mne.read_cov(noise_covPath)
# stc_AB_sum = (stc_A + stc_B) / (stc_noise)
# stc_A = stc_A
# stc_B = stc_B

# stc_AB_sum.plot(
#     subjects_dir=subjects_dir,
#     surface='inflated',
#     hemi='split',
#     cortex='classic',
# )



# stc_A.plot(
#     subjects_dir=subjects_dir,
#     surface='inflated',
#     hemi='split',
#     cortex='classic',
# )



# stc_B.plot(
#     subjects_dir=subjects_dir,
#     surface='inflated',
#     hemi='split',
#     cortex='classic',
# )


# %% TIME COURSE


if plot_time:

    print('\n\n ---------- Plotting Source-Localized Activation ---------- ')

    cov_noise = mne.read_cov(noise_covPath)
    cov_epoch = mne.compute_covariance(epochs, method="shrunk")

    filter = make_lcmv(
        epochs.info,
        fwd,
        cov_epoch,
        reg=0.05,
        noise_cov=cov_noise,
        pick_ori="max-power",
        weight_norm="nai",
    )

    stc_A = apply_lcmv(epochs[conds[0]].average(), filter)
    stc_B = apply_lcmv(epochs[conds[1]].average(), filter)
    stc_all = apply_lcmv(epochs.average(), filter)

    stc_A = stc_A
    stc_B = stc_B
    stc_AB_diff = (stc_A - stc_B)

    stc_all.plot(
        subjects_dir=subjects_dir,
        surface='inflated',
        hemi='split',
        cortex='classic',
    )



# %%
