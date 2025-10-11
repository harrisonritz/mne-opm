# %% 
import mne
import  matplotlib.pyplot as plt
# set matplotlib backend
%matplotlib qt

subject = '005'


# %% plot raw

# raw = mne.io.read_raw(
#     f'/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/trial-repeat/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-TSXpilot_run-01_proc-clean_raw.fif', 
#     preload=True)

# raw.plot(show_scrollbars=True,scalings = {'mag' : 1e-12})
# plt.show()

# %% plot ICA

ica_path = f'/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/trial-repeat/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-TSXpilot_proc-ica_ica.fif'
epo_path = f'/Users/hr0283/Projects/TSX_OPM/data/bids/derivatives/trial-repeat/sub-{subject}/ses-01/meg/sub-{subject}_ses-01_task-TSXpilot_proc-ica_epo.fif'


ica = mne.preprocessing.read_ica(ica_path)
ica_epo = mne.read_epochs(epo_path, preload=True)

# plot in window
ica.copy().plot_sources(ica_epo.copy(), show_scrollbars=False)
plt.show()

# %%
