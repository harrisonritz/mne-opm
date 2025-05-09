# %%
import mne

# %%

fld = "/Volumes/hritz/2025_TSX_Pilot/bids/derivatives/preproc-trial/sub-003/ses-01/meg/"


# load ica
ica = mne.preprocessing.read_ica(
    fld + "sub-003_ses-01_task-TSXpilot_proc-icafit_ica.fif",
)

epochs = mne.read_epochs(
    fld + "sub-003_ses-01_task-TSXpilot_proc-icafit_epo.fif",
)

# %%

%matplotlib qt
ica.plot_sources(epochs)
