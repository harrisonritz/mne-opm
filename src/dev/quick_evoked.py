# %%
import matplotlib
import mne
import numpy as np
%matplotlib qt

# %%

fn = '/Volumes/hritz/2025_TSX_Pilot/bids/derivatives/preproc-trial/sub-003/ses-01/meg/sub-003_ses-01_task-TSXpilot_proc-clean_epo.fif'

epochs = mne.read_epochs(
    fn,
    preload=True,
)

epochs.apply_baseline((-.150, -.050))


# %% create diff evoked (listen vs read)

listen = epochs['listen_listen']
read = epochs['read_read']

# plot epochs
# mne.viz.plot_epochs_image(listen, picks="mag", evoked=True)
# mne.viz.plot_epochs_image(read, picks="mag", evoked=True)




# -- COMPARE MRF ---------------------
evoked_list = [
    listen.average(),
    read.average(),
]


mne.viz.plot_compare_evokeds(evoked_list, 
                             picks='mag', 
                             colors=['blue', 'red'], 
                             show_sensors='upper right', 
                             title="listen vs read", 
                             time_unit='s')





# -- PLOT JOINT DIFFERENCE ---------------------

total_epochs = len(listen) + len(read)
weights = [len(listen)/total_epochs, -len(read)/total_epochs]

evoked_diff = mne.combine_evoked(
    evoked_list, weights=weights
) 

# We need an evoked object to plot the image to be masked
evoked_diff = mne.combine_evoked(
    [listen.average(), read.average()], weights=[1, -1]
)  # calculate difference wave

topo_args = dict(time_unit="s", vlim=(-200,200))


evoked_diff.plot_joint(
    times=(.2, .4, .5, .6, .8),
    picks='mag',
    title="listen vs read", 
    ts_args=dict(time_unit="s"), 
    topomap_args=topo_args
)  # show difference wave


# %%
