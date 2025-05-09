
# %%
import mne
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

%matplotlib qt



# %% load data
raw = mne.io.read_raw_fif(
    '/Users/hr0283/Projects/TSX_OPM/analysis/scripts/20250411_153733_meg.fif', 
    preload=True,
    )

# %%

# List the annotations
print("Annotations in the raw data:")
print(raw.annotations)

# Display more details about each annotation
# Get summary of unique annotation descriptions and their counts
descriptions = raw.annotations.description
unique_descriptions = set(descriptions)
print("\nUnique annotation types:")
for desc in unique_descriptions:
    count = sum(descriptions == desc)
    print(f"- {desc}: {count} occurrences")


# %%

# Plot the raw data in a new window
# Switch to interactive backend

# Plot the raw data in interactive mode
# raw.plot(block=True, show=True, n_channels=30, title='Raw MEG Data', duration=10.0)


# %% create binary channel

trigger_channels = [f'Trigger {i}' for i in range(1, 9)]  # Example trigger channels

# Extract data from trigger channels
trigger_data = []
for ch_name in trigger_channels:
    trigger_data.append(raw.get_data(ch_name))

# Convert binary pattern to integer at each time point
# Stack the channels together (shape: n_channels x n_timepoints)
stacked_triggers = np.vstack(trigger_data)

# plot histogram of stacked_triggers
plt.figure(figsize=(10, 4))
plt.hist(stacked_triggers.flatten(), bins=50, color='gray', alpha=0.7)
plt.title('Histogram of Stacked Trigger Channels')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.ylim(0, 1000)
plt.axvline(0, color='red', linestyle='--', label='Zero Line')
plt.legend()
plt.show()


# convert to binary
stacked_triggers[stacked_triggers < 1] = 0
stacked_triggers[stacked_triggers > 1] = 1

# plot stacked_triggers as heatmap
# plt.figure(figsize=(10, 4))
# sns.heatmap(stacked_triggers, cmap='gray', cbar=False)
# plt.title('Stacked Trigger Channels')
# plt.xlabel('Time Points')
# plt.ylabel('Trigger Channels')
# plt.xticks(ticks=np.arange(len(trigger_channels)), labels=trigger_channels)
# plt.yticks(ticks=np.arange(stacked_triggers.shape[0]), labels=trigger_channels)
# plt.show()

# Convert binary pattern to integer (using powers of 2)
powers_of_two = 2 ** np.arange(len(trigger_channels))[:, np.newaxis]  # Column vector
integer_triggers = np.sum(stacked_triggers * powers_of_two, axis=0).astype(int)

# plot integer_triggers

plt.figure(figsize=(10, 4))
plt.plot(integer_triggers, label='Integer Triggers')
plt.title('Integer Trigger Plot')
plt.xlabel('Time Points')
plt.ylabel('Trigger Value')
plt.legend()
plt.show()

# %% add new channel, get events

raw.add_channels([mne.io.RawArray(
    integer_triggers.reshape(1, -1), 
    mne.create_info(['Trigger Combined'], raw.info['sfreq'], ['stim'])
)],
    force_update_info=True,
)


events = mne.find_events(raw, stim_channel='Trigger Combined', min_duration=0.01, consecutive=True)

# plot events
mne.viz.plot_events(events, sfreq=raw.info['sfreq'])

# %% convert events to annotations
# 'feedback_col': [0,0,1],
# 'ITI_col': [0,0,2],
# 'read_noresp_trial_col': [0,0,4],
# 'listen_noresp_trial_col': [0,0,8],
# 'av_noresp_trial_col': [0,0,16],
# 'unimodal_read_col': [0,0,32],
# 'bimodal_read_col': [0,0,64],

# 'unimodal_listen_col': [0,0,6],
# 'bimodal_listen_col': [0,0,10],

# 'read_read_col': [0,0,9],
# 'listen_listen_col': [0,0,17],
# 'read_listen_col': [0,0,33],
# 'listen_read_col': [0,0,65],

# 'resp_col': [0,0,3], 
# 'CSI_col': [0,0,5],

annot_dict = {
    1: 'feedback',
    2: 'ITI',
    4: 'read_noresp_trial',
    8: 'listen_noresp_trial',
    16: 'av_noresp_trial',
    32: 'unimodal_read',
    64: 'bimodal_read',

    6: 'unimodal_listen',
    10: 'bimodal_listen',

    9: 'read_read',
    17: 'listen_listen',
    33: 'read_listen',
    65: 'listen_read',

    5: 'CSI',
}

annotations = mne.annotations_from_events(
    events,
    event_desc=annot_dict,
    sfreq=raw.info['sfreq'],
    orig_time=raw.info['meas_date'],
)
print(annotations)

# %%

test_raw= raw.copy()

test_raw.set_annotations(annotations + raw.copy().annotations)
test_raw.plot(start=2, duration=6)
