# Main analysis config (template)
"""
Copy to <CONFIG_DIR>/<EXPERIMENT>/config-CSI.py and edit values.
This config expects environment variables exported by mne-opm.sh.
"""

import os

# Core paths (BIDS root should match DATA_BASE/EXPERIMENT/bids)
bids_root = os.getenv("BIDS_DIR", "")  # set by mne-opm.sh
subjects_dir = os.getenv("SUBJECTS_DIR", "")  # FreeSurfer SUBJECTS_DIR

# Derivatives root for this analysis
_analysis_name = 'CSI'
deriv_root = f"{bids_root}/derivatives/{_analysis_name}"

# Participants
subjects = ['007']
sessions = ['01']

# Recording/task
process_empty_room = True
ch_types = ['mag']
# Set your task label (used for BIDS lookup and pipeline config)
task = 'TSX'

# Filtering / preprocessing
l_freq = 1.0
h_freq = 90.0

# Manual steps toggles
_manual_bads = False   # enable manual bad channel marking
_manual_ica = False    # enable manual ICA selection

# Optional: skip certain steps if derivatives exist
_skip_on_deriv = True

# OSL/auxiliary options
find_breaks = True
min_break_duration = 6
t_break_annot_start_after_previous_event = 1.5
t_break_annot_stop_before_next_event = 1.5

# Spatial filter (for ICA stages in mne-bids-pipeline)
spatial_filter = 'ica'

# Maxwell filter toggle (if applicable in your setup)
use_maxwell_filter = True

# Eye-tracking and custom metadata hooks (optional)
epochs_custom_metadata = None
epochs_metadata_query = None
